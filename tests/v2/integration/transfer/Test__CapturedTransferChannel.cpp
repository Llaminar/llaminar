/**
 * @file Test__CapturedTransferChannel.cpp
 * @brief Real captured round trips across independently executing GPU endpoints.
 *
 * Two retained per-device graph families exchange bytes and return them without
 * a host relay. Alternating captured patterns distinguish fresh publication
 * from stale replay. Odd sizes, unaligned regions and guards prove exact extents.
 * All channel, scratch, readback and native graph storage is PMA-admitted.
 * Timings are diagnostic only; production preflight has no performance threshold.
 * An opt-in Release performance role reuses these exact channels and graphs at
 * ngram payload sizes. Source-clock timing ends after return consumption; wall
 * time includes both submissions/completions. Poison, readback and byte checks
 * stay outside those timers. This floor omits pooled lookup/decoding work.
 */
#include "transfer/CapturedTransferChannel.h"
#include "CapturedChannelGraphProof.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "collective/LocalTPCollectiveInventory.h"
#include "collective/ILocalTPContext.h"
#include "planning/CollectiveMemoryEstimator.h"
#include <barrier>
#include <functional>
#include <mutex>
#include "CapturedTransferChannelReplayProbe.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "planning/PlanningObservedResource.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Propagate native failure before recording/launching any dependent node. */
    void require(bool condition, const char *message)
    { if (!condition) throw std::runtime_error(message); }

    /** @brief Keep native graphs/buffers alive until an interrupted test's original streams drain. */
    struct NativeReplayFailureDrain final
    {
        std::function<void()> abort_and_join;
        bool completed = false; ///< Test lifetime only; never device count or logical success state.
        /** @brief Cancel the original communicator and require actual terminal proof before destruction. */
        ~NativeReplayFailureDrain() noexcept
        {
            if (!completed && abort_and_join)
                try { abort_and_join(); }
                catch (const std::exception& error)
                {
                    std::cerr << "Native counted fixture cannot prove physical retirement: " << error.what() << '\n';
                    std::terminate();
                }
        }
    };

    /** @return An offset-sensitive byte pattern which also changes between retained graphs. */
    unsigned char patternByte(size_t index, size_t pattern)
    {
        std::uint32_t value = static_cast<std::uint32_t>(index) * 0x9e3779b9u +
            (pattern ? 0xb7e15162u : 0x243f6a88u);
        value ^= value >> 16;
        value *= 0x85ebca6bu;
        value ^= value >> 13;
        return static_cast<unsigned char>(value ^ (value >> 8));
    }

    /** @brief Exact final-event owner; never performs a stream-wide synchronization. */
    struct Terminal
    {
        IBackend *backend;
        DeviceId device;
        void *event;
        /** @brief Retire only after the final observation, without a hidden host wait. */
        ~Terminal() { if (event) backend->destroyEvent(event, device.ordinal); }
        /** @brief Bounded test-only terminal observation after submitted GPU work.
         * @param timeout Existing fixture bound, with explicit 30s cancellation checks. */
        void await(std::chrono::seconds timeout = std::chrono::seconds(35)) const
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            bool ready = false;
            while (!ready)
            {
                require(backend->queryEvent(event, device.ordinal, &ready), "Captured channel terminal query failed");
                require(ready || std::chrono::steady_clock::now() < deadline, "Captured channel terminal timed out");
                if (!ready) std::this_thread::yield();
            }
        }
    };

    /**
     * @brief Fixture-only final cancellation and exact-event retirement before owner destruction.
     * All fields borrow the original workers/streams/events; this never changes
     * protocol state, substitutes a peer close or reports request success.
     */
    struct CancelledChannelRetirement final
    {
        TransferEngine &transfer;
        std::shared_ptr<CapturedTransferChannel> channel;
        CapturedTransferInvocation original;
        IWorkerGPUContext &producer;
        IWorkerGPUContext &consumer;
        void *producer_stream;
        void *consumer_stream;
        Terminal &producer_done;
        Terminal &consumer_done;
        /** @brief Cancel before exact terminal joins, including exceptional fixture exits. */
        ~CancelledChannelRetirement() noexcept
        {
            try
            {
                require(transfer.cancelCapturedTransferInvocation(channel, original), "Cancellation retirement lost its original admission");
                producer.submitAndWait([&] {
                    require(producer_done.backend->recordEvent(producer_done.event, producer_done.device.ordinal, producer_stream),
                        "Cancellation retirement producer event failed");
                });
                consumer.submitAndWait([&] {
                    require(consumer_done.backend->recordEvent(consumer_done.event, consumer_done.device.ordinal, consumer_stream),
                        "Cancellation retirement consumer event failed");
                });
                producer_done.await(std::chrono::seconds(30));
                consumer_done.await(std::chrono::seconds(30));
            }
            catch (...) { std::terminate(); } // Never release original backing without native completion.
        }
    };

    /** @brief Original request failure point; terminal cancellation is never a channel reset. */
    enum class CancellationBoundary { BeforeAcquire, PublishedPeerAbsent, PublishedPeerLate };

    /**
     * @brief Run original captured endpoints through exact absorbing CPU cancellation.
     * @param first Original producer device.
     * @param second Original consumer device.
     * @param boundary Cancel before acquisition or after a real producer release.
     * @note An external event between two launches of the same captured
     * operation proves first release before the second pending acquisition. The host never reads a live cursor or
     * selects progress from GPU values. All snapshots below follow terminal joins.
     */
    void cancellationReplay(DeviceId first, DeviceId second, CancellationBoundary boundary,
        CapturedTransferPayloadStorage payload_storage=CapturedTransferPayloadStorage::MappedHost)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &rank = inventory->ranks.at(mpi->rank());
        auto *producer_backend = getBackendFor(first);
        auto *consumer_backend = getBackendFor(second);
        ASSERT_NE(producer_backend, nullptr);
        ASSERT_NE(consumer_backend, nullptr);
        ASSERT_GT(producer_backend->deviceCount(), first.ordinal);
        ASSERT_GT(consumer_backend->deviceCount(), second.ordinal);
        auto &producer_worker = GPUDeviceContextPool::instance().getContext(first);
        auto &consumer_worker = GPUDeviceContextPool::instance().getContext(second);
        void *producer_stream = nullptr, *consumer_stream = nullptr;
        producer_worker.submitAndWait([&] { producer_stream = producer_worker.getOrCreateAuxiliaryStream("captured_channel_cancel"); });
        consumer_worker.submitAndWait([&] { consumer_stream = consumer_worker.getOrCreateAuxiliaryStream("captured_channel_cancel"); });
        ASSERT_NE(producer_stream, nullptr);
        ASSERT_NE(consumer_stream, nullptr);
        auto &transfer = TransferEngine::instance();
        // Both declared shapes reuse one original admission. Their storage is
        // mutually exclusive, so the explicit maximum is the same-PMA envelope.
        constexpr size_t planned_capacity = 8193 + 32;
        const size_t planned_snapshot_bytes = 2 * sizeof(CapturedTransferCursor) + sizeof(uint64_t) + planned_capacity;
        const auto geometry = CapturedTransferChannel::memoryFor(planned_capacity,payload_storage);
        PhysicalMemoryPlanBuilder builder;
        builder.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            geometry.mapped_host_bytes + TransferEngine::mappedHostRegionAllocationBytes(planned_capacity) +
            TransferEngine::mappedHostRegionAllocationBytes(planned_snapshot_bytes));
        builder.add(planningObservedResource(rank, first), PhysicalMemoryOwner::ActivationTransportStaging,
            geometry.cursor_bytes_per_device + planned_capacity + geometry.producer_payload_bytes);
        builder.add(planningObservedResource(rank, second), PhysicalMemoryOwner::ActivationTransportStaging,
            geometry.cursor_bytes_per_device + planned_capacity + sizeof(uint64_t));
        for (auto device : {first, second})
            builder.add(planningObservedResource(rank, device), PhysicalMemoryOwner::NativeGraphExecutable,
                GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        for (const size_t bytes : {size_t{257}, size_t{8193}})
        {
            SCOPED_TRACE(bytes); // Both original fused and parallel-copy paths.
            const size_t capacity = bytes + 32;
            const size_t cursor_bytes = sizeof(CapturedTransferCursor);
            const size_t count_offset = 2 * cursor_bytes;
            const size_t payload_offset = count_offset + sizeof(uint64_t);
            const size_t snapshot_bytes = payload_offset + capacity;
            auto channel=payload_storage==CapturedTransferPayloadStorage::PeerDevice
                ? transfer.createPeerCapturedTransferChannel(*authority,DeviceId::cpu(),
                    first,producer_stream,second,consumer_stream,capacity,rank)
                : transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),
                    first,producer_stream,second,consumer_stream,capacity);
            auto source_claim = authority->claimNewAllocation(first, PhysicalMemoryOwner::ActivationTransportStaging, capacity);
            auto destination_claim = authority->claimNewAllocation(second, PhysicalMemoryOwner::ActivationTransportStaging, capacity + sizeof(uint64_t));
            auto host_claim = authority->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(capacity) + TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
            auto source = transfer.allocateDeviceTransferBuffer(capacity, first);
            auto destination = transfer.allocateDeviceTransferBuffer(capacity, second);
            auto count = transfer.allocateDeviceTransferBuffer(sizeof(uint64_t), second);
            const std::array source_devices{first};
            const std::array both_devices{first, second};
            auto input = transfer.allocateMappedHostRegion(capacity, source_devices);
            auto snapshot = transfer.allocateMappedHostRegion(snapshot_bytes, both_devices);
            auto *input_bytes = static_cast<unsigned char *>(input->mutableHostData());
            for (size_t i = 0; i < capacity; ++i) input_bytes[i] = patternByte(i, 0);
            std::span<std::byte> observed_wire;
            if(payload_storage==CapturedTransferPayloadStorage::MappedHost){
                observed_wire=CapturedTransferChannelReplayProbe::payload(*channel);
                std::fill(observed_wire.begin(),observed_wire.end(),std::byte{0xa5});
            }else{
                // Reuse the original input mapping for a terminal observation;
                // it is no longer a producer input after the initial upload.
                observed_wire={static_cast<std::byte*>(input->mutableHostData()),capacity};
            }
            producer_worker.submitAndWait([&] {
                transfer.enqueueMappedHostToDevice(*input,0,*source,0,capacity,first,producer_stream);
                if(payload_storage==CapturedTransferPayloadStorage::PeerDevice){
                    const auto wire=CapturedTransferChannelReplayProbe::terminalPeerPayload(*channel);
                    require(producer_backend->memset(wire->mutableDeviceData(),0xa5,capacity,first.ordinal,producer_stream),
                        "Cancellation peer payload poison failed");
                }
            });
            consumer_worker.submitAndWait([&] {
                require(consumer_backend->memset(destination->mutableDeviceData(), 0xa5, capacity, second.ordinal, consumer_stream), "Cancellation destination poison failed");
                require(consumer_backend->memset(count->mutableDeviceData(), 0xa5, sizeof(uint64_t), second.ordinal, consumer_stream), "Cancellation extent poison failed");
            });
            Terminal progress{producer_backend, first, producer_backend->createEvent(first.ordinal)};
            Terminal producer_done{producer_backend, first, producer_backend->createEvent(first.ordinal)};
            Terminal consumer_done{consumer_backend, second, consumer_backend->createEvent(second.ordinal)};
            ASSERT_NE(progress.event, nullptr);
            ASSERT_NE(producer_done.event, nullptr);
            ASSERT_NE(consumer_done.event, nullptr);
            const auto send = transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Producer, {71, bytes}, source);
            const auto receive = transfer.bindDeviceCountedTransfer(
                transfer.bindCapturedTransfer(channel, CapturedTransferEndpoint::Consumer, {71, bytes}, destination), count);
            std::unique_ptr<IGPUGraphCapture> producer_graph, consumer_graph;
            producer_worker.submitAndWait([&] {
                producer_graph = producer_worker.createGraphCapture(producer_stream);
                require(bool(producer_graph), "Cancellation producer graph missing");
                ScopedBackendGraphCapture recording(producer_worker, *producer_graph, "captured cancellation producer");
                require(recording.begin(), "Cancellation producer capture failed");
                transfer.enqueueCapturedTransfer(send, producer_stream);
                recording.finish();
                counted_channel_test::verifyCapturedChannelGraph(*producer_graph, bytes, 1);
                require(producer_graph->instantiate(), "Cancellation producer instantiate failed");
            });
            consumer_worker.submitAndWait([&] {
                consumer_graph = consumer_worker.createGraphCapture(consumer_stream);
                require(bool(consumer_graph), "Cancellation consumer graph missing");
                ScopedBackendGraphCapture recording(consumer_worker, *consumer_graph, "captured cancellation consumer");
                require(recording.begin(), "Cancellation consumer capture failed");
                transfer.enqueueCapturedTransfer(receive, consumer_stream);
                recording.finish();
                counted_channel_test::verifyCapturedChannelGraph(*consumer_graph, bytes, 1);
                require(consumer_graph->instantiate(), "Cancellation consumer instantiate failed");
            });
            const CapturedTransferInvocation original{17, 29};
            transfer.beginCapturedTransferInvocation(channel, original);
            CancelledChannelRetirement terminal_owner{transfer, channel, original,
                producer_worker, consumer_worker, producer_stream, consumer_stream, producer_done, consumer_done};
            EXPECT_FALSE(transfer.cancelCapturedTransferInvocation(channel, {18, 29}));
            EXPECT_FALSE(transfer.cancelCapturedTransferInvocation(channel, {17, 30}));
            if (boundary == CancellationBoundary::BeforeAcquire)
                ASSERT_TRUE(transfer.cancelCapturedTransferInvocation(channel, original));
            producer_worker.submitAndWait([&] {
                require(producer_graph->launch(), "Cancellation producer submit failed");
                // This is an external completion receipt BETWEEN retained
                // submissions, not a capture-internal dependency event. Its
                // exact stream proves epoch1 publication before cancellation.
                require(producer_backend->recordEvent(progress.event, first.ordinal, producer_stream), "Cancellation progress event recording failed");
                // The same captured operation now waits for the absent peer's
                // epoch1 acknowledgement before it can publish epoch2.
                require(producer_graph->launch(), "Cancellation pending producer submit failed");
                require(producer_backend->recordEvent(producer_done.event, first.ordinal, producer_stream), "Cancellation producer terminal recording failed");
            });
            if (boundary != CancellationBoundary::BeforeAcquire)
            {
                progress.await(std::chrono::seconds(30));
                ASSERT_TRUE(transfer.cancelCapturedTransferInvocation(channel, original));
            }
            const bool submit_consumer = boundary != CancellationBoundary::PublishedPeerAbsent;
            if (submit_consumer)
                consumer_worker.submitAndWait([&] {
                    require(consumer_graph->launch(), "Cancellation consumer submit failed");
                    require(consumer_backend->recordEvent(consumer_done.event, second.ordinal, consumer_stream), "Cancellation consumer terminal recording failed");
                });
            producer_done.await(std::chrono::seconds(30));
            if (submit_consumer) consumer_done.await(std::chrono::seconds(30));
            // The exact cancelled graphs remain absorbing through repeated
            // replay. No channel reset or new admission disguises failure.
            for (int replay = 0; replay < 20; ++replay)
            {
                producer_worker.submitAndWait([&] {
                    require(producer_graph->launch(), "Cancelled producer replay failed");
                    require(producer_graph->launch(), "Cancelled pending producer replay failed");
                    require(producer_backend->recordEvent(producer_done.event, first.ordinal, producer_stream), "Cancelled producer replay terminal failed");
                });
                if (submit_consumer)
                    consumer_worker.submitAndWait([&] {
                        require(consumer_graph->launch(), "Cancelled consumer replay failed");
                        require(consumer_backend->recordEvent(consumer_done.event, second.ordinal, consumer_stream), "Cancelled consumer replay terminal failed");
                    });
                producer_done.await(std::chrono::seconds(30));
                if (submit_consumer) consumer_done.await(std::chrono::seconds(30));
            }
            const auto producer_cursor = CapturedTransferChannelReplayProbe::terminalCursor(*channel, CapturedTransferEndpoint::Producer);
            const auto consumer_cursor = CapturedTransferChannelReplayProbe::terminalCursor(*channel, CapturedTransferEndpoint::Consumer);
            producer_worker.submitAndWait([&] {
                transfer.enqueueDeviceToMappedHost(*producer_cursor,0,*snapshot,0,cursor_bytes,first,producer_stream);
                if(payload_storage==CapturedTransferPayloadStorage::PeerDevice){
                    const auto wire=CapturedTransferChannelReplayProbe::terminalPeerPayload(*channel);
                    transfer.enqueueDeviceToMappedHost(*wire,0,*input,0,capacity,first,producer_stream);
                }
                require(producer_backend->recordEvent(producer_done.event, first.ordinal, producer_stream), "Producer cursor terminal failed");
            });
            consumer_worker.submitAndWait([&] {
                transfer.enqueueDeviceToMappedHost(*consumer_cursor, 0, *snapshot, cursor_bytes, cursor_bytes, second, consumer_stream);
                transfer.enqueueDeviceToMappedHost(*count, 0, *snapshot, count_offset, sizeof(uint64_t), second, consumer_stream);
                transfer.enqueueDeviceToMappedHost(*destination, 0, *snapshot, payload_offset, capacity, second, consumer_stream);
                require(consumer_backend->recordEvent(consumer_done.event, second.ordinal, consumer_stream), "Consumer cursor terminal failed");
            });
            producer_done.await(std::chrono::seconds(30));
            consumer_done.await(std::chrono::seconds(30));
            CapturedTransferCursor producer_state{}, consumer_state{};
            uint64_t consumer_bytes = 0;
            auto *observed = static_cast<const std::byte *>(snapshot->mutableHostData());
            std::memcpy(&producer_state, observed, cursor_bytes);
            std::memcpy(&consumer_state, observed + cursor_bytes, cursor_bytes);
            std::memcpy(&consumer_bytes, observed + count_offset, sizeof(uint64_t));
            EXPECT_EQ(producer_state.phase, CapturedTransferPhase::Cancelled);
            EXPECT_EQ(producer_state.completed_epoch, boundary == CancellationBoundary::BeforeAcquire ? 0u : 1u);
            EXPECT_EQ(consumer_state.phase, submit_consumer ? CapturedTransferPhase::Cancelled : CapturedTransferPhase::Idle);
            EXPECT_EQ(consumer_state.completed_epoch, 0u);
            EXPECT_EQ(consumer_bytes, submit_consumer ? 0u : UINT64_C(0xa5a5a5a5a5a5a5a5));
            const auto &control = CapturedTransferChannelReplayProbe::terminalControl(*channel);
            EXPECT_EQ(control.cancelled_nonce, channel->identity().nonce);
            EXPECT_EQ(control.producer.epoch, kCapturedTransferAbortEpoch);
            EXPECT_EQ(control.consumer.epoch, submit_consumer ? kCapturedTransferAbortEpoch : 0u);
            for (size_t i = 0; i < capacity; ++i)
            {
                EXPECT_EQ(observed[payload_offset + i], std::byte{0xa5}) << i;
                EXPECT_EQ(observed_wire[i], boundary != CancellationBoundary::BeforeAcquire && i < bytes
                    ? std::byte{patternByte(i, 0)} : std::byte{0xa5}) << i;
            }
            EXPECT_TRUE(transfer.cancelCapturedTransferInvocation(channel, original));
            EXPECT_EQ(transfer.finishCapturedTransferInvocation(channel, original), CapturedTransferInvocationRetirement::Cancelled);
            EXPECT_THROW(transfer.finishCapturedTransferInvocation(channel, original), std::logic_error);
            EXPECT_THROW(transfer.beginCapturedTransferInvocation(channel, {17, 30}), std::logic_error);
            // Real Failed/InvalidPhase remains fatal in native lowering; this
            // separate actual-law control does NOT execute a device trap or
            // claim native fault recovery as part of explicit cancellation.
            CapturedTransferCursor prior{};
            prior.phase = CapturedTransferPhase::Failed;
            const auto failed = CapturedTransferChannelProtocol::acquire(CapturedTransferEndpoint::Producer,
                channel->identity(), channel->identity(), prior, {}, {71, bytes},
                CapturedTransferExtentSource::FixedMessage, 0, 0, 0, 0, channel->identity().nonce);
            EXPECT_EQ(failed.status, CapturedTransferStatus::InvalidPhase);
            EXPECT_EQ(prior.phase, CapturedTransferPhase::Failed);
            EXPECT_FALSE(CapturedTransferChannelProtocol::drainsCancellation(prior.phase, CapturedTransferStatus::Cancelled, true));
        }
    }

    /** @brief Physical owner exercised by the peer's two captured message endpoints. */
    enum class PeerStorage { PrivateScratch, RetainedWorkspace };

    /** @brief Immutable fixture geometry; profiler probes never change production lowering. */
    struct ChannelWorkload
    {
        size_t capacity = 1024 * 1024 + 3;
        std::vector<size_t> bytes{1, 257, 4099, 1024 * 1024 + 3};
        std::vector<size_t> offsets{1, 16};
        size_t replays = 64;
        size_t patterns = 2;
        PeerStorage peer_storage = PeerStorage::PrivateScratch;
        bool measure_floor = false; ///< Opt-in performance role; functional timing remains unchanged.
        CapturedTransferPayloadStorage payload_storage = CapturedTransferPayloadStorage::MappedHost;
        bool fixed_native_collective = false; ///< Fixed byte transport floor only; never a device-counted ngram policy.
        size_t warmups = 0; ///< Unreported replays on the same original cursor/graph family.
    };

    /**
     * @brief Prove exact per-device retained graphs without observing either live cursor.
     * @param first Source and round-trip result GPU, independent of vendor ordering.
     * @param second Peer receiving and returning bytes through its own private bank.
     * @param workload Exact geometry; the default is the full functional preflight sweep.
     */
    void roundTrip(DeviceId first, DeviceId second, const ChannelWorkload &workload = {})
    {
        ASSERT_GT(workload.capacity, 0u);
        ASSERT_GT(workload.replays, 0u);
        ASSERT_GT(workload.patterns, 0u);
        ASSERT_LE(workload.patterns, 2u);
        ASSERT_FALSE(workload.bytes.empty());
        ASSERT_FALSE(workload.offsets.empty());
        for (const auto bytes : workload.bytes)
        { ASSERT_GT(bytes, 0u); ASSERT_LE(bytes, workload.capacity); }
        for (const auto offset : workload.offsets) ASSERT_LE(offset, 64u);
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &rank = inventory->ranks.at(mpi->rank());
        auto *first_backend = getBackendFor(first);
        auto *second_backend = getBackendFor(second);
        ASSERT_NE(first_backend, nullptr);
        ASSERT_NE(second_backend, nullptr);
        ASSERT_GT(first_backend->deviceCount(), first.ordinal);
        ASSERT_GT(second_backend->deviceCount(), second.ordinal);
        std::unique_ptr<test::PreparedNativeLookupFixture> original;
        if (workload.fixed_native_collective)
        {
            const std::array devices{first, second};
            original = test::PreparedNativeLookupFixture::prepare(devices);
        }
        const auto resourceFor = [&](DeviceId device) {
            return original ? original->resource(device) : planningObservedResource(rank, device);
        };
        auto &first_worker = GPUDeviceContextPool::instance().getContext(first);
        auto &second_worker = GPUDeviceContextPool::instance().getContext(second);
        void *first_stream = nullptr, *second_stream = nullptr;
        first_worker.submitAndWait([&] {
            // A prepared participant graph borrows its original admitted
            // recording stream. The mapped fixture owns its separate lane.
            first_stream = original ? original->permit(first).worker()->defaultStream()
                : first_worker.getOrCreateAuxiliaryStream("captured_channel_round_trip");
        });
        second_worker.submitAndWait([&] {
            second_stream = original ? original->permit(second).worker()->defaultStream()
                : second_worker.getOrCreateAuxiliaryStream("captured_channel_round_trip");
        });
        ASSERT_NE(first_stream, nullptr);
        ASSERT_NE(second_stream, nullptr);

        const size_t capacity = workload.capacity;
        const bool workspace_peer = workload.peer_storage == PeerStorage::RetainedWorkspace;
        const size_t buffer_bytes = capacity + 64;
        const auto geometry = workload.fixed_native_collective ? CapturedTransferChannelMemory{} :
            CapturedTransferChannel::memoryFor(capacity, workload.payload_storage);
        std::unique_ptr<ILocalTPContext> fixed_collective;
        LocalTPCollectiveMemoryBOM collective_bom{};
        if (workload.fixed_native_collective) {
            require(first.type == second.type && first != second && workload.measure_floor && !workspace_peer,
                "Native fixed floor requires distinct homogeneous endpoints and private scratch");
            require(workload.bytes.size() == 1 && workload.offsets.size() == 1,
                "Native fixed floor requires its declared two-definition family");
            collective_bom = CollectiveMemoryEstimator::localTP(1, static_cast<int>(capacity),
                first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        }
        const auto readback_bytes = TransferEngine::mappedHostRegionAllocationBytes(buffer_bytes);
        PhysicalMemoryPlanBuilder builder;
        builder.add(resourceFor(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            2 * geometry.mapped_host_bytes + 3 * readback_bytes);
        for (auto device : {first, second})
        {
            const auto resource = resourceFor(device);
            builder.add(resource, PhysicalMemoryOwner::ActivationTransportStaging,
                2 * geometry.cursor_bytes_per_device + geometry.producer_payload_bytes + (device == first ? 3 : workspace_peer ? 0 : 1) * buffer_bytes);
            if (device == second && workspace_peer)
                builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, buffer_bytes);
            if (original)
            {
                const auto graph_bom = RetainedParticipantGraphExecutable::generalBOM(
                    resourceFor(DeviceId::cpu()), resource, 2);
                for (const auto& row : graph_bom.resources()) builder.add(row);
            }
            else
                builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable,
                    2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
            if (original)
                builder.add(resource, PhysicalMemoryOwner::LocalCollective, collective_bom.perDeviceBytes());
        }
        if (original) original->admit(builder.build());
        auto authority = original ? original->memory() : std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        auto &transfer = TransferEngine::instance();
        if (original)
        {
            const std::vector addresses{first.is_cuda() ? GlobalDeviceAddress::cuda(first.ordinal) : GlobalDeviceAddress::rocm(first.ordinal),
                second.is_cuda() ? GlobalDeviceAddress::cuda(second.ordinal) : GlobalDeviceAddress::rocm(second.ordinal)};
            fixed_collective = createLocalTPContext(addresses, {},
                first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
            require(bool(fixed_collective), "Native fixed floor communicator creation failed");
            require(fixed_collective->reserveCollectiveResources(collective_bom.backend_payload_capacity_bytes,
                collective_bom.fp16_scratch_elements, authority), "Native fixed floor collective admission failed");
        }
        {
            const auto create=[&](DeviceId producer,void* producerStream,DeviceId consumer,void* consumerStream){
                if(workload.payload_storage==CapturedTransferPayloadStorage::PeerDevice)
                    return transfer.createPeerCapturedTransferChannel(*authority,DeviceId::cpu(),
                        producer,producerStream,consumer,consumerStream,capacity,rank);
                return transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),
                    producer,producerStream,consumer,consumerStream,capacity);
            };
            auto forward = fixed_collective ? std::shared_ptr<CapturedTransferChannel>{} : create(first,first_stream,second,second_stream);
            auto backward = fixed_collective ? std::shared_ptr<CapturedTransferChannel>{} : create(second,second_stream,first,first_stream);
            // Scratch leases precede storage owners so reverse destruction
            // releases physical bytes before releasing their canonical charge.
            auto first_claim = authority->claimNewAllocation(first, PhysicalMemoryOwner::ActivationTransportStaging, 3 * buffer_bytes);
            PhysicalMemoryAllocationLease second_claim;
            auto readback_claim = authority->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, 3 * readback_bytes);
            std::array<std::shared_ptr<DeviceTransferBuffer>, 2> sources;
            auto returned = transfer.allocateDeviceTransferBuffer(buffer_bytes, first);
            std::shared_ptr<DeviceTransferBuffer> peer;
            std::shared_ptr<const WorkspaceBufferLease> peer_region;
            if (workspace_peer)
            {
                DeviceWorkspaceManager workspace(second, buffer_bytes, authority);
                WorkspaceRequirements reqs;
                reqs.buffers.push_back({"pipeline_metadata", buffer_bytes, 1, true});
                ASSERT_TRUE(workspace.allocate(reqs));
                peer_region = workspace.retainBuffer("pipeline_metadata", 0, buffer_bytes);
                // Retire the namespace before any capture. The binding must
                // own actual bytes, not rely on a manager that happens to live.
                workspace.release();
                EXPECT_EQ(authority->claimedBytes(second, PhysicalMemoryOwner::ExecutionWorkspace,
                    PhysicalMemoryMaterializationKind::NewAllocation), buffer_bytes);
                EXPECT_THROW((void)transfer.bindCapturedTransfer(forward, CapturedTransferEndpoint::Producer,
                    {11, 1}, peer_region), std::invalid_argument);
                EXPECT_THROW((void)transfer.bindCapturedTransfer(forward, CapturedTransferEndpoint::Consumer,
                    {11, 1}, peer_region, buffer_bytes), std::out_of_range);
            }
            else
            {
                second_claim = authority->claimNewAllocation(second, PhysicalMemoryOwner::ActivationTransportStaging, buffer_bytes);
                peer = transfer.allocateDeviceTransferBuffer(buffer_bytes, second);
            }
            const std::array readback_devices{first};
            auto readback = transfer.allocateMappedHostRegion(buffer_bytes, readback_devices);
            std::array<std::shared_ptr<MappedHostTransferRegion>, 2> inputs;
            for (size_t pattern = 0; pattern < sources.size(); ++pattern)
            {
                sources[pattern] = transfer.allocateDeviceTransferBuffer(buffer_bytes, first);
                inputs[pattern] = transfer.allocateMappedHostRegion(buffer_bytes, readback_devices);
                auto *data = static_cast<unsigned char *>(inputs[pattern]->mutableHostData());
                for (size_t i = 0; i < buffer_bytes; ++i) data[i] = patternByte(i, pattern);
                // Setup uploads precede every launch on this same exact stream.
                // The host sources remain immutable and retained until teardown.
                transfer.enqueueMappedHostToDevice(*inputs[pattern], 0, *sources[pattern], 0,
                    buffer_bytes, first, first_stream);
            }
            Terminal first_done{first_backend, first, first_backend->createEvent(first.ordinal)};
            Terminal second_done{second_backend, second, second_backend->createEvent(second.ordinal)};
            ASSERT_NE(first_done.event, nullptr);
            ASSERT_NE(second_done.event, nullptr);
            // Completion events retain their original protocol role. Timing
            // belongs to the source clock only, never mixed CUDA/HIP timestamps.
            Terminal timing_start{first_backend, first,
                workload.measure_floor ? first_backend->createTimingEvent(first.ordinal) : nullptr};
            Terminal timing_stop{first_backend, first,
                workload.measure_floor ? first_backend->createTimingEvent(first.ordinal) : nullptr};
            if (workload.measure_floor) {
                ASSERT_NE(timing_start.event, nullptr);
                ASSERT_NE(timing_stop.event, nullptr);
            }

            // Geometry changes create distinct recordings, while channel epochs
            // continue monotonically across graph-family/request-like boundaries.
            for (const size_t bytes : workload.bytes)
                for (const size_t offset : workload.offsets)
                {
                    PhysicalMemoryOwnerReservation first_graph_claim, second_graph_claim;
                    if (!original)
                    {
                        first_graph_claim = authority->reserveNewAllocations(first, PhysicalMemoryOwner::NativeGraphExecutable,
                            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(first));
                        second_graph_claim = authority->reserveNewAllocations(second, PhysicalMemoryOwner::NativeGraphExecutable,
                            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(second));
                    }
                    std::array<std::array<std::shared_ptr<RetainedParticipantGraphExecutable>, 2>, 2> native_owners;
                    std::array<std::array<IGPUGraphCapture*, 2>, 2> native_definitions{};
                    if (original)
                        for (size_t participant = 0; participant < 2; ++participant)
                            for (size_t pattern = 0; pattern < 2; ++pattern)
                                native_owners[participant][pattern] = test::RetainedParticipantGraphExecutableFixtureAccess::original(
                                    original->permit(participant == 0 ? first : second),
                                    participant == 0 ? first_stream : second_stream);
                    std::array<std::unique_ptr<IGPUGraphCapture>, 2> first_graphs, second_graphs;
                    NativeReplayFailureDrain failure_drain;
                    if (original)
                        failure_drain.abort_and_join = [&] {
                            fixed_collective->requestAbort();
                            first_worker.submitAndWait([&] {
                                require(first_backend->recordEvent(first_done.event, first.ordinal, first_stream), "Fixed failure source terminal failed");
                            });
                            second_worker.submitAndWait([&] {
                                require(second_backend->recordEvent(second_done.event, second.ordinal, second_stream), "Fixed failure peer terminal failed");
                            });
                            first_done.await();
                            second_done.await();
                            for (const auto& participant : native_owners)
                                for (const auto& owner : participant)
                                    require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owner),
                                        "Fixed failure original stream retirement unproved");
                        };
                    if (fixed_collective) {
                        // Both native captures stay open until every matching library
                        // operation is recorded. Dependent return is a SECOND group
                        // on the same stream; one group cannot order receive→send.
                        for (size_t pattern = 0; pattern < 2; ++pattern) {
                            std::barrier begun(2), recorded(2);
                            std::array<std::exception_ptr, 2> errors{};
                            std::array<bool, 2> ready{};
                            std::mutex error_mutex;
                            std::exception_ptr first_error;
                            const auto remember_error = [&](int slot) {
                                const auto error = std::current_exception();
                                fixed_collective->requestAbort();
                                if (!errors[slot]) errors[slot] = error;
                                std::lock_guard lock(error_mutex);
                                if (!first_error) first_error = error;
                            };
                            const auto record = [&](int slot) {
                                auto& worker = slot == 0 ? first_worker : second_worker;
                                auto& graph = native_definitions[slot][pattern];
                                void* stream = slot == 0 ? first_stream : second_stream;
                                std::unique_ptr<ScopedBackendGraphCapture> capture;
                                try {
                                    graph = &test::RetainedParticipantGraphExecutableFixtureAccess::capture(native_owners[slot][pattern]);
                                    require(graph != nullptr, "Native fixed floor capture owner missing");
                                    capture = std::make_unique<ScopedBackendGraphCapture>(worker, *graph, "native fixed ngram transfer floor");
                                    require(capture->begin(), "Native fixed floor begin failed");
                                    ready[slot] = true;
                                } catch (...) { remember_error(slot); }
                                begun.arrive_and_wait();
                                if (ready[0] && ready[1]) try {
                                    auto* send = static_cast<std::byte*>(sources[pattern]->mutableDeviceData()) + offset;
                                    auto* middle = static_cast<std::byte*>(peer->mutableDeviceData()) + offset;
                                    auto* result = static_cast<std::byte*>(returned->mutableDeviceData()) + offset;
                                    const CollectiveP2POp forward_op{slot == 0 ? CollectiveP2POpKind::Send : CollectiveP2POpKind::Recv,
                                        slot == 0 ? send : nullptr, slot == 1 ? middle : nullptr, bytes, CollectiveDataType::INT8, 1 - slot};
                                    const CollectiveP2POp reverse_op{slot == 1 ? CollectiveP2POpKind::Send : CollectiveP2POpKind::Recv,
                                        slot == 1 ? middle : nullptr, slot == 0 ? result : nullptr, bytes, CollectiveDataType::INT8, 1 - slot};
                                    require(fixed_collective->groupedP2PRawOnStream({forward_op}, slot, stream, "ngram_fixed_floor_forward"),
                                        "Native fixed floor forward failed");
                                    require(fixed_collective->groupedP2PRawOnStream({reverse_op}, slot, stream, "ngram_fixed_floor_return"),
                                        "Native fixed floor return failed");
                                } catch (...) { remember_error(slot); }
                                recorded.arrive_and_wait();
                                if (capture) try { capture->finish(); }
                                    catch (...) { if (!errors[slot]) remember_error(slot); }
                                if (!errors[slot]) try {
                                    require(graph->prepareRuntimeContextStorage(authority), "Native fixed floor runtime storage preparation failed");
                                    require(graph->instantiate(), "Native fixed floor instantiate failed");
                                }
                                    catch (...) { remember_error(slot); }
                            };
                            auto first_record = first_worker.submitAsync([&] { record(0); });
                            auto second_record = second_worker.submitAsync([&] { record(1); });
                            first_record.get();
                            second_record.get();
                            if (first_error) std::rethrow_exception(first_error);
                        }
                    } else {
                    const std::array sends{
                        transfer.bindCapturedTransfer(forward, CapturedTransferEndpoint::Producer, {11, bytes}, sources[0], offset),
                        transfer.bindCapturedTransfer(forward, CapturedTransferEndpoint::Producer, {11, bytes}, sources[1], offset)};
                    const auto receive = workspace_peer
                        ? transfer.bindCapturedTransfer(forward, CapturedTransferEndpoint::Consumer, {11, bytes}, peer_region, offset)
                        : transfer.bindCapturedTransfer(forward, CapturedTransferEndpoint::Consumer, {11, bytes}, peer, offset);
                    const auto reply = workspace_peer
                        ? transfer.bindCapturedTransfer(backward, CapturedTransferEndpoint::Producer, {12, bytes}, peer_region, offset)
                        : transfer.bindCapturedTransfer(backward, CapturedTransferEndpoint::Producer, {12, bytes}, peer, offset);
                    const auto collect = transfer.bindCapturedTransfer(backward, CapturedTransferEndpoint::Consumer, {12, bytes}, returned, offset);
                    for (size_t pattern = 0; pattern < 2; ++pattern)
                    {
                        first_worker.submitAndWait([&] {
                            auto &graph = first_graphs[pattern];
                            graph = first_worker.createGraphCapture(first_stream);
                            require(bool(graph), "Missing source native capture");
                            {
                                ScopedBackendGraphCapture capture(first_worker, *graph, "captured channel round trip source");
                                require(capture.begin(), "Source capture begin failed");
                                if (!workload.measure_floor)
                                    require(first_backend->memset(returned->mutableDeviceData(), 0xa5,
                                        buffer_bytes, first.ordinal, first_stream), "Returned guard initialization failed");
                                transfer.enqueueCapturedTransfer(sends[pattern], first_stream);
                                transfer.enqueueCapturedTransfer(collect, first_stream);
                                capture.finish();
                            }
                            counted_channel_test::verifyCapturedChannelGraph(*graph, bytes, 2);
                            require(graph->instantiate(), "Source graph incomplete");
                        });
                        second_worker.submitAndWait([&] {
                            auto &graph = second_graphs[pattern];
                            graph = second_worker.createGraphCapture(second_stream);
                            require(bool(graph), "Missing peer native capture");
                            {
                                ScopedBackendGraphCapture capture(second_worker, *graph, "captured channel round trip peer");
                                require(capture.begin(), "Peer capture begin failed");
                                transfer.enqueueCapturedTransfer(receive, second_stream);
                                transfer.enqueueCapturedTransfer(reply, second_stream);
                                capture.finish();
                            }
                            counted_channel_test::verifyCapturedChannelGraph(*graph, bytes, 2);
                            require(graph->instantiate(), "Peer graph incomplete");
                        });
                    }
                    }
                    if (original)
                    {
                        // First-use communicator storage may appear at instantiate;
                        // report this mixed lifetime interval without attributing it
                        // to graph slots or changing any original PMA allowance.
                        for (size_t participant = 0; participant < 2; ++participant)
                            ::testing::Test::RecordProperty("native_fixed_first_use_mixed_bytes_" + std::to_string(participant),
                                std::to_string(native_definitions[participant][0]->residentMemoryBytes() +
                                    native_definitions[participant][1]->residentMemoryBytes()));
                    }
                    else
                    {
                    // Native graph pools grow in slabs. Attest the complete
                    // retained family, never attribute one slab to one slot.
                    EXPECT_TRUE(GPUGraphMemoryContract::acceptsFamilyObservation(first,
                        first_graphs[0]->residentMemoryBytes() + first_graphs[1]->residentMemoryBytes(),
                        2 * GPUGraphMemoryContract::reservationBytesPerExecutable(first)))
                        << "source family observed=" << first_graphs[0]->residentMemoryBytes()
                        << "+" << first_graphs[1]->residentMemoryBytes() << " admitted="
                        << 2 * GPUGraphMemoryContract::reservationBytesPerExecutable(first);
                    EXPECT_TRUE(GPUGraphMemoryContract::acceptsFamilyObservation(second,
                        second_graphs[0]->residentMemoryBytes() + second_graphs[1]->residentMemoryBytes(),
                        2 * GPUGraphMemoryContract::reservationBytesPerExecutable(second)))
                        << "peer family observed=" << second_graphs[0]->residentMemoryBytes()
                        << "+" << second_graphs[1]->residentMemoryBytes() << " admitted="
                        << 2 * GPUGraphMemoryContract::reservationBytesPerExecutable(second);
                    }
                    const auto started = std::chrono::steady_clock::now();
                    for (size_t replay = 0; replay < workload.warmups + workload.replays; ++replay)
                    {
                        const auto pattern = replay % workload.patterns;
                        // Each retained definition owns its submission sequence.
                        // Alternating immutable inputs must not skip ordinal 1
                        // on the second graph or ordinal 2 on the first graph.
                        const auto definition_ordinal = replay / workload.patterns + 1;
                        if (workload.measure_floor) {
                            // Re-poison every inactive byte before timing. The
                            // timed retained program contains channel endpoints
                            // only; immutable source uploads occurred at setup.
                            first_worker.submitAndWait([&] {
                                require(first_backend->memset(returned->mutableDeviceData(), 0xa5,
                                    buffer_bytes, first.ordinal, first_stream), "Floor guard initialization failed");
                                require(first_backend->recordEvent(first_done.event, first.ordinal, first_stream), "Floor guard completion failed");
                            });
                            first_done.await();
                        }
                        const auto wall_start = std::chrono::steady_clock::now();
                        const auto submit_first = [&] { first_worker.submitAndWait([&] {
                            if (workload.measure_floor)
                                require(first_backend->recordEvent(timing_start.event, first.ordinal, first_stream), "Floor source timing start failed");
                            require(original ? test::RetainedParticipantGraphExecutableFixtureAccess::submit(native_owners[0][pattern], definition_ordinal)
                                : first_graphs[pattern]->launch(), "Source retained launch failed");
                            if (workload.measure_floor)
                                require(first_backend->recordEvent(timing_stop.event, first.ordinal, first_stream), "Floor source timing stop failed");
                            else
                                transfer.enqueueDeviceToMappedHost(*returned, 0, *readback, 0, buffer_bytes, first, first_stream);
                            require(first_backend->recordEvent(first_done.event, first.ordinal, first_stream), "Source terminal publication failed");
                        }); };
                        const auto submit_second = [&] { second_worker.submitAndWait([&] {
                            require(original ? test::RetainedParticipantGraphExecutableFixtureAccess::submit(native_owners[1][pattern], definition_ordinal)
                                : second_graphs[pattern]->launch(), "Peer retained launch failed");
                            require(second_backend->recordEvent(second_done.event, second.ordinal, second_stream), "Peer terminal publication failed");
                        }); };
                        // Either endpoint may be queued first. No observed model
                        // state, host copy or terminal wait chooses the peer work.
                        if (workload.measure_floor) { submit_second(); submit_first(); }
                        else if (replay % 4 < 2) { submit_first(); submit_second(); }
                        else { submit_second(); submit_first(); }
                        first_done.await();
                        second_done.await();
                        if (original)
                            for (size_t participant = 0; participant < 2; ++participant)
                                require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(native_owners[participant][pattern]),
                                    "Native fixed floor original terminal observation failed");
                        const auto wall_stop = std::chrono::steady_clock::now();
                        float event_ms = 0;
                        if (workload.measure_floor) {
                            require(first_backend->eventElapsedTimeMs(timing_start.event, timing_stop.event,
                                first.ordinal, &event_ms) && event_ms > 0, "Floor source event duration unavailable");
                            // Validation is not part of either timer. Retain the
                            // original readback owner and join its exact stream.
                            first_worker.submitAndWait([&] {
                                transfer.enqueueDeviceToMappedHost(*returned, 0, *readback, 0, buffer_bytes, first, first_stream);
                                require(first_backend->recordEvent(first_done.event, first.ordinal, first_stream), "Floor readback completion failed");
                            });
                            first_done.await();
                        }
                        const auto *actual = static_cast<const unsigned char *>(readback->mutableHostData());
                        for (size_t i = 0; i < buffer_bytes; ++i)
                            if (actual[i] != (i >= offset && i < offset + bytes ? patternByte(i, pattern) : 0xa5))
                                FAIL() << first.toString() << " -> " << second.toString() << " replay=" << replay
                                    << " bytes=" << bytes << " offset=" << offset << " first_bad_byte=" << i;
                        if (workload.measure_floor && replay >= workload.warmups)
                            std::cout << "NGRAM_CHANNEL_FLOOR_SAMPLE {\"schema\":1,\"source\":\"" << first.toString()
                                << "\",\"peer\":\"" << second.toString()
                                << "\",\"transport\":\"" << (fixed_collective ? (first.is_cuda() ? "nccl_fixed_p2p" : "rccl_fixed_p2p") : "captured_channel")
                                << "\",\"scope\":\"" << (fixed_collective
                                    ? "fixed_native_payload_round_trip_not_variable_count_ngram_qualification"
                                    : "complete_original_channel_round_trip_without_pool_lookup_or_decoding") << "\""
                                << ",\"source_graph_nodes\":" << (original ? native_definitions[0][pattern]->nodeCount() : first_graphs[pattern]->nodeCount())
                                << ",\"peer_graph_nodes\":" << (original ? native_definitions[1][pattern]->nodeCount() : second_graphs[pattern]->nodeCount())
                                << ",\"payload_bytes_per_direction\":" << bytes << ",\"logical_wire_payload_bytes\":" << 2 * bytes
                                << ",\"offset\":" << offset << ",\"sample\":" << replay - workload.warmups
                                << ",\"trajectory_index\":" << replay << ",\"pattern\":" << pattern
                                << ",\"source_event_us\":" << std::setprecision(10) << event_ms * 1000.0
                                << ",\"wall_submit_to_both_complete_us\":" << std::chrono::duration<double, std::micro>(wall_stop-wall_start).count()
                                << ",\"event_scope\":\"source_send_through_return_consumption_peer_preposted\",\"oracle\":\"exact_live_bytes_and_replaced_inactive_guards_outside_timing\"}\n";
                    }
                    failure_drain.completed = true;
                    std::cout << "CAPTURED_CHANNEL_ROUND_TRIP first=" << first.toString() << " second=" << second.toString()
                        << " bytes=" << bytes << " offset=" << offset << " replays=" << workload.replays << " verification_wall_ms="
                        << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() << '\n';
                }
        }
        fixed_collective.reset();
        for (const auto device : {DeviceId::cpu(), first, second})
            EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        if (original)
        {
            original->requirePayloadRetired();
            original->retire();
            original->requireExactRetired();
        }
        else
            for (const auto device : {first, second})
                EXPECT_EQ(authority->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
        if (workspace_peer)
            EXPECT_EQ(authority->claimedBytes(second, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }

    /** @brief Exercise captured metadata with the original workspace manager already retired. */
    void workspaceRoundTrip(DeviceId first, DeviceId second)
    {
        roundTrip(first, second, {.capacity = 257, .bytes = {4, 17, 257},
            .offsets = {1, 16}, .replays = 8, .patterns = 2,
            .peer_storage = PeerStorage::RetainedWorkspace});
    }
}

#ifdef HAVE_CUDA
TEST(CapturedTransferChannelDevices, CUDA_CUDA_RetainedRoundTrip) { roundTrip(DeviceId::cuda(0), DeviceId::cuda(1)); }
TEST(CapturedTransferChannelDevices, CUDA_CUDA_RetainedWorkspaceRoundTrip) { workspaceRoundTrip(DeviceId::cuda(0), DeviceId::cuda(1)); }
#endif
#ifdef HAVE_ROCM
TEST(CapturedTransferChannelDevices, ROCm_ROCm_RetainedRoundTrip) { roundTrip(DeviceId::rocm(0), DeviceId::rocm(1)); }
TEST(CapturedTransferChannelDevices, ROCm_ROCm_RetainedWorkspaceRoundTrip) { workspaceRoundTrip(DeviceId::rocm(0), DeviceId::rocm(1)); }
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
TEST(CapturedTransferChannelDevices, CUDA_ROCm_RetainedRoundTrip) { roundTrip(DeviceId::cuda(0), DeviceId::rocm(0)); }
TEST(CapturedTransferChannelDevices, ROCm_CUDA_RetainedRoundTrip) { roundTrip(DeviceId::rocm(0), DeviceId::cuda(0)); }
TEST(CapturedTransferChannelDevices, CUDA_ROCm_RetainedWorkspaceRoundTrip) { workspaceRoundTrip(DeviceId::cuda(0), DeviceId::rocm(0)); }
TEST(CapturedTransferChannelDevices, ROCm_CUDA_RetainedWorkspaceRoundTrip) { workspaceRoundTrip(DeviceId::rocm(0), DeviceId::cuda(0)); }

/** @test Exact ngram decode physical-channel floor with both original participant captures. */
TEST(CapturedTransferChannelFloor, CUDA_ROCm_Decode1440) {
    roundTrip(DeviceId::cuda(0), DeviceId::rocm(0),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.warmups=3});
}
/** @test Symmetric original channel floor; no cross-clock event arithmetic. */
TEST(CapturedTransferChannelFloor, ROCm_CUDA_Decode1440) {
    roundTrip(DeviceId::rocm(0), DeviceId::cuda(0),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.warmups=3});
}
/** @test Exact512token/16head/native90byte payload, without pooled metadata or decoding. */
TEST(CapturedTransferChannelFloor, CUDA_ROCm_M512737280) {
    roundTrip(DeviceId::cuda(0), DeviceId::rocm(0),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.warmups=3});
}
/** @test Symmetric complete M512 round trip with exact useful payload extent. */
TEST(CapturedTransferChannelFloor, ROCm_CUDA_M512737280) {
    roundTrip(DeviceId::rocm(0), DeviceId::cuda(0),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.warmups=3});
}

/**
 * @test Opt-in single-geometry probe for application-replay/native dispatch profilers.
 *
 * Not registered in CTest or preflight. One immutable source and one replay
 * isolate a 4099-byte aligned invocation. Native kernel replay is invalid for
 * a stateful peer protocol; use application replay or a non-replaying trace.
 */
TEST(CapturedTransferChannelProfiles, DISABLED_CUDA_ROCm_Aligned4099)
{
    roundTrip(DeviceId::cuda(0), DeviceId::rocm(0),
        {.capacity = 4099, .bytes = {4099}, .offsets = {16}, .replays = 1, .patterns = 1});
}

/** @test Symmetric ROCm-first isolated capture for per-dispatch attribution. */
TEST(CapturedTransferChannelProfiles, DISABLED_ROCm_CUDA_Aligned4099)
{
    roundTrip(DeviceId::rocm(0), DeviceId::cuda(0),
        {.capacity = 4099, .bytes = {4099}, .offsets = {16}, .replays = 1, .patterns = 1});
}
#endif

#ifdef HAVE_CUDA
TEST(CapturedTransferCancellation, CUDABeforeAcquire) { cancellationReplay(DeviceId::cuda(0), DeviceId::cuda(1), CancellationBoundary::BeforeAcquire); }
TEST(CapturedTransferCancellation, CUDAPublishedPeerAbsent) { cancellationReplay(DeviceId::cuda(0), DeviceId::cuda(1), CancellationBoundary::PublishedPeerAbsent); }
TEST(CapturedTransferCancellation, CUDAPublishedPeerLate) { cancellationReplay(DeviceId::cuda(0), DeviceId::cuda(1), CancellationBoundary::PublishedPeerLate); }
#endif
#ifdef HAVE_ROCM
TEST(CapturedTransferCancellation, ROCmBeforeAcquire) { cancellationReplay(DeviceId::rocm(0), DeviceId::rocm(1), CancellationBoundary::BeforeAcquire); }
TEST(CapturedTransferCancellation, ROCmPublishedPeerAbsent) { cancellationReplay(DeviceId::rocm(0), DeviceId::rocm(1), CancellationBoundary::PublishedPeerAbsent); }
TEST(CapturedTransferCancellation, ROCmPublishedPeerLate) { cancellationReplay(DeviceId::rocm(0), DeviceId::rocm(1), CancellationBoundary::PublishedPeerLate); }
#endif

#ifdef HAVE_CUDA
/** @brief Strict native peer-VRAM route reuses the original captured byte/extent/reset oracle. */
TEST(CapturedTransferChannelDevices, CUDA_CUDA_PeerPayloadRetainedRoundTrip)
{ roundTrip(DeviceId::cuda(0), DeviceId::cuda(1), {.measure_floor=true,.payload_storage=CapturedTransferPayloadStorage::PeerDevice,.warmups=5}); }
#endif
#ifdef HAVE_ROCM
/** @brief Symmetric native peer-VRAM route; absent peer access fails rather than selecting host payload. */
TEST(CapturedTransferChannelDevices, ROCm_ROCm_PeerPayloadRetainedRoundTrip)
{ roundTrip(DeviceId::rocm(0), DeviceId::rocm(1), {.measure_floor=true,.payload_storage=CapturedTransferPayloadStorage::PeerDevice,.warmups=5}); }
#endif

#ifdef HAVE_CUDA
/** @brief Explicit peer route cancellation preserves the original absent/late endpoint and sticky-failure oracle. */
TEST(CapturedTransferCancellation,CUDAPeerBeforeAcquire)
{ cancellationReplay(DeviceId::cuda(0),DeviceId::cuda(1),CancellationBoundary::BeforeAcquire,CapturedTransferPayloadStorage::PeerDevice); }
TEST(CapturedTransferCancellation,CUDAPeerPublishedPeerAbsent)
{ cancellationReplay(DeviceId::cuda(0),DeviceId::cuda(1),CancellationBoundary::PublishedPeerAbsent,CapturedTransferPayloadStorage::PeerDevice); }
TEST(CapturedTransferCancellation,CUDAPeerPublishedPeerLate)
{ cancellationReplay(DeviceId::cuda(0),DeviceId::cuda(1),CancellationBoundary::PublishedPeerLate,CapturedTransferPayloadStorage::PeerDevice); }
#endif
#ifdef HAVE_ROCM
/** @brief The same original cancellation law and terminal proof apply to native HIP peer payload. */
TEST(CapturedTransferCancellation,ROCmPeerBeforeAcquire)
{ cancellationReplay(DeviceId::rocm(0),DeviceId::rocm(1),CancellationBoundary::BeforeAcquire,CapturedTransferPayloadStorage::PeerDevice); }
TEST(CapturedTransferCancellation,ROCmPeerPublishedPeerAbsent)
{ cancellationReplay(DeviceId::rocm(0),DeviceId::rocm(1),CancellationBoundary::PublishedPeerAbsent,CapturedTransferPayloadStorage::PeerDevice); }
TEST(CapturedTransferCancellation,ROCmPeerPublishedPeerLate)
{ cancellationReplay(DeviceId::rocm(0),DeviceId::rocm(1),CancellationBoundary::PublishedPeerLate,CapturedTransferPayloadStorage::PeerDevice); }
#endif

namespace {
/**
 * @brief Exercise the actual cached hardware admission law without pretending unsupported peer payload ran.
 * @param first Original producer from the native fixture's MPI inventory.
 * @param second Original consumer from that same discovery owner.
 *
 * Complete executes the same strict native payload/extent/reset oracles. None
 * and Partial prove rejection before channel materialization; they never enter
 * the mapped path. Observations are terminal test properties, not scheduling state.
 */
void peerInventoryAdmission(DeviceId first,DeviceId second)
{
    const auto mpi=MPIContextFactory::global();
    const auto inventory=mpi->clusterInventory();
    ASSERT_TRUE(inventory);
    const auto& rank=inventory->ranks.at(mpi->rank());
    const auto coverage=localTPPeerAccessCoverage(rank,{first,second});
    ::testing::Test::RecordProperty("peer_access_coverage",peerAccessCoverageName(coverage));
    ::testing::Test::RecordProperty("peer_payload_executed",0);
    std::cout<<"CAPTURED_PEER_ADMISSION coverage="<<peerAccessCoverageName(coverage)
        <<" peer_payload_executed=0\n";
    if(coverage==PeerAccessCoverage::Complete){
        ASSERT_NO_FATAL_FAILURE(roundTrip(first,second,
            {.measure_floor=true,.payload_storage=CapturedTransferPayloadStorage::PeerDevice,.warmups=5}));
        ASSERT_NO_FATAL_FAILURE(counted_channel_test::verifyPeerPayloadReplay(first,second));
        ::testing::Test::RecordProperty("peer_payload_executed",1);
        std::cout<<"CAPTURED_PEER_ADMISSION coverage=complete peer_payload_executed=1\n";
        return;
    }
    ASSERT_TRUE(coverage==PeerAccessCoverage::None || coverage==PeerAccessCoverage::Partial);
    auto& producer=GPUDeviceContextPool::instance().getContext(first);
    auto& consumer=GPUDeviceContextPool::instance().getContext(second);
    void* producerStream=nullptr;
    void* consumerStream=nullptr;
    producer.submitAndWait([&]{producerStream=producer.getOrCreateAuxiliaryStream("captured_peer_admission");});
    consumer.submitAndWait([&]{consumerStream=consumer.getOrCreateAuxiliaryStream("captured_peer_admission");});
    ASSERT_NE(producerStream,nullptr);
    ASSERT_NE(consumerStream,nullptr);
    constexpr std::size_t capacity=4099;
    const auto geometry=CapturedTransferChannel::memoryFor(capacity,CapturedTransferPayloadStorage::PeerDevice);
    PhysicalMemoryPlanBuilder plan;
    plan.add(planningObservedResource(rank,DeviceId::cpu()),PhysicalMemoryOwner::ActivationTransportStaging,geometry.mapped_host_bytes);
    plan.add(planningObservedResource(rank,first),PhysicalMemoryOwner::ActivationTransportStaging,
        geometry.cursor_bytes_per_device+geometry.producer_payload_bytes);
    plan.add(planningObservedResource(rank,second),PhysicalMemoryOwner::ActivationTransportStaging,geometry.cursor_bytes_per_device);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()),rank.rank);
    try{
        (void)TransferEngine::instance().createPeerCapturedTransferChannel(*memory,DeviceId::cpu(),
            first,producerStream,second,consumerStream,capacity,rank);
        FAIL()<<"Unavailable peer coverage entered channel materialization";
    }catch(const std::invalid_argument& error){
        EXPECT_NE(std::string(error.what()).find("cached complete bidirectional coverage"),std::string::npos)<<error.what();
    }
    for(const auto device:{DeviceId::cpu(),first,second})
        EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
} // namespace
#ifdef HAVE_CUDA
TEST(CapturedTransferPeerAdmission,CUDAOriginalInventory)
{ peerInventoryAdmission(DeviceId::cuda(0),DeviceId::cuda(1)); }
#endif
#ifdef HAVE_ROCM
TEST(CapturedTransferPeerAdmission,ROCmOriginalInventory)
{ peerInventoryAdmission(DeviceId::rocm(0),DeviceId::rocm(1)); }
#endif

#include "PooledReplicatedReaderNativeReplay.inc"

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeForward_Header64) {
    roundTrip(DeviceId::cuda(0), DeviceId::cuda(1),
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeForward_Decode1440) {
    roundTrip(DeviceId::cuda(0), DeviceId::cuda(1),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeForward_M512737280) {
    roundTrip(DeviceId::cuda(0), DeviceId::cuda(1),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeReverse_Header64) {
    roundTrip(DeviceId::cuda(1), DeviceId::cuda(0),
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeReverse_Decode1440) {
    roundTrip(DeviceId::cuda(1), DeviceId::cuda(0),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeReverse_M512737280) {
    roundTrip(DeviceId::cuda(1), DeviceId::cuda(0),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeForward_Header64) {
    roundTrip(DeviceId::rocm(0), DeviceId::rocm(1),
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeForward_Decode1440) {
    roundTrip(DeviceId::rocm(0), DeviceId::rocm(1),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeForward_M512737280) {
    roundTrip(DeviceId::rocm(0), DeviceId::rocm(1),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeReverse_Header64) {
    roundTrip(DeviceId::rocm(1), DeviceId::rocm(0),
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeReverse_Decode1440) {
    roundTrip(DeviceId::rocm(1), DeviceId::rocm(0),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeReverse_M512737280) {
    roundTrip(DeviceId::rocm(1), DeviceId::rocm(0),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedForward_Header64) {
    roundTrip(DeviceId::cuda(0), DeviceId::cuda(1),
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedForward_Decode1440) {
    roundTrip(DeviceId::cuda(0), DeviceId::cuda(1),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedForward_M512737280) {
    roundTrip(DeviceId::cuda(0), DeviceId::cuda(1),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedReverse_Header64) {
    roundTrip(DeviceId::cuda(1), DeviceId::cuda(0),
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedReverse_Decode1440) {
    roundTrip(DeviceId::cuda(1), DeviceId::cuda(0),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedReverse_M512737280) {
    roundTrip(DeviceId::cuda(1), DeviceId::cuda(0),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedForward_Header64) {
    roundTrip(DeviceId::rocm(0), DeviceId::rocm(1),
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedForward_Decode1440) {
    roundTrip(DeviceId::rocm(0), DeviceId::rocm(1),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedForward_M512737280) {
    roundTrip(DeviceId::rocm(0), DeviceId::rocm(1),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedReverse_Header64) {
    roundTrip(DeviceId::rocm(1), DeviceId::rocm(0),
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedReverse_Decode1440) {
    roundTrip(DeviceId::rocm(1), DeviceId::rocm(0),
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedReverse_M512737280) {
    roundTrip(DeviceId::rocm(1), DeviceId::rocm(0),
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});
}
