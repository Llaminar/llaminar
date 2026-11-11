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
#include "execution/local_execution/orchestrators/PipelineActivationFanout.h"
#include "planning/PipelineTransferMemory.h"
#include "CapturedChannelGraphProof.h"
#include "CountedChannelTestKernels.h"
#include "backends/GPUGraphMemoryContract.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "planning/PlanningPublication.h"
#include "planning/RankHardwareOwnership.h"
#include "tensors/TensorClasses.h"
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
#include "backends/NativeContextOpaqueContract.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "planning/PlanningObservedResource.h"
#include "utils/MPIContext.h"
#include "fixtures/backends/OriginalRankLocalNativeReplay.h"
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

    using llaminar2::test::originalRankLocalReplay;

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
    void cancellationReplay(DeviceId first, DeviceId second, CancellationBoundary boundary)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &rank = inventory->ranks.at(mpi->rank());
        for (const auto device : {first, second}) {
            bool owned = false;
            for (const auto &gpu : rank.gpus)
                if (gpu.type == (device.is_cuda() ? DeviceType::CUDA : DeviceType::ROCm) && gpu.local_device_id == device.ordinal && rankOwnsGPUDevice(rank, gpu))
                    owned = true;
            require(owned, "Captured local transfer endpoint is not owned by this original NUMA-local rank");
        }
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
        const auto geometry = CapturedTransferChannel::memoryFor(planned_capacity);
        PhysicalMemoryPlanBuilder builder;
        builder.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            geometry.mapped_host_bytes + TransferEngine::mappedHostRegionAllocationBytes(planned_capacity) +
            TransferEngine::mappedHostRegionAllocationBytes(planned_snapshot_bytes));
        builder.add(planningObservedResource(rank, first), PhysicalMemoryOwner::ActivationTransportStaging,
            geometry.cursor_bytes_per_device + planned_capacity);
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
            auto channel=transfer.createCapturedTransferChannel(*authority,DeviceId::cpu(),
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

            observed_wire=CapturedTransferChannelReplayProbe::payload(*channel);
            std::fill(observed_wire.begin(),observed_wire.end(),std::byte{0xa5});

            producer_worker.submitAndWait([&] {
                transfer.enqueueMappedHostToDevice(*input,0,*source,0,capacity,first,producer_stream);

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
        for (const auto device : {first, second}) {
            bool owned = false;
            for (const auto &gpu : rank.gpus)
                if (gpu.type == (device.is_cuda() ? DeviceType::CUDA : DeviceType::ROCm) && gpu.local_device_id == device.ordinal && rankOwnsGPUDevice(rank, gpu))
                    owned = true;
            require(owned, "Captured local transfer endpoint is not owned by this original NUMA-local rank");
        }
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
            original = test::PreparedNativeLookupFixture::prepare(devices, mpi->rank());
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
            CapturedTransferChannel::memoryFor(capacity);
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
                2 * geometry.cursor_bytes_per_device + (device == first ? 3 : workspace_peer ? 0 : 1) * buffer_bytes);
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
#ifdef HAVE_CUDA
        if(original && first.is_cuda()) {
            // Verify birth capacity after final admission and BEFORE any NCCL
            // communicator exists. This
            // must fail under the original unprovisioned constructor, rather
            // than depending on a library warmup or a late graph byte claim.
            for(const auto device:{first,second}) {
                auto& worker=GPUDeviceContextPool::instance().getContext(device);
                auto* backend=dynamic_cast<CUDABackend*>(getBackendFor(device));
                ASSERT_NE(backend,nullptr);
                worker.submitAndWait([&] {
                    backend->prepareNativeExecutionContextStorage(CUDANativeProgramStack::capacityBytes,
                        device.ordinal,worker.defaultStream(),original->permit(device).memory());
                    EXPECT_THROW(backend->prepareNativeExecutionContextStorage(CUDANativeProgramStack::capacityBytes+1u,
                        device.ordinal,worker.defaultStream(),original->permit(device).memory()),std::runtime_error);
                    // Original late-growth prohibition remains unchanged.
                    EXPECT_THROW(GPUDeviceContextPool::instance().requireOrdinaryContextStackGrowth(
                        device,original->permit(device).memory()),std::runtime_error);
                });
            }
        }
#endif
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
            const std::array diagnostic_devices{first, second};
            auto readback = transfer.allocateMappedHostRegion(buffer_bytes, diagnostic_devices);
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
                            {
                                const unsigned observed = actual[i];
                                // All original graphs are terminal. Reuse the admitted
                                // readback for one failing byte at each physical hop;
                                // this is failure evidence, never live protocol state.

                                FAIL() << first.toString() << " -> " << second.toString() << " replay=" << replay
                                    << " bytes=" << bytes << " offset=" << offset << " first_bad_byte=" << i
                                    << " expected=" << unsigned(i >= offset && i < offset + bytes ? patternByte(i, pattern) : 0xa5)
                                    << " actual=" << observed;
                            }
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
TEST(CapturedTransferChannelDevices, CUDA_CUDA_RetainedRoundTrip) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) { roundTrip(cohort[0], cohort[1]);
    });
}
TEST(CapturedTransferChannelDevices, CUDA_CUDA_RetainedWorkspaceRoundTrip) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) { workspaceRoundTrip(cohort[0], cohort[1]);
    });
}
#endif
#ifdef HAVE_ROCM
TEST(CapturedTransferChannelDevices, ROCm_ROCm_RetainedRoundTrip) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) { roundTrip(cohort[0], cohort[1]);
    });
}
TEST(CapturedTransferChannelDevices, ROCm_ROCm_RetainedWorkspaceRoundTrip) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) { workspaceRoundTrip(cohort[0], cohort[1]);
    });
}
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
TEST(CapturedTransferChannelDevices, CUDA_ROCm_RetainedRoundTrip)
{
    const std::array families{test::NativeReplayFamily{DeviceType::CUDA, 1}, test::NativeReplayFamily{DeviceType::ROCm, 1}};
    originalRankLocalReplay(std::span<const test::NativeReplayFamily>(families), [&](const auto &cohort) {
        roundTrip(cohort[0], cohort[1]);
    });
}
TEST(CapturedTransferChannelDevices, ROCm_CUDA_RetainedRoundTrip)
{
    const std::array families{test::NativeReplayFamily{DeviceType::ROCm, 1}, test::NativeReplayFamily{DeviceType::CUDA, 1}};
    originalRankLocalReplay(std::span<const test::NativeReplayFamily>(families), [&](const auto &cohort) {
        roundTrip(cohort[0], cohort[1]);
    });
}
TEST(CapturedTransferChannelDevices, CUDA_ROCm_RetainedWorkspaceRoundTrip)
{
    const std::array families{test::NativeReplayFamily{DeviceType::CUDA, 1}, test::NativeReplayFamily{DeviceType::ROCm, 1}};
    originalRankLocalReplay(std::span<const test::NativeReplayFamily>(families), [&](const auto &cohort) {
        workspaceRoundTrip(cohort[0], cohort[1]);
    });
}
TEST(CapturedTransferChannelDevices, ROCm_CUDA_RetainedWorkspaceRoundTrip)
{
    const std::array families{test::NativeReplayFamily{DeviceType::ROCm, 1}, test::NativeReplayFamily{DeviceType::CUDA, 1}};
    originalRankLocalReplay(std::span<const test::NativeReplayFamily>(families), [&](const auto &cohort) {
        workspaceRoundTrip(cohort[0], cohort[1]);
    });
}

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
TEST(CapturedTransferCancellation, CUDABeforeAcquire) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) { cancellationReplay(cohort[0], cohort[1], CancellationBoundary::BeforeAcquire);
    });
}
TEST(CapturedTransferCancellation, CUDAPublishedPeerAbsent) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) { cancellationReplay(cohort[0], cohort[1], CancellationBoundary::PublishedPeerAbsent);
    });
}
TEST(CapturedTransferCancellation, CUDAPublishedPeerLate) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) { cancellationReplay(cohort[0], cohort[1], CancellationBoundary::PublishedPeerLate);
    });
}
#endif
#ifdef HAVE_ROCM
TEST(CapturedTransferCancellation, ROCmBeforeAcquire) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) { cancellationReplay(cohort[0], cohort[1], CancellationBoundary::BeforeAcquire);
    });
}
TEST(CapturedTransferCancellation, ROCmPublishedPeerAbsent) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) { cancellationReplay(cohort[0], cohort[1], CancellationBoundary::PublishedPeerAbsent);
    });
}
TEST(CapturedTransferCancellation, ROCmPublishedPeerLate) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) { cancellationReplay(cohort[0], cohort[1], CancellationBoundary::PublishedPeerLate);
    });
}
#endif

#ifdef HAVE_ROCM
/**
 * @test Public native RCCL copies the failing peer case's exact byte geometry.
 *
 * Keep the original full source/return allocation capacity, alternating immutable
 * patterns, unaligned offset and terminal byte/guard oracle. The archived causal red remains independent evidence after custom transport
 * retirement; this native control claims only its own exact selected route.
 */
TEST(CapturedTransferNativePeerVisibility, ROCmOriginalInventory257Offset1)
{
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
        roundTrip(cohort[0], cohort[1],
            {.capacity = 1024 * 1024 + 3, .bytes = {257}, .offsets = {1},
             .replays = 64, .patterns = 2, .peer_storage = PeerStorage::PrivateScratch,
             .measure_floor = true, .fixed_native_collective = true, .warmups = 5});
    });
}
#endif

/** @test Original source birth plus two native patterns replay without late stack admission or warmup. */
TEST(CapturedTransferNativeProgramStack, CUDAOriginalSourceReplay) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[0],cohort[1],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=3,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=0});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeForward_Header64) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeForward_Decode1440) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeForward_M512737280) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeReverse_Header64) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeReverse_Decode1440) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, CUDA_NativeReverse_M512737280) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeForward_Header64) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeForward_Decode1440) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeForward_M512737280) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeReverse_Header64) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeReverse_Decode1440) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Fixed native library floor, not variable-count ngram qualification. */
TEST(CapturedTransferChannelFloor, ROCm_NativeReverse_M512737280) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=true,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedForward_Header64) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedForward_Decode1440) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedForward_M512737280) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedReverse_Header64) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedReverse_Decode1440) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, CUDA_MappedReverse_M512737280) {
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedForward_Header64) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedForward_Decode1440) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedForward_M512737280) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[0], cohort[1],
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedReverse_Header64) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=64,.bytes={64},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedReverse_Decode1440) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=1440,.bytes={1440},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

/** @test Diagnostic same-pair mapped comparator; never selects production ngram transport. */
TEST(CapturedTransferChannelFloor, ROCm_MappedReverse_M512737280) {
    originalRankLocalReplay(DeviceType::ROCm, 2, [&](const auto &cohort) {
    roundTrip(cohort[1], cohort[0],
        {.capacity=737280,.bytes={737280},.offsets={16},.replays=25,.patterns=2,
         .peer_storage=PeerStorage::PrivateScratch,.measure_floor=true,.fixed_native_collective=false,.warmups=3});

    });
}

#include "NativeCompletedReaderReplay.inc"

namespace
{
/** @brief Select distinct physical GPUs only from their original rank-local observations.
 * @param inventory Original context-owned complete discovery publication.
 * @param first Required first GPU family.
 * @param second Required second GPU family.
 * @return Actual original rank/device endpoints; no rank/NUMA/ordinal arithmetic establishes ownership.
 * @throws std::runtime_error When this exact native placement is unavailable.
 */
std::array<CapturedTransferRankGPU, 2> rankReplayEndpoints(
    const ClusterInventory& inventory, DeviceType first, DeviceType second)
{
    for (const auto& source : inventory.ranks)
        for (const auto& source_gpu : source.gpus)
            if (source_gpu.type == first && rankOwnsGPUDevice(source, source_gpu))
                for (const auto& consumer : inventory.ranks)
                    if (source.rank != consumer.rank &&
                        inventory.connectionBetweenRanks(source.rank, consumer.rank).locality() == RankConnectionLocality::SameNode)
                        for (const auto& consumer_gpu : consumer.gpus)
                            if (consumer_gpu.type == second && rankOwnsGPUDevice(consumer, consumer_gpu) &&
                                !source_gpu.uuid.empty() && !consumer_gpu.uuid.empty() && source_gpu.uuid != consumer_gpu.uuid)
                                return {{{source.rank, {first, source_gpu.local_device_id}},
                                         {consumer.rank, {second, consumer_gpu.local_device_id}}}};
    std::string detail="Required distinct-rank, NUMA-local physical GPU pair is absent from original inventory";
    for(const auto& rank:inventory.ranks) {
        detail+="; original rank="+std::to_string(rank.rank)+" CPU_NUMA="+std::to_string(rank.cpu.numa_node)+" local endpoints=";
        bool present=false;
        for(const auto& gpu:rank.gpus)
            if((gpu.type==first || gpu.type==second) && rankOwnsGPUDevice(rank,gpu)) {
                detail+=std::string(deviceTypeToString(gpu.type))+":"+std::to_string(gpu.local_device_id)+
                    "(NUMA="+std::to_string(gpu.numa_node)+",UUID="+gpu.uuid+") ";
                present=true;
            }
        if(!present)detail+="none";
    }
    throw std::runtime_error(detail);
}

/** @brief Replay recorded row inputs through real rank-local captured byte kernels.
 * @param first Required original source family for the first direction.
 * @param second Required original source family for the reverse direction.
 *
 * Recorded inputs are restored only at explicit test request admission. The
 * captured GPU kernel derives live bytes from the retained INT32 row input;
 * no device count is read by the host to schedule execution. This is native
 * transport/reset proof, not model generation or prefill-materializer proof.
 */
void rankCapturedRecordedRowsReplay(DeviceType first, DeviceType second)
{
    const auto mpi = MPIContextFactory::global();
    require(mpi && mpi->world_size() >= 2, "Rank captured replay requires original multi-rank MPI ownership");
    const auto inventory = mpi->clusterInventory();
    require(bool(inventory), "Rank captured replay requires original discovery publication");
    const auto endpoints = rankReplayEndpoints(*inventory, first, second);
    auto session = CapturedTransferRankSession::create(mpi);
    constexpr std::size_t row_bytes = 17 * sizeof(float);
    constexpr std::size_t capacity = 65 * row_bytes;
    constexpr std::size_t storage_bytes = capacity + 2 * row_bytes;
    constexpr std::size_t row_words = 4;
    constexpr std::size_t row_offset = sizeof(std::int32_t);
    const auto geometry = CapturedTransferChannel::memoryFor(capacity);
    const auto snapshot_bytes = TransferEngine::mappedHostRegionAllocationBytes(storage_bytes + row_words * sizeof(std::int32_t));
    const auto graph_bytes = GPUGraphMemoryContract::reservationBytesPerExecutable(
        endpoints[0].rank == mpi->rank() ? endpoints[0].device : endpoints[1].device,
        GPUGraphExecutableClass::BoundedFlatHelper);
    const std::array plans{
        CapturedTransferRankPlan::fromInventory(*inventory, {session->originalGeneration(), 701}, endpoints[0], endpoints[1], capacity),
        CapturedTransferRankPlan::fromInventory(*inventory, {session->originalGeneration(), 702}, endpoints[1], endpoints[0], capacity)};
    const auto role = plans[0].localRole(mpi->rank());
    const auto local = role ? plans[0].endpoint(*role).device : DeviceId::invalid();
    std::unique_ptr<llaminar2::test::PreparedNativeLookupFixture> original;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    IWorkerGPUContext* worker = nullptr;
    IBackend* backend = nullptr;
    void* stream = nullptr;
    const CPUTensorStorageDeclaration rows_declaration(1, row_words, BufferTensorType::INT32,
        PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
    // Original all-rank failure consensus surrounds actual rank-local source
    // construction. Idle ranks do not create a source, PMA, GPU cursor or worker.
    exchangePlanningSamples(mpi, [&](int count) {
        return std::vector<std::vector<std::uint8_t>>(count, plans[0].serialize());
    }, [&](auto wire) {
        require(std::vector<std::uint8_t>(wire.begin(), wire.end()) == plans[0].serialize(), "Native replay role declarations disagree");
        if (role)
        {
            const std::array devices{local};
            original = llaminar2::test::PreparedNativeLookupFixture::prepare(devices, mpi->rank());
            PhysicalMemoryPlanBuilder builder;
            for (const auto& plan : plans)
                plan.contributeMemory(*plan.localRole(mpi->rank()), original->resource(local),
                    plan.hostRank() == mpi->rank() ? std::optional{original->resource(DeviceId::cpu())} : std::nullopt, builder);
            builder.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace, rows_declaration.physicalBytes());
            builder.add(original->resource(local), PhysicalMemoryOwner::ExecutionWorkspace, row_words * sizeof(std::int32_t));
            builder.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging, snapshot_bytes);
            builder.add(original->resource(local), PhysicalMemoryOwner::ActivationTransportStaging, storage_bytes + 2 * sizeof(std::uint64_t));
            builder.add(original->resource(local), PhysicalMemoryOwner::NativeGraphExecutable, graph_bytes);
            original->admit(builder.build());
            memory = original->memory();
            worker = original->permit(local).worker().get();
            backend = getBackendFor(local);
            require(worker && backend && memory->worldRank() == mpi->rank(), "Native replay lost its original local source");
            worker->submitAndWait([&] {
                stream = worker->getOrCreateAuxiliaryStream("rank_captured_recorded_rows");
#ifdef HAVE_CUDA
                if (local.is_cuda()) require(counted_channel_test::prepareKernelsCUDA(), "Prepare original CUDA byte writer");
#endif
#ifdef HAVE_ROCM
                if (local.is_rocm()) require(counted_channel_test::prepareKernelsROCm(), "Prepare original HIP byte writer");
#endif
            });
        }
        return std::vector<std::uint8_t>{1};
    }, [](auto receipts) {
        for (const auto& receipt : receipts)
            require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Native rank source preparation omitted a participant");
    });
    auto& transfer = TransferEngine::instance();
    std::array<std::shared_ptr<CapturedTransferChannel>, 2> channels;
    for (std::size_t i = 0; i < plans.size(); ++i)
    {
        channels[i] = transfer.createRankCapturedTransferChannel(plans[i], session, memory, stream);
        require(bool(channels[i]) == bool(role), "Inactive rank acquired a native channel owner");
        if (!role) continue;
        const auto own_role = *plans[i].localRole(mpi->rank());
        const auto peer_role = own_role == CapturedTransferEndpoint::Producer
            ? CapturedTransferEndpoint::Consumer : CapturedTransferEndpoint::Producer;
        require(channels[i]->endpointDevice(own_role) == local, "Rank channel changed its local GPU");
        require(!CapturedTransferChannelReplayProbe::terminalCursor(*channels[i], peer_role), "Rank channel created a foreign GPU cursor");
        const auto peer = plans[i].endpoint(peer_role).device;
        if (peer != local)
        {
            bool absent = false;
            try { (void)GPUDeviceContextPool::instance().retainExistingContextProvenance(peer); }
            catch (const std::logic_error&) { absent = true; }
            require(absent, "Rank channel created a hidden foreign worker");
        }
        const auto& control = CapturedTransferChannelReplayProbe::terminalControl(*channels[i]);
        require(control.producer.epoch == 0 && control.consumer.epoch == 0 && control.identity.nonce == channels[i]->identity().nonce &&
            control.identity.capacity == channels[i]->identity().capacity,
            "Rank channel attachment changed its source-initialized control");
    }
    // The original cohort rejects release while even a local channel/graph
    // borrow remains. Rejection must not drop the actual source host claim.
    bool live_rejected = false;
    try { session->retireAfterOriginalGraphRelease(); }
    catch (const std::exception&) { live_rejected = true; }
    require(live_rejected && !session->retired(), "Rank source backing retired before original readers");
    for (std::size_t direction = 0; direction < plans.size(); ++direction)
    {
        if (!role)
        {
            // Idle original ranks join exactly the same terminal boundaries;
            // they never acquire a GPU alias, worker or physical-memory claim.
            for (const auto ignored : {65, 3, 1, 0, 65})
            {
                (void)ignored;
                mpi->barrier();
            }
            continue;
        }
        const auto own_role = *plans[direction].localRole(mpi->rank());
        const bool sends = own_role == CapturedTransferEndpoint::Producer;
        auto payload_claim = memory->claimNewAllocation(local, PhysicalMemoryOwner::ActivationTransportStaging,
            storage_bytes + 2 * sizeof(std::uint64_t));
        auto snapshot_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, snapshot_bytes);
        auto payload = transfer.allocateDeviceTransferBuffer(storage_bytes, local);
        auto sequence = transfer.allocateDeviceTransferBuffer(sizeof(std::uint64_t), local);
        auto ignored_extent = transfer.allocateDeviceTransferBuffer(sizeof(std::uint64_t), local);
        std::shared_ptr<TensorBase> rows(rows_declaration.materialize(memory));
        const std::array devices{local};
        auto snapshot = transfer.allocateMappedHostRegion(storage_bytes + row_words * sizeof(std::int32_t), devices);
        worker->submitAndWait([&] {
            require(backend->memset(sequence->mutableDeviceData(), 0, sizeof(std::uint64_t), local.ordinal, stream), "Initial native sequence reset failed");
            auto* recorded = static_cast<std::int32_t*>(rows->raw_mutable_data());
            recorded[0] = 0x1357; recorded[1] = 65; recorded[2] = 0x2468; recorded[3] = 0x369;
            TransferEngine::publishHostWrite(rows.get());
            TransferEngine::prepareDeviceInput(rows.get(), local, stream);
            TransferEngine::requireDeviceInput(rows.get(), local, stream);
        });
        const auto binding = transfer.bindDeviceRowCountedTransfer(transfer.bindCapturedTransfer(channels[direction],
            own_role, {991, capacity}, payload, row_bytes), rows, row_offset, row_bytes);
        const auto frozen_identity = binding.captureIdentity();
        auto graph_claim = memory->reserveNewAllocations(local, PhysicalMemoryOwner::NativeGraphExecutable, graph_bytes);
        // Original graph factory ownership is host setup; only recording and
        // execution enter the exact local GPU worker.
        auto graph = worker->createGraphCapture(stream);
        require(bool(graph), "Missing original rank-local graph factory");
        Terminal terminal{backend, local, backend->createEvent(local.ordinal)};
        require(terminal.event != nullptr, "Missing original rank-local terminal event");
        NativeReplayFailureDrain fatal{[mpi] {
            MPI_Abort(mpi->communicator(), 2);
            std::terminate();
        }};
        worker->submitAndWait([&] {
            ScopedBackendGraphCapture capture(*worker, *graph, "rank-local recorded activation rows");
            require(capture.begin(), "Rank channel capture begin failed");
            require(backend->memset(payload->mutableDeviceData(), 0xa5, storage_bytes, local.ordinal, stream), "Captured guard reset failed");
            if (sends)
            {
#ifdef HAVE_CUDA
                if (local.is_cuda()) require(counted_channel_test::prepareCUDA(payload->mutableDeviceData(row_bytes),
                    static_cast<std::uint64_t*>(ignored_extent->mutableDeviceData()), static_cast<const std::uint64_t*>(sequence->deviceData()), capacity, stream), "Captured CUDA payload writer failed");
#endif
#ifdef HAVE_ROCM
                if (local.is_rocm()) require(counted_channel_test::prepareROCm(payload->mutableDeviceData(row_bytes),
                    static_cast<std::uint64_t*>(ignored_extent->mutableDeviceData()), static_cast<const std::uint64_t*>(sequence->deviceData()), capacity, stream), "Captured HIP payload writer failed");
#endif
            }
            transfer.enqueueCapturedTransfer(binding, stream);
            if (sends)
            {
#ifdef HAVE_CUDA
                if (local.is_cuda()) require(counted_channel_test::advanceCUDA(static_cast<std::uint64_t*>(sequence->mutableDeviceData()), stream), "Captured CUDA sequence advance failed");
#endif
#ifdef HAVE_ROCM
                if (local.is_rocm()) require(counted_channel_test::advanceROCm(static_cast<std::uint64_t*>(sequence->mutableDeviceData()), stream), "Captured HIP sequence advance failed");
#endif
            }
            capture.finish();
            counted_channel_test::verifyCapturedChannelGraph(*graph, capacity, 1);
            std::string error;
            require(graph->validateExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper, &error), error.c_str());
            require(graph->prepareRuntimeContextStorage(memory) && graph->instantiate(), "Rank-local native graph instantiation failed");
        });
        std::size_t replay = 0;
        for (const std::int32_t live_rows : {65, 3, 1, 0, 65})
        {
            // Recorded initial inputs at an explicit test request boundary;
            // no GPU state is downloaded to decide what transaction to launch.
            require(rows->hostValid(), "Read-only expected row input was falsely published as device-written");
            auto* recorded = static_cast<std::int32_t*>(rows->raw_mutable_data());
            recorded[1] = live_rows;
            TransferEngine::publishHostWrite(rows.get());
            const CapturedTransferInvocation invocation{session->originalGeneration(), replay + 1};
            transfer.beginCapturedTransferInvocation(channels[direction], invocation);
            worker->submitAndWait([&] {
                TransferEngine::prepareDeviceInput(rows.get(), local, stream);
                TransferEngine::requireDeviceInput(rows.get(), local, stream);
                require(graph->launch(), "Rank retained replay launch failed");
                transfer.enqueueDeviceToMappedHost(*payload, 0, *snapshot, 0, storage_bytes, local, stream);
                require(backend->deviceToHostOnStream(static_cast<std::byte*>(snapshot->mutableHostData()) + storage_bytes,
                    rows->gpu_data_ptr(), row_words * sizeof(std::int32_t), local.ordinal, stream), "Observe terminal native row input");
                require(backend->recordEvent(terminal.event, local.ordinal, stream), "Rank native terminal publication failed");
            });
            terminal.await(std::chrono::seconds(30));
            mpi->barrier(); // Explicit test terminal boundary, never production hot execution.
            const auto& control = CapturedTransferChannelReplayProbe::terminalControl(*channels[direction]);
            const auto live_bytes = std::size_t(live_rows) * row_bytes;
            require(control.producer.message.bytes == live_bytes && control.consumer.message.bytes == live_bytes &&
                control.producer.epoch == replay + 1 && control.consumer.epoch == replay + 1,
                "Native shared header communicated a capacity/stale extent");
            const auto* actual = static_cast<const std::uint8_t*>(snapshot->mutableHostData());
            for (std::size_t index = 0; index < storage_bytes; ++index)
            {
                const bool active = index >= row_bytes && index - row_bytes < (sends ? capacity : live_bytes);
                const auto expected = active ? counted_channel_test::byte(replay, index - row_bytes) : std::uint8_t{0xa5};
                require(actual[index] == expected, "Native rank payload byte or poison-tail mismatch");
            }
            std::array<std::int32_t, row_words> actual_rows{};
            std::memcpy(actual_rows.data(), actual + storage_bytes, sizeof(actual_rows));
            require(actual_rows == std::array<std::int32_t, row_words>{0x1357, live_rows, 0x2468, 0x369},
                "Native row-count input or adjacent INT32 guard was overwritten");
            require(binding.captureIdentity() == frozen_identity, "Rank replay changed its capture identity");
            require(transfer.finishCapturedTransferInvocation(channels[direction], invocation) ==
                CapturedTransferInvocationRetirement::Completed, "Rank request retirement lost original admission");
            ++replay;
        }
        worker->submitAndWait([&] { graph.reset(); });
        fatal.completed = true;
    }
    mpi->barrier();
    channels = {};
    if (role)
        require(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation) == geometry.mapped_host_bytes,
            "Original host claim retired before peer graph/alias consensus");
    session->retireAfterOriginalGraphRelease();
    require(session->retired(), "Original native cohort did not retire shared mappings");
    if (original)
    {
        original->requirePayloadRetired();
        original->retire();
        original->requireExactRetired();
    }
}
} // namespace

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
/** @test Real node-local mixed-rank payload and request-data reset; no real-model qualification. */
TEST(CapturedTransferRankNative, MixedCUDA_ROCmRecordedRowsAndRetirement)
{
    rankCapturedRecordedRowsReplay(DeviceType::CUDA, DeviceType::ROCm);
}
#endif
#ifdef HAVE_ROCM
/** @test Real distinct-rank ROCm physical pair, including both endpoint directions. */
TEST(CapturedTransferRankNative, ROCmRecordedRowsAndRetirement)
{
    rankCapturedRecordedRowsReplay(DeviceType::ROCm, DeviceType::ROCm);
}
#endif

namespace
{
/** @brief Real local TP2 fanout after original captured channel receipt, inside the original native owner.
 * @param family Exact backend family. Original NUMA ownership chooses a real TP2 cohort; ranks never impersonate devices.
 * Recorded request inputs and independent per-member poison prove compactness
 * even when the source and one native reader share the same physical GPU.
 */
void pipelineActivationCompletedFanoutReplay(DeviceType family)
{
    const auto mpi = MPIContextFactory::global();
    originalRankLocalReplay(family, 2, [&](const auto &devices) {
        const int owner_rank = mpi->rank();
        constexpr size_t row_bytes = 17 * sizeof(float), capacity = 65 * row_bytes;
        constexpr size_t storage_bytes = 67 * row_bytes, count_bytes = 4 * sizeof(int32_t);
        constexpr size_t host_snapshot_bytes = 2 * (storage_bytes + count_bytes);
        constexpr uint64_t message_key = 0x4143540000000041ULL;
        const CPUTensorStorageDeclaration hidden_declaration(67, 17, BufferTensorType::FP32,
            PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        const CPUTensorStorageDeclaration rows_declaration(1, 4, BufferTensorType::INT32,
            PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        auto original = test::PreparedNativeLookupFixture::prepare(devices, owner_rank);
        std::vector<GlobalDeviceAddress> addresses;
        for (const auto device : devices)
            addresses.push_back(device.is_cuda() ? GlobalDeviceAddress::cuda(device.ordinal) : GlobalDeviceAddress::rocm(device.ordinal));
        auto collective = createLocalTPContext(addresses, {}, family == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        require(bool(collective), "Original fanout native context is absent");
        const auto native_bom = CollectiveMemoryEstimator::localTP(1, capacity,
            family == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        const auto channel_bom = CapturedTransferChannel::memoryFor(capacity);
        PhysicalMemoryPlanBuilder plan;
        plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            channel_bom.mapped_host_bytes + 2 * PipelineTransferMemory::fanoutHostBytes() +
                TransferEngine::mappedHostRegionAllocationBytes(host_snapshot_bytes));
        plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace,
            2 * (hidden_declaration.physicalBytes() + rows_declaration.physicalBytes()));
        for (const auto device : devices)
        {
            plan.add(original->resource(device), PhysicalMemoryOwner::ActivationTransportStaging,
                channel_bom.cursor_bytes_per_device + PipelineTransferMemory::fanoutDeviceBytes() +
                    (device == devices[1] ? capacity + 2 * sizeof(uint64_t) : 0));
            plan.add(original->resource(device), PhysicalMemoryOwner::ExecutionWorkspace, storage_bytes + count_bytes);
            plan.add(original->resource(device), PhysicalMemoryOwner::LocalCollective, native_bom.perDeviceBytes());
            const auto graph_bom = RetainedParticipantGraphExecutable::generalBOM(
                original->resource(DeviceId::cpu()), original->resource(device), 1);
            for (const auto &row : graph_bom.resources()) plan.add(row);
        }
        original->admit(plan.build());
        auto memory = original->memory();
        require(collective->reserveCollectiveResources(native_bom.backend_payload_capacity_bytes,
            native_bom.fp16_scratch_elements, memory), "Fanout original collective resources were not admitted");
        {
            auto &transfer = TransferEngine::instance();
            std::array<std::shared_ptr<IWorkerGPUContext>, 2> workers{original->permit(devices[0]).worker(), original->permit(devices[1]).worker()};
            std::array<IBackend *, 2> backends{getBackendFor(devices[0]), getBackendFor(devices[1])};
            std::array<void *, 2> streams{workers[0]->defaultStream(), workers[1]->defaultStream()};
            std::array<std::shared_ptr<TensorBase>, 2> hidden, rows;
            for (size_t i = 0; i < 2; ++i)
            {
                hidden[i] = hidden_declaration.materialize(memory);
                rows[i] = rows_declaration.materialize(memory);
                workers[i]->submitAndWait([&] {
                    auto *input = static_cast<int32_t *>(rows[i]->raw_mutable_data());
                    input[0] = 0x1357; input[1] = 65; input[2] = 0x2468; input[3] = 0x369;
                    TransferEngine::publishHostWrite(rows[i].get());
                    TransferEngine::prepareDeviceInput(rows[i].get(), devices[i], streams[i]);
                    TransferEngine::prepareDeviceOutput(hidden[i].get(), devices[i], streams[i]);
                });
            }
            auto channel = transfer.createCapturedTransferChannel(*memory, DeviceId::cpu(),
                devices[1], streams[1], devices[0], streams[0], capacity);
            std::array<std::shared_ptr<PipelineActivationFanout>, 2> fanout;
            for (size_t i = 0; i < 2; ++i)
                fanout[i] = PipelineActivationFanout::create(devices[i], *collective, int(i), channel, memory, streams[i]);
            auto producer_claim = memory->claimNewAllocation(devices[1], PhysicalMemoryOwner::ActivationTransportStaging,
                capacity + 2 * sizeof(uint64_t));
            auto producer = transfer.allocateDeviceTransferBuffer(capacity, devices[1]);
            auto sequence = transfer.allocateDeviceTransferBuffer(sizeof(uint64_t), devices[1]);
            auto ignored_extent = transfer.allocateDeviceTransferBuffer(sizeof(uint64_t), devices[1]);
            workers[1]->submitAndWait([&] {
                require(backends[1]->memset(sequence->mutableDeviceData(), 0, sizeof(uint64_t), devices[1].ordinal, streams[1]), "Fanout original sequence initialization failed");
#ifdef HAVE_CUDA
                if (devices[1].is_cuda()) require(counted_channel_test::prepareKernelsCUDA(), "Fanout CUDA writer preparation failed");
#endif
#ifdef HAVE_ROCM
                if (devices[1].is_rocm()) require(counted_channel_test::prepareKernelsROCm(), "Fanout HIP writer preparation failed");
#endif
            });
            const auto outgoing = transfer.bindDeviceRowCountedTransfer(transfer.bindCapturedTransfer(channel,
                CapturedTransferEndpoint::Producer, {message_key, capacity}, producer), rows[1], sizeof(int32_t), row_bytes);
            const auto incoming = transfer.bindDeviceRowCountedTransfer(transfer.bindCapturedTransfer(channel,
                CapturedTransferEndpoint::Consumer, {message_key, capacity}, hidden[0]), rows[0], sizeof(int32_t), row_bytes);
            const std::array identities{outgoing.captureIdentity(), incoming.captureIdentity()};
            auto snapshot_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(host_snapshot_bytes));
            auto snapshot = transfer.allocateMappedHostRegion(host_snapshot_bytes, devices);
            std::array<std::shared_ptr<RetainedParticipantGraphExecutable>, 2> owners;
            for (size_t i = 0; i < 2; ++i)
                owners[i] = test::RetainedParticipantGraphExecutableFixtureAccess::original(original->permit(devices[i]), streams[i]);
            const auto await = [&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                for (size_t i = 0; i < 2; ++i)
                    for (;;)
                    {
                        GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
                        workers[i]->submitAndWait([&] { state = workers[i]->queryStreamExecutionState(streams[i], "pipeline fanout original terminal"); });
                        if (state == GPUStreamExecutionState::Complete) break;
                        require(std::chrono::steady_clock::now() < deadline, "Fanout exact original terminal exceeded canonical deadline");
                        std::this_thread::yield();
                    }
            };
            CapturedTransferInvocation active_invocation{};
            NativeReplayFailureDrain drain{[&] {
                if (active_invocation.valid()) transfer.cancelCapturedTransferInvocation(channel, active_invocation);
                require(collective->signalNativeCompletionAbort(), "Fanout original native abort failed");
                await();
            }};
            std::barrier begun(2), recorded(2), ended(2);
            std::array<bool, 2> began{};
            std::array<std::exception_ptr, 2> errors{};
            std::array<bool, 2> failed{};
            const auto capture = [&](size_t i) {
                auto &definition = test::RetainedParticipantGraphExecutableFixtureAccess::capture(owners[i]);
                std::unique_ptr<GraphCaptureDependencyLedger> ledger;
                std::unique_ptr<ScopedBackendGraphCapture> scope;
                try {
                    // The recorded row bank is an external input. Join its
                    // upload event on this exact worker/stream before capture,
                    // matching the production executor's arena-frontier pass.
                    TransferEngine::requireDeviceInput(rows[i].get(), devices[i], streams[i]);
                    using StagePlan = GraphCaptureDependencyLedger::StagePlan;
                    std::vector<StagePlan> stages;
                    stages.push_back({.stage_identity = fanout[i].get(), .stage_name = "original completed activation fanout",
                        .external_inputs = {rows[i].get()}, .outputs = {hidden[i].get()}});
                    ledger = std::make_unique<GraphCaptureDependencyLedger>(devices[i], streams[i], std::move(stages),
                        "pipeline fanout recorded row component");
                    scope = std::make_unique<ScopedBackendGraphCapture>(*workers[i], definition,
                        "original pipeline completed activation fanout", ledger.get());
                    require(scope->begin(), "Fanout original capture begin failed"); began[i] = true;
                } catch (...) { errors[i] = std::current_exception(); }
                begun.arrive_and_wait();
                if (began[0] && began[1]) try {
                    ScopedGraphCaptureStage stage(fanout[i].get());
                    require(backends[i]->memset(hidden[i]->gpu_data_ptr(), i ? 0x5a : 0xa5, storage_bytes, devices[i].ordinal, streams[i]), "Fanout captured poison failed");
                    if (i == 1)
                    {
#ifdef HAVE_CUDA
                        if (devices[i].is_cuda()) require(counted_channel_test::prepareCUDA(producer->mutableDeviceData(),
                            static_cast<uint64_t *>(ignored_extent->mutableDeviceData()), static_cast<const uint64_t *>(sequence->deviceData()), capacity, streams[i]), "Fanout captured CUDA source failed");
#endif
#ifdef HAVE_ROCM
                        if (devices[i].is_rocm()) require(counted_channel_test::prepareROCm(producer->mutableDeviceData(),
                            static_cast<uint64_t *>(ignored_extent->mutableDeviceData()), static_cast<const uint64_t *>(sequence->deviceData()), capacity, streams[i]), "Fanout captured HIP source failed");
#endif
                        transfer.enqueueCapturedTransfer(outgoing, streams[i]);
                    }
                    else transfer.enqueueCapturedTransfer(incoming, streams[i]);
                    const auto receipt = i == 0 ? transfer.capturedReceiveCompletion(incoming, streams[i]) : CapturedTransferReceiveCompletion::notRequired();
                    require(fanout[i]->enqueue(*hidden[i], *rows[i], sizeof(int32_t), row_bytes,
                        {message_key, capacity}, receipt, streams[i]), "Fanout original live native operations failed");
                    if (i == 1)
                    {
#ifdef HAVE_CUDA
                        if (devices[i].is_cuda()) require(counted_channel_test::advanceCUDA(static_cast<uint64_t *>(sequence->mutableDeviceData()), streams[i]), "Fanout CUDA original sequence advance failed");
#endif
#ifdef HAVE_ROCM
                        if (devices[i].is_rocm()) require(counted_channel_test::advanceROCm(static_cast<uint64_t *>(sequence->mutableDeviceData()), streams[i]), "Fanout HIP original sequence advance failed");
#endif
                    }
                    stage.complete();
                } catch (...) { errors[i] = std::current_exception(); collective->signalNativeCompletionAbort(); }
                recorded.arrive_and_wait();
                if (scope) try { scope->finish(); } catch (...) { if (!errors[i]) errors[i] = std::current_exception(); collective->signalNativeCompletionAbort(); }
                failed[i] = bool(errors[i]);
                ended.arrive_and_wait();
                if (!failed[0] && !failed[1]) try {
                    require(definition.prepareRuntimeContextStorage(memory) && definition.instantiate(), "Fanout original graph instantiation failed");
                } catch (...) { errors[i] = std::current_exception(); collective->signalNativeCompletionAbort(); }
            };
            auto first = workers[0]->submitAsync([&] { capture(0); });
            auto second = workers[1]->submitAsync([&] { capture(1); });
            first.get(); second.get();
            for (const auto &error : errors) if (error) std::rethrow_exception(error);
            size_t replay = 0;
            for (const int32_t live : {65, 3, 1, 0, 65})
            {
                for (size_t i = 0; i < 2; ++i)
                {
                    require(rows[i]->hostValid(), "Fanout read-only INT32 input was falsely published as output");
                    static_cast<int32_t *>(rows[i]->raw_mutable_data())[1] = live;
                    TransferEngine::publishHostWrite(rows[i].get());
                    workers[i]->submitAndWait([&] { TransferEngine::prepareDeviceInput(rows[i].get(), devices[i], streams[i]); });
                }
                const CapturedTransferInvocation invocation{channel->identity().nonce, replay + 1};
                active_invocation = invocation;
                transfer.beginCapturedTransferInvocation(channel, invocation);
                for (const auto i : replay % 2 ? std::array<size_t, 2>{1, 0} : std::array<size_t, 2>{0, 1})
                    workers[i]->submitAndWait([&] { require(test::RetainedParticipantGraphExecutableFixtureAccess::submit(owners[i], replay + 1), "Fanout retained original replay submission failed"); });
                await();
                for (size_t i = 0; i < 2; ++i)
                {
                    require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owners[i]), "Fanout original graph lacks terminal proof");
                    workers[i]->submitAndWait([&] {
                        require(backends[i]->deviceToHostOnStream(static_cast<std::byte *>(snapshot->mutableHostData()) + i * (storage_bytes + count_bytes),
                            hidden[i]->gpu_data_ptr(), storage_bytes, devices[i].ordinal, streams[i]), "Fanout terminal hidden readback failed");
                        require(backends[i]->deviceToHostOnStream(static_cast<std::byte *>(snapshot->mutableHostData()) + i * (storage_bytes + count_bytes) + storage_bytes,
                            rows[i]->gpu_data_ptr(), count_bytes, devices[i].ordinal, streams[i]), "Fanout terminal original row readback failed");
                    });
                }
                await();
                for (size_t i = 0; i < 2; ++i)
                {
                    const auto *actual = static_cast<const uint8_t *>(snapshot->mutableHostData()) + i * (storage_bytes + count_bytes);
                    for (size_t byte = 0; byte < storage_bytes; ++byte)
                        require(actual[byte] == (byte < size_t(live) * row_bytes ? counted_channel_test::byte(replay, byte) : uint8_t(i ? 0x5a : 0xa5)),
                            "Fanout copied capacity/padding or changed an original live byte");
                    std::array<int32_t, 4> observed{};
                    std::memcpy(observed.data(), actual + storage_bytes, count_bytes);
                    require(observed == std::array<int32_t, 4>{0x1357, live, 0x2468, 0x369}, "Fanout changed original INT32 input guards");
                }
                const auto &control = CapturedTransferChannelReplayProbe::terminalControl(*channel);
                require(control.consumer.message.bytes == size_t(live) * row_bytes && control.consumer.epoch == replay + 1,
                    "Fanout source header retained stale/capacity extent");
                require(outgoing.captureIdentity() == identities[0] && incoming.captureIdentity() == identities[1], "Fanout reset changed original captured identity");
                require(transfer.finishCapturedTransferInvocation(channel, invocation) == CapturedTransferInvocationRetirement::Completed,
                    "Fanout failed original invocation retirement");
                ++replay;
            }
            ::testing::Test::RecordProperty("fanout_wire_contract", "actual_publication_header24_then_completed_native_live_prefix;independent_member_poison");
            drain.completed = true;
        }
        collective.reset();
        original->requirePayloadRetired(); original->retire(); original->requireExactRetired();
    });
}
}
#ifdef HAVE_CUDA
/** @test Real completed native CUDA prefix with recorded live rows and distinct TP padding poisons. */
TEST(PipelineActivationFanoutNative, CUDARecordedRowsAndPoisonReplay)
{ pipelineActivationCompletedFanoutReplay(DeviceType::CUDA); }
#endif
#ifdef HAVE_ROCM
/** @test Symmetric actual ROCm native TP owner, selected only from original NUMA observations. */
TEST(PipelineActivationFanoutNative, ROCmRecordedRowsAndPoisonReplay)
{ pipelineActivationCompletedFanoutReplay(DeviceType::ROCm); }
#endif

#ifdef HAVE_CUDA
/** @test The same original native visibility law on the CUDA/NCCL backend. */
TEST(CapturedTransferNativePeerVisibility, CUDAOriginalInventory257Offset1)
{
    originalRankLocalReplay(DeviceType::CUDA, 2, [&](const auto &cohort) {
        roundTrip(cohort[0], cohort[1],
            {.capacity = 1024 * 1024 + 3, .bytes = {257}, .offsets = {1},
             .replays = 64, .patterns = 2, .peer_storage = PeerStorage::PrivateScratch,
             .measure_floor = true, .fixed_native_collective = true, .warmups = 5});
    });
}
#endif

#include "NativePipelineByteReplay.inc"
#include "NativeReductionOpcodeReplay.inc"
#include "NativeFixedHeadPipelineReplay.inc"

#include "CapturedTransferRankRoundTripReplay.inc"

#include "CapturedTransferRankCountedReplay.inc"

#include "NativeMoECompletedDecisionReplay.inc"

#include "NativeExactRowsMappedPacketReplay.inc"

#include "MoERecurrentOriginalSlotPacketReplay.inc"

#include "NativeSharedUnionPacketReplay.inc"
