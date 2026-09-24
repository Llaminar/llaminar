/**
 * @file Test__CapturedTransferChannel.cpp
 * @brief Real captured round trips across independently executing GPU endpoints.
 *
 * Two retained per-device graph families exchange bytes and return them without
 * a host relay. Alternating captured patterns distinguish fresh publication
 * from stale replay. Odd sizes, unaligned regions and guards prove exact extents.
 * All channel, scratch, readback and native graph storage is PMA-admitted.
 * Timings are diagnostic only; production preflight has no performance threshold.
 */
#include "transfer/CapturedTransferChannel.h"
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
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Propagate native failure before recording/launching any dependent node. */
    void require(bool condition, const char *message)
    { if (!condition) throw std::runtime_error(message); }

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
        /** @brief Bounded test-only terminal observation after both GPUs were submitted. */
        void await() const
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(35);
            bool ready = false;
            while (!ready)
            {
                require(backend->queryEvent(event, device.ordinal, &ready), "Captured channel terminal query failed");
                require(ready || std::chrono::steady_clock::now() < deadline, "Captured channel terminal timed out");
                if (!ready) std::this_thread::yield();
            }
        }
    };

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
        auto &first_worker = GPUDeviceContextPool::instance().getContext(first);
        auto &second_worker = GPUDeviceContextPool::instance().getContext(second);
        void *first_stream = nullptr, *second_stream = nullptr;
        first_worker.submitAndWait([&] {
            first_stream = first_worker.getOrCreateAuxiliaryStream("captured_channel_round_trip");
        });
        second_worker.submitAndWait([&] {
            second_stream = second_worker.getOrCreateAuxiliaryStream("captured_channel_round_trip");
        });
        ASSERT_NE(first_stream, nullptr);
        ASSERT_NE(second_stream, nullptr);

        const size_t capacity = workload.capacity;
        const bool workspace_peer = workload.peer_storage == PeerStorage::RetainedWorkspace;
        const size_t buffer_bytes = capacity + 64;
        const auto geometry = CapturedTransferChannel::memoryFor(capacity);
        const auto readback_bytes = TransferEngine::mappedHostRegionAllocationBytes(buffer_bytes);
        PhysicalMemoryPlanBuilder builder;
        builder.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            2 * geometry.mapped_host_bytes + 3 * readback_bytes);
        for (auto device : {first, second})
        {
            const auto resource = planningObservedResource(rank, device);
            builder.add(resource, PhysicalMemoryOwner::ActivationTransportStaging,
                2 * geometry.cursor_bytes_per_device + (device == first ? 3 : workspace_peer ? 0 : 1) * buffer_bytes);
            if (device == second && workspace_peer)
                builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, buffer_bytes);
            builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable,
                2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        }
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        auto &transfer = TransferEngine::instance();
        {
            auto forward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
                first, first_stream, second, second_stream, capacity);
            auto backward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
                second, second_stream, first, first_stream, capacity);
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

            // Geometry changes create distinct recordings, while channel epochs
            // continue monotonically across graph-family/request-like boundaries.
            for (const size_t bytes : workload.bytes)
                for (const size_t offset : workload.offsets)
                {
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
                    auto first_graph_claim = authority->reserveNewAllocations(first, PhysicalMemoryOwner::NativeGraphExecutable,
                        2 * GPUGraphMemoryContract::reservationBytesPerExecutable(first));
                    auto second_graph_claim = authority->reserveNewAllocations(second, PhysicalMemoryOwner::NativeGraphExecutable,
                        2 * GPUGraphMemoryContract::reservationBytesPerExecutable(second));
                    std::array<std::unique_ptr<IGPUGraphCapture>, 2> first_graphs, second_graphs;
                    for (size_t pattern = 0; pattern < 2; ++pattern)
                    {
                        first_worker.submitAndWait([&] {
                            auto &graph = first_graphs[pattern];
                            graph = first_worker.createGraphCapture(first_stream);
                            require(bool(graph), "Missing source native capture");
                            {
                                ScopedBackendGraphCapture capture(first_worker, *graph, "captured channel round trip source");
                                require(capture.begin(), "Source capture begin failed");
                                require(first_backend->memset(returned->mutableDeviceData(), 0xa5,
                                    buffer_bytes, first.ordinal, first_stream), "Returned guard initialization failed");
                                transfer.enqueueCapturedTransfer(sends[pattern], first_stream);
                                transfer.enqueueCapturedTransfer(collect, first_stream);
                                capture.finish();
                            }
                            require(graph->instantiate() && graph->nodeCount() >= 6, "Source graph incomplete");
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
                            require(graph->instantiate() && graph->nodeCount() >= 6, "Peer graph incomplete");
                        });
                    }
                    // Native graph pools grow in slabs. Attest the complete
                    // retained family, never attribute one slab to one slot.
                    EXPECT_TRUE(GPUGraphMemoryContract::acceptsFamilyObservation(first,
                        first_graphs[0]->residentMemoryBytes() + first_graphs[1]->residentMemoryBytes(),
                        2 * GPUGraphMemoryContract::reservationBytesPerExecutable(first)));
                    EXPECT_TRUE(GPUGraphMemoryContract::acceptsFamilyObservation(second,
                        second_graphs[0]->residentMemoryBytes() + second_graphs[1]->residentMemoryBytes(),
                        2 * GPUGraphMemoryContract::reservationBytesPerExecutable(second)));
                    const auto started = std::chrono::steady_clock::now();
                    for (size_t replay = 0; replay < workload.replays; ++replay)
                    {
                        const auto pattern = replay % workload.patterns;
                        const auto submit_first = [&] { first_worker.submitAndWait([&] {
                            require(first_graphs[pattern]->launch(), "Source retained launch failed");
                            transfer.enqueueDeviceToMappedHost(*returned, 0, *readback, 0, buffer_bytes, first, first_stream);
                            require(first_backend->recordEvent(first_done.event, first.ordinal, first_stream), "Source terminal publication failed");
                        }); };
                        const auto submit_second = [&] { second_worker.submitAndWait([&] {
                            require(second_graphs[pattern]->launch(), "Peer retained launch failed");
                            require(second_backend->recordEvent(second_done.event, second.ordinal, second_stream), "Peer terminal publication failed");
                        }); };
                        // Either endpoint may be queued first. No observed model
                        // state, host copy or terminal wait chooses the peer work.
                        if (replay % 4 < 2) { submit_first(); submit_second(); }
                        else { submit_second(); submit_first(); }
                        first_done.await();
                        second_done.await();
                        const auto *actual = static_cast<const unsigned char *>(readback->mutableHostData());
                        for (size_t i = 0; i < buffer_bytes; ++i)
                            if (actual[i] != (i >= offset && i < offset + bytes ? patternByte(i, pattern) : 0xa5))
                                FAIL() << first.toString() << " -> " << second.toString() << " replay=" << replay
                                    << " bytes=" << bytes << " offset=" << offset << " first_bad_byte=" << i;
                    }
                    std::cout << "CAPTURED_CHANNEL_ROUND_TRIP first=" << first.toString() << " second=" << second.toString()
                        << " bytes=" << bytes << " offset=" << offset << " replays=" << workload.replays << " verification_wall_ms="
                        << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() << '\n';
                }
        }
        for (const auto device : {DeviceId::cpu(), first, second})
            EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
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
