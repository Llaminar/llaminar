/**
 * @file Test__DeviceCountedTransferChannel.cpp
 * @brief One retained graph transports GPU-selected extents across real devices.
 *
 * The source graph authors a different count/pattern on each replay. The peer
 * adopts that published extent and echoes only those bytes. Full/empty/odd
 * transitions, opposite submission orders and untouched tails exercise the
 * exact production acquire/copy/publish protocol without CPU count selection.
 * PMA owns every allocation; workspace leases outlive the retired name maps.
 */
#include "CountedChannelTestKernels.h"
#include "CapturedChannelGraphProof.h"
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
#include <cstdint>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Stop the fixture immediately on a failed native contract. */
    void require(bool success, const char *message)
    { if (!success) throw std::runtime_error(message); }

    /** @brief Test-only exact-event observer; never a production synchronization. */
    struct Terminal final
    {
        IBackend *backend;
        DeviceId device;
        void *event;
        /** @brief Release the event after the caller's last explicit observation. */
        ~Terminal() { if (event) backend->destroyEvent(event, device.ordinal); }
        /** @brief Bound observations to the normal protocol timeout plus diagnostic slack. */
        void await() const
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(35);
            bool ready = false;
            while (!ready)
            {
                require(backend->queryEvent(event, device.ordinal, &ready), "Counted channel terminal query failed");
                require(ready || std::chrono::steady_clock::now() < deadline, "Counted channel terminal timed out");
                if (!ready) std::this_thread::yield();
            }
        }
    };

    /** @brief Dispatch only the fixture's device-owned input writer. */
    void preparePayload(DeviceId device, void *payload, std::uint64_t *count,
        const std::uint64_t *sequence, size_t capacity, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) { require(counted_channel_test::prepareCUDA(payload, count, sequence, capacity, stream), "CUDA count writer"); return; }
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) { require(counted_channel_test::prepareROCm(payload, count, sequence, capacity, stream), "ROCm count writer"); return; }
#endif
        throw std::invalid_argument("Counted channel fixture requires GPU endpoints");
    }

    /** @brief Advance the fixture sequence as the captured round trip's last device node. */
    void advanceSequence(DeviceId device, std::uint64_t *sequence, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) { require(counted_channel_test::advanceCUDA(sequence, stream), "CUDA sequence advance"); return; }
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) { require(counted_channel_test::advanceROCm(sequence, stream), "ROCm sequence advance"); return; }
#endif
        throw std::invalid_argument("Counted channel fixture requires GPU endpoints");
    }

    /** @brief Exercise 128 changing extents in one captured graph per physical alignment.
     * @param first Producer and result endpoint; ordering is independent of vendor.
     * @param second Receiver and echo endpoint, with its own graph and private count.
     * @param capacity Immutable upper bound, exercising either complete lowering. */
    void verifyCountedRoundTrip(DeviceId first, DeviceId second, size_t capacity = 1024 * 1024 + 3)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &rank = inventory->ranks.at(mpi->rank());
        const std::array devices{first, second};
        std::array<IBackend *, 2> backends{getBackendFor(first), getBackendFor(second)};
        std::array<IWorkerGPUContext *, 2> workers{};
        std::array<void *, 2> streams{};
        for (size_t i = 0; i < 2; ++i)
        {
            ASSERT_NE(backends[i], nullptr);
            ASSERT_GT(backends[i]->deviceCount(), devices[i].ordinal);
            workers[i] = &GPUDeviceContextPool::instance().getContext(devices[i]);
            workers[i]->submitAndWait([&] {
                streams[i] = workers[i]->getOrCreateAuxiliaryStream("device_counted_channel");
#ifdef HAVE_CUDA
                if (devices[i].is_cuda()) require(counted_channel_test::prepareKernelsCUDA(), "Prepare CUDA fixture kernels");
#endif
#ifdef HAVE_ROCM
                if (devices[i].is_rocm()) require(counted_channel_test::prepareKernelsROCm(), "Prepare ROCm fixture kernels");
#endif
            });
        }
        ASSERT_GE(capacity, 257u);
        const size_t storage_bytes = capacity + 64;
        WorkspaceRequirements requirements;
        requirements.buffers = {{"payload", storage_bytes, 16, true}, {"returned", storage_bytes, 16, true},
            {"counts", 3 * sizeof(std::uint64_t), 8, true}};
        const auto workspace_bytes = requirements.total_bytes_with_alignment();
        const auto geometry = CapturedTransferChannel::memoryFor(capacity);
        PhysicalMemoryPlanBuilder builder;
        builder.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            2 * geometry.mapped_host_bytes);
        for (const auto device : devices)
        {
            const auto resource = planningObservedResource(rank, device);
            builder.add(resource, PhysicalMemoryOwner::ActivationTransportStaging, 2 * geometry.cursor_bytes_per_device);
            builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, workspace_bytes);
            builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable, GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        }
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        auto &transfer = TransferEngine::instance();
        {
            auto forward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), first, streams[0], second, streams[1], capacity);
            auto backward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(), second, streams[1], first, streams[0], capacity);
            std::array<std::shared_ptr<const WorkspaceBufferLease>, 2> payloads, returned, counts;
            for (size_t i = 0; i < 2; ++i)
            {
                DeviceWorkspaceManager workspace(devices[i], workspace_bytes, authority);
                ASSERT_TRUE(workspace.allocate(requirements));
                payloads[i] = workspace.retainBuffer("payload", 0, storage_bytes);
                returned[i] = workspace.retainBuffer("returned", 0, storage_bytes);
                counts[i] = workspace.retainBuffer("counts", 0, 3 * sizeof(std::uint64_t));
                workspace.release();
            }
            Terminal first_done{backends[0], first, backends[0]->createEvent(first.ordinal)};
            Terminal second_done{backends[1], second, backends[1]->createEvent(second.ordinal)};
            ASSERT_NE(first_done.event, nullptr);
            ASSERT_NE(second_done.event, nullptr);
            for (const size_t offset : {size_t{1}, size_t{16}})
            {
                // These are the ONLY captures for this alignment. The device
                // sequence selects all 128 live sizes through the same graphs.
                const auto send = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(forward,
                    CapturedTransferEndpoint::Producer, {51, capacity}, payloads[0], offset), counts[0], 0);
                const auto receive = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(forward,
                    CapturedTransferEndpoint::Consumer, {51, capacity}, payloads[1], offset), counts[1], 0);
                const auto reply = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(backward,
                    CapturedTransferEndpoint::Producer, {52, capacity}, payloads[1], offset), counts[1], 0);
                const auto collect = transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(backward,
                    CapturedTransferEndpoint::Consumer, {52, capacity}, returned[0], offset), counts[0], sizeof(std::uint64_t));
                auto *sequence = static_cast<std::uint64_t *>(counts[0]->data(2 * sizeof(std::uint64_t)));
                require(backends[0]->memset(sequence, 0, sizeof(*sequence), first.ordinal, streams[0]), "Initialize device sequence");
                require(backends[0]->recordEvent(first_done.event, first.ordinal, streams[0]), "Sequence setup event");
                first_done.await();
                auto first_graph_claim = authority->reserveNewAllocations(first, PhysicalMemoryOwner::NativeGraphExecutable,
                    GPUGraphMemoryContract::reservationBytesPerExecutable(first));
                auto second_graph_claim = authority->reserveNewAllocations(second, PhysicalMemoryOwner::NativeGraphExecutable,
                    GPUGraphMemoryContract::reservationBytesPerExecutable(second));
                std::array<std::unique_ptr<IGPUGraphCapture>, 2> graphs;
                for (size_t i = 0; i < 2; ++i)
                    workers[i]->submitAndWait([&] {
                        graphs[i] = workers[i]->createGraphCapture(streams[i]);
                        require(bool(graphs[i]), "Missing counted channel graph");
                        ScopedBackendGraphCapture capture(*workers[i], *graphs[i], "device-counted round trip");
                        require(capture.begin(), "Counted capture begin");
                        if (i == 0)
                        {
                            require(backends[i]->memset(returned[i]->data(), 0xa5, storage_bytes, first.ordinal, streams[i]), "Reset returned guards");
                            preparePayload(first, payloads[0]->data(offset), static_cast<std::uint64_t *>(counts[0]->data()), sequence, capacity, streams[0]);
                            transfer.enqueueCapturedTransfer(send, streams[0]);
                            transfer.enqueueCapturedTransfer(collect, streams[0]);
                            advanceSequence(first, sequence, streams[0]);
                        }
                        else
                        {
                            require(backends[i]->memset(payloads[i]->data(), 0xa5, storage_bytes, second.ordinal, streams[i]), "Reset peer guards");
                            transfer.enqueueCapturedTransfer(receive, streams[1]);
                            transfer.enqueueCapturedTransfer(reply, streams[1]);
                        }
                        capture.finish();
                        counted_channel_test::verifyCapturedChannelGraph(*graphs[i], capacity, 2);
                        require(graphs[i]->instantiate(), "Incomplete counted capture");
                    });
                std::array<std::vector<std::uint8_t>, 2> actual{std::vector<std::uint8_t>(storage_bytes), std::vector<std::uint8_t>(storage_bytes)};
                std::array<std::uint64_t, 3> observed_counts{};
                for (size_t replay = 0; replay < 128; ++replay)
                {
                    const auto order = replay % 2 ? std::array<size_t, 2>{1, 0} : std::array<size_t, 2>{0, 1};
                    for (const auto i : order)
                    {
                        workers[i]->submitAndWait([&] {
                            require(graphs[i]->launch(), "Counted retained replay failed");
                        });
                        // Deliberately leave an already-running endpoint
                        // waiting for its peer. This tests publication under
                        // scheduling skew, not just two nearly simultaneous launches.
                        if (i == order[0] && replay % 7 == 0)
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    // A pageable test readback may block inside the runtime.
                    // Submit BOTH complete graphs before any such observation;
                    // otherwise the first endpoint waits for an unsubmitted peer.
                    for (size_t i = 0; i < 2; ++i)
                        workers[i]->submitAndWait([&] {
                            require(backends[i]->deviceToHostOnStream(actual[i].data(), i == 0 ? returned[0]->data() : payloads[1]->data(),
                                storage_bytes, devices[i].ordinal, streams[i]), "Observe counted payload");
                            if (i == 0)
                                require(backends[i]->deviceToHostOnStream(observed_counts.data(), counts[0]->data(), sizeof(observed_counts),
                                    first.ordinal, streams[i]), "Observe terminal counts");
                            require(backends[i]->recordEvent(i == 0 ? first_done.event : second_done.event, devices[i].ordinal, streams[i]), "Counted terminal event");
                        });
                    first_done.await(); second_done.await();
                    const auto bytes = counted_channel_test::extent(replay, capacity);
                    ASSERT_EQ(observed_counts[0], bytes);
                    ASSERT_EQ(observed_counts[1], bytes);
                    ASSERT_EQ(observed_counts[2], replay + 1);
                    for (size_t peer = 0; peer < 2; ++peer)
                        for (size_t byte = 0; byte < storage_bytes; ++byte)
                        {
                            const auto expected = byte >= offset && byte - offset < bytes
                                ? counted_channel_test::byte(replay, byte - offset) : std::uint8_t{0xa5};
                            if (actual[peer][byte] != expected)
                                FAIL() << first.toString() << " -> " << second.toString() << " replay=" << replay << " peer=" << peer
                                    << " offset=" << offset << " live_bytes=" << bytes << " first_bad_byte=" << byte;
                        }
                }
            }
        }
        for (auto device : devices)
        {
            EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ExecutionWorkspace, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            EXPECT_EQ(authority->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
        }
        EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }

    /** @brief Cross the exact dispatch boundary without changing the protocol oracle. */
    void verifyFusedBoundaries(DeviceId first, DeviceId second)
    {
        for (const size_t capacity : {size_t{257}, size_t{4099},
                size_t{kCapturedTransferSingleBlockBytes}, size_t{kCapturedTransferSingleBlockBytes + 1}})
            verifyCountedRoundTrip(first, second, capacity);
    }
}

#ifdef HAVE_CUDA
TEST(DeviceCountedTransferChannel, CUDA_CUDA) { verifyCountedRoundTrip(DeviceId::cuda(0), DeviceId::cuda(1)); }
TEST(DeviceCountedTransferChannel, CUDA_CUDA_FusedBoundaries) { verifyFusedBoundaries(DeviceId::cuda(0), DeviceId::cuda(1)); }
#endif
#ifdef HAVE_ROCM
TEST(DeviceCountedTransferChannel, ROCm_ROCm) { verifyCountedRoundTrip(DeviceId::rocm(0), DeviceId::rocm(1)); }
TEST(DeviceCountedTransferChannel, ROCm_ROCm_FusedBoundaries) { verifyFusedBoundaries(DeviceId::rocm(0), DeviceId::rocm(1)); }
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
TEST(DeviceCountedTransferChannel, CUDA_ROCm) { verifyCountedRoundTrip(DeviceId::cuda(0), DeviceId::rocm(0)); }
TEST(DeviceCountedTransferChannel, ROCm_CUDA) { verifyCountedRoundTrip(DeviceId::rocm(0), DeviceId::cuda(0)); }
TEST(DeviceCountedTransferChannel, CUDA_ROCm_FusedBoundaries) { verifyFusedBoundaries(DeviceId::cuda(0), DeviceId::rocm(0)); }
TEST(DeviceCountedTransferChannel, ROCm_CUDA_FusedBoundaries) { verifyFusedBoundaries(DeviceId::rocm(0), DeviceId::cuda(0)); }
#endif
