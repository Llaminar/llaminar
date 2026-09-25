/**
 * @file Test__GPUHostRegistrationLifetime.cpp
 * @brief Repeated page registration, captured DMA and retirement on every GPU.
 *
 * A four-MI50 startup probe exposed an ATS interrupt storm when HIP registered
 * ordinary host pages through HMM/SVM. Exercise both tensor uploads and public
 * external registration with the identical byte contract on CUDA and ROCm.
 * Small odd tails and cache-exceeding regions cover page and bulk lifetimes;
 * no model, weight format, pageable-copy substitution or serial compute oracle
 * is involved. CPU waits occur only at the test's terminal result boundary.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "../../utils/HostPageMappingTestUtils.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <array>

using namespace llaminar2;

namespace
{
    /** @brief The two production owners that register caller-allocated pages. */
    enum class SourceOwner { TensorUpload, ExternalChannel };

    /** @return Exact backend selected by the CMake-owned split executable. */
    DeviceId deviceAt(int ordinal)
    {
#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
        return DeviceId::rocm(ordinal);
#elif defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
        return DeviceId::cuda(ordinal);
#else
        return DeviceId::invalid();
#endif
    }

    /**
     * @brief Retire/recreate host ranges across every device, proving exact DMA.
     * @param owner Public transfer path whose page-registration lifetime is tested.
     *
     * Each captured graph dies before its pinned owners. Changing a source
     * between replays occurs only after the terminal event, so the second replay
     * also proves that retained graphs consume live bytes, not a setup snapshot.
     */
    void proveRegistrationLifetime(SourceOwner owner)
    {
        const auto first = deviceAt(0);
        if (!first.is_gpu()) GTEST_SKIP() << "No GPU backend compiled";
        auto *backend = getBackendFor(first);
        ASSERT_NE(backend, nullptr);
        ASSERT_GT(backend->deviceCount(), 0);
        TransferEngine transfers;
        for (int round = 0; round < 2; ++round)
            for (int ordinal = 0; ordinal < backend->deviceCount(); ++ordinal)
                for (const std::size_t bytes : {4093u, 32u * 1024u * 1024u + 13u})
                {
                    const auto device = deviceAt(ordinal);
                    SCOPED_TRACE(device.toString() + " bytes=" + std::to_string(bytes) +
                        " round=" + std::to_string(round));
                    auto &worker = GPUDeviceContextPool::instance().getContext(device);
                    const auto lanes = transfers.allocatePersistentTransferExecutionLanes(
                        1u, device, "host_registration_lifetime");
                    auto download = transfers.allocatePinnedHostBuffer(bytes, device);
                    worker.submitAndWait([&] {
                        const auto stream = lanes.front().stream();
                        const std::size_t elements = (bytes + sizeof(float) - 1u) / sizeof(float);
                        // A fresh page-owned tensor reproduces the planner's
                        // actual input lifetime, including unregistration before
                        // the derived AlignedVector unmaps the source pages.
                        FP32Tensor tensor({1u, elements},
                            AlignedVector<float>::pageMappedUninitialized(elements));
                        auto *tensor_bytes = static_cast<unsigned char *>(tensor.raw_mutable_data());
                        for (std::size_t index = 0; index < tensor.size_bytes(); ++index)
                            tensor_bytes[index] = static_cast<unsigned char>(
                                index * 43u + (index >> 7u) + 0x31u + round + ordinal);
                        std::shared_ptr<AlignedVector<unsigned char>> external;
                        std::shared_ptr<PinnedHostTransferBuffer> registered;
                        if (owner == SourceOwner::ExternalChannel)
                        {
                            external = std::make_shared<AlignedVector<unsigned char>>(
                                AlignedVector<unsigned char>::pageMappedUninitialized(bytes));
                            std::memcpy(external->data(), tensor_bytes, bytes);
                            registered = transfers.registerExternalPinnedHostBuffer(
                                external->data(), bytes, device, external);
                            TransferEngine::prepareDeviceOutput(&tensor, device, stream);
                        }
                        else
                        {
                            TransferEngine::prepareDeviceInput(&tensor, device, stream);
                            TransferEngine::requireDeviceInput(&tensor, device, stream);
                        }

                        auto graph = worker.createGraphCapture(stream);
                        ASSERT_NE(graph, nullptr);
                        GraphCaptureDependencyLedger::StagePlan transaction{
                            .stage_identity = &tensor,
                            .stage_name = "captured host registration roundtrip",
                        };
                        if (registered) transaction.outputs = {&tensor};
                        else transaction.external_inputs = {&tensor};
                        GraphCaptureDependencyLedger ledger(device, stream,
                            {std::move(transaction)}, "host registration lifetime");
                        {
                            ScopedBackendGraphCapture capture(worker, *graph,
                                "host registration lifetime", &ledger);
                            ASSERT_TRUE(capture.begin());
                            ScopedGraphCaptureStage stage(&tensor);
                            if (registered)
                                transfers.enqueuePinnedHostToDevice(*registered, 0u,
                                    &tensor, 0u, bytes, device, stream);
                            transfers.enqueueDeviceToPinnedHost(&tensor, 0u,
                                *download, 0u, bytes, device, stream);
                            stage.complete();
                            capture.finish();
                        }
                        ASSERT_TRUE(graph->instantiate());
                        void *terminal = worker.createEvent();
                        ASSERT_NE(terminal, nullptr);
                        for (int replay = 0; replay < 2; ++replay)
                        {
                            if (external)
                                for (std::size_t index = 0; index < bytes; ++index)
                                    (*external)[index] = static_cast<unsigned char>(
                                        index * 43u + (index >> 7u) + round + ordinal + replay);
                            std::memset(download->mutableData(), 0, bytes);
                            ASSERT_TRUE(graph->launch());
                            const bool recorded = worker.recordEventChecked(terminal, stream);
                            const bool completed = recorded && worker.synchronizeEventChecked(terminal);
                            ASSERT_TRUE(completed);
                            const void *expected = external ? external->data() : tensor_bytes;
                            EXPECT_EQ(std::memcmp(download->data(), expected, bytes), 0);
                        }
                        worker.destroyEvent(terminal);
                        // Captured raw pointers must be retired before dropping
                        // the last registered source owner. Registration itself
                        // retains the external mapping, even after its caller
                        // releases the original handle.
                        graph.reset();
                        if (registered)
                        {
                            std::weak_ptr<AlignedVector<unsigned char>> lifetime = external;
                            external.reset();
                            EXPECT_FALSE(lifetime.expired());
                            registered.reset();
                            EXPECT_TRUE(lifetime.expired());
                        }
                    });
                }
    }
}

TEST(GPUHostRegistrationLifetime, TensorUploadAcrossAllDevices)
{
    proveRegistrationLifetime(SourceOwner::TensorUpload);
}

TEST(GPUHostRegistrationLifetime, ExternalChannelCapturedReplayAcrossAllDevices)
{
    proveRegistrationLifetime(SourceOwner::ExternalChannel);
}

TEST(GPUHostRegistrationLifetime, NeighbourReclaimPreservesCapturedRegisteredPayload)
{
    const auto first = deviceAt(0);
    auto *backend = getBackendFor(first);
    ASSERT_NE(backend, nullptr);
    ASSERT_GT(backend->deviceCount(), 0);
    TransferEngine transfers;
    for (int ordinal = 0; ordinal < backend->deviceCount(); ++ordinal)
        for (const bool mapped_channel : {false, true})
        {
            const auto device = deviceAt(ordinal);
            SCOPED_TRACE(device.toString() + (mapped_channel ? " mapped" : " pinned"));
            auto &worker = GPUDeviceContextPool::instance().getContext(device);
            const auto lanes = transfers.allocatePersistentTransferExecutionLanes(
                1u, device, "host_registration_neighbour_reclaim");
            worker.submitAndWait([&] {
                // The unrelated tail shares an initial huge-page VMA. Reclaim
                // is legal there even while this live prefix is GPU-registered.
                auto host = std::make_shared<test::HostPageMapping>();
                const auto page = test::HostPageMapping::pageBytes();
                const auto bytes = test::HostPageMapping::live_bytes;
                std::shared_ptr<PinnedHostTransferBuffer> pinned;
                std::shared_ptr<MappedHostTransferRegion> mapped;
                if (mapped_channel)
                    mapped = transfers.registerExternalMappedHostRegion(
                        host->data(), bytes, std::array{device}, host);
                else
                    pinned = transfers.registerExternalPinnedHostBuffer(
                        host->data(), bytes, device, host);
#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
                // Assert the installed production policy, not a source scan
                // or absence of a rate-limited driver warning. NVIDIA's
                // long-term physical pinning has no KFD USERPTR eviction path.
                EXPECT_TRUE(test::HostPageMapping::hasFlag(host->data(), "nh"));
                EXPECT_TRUE(test::HostPageMapping::hasFlag(host->data() + bytes - 1u, "nh"));
                EXPECT_TRUE(test::HostPageMapping::hasFlag(
                    host->data() + 2u * test::HostPageMapping::huge_bytes, "hg"));
#endif
                FP32Tensor output({1u, page / sizeof(float)});
                auto download = transfers.allocatePinnedHostBuffer(page, device);
                const auto stream = lanes.front().stream();
                TransferEngine::prepareDeviceOutput(&output, device, stream);
                auto graph = worker.createGraphCapture(stream);
                ASSERT_NE(graph, nullptr);
                GraphCaptureDependencyLedger::StagePlan transaction{
                    .stage_identity = &output,
                    .stage_name = "registered payload neighbour reclaim",
                    .outputs = {&output},
                };
                GraphCaptureDependencyLedger ledger(device, stream,
                    {std::move(transaction)}, "registered payload neighbour reclaim");
                {
                    ScopedBackendGraphCapture capture(worker, *graph,
                        "registered payload neighbour reclaim", &ledger);
                    ASSERT_TRUE(capture.begin());
                    ScopedGraphCaptureStage stage(&output);
                    if (mapped)
                        transfers.enqueueMappedHostToDevice(*mapped, bytes - page,
                            &output, 0u, page, device, stream);
                    else
                        transfers.enqueuePinnedHostToDevice(*pinned, bytes - page,
                            &output, 0u, page, device, stream);
                    transfers.enqueueDeviceToPinnedHost(&output, 0u, *download,
                        0u, page, device, stream);
                    stage.complete();
                    capture.finish();
                }
                ASSERT_TRUE(graph->instantiate());
                const auto destroy_event = [&](void *event) { worker.destroyEvent(event); };
                std::unique_ptr<void, decltype(destroy_event)> terminal(
                    worker.createEvent(), destroy_event);
                ASSERT_NE(terminal, nullptr);
                for (int replay = 0; replay < 3; ++replay)
                {
                    host->reclaimNeighbour();
                    ASSERT_TRUE(graph->launch());
                    ASSERT_TRUE(worker.recordEventChecked(terminal.get(), stream));
                    ASSERT_TRUE(worker.synchronizeEventChecked(terminal.get()));
                    EXPECT_EQ(std::memcmp(download->data(), host->data() + bytes - page, page), 0);
                }
                // Graph and terminal event retire before their registered host
                // owner; no runtime free or registration occurs inside capture.
                graph.reset();
            });
        }
}
