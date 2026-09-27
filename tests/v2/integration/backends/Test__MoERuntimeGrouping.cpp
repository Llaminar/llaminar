/**
 * @file Test__MoERuntimeGrouping.cpp
 * @brief Captured runtime route counts and stable grouping under adversarial skew.
 *
 * One retained graph cycles through empty, maximally skewed, distributed and
 * sparse routing inputs. Counts use bounded integer arithmetic; grouped rows
 * must retain serial original-route order and exact weight bits regardless of
 * integer update order. The public runtime table owns all device publications.
 * This model-free regression intentionally has no timing threshold.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2::test
{
    /** @brief One symmetric captured protocol proof per accelerator backend. */
    class MoERuntimeGrouping : public ::testing::TestWithParam<std::string> {};

    /** @test Integer counts and stable placement survive 20 adversarial replays. */
    TEST_P(MoERuntimeGrouping, CapturedSkewAndSparseCountsAreExact)
    {
        const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            auto *backend = getBackendFor(device);
            auto *stream = context.defaultStream();
            ASSERT_NE(backend, nullptr);
            ASSERT_NE(stream, nullptr);
            constexpr int top_k = 8;
            for (int experts : {8, 17, 256})
                for (int rows : {1, 2, 16, 32})
                {
                    SCOPED_TRACE("experts=" + std::to_string(experts) + " rows=" + std::to_string(rows));
                    const int slots = rows * top_k;
                    DeviceMoERuntimeTable::Config config;
                    config.device_id = device;
                    config.num_layers = 1;
                    config.num_experts = experts;
                    config.top_k = top_k;
                    config.mirror_to_device = true;
                    config.prefill_token_capacity = rows;
                    MoERuntimeTable runtime(config);
                    const auto state = runtime.hostLayerState(0);
                    const auto requirements = device.is_cuda()
                        ? MoEWorkspaceBuffers::cudaMoE(rows, 256, 256, experts, top_k)
                        : MoEWorkspaceBuffers::rocmMoE(rows, 256, 256, experts, top_k);
                    DeviceWorkspaceManager workspace(device,
                        requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
                    ASSERT_TRUE(workspace.allocate(requirements));
                    auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
                    ASSERT_NE(kernel, nullptr);
                    kernel->setGPUStream(stream);
                    auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
                    ASSERT_NE(consumer, nullptr);
                    consumer->bindWorkspace(&workspace);
                    auto indices = TestTensorFactory::createFP32({static_cast<size_t>(rows), top_k});
                    auto weights = TestTensorFactory::createFP32({static_cast<size_t>(rows), top_k});
                    std::fill_n(indices->mutable_data(), slots, -1.0f);
                    std::fill_n(weights->mutable_data(), slots, 0.0f);
                    ASSERT_TRUE(indices->ensureOnDevice(device, stream));
                    ASSERT_TRUE(weights->ensureOnDevice(device, stream));
                    const auto enqueue = [&] {
                        return kernel->groupPrefillRoutes(runtime.deviceLayerState(0),
                            indices.get(), weights.get(), rows, rows, experts, top_k);
                    };
                    ASSERT_TRUE(enqueue());
                    ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                    auto graph = context.createGraphCapture(stream);
                    ASSERT_NE(graph, nullptr);
                    {
                        GraphCaptureGuard recording;
                        ASSERT_TRUE(graph->beginCapture());
                        ASSERT_TRUE(enqueue());
                        ASSERT_TRUE(graph->endCapture());
                    }
                    ASSERT_TRUE(graph->instantiate());
                    for (int replay = 0; replay < 20; ++replay)
                    {
                        SCOPED_TRACE("replay=" + std::to_string(replay));
                        std::vector<float> ids(slots), probabilities(slots);
                        for (int slot = 0; slot < slots; ++slot)
                        {
                            const int pattern = replay % 4;
                            ids[slot] = pattern == 0 ? 0.0f :
                                pattern == 1 || (pattern == 3 && slot % 3 != 0) ? -1.0f :
                                static_cast<float>((slot * 37 + replay) % experts);
                            probabilities[slot] = static_cast<float>(slot + 1) / 512.0f;
                        }
                        std::vector<int32_t> expected_counts(experts), expected_offsets(experts);
                        std::vector<int32_t> expected_rows(slots), actual_rows(slots), actual_counts(experts), actual_offsets(experts);
                        std::vector<float> expected_weights(slots), actual_weights(slots);
                        int grouped = 0;
                        for (int expert = 0; expert < experts; ++expert)
                        {
                            expected_offsets[expert] = grouped;
                            for (int slot = 0; slot < slots; ++slot)
                                if (ids[slot] == static_cast<float>(expert))
                                {
                                    ++expected_counts[expert];
                                    expected_rows[grouped] = slot;
                                    expected_weights[grouped++] = probabilities[slot];
                                }
                        }
                        ASSERT_TRUE(backend->hostToDevice(indices->gpu_data_ptr(), ids.data(), slots * sizeof(float), device.ordinal, stream));
                        ASSERT_TRUE(backend->hostToDevice(weights->gpu_data_ptr(), probabilities.data(), slots * sizeof(float), device.ordinal, stream));
                        ASSERT_TRUE(graph->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual_counts.data(), state.expert_counts, experts * sizeof(int32_t), device.ordinal, stream));
                        ASSERT_TRUE(backend->deviceToHost(actual_offsets.data(), state.expert_offsets, experts * sizeof(int32_t), device.ordinal, stream));
                        ASSERT_TRUE(backend->deviceToHost(actual_rows.data(), state.grouped_token_ids, slots * sizeof(int32_t), device.ordinal, stream));
                        ASSERT_TRUE(backend->deviceToHost(actual_weights.data(), state.grouped_route_weights, slots * sizeof(float), device.ordinal, stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                        EXPECT_EQ(actual_counts, expected_counts);
                        EXPECT_EQ(actual_offsets, expected_offsets);
                        EXPECT_EQ(actual_rows, expected_rows);
                        for (int slot = 0; slot < slots; ++slot)
                            EXPECT_EQ(std::bit_cast<uint32_t>(actual_weights[slot]),
                                std::bit_cast<uint32_t>(expected_weights[slot])) << "grouped slot=" << slot;
                    }
                }
        });
    }

    INSTANTIATE_TEST_SUITE_P(Backends, MoERuntimeGrouping, ::testing::Values("CUDA", "ROCm"),
        [](const ::testing::TestParamInfo<std::string> &info) { return info.param; });
}
