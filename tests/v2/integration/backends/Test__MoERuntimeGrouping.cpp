/**
 * @file Test__MoERuntimeGrouping.cpp
 * @brief Captured runtime route counts and stable grouping under adversarial skew.
 *
 * One retained graph cycles through empty, maximally skewed, distributed and
 * sparse routing inputs. Counts use bounded integer arithmetic; grouped rows
 * must retain serial original-route order and exact weight bits regardless of
 * integer update order. The public runtime table owns all device publications.
 * This model-free regression intentionally has no timing threshold.
 * The standalone expert-prefix bridge also sweeps warp/block boundaries,
 * multiple chunks, near-limit integer totals and poisoned replay storage.
 * Grouping initialization proves one captured kernel replaces native memsets
 * while preserving all optional publications, independent extents and guards.
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
#include "kernels/common/MoEGroupingInitialization.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// These are the production grouping launch bridges, not test scan kernels.
#ifdef HAVE_CUDA
extern "C" bool cudaMoE_exclusive_scan(const int *, int *, int, int, void *);
#endif
#ifdef HAVE_ROCM
extern "C" bool hipMoE_exclusive_scan(int *, int *, int, int, void *);
#endif

namespace llaminar2::test
{
    /** @brief One symmetric captured protocol proof per accelerator backend. */
    class MoERuntimeGrouping : public ::testing::TestWithParam<std::string> {};

    /** @brief Independently registered reset-publication proof for both vendors. */
    class MoEGroupingInitializationTest : public ::testing::TestWithParam<std::string> {};

    /**
     * @test One captured reset clears every present bank and preserves guards.
     *
     * Poison all six banks before each of 20 replays, including optional absent
     * arrays. Expert and slot extents alternate which is larger and cross wave
     * and block boundaries. Node inventory proves the optimized one-kernel path;
     * this is a functional assertion, not a timing threshold in preflight.
     */
    TEST_P(MoEGroupingInitializationTest, CapturedSentinelsAndGuards)
    {
        const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            auto *backend = getBackendFor(device);
            auto *stream = context.defaultStream();
            ASSERT_NE(backend, nullptr);
            ASSERT_NE(stream, nullptr);
            for (const auto [experts, slots] : std::vector<std::array<int, 2>>{
                     {1, 1}, {1, 33}, {33, 1}, {31, 63}, {65, 32},
                     {255, 256}, {257, 255}, {512, 513}, {513, 512},
                     {256, 4096}, {1024, 32767}})
            {
                SCOPED_TRACE("experts=" + std::to_string(experts) + " slots=" + std::to_string(slots));
                const size_t stride = static_cast<size_t>(std::max(experts, slots)) + 2;
                const size_t words = 6 * stride;
                auto storage = TestTensorFactory::createFP32({6, stride});
                std::fill_n(storage->mutable_data(), words, 0.0f);
                ASSERT_TRUE(storage->ensureOnDevice(device, stream));
                auto *base = static_cast<int *>(storage->gpu_data_ptr());
                for (unsigned int mask = 0; mask < 8; ++mask)
                {
                    SCOPED_TRACE("optional_mask=" + std::to_string(mask));
                    const MoEGroupingInitialization binding{
                        .expert_counts = base + 1,
                        .write_heads = (mask & 1) ? base + stride + 1 : nullptr,
                        .original_to_grouped = (mask & 2) ? base + 2 * stride + 1 : nullptr,
                        .original_expert_ids = (mask & 4) ? base + 3 * stride + 1 : nullptr,
                        .grouped_token_indices = base + 4 * stride + 1,
                        .grouped_weights = reinterpret_cast<float *>(base + 5 * stride + 1),
                        .num_experts = experts,
                        .total_slots = slots};
                    const auto enqueue = [&](MoEGroupingInitialization value, void *launch_stream) {
#ifdef HAVE_CUDA
                        if (device.is_cuda())
                            return cudaMoE_initialize_grouping(value, device.ordinal, launch_stream);
#endif
#ifdef HAVE_ROCM
                        if (device.is_rocm())
                            return hipMoE_initialize_grouping(value, device.ordinal, launch_stream);
#endif
                        return false;
                    };
                    ASSERT_TRUE(binding.valid());
                    EXPECT_FALSE(enqueue(binding, nullptr));
                    auto invalid = binding;
                    invalid.total_slots = 0;
                    EXPECT_FALSE(enqueue(invalid, stream));
                    invalid = binding;
                    invalid.expert_counts = nullptr;
                    EXPECT_FALSE(enqueue(invalid, stream));
                    ASSERT_TRUE(enqueue(binding, stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                    auto graph = context.createGraphCapture(stream);
                    ASSERT_NE(graph, nullptr);
                    {
                        ScopedBackendGraphCapture recording(context, *graph, "grouping initialization proof");
                        ASSERT_TRUE(recording.begin());
                        ASSERT_TRUE(enqueue(binding, stream));
                        recording.finish();
                    }
                    EXPECT_EQ(graph->nodeCount(), 1u);
                    std::vector<GPUGraphKernelNodeInfo> kernels;
                    ASSERT_TRUE(graph->inspectKernelNodes(kernels));
                    ASSERT_EQ(kernels.size(), 1u);
                    EXPECT_TRUE(kernels.front().valid());
                    EXPECT_EQ(kernels.front().local_memory_bytes_per_thread, 0u);
                    EXPECT_EQ(kernels.front().block_x, 256u);
                    EXPECT_EQ(kernels.front().grid_x,
                        static_cast<unsigned int>(1 + (binding.workItems() - 1) / 256));
                    ASSERT_TRUE(graph->instantiate());
                    for (unsigned int replay = 0; replay < 20; ++replay)
                    {
                        SCOPED_TRACE("replay=" + std::to_string(replay));
                        const uint32_t poison = 0x13579bdfu + replay;
                        std::vector<uint32_t> input(words, poison), expected(input), actual(words);
                        for (size_t bank = 0; bank < 6; ++bank)
                        {
                            if (bank >= 1 && bank <= 3 && !(mask & (1u << (bank - 1))))
                                continue;
                            const int extent = bank < 2 ? experts : slots;
                            const uint32_t fill = bank == 2 || bank == 3 ? 0xffffffffu : 0;
                            std::fill_n(expected.begin() + bank * stride + 1, extent, fill);
                        }
                        ASSERT_TRUE(backend->hostToDevice(storage->gpu_data_ptr(), input.data(),
                            words * sizeof(uint32_t), device.ordinal, stream));
                        ASSERT_TRUE(graph->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(), storage->gpu_data_ptr(),
                            words * sizeof(uint32_t), device.ordinal, stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                        EXPECT_EQ(actual, expected);
                    }
                }
            }
        });
    }

    /**
     * @test The production prefix bridge is exact across wave and block tails.
     *
     * One captured graph per geometry sees 20 changing publications. Guard
     * words prove short tails cannot overwrite adjacent storage; input bytes
     * prove cooperative staging never destroys counts needed by later stages.
     */
    TEST_P(MoERuntimeGrouping, CapturedStandalonePrefixIsExact)
    {
        const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            auto *backend = getBackendFor(device);
            auto *stream = context.defaultStream();
            ASSERT_NE(backend, nullptr);
            ASSERT_NE(stream, nullptr);
            for (const int experts : {1, 17, 31, 32, 33, 63, 64, 65,
                                      255, 256, 257, 511, 512, 513, 1023, 1024})
            {
                SCOPED_TRACE("experts=" + std::to_string(experts));
                const auto elements = static_cast<size_t>(experts + 2);
                auto counts = TestTensorFactory::createFP32({elements, 1});
                auto offsets = TestTensorFactory::createFP32({elements, 1});
                std::fill_n(counts->mutable_data(), elements, 0.0f);
                std::fill_n(offsets->mutable_data(), elements, 0.0f);
                ASSERT_TRUE(counts->ensureOnDevice(device, stream));
                ASSERT_TRUE(offsets->ensureOnDevice(device, stream));
                auto *input = static_cast<int *>(counts->gpu_data_ptr()) + 1;
                auto *output = static_cast<int *>(offsets->gpu_data_ptr()) + 1;
                const auto enqueue = [&] {
#ifdef HAVE_CUDA
                    if (device.is_cuda())
                        return cudaMoE_exclusive_scan(input, output, experts, device.ordinal, stream);
#endif
#ifdef HAVE_ROCM
                    if (device.is_rocm())
                        return hipMoE_exclusive_scan(input, output, experts, device.ordinal, stream);
#endif
                    return false;
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
                    constexpr int32_t guard = 0x13579bdf;
                    std::vector<int32_t> values(elements, guard), expected(elements, guard);
                    int running = 0;
                    for (int expert = 0; expert < experts; ++expert)
                    {
                        const int value = replay % 4 == 0 ? 0 :
                            replay % 4 == 1 ? (expert == experts - 1 ? 8191 : 0) :
                            replay % 4 == 2 ? (expert * 37 + replay) % 103 :
                            std::numeric_limits<int>::max() / experts;
                        values[expert + 1] = value;
                        expected[expert + 1] = running;
                        running += value;
                    }
                    std::vector<int32_t> poison(elements, guard), actual(elements), input_after(elements);
                    ASSERT_TRUE(backend->hostToDevice(counts->gpu_data_ptr(), values.data(),
                        elements * sizeof(int32_t), device.ordinal, stream));
                    ASSERT_TRUE(backend->hostToDevice(offsets->gpu_data_ptr(), poison.data(),
                        elements * sizeof(int32_t), device.ordinal, stream));
                    ASSERT_TRUE(graph->launch());
                    ASSERT_TRUE(backend->deviceToHost(actual.data(), offsets->gpu_data_ptr(),
                        elements * sizeof(int32_t), device.ordinal, stream));
                    ASSERT_TRUE(backend->deviceToHost(input_after.data(), counts->gpu_data_ptr(),
                        elements * sizeof(int32_t), device.ordinal, stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                    EXPECT_EQ(actual, expected);
                    EXPECT_EQ(input_after, values);
                }
            }
        });
    }

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
            for (int experts : {8, 17, 31, 32, 33, 63, 64, 65, 255, 256})
                // Cross both the compact-kernel boundary (256 route slots)
                // and partial wave/workgroup tails in the scalable scatter.
                // Odd top-k also prevents row alignment hiding a lane bug.
                for (const auto [rows, live_rows, top_k] :
                     std::vector<std::array<int, 3>>{
                         {1, 1, 8}, {2, 2, 8}, {16, 16, 8}, {32, 32, 8},
                         {33, 33, 8}, {63, 63, 8}, {64, 64, 8}, {65, 65, 8},
                         {129, 127, 8}, {512, 448, 8}, {512, 512, 8},
                         {2048, 2048, 8}, {257, 257, 1}, {65, 63, 7},
                         {33, 33, 16}, {512, 33, 8}, {4096, 65, 7}})
                {
                    SCOPED_TRACE("experts=" + std::to_string(experts) +
                        " rows=" + std::to_string(rows) +
                        " live_rows=" + std::to_string(live_rows) +
                        " top_k=" + std::to_string(top_k));
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
                    auto indices = TestTensorFactory::createFP32(
                        {static_cast<size_t>(rows), static_cast<size_t>(top_k)});
                    auto weights = TestTensorFactory::createFP32(
                        {static_cast<size_t>(rows), static_cast<size_t>(top_k)});
                    std::fill_n(indices->mutable_data(), slots, -1.0f);
                    std::fill_n(weights->mutable_data(), slots, 0.0f);
                    ASSERT_TRUE(indices->ensureOnDevice(device, stream));
                    ASSERT_TRUE(weights->ensureOnDevice(device, stream));
                    const auto enqueue = [&] {
                        return kernel->groupPrefillRoutes(runtime.deviceLayerState(0),
                            indices.get(), weights.get(), live_rows, rows, experts, top_k);
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
                            // No arithmetic belongs in grouping: preserve even
                            // the low mantissa bits of each selected weight.
                            probabilities[slot] = std::bit_cast<float>(
                                uint32_t{0x3e800000} + static_cast<uint32_t>(slot * 13 + replay));
                        }
                        std::vector<int32_t> expected_counts(experts), expected_offsets(experts);
                        std::vector<int32_t> expected_rows(slots), actual_rows(slots), actual_counts(experts), actual_offsets(experts);
                        std::vector<float> expected_weights(slots), actual_weights(slots);
                        int grouped = 0;
                        for (int expert = 0; expert < experts; ++expert)
                        {
                            expected_offsets[expert] = grouped;
                            for (int slot = 0; slot < live_rows * top_k; ++slot)
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
    INSTANTIATE_TEST_SUITE_P(Backends, MoEGroupingInitializationTest, ::testing::Values("CUDA", "ROCm"),
        [](const ::testing::TestParamInfo<std::string> &info) { return info.param; });
}
