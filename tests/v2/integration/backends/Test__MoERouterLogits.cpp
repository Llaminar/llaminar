/**
 * @file Test__MoERouterLogits.cpp
 * @brief Byte-exact captured router logits across serial-partition boundaries.
 *
 * Prepared Q8 router bytes are an internal representation, not a restriction
 * on the GGUF expert codebook. This test calls the same production bridges as
 * routing, with independent serial decode as the numerical oracle. Retained
 * graphs must survive changed input, empty/partial/full live counts, signed
 * zeros and K widths on both sides of the reduced-workgroup boundary. Native
 * node inspection also proves that an empty serial wave was actually removed;
 * a tolerance-only output test cannot detect that economy regression.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" bool hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
    const float *, int8_t *, float *, const int8_t *, const float *, float *,
    int, int, int, int, void *, const int *);
extern "C" bool hipMoE_gate_logits_single_token_q8_weights(
    const float *, int8_t *, float *, const int8_t *, const float *, float *,
    int, int, int, void *);

namespace llaminar2::test
{
    /**
     * @brief Throw a failed setup/observation through the context's owner thread.
     * @param condition Result of the exact backend operation being checked.
     * @param operation Stable diagnostic naming the failed lifecycle edge.
     */
    void requireRouterLogits(bool condition, const char *operation)
    {
        if (!condition) throw std::runtime_error(operation);
    }

    /** @brief Join observations before retiring graph bindings, even on failure. */
    struct RouterObservationJoin
    {
        IBackend &backend;
        DeviceId device;
        void *stream;

        /** @brief This test-only boundary drains any partially submitted replay. */
        ~RouterObservationJoin() { (void)backend.synchronizeStream(stream, device.ordinal); }
    };

    /** @brief One immutable captured geometry; live row count is device-owned. */
    struct RouterGeometry
    {
        int rows;     ///< Captured capacity, independent of the mutable live count.
        int width;    ///< Hidden columns; each Q8 block owns 32 consecutive values.
        int experts;  ///< Independent output columns in each row.
    };

    /**
     * @brief Authenticate launch geometry and every live/inactive output bit.
     * @param geometry Immutable capacity, hidden width and expert count.
     *
     * The scalar oracle uses its unmodified 128-thread reduction. Its Q8
     * workspace is independent of the grouped route, so an overwrite cannot
     * accidentally turn the expected answer into the candidate's answer.
     */
    void proveRouterLogits(RouterGeometry geometry)
    {
        const DeviceId device = DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&]
        {
            SCOPED_TRACE(::testing::Message() << "rows=" << geometry.rows
                << " K=" << geometry.width << " experts=" << geometry.experts);
            auto *backend = getBackendFor(device);
            void *stream = context.defaultStream();
            requireRouterLogits(backend && stream, "router logits exact context stream");
            const auto [rows, width, experts] = geometry;
            const size_t input_count = size_t(rows) * width;
            const size_t weight_count = size_t(experts) * width;
            const size_t output_count = size_t(rows + 1) * experts;
            auto hidden = TestTensorFactory::createFP32({input_count});
            auto scales = TestTensorFactory::createFP32({input_count / 32});
            auto serial_scales = TestTensorFactory::createFP32({input_count / 32});
            auto gate_scales = TestTensorFactory::createFP32({weight_count / 32});
            auto output = TestTensorFactory::createFP32({output_count});
            auto serial = TestTensorFactory::createFP32({size_t(rows) * experts});
            auto q8 = TestTensorFactory::createINT32({input_count / sizeof(int32_t)});
            auto serial_q8 = TestTensorFactory::createINT32({input_count / sizeof(int32_t)});
            auto gate_q8 = TestTensorFactory::createINT32({weight_count / sizeof(int32_t)});
            auto live_rows = TestTensorFactory::createINT32({1});
            for (auto *tensor : {hidden.get(), scales.get(), serial_scales.get(),
                                gate_scales.get(), output.get(), serial.get()})
            {
                std::fill_n(tensor->mutable_data(), tensor->numel(), 0.f);
                requireRouterLogits(tensor->ensureOnDevice(device, stream), "router logits float storage");
            }
            for (auto *tensor : {q8.get(), serial_q8.get(), gate_q8.get(), live_rows.get()})
            {
                std::fill_n(tensor->mutable_int32_data(), tensor->numel(), 0);
                requireRouterLogits(tensor->ensureOnDevice(device, stream), "router logits integer storage");
            }
            const auto grouped = [&]
            {
                return hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
                    static_cast<const float *>(hidden->gpu_data_ptr()),
                    static_cast<int8_t *>(q8->gpu_data_ptr()),
                    static_cast<float *>(scales->gpu_data_ptr()),
                    static_cast<const int8_t *>(gate_q8->gpu_data_ptr()),
                    static_cast<const float *>(gate_scales->gpu_data_ptr()),
                    static_cast<float *>(output->gpu_data_ptr()),
                    rows, width, experts, device.ordinal, stream,
                    static_cast<const int *>(live_rows->gpu_data_ptr()));
            };
            auto graph = context.createGraphCapture(stream);
            requireRouterLogits(graph != nullptr, "router logits graph owner");
            std::vector<float> input(input_count), weight_scales(weight_count / 32);
            std::vector<int8_t> weights(weight_count);
            std::vector<float> expected(size_t(rows) * experts), actual(output_count);
            // Declared after all asynchronous bindings: an assertion or throw
            // joins before graph, device tensors or host staging can disappear.
            RouterObservationJoin join{*backend, device, stream};
            requireRouterLogits(grouped() && backend->synchronizeStream(stream, device.ordinal),
                "router logits warmup");
            {
                ScopedBackendGraphCapture recording(context, *graph, "router logits partition proof");
                requireRouterLogits(recording.begin(), "router logits begin capture");
                requireRouterLogits(grouped(), "router logits captured body");
                recording.finish();
            }
            ASSERT_EQ(graph->nodeCount(), 2u);
            std::vector<GPUGraphKernelNodeInfo> nodes;
            ASSERT_TRUE(graph->inspectKernelNodes(nodes));
            ASSERT_EQ(nodes.size(), 2u);
            const auto logits_node = std::find_if(nodes.begin(), nodes.end(), [](const auto &node)
            {
                return node.name.find("gate_logits_q8_grouped_verifier_kernel") != std::string::npos;
            });
            ASSERT_NE(logits_node, nodes.end());
            EXPECT_EQ(logits_node->block_x, width <= 2048 ? 64u : 128u);
            EXPECT_EQ(logits_node->local_memory_bytes_per_thread, 0u);
            EXPECT_EQ(logits_node->grid_x, static_cast<unsigned>(experts));
            EXPECT_EQ(logits_node->grid_y, static_cast<unsigned>((rows + 15) / 16));
            requireRouterLogits(graph->instantiate(), "router logits instantiate");

            std::vector<int> counts{rows, 0, rows / 2};
            if (rows == 65)
                for (int count = 1; count <= rows; ++count) counts.push_back(count);
            else
                for (int count : {1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 447, 448, 449, rows - 1})
                    if (count <= rows) counts.push_back(count);
            counts.insert(counts.end(), {rows, rows + 17, -3, rows});
            constexpr uint32_t poison = 0x7fc01234u;
            for (int pattern = 0; pattern < 3; ++pattern)
            {
                SCOPED_TRACE(::testing::Message() << "pattern=" << pattern);
                for (size_t i = 0; i < input.size(); ++i)
                {
                    const float magnitude = std::ldexp(1.f, int((i / 32 + pattern) % 5) * 3 - 10);
                    input[i] = pattern == 2 && i / 32 % 3 == 0 ? (i % 2 ? -0.f : 0.f)
                        : magnitude * std::sin(float(i + 19 * pattern) * .021f);
                }
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = static_cast<int8_t>(int((i * 71 + pattern * 17) % 255) - 127);
                for (size_t i = 0; i < weight_scales.size(); ++i)
                    weight_scales[i] = std::ldexp(float(1 + (i + pattern) % 7), -12);
                requireRouterLogits(backend->hostToDevice(hidden->gpu_data_ptr(), input.data(),
                    input.size() * sizeof(float), device.ordinal, stream) &&
                    backend->hostToDevice(gate_q8->gpu_data_ptr(), weights.data(), weights.size(), device.ordinal, stream) &&
                    backend->hostToDevice(gate_scales->gpu_data_ptr(), weight_scales.data(),
                        weight_scales.size() * sizeof(float), device.ordinal, stream), "router logits changed inputs");
                for (int row = 0; row < rows; ++row)
                    requireRouterLogits(hipMoE_gate_logits_single_token_q8_weights(
                        static_cast<const float *>(hidden->gpu_data_ptr()) + size_t(row) * width,
                        static_cast<int8_t *>(serial_q8->gpu_data_ptr()) + size_t(row) * width,
                        static_cast<float *>(serial_scales->gpu_data_ptr()) + size_t(row) * width / 32,
                        static_cast<const int8_t *>(gate_q8->gpu_data_ptr()),
                        static_cast<const float *>(gate_scales->gpu_data_ptr()),
                        static_cast<float *>(serial->gpu_data_ptr()) + size_t(row) * experts,
                        width, experts, device.ordinal, stream), "router logits serial oracle");
                requireRouterLogits(backend->deviceToHost(expected.data(), serial->gpu_data_ptr(),
                    expected.size() * sizeof(float), device.ordinal, stream) &&
                    backend->synchronizeStream(stream, device.ordinal), "router logits serial observation");
                for (const int count : counts)
                {
                    SCOPED_TRACE(::testing::Message() << "live=" << count);
                    std::fill(actual.begin(), actual.end(), std::bit_cast<float>(poison));
                    requireRouterLogits(backend->hostToDevice(output->gpu_data_ptr(), actual.data(),
                        actual.size() * sizeof(float), device.ordinal, stream) &&
                        backend->hostToDevice(live_rows->gpu_data_ptr(), &count, sizeof(count), device.ordinal, stream),
                        "router logits retained reset");
                    requireRouterLogits(graph->launch(), "router logits retained replay");
                    requireRouterLogits(backend->deviceToHost(actual.data(), output->gpu_data_ptr(),
                        actual.size() * sizeof(float), device.ordinal, stream) &&
                        backend->synchronizeStream(stream, device.ordinal), "router logits grouped observation");
                    const size_t live_outputs = size_t(std::clamp(count, 0, rows)) * experts;
                    for (size_t i = 0; i < actual.size(); ++i)
                        ASSERT_EQ(std::bit_cast<uint32_t>(actual[i]),
                            i < live_outputs ? std::bit_cast<uint32_t>(expected[i]) : poison) << "element=" << i;
                }
            }
        });
    }

    /** @test Every small live M, physical tail, K partition and reset remains exact. */
    TEST(MoERouterLogits, ROCmCapturedSerialPartitions)
    {
        for (int width : {32, 96, 2016, 2048, 2080, 3072, 4096, 4128, 8192})
            proveRouterLogits({.rows = 65, .width = width, .experts = 17});
        for (int width : {2048, 3072})
            proveRouterLogits({.rows = 512, .width = width, .experts = 256});
    }
} // namespace llaminar2::test
