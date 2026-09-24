/**
 * @file Test__ROCmAttentionKeyQ8.cpp
 * @brief Real-device ROCm conformance tests for attention-key Q8 kernels.
 *
 * The suite mirrors CUDA: all supported head widths must produce scalar-exact
 * physical and decoded bytes, reject the default stream, and replay as one
 * retained HIP graph. Host waits exist only at the test evidence boundary.
 */

#include <gtest/gtest.h>

#include "../AttentionKeyQ8DeviceTestCommon.h"
#include "kernels/rocm/kvcache/ROCmAttentionKeyQ8Kernels.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstring>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Return whether at least one usable ROCm participant exists. */
        bool hasROCmDevice()
        {
            int count = 0;
            return hipGetDeviceCount(&count) == hipSuccess && count > 0;
        }

        /** @brief Run one head width through eager or captured HIP conformance. */
        template <int D>
        void runROCmConformance(bool captured)
        {
            constexpr int kBlockCount = 19;
            const std::vector<float> host_input =
                test::makeAttentionKeyQ8DeviceInput<D>(kBlockCount);
            const auto expected_blocks =
                test::makeAttentionKeyQ8ReferenceBlocks<D>(host_input);
            const std::vector<float> expected_decoded =
                test::makeAttentionKeyQ8ReferenceDecoded<D>(expected_blocks);

            hipStream_t stream = nullptr;
            ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
            float *device_input = nullptr;
            float *device_decoded = nullptr;
            AttentionKeyQ8Block<D> *device_blocks = nullptr;
            ASSERT_EQ(hipMalloc(&device_input, host_input.size() * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&device_decoded, host_input.size() * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&device_blocks, expected_blocks.size() * sizeof(expected_blocks[0])),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                          device_input,
                          host_input.data(),
                          host_input.size() * sizeof(float),
                          hipMemcpyHostToDevice,
                          stream),
                      hipSuccess);

            hipGraph_t graph = nullptr;
            hipGraphExec_t executable = nullptr;
            if (captured)
            {
                ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
                          hipSuccess);
            }
            ASSERT_TRUE(rocmAttentionKeyQ8Quantize(
                device_input, device_blocks, kBlockCount, D, stream));
            ASSERT_TRUE(rocmAttentionKeyQ8Dequantize(
                device_blocks, device_decoded, kBlockCount, D, stream));
            if (captured)
            {
                ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
                size_t node_count = 0;
                ASSERT_EQ(hipGraphGetNodes(graph, nullptr, &node_count), hipSuccess);
                EXPECT_EQ(node_count, 2U);
                ASSERT_EQ(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                          hipSuccess);
                ASSERT_EQ(hipGraphLaunch(executable, stream), hipSuccess);
            }

            std::vector<AttentionKeyQ8Block<D>> actual_blocks(kBlockCount);
            std::vector<float> actual_decoded(host_input.size());
            ASSERT_EQ(hipMemcpyAsync(
                          actual_blocks.data(),
                          device_blocks,
                          actual_blocks.size() * sizeof(actual_blocks[0]),
                          hipMemcpyDeviceToHost,
                          stream),
                      hipSuccess);
            ASSERT_EQ(hipMemcpyAsync(
                          actual_decoded.data(),
                          device_decoded,
                          actual_decoded.size() * sizeof(float),
                          hipMemcpyDeviceToHost,
                          stream),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

            EXPECT_EQ(
                std::memcmp(
                    actual_blocks.data(),
                    expected_blocks.data(),
                    actual_blocks.size() * sizeof(actual_blocks[0])),
                0)
                << "ROCm physical bytes differ for head_dim=" << D;
            EXPECT_EQ(
                std::memcmp(
                    actual_decoded.data(),
                    expected_decoded.data(),
                    actual_decoded.size() * sizeof(float)),
                0)
                << "ROCm decoded FP32 bytes differ for head_dim=" << D;

            if (executable)
            {
                EXPECT_EQ(hipGraphExecDestroy(executable), hipSuccess);
            }
            if (graph)
            {
                EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
            }
            EXPECT_EQ(hipFree(device_blocks), hipSuccess);
            EXPECT_EQ(hipFree(device_decoded), hipSuccess);
            EXPECT_EQ(hipFree(device_input), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        }
    } // namespace

    TEST(Test__ROCmAttentionKeyQ8, AllHeadWidthsMatchScalarPhysicalAndDecodedBytes)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        runROCmConformance<64>(false);
        runROCmConformance<128>(false);
        runROCmConformance<256>(false);
    }

    TEST(Test__ROCmAttentionKeyQ8, CapturedEncodeDecodeGraphReplaysExactBytes)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        runROCmConformance<64>(true);
        runROCmConformance<128>(true);
        runROCmConformance<256>(true);
    }

    TEST(Test__ROCmAttentionKeyQ8, RejectsDefaultStreamAndUnsupportedWidth)
    {
        const auto *input = reinterpret_cast<const float *>(uintptr_t{1});
        auto *output = reinterpret_cast<void *>(uintptr_t{1});
        EXPECT_FALSE(rocmAttentionKeyQ8Quantize(input, output, 1, 64, nullptr));
        EXPECT_FALSE(rocmAttentionKeyQ8Dequantize(output, const_cast<float *>(input), 1, 64, nullptr));

        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available for explicit-stream validation";
        }
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        EXPECT_FALSE(rocmAttentionKeyQ8Quantize(input, output, 1, 96, stream));
        EXPECT_FALSE(rocmAttentionKeyQ8Dequantize(output, const_cast<float *>(input), 1, 96, stream));
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    }
} // namespace llaminar2
