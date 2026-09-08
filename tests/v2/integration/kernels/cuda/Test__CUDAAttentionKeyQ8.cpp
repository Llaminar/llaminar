/**
 * @file Test__CUDAAttentionKeyQ8.cpp
 * @brief Real-device CUDA conformance tests for attention-key Q8 kernels.
 *
 * The suite proves scalar-byte equivalence at every supported head width,
 * exact decoded output, explicit-stream validation, and retained graph replay.
 * Host synchronization is confined to test result collection.
 */

#include <gtest/gtest.h>

#include "../AttentionKeyQ8DeviceTestCommon.h"
#include "kernels/cuda/kvcache/CUDAAttentionKeyQ8Kernels.h"

#include <cuda_runtime.h>

#include <bit>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Return whether at least one usable CUDA participant exists. */
        bool hasCUDADevice()
        {
            int count = 0;
            return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
        }

        /**
         * @brief Run one width through eager or captured encode/decode and compare bytes.
         *
         * @tparam D Attention-head width.
         * @param captured Whether both production launches execute from one CUDA graph.
         */
        template <int D>
        void runCUDAConformance(bool captured)
        {
            constexpr int kBlockCount = 19;
            const std::vector<float> host_input =
                test::makeAttentionKeyQ8DeviceInput<D>(kBlockCount);
            const auto expected_blocks =
                test::makeAttentionKeyQ8ReferenceBlocks<D>(host_input);
            const std::vector<float> expected_decoded =
                test::makeAttentionKeyQ8ReferenceDecoded<D>(expected_blocks);

            cudaStream_t stream = nullptr;
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            float *device_input = nullptr;
            float *device_decoded = nullptr;
            AttentionKeyQ8Block<D> *device_blocks = nullptr;
            ASSERT_EQ(cudaMalloc(&device_input, host_input.size() * sizeof(float)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&device_decoded, host_input.size() * sizeof(float)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&device_blocks, expected_blocks.size() * sizeof(expected_blocks[0])),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          device_input,
                          host_input.data(),
                          host_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice,
                          stream),
                      cudaSuccess);

            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            if (captured)
            {
                ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                          cudaSuccess);
            }
            ASSERT_TRUE(cudaAttentionKeyQ8Quantize(
                device_input, device_blocks, kBlockCount, D, stream));
            ASSERT_TRUE(cudaAttentionKeyQ8Dequantize(
                device_blocks, device_decoded, kBlockCount, D, stream));
            if (captured)
            {
                ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
                size_t node_count = 0;
                ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &node_count), cudaSuccess);
                EXPECT_EQ(node_count, 2U);
                ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                          cudaSuccess);
                ASSERT_EQ(cudaGraphLaunch(executable, stream), cudaSuccess);
            }

            std::vector<AttentionKeyQ8Block<D>> actual_blocks(kBlockCount);
            std::vector<float> actual_decoded(host_input.size());
            ASSERT_EQ(cudaMemcpyAsync(
                          actual_blocks.data(),
                          device_blocks,
                          actual_blocks.size() * sizeof(actual_blocks[0]),
                          cudaMemcpyDeviceToHost,
                          stream),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(
                          actual_decoded.data(),
                          device_decoded,
                          actual_decoded.size() * sizeof(float),
                          cudaMemcpyDeviceToHost,
                          stream),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

            if (std::memcmp(
                    actual_blocks.data(),
                    expected_blocks.data(),
                    actual_blocks.size() * sizeof(actual_blocks[0])) != 0)
            {
                for (size_t block = 0; block < actual_blocks.size(); ++block)
                {
                    if (std::bit_cast<uint32_t>(actual_blocks[block].quadratic_scale) !=
                        std::bit_cast<uint32_t>(expected_blocks[block].quadratic_scale))
                    {
                        ADD_FAILURE()
                            << "first CUDA scale mismatch head_dim=" << D
                            << " block=" << block
                            << " actual_bits=0x" << std::hex
                            << std::bit_cast<uint32_t>(actual_blocks[block].quadratic_scale)
                            << " expected_bits=0x"
                            << std::bit_cast<uint32_t>(expected_blocks[block].quadratic_scale);
                        break;
                    }
                    for (int coordinate = 0; coordinate < D; ++coordinate)
                    {
                        if (actual_blocks[block].codes[coordinate] !=
                            expected_blocks[block].codes[coordinate])
                        {
                            ADD_FAILURE()
                                << "first CUDA code mismatch head_dim=" << D
                                << " block=" << block
                                << " coordinate=" << coordinate
                                << " actual="
                                << static_cast<int>(actual_blocks[block].codes[coordinate])
                                << " expected="
                                << static_cast<int>(expected_blocks[block].codes[coordinate]);
                            block = actual_blocks.size();
                            break;
                        }
                    }
                }
            }

            EXPECT_EQ(
                std::memcmp(
                    actual_blocks.data(),
                    expected_blocks.data(),
                    actual_blocks.size() * sizeof(actual_blocks[0])),
                0)
                << "CUDA physical bytes differ for head_dim=" << D;
            EXPECT_EQ(
                std::memcmp(
                    actual_decoded.data(),
                    expected_decoded.data(),
                    actual_decoded.size() * sizeof(float)),
                0)
                << "CUDA decoded FP32 bytes differ for head_dim=" << D;

            if (executable)
            {
                EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
            }
            if (graph)
            {
                EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
            }
            EXPECT_EQ(cudaFree(device_blocks), cudaSuccess);
            EXPECT_EQ(cudaFree(device_decoded), cudaSuccess);
            EXPECT_EQ(cudaFree(device_input), cudaSuccess);
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
    } // namespace

    TEST(Test__CUDAAttentionKeyQ8, AllHeadWidthsMatchScalarPhysicalAndDecodedBytes)
    {
        if (!hasCUDADevice())
        {
            GTEST_SKIP() << "No CUDA device available";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        runCUDAConformance<64>(false);
        runCUDAConformance<128>(false);
        runCUDAConformance<256>(false);
    }

    TEST(Test__CUDAAttentionKeyQ8, CapturedEncodeDecodeGraphReplaysExactBytes)
    {
        if (!hasCUDADevice())
        {
            GTEST_SKIP() << "No CUDA device available";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        runCUDAConformance<64>(true);
        runCUDAConformance<128>(true);
        runCUDAConformance<256>(true);
    }

    TEST(Test__CUDAAttentionKeyQ8, RejectsDefaultStreamAndUnsupportedWidth)
    {
        const auto *input = reinterpret_cast<const float *>(uintptr_t{1});
        auto *output = reinterpret_cast<void *>(uintptr_t{1});
        EXPECT_FALSE(cudaAttentionKeyQ8Quantize(input, output, 1, 64, nullptr));
        EXPECT_FALSE(cudaAttentionKeyQ8Dequantize(output, const_cast<float *>(input), 1, 64, nullptr));

        if (!hasCUDADevice())
        {
            GTEST_SKIP() << "No CUDA device available for explicit-stream validation";
        }
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        EXPECT_FALSE(cudaAttentionKeyQ8Quantize(input, output, 1, 96, stream));
        EXPECT_FALSE(cudaAttentionKeyQ8Dequantize(output, const_cast<float *>(input), 1, 96, stream));
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
} // namespace llaminar2
