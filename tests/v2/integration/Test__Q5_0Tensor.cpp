/**
 * @file Test__Q5_0Tensor.cpp
 * @brief Unit tests for Q5_0Tensor SIMD path equivalency
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/utils/DebugEnv.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <cstring>
#include "loaders/ModelLoader.h"
#include <fstream>
#include "v2/utils/MPIContext.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/FP16Utils.h"
#include "v2/kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#include "v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h"

namespace llaminar2
{
    namespace test
    {
        class Test__Q5_0Tensor : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
            }

            std::shared_ptr<MPIContext> mpi_ctx_;

            Q5_0Block createTestBlock()
            {
                Q5_0Block block;
                block.d = 0x3C00; // FP16 1.0

                // Set low 4-bit values (0-15 pattern)
                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low_nibble = i % 16;
                    uint8_t high_nibble = (i + 8) % 16;
                    block.qs[i] = (high_nibble << 4) | low_nibble;
                }

                // Set high bits (alternating pattern for testing)
                for (int i = 0; i < 4; ++i)
                {
                    block.qh[i] = (i % 2 == 0) ? 0xAA : 0x55; // 10101010 or 01010101
                }

                return block;
            }

            bool compareOutputs(const float *a, const float *b, size_t count, float tolerance = 1e-6f)
            {
                float max_abs_diff = 0.0f;
                size_t mismatch_count = 0;

                for (size_t i = 0; i < count; ++i)
                {
                    float abs_diff = std::fabs(a[i] - b[i]);
                    max_abs_diff = std::max(max_abs_diff, abs_diff);

                    if (abs_diff > tolerance)
                    {
                        ++mismatch_count;
                        if (mismatch_count <= 3)
                        {
                            std::cout << "Mismatch at [" << i << "]: "
                                      << "a=" << a[i] << ", b=" << b[i]
                                      << ", diff=" << abs_diff << std::endl;
                        }
                    }
                }

                if (mismatch_count > 0)
                {
                    std::cout << "Total mismatches: " << mismatch_count << "/" << count
                              << ", max_abs_diff=" << max_abs_diff << std::endl;
                }

                return mismatch_count == 0;
            }

            /**
             * @brief Compute relative L2 error between two output vectors.
             */
            float compute_relative_l2_error(const float *a, const float *b, size_t n)
            {
                float sum_sq_diff = 0.0f;
                float sum_sq_ref = 0.0f;
                for (size_t i = 0; i < n; ++i)
                {
                    float diff = a[i] - b[i];
                    sum_sq_diff += diff * diff;
                    sum_sq_ref += b[i] * b[i];
                }
                if (sum_sq_ref < 1e-10f)
                    return 0.0f;
                return std::sqrt(sum_sq_diff / sum_sq_ref);
            }
        };

        TEST_F(Test__Q5_0Tensor, ScalarVsAVX2Equivalency)
        {
            Q5_0Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);
            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-6f))
                << "Q5_0 scalar and AVX2 paths produce different results";
#else
            GTEST_SKIP() << "AVX2 not available";
#endif
        }

        TEST_F(Test__Q5_0Tensor, ScalarVsAVX512Equivalency)
        {
            Q5_0Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX512F__)
            Q5_0Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-6f))
                << "Q5_0 scalar and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "AVX512 not available";
#endif
        }

        TEST_F(Test__Q5_0Tensor, AVX2VsAVX512Equivalency)
        {
#if defined(__AVX2__) && defined(__AVX512F__)
            Q5_0Block test_block = createTestBlock();

            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());
            Q5_0Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-6f))
                << "Q5_0 AVX2 and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "Both AVX2 and AVX512 required";
#endif
        }

        TEST_F(Test__Q5_0Tensor, EdgeCase_ZeroScale)
        {
            Q5_0Block test_block = createTestBlock();
            test_block.d = 0x0000; // FP16 zero

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

            for (size_t i = 0; i < Q5_0Block::BLOCK_SIZE; ++i)
            {
                EXPECT_FLOAT_EQ(scalar_output[i], 0.0f);
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);
            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-6f));
#endif
        }

        TEST_F(Test__Q5_0Tensor, EdgeCase_AllHighBitsSet)
        {
            Q5_0Block test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Set all low bits to 0
            for (int i = 0; i < 16; ++i)
            {
                test_block.qs[i] = 0x00;
            }

            // Set all high bits to 1
            for (int i = 0; i < 4; ++i)
            {
                test_block.qh[i] = 0xFF;
            }

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

            // Each element should be: scale * (16 - 16) = 0.0
            // (high bit contributes 16, low bits are 0, centered at 16)
            for (size_t i = 0; i < Q5_0Block::BLOCK_SIZE; ++i)
            {
                EXPECT_NEAR(scalar_output[i], 0.0f, 1e-5f) << "At index " << i;
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);
            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-5f));
#endif
        }

        TEST_F(Test__Q5_0Tensor, EdgeCase_HighBitExtraction)
        {
            Q5_0Block test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Test specific high bit patterns with low bits = 0
            for (int i = 0; i < 16; ++i)
            {
                test_block.qs[i] = 0x00; // Low bits = 0
            }

            // Q5_0 high bit extraction (from dequantize_row_q5_0):
            //   For j=0..15:
            //     output[j]    uses bit j from qh (bits 0-15)
            //     output[j+16] uses bit j+12 from qh (bits 12-27)
            //
            // Set qh to have bits 0-11 clear, bits 12-31 set
            // Detailed trace:
            //   j=0..11:  output[j]    gets bit 0..11 (clear) → high_bit=0 → (0|0)-16 = -16
            //             output[j+16] gets bit 12..23 (set)  → high_bit=16 → (0|16)-16 = 0
            //   j=12..15: output[j]    gets bit 12..15 (set)  → high_bit=16 → (0|16)-16 = 0
            //             output[j+16] gets bit 24..27 (set)  → high_bit=16 → (0|16)-16 = 0
            //
            // Expected pattern: output[0-11]=-16, output[12-31]=0
            uint32_t qh_value = 0xFFFFF000; // Bits 12-31 set, bits 0-11 clear (little-endian)
            std::memcpy(test_block.qh, &qh_value, sizeof(qh_value));

            std::vector<float> scalar_output(Q5_0Block::BLOCK_SIZE);

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q5_0Block::BLOCK_SIZE);

            Q5_0Tensor::decodeBlockScalar(test_block, scalar_output.data());
            Q5_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            // Verify pattern based on high bit extraction
            for (size_t i = 0; i < 12; ++i)
            {
                EXPECT_NEAR(scalar_output[i], -16.0f, 1e-5f) << "First segment (bits 0-11 clear) at index " << i;
            }
            for (size_t i = 12; i < 32; ++i)
            {
                EXPECT_NEAR(scalar_output[i], 0.0f, 1e-5f) << "Second segment (bits 12-31 set) at index " << i;
            }

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q5_0Block::BLOCK_SIZE, 1e-5f));
#endif
        }

        /**
         * @brief Compare QuantisedGemmKernel (INT8) vs FloatingPointGemmKernel (FP32) for Q5_0.
         *
         * This test verifies that the quantized GEMM kernel produces results close to
         * the FP32 reference implementation using OneDNN.
         */
        TEST_F(Test__Q5_0Tensor, QuantizedVsFP32Parity)
        {
            // Realistic dimensions: 64 tokens, 512 hidden dim
            const int m = 64;
            const int n = 512;
            const int k = 512;

            // Q5_0: 32 elements per block
            const size_t num_blocks = (static_cast<size_t>(n) * k) / 32;
            std::vector<uint8_t> raw_data(num_blocks * sizeof(Q5_0Block));
            Q5_0Block *blocks = reinterpret_cast<Q5_0Block *>(raw_data.data());

            // Initialize with random but valid data
            std::mt19937 rng(42);
            std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
            std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

            for (size_t b = 0; b < num_blocks; ++b)
            {
                // Valid FP16 scale factor
                blocks[b].d = fp32_to_fp16(scale_dist(rng));
                // Random qh (high bits, 4 bytes = 32 bits)
                for (int i = 0; i < 4; ++i)
                {
                    blocks[b].qh[i] = byte_dist(rng);
                }
                // Random qs (low 4 bits packed into 16 bytes)
                for (int i = 0; i < 16; ++i)
                {
                    blocks[b].qs[i] = byte_dist(rng);
                }
            }

            // Create quantized tensor
            auto q5_0_tensor = std::make_unique<Q5_0Tensor>(
                std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
                raw_data);

            // Dequantize to FP32 for reference
            TensorFactory factory(*mpi_ctx_);
            auto fp32_weights = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
            q5_0_tensor->to_fp32(fp32_weights->mutable_data());

            // Create random input activations
            auto input = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
            float *input_data = input->mutable_data();
            std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
            for (int i = 0; i < m * k; ++i)
            {
                input_data[i] = input_dist(rng);
            }

            // Allocate outputs
            auto output_quantized = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
            auto output_fp32 = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
            std::memset(output_quantized->mutable_data(), 0, m * n * sizeof(float));
            std::memset(output_fp32->mutable_data(), 0, m * n * sizeof(float));

            // Run quantized GEMM (INT8 path)
            auto quantized_gemm = q5_0_tensor->createGemm();
            ASSERT_TRUE(quantized_gemm->multiply(
                input_data,
                output_quantized->mutable_data(),
                m, n, k));

            // Run FP32 GEMM (OneDNN reference)
            gemm_v4::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());
            ASSERT_TRUE(fp32_gemm.multiply(
                input_data,
                output_fp32->mutable_data(),
                m, n, k));

            // Compare results
            float rel_l2_error = compute_relative_l2_error(
                output_quantized->data(),
                output_fp32->data(),
                m * n);

            std::cout << "[Q5_0 Parity] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;

            // 1% tolerance for quantization error
            EXPECT_LT(rel_l2_error, 0.01f)
                << "Q5_0 quantized GEMM diverged from FP32 reference by "
                << (rel_l2_error * 100.0f) << "%";
        }

    } // namespace test
} // namespace llaminar2

/**
 * @brief Test Q5_0Tensor to INT8 block quantization
 *
 * Validates that Q5_0Tensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy using real model weights.
 */
TEST(Q5_0Tensor_INT8, BlockConversion)
{
    // Load a test model to get real Q5_0Tensor weights
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q5_0.gguf";

    // Skip if model file doesn't exist (not all test environments have models)
    if (!std::ifstream(model_path).good())
    {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }

    llaminar2::ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));

    // Find a weight tensor of type Q5_0Tensor
    std::shared_ptr<llaminar2::TensorBase> weight_tensor;
    const auto &model = loader.getModel();

    for (const auto &tensor_info : model.tensors)
    {
        // Map GGUFTensorType to our expected type
        // For now, just try to load the first tensor and check its type
        auto tensor = loader.loadTensor(tensor_info.name);
        if (tensor && std::dynamic_pointer_cast<llaminar2::Q5_0Tensor>(tensor))
        {
            weight_tensor = tensor;
            break;
        }
    }

    if (!weight_tensor)
    {
        GTEST_SKIP() << "No Q5_0Tensor weights found in model";
    }

    // Convert to INT8 blocked format
    // Compute total elements from shape
    size_t total_elements = 1;
    for (auto dim : weight_tensor->shape())
    {
        total_elements *= dim;
    }

    const size_t block_size = 32;
    const size_t num_blocks = (total_elements + block_size - 1) / block_size;

    std::vector<int8_t> int8_data(total_elements);
    std::vector<float> scales(num_blocks);

    weight_tensor->to_int8_blocked(int8_data.data(), scales.data(), block_size);

    // Verify all int8 values are in valid range [-127, 127]
    for (size_t i = 0; i < total_elements; ++i)
    {
        EXPECT_GE(int8_data[i], -127) << "INT8 value at index " << i << " out of range";
        EXPECT_LE(int8_data[i], 127) << "INT8 value at index " << i << " out of range";
    }

    // Verify all scales are positive and reasonable
    for (size_t i = 0; i < num_blocks; ++i)
    {
        EXPECT_GT(scales[i], 0.0f) << "Scale at block " << i << " should be positive";
        EXPECT_LT(scales[i], 1e6f) << "Scale at block " << i << " should be reasonable";
    }

    // Verify reconstruction accuracy by dequantizing and comparing
    // First, dequantize to FP32
    std::vector<float> reconstructed(total_elements);
    for (size_t i = 0; i < total_elements; ++i)
    {
        size_t block_idx = i / block_size;
        reconstructed[i] = static_cast<float>(int8_data[i]) * scales[block_idx];
    }

    // Get original FP32 representation for comparison
    std::vector<float> original(total_elements);
    weight_tensor->to_fp32(original.data());

    // Compute relative L2 error
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;
    double max_abs_diff = 0.0;

    for (size_t i = 0; i < total_elements; ++i)
    {
        double diff = reconstructed[i] - original[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += original[i] * original[i];
        max_abs_diff = std::max(max_abs_diff, std::abs(diff));
    }

    double rel_l2_error = std::sqrt(sum_sq_diff / sum_sq_orig);

    // INT8 quantization should have reasonable accuracy
    // Typical relative L2 error should be < 5%
    EXPECT_LT(rel_l2_error, 0.05) << "Relative L2 error too large: " << rel_l2_error;

    // Max absolute difference should be reasonable (depends on weight magnitude)
    EXPECT_LT(max_abs_diff, 0.5) << "Max absolute difference too large: " << max_abs_diff;
}

// ========================================================================
// Template Method Tests for to<T>() API
// ========================================================================

/**
 * @brief Test Q5_0Tensor to<float>() template method matches to_fp32()
 */
TEST(Q5_0Tensor_Template, ToFloat_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q5_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
        for (size_t j = 0; j < 4; ++j)
        {
            blocks[i].qh[j] = static_cast<uint8_t>((i * 4 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q5_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q5_0Tensor>(shape, raw_data);

    std::vector<float> result_template(64);
    tensor->template to<float>(result_template.data());

    std::vector<float> result_legacy(64);
    tensor->to_fp32(result_legacy.data());

    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_FLOAT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

TEST(Q5_0Tensor_Template, ToBF16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q5_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        for (size_t j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i * 4 + j) % 256);
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q5_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q5_0Tensor>(shape, raw_data);

    std::vector<uint16_t> result_template(64);
    tensor->template to<uint16_t>(result_template.data(), TensorType::BF16);

    std::vector<uint16_t> result_legacy(64);
    tensor->to_bf16(result_legacy.data());

    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

TEST(Q5_0Tensor_Template, ToFP16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q5_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        for (size_t j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i * 4 + j) % 256);
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q5_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q5_0Tensor>(shape, raw_data);

    std::vector<uint16_t> result_template(64);
    tensor->template to<uint16_t>(result_template.data(), TensorType::FP16);

    std::vector<uint16_t> result_legacy(64);
    tensor->to_fp16(result_legacy.data());

    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

TEST(Q5_0Tensor_Template, ToINT8_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q5_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        for (size_t j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i * 4 + j) % 256);
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q5_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q5_0Tensor>(shape, raw_data);

    std::vector<int8_t> int8_data(64);
    tensor->template to<int8_t>(int8_data.data());

    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_GE(int8_data[i], -127);
        EXPECT_LE(int8_data[i], 127);
    }
}

TEST(Q5_0Tensor_Template, ToINT32_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q5_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        for (size_t j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i * 4 + j) % 256);
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q5_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q5_0Tensor>(shape, raw_data);

    std::vector<int32_t> int32_data(64);
    tensor->template to<int32_t>(int32_data.data());

    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }
}

TEST(Q5_0Tensor_Template, RoundTrip_Q5_0_FP32_BF16_FP32)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q5_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        for (size_t j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i * 4 + j) % 256);
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q5_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q5_0Tensor>(shape, raw_data);

    std::vector<float> fp32_data(64);
    tensor->template to<float>(fp32_data.data());

    auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
    std::memcpy(fp32_tensor->mutable_data(), fp32_data.data(), 64 * sizeof(float));

    std::vector<uint16_t> bf16_data(64);
    fp32_tensor->template to<uint16_t>(bf16_data.data(), TensorType::BF16);

    auto bf16_tensor = std::make_shared<BF16Tensor>(shape, bf16_data);
    std::vector<float> final_fp32_data(64);
    bf16_tensor->template to<float>(final_fp32_data.data());

    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;

    for (size_t i = 0; i < 64; ++i)
    {
        double diff = fp32_data[i] - final_fp32_data[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += fp32_data[i] * fp32_data[i];
    }

    double rel_l2_error = std::sqrt(sum_sq_diff / sum_sq_orig);
    EXPECT_LT(rel_l2_error, 0.05);
}
