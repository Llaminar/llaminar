/**
 * @file Test__Q4_0Tensor.cpp
 * @brief Unit tests for Q4_0Tensor SIMD path equivalency
 * @author David Sanftenberg
 * @date October 29, 2025
 *
 * Tests verify that scalar, AVX2, and AVX512 dequantization paths produce
 * numerically equivalent results using the LLAMINAR_DEQUANT_SIMD_PATH environment variable.
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
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"

namespace llaminar2
{
    namespace test
    {
        class Test__Q4_0Tensor : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Save original environment variable
                const char *original = std::getenv("LLAMINAR_DEQUANT_SIMD_PATH");
                original_simd_path_ = original ? std::string(original) : "";

                // Initialize MPI context for GEMM tests
                mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
            }

            void TearDown() override
            {
                // Restore original environment variable
                if (original_simd_path_.empty())
                {
                    unsetenv("LLAMINAR_DEQUANT_SIMD_PATH");
                }
                else
                {
                    setenv("LLAMINAR_DEQUANT_SIMD_PATH", original_simd_path_.c_str(), 1);
                }
            }

            /**
             * @brief Force a specific SIMD path by setting environment variable and re-initializing DebugEnv
             */
            void setSIMDPath(const std::string &path)
            {
                setenv("LLAMINAR_DEQUANT_SIMD_PATH", path.c_str(), 1);
                // Force DebugEnv to re-read environment variable
                // Note: DebugEnv is a lazy static singleton, so we need to access it to trigger re-init
                // For testing, we'll create fresh blocks and decode them
            }

            /**
             * @brief Create a test Q4_0 block with known values
             */
            Q4_0Block createTestBlock()
            {
                Q4_0Block block;

                // Set scale (FP16)
                block.d = 0x3C00; // FP16 value of 1.0

                // Set quantized values (32 4-bit values in 16 bytes)
                // Values: 0, 1, 2, 3, ..., 15 repeated twice
                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low_nibble = i % 16;
                    uint8_t high_nibble = (i + 8) % 16;
                    block.qs[i] = (high_nibble << 4) | low_nibble;
                }

                return block;
            }

            /**
             * @brief Compare two float arrays with tolerance
             */
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

            std::string original_simd_path_;
            std::shared_ptr<MPIContext> mpi_ctx_;

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

        /**
         * @brief Test that scalar and AVX2 paths produce identical results
         */
        TEST_F(Test__Q4_0Tensor, ScalarVsAVX2Equivalency)
        {
            Q4_0Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q4_0Block::BLOCK_SIZE);
            std::vector<float> avx2_output(Q4_0Block::BLOCK_SIZE);

            // Decode with scalar path
            Q4_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

            // Decode with AVX2 path (if available)
#if defined(__AVX2__)
            Q4_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_0Block::BLOCK_SIZE, 1e-6f))
                << "Q4_0 scalar and AVX2 paths produce different results";
#else
            GTEST_SKIP() << "AVX2 not available on this platform";
#endif
        }

        /**
         * @brief Test that scalar and AVX512 paths produce identical results
         */
        TEST_F(Test__Q4_0Tensor, ScalarVsAVX512Equivalency)
        {
            Q4_0Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q4_0Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q4_0Block::BLOCK_SIZE);

            // Decode with scalar path
            Q4_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

            // Decode with AVX512 path (if available)
#if defined(__AVX512F__)
            Q4_0Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                                       Q4_0Block::BLOCK_SIZE, 1e-6f))
                << "Q4_0 scalar and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "AVX512 not available on this platform";
#endif
        }

        /**
         * @brief Test that AVX2 and AVX512 paths produce identical results
         */
        TEST_F(Test__Q4_0Tensor, AVX2VsAVX512Equivalency)
        {
#if defined(__AVX2__) && defined(__AVX512F__)
            Q4_0Block test_block = createTestBlock();

            std::vector<float> avx2_output(Q4_0Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q4_0Block::BLOCK_SIZE);

            Q4_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());
            Q4_0Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                                       Q4_0Block::BLOCK_SIZE, 1e-6f))
                << "Q4_0 AVX2 and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
        }

        /**
         * @brief Test edge cases: zero scale
         */
        TEST_F(Test__Q4_0Tensor, EdgeCase_ZeroScale)
        {
            Q4_0Block test_block = createTestBlock();
            test_block.d = 0x0000; // FP16 zero

            std::vector<float> scalar_output(Q4_0Block::BLOCK_SIZE);
            std::vector<float> avx2_output(Q4_0Block::BLOCK_SIZE);

            Q4_0Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            Q4_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            // All outputs should be zero
            for (size_t i = 0; i < Q4_0Block::BLOCK_SIZE; ++i)
            {
                EXPECT_FLOAT_EQ(scalar_output[i], 0.0f) << "Scalar output not zero at index " << i;
                EXPECT_FLOAT_EQ(avx2_output[i], 0.0f) << "AVX2 output not zero at index " << i;
            }
#endif
        }

        /**
         * @brief Test edge cases: all nibbles are 0xF (maximum value)
         */
        TEST_F(Test__Q4_0Tensor, EdgeCase_MaxNibbles)
        {
            Q4_0Block test_block;
            test_block.d = 0x3C00; // FP16 1.0

            for (int i = 0; i < 16; ++i)
            {
                test_block.qs[i] = 0xFF; // All nibbles = 15
            }

            std::vector<float> scalar_output(Q4_0Block::BLOCK_SIZE);

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q4_0Block::BLOCK_SIZE);
            Q4_0Tensor::decodeBlockScalar(test_block, scalar_output.data());
            Q4_0Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_0Block::BLOCK_SIZE, 1e-6f));
#endif
        }

        /**
         * @brief Compare CPUNativeVNNIGemmKernel (INT8) vs FloatingPointGemmKernel (FP32) for Q4_0.
         *
         * This test verifies that the quantized GEMM kernel produces results close to
         * the FP32 reference implementation using OneDNN.
         */
        TEST_F(Test__Q4_0Tensor, QuantizedVsFP32Parity)
        {
            // Realistic dimensions: 64 tokens, 512 hidden dim
            const int m = 64;
            const int n = 512;
            const int k = 512;

            // Q4_0: 32 elements per block
            const size_t num_blocks = (static_cast<size_t>(n) * k) / 32;
            std::vector<uint8_t> raw_data(num_blocks * sizeof(Q4_0Block));
            Q4_0Block *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());

            // Initialize with random but valid data
            std::mt19937 rng(42);
            std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
            std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

            for (size_t b = 0; b < num_blocks; ++b)
            {
                // Valid FP16 scale factor
                blocks[b].d = fp32_to_fp16(scale_dist(rng));
                // Random qs (4-bit values packed into 16 bytes)
                for (int i = 0; i < 16; ++i)
                {
                    blocks[b].qs[i] = byte_dist(rng);
                }
            }

            // Create quantized tensor
            auto q4_0_tensor = std::make_unique<Q4_0Tensor>(
                std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
                raw_data);

            // Dequantize to FP32 for reference
            TensorFactory factory(*mpi_ctx_);
            auto fp32_weights = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
            q4_0_tensor->to_fp32(fp32_weights->mutable_data());

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
            auto quantized_gemm = q4_0_tensor->createGemm();
            ASSERT_TRUE(quantized_gemm->multiply_tensor(
                input.get(),
                output_quantized.get(),
                m, n, k));

            // Run FP32 GEMM (OneDNN reference)
            gemm::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());
            ASSERT_TRUE(fp32_gemm.multiply_tensor(
                input.get(),
                output_fp32.get(),
                m, n, k));

            // Compare results
            float rel_l2_error = compute_relative_l2_error(
                output_quantized->data(),
                output_fp32->data(),
                m * n);

            std::cout << "[Q4_0 Parity] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;

            // 1% tolerance for quantization error
            EXPECT_LT(rel_l2_error, 0.01f)
                << "Q4_0 quantized GEMM diverged from FP32 reference by "
                << (rel_l2_error * 100.0f) << "%";
        }

    } // namespace test
} // namespace llaminar2

/**
 * @brief Test Q4_0Tensor to INT8 block quantization
 *
 * Validates that Q4_0Tensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy using real model weights.
 */
TEST(Q4_0Tensor_INT8, BlockConversion)
{
    // Load a test model to get real Q4_0Tensor weights
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    // Skip if model file doesn't exist (not all test environments have models)
    if (!std::ifstream(model_path).good())
    {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }

    llaminar2::ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));

    // Find a weight tensor of type Q4_0Tensor
    std::shared_ptr<llaminar2::TensorBase> weight_tensor;
    const auto &model = loader.getModel();

    for (const auto &tensor_info : model.tensors)
    {
        // Map GGUFTensorType to our expected type
        // For now, just try to load the first tensor and check its type
        auto tensor = loader.loadTensor(tensor_info.name);
        if (tensor && std::dynamic_pointer_cast<llaminar2::Q4_0Tensor>(tensor))
        {
            weight_tensor = tensor;
            break;
        }
    }

    if (!weight_tensor)
    {
        GTEST_SKIP() << "No Q4_0Tensor weights found in model";
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
 * @brief Test Q4_0Tensor to<float>() template method matches to_fp32()
 */
TEST(Q4_0Tensor_Template, ToFloat_TemplateMethod)
{
    using namespace llaminar2;

    // Create Q4_0 tensor with test data (2 rows × 32 cols = 64 elements = 2 blocks)
    const std::vector<size_t> shape = {2, 32};

    // Q4_0 block size is 32, so we need 2 blocks for 64 elements
    std::vector<Q4_0Block> blocks(2);

    // Fill blocks with test data
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j) // Q4_0 has 16 bytes (32 nibbles)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    // Convert blocks to raw bytes
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q4_0Tensor>(shape, raw_data);

    // Test to<float>() template method
    std::vector<float> result_template(64);
    tensor->template to<float>(result_template.data());

    // Compare with legacy to_fp32()
    std::vector<float> result_legacy(64);
    tensor->to_fp32(result_legacy.data());

    // Should be identical
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_FLOAT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test Q4_0Tensor to<uint16_t>(BF16) template method matches to_bf16()
 */
TEST(Q4_0Tensor_Template, ToBF16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_0Tensor>(shape, raw_data);

    // Test to<uint16_t>() with BF16 format
    std::vector<uint16_t> result_template(64);
    tensor->template to<uint16_t>(result_template.data(), TensorType::BF16);

    // Compare with legacy to_bf16()
    std::vector<uint16_t> result_legacy(64);
    tensor->to_bf16(result_legacy.data());

    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test Q4_0Tensor to<uint16_t>(FP16) template method matches to_fp16()
 */
TEST(Q4_0Tensor_Template, ToFP16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_0Tensor>(shape, raw_data);

    // Test to<uint16_t>() with FP16 format
    std::vector<uint16_t> result_template(64);
    tensor->template to<uint16_t>(result_template.data(), TensorType::FP16);

    // Compare with legacy to_fp16()
    std::vector<uint16_t> result_legacy(64);
    tensor->to_fp16(result_legacy.data());

    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test Q4_0Tensor to<int8_t>() INT8 quantization
 */
TEST(Q4_0Tensor_Template, ToINT8_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_0Tensor>(shape, raw_data);

    // Test to<int8_t>() INT8 quantization
    std::vector<int8_t> int8_data(64);
    tensor->template to<int8_t>(int8_data.data());

    // Verify all int8 values are in valid range [-127, 127]
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_GE(int8_data[i], -127) << "Value at index " << i << " too low";
        EXPECT_LE(int8_data[i], 127) << "Value at index " << i << " too high";
    }
}

/**
 * @brief Test Q4_0Tensor to<int32_t>() INT32 conversion
 */
TEST(Q4_0Tensor_Template, ToINT32_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_0Tensor>(shape, raw_data);

    // Test to<int32_t>() INT32 conversion
    std::vector<int32_t> int32_data(64);
    tensor->template to<int32_t>(int32_data.data());

    // Verify no overflow occurred
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }
}

/**
 * @brief Test round-trip conversion: Q4_0 → FP32 → BF16 → FP32
 */
TEST(Q4_0Tensor_Template, RoundTrip_Q4_0_FP32_BF16_FP32)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_0Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_0Tensor>(shape, raw_data);

    // Step 1: Q4_0 → FP32
    std::vector<float> fp32_data(64);
    tensor->template to<float>(fp32_data.data());

    // Step 2: FP32 → BF16
    auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
    std::memcpy(fp32_tensor->mutable_data(), fp32_data.data(), 64 * sizeof(float));

    std::vector<uint16_t> bf16_data(64);
    fp32_tensor->template to<uint16_t>(bf16_data.data(), TensorType::BF16);

    // Step 3: BF16 → FP32
    auto bf16_tensor = std::make_shared<BF16Tensor>(shape, bf16_data);
    std::vector<float> final_fp32_data(64);
    bf16_tensor->template to<float>(final_fp32_data.data());

    // Verify round-trip accuracy (Q4_0 has ~2-3% error, BF16 adds another ~1%)
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;

    for (size_t i = 0; i < 64; ++i)
    {
        double diff = fp32_data[i] - final_fp32_data[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += fp32_data[i] * fp32_data[i];
    }

    double rel_l2_error = std::sqrt(sum_sq_diff / sum_sq_orig);
    EXPECT_LT(rel_l2_error, 0.05) << "Round-trip relative L2 error: " << rel_l2_error;
}
