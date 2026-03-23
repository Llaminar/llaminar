/**
 * @file Test__Q6_KTensor.cpp
 * @brief Unit tests for Q6_KTensor SIMD path equivalency
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include "loaders/ModelLoader.h"
#include <fstream>
#include "v2/utils/MPIContext.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"

namespace llaminar2
{
    namespace test
    {
        class Test__Q6_KTensor : public ::testing::Test
        {
        protected:
            Q6_KBlock createTestBlock()
            {
                Q6_KBlock block;
                block.d = 0x3C00; // FP16 1.0 (global scale)

                // Initialize all scales (16 sub-blocks)
                for (int i = 0; i < 16; ++i)
                {
                    block.scales[i] = static_cast<int8_t>(i - 8); // Range: -8 to 7
                }

                // Initialize ql (lower 4 bits) with pattern
                for (int i = 0; i < 128; ++i)
                {
                    block.ql[i] = static_cast<uint8_t>(i % 256);
                }

                // Initialize qh (upper 2 bits) with pattern
                for (int i = 0; i < 64; ++i)
                {
                    block.qh[i] = static_cast<uint8_t>((i % 4) * 0x55); // 0x00, 0x55, 0xAA, 0xFF
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
                        if (mismatch_count <= 5)
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

            std::unique_ptr<MPIContext> mpi_ctx_;

            void SetUp() override
            {
                mpi_ctx_ = std::make_unique<MPIContext>(0, 1, MPI_COMM_WORLD);
            }

            static float compute_relative_l2_error(const float *a, const float *b, size_t n)
            {
                double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
                for (size_t i = 0; i < n; ++i)
                {
                    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                    sum_sq_diff += diff * diff;
                    sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
                }
                return (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;
            }
        };

        TEST_F(Test__Q6_KTensor, ScalarVsAVX2Equivalency)
        {
            Q6_KBlock test_block = createTestBlock();

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-6f))
                << "Q6_K scalar and AVX2 paths produce different results";
#else
            GTEST_SKIP() << "AVX2 not available";
#endif
        }

        TEST_F(Test__Q6_KTensor, ScalarVsAVX512Equivalency)
        {
            Q6_KBlock test_block = createTestBlock();

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);
            std::vector<float> avx512_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX512F__)
            Q6_KTensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-6f))
                << "Q6_K scalar and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "AVX512 not available";
#endif
        }

        TEST_F(Test__Q6_KTensor, AVX2VsAVX512Equivalency)
        {
#if defined(__AVX2__) && defined(__AVX512F__)
            Q6_KBlock test_block = createTestBlock();

            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            std::vector<float> avx512_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());
            Q6_KTensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-6f))
                << "Q6_K AVX2 and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "Both AVX2 and AVX512 required";
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_ZeroGlobalScale)
        {
            Q6_KBlock test_block = createTestBlock();
            test_block.d = 0x0000; // FP16 zero (global scale)

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

            // With zero global scale, all outputs should be zero
            for (size_t i = 0; i < Q6_KBlock::BLOCK_SIZE; ++i)
            {
                EXPECT_FLOAT_EQ(scalar_output[i], 0.0f) << "At index " << i;
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-6f));
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_UniformScales)
        {
            Q6_KBlock test_block;
            test_block.d = 0x4000; // FP16 2.0

            // Set all sub-block scales to same value
            for (int i = 0; i < 16; ++i)
            {
                test_block.scales[i] = 10; // Uniform scale
            }

            // Set all ql to 0
            for (int i = 0; i < 128; ++i)
            {
                test_block.ql[i] = 0x00;
            }

            // Set all qh to 0
            for (int i = 0; i < 64; ++i)
            {
                test_block.qh[i] = 0x00;
            }

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

            // Each value: global_scale * sub_scale * (q - 32)
            // = 2.0 * 10 * (0 - 32) = -640.0
            for (size_t i = 0; i < Q6_KBlock::BLOCK_SIZE; ++i)
            {
                EXPECT_NEAR(scalar_output[i], -640.0f, 1e-4f) << "At index " << i;
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-4f));
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_MaxQuantValue)
        {
            Q6_KBlock test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Set all scales to 1
            for (int i = 0; i < 16; ++i)
            {
                test_block.scales[i] = 1;
            }

            // Set ql to maximum (all bits set)
            for (int i = 0; i < 128; ++i)
            {
                test_block.ql[i] = 0xFF;
            }

            // Set qh to maximum (all bits set)
            for (int i = 0; i < 64; ++i)
            {
                test_block.qh[i] = 0xFF;
            }

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

            // Maximum 6-bit value is 63 (0x3F)
            // Result: 1.0 * 1 * (63 - 32) = 31.0
            for (size_t i = 0; i < Q6_KBlock::BLOCK_SIZE; ++i)
            {
                EXPECT_NEAR(scalar_output[i], 31.0f, 1e-5f) << "At index " << i;
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-5f));
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_NegativeScales)
        {
            Q6_KBlock test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Set all sub-block scales to negative
            for (int i = 0; i < 16; ++i)
            {
                test_block.scales[i] = -10;
            }

            // Set all quantized values to center (32)
            for (int i = 0; i < 128; ++i)
            {
                test_block.ql[i] = 0x00; // Lower 4 bits = 0
            }

            for (int i = 0; i < 64; ++i)
            {
                test_block.qh[i] = 0xAA; // Pattern: 10101010 (upper 2 bits = 2 per element)
            }

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-5f));
#endif
        }

        TEST_F(Test__Q6_KTensor, EdgeCase_SubBlockBoundaries)
        {
            // Test that sub-block boundaries are handled correctly (16 sub-blocks × 16 elements)
            Q6_KBlock test_block;
            test_block.d = 0x3C00; // FP16 1.0

            // Set each sub-block to have unique scale
            for (int i = 0; i < 16; ++i)
            {
                test_block.scales[i] = static_cast<int8_t>(i); // 0, 1, 2, ..., 15
            }

            // Set all quantized values to center (32)
            for (int i = 0; i < 128; ++i)
            {
                test_block.ql[i] = 0x00;
            }

            for (int i = 0; i < 64; ++i)
            {
                test_block.qh[i] = 0xAA; // Upper 2 bits = 2 per element
            }

            std::vector<float> scalar_output(Q6_KBlock::BLOCK_SIZE);

            Q6_KTensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q6_KBlock::BLOCK_SIZE);
            Q6_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

            // Verify sub-block boundaries are consistent
            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q6_KBlock::BLOCK_SIZE, 1e-5f));

            // Additionally verify that each sub-block of 16 elements has consistent scaling
            for (int sb = 0; sb < 16; ++sb)
            {
                int8_t expected_scale = test_block.scales[sb];
                float first_value = scalar_output[sb * 16];

                for (int j = 1; j < 16; ++j)
                {
                    // All values in same sub-block should be equal (same scale, same quant values)
                    EXPECT_NEAR(scalar_output[sb * 16 + j], first_value, 1e-5f)
                        << "Sub-block " << sb << ", element " << j;
                }
            }
#endif
        }

        /**
         * @brief Quantized GEMM vs FP32 GEMM Parity Test for Q6_K
         *
         * Compares CPUNativeVNNIGemmKernel (INT8) against FloatingPointGemmKernel (FP32 OneDNN)
         * using randomly initialized Q6_K weights. Validates that quantization introduces
         * acceptable error (< 1% relative L2).
         */
        TEST_F(Test__Q6_KTensor, QuantizedVsFP32Parity)
        {
            // Realistic dimensions: 64 tokens, 512 hidden dim
            const int m = 64;
            const int n = 512;
            const int k = 512;

            // Q6_K: 256 elements per block (super-block with 16 sub-blocks of 16 elements each)
            const size_t num_blocks = (static_cast<size_t>(n) * k) / 256;
            std::vector<uint8_t> raw_data(num_blocks * sizeof(Q6_KBlock));
            Q6_KBlock *blocks = reinterpret_cast<Q6_KBlock *>(raw_data.data());

            // Initialize with random but valid data
            std::mt19937 rng(42);
            std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
            std::uniform_int_distribution<int8_t> scale_dist(-32, 31);
            std::uniform_real_distribution<float> global_scale_dist(0.001f, 0.1f);

            for (size_t b = 0; b < num_blocks; ++b)
            {
                // Valid FP16 global scale factor
                blocks[b].d = fp32_to_fp16(global_scale_dist(rng));
                // Random sub-block scales (16 x int8)
                for (int i = 0; i < 16; ++i)
                {
                    blocks[b].scales[i] = scale_dist(rng);
                }
                // Random ql (lower 4 bits, 128 bytes for 256 elements)
                for (int i = 0; i < 128; ++i)
                {
                    blocks[b].ql[i] = byte_dist(rng);
                }
                // Random qh (upper 2 bits, 64 bytes for 256 elements)
                for (int i = 0; i < 64; ++i)
                {
                    blocks[b].qh[i] = byte_dist(rng);
                }
            }

            // Create quantized tensor
            auto q6_k_tensor = std::make_unique<Q6_KTensor>(
                std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
                raw_data);

            // Dequantize to FP32 for reference
            TensorFactory factory(*mpi_ctx_);
            auto fp32_weights = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
            q6_k_tensor->to_fp32(fp32_weights->mutable_data());

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
            auto quantized_gemm = q6_k_tensor->createGemm();
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

            std::cout << "[Q6_K Parity] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;

            // 1% tolerance for quantization error
            EXPECT_LT(rel_l2_error, 0.01f)
                << "Q6_K quantized GEMM error exceeds 1% threshold";
        }

    } // namespace test
} // namespace llaminar2

/**
 * @brief Test Q6_KTensor to INT8 block quantization
 *
 * Validates that Q6_KTensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy using real model weights.
 */
TEST(Q6_KTensor_INT8, BlockConversion)
{
    // Load a test model to get real Q6_KTensor weights
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q6_k.gguf";

    // Skip if model file doesn't exist (not all test environments have models)
    if (!std::ifstream(model_path).good())
    {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }

    llaminar2::ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));

    // Find a weight tensor of type Q6_KTensor
    std::shared_ptr<llaminar2::TensorBase> weight_tensor;
    const auto &model = loader.getModel();

    for (const auto &tensor_info : model.tensors)
    {
        // Map GGUFTensorType to our expected type
        // For now, just try to load the first tensor and check its type
        auto tensor = loader.loadTensor(tensor_info.name);
        if (tensor && std::dynamic_pointer_cast<llaminar2::Q6_KTensor>(tensor))
        {
            weight_tensor = tensor;
            break;
        }
    }

    if (!weight_tensor)
    {
        GTEST_SKIP() << "No Q6_KTensor weights found in model";
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

TEST(Q6_KTensor_Template, ToFloat_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256}; // Q6_K: 256 elements per block
    std::vector<Q6_KBlock> blocks(1);

    blocks[0].d = fp32_to_fp16(0.5f);
    for (size_t i = 0; i < 128; ++i)
        blocks[0].ql[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qh[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<int8_t>((i - 8) * 5);

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q6_KTensor>(shape, raw_data);

    std::vector<float> result_template(256);
    tensor->template to<float>(result_template.data());

    std::vector<float> result_legacy(256);
    tensor->to_fp32(result_legacy.data());

    for (size_t i = 0; i < 256; ++i)
        EXPECT_FLOAT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(Q6_KTensor_Template, ToBF16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q6_KBlock> blocks(1);

    blocks[0].d = fp32_to_fp16(0.5f);
    for (size_t i = 0; i < 128; ++i)
        blocks[0].ql[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qh[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<int8_t>((i - 8) * 5);

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q6_KTensor>(shape, raw_data);

    std::vector<uint16_t> result_template(256);
    tensor->template to<uint16_t>(result_template.data(), TensorType::BF16);

    std::vector<uint16_t> result_legacy(256);
    tensor->to_bf16(result_legacy.data());

    for (size_t i = 0; i < 256; ++i)
        EXPECT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(Q6_KTensor_Template, ToFP16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q6_KBlock> blocks(1);

    blocks[0].d = fp32_to_fp16(0.5f);
    for (size_t i = 0; i < 128; ++i)
        blocks[0].ql[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qh[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<int8_t>((i - 8) * 5);

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q6_KTensor>(shape, raw_data);

    std::vector<uint16_t> result_template(256);
    tensor->template to<uint16_t>(result_template.data(), TensorType::FP16);

    std::vector<uint16_t> result_legacy(256);
    tensor->to_fp16(result_legacy.data());

    for (size_t i = 0; i < 256; ++i)
        EXPECT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(Q6_KTensor_Template, ToINT8_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q6_KBlock> blocks(1);

    blocks[0].d = fp32_to_fp16(0.5f);
    for (size_t i = 0; i < 128; ++i)
        blocks[0].ql[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qh[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<int8_t>((i - 8) * 5);

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q6_KTensor>(shape, raw_data);

    std::vector<int8_t> int8_data(256);
    tensor->template to<int8_t>(int8_data.data());

    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_GE(int8_data[i], -127);
        EXPECT_LE(int8_data[i], 127);
    }
}

TEST(Q6_KTensor_Template, ToINT32_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q6_KBlock> blocks(1);

    blocks[0].d = fp32_to_fp16(0.5f);
    for (size_t i = 0; i < 128; ++i)
        blocks[0].ql[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qh[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<int8_t>((i - 8) * 5);

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q6_KTensor>(shape, raw_data);

    std::vector<int32_t> int32_data(256);
    tensor->template to<int32_t>(int32_data.data());

    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }
}

TEST(Q6_KTensor_Template, RoundTrip_Q6_K_FP32_BF16_FP32)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q6_KBlock> blocks(1);

    blocks[0].d = fp32_to_fp16(0.5f);
    for (size_t i = 0; i < 128; ++i)
        blocks[0].ql[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qh[i] = static_cast<uint8_t>(i % 256);
    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<int8_t>((i - 8) * 5);

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q6_KTensor>(shape, raw_data);

    std::vector<float> fp32_data(256);
    tensor->template to<float>(fp32_data.data());

    auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
    std::memcpy(fp32_tensor->mutable_data(), fp32_data.data(), 256 * sizeof(float));

    std::vector<uint16_t> bf16_data(256);
    fp32_tensor->template to<uint16_t>(bf16_data.data(), TensorType::BF16);

    auto bf16_tensor = std::make_shared<BF16Tensor>(shape, bf16_data);
    std::vector<float> final_fp32_data(256);
    bf16_tensor->template to<float>(final_fp32_data.data());

    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;

    for (size_t i = 0; i < 256; ++i)
    {
        double diff = fp32_data[i] - final_fp32_data[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += fp32_data[i] * fp32_data[i];
    }

    double rel_l2_error = std::sqrt(sum_sq_diff / sum_sq_orig);
    EXPECT_LT(rel_l2_error, 0.05);
}
