/**
 * @file Test__Q4_1Tensor.cpp
 * @brief Unit tests for Q4_1Tensor SIMD path equivalency
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include "loaders/ModelLoader.h"
#include <fstream>

namespace llaminar2
{
    namespace test
    {
        class Test__Q4_1Tensor : public ::testing::Test
        {
        protected:
            Q4_1Block createTestBlock()
            {
                Q4_1Block block;
                block.d = 0x3C00; // FP16 1.0 (scale)
                block.m = 0x3800; // FP16 0.5 (min)

                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low_nibble = i % 16;
                    uint8_t high_nibble = (i + 8) % 16;
                    block.qs[i] = (high_nibble << 4) | low_nibble;
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
        };

        TEST_F(Test__Q4_1Tensor, ScalarVsAVX2Equivalency)
        {
            Q4_1Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q4_1Block::BLOCK_SIZE);
            std::vector<float> avx2_output(Q4_1Block::BLOCK_SIZE);

            Q4_1Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX2__)
            Q4_1Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-6f))
                << "Q4_1 scalar and AVX2 paths produce different results";
#else
            GTEST_SKIP() << "AVX2 not available";
#endif
        }

        TEST_F(Test__Q4_1Tensor, ScalarVsAVX512Equivalency)
        {
            Q4_1Block test_block = createTestBlock();

            std::vector<float> scalar_output(Q4_1Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q4_1Block::BLOCK_SIZE);

            Q4_1Tensor::decodeBlockScalar(test_block, scalar_output.data());

#if defined(__AVX512F__)
            Q4_1Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-6f))
                << "Q4_1 scalar and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "AVX512 not available";
#endif
        }

        TEST_F(Test__Q4_1Tensor, AVX2VsAVX512Equivalency)
        {
#if defined(__AVX2__) && defined(__AVX512F__)
            Q4_1Block test_block = createTestBlock();

            std::vector<float> avx2_output(Q4_1Block::BLOCK_SIZE);
            std::vector<float> avx512_output(Q4_1Block::BLOCK_SIZE);

            Q4_1Tensor::decodeBlockAVX2(test_block, avx2_output.data());
            Q4_1Tensor::decodeBlockAVX512(test_block, avx512_output.data());

            EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-6f))
                << "Q4_1 AVX2 and AVX512 paths produce different results";
#else
            GTEST_SKIP() << "Both AVX2 and AVX512 required";
#endif
        }

        TEST_F(Test__Q4_1Tensor, EdgeCase_ZeroScaleAndMin)
        {
            Q4_1Block test_block = createTestBlock();
            test_block.d = 0x0000; // FP16 zero (scale)
            test_block.m = 0x0000; // FP16 zero (min)

            std::vector<float> scalar_output(Q4_1Block::BLOCK_SIZE);

            Q4_1Tensor::decodeBlockScalar(test_block, scalar_output.data());

            for (size_t i = 0; i < Q4_1Block::BLOCK_SIZE; ++i)
            {
                EXPECT_FLOAT_EQ(scalar_output[i], 0.0f);
            }

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q4_1Block::BLOCK_SIZE);
            Q4_1Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-6f));
#endif
        }

        TEST_F(Test__Q4_1Tensor, EdgeCase_NonZeroMin)
        {
            Q4_1Block test_block;
            test_block.d = 0x4000; // FP16 2.0
            test_block.m = 0xC000; // FP16 -2.0

            for (int i = 0; i < 16; ++i)
            {
                test_block.qs[i] = 0x00; // All nibbles = 0
            }

            std::vector<float> scalar_output(Q4_1Block::BLOCK_SIZE);

#if defined(__AVX2__)
            std::vector<float> avx2_output(Q4_1Block::BLOCK_SIZE);
            Q4_1Tensor::decodeBlockScalar(test_block, scalar_output.data());
            Q4_1Tensor::decodeBlockAVX2(test_block, avx2_output.data());

            // Result should be: 2.0 * 0 + (-2.0) = -2.0 for all elements
            for (size_t i = 0; i < Q4_1Block::BLOCK_SIZE; ++i)
            {
                EXPECT_NEAR(scalar_output[i], -2.0f, 1e-5f);
            }

            EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                                       Q4_1Block::BLOCK_SIZE, 1e-5f));
#endif
        }

    } // namespace test
} // namespace llaminar2

/**
 * @brief Test Q4_1Tensor to INT8 block quantization
 *
 * Validates that Q4_1Tensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy using real model weights.
 */
TEST(Q4_1Tensor_INT8, BlockConversion)
{
    // Load a test model to get real Q4_1Tensor weights
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    // Skip if model file doesn't exist (not all test environments have models)
    if (!std::ifstream(model_path).good())
    {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }

    llaminar2::ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));

    // Find a weight tensor of type Q4_1Tensor
    std::shared_ptr<llaminar2::TensorBase> weight_tensor;
    const auto &model = loader.getModel();

    for (const auto &tensor_info : model.tensors)
    {
        // Map GGUFTensorType to our expected type
        // For now, just try to load the first tensor and check its type
        auto tensor = loader.loadTensor(tensor_info.name);
        if (tensor && std::dynamic_pointer_cast<llaminar2::Q4_1Tensor>(tensor))
        {
            weight_tensor = tensor;
            break;
        }
    }

    if (!weight_tensor)
    {
        GTEST_SKIP() << "No Q4_1Tensor weights found in model";
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
 * @brief Test Q4_1Tensor to<float>() template method matches to_fp32()
 */
TEST(Q4_1Tensor_Template, ToFloat_TemplateMethod)
{
    using namespace llaminar2;

    // Create Q4_1 tensor with test data (2 rows × 32 cols = 64 elements = 2 blocks)
    const std::vector<size_t> shape = {2, 32};

    // Q4_1 block size is 32, so we need 2 blocks for 64 elements
    std::vector<Q4_1Block> blocks(2);

    // Fill blocks with test data
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f); // scale
        blocks[i].m = fp32_to_fp16(0.2f);                                // minimum offset
        for (size_t j = 0; j < 16; ++j)                                  // Q4_1 has 16 bytes (32 nibbles)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    // Convert blocks to raw bytes
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_1Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q4_1Tensor>(shape, raw_data);

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
 * @brief Test Q4_1Tensor to<uint16_t>(BF16) template method matches to_bf16()
 */
TEST(Q4_1Tensor_Template, ToBF16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_1Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].m = fp32_to_fp16(0.2f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_1Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_1Tensor>(shape, raw_data);

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
 * @brief Test Q4_1Tensor to<uint16_t>(FP16) template method matches to_fp16()
 */
TEST(Q4_1Tensor_Template, ToFP16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_1Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].m = fp32_to_fp16(0.2f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_1Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_1Tensor>(shape, raw_data);

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
 * @brief Test Q4_1Tensor to<int8_t>() INT8 quantization
 */
TEST(Q4_1Tensor_Template, ToINT8_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_1Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].m = fp32_to_fp16(0.2f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_1Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_1Tensor>(shape, raw_data);

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
 * @brief Test Q4_1Tensor to<int32_t>() INT32 conversion
 */
TEST(Q4_1Tensor_Template, ToINT32_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_1Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].m = fp32_to_fp16(0.2f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_1Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_1Tensor>(shape, raw_data);

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
 * @brief Test round-trip conversion: Q4_1 → FP32 → BF16 → FP32
 */
TEST(Q4_1Tensor_Template, RoundTrip_Q4_1_FP32_BF16_FP32)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {2, 32};
    std::vector<Q4_1Block> blocks(2);

    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].m = fp32_to_fp16(0.2f);
        for (size_t j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) % 256);
        }
    }

    std::vector<uint8_t> raw_data(2 * sizeof(Q4_1Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_1Tensor>(shape, raw_data);

    // Step 1: Q4_1 → FP32
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

    // Verify round-trip accuracy (Q4_1 has ~2-3% error, BF16 adds another ~1%)
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
