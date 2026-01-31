/**
 * @file Test__Q2_KTensor.cpp
 * @brief Unit tests for Q2_K SIMD equivalency
 * @author David Sanftenberg
 *
 * Tests that scalar, AVX2, and AVX512 implementations of Q2_K dequantization
 * produce identical numerical results.
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <vector>
#include <cmath>
#include <cstring>
#include "loaders/ModelLoader.h"
#include <fstream>

using namespace llaminar2;

class Test__Q2_KTensor : public ::testing::Test
{
protected:
    // Helper to compare two float arrays with tolerance
    bool compareOutputs(const float *a, const float *b, size_t count, float tolerance = 1e-5f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::fabs(a[i] - b[i]) > tolerance)
            {
                std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i]
                          << " (diff = " << std::fabs(a[i] - b[i]) << ")" << std::endl;
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// SIMD Equivalency Tests
// ============================================================================

#if defined(__AVX2__)
TEST_F(Test__Q2_KTensor, ScalarVsAVX2Equivalency)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    // Set scales (4 bits each for scale and min)
    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x85; // scale=5, min=8
    }

    // Set 2-bit quantized values (pattern: 0, 1, 2, 3 repeating)
    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xE4; // 11 10 01 00 in binary = 3,2,1,0
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);

    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q2_KTensor, ScalarVsAVX512Equivalency)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x85; // scale=5, min=8
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xE4; // Pattern: 3,2,1,0
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);
    std::vector<float> avx512_output(Q2_KBlock::BLOCK_SIZE);

    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX512(test_block, avx512_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
}
#endif

#if defined(__AVX2__) && defined(__AVX512F__)
TEST_F(Test__Q2_KTensor, AVX2VsAVX512Equivalency)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x85;
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xE4;
    }

    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    std::vector<float> avx512_output(Q2_KBlock::BLOCK_SIZE);

    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());
    Q2_KTensor::decodeBlockAVX512(test_block, avx512_output.data());

    EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
}
#endif

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(Test__Q2_KTensor, EdgeCase_ZeroScales)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    // All scales zero
    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x00;
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xFF;
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    // With zero scales, all outputs should be zero (or -min)
    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

TEST_F(Test__Q2_KTensor, EdgeCase_Max2BitValues)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0xFF; // max scale and min (15 each)
    }

    // All 2-bit values = 3 (maximum)
    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xFF; // 11 11 11 11 = all 3's
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

TEST_F(Test__Q2_KTensor, EdgeCase_SuperBlockScales)
{
    Q2_KBlock test_block;
    test_block.d = 0x4000;    // FP16 2.0 (higher super-block scale)
    test_block.dmin = 0x3000; // FP16 0.25 (lower min scale)

    // Varying scales across groups
    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = (i << 4) | (15 - i); // Varying pattern
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0x1B; // 00 01 10 11 pattern
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

TEST_F(Test__Q2_KTensor, EdgeCase_ChunkBoundary)
{
    // Test the boundary between the 2 chunks of 128 elements
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    // Different scales for first and second chunk
    for (int i = 0; i < 8; ++i)
    {
        test_block.scales[i] = 0x3C; // First chunk: scale=12, min=3
    }
    for (int i = 8; i < 16; ++i)
    {
        test_block.scales[i] = 0xA5; // Second chunk: scale=5, min=10
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = (i & 1) ? 0xAA : 0x55; // Alternating pattern
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

TEST_F(Test__Q2_KTensor, EdgeCase_ShiftPattern)
{
    // Test all 4 shift positions (0, 2, 4, 6 bits)
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x77; // scale=7, min=7
    }

    // Each byte has different 2-bit values at each shift position
    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0x1B; // 00 01 10 11 (tests all shift patterns)
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    // Verify the shift pattern is correctly extracted
    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

/**
 * @brief Test Q2_KTensor to INT8 block quantization
 *
 * Validates that Q2_KTensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy using real model weights.
 */
TEST(Q2_KTensor_INT8, BlockConversion)
{
    // Load a test model to get real Q2_KTensor weights
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q2_k.gguf";

    // Skip if model file doesn't exist (not all test environments have models)
    if (!std::ifstream(model_path).good())
    {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }

    llaminar2::ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));

    // Find a weight tensor of type Q2_KTensor
    std::shared_ptr<llaminar2::TensorBase> weight_tensor;
    const auto &model = loader.getModel();

    for (const auto &tensor_info : model.tensors)
    {
        // Map GGUFTensorType to our expected type
        // For now, just try to load the first tensor and check its type
        auto tensor = loader.loadTensor(tensor_info.name);
        if (tensor && std::dynamic_pointer_cast<llaminar2::Q2_KTensor>(tensor))
        {
            weight_tensor = tensor;
            break;
        }
    }

    if (!weight_tensor)
    {
        GTEST_SKIP() << "No Q2_KTensor weights found in model";
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

TEST(Q2_KTensor_Template, ToFloat_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q2_KBlock> blocks(1);

    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<uint8_t>(i * 10);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qs[i] = static_cast<uint8_t>(i % 256);
    blocks[0].d = fp32_to_fp16(0.5f);
    blocks[0].dmin = fp32_to_fp16(0.1f);

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q2_KTensor>(shape, raw_data);

    std::vector<float> result_template(256);
    tensor->template to<float>(result_template.data());

    std::vector<float> result_legacy(256);
    tensor->to_fp32(result_legacy.data());

    for (size_t i = 0; i < 256; ++i)
        EXPECT_FLOAT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(Q2_KTensor_Template, ToBF16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q2_KBlock> blocks(1);

    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<uint8_t>(i * 10);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qs[i] = static_cast<uint8_t>(i % 256);
    blocks[0].d = fp32_to_fp16(0.5f);
    blocks[0].dmin = fp32_to_fp16(0.1f);

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q2_KTensor>(shape, raw_data);

    std::vector<uint16_t> result_template(256);
    tensor->template to<uint16_t>(result_template.data(), TensorType::BF16);

    std::vector<uint16_t> result_legacy(256);
    tensor->to_bf16(result_legacy.data());

    for (size_t i = 0; i < 256; ++i)
        EXPECT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(Q2_KTensor_Template, ToFP16_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q2_KBlock> blocks(1);

    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<uint8_t>(i * 10);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qs[i] = static_cast<uint8_t>(i % 256);
    blocks[0].d = fp32_to_fp16(0.5f);
    blocks[0].dmin = fp32_to_fp16(0.1f);

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q2_KTensor>(shape, raw_data);

    std::vector<uint16_t> result_template(256);
    tensor->template to<uint16_t>(result_template.data(), TensorType::FP16);

    std::vector<uint16_t> result_legacy(256);
    tensor->to_fp16(result_legacy.data());

    for (size_t i = 0; i < 256; ++i)
        EXPECT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(Q2_KTensor_Template, ToINT8_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q2_KBlock> blocks(1);

    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<uint8_t>(i * 10);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qs[i] = static_cast<uint8_t>(i % 256);
    blocks[0].d = fp32_to_fp16(0.5f);
    blocks[0].dmin = fp32_to_fp16(0.1f);

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q2_KTensor>(shape, raw_data);

    std::vector<int8_t> int8_data(256);
    tensor->template to<int8_t>(int8_data.data());

    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_GE(int8_data[i], -127);
        EXPECT_LE(int8_data[i], 127);
    }
}

TEST(Q2_KTensor_Template, ToINT32_TemplateMethod)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q2_KBlock> blocks(1);

    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<uint8_t>(i * 10);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qs[i] = static_cast<uint8_t>(i % 256);
    blocks[0].d = fp32_to_fp16(0.5f);
    blocks[0].dmin = fp32_to_fp16(0.1f);

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q2_KTensor>(shape, raw_data);

    std::vector<int32_t> int32_data(256);
    tensor->template to<int32_t>(int32_data.data());

    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }
}

TEST(Q2_KTensor_Template, RoundTrip_Q2_K_FP32_BF16_FP32)
{
    using namespace llaminar2;

    const std::vector<size_t> shape = {1, 256};
    std::vector<Q2_KBlock> blocks(1);

    for (size_t i = 0; i < 16; ++i)
        blocks[0].scales[i] = static_cast<uint8_t>(i * 10);
    for (size_t i = 0; i < 64; ++i)
        blocks[0].qs[i] = static_cast<uint8_t>(i % 256);
    blocks[0].d = fp32_to_fp16(0.5f);
    blocks[0].dmin = fp32_to_fp16(0.1f);

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q2_KTensor>(shape, raw_data);

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
