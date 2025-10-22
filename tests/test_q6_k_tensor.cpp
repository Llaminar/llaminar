/**
 * @file test_q6_k_tensor.cpp
 * @brief Unit tests for Q6_KTensor implementation
 *
 * Validates Q6_KTensor::decodeRow() produces correct dequantization results
 * for 6-bit K-quant format.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Q6_KTensor.h"
#include "tensors/TensorFactory.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar;

/**
 * @brief Helper: Convert FP32 to FP16
 */
uint16_t fp32_to_fp16_q6k(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4);

    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t exp = ((bits >> 23) & 0xFF);
    uint32_t mant = (bits & 0x7FFFFF);

    if (exp == 0)
    {
        return sign; // Zero or denormal
    }
    else if (exp == 255)
    {
        return sign | 0x7C00 | (mant ? 0x0200 : 0); // Inf or NaN
    }
    else
    {
        int32_t new_exp = exp - 127 + 15;
        if (new_exp <= 0)
        {
            return sign; // Underflow
        }
        else if (new_exp >= 31)
        {
            return sign | 0x7C00; // Overflow
        }
        else
        {
            return sign | (new_exp << 10) | (mant >> 13);
        }
    }
}

/**
 * @brief Helper: Create Q6_K block data
 *
 * Q6_K block format (256 elements):
 *   - 128 × uint8_t: lower 4 bits (ql)
 *   - 64 × uint8_t: upper 2 bits (qh)
 *   - 16 × int8_t: quantized scales
 *   - 1 × FP16: super-block scale (d)
 *   - Total: 210 bytes
 */
std::vector<uint8_t> create_q6_k_blocks(
    const std::vector<float> &super_scales,
    const std::vector<std::vector<int8_t>> &scales,
    const std::vector<std::vector<int>> &values)
{
    std::vector<uint8_t> raw_data;

    for (size_t block_idx = 0; block_idx < super_scales.size(); block_idx++)
    {
        // 1. Pack lower 4 bits (ql): 256 values → 128 bytes
        // Each byte stores lower 4 bits of 2 values
        for (int i = 0; i < 128; i++)
        {
            int val0_low = values[block_idx][i * 2] & 0x0F;
            int val1_low = values[block_idx][i * 2 + 1] & 0x0F;
            uint8_t packed = val0_low | (val1_low << 4);
            raw_data.push_back(packed);
        }

        // 2. Pack upper 2 bits (qh): 256 values → 64 bytes
        // Each byte stores upper 2 bits of 4 values
        for (int i = 0; i < 64; i++)
        {
            int val0_high = (values[block_idx][i * 4] >> 4) & 0x03;
            int val1_high = (values[block_idx][i * 4 + 1] >> 4) & 0x03;
            int val2_high = (values[block_idx][i * 4 + 2] >> 4) & 0x03;
            int val3_high = (values[block_idx][i * 4 + 3] >> 4) & 0x03;
            uint8_t packed = val0_high | (val1_high << 2) | (val2_high << 4) | (val3_high << 6);
            raw_data.push_back(packed);
        }

        // 3. Write 16 scales (int8_t)
        for (int i = 0; i < 16; i++)
        {
            raw_data.push_back(static_cast<uint8_t>(scales[block_idx][i]));
        }

        // 4. Write super-block scale (FP16)
        uint16_t fp16 = fp32_to_fp16_q6k(super_scales[block_idx]);
        raw_data.push_back(fp16 & 0xFF);
        raw_data.push_back((fp16 >> 8) & 0xFF);
    }

    return raw_data;
}

/**
 * @brief Test: Basic construction and metadata
 */
TEST(Q6_KTensorTest, BasicConstruction)
{
    // Create 1x256 tensor (1 super-block)
    std::vector<float> super_scales = {1.0f};
    std::vector<std::vector<int8_t>> scales = {std::vector<int8_t>(16, 1)};
    std::vector<std::vector<int>> values = {std::vector<int>(256, 32)}; // All zeros (32 = bias)

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);
    Q6_KTensor tensor({1, 256}, raw_data);

    // Validate metadata
    EXPECT_EQ(tensor.shape()[0], 1);
    EXPECT_EQ(tensor.shape()[1], 256);
    EXPECT_EQ(tensor.size(), 256);
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.quant_type(), QuantType::Q6_K);
    EXPECT_NEAR(tensor.compression_ratio(), 5.33f, 0.01f);

    // Validate block descriptor
    auto desc = tensor.block_descriptor();
    EXPECT_EQ(desc.elements_per_block, 256);
    EXPECT_EQ(desc.bytes_per_block, 210); // 128 + 64 + 16 + 2
    EXPECT_EQ(desc.scale_count, 16);
    EXPECT_EQ(desc.bits_per_value, 6);
    EXPECT_TRUE(desc.is_k_quant);
}

/**
 * @brief Test: DecodeRow with zero values (bias = 32)
 */
TEST(Q6_KTensorTest, DecodeRowZeros)
{
    std::vector<float> super_scales = {1.0f};
    std::vector<std::vector<int8_t>> scales = {std::vector<int8_t>(16, 1)};
    std::vector<std::vector<int>> values = {std::vector<int>(256, 32)}; // 32 = zero bias

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);
    Q6_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());

    // All values should be zero: d * scale * (32 - 32) = 0
    for (int i = 0; i < 256; i++)
    {
        EXPECT_FLOAT_EQ(decoded[i], 0.0f) << "Element " << i << " should be zero";
    }
}

/**
 * @brief Test: DecodeRow with known pattern
 */
TEST(Q6_KTensorTest, DecodeRowKnownPattern)
{
    std::vector<float> super_scales = {0.5f};
    std::vector<std::vector<int8_t>> scales(1, std::vector<int8_t>(16));

    // Set scales: segment 0 = scale 2, segment 1 = scale 3, etc.
    for (int i = 0; i < 16; i++)
    {
        scales[0][i] = i + 2;
    }

    // Set values: each segment (16 elements) has value 33 (33 - 32 = 1 after bias)
    std::vector<std::vector<int>> values = {std::vector<int>(256, 33)};

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);
    Q6_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());

    // Validate: d * scale[i/16] * (33 - 32) = 0.5 * (i/16 + 2) * 1
    for (int i = 0; i < 256; i++)
    {
        int scale_idx = i / 16;
        float expected = 0.5f * (scale_idx + 2) * 1.0f;
        EXPECT_NEAR(decoded[i], expected, 1e-3f) << "Element " << i << ", scale_idx " << scale_idx;
    }
}

/**
 * @brief Test: DecodeRow with varied 6-bit values
 */
TEST(Q6_KTensorTest, DecodeRowVariedValues)
{
    std::vector<float> super_scales = {1.0f};
    std::vector<std::vector<int8_t>> scales = {std::vector<int8_t>(16, 1)}; // Uniform scale

    // Create pattern: 0, 1, 2, ..., 63, 0, 1, ... (4 full cycles)
    std::vector<std::vector<int>> values(1);
    for (int i = 0; i < 256; i++)
    {
        values[0].push_back(i % 64);
    }

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);
    Q6_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);
    tensor.decodeRow(0, decoded.data());

    // Validate: d * scale * (value - 32) = 1.0 * 1 * (value - 32)
    for (int i = 0; i < 256; i++)
    {
        int value = i % 64;
        float expected = 1.0f * 1.0f * (value - 32);
        EXPECT_NEAR(decoded[i], expected, 1e-3f) << "Element " << i;
    }
}

/**
 * @brief Test: DecodeSpan
 */
TEST(Q6_KTensorTest, DecodeSpan)
{
    std::vector<float> super_scales = {1.0f};
    std::vector<std::vector<int8_t>> scales = {std::vector<int8_t>(16, 2)};
    std::vector<std::vector<int>> values = {std::vector<int>(256, 40)}; // 40 - 32 = 8

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);
    Q6_KTensor tensor({1, 256}, raw_data);

    // Decode span from middle: elements 100-120
    std::vector<float> decoded(20);
    tensor.decodeSpan(100, 20, decoded.data());

    // Expected: 1.0 * 2 * 8 = 16.0
    for (int i = 0; i < 20; i++)
    {
        EXPECT_NEAR(decoded[i], 16.0f, 1e-3f) << "Span element " << i;
    }
}

/**
 * @brief Test: DecodeRowToBF16
 */
TEST(Q6_KTensorTest, DecodeRowToBF16)
{
    std::vector<float> super_scales = {1.0f};
    std::vector<std::vector<int8_t>> scales = {std::vector<int8_t>(16, 3)};
    std::vector<std::vector<int>> values = {std::vector<int>(256, 35)}; // 35 - 32 = 3

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);
    Q6_KTensor tensor({1, 256}, raw_data);

    std::vector<bfloat16> decoded_bf16(256);
    tensor.decodeRowToBF16(0, decoded_bf16.data());

    // Compare against FP32 decode
    std::vector<float> decoded_fp32(256);
    tensor.decodeRow(0, decoded_fp32.data());

    // Expected: 1.0 * 3 * 3 = 9.0
    for (int i = 0; i < 256; i++)
    {
        float bf16_val = static_cast<float>(decoded_bf16[i]);
        EXPECT_NEAR(bf16_val, decoded_fp32[i], 1e-2f) << "BF16 element " << i;
        EXPECT_NEAR(bf16_val, 9.0f, 1e-1f) << "BF16 value check " << i;
    }
}

/**
 * @brief Test: Multiple rows
 */
TEST(Q6_KTensorTest, MultipleRows)
{
    // Create 2x256 tensor (2 super-blocks)
    std::vector<float> super_scales = {1.0f, 2.0f};
    std::vector<std::vector<int8_t>> scales = {
        std::vector<int8_t>(16, 1),
        std::vector<int8_t>(16, 2)};
    std::vector<std::vector<int>> values = {
        std::vector<int>(256, 33), // Row 0: value 1
        std::vector<int>(256, 34)  // Row 1: value 2
    };

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);
    Q6_KTensor tensor({2, 256}, raw_data);

    // Decode row 0
    std::vector<float> decoded0(256);
    tensor.decodeRow(0, decoded0.data());

    for (int i = 0; i < 256; i++)
    {
        // Expected: 1.0 * 1 * (33 - 32) = 1.0
        EXPECT_NEAR(decoded0[i], 1.0f, 1e-3f) << "Row 0 element " << i;
    }

    // Decode row 1
    std::vector<float> decoded1(256);
    tensor.decodeRow(1, decoded1.data());

    for (int i = 0; i < 256; i++)
    {
        // Expected: 2.0 * 2 * (34 - 32) = 8.0
        EXPECT_NEAR(decoded1[i], 8.0f, 1e-3f) << "Row 1 element " << i;
    }
}

/**
 * @brief Test: Out of bounds access
 */
TEST(Q6_KTensorTest, OutOfBoundsAccess)
{
    std::vector<float> super_scales = {1.0f};
    std::vector<std::vector<int8_t>> scales = {std::vector<int8_t>(16, 1)};
    std::vector<std::vector<int>> values = {std::vector<int>(256, 32)};

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);
    Q6_KTensor tensor({1, 256}, raw_data);

    std::vector<float> decoded(256);

    // Try to access row 1 (should throw)
    EXPECT_THROW(tensor.decodeRow(1, decoded.data()), std::out_of_range);
}

/**
 * @brief Test: Invalid construction (size mismatch)
 */
TEST(Q6_KTensorTest, InvalidSizeMismatch)
{
    // Create data for 1 block (256 elements) but claim shape is 2x256
    std::vector<float> super_scales = {1.0f};
    std::vector<std::vector<int8_t>> scales = {std::vector<int8_t>(16, 1)};
    std::vector<std::vector<int>> values = {std::vector<int>(256, 32)};

    auto raw_data = create_q6_k_blocks(super_scales, scales, values);

    // Should throw due to size mismatch
    EXPECT_THROW(Q6_KTensor tensor({2, 256}, raw_data), std::invalid_argument);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
