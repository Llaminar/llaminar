/**
 * @file test_q4_0_tensor.cpp
 * @brief Unit tests for Q4_0Tensor implementation
 *
 * Validates Q4_0Tensor::decodeRow() produces correct dequantization results
 * for 4-bit uniform quantization.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Q4_0Tensor.h"
#include "tensors/TensorFactory.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar;

/**
 * @brief Helper: Convert FP32 to FP16
 */
uint16_t fp32_to_fp16(float value)
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
 * @brief Helper: Create Q4_0 block data with known scale and values
 *
 * Q4_0 block format:
 *   - 1 × FP16 scale (2 bytes)
 *   - 16 × uint8_t storing nibbles (16 bytes, 2 4-bit values per byte)
 *   - Total: 18 bytes per block (32 elements)
 */
std::vector<uint8_t> create_q4_0_blocks(
    const std::vector<float> &scales,
    const std::vector<std::vector<int8_t>> &block_values)
{
    std::vector<uint8_t> raw_data;

    for (size_t i = 0; i < scales.size(); i++)
    {
        // Write FP16 scale (2 bytes)
        uint16_t fp16 = fp32_to_fp16(scales[i]);
        raw_data.push_back(fp16 & 0xFF);
        raw_data.push_back((fp16 >> 8) & 0xFF);

        // Write 32 4-bit values as 16 bytes (nibbles)
        for (int j = 0; j < 16; j++)
        {
            int8_t val0 = block_values[i][j * 2] + 8;     // First nibble (add 8 for unsigned)
            int8_t val1 = block_values[i][j * 2 + 1] + 8; // Second nibble (add 8 for unsigned)
            uint8_t packed = (val0 & 0x0F) | ((val1 & 0x0F) << 4);
            raw_data.push_back(packed);
        }
    }

    return raw_data;
}

/**
 * @brief Test: Basic construction and metadata
 */
TEST(Q4_0TensorTest, BasicConstruction)
{
    // Create 2x32 tensor (2 blocks)
    std::vector<float> scales = {1.0f, 2.0f};
    std::vector<std::vector<int8_t>> values(2, std::vector<int8_t>(32, 0));

    auto raw_data = create_q4_0_blocks(scales, values);
    Q4_0Tensor tensor({2, 32}, raw_data);

    // Validate metadata
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 32);
    EXPECT_EQ(tensor.size(), 64);
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.quant_type(), QuantType::Q4_0);
    EXPECT_FLOAT_EQ(tensor.compression_ratio(), 8.0f); // 32-bit → 4-bit

    // Validate block descriptor
    auto desc = tensor.block_descriptor();
    EXPECT_EQ(desc.elements_per_block, 32);
    EXPECT_EQ(desc.bytes_per_block, 18); // 2 bytes scale + 16 bytes nibbles
    EXPECT_EQ(desc.scale_count, 1);
    EXPECT_EQ(desc.bits_per_value, 4);
    EXPECT_FALSE(desc.is_k_quant);
}

/**
 * @brief Test: DecodeRow with zero values
 */
TEST(Q4_0TensorTest, DecodeRowZeros)
{
    std::vector<float> scales = {1.0f};
    std::vector<std::vector<int8_t>> values = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

    auto raw_data = create_q4_0_blocks(scales, values);
    Q4_0Tensor tensor({1, 32}, raw_data);

    std::vector<float> decoded(32);
    tensor.decodeRow(0, decoded.data());

    // All values should be zero
    for (int i = 0; i < 32; i++)
    {
        EXPECT_FLOAT_EQ(decoded[i], 0.0f) << "Element " << i << " should be zero";
    }
}

/**
 * @brief Test: DecodeRow with known pattern
 */
TEST(Q4_0TensorTest, DecodeRowKnownPattern)
{
    std::vector<float> scales = {0.5f};
    std::vector<int8_t> values;
    // Create pattern: -8, -7, -6, ..., 7 (full 4-bit signed range)
    for (int i = -8; i < 8; i++)
    {
        values.push_back(i);
    }
    for (int i = -8; i < 8; i++)
    {
        values.push_back(i);
    }

    std::vector<std::vector<int8_t>> block_vals = {values};
    auto raw_data = create_q4_0_blocks(scales, block_vals);
    Q4_0Tensor tensor({1, 32}, raw_data);

    std::vector<float> decoded(32);
    tensor.decodeRow(0, decoded.data());

    // Validate decoded values: scale * quantized
    for (int i = 0; i < 32; i++)
    {
        float expected = 0.5f * values[i];
        EXPECT_NEAR(decoded[i], expected, 1e-4f) << "Element " << i;
    }
}

/**
 * @brief Test: DecodeRow with multiple blocks
 */
TEST(Q4_0TensorTest, DecodeRowMultipleBlocks)
{
    // Create 1x64 tensor (2 blocks)
    std::vector<float> scales = {1.0f, 2.0f};
    std::vector<std::vector<int8_t>> values(2);

    // Block 0: all 1s
    values[0] = std::vector<int8_t>(32, 1);
    // Block 1: all 2s
    values[1] = std::vector<int8_t>(32, 2);

    auto raw_data = create_q4_0_blocks(scales, values);
    Q4_0Tensor tensor({1, 64}, raw_data);

    std::vector<float> decoded(64);
    tensor.decodeRow(0, decoded.data());

    // Block 0: 1.0 * 1 = 1.0
    for (int i = 0; i < 32; i++)
    {
        EXPECT_NEAR(decoded[i], 1.0f, 1e-4f) << "Block 0 element " << i;
    }

    // Block 1: 2.0 * 2 = 4.0
    for (int i = 32; i < 64; i++)
    {
        EXPECT_NEAR(decoded[i], 4.0f, 1e-4f) << "Block 1 element " << i;
    }
}

/**
 * @brief Test: DecodeSpan
 */
TEST(Q4_0TensorTest, DecodeSpan)
{
    std::vector<float> scales = {1.0f};
    std::vector<int8_t> values;
    // Q4_0 can only represent -8 to +7 (4 bits)
    for (int i = 0; i < 32; i++)
    {
        values.push_back((i % 16) - 8); // -8 to +7, repeating
    }

    std::vector<std::vector<int8_t>> block_vals = {values};
    auto raw_data = create_q4_0_blocks(scales, block_vals);
    Q4_0Tensor tensor({1, 32}, raw_data);

    // Decode span from middle: elements 10-20
    std::vector<float> decoded(10);
    tensor.decodeSpan(10, 10, decoded.data());

    for (int i = 0; i < 10; i++)
    {
        int idx = 10 + i;
        float expected = 1.0f * ((idx % 16) - 8); // values[10 + i]
        EXPECT_NEAR(decoded[i], expected, 1e-4f) << "Span element " << i;
    }
}

/**
 * @brief Test: DecodeRowToBF16
 */
TEST(Q4_0TensorTest, DecodeRowToBF16)
{
    std::vector<float> scales = {1.0f};
    std::vector<int8_t> values;
    for (int i = 0; i < 32; i++)
    {
        values.push_back(i - 16);
    }

    std::vector<std::vector<int8_t>> block_vals = {values};
    auto raw_data = create_q4_0_blocks(scales, block_vals);
    Q4_0Tensor tensor({1, 32}, raw_data);

    std::vector<bfloat16> decoded_bf16(32);
    tensor.decodeRowToBF16(0, decoded_bf16.data());

    // Compare against FP32 decode
    std::vector<float> decoded_fp32(32);
    tensor.decodeRow(0, decoded_fp32.data());

    for (int i = 0; i < 32; i++)
    {
        float bf16_val = static_cast<float>(decoded_bf16[i]);
        EXPECT_NEAR(bf16_val, decoded_fp32[i], 1e-2f) << "BF16 element " << i;
    }
}

/**
 * @brief Test: Out of bounds access
 */
TEST(Q4_0TensorTest, OutOfBoundsAccess)
{
    std::vector<float> scales = {1.0f, 1.0f}; // 2 blocks for 2 rows
    std::vector<std::vector<int8_t>> values = {
        std::vector<int8_t>(32, 0),
        std::vector<int8_t>(32, 0)
    };

    auto raw_data = create_q4_0_blocks(scales, values);
    Q4_0Tensor tensor({2, 32}, raw_data);

    std::vector<float> decoded(32);

    // Try to access row 2 (should throw)
    EXPECT_THROW(tensor.decodeRow(2, decoded.data()), std::out_of_range);
}

/**
 * @brief Test: Invalid construction (size mismatch)
 */
TEST(Q4_0TensorTest, InvalidSizeMismatch)
{
    // Create data for 1 block (32 elements) but claim shape is 2x32
    std::vector<float> scales = {1.0f};
    std::vector<std::vector<int8_t>> values(1, std::vector<int8_t>(32, 0));

    auto raw_data = create_q4_0_blocks(scales, values);

    // Should throw due to size mismatch
    EXPECT_THROW(Q4_0Tensor tensor({2, 32}, raw_data), std::invalid_argument);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
