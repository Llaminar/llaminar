#include "../src/tensors/Q4_1Tensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>

using namespace llaminar;

/**
 * @brief Test suite for Q4_1 quantized tensor
 *
 * Tests decoding of 4-bit quantized values with scale + min
 */
class Q4_1TensorTest : public ::testing::Test
{
protected:
    // Helper to create a Q4_1 block with known values
    std::vector<uint8_t> create_q4_1_block(float scale, float min, const std::vector<int8_t> &nibbles)
    {
        std::vector<uint8_t> block_data(20); // 20 bytes per Q4_1 block

        // Convert FP32 to FP16
        auto fp32_to_fp16 = [](float f) -> uint16_t
        {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(float));
            uint32_t sign = (bits & 0x80000000) >> 16;
            int32_t exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
            uint32_t mantissa = (bits & 0x007FFFFF) >> 13;

            if (exponent <= 0)
            {
                return sign;
            }
            if (exponent >= 31)
            {
                return sign | 0x7C00;
            }

            return sign | (exponent << 10) | mantissa;
        };

        uint16_t scale_fp16 = fp32_to_fp16(scale);
        uint16_t min_fp16 = fp32_to_fp16(min);

        std::memcpy(block_data.data(), &scale_fp16, 2);
        std::memcpy(block_data.data() + 2, &min_fp16, 2);

        // Pack nibbles (32 nibbles → 16 bytes)
        for (size_t i = 0; i < 16; i++)
        {
            uint8_t low_nibble = nibbles[i * 2] & 0x0F;
            uint8_t high_nibble = nibbles[i * 2 + 1] & 0x0F;
            block_data[4 + i] = low_nibble | (high_nibble << 4);
        }

        return block_data;
    }
};

TEST_F(Q4_1TensorTest, BasicConstruction)
{
    std::vector<int> shape = {2, 32};
    std::vector<uint8_t> raw_data(40); // 2 blocks × 20 bytes

    Q4_1Tensor tensor(shape, raw_data);

    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 32);
    EXPECT_EQ(tensor.size(), 64);
    EXPECT_EQ(tensor.quant_type(), QuantType::Q4_1);
    EXPECT_NEAR(tensor.compression_ratio(), 6.4f, 0.1f);
}

TEST_F(Q4_1TensorTest, DecodeRowZeros)
{
    std::vector<int> shape = {1, 32};
    
    // Create block with scale=1.0, min=0.0, all nibbles=0
    std::vector<int8_t> nibbles(32, 0);
    auto block_data = create_q4_1_block(1.0f, 0.0f, nibbles);

    Q4_1Tensor tensor(shape, block_data);

    std::vector<float> output(32);
    tensor.decodeRow(0, output.data());

    // Expected: scale * 0 + min = 1.0 * 0 + 0.0 = 0.0
    for (int i = 0; i < 32; i++)
    {
        EXPECT_NEAR(output[i], 0.0f, 1e-5f) << "Index " << i;
    }
}

TEST_F(Q4_1TensorTest, DecodeRowKnownPattern)
{
    std::vector<int> shape = {1, 32};
    
    // scale=2.0, min=1.0
    // Formula: value = scale * quant + min = 2.0 * quant + 1.0
    float scale = 2.0f;
    float min = 1.0f;
    
    std::vector<int8_t> nibbles(32);
    for (int i = 0; i < 32; i++)
    {
        nibbles[i] = i % 16; // Cycle through 0-15
    }
    
    auto block_data = create_q4_1_block(scale, min, nibbles);
    Q4_1Tensor tensor(shape, block_data);

    std::vector<float> output(32);
    tensor.decodeRow(0, output.data());

    // Verify decoding: value = 2.0 * quant + 1.0
    for (int i = 0; i < 32; i++)
    {
        float expected = scale * (i % 16) + min;
        EXPECT_NEAR(output[i], expected, 1e-3f) << "Index " << i;
    }
}

TEST_F(Q4_1TensorTest, DecodeRowMultipleBlocks)
{
    std::vector<int> shape = {1, 64};
    
    std::vector<int8_t> nibbles1(32);
    std::vector<int8_t> nibbles2(32);
    
    for (int i = 0; i < 32; i++)
    {
        nibbles1[i] = i % 8;
        nibbles2[i] = (i % 8) + 8;
    }
    
    auto block1 = create_q4_1_block(1.0f, 0.0f, nibbles1);
    auto block2 = create_q4_1_block(2.0f, 5.0f, nibbles2);
    
    std::vector<uint8_t> raw_data;
    raw_data.insert(raw_data.end(), block1.begin(), block1.end());
    raw_data.insert(raw_data.end(), block2.begin(), block2.end());
    
    Q4_1Tensor tensor(shape, raw_data);

    std::vector<float> output(64);
    tensor.decodeRow(0, output.data());

    // First block: scale=1.0, min=0.0
    for (int i = 0; i < 32; i++)
    {
        float expected = 1.0f * (i % 8) + 0.0f;
        EXPECT_NEAR(output[i], expected, 1e-3f) << "Block 1, index " << i;
    }

    // Second block: scale=2.0, min=5.0
    for (int i = 0; i < 32; i++)
    {
        float expected = 2.0f * ((i % 8) + 8) + 5.0f;
        EXPECT_NEAR(output[32 + i], expected, 1e-3f) << "Block 2, index " << i;
    }
}

TEST_F(Q4_1TensorTest, DecodeSpan)
{
    std::vector<int> shape = {1, 64};
    
    std::vector<int8_t> nibbles(32, 5); // All quants = 5
    auto block1 = create_q4_1_block(1.0f, 2.0f, nibbles);
    auto block2 = create_q4_1_block(3.0f, 4.0f, nibbles);
    
    std::vector<uint8_t> raw_data;
    raw_data.insert(raw_data.end(), block1.begin(), block1.end());
    raw_data.insert(raw_data.end(), block2.begin(), block2.end());
    
    Q4_1Tensor tensor(shape, raw_data);

    // Decode span across block boundary
    std::vector<float> output(16);
    tensor.decodeSpan(24, 16, output.data());

    // First 8 from block 1: 1.0 * 5 + 2.0 = 7.0
    for (int i = 0; i < 8; i++)
    {
        EXPECT_NEAR(output[i], 7.0f, 1e-3f) << "Block 1, index " << i;
    }

    // Next 8 from block 2: 3.0 * 5 + 4.0 = 19.0
    for (int i = 8; i < 16; i++)
    {
        EXPECT_NEAR(output[i], 19.0f, 1e-3f) << "Block 2, index " << i;
    }
}

TEST_F(Q4_1TensorTest, DecodeRowToBF16)
{
    std::vector<int> shape = {1, 32};
    
    std::vector<int8_t> nibbles(32);
    for (int i = 0; i < 32; i++)
    {
        nibbles[i] = i % 10;
    }
    
    auto block_data = create_q4_1_block(0.5f, 1.5f, nibbles);
    Q4_1Tensor tensor(shape, block_data);

    std::vector<bfloat16> bf16_output(32);
    tensor.decodeRowToBF16(0, bf16_output.data());

    // Verify BF16 values are non-zero (actual conversion tested elsewhere)
    for (int i = 0; i < 32; i++)
    {
        EXPECT_NE(bf16_output[i], 0) << "Index " << i;
    }
}

TEST_F(Q4_1TensorTest, OutOfBoundsAccess)
{
    std::vector<int> shape = {2, 32};
    std::vector<uint8_t> raw_data(40); // 2 blocks

    Q4_1Tensor tensor(shape, raw_data);

    std::vector<float> buffer(32);
    EXPECT_THROW(tensor.decodeRow(2, buffer.data()), std::out_of_range);
    EXPECT_THROW(tensor.decodeRow(10, buffer.data()), std::out_of_range);
}

TEST_F(Q4_1TensorTest, InvalidSizeMismatch)
{
    std::vector<int> shape = {1, 32};
    std::vector<uint8_t> raw_data(10); // Too small

    EXPECT_THROW(Q4_1Tensor(shape, raw_data), std::invalid_argument);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
