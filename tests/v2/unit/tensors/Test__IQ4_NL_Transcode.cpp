#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <iostream>
#include <cstring>

#include "v2/tensors/Tensors.h"
#include "v2/tensors/IQQuantTables.h"
#include "v2/tensors/FP16Utils.h"
#include "v2/tensors/SIMDHelpers.h"

using namespace llaminar2;

class IQ4_NL_TranscodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize random number generator
        std::random_device rd;
        rng.seed(rd());
    }

    std::mt19937 rng;
};

TEST_F(IQ4_NL_TranscodeTest, UnpackBlockToInt8)
{
    // Create a random block
    IQ4_NLBlock block;
    block.d = fp32_to_fp16(1.5f); // Arbitrary scale

    // Fill with random indices
    std::uniform_int_distribution<int> dist(0, 255);
    for (int i = 0; i < 16; ++i)
    {
        block.qs[i] = static_cast<uint8_t>(dist(rng));
    }

    // Expected values
    int8_t expected[32];
    for (int i = 0; i < 16; ++i)
    {
        uint8_t qbyte = block.qs[i];
        expected[i] = kvalues_iq4nl_i8[qbyte & 0x0F];
        expected[i + 16] = kvalues_iq4nl_i8[qbyte >> 4];
    }

    // Actual values
    int8_t actual[32];
    simd::unpack_iq4_nl_to_int8(block, actual);

    // Verify
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(IQ4_NL_TranscodeTest, TensorInterface)
{
    // Create a small tensor
    size_t rows = 2;
    size_t cols = 64; // 2 blocks per row
    std::vector<size_t> shape = {rows, cols};

    size_t num_blocks = rows * (cols / 32);
    std::vector<uint8_t> raw_data(num_blocks * sizeof(IQ4_NLBlock));
    IQ4_NLBlock *blocks = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());

    // Initialize blocks
    for (size_t i = 0; i < num_blocks; ++i)
    {
        blocks[i].d = fp32_to_fp16(1.0f + i * 0.1f);
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>(i + j);
        }
    }

    IQ4_NLTensor tensor(shape, raw_data);

    // Test IINT8Unpackable interface
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t kb = 0; kb < cols / 32; ++kb)
        {
            size_t block_idx = r * (cols / 32) + kb;
            const IQ4_NLBlock &block = blocks[block_idx];

            // Check scale
            float expected_scale = fp16_to_fp32(block.d);
            EXPECT_FLOAT_EQ(tensor.get_block_scale(r, kb), expected_scale);

            // Check min
            EXPECT_FLOAT_EQ(tensor.get_block_min(r, kb), 0.0f);

            // Check unpack
            int8_t actual[32];
            tensor.unpack_block_to_int8(r, kb, actual);

            for (int j = 0; j < 16; ++j)
            {
                uint8_t qbyte = block.qs[j];
                EXPECT_EQ(actual[j], kvalues_iq4nl_i8[qbyte & 0x0F]);
                EXPECT_EQ(actual[j + 16], kvalues_iq4nl_i8[qbyte >> 4]);
            }
        }
    }
}