/**
 * @file Test__Q5_1_Transcode.cpp
 * @brief Unit tests for Q5_1 transcoding and SIMD unpacking
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "utils/MPIContext.h"
#include <vector>
#include <random>
#include <cstring>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q5_1_Transcode : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gen_.seed(42);
    }

    Q5_1Block create_random_block()
    {
        Q5_1Block block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        std::uniform_int_distribution<uint8_t> dist_qh(0, 255);

        for (size_t i = 0; i < 16; ++i)
            block.qs[i] = dist(gen_);
        for (size_t i = 0; i < 4; ++i)
            block.qh[i] = dist_qh(gen_);

        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.5f);
        return block;
    }

    std::mt19937 gen_;
};

TEST_F(Q5_1_Transcode, ScalarUnpackCorrectness)
{
    Q5_1Block block;
    // Q5_1 layout:
    //   output[j] = low nibble of qs[j] | high bit from qh bit j
    //   output[j+16] = high nibble of qs[j] | high bit from qh bit j+12
    //
    // Setting up:
    //   qs[0] = 0xF0 -> low nibble = 0, high nibble = 15
    //   qh = 0x00010000 -> bit 16 = 1 (for output[16])
    //
    // Expected:
    //   output[0] = (qs[0] & 0x0F) | ((qh >> 0) << 4 & 0x10) = 0 | 0 = 0
    //   output[16] = (qs[0] >> 4) | ((qh >> 12) & 0x10) = 15 | 16 = 31

    std::memset(&block, 0, sizeof(block));
    block.qs[0] = 0xF0; // Low=0, High=15
    // qh is little-endian uint32: we need bit 16 to be set
    // bit 16 is used for output[16] via ((qh >> 12) & 0x10)
    // qh >> 12 needs bit 4 set, so qh needs bit 16 set
    block.qh[2] = 0x01; // Sets bit 16 of the 32-bit qh

    int8_t output[32];
    unpack_q5_1_to_int8_scalar(block, output);

    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[16], 31); // High nibble + high bit
}

TEST_F(Q5_1_Transcode, TensorInterface)
{
    // Create a small Q5_1 tensor (1 row, 32 cols = 1 block)
    std::vector<size_t> shape = {1, 32};
    std::vector<uint8_t> raw_data(sizeof(Q5_1Block));

    Q5_1Block *block = reinterpret_cast<Q5_1Block *>(raw_data.data());
    block->d = fp32_to_fp16(2.5f);
    block->m = fp32_to_fp16(0.5f);
    block->qs[0] = 0xF0; // Low nibble=0, High nibble=15
    block->qh[2] = 0x01; // Bit 16 set for output[16]

    Q5_1Tensor tensor(shape, raw_data);

    // Test get_block_scale
    EXPECT_FLOAT_EQ(tensor.get_block_scale(0, 0), 2.5f);
    EXPECT_FLOAT_EQ(tensor.get_block_min(0, 0), 0.5f);

    // Test unpack_block_to_int8
    int8_t output[32];
    tensor.unpack_block_to_int8(0, 0, output);

    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[16], 31); // High nibble (15) + high bit (16) = 31
}

TEST_F(Q5_1_Transcode, DispatcherConsistency)
{
    Q5_1Block block = create_random_block();
    int8_t output_scalar[32];
    int8_t output_dispatch[32];

    unpack_q5_1_to_int8_scalar(block, output_scalar);
    unpack_q5_1_to_int8(block, output_dispatch);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_dispatch[i]);
    }
}
