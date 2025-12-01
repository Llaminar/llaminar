/**
 * @file Test__Q5_0_Transcode.cpp
 * @brief Unit tests for Q5_0 transcoding and SIMD unpacking
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/tensors/SIMDHelpers.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include <vector>
#include <random>
#include <cstring>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q5_0_Transcode : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gen_.seed(42);
    }

    Q5_0Block create_random_block()
    {
        Q5_0Block block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        std::uniform_int_distribution<uint8_t> dist_qh(0, 255);

        for (size_t i = 0; i < 16; ++i)
            block.qs[i] = dist(gen_);
        for (size_t i = 0; i < 4; ++i)
            block.qh[i] = dist_qh(gen_);

        block.d = fp32_to_fp16(1.0f);
        return block;
    }

    std::mt19937 gen_;
};

TEST_F(Q5_0_Transcode, ScalarUnpackCorrectness)
{
    Q5_0Block block;
    // Set specific pattern matching llama.cpp dequantize_row_q5_0 layout:
    // - output[0..15] = low nibbles of qs[0..15] + high bits from qh
    // - output[16..31] = high nibbles of qs[0..15] + high bits from qh
    //
    // From decodeBlockScalar:
    //   xh_0 = ((qh >> (j + 0)) << 4) & 0x10  -> for low nibble
    //   xh_1 = ((qh >> (j + 12)) & 0x10)       -> for high nibble
    //
    // For j=0 (output[0] and output[16]):
    //   xh_0 = ((qh >> 0) << 4) & 0x10 -> bit 0 of qh shifted to bit 4
    //   xh_1 = ((qh >> 12) & 0x10)     -> bit 16 of qh (shift 12, then mask bit 4)
    //
    // To get xh_1 = 0x10, we need qh bit 16 set (byte 2, bit 0)

    std::memset(&block, 0, sizeof(block));
    block.qs[0] = 0xF0; // Low=0, High=15
    block.qh[2] = 0x01; // Set bit 16 (byte 2, bit 0) for high nibble of output[16]

    int8_t output[32];
    unpack_q5_0_to_int8_scalar(block, output);

    // output[0] = (qs[0] & 0x0F) | xh_0 - 16 = (0 | 0) - 16 = -16
    // output[16] = (qs[0] >> 4) | xh_1 - 16 = (15 | 16) - 16 = 15
    EXPECT_EQ(output[0], -16);
    EXPECT_EQ(output[16], 15);
}

TEST_F(Q5_0_Transcode, TensorInterface)
{
    // Create a small Q5_0 tensor (1 row, 32 cols = 1 block)
    std::vector<size_t> shape = {1, 32};
    std::vector<uint8_t> raw_data(sizeof(Q5_0Block));

    Q5_0Block *block = reinterpret_cast<Q5_0Block *>(raw_data.data());
    std::memset(block, 0, sizeof(Q5_0Block));
    block->d = fp32_to_fp16(2.5f);
    block->qs[0] = 0xF0; // Low=0, High=15
    block->qh[2] = 0x01; // Set bit 16 (byte 2, bit 0) for high nibble of output[16]

    Q5_0Tensor tensor(shape, raw_data);

    // Test get_block_scale
    EXPECT_FLOAT_EQ(tensor.get_block_scale(0, 0), 2.5f);
    EXPECT_FLOAT_EQ(tensor.get_block_min(0, 0), 0.0f);

    // Test unpack_block_to_int8 - uses new llama.cpp-compatible layout
    int8_t output[32];
    tensor.unpack_block_to_int8(0, 0, output);

    EXPECT_EQ(output[0], -16); // Low nibble of qs[0] with qh bit 0 = 0
    EXPECT_EQ(output[16], 15); // High nibble of qs[0] with qh bit 16 = 1
}

TEST_F(Q5_0_Transcode, DispatcherConsistency)
{
    Q5_0Block block = create_random_block();
    int8_t output_scalar[32];
    int8_t output_dispatch[32];

    unpack_q5_0_to_int8_scalar(block, output_scalar);
    unpack_q5_0_to_int8(block, output_dispatch);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_dispatch[i]);
    }
}
