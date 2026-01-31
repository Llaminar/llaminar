/**
 * @file Test__Q5_K_Transcode.cpp
 * @brief Unit tests for Q5_K transcoding and SIMD unpacking
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

class Q5_K_Transcode : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gen_.seed(42);
    }

    Q5_KBlock create_random_block()
    {
        Q5_KBlock block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        std::uniform_int_distribution<uint16_t> dist16(0, 65535);

        block.d = fp32_to_fp16(1.0f);
        block.dmin = fp32_to_fp16(0.5f);

        for (size_t i = 0; i < 12; ++i)
            block.scales[i] = dist(gen_);
        for (size_t i = 0; i < 32; ++i)
            block.qh[i] = dist(gen_);
        for (size_t i = 0; i < 128; ++i)
            block.qs[i] = dist(gen_);

        return block;
    }

    std::mt19937 gen_;
};

TEST_F(Q5_K_Transcode, ScalarUnpackCorrectness)
{
    Q5_KBlock block;
    std::memset(&block, 0, sizeof(block));

    // Sub-block 0 (first 32 elements)
    // qs[0] low nibble = 0, qh[0] bit 0 = 0 -> val = 0
    // qs[0] high nibble = 15 (ignored for sub-block 0)
    // qs[1] low nibble = 5, qh[1] bit 0 = 1 -> val = 5 + 16 = 21

    block.qs[0] = 0xF0; // Low=0, High=15
    block.qs[1] = 0x05; // Low=5, High=0

    block.qh[0] = 0x00; // Bit 0=0
    block.qh[1] = 0x01; // Bit 0=1

    int8_t output[32];
    unpack_q5_k_to_int8_scalar(block, 0, output);

    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 21);

    // Sub-block 1 (next 32 elements)
    // Uses HIGH nibbles of qs[0..31]
    // qs[0] high nibble = 15, qh[0] bit 1 = 0 -> val = 15
    // qs[1] high nibble = 0, qh[1] bit 1 = 1 -> val = 0 + 16 = 16

    block.qh[0] = 0x00; // Bit 1=0
    block.qh[1] = 0x02; // Bit 1=1

    unpack_q5_k_to_int8_scalar(block, 1, output);

    EXPECT_EQ(output[0], 15);
    EXPECT_EQ(output[1], 16);
}

TEST_F(Q5_K_Transcode, TensorInterface)
{
    // Create a Q5_K tensor (1 row, 256 cols = 1 block)
    std::vector<size_t> shape = {1, 256};
    std::vector<uint8_t> raw_data(sizeof(Q5_KBlock));

    Q5_KBlock *block = reinterpret_cast<Q5_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q5_KBlock));

    block->d = fp32_to_fp16(2.0f);
    block->dmin = fp32_to_fp16(1.0f);

    // Set scales for sub-block 0
    // get_scale_min_k4(0, scales, &sc, &m)
    // j=0 < 4. d = scales[0] & 63. m = scales[4] & 63.
    block->scales[0] = 10; // sc = 10
    block->scales[4] = 5;  // m = 5

    // Set qs for unpack test
    block->qs[0] = 0x07; // Low=7
    block->qh[0] = 0x00; // Bit 0=0

    Q5_KTensor tensor(shape, raw_data);

    // Test get_block_scale/min for sub-block 0 (k_block_offset=0)
    // scale = d * sc = 2.0 * 10 = 20.0
    // min = -dmin * m = -1.0 * 5 = -5.0

    EXPECT_FLOAT_EQ(tensor.get_block_scale(0, 0), 20.0f);
    EXPECT_FLOAT_EQ(tensor.get_block_min(0, 0), -5.0f);

    // Test unpack_block_to_int8
    int8_t output[32];
    tensor.unpack_block_to_int8(0, 0, output);

    EXPECT_EQ(output[0], 7);
}

TEST_F(Q5_K_Transcode, DispatcherConsistency)
{
    Q5_KBlock block = create_random_block();
    int8_t output_scalar[32];
    int8_t output_dispatch[32];

    for (int i = 0; i < 8; ++i)
    {
        unpack_q5_k_to_int8_scalar(block, i, output_scalar);
        unpack_q5_k_to_int8(block, i, output_dispatch);

        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(output_scalar[j], output_dispatch[j]) << "Mismatch at sub-block " << i << " element " << j;
        }
    }
}

#if defined(__AVX2__)
TEST_F(Q5_K_Transcode, AVX2Correctness)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported";
    }

    Q5_KBlock block = create_random_block();
    int8_t output_scalar[32];
    int8_t output_avx2[32];

    for (int i = 0; i < 8; ++i)
    {
        unpack_q5_k_to_int8_scalar(block, i, output_scalar);
        unpack_q5_k_to_int8_avx2(block, i, output_avx2);

        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(output_scalar[j], output_avx2[j]) << "Mismatch at sub-block " << i << " element " << j;
        }
    }
}
#endif
