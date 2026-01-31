/**
 * @file Test__IQ2_XS_UnpackVectorization.cpp
 * @brief Test suite for IQ2_XS unpack_superblock_to_int8 vectorization
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/IQQuantTables.h"
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <cstring>

using namespace llaminar2;

class IQ2_XSUnpackTest : public ::testing::Test
{
protected:
    static constexpr size_t SUPER_BLOCK_SIZE = 256;
    static constexpr size_t SUB_BLOCKS_PER_SUPER = 8;
    static constexpr size_t SUB_BLOCK_SIZE = 32;

    void SetUp() override
    {
        gen_.seed(42);
    }

    IQ2_XSBlock createRandomBlock()
    {
        IQ2_XSBlock block;
        std::uniform_real_distribution<float> scale_dist(0.001f, 2.0f);
        block.d = fp32_to_fp16(scale_dist(gen_));

        std::uniform_int_distribution<uint16_t> data_dist(0, 65535);
        for (size_t i = 0; i < 32; ++i)
        {
            block.qs[i] = data_dist(gen_);
        }

        std::uniform_int_distribution<uint16_t> scale_byte_dist(0, 255);
        for (size_t i = 0; i < 8; ++i)
        {
            block.scales[i] = (uint8_t)scale_byte_dist(gen_);
        }

        return block;
    }

    std::mt19937 gen_;
};

TEST_F(IQ2_XSUnpackTest, AVX512Parity)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX512 not supported";
    }

    IQ2_XSBlock block = createRandomBlock();

    // Reference (Scalar/Loop)
    int8_t ref_output[SUPER_BLOCK_SIZE];
    float ref_scales[SUB_BLOCKS_PER_SUPER];
    float ref_mins[SUB_BLOCKS_PER_SUPER];

    // Force scalar path by calling decode_iq2xs_to_q8_0 manually
    for (int i = 0; i < 8; ++i)
    {
        uint16_t scale_fp16;
        llaminar2::simd::decode_iq2xs_to_q8_0(block, i, ref_output + i * 32, &scale_fp16);
        ref_scales[i] = fp16_to_fp32(scale_fp16);
        ref_mins[i] = 0.0f;
    }

    // Optimized (AVX512)
    int8_t opt_output[SUPER_BLOCK_SIZE];
    float opt_scales[SUB_BLOCKS_PER_SUPER];
    float opt_mins[SUB_BLOCKS_PER_SUPER];

    llaminar2::simd::unpack_iq2_xs_superblock_to_int8_avx512(block, opt_output, opt_scales, opt_mins);

    // Compare
    for (int i = 0; i < SUPER_BLOCK_SIZE; ++i)
    {
        int diff = std::abs(ref_output[i] - opt_output[i]);
        EXPECT_LE(diff, 1) << "Mismatch at index " << i << ": ref=" << (int)ref_output[i] << " opt=" << (int)opt_output[i];
    }

    for (int i = 0; i < SUB_BLOCKS_PER_SUPER; ++i)
    {
        EXPECT_NEAR(ref_scales[i], opt_scales[i], 0.05f * ref_scales[i] + 1e-4f) << "Scale mismatch at subblock " << i;
        EXPECT_EQ(ref_mins[i], opt_mins[i]) << "Min mismatch at subblock " << i;
    }
}
