#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>

#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/FP16Utils.h"

using namespace llaminar2;

class Q4_K_TranscodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::random_device rd;
        rng.seed(rd());
    }

    float fp16_to_fp32(uint16_t h)
    {
        return llaminar2::fp16_to_fp32(h);
    }

    uint16_t fp32_to_fp16(float f)
    {
        return llaminar2::fp32_to_fp16(f);
    }

    // Scalar reference implementation
    void RunScalar(const Q4_KBlock &block, int sub_block_idx, float *out_scale, float *out_min)
    {
        // Use the scalar implementation from SIMDHelpers directly as reference for AVX2
        llaminar2::simd::transcode_q4_k_to_int8_scalar(block, sub_block_idx, expected_int8, out_scale, out_min);
    }

    std::mt19937 rng;
    int8_t expected_int8[32];
    int8_t actual_int8[32];
    float expected_scale, expected_min;
    float actual_scale, actual_min;
};

TEST_F(Q4_K_TranscodeTest, ScalarVsAVX2_Random)
{
#if defined(__AVX2__)
    std::uniform_real_distribution<float> dist_d(0.5f, 2.0f);
    std::uniform_int_distribution<int> dist_q(0, 255);
    std::uniform_int_distribution<int> dist_sub(0, 7);

    for (int i = 0; i < 100; ++i)
    {
        Q4_KBlock block;
        block.d = fp32_to_fp16(dist_d(rng));
        block.dmin = fp32_to_fp16(dist_d(rng));

        for (int j = 0; j < 12; ++j)
            block.scales[j] = dist_q(rng);
        for (int j = 0; j < 128; ++j)
            block.qs[j] = dist_q(rng);

        int sub_block_idx = dist_sub(rng);

        RunScalar(block, sub_block_idx, &expected_scale, &expected_min);
        llaminar2::simd::transcode_q4_k_to_int8_avx2(block, sub_block_idx, actual_int8, &actual_scale, &actual_min);

        EXPECT_NEAR(expected_min, actual_min, 1e-3f);
        EXPECT_NEAR(expected_scale, actual_scale, 1e-3f);

        for (int j = 0; j < 32; ++j)
        {
            int diff = std::abs(expected_int8[j] - actual_int8[j]);
            EXPECT_LE(diff, 0) << "Mismatch at index " << j << " iteration " << i;
        }
    }
#else
    GTEST_SKIP() << "AVX2 not supported";
#endif
}

TEST_F(Q4_K_TranscodeTest, ScalarVsAVX512_Random)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    std::uniform_real_distribution<float> dist_d(0.5f, 2.0f);
    std::uniform_int_distribution<int> dist_q(0, 255);
    std::uniform_int_distribution<int> dist_sub(0, 7);

    for (int i = 0; i < 100; ++i)
    {
        Q4_KBlock block;
        block.d = fp32_to_fp16(dist_d(rng));
        block.dmin = fp32_to_fp16(dist_d(rng));

        for (int j = 0; j < 12; ++j)
            block.scales[j] = dist_q(rng);
        for (int j = 0; j < 128; ++j)
            block.qs[j] = dist_q(rng);

        int sub_block_idx = dist_sub(rng);

        RunScalar(block, sub_block_idx, &expected_scale, &expected_min);
        llaminar2::simd::transcode_q4_k_to_int8_avx512(block, sub_block_idx, actual_int8, &actual_scale, &actual_min);

        EXPECT_NEAR(expected_min, actual_min, 1e-3f);
        EXPECT_NEAR(expected_scale, actual_scale, 1e-3f);

        for (int j = 0; j < 32; ++j)
        {
            int diff = std::abs(expected_int8[j] - actual_int8[j]);
            EXPECT_LE(diff, 0) << "Mismatch at index " << j << " iteration " << i;
        }
    }
#else
    GTEST_SKIP() << "AVX512 not supported";
#endif
}
