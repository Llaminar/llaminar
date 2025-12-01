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
#include "v2/tensors/Tensors.h"

using namespace llaminar2;

class Q6_K_TranscodeTest : public ::testing::Test
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
    void RunScalar(const Q6_KBlock &block, int sub_block_idx, float *out_scale, float *out_min)
    {
        llaminar2::simd::transcode_q6_k_to_int8_scalar(block, sub_block_idx, expected_int8, out_scale, out_min);
    }

    std::mt19937 rng;
    int8_t expected_int8[32];
    int8_t actual_int8[32];
    float expected_scale, expected_min;
    float actual_scale, actual_min;
};

TEST_F(Q6_K_TranscodeTest, ScalarVsAVX2_Random)
{
#if defined(__AVX2__)
    std::uniform_real_distribution<float> dist_d(0.5f, 2.0f);
    std::uniform_int_distribution<int> dist_q(0, 255);
    std::uniform_int_distribution<int> dist_sc(-128, 127);
    std::uniform_int_distribution<int> dist_sub(0, 7);

    for (int i = 0; i < 100; ++i)
    {
        Q6_KBlock block;
        block.d = fp32_to_fp16(dist_d(rng));

        for (int j = 0; j < 16; ++j)
            block.scales[j] = dist_sc(rng);
        for (int j = 0; j < 128; ++j)
            block.ql[j] = dist_q(rng);
        for (int j = 0; j < 64; ++j)
            block.qh[j] = dist_q(rng);

        int sub_block_idx = dist_sub(rng);

        RunScalar(block, sub_block_idx, &expected_scale, &expected_min);
        llaminar2::simd::transcode_q6_k_to_int8_avx2(block, sub_block_idx, actual_int8, &actual_scale, &actual_min);

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

TEST_F(Q6_K_TranscodeTest, ScalarVsAVX512_Random)
{
#if defined(__AVX512F__)
    std::uniform_real_distribution<float> dist_d(0.5f, 2.0f);
    std::uniform_int_distribution<int> dist_q(0, 255);
    std::uniform_int_distribution<int> dist_sc(-128, 127);
    std::uniform_int_distribution<int> dist_sub(0, 7);

    for (int i = 0; i < 100; ++i)
    {
        Q6_KBlock block;
        block.d = fp32_to_fp16(dist_d(rng));

        for (int j = 0; j < 16; ++j)
            block.scales[j] = dist_sc(rng);
        for (int j = 0; j < 128; ++j)
            block.ql[j] = dist_q(rng);
        for (int j = 0; j < 64; ++j)
            block.qh[j] = dist_q(rng);

        int sub_block_idx = dist_sub(rng);

        RunScalar(block, sub_block_idx, &expected_scale, &expected_min);
        llaminar2::simd::transcode_q6_k_to_int8_avx512(block, sub_block_idx, actual_int8, &actual_scale, &actual_min);

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

TEST_F(Q6_K_TranscodeTest, Q6_KTensor_Unpack)
{
    // Create a Q6_KTensor with 1 row, 256 columns (1 super-block)
    std::vector<size_t> shape = {1, 256};
    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));

    // Initialize with random data
    Q6_KBlock *block = reinterpret_cast<Q6_KBlock *>(raw_data.data());
    std::uniform_real_distribution<float> dist_d(0.5f, 2.0f);
    std::uniform_int_distribution<int> dist_q(0, 255);
    std::uniform_int_distribution<int> dist_sc(-128, 127);

    block->d = fp32_to_fp16(dist_d(rng));
    for (int j = 0; j < 16; ++j)
        block->scales[j] = dist_sc(rng);
    for (int j = 0; j < 128; ++j)
        block->ql[j] = dist_q(rng);
    for (int j = 0; j < 64; ++j)
        block->qh[j] = dist_q(rng);

    Q6_KTensor tensor(shape, raw_data);

    // Test unpacking each sub-block
    for (int sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        int8_t output[32];
        tensor.unpack_block_to_int8(0, sub_idx, output);

        // Verify against scalar implementation
        RunScalar(*block, sub_idx, &expected_scale, &expected_min);

        for (int j = 0; j < 32; ++j)
        {
            EXPECT_EQ(output[j], expected_int8[j]);
        }

        float scale = tensor.get_block_scale(0, sub_idx);
        EXPECT_NEAR(scale, expected_scale, 1e-3f);

        float min = tensor.get_block_min(0, sub_idx);
        EXPECT_NEAR(min, expected_min, 1e-3f);
    }
}
