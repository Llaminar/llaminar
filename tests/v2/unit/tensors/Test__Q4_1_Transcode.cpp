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

class Q4_1_TranscodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize random number generator
        std::random_device rd;
        rng.seed(rd());
    }

    // Helper to convert fp16 to float (simple version for test)
    float fp16_to_fp32(uint16_t h)
    {
        return llaminar2::fp16_to_fp32(h);
    }

    uint16_t fp32_to_fp16(float f)
    {
        return llaminar2::fp32_to_fp16(f);
    }

    // Scalar implementation of Q4_1 -> INT8 transcoding for verification
    void RunScalar(const Q4_1Block &block, float *out_scale, float *out_min)
    {
        float d = fp16_to_fp32(block.d);
        float m = fp16_to_fp32(block.m);

        // 1. Dequantize to FP32
        std::vector<float> temp_fp32(32);
        uint8_t qs[32];

        for (int j = 0; j < 16; ++j)
        {
            qs[j] = block.qs[j] & 0xF;
            qs[j + 16] = block.qs[j] >> 4;
        }

        for (int j = 0; j < 32; ++j)
        {
            temp_fp32[j] = d * qs[j] + m;
        }

        // 2. Find min/max for requantization
        float min_val = temp_fp32[0];
        float max_val = temp_fp32[0];
        for (float val : temp_fp32)
        {
            if (val < min_val)
                min_val = val;
            if (val > max_val)
                max_val = val;
        }

        float range = max_val - min_val;

        // Handle constant value case (range ~ 0)
        // The AVX512 implementation checks range_q < 0.5f.
        // range_q = (max_val - min_val) / d.
        // So if range < 0.5f * d, it treats it as constant.
        // However, for exact comparison, let's just check if range is very small.

        if (range < 1e-5f)
        {
            *out_scale = 0.0f;
            *out_min = min_val;
            for (int j = 0; j < 32; ++j)
            {
                expected_int8[j] = -128;
            }
            return;
        }

        // 3. Compute scale and zero point for INT8
        // Quantization parameters (float -> int8)
        float quant_scale = 255.0f / range;
        float quant_bias = -min_val * quant_scale - 128.0f;

        // Dequantization parameters (int8 -> float) - what AVX512 returns
        *out_scale = range / 255.0f;
        *out_min = min_val + (*out_scale) * 128.0f;

        for (int j = 0; j < 32; ++j)
        {
            float val = temp_fp32[j];
            float f = val * quant_scale + quant_bias;
            int32_t i = static_cast<int32_t>(std::round(f));
            // Clamp to [-128, 127]
            if (i < -128)
                i = -128;
            if (i > 127)
                i = 127;
            expected_int8[j] = static_cast<int8_t>(i);
        }
    }

    std::mt19937 rng;
    int8_t expected_int8[32];
    int8_t actual_int8[32];
    float expected_scale, expected_min;
    float actual_scale, actual_min;
};

TEST_F(Q4_1_TranscodeTest, ScalarVsAVX512_Random)
{
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    std::uniform_real_distribution<float> dist_d(0.1f, 5.0f);
    std::uniform_real_distribution<float> dist_m(-2.0f, 2.0f);
    std::uniform_int_distribution<int> dist_q(0, 255);

    for (int i = 0; i < 100; ++i)
    {
        Q4_1Block block;
        float d_val = dist_d(rng);
        float m_val = dist_m(rng);

        uint16_t d_fp16 = fp32_to_fp16(d_val);
        uint16_t m_fp16 = fp32_to_fp16(m_val);

        block.d = d_fp16;
        block.m = m_fp16;

        for (int j = 0; j < 16; ++j)
        {
            block.qs[j] = static_cast<uint8_t>(dist_q(rng));
        }

        // Run Scalar Reference
        RunScalar(block, &expected_scale, &expected_min);

        // Run AVX512 Implementation
        llaminar2::simd::transcode_q4_1_to_int8_avx512(block, actual_int8, &actual_scale, &actual_min);

        // Compare Scale and Min
        // Note: AVX512 reduction might be slightly different than scalar loop due to order of operations
        // but should be very close.
        EXPECT_NEAR(expected_min, actual_min, 1e-3f) << "Min mismatch at iteration " << i;
        EXPECT_NEAR(expected_scale, actual_scale, 1e-3f) << "Scale mismatch at iteration " << i;

        // Compare INT8 values
        for (int j = 0; j < 32; ++j)
        {
            // Allow off-by-one error due to floating point rounding differences
            int diff = std::abs(expected_int8[j] - actual_int8[j]);
            EXPECT_LE(diff, 1) << "Mismatch at index " << j << " iteration " << i
                               << " Expected: " << (int)expected_int8[j]
                               << " Actual: " << (int)actual_int8[j];
        }
    }
#else
    GTEST_SKIP() << "AVX512 not supported";
#endif
}

TEST_F(Q4_1_TranscodeTest, ConstantValue)
{
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    // Test with a block where all values are the same
    float d_val = 1.0f;
    float m_val = 0.0f;
    uint16_t d_fp16 = fp32_to_fp16(d_val);
    uint16_t m_fp16 = fp32_to_fp16(m_val);

    Q4_1Block block;
    block.d = d_fp16;
    block.m = m_fp16;
    // Set all nibbles to 7
    for (int j = 0; j < 16; ++j)
    {
        block.qs[j] = (7) | (7 << 4);
    }

    RunScalar(block, &expected_scale, &expected_min);
    llaminar2::simd::transcode_q4_1_to_int8_avx512(block, actual_int8, &actual_scale, &actual_min);

    EXPECT_NEAR(expected_min, actual_min, 1e-5f);
    // Scale might be different if range is 0 (handled by epsilon), but should be consistent
    EXPECT_NEAR(expected_scale, actual_scale, 1e-5f);

    for (int j = 0; j < 32; ++j)
    {
        EXPECT_EQ(expected_int8[j], actual_int8[j]);
    }
#else
    GTEST_SKIP() << "AVX512 not supported";
#endif
}
