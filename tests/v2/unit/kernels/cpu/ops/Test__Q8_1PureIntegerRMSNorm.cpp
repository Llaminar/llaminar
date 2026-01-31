/**
 * @file Test__Q8_1PureIntegerRMSNorm.cpp
 * @brief Unit tests for pure-integer Q8_1 RMSNorm implementation
 * @author David Sanftenberg
 * @date 2025-12-04
 *
 * Tests the experimental pure-integer RMSNorm that avoids ALL floating-point
 * operations by using fixed-point arithmetic and pre-quantized gamma weights.
 */

#include <gtest/gtest.h>
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "tensors/BlockStructures.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>

using namespace llaminar2;
using namespace llaminar2::primitives;

class Q8_1PureIntegerRMSNormTest : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // Helper: Quantize FP32 gamma to Q8 fixed-point (int16 with scale 256)
    void quantize_gamma_to_q16(const float *gamma_fp32, int16_t *gamma_q16, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float scaled = gamma_fp32[i] * 256.0f;
            int32_t q = static_cast<int32_t>(std::round(scaled));
            q = std::max(-32767, std::min(32767, q));
            gamma_q16[i] = static_cast<int16_t>(q);
        }
    }

    // Helper: Create Q8_1 blocks from FP32 data
    void quantize_fp32_to_q8_1(const float *fp32, Q8_1Block *blocks, size_t count)
    {
        size_t n_blocks = count / 32;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *src = fp32 + b * 32;
            Q8_1Block &blk = blocks[b];

            // Find max for scale
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 0.0f;
            float inv_scale = (scale > 0.0f) ? 127.0f / max_abs : 0.0f;

            // Pack scale as FP16
            uint32_t scale_bits;
            std::memcpy(&scale_bits, &scale, sizeof(float));
            uint32_t sign = (scale_bits >> 16) & 0x8000U;
            int32_t exp = ((scale_bits >> 23) & 0xFFU) - 127 + 15;
            uint32_t mant = (scale_bits >> 13) & 0x03FFU;
            if (exp <= 0)
            {
                blk.d = 0;
            }
            else if (exp >= 31)
            {
                blk.d = sign | 0x7C00U;
            }
            else
            {
                blk.d = sign | (exp << 10) | mant;
            }

            int32_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale));
                q = std::max(-127, std::min(127, q));
                blk.qs[i] = static_cast<int8_t>(q);
                sum += blk.qs[i];
            }
            blk.sum_qs = static_cast<int16_t>(sum);
        }
    }

    // Helper: Dequantize Q8_1 to FP32
    void dequantize_q8_1_to_fp32(const Q8_1Block *blocks, float *fp32, size_t count)
    {
        size_t n_blocks = count / 32;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const Q8_1Block &blk = blocks[b];
            float *dst = fp32 + b * 32;

            // Decode FP16 scale
            uint16_t d_bits = blk.d;
            int exp = (d_bits & 0x7C00) >> 10;
            uint32_t mant = d_bits & 0x03FF;
            float d;
            if (exp == 0)
            {
                d = (mant == 0) ? 0.0f : std::ldexp(static_cast<float>(mant), -24);
            }
            else if (exp == 31)
            {
                d = std::numeric_limits<float>::infinity();
            }
            else
            {
                d = std::ldexp(static_cast<float>(mant | 0x0400), exp - 25);
            }

            for (int i = 0; i < 32; ++i)
            {
                dst[i] = d * static_cast<float>(blk.qs[i]);
            }
        }
    }

    // Helper: Compute cosine similarity
    float cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 1.0f;
        return static_cast<float>(dot / std::sqrt(norm_a * norm_b));
    }

    // Helper: Reference FP32 RMSNorm
    void rmsnorm_reference(const float *input, const float *gamma, float *output,
                           size_t cols, float epsilon)
    {
        double sum_sq = 0.0;
        for (size_t i = 0; i < cols; ++i)
        {
            sum_sq += input[i] * input[i];
        }
        float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        for (size_t i = 0; i < cols; ++i)
        {
            output[i] = input[i] * inv_rms * gamma[i];
        }
    }
};

TEST_F(Q8_1PureIntegerRMSNormTest, BasicCorrectness_SingleBlock)
{
    const size_t cols = 32;
    const size_t blocks_per_row = 1;
    const float epsilon = 1e-6f;
    const int32_t epsilon_scaled = 1; // Small value for stability

    // Create input data
    std::vector<float> input_fp32(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : input_fp32)
        v = dist(rng_);

    // Create gamma
    std::vector<float> gamma_fp32(cols, 1.0f);
    std::vector<int16_t> gamma_q8(cols);
    quantize_gamma_to_q16(gamma_fp32.data(), gamma_q8.data(), cols);

    // Quantize input
    std::vector<Q8_1Block> input_q8(blocks_per_row);
    quantize_fp32_to_q8_1(input_fp32.data(), input_q8.data(), cols);

    // Run pure-integer RMSNorm
    std::vector<Q8_1Block> output_q8(blocks_per_row);
    rmsnorm_q8_1_pure_integer_row(input_q8.data(), gamma_q8.data(),
                                  output_q8.data(), blocks_per_row, epsilon_scaled);

    // Compute FP32 reference
    std::vector<float> ref_output(cols);
    rmsnorm_reference(input_fp32.data(), gamma_fp32.data(), ref_output.data(), cols, epsilon);

    // Dequantize pure-integer output
    std::vector<float> out_pure(cols);
    dequantize_q8_1_to_fp32(output_q8.data(), out_pure.data(), cols);

    float cosine = cosine_similarity(out_pure.data(), ref_output.data(), cols);
    std::cout << "SingleBlock: cosine=" << cosine << std::endl;

    // Pure integer should be very close (>0.999 cosine)
    EXPECT_GE(cosine, 0.999f) << "Pure integer implementation diverged too much";
}

TEST_F(Q8_1PureIntegerRMSNormTest, BasicCorrectness_Qwen05B_DModel)
{
    const size_t cols = 896; // Qwen 0.5B d_model
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;
    const int32_t epsilon_scaled = 1;

    // Create input data
    std::vector<float> input_fp32(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : input_fp32)
        v = dist(rng_);

    // Create gamma with varying values
    std::vector<float> gamma_fp32(cols);
    std::uniform_real_distribution<float> gamma_dist(0.5f, 1.5f);
    for (auto &v : gamma_fp32)
        v = gamma_dist(rng_);

    std::vector<int16_t> gamma_q8(cols);
    quantize_gamma_to_q16(gamma_fp32.data(), gamma_q8.data(), cols);

    // Quantize input
    std::vector<Q8_1Block> input_q8(blocks_per_row);
    quantize_fp32_to_q8_1(input_fp32.data(), input_q8.data(), cols);

    // Run pure-integer RMSNorm
    std::vector<Q8_1Block> output_q8(blocks_per_row);
    rmsnorm_q8_1_pure_integer_row(input_q8.data(), gamma_q8.data(),
                                  output_q8.data(), blocks_per_row, epsilon_scaled);

    // Compute FP32 reference
    std::vector<float> ref_output(cols);
    rmsnorm_reference(input_fp32.data(), gamma_fp32.data(), ref_output.data(), cols, epsilon);

    // Dequantize pure-integer output
    std::vector<float> out_pure(cols);
    dequantize_q8_1_to_fp32(output_q8.data(), out_pure.data(), cols);

    float cosine = cosine_similarity(out_pure.data(), ref_output.data(), cols);
    std::cout << "Qwen05B (d_model=896): cosine=" << cosine << std::endl;

    EXPECT_GE(cosine, 0.999f);
}

TEST_F(Q8_1PureIntegerRMSNormTest, PerformanceComparison_SingleRow)
{
    const size_t cols = 896;
    const size_t blocks_per_row = cols / 32;
    const float epsilon = 1e-6f;
    const int32_t epsilon_scaled = 1;
    const int warmup = 100;
    const int iterations = 1000;

    // Create input data
    std::vector<float> input_fp32(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : input_fp32)
        v = dist(rng_);

    std::vector<float> gamma_fp32(cols, 1.0f);
    std::vector<int16_t> gamma_q8(cols);
    quantize_gamma_to_q16(gamma_fp32.data(), gamma_q8.data(), cols);

    std::vector<Q8_1Block> input_q8(blocks_per_row);
    quantize_fp32_to_q8_1(input_fp32.data(), input_q8.data(), cols);
    std::vector<Q8_1Block> output_q8(blocks_per_row);

    // Benchmark pure-integer
    for (int i = 0; i < warmup; ++i)
    {
        rmsnorm_q8_1_pure_integer_row(input_q8.data(), gamma_q8.data(),
                                      output_q8.data(), blocks_per_row, epsilon_scaled);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        rmsnorm_q8_1_pure_integer_row(input_q8.data(), gamma_q8.data(),
                                      output_q8.data(), blocks_per_row, epsilon_scaled);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double pure_int_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    std::cout << "\n=== Performance (Single Row, d_model=" << cols << ") ===" << std::endl;
    std::cout << "Pure Integer:     " << pure_int_us << " us/row" << std::endl;

    // Verify correctness against FP32 reference
    std::vector<Q8_1Block> out_pure(blocks_per_row);
    rmsnorm_q8_1_pure_integer_row(input_q8.data(), gamma_q8.data(),
                                  out_pure.data(), blocks_per_row, epsilon_scaled);

    // Compute FP32 reference
    std::vector<float> input_fp32_for_ref(cols);
    dequantize_q8_1_to_fp32(input_q8.data(), input_fp32_for_ref.data(), cols);
    std::vector<float> ref_output(cols);
    rmsnorm_reference(input_fp32_for_ref.data(), gamma_fp32.data(), ref_output.data(), cols, epsilon);

    std::vector<float> fp32_pure(cols);
    dequantize_q8_1_to_fp32(out_pure.data(), fp32_pure.data(), cols);
    float cosine = cosine_similarity(fp32_pure.data(), ref_output.data(), cols);
    std::cout << "Cosine similarity vs FP32 ref: " << cosine << std::endl;
}

TEST_F(Q8_1PureIntegerRMSNormTest, EdgeCase_AllZeros)
{
    const size_t cols = 32;
    const size_t blocks_per_row = 1;
    const int32_t epsilon_scaled = 1;

    std::vector<Q8_1Block> input_q8(blocks_per_row);
    std::memset(input_q8.data(), 0, sizeof(Q8_1Block));

    std::vector<int16_t> gamma_q8(cols, 256); // gamma = 1.0 in Q8
    std::vector<Q8_1Block> output_q8(blocks_per_row);

    // Should not crash
    rmsnorm_q8_1_pure_integer_row(input_q8.data(), gamma_q8.data(),
                                  output_q8.data(), blocks_per_row, epsilon_scaled);

    // Output should be zero or near-zero
    EXPECT_EQ(output_q8[0].d, 0);
}

TEST_F(Q8_1PureIntegerRMSNormTest, EdgeCase_LargeValues)
{
    const size_t cols = 32;
    const size_t blocks_per_row = 1;
    const int32_t epsilon_scaled = 1;

    // Create input with large values
    std::vector<float> input_fp32(cols);
    for (size_t i = 0; i < cols; ++i)
    {
        input_fp32[i] = (i % 2 == 0) ? 100.0f : -100.0f;
    }

    std::vector<float> gamma_fp32(cols, 1.0f);
    std::vector<int16_t> gamma_q8(cols);
    quantize_gamma_to_q16(gamma_fp32.data(), gamma_q8.data(), cols);

    std::vector<Q8_1Block> input_q8(blocks_per_row);
    quantize_fp32_to_q8_1(input_fp32.data(), input_q8.data(), cols);

    std::vector<Q8_1Block> output_q8(blocks_per_row);
    rmsnorm_q8_1_pure_integer_row(input_q8.data(), gamma_q8.data(),
                                  output_q8.data(), blocks_per_row, epsilon_scaled);

    // Should produce valid output
    EXPECT_NE(output_q8[0].d, 0) << "Output scale should be non-zero";
}
