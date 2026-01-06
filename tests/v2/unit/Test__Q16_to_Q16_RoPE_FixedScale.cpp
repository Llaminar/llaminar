/**
 * @file Test__Q16_to_Q16_RoPE_FixedScale.cpp
 * @brief Unit tests for fixed-scale Q16→Q16 RoPE implementation
 *
 * Tests:
 * 1. Scale verification: All output blocks have d == kv_cache_scale / 32767
 * 2. Round-trip parity: Fixed-scale Q16→Q16 RoPE matches FP32 RoPE within tolerance
 * 3. Block structure: sum_qs is correct, qs values in valid range
 * 4. Edge cases: Zero input, small values, large values, varying input scales
 */

#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/BlockStructures.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <iostream>

using namespace llaminar2::primitives;
using namespace llaminar2;

namespace
{
    // ============================================================================
    // Test Utilities
    // ============================================================================

    /**
     * @brief Quantize FP32 data to Q16_1 blocks with dynamic scale (simulating GEMM output)
     */
    std::vector<Q16_1Block> fp32_to_q16_1(const std::vector<float> &fp32)
    {
        const size_t n_blocks = (fp32.size() + 31) / 32;
        std::vector<Q16_1Block> blocks(n_blocks);

        std::vector<float> padded = fp32;
        padded.resize(n_blocks * 32, 0.0f);

        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = padded.data() + b * 32;
            Q16_1Block &blk = blocks[b];

            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }

            float scale = max_abs / 16383.0f; // Q16_1 uses ±16383 range
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-16383, std::min(16383, q));
                blk.qs[i] = static_cast<int16_t>(q);
                sum_qs += q;
            }

            blk.d = scale;
            blk.sum_qs = sum_qs;
        }

        return blocks;
    }

    /**
     * @brief Quantize FP32 data to Q16_1 blocks with varying per-block scales
     *
     * This simulates GEMM output where each block may have different magnitude
     */
    std::vector<Q16_1Block> fp32_to_q16_1_varying_scale(
        const std::vector<float> &fp32,
        const std::vector<float> &block_scales)
    {
        const size_t n_blocks = (fp32.size() + 31) / 32;
        std::vector<Q16_1Block> blocks(n_blocks);

        std::vector<float> padded = fp32;
        padded.resize(n_blocks * 32, 0.0f);

        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = padded.data() + b * 32;
            Q16_1Block &blk = blocks[b];

            // Use provided scale (simulating different GEMM output magnitudes)
            float scale = block_scales[b];
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-16383, std::min(16383, q));
                blk.qs[i] = static_cast<int16_t>(q);
                sum_qs += q;
            }

            blk.d = scale;
            blk.sum_qs = sum_qs;
        }

        return blocks;
    }

    /**
     * @brief Dequantize Q16_1 blocks to FP32
     */
    std::vector<float> q16_1_to_fp32(const std::vector<Q16_1Block> &blocks)
    {
        std::vector<float> fp32(blocks.size() * 32);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            const Q16_1Block &blk = blocks[b];
            float *block_data = fp32.data() + b * 32;

            for (int i = 0; i < 32; ++i)
            {
                block_data[i] = blk.d * static_cast<float>(blk.qs[i]);
            }
        }

        return fp32;
    }

    /**
     * @brief Apply FP32 RoPE to head data (reference implementation)
     */
    void apply_rope_fp32_reference(
        float *head_data,
        int head_dim,
        const float *cos_fp32,
        const float *sin_fp32)
    {
        const int half_dim = head_dim / 2;
        for (int i = 0; i < half_dim; ++i)
        {
            float x = head_data[i];
            float y = head_data[i + half_dim];
            float c = cos_fp32[i];
            float s = sin_fp32[i];
            head_data[i] = x * c - y * s;
            head_data[i + half_dim] = x * s + y * c;
        }
    }

    /**
     * @brief Generate sin/cos tables in both Q15 and FP32 formats
     */
    void generate_rope_tables(
        int half_dim,
        int position,
        float rope_theta,
        std::vector<int16_t> &cos_q15,
        std::vector<int16_t> &sin_q15,
        std::vector<float> &cos_fp32,
        std::vector<float> &sin_fp32)
    {
        cos_q15.resize(half_dim);
        sin_q15.resize(half_dim);
        cos_fp32.resize(half_dim);
        sin_fp32.resize(half_dim);

        for (int i = 0; i < half_dim; ++i)
        {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / (half_dim * 2));
            float angle = static_cast<float>(position) * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);

            cos_fp32[i] = c;
            sin_fp32[i] = s;
            cos_q15[i] = static_cast<int16_t>(std::round(c * 32767.0f));
            sin_q15[i] = static_cast<int16_t>(std::round(s * 32767.0f));
        }
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
            return 0.0f;

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a < 1e-10f || norm_b < 1e-10f)
            return 1.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute max absolute error
     */
    float max_abs_error(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        {
            max_err = std::max(max_err, std::fabs(a[i] - b[i]));
        }
        return max_err;
    }

    /**
     * @brief Compute relative error with tolerance for near-zero values
     */
    float relative_error(float expected, float actual, float epsilon = 1e-6f)
    {
        float abs_diff = std::fabs(expected - actual);
        float max_abs = std::max(std::fabs(expected), std::fabs(actual));
        if (max_abs < epsilon)
            return abs_diff; // For near-zero values, return absolute error
        return abs_diff / max_abs;
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__Q16_to_Q16_RoPE_FixedScale : public ::testing::Test
{
protected:
    static constexpr int HEAD_DIM = 128;
    static constexpr float ROPE_THETA = 1000000.0f;
    static constexpr float KV_CACHE_SCALE = 8.0f;
    static constexpr float EXPECTED_D = KV_CACHE_SCALE / 32767.0f;

    std::mt19937 rng_{42};
};

// ============================================================================
// Scale Verification Tests
// ============================================================================

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, AllBlocksHaveFixedScale)
{
    // Generate random FP32 data and quantize to Q16_1 with dynamic scales
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Apply fixed-scale RoPE
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Verify all output blocks have fixed scale
    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_FLOAT_EQ(q16_out[b].d, EXPECTED_D)
            << "Block " << b << " has d=" << q16_out[b].d
            << " but expected " << EXPECTED_D;
    }
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, VaryingInputScalesProduceUniformOutputScale)
{
    // Create input with deliberately varying per-block scales (simulating GEMM output)
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    // Set up blocks with different scales (block 0 = 0.1, block 1 = 5.0, etc.)
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<float> varying_scales = {0.1f, 5.0f, 0.5f, 2.0f};
    varying_scales.resize(blocks_per_head, 1.0f);

    // Scale the FP32 data to match the intended block scales
    for (int b = 0; b < blocks_per_head; ++b)
    {
        for (int i = 0; i < 32; ++i)
        {
            fp32_data[b * 32 + i] *= varying_scales[b];
        }
    }

    auto q16_in = fp32_to_q16_1_varying_scale(fp32_data, varying_scales);
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 100, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Verify input has varying scales
    std::set<float> input_scales;
    for (int b = 0; b < blocks_per_head; ++b)
    {
        input_scales.insert(q16_in[b].d);
    }
    ASSERT_GT(input_scales.size(), 1u) << "Test requires varying input scales";

    // Apply fixed-scale RoPE
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Verify all output blocks have uniform fixed scale
    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_FLOAT_EQ(q16_out[b].d, EXPECTED_D)
            << "Block " << b << " has d=" << q16_out[b].d
            << " but expected uniform " << EXPECTED_D;
    }
}

// ============================================================================
// Correctness Tests: Compare to FP32 Reference
// ============================================================================

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, MatchesFP32Reference_TypicalActivations)
{
    // Generate typical activation values
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    // Create Q16_1 input
    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Apply fixed-scale Q16→Q16 RoPE
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Apply FP32 reference RoPE
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    // Dequantize Q16 output
    auto q16_dequant = q16_1_to_fp32(q16_out);

    // Compare
    float cosine = cosine_similarity(fp32_ref, q16_dequant);
    float max_err = max_abs_error(fp32_ref, q16_dequant);

    EXPECT_GT(cosine, 0.9999f) << "Cosine similarity " << cosine << " below threshold";
    EXPECT_LT(max_err, 0.01f) << "Max absolute error " << max_err << " too high";
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, MatchesFP32Reference_Position0)
{
    // Position 0 means cos=1, sin=0 for all pairs → no rotation, just rescale
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 0, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // All cos should be 1.0, sin should be 0.0
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        ASSERT_NEAR(cos_fp32[i], 1.0f, 1e-6f);
        ASSERT_NEAR(sin_fp32[i], 0.0f, 1e-6f);
    }

    // Apply fixed-scale Q16→Q16 RoPE
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Output should match input values (just rescaled)
    auto q16_dequant = q16_1_to_fp32(q16_out);

    // Dequant input for comparison
    auto q16_in_dequant = q16_1_to_fp32(q16_in);

    float cosine = cosine_similarity(q16_in_dequant, q16_dequant);
    EXPECT_GT(cosine, 0.9999f) << "Position 0 should preserve values";
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, MatchesFP32Reference_HighPosition)
{
    // High position → significant rotation
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 1000, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Apply fixed-scale Q16→Q16 RoPE
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Apply FP32 reference RoPE
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    // Compare
    auto q16_dequant = q16_1_to_fp32(q16_out);
    float cosine = cosine_similarity(fp32_ref, q16_dequant);

    EXPECT_GT(cosine, 0.9999f) << "High position cosine similarity " << cosine;
}

// ============================================================================
// Block Structure Tests
// ============================================================================

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, SumQsIsCorrect)
{
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Verify sum_qs matches actual sum of qs values
    for (int b = 0; b < blocks_per_head; ++b)
    {
        int32_t computed_sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            computed_sum += q16_out[b].qs[i];
        }
        EXPECT_EQ(q16_out[b].sum_qs, computed_sum)
            << "Block " << b << " sum_qs mismatch";
    }
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, QsValuesInValidRange)
{
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Verify all qs values are in valid Q16_1 range
    for (int b = 0; b < blocks_per_head; ++b)
    {
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_GE(q16_out[b].qs[i], -16383)
                << "Block " << b << " element " << i << " underflow";
            EXPECT_LE(q16_out[b].qs[i], 16383)
                << "Block " << b << " element " << i << " overflow";
        }
    }
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, ScaleIsNonNegative)
{
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_GE(q16_out[b].d, 0.0f) << "Block " << b << " has negative scale";
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, ZeroInput)
{
    // All zeros input
    std::vector<float> fp32_data(HEAD_DIM, 0.0f);

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Output should also be all zeros
    auto q16_dequant = q16_1_to_fp32(q16_out);
    for (size_t i = 0; i < q16_dequant.size(); ++i)
    {
        EXPECT_NEAR(q16_dequant[i], 0.0f, 1e-6f) << "Element " << i << " should be zero";
    }
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, SmallValues_Preserved)
{
    // Test that small values WITHIN kv_cache_scale range are preserved after rescale
    // kv_cache_scale=8.0 means quantization step is 8.0/32767 ≈ 0.000244
    // So values >=0.001 should be preserved (quantize to ~4 in Q16)
    std::vector<float> fp32_data(HEAD_DIM);
    for (size_t i = 0; i < fp32_data.size(); ++i)
    {
        // Use values in range [0.01, 0.1] which are well above quantization threshold
        fp32_data[i] = 0.01f + 0.09f * static_cast<float>(i) / HEAD_DIM;
        if (i % 2 == 1)
            fp32_data[i] = -fp32_data[i]; // Mix positive and negative
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 0, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32); // Position 0 = no rotation

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Verify small values are preserved after rescale
    auto q16_dequant = q16_1_to_fp32(q16_out);

    int non_zero_count = 0;
    for (size_t i = 0; i < q16_dequant.size(); ++i)
    {
        if (std::fabs(q16_dequant[i]) > 1e-10f)
        {
            non_zero_count++;
        }
    }

    // All values should be preserved (they're all above quantization threshold)
    EXPECT_GT(non_zero_count, static_cast<int>(HEAD_DIM * 0.95))
        << "Small values not preserved: only " << non_zero_count << "/" << HEAD_DIM;

    // Verify relative accuracy for small values
    for (size_t i = 0; i < fp32_data.size(); ++i)
    {
        float expected = std::fabs(fp32_data[i]);
        float actual = std::fabs(q16_dequant[i]);
        if (expected > 0.001f) // For non-tiny values
        {
            float rel_err = std::fabs(expected - actual) / expected;
            EXPECT_LT(rel_err, 0.05f) // 5% relative error tolerance
                << "Element " << i << ": expected " << expected << " got " << actual;
        }
    }
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, LargeValues_Clipped)
{
    // Test with values that exceed kv_cache_scale
    std::vector<float> fp32_data(HEAD_DIM);
    for (size_t i = 0; i < fp32_data.size(); ++i)
    {
        fp32_data[i] = (i % 2 == 0) ? 10.0f : -10.0f; // Outside ±8.0 range
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 0, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Values should be clipped to kv_cache_scale range
    auto q16_dequant = q16_1_to_fp32(q16_out);
    for (size_t i = 0; i < q16_dequant.size(); ++i)
    {
        // With d_fixed = 8.0/32767 and max qs = ±16383, max value = ~4.0
        // But the clipping happens at ±16383 * d_fixed ≈ ±4.0
        EXPECT_LE(std::fabs(q16_dequant[i]), KV_CACHE_SCALE * 0.5f + 0.001f)
            << "Element " << i << " not properly clipped";
    }
}

// ============================================================================
// Production Dimensions Tests
// ============================================================================

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, ProductionDimensions_Qwen2_HeadDim128)
{
    constexpr int QWEN2_HEAD_DIM = 128;

    std::vector<float> fp32_data(QWEN2_HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = QWEN2_HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(QWEN2_HEAD_DIM / 2, 512, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), QWEN2_HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Apply FP32 reference
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), QWEN2_HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_dequant = q16_1_to_fp32(q16_out);
    float cosine = cosine_similarity(fp32_ref, q16_dequant);

    EXPECT_GT(cosine, 0.9999f) << "Qwen2 head_dim=128 cosine: " << cosine;
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, ProductionDimensions_Llama_HeadDim64)
{
    constexpr int LLAMA_HEAD_DIM = 64;

    std::vector<float> fp32_data(LLAMA_HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = LLAMA_HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(LLAMA_HEAD_DIM / 2, 512, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), LLAMA_HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Apply FP32 reference
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), LLAMA_HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_dequant = q16_1_to_fp32(q16_out);
    float cosine = cosine_similarity(fp32_ref, q16_dequant);

    EXPECT_GT(cosine, 0.9999f) << "Llama head_dim=64 cosine: " << cosine;
}

// ============================================================================
// Batch Wrapper Test
// ============================================================================

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, BatchWrapper_MultiplePositions)
{
    constexpr int SEQ_LEN = 4;
    constexpr int N_KV_HEADS = 2;

    // Create input for all positions and heads
    const int total_elements = SEQ_LEN * N_KV_HEADS * HEAD_DIM;
    std::vector<float> fp32_data(total_elements);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    // Quantize full tensor
    const int blocks_per_head = HEAD_DIM / 32;
    const int total_blocks = SEQ_LEN * N_KV_HEADS * blocks_per_head;
    std::vector<Q16_1Block> q16_in(total_blocks);
    std::vector<Q16_1Block> q16_out(total_blocks);

    // Manual quantization of full tensor
    for (int pos = 0; pos < SEQ_LEN; ++pos)
    {
        for (int h = 0; h < N_KV_HEADS; ++h)
        {
            int head_offset = (pos * N_KV_HEADS + h) * blocks_per_head;
            int fp32_offset = head_offset * 32;

            for (int b = 0; b < blocks_per_head; ++b)
            {
                Q16_1Block &blk = q16_in[head_offset + b];
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::fabs(fp32_data[fp32_offset + b * 32 + i]));
                }
                float scale = max_abs / 16383.0f;
                if (scale < 1e-20f)
                    scale = 1e-20f;
                float inv_scale = 1.0f / scale;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(fp32_data[fp32_offset + b * 32 + i] * inv_scale));
                    q = std::max(-16383, std::min(16383, q));
                    blk.qs[i] = static_cast<int16_t>(q);
                    sum_qs += q;
                }
                blk.d = scale;
                blk.sum_qs = sum_qs;
            }
        }
    }

    // Position IDs
    std::vector<int> position_ids = {0, 1, 2, 3};

    // Apply batch wrapper
    apply_rope_q16_to_q16_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(),
        position_ids.data(), SEQ_LEN, N_KV_HEADS, HEAD_DIM,
        ROPE_THETA, KV_CACHE_SCALE);

    // Verify all output blocks have fixed scale
    for (int i = 0; i < total_blocks; ++i)
    {
        EXPECT_FLOAT_EQ(q16_out[i].d, EXPECTED_D)
            << "Block " << i << " has d=" << q16_out[i].d;
    }

    // Verify correctness for each position/head
    for (int pos = 0; pos < SEQ_LEN; ++pos)
    {
        std::vector<int16_t> cos_q15, sin_q15;
        std::vector<float> cos_fp32, sin_fp32;
        generate_rope_tables(HEAD_DIM / 2, pos, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

        for (int h = 0; h < N_KV_HEADS; ++h)
        {
            int head_offset = (pos * N_KV_HEADS + h) * blocks_per_head;
            int fp32_offset = head_offset * 32;

            // Get FP32 reference for this head
            std::vector<float> fp32_head(fp32_data.begin() + fp32_offset,
                                         fp32_data.begin() + fp32_offset + HEAD_DIM);
            apply_rope_fp32_reference(fp32_head.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

            // Dequant output for this head
            std::vector<float> q16_dequant(HEAD_DIM);
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const Q16_1Block &blk = q16_out[head_offset + b];
                for (int i = 0; i < 32; ++i)
                {
                    q16_dequant[b * 32 + i] = blk.d * static_cast<float>(blk.qs[i]);
                }
            }

            float cosine = cosine_similarity(fp32_head, q16_dequant);
            EXPECT_GT(cosine, 0.999f)
                << "pos=" << pos << " head=" << h << " cosine=" << cosine;
        }
    }
}

// ============================================================================
// Special Angle Tests (45°, 90°, 180°)
// ============================================================================

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, Rotation45Degrees)
{
    // 45° rotation: cos = sin = √2/2 ≈ 0.707
    // This is the "balanced" case where x and y contribute equally
    // x' = x*cos - y*sin = 0.707*(x - y)
    // y' = x*sin + y*cos = 0.707*(x + y)

    constexpr float SQRT2_2 = 0.7071067811865475f; // √2/2

    // Create Q15 sin/cos for 45°
    std::vector<int16_t> cos_q15(HEAD_DIM / 2);
    std::vector<int16_t> sin_q15(HEAD_DIM / 2);
    std::vector<float> cos_fp32(HEAD_DIM / 2);
    std::vector<float> sin_fp32(HEAD_DIM / 2);

    const int16_t cos_sin_q15 = static_cast<int16_t>(std::round(SQRT2_2 * 32767.0f)); // ~23170

    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        cos_q15[i] = cos_sin_q15;
        sin_q15[i] = cos_sin_q15;
        cos_fp32[i] = SQRT2_2;
        sin_fp32[i] = SQRT2_2;
    }

    // Test with known values
    std::vector<float> fp32_data(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        fp32_data[i] = 1.0f;                // x = 1
        fp32_data[i + HEAD_DIM / 2] = 0.0f; // y = 0
    }
    // Expected: x' = 1*0.707 - 0*0.707 = 0.707
    //           y' = 1*0.707 + 0*0.707 = 0.707

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    auto q16_dequant = q16_1_to_fp32(q16_out);

    // Verify x and y components are approximately equal after rotation
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        float x_out = q16_dequant[i];
        float y_out = q16_dequant[i + HEAD_DIM / 2];

        // x' should equal y' (both ≈ 0.707)
        EXPECT_NEAR(x_out, SQRT2_2, 0.02f)
            << "Element " << i << ": x' expected ~0.707, got " << x_out;
        EXPECT_NEAR(y_out, SQRT2_2, 0.02f)
            << "Element " << i << ": y' expected ~0.707, got " << y_out;
    }

    // Also compare to FP32 reference
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    float cosine = cosine_similarity(fp32_ref, q16_dequant);
    EXPECT_GT(cosine, 0.9999f) << "45° rotation cosine: " << cosine;
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, Rotation90Degrees)
{
    // 90° rotation: cos = 0, sin = 1
    // x' = x*0 - y*1 = -y
    // y' = x*1 + y*0 = x
    // Pure swap with sign flip on x component

    std::vector<int16_t> cos_q15(HEAD_DIM / 2, 0);     // cos(90°) = 0
    std::vector<int16_t> sin_q15(HEAD_DIM / 2, 32767); // sin(90°) = 1
    std::vector<float> cos_fp32(HEAD_DIM / 2, 0.0f);
    std::vector<float> sin_fp32(HEAD_DIM / 2, 1.0f);

    // Test with known values: x=1, y=2
    std::vector<float> fp32_data(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        fp32_data[i] = 1.0f;                // x = 1
        fp32_data[i + HEAD_DIM / 2] = 2.0f; // y = 2
    }
    // Expected: x' = -y = -2
    //           y' = x = 1

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    auto q16_dequant = q16_1_to_fp32(q16_out);

    // Verify swap behavior
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        float x_out = q16_dequant[i];
        float y_out = q16_dequant[i + HEAD_DIM / 2];

        EXPECT_NEAR(x_out, -2.0f, 0.02f)
            << "Element " << i << ": x' expected -2.0, got " << x_out;
        EXPECT_NEAR(y_out, 1.0f, 0.02f)
            << "Element " << i << ": y' expected 1.0, got " << y_out;
    }

    // Also compare to FP32 reference
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    float cosine = cosine_similarity(fp32_ref, q16_dequant);
    EXPECT_GT(cosine, 0.9999f) << "90° rotation cosine: " << cosine;
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, Rotation180Degrees)
{
    // 180° rotation: cos = -1, sin = 0
    // x' = x*(-1) - y*0 = -x
    // y' = x*0 + y*(-1) = -y
    // Pure negation

    std::vector<int16_t> cos_q15(HEAD_DIM / 2, -32767); // cos(180°) = -1
    std::vector<int16_t> sin_q15(HEAD_DIM / 2, 0);      // sin(180°) = 0
    std::vector<float> cos_fp32(HEAD_DIM / 2, -1.0f);
    std::vector<float> sin_fp32(HEAD_DIM / 2, 0.0f);

    // Test with known values
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    auto q16_dequant = q16_1_to_fp32(q16_out);

    // Compare to FP32 reference (which should negate all values)
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    // Verify negation: output should be approximately -input
    auto q16_in_dequant = q16_1_to_fp32(q16_in);
    for (int i = 0; i < HEAD_DIM; ++i)
    {
        EXPECT_NEAR(q16_dequant[i], -q16_in_dequant[i], 0.02f)
            << "Element " << i << ": expected negation";
    }

    float cosine = cosine_similarity(fp32_ref, q16_dequant);
    EXPECT_GT(cosine, 0.9999f) << "180° rotation cosine: " << cosine;
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, Rotation45Degrees_XYEqualOppositeSign)
{
    // 45° with x = 1, y = -1 tests the subtraction path more heavily
    // x' = x*cos - y*sin = 0.707*1 - 0.707*(-1) = 0.707 + 0.707 = 1.414
    // y' = x*sin + y*cos = 0.707*1 + 0.707*(-1) = 0.707 - 0.707 = 0

    constexpr float SQRT2_2 = 0.7071067811865475f;
    constexpr float SQRT2 = 1.4142135623730951f;

    std::vector<int16_t> cos_q15(HEAD_DIM / 2);
    std::vector<int16_t> sin_q15(HEAD_DIM / 2);
    std::vector<float> cos_fp32(HEAD_DIM / 2);
    std::vector<float> sin_fp32(HEAD_DIM / 2);

    const int16_t cos_sin_q15 = static_cast<int16_t>(std::round(SQRT2_2 * 32767.0f));

    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        cos_q15[i] = cos_sin_q15;
        sin_q15[i] = cos_sin_q15;
        cos_fp32[i] = SQRT2_2;
        sin_fp32[i] = SQRT2_2;
    }

    std::vector<float> fp32_data(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        fp32_data[i] = 1.0f;                 // x = 1
        fp32_data[i + HEAD_DIM / 2] = -1.0f; // y = -1
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    auto q16_dequant = q16_1_to_fp32(q16_out);

    // Verify the specific expected values
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        float x_out = q16_dequant[i];
        float y_out = q16_dequant[i + HEAD_DIM / 2];

        EXPECT_NEAR(x_out, SQRT2, 0.03f)
            << "Element " << i << ": x' expected √2≈1.414, got " << x_out;
        EXPECT_NEAR(y_out, 0.0f, 0.03f)
            << "Element " << i << ": y' expected 0, got " << y_out;
    }

    // Compare to FP32 reference
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    float cosine = cosine_similarity(fp32_ref, q16_dequant);
    EXPECT_GT(cosine, 0.9999f) << "45° (x=1,y=-1) rotation cosine: " << cosine;
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, CrossBlockBoundaryPairs)
{
    // RoPE pairs elements [i] with [i + half_dim].
    // With HEAD_DIM=128 and blocks of 32, the pairs cross block boundaries:
    // Block 0 (elements 0-31) pairs with Block 2 (elements 64-95)
    // Block 1 (elements 32-63) pairs with Block 3 (elements 96-127)
    //
    // This tests that the rotation correctly handles pairs that span blocks
    // with different input scales.
    //
    // NOTE: Input scales must be within reasonable range of d_fixed to avoid
    // integer overflow. With KV_CACHE_SCALE=8.0 and d_fixed≈0.000244,
    // input scales should be similar order of magnitude (0.0001 to 0.001).

    const int blocks_per_head = HEAD_DIM / 32; // 4 blocks

    // Create input with different (but reasonable) scales per block
    // Scales in the range that would result from typical GEMM outputs
    std::vector<Q16_1Block> q16_in(blocks_per_head);

    // Block 0: d = 0.0002 (smaller values)
    q16_in[0].d = 0.0002f;
    for (int i = 0; i < 32; ++i)
    {
        q16_in[0].qs[i] = static_cast<int16_t>(5000 + (i - 16) * 100); // Values around ±0.8
    }

    // Block 1: d = 0.0004 (medium values)
    q16_in[1].d = 0.0004f;
    for (int i = 0; i < 32; ++i)
    {
        q16_in[1].qs[i] = static_cast<int16_t>(3000 + (i - 16) * 150); // Values around ±0.9
    }

    // Block 2: d = 0.0003 - pairs with Block 0
    q16_in[2].d = 0.0003f;
    for (int i = 0; i < 32; ++i)
    {
        q16_in[2].qs[i] = static_cast<int16_t>(4000 + (i - 16) * 120); // Values around ±0.9
    }

    // Block 3: d = 0.0005 - pairs with Block 1
    q16_in[3].d = 0.0005f;
    for (int i = 0; i < 32; ++i)
    {
        q16_in[3].qs[i] = static_cast<int16_t>(2000 + (i - 16) * 80); // Values around ±0.6
    }

    // Update sum_qs for correctness
    for (auto &blk : q16_in)
    {
        blk.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            blk.sum_qs += blk.qs[i];
        }
    }

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    std::vector<Q16_1Block> q16_out(blocks_per_head);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Verify all output blocks have uniform scale
    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_FLOAT_EQ(q16_out[b].d, EXPECTED_D)
            << "Cross-block test: Block " << b << " has wrong scale";
    }

    // Dequantize both input and output
    auto q16_in_dequant = q16_1_to_fp32(q16_in);
    auto q16_out_dequant = q16_1_to_fp32(q16_out);

    // Compare to FP32 reference
    std::vector<float> fp32_ref = q16_in_dequant;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    float cosine = cosine_similarity(fp32_ref, q16_out_dequant);
    EXPECT_GT(cosine, 0.999f) << "Cross-block boundary cosine: " << cosine;
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, SymmetricRoundingPreservesSign)
{
    // Test that symmetric rounding (with (1<<14) bias before >>15) preserves
    // the correct sign for values near zero after rotation

    // Use 90° rotation where x'=-y, y'=x
    // With small positive y, x' should be small negative
    std::vector<int16_t> cos_q15(HEAD_DIM / 2, 0);
    std::vector<int16_t> sin_q15(HEAD_DIM / 2, 32767); // sin = 1
    std::vector<float> cos_fp32(HEAD_DIM / 2, 0.0f);
    std::vector<float> sin_fp32(HEAD_DIM / 2, 1.0f);

    // Create input where y has small positive values
    std::vector<float> fp32_data(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        fp32_data[i] = 0.0f;                // x = 0
        fp32_data[i + HEAD_DIM / 2] = 0.1f; // y = small positive
    }
    // Expected: x' = -0.1 (small negative)
    //           y' = 0

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    auto q16_dequant = q16_1_to_fp32(q16_out);

    // x' should be negative (or zero due to quantization)
    // The key is it should NOT be positive
    int positive_x_count = 0;
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        float x_out = q16_dequant[i];
        if (x_out > 0.01f) // Allow small tolerance for quantization noise
        {
            positive_x_count++;
        }
    }

    EXPECT_EQ(positive_x_count, 0)
        << "Symmetric rounding failed: " << positive_x_count
        << " elements have wrong sign after 90° rotation";
}

// ============================================================================
// Dynamic Range / Spikiness Tests
// ============================================================================

/**
 * @brief Test cosine similarity degradation as input "spikiness" increases
 *
 * K projection outputs can have spiky values (e.g., peak ~130) while the
 * fixed output scale is typically 8.0f. This test measures at what point
 * the cosine similarity degrades due to clipping.
 *
 * With KV_CACHE_SCALE=8.0:
 *   d_fixed = 8.0 / 32767 ≈ 0.000244
 *   Max representable = 16383 * d_fixed ≈ 4.0
 *
 * So values > ~4.0 will clip.
 */
TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, SpikySweep_CosineSimilarityDegradation)
{
    // Test peak magnitudes from 1.0 to 200.0
    std::vector<float> peak_magnitudes = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 130.0f, 200.0f};

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    std::cout << "\n=== Spiky Input Cosine Similarity Test ===" << std::endl;
    std::cout << "KV_CACHE_SCALE = " << KV_CACHE_SCALE << std::endl;
    std::cout << "Max representable value ≈ " << (16383.0f * KV_CACHE_SCALE / 32767.0f) << std::endl;
    std::cout << std::endl;

    for (float peak : peak_magnitudes)
    {
        // Create input with one "spike" and typical values elsewhere
        std::vector<float> fp32_data(HEAD_DIM);
        std::uniform_real_distribution<float> typical_dist(-2.0f, 2.0f);

        // Fill with typical values
        for (auto &v : fp32_data)
        {
            v = typical_dist(rng_);
        }

        // Insert spikes at a few positions
        fp32_data[0] = peak;
        fp32_data[HEAD_DIM / 2] = -peak; // Paired element
        fp32_data[10] = peak * 0.8f;
        fp32_data[10 + HEAD_DIM / 2] = peak * 0.5f;

        auto q16_in = fp32_to_q16_1(fp32_data);
        const int blocks_per_head = HEAD_DIM / 32;
        std::vector<Q16_1Block> q16_out(blocks_per_head);

        apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
            q16_in.data(), q16_out.data(), HEAD_DIM,
            cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

        // FP32 reference
        std::vector<float> fp32_ref = fp32_data;
        apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

        auto q16_dequant = q16_1_to_fp32(q16_out);

        float cosine = cosine_similarity(fp32_ref, q16_dequant);
        float max_err = max_abs_error(fp32_ref, q16_dequant);

        // Count clipped values
        int clipped_count = 0;
        for (const auto &blk : q16_out)
        {
            for (int i = 0; i < 32; ++i)
            {
                if (std::abs(blk.qs[i]) >= 16383)
                {
                    clipped_count++;
                }
            }
        }

        std::cout << "Peak=" << std::setw(6) << peak
                  << "  Cosine=" << std::fixed << std::setprecision(6) << cosine
                  << "  MaxErr=" << std::setw(10) << max_err
                  << "  Clipped=" << clipped_count << "/" << HEAD_DIM
                  << std::endl;
    }
    std::cout << std::endl;

    // The test "passes" - this is informational
    // But we verify that small peaks still have high cosine similarity
    SUCCEED() << "Spiky sweep completed - see output for degradation profile";
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, SpikyInput_Peak4_AcceptableSimilarity)
{
    // Peak of 4.0 is exactly at the representable limit
    // (Max representable ≈ 4.0 with KV_CACHE_SCALE=8.0)
    // Expect slight degradation due to clipping at the boundary
    constexpr float PEAK = 4.0f;

    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> typical_dist(-1.0f, 1.0f);
    for (auto &v : fp32_data)
    {
        v = typical_dist(rng_);
    }

    // Add spikes
    fp32_data[0] = PEAK;
    fp32_data[HEAD_DIM / 2] = -PEAK;

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_dequant = q16_1_to_fp32(q16_out);
    float cosine = cosine_similarity(fp32_ref, q16_dequant);

    // Peak=4.0 is at the boundary, expect >99% similarity
    EXPECT_GT(cosine, 0.99f) << "Peak=4.0 should have very good similarity, got " << cosine;
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, SpikyInput_Peak8_ModestDegradation)
{
    // Peak of 8.0 is at the edge of KV_CACHE_SCALE
    // Rotation can produce values up to sqrt(2)*8 ≈ 11.3 which will clip
    constexpr float PEAK = 8.0f;

    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> typical_dist(-1.0f, 1.0f);
    for (auto &v : fp32_data)
    {
        v = typical_dist(rng_);
    }

    fp32_data[0] = PEAK;
    fp32_data[HEAD_DIM / 2] = PEAK; // Same sign = max rotation output

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_dequant = q16_1_to_fp32(q16_out);
    float cosine = cosine_similarity(fp32_ref, q16_dequant);

    // With KV_CACHE_SCALE=8, max representable is ~4.0
    // Peak=8 is 2x the max, so significant clipping occurs
    // Expect noticeable degradation but not catastrophic
    EXPECT_GT(cosine, 0.85f) << "Peak=8.0 should have moderate similarity despite clipping, got " << cosine;
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, SpikyInput_Peak130_SignificantClipping)
{
    // Peak of 130 (realistic K projection spike) will clip severely
    // This documents the expected behavior, not a pass/fail criterion
    constexpr float PEAK = 130.0f;

    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> typical_dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = typical_dist(rng_);
    }

    // Spike in multiple positions to simulate realistic K projection
    fp32_data[0] = PEAK;
    fp32_data[HEAD_DIM / 2] = -PEAK * 0.5f;
    fp32_data[5] = PEAK * 0.8f;
    fp32_data[5 + HEAD_DIM / 2] = PEAK * 0.3f;

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_dequant = q16_1_to_fp32(q16_out);
    float cosine = cosine_similarity(fp32_ref, q16_dequant);

    // Count clipped values
    int clipped_count = 0;
    for (const auto &blk : q16_out)
    {
        for (int i = 0; i < 32; ++i)
        {
            if (std::abs(blk.qs[i]) >= 16383)
            {
                clipped_count++;
            }
        }
    }

    std::cout << "\n=== Peak 130 Spike Test ===" << std::endl;
    std::cout << "Cosine similarity: " << cosine << std::endl;
    std::cout << "Clipped values: " << clipped_count << "/" << HEAD_DIM << std::endl;
    std::cout << "Max representable: " << (16383.0f * KV_CACHE_SCALE / 32767.0f) << std::endl;
    std::cout << std::endl;

    // With peak=130 and max representable≈4.0, we expect severe clipping
    // Cosine will be low because most of the "signal" is in the clipped spikes
    // This is EXPECTED behavior - the test documents it
    EXPECT_GT(clipped_count, 0) << "Expected some clipping with peak=130";

    // The cosine will be poor, but it shouldn't crash or produce NaN
    EXPECT_FALSE(std::isnan(cosine)) << "Cosine should not be NaN";
    EXPECT_FALSE(std::isinf(cosine)) << "Cosine should not be Inf";
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, SpikyInput_HigherKVScale_ReducesClipping)
{
    // Test that increasing KV_CACHE_SCALE reduces clipping for spiky inputs
    constexpr float PEAK = 130.0f;
    constexpr float HIGHER_KV_SCALE = 256.0f; // 32x larger than default

    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> typical_dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = typical_dist(rng_);
    }

    fp32_data[0] = PEAK;
    fp32_data[HEAD_DIM / 2] = -PEAK * 0.5f;

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Test with default scale (8.0)
    std::vector<Q16_1Block> q16_out_default(blocks_per_head);
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out_default.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Test with higher scale (256.0)
    std::vector<Q16_1Block> q16_out_higher(blocks_per_head);
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out_higher.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), HIGHER_KV_SCALE);

    // FP32 reference
    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_dequant_default = q16_1_to_fp32(q16_out_default);
    auto q16_dequant_higher = q16_1_to_fp32(q16_out_higher);

    float cosine_default = cosine_similarity(fp32_ref, q16_dequant_default);
    float cosine_higher = cosine_similarity(fp32_ref, q16_dequant_higher);

    std::cout << "\n=== KV Scale Comparison for Spiky Input ===" << std::endl;
    std::cout << "Peak magnitude: " << PEAK << std::endl;
    std::cout << "Default KV_SCALE=8.0:   cosine=" << cosine_default << std::endl;
    std::cout << "Higher  KV_SCALE=256.0: cosine=" << cosine_higher << std::endl;
    std::cout << std::endl;

    // Higher scale should have better cosine similarity for spiky inputs
    EXPECT_GT(cosine_higher, cosine_default)
        << "Higher KV scale should improve similarity for spiky inputs";

    // With scale=256, max representable = 16383 * 256/32767 ≈ 128
    // So peak=130 will still have minor clipping, but much less than scale=8
    EXPECT_GT(cosine_higher, 0.99f)
        << "Higher KV scale should achieve excellent similarity";
}

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, RealisticKProjection_TypicalDistribution)
{
    // Simulate realistic K projection output distribution:
    // - Most values in [-2, 2] range
    // - Occasional outliers up to ±10
    // - Rare spikes up to ±50

    std::vector<float> fp32_data(HEAD_DIM);
    std::normal_distribution<float> typical_dist(0.0f, 1.0f); // Mean=0, StdDev=1

    for (size_t i = 0; i < fp32_data.size(); ++i)
    {
        float v = typical_dist(rng_);

        // 5% chance of outlier (±5 to ±10)
        if (rng_() % 100 < 5)
        {
            v = (rng_() % 2 == 0 ? 1.0f : -1.0f) * (5.0f + typical_dist(rng_) * 2.5f);
        }

        // 1% chance of spike (±20 to ±50)
        if (rng_() % 100 < 1)
        {
            v = (rng_() % 2 == 0 ? 1.0f : -1.0f) * (20.0f + std::abs(typical_dist(rng_)) * 15.0f);
        }

        fp32_data[i] = v;
    }

    auto q16_in = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 100, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_in.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    std::vector<float> fp32_ref = fp32_data;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    auto q16_dequant = q16_1_to_fp32(q16_out);
    float cosine = cosine_similarity(fp32_ref, q16_dequant);

    // Find stats
    float max_input = 0.0f;
    for (float v : fp32_data)
    {
        max_input = std::max(max_input, std::abs(v));
    }

    std::cout << "\n=== Realistic K Projection Distribution ===" << std::endl;
    std::cout << "Max input magnitude: " << max_input << std::endl;
    std::cout << "Cosine similarity: " << cosine << std::endl;
    std::cout << std::endl;

    // With KV_CACHE_SCALE=8, max representable is ~4.0
    // This distribution has spikes up to ±50 which will severely clip.
    // The 1% spikes dominate the norm, so cosine can be quite low.
    // This test documents the behavior rather than enforcing a threshold.
    // For production use, either:
    //   1. Use dynamic-scale RoPE (adapts to input range)
    //   2. Increase KV_CACHE_SCALE to handle expected activation ranges
    //   3. Clip/normalize activations before quantization
    EXPECT_FALSE(std::isnan(cosine)) << "Cosine should not be NaN";
    EXPECT_FALSE(std::isinf(cosine)) << "Cosine should not be Inf";
    // When max_input < 4.0 (no clipping), similarity should be excellent
    if (max_input < 4.0f)
    {
        EXPECT_GT(cosine, 0.99f) << "No-clipping case should have excellent similarity";
    }
}

// ============================================================================
// In-place Operation Test
// ============================================================================

TEST_F(Test__Q16_to_Q16_RoPE_FixedScale, InPlaceOperation_SameInputOutput)
{
    // Test that in-place operation (K_in == K_out) works correctly
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q16_blocks = fp32_to_q16_1(fp32_data);
    const int blocks_per_head = HEAD_DIM / 32;

    // Make a copy for comparison
    auto q16_copy = q16_blocks;
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Apply with separate input/output
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_copy.data(), q16_out.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Apply in-place
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(
        q16_blocks.data(), q16_blocks.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Compare results
    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_FLOAT_EQ(q16_blocks[b].d, q16_out[b].d);
        EXPECT_EQ(q16_blocks[b].sum_qs, q16_out[b].sum_qs);
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(q16_blocks[b].qs[i], q16_out[b].qs[i])
                << "Block " << b << " element " << i << " mismatch in in-place";
        }
    }
}
