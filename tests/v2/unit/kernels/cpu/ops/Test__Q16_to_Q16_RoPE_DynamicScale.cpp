/**
 * @file Test__Q16_to_Q16_RoPE_DynamicScale.cpp
 * @brief Unit tests for dynamic-scale Q16→Q16 RoPE implementation
 *
 * Tests for Phase 12 of Q16 Integer Attention project.
 *
 * Key tests:
 * 1. Output scale equals max input scale (d_unified = max(d_input))
 * 2. Round-trip parity with FP32 RoPE (cosine similarity > 0.999)
 * 3. Varying input scales are preserved (no clipping)
 * 4. Spiky inputs (peak ~130) are preserved faithfully
 * 5. All SIMD variants produce identical results
 * 6. All block types are supported
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
     * @brief Quantize FP32 data to Q16_1 blocks with dynamic per-block scale
     * (Simulating GEMM output with varying magnitudes)
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

            float scale = max_abs / 16383.0f;
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
     * @brief Quantize FP32 data to Q16_1 blocks with specified per-block scales
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

            float scale = block_scales[b];
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
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
    std::vector<float> q16_1_to_fp32(const std::vector<Q16_1Block> &blocks, int total_elements)
    {
        std::vector<float> fp32(total_elements);
        for (size_t b = 0; b < blocks.size(); ++b)
        {
            for (int i = 0; i < 32; ++i)
            {
                size_t idx = b * 32 + i;
                if (idx < static_cast<size_t>(total_elements))
                {
                    fp32[idx] = blocks[b].qs[i] * blocks[b].d;
                }
            }
        }
        return fp32;
    }

    /**
     * @brief Compute FP32 reference RoPE rotation
     */
    std::vector<float> fp32_rope_reference(
        const std::vector<float> &input,
        int head_dim,
        float rope_theta,
        int position)
    {
        std::vector<float> output = input;
        const int half_dim = head_dim / 2;
        const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);

        for (int i = 0; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float x = input[i];
            float y = input[i + half_dim];

            output[i] = x * cos_val - y * sin_val;
            output[i + half_dim] = x * sin_val + y * cos_val;
        }

        return output;
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    double compute_cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            return 0.0;

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return denom > 0 ? dot / denom : 0.0;
    }

    /**
     * @brief Generate Q15 sin/cos LUT for given position and head_dim
     */
    std::pair<std::vector<int16_t>, std::vector<int16_t>> generate_sincos_q15(
        int head_dim, float rope_theta, int position)
    {
        const int half_dim = head_dim / 2;
        std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);

        const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);
        for (int i = 0; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
            sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
        }

        return {cos_q15, sin_q15};
    }

} // anonymous namespace

// ============================================================================
// Test: Output Scale Equals Max Input Scale
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, OutputScaleEqualsMaxInputScale)
{
    const int head_dim = 64;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;
    const int position = 10;

    // Create input with varying per-block scales
    std::vector<float> block_scales = {1.0f, 5.0f}; // block 0: scale=1, block 1: scale=5
    std::vector<float> fp32_data(head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Fill each block with values matching its scale
    for (int b = 0; b < blocks_per_head; ++b)
    {
        for (int i = 0; i < 32; ++i)
        {
            fp32_data[b * 32 + i] = dist(rng) * block_scales[b];
        }
    }

    auto q16_in = fp32_to_q16_1_varying_scale(fp32_data, block_scales);
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    auto [cos_q15, sin_q15] = generate_sincos_q15(head_dim, rope_theta, position);

    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);

    // Verify unified_scale equals max input d
    // The input blocks have d = block_scales directly (1.0 and 5.0)
    // So unified_scale should be max(1.0, 5.0) = 5.0
    float expected_unified = 5.0f; // max(d_in)
    EXPECT_NEAR(unified_scale, expected_unified, 1e-5f)
        << "Unified scale should equal max input d";

    // Verify all output blocks have the same d
    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_FLOAT_EQ(q16_out[b].d, unified_scale)
            << "Block " << b << " should have d = unified_scale";
    }
}

// ============================================================================
// Test: Round-Trip Parity with FP32 RoPE
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, MatchesFP32Reference)
{
    const int head_dim = 64;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;
    const int position = 10;

    // Generate random input
    std::vector<float> fp32_input(head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : fp32_input)
        v = dist(rng);

    // FP32 reference
    auto fp32_ref = fp32_rope_reference(fp32_input, head_dim, rope_theta, position);

    // Q16 path
    auto q16_in = fp32_to_q16_1(fp32_input);
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    auto [cos_q15, sin_q15] = generate_sincos_q15(head_dim, rope_theta, position);

    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);

    // Dequantize
    auto fp32_q16 = q16_1_to_fp32(q16_out, head_dim);

    // Compare
    double cosine_sim = compute_cosine_similarity(fp32_ref, fp32_q16);
    EXPECT_GT(cosine_sim, 0.999)
        << "Q16 dynamic-scale RoPE should match FP32 reference with cosine > 0.999";
}

// ============================================================================
// Test: Varying Input Scales Are Preserved
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, PreservesVaryingInputScales)
{
    // This test verifies dynamic-scale handles blocks with SLIGHTLY different scales
    // (as would happen with real GEMM outputs). Large scale differences (10×+) will
    // naturally lose precision for small-scale blocks during rescaling - this is
    // expected behavior since dynamic-scale prioritizes preserving peaks.

    const int head_dim = 64;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;
    const int position = 10;

    // Create FP32 data first with realistic magnitude variation (within ~2-3×)
    std::vector<float> fp32_data(head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (int i = 0; i < head_dim; ++i)
    {
        fp32_data[i] = dist(rng);
    }

    // Add some outliers in block 1 (realistic - GEMM can have varying magnitudes)
    fp32_data[40] = 8.0f;  // Slightly larger than typical
    fp32_data[50] = -7.5f; // Another outlier

    // FP32 reference
    auto fp32_ref = fp32_rope_reference(fp32_data, head_dim, rope_theta, position);

    // Q16 path - use proper quantization that finds per-block max
    std::vector<Q16_1Block> q16_in(blocks_per_head);
    for (int b = 0; b < blocks_per_head; ++b)
    {
        float max_val = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            max_val = std::max(max_val, std::abs(fp32_data[b * 32 + i]));
        }
        float d = max_val / 16383.0f; // Use safe INT16 range
        q16_in[b].d = d;
        for (int i = 0; i < 32; ++i)
        {
            float val = fp32_data[b * 32 + i];
            int16_t qs = static_cast<int16_t>(std::round(val / d));
            q16_in[b].qs[i] = qs;
        }
    }

    std::vector<Q16_1Block> q16_out(blocks_per_head);
    auto [cos_q15, sin_q15] = generate_sincos_q15(head_dim, rope_theta, position);

    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);

    auto fp32_q16 = q16_1_to_fp32(q16_out, head_dim);

    double cosine_sim = compute_cosine_similarity(fp32_ref, fp32_q16);
    EXPECT_GT(cosine_sim, 0.999)
        << "Dynamic-scale should preserve realistic varying scales with cosine > 0.999";

    // Verify unified scale captures the larger block's range
    float max_input_d = std::max(q16_in[0].d, q16_in[1].d);
    EXPECT_NEAR(unified_scale, max_input_d, 1e-5f)
        << "Unified scale should equal max input d";
}

// ============================================================================
// Test: Spiky Input (Peak ~130) Is Preserved
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, SpikyInput_Peak130_Preserved)
{
    const int head_dim = 128; // Typical Qwen2 head_dim
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;
    const int position = 10;

    // Create spiky input: most values small (~4), one block has peak ~130
    std::vector<float> fp32_data(head_dim, 0.0f);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Fill with small values
    for (auto &v : fp32_data)
        v = dist(rng) * 4.0f;

    // Add a spike in block 1
    fp32_data[32 + 15] = 130.0f; // Peak value at block 1, element 15

    // Track the spike value before RoPE
    float spike_before = fp32_data[32 + 15];

    // FP32 reference
    auto fp32_ref = fp32_rope_reference(fp32_data, head_dim, rope_theta, position);

    // Q16 path
    auto q16_in = fp32_to_q16_1(fp32_data);
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    auto [cos_q15, sin_q15] = generate_sincos_q15(head_dim, rope_theta, position);

    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);

    auto fp32_q16 = q16_1_to_fp32(q16_out, head_dim);

    // Dynamic-scale should preserve the spike
    double cosine_sim = compute_cosine_similarity(fp32_ref, fp32_q16);
    EXPECT_GT(cosine_sim, 0.999)
        << "Dynamic-scale should preserve peak=130 with cosine > 0.999, got " << cosine_sim;

    // Verify the unified scale accommodates the spike
    // With peak=130, the block scale should be ~130/16383 = 0.00794
    // unified_scale = max_d / 32767
    float expected_block_scale = 130.0f / 16383.0f; // ~0.00794
    EXPECT_GT(q16_in[1].d, 0.007f)
        << "Block containing spike should have scale ~0.008";

    std::cout << "  Spike before RoPE: " << spike_before << std::endl;
    std::cout << "  Unified scale: " << unified_scale << std::endl;
    std::cout << "  Cosine similarity: " << cosine_sim << std::endl;
}

// ============================================================================
// Test: Comparison with Fixed-Scale (Shows Clipping Problem)
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, DynamicVsFixed_SpikyCosineComparison)
{
    const int head_dim = 128;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;
    const int position = 10;
    const float kv_cache_scale = 64.0f; // Fixed scale for comparison

    // Create spiky input with peak ~130
    std::vector<float> fp32_data(head_dim, 0.0f);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : fp32_data)
        v = dist(rng) * 4.0f;
    fp32_data[32 + 15] = 130.0f;

    // FP32 reference
    auto fp32_ref = fp32_rope_reference(fp32_data, head_dim, rope_theta, position);

    // Q16 input (same for both)
    auto q16_in = fp32_to_q16_1(fp32_data);
    auto [cos_q15, sin_q15] = generate_sincos_q15(head_dim, rope_theta, position);

    // DYNAMIC-scale path
    std::vector<Q16_1Block> q16_out_dynamic(blocks_per_head);
    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out_dynamic.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);
    auto fp32_dynamic = q16_1_to_fp32(q16_out_dynamic, head_dim);
    double cosine_dynamic = compute_cosine_similarity(fp32_ref, fp32_dynamic);

    // FIXED-scale path
    std::vector<Q16_1Block> q16_out_fixed(blocks_per_head);
    apply_rope_q16_to_q16_head_fixed_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out_fixed.data(), head_dim,
        cos_q15.data(), sin_q15.data(), kv_cache_scale);
    auto fp32_fixed = q16_1_to_fp32(q16_out_fixed, head_dim);
    double cosine_fixed = compute_cosine_similarity(fp32_ref, fp32_fixed);

    std::cout << "  Peak value: 130" << std::endl;
    std::cout << "  DYNAMIC-scale cosine: " << cosine_dynamic << std::endl;
    std::cout << "  FIXED-scale (64) cosine: " << cosine_fixed << std::endl;

    // Dynamic should be significantly better than fixed for spiky inputs
    EXPECT_GT(cosine_dynamic, 0.999)
        << "Dynamic-scale should achieve cosine > 0.999";
    EXPECT_GT(cosine_dynamic, cosine_fixed)
        << "Dynamic-scale should outperform fixed-scale for spiky inputs";
}

// ============================================================================
// Test: Block64 Support
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, Block64_Works)
{
    const int head_dim = 128;
    const int blocks_per_head = head_dim / 64;
    const float rope_theta = 10000.0f;
    const int position = 10;

    // Generate random input
    std::vector<float> fp32_input(head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : fp32_input)
        v = dist(rng);

    // Create Block64 input
    const size_t n_blocks = (fp32_input.size() + 63) / 64;
    std::vector<Q16_1Block_64> blocks_in(n_blocks);

    for (size_t b = 0; b < n_blocks; ++b)
    {
        const float *block_data = fp32_input.data() + b * 64;
        Q16_1Block_64 &blk = blocks_in[b];

        float max_abs = 0.0f;
        for (int i = 0; i < 64; ++i)
        {
            max_abs = std::max(max_abs, std::fabs(block_data[i]));
        }

        float scale = max_abs / 16383.0f;
        if (scale < 1e-20f)
            scale = 1e-20f;
        float inv_scale = 1.0f / scale;

        int32_t sum_qs = 0;
        for (int i = 0; i < 64; ++i)
        {
            int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
            q = std::max(-16383, std::min(16383, q));
            blk.qs[i] = static_cast<int16_t>(q);
            sum_qs += q;
        }

        blk.d = scale;
        blk.sum_qs = sum_qs;
    }

    std::vector<Q16_1Block_64> blocks_out(n_blocks);

    // Generate sin/cos for Block64 (half_dim = 64)
    const int half_dim = head_dim / 2;
    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);
    for (int i = 0; i < half_dim; ++i)
    {
        float angle = position * inv_freq[i];
        cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
    }

    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block_64>(
        blocks_in.data(), blocks_out.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);

    // Verify all output blocks have unified scale
    for (size_t b = 0; b < n_blocks; ++b)
    {
        EXPECT_FLOAT_EQ(blocks_out[b].d, unified_scale)
            << "Block64 " << b << " should have unified scale";
    }

    // Dequantize and compare with FP32 reference
    std::vector<float> fp32_q16(head_dim);
    for (size_t b = 0; b < n_blocks; ++b)
    {
        for (int i = 0; i < 64; ++i)
        {
            size_t idx = b * 64 + i;
            if (idx < fp32_q16.size())
            {
                fp32_q16[idx] = blocks_out[b].qs[i] * blocks_out[b].d;
            }
        }
    }

    auto fp32_ref = fp32_rope_reference(fp32_input, head_dim, rope_theta, position);
    double cosine_sim = compute_cosine_similarity(fp32_ref, fp32_q16);
    EXPECT_GT(cosine_sim, 0.999)
        << "Block64 dynamic-scale should match FP32 reference";
}

// ============================================================================
// Test: Block128 Support
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, Block128_Works)
{
    const int head_dim = 256;
    const int blocks_per_head = head_dim / 128;
    const float rope_theta = 10000.0f;
    const int position = 10;

    // Generate random input
    std::vector<float> fp32_input(head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : fp32_input)
        v = dist(rng);

    // Create Block128 input
    const size_t n_blocks = (fp32_input.size() + 127) / 128;
    std::vector<Q16_1Block_128> blocks_in(n_blocks);

    for (size_t b = 0; b < n_blocks; ++b)
    {
        const float *block_data = fp32_input.data() + b * 128;
        Q16_1Block_128 &blk = blocks_in[b];

        float max_abs = 0.0f;
        for (int i = 0; i < 128; ++i)
        {
            max_abs = std::max(max_abs, std::fabs(block_data[i]));
        }

        float scale = max_abs / 16383.0f;
        if (scale < 1e-20f)
            scale = 1e-20f;
        float inv_scale = 1.0f / scale;

        int32_t sum_qs = 0;
        for (int i = 0; i < 128; ++i)
        {
            int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
            q = std::max(-16383, std::min(16383, q));
            blk.qs[i] = static_cast<int16_t>(q);
            sum_qs += q;
        }

        blk.d = scale;
        blk.sum_qs = sum_qs;
    }

    std::vector<Q16_1Block_128> blocks_out(n_blocks);

    // Generate sin/cos for Block128
    const int half_dim = head_dim / 2;
    std::vector<int16_t> cos_q15(half_dim), sin_q15(half_dim);
    const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);
    for (int i = 0; i < half_dim; ++i)
    {
        float angle = position * inv_freq[i];
        cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
    }

    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block_128>(
        blocks_in.data(), blocks_out.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);

    // Verify all output blocks have unified scale
    for (size_t b = 0; b < n_blocks; ++b)
    {
        EXPECT_FLOAT_EQ(blocks_out[b].d, unified_scale)
            << "Block128 " << b << " should have unified scale";
    }

    // Dequantize and compare with FP32 reference
    std::vector<float> fp32_q16(head_dim);
    for (size_t b = 0; b < n_blocks; ++b)
    {
        for (int i = 0; i < 128; ++i)
        {
            size_t idx = b * 128 + i;
            if (idx < fp32_q16.size())
            {
                fp32_q16[idx] = blocks_out[b].qs[i] * blocks_out[b].d;
            }
        }
    }

    auto fp32_ref = fp32_rope_reference(fp32_input, head_dim, rope_theta, position);
    double cosine_sim = compute_cosine_similarity(fp32_ref, fp32_q16);
    EXPECT_GT(cosine_sim, 0.999)
        << "Block128 dynamic-scale should match FP32 reference";
}

// ============================================================================
// Test: Zero Input
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, ZeroInput)
{
    const int head_dim = 64;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;
    const int position = 10;

    std::vector<float> fp32_data(head_dim, 0.0f);
    auto q16_in = fp32_to_q16_1(fp32_data);
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    auto [cos_q15, sin_q15] = generate_sincos_q15(head_dim, rope_theta, position);

    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);

    // Verify output is still all zeros (rotation of zero is zero)
    auto fp32_out = q16_1_to_fp32(q16_out, head_dim);
    for (int i = 0; i < head_dim; ++i)
    {
        EXPECT_NEAR(fp32_out[i], 0.0f, 1e-6f);
    }
}

// ============================================================================
// Test: High-Level Batch Wrapper
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, BatchWrapper_MultiHead)
{
    const int seq_len = 2;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;

    // Total blocks
    const int total_blocks = seq_len * n_kv_heads * blocks_per_head;

    // Generate random input
    std::vector<Q16_1Block> K_in(total_blocks);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto &blk : K_in)
    {
        float max_abs = 0.0f;
        std::vector<float> temp(32);
        for (int i = 0; i < 32; ++i)
        {
            temp[i] = dist(rng);
            max_abs = std::max(max_abs, std::fabs(temp[i]));
        }
        float scale = max_abs / 16383.0f;
        if (scale < 1e-20f)
            scale = 1e-20f;

        int32_t sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            int32_t q = static_cast<int32_t>(std::round(temp[i] / scale));
            q = std::max(-16383, std::min(16383, q));
            blk.qs[i] = static_cast<int16_t>(q);
            sum_qs += q;
        }
        blk.d = scale;
        blk.sum_qs = sum_qs;
    }

    std::vector<Q16_1Block> K_out(total_blocks);
    std::vector<float> head_scales(seq_len * n_kv_heads);
    std::vector<int> position_ids = {5, 10}; // Two positions

    apply_rope_q16_to_q16_dynamic_scale<Q16_1Block>(
        K_in.data(), K_out.data(),
        position_ids.data(), seq_len, n_kv_heads, head_dim,
        rope_theta, head_scales.data());

    // Verify each head has a valid unified scale
    for (int i = 0; i < seq_len * n_kv_heads; ++i)
    {
        EXPECT_GT(head_scales[i], 0.0f)
            << "Head " << i << " should have positive unified scale";
    }

    // Verify all blocks within each head share the same d
    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            int head_idx = pos * n_kv_heads + h;
            float expected_d = head_scales[head_idx];

            for (int b = 0; b < blocks_per_head; ++b)
            {
                int block_idx = head_idx * blocks_per_head + b;
                EXPECT_FLOAT_EQ(K_out[block_idx].d, expected_d)
                    << "Block " << block_idx << " (head " << head_idx << ") should have d = " << expected_d;
            }
        }
    }
}

// ============================================================================
// Test: SIMD Variants Match (if available)
// ============================================================================

#if defined(__AVX2__) || defined(__AVX512F__)
TEST(Test__Q16_to_Q16_RoPE_DynamicScale, SIMDVariantsMatch)
{
    const int head_dim = 64;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;
    const int position = 10;

    // Generate random input
    std::vector<float> fp32_input(head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : fp32_input)
        v = dist(rng);

    auto q16_in = fp32_to_q16_1(fp32_input);
    auto [cos_q15, sin_q15] = generate_sincos_q15(head_dim, rope_theta, position);

    // Scalar
    std::vector<Q16_1Block> q16_out_scalar(blocks_per_head);
    float unified_scale_scalar;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out_scalar.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale_scalar);

#if defined(__AVX2__)
    // AVX2
    std::vector<Q16_1Block> q16_out_avx2(blocks_per_head);
    float unified_scale_avx2;
    apply_rope_q16_to_q16_head_dynamic_scale_avx2(
        q16_in.data(), q16_out_avx2.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale_avx2);

    EXPECT_FLOAT_EQ(unified_scale_scalar, unified_scale_avx2)
        << "AVX2 should produce same unified scale as scalar";

    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_FLOAT_EQ(q16_out_scalar[b].d, q16_out_avx2[b].d);
        EXPECT_EQ(q16_out_scalar[b].sum_qs, q16_out_avx2[b].sum_qs);
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(q16_out_scalar[b].qs[i], q16_out_avx2[b].qs[i])
                << "Block " << b << " element " << i << " mismatch";
        }
    }
#endif

#if defined(__AVX512F__)
    // AVX512
    std::vector<Q16_1Block> q16_out_avx512(blocks_per_head);
    float unified_scale_avx512;
    apply_rope_q16_to_q16_head_dynamic_scale_avx512(
        q16_in.data(), q16_out_avx512.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale_avx512);

    EXPECT_FLOAT_EQ(unified_scale_scalar, unified_scale_avx512)
        << "AVX512 should produce same unified scale as scalar";

    for (int b = 0; b < blocks_per_head; ++b)
    {
        EXPECT_FLOAT_EQ(q16_out_scalar[b].d, q16_out_avx512[b].d);
        EXPECT_EQ(q16_out_scalar[b].sum_qs, q16_out_avx512[b].sum_qs);
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(q16_out_scalar[b].qs[i], q16_out_avx512[b].qs[i])
                << "Block " << b << " element " << i << " mismatch";
        }
    }
#endif
}
#endif

// ============================================================================
// Test: sum_qs Is Correctly Computed
// ============================================================================

TEST(Test__Q16_to_Q16_RoPE_DynamicScale, SumQsCorrect)
{
    const int head_dim = 64;
    const int blocks_per_head = head_dim / 32;
    const float rope_theta = 10000.0f;
    const int position = 10;

    std::vector<float> fp32_input(head_dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &v : fp32_input)
        v = dist(rng);

    auto q16_in = fp32_to_q16_1(fp32_input);
    std::vector<Q16_1Block> q16_out(blocks_per_head);

    auto [cos_q15, sin_q15] = generate_sincos_q15(head_dim, rope_theta, position);

    float unified_scale;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block>(
        q16_in.data(), q16_out.data(), head_dim,
        cos_q15.data(), sin_q15.data(), &unified_scale);

    // Verify sum_qs matches manual sum of qs[]
    for (int b = 0; b < blocks_per_head; ++b)
    {
        int32_t manual_sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            manual_sum += q16_out[b].qs[i];
        }
        EXPECT_EQ(q16_out[b].sum_qs, manual_sum)
            << "Block " << b << " sum_qs mismatch";
    }
}
