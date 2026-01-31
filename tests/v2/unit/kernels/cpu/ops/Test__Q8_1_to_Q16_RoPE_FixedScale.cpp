/**
 * @file Test__Q8_1_to_Q16_RoPE_FixedScale.cpp
 * @brief Unit tests for fixed-scale Q8_1→Q16 RoPE implementation
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Scale verification: All output blocks have d == kv_cache_scale / 32767
 * 2. Round-trip parity: Fixed-scale Q8→Q16 RoPE matches FP32 RoPE within tolerance
 * 3. No clipping: Typical activation ranges don't clip
 * 4. Integer arithmetic correctness: Verify pure integer rotation matches FP32
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

using namespace llaminar2::primitives;
using namespace llaminar2;

namespace
{
    // ============================================================================
    // FP16 Conversion Helpers
    // ============================================================================

    inline uint16_t fp32_to_fp16(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (bits >> 13) & 0x3FF;
        if (exp <= 0)
            return static_cast<uint16_t>(sign);
        if (exp >= 31)
            return static_cast<uint16_t>(sign | 0x7C00);
        return static_cast<uint16_t>(sign | (exp << 10) | frac);
    }

    inline float fp16_to_fp32(uint16_t value)
    {
        uint32_t sign = (value & 0x8000) << 16;
        uint32_t exp = (value >> 10) & 0x1F;
        uint32_t frac = value & 0x3FF;

        if (exp == 0)
        {
            if (frac == 0)
            {
                uint32_t result = sign;
                float f;
                std::memcpy(&f, &result, sizeof(float));
                return f;
            }
            exp = 1;
            while (!(frac & 0x400))
            {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3FF;
        }
        else if (exp == 31)
        {
            uint32_t result = sign | 0x7F800000 | (frac << 13);
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }

        uint32_t result = sign | ((exp + 127 - 15) << 23) | (frac << 13);
        float f;
        std::memcpy(&f, &result, sizeof(float));
        return f;
    }

    // ============================================================================
    // Test Utilities
    // ============================================================================

    /**
     * @brief Quantize FP32 data to Q8_1 blocks
     */
    std::vector<Q8_1Block> fp32_to_q8_1(const std::vector<float> &fp32)
    {
        const size_t n_blocks = (fp32.size() + 31) / 32;
        std::vector<Q8_1Block> blocks(n_blocks);

        std::vector<float> padded = fp32;
        padded.resize(n_blocks * 32, 0.0f);

        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = padded.data() + b * 32;
            Q8_1Block &blk = blocks[b];

            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }

            float scale = max_abs / 127.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-127, std::min(127, q));
                blk.qs[i] = static_cast<int8_t>(q);
                sum_qs += q;
            }

            blk.d = fp32_to_fp16(scale);
            blk.sum_qs = static_cast<int16_t>(sum_qs);
        }

        return blocks;
    }

    /**
     * @brief Dequantize Q8_1 blocks to FP32
     */
    std::vector<float> q8_1_to_fp32(const std::vector<Q8_1Block> &blocks)
    {
        std::vector<float> fp32(blocks.size() * 32);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            const Q8_1Block &blk = blocks[b];
            float *block_data = fp32.data() + b * 32;
            float scale = fp16_to_fp32(blk.d);

            for (int i = 0; i < 32; ++i)
            {
                block_data[i] = scale * static_cast<float>(blk.qs[i]);
            }
        }

        return fp32;
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

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__Q8_1_to_Q16_RoPE_FixedScale : public ::testing::Test
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

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, AllBlocksHaveFixedScale)
{
    // Generate random Q8_1 input data with typical activation range
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q8_blocks = fp32_to_q8_1(fp32_data);
    const int q8_blocks_per_head = HEAD_DIM / 32;
    const int q16_blocks_per_head = HEAD_DIM / 32;
    std::vector<Q16_1Block> q16_blocks(q16_blocks_per_head);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Apply fixed-scale RoPE
    apply_rope_q8_1_to_q16_head_fixed_scale<Q16_1Block>(
        q8_blocks.data(), q16_blocks.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Verify ALL blocks have exactly the fixed scale
    for (int b = 0; b < q16_blocks_per_head; ++b)
    {
        EXPECT_FLOAT_EQ(q16_blocks[b].d, EXPECTED_D)
            << "Block " << b << " has wrong scale: " << q16_blocks[b].d
            << " vs expected " << EXPECTED_D;
    }
}

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, ScaleConsistentAcrossPositions)
{
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q8_blocks = fp32_to_q8_1(fp32_data);
    const int q16_blocks_per_head = HEAD_DIM / 32;

    // Test multiple positions
    for (int pos : {0, 1, 10, 100, 1000, 4096})
    {
        std::vector<Q16_1Block> q16_blocks(q16_blocks_per_head);
        std::vector<int16_t> cos_q15, sin_q15;
        std::vector<float> cos_fp32, sin_fp32;
        generate_rope_tables(HEAD_DIM / 2, pos, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

        apply_rope_q8_1_to_q16_head_fixed_scale<Q16_1Block>(
            q8_blocks.data(), q16_blocks.data(), HEAD_DIM,
            cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

        for (int b = 0; b < q16_blocks_per_head; ++b)
        {
            EXPECT_FLOAT_EQ(q16_blocks[b].d, EXPECTED_D)
                << "Position " << pos << ", block " << b << " has wrong scale";
        }
    }
}

// ============================================================================
// Round-Trip Parity Tests
// ============================================================================

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, MatchesFP32Reference)
{
    // Test that fixed-scale Q8→Q16 RoPE matches FP32 RoPE within tolerance
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    // Create Q8 input
    auto q8_blocks = fp32_to_q8_1(fp32_data);

    // Dequantize Q8 to get input for FP32 reference
    auto q8_dequant = q8_1_to_fp32(q8_blocks);

    // Generate tables
    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // Apply FP32 RoPE to dequantized data
    apply_rope_fp32_reference(q8_dequant.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    // Apply fixed-scale Q8→Q16 RoPE
    std::vector<Q16_1Block> q16_blocks(HEAD_DIM / 32);
    apply_rope_q8_1_to_q16_head_fixed_scale<Q16_1Block>(
        q8_blocks.data(), q16_blocks.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Dequantize Q16 output
    auto q16_dequant = q16_1_to_fp32(q16_blocks);

    // Compare
    float cos_sim = cosine_similarity(q8_dequant, q16_dequant);
    float max_err = max_abs_error(q8_dequant, q16_dequant);

    std::cout << "[FixedScale] Cosine similarity: " << cos_sim << std::endl;
    std::cout << "[FixedScale] Max abs error: " << max_err << std::endl;

    // Expect high similarity - fixed scale should still produce accurate results
    // The tolerance is slightly higher than data-adaptive because we're using a fixed scale
    EXPECT_GT(cos_sim, 0.995f) << "Fixed-scale RoPE should match FP32 reference closely";
    EXPECT_LT(max_err, 0.1f) << "Max error should be small for typical activations";
}

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, ParityAcrossPositions)
{
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q8_blocks = fp32_to_q8_1(fp32_data);

    for (int pos : {0, 1, 10, 100, 500})
    {
        auto q8_dequant = q8_1_to_fp32(q8_blocks);

        std::vector<int16_t> cos_q15, sin_q15;
        std::vector<float> cos_fp32, sin_fp32;
        generate_rope_tables(HEAD_DIM / 2, pos, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

        // FP32 reference
        apply_rope_fp32_reference(q8_dequant.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

        // Fixed-scale Q8→Q16
        std::vector<Q16_1Block> q16_blocks(HEAD_DIM / 32);
        apply_rope_q8_1_to_q16_head_fixed_scale<Q16_1Block>(
            q8_blocks.data(), q16_blocks.data(), HEAD_DIM,
            cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

        auto q16_dequant = q16_1_to_fp32(q16_blocks);

        float cos_sim = cosine_similarity(q8_dequant, q16_dequant);
        EXPECT_GT(cos_sim, 0.995f) << "Position " << pos << " should match FP32";
    }
}

// ============================================================================
// No Clipping Tests
// ============================================================================

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, NoClippingForTypicalActivations)
{
    // Test that typical activation ranges don't cause clipping
    // Qwen2 activations typically stay in [-2.0, 2.0] range

    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q8_blocks = fp32_to_q8_1(fp32_data);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    std::vector<Q16_1Block> q16_blocks(HEAD_DIM / 32);
    apply_rope_q8_1_to_q16_head_fixed_scale<Q16_1Block>(
        q8_blocks.data(), q16_blocks.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Check that no values are at the clipping boundary (±16383)
    int clipped_count = 0;
    for (const auto &blk : q16_blocks)
    {
        for (int i = 0; i < 32; ++i)
        {
            if (std::abs(blk.qs[i]) >= 16383)
            {
                clipped_count++;
            }
        }
    }

    EXPECT_EQ(clipped_count, 0)
        << "Typical activations (-2 to +2) should not clip with KV_CACHE_SCALE=8.0";
}

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, ClippingForExtremeActivations)
{
    // Test behavior with extreme activations (should clip but not crash)
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f); // Way beyond typical range
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q8_blocks = fp32_to_q8_1(fp32_data);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    std::vector<Q16_1Block> q16_blocks(HEAD_DIM / 32);
    // Should not crash even with extreme values
    EXPECT_NO_THROW(apply_rope_q8_1_to_q16_head_fixed_scale<Q16_1Block>(
        q8_blocks.data(), q16_blocks.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE));

    // Verify all values are within int16 range
    for (const auto &blk : q16_blocks)
    {
        EXPECT_FLOAT_EQ(blk.d, EXPECTED_D);
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_LE(std::abs(blk.qs[i]), 16383);
        }
    }
}

// ============================================================================
// Sum_qs Verification
// ============================================================================

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, SumQsCorrect)
{
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q8_blocks = fp32_to_q8_1(fp32_data);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    std::vector<Q16_1Block> q16_blocks(HEAD_DIM / 32);
    apply_rope_q8_1_to_q16_head_fixed_scale<Q16_1Block>(
        q8_blocks.data(), q16_blocks.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);

    // Verify sum_qs matches actual sum
    for (size_t b = 0; b < q16_blocks.size(); ++b)
    {
        int32_t actual_sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            actual_sum += q16_blocks[b].qs[i];
        }
        EXPECT_EQ(q16_blocks[b].sum_qs, actual_sum)
            << "Block " << b << " sum_qs mismatch";
    }
}

// ============================================================================
// High-Level Wrapper Tests
// ============================================================================

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, HighLevelWrapperWorks)
{
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int q8_blocks_per_head = HEAD_DIM / 32;
    const int q16_blocks_per_head = HEAD_DIM / 32;

    // Generate random input for Q and K
    std::vector<float> q_fp32(seq_len * n_heads * HEAD_DIM);
    std::vector<float> k_fp32(seq_len * n_kv_heads * HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : q_fp32)
        v = dist(rng_);
    for (auto &v : k_fp32)
        v = dist(rng_);

    // Quantize to Q8
    auto q_q8 = fp32_to_q8_1(q_fp32);
    auto k_q8 = fp32_to_q8_1(k_fp32);

    // Output buffers
    std::vector<Q16_1Block> q_out(seq_len * n_heads * q16_blocks_per_head);
    std::vector<Q16_1Block> k_out(seq_len * n_kv_heads * q16_blocks_per_head);

    // Position IDs
    std::vector<int> position_ids(seq_len);
    std::iota(position_ids.begin(), position_ids.end(), 0);

    // Apply fixed-scale RoPE
    apply_rope_q8_1_to_q16_fixed_scale<Q16_1Block>(
        q_q8.data(), k_q8.data(),
        q_out.data(), k_out.data(),
        position_ids.data(),
        seq_len, n_heads, n_kv_heads, HEAD_DIM,
        ROPE_THETA, KV_CACHE_SCALE);

    // Verify all output blocks have fixed scale
    for (const auto &blk : q_out)
    {
        EXPECT_FLOAT_EQ(blk.d, EXPECTED_D);
    }
    for (const auto &blk : k_out)
    {
        EXPECT_FLOAT_EQ(blk.d, EXPECTED_D);
    }
}

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, DispatchWorks)
{
    const int seq_len = 2;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int q16_blocks_per_head = HEAD_DIM / 32;

    std::vector<float> q_fp32(seq_len * n_heads * HEAD_DIM);
    std::vector<float> k_fp32(seq_len * n_kv_heads * HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : q_fp32)
        v = dist(rng_);
    for (auto &v : k_fp32)
        v = dist(rng_);

    auto q_q8 = fp32_to_q8_1(q_fp32);
    auto k_q8 = fp32_to_q8_1(k_fp32);

    std::vector<Q16_1Block> q_out(seq_len * n_heads * q16_blocks_per_head);
    std::vector<Q16_1Block> k_out(seq_len * n_kv_heads * q16_blocks_per_head);

    std::vector<int> position_ids(seq_len);
    std::iota(position_ids.begin(), position_ids.end(), 0);

    // Use dispatch function
    apply_rope_q8_1_to_q16_fixed_scale_dispatch(
        q_q8.data(), k_q8.data(),
        q_out.data(), k_out.data(),
        Q16BlockSize::BLOCK_32,
        position_ids.data(),
        seq_len, n_heads, n_kv_heads, HEAD_DIM,
        ROPE_THETA, KV_CACHE_SCALE);

    // Verify scales
    for (const auto &blk : q_out)
    {
        EXPECT_FLOAT_EQ(blk.d, EXPECTED_D);
    }
}

// ============================================================================
// Comparison with Data-Adaptive Scale
// ============================================================================

TEST_F(Test__Q8_1_to_Q16_RoPE_FixedScale, CompareWithDataAdaptive)
{
    // Compare fixed-scale vs data-adaptive to understand the tradeoff
    std::vector<float> fp32_data(HEAD_DIM);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &v : fp32_data)
    {
        v = dist(rng_);
    }

    auto q8_blocks = fp32_to_q8_1(fp32_data);
    auto q8_dequant = q8_1_to_fp32(q8_blocks);

    std::vector<int16_t> cos_q15, sin_q15;
    std::vector<float> cos_fp32, sin_fp32;
    generate_rope_tables(HEAD_DIM / 2, 42, ROPE_THETA, cos_q15, sin_q15, cos_fp32, sin_fp32);

    // FP32 reference
    auto fp32_ref = q8_dequant;
    apply_rope_fp32_reference(fp32_ref.data(), HEAD_DIM, cos_fp32.data(), sin_fp32.data());

    // Fixed-scale
    std::vector<Q16_1Block> q16_fixed(HEAD_DIM / 32);
    apply_rope_q8_1_to_q16_head_fixed_scale<Q16_1Block>(
        q8_blocks.data(), q16_fixed.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);
    auto fixed_dequant = q16_1_to_fp32(q16_fixed);

    // Data-adaptive
    std::vector<Q16_1Block> q16_adaptive(HEAD_DIM / 32);
    float adaptive_scale = apply_rope_q8_1_to_q16_head_scalar<Q16_1Block>(
        q8_blocks.data(), q16_adaptive.data(), HEAD_DIM,
        cos_q15.data(), sin_q15.data());
    auto adaptive_dequant = q16_1_to_fp32(q16_adaptive);

    float fixed_cos_sim = cosine_similarity(fp32_ref, fixed_dequant);
    float adaptive_cos_sim = cosine_similarity(fp32_ref, adaptive_dequant);

    std::cout << "\n=== Fixed-Scale vs Data-Adaptive Comparison ===" << std::endl;
    std::cout << "Fixed scale: " << EXPECTED_D << std::endl;
    std::cout << "Adaptive scale: " << adaptive_scale << std::endl;
    std::cout << "Fixed cosine sim: " << fixed_cos_sim << std::endl;
    std::cout << "Adaptive cosine sim: " << adaptive_cos_sim << std::endl;

    // Both should be high quality
    EXPECT_GT(fixed_cos_sim, 0.99f);
    EXPECT_GT(adaptive_cos_sim, 0.99f);

    // Data-adaptive may have slightly better precision for this data,
    // but fixed-scale enables integer attention
    std::cout << "Note: Fixed-scale enables integer attention with consistent Q/K/V scales" << std::endl;
}
