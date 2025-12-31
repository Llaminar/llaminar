/**
 * @file Test__Q16FixedScaleSIMD.cpp
 * @brief Unit tests for SIMD fixed-scale quantization correctness
 *
 * Verifies that AVX2 and AVX512 implementations produce identical results
 * to the scalar reference implementation for fixed-scale quantization.
 *
 * @author Llaminar Team
 * @date 2025-12-31
 */

#include <gtest/gtest.h>
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q16FixedScaleSIMDTest : public ::testing::Test
{
protected:
    static constexpr float KV_CACHE_SCALE = 8.0f;
    static constexpr int16_t MAX_SAFE_INT16_64 = 23170;  // For head_dim=64
    static constexpr int16_t MAX_SAFE_INT16_128 = 16383; // For head_dim=128

    std::mt19937 rng{42};
    std::uniform_real_distribution<float> normal_dist{-8.0f, 8.0f};
    std::uniform_real_distribution<float> outlier_dist{-20.0f, 20.0f};

    // Generate random FP32 data with some outliers
    template <typename BlockType>
    std::vector<float> generate_test_data(size_t num_blocks, float outlier_ratio = 0.1f)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        std::vector<float> data(num_blocks * BLOCK_SIZE);
        std::uniform_real_distribution<float> coin(0.0f, 1.0f);

        for (auto &v : data)
        {
            if (coin(rng) < outlier_ratio)
            {
                v = outlier_dist(rng);
            }
            else
            {
                v = normal_dist(rng);
            }
        }
        return data;
    }

    // Compare two blocks for equality
    template <typename BlockType>
    bool blocks_equal(const BlockType &a, const BlockType &b, float d_tolerance = 1e-6f)
    {
        if (std::abs(a.d - b.d) > d_tolerance)
        {
            return false;
        }
        if (a.sum_qs != b.sum_qs)
        {
            return false;
        }
        for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
        {
            if (a.qs[i] != b.qs[i])
            {
                return false;
            }
        }
        return true;
    }

    // Verify all INT16 values are within safe limit
    template <typename BlockType>
    bool verify_within_limit(const BlockType &block, int16_t limit)
    {
        for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
        {
            if (std::abs(block.qs[i]) > limit)
            {
                return false;
            }
        }
        return true;
    }

    // Verify sum_qs is correct
    template <typename BlockType>
    bool verify_sum(const BlockType &block)
    {
        int32_t computed_sum = 0;
        for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
        {
            computed_sum += block.qs[i];
        }
        return computed_sum == block.sum_qs;
    }
};

// ============================================================================
// Block32 Fixed-Scale Tests
// ============================================================================

TEST_F(Q16FixedScaleSIMDTest, Block32_ScalarProducesCorrectResults)
{
    const size_t num_blocks = 64;
    auto input = generate_test_data<Q16_1Block>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block block;
        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block>(
            input.data() + b * Q16_1Block::BLOCK_SIZE, block, d, inv_d, MAX_SAFE_INT16_128);

        // Verify scale is fixed
        EXPECT_FLOAT_EQ(block.d, d);

        // Verify all values within VNNI-safe limit
        EXPECT_TRUE(verify_within_limit(block, MAX_SAFE_INT16_128));

        // Verify sum is correct
        EXPECT_TRUE(verify_sum(block));
    }
}

#if defined(__AVX2__)
TEST_F(Q16FixedScaleSIMDTest, Block32_AVX2_MatchesScalar)
{
    const size_t num_blocks = 64;
    auto input = generate_test_data<Q16_1Block>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block scalar_block, avx2_block;

        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block>(
            input.data() + b * Q16_1Block::BLOCK_SIZE, scalar_block, d, inv_d, MAX_SAFE_INT16_128);

        quantize_fp32_to_q16_block_fixed_scale_avx2<Q16_1Block>(
            input.data() + b * Q16_1Block::BLOCK_SIZE, avx2_block, d, inv_d, MAX_SAFE_INT16_128);

        EXPECT_TRUE(blocks_equal(scalar_block, avx2_block))
            << "AVX2 mismatch at block " << b;
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16FixedScaleSIMDTest, Block32_AVX512_MatchesScalar)
{
    const size_t num_blocks = 64;
    auto input = generate_test_data<Q16_1Block>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block scalar_block, avx512_block;

        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block>(
            input.data() + b * Q16_1Block::BLOCK_SIZE, scalar_block, d, inv_d, MAX_SAFE_INT16_128);

        quantize_fp32_to_q16_block_fixed_scale_avx512<Q16_1Block>(
            input.data() + b * Q16_1Block::BLOCK_SIZE, avx512_block, d, inv_d, MAX_SAFE_INT16_128);

        EXPECT_TRUE(blocks_equal(scalar_block, avx512_block))
            << "AVX512 mismatch at block " << b;
    }
}
#endif

// ============================================================================
// Block64 Fixed-Scale Tests
// ============================================================================

TEST_F(Q16FixedScaleSIMDTest, Block64_ScalarProducesCorrectResults)
{
    const size_t num_blocks = 32;
    auto input = generate_test_data<Q16_1Block_64>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block_64 block;
        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block_64>(
            input.data() + b * Q16_1Block_64::BLOCK_SIZE, block, d, inv_d, MAX_SAFE_INT16_64);

        EXPECT_FLOAT_EQ(block.d, d);
        EXPECT_TRUE(verify_within_limit(block, MAX_SAFE_INT16_64));
        EXPECT_TRUE(verify_sum(block));
    }
}

#if defined(__AVX2__)
TEST_F(Q16FixedScaleSIMDTest, Block64_AVX2_MatchesScalar)
{
    const size_t num_blocks = 32;
    auto input = generate_test_data<Q16_1Block_64>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block_64 scalar_block, avx2_block;

        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block_64>(
            input.data() + b * Q16_1Block_64::BLOCK_SIZE, scalar_block, d, inv_d, MAX_SAFE_INT16_64);

        quantize_fp32_to_q16_block_fixed_scale_avx2<Q16_1Block_64>(
            input.data() + b * Q16_1Block_64::BLOCK_SIZE, avx2_block, d, inv_d, MAX_SAFE_INT16_64);

        EXPECT_TRUE(blocks_equal(scalar_block, avx2_block))
            << "AVX2 mismatch at block " << b;
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16FixedScaleSIMDTest, Block64_AVX512_MatchesScalar)
{
    const size_t num_blocks = 32;
    auto input = generate_test_data<Q16_1Block_64>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block_64 scalar_block, avx512_block;

        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block_64>(
            input.data() + b * Q16_1Block_64::BLOCK_SIZE, scalar_block, d, inv_d, MAX_SAFE_INT16_64);

        quantize_fp32_to_q16_block_fixed_scale_avx512<Q16_1Block_64>(
            input.data() + b * Q16_1Block_64::BLOCK_SIZE, avx512_block, d, inv_d, MAX_SAFE_INT16_64);

        EXPECT_TRUE(blocks_equal(scalar_block, avx512_block))
            << "AVX512 mismatch at block " << b;
    }
}
#endif

// ============================================================================
// Block128 Fixed-Scale Tests
// ============================================================================

TEST_F(Q16FixedScaleSIMDTest, Block128_ScalarProducesCorrectResults)
{
    const size_t num_blocks = 16;
    auto input = generate_test_data<Q16_1Block_128>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block_128 block;
        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block_128>(
            input.data() + b * Q16_1Block_128::BLOCK_SIZE, block, d, inv_d, MAX_SAFE_INT16_128);

        EXPECT_FLOAT_EQ(block.d, d);
        EXPECT_TRUE(verify_within_limit(block, MAX_SAFE_INT16_128));
        EXPECT_TRUE(verify_sum(block));
    }
}

#if defined(__AVX2__)
TEST_F(Q16FixedScaleSIMDTest, Block128_AVX2_MatchesScalar)
{
    const size_t num_blocks = 16;
    auto input = generate_test_data<Q16_1Block_128>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block_128 scalar_block, avx2_block;

        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block_128>(
            input.data() + b * Q16_1Block_128::BLOCK_SIZE, scalar_block, d, inv_d, MAX_SAFE_INT16_128);

        quantize_fp32_to_q16_block_fixed_scale_avx2<Q16_1Block_128>(
            input.data() + b * Q16_1Block_128::BLOCK_SIZE, avx2_block, d, inv_d, MAX_SAFE_INT16_128);

        EXPECT_TRUE(blocks_equal(scalar_block, avx2_block))
            << "AVX2 mismatch at block " << b;
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16FixedScaleSIMDTest, Block128_AVX512_MatchesScalar)
{
    const size_t num_blocks = 16;
    auto input = generate_test_data<Q16_1Block_128>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block_128 scalar_block, avx512_block;

        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block_128>(
            input.data() + b * Q16_1Block_128::BLOCK_SIZE, scalar_block, d, inv_d, MAX_SAFE_INT16_128);

        quantize_fp32_to_q16_block_fixed_scale_avx512<Q16_1Block_128>(
            input.data() + b * Q16_1Block_128::BLOCK_SIZE, avx512_block, d, inv_d, MAX_SAFE_INT16_128);

        EXPECT_TRUE(blocks_equal(scalar_block, avx512_block))
            << "AVX512 mismatch at block " << b;
    }
}
#endif

// ============================================================================
// Partial Block Tests
// ============================================================================

TEST_F(Q16FixedScaleSIMDTest, Block64_Partial_ScalarCorrectness)
{
    const size_t count = 48; // Partial block (48 of 64)
    std::vector<float> input(64);
    for (size_t i = 0; i < count; ++i)
    {
        input[i] = normal_dist(rng);
    }

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    Q16_1Block_64 block;
    quantize_fp32_to_q16_block_partial_fixed_scale_scalar<Q16_1Block_64>(
        input.data(), count, block, d, inv_d, MAX_SAFE_INT16_64);

    EXPECT_FLOAT_EQ(block.d, d);

    // Verify non-zero values within limit
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_LE(std::abs(block.qs[i]), MAX_SAFE_INT16_64);
    }

    // Verify zero-filled remainder
    for (size_t i = count; i < 64; ++i)
    {
        EXPECT_EQ(block.qs[i], 0);
    }

    // Verify sum (should only include first 'count' elements)
    int32_t expected_sum = 0;
    for (size_t i = 0; i < count; ++i)
    {
        expected_sum += block.qs[i];
    }
    EXPECT_EQ(block.sum_qs, expected_sum);
}

#if defined(__AVX512F__)
TEST_F(Q16FixedScaleSIMDTest, Block64_Partial_AVX512_MatchesScalar)
{
    // Test various partial sizes
    const std::vector<size_t> test_counts = {16, 32, 48, 60};

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t count : test_counts)
    {
        std::vector<float> input(64);
        for (size_t i = 0; i < count; ++i)
        {
            input[i] = normal_dist(rng);
        }

        Q16_1Block_64 scalar_block, avx512_block;

        quantize_fp32_to_q16_block_partial_fixed_scale_scalar<Q16_1Block_64>(
            input.data(), count, scalar_block, d, inv_d, MAX_SAFE_INT16_64);

        quantize_fp32_to_q16_block_partial_fixed_scale_avx512<Q16_1Block_64>(
            input.data(), count, avx512_block, d, inv_d, MAX_SAFE_INT16_64);

        EXPECT_TRUE(blocks_equal(scalar_block, avx512_block))
            << "AVX512 partial mismatch at count=" << count;
    }
}
#endif

#if defined(__AVX2__)
TEST_F(Q16FixedScaleSIMDTest, Block64_Partial_AVX2_MatchesScalar)
{
    // Test various partial sizes
    const std::vector<size_t> test_counts = {8, 16, 32, 48, 60};

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t count : test_counts)
    {
        std::vector<float> input(64);
        for (size_t i = 0; i < count; ++i)
        {
            input[i] = normal_dist(rng);
        }

        Q16_1Block_64 scalar_block, avx2_block;

        quantize_fp32_to_q16_block_partial_fixed_scale_scalar<Q16_1Block_64>(
            input.data(), count, scalar_block, d, inv_d, MAX_SAFE_INT16_64);

        quantize_fp32_to_q16_block_partial_fixed_scale_avx2<Q16_1Block_64>(
            input.data(), count, avx2_block, d, inv_d, MAX_SAFE_INT16_64);

        EXPECT_TRUE(blocks_equal(scalar_block, avx2_block))
            << "AVX2 partial mismatch at count=" << count;
    }
}
#endif

// ============================================================================
// Clipping Tests (verify VNNI-safe clipping works)
// ============================================================================

TEST_F(Q16FixedScaleSIMDTest, ClippingToVNNISafeLimit)
{
    // Create input that would exceed VNNI-safe limit without clipping
    // With scale=8.0 and max_safe=16383, values beyond ~4.0 would exceed limit
    std::vector<float> input(32);
    for (size_t i = 0; i < 32; ++i)
    {
        input[i] = (i % 2 == 0) ? 10.0f : -10.0f; // Values that need clipping
    }

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    Q16_1Block scalar_block;
    quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block>(
        input.data(), scalar_block, d, inv_d, MAX_SAFE_INT16_128);

    // All values should be clipped to ±16383
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_LE(std::abs(scalar_block.qs[i]), MAX_SAFE_INT16_128)
            << "Value at " << i << " exceeds VNNI-safe limit";
        if (input[i] > 0)
        {
            EXPECT_EQ(scalar_block.qs[i], MAX_SAFE_INT16_128);
        }
        else
        {
            EXPECT_EQ(scalar_block.qs[i], -MAX_SAFE_INT16_128);
        }
    }

#if defined(__AVX512F__)
    Q16_1Block avx512_block;
    quantize_fp32_to_q16_block_fixed_scale_avx512<Q16_1Block>(
        input.data(), avx512_block, d, inv_d, MAX_SAFE_INT16_128);

    // AVX512 should produce identical clipping
    EXPECT_TRUE(blocks_equal(scalar_block, avx512_block))
        << "AVX512 clipping differs from scalar";
#endif

#if defined(__AVX2__)
    Q16_1Block avx2_block;
    quantize_fp32_to_q16_block_fixed_scale_avx2<Q16_1Block>(
        input.data(), avx2_block, d, inv_d, MAX_SAFE_INT16_128);

    // AVX2 should produce identical clipping
    EXPECT_TRUE(blocks_equal(scalar_block, avx2_block))
        << "AVX2 clipping differs from scalar";
#endif
}

// ============================================================================
// Auto-Dispatch Test
// ============================================================================

TEST_F(Q16FixedScaleSIMDTest, AutoDispatch_ProducesCorrectResults)
{
    const size_t num_blocks = 32;
    auto input = generate_test_data<Q16_1Block_64>(num_blocks);

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        Q16_1Block_64 scalar_block, auto_block;

        // Scalar reference
        quantize_fp32_to_q16_block_fixed_scale_scalar<Q16_1Block_64>(
            input.data() + b * Q16_1Block_64::BLOCK_SIZE, scalar_block, d, inv_d, MAX_SAFE_INT16_64);

        // Auto-dispatch (uses best available)
        quantize_fp32_to_q16_block_fixed_scale<Q16_1Block_64>(
            input.data() + b * Q16_1Block_64::BLOCK_SIZE, auto_block, d, inv_d, MAX_SAFE_INT16_64);

        EXPECT_TRUE(blocks_equal(scalar_block, auto_block))
            << "Auto-dispatch mismatch at block " << b;
    }
}
