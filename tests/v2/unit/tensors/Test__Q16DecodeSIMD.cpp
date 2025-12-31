/**
 * @file Test__Q16DecodeSIMD.cpp
 * @brief Unit tests for SIMD Q16 block decode (dequantization) correctness
 *
 * Verifies that AVX2 and AVX512 implementations produce identical results
 * to the scalar reference implementation for Q16 block dequantization.
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

class Q16DecodeSIMDTest : public ::testing::Test
{
protected:
    std::mt19937 rng{42};
    std::uniform_int_distribution<int16_t> int16_dist{-32000, 32000};
    std::uniform_real_distribution<float> scale_dist{0.0001f, 0.1f};

    // Generate a random Q16 block
    template <typename BlockType>
    BlockType generate_random_block()
    {
        BlockType block{};
        block.d = scale_dist(rng);
        int32_t sum = 0;
        for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
        {
            block.qs[i] = int16_dist(rng);
            sum += block.qs[i];
        }
        block.sum_qs = sum;
        return block;
    }

    // Compare two float arrays for near-equality
    template <size_t N>
    bool arrays_near_equal(const float *a, const float *b, float rtol = 1e-5f, float atol = 1e-7f)
    {
        for (size_t i = 0; i < N; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
            if (diff > atol + rtol * max_val)
            {
                return false;
            }
        }
        return true;
    }

    // Compute max absolute difference between arrays
    template <size_t N>
    float max_abs_diff(const float *a, const float *b)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < N; ++i)
        {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        return max_diff;
    }
};

// ============================================================================
// Block32 Decode Tests
// ============================================================================

TEST_F(Q16DecodeSIMDTest, Block32_ScalarMatchesExpected)
{
    Q16_1Block block = generate_random_block<Q16_1Block>();

    alignas(64) float output[Q16_1Block::BLOCK_SIZE];
    decode_q16_block_to_fp32_scalar<Q16_1Block>(block, output);

    // Verify against manual calculation
    for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
    {
        float expected = block.d * static_cast<float>(block.qs[i]);
        EXPECT_FLOAT_EQ(output[i], expected) << "Mismatch at index " << i;
    }
}

#if defined(__AVX2__)
TEST_F(Q16DecodeSIMDTest, Block32_AVX2MatchesScalar)
{
    constexpr size_t NUM_BLOCKS = 1000;

    for (size_t b = 0; b < NUM_BLOCKS; ++b)
    {
        Q16_1Block block = generate_random_block<Q16_1Block>();

        alignas(64) float scalar_output[Q16_1Block::BLOCK_SIZE];
        alignas(64) float avx2_output[Q16_1Block::BLOCK_SIZE];

        decode_q16_block_to_fp32_scalar<Q16_1Block>(block, scalar_output);
        decode_q16_block_to_fp32_avx2<Q16_1Block>(block, avx2_output);

        ASSERT_TRUE((arrays_near_equal<Q16_1Block::BLOCK_SIZE>(scalar_output, avx2_output)))
            << "Block " << b << " mismatch, max diff: "
            << max_abs_diff<Q16_1Block::BLOCK_SIZE>(scalar_output, avx2_output);
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16DecodeSIMDTest, Block32_AVX512MatchesScalar)
{
    constexpr size_t NUM_BLOCKS = 1000;

    for (size_t b = 0; b < NUM_BLOCKS; ++b)
    {
        Q16_1Block block = generate_random_block<Q16_1Block>();

        alignas(64) float scalar_output[Q16_1Block::BLOCK_SIZE];
        alignas(64) float avx512_output[Q16_1Block::BLOCK_SIZE];

        decode_q16_block_to_fp32_scalar<Q16_1Block>(block, scalar_output);
        decode_q16_block_to_fp32_avx512<Q16_1Block>(block, avx512_output);

        ASSERT_TRUE((arrays_near_equal<Q16_1Block::BLOCK_SIZE>(scalar_output, avx512_output)))
            << "Block " << b << " mismatch, max diff: "
            << max_abs_diff<Q16_1Block::BLOCK_SIZE>(scalar_output, avx512_output);
    }
}
#endif

// ============================================================================
// Block64 Decode Tests
// ============================================================================

TEST_F(Q16DecodeSIMDTest, Block64_ScalarMatchesExpected)
{
    Q16_1Block_64 block = generate_random_block<Q16_1Block_64>();

    alignas(64) float output[Q16_1Block_64::BLOCK_SIZE];
    decode_q16_block_to_fp32_scalar<Q16_1Block_64>(block, output);

    for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
    {
        float expected = block.d * static_cast<float>(block.qs[i]);
        EXPECT_FLOAT_EQ(output[i], expected) << "Mismatch at index " << i;
    }
}

#if defined(__AVX2__)
TEST_F(Q16DecodeSIMDTest, Block64_AVX2MatchesScalar)
{
    constexpr size_t NUM_BLOCKS = 1000;

    for (size_t b = 0; b < NUM_BLOCKS; ++b)
    {
        Q16_1Block_64 block = generate_random_block<Q16_1Block_64>();

        alignas(64) float scalar_output[Q16_1Block_64::BLOCK_SIZE];
        alignas(64) float avx2_output[Q16_1Block_64::BLOCK_SIZE];

        decode_q16_block_to_fp32_scalar<Q16_1Block_64>(block, scalar_output);
        decode_q16_block_to_fp32_avx2<Q16_1Block_64>(block, avx2_output);

        ASSERT_TRUE((arrays_near_equal<Q16_1Block_64::BLOCK_SIZE>(scalar_output, avx2_output)))
            << "Block " << b << " mismatch, max diff: "
            << max_abs_diff<Q16_1Block_64::BLOCK_SIZE>(scalar_output, avx2_output);
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16DecodeSIMDTest, Block64_AVX512MatchesScalar)
{
    constexpr size_t NUM_BLOCKS = 1000;

    for (size_t b = 0; b < NUM_BLOCKS; ++b)
    {
        Q16_1Block_64 block = generate_random_block<Q16_1Block_64>();

        alignas(64) float scalar_output[Q16_1Block_64::BLOCK_SIZE];
        alignas(64) float avx512_output[Q16_1Block_64::BLOCK_SIZE];

        decode_q16_block_to_fp32_scalar<Q16_1Block_64>(block, scalar_output);
        decode_q16_block_to_fp32_avx512<Q16_1Block_64>(block, avx512_output);

        ASSERT_TRUE((arrays_near_equal<Q16_1Block_64::BLOCK_SIZE>(scalar_output, avx512_output)))
            << "Block " << b << " mismatch, max diff: "
            << max_abs_diff<Q16_1Block_64::BLOCK_SIZE>(scalar_output, avx512_output);
    }
}
#endif

// ============================================================================
// Block128 Decode Tests
// ============================================================================

TEST_F(Q16DecodeSIMDTest, Block128_ScalarMatchesExpected)
{
    Q16_1Block_128 block = generate_random_block<Q16_1Block_128>();

    alignas(64) float output[Q16_1Block_128::BLOCK_SIZE];
    decode_q16_block_to_fp32_scalar<Q16_1Block_128>(block, output);

    for (size_t i = 0; i < Q16_1Block_128::BLOCK_SIZE; ++i)
    {
        float expected = block.d * static_cast<float>(block.qs[i]);
        EXPECT_FLOAT_EQ(output[i], expected) << "Mismatch at index " << i;
    }
}

#if defined(__AVX2__)
TEST_F(Q16DecodeSIMDTest, Block128_AVX2MatchesScalar)
{
    constexpr size_t NUM_BLOCKS = 1000;

    for (size_t b = 0; b < NUM_BLOCKS; ++b)
    {
        Q16_1Block_128 block = generate_random_block<Q16_1Block_128>();

        alignas(64) float scalar_output[Q16_1Block_128::BLOCK_SIZE];
        alignas(64) float avx2_output[Q16_1Block_128::BLOCK_SIZE];

        decode_q16_block_to_fp32_scalar<Q16_1Block_128>(block, scalar_output);
        decode_q16_block_to_fp32_avx2<Q16_1Block_128>(block, avx2_output);

        ASSERT_TRUE((arrays_near_equal<Q16_1Block_128::BLOCK_SIZE>(scalar_output, avx2_output)))
            << "Block " << b << " mismatch, max diff: "
            << max_abs_diff<Q16_1Block_128::BLOCK_SIZE>(scalar_output, avx2_output);
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16DecodeSIMDTest, Block128_AVX512MatchesScalar)
{
    constexpr size_t NUM_BLOCKS = 1000;

    for (size_t b = 0; b < NUM_BLOCKS; ++b)
    {
        Q16_1Block_128 block = generate_random_block<Q16_1Block_128>();

        alignas(64) float scalar_output[Q16_1Block_128::BLOCK_SIZE];
        alignas(64) float avx512_output[Q16_1Block_128::BLOCK_SIZE];

        decode_q16_block_to_fp32_scalar<Q16_1Block_128>(block, scalar_output);
        decode_q16_block_to_fp32_avx512<Q16_1Block_128>(block, avx512_output);

        ASSERT_TRUE((arrays_near_equal<Q16_1Block_128::BLOCK_SIZE>(scalar_output, avx512_output)))
            << "Block " << b << " mismatch, max diff: "
            << max_abs_diff<Q16_1Block_128::BLOCK_SIZE>(scalar_output, avx512_output);
    }
}
#endif

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Q16DecodeSIMDTest, ZeroScale_AllZeroOutput)
{
    Q16_1Block_64 block{};
    block.d = 0.0f;
    for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
    {
        block.qs[i] = static_cast<int16_t>(i * 100); // Non-zero values
    }

    alignas(64) float scalar_output[Q16_1Block_64::BLOCK_SIZE];
    decode_q16_block_to_fp32_scalar<Q16_1Block_64>(block, scalar_output);

    for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(scalar_output[i], 0.0f) << "Expected zero at index " << i;
    }

#if defined(__AVX512F__)
    alignas(64) float avx512_output[Q16_1Block_64::BLOCK_SIZE];
    decode_q16_block_to_fp32_avx512<Q16_1Block_64>(block, avx512_output);

    for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(avx512_output[i], 0.0f) << "AVX512: Expected zero at index " << i;
    }
#endif

#if defined(__AVX2__)
    alignas(64) float avx2_output[Q16_1Block_64::BLOCK_SIZE];
    decode_q16_block_to_fp32_avx2<Q16_1Block_64>(block, avx2_output);

    for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(avx2_output[i], 0.0f) << "AVX2: Expected zero at index " << i;
    }
#endif
}

TEST_F(Q16DecodeSIMDTest, ExtremeValues_MaxMin)
{
    Q16_1Block_64 block{};
    block.d = 0.01f;

    // Fill with alternating max and min INT16 values
    for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
    {
        block.qs[i] = (i % 2 == 0) ? 32767 : -32768;
    }

    alignas(64) float scalar_output[Q16_1Block_64::BLOCK_SIZE];
    decode_q16_block_to_fp32_scalar<Q16_1Block_64>(block, scalar_output);

    // Verify values
    for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
    {
        float expected = block.d * static_cast<float>(block.qs[i]);
        EXPECT_FLOAT_EQ(scalar_output[i], expected);
    }

#if defined(__AVX512F__)
    alignas(64) float avx512_output[Q16_1Block_64::BLOCK_SIZE];
    decode_q16_block_to_fp32_avx512<Q16_1Block_64>(block, avx512_output);

    ASSERT_TRUE((arrays_near_equal<Q16_1Block_64::BLOCK_SIZE>(scalar_output, avx512_output)))
        << "AVX512 mismatch with extreme values";
#endif

#if defined(__AVX2__)
    alignas(64) float avx2_output[Q16_1Block_64::BLOCK_SIZE];
    decode_q16_block_to_fp32_avx2<Q16_1Block_64>(block, avx2_output);

    ASSERT_TRUE((arrays_near_equal<Q16_1Block_64::BLOCK_SIZE>(scalar_output, avx2_output)))
        << "AVX2 mismatch with extreme values";
#endif
}

TEST_F(Q16DecodeSIMDTest, VerySmallScale_Precision)
{
    Q16_1Block_128 block{};
    block.d = 1e-7f; // Very small scale

    for (size_t i = 0; i < Q16_1Block_128::BLOCK_SIZE; ++i)
    {
        block.qs[i] = static_cast<int16_t>(i - 64);
    }

    alignas(64) float scalar_output[Q16_1Block_128::BLOCK_SIZE];
    alignas(64) float simd_output[Q16_1Block_128::BLOCK_SIZE];

    decode_q16_block_to_fp32_scalar<Q16_1Block_128>(block, scalar_output);

#if defined(__AVX512F__)
    decode_q16_block_to_fp32_avx512<Q16_1Block_128>(block, simd_output);
    ASSERT_TRUE((arrays_near_equal<Q16_1Block_128::BLOCK_SIZE>(scalar_output, simd_output)))
        << "AVX512 mismatch with very small scale";
#elif defined(__AVX2__)
    decode_q16_block_to_fp32_avx2<Q16_1Block_128>(block, simd_output);
    ASSERT_TRUE((arrays_near_equal<Q16_1Block_128::BLOCK_SIZE>(scalar_output, simd_output)))
        << "AVX2 mismatch with very small scale";
#endif
}

TEST_F(Q16DecodeSIMDTest, AutoDispatch_MatchesScalar)
{
    // Test the auto-dispatch function across all block sizes
    Q16_1Block block32 = generate_random_block<Q16_1Block>();
    Q16_1Block_64 block64 = generate_random_block<Q16_1Block_64>();
    Q16_1Block_128 block128 = generate_random_block<Q16_1Block_128>();

    alignas(64) float scalar32[Q16_1Block::BLOCK_SIZE];
    alignas(64) float auto32[Q16_1Block::BLOCK_SIZE];
    decode_q16_block_to_fp32_scalar<Q16_1Block>(block32, scalar32);
    decode_q16_block_to_fp32<Q16_1Block>(block32, auto32);
    EXPECT_TRUE((arrays_near_equal<Q16_1Block::BLOCK_SIZE>(scalar32, auto32)));

    alignas(64) float scalar64[Q16_1Block_64::BLOCK_SIZE];
    alignas(64) float auto64[Q16_1Block_64::BLOCK_SIZE];
    decode_q16_block_to_fp32_scalar<Q16_1Block_64>(block64, scalar64);
    decode_q16_block_to_fp32<Q16_1Block_64>(block64, auto64);
    EXPECT_TRUE((arrays_near_equal<Q16_1Block_64::BLOCK_SIZE>(scalar64, auto64)));

    alignas(64) float scalar128[Q16_1Block_128::BLOCK_SIZE];
    alignas(64) float auto128[Q16_1Block_128::BLOCK_SIZE];
    decode_q16_block_to_fp32_scalar<Q16_1Block_128>(block128, scalar128);
    decode_q16_block_to_fp32<Q16_1Block_128>(block128, auto128);
    EXPECT_TRUE((arrays_near_equal<Q16_1Block_128::BLOCK_SIZE>(scalar128, auto128)));
}
