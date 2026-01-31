/**
 * @file Test__Q4_1_SIMD_Unpack.cpp
 * @brief Unit tests for Q4_1 SIMD unpacking functions
 * @author David Sanftenberg
 *
 * Tests verify that all SIMD implementations (AVX-512, AVX2, SSE, scalar)
 * produce identical results for Q4_1 unpacking operations.
 */

#include <gtest/gtest.h>
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"
#include <vector>
#include <cstring>
#include <random>
#include <algorithm>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q4_1_SIMD_Unpack : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed for reproducibility
        gen_.seed(42);
    }

    /**
     * @brief Create a test Q4_1 block with known pattern
     *
     * Pattern: Nibbles cycle through [0, 15] repeatedly
     * Expected int8 output: [0, 1, 2, ..., 15, 0, 1, ...] (no bias subtraction)
     */
    Q4_1Block create_test_block_pattern()
    {
        Q4_1Block block;

        // Fill with pattern: nibbles [0, 1, 2, ..., 15, 0, 1, ...]
        for (size_t i = 0; i < 16; ++i)
        {
            uint8_t nibble_low = (i * 2) % 16;
            uint8_t nibble_high = (i * 2 + 1) % 16;
            block.qs[i] = (nibble_high << 4) | nibble_low;
        }

        block.d = fp32_to_fp16(1.0f); // Scale doesn't affect unpacking
        block.m = fp32_to_fp16(0.0f); // Min doesn't affect unpacking
        return block;
    }

    /**
     * @brief Create a random Q4_1 block
     */
    Q4_1Block create_random_block()
    {
        Q4_1Block block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);

        for (size_t i = 0; i < 16; ++i)
        {
            block.qs[i] = dist(gen_);
        }

        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);
        return block;
    }

    /**
     * @brief Verify two int8 arrays are identical
     */
    void verify_arrays_equal(const int8_t *expected, const int8_t *actual, size_t count, const std::string &context)
    {
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_EQ(expected[i], actual[i])
                << context << ": Mismatch at index " << i
                << " (expected " << static_cast<int>(expected[i])
                << ", got " << static_cast<int>(actual[i]) << ")";
        }
    }

    std::mt19937 gen_;
};

// ============================================================================
// Scalar Reference Tests
// ============================================================================

TEST_F(Q4_1_SIMD_Unpack, ScalarReference_Pattern)
{
    Q4_1Block block = create_test_block_pattern();
    int8_t output[32];

    unpack_q4_1_to_int8_scalar(block, output);

    // Verify pattern with split layout (GGUF Q4_1 spec):
    // Indices 0-15: Low nibbles of qs[0..15] -> even numbers 0, 2, 4...
    // Indices 16-31: High nibbles of qs[0..15] -> odd numbers 1, 3, 5...
    for (size_t i = 0; i < 32; ++i)
    {
        uint8_t nibble;
        if (i < 16)
        {
            // Low nibbles: (i * 2) % 16
            nibble = (i * 2) % 16;
        }
        else
        {
            // High nibbles: ((i - 16) * 2 + 1) % 16
            nibble = ((i - 16) * 2 + 1) % 16;
        }

        int8_t expected = static_cast<int8_t>(nibble); // No bias subtraction for Q4_1
        EXPECT_EQ(expected, output[i])
            << "Scalar pattern mismatch at index " << i;
    }
}

TEST_F(Q4_1_SIMD_Unpack, ScalarReference_AllZeros)
{
    Q4_1Block block;
    std::memset(&block, 0, sizeof(block));
    block.d = fp32_to_fp16(1.0f);
    block.m = fp32_to_fp16(0.0f);

    int8_t output[32];
    unpack_q4_1_to_int8_scalar(block, output);

    // All nibbles = 0 → all output = 0
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_EQ(0, output[i])
            << "Scalar all-zeros mismatch at index " << i;
    }
}

TEST_F(Q4_1_SIMD_Unpack, ScalarReference_AllOnes)
{
    Q4_1Block block;
    std::memset(block.qs, 0xFF, 16); // All nibbles = 15
    block.d = fp32_to_fp16(1.0f);
    block.m = fp32_to_fp16(0.0f);

    int8_t output[32];
    unpack_q4_1_to_int8_scalar(block, output);

    // All nibbles = 15 → all output = 15
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_EQ(15, output[i])
            << "Scalar all-ones mismatch at index " << i;
    }
}

TEST_F(Q4_1_SIMD_Unpack, ScalarReference_Range)
{
    Q4_1Block block = create_test_block_pattern();
    int8_t output[32];

    unpack_q4_1_to_int8_scalar(block, output);

    // Verify range: all values in [0, 15]
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_GE(output[i], 0) << "Value below range at index " << i;
        EXPECT_LE(output[i], 15) << "Value above range at index " << i;
    }
}

// ============================================================================
// AVX2 vs Scalar Parity
// ============================================================================

#if defined(__AVX2__)
TEST_F(Q4_1_SIMD_Unpack, AVX2_vs_Scalar_Pattern)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    Q4_1Block block = create_test_block_pattern();
    int8_t scalar_output[32];
    int8_t avx2_output[32];

    unpack_q4_1_to_int8_scalar(block, scalar_output);
    unpack_q4_1_to_int8_avx2(block, avx2_output);

    verify_arrays_equal(scalar_output, avx2_output, 32, "AVX2 vs Scalar (pattern)");
}

TEST_F(Q4_1_SIMD_Unpack, AVX2_vs_Scalar_Random)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    // Test 100 random blocks
    for (int trial = 0; trial < 100; ++trial)
    {
        Q4_1Block block = create_random_block();
        int8_t scalar_output[32];
        int8_t avx2_output[32];

        unpack_q4_1_to_int8_scalar(block, scalar_output);
        unpack_q4_1_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32,
                            "AVX2 vs Scalar (random trial " + std::to_string(trial) + ")");
    }
}

TEST_F(Q4_1_SIMD_Unpack, AVX2_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    // Edge case 1: All zeros
    {
        Q4_1Block block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);

        int8_t scalar_output[32];
        int8_t avx2_output[32];

        unpack_q4_1_to_int8_scalar(block, scalar_output);
        unpack_q4_1_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32, "AVX2 vs Scalar (all zeros)");
    }

    // Edge case 2: All ones (0xFF)
    {
        Q4_1Block block;
        std::memset(block.qs, 0xFF, 16);
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);

        int8_t scalar_output[32];
        int8_t avx2_output[32];

        unpack_q4_1_to_int8_scalar(block, scalar_output);
        unpack_q4_1_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32, "AVX2 vs Scalar (all ones)");
    }

    // Edge case 3: Alternating 0x0F and 0xF0
    {
        Q4_1Block block;
        for (size_t i = 0; i < 16; ++i)
        {
            block.qs[i] = (i % 2 == 0) ? 0x0F : 0xF0;
        }
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);

        int8_t scalar_output[32];
        int8_t avx2_output[32];

        unpack_q4_1_to_int8_scalar(block, scalar_output);
        unpack_q4_1_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32, "AVX2 vs Scalar (alternating)");
    }
}
#endif

// ============================================================================
// AVX-512 vs Scalar Parity
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST_F(Q4_1_SIMD_Unpack, AVX512_vs_Scalar_Pattern)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    Q4_1Block block = create_test_block_pattern();
    int8_t scalar_output[32];
    int8_t avx512_output[32];

    unpack_q4_1_to_int8_scalar(block, scalar_output);
    unpack_q4_1_to_int8_avx512(block, avx512_output);

    verify_arrays_equal(scalar_output, avx512_output, 32, "AVX-512 vs Scalar (pattern)");
}

TEST_F(Q4_1_SIMD_Unpack, AVX512_vs_Scalar_Random)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    // Test 100 random blocks
    for (int trial = 0; trial < 100; ++trial)
    {
        Q4_1Block block = create_random_block();
        int8_t scalar_output[32];
        int8_t avx512_output[32];

        unpack_q4_1_to_int8_scalar(block, scalar_output);
        unpack_q4_1_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32,
                            "AVX-512 vs Scalar (random trial " + std::to_string(trial) + ")");
    }
}

TEST_F(Q4_1_SIMD_Unpack, AVX512_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    // Edge case 1: All zeros
    {
        Q4_1Block block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);

        int8_t scalar_output[32];
        int8_t avx512_output[32];

        unpack_q4_1_to_int8_scalar(block, scalar_output);
        unpack_q4_1_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32, "AVX-512 vs Scalar (all zeros)");
    }

    // Edge case 2: All ones (0xFF)
    {
        Q4_1Block block;
        std::memset(block.qs, 0xFF, 16);
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);

        int8_t scalar_output[32];
        int8_t avx512_output[32];

        unpack_q4_1_to_int8_scalar(block, scalar_output);
        unpack_q4_1_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32, "AVX-512 vs Scalar (all ones)");
    }
}
#endif

// ============================================================================
// Auto-Dispatch Tests
// ============================================================================

TEST_F(Q4_1_SIMD_Unpack, AutoDispatch_Pattern)
{
    Q4_1Block block = create_test_block_pattern();
    int8_t scalar_output[32];
    int8_t dispatch_output[32];

    unpack_q4_1_to_int8_scalar(block, scalar_output);
    unpack_q4_1_to_int8(block, dispatch_output);

    verify_arrays_equal(scalar_output, dispatch_output, 32, "Auto-dispatch vs Scalar (pattern)");
}

TEST_F(Q4_1_SIMD_Unpack, AutoDispatch_Random)
{
    // Test 100 random blocks
    for (int trial = 0; trial < 100; ++trial)
    {
        Q4_1Block block = create_random_block();
        int8_t scalar_output[32];
        int8_t dispatch_output[32];

        unpack_q4_1_to_int8_scalar(block, scalar_output);
        unpack_q4_1_to_int8(block, dispatch_output);

        verify_arrays_equal(scalar_output, dispatch_output, 32,
                            "Auto-dispatch vs Scalar (random trial " + std::to_string(trial) + ")");
    }
}
