/**
 * @file Test__Q8_0_SIMD_Accuracy.cpp
 * @brief Unit tests for Q8_0 decode SIMD accuracy (FP32/FP16/BF16/Q8_K)
 *
 * Validates that AVX512, AVX2, and scalar implementations of Q8_0 quantization
 * produce identical or nearly-identical results for FP32Tensor, FP16Tensor,
 * BF16Tensor, and Q8_KTensor decode_to_q8_0() methods.
 *
 * Tests cover:
 * - FP32 → Q8_0 quantization (direct)
 * - FP16 → Q8_0 quantization (decode + quantize)
 * - BF16 → Q8_0 quantization (decode + quantize)
 * - Q8_K → Q8_0 extraction (sub-block extraction)
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/FP16Utils.h"
#include "tensors/BlockStructures.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::simd;

// ============================================================================
// Test Fixture
// ============================================================================

class Q8_0_SIMD_Accuracy_Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rng_.seed(42);
    }

    std::mt19937 rng_;

    // Helper: Compare Q8_0 blocks with tolerance
    struct ComparisonResult
    {
        bool passed;
        int max_diff;
        int num_mismatches;
        float scale_diff;
        std::string error_msg;
    };

    ComparisonResult compare_q8_0_blocks(
        const Q8_0Block &a,
        const Q8_0Block &b,
        int tolerance = 1) // Tolerance in quantized int8 domain
    {
        ComparisonResult result{true, 0, 0, 0.0f, ""};

        // Compare scales
        float scale_a = fp16_to_fp32(a.d);
        float scale_b = fp16_to_fp32(b.d);
        result.scale_diff = std::fabs(scale_a - scale_b);

        // Compare quantized values
        for (int i = 0; i < 32; ++i)
        {
            int diff = std::abs(static_cast<int>(a.qs[i]) - static_cast<int>(b.qs[i]));
            result.max_diff = std::max(result.max_diff, diff);
            if (diff > tolerance)
            {
                result.num_mismatches++;
                result.passed = false;
                if (result.error_msg.empty())
                {
                    result.error_msg = "First mismatch at index " + std::to_string(i) +
                                       ": " + std::to_string(static_cast<int>(a.qs[i])) +
                                       " vs " + std::to_string(static_cast<int>(b.qs[i]));
                }
            }
        }

        return result;
    }

    // Helper: Generate random FP32 data
    void generate_random_fp32(float *data, size_t count, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // Helper: Generate random FP16 data
    void generate_random_fp16(uint16_t *data, size_t count, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = fp32_to_fp16(dist(rng_));
        }
    }

    // Helper: Generate random BF16 data
    void generate_random_bf16(uint16_t *data, size_t count, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = fp32_to_bf16(dist(rng_));
        }
    }
};

// ============================================================================
// FP32 → Q8_0 SIMD Accuracy Tests
// ============================================================================

TEST_F(Q8_0_SIMD_Accuracy_Test, FP32_Scalar_vs_AVX2)
{
#ifdef __AVX2__
    constexpr size_t BLOCK_SIZE = 32;
    float fp32_data[BLOCK_SIZE];
    generate_random_fp32(fp32_data, BLOCK_SIZE, -10.0f, 10.0f);

    // Scalar implementation
    Q8_0Block scalar_result;
    decode_fp32_to_q8_0_scalar(fp32_data, scalar_result.qs, &scalar_result.d);

    // AVX2 implementation
    Q8_0Block avx2_result;
    decode_fp32_to_q8_0_avx2(fp32_data, avx2_result.qs, &avx2_result.d);

    // Compare results (should be identical)
    auto cmp = compare_q8_0_blocks(scalar_result, avx2_result, 0);
    EXPECT_TRUE(cmp.passed) << cmp.error_msg;
    EXPECT_EQ(cmp.max_diff, 0) << "Scalar and AVX2 should produce identical results";
    EXPECT_LT(cmp.scale_diff, 1e-6f) << "Scales should be nearly identical";
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, FP32_Scalar_vs_AVX512)
{
#ifdef __AVX512F__
    constexpr size_t BLOCK_SIZE = 32;
    float fp32_data[BLOCK_SIZE];
    generate_random_fp32(fp32_data, BLOCK_SIZE, -10.0f, 10.0f);

    // Scalar implementation
    Q8_0Block scalar_result;
    decode_fp32_to_q8_0_scalar(fp32_data, scalar_result.qs, &scalar_result.d);

    // AVX512 implementation
    Q8_0Block avx512_result;
    decode_fp32_to_q8_0_avx512(fp32_data, avx512_result.qs, &avx512_result.d);

    // Compare results (should be identical)
    auto cmp = compare_q8_0_blocks(scalar_result, avx512_result, 0);
    EXPECT_TRUE(cmp.passed) << cmp.error_msg;
    EXPECT_EQ(cmp.max_diff, 0) << "Scalar and AVX512 should produce identical results";
    EXPECT_LT(cmp.scale_diff, 1e-6f) << "Scales should be nearly identical";
#else
    GTEST_SKIP() << "AVX512 not available";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, FP32_EdgeCases_AllZeros)
{
    constexpr size_t BLOCK_SIZE = 32;
    float fp32_data[BLOCK_SIZE];
    std::memset(fp32_data, 0, sizeof(fp32_data));

    // Scalar
    Q8_0Block scalar_result;
    decode_fp32_to_q8_0_scalar(fp32_data, scalar_result.qs, &scalar_result.d);

#ifdef __AVX2__
    // AVX2
    Q8_0Block avx2_result;
    decode_fp32_to_q8_0_avx2(fp32_data, avx2_result.qs, &avx2_result.d);
    auto cmp_avx2 = compare_q8_0_blocks(scalar_result, avx2_result, 0);
    EXPECT_TRUE(cmp_avx2.passed) << "All-zeros: Scalar vs AVX2";
#endif

#ifdef __AVX512F__
    // AVX512
    Q8_0Block avx512_result;
    decode_fp32_to_q8_0_avx512(fp32_data, avx512_result.qs, &avx512_result.d);
    auto cmp_avx512 = compare_q8_0_blocks(scalar_result, avx512_result, 0);
    EXPECT_TRUE(cmp_avx512.passed) << "All-zeros: Scalar vs AVX512";
#endif

    // All quantized values should be 0
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(scalar_result.qs[i], 0);
    }
}

TEST_F(Q8_0_SIMD_Accuracy_Test, FP32_EdgeCases_SingleLargeValue)
{
    constexpr size_t BLOCK_SIZE = 32;
    float fp32_data[BLOCK_SIZE] = {0};
    fp32_data[15] = 100.0f; // Single large value in the middle

    Q8_0Block scalar_result, simd_result;
    decode_fp32_to_q8_0_scalar(fp32_data, scalar_result.qs, &scalar_result.d);

#ifdef __AVX512F__
    decode_fp32_to_q8_0_avx512(fp32_data, simd_result.qs, &simd_result.d);
    auto cmp = compare_q8_0_blocks(scalar_result, simd_result, 0);
    EXPECT_TRUE(cmp.passed) << "Single large value test failed";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, FP32_RandomData_MultipleSamples)
{
    // Test 100 random blocks to ensure robustness
    constexpr int NUM_SAMPLES = 100;
    constexpr size_t BLOCK_SIZE = 32;

    int scalar_avx2_mismatches = 0;
    int scalar_avx512_mismatches = 0;

    for (int sample = 0; sample < NUM_SAMPLES; ++sample)
    {
        float fp32_data[BLOCK_SIZE];
        generate_random_fp32(fp32_data, BLOCK_SIZE, -50.0f, 50.0f);

        Q8_0Block scalar_result;
        decode_fp32_to_q8_0_scalar(fp32_data, scalar_result.qs, &scalar_result.d);

#ifdef __AVX2__
        Q8_0Block avx2_result;
        decode_fp32_to_q8_0_avx2(fp32_data, avx2_result.qs, &avx2_result.d);
        auto cmp_avx2 = compare_q8_0_blocks(scalar_result, avx2_result, 1);
        if (!cmp_avx2.passed)
            scalar_avx2_mismatches++;
#endif

#ifdef __AVX512F__
        Q8_0Block avx512_result;
        decode_fp32_to_q8_0_avx512(fp32_data, avx512_result.qs, &avx512_result.d);
        auto cmp_avx512 = compare_q8_0_blocks(scalar_result, avx512_result, 1);
        if (!cmp_avx512.passed)
            scalar_avx512_mismatches++;
#endif
    }

    // Allow at most 1% mismatch rate (quantization rounding differences)
    EXPECT_LE(scalar_avx2_mismatches, NUM_SAMPLES / 100)
        << "Too many AVX2 mismatches: " << scalar_avx2_mismatches << "/" << NUM_SAMPLES;
    EXPECT_LE(scalar_avx512_mismatches, NUM_SAMPLES / 100)
        << "Too many AVX512 mismatches: " << scalar_avx512_mismatches << "/" << NUM_SAMPLES;
}

// ============================================================================
// FP16 → Q8_0 SIMD Accuracy Tests
// ============================================================================

TEST_F(Q8_0_SIMD_Accuracy_Test, FP16_Scalar_vs_AVX2)
{
#ifdef __AVX2__
    constexpr size_t BLOCK_SIZE = 32;
    uint16_t fp16_data[BLOCK_SIZE];
    generate_random_fp16(fp16_data, BLOCK_SIZE, -10.0f, 10.0f);

    // Scalar implementation
    Q8_0Block scalar_result;
    decode_fp16_to_q8_0_scalar(fp16_data, scalar_result.qs, &scalar_result.d);

    // AVX2 implementation
    Q8_0Block avx2_result;
    decode_fp16_to_q8_0_avx2(fp16_data, avx2_result.qs, &avx2_result.d);

    // Compare results (allow small tolerance due to FP16→FP32 conversion)
    auto cmp = compare_q8_0_blocks(scalar_result, avx2_result, 1);
    EXPECT_TRUE(cmp.passed) << cmp.error_msg;
    EXPECT_LE(cmp.max_diff, 1) << "Scalar and AVX2 should produce nearly identical results";
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, FP16_Scalar_vs_AVX512)
{
#if defined(__AVX512F__) && defined(__AVX512FP16__)
    constexpr size_t BLOCK_SIZE = 32;
    uint16_t fp16_data[BLOCK_SIZE];
    generate_random_fp16(fp16_data, BLOCK_SIZE, -10.0f, 10.0f);

    // Scalar implementation
    Q8_0Block scalar_result;
    decode_fp16_to_q8_0_scalar(fp16_data, scalar_result.qs, &scalar_result.d);

    // AVX512 implementation (with FP16 support)
    Q8_0Block avx512_result;
    decode_fp16_to_q8_0_avx512(fp16_data, avx512_result.qs, &avx512_result.d);

    // Compare results
    auto cmp = compare_q8_0_blocks(scalar_result, avx512_result, 1);
    EXPECT_TRUE(cmp.passed) << cmp.error_msg;
    EXPECT_LE(cmp.max_diff, 1) << "Scalar and AVX512 should produce nearly identical results";
#else
    GTEST_SKIP() << "AVX512FP16 not available";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, FP16_RandomData_MultipleSamples)
{
    constexpr int NUM_SAMPLES = 100;
    constexpr size_t BLOCK_SIZE = 32;

    int scalar_avx2_mismatches = 0;
    int scalar_avx512_mismatches = 0;

    for (int sample = 0; sample < NUM_SAMPLES; ++sample)
    {
        uint16_t fp16_data[BLOCK_SIZE];
        generate_random_fp16(fp16_data, BLOCK_SIZE, -20.0f, 20.0f);

        Q8_0Block scalar_result;
        decode_fp16_to_q8_0_scalar(fp16_data, scalar_result.qs, &scalar_result.d);

#ifdef __AVX2__
        Q8_0Block avx2_result;
        decode_fp16_to_q8_0_avx2(fp16_data, avx2_result.qs, &avx2_result.d);
        auto cmp_avx2 = compare_q8_0_blocks(scalar_result, avx2_result, 1);
        if (!cmp_avx2.passed)
            scalar_avx2_mismatches++;
#endif

#if defined(__AVX512F__) && defined(__AVX512FP16__)
        Q8_0Block avx512_result;
        decode_fp16_to_q8_0_avx512(fp16_data, avx512_result.qs, &avx512_result.d);
        auto cmp_avx512 = compare_q8_0_blocks(scalar_result, avx512_result, 1);
        if (!cmp_avx512.passed)
            scalar_avx512_mismatches++;
#endif
    }

    EXPECT_LE(scalar_avx2_mismatches, NUM_SAMPLES / 100);
    EXPECT_LE(scalar_avx512_mismatches, NUM_SAMPLES / 100);
}

// ============================================================================
// BF16 → Q8_0 SIMD Accuracy Tests
// ============================================================================

TEST_F(Q8_0_SIMD_Accuracy_Test, BF16_Scalar_vs_AVX2)
{
#ifdef __AVX2__
    constexpr size_t BLOCK_SIZE = 32;
    uint16_t bf16_data[BLOCK_SIZE];
    generate_random_bf16(bf16_data, BLOCK_SIZE, -10.0f, 10.0f);

    // Scalar implementation
    Q8_0Block scalar_result;
    decode_bf16_to_q8_0_scalar(bf16_data, scalar_result.qs, &scalar_result.d);

    // AVX2 implementation
    Q8_0Block avx2_result;
    decode_bf16_to_q8_0_avx2(bf16_data, avx2_result.qs, &avx2_result.d);

    // Compare results
    auto cmp = compare_q8_0_blocks(scalar_result, avx2_result, 1);
    EXPECT_TRUE(cmp.passed) << cmp.error_msg;
    EXPECT_LE(cmp.max_diff, 1) << "Scalar and AVX2 should produce nearly identical results";
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, BF16_Scalar_vs_AVX512)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    constexpr size_t BLOCK_SIZE = 32;
    uint16_t bf16_data[BLOCK_SIZE];
    generate_random_bf16(bf16_data, BLOCK_SIZE, -10.0f, 10.0f);

    // Scalar implementation
    Q8_0Block scalar_result;
    decode_bf16_to_q8_0_scalar(bf16_data, scalar_result.qs, &scalar_result.d);

    // AVX512 implementation
    Q8_0Block avx512_result;
    decode_bf16_to_q8_0_avx512(bf16_data, avx512_result.qs, &avx512_result.d);

    // Compare results
    auto cmp = compare_q8_0_blocks(scalar_result, avx512_result, 1);
    EXPECT_TRUE(cmp.passed) << cmp.error_msg;
    EXPECT_LE(cmp.max_diff, 1) << "Scalar and AVX512 should produce nearly identical results";
#else
    GTEST_SKIP() << "AVX512BW not available";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, BF16_RandomData_MultipleSamples)
{
    constexpr int NUM_SAMPLES = 100;
    constexpr size_t BLOCK_SIZE = 32;

    int scalar_avx2_mismatches = 0;
    int scalar_avx512_mismatches = 0;

    for (int sample = 0; sample < NUM_SAMPLES; ++sample)
    {
        uint16_t bf16_data[BLOCK_SIZE];
        generate_random_bf16(bf16_data, BLOCK_SIZE, -30.0f, 30.0f);

        Q8_0Block scalar_result;
        decode_bf16_to_q8_0_scalar(bf16_data, scalar_result.qs, &scalar_result.d);

#ifdef __AVX2__
        Q8_0Block avx2_result;
        decode_bf16_to_q8_0_avx2(bf16_data, avx2_result.qs, &avx2_result.d);
        auto cmp_avx2 = compare_q8_0_blocks(scalar_result, avx2_result, 1);
        if (!cmp_avx2.passed)
            scalar_avx2_mismatches++;
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
        Q8_0Block avx512_result;
        decode_bf16_to_q8_0_avx512(bf16_data, avx512_result.qs, &avx512_result.d);
        auto cmp_avx512 = compare_q8_0_blocks(scalar_result, avx512_result, 1);
        if (!cmp_avx512.passed)
            scalar_avx512_mismatches++;
#endif
    }

    EXPECT_LE(scalar_avx2_mismatches, NUM_SAMPLES / 100);
    EXPECT_LE(scalar_avx512_mismatches, NUM_SAMPLES / 100);
}

TEST_F(Q8_0_SIMD_Accuracy_Test, BF16_EdgeCases_Denormals)
{
    constexpr size_t BLOCK_SIZE = 32;
    uint16_t bf16_data[BLOCK_SIZE];

    // Create denormal BF16 values (small exponent)
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        bf16_data[i] = fp32_to_bf16(1e-6f * (i + 1)); // Very small values
    }

    Q8_0Block scalar_result;
    decode_bf16_to_q8_0_scalar(bf16_data, scalar_result.qs, &scalar_result.d);

#ifdef __AVX2__
    Q8_0Block avx2_result;
    decode_bf16_to_q8_0_avx2(bf16_data, avx2_result.qs, &avx2_result.d);
    auto cmp = compare_q8_0_blocks(scalar_result, avx2_result, 1);
    EXPECT_TRUE(cmp.passed) << "BF16 denormals: Scalar vs AVX2";
#endif
}

// ============================================================================
// Q8_K → Q8_0 SIMD Accuracy Tests
// ============================================================================

TEST_F(Q8_0_SIMD_Accuracy_Test, Q8_K_Scalar_vs_AVX2)
{
#ifdef __AVX2__
    // Create a Q8_K block with random data
    Q8_KBlock q8k_block;

    // Initialize with random quantized data (256 int8 values)
    for (int i = 0; i < Q8_KBlock::BLOCK_SIZE; ++i)
    {
        q8k_block.qs[i] = static_cast<int8_t>((rng_() % 255) - 127);
    }

    // Initialize block sums (16 values)
    for (int i = 0; i < 16; ++i)
    {
        q8k_block.bsums[i] = static_cast<int16_t>((rng_() % 1000) - 500);
    }

    // Test each of the 8 sub-blocks (256 elements / 32 per block = 8 sub-blocks)
    for (size_t subblock_idx = 0; subblock_idx < 8; ++subblock_idx)
    {
        // Scalar implementation
        Q8_0Block scalar_result;
        decode_q8_k_to_q8_0_scalar(q8k_block, subblock_idx, scalar_result.qs, &scalar_result.d);

        // AVX2 implementation
        Q8_0Block avx2_result;
        decode_q8_k_to_q8_0_avx2(q8k_block, subblock_idx, avx2_result.qs, &avx2_result.d);

        // Compare results (should be identical - just extraction)
        auto cmp = compare_q8_0_blocks(scalar_result, avx2_result, 0);
        EXPECT_TRUE(cmp.passed) << "Sub-block " << subblock_idx << ": " << cmp.error_msg;
        EXPECT_EQ(cmp.max_diff, 0) << "Sub-block " << subblock_idx << ": should be identical";
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, Q8_K_Scalar_vs_AVX512)
{
#ifdef __AVX512F__
    // Create a Q8_K block with random data
    Q8_KBlock q8k_block;

    // Initialize with random quantized data (256 int8 values)
    for (int i = 0; i < Q8_KBlock::BLOCK_SIZE; ++i)
    {
        q8k_block.qs[i] = static_cast<int8_t>((rng_() % 255) - 127);
    }

    // Initialize block sums
    for (int i = 0; i < 16; ++i)
    {
        q8k_block.bsums[i] = static_cast<int16_t>((rng_() % 1000) - 500);
    }

    // Test each of the 8 sub-blocks
    for (size_t subblock_idx = 0; subblock_idx < 8; ++subblock_idx)
    {
        // Scalar implementation
        Q8_0Block scalar_result;
        decode_q8_k_to_q8_0_scalar(q8k_block, subblock_idx, scalar_result.qs, &scalar_result.d);

        // AVX512 implementation
        Q8_0Block avx512_result;
        decode_q8_k_to_q8_0_avx512(q8k_block, subblock_idx, avx512_result.qs, &avx512_result.d);

        // Compare results
        auto cmp = compare_q8_0_blocks(scalar_result, avx512_result, 0);
        EXPECT_TRUE(cmp.passed) << "Sub-block " << subblock_idx << ": " << cmp.error_msg;
        EXPECT_EQ(cmp.max_diff, 0) << "Sub-block " << subblock_idx << ": should be identical";
    }
#else
    GTEST_SKIP() << "AVX512 not available";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, Q8_K_MultipleSuperblocks)
{
    // Test multiple Q8_K blocks to ensure robustness
    constexpr int NUM_BLOCKS = 20;

    int scalar_avx2_mismatches = 0;
    int scalar_avx512_mismatches = 0;

    for (int block_idx = 0; block_idx < NUM_BLOCKS; ++block_idx)
    {
        Q8_KBlock q8k_block;

        // Random quantized values (256 int8 values)
        for (int i = 0; i < Q8_KBlock::BLOCK_SIZE; ++i)
        {
            q8k_block.qs[i] = static_cast<int8_t>((rng_() % 255) - 127);
        }

        // Random block sums
        for (int i = 0; i < 16; ++i)
        {
            q8k_block.bsums[i] = static_cast<int16_t>((rng_() % 2000) - 1000);
        }

        // Test all 8 sub-blocks
        for (size_t subblock_idx = 0; subblock_idx < 8; ++subblock_idx)
        {
            Q8_0Block scalar_result;
            decode_q8_k_to_q8_0_scalar(q8k_block, subblock_idx, scalar_result.qs, &scalar_result.d);

#ifdef __AVX2__
            Q8_0Block avx2_result;
            decode_q8_k_to_q8_0_avx2(q8k_block, subblock_idx, avx2_result.qs, &avx2_result.d);
            auto cmp_avx2 = compare_q8_0_blocks(scalar_result, avx2_result, 0);
            if (!cmp_avx2.passed)
                scalar_avx2_mismatches++;
#endif

#ifdef __AVX512F__
            Q8_0Block avx512_result;
            decode_q8_k_to_q8_0_avx512(q8k_block, subblock_idx, avx512_result.qs, &avx512_result.d);
            auto cmp_avx512 = compare_q8_0_blocks(scalar_result, avx512_result, 0);
            if (!cmp_avx512.passed)
                scalar_avx512_mismatches++;
#endif
        }
    }

    // Q8_K extraction should be bit-exact (no quantization rounding)
    EXPECT_EQ(scalar_avx2_mismatches, 0) << "Q8_K extraction should be exact";
    EXPECT_EQ(scalar_avx512_mismatches, 0) << "Q8_K extraction should be exact";
}

// ============================================================================
// ISA Parity Tests — detail:: Q8_0 find_max_abs / requantize
// ============================================================================

#include "tensors/Q8_0Tensor_detail.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2::detail;

TEST_F(Q8_0_SIMD_Accuracy_Test, ISAParity_FindMaxAbs)
{
    const size_t blocks_per_row = 8;
    std::vector<Q8_0Block> blocks(blocks_per_row);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    for (auto &blk : blocks)
    {
        float max_val = 0.0f;
        std::array<float, 32> vals;
        for (int i = 0; i < 32; ++i)
        {
            vals[i] = dist(rng_);
            max_val = std::max(max_val, std::abs(vals[i]));
        }
        float scale = max_val / 127.0f;
        if (scale < 1e-10f)
            scale = 1e-10f;
        blk.d = llaminar2::fp32_to_fp16(scale);
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = static_cast<int8_t>(std::round(vals[i] / scale));
        }
    }

    float ref = q8_0_find_max_abs_scalar(blocks.data(), blocks_per_row);

#if defined(__AVX2__)
    float avx2_val = q8_0_find_max_abs_avx2(blocks.data(), blocks_per_row);
    EXPECT_NEAR(ref, avx2_val, 1e-4f) << "q8_0_find_max_abs scalar vs AVX2";
#endif
}

TEST_F(Q8_0_SIMD_Accuracy_Test, ISAParity_Requantize)
{
    const size_t blocks_per_row = 8;
    std::vector<Q8_0Block> blocks(blocks_per_row);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (auto &blk : blocks)
    {
        float max_val = 0.0f;
        std::array<float, 32> vals;
        for (int i = 0; i < 32; ++i)
        {
            vals[i] = dist(rng_);
            max_val = std::max(max_val, std::abs(vals[i]));
        }
        float scale = max_val / 127.0f;
        if (scale < 1e-10f)
            scale = 1e-10f;
        blk.d = llaminar2::fp32_to_fp16(scale);
        for (int i = 0; i < 32; ++i)
        {
            blk.qs[i] = static_cast<int8_t>(std::round(vals[i] / scale));
        }
    }

    float inv_scale = 0.5f;
    const size_t total = blocks_per_row * 32;
    std::vector<int8_t> ref(total), avx2_out(total), avx512_out(total);

    q8_0_requantize_scalar(blocks.data(), blocks_per_row, inv_scale, ref.data());

#if defined(__AVX2__)
    q8_0_requantize_avx2(blocks.data(), blocks_per_row, inv_scale, avx2_out.data());
    int max_diff = 0;
    for (size_t i = 0; i < total; ++i)
    {
        max_diff = std::max(max_diff, std::abs(static_cast<int>(ref[i]) - static_cast<int>(avx2_out[i])));
    }
    EXPECT_LE(max_diff, 1) << "q8_0_requantize scalar vs AVX2";
#endif

#if defined(__AVX512F__)
    q8_0_requantize_avx512(blocks.data(), blocks_per_row, inv_scale, avx512_out.data());
    int max_diff512 = 0;
    for (size_t i = 0; i < total; ++i)
    {
        max_diff512 = std::max(max_diff512, std::abs(static_cast<int>(ref[i]) - static_cast<int>(avx512_out[i])));
    }
    EXPECT_LE(max_diff512, 1) << "q8_0_requantize scalar vs AVX512";
#endif
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
