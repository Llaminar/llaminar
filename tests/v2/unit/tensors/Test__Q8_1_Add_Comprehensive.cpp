/**
 * @file Test__Q8_1_Add_Comprehensive.cpp
 * @brief Comprehensive unit tests for Q8_1 + Q8_1 residual addition
 * @author David Sanftenberg
 *
 * This test suite validates the q8_1_add_q8_1() function under various
 * extreme and edge case scenarios, ensuring numerical correctness
 * and bounded error behavior.
 *
 * Key scenarios tested:
 * 1. Basic correctness (same scale, different scales)
 * 2. Extreme values (saturation, near-zero)
 * 3. Catastrophic cancellation (a + (-a))
 * 4. Scale mismatch scenarios
 * 5. Boundary conditions (overflow, underflow)
 * 6. Random statistical validation
 * 7. Real-world patterns (layer 21 style)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/Tensors.h"

using namespace llaminar2;

/**
 * @class Test__Q8_1_Add_Comprehensive
 * @brief Test fixture for Q8_1 addition tests
 */
class Test__Q8_1_Add_Comprehensive : public ::testing::Test
{
protected:
    // Helper: Compute cosine similarity between two vectors
    static double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0, norm_a = 0, norm_b = 0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        if (norm_a < 1e-12 && norm_b < 1e-12)
            return 1.0; // Both zero
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0; // One zero
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
    }

    // Helper: Compute mean absolute error
    static double mean_abs_error(const float *a, const float *b, size_t n)
    {
        double sum = 0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(static_cast<double>(a[i]) - b[i]);
        }
        return sum / n;
    }

    // Helper: Compute max absolute error
    static double max_abs_error(const float *a, const float *b, size_t n)
    {
        double max_err = 0;
        for (size_t i = 0; i < n; ++i)
        {
            max_err = std::max(max_err, std::abs(static_cast<double>(a[i]) - b[i]));
        }
        return max_err;
    }

    // Helper: Quantize FP32 buffer to Q8_1 blocks
    static void quantize_to_q8_1(const float *src, Q8_1Block *dst, size_t n_elements)
    {
        simd::quantize_fp32_to_q8_1_blocks(src, dst, n_elements);
    }

    // Helper: Dequantize Q8_1 blocks to FP32 buffer
    static void dequantize_q8_1(const Q8_1Block *src, float *dst, size_t n_elements)
    {
        const size_t n_blocks = n_elements / 32;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            float scale = simd::fp16_to_fp32(src[b].d);
            for (size_t i = 0; i < 32; ++i)
            {
                dst[b * 32 + i] = src[b].qs[i] * scale;
            }
        }
    }

    // Helper: Compute expected quantization error bound
    // For Q8_1: error <= scale/2 per element
    static double expected_quant_error(float max_abs_value)
    {
        float scale = max_abs_value / 127.0f;
        return scale / 2.0; // Rounding error bound
    }
};

// ============================================================================
// SECTION 1: Basic Correctness Tests
// ============================================================================

TEST_F(Test__Q8_1_Add_Comprehensive, Basic_SameScale)
{
    // Two blocks with identical scale - simplest case
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 10.0f * (i - 16); // Range: [-160, 150]
        b[i] = 5.0f * (i - 16);  // Range: [-80, 75]
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);
    double mae = mean_abs_error(expected.data(), result.data(), 32);

    EXPECT_GE(cos, 0.9999) << "Cosine similarity too low for same-scale case";
    EXPECT_LE(mae, 5.0) << "Mean absolute error too high";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Basic_DifferentScales)
{
    // Two blocks with very different scales (10x ratio)
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 100.0f * (i - 16); // Range: [-1600, 1500], scale ~12.6
        b[i] = 10.0f * (i - 16);  // Range: [-160, 150], scale ~1.26
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);

    // With 10x scale difference, smaller values get less precision
    EXPECT_GE(cos, 0.999) << "Cosine similarity too low for different-scale case";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Basic_MultipleBlocks)
{
    // Test with multiple blocks
    const size_t n_blocks = 16;
    const size_t n_elements = n_blocks * 32;

    std::vector<float> a(n_elements), b(n_elements), expected(n_elements);
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 100.0f);

    for (size_t i = 0; i < n_elements; ++i)
    {
        a[i] = dist(rng);
        b[i] = dist(rng);
        expected[i] = a[i] + b[i];
    }

    std::vector<Q8_1Block> blocks_a(n_blocks), blocks_b(n_blocks), blocks_out(n_blocks);
    quantize_to_q8_1(a.data(), blocks_a.data(), n_elements);
    quantize_to_q8_1(b.data(), blocks_b.data(), n_elements);

    simd::q8_1_add_q8_1(blocks_a.data(), blocks_b.data(), blocks_out.data(), n_elements);

    std::vector<float> result(n_elements);
    dequantize_q8_1(blocks_out.data(), result.data(), n_elements);

    double cos = cosine_similarity(expected.data(), result.data(), n_elements);
    EXPECT_GE(cos, 0.999) << "Multi-block cosine similarity too low";
}

// ============================================================================
// SECTION 2: Extreme Value Tests
// ============================================================================

TEST_F(Test__Q8_1_Add_Comprehensive, Extreme_MaxFP16Scale)
{
    // Test values near FP16 max (~65504)
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        // Max representable in Q8_1 with FP16 scale: 127 * 65504 ≈ 8.3M
        // But we'll stay below FP16 overflow
        a[i] = 30000.0f + i * 100.0f;
        b[i] = 25000.0f + i * 50.0f;
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.99) << "Large value cosine too low";

    // Check no overflow occurred (result should be close to expected)
    double mae = mean_abs_error(expected.data(), result.data(), 32);
    EXPECT_LT(mae, 1000.0) << "Large value error too high - possible overflow";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Extreme_VerySmallValues)
{
    // Test with very small values (near FP16 subnormal range)
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 1e-5f * (i + 1); // Range: [1e-5, 3.2e-4]
        b[i] = 2e-5f * (i + 1);
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    // For very small values, relative error matters more than absolute
    double cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.95) << "Small value direction preservation failed";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Extreme_NearZero)
{
    // Test with values very close to zero (should use zero scale path)
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 1e-8f * (i - 16); // Below MIN_SCALE_THRESHOLD (1e-6)
        b[i] = 1e-8f * (16 - i);
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    // Near-zero values should quantize to zero
    double max_err = max_abs_error(expected.data(), result.data(), 32);
    EXPECT_LT(max_err, 1e-5) << "Near-zero handling incorrect";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Extreme_AllZeros)
{
    // Both inputs are all zeros
    std::vector<float> a(32, 0.0f), b(32, 0.0f), expected(32, 0.0f);

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], 0.0f) << "Zero input at index " << i;
    }
}

TEST_F(Test__Q8_1_Add_Comprehensive, Extreme_SingleNonzero)
{
    // Only one element is nonzero per block
    std::vector<float> a(32, 0.0f), b(32, 0.0f), expected(32, 0.0f);
    a[0] = 100.0f;
    b[15] = 200.0f;
    expected[0] = 100.0f;
    expected[15] = 200.0f;

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.99) << "Single nonzero handling incorrect";
}

// ============================================================================
// SECTION 3: Catastrophic Cancellation Tests
// ============================================================================

TEST_F(Test__Q8_1_Add_Comprehensive, Cancellation_ExactOpposite)
{
    // a + (-a) should be close to zero
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 100.0f + i * 10.0f;
        b[i] = -a[i]; // Exact opposite
        expected[i] = 0.0f;
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    // Due to quantization, result won't be exactly zero
    // But should be bounded by combined quantization error
    float scale_a = simd::fp16_to_fp32(blk_a.d);
    float scale_b = simd::fp16_to_fp32(blk_b.d);
    float max_quant_error = (scale_a + scale_b); // Conservative bound

    double max_err = max_abs_error(expected.data(), result.data(), 32);

    std::cout << "Exact opposite cancellation:\n";
    std::cout << "  scale_a=" << scale_a << " scale_b=" << scale_b << "\n";
    std::cout << "  max_quant_error bound=" << max_quant_error << "\n";
    std::cout << "  actual max_err=" << max_err << "\n";

    EXPECT_LT(max_err, max_quant_error * 2) << "Cancellation error exceeds expected bound";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Cancellation_NearOpposite)
{
    // a + (-a + small_residue) - tests precision loss in cancellation
    std::vector<float> a(32), b(32), expected(32);
    const float residue = 5.0f; // Small residue

    for (int i = 0; i < 32; ++i)
    {
        a[i] = 1000.0f + i * 10.0f; // Large values
        b[i] = -a[i] + residue;     // Near opposite with small residue
        expected[i] = residue;
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    // With 1000x scale difference (1000 vs 5), expect significant relative error
    double cos = cosine_similarity(expected.data(), result.data(), 32);

    std::cout << "Near-opposite cancellation (residue=" << residue << "):\n";
    std::cout << "  cosine=" << cos << "\n";
    std::cout << "  expected[0]=" << expected[0] << " result[0]=" << result[0] << "\n";

    // This is a known limitation - cosine may be poor due to quantization noise
    // But we want to ensure no catastrophic failures (e.g., wrong sign)
    float mean_result = std::accumulate(result.begin(), result.end(), 0.0f) / 32;
    EXPECT_GT(mean_result, 0.0f) << "Mean should be positive (near residue)";
    EXPECT_LT(mean_result, residue * 3) << "Mean should be near residue value";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Cancellation_LargeScaleDifference)
{
    // Large value + Small opposite value - tests dynamic range
    std::vector<float> a(32), b(32), expected(32);

    for (int i = 0; i < 32; ++i)
    {
        a[i] = 10000.0f;            // All same large value
        b[i] = -9990.0f - i * 0.1f; // Nearly canceling with small variation
        expected[i] = a[i] + b[i];  // ~10 + small variation
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    // Verify result is in reasonable range
    float min_exp = *std::min_element(expected.begin(), expected.end());
    float max_exp = *std::max_element(expected.begin(), expected.end());
    float min_res = *std::min_element(result.begin(), result.end());
    float max_res = *std::max_element(result.begin(), result.end());

    std::cout << "Large scale difference cancellation:\n";
    std::cout << "  Expected range: [" << min_exp << ", " << max_exp << "]\n";
    std::cout << "  Result range:   [" << min_res << ", " << max_res << "]\n";

    // Result should be positive (same sign as expected)
    EXPECT_GT(min_res, -100.0f) << "Result went too negative";
    EXPECT_LT(max_res, 200.0f) << "Result went too positive";
}

// ============================================================================
// SECTION 4: Scale Mismatch Scenarios
// ============================================================================

TEST_F(Test__Q8_1_Add_Comprehensive, ScaleMismatch_100x)
{
    // 100x scale difference between inputs
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 1000.0f + i * 10.0f; // Scale ~10
        b[i] = 10.0f + i * 0.1f;    // Scale ~0.1
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    float scale_a = simd::fp16_to_fp32(blk_a.d);
    float scale_b = simd::fp16_to_fp32(blk_b.d);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);

    std::cout << "100x scale mismatch:\n";
    std::cout << "  scale_a=" << scale_a << " scale_b=" << scale_b << "\n";
    std::cout << "  ratio=" << scale_a / scale_b << "\n";
    std::cout << "  cosine=" << cos << "\n";

    // With 100x scale difference, smaller values contribute negligibly
    EXPECT_GE(cos, 0.9999) << "Large inputs dominate - should have good cosine";
}

TEST_F(Test__Q8_1_Add_Comprehensive, ScaleMismatch_OppositeSign)
{
    // Different scales with opposite signs
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 500.0f + i * 5.0f; // Positive, scale ~5
        b[i] = -50.0f - i * 0.5f; // Negative, scale ~0.5
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.999) << "Opposite sign scale mismatch failed";
}

// ============================================================================
// SECTION 5: Boundary Conditions
// ============================================================================

TEST_F(Test__Q8_1_Add_Comprehensive, Boundary_Saturation)
{
    // Test saturation behavior (both inputs at max representable)
    std::vector<float> a(32), b(32), expected(32);
    const float max_val = 127.0f * 500.0f; // Near max with reasonable scale

    for (int i = 0; i < 32; ++i)
    {
        a[i] = max_val;
        b[i] = max_val * 0.5f;
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    // Verify no overflow/underflow
    EXPECT_GT(result[0], 0) << "Result should be positive";

    double cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.99) << "Saturation case cosine too low";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Boundary_MixedSigns)
{
    // Half positive, half negative with large magnitudes
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = (i < 16) ? 1000.0f : -1000.0f;
        b[i] = (i < 16) ? -500.0f : 500.0f;
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.999) << "Mixed signs case failed";
}

// ============================================================================
// SECTION 6: Statistical Validation
// ============================================================================

TEST_F(Test__Q8_1_Add_Comprehensive, Statistical_GaussianInputs)
{
    // Large-scale test with Gaussian inputs
    const size_t n_elements = 1024 * 32; // 1024 blocks
    const size_t n_blocks = n_elements / 32;

    std::vector<float> a(n_elements), b(n_elements), expected(n_elements);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 100.0f);

    for (size_t i = 0; i < n_elements; ++i)
    {
        a[i] = dist(rng);
        b[i] = dist(rng);
        expected[i] = a[i] + b[i];
    }

    std::vector<Q8_1Block> blocks_a(n_blocks), blocks_b(n_blocks), blocks_out(n_blocks);
    quantize_to_q8_1(a.data(), blocks_a.data(), n_elements);
    quantize_to_q8_1(b.data(), blocks_b.data(), n_elements);

    simd::q8_1_add_q8_1(blocks_a.data(), blocks_b.data(), blocks_out.data(), n_elements);

    std::vector<float> result(n_elements);
    dequantize_q8_1(blocks_out.data(), result.data(), n_elements);

    double cos = cosine_similarity(expected.data(), result.data(), n_elements);
    double mae = mean_abs_error(expected.data(), result.data(), n_elements);
    double max_err = max_abs_error(expected.data(), result.data(), n_elements);

    std::cout << "Gaussian inputs (N=" << n_elements << "):\n";
    std::cout << "  cosine=" << cos << "\n";
    std::cout << "  MAE=" << mae << "\n";
    std::cout << "  max_err=" << max_err << "\n";

    EXPECT_GE(cos, 0.9999) << "Gaussian input cosine too low";
    EXPECT_LT(mae, 5.0) << "Gaussian input MAE too high";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Statistical_UniformInputs)
{
    // Uniform distribution over full range
    const size_t n_elements = 512 * 32;
    const size_t n_blocks = n_elements / 32;

    std::vector<float> a(n_elements), b(n_elements), expected(n_elements);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (size_t i = 0; i < n_elements; ++i)
    {
        a[i] = dist(rng);
        b[i] = dist(rng);
        expected[i] = a[i] + b[i];
    }

    std::vector<Q8_1Block> blocks_a(n_blocks), blocks_b(n_blocks), blocks_out(n_blocks);
    quantize_to_q8_1(a.data(), blocks_a.data(), n_elements);
    quantize_to_q8_1(b.data(), blocks_b.data(), n_elements);

    simd::q8_1_add_q8_1(blocks_a.data(), blocks_b.data(), blocks_out.data(), n_elements);

    std::vector<float> result(n_elements);
    dequantize_q8_1(blocks_out.data(), result.data(), n_elements);

    double cos = cosine_similarity(expected.data(), result.data(), n_elements);

    EXPECT_GE(cos, 0.9999) << "Uniform input cosine too low";
}

// ============================================================================
// SECTION 7: Real-World Patterns
// ============================================================================

TEST_F(Test__Q8_1_Add_Comprehensive, RealWorld_Layer21Pattern)
{
    // Simulates the layer 21 FFN residual add pattern
    // Characterized by:
    // - Attention residual: mean~0.3, std~17, occasional large spikes
    // - FFN down: mean~-0.2, std~16, tends to cancel attention residual
    // - Result: near-zero mean with small std (~2.4)

    const size_t n_elements = 9 * 896; // Typical layer 21 shape
    const size_t n_blocks = (n_elements + 31) / 32;

    std::vector<float> attn_res(n_elements), ffn_down(n_elements), expected(n_elements);
    std::mt19937 rng(42);
    std::normal_distribution<float> attn_dist(0.3f, 17.0f);
    std::normal_distribution<float> noise_dist(0.0f, 2.4f);

    for (size_t i = 0; i < n_elements; ++i)
    {
        attn_res[i] = attn_dist(rng);
        // FFN down mostly cancels attention residual, leaving small residue
        ffn_down[i] = -attn_res[i] + noise_dist(rng);
        expected[i] = attn_res[i] + ffn_down[i]; // ≈ noise
    }

    std::vector<Q8_1Block> blocks_a(n_blocks), blocks_b(n_blocks), blocks_out(n_blocks);
    quantize_to_q8_1(attn_res.data(), blocks_a.data(), n_elements);
    quantize_to_q8_1(ffn_down.data(), blocks_b.data(), n_elements);

    simd::q8_1_add_q8_1(blocks_a.data(), blocks_b.data(), blocks_out.data(), n_elements);

    std::vector<float> result(n_elements);
    dequantize_q8_1(blocks_out.data(), result.data(), n_elements);

    double cos = cosine_similarity(expected.data(), result.data(), n_elements);
    double mae = mean_abs_error(expected.data(), result.data(), n_elements);

    std::cout << "Layer 21 pattern simulation:\n";
    std::cout << "  cosine=" << cos << "\n";
    std::cout << "  MAE=" << mae << "\n";

    // Due to cancellation, cosine will be lower (~0.9)
    EXPECT_GE(cos, 0.85) << "Layer 21 pattern cosine too low";

    // But mean should be close to zero (cancellation works)
    float mean_exp = std::accumulate(expected.begin(), expected.end(), 0.0f) / n_elements;
    float mean_res = std::accumulate(result.begin(), result.end(), 0.0f) / n_elements;
    EXPECT_NEAR(mean_res, mean_exp, 5.0f) << "Mean preservation failed";
}

TEST_F(Test__Q8_1_Add_Comprehensive, RealWorld_LargeActivation)
{
    // Simulates large activation + small correction pattern
    // Common in residual connections where main signal is large

    const size_t n_elements = 32 * 100;
    const size_t n_blocks = n_elements / 32;

    std::vector<float> main_signal(n_elements), correction(n_elements), expected(n_elements);
    std::mt19937 rng(999);
    std::normal_distribution<float> main_dist(500.0f, 100.0f);
    std::normal_distribution<float> corr_dist(0.0f, 10.0f);

    for (size_t i = 0; i < n_elements; ++i)
    {
        main_signal[i] = main_dist(rng);
        correction[i] = corr_dist(rng);
        expected[i] = main_signal[i] + correction[i];
    }

    std::vector<Q8_1Block> blocks_a(n_blocks), blocks_b(n_blocks), blocks_out(n_blocks);
    quantize_to_q8_1(main_signal.data(), blocks_a.data(), n_elements);
    quantize_to_q8_1(correction.data(), blocks_b.data(), n_elements);

    simd::q8_1_add_q8_1(blocks_a.data(), blocks_b.data(), blocks_out.data(), n_elements);

    std::vector<float> result(n_elements);
    dequantize_q8_1(blocks_out.data(), result.data(), n_elements);

    double cos = cosine_similarity(expected.data(), result.data(), n_elements);

    // Main signal dominates, so should have excellent cosine
    EXPECT_GE(cos, 0.9999) << "Large activation + small correction failed";
}

// ============================================================================
// SECTION 8: Edge Cases
// ============================================================================

TEST_F(Test__Q8_1_Add_Comprehensive, Edge_LargeButFiniteValues)
{
    // Test with large but realistic activation magnitudes
    // (Real inference activations are typically < 1000 in magnitude)

    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = 50000.0f + static_cast<float>(i) * 100.0f; // Large positive
        b[i] = 50000.0f + static_cast<float>(i) * 100.0f;
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    // Verify no NaN/Inf in output
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_FALSE(std::isnan(result[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(result[i])) << "Inf at index " << i;
    }

    // Check reasonable accuracy
    float cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.99) << "Large values should be handled correctly";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Edge_Alternating_Signs)
{
    // Alternating positive/negative with same magnitude
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = (i % 2 == 0) ? 100.0f : -100.0f;
        b[i] = (i % 2 == 0) ? 50.0f : -50.0f;
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.9999) << "Alternating signs failed";
}

TEST_F(Test__Q8_1_Add_Comprehensive, Edge_GradualRamp)
{
    // Linear ramp - tests full dynamic range within block
    std::vector<float> a(32), b(32), expected(32);
    for (int i = 0; i < 32; ++i)
    {
        a[i] = -500.0f + i * 32.0f; // -500 to 492
        b[i] = 250.0f - i * 16.0f;  // 250 to -246
        expected[i] = a[i] + b[i];
    }

    Q8_1Block blk_a, blk_b, blk_out;
    quantize_to_q8_1(a.data(), &blk_a, 32);
    quantize_to_q8_1(b.data(), &blk_b, 32);

    simd::q8_1_add_q8_1(&blk_a, &blk_b, &blk_out, 32);

    std::vector<float> result(32);
    dequantize_q8_1(&blk_out, result.data(), 32);

    double cos = cosine_similarity(expected.data(), result.data(), 32);
    EXPECT_GE(cos, 0.9999) << "Gradual ramp failed";
}
