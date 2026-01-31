/**
 * @file Test__Q8_1_QuantizeSIMD.cpp
 * @brief Unit tests for Q8_1 quantization SIMD implementations
 *
 * Validates that AVX512, AVX2, and scalar implementations of FP32→Q8_1
 * quantization produce identical results. Tests the quantize_single_block_*
 * functions added to SIMDHelpers.h.
 *
 * Tests cover:
 * - Scalar, AVX2, AVX512 agreement for full 32-element blocks
 * - Partial block handling (boundary cases)
 * - Edge cases (zeros, near-zero, extreme values, uniform data)
 * - Dequantization round-trip accuracy
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "tensors/Tensors.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <sstream>

using namespace llaminar2;
using namespace llaminar2::simd;

// ============================================================================
// Test Fixture
// ============================================================================

class Q8_1_QuantizeSIMD_Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rng_.seed(42);
    }

    std::mt19937 rng_;

    // ========================================================================
    // Helper: Compare Q8_1 blocks with detailed results
    // ========================================================================
    struct ComparisonResult
    {
        bool passed;
        int max_qs_diff;       ///< Maximum difference in quantized values
        int num_qs_mismatches; ///< Number of mismatched quantized values
        int sum_diff;          ///< Difference in sum_qs field
        float scale_diff;      ///< Difference in scale (d field)
        std::string error_msg;

        std::string to_string() const
        {
            std::ostringstream oss;
            oss << "passed=" << (passed ? "true" : "false")
                << ", max_qs_diff=" << max_qs_diff
                << ", num_qs_mismatches=" << num_qs_mismatches
                << ", sum_diff=" << sum_diff
                << ", scale_diff=" << scale_diff;
            if (!error_msg.empty())
            {
                oss << ", error=" << error_msg;
            }
            return oss.str();
        }
    };

    /**
     * @brief Compare two Q8_1 blocks for equality
     *
     * Q8_1 format:
     *   - d: FP16 scale factor
     *   - sum_qs: INT16 sum of quantized values
     *   - qs[32]: INT8 quantized values
     *
     * @param a First block
     * @param b Second block
     * @param qs_tolerance Allowed difference in quantized int8 values (typically 0 or 1)
     * @param sum_tolerance Allowed difference in sum_qs (typically 0 or small)
     * @return ComparisonResult with detailed comparison info
     */
    ComparisonResult compare_q8_1_blocks(
        const Q8_1Block &a,
        const Q8_1Block &b,
        int qs_tolerance = 0,
        int sum_tolerance = 0)
    {
        ComparisonResult result{true, 0, 0, 0, 0.0f, ""};

        // Compare scales (FP16 stored as uint16_t)
        float scale_a = fp16_to_fp32(a.d);
        float scale_b = fp16_to_fp32(b.d);
        result.scale_diff = std::fabs(scale_a - scale_b);

        // Scales must be exactly equal (same computation path)
        if (a.d != b.d)
        {
            result.passed = false;
            result.error_msg = "Scale mismatch: " + std::to_string(scale_a) +
                               " vs " + std::to_string(scale_b);
        }

        // Compare quantized values
        for (int i = 0; i < 32; ++i)
        {
            int diff = std::abs(static_cast<int>(a.qs[i]) - static_cast<int>(b.qs[i]));
            result.max_qs_diff = std::max(result.max_qs_diff, diff);
            if (diff > qs_tolerance)
            {
                result.num_qs_mismatches++;
                result.passed = false;
                if (result.error_msg.empty())
                {
                    result.error_msg = "qs mismatch at index " + std::to_string(i) +
                                       ": " + std::to_string(static_cast<int>(a.qs[i])) +
                                       " vs " + std::to_string(static_cast<int>(b.qs[i]));
                }
            }
        }

        // Compare sum_qs
        result.sum_diff = std::abs(static_cast<int>(a.sum_qs) - static_cast<int>(b.sum_qs));
        if (result.sum_diff > sum_tolerance)
        {
            result.passed = false;
            if (result.error_msg.empty())
            {
                result.error_msg = "sum_qs mismatch: " + std::to_string(a.sum_qs) +
                                   " vs " + std::to_string(b.sum_qs);
            }
        }

        return result;
    }

    // ========================================================================
    // Helper: Generate random FP32 data
    // ========================================================================
    void generate_random_fp32(float *data, size_t count, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // ========================================================================
    // Helper: Generate specific test patterns
    // ========================================================================
    void generate_zeros(float *data, size_t count)
    {
        std::memset(data, 0, count * sizeof(float));
    }

    void generate_constant(float *data, size_t count, float value)
    {
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = value;
        }
    }

    void generate_alternating(float *data, size_t count, float a, float b)
    {
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = (i % 2 == 0) ? a : b;
        }
    }

    void generate_ramp(float *data, size_t count, float start, float step)
    {
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = start + i * step;
        }
    }

    // ========================================================================
    // Helper: Dequantize Q8_1 block to FP32 for round-trip testing
    // ========================================================================
    void dequantize_q8_1_block(const Q8_1Block &block, float *output)
    {
        float scale = fp16_to_fp32(block.d);
        for (int i = 0; i < 32; ++i)
        {
            output[i] = static_cast<float>(block.qs[i]) * scale;
        }
    }

    // ========================================================================
    // Helper: Compute round-trip error
    // ========================================================================
    float compute_max_round_trip_error(const float *original, const float *reconstructed, size_t count)
    {
        float max_error = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float error = std::fabs(original[i] - reconstructed[i]);
            max_error = std::max(max_error, error);
        }
        return max_error;
    }

    float compute_rms_round_trip_error(const float *original, const float *reconstructed, size_t count)
    {
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double error = original[i] - reconstructed[i];
            sum_sq += error * error;
        }
        return static_cast<float>(std::sqrt(sum_sq / count));
    }
};

// ============================================================================
// Test: All implementations agree on random data
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, AllImplementationsAgree_RandomData)
{
    constexpr int NUM_ITERATIONS = 1000;
    constexpr int BLOCK_SIZE = 32;

    // Allow ±1 difference due to FP rounding differences between
    // scalar std::round() and SIMD vroundps/roundscale_ps
    // This is expected and acceptable for ML workloads
    constexpr int QS_TOLERANCE = 1;
    constexpr int SUM_TOLERANCE = 32; // At most 32 elements could differ by 1

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        // Generate random input
        alignas(64) float input[BLOCK_SIZE];
        generate_random_fp32(input, BLOCK_SIZE, -10.0f, 10.0f);

        // Quantize with each implementation
        Q8_1Block scalar_result, avx2_result, avx512_result;
        std::memset(&scalar_result, 0, sizeof(Q8_1Block));
        std::memset(&avx2_result, 0, sizeof(Q8_1Block));
        std::memset(&avx512_result, 0, sizeof(Q8_1Block));

        // Always run scalar (baseline)
        quantize_single_block_scalar(input, scalar_result, BLOCK_SIZE);

#if defined(__AVX2__)
        quantize_single_block_avx2(input, avx2_result);

        // Compare scalar vs AVX2
        auto cmp_avx2 = compare_q8_1_blocks(scalar_result, avx2_result, QS_TOLERANCE, SUM_TOLERANCE);
        EXPECT_TRUE(cmp_avx2.passed)
            << "Iteration " << iter << " (scalar vs AVX2): " << cmp_avx2.to_string();
#endif

#if defined(__AVX512F__)
        quantize_single_block_avx512(input, avx512_result);

        // Compare scalar vs AVX512
        auto cmp_avx512 = compare_q8_1_blocks(scalar_result, avx512_result, QS_TOLERANCE, SUM_TOLERANCE);
        EXPECT_TRUE(cmp_avx512.passed)
            << "Iteration " << iter << " (scalar vs AVX512): " << cmp_avx512.to_string();
#endif

#if defined(__AVX2__) && defined(__AVX512F__)
        // Compare AVX2 vs AVX512 (should be identical since both use vroundps)
        auto cmp_avx2_512 = compare_q8_1_blocks(avx2_result, avx512_result);
        EXPECT_TRUE(cmp_avx2_512.passed)
            << "Iteration " << iter << " (AVX2 vs AVX512): " << cmp_avx2_512.to_string();
#endif
    }
}

// ============================================================================
// Test: All implementations agree on uniform data
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, AllImplementationsAgree_UniformData)
{
    constexpr int BLOCK_SIZE = 32;
    const std::vector<float> test_values = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 127.0f, -127.0f, 0.001f, -0.001f};

    for (float val : test_values)
    {
        alignas(64) float input[BLOCK_SIZE];
        generate_constant(input, BLOCK_SIZE, val);

        Q8_1Block scalar_result, avx2_result, avx512_result;
        std::memset(&scalar_result, 0, sizeof(Q8_1Block));
        std::memset(&avx2_result, 0, sizeof(Q8_1Block));
        std::memset(&avx512_result, 0, sizeof(Q8_1Block));

        quantize_single_block_scalar(input, scalar_result, BLOCK_SIZE);

#if defined(__AVX2__)
        quantize_single_block_avx2(input, avx2_result);
        auto cmp_avx2 = compare_q8_1_blocks(scalar_result, avx2_result);
        EXPECT_TRUE(cmp_avx2.passed)
            << "Uniform value " << val << " (scalar vs AVX2): " << cmp_avx2.to_string();
#endif

#if defined(__AVX512F__)
        quantize_single_block_avx512(input, avx512_result);
        auto cmp_avx512 = compare_q8_1_blocks(scalar_result, avx512_result);
        EXPECT_TRUE(cmp_avx512.passed)
            << "Uniform value " << val << " (scalar vs AVX512): " << cmp_avx512.to_string();
#endif
    }
}

// ============================================================================
// Test: All implementations agree on alternating patterns
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, AllImplementationsAgree_AlternatingPatterns)
{
    constexpr int BLOCK_SIZE = 32;
    const std::vector<std::pair<float, float>> test_patterns = {
        {1.0f, -1.0f},
        {0.5f, -0.5f},
        {100.0f, -100.0f},
        {0.0f, 1.0f},
        {127.0f, 0.0f}};

    for (const auto &pattern : test_patterns)
    {
        alignas(64) float input[BLOCK_SIZE];
        generate_alternating(input, BLOCK_SIZE, pattern.first, pattern.second);

        Q8_1Block scalar_result, avx2_result, avx512_result;
        std::memset(&scalar_result, 0, sizeof(Q8_1Block));
        std::memset(&avx2_result, 0, sizeof(Q8_1Block));
        std::memset(&avx512_result, 0, sizeof(Q8_1Block));

        quantize_single_block_scalar(input, scalar_result, BLOCK_SIZE);

#if defined(__AVX2__)
        quantize_single_block_avx2(input, avx2_result);
        auto cmp_avx2 = compare_q8_1_blocks(scalar_result, avx2_result);
        EXPECT_TRUE(cmp_avx2.passed)
            << "Alternating (" << pattern.first << ", " << pattern.second
            << ") (scalar vs AVX2): " << cmp_avx2.to_string();
#endif

#if defined(__AVX512F__)
        quantize_single_block_avx512(input, avx512_result);
        auto cmp_avx512 = compare_q8_1_blocks(scalar_result, avx512_result);
        EXPECT_TRUE(cmp_avx512.passed)
            << "Alternating (" << pattern.first << ", " << pattern.second
            << ") (scalar vs AVX512): " << cmp_avx512.to_string();
#endif
    }
}

// ============================================================================
// Test: All implementations agree on ramp data (tests all quantization levels)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, AllImplementationsAgree_RampData)
{
    constexpr int BLOCK_SIZE = 32;
    const std::vector<std::pair<float, float>> test_ramps = {
        {-1.0f, 2.0f / 31.0f},     // -1 to +1
        {0.0f, 10.0f / 31.0f},     // 0 to ~10
        {-127.0f, 254.0f / 31.0f}, // Full range
        {-0.5f, 1.0f / 31.0f}};    // Narrow range

    for (const auto &ramp : test_ramps)
    {
        alignas(64) float input[BLOCK_SIZE];
        generate_ramp(input, BLOCK_SIZE, ramp.first, ramp.second);

        Q8_1Block scalar_result, avx2_result, avx512_result;
        std::memset(&scalar_result, 0, sizeof(Q8_1Block));
        std::memset(&avx2_result, 0, sizeof(Q8_1Block));
        std::memset(&avx512_result, 0, sizeof(Q8_1Block));

        quantize_single_block_scalar(input, scalar_result, BLOCK_SIZE);

#if defined(__AVX2__)
        quantize_single_block_avx2(input, avx2_result);
        auto cmp_avx2 = compare_q8_1_blocks(scalar_result, avx2_result);
        EXPECT_TRUE(cmp_avx2.passed)
            << "Ramp (start=" << ramp.first << ", step=" << ramp.second
            << ") (scalar vs AVX2): " << cmp_avx2.to_string();
#endif

#if defined(__AVX512F__)
        quantize_single_block_avx512(input, avx512_result);
        auto cmp_avx512 = compare_q8_1_blocks(scalar_result, avx512_result);
        EXPECT_TRUE(cmp_avx512.passed)
            << "Ramp (start=" << ramp.first << ", step=" << ramp.second
            << ") (scalar vs AVX512): " << cmp_avx512.to_string();
#endif
    }
}

// ============================================================================
// Test: AVX2 and AVX512 produce IDENTICAL results (bit-exact)
// Both use the same SIMD rounding mode (vroundps), so should be identical
// ============================================================================

#if defined(__AVX2__) && defined(__AVX512F__)
TEST_F(Q8_1_QuantizeSIMD_Test, AVX2_AVX512_BitExactMatch)
{
    constexpr int BLOCK_SIZE = 32;
    constexpr int NUM_ITERATIONS = 100;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        alignas(64) float input[BLOCK_SIZE];
        generate_random_fp32(input, BLOCK_SIZE);

        Q8_1Block avx2_result, avx512_result;
        std::memset(&avx2_result, 0, sizeof(Q8_1Block));
        std::memset(&avx512_result, 0, sizeof(Q8_1Block));

        quantize_single_block_avx2(input, avx2_result);
        quantize_single_block_avx512(input, avx512_result);

        // AVX2 and AVX512 should produce BIT-EXACT identical results
        // No tolerance - they use the same algorithm and rounding mode
        auto cmp = compare_q8_1_blocks(avx2_result, avx512_result, 0 /* no qs tolerance */, 0 /* no sum tolerance */);
        EXPECT_TRUE(cmp.passed)
            << "Iteration " << iter << " (AVX2 vs AVX512 - should be bit-exact): " << cmp.to_string();
    }
}
#endif

// ============================================================================
// Test: Zero input produces zero output
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, ZeroInput_ProducesZeroOutput)
{
    constexpr int BLOCK_SIZE = 32;
    alignas(64) float input[BLOCK_SIZE];
    generate_zeros(input, BLOCK_SIZE);

    Q8_1Block result;
    quantize_single_block_scalar(input, result, BLOCK_SIZE);

    // Zero input should produce:
    // - d = 0 (scale is zero)
    // - sum_qs = 0
    // - all qs[i] = 0
    EXPECT_EQ(result.d, 0) << "Scale should be zero for zero input";
    EXPECT_EQ(result.sum_qs, 0) << "Sum should be zero for zero input";
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(result.qs[i], 0) << "qs[" << i << "] should be zero for zero input";
    }

#if defined(__AVX2__)
    Q8_1Block avx2_result;
    quantize_single_block_avx2(input, avx2_result);
    EXPECT_EQ(avx2_result.d, 0);
    EXPECT_EQ(avx2_result.sum_qs, 0);
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(avx2_result.qs[i], 0);
    }
#endif

#if defined(__AVX512F__)
    Q8_1Block avx512_result;
    quantize_single_block_avx512(input, avx512_result);
    EXPECT_EQ(avx512_result.d, 0);
    EXPECT_EQ(avx512_result.sum_qs, 0);
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(avx512_result.qs[i], 0);
    }
#endif
}

// ============================================================================
// Test: Near-zero input (below MIN_SCALE_THRESHOLD)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, NearZeroInput_BelowThreshold)
{
    constexpr int BLOCK_SIZE = 32;
    alignas(64) float input[BLOCK_SIZE];

    // Values below MIN_SCALE_THRESHOLD (1e-10f)
    generate_constant(input, BLOCK_SIZE, 1e-12f);

    Q8_1Block result;
    quantize_single_block_scalar(input, result, BLOCK_SIZE);

    EXPECT_EQ(result.d, 0) << "Scale should be zero for tiny input";
    EXPECT_EQ(result.sum_qs, 0) << "Sum should be zero for tiny input";

#if defined(__AVX2__)
    Q8_1Block avx2_result;
    quantize_single_block_avx2(input, avx2_result);
    EXPECT_EQ(avx2_result.d, 0);
#endif

#if defined(__AVX512F__)
    Q8_1Block avx512_result;
    quantize_single_block_avx512(input, avx512_result);
    EXPECT_EQ(avx512_result.d, 0);
#endif
}

// ============================================================================
// Test: Maximum value input (tests clamping to [-127, 127])
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, MaxValueInput_ProperClamping)
{
    constexpr int BLOCK_SIZE = 32;
    alignas(64) float input[BLOCK_SIZE];
    generate_constant(input, BLOCK_SIZE, 1000.0f);

    Q8_1Block scalar_result;
    quantize_single_block_scalar(input, scalar_result, BLOCK_SIZE);

    // With all values = 1000.0f, max_abs = 1000.0f
    // scale = 1000.0f / 127.0f ≈ 7.874f
    // Each quantized value = round(1000.0f / scale) = round(127) = 127
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(scalar_result.qs[i], 127) << "qs[" << i << "] should be 127 for large positive input";
    }
    EXPECT_EQ(scalar_result.sum_qs, 32 * 127) << "Sum should be 32 * 127";

#if defined(__AVX2__)
    Q8_1Block avx2_result;
    quantize_single_block_avx2(input, avx2_result);
    auto cmp_avx2 = compare_q8_1_blocks(scalar_result, avx2_result);
    EXPECT_TRUE(cmp_avx2.passed) << "Max value (AVX2): " << cmp_avx2.to_string();
#endif

#if defined(__AVX512F__)
    Q8_1Block avx512_result;
    quantize_single_block_avx512(input, avx512_result);
    auto cmp_avx512 = compare_q8_1_blocks(scalar_result, avx512_result);
    EXPECT_TRUE(cmp_avx512.passed) << "Max value (AVX512): " << cmp_avx512.to_string();
#endif
}

// ============================================================================
// Test: Minimum value input (negative clamping)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, MinValueInput_ProperClamping)
{
    constexpr int BLOCK_SIZE = 32;
    alignas(64) float input[BLOCK_SIZE];
    generate_constant(input, BLOCK_SIZE, -1000.0f);

    Q8_1Block scalar_result;
    quantize_single_block_scalar(input, scalar_result, BLOCK_SIZE);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(scalar_result.qs[i], -127) << "qs[" << i << "] should be -127 for large negative input";
    }
    EXPECT_EQ(scalar_result.sum_qs, 32 * -127) << "Sum should be 32 * -127";

#if defined(__AVX2__)
    Q8_1Block avx2_result;
    quantize_single_block_avx2(input, avx2_result);
    auto cmp_avx2 = compare_q8_1_blocks(scalar_result, avx2_result);
    EXPECT_TRUE(cmp_avx2.passed) << "Min value (AVX2): " << cmp_avx2.to_string();
#endif

#if defined(__AVX512F__)
    Q8_1Block avx512_result;
    quantize_single_block_avx512(input, avx512_result);
    auto cmp_avx512 = compare_q8_1_blocks(scalar_result, avx512_result);
    EXPECT_TRUE(cmp_avx512.passed) << "Min value (AVX512): " << cmp_avx512.to_string();
#endif
}

// ============================================================================
// Test: Partial block handling (scalar only supports this)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, PartialBlock_ScalarHandling)
{
    const std::vector<int> valid_element_counts = {1, 7, 15, 16, 17, 24, 31};

    for (int valid_elements : valid_element_counts)
    {
        alignas(64) float input[32];
        generate_random_fp32(input, 32, -5.0f, 5.0f);

        Q8_1Block result;
        std::memset(&result, 0xAA, sizeof(result)); // Fill with sentinel

        quantize_single_block_scalar(input, result, valid_elements);

        // Verify: elements beyond valid_elements should be zero
        for (int i = valid_elements; i < 32; ++i)
        {
            EXPECT_EQ(result.qs[i], 0)
                << "qs[" << i << "] should be zero-padded for valid_elements=" << valid_elements;
        }

        // Verify: sum only includes valid elements
        int32_t expected_sum = 0;
        for (int i = 0; i < valid_elements; ++i)
        {
            expected_sum += result.qs[i];
        }
        EXPECT_EQ(result.sum_qs, static_cast<int16_t>(expected_sum))
            << "Sum mismatch for valid_elements=" << valid_elements;
    }
}

// ============================================================================
// Test: Auto-dispatch uses scalar for partial blocks
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, AutoDispatch_UsesScalarForPartial)
{
    alignas(64) float input[32];
    generate_random_fp32(input, 32, -5.0f, 5.0f);

    Q8_1Block dispatch_result, scalar_result;

    // Auto-dispatch with partial block
    quantize_single_block(input, dispatch_result, 16);

    // Direct scalar call
    quantize_single_block_scalar(input, scalar_result, 16);

    // They should be identical (dispatch should use scalar for partial)
    auto cmp = compare_q8_1_blocks(dispatch_result, scalar_result);
    EXPECT_TRUE(cmp.passed) << "Auto-dispatch partial block: " << cmp.to_string();
}

// ============================================================================
// Test: Auto-dispatch produces same result as SIMD for full blocks
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, AutoDispatch_MatchesSIMD_FullBlocks)
{
    constexpr int NUM_ITERATIONS = 100;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        alignas(64) float input[32];
        generate_random_fp32(input, 32, -10.0f, 10.0f);

        Q8_1Block dispatch_result, scalar_result;
        quantize_single_block(input, dispatch_result, 32);
        quantize_single_block_scalar(input, scalar_result, 32);

        auto cmp = compare_q8_1_blocks(dispatch_result, scalar_result);
        EXPECT_TRUE(cmp.passed)
            << "Iteration " << iter << " (auto-dispatch vs scalar): " << cmp.to_string();
    }
}

// ============================================================================
// Test: Round-trip accuracy (quantize → dequantize → compare)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, RoundTripAccuracy_RandomData)
{
    constexpr int NUM_ITERATIONS = 100;
    constexpr int BLOCK_SIZE = 32;

    float total_max_error = 0.0f;
    float total_rms_error = 0.0f;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        alignas(64) float input[BLOCK_SIZE];
        generate_random_fp32(input, BLOCK_SIZE, -10.0f, 10.0f);

        // Find input range for expected error calculation
        float max_abs = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            max_abs = std::max(max_abs, std::fabs(input[i]));
        }

        Q8_1Block quantized;
        quantize_single_block_scalar(input, quantized, BLOCK_SIZE);

        float reconstructed[BLOCK_SIZE];
        dequantize_q8_1_block(quantized, reconstructed);

        float max_error = compute_max_round_trip_error(input, reconstructed, BLOCK_SIZE);
        float rms_error = compute_rms_round_trip_error(input, reconstructed, BLOCK_SIZE);

        // Q8_1 quantizes to [-127, 127], so max error per element is scale/2
        // Scale = max_abs / 127, so expected max error ≈ max_abs / 254
        float expected_max_error = max_abs / 127.0f; // Conservative bound

        EXPECT_LE(max_error, expected_max_error * 1.1f) // 10% margin
            << "Iteration " << iter << ": max_error=" << max_error
            << ", expected_max_error=" << expected_max_error;

        total_max_error += max_error;
        total_rms_error += rms_error;
    }

    // Print average errors (informational)
    float avg_max_error = total_max_error / NUM_ITERATIONS;
    float avg_rms_error = total_rms_error / NUM_ITERATIONS;
    std::cout << "[INFO] Average round-trip max error: " << avg_max_error << std::endl;
    std::cout << "[INFO] Average round-trip RMS error: " << avg_rms_error << std::endl;
}

// ============================================================================
// Test: Sum computation correctness
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, SumComputation_Correctness)
{
    constexpr int NUM_ITERATIONS = 100;
    constexpr int BLOCK_SIZE = 32;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        alignas(64) float input[BLOCK_SIZE];
        generate_random_fp32(input, BLOCK_SIZE, -10.0f, 10.0f);

        Q8_1Block result;
        quantize_single_block_scalar(input, result, BLOCK_SIZE);

        // Manually compute sum
        int32_t expected_sum = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            expected_sum += result.qs[i];
        }

        EXPECT_EQ(result.sum_qs, static_cast<int16_t>(expected_sum))
            << "Iteration " << iter << ": sum_qs mismatch";

#if defined(__AVX2__)
        Q8_1Block avx2_result;
        quantize_single_block_avx2(input, avx2_result);
        int32_t avx2_sum = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            avx2_sum += avx2_result.qs[i];
        }
        EXPECT_EQ(avx2_result.sum_qs, static_cast<int16_t>(avx2_sum))
            << "Iteration " << iter << ": AVX2 sum_qs mismatch";
#endif

#if defined(__AVX512F__)
        Q8_1Block avx512_result;
        quantize_single_block_avx512(input, avx512_result);
        int32_t avx512_sum = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            avx512_sum += avx512_result.qs[i];
        }
        EXPECT_EQ(avx512_result.sum_qs, static_cast<int16_t>(avx512_sum))
            << "Iteration " << iter << ": AVX512 sum_qs mismatch";
#endif
    }
}

// ============================================================================
// Test: Single extreme value in block (tests max finding)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, SingleExtremeValue_MaxFinding)
{
    constexpr int BLOCK_SIZE = 32;

    // Test with single large value at different positions
    for (int pos = 0; pos < BLOCK_SIZE; ++pos)
    {
        alignas(64) float input[BLOCK_SIZE];
        generate_zeros(input, BLOCK_SIZE);
        input[pos] = 100.0f;

        Q8_1Block scalar_result;
        quantize_single_block_scalar(input, scalar_result, BLOCK_SIZE);

        // The large value should be quantized to 127
        EXPECT_EQ(scalar_result.qs[pos], 127)
            << "Position " << pos << ": extreme value should quantize to 127";

        // All other values should be 0
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            if (i != pos)
            {
                EXPECT_EQ(scalar_result.qs[i], 0)
                    << "Position " << pos << ": qs[" << i << "] should be 0";
            }
        }

#if defined(__AVX2__)
        Q8_1Block avx2_result;
        quantize_single_block_avx2(input, avx2_result);
        auto cmp_avx2 = compare_q8_1_blocks(scalar_result, avx2_result);
        EXPECT_TRUE(cmp_avx2.passed) << "Position " << pos << " (AVX2): " << cmp_avx2.to_string();
#endif

#if defined(__AVX512F__)
        Q8_1Block avx512_result;
        quantize_single_block_avx512(input, avx512_result);
        auto cmp_avx512 = compare_q8_1_blocks(scalar_result, avx512_result);
        EXPECT_TRUE(cmp_avx512.passed) << "Position " << pos << " (AVX512): " << cmp_avx512.to_string();
#endif
    }
}

// ============================================================================
// Test: Negative extreme value (tests abs for max finding)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, NegativeExtremeValue_AbsMaxFinding)
{
    constexpr int BLOCK_SIZE = 32;

    for (int pos = 0; pos < BLOCK_SIZE; ++pos)
    {
        alignas(64) float input[BLOCK_SIZE];
        generate_zeros(input, BLOCK_SIZE);
        input[pos] = -100.0f; // Negative extreme

        Q8_1Block scalar_result;
        quantize_single_block_scalar(input, scalar_result, BLOCK_SIZE);

        // The negative value should be quantized to -127
        EXPECT_EQ(scalar_result.qs[pos], -127)
            << "Position " << pos << ": negative extreme value should quantize to -127";

#if defined(__AVX2__)
        Q8_1Block avx2_result;
        quantize_single_block_avx2(input, avx2_result);
        auto cmp_avx2 = compare_q8_1_blocks(scalar_result, avx2_result);
        EXPECT_TRUE(cmp_avx2.passed) << "Position " << pos << " (AVX2 negative): " << cmp_avx2.to_string();
#endif

#if defined(__AVX512F__)
        Q8_1Block avx512_result;
        quantize_single_block_avx512(input, avx512_result);
        auto cmp_avx512 = compare_q8_1_blocks(scalar_result, avx512_result);
        EXPECT_TRUE(cmp_avx512.passed) << "Position " << pos << " (AVX512 negative): " << cmp_avx512.to_string();
#endif
    }
}

// ============================================================================
// Test: FP32Tensor::quantize_to_q8_1 integration (uses the SIMD functions)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, FP32Tensor_Integration)
{
    constexpr int M = 4;  // rows
    constexpr int K = 64; // cols (2 blocks per row)

    // Create FP32 tensor
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K}, DeviceId::cpu());

    // Fill with random data
    float *data = tensor->mutable_data();
    generate_random_fp32(data, M * K, -5.0f, 5.0f);

    // Quantize to Q8_1
    int k_blocks = (K + 31) / 32;
    std::vector<Q8_1Block> q8_buffer(M * k_blocks);

    bool success = tensor->quantize_to_q8_1(q8_buffer.data(), M, K);
    ASSERT_TRUE(success) << "FP32Tensor::quantize_to_q8_1 failed";

    // Verify by comparing with direct scalar quantization
    for (int row = 0; row < M; ++row)
    {
        for (int blk = 0; blk < k_blocks; ++blk)
        {
            const float *src = data + row * K + blk * 32;
            int valid_elements = std::min(32, K - blk * 32);

            Q8_1Block expected;
            quantize_single_block_scalar(src, expected, valid_elements);

            const Q8_1Block &actual = q8_buffer[row * k_blocks + blk];

            auto cmp = compare_q8_1_blocks(expected, actual);
            EXPECT_TRUE(cmp.passed)
                << "Row " << row << ", block " << blk << ": " << cmp.to_string();
        }
    }
}

// ============================================================================
// Test: Large matrix quantization (stress test)
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Test, LargeMatrixQuantization_StressTest)
{
    constexpr int M = 128;
    constexpr int K = 4096;

    // Allow ±1 difference due to FP rounding differences
    constexpr int QS_TOLERANCE = 1;
    constexpr int SUM_TOLERANCE = 32;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K}, DeviceId::cpu());
    float *data = tensor->mutable_data();
    generate_random_fp32(data, M * K, -10.0f, 10.0f);

    int k_blocks = (K + 31) / 32;
    std::vector<Q8_1Block> q8_buffer(M * k_blocks);

    bool success = tensor->quantize_to_q8_1(q8_buffer.data(), M, K);
    ASSERT_TRUE(success) << "Large matrix quantization failed";

    // Spot-check a few random blocks
    std::uniform_int_distribution<int> row_dist(0, M - 1);
    std::uniform_int_distribution<int> blk_dist(0, k_blocks - 1);

    for (int check = 0; check < 100; ++check)
    {
        int row = row_dist(rng_);
        int blk = blk_dist(rng_);

        const float *src = data + row * K + blk * 32;
        int valid_elements = std::min(32, K - blk * 32);

        Q8_1Block expected;
        quantize_single_block_scalar(src, expected, valid_elements);

        const Q8_1Block &actual = q8_buffer[row * k_blocks + blk];
        auto cmp = compare_q8_1_blocks(expected, actual, QS_TOLERANCE, SUM_TOLERANCE);
        EXPECT_TRUE(cmp.passed)
            << "Check " << check << " (row=" << row << ", blk=" << blk << "): " << cmp.to_string();
    }
}
