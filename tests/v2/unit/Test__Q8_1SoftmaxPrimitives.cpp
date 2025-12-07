/**
 * @file Test__Q8_1SoftmaxPrimitives.cpp
 * @brief Unit tests for Q8_1 integer-aware softmax primitives
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Q8_1 SIMD variant parity (scalar vs AVX2 vs AVX512)
 * 2. Q8_1 vs FP32 reference parity (with conversion tolerance)
 * 3. Numerical accuracy analysis (cosine similarity, max absolute error)
 * 4. Causal masking correctness
 * 5. Edge cases (uniform inputs, single max, large scale differences)
 * 6. Requantization accuracy (scale factor, sum_qs)
 * 7. Multi-row batch correctness
 *
 * The Q8_1 softmax operates on quantized integer data, computing softmax
 * with minimal floating-point conversions and requantizing the output.
 * We test accuracy against a FP32 reference (dequant → softmax → requant).
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>
#include <random>

#include "v2/kernels/cpu/primitives/SoftmaxPrimitives_New.h"
#include "v2/tensors/BlockStructures.h"

using namespace llaminar2::primitives;
using namespace llaminar2;

namespace
{
    // ============================================================================
    // Test Tolerances
    // ============================================================================

    // Q8_1 has 8-bit precision per element, scale factor in FP16
    // Requantization introduces additional error
    constexpr float Q8_1_COSINE_TOLERANCE = 0.98f;   // Cosine similarity should be high
    constexpr float Q8_1_MAX_RELATIVE_ERROR = 0.15f; // Max relative error for probabilities > 0.01
    constexpr float Q8_1_ABSOLUTE_TOLERANCE = 0.02f; // Absolute tolerance for small probabilities

    // ============================================================================
    // Test Utilities
    // ============================================================================

    /**
     * @brief Dequantize Q8_1 block to FP32 values
     */
    std::vector<float> dequantize_q8_1_row(const Q8_1Block *row, int n_blocks)
    {
        std::vector<float> fp32(n_blocks * 32);
        for (int b = 0; b < n_blocks; ++b)
        {
            const Q8_1Block &block = row[b];
            // Convert FP16 scale to FP32 (stored as BF16 in our implementation)
            uint32_t fp32_bits = static_cast<uint32_t>(block.d) << 16;
            float d;
            std::memcpy(&d, &fp32_bits, sizeof(float));

            for (int i = 0; i < 32; ++i)
            {
                fp32[b * 32 + i] = static_cast<float>(block.qs[i]) * d;
            }
        }
        return fp32;
    }

    /**
     * @brief Quantize FP32 values to Q8_1 blocks
     */
    void quantize_fp32_to_q8_1(const float *fp32, Q8_1Block *row, int n_blocks)
    {
        for (int b = 0; b < n_blocks; ++b)
        {
            Q8_1Block &block = row[b];
            const float *block_data = fp32 + b * 32;

            // Find max absolute value for scale
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                float abs_val = std::abs(block_data[i]);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }

            // Compute scale (max maps to 127)
            float d = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f / 127.0f;
            float inv_d = 1.0f / d;

            // Convert scale to BF16 format
            uint32_t d_bits;
            std::memcpy(&d_bits, &d, sizeof(float));
            uint16_t d_bf16 = static_cast<uint16_t>(d_bits >> 16);
            block.d = d_bf16;

            // Quantize values
            int16_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_d));
                q = std::max(-128, std::min(127, q));
                block.qs[i] = static_cast<int8_t>(q);
                sum_qs += block.qs[i];
            }
            block.sum_qs = sum_qs;
        }
    }

    /**
     * @brief Reference FP32 softmax (for comparison)
     */
    void reference_softmax_fp32(float *row, int cols, bool causal, float scale, int row_idx)
    {
        // Pass 1: Find max
        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            if (v > row_max)
                row_max = v;
        }
        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            sum += std::exp(v - row_max);
        }
        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0.0f;
            }
            else
            {
                float v = row[c] * scale;
                row[c] = std::exp(v - row_max) * inv;
            }
        }
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    float cosine_similarity(const float *a, const float *b, int n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-10 || norm_b < 1e-10)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    /**
     * @brief Compute max relative error for non-zero values
     */
    float max_relative_error(const float *expected, const float *actual, int n, float threshold = 0.01f)
    {
        float max_rel = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            if (std::abs(expected[i]) > threshold)
            {
                float rel = std::abs(expected[i] - actual[i]) / std::abs(expected[i]);
                if (rel > max_rel)
                    max_rel = rel;
            }
        }
        return max_rel;
    }

    /**
     * @brief Create random Q8_1 row with specified characteristics
     */
    std::vector<Q8_1Block> create_random_q8_1_row(int n_blocks, std::mt19937 &rng,
                                                  float scale_min = 0.001f, float scale_max = 1.0f)
    {
        std::vector<Q8_1Block> row(n_blocks);
        std::uniform_real_distribution<float> scale_dist(scale_min, scale_max);
        std::uniform_int_distribution<int8_t> qs_dist(-100, 100); // Leave headroom from extremes

        for (int b = 0; b < n_blocks; ++b)
        {
            float d = scale_dist(rng);
            // Convert to BF16
            uint32_t d_bits;
            std::memcpy(&d_bits, &d, sizeof(float));
            row[b].d = static_cast<uint16_t>(d_bits >> 16);

            int16_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                row[b].qs[i] = qs_dist(rng);
                sum_qs += row[b].qs[i];
            }
            row[b].sum_qs = sum_qs;
        }
        return row;
    }

    /**
     * @brief Check that softmax output sums to ~1.0 (after dequantization)
     */
    void expect_q8_1_sums_to_one(const Q8_1Block *row, int n_blocks, float tolerance = 0.05f)
    {
        auto fp32 = dequantize_q8_1_row(row, n_blocks);
        float sum = 0.0f;
        for (float v : fp32)
            sum += v;
        EXPECT_NEAR(sum, 1.0f, tolerance) << "Q8_1 softmax output should sum to ~1.0";
    }

    /**
     * @brief Check that all dequantized values are valid probabilities
     */
    void expect_q8_1_valid_probabilities(const Q8_1Block *row, int n_blocks)
    {
        auto fp32 = dequantize_q8_1_row(row, n_blocks);
        for (int i = 0; i < static_cast<int>(fp32.size()); ++i)
        {
            EXPECT_GE(fp32[i], -0.01f) << "Q8_1 probability at " << i << " is negative: " << fp32[i];
            EXPECT_LE(fp32[i], 1.05f) << "Q8_1 probability at " << i << " exceeds 1.0: " << fp32[i];
        }
    }

} // anonymous namespace

// ============================================================================
// Test Class
// ============================================================================

class Q8_1SoftmaxPrimitivesTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rng_.seed(42); // Deterministic tests
    }

    std::mt19937 rng_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(Q8_1SoftmaxPrimitivesTest, ScalarBasicFunctionality)
{
    const int n_blocks = 4; // 128 elements

    // Create input with clear max at position 0
    // Use larger scale and negative values for others to make max dominant
    std::vector<Q8_1Block> input(n_blocks);
    for (int b = 0; b < n_blocks; ++b)
    {
        float d = 0.1f; // Larger scale
        uint32_t d_bits;
        std::memcpy(&d_bits, &d, sizeof(float));
        input[b].d = static_cast<uint16_t>(d_bits >> 16);

        int16_t sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            // Position 0 in block 0 is max (127), others are negative (-50)
            // Difference = 127*0.1 - (-50*0.1) = 12.7 - (-5) = 17.7
            // exp(17.7) >> exp(0), so max should dominate
            input[b].qs[i] = (b == 0 && i == 0) ? 127 : -50;
            sum_qs += input[b].qs[i];
        }
        input[b].sum_qs = sum_qs;
    }

    // Apply softmax
    softmax_row_q8_1_scalar(input.data(), n_blocks, false, 1.0f, 0);

    // Check basic properties
    expect_q8_1_valid_probabilities(input.data(), n_blocks);
    expect_q8_1_sums_to_one(input.data(), n_blocks);

    // Max element should have highest probability
    auto fp32 = dequantize_q8_1_row(input.data(), n_blocks);
    float max_prob = *std::max_element(fp32.begin(), fp32.end());
    EXPECT_GT(max_prob, 0.5f) << "Max probability should be dominant";
}

TEST_F(Q8_1SoftmaxPrimitivesTest, AVX2BasicFunctionality)
{
    const int n_blocks = 4;
    auto input = create_random_q8_1_row(n_blocks, rng_);

    softmax_row_q8_1_avx2(input.data(), n_blocks, false, 1.0f, 0);

    expect_q8_1_valid_probabilities(input.data(), n_blocks);
    expect_q8_1_sums_to_one(input.data(), n_blocks);
}

TEST_F(Q8_1SoftmaxPrimitivesTest, AVX512BasicFunctionality)
{
    const int n_blocks = 4;
    auto input = create_random_q8_1_row(n_blocks, rng_);

    softmax_row_q8_1_avx512(input.data(), n_blocks, false, 1.0f, 0);

    expect_q8_1_valid_probabilities(input.data(), n_blocks);
    expect_q8_1_sums_to_one(input.data(), n_blocks);
}

// ============================================================================
// SIMD Variant Parity Tests
// ============================================================================

TEST_F(Q8_1SoftmaxPrimitivesTest, SIMDVariantParity_SmallInput)
{
    const int n_blocks = 2; // 64 elements
    auto input_scalar = create_random_q8_1_row(n_blocks, rng_);
    auto input_avx2 = input_scalar;
    auto input_avx512 = input_scalar;

    softmax_row_q8_1_scalar(input_scalar.data(), n_blocks, false, 1.0f, 0);
    softmax_row_q8_1_avx2(input_avx2.data(), n_blocks, false, 1.0f, 0);
    softmax_row_q8_1_avx512(input_avx512.data(), n_blocks, false, 1.0f, 0);

    auto fp32_scalar = dequantize_q8_1_row(input_scalar.data(), n_blocks);
    auto fp32_avx2 = dequantize_q8_1_row(input_avx2.data(), n_blocks);
    auto fp32_avx512 = dequantize_q8_1_row(input_avx512.data(), n_blocks);

    float cos_avx2 = cosine_similarity(fp32_scalar.data(), fp32_avx2.data(), n_blocks * 32);
    float cos_avx512 = cosine_similarity(fp32_scalar.data(), fp32_avx512.data(), n_blocks * 32);

    EXPECT_GE(cos_avx2, 0.999f) << "AVX2 should match scalar closely";
    EXPECT_GE(cos_avx512, 0.999f) << "AVX512 should match scalar closely";
}

TEST_F(Q8_1SoftmaxPrimitivesTest, SIMDVariantParity_LargeInput)
{
    const int n_blocks = 16; // 512 elements
    auto input_scalar = create_random_q8_1_row(n_blocks, rng_);
    auto input_avx2 = input_scalar;
    auto input_avx512 = input_scalar;

    softmax_row_q8_1_scalar(input_scalar.data(), n_blocks, false, 1.0f, 0);
    softmax_row_q8_1_avx2(input_avx2.data(), n_blocks, false, 1.0f, 0);
    softmax_row_q8_1_avx512(input_avx512.data(), n_blocks, false, 1.0f, 0);

    auto fp32_scalar = dequantize_q8_1_row(input_scalar.data(), n_blocks);
    auto fp32_avx2 = dequantize_q8_1_row(input_avx2.data(), n_blocks);
    auto fp32_avx512 = dequantize_q8_1_row(input_avx512.data(), n_blocks);

    float cos_avx2 = cosine_similarity(fp32_scalar.data(), fp32_avx2.data(), n_blocks * 32);
    float cos_avx512 = cosine_similarity(fp32_scalar.data(), fp32_avx512.data(), n_blocks * 32);

    EXPECT_GE(cos_avx2, 0.999f) << "AVX2 should match scalar closely";
    EXPECT_GE(cos_avx512, 0.999f) << "AVX512 should match scalar closely";
}

// ============================================================================
// Q8_1 vs FP32 Reference Parity Tests
// ============================================================================

TEST_F(Q8_1SoftmaxPrimitivesTest, VsFP32Reference_RandomInput)
{
    const int n_blocks = 4;
    auto input_q8 = create_random_q8_1_row(n_blocks, rng_);

    // Dequant → FP32 softmax → requant (reference)
    auto fp32_input = dequantize_q8_1_row(input_q8.data(), n_blocks);
    auto fp32_ref = fp32_input;
    reference_softmax_fp32(fp32_ref.data(), n_blocks * 32, false, 1.0f, 0);

    // Q8_1 native softmax
    softmax_row_q8_1(input_q8.data(), n_blocks, false, 1.0f, 0);
    auto fp32_q8 = dequantize_q8_1_row(input_q8.data(), n_blocks);

    // Compare
    float cos_sim = cosine_similarity(fp32_ref.data(), fp32_q8.data(), n_blocks * 32);
    float max_rel = max_relative_error(fp32_ref.data(), fp32_q8.data(), n_blocks * 32);

    EXPECT_GE(cos_sim, Q8_1_COSINE_TOLERANCE)
        << "Cosine similarity vs FP32: " << cos_sim;
    EXPECT_LE(max_rel, Q8_1_MAX_RELATIVE_ERROR)
        << "Max relative error vs FP32: " << max_rel;
}

TEST_F(Q8_1SoftmaxPrimitivesTest, VsFP32Reference_UniformInput)
{
    const int n_blocks = 4;

    // Create uniform input (all same value)
    std::vector<Q8_1Block> input_q8(n_blocks);
    for (int b = 0; b < n_blocks; ++b)
    {
        float d = 0.01f;
        uint32_t d_bits;
        std::memcpy(&d_bits, &d, sizeof(float));
        input_q8[b].d = static_cast<uint16_t>(d_bits >> 16);

        int16_t sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            input_q8[b].qs[i] = 50; // All same
            sum_qs += input_q8[b].qs[i];
        }
        input_q8[b].sum_qs = sum_qs;
    }

    // Reference
    auto fp32_ref = dequantize_q8_1_row(input_q8.data(), n_blocks);
    reference_softmax_fp32(fp32_ref.data(), n_blocks * 32, false, 1.0f, 0);

    // Q8_1 native
    softmax_row_q8_1(input_q8.data(), n_blocks, false, 1.0f, 0);
    auto fp32_q8 = dequantize_q8_1_row(input_q8.data(), n_blocks);

    // Uniform input should give uniform output (~1/128 each)
    float expected_prob = 1.0f / (n_blocks * 32);
    for (int i = 0; i < n_blocks * 32; ++i)
    {
        EXPECT_NEAR(fp32_q8[i], expected_prob, 0.02f)
            << "Uniform input should give uniform probabilities at index " << i;
    }
}

TEST_F(Q8_1SoftmaxPrimitivesTest, VsFP32Reference_PeakyInput)
{
    const int n_blocks = 4;

    // Create input with one clear max (peaky distribution)
    std::vector<Q8_1Block> input_q8(n_blocks);
    for (int b = 0; b < n_blocks; ++b)
    {
        float d = 0.1f;
        uint32_t d_bits;
        std::memcpy(&d_bits, &d, sizeof(float));
        input_q8[b].d = static_cast<uint16_t>(d_bits >> 16);

        int16_t sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            // Large positive at position 0 of block 0
            input_q8[b].qs[i] = (b == 0 && i == 0) ? 127 : -50;
            sum_qs += input_q8[b].qs[i];
        }
        input_q8[b].sum_qs = sum_qs;
    }

    // Reference
    auto fp32_ref = dequantize_q8_1_row(input_q8.data(), n_blocks);
    reference_softmax_fp32(fp32_ref.data(), n_blocks * 32, false, 1.0f, 0);

    // Q8_1 native
    softmax_row_q8_1(input_q8.data(), n_blocks, false, 1.0f, 0);
    auto fp32_q8 = dequantize_q8_1_row(input_q8.data(), n_blocks);

    // Max element should dominate
    EXPECT_GT(fp32_q8[0], 0.9f) << "Max element should have probability > 0.9";

    float cos_sim = cosine_similarity(fp32_ref.data(), fp32_q8.data(), n_blocks * 32);
    EXPECT_GE(cos_sim, Q8_1_COSINE_TOLERANCE) << "Cosine similarity: " << cos_sim;
}

// ============================================================================
// Causal Masking Tests
// ============================================================================

TEST_F(Q8_1SoftmaxPrimitivesTest, CausalMasking_Basic)
{
    const int n_blocks = 4; // 128 elements
    auto input_q8 = create_random_q8_1_row(n_blocks, rng_);

    // Apply causal mask at position 63 (halfway)
    softmax_row_q8_1(input_q8.data(), n_blocks, true, 1.0f, 63);
    auto fp32 = dequantize_q8_1_row(input_q8.data(), n_blocks);

    // Check that positions > 63 are zero
    for (int i = 64; i < n_blocks * 32; ++i)
    {
        EXPECT_NEAR(fp32[i], 0.0f, 0.01f)
            << "Position " << i << " should be masked (zero) but is " << fp32[i];
    }

    // Check that positions <= 63 sum to ~1
    float sum_valid = 0.0f;
    for (int i = 0; i <= 63; ++i)
    {
        sum_valid += fp32[i];
    }
    EXPECT_NEAR(sum_valid, 1.0f, 0.05f) << "Valid positions should sum to ~1.0";
}

TEST_F(Q8_1SoftmaxPrimitivesTest, CausalMasking_AllMasked)
{
    const int n_blocks = 2;
    auto input_q8 = create_random_q8_1_row(n_blocks, rng_);

    // Causal mask at -1 (all masked)
    softmax_row_q8_1(input_q8.data(), n_blocks, true, 1.0f, -1);
    auto fp32 = dequantize_q8_1_row(input_q8.data(), n_blocks);

    // All should be zero (or numerically close)
    for (int i = 0; i < n_blocks * 32; ++i)
    {
        EXPECT_NEAR(fp32[i], 0.0f, 0.01f) << "All positions should be masked";
    }
}

TEST_F(Q8_1SoftmaxPrimitivesTest, CausalMasking_SIMDVariantParity)
{
    const int n_blocks = 4;
    auto input_scalar = create_random_q8_1_row(n_blocks, rng_);
    auto input_avx2 = input_scalar;
    auto input_avx512 = input_scalar;

    const int row_idx = 50; // Causal position

    softmax_row_q8_1_scalar(input_scalar.data(), n_blocks, true, 1.0f, row_idx);
    softmax_row_q8_1_avx2(input_avx2.data(), n_blocks, true, 1.0f, row_idx);
    softmax_row_q8_1_avx512(input_avx512.data(), n_blocks, true, 1.0f, row_idx);

    auto fp32_scalar = dequantize_q8_1_row(input_scalar.data(), n_blocks);
    auto fp32_avx2 = dequantize_q8_1_row(input_avx2.data(), n_blocks);
    auto fp32_avx512 = dequantize_q8_1_row(input_avx512.data(), n_blocks);

    float cos_avx2 = cosine_similarity(fp32_scalar.data(), fp32_avx2.data(), n_blocks * 32);
    float cos_avx512 = cosine_similarity(fp32_scalar.data(), fp32_avx512.data(), n_blocks * 32);

    EXPECT_GE(cos_avx2, 0.999f) << "AVX2 causal should match scalar";
    EXPECT_GE(cos_avx512, 0.999f) << "AVX512 causal should match scalar";
}

// ============================================================================
// Scale Factor Tests
// ============================================================================

TEST_F(Q8_1SoftmaxPrimitivesTest, ScaleFactor_AttentionScale)
{
    const int n_blocks = 4;
    auto input_q8 = create_random_q8_1_row(n_blocks, rng_);

    // Typical attention scale: 1/sqrt(head_dim)
    const float head_dim = 64.0f;
    const float scale = 1.0f / std::sqrt(head_dim);

    softmax_row_q8_1(input_q8.data(), n_blocks, false, scale, 0);

    expect_q8_1_valid_probabilities(input_q8.data(), n_blocks);
    expect_q8_1_sums_to_one(input_q8.data(), n_blocks);
}

TEST_F(Q8_1SoftmaxPrimitivesTest, ScaleFactor_LargeScale)
{
    const int n_blocks = 2;
    auto input_q8 = create_random_q8_1_row(n_blocks, rng_, 0.001f, 0.01f); // Small values

    // Large scale factor
    const float scale = 10.0f;

    softmax_row_q8_1(input_q8.data(), n_blocks, false, scale, 0);

    expect_q8_1_valid_probabilities(input_q8.data(), n_blocks);
    expect_q8_1_sums_to_one(input_q8.data(), n_blocks, 0.1f); // Slightly relaxed
}

// ============================================================================
// Requantization Tests
// ============================================================================

TEST_F(Q8_1SoftmaxPrimitivesTest, Requantization_ScaleComputation)
{
    const int n_blocks = 4;

    // Create peaky input
    std::vector<Q8_1Block> input_q8(n_blocks);
    for (int b = 0; b < n_blocks; ++b)
    {
        float d = 0.1f;
        uint32_t d_bits;
        std::memcpy(&d_bits, &d, sizeof(float));
        input_q8[b].d = static_cast<uint16_t>(d_bits >> 16);

        for (int i = 0; i < 32; ++i)
        {
            input_q8[b].qs[i] = (b == 0 && i == 0) ? 127 : 0;
        }
        input_q8[b].sum_qs = 127;
    }

    softmax_row_q8_1(input_q8.data(), n_blocks, false, 1.0f, 0);

    // After softmax, max probability is close to 1.0
    // Scale should be approximately 1/127 = 0.00787...
    // But stored as BF16, so check it's in reasonable range
    for (int b = 0; b < n_blocks; ++b)
    {
        uint32_t d_bits = static_cast<uint32_t>(input_q8[b].d) << 16;
        float d;
        std::memcpy(&d, &d_bits, sizeof(float));

        EXPECT_GT(d, 0.0f) << "Scale should be positive for block " << b;
        EXPECT_LT(d, 1.0f) << "Scale should be < 1 for probability block " << b;
    }
}

TEST_F(Q8_1SoftmaxPrimitivesTest, Requantization_SumQsConsistency)
{
    const int n_blocks = 4;
    auto input_q8 = create_random_q8_1_row(n_blocks, rng_);

    softmax_row_q8_1(input_q8.data(), n_blocks, false, 1.0f, 0);

    // Verify sum_qs matches actual sum
    for (int b = 0; b < n_blocks; ++b)
    {
        int16_t actual_sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            actual_sum += input_q8[b].qs[i];
        }
        EXPECT_EQ(input_q8[b].sum_qs, actual_sum)
            << "sum_qs should match actual sum for block " << b;
    }
}

// ============================================================================
// Multi-Row Batch Tests
// ============================================================================

TEST_F(Q8_1SoftmaxPrimitivesTest, BatchSoftmax_Basic)
{
    const int rows = 4;
    const int n_blocks_per_row = 2;

    std::vector<Q8_1Block> batch(rows * n_blocks_per_row);
    for (int r = 0; r < rows; ++r)
    {
        auto row = create_random_q8_1_row(n_blocks_per_row, rng_);
        std::copy(row.begin(), row.end(), batch.begin() + r * n_blocks_per_row);
    }

    softmax_row_major_q8_1(batch.data(), rows, n_blocks_per_row, false, 1.0f, true);

    // Each row should independently sum to ~1
    for (int r = 0; r < rows; ++r)
    {
        Q8_1Block *row_ptr = batch.data() + r * n_blocks_per_row;
        expect_q8_1_sums_to_one(row_ptr, n_blocks_per_row);
    }
}

TEST_F(Q8_1SoftmaxPrimitivesTest, BatchSoftmax_CausalMultiRow)
{
    const int rows = 4;
    const int n_blocks_per_row = 2;

    std::vector<Q8_1Block> batch(rows * n_blocks_per_row);
    for (int r = 0; r < rows; ++r)
    {
        auto row = create_random_q8_1_row(n_blocks_per_row, rng_);
        std::copy(row.begin(), row.end(), batch.begin() + r * n_blocks_per_row);
    }

    // Causal mask varies by row
    softmax_row_major_q8_1(batch.data(), rows, n_blocks_per_row, true, 1.0f, true);

    // Each row should have causal mask at its row index
    for (int r = 0; r < rows; ++r)
    {
        Q8_1Block *row_ptr = batch.data() + r * n_blocks_per_row;
        auto fp32 = dequantize_q8_1_row(row_ptr, n_blocks_per_row);

        // Positions > r should be zero
        for (int c = r + 1; c < n_blocks_per_row * 32; ++c)
        {
            EXPECT_NEAR(fp32[c], 0.0f, 0.01f)
                << "Row " << r << ", position " << c << " should be masked";
        }
    }
}

// ============================================================================
// Numerical Accuracy Statistics Test
// ============================================================================

TEST_F(Q8_1SoftmaxPrimitivesTest, AccuracyStatistics_MultipleTrials)
{
    const int n_trials = 20;
    const int n_blocks = 8;

    float total_cos_sim = 0.0f;
    float min_cos_sim = 1.0f;
    float total_max_rel = 0.0f;
    float worst_max_rel = 0.0f;

    for (int trial = 0; trial < n_trials; ++trial)
    {
        auto input_q8 = create_random_q8_1_row(n_blocks, rng_);

        // FP32 reference
        auto fp32_ref = dequantize_q8_1_row(input_q8.data(), n_blocks);
        reference_softmax_fp32(fp32_ref.data(), n_blocks * 32, false, 1.0f, 0);

        // Q8_1 native
        softmax_row_q8_1(input_q8.data(), n_blocks, false, 1.0f, 0);
        auto fp32_q8 = dequantize_q8_1_row(input_q8.data(), n_blocks);

        float cos_sim = cosine_similarity(fp32_ref.data(), fp32_q8.data(), n_blocks * 32);
        float max_rel = max_relative_error(fp32_ref.data(), fp32_q8.data(), n_blocks * 32);

        total_cos_sim += cos_sim;
        min_cos_sim = std::min(min_cos_sim, cos_sim);
        total_max_rel += max_rel;
        worst_max_rel = std::max(worst_max_rel, max_rel);
    }

    float avg_cos_sim = total_cos_sim / n_trials;
    float avg_max_rel = total_max_rel / n_trials;

    std::cout << "Q8_1 Softmax Accuracy Statistics over " << n_trials << " trials:" << std::endl;
    std::cout << "  Average cosine similarity: " << avg_cos_sim << std::endl;
    std::cout << "  Minimum cosine similarity: " << min_cos_sim << std::endl;
    std::cout << "  Average max relative error: " << avg_max_rel << std::endl;
    std::cout << "  Worst max relative error: " << worst_max_rel << std::endl;

    EXPECT_GE(avg_cos_sim, 0.98f) << "Average cosine similarity should be high";
    EXPECT_GE(min_cos_sim, Q8_1_COSINE_TOLERANCE) << "Minimum cosine similarity should meet threshold";
}
