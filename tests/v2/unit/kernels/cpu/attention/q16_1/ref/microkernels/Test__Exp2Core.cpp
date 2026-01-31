/**
 * @file Test__Exp2Core.cpp
 * @brief Unit tests for the shared Exp2Core primitives
 *
 * Tests the core exp2 LUT-based softmax primitives that are shared across:
 * - Exp2FixedSoftmax (standalone batch softmax)
 * - OnlineSoftmax (FlashAttention-style online softmax)
 * - Q16IntegerAttentionRef (snapshot computation)
 *
 * These tests focus on the individual primitive functions to ensure they
 * behave correctly across a variety of input conditions.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/attention/q16_1/ref/microkernels/Exp2Core.h"

using namespace llaminar2::kernels::q16_1::microkernels;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__Exp2Core : public ::testing::Test
{
protected:
    static constexpr int32_t MASKED = std::numeric_limits<int32_t>::min();
    static constexpr int LUT_VALUE_BITS = 30;
    static constexpr uint32_t ONE = 1U << LUT_VALUE_BITS;

    void SetUp() override
    {
        // Ensure LUT is initialized for all tests
        ensure_exp2_lut_initialized(LUT_VALUE_BITS);
    }

    // Reference exp2 computation using standard library
    double reference_exp2(double t)
    {
        return std::pow(2.0, -t);
    }

    // Reference softmax probability
    double reference_exp(double alpha, double delta)
    {
        return std::exp(-alpha * delta);
    }
};

// ============================================================================
// LUT Initialization Tests
// ============================================================================

TEST_F(Test__Exp2Core, LUTInitialization_IsThreadSafe)
{
    // Multiple calls should succeed without error
    ensure_exp2_lut_initialized(30);
    ensure_exp2_lut_initialized(30);
    ensure_exp2_lut_initialized(30);

    const uint32_t *lut = get_exp2_lut_data();
    ASSERT_NE(lut, nullptr);
}

TEST_F(Test__Exp2Core, LUTValues_AtIndex0)
{
    // LUT[0] = 2^(-0) = 1.0, scaled by 2^30
    const uint32_t *lut = get_exp2_lut_data();
    ASSERT_NE(lut, nullptr);

    EXPECT_EQ(lut[0], ONE) << "LUT[0] should be exactly 2^30";
}

TEST_F(Test__Exp2Core, LUTValues_AtIndex1024)
{
    // LUT[1024] = 2^(-0.5) ≈ 0.7071
    // 1024/2048 = 0.5
    const uint32_t *lut = get_exp2_lut_data();

    double expected = std::pow(2.0, -0.5);
    double actual = static_cast<double>(lut[1024]) / static_cast<double>(ONE);

    EXPECT_NEAR(actual, expected, 1e-6)
        << "LUT[1024] should be 2^(-0.5), expected " << expected << " got " << actual;
}

TEST_F(Test__Exp2Core, LUTValues_AtIndex2047)
{
    // LUT[2047] = 2^(-2047/2048) ≈ 0.5002
    const uint32_t *lut = get_exp2_lut_data();

    double expected = std::pow(2.0, -2047.0 / 2048.0);
    double actual = static_cast<double>(lut[2047]) / static_cast<double>(ONE);

    EXPECT_NEAR(actual, expected, 1e-6)
        << "LUT[2047] should be 2^(-2047/2048), expected " << expected << " got " << actual;
}

TEST_F(Test__Exp2Core, LUTValues_MonotonicallyDecreasing)
{
    // LUT should be monotonically decreasing since 2^(-x) decreases as x increases
    const uint32_t *lut = get_exp2_lut_data();

    for (int i = 1; i < EXP2_LUT_SIZE; ++i)
    {
        ASSERT_LE(lut[i], lut[i - 1])
            << "LUT should be monotonically decreasing at index " << i;
    }
}

// ============================================================================
// AdaptiveAlphaConfig Tests
// ============================================================================

TEST_F(Test__Exp2Core, AdaptiveAlphaConfig_VerySmallAlpha)
{
    // Very small alpha like 7.45e-9 (typical for KV_CACHE_SCALE=8.0)
    float alpha = 7.45e-9f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    EXPECT_EQ(config.original_alpha, alpha);
    EXPECT_EQ(config.alpha_shift, 30); // Should use max shift for tiny alpha
    EXPECT_GT(config.effective_alpha, 0);

    // Verify we can recover approximate alpha
    double recovered = static_cast<double>(config.effective_alpha) / static_cast<double>(1ULL << 30);
    EXPECT_NEAR(recovered, alpha, alpha * 0.01) << "Should recover original alpha";
}

TEST_F(Test__Exp2Core, AdaptiveAlphaConfig_SmallAlpha)
{
    float alpha = 1e-5f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    // 1e-5 is not < 1e-6, so it uses alpha_shift=20
    EXPECT_EQ(config.alpha_shift, 20);
    EXPECT_GT(config.effective_alpha, 0);
}

TEST_F(Test__Exp2Core, AdaptiveAlphaConfig_NormalAlpha)
{
    // Normal alpha like 1/sqrt(128) ≈ 0.0884
    float alpha = 1.0f / std::sqrt(128.0f);
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    EXPECT_EQ(config.original_alpha, alpha);
    EXPECT_EQ(config.alpha_shift, 10); // Normal range uses smaller shift
    EXPECT_GT(config.effective_alpha, 0);

    double recovered = static_cast<double>(config.effective_alpha) / static_cast<double>(1ULL << 10);
    EXPECT_NEAR(recovered, alpha, alpha * 0.01);
}

TEST_F(Test__Exp2Core, AdaptiveAlphaConfig_LargeAlpha)
{
    float alpha = 5.0f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    EXPECT_EQ(config.alpha_shift, 0); // Large alpha uses no shift
    EXPECT_NEAR(static_cast<float>(config.effective_alpha), alpha, alpha * 0.01);
}

// ============================================================================
// Weight Computation Tests
// ============================================================================

TEST_F(Test__Exp2Core, ComputeWeight_DeltaZero)
{
    // delta=0 means this is the max, should return full weight
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(0.1f);
    uint32_t weight = exp2_compute_weight(0, config);

    EXPECT_EQ(weight, ONE) << "Delta=0 should give weight=1 (scaled)";
}

TEST_F(Test__Exp2Core, ComputeWeight_SmallDelta)
{
    float alpha = 0.1f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    // For delta=10 and alpha=0.1:
    // exp(-0.1 * 10) = exp(-1) ≈ 0.368
    int32_t delta = 10;
    uint32_t weight = exp2_compute_weight(delta, config);

    double expected = std::exp(-alpha * delta);
    double actual = static_cast<double>(weight) / static_cast<double>(ONE);

    // Allow 5% error due to LUT approximation
    EXPECT_NEAR(actual, expected, expected * 0.05)
        << "Weight computation for delta=" << delta << " alpha=" << alpha;
}

TEST_F(Test__Exp2Core, ComputeWeight_LargeDelta)
{
    float alpha = 0.01f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    // For delta=1000 and alpha=0.01:
    // exp(-0.01 * 1000) = exp(-10) ≈ 4.5e-5
    int32_t delta = 1000;
    uint32_t weight = exp2_compute_weight(delta, config);

    double expected = std::exp(-alpha * delta);
    double actual = static_cast<double>(weight) / static_cast<double>(ONE);

    // LUT approximation has ~20-30% error for very small weights
    // This is acceptable since these weights contribute very little to the final sum
    EXPECT_NEAR(actual, expected, expected * 0.5);
}

TEST_F(Test__Exp2Core, ComputeWeight_Underflow)
{
    // Very large delta should underflow to 0
    float alpha = 1.0f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    int32_t delta = 100; // exp(-100) ≈ 3.7e-44, will underflow
    uint32_t weight = exp2_compute_weight(delta, config);

    EXPECT_EQ(weight, 0) << "Large delta should underflow to 0";
}

TEST_F(Test__Exp2Core, ComputeWeight_NegativeDelta)
{
    // Negative delta should be treated same as delta=0
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(0.1f);
    uint32_t weight = exp2_compute_weight(-10, config);

    EXPECT_EQ(weight, ONE) << "Negative delta should give weight=1";
}

// ============================================================================
// Rescale Factor Tests
// ============================================================================

TEST_F(Test__Exp2Core, ComputeRescale_NoChange)
{
    float alpha = 0.1f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    int32_t scale_num;
    int scale_shift;
    exp2_compute_rescale(100, 100, config, scale_num, scale_shift);

    // No change in max should give identity rescale
    EXPECT_EQ(scale_num, static_cast<int32_t>(ONE));
    EXPECT_EQ(scale_shift, LUT_VALUE_BITS);

    // Verify: (x * scale_num) >> scale_shift = x
    int64_t x = 12345;
    int64_t result = (x * scale_num) >> scale_shift;
    EXPECT_EQ(result, x);
}

TEST_F(Test__Exp2Core, ComputeRescale_SmallIncrease)
{
    float alpha = 0.01f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    int32_t m_old = 100;
    int32_t m_new = 110;
    int32_t scale_num;
    int scale_shift;

    exp2_compute_rescale(m_old, m_new, config, scale_num, scale_shift);

    // Rescale factor = exp((m_old - m_new) * alpha) = exp(-0.1) ≈ 0.905
    double expected = std::exp(-(m_new - m_old) * alpha);

    // Apply rescale to test value
    int64_t test_val = 1000000;
    int64_t rescaled = (test_val * scale_num) >> scale_shift;
    double actual_factor = static_cast<double>(rescaled) / static_cast<double>(test_val);

    EXPECT_NEAR(actual_factor, expected, expected * 0.05);
}

TEST_F(Test__Exp2Core, ComputeRescale_LargeIncrease)
{
    float alpha = 0.01f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    int32_t m_old = 0;
    int32_t m_new = 500;
    int32_t scale_num;
    int scale_shift;

    exp2_compute_rescale(m_old, m_new, config, scale_num, scale_shift);

    // Rescale factor = exp(-500 * 0.01) = exp(-5) ≈ 0.0067
    double expected = std::exp(-5.0);

    int64_t test_val = 100000000LL;
    int64_t rescaled = (test_val * scale_num) >> scale_shift;
    double actual_factor = static_cast<double>(rescaled) / static_cast<double>(test_val);

    // Allow 20% error for LUT approximation at edges
    EXPECT_NEAR(actual_factor, expected, expected * 0.2);
}

TEST_F(Test__Exp2Core, ComputeRescale_MaskedOldMax)
{
    // When m_old is MASKED, should return identity
    float alpha = 0.1f;
    AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);

    int32_t scale_num;
    int scale_shift;
    exp2_compute_rescale(MASKED, 100, config, scale_num, scale_shift);

    EXPECT_EQ(scale_num, static_cast<int32_t>(ONE));
    EXPECT_EQ(scale_shift, LUT_VALUE_BITS);
}

// ============================================================================
// Block Weight Computation Tests
// ============================================================================

TEST_F(Test__Exp2Core, ComputeBlockWeights_SingleElement)
{
    std::vector<int32_t> scores = {1000};
    std::vector<int32_t> weights(1);

    int64_t sum = exp2_compute_block_weights(
        scores.data(), weights.data(), 1, 1000, 0.1f);

    EXPECT_EQ(weights[0], static_cast<int32_t>(ONE));
    EXPECT_EQ(sum, ONE);
}

TEST_F(Test__Exp2Core, ComputeBlockWeights_TwoEqualScores)
{
    std::vector<int32_t> scores = {500, 500};
    std::vector<int32_t> weights(2);

    int64_t sum = exp2_compute_block_weights(
        scores.data(), weights.data(), 2, 500, 0.1f);

    // Equal scores should give equal weights
    EXPECT_EQ(weights[0], weights[1]);

    // Sum should be 2 * weight (use int64_t to avoid overflow)
    EXPECT_EQ(sum, static_cast<int64_t>(weights[0]) + static_cast<int64_t>(weights[1]));
}

TEST_F(Test__Exp2Core, ComputeBlockWeights_MaskedElements)
{
    std::vector<int32_t> scores = {100, MASKED, 50, MASKED};
    std::vector<int32_t> weights(4);

    int64_t sum = exp2_compute_block_weights(
        scores.data(), weights.data(), 4, 100, 0.01f);

    // Masked elements should have zero weight
    EXPECT_EQ(weights[1], 0);
    EXPECT_EQ(weights[3], 0);

    // Unmasked elements should have non-zero weight
    EXPECT_GT(weights[0], 0);
    EXPECT_GT(weights[2], 0);

    // Sum should only include unmasked
    EXPECT_EQ(sum, weights[0] + weights[2]);
}

TEST_F(Test__Exp2Core, ComputeBlockWeights_AllMasked)
{
    std::vector<int32_t> scores = {MASKED, MASKED, MASKED};
    std::vector<int32_t> weights(3, 999);

    int64_t sum = exp2_compute_block_weights(
        scores.data(), weights.data(), 3, 0, 0.1f);

    EXPECT_EQ(weights[0], 0);
    EXPECT_EQ(weights[1], 0);
    EXPECT_EQ(weights[2], 0);
    EXPECT_EQ(sum, 0);
}

TEST_F(Test__Exp2Core, ComputeBlockWeights_VsReference)
{
    // Compare block weights against FP32 reference
    std::vector<int32_t> scores = {100, 90, 80, 70, 60};
    std::vector<int32_t> weights(5);
    float alpha = 0.05f;

    int32_t max_score = 100;
    int64_t sum = exp2_compute_block_weights(
        scores.data(), weights.data(), 5, max_score, alpha);

    // Compute reference
    float ref_sum = 0;
    std::vector<float> ref_weights(5);
    for (int i = 0; i < 5; ++i)
    {
        ref_weights[i] = std::exp(-alpha * (max_score - scores[i]));
        ref_sum += ref_weights[i];
    }

    // Compare normalized distributions
    for (int i = 0; i < 5; ++i)
    {
        float expected = ref_weights[i] / ref_sum;
        float actual = static_cast<float>(weights[i]) / static_cast<float>(sum);
        EXPECT_NEAR(actual, expected, 0.02)
            << "Mismatch at index " << i;
    }
}

// ============================================================================
// Find Block Max Tests
// ============================================================================

TEST_F(Test__Exp2Core, FindBlockMax_BasicCase)
{
    std::vector<int32_t> scores = {10, 50, 30, 20};
    int32_t max = exp2_find_block_max(scores.data(), 4);
    EXPECT_EQ(max, 50);
}

TEST_F(Test__Exp2Core, FindBlockMax_WithMasked)
{
    std::vector<int32_t> scores = {10, MASKED, 30, MASKED};
    int32_t max = exp2_find_block_max(scores.data(), 4);
    EXPECT_EQ(max, 30);
}

TEST_F(Test__Exp2Core, FindBlockMax_AllMasked)
{
    std::vector<int32_t> scores = {MASKED, MASKED, MASKED};
    int32_t max = exp2_find_block_max(scores.data(), 3);
    EXPECT_EQ(max, MASKED);
}

TEST_F(Test__Exp2Core, FindBlockMax_NegativeScores)
{
    std::vector<int32_t> scores = {-100, -50, -200, -10};
    int32_t max = exp2_find_block_max(scores.data(), 4);
    EXPECT_EQ(max, -10);
}

TEST_F(Test__Exp2Core, FindBlockMax_SingleElement)
{
    std::vector<int32_t> scores = {42};
    int32_t max = exp2_find_block_max(scores.data(), 1);
    EXPECT_EQ(max, 42);
}

// ============================================================================
// Weight Normalization Tests
// ============================================================================

TEST_F(Test__Exp2Core, NormalizeWeights_SingleElement)
{
    std::vector<int32_t> block_weights = {static_cast<int32_t>(ONE)};
    std::vector<int16_t> final_weights(1);

    int32_t sum = exp2_normalize_weights(
        block_weights.data(), final_weights.data(), 1, ONE);

    EXPECT_EQ(final_weights[0], 32767);
    EXPECT_EQ(sum, 32767);
}

TEST_F(Test__Exp2Core, NormalizeWeights_TwoEqual)
{
    std::vector<int32_t> block_weights = {100, 100};
    std::vector<int16_t> final_weights(2);

    int32_t sum = exp2_normalize_weights(
        block_weights.data(), final_weights.data(), 2, 200);

    // Each should get half
    EXPECT_NEAR(final_weights[0], 32767 / 2, 1);
    EXPECT_NEAR(final_weights[1], 32767 / 2, 1);
}

TEST_F(Test__Exp2Core, NormalizeWeights_WithZeros)
{
    std::vector<int32_t> block_weights = {100, 0, 50, 0};
    std::vector<int16_t> final_weights(4);

    int32_t sum = exp2_normalize_weights(
        block_weights.data(), final_weights.data(), 4, 150);

    // Zeros should remain zero
    EXPECT_EQ(final_weights[1], 0);
    EXPECT_EQ(final_weights[3], 0);

    // Non-zeros should be proportional
    // 100/(100+50) ≈ 0.667 -> 21844
    // 50/(100+50) ≈ 0.333 -> 10922
    EXPECT_NEAR(final_weights[0], 21845, 10);
    EXPECT_NEAR(final_weights[2], 10922, 10);
}

TEST_F(Test__Exp2Core, NormalizeWeights_AllZero)
{
    std::vector<int32_t> block_weights = {0, 0, 0};
    std::vector<int16_t> final_weights(3, 999);

    int32_t sum = exp2_normalize_weights(
        block_weights.data(), final_weights.data(), 3, 0);

    EXPECT_EQ(final_weights[0], 0);
    EXPECT_EQ(final_weights[1], 0);
    EXPECT_EQ(final_weights[2], 0);
    EXPECT_EQ(sum, 0);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(Test__Exp2Core, EndToEnd_SimpleVector)
{
    // Complete flow: find max -> compute weights -> normalize

    std::vector<int32_t> scores = {100, 90, 80, 70};
    float alpha = 0.1f;

    // Step 1: Find max
    int32_t max = exp2_find_block_max(scores.data(), 4);
    EXPECT_EQ(max, 100);

    // Step 2: Compute weights
    std::vector<int32_t> block_weights(4);
    int64_t sum = exp2_compute_block_weights(
        scores.data(), block_weights.data(), 4, max, alpha);
    EXPECT_GT(sum, 0);

    // Step 3: Normalize
    std::vector<int16_t> final_weights(4);
    int32_t final_sum = exp2_normalize_weights(
        block_weights.data(), final_weights.data(), 4, sum);

    // Final weights should sum close to 32767
    int total = 0;
    for (int i = 0; i < 4; ++i)
    {
        total += final_weights[i];
    }
    EXPECT_NEAR(total, 32767, 4);
    EXPECT_NEAR(final_sum, 32767, 4);

    // Weights should be monotonically decreasing (higher score = higher weight)
    EXPECT_GT(final_weights[0], final_weights[1]);
    EXPECT_GT(final_weights[1], final_weights[2]);
    EXPECT_GT(final_weights[2], final_weights[3]);
}

TEST_F(Test__Exp2Core, EndToEnd_WithMasking)
{
    std::vector<int32_t> scores = {100, MASKED, 80, MASKED, 60};
    float alpha = 0.05f;

    int32_t max = exp2_find_block_max(scores.data(), 5);
    EXPECT_EQ(max, 100);

    std::vector<int32_t> block_weights(5);
    int64_t sum = exp2_compute_block_weights(
        scores.data(), block_weights.data(), 5, max, alpha);

    EXPECT_EQ(block_weights[1], 0);
    EXPECT_EQ(block_weights[3], 0);

    std::vector<int16_t> final_weights(5);
    exp2_normalize_weights(
        block_weights.data(), final_weights.data(), 5, sum);

    EXPECT_EQ(final_weights[1], 0);
    EXPECT_EQ(final_weights[3], 0);

    // Unmasked should have weight
    EXPECT_GT(final_weights[0], 0);
    EXPECT_GT(final_weights[2], 0);
    EXPECT_GT(final_weights[4], 0);
}

// ============================================================================
// Random Stress Tests
// ============================================================================

TEST_F(Test__Exp2Core, RandomStress_WeightComputation)
{
    // Test weight computation with random deltas and alphas
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> delta_dist(0, 10000);
    std::uniform_real_distribution<float> alpha_dist(0.001f, 1.0f);

    int errors = 0;
    for (int i = 0; i < 1000; ++i)
    {
        int32_t delta = delta_dist(rng);
        float alpha = alpha_dist(rng);

        AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);
        uint32_t weight = exp2_compute_weight(delta, config);

        // Reference
        double expected = std::exp(-alpha * delta);
        double actual = static_cast<double>(weight) / static_cast<double>(ONE);

        // Large tolerance for very small weights
        double tolerance = std::max(expected * 0.1, 1e-8);
        if (std::abs(actual - expected) > tolerance)
        {
            errors++;
        }
    }

    EXPECT_LT(errors, 10) << "Too many weight computation errors";
}

TEST_F(Test__Exp2Core, RandomStress_BlockWeights)
{
    // Test block weight computation with random vectors
    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> score_dist(-10000, 10000);
    std::uniform_real_distribution<float> alpha_dist(0.001f, 0.5f);
    std::uniform_int_distribution<int> size_dist(4, 128);

    for (int trial = 0; trial < 100; ++trial)
    {
        int n = size_dist(rng);
        float alpha = alpha_dist(rng);

        std::vector<int32_t> scores(n);
        for (int i = 0; i < n; ++i)
        {
            scores[i] = score_dist(rng);
        }

        int32_t max = exp2_find_block_max(scores.data(), n);
        std::vector<int32_t> weights(n);
        int64_t sum = exp2_compute_block_weights(
            scores.data(), weights.data(), n, max, alpha);

        // Sum should be positive if any scores exist
        ASSERT_GT(sum, 0) << "Trial " << trial;

        // All weights should be non-negative
        for (int i = 0; i < n; ++i)
        {
            ASSERT_GE(weights[i], 0) << "Trial " << trial << " index " << i;
        }

        // Max score should have highest weight
        for (int i = 0; i < n; ++i)
        {
            if (scores[i] == max)
            {
                EXPECT_EQ(weights[i], static_cast<int32_t>(ONE))
                    << "Max score should have full weight";
            }
        }
    }
}
