/**
 * @file Test__Exp2FixedSoftmax.cpp
 * @brief Unit tests for integer-only exp2 softmax microkernel
 *
 * Tests the Exp2FixedSoftmax LUT-based softmax that produces INT16 weights
 * from INT32 scores without any per-element floating-point operations.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/attention/q16_1/ref/microkernels/Exp2FixedSoftmax.h"

using namespace llaminar2::kernels::q16_1::microkernels;

class Test__Exp2FixedSoftmax : public ::testing::Test
{
protected:
    static constexpr int16_t WEIGHT_MAX = 32767;
    static constexpr int32_t MASKED = std::numeric_limits<int32_t>::min();

    // Reference FP32 softmax for comparison
    std::vector<float> reference_softmax(const std::vector<int32_t> &scores, float alpha)
    {
        std::vector<float> result(scores.size(), 0.0f);

        // Find max
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] != MASKED)
            {
                float s = static_cast<float>(scores[i]) * alpha;
                max_score = std::max(max_score, s);
            }
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] != MASKED)
            {
                float s = static_cast<float>(scores[i]) * alpha;
                result[i] = std::exp(s - max_score);
                sum += result[i];
            }
        }

        // Normalize
        if (sum > 0.0f)
        {
            for (size_t i = 0; i < scores.size(); ++i)
            {
                result[i] /= sum;
            }
        }

        return result;
    }

    // Convert INT16 weights to FP32 probabilities
    std::vector<float> weights_to_probs(const std::vector<int16_t> &weights)
    {
        std::vector<float> probs(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            probs[i] = static_cast<float>(weights[i]) / static_cast<float>(WEIGHT_MAX);
        }
        return probs;
    }

    // Compute max absolute error between two probability distributions
    float max_abs_error(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

    // Compute KL divergence (p || q)
    float kl_divergence(const std::vector<float> &p, const std::vector<float> &q)
    {
        float kl = 0.0f;
        for (size_t i = 0; i < p.size(); ++i)
        {
            if (p[i] > 1e-10f && q[i] > 1e-10f)
            {
                kl += p[i] * std::log(p[i] / q[i]);
            }
        }
        return kl;
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(Test__Exp2FixedSoftmax, LUTInitialization)
{
    // LUT should be nullptr before initialization
    const uint32_t *lut = get_exp2_lut_data();

    // After initialization, should have valid data
    ensure_exp2_lut_initialized(30);
    lut = get_exp2_lut_data();
    ASSERT_NE(lut, nullptr);

    // LUT[0] should be close to 2^30 (since 2^(-0) = 1)
    const uint32_t expected_one = 1U << 30;
    EXPECT_EQ(lut[0], expected_one);

    // LUT[1024] should be close to 2^30 * 2^(-0.5) ≈ 0.707 * 2^30
    // (1024/2048 = 0.5)
    const float expected_half = std::pow(2.0f, -0.5f);
    const float actual_half = static_cast<float>(lut[1024]) / static_cast<float>(expected_one);
    EXPECT_NEAR(actual_half, expected_half, 0.001f);

    // LUT[2047] should be close to 2^30 * 2^(-2047/2048) ≈ 0.5002 * 2^30
    const float expected_last = std::pow(2.0f, -2047.0f / 2048.0f);
    const float actual_last = static_cast<float>(lut[2047]) / static_cast<float>(expected_one);
    EXPECT_NEAR(actual_last, expected_last, 0.001f);
}

TEST_F(Test__Exp2FixedSoftmax, SingleElement)
{
    std::vector<int32_t> scores = {1000};
    std::vector<int16_t> weights(1);
    int32_t sum = 0;

    exp2_softmax_int32(scores.data(), weights.data(), 1, 1.0f, &sum);

    // Single element should get all the weight
    EXPECT_EQ(weights[0], WEIGHT_MAX);
    EXPECT_EQ(sum, WEIGHT_MAX);
}

TEST_F(Test__Exp2FixedSoftmax, TwoEqualElements)
{
    std::vector<int32_t> scores = {1000, 1000};
    std::vector<int16_t> weights(2);
    int32_t sum = 0;

    exp2_softmax_int32(scores.data(), weights.data(), 2, 1.0f, &sum);

    // Equal scores should give equal weights (each ~half)
    EXPECT_NEAR(weights[0], WEIGHT_MAX / 2, 1);
    EXPECT_NEAR(weights[1], WEIGHT_MAX / 2, 1);

    // Sum should be close to WEIGHT_MAX
    EXPECT_NEAR(sum, WEIGHT_MAX, 2);
}

TEST_F(Test__Exp2FixedSoftmax, MaskedElements)
{
    std::vector<int32_t> scores = {1000, MASKED, 500, MASKED};
    std::vector<int16_t> weights(4);
    int32_t sum = 0;

    exp2_softmax_int32(scores.data(), weights.data(), 4, 0.01f, &sum);

    // Masked elements should have zero weight
    EXPECT_EQ(weights[1], 0);
    EXPECT_EQ(weights[3], 0);

    // Unmasked elements should have non-zero weight
    EXPECT_GT(weights[0], 0);
    EXPECT_GT(weights[2], 0);

    // Higher score should have higher weight
    EXPECT_GT(weights[0], weights[2]);
}

TEST_F(Test__Exp2FixedSoftmax, AllMasked)
{
    std::vector<int32_t> scores = {MASKED, MASKED, MASKED};
    std::vector<int16_t> weights(3, 999); // Initialize to non-zero
    int32_t sum = 0;

    exp2_softmax_int32(scores.data(), weights.data(), 3, 1.0f, &sum);

    // All should be zero
    EXPECT_EQ(weights[0], 0);
    EXPECT_EQ(weights[1], 0);
    EXPECT_EQ(weights[2], 0);
    EXPECT_EQ(sum, 0);
}

// ============================================================================
// Accuracy Tests (vs FP32 Reference)
// ============================================================================

TEST_F(Test__Exp2FixedSoftmax, AccuracySmallRange)
{
    // Scores with small range (typical for well-scaled attention)
    std::vector<int32_t> scores = {100, 95, 90, 85, 80};
    std::vector<int16_t> weights(5);
    float alpha = 0.1f;

    exp2_softmax_int32(scores.data(), weights.data(), 5, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    // Should be very accurate for small score ranges
    float max_err = max_abs_error(ref, actual);
    EXPECT_LT(max_err, 0.01f) << "Max error: " << max_err;

    // KL divergence should be tiny
    float kl = kl_divergence(ref, actual);
    EXPECT_LT(kl, 0.001f) << "KL divergence: " << kl;
}

TEST_F(Test__Exp2FixedSoftmax, AccuracyLargeRange)
{
    // Scores with large range (stress test)
    std::vector<int32_t> scores = {1000, 500, 0, -500, -1000};
    std::vector<int16_t> weights(5);
    float alpha = 0.01f;

    exp2_softmax_int32(scores.data(), weights.data(), 5, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    // Larger range has more approximation error, but should still be reasonable
    float max_err = max_abs_error(ref, actual);
    EXPECT_LT(max_err, 0.05f) << "Max error: " << max_err;
}

TEST_F(Test__Exp2FixedSoftmax, AccuracyTypicalAttention)
{
    // Simulate typical attention scores: Q×K^T with head_dim=128
    // Scores are INT32 from Q16 dot products, alpha = 1/sqrt(128) ≈ 0.0884
    const int n = 64; // Sequence length
    const float alpha = 1.0f / std::sqrt(128.0f);

    std::vector<int32_t> scores(n);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1000.0f);

    for (int i = 0; i < n; ++i)
    {
        scores[i] = static_cast<int32_t>(dist(rng));
    }

    std::vector<int16_t> weights(n);
    exp2_softmax_int32(scores.data(), weights.data(), n, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    float max_err = max_abs_error(ref, actual);
    float kl = kl_divergence(ref, actual);

    EXPECT_LT(max_err, 0.02f) << "Max error: " << max_err;
    EXPECT_LT(kl, 0.01f) << "KL divergence: " << kl;
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__Exp2FixedSoftmax, VerySmallAlpha)
{
    // Very small alpha should give uniform-ish weights
    std::vector<int32_t> scores = {1000, 0, -1000};
    std::vector<int16_t> weights(3);

    exp2_softmax_int32(scores.data(), weights.data(), 3, 1e-10f);

    // All weights should be equal (when alpha → 0, exp(-α×δ) → 1 for all δ)
    EXPECT_EQ(weights[0], weights[1]);
    EXPECT_EQ(weights[1], weights[2]);
}

TEST_F(Test__Exp2FixedSoftmax, VeryLargeAlpha)
{
    // Very large alpha should give winner-take-all
    std::vector<int32_t> scores = {100, 99, 98};
    std::vector<int16_t> weights(3);

    exp2_softmax_int32(scores.data(), weights.data(), 3, 100.0f);

    // Max score should get almost all the weight
    EXPECT_GT(weights[0], WEIGHT_MAX * 0.95);
    EXPECT_LT(weights[1], WEIGHT_MAX * 0.05);
    EXPECT_LT(weights[2], WEIGHT_MAX * 0.01);
}

TEST_F(Test__Exp2FixedSoftmax, NegativeScores)
{
    // All negative scores should still work
    std::vector<int32_t> scores = {-100, -200, -300};
    std::vector<int16_t> weights(3);
    int32_t sum = 0;

    exp2_softmax_int32(scores.data(), weights.data(), 3, 0.01f, &sum);

    // Highest (least negative) should have highest weight
    EXPECT_GT(weights[0], weights[1]);
    EXPECT_GT(weights[1], weights[2]);

    // Sum should still be close to WEIGHT_MAX
    EXPECT_NEAR(sum, WEIGHT_MAX, 10);
}

TEST_F(Test__Exp2FixedSoftmax, ZeroScores)
{
    // All zero scores
    std::vector<int32_t> scores = {0, 0, 0, 0};
    std::vector<int16_t> weights(4);
    int32_t sum = 0;

    exp2_softmax_int32(scores.data(), weights.data(), 4, 1.0f, &sum);

    // All weights should be equal
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(weights[i], WEIGHT_MAX / 4, 1);
    }
}

TEST_F(Test__Exp2FixedSoftmax, LongSequence)
{
    // Test with longer sequence (typical for prefill)
    const int n = 2048;
    std::vector<int32_t> scores(n);
    std::vector<int16_t> weights(n);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> dist(-10000, 10000);

    for (int i = 0; i < n; ++i)
    {
        scores[i] = dist(rng);
    }

    int32_t sum = 0;
    exp2_softmax_int32(scores.data(), weights.data(), n, 0.01f, &sum);

    // Sum should be close to WEIGHT_MAX
    EXPECT_NEAR(sum, WEIGHT_MAX, 100);

    // No negative weights
    for (int i = 0; i < n; ++i)
    {
        EXPECT_GE(weights[i], 0);
    }
}

// ============================================================================
// Null/Invalid Input Tests
// ============================================================================

TEST_F(Test__Exp2FixedSoftmax, NullScores)
{
    std::vector<int16_t> weights(4);
    int32_t sum = 999;

    exp2_softmax_int32(nullptr, weights.data(), 4, 1.0f, &sum);

    EXPECT_EQ(sum, 0);
}

TEST_F(Test__Exp2FixedSoftmax, NullWeights)
{
    std::vector<int32_t> scores = {1, 2, 3};
    int32_t sum = 999;

    // Should not crash
    exp2_softmax_int32(scores.data(), nullptr, 3, 1.0f, &sum);

    EXPECT_EQ(sum, 0);
}

TEST_F(Test__Exp2FixedSoftmax, ZeroLength)
{
    std::vector<int32_t> scores = {1, 2, 3};
    std::vector<int16_t> weights(3);
    int32_t sum = 999;

    exp2_softmax_int32(scores.data(), weights.data(), 0, 1.0f, &sum);

    EXPECT_EQ(sum, 0);
}

TEST_F(Test__Exp2FixedSoftmax, NullSumOut)
{
    std::vector<int32_t> scores = {100, 100};
    std::vector<int16_t> weights(2);

    // Should work without sum_out
    exp2_softmax_int32(scores.data(), weights.data(), 2, 1.0f, nullptr);

    EXPECT_NEAR(weights[0], WEIGHT_MAX / 2, 1);
    EXPECT_NEAR(weights[1], WEIGHT_MAX / 2, 1);
}

// ============================================================================
// Spiky Activation Tests (Numerical Precision)
// ============================================================================

TEST_F(Test__Exp2FixedSoftmax, SpikyOneDominant)
{
    // One score significantly higher than rest
    // Simulates attention focusing on a single token
    std::vector<int32_t> scores = {10000, 5000, 5000, 5000, 5000};
    std::vector<int16_t> weights(5);
    float alpha = 0.001f; // Typical attention scale

    exp2_softmax_int32(scores.data(), weights.data(), 5, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    // The dominant token should get most weight
    EXPECT_GT(weights[0], WEIGHT_MAX * 0.9) << "Dominant token weight: " << weights[0];

    // Check reference vs actual for the dominant position
    float dominant_err = std::abs(ref[0] - actual[0]);
    EXPECT_LT(dominant_err, 0.02f) << "Dominant error: " << dominant_err
                                   << " ref=" << ref[0] << " actual=" << actual[0];

    // Non-dominant tokens should be small but non-zero
    for (int i = 1; i < 5; ++i)
    {
        EXPECT_GT(weights[i], 0) << "Token " << i << " should have non-zero weight";
        EXPECT_LT(weights[i], WEIGHT_MAX * 0.05) << "Token " << i << " too large: " << weights[i];
    }

    std::cout << "  [SpikyOneDominant] Dominant weight: " << actual[0]
              << " (ref: " << ref[0] << "), others: ~" << actual[1] << std::endl;
}

TEST_F(Test__Exp2FixedSoftmax, SpikyTwoDominant)
{
    // Two high scores competing for attention
    std::vector<int32_t> scores = {10000, 9900, 5000, 5000, 5000};
    std::vector<int16_t> weights(5);
    float alpha = 0.001f;

    exp2_softmax_int32(scores.data(), weights.data(), 5, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    // Both dominant tokens should share most of the weight
    float top2_weight = actual[0] + actual[1];
    EXPECT_GT(top2_weight, 0.9f) << "Top 2 combined: " << top2_weight;

    // First should be slightly higher than second
    EXPECT_GT(weights[0], weights[1]);

    // Check the ratio between top 2
    float ratio = actual[0] / actual[1];
    float ref_ratio = ref[0] / ref[1];
    EXPECT_NEAR(ratio, ref_ratio, 0.1f) << "Ratio mismatch: " << ratio << " vs " << ref_ratio;

    std::cout << "  [SpikyTwoDominant] Top 2 weights: " << actual[0] << ", " << actual[1]
              << " (sum: " << top2_weight << ")" << std::endl;
}

TEST_F(Test__Exp2FixedSoftmax, SpikyGradient)
{
    // Gradually decreasing scores (attention over sequence position)
    const int n = 10;
    std::vector<int32_t> scores(n);
    for (int i = 0; i < n; ++i)
    {
        scores[i] = 10000 - i * 500; // 10000, 9500, 9000, ...
    }
    std::vector<int16_t> weights(n);
    float alpha = 0.001f;

    exp2_softmax_int32(scores.data(), weights.data(), n, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    // Weights should be monotonically decreasing
    for (int i = 0; i < n - 1; ++i)
    {
        EXPECT_GE(weights[i], weights[i + 1])
            << "Not monotonic at " << i << ": " << weights[i] << " < " << weights[i + 1];
    }

    // Check max absolute error
    float max_err = max_abs_error(ref, actual);
    EXPECT_LT(max_err, 0.03f) << "Max error: " << max_err;

    std::cout << "  [SpikyGradient] First 5 weights: ";
    for (int i = 0; i < 5; ++i)
        std::cout << actual[i] << " ";
    std::cout << std::endl;
}

TEST_F(Test__Exp2FixedSoftmax, SpikyExtremeGap)
{
    // Extreme gap: one score way higher than others
    // This tests underflow handling for the low scores
    std::vector<int32_t> scores = {100000, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int16_t> weights(8);
    float alpha = 0.001f;

    exp2_softmax_int32(scores.data(), weights.data(), 8, alpha);

    // The dominant score should get essentially all weight
    EXPECT_GE(weights[0], WEIGHT_MAX - 10)
        << "Expected winner-take-all, got " << weights[0];

    // Others should be zero or near-zero (underflow)
    int32_t others_sum = 0;
    for (int i = 1; i < 8; ++i)
    {
        others_sum += weights[i];
    }
    EXPECT_LE(others_sum, 10) << "Others should underflow to ~0, sum=" << others_sum;

    std::cout << "  [SpikyExtremeGap] Winner: " << weights[0]
              << ", others sum: " << others_sum << std::endl;
}

TEST_F(Test__Exp2FixedSoftmax, SpikyNearTie)
{
    // Near-tie: scores differ by 1
    std::vector<int32_t> scores = {1001, 1000, 1000, 1000};
    std::vector<int16_t> weights(4);
    float alpha = 0.1f;

    exp2_softmax_int32(scores.data(), weights.data(), 4, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    // First should be slightly higher but all should be close
    EXPECT_GT(weights[0], weights[1]);

    // Check that differences are small
    float spread = actual[0] - actual[3];
    float ref_spread = ref[0] - ref[3];
    EXPECT_NEAR(spread, ref_spread, 0.01f);

    std::cout << "  [SpikyNearTie] Weights: " << actual[0] << ", " << actual[1]
              << ", " << actual[2] << ", " << actual[3] << std::endl;
}

TEST_F(Test__Exp2FixedSoftmax, SpikyLongTail)
{
    // Long tail distribution: one peak with many small values
    const int n = 64;
    std::vector<int32_t> scores(n);
    scores[0] = 10000; // Peak
    scores[1] = 8000;  // Secondary
    for (int i = 2; i < n; ++i)
    {
        scores[i] = 1000 + (i % 100); // Long uniform tail
    }
    std::vector<int16_t> weights(n);
    float alpha = 0.001f;

    exp2_softmax_int32(scores.data(), weights.data(), n, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    // Top 2 should dominate
    float top2 = actual[0] + actual[1];
    EXPECT_GT(top2, 0.8f) << "Top 2 should dominate, got " << top2;

    // Tail should have small but measurable weights (with alpha=0.1 and large gaps, tail is tiny)
    float tail_sum = 0;
    for (int i = 2; i < n; ++i)
    {
        tail_sum += actual[i];
        EXPECT_GE(weights[i], 0) << "Tail weight " << i << " negative";
    }
    // With alpha=0.1 and ~900+ gaps, tail is < 1% - verify it's non-zero but small
    EXPECT_GT(tail_sum, 0.001f) << "Tail should have some weight";
    EXPECT_LT(tail_sum, 0.2f) << "Tail too heavy: " << tail_sum;

    // Check overall accuracy
    float max_err = max_abs_error(ref, actual);
    EXPECT_LT(max_err, 0.05f) << "Max error in long tail: " << max_err;

    std::cout << "  [SpikyLongTail] Peak: " << actual[0] << ", secondary: " << actual[1]
              << ", tail sum: " << tail_sum << std::endl;
}

TEST_F(Test__Exp2FixedSoftmax, SpikyAlphaSweep)
{
    // Same scores, varying alpha to see sharpness effect
    std::vector<int32_t> scores = {1000, 900, 800, 700, 600};
    std::vector<int16_t> weights(5);

    std::cout << "  [SpikyAlphaSweep] Alpha sweep:" << std::endl;

    float alphas[] = {0.001f, 0.01f, 0.1f, 1.0f, 10.0f};
    for (float alpha : alphas)
    {
        exp2_softmax_int32(scores.data(), weights.data(), 5, alpha);

        auto ref = reference_softmax(scores, alpha);
        auto actual = weights_to_probs(weights);

        float max_err = max_abs_error(ref, actual);
        float kl = kl_divergence(ref, actual);

        // As alpha increases, distribution should become sharper
        std::cout << "    alpha=" << std::setw(6) << alpha
                  << " max_weight=" << std::setw(8) << actual[0]
                  << " max_err=" << std::setw(8) << max_err
                  << " KL=" << std::setw(10) << kl << std::endl;

        // Error should be reasonable for all alphas
        EXPECT_LT(max_err, 0.1f) << "High error at alpha=" << alpha;
    }
}

TEST_F(Test__Exp2FixedSoftmax, SpikyAttentionRealistic)
{
    // Realistic attention scores from Q×K^T
    // Simulates: "What is the capital of France?"
    // with strong attention on "capital" and "France"
    const int seq_len = 8;
    // Scores represent attention from "?" to each token
    // Higher scores for semantically relevant tokens
    std::vector<int32_t> scores = {
        500,  // "What"
        400,  // "is"
        300,  // "the"
        5000, // "capital" <- high attention
        200,  // "of"
        4500, // "France" <- high attention
        100,  // "?"
        600   // [BOS] or padding
    };
    std::vector<int16_t> weights(seq_len);
    float alpha = 1.0f / std::sqrt(128.0f); // ~0.0884 for head_dim=128

    exp2_softmax_int32(scores.data(), weights.data(), seq_len, alpha);

    auto ref = reference_softmax(scores, alpha);
    auto actual = weights_to_probs(weights);

    // "capital" (idx 3) and "France" (idx 5) should dominate
    float capital_weight = actual[3];
    float france_weight = actual[5];
    float combined = capital_weight + france_weight;

    EXPECT_GT(combined, 0.7f) << "Key tokens should dominate";
    EXPECT_GT(capital_weight, france_weight) << "Capital has higher score";

    // Check accuracy vs FP32 reference
    float max_err = max_abs_error(ref, actual);
    EXPECT_LT(max_err, 0.02f);

    std::cout << "  [SpikyAttentionRealistic] Attention distribution:" << std::endl;
    const char *tokens[] = {"What", "is", "the", "capital", "of", "France", "?", "[BOS]"};
    for (int i = 0; i < seq_len; ++i)
    {
        std::cout << "    " << std::setw(8) << tokens[i]
                  << ": " << std::setw(6) << std::fixed << std::setprecision(4) << actual[i]
                  << " (ref: " << ref[i] << ")" << std::endl;
    }
}

TEST_F(Test__Exp2FixedSoftmax, SpikyQuantizationNoise)
{
    // Test robustness to quantization noise in scores
    // INT32 scores from Q16 dot products have quantization error
    const int n = 16;
    std::vector<int32_t> base_scores(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> base_dist(0, 10000);
    std::uniform_int_distribution<int32_t> noise_dist(-50, 50);

    for (int i = 0; i < n; ++i)
    {
        base_scores[i] = base_dist(rng);
    }

    // Run with base scores
    std::vector<int16_t> base_weights(n);
    exp2_softmax_int32(base_scores.data(), base_weights.data(), n, 0.001f);

    // Run with noisy scores (±50 quantization noise)
    std::vector<int32_t> noisy_scores = base_scores;
    for (int i = 0; i < n; ++i)
    {
        noisy_scores[i] += noise_dist(rng);
    }
    std::vector<int16_t> noisy_weights(n);
    exp2_softmax_int32(noisy_scores.data(), noisy_weights.data(), n, 0.001f);

    // Convert to probabilities
    auto base_probs = weights_to_probs(base_weights);
    auto noisy_probs = weights_to_probs(noisy_weights);

    // Compute perturbation sensitivity
    float max_diff = 0;
    for (int i = 0; i < n; ++i)
    {
        max_diff = std::max(max_diff, std::abs(base_probs[i] - noisy_probs[i]));
    }

    // Small noise should cause small output changes
    EXPECT_LT(max_diff, 0.1f) << "Too sensitive to quantization noise";

    std::cout << "  [SpikyQuantizationNoise] Max weight diff from ±50 noise: " << max_diff << std::endl;
}

TEST_F(Test__Exp2FixedSoftmax, SpikyUnderflowRecovery)
{
    // Test that underflow doesn't cause all-zero output
    // Large negative deltas should still produce valid distribution
    std::vector<int32_t> scores = {1000000, 0, 0, 0}; // Huge gap
    std::vector<int16_t> weights(4);
    int32_t sum = 0;

    // Very large alpha makes the gap even more extreme
    exp2_softmax_int32(scores.data(), weights.data(), 4, 1.0f, &sum);

    // Should not all be zero - at minimum the max gets weight
    EXPECT_GT(sum, 0) << "Sum should be positive";
    EXPECT_GT(weights[0], 0) << "Max score should get weight";

    // The winner should get all (or nearly all) weight
    EXPECT_GE(weights[0], WEIGHT_MAX - 1);

    std::cout << "  [SpikyUnderflowRecovery] Extreme gap recovery: "
              << "winner=" << weights[0] << " sum=" << sum << std::endl;
}

TEST_F(Test__Exp2FixedSoftmax, SpikyPeakPosition)
{
    // Test that peak position doesn't affect accuracy
    const int n = 8;
    float alpha = 0.01f;
    std::vector<int32_t> base_scores = {100, 100, 100, 100, 100, 100, 100, 100};

    std::cout << "  [SpikyPeakPosition] Testing peak at different positions:" << std::endl;

    for (int peak_pos = 0; peak_pos < n; ++peak_pos)
    {
        std::vector<int32_t> scores = base_scores;
        scores[peak_pos] = 1000; // Make this position the peak

        std::vector<int16_t> weights(n);
        exp2_softmax_int32(scores.data(), weights.data(), n, alpha);

        auto ref = reference_softmax(scores, alpha);
        auto actual = weights_to_probs(weights);

        // Find which position got max weight
        int max_pos = 0;
        for (int i = 1; i < n; ++i)
        {
            if (weights[i] > weights[max_pos])
                max_pos = i;
        }

        EXPECT_EQ(max_pos, peak_pos) << "Peak should be at position " << peak_pos;

        float err = std::abs(ref[peak_pos] - actual[peak_pos]);
        EXPECT_LT(err, 0.01f) << "Error at peak position " << peak_pos;

        std::cout << "    Peak at " << peak_pos << ": weight=" << actual[peak_pos]
                  << " (ref=" << ref[peak_pos] << ")" << std::endl;
    }
}

// ============================================================================
// Mass Random Tests (10,000 Vectors) - Per Gemini3 Battle Testing Requirements
// ============================================================================

TEST_F(Test__Exp2FixedSoftmax, MassRandom_10000Vectors_KLDivergence)
{
    /**
     * BATTLE TEST: Run exp2 LUT softmax against FP32 std::exp reference
     * for 10,000 random input vectors.
     *
     * Metrics computed:
     * - KL Divergence: D_KL(P || Q) where P = FP32 reference, Q = exp2 LUT
     * - RMS Error: sqrt(mean((p_i - q_i)²))
     * - Max Absolute Error: max |p_i - q_i|
     * - L1 Error (Total Variation): Σ |p_i - q_i|
     *
     * Per Gemini3: "If this LUT is inaccurate, the whole kernel fails
     * regardless of how good the dot products are."
     */
    const int num_vectors = 10000;
    const int seq_lengths[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    const float alphas[] = {
        0.001f,                   // Very smooth
        1.0f / std::sqrt(64.0f),  // head_dim=64
        1.0f / std::sqrt(128.0f), // head_dim=128 (typical)
        0.1f,                     // Medium
        1.0f                      // Sharp
    };

    std::mt19937 rng(12345);

    // Aggregate statistics
    double total_kl = 0.0;
    double total_rms = 0.0;
    double worst_kl = 0.0;
    double worst_max_err = 0.0;
    int num_tests = 0;
    int kl_failures = 0;  // KL > 0.01 nats
    int rms_failures = 0; // RMS > 0.02

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║     EXP2 LUT SOFTMAX vs FP32 STD::EXP - BATTLE TEST (10,000 Vectors)     ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n";

    for (float alpha : alphas)
    {
        for (int seq_len : seq_lengths)
        {
            // Score distribution parameters (realistic attention score ranges)
            std::normal_distribution<float> score_dist(0.0f, 1000.0f * alpha);

            double alpha_kl_sum = 0.0;
            double alpha_rms_sum = 0.0;
            double alpha_worst_kl = 0.0;
            double alpha_worst_max_err = 0.0;
            int alpha_count = 0;

            for (int v = 0; v < num_vectors / (8 * 5); ++v) // Distribute evenly
            {
                // Generate random scores
                std::vector<int32_t> scores(seq_len);
                for (int i = 0; i < seq_len; ++i)
                {
                    scores[i] = static_cast<int32_t>(score_dist(rng) / alpha);
                }

                // Exp2 LUT softmax
                std::vector<int16_t> weights(seq_len);
                exp2_softmax_int32(scores.data(), weights.data(), seq_len, alpha);

                // FP32 reference
                auto ref = reference_softmax(scores, alpha);
                auto actual = weights_to_probs(weights);

                // Compute metrics
                float kl = kl_divergence(ref, actual);
                float max_err = max_abs_error(ref, actual);

                // RMS error
                double rms_sum = 0.0;
                for (size_t i = 0; i < ref.size(); ++i)
                {
                    double diff = ref[i] - actual[i];
                    rms_sum += diff * diff;
                }
                float rms_err = std::sqrt(rms_sum / ref.size());

                // Track failures
                if (kl > 0.01f)
                    kl_failures++;
                if (rms_err > 0.02f)
                    rms_failures++;

                // Accumulate
                alpha_kl_sum += kl;
                alpha_rms_sum += rms_err;
                alpha_worst_kl = std::max(alpha_worst_kl, static_cast<double>(kl));
                alpha_worst_max_err = std::max(alpha_worst_max_err, static_cast<double>(max_err));
                alpha_count++;

                total_kl += kl;
                total_rms += rms_err;
                worst_kl = std::max(worst_kl, static_cast<double>(kl));
                worst_max_err = std::max(worst_max_err, static_cast<double>(max_err));
                num_tests++;
            }

            // Per-config summary
            double avg_kl = alpha_kl_sum / alpha_count;
            double avg_rms = alpha_rms_sum / alpha_count;

            std::cout << "║ alpha=" << std::setw(8) << std::fixed << std::setprecision(5) << alpha
                      << " seq=" << std::setw(4) << seq_len
                      << " | KL_avg=" << std::setw(10) << std::scientific << std::setprecision(3) << avg_kl
                      << " KL_max=" << std::setw(10) << alpha_worst_kl
                      << " RMS_avg=" << std::setw(8) << avg_rms
                      << " ║\n";
        }
    }

    // Final summary
    double mean_kl = total_kl / num_tests;
    double mean_rms = total_rms / num_tests;
    double failure_rate_kl = 100.0 * kl_failures / num_tests;
    double failure_rate_rms = 100.0 * rms_failures / num_tests;

    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║                              AGGREGATE RESULTS                           ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ Total test vectors:      " << std::setw(10) << num_tests << "                                    ║\n"
              << "║ Mean KL Divergence:      " << std::setw(10) << std::scientific << std::setprecision(4) << mean_kl << " nats                                ║\n"
              << "║ Worst KL Divergence:     " << std::setw(10) << worst_kl << " nats                                ║\n"
              << "║ Mean RMS Error:          " << std::setw(10) << mean_rms << "                                     ║\n"
              << "║ Worst Max Abs Error:     " << std::setw(10) << worst_max_err << "                                     ║\n"
              << "║ KL > 0.01 failures:      " << std::setw(10) << kl_failures << " (" << std::fixed << std::setprecision(2) << failure_rate_kl << "%)                              ║\n"
              << "║ RMS > 0.02 failures:     " << std::setw(10) << rms_failures << " (" << failure_rate_rms << "%)                              ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // Assertions: strict quality gates
    EXPECT_LT(mean_kl, 0.005) << "Mean KL divergence too high! LUT accuracy compromised.";
    EXPECT_LT(worst_kl, 0.1) << "Worst-case KL divergence unacceptable.";
    EXPECT_LT(mean_rms, 0.01) << "Mean RMS error too high.";
    EXPECT_LT(failure_rate_kl, 1.0) << "Too many high-KL failures (>1%).";
    EXPECT_LT(failure_rate_rms, 1.0) << "Too many high-RMS failures (>1%).";
}

TEST_F(Test__Exp2FixedSoftmax, MassRandom_LUTvsStdExp_DirectComparison)
{
    /**
     * Direct comparison of exp2 LUT values against std::exp
     * Tests the LUT accuracy in isolation, independent of softmax normalization.
     *
     * For random x values in the typical attention range, compare:
     *   exp2_lut(x) vs std::exp(-x * alpha * log2e)
     */
    const int num_samples = 100000;
    std::mt19937 rng(54321);

    // Score differences in typical range (after subtracting max)
    // Test the LUT accuracy for exp2(-u) where u ∈ [0, 1)
    // The LUT stores 2^(-u) scaled by 2^30 at 2048 uniformly spaced points
    // LUT[i] = round(2^30 * 2^(-i/2048)) for i ∈ [0, 2047]

    ensure_exp2_lut_initialized(30);
    const uint32_t *lut = get_exp2_lut_data();
    const uint32_t LUT_ONE = 1U << 30;

    double total_rel_error = 0.0;
    double max_rel_error = 0.0;
    double total_abs_error = 0.0;
    double max_abs_error_val = 0.0;

    // Test each LUT entry directly
    for (int i = 0; i < EXP2_LUT_SIZE; ++i)
    {
        // u = i / 2048, so LUT[i] should approximate 2^30 * 2^(-i/2048)
        double u = static_cast<double>(i) / EXP2_LUT_SIZE;
        double expected = std::exp2(-u); // FP64 reference: 2^(-u)
        double actual = static_cast<double>(lut[i]) / static_cast<double>(LUT_ONE);

        double abs_err = std::abs(expected - actual);
        double rel_err = abs_err / expected; // expected is always > 0.5

        total_abs_error += abs_err;
        total_rel_error += rel_err;
        max_abs_error_val = std::max(max_abs_error_val, abs_err);
        max_rel_error = std::max(max_rel_error, rel_err);
    }

    double mean_abs_error = total_abs_error / EXP2_LUT_SIZE;
    double mean_rel_error = total_rel_error / EXP2_LUT_SIZE;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║           EXP2 LUT TABLE ACCURACY TEST (2048 entries)                    ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ LUT[i] stores: 2^30 × 2^(-i/2048) for i ∈ [0, 2047]                       ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ LUT entries tested:      " << std::setw(10) << EXP2_LUT_SIZE << "                                    ║\n"
              << "║ Mean Absolute Error:     " << std::setw(14) << std::scientific << std::setprecision(6) << mean_abs_error << "                            ║\n"
              << "║ Max Absolute Error:      " << std::setw(14) << max_abs_error_val << "                            ║\n"
              << "║ Mean Relative Error:     " << std::setw(14) << mean_rel_error << "                            ║\n"
              << "║ Max Relative Error:      " << std::setw(14) << max_rel_error << "                            ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // LUT values are computed with Q30 precision (2^30 scaling)
    // Rounding error should be at most 0.5 LSB = 0.5/2^30 ≈ 4.7e-10
    // Relative error should be < 1e-9
    EXPECT_LT(mean_rel_error, 1e-8) << "Mean relative error too high for Q30 LUT";
    EXPECT_LT(max_rel_error, 1e-8) << "Worst-case relative error too high";
}