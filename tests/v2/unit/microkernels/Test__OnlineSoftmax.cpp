/**
 * @file Test__OnlineSoftmax.cpp
 * @brief Unit tests for online softmax microkernel
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "kernels/cpu/attention/q8_1/microkernels/OnlineSoftmax.h"

#include <cmath>
#include <numeric>
#include <vector>

using namespace llaminar::v2::kernels::microkernels;

class Test__OnlineSoftmax : public ::testing::Test
{
protected:
    // Compute offline softmax for reference
    std::vector<float> softmax_reference(const std::vector<float> &scores)
    {
        if (scores.empty())
            return {};

        // Find max for numerical stability
        float max_score = *std::max_element(scores.begin(), scores.end());

        // Compute exp(x - max) and sum
        std::vector<float> exp_scores(scores.size());
        float sum_exp = 0.0f;
        for (size_t i = 0; i < scores.size(); ++i)
        {
            exp_scores[i] = std::exp(scores[i] - max_score);
            sum_exp += exp_scores[i];
        }

        // Normalize
        for (auto &e : exp_scores)
        {
            e /= sum_exp;
        }

        return exp_scores;
    }
};

TEST_F(Test__OnlineSoftmax, SingleScore)
{
    OnlineSoftmaxState state = online_softmax_init();

    float weight = online_softmax_update(state, 1.0f);
    float inv_sum = online_softmax_finalize(state);

    // Single score: softmax([x]) = [1.0]
    EXPECT_FLOAT_EQ(weight * inv_sum, 1.0f);
}

TEST_F(Test__OnlineSoftmax, TwoEqualScores)
{
    OnlineSoftmaxState state = online_softmax_init();

    float weight1 = online_softmax_update(state, 5.0f);
    float prev_max = state.max_score;
    float weight2 = online_softmax_update(state, 5.0f);

    // No max change, so no correction needed
    EXPECT_FALSE(online_softmax_max_changed(state, prev_max));

    float inv_sum = online_softmax_finalize(state);

    // Two equal scores: softmax([x, x]) = [0.5, 0.5]
    // Both weights should be 1 (before normalization), sum_exp = 2
    EXPECT_NEAR(weight1 * inv_sum, 0.5f, 1e-6f);
    EXPECT_NEAR(weight2 * inv_sum, 0.5f, 1e-6f);
}

TEST_F(Test__OnlineSoftmax, IncreasingSequence)
{
    // [1, 2, 3] - each score triggers max update
    std::vector<float> scores = {1.0f, 2.0f, 3.0f};
    auto reference = softmax_reference(scores);

    OnlineSoftmaxState state = online_softmax_init();
    std::vector<float> weights(3);

    // Track accumulated weighted values (simulating V accumulation)
    std::vector<float> accum = {1.0f, 1.0f, 1.0f}; // Pretend V = [1, 1, 1]

    for (size_t i = 0; i < scores.size(); ++i)
    {
        float prev_max = state.max_score;
        weights[i] = online_softmax_update(state, scores[i]);

        if (online_softmax_max_changed(state, prev_max) && i > 0)
        {
            // Apply correction to previous accumulations
            float correction = online_softmax_correction(prev_max, state.max_score);
            for (size_t j = 0; j < i; ++j)
            {
                weights[j] *= correction;
            }
        }
    }

    float inv_sum = online_softmax_finalize(state);

    // Verify final normalized weights match reference
    for (size_t i = 0; i < scores.size(); ++i)
    {
        EXPECT_NEAR(weights[i] * inv_sum, reference[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST_F(Test__OnlineSoftmax, DecreasingSequence)
{
    // [3, 2, 1] - no max updates after first
    std::vector<float> scores = {3.0f, 2.0f, 1.0f};
    auto reference = softmax_reference(scores);

    OnlineSoftmaxState state = online_softmax_init();
    std::vector<float> weights(3);

    for (size_t i = 0; i < scores.size(); ++i)
    {
        float prev_max = state.max_score;
        weights[i] = online_softmax_update(state, scores[i]);

        // After first score, max shouldn't change
        if (i > 0)
        {
            EXPECT_FALSE(online_softmax_max_changed(state, prev_max));
        }
    }

    float inv_sum = online_softmax_finalize(state);

    for (size_t i = 0; i < scores.size(); ++i)
    {
        EXPECT_NEAR(weights[i] * inv_sum, reference[i], 1e-5f);
    }
}

TEST_F(Test__OnlineSoftmax, LargeDynamicRange)
{
    // Test numerical stability with large differences
    std::vector<float> scores = {-100.0f, 0.0f, 10.0f};
    auto reference = softmax_reference(scores);

    OnlineSoftmaxState state = online_softmax_init();
    std::vector<float> weights(3);

    for (size_t i = 0; i < scores.size(); ++i)
    {
        float prev_max = state.max_score;
        weights[i] = online_softmax_update(state, scores[i]);

        if (online_softmax_max_changed(state, prev_max) && state.initialized)
        {
            float correction = online_softmax_correction(prev_max, state.max_score);
            for (size_t j = 0; j < i; ++j)
            {
                weights[j] *= correction;
            }
        }
    }

    float inv_sum = online_softmax_finalize(state);

    // exp(-100) ≈ 0, so first weight should be ~0
    // exp(10) >> exp(0), so third weight dominates
    EXPECT_NEAR(weights[0] * inv_sum, reference[0], 1e-10f);
    EXPECT_NEAR(weights[1] * inv_sum, reference[1], 1e-5f);
    EXPECT_NEAR(weights[2] * inv_sum, reference[2], 1e-5f);
}

TEST_F(Test__OnlineSoftmax, RandomSequence)
{
    // Test with a longer random-ish sequence
    std::vector<float> scores = {2.1f, -0.5f, 3.7f, 1.2f, 2.8f, -1.3f, 4.0f, 0.0f};
    auto reference = softmax_reference(scores);

    OnlineSoftmaxState state = online_softmax_init();
    std::vector<float> weights(scores.size());

    for (size_t i = 0; i < scores.size(); ++i)
    {
        float prev_max = state.max_score;
        weights[i] = online_softmax_update(state, scores[i]);

        if (online_softmax_max_changed(state, prev_max) && i > 0)
        {
            float correction = online_softmax_correction(prev_max, state.max_score);
            for (size_t j = 0; j < i; ++j)
            {
                weights[j] *= correction;
            }
        }
    }

    float inv_sum = online_softmax_finalize(state);

    // Verify sum of weights is 1.0
    float sum = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        weights[i] *= inv_sum;
        sum += weights[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Verify each weight
    for (size_t i = 0; i < scores.size(); ++i)
    {
        EXPECT_NEAR(weights[i], reference[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST_F(Test__OnlineSoftmax, CorrectionFactor)
{
    // Directly test correction factor computation
    float old_max = 5.0f;
    float new_max = 8.0f;

    float correction = online_softmax_correction(old_max, new_max);
    float expected = std::exp(5.0f - 8.0f); // exp(-3)

    EXPECT_NEAR(correction, expected, 1e-6f);
}

TEST_F(Test__OnlineSoftmax, FinalizeEmptyState)
{
    // Empty state (no scores) should return 0 to avoid division by zero
    OnlineSoftmaxState state = online_softmax_init();
    float inv_sum = online_softmax_finalize(state);

    EXPECT_FLOAT_EQ(inv_sum, 0.0f);
}

TEST_F(Test__OnlineSoftmax, AttentionTypicalRange)
{
    // Test with values typical of attention scores (after scale by 1/sqrt(d))
    // For head_dim=64, scale = 1/8 = 0.125
    // Raw Q·K products might be -200 to 200, scaled to -25 to 25

    std::vector<float> scores = {-5.0f, -2.0f, 0.0f, 1.5f, 3.0f, -10.0f, 2.0f, -1.0f};
    auto reference = softmax_reference(scores);

    OnlineSoftmaxState state = online_softmax_init();
    std::vector<float> weights(scores.size());

    for (size_t i = 0; i < scores.size(); ++i)
    {
        float prev_max = state.max_score;
        weights[i] = online_softmax_update(state, scores[i]);

        if (online_softmax_max_changed(state, prev_max) && i > 0)
        {
            float correction = online_softmax_correction(prev_max, state.max_score);
            for (size_t j = 0; j < i; ++j)
            {
                weights[j] *= correction;
            }
        }
    }

    float inv_sum = online_softmax_finalize(state);

    for (size_t i = 0; i < scores.size(); ++i)
    {
        EXPECT_NEAR(weights[i] * inv_sum, reference[i], 1e-5f);
    }
}
