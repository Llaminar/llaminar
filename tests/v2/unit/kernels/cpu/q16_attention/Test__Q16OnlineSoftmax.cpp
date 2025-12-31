/**
 * @file Test__Q16OnlineSoftmax.cpp
 * @brief Comprehensive unit tests for Q16 FlashDecode online softmax microkernel
 *
 * Tests the OnlineSoftmax.h/cpp functions for the Q16 integer attention path.
 * Includes mass random testing (10,000 vectors) comparing against FP32 reference.
 *
 * Per Gemini3: "The Exp2FixedSoftmax needs to be battle-tested."
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Exp2FixedSoftmax.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2::kernels::q16_1::microkernels;
using namespace llaminar2;

class Test__Q16OnlineSoftmax : public ::testing::Test
{
protected:
    static constexpr int32_t MASKED = std::numeric_limits<int32_t>::min();

    /**
     * @brief FP32 reference softmax for comparison
     *
     * Standard two-pass softmax with numerical stability.
     */
    std::vector<float> reference_softmax_fp32(
        const std::vector<int32_t> &scores,
        float alpha)
    {
        std::vector<float> result(scores.size(), 0.0f);

        // Find max (excluding masked)
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] != MASKED)
            {
                float s = static_cast<float>(scores[i]) * alpha;
                max_score = std::max(max_score, s);
            }
        }

        if (max_score == -std::numeric_limits<float>::infinity())
        {
            return result; // All masked
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

    /**
     * @brief Simulate online softmax using the Q16 microkernels
     *
     * Processes scores in blocks, mimicking the FlashDecode pattern.
     */
    std::vector<float> online_softmax_q16(
        const std::vector<int32_t> &scores,
        float alpha,
        int block_size = 32)
    {
        const int n = static_cast<int>(scores.size());
        std::vector<float> result(n, 0.0f);

        if (n == 0)
            return result;

        // Initialize state
        OnlineSoftmaxState state;
        state.reset();

        // We'll track unnormalized weights per position
        std::vector<int64_t> unnorm_weights(n, 0);

        // Process in blocks
        for (int block_start = 0; block_start < n; block_start += block_size)
        {
            int block_end = std::min(block_start + block_size, n);
            int actual_block_size = block_end - block_start;

            // Find max in this block
            int32_t block_max = online_softmax_find_block_max(
                scores.data() + block_start,
                actual_block_size,
                MASKED);

            if (block_max == MASKED)
                continue; // All masked in this block

            // Update running max
            int32_t m_old = state.m;
            int32_t m_new = std::max(m_old, block_max);

            // If max increased, rescale previous state
            if (m_new > m_old && state.count > 0)
            {
                int32_t scale_num;
                int scale_shift;
                online_softmax_compute_rescale(
                    m_old, m_new, alpha,
                    scale_num, scale_shift,
                    state.frac_bits, state.lut_value_bits);

                // Rescale previous weights
                for (int i = 0; i < block_start; ++i)
                {
                    unnorm_weights[i] = (unnorm_weights[i] * scale_num) >> scale_shift;
                }

                // Rescale l
                state.l = (state.l * scale_num) >> scale_shift;
            }

            state.m = m_new;

            // Compute weights for this block
            std::vector<int32_t> block_weights(actual_block_size);
            int64_t block_sum = online_softmax_compute_block_weights(
                scores.data() + block_start,
                block_weights.data(),
                actual_block_size,
                state.m,
                alpha,
                state.frac_bits,
                state.lut_value_bits);

            // Store weights and update state
            for (int i = 0; i < actual_block_size; ++i)
            {
                if (scores[block_start + i] != MASKED)
                {
                    unnorm_weights[block_start + i] = block_weights[i];
                    state.count++;
                }
            }
            state.l += block_sum;
        }

        // Normalize
        if (state.l > 0)
        {
            for (int i = 0; i < n; ++i)
            {
                result[i] = static_cast<float>(unnorm_weights[i]) / static_cast<float>(state.l);
            }
        }

        return result;
    }

    /**
     * @brief Compute KL divergence D_KL(P || Q)
     */
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

    /**
     * @brief Compute max absolute error
     */
    float max_abs_error(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

    /**
     * @brief Compute RMS error
     */
    float rms_error(const std::vector<float> &a, const std::vector<float> &b)
    {
        double sum_sq = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        return static_cast<float>(std::sqrt(sum_sq / a.size()));
    }

    std::mt19937 rng_{42};
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(Test__Q16OnlineSoftmax, FindBlockMax_AllPositive)
{
    std::vector<int32_t> scores = {100, 500, 200, 800, 300};
    int32_t max = online_softmax_find_block_max(scores.data(), 5);
    EXPECT_EQ(max, 800);
}

TEST_F(Test__Q16OnlineSoftmax, FindBlockMax_WithMasked)
{
    std::vector<int32_t> scores = {100, MASKED, 200, MASKED, 150};
    int32_t max = online_softmax_find_block_max(scores.data(), 5);
    EXPECT_EQ(max, 200);
}

TEST_F(Test__Q16OnlineSoftmax, FindBlockMax_AllMasked)
{
    std::vector<int32_t> scores = {MASKED, MASKED, MASKED};
    int32_t max = online_softmax_find_block_max(scores.data(), 3);
    EXPECT_EQ(max, MASKED);
}

TEST_F(Test__Q16OnlineSoftmax, FindBlockMax_Negative)
{
    std::vector<int32_t> scores = {-100, -500, -200, -50};
    int32_t max = online_softmax_find_block_max(scores.data(), 4);
    EXPECT_EQ(max, -50);
}

TEST_F(Test__Q16OnlineSoftmax, ComputeRescale_NoChange)
{
    int32_t scale_num;
    int scale_shift;

    // Same max -> rescale factor = 1
    online_softmax_compute_rescale(1000, 1000, 0.01f, scale_num, scale_shift);

    // scale_num >> scale_shift ≈ 1.0 when interpreted as fixed-point
    // Use 1LL to avoid signed overflow when scale_shift >= 31
    float factor = static_cast<float>(scale_num) / static_cast<float>(1LL << scale_shift);
    EXPECT_NEAR(factor, 1.0f, 0.01f);
}

TEST_F(Test__Q16OnlineSoftmax, ComputeRescale_MaxIncreased)
{
    int32_t scale_num;
    int scale_shift;

    // Max increased by 100, with alpha = 0.01
    // correction = exp(-100 * 0.01) = exp(-1) ≈ 0.368
    online_softmax_compute_rescale(1000, 1100, 0.01f, scale_num, scale_shift);

    // Use 1LL to avoid signed overflow when scale_shift >= 31
    // (default lut_value_bits=30, so scale_shift = 30 + ip can reach 31+)
    float factor = static_cast<float>(scale_num) / static_cast<float>(1LL << scale_shift);
    float expected = std::exp(-100.0f * 0.01f);

    EXPECT_NEAR(factor, expected, 0.05f) << "Rescale factor mismatch";
}

// ============================================================================
// Online vs Offline Comparison Tests
// ============================================================================

TEST_F(Test__Q16OnlineSoftmax, SimpleSequence_MatchesReference)
{
    std::vector<int32_t> scores = {1000, 900, 800, 700, 600};
    float alpha = 0.01f;

    auto ref = reference_softmax_fp32(scores, alpha);
    auto online = online_softmax_q16(scores, alpha, 2); // Small block size

    float max_err = max_abs_error(ref, online);
    EXPECT_LT(max_err, 0.05f) << "Online softmax diverged from reference";
}

TEST_F(Test__Q16OnlineSoftmax, IncreasingSequence_TriggersRescaling)
{
    // Increasing scores trigger max updates
    std::vector<int32_t> scores = {100, 200, 300, 400, 500, 600, 700, 800};
    float alpha = 0.01f;

    auto ref = reference_softmax_fp32(scores, alpha);
    auto online = online_softmax_q16(scores, alpha, 4);

    float kl = kl_divergence(ref, online);
    EXPECT_LT(kl, 0.01f) << "KL divergence too high after rescaling";
}

TEST_F(Test__Q16OnlineSoftmax, DecreasingSequence_NoRescaling)
{
    // Decreasing scores don't trigger rescaling after first block
    std::vector<int32_t> scores = {800, 700, 600, 500, 400, 300, 200, 100};
    float alpha = 0.01f;

    auto ref = reference_softmax_fp32(scores, alpha);
    auto online = online_softmax_q16(scores, alpha, 4);

    float kl = kl_divergence(ref, online);
    EXPECT_LT(kl, 0.01f);
}

TEST_F(Test__Q16OnlineSoftmax, WithMaskedPositions)
{
    std::vector<int32_t> scores = {500, MASKED, 300, MASKED, 400, 200};
    float alpha = 0.01f;

    auto ref = reference_softmax_fp32(scores, alpha);
    auto online = online_softmax_q16(scores, alpha, 2);

    // Masked positions should have zero weight
    EXPECT_NEAR(online[1], 0.0f, 1e-6f);
    EXPECT_NEAR(online[3], 0.0f, 1e-6f);

    // Others should match reference
    EXPECT_NEAR(online[0], ref[0], 0.02f);
    EXPECT_NEAR(online[2], ref[2], 0.02f);
}

TEST_F(Test__Q16OnlineSoftmax, SpikySingleDominant)
{
    // One very high score (typical attention pattern)
    std::vector<int32_t> scores = {10000, 100, 100, 100, 100, 100, 100, 100};
    float alpha = 0.001f;

    auto ref = reference_softmax_fp32(scores, alpha);
    auto online = online_softmax_q16(scores, alpha, 4);

    // Dominant position should get most weight
    EXPECT_GT(online[0], 0.9f) << "Dominant position should dominate";
    EXPECT_GT(online[0], ref[0] - 0.05f) << "Should match reference dominant weight";
}

// ============================================================================
// Mass Random Tests (10,000 Vectors) - Battle Testing
// ============================================================================

TEST_F(Test__Q16OnlineSoftmax, MassRandom_10000Vectors_KLDivergence)
{
    /**
     * BATTLE TEST: Compare block-wise online softmax against FP32 offline softmax
     * for 10,000 random input vectors.
     *
     * This tests:
     * 1. Online softmax correctness with max rescaling
     * 2. Block-wise processing accuracy
     * 3. Numerical stability across score ranges
     *
     * Per Gemini3: "If this LUT is inaccurate, the whole kernel fails."
     */
    const int num_vectors = 10000;
    const int seq_lengths[] = {8, 16, 32, 64, 128, 256};
    const int block_sizes[] = {8, 16, 32, 64};
    const float alphas[] = {
        0.001f,                   // Very smooth
        1.0f / std::sqrt(64.0f),  // head_dim=64
        1.0f / std::sqrt(128.0f), // head_dim=128 (typical)
        0.1f                      // Medium
    };

    std::mt19937 rng(98765);

    // Aggregate statistics
    double total_kl = 0.0;
    double total_rms = 0.0;
    double worst_kl = 0.0;
    double worst_max_err = 0.0;
    int num_tests = 0;
    int kl_failures = 0;  // KL > 0.02 nats (more lenient for online)
    int rms_failures = 0; // RMS > 0.05

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║  Q16 ONLINE SOFTMAX vs FP32 OFFLINE - BATTLE TEST (10,000 Vectors)       ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n";

    for (float alpha : alphas)
    {
        for (int seq_len : seq_lengths)
        {
            for (int block_size : block_sizes)
            {
                if (block_size > seq_len)
                    continue;

                // Score distribution (realistic attention score ranges)
                std::normal_distribution<float> score_dist(0.0f, 1000.0f);

                int vectors_per_config = num_vectors / (4 * 6 * 4);

                for (int v = 0; v < vectors_per_config; ++v)
                {
                    // Generate random scores
                    std::vector<int32_t> scores(seq_len);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        scores[i] = static_cast<int32_t>(score_dist(rng));
                    }

                    // FP32 reference
                    auto ref = reference_softmax_fp32(scores, alpha);

                    // Q16 online softmax
                    auto online = online_softmax_q16(scores, alpha, block_size);

                    // Compute metrics
                    float kl = kl_divergence(ref, online);
                    float max_err = max_abs_error(ref, online);
                    float rms = rms_error(ref, online);

                    // Track failures
                    if (kl > 0.02f)
                        kl_failures++;
                    if (rms > 0.05f)
                        rms_failures++;

                    // Accumulate
                    total_kl += kl;
                    total_rms += rms;
                    worst_kl = std::max(worst_kl, static_cast<double>(kl));
                    worst_max_err = std::max(worst_max_err, static_cast<double>(max_err));
                    num_tests++;
                }
            }
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
              << "║ KL > 0.02 failures:      " << std::setw(10) << kl_failures << " (" << std::fixed << std::setprecision(2) << failure_rate_kl << "%)                              ║\n"
              << "║ RMS > 0.05 failures:     " << std::setw(10) << rms_failures << " (" << failure_rate_rms << "%)                              ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // Assertions: Quality gates for online softmax
    // Online has more accumulated error from rescaling, so we're slightly more lenient
    EXPECT_LT(mean_kl, 0.01) << "Mean KL divergence too high for online softmax!";
    EXPECT_LT(worst_kl, 0.2) << "Worst-case KL divergence unacceptable.";
    EXPECT_LT(mean_rms, 0.02) << "Mean RMS error too high.";
    EXPECT_LT(failure_rate_kl, 2.0) << "Too many high-KL failures (>2%).";
    EXPECT_LT(failure_rate_rms, 2.0) << "Too many high-RMS failures (>2%).";
}

TEST_F(Test__Q16OnlineSoftmax, MassRandom_RescalingStressTest)
{
    /**
     * Stress test: Sequences designed to trigger many rescaling events.
     *
     * Creates sequences where the max increases frequently, forcing
     * the online softmax to rescale accumulated state repeatedly.
     */
    const int num_vectors = 1000;
    const int seq_len = 128;
    const int block_size = 8; // Small blocks = more rescaling
    const float alpha = 0.01f;

    std::mt19937 rng(11111);

    int rescale_triggers = 0;
    double total_kl = 0.0;
    int num_tests = 0;

    for (int v = 0; v < num_vectors; ++v)
    {
        // Generate INCREASING sequence (worst case for rescaling)
        std::vector<int32_t> scores(seq_len);
        int32_t base = -5000;
        for (int i = 0; i < seq_len; ++i)
        {
            // Add noise but ensure overall increasing trend
            std::uniform_int_distribution<int32_t> noise(-50, 100);
            base += noise(rng);
            scores[i] = base;
        }

        // Count expected rescale events (every time block max > running max)
        int expected_rescales = 0;
        int32_t running_max = MASKED;
        for (int b = 0; b < seq_len; b += block_size)
        {
            int32_t block_max = MASKED;
            for (int i = b; i < std::min(b + block_size, seq_len); ++i)
            {
                block_max = std::max(block_max, scores[i]);
            }
            if (block_max > running_max)
            {
                if (running_max != MASKED)
                    expected_rescales++;
                running_max = block_max;
            }
        }
        rescale_triggers += expected_rescales;

        // Test
        auto ref = reference_softmax_fp32(scores, alpha);
        auto online = online_softmax_q16(scores, alpha, block_size);

        float kl = kl_divergence(ref, online);
        total_kl += kl;
        num_tests++;
    }

    double mean_kl = total_kl / num_tests;
    double avg_rescales = static_cast<double>(rescale_triggers) / num_vectors;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "║             RESCALING STRESS TEST (INCREASING SEQUENCES)                 ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════╣\n"
              << "║ Test vectors:            " << std::setw(10) << num_vectors << "                                    ║\n"
              << "║ Sequence length:         " << std::setw(10) << seq_len << "                                    ║\n"
              << "║ Block size:              " << std::setw(10) << block_size << "                                    ║\n"
              << "║ Total rescale triggers:  " << std::setw(10) << rescale_triggers << "                                    ║\n"
              << "║ Avg rescales per seq:    " << std::setw(10) << std::fixed << std::setprecision(2) << avg_rescales << "                                    ║\n"
              << "║ Mean KL Divergence:      " << std::setw(10) << std::scientific << std::setprecision(4) << mean_kl << " nats                                ║\n"
              << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

    // Even with many rescales, should maintain accuracy
    EXPECT_LT(mean_kl, 0.02) << "Mean KL too high after heavy rescaling";
    EXPECT_GT(avg_rescales, 5.0) << "Test didn't trigger enough rescales";
}

TEST_F(Test__Q16OnlineSoftmax, EdgeCase_AllSameScore)
{
    // All same scores -> uniform distribution
    const int n = 64;
    std::vector<int32_t> scores(n, 1000);
    float alpha = 0.01f;

    auto ref = reference_softmax_fp32(scores, alpha);
    auto online = online_softmax_q16(scores, alpha, 8);

    // Should all be 1/n
    float expected = 1.0f / n;
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(online[i], expected, 0.01f)
            << "Position " << i << " should be " << expected;
    }
}

TEST_F(Test__Q16OnlineSoftmax, EdgeCase_SinglePosition)
{
    std::vector<int32_t> scores = {1000};
    float alpha = 0.01f;

    auto online = online_softmax_q16(scores, alpha, 1);

    EXPECT_NEAR(online[0], 1.0f, 1e-6f) << "Single position should get all weight";
}

TEST_F(Test__Q16OnlineSoftmax, EdgeCase_ExtremeScoreGap)
{
    // Extreme gap tests underflow handling
    std::vector<int32_t> scores = {100000, 0, 0, 0, 0, 0, 0, 0};
    float alpha = 0.001f;

    auto ref = reference_softmax_fp32(scores, alpha);
    auto online = online_softmax_q16(scores, alpha, 4);

    // Winner should get essentially everything
    EXPECT_GT(online[0], 0.99f) << "Extreme winner should dominate";

    // Match reference
    EXPECT_NEAR(online[0], ref[0], 0.01f);
}

// ============================================================================
// online_softmax_update_block Tests (Direct API Coverage)
// ============================================================================

TEST_F(Test__Q16OnlineSoftmax, UpdateBlock_FirstBlock_NoRescale)
{
    // First block should never need rescaling
    OnlineSoftmaxState state;
    state.reset();

    std::vector<int32_t> scores = {1000, 800, 600, 400};
    std::vector<int32_t> weights(4);
    float alpha = 0.01f;

    int32_t scale_num;
    int scale_shift;

    bool needs_rescale = online_softmax_update_block(
        state, scores.data(), weights.data(),
        4, alpha, scale_num, scale_shift);

    EXPECT_FALSE(needs_rescale) << "First block should never need rescaling";
    EXPECT_EQ(state.m, 1000) << "Max should be highest score";
    EXPECT_GT(state.l, 0) << "Sum should be positive";
    EXPECT_EQ(state.count, 4) << "Should count all unmasked positions";
}

TEST_F(Test__Q16OnlineSoftmax, UpdateBlock_SecondBlock_TriggersRescale)
{
    // Process first block, then second with higher max
    OnlineSoftmaxState state;
    state.reset();
    float alpha = 0.01f;

    // First block
    std::vector<int32_t> scores1 = {500, 400, 300, 200};
    std::vector<int32_t> weights1(4);
    int32_t scale_num;
    int scale_shift;

    online_softmax_update_block(
        state, scores1.data(), weights1.data(),
        4, alpha, scale_num, scale_shift);

    int64_t l_after_first = state.l;
    EXPECT_EQ(state.m, 500);

    // Second block with higher max
    std::vector<int32_t> scores2 = {1000, 900, 800, 700};
    std::vector<int32_t> weights2(4);

    bool needs_rescale = online_softmax_update_block(
        state, scores2.data(), weights2.data(),
        4, alpha, scale_num, scale_shift);

    EXPECT_TRUE(needs_rescale) << "Should need rescaling when max increases";
    EXPECT_EQ(state.m, 1000) << "Max should update to 1000";
    EXPECT_LT(state.l, l_after_first * 2) << "Old sum should be rescaled down";
    EXPECT_EQ(state.count, 8) << "Should count all positions";
}

TEST_F(Test__Q16OnlineSoftmax, UpdateBlock_AllMasked)
{
    OnlineSoftmaxState state;
    state.reset();

    std::vector<int32_t> scores = {MASKED, MASKED, MASKED, MASKED};
    std::vector<int32_t> weights(4);
    float alpha = 0.01f;

    int32_t scale_num;
    int scale_shift;

    bool needs_rescale = online_softmax_update_block(
        state, scores.data(), weights.data(),
        4, alpha, scale_num, scale_shift);

    EXPECT_FALSE(needs_rescale);
    EXPECT_EQ(state.count, 0) << "No positions should be counted";

    // All weights should be zero
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(weights[i], 0) << "Masked position " << i << " should have zero weight";
    }
}

TEST_F(Test__Q16OnlineSoftmax, UpdateBlock_PartialMask)
{
    OnlineSoftmaxState state;
    state.reset();

    std::vector<int32_t> scores = {1000, MASKED, 800, MASKED};
    std::vector<int32_t> weights(4);
    float alpha = 0.01f;

    int32_t scale_num;
    int scale_shift;

    online_softmax_update_block(
        state, scores.data(), weights.data(),
        4, alpha, scale_num, scale_shift);

    EXPECT_EQ(state.m, 1000) << "Max should ignore masked";
    EXPECT_EQ(state.count, 2) << "Only unmasked positions counted";
    EXPECT_EQ(weights[1], 0) << "Masked should be zero";
    EXPECT_EQ(weights[3], 0) << "Masked should be zero";
    EXPECT_GT(weights[0], 0) << "Unmasked should have weight";
    EXPECT_GT(weights[2], 0) << "Unmasked should have weight";
}

// ============================================================================
// online_softmax_finalize_weights Tests
// ============================================================================

TEST_F(Test__Q16OnlineSoftmax, FinalizeWeights_Normalization)
{
    // Setup state manually
    OnlineSoftmaxState state;
    state.reset();
    state.l = 100000; // Sum of unnormalized weights
    state.count = 4;

    // Input weights (unnormalized INT32)
    std::vector<int32_t> block_weights = {40000, 30000, 20000, 10000};

    // Output weights (normalized INT16)
    std::vector<int16_t> final_weights(4);
    int16_t weight_max = 32767;

    int32_t total = online_softmax_finalize_weights(
        state, block_weights.data(), final_weights.data(),
        4, weight_max);

    // Check normalization: w_final = block_weights * weight_max / l
    // Expected: 40000 * 32767 / 100000 ≈ 13107
    EXPECT_NEAR(final_weights[0], 13107, 100);
    EXPECT_NEAR(final_weights[1], 9830, 100);
    EXPECT_NEAR(final_weights[2], 6553, 100);
    EXPECT_NEAR(final_weights[3], 3277, 100);

    // Total should be close to weight_max (but not exact due to rounding)
    EXPECT_GT(total, 0);
    EXPECT_LE(total, weight_max * 4);
}

TEST_F(Test__Q16OnlineSoftmax, FinalizeWeights_ZeroSum)
{
    OnlineSoftmaxState state;
    state.reset();
    state.l = 0; // No unmasked positions

    std::vector<int32_t> block_weights = {0, 0, 0, 0};
    std::vector<int16_t> final_weights(4, 999); // Initialize to non-zero

    int32_t total = online_softmax_finalize_weights(
        state, block_weights.data(), final_weights.data(),
        4, 32767);

    EXPECT_EQ(total, 0);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(final_weights[i], 0) << "Position " << i << " should be zero";
    }
}

TEST_F(Test__Q16OnlineSoftmax, FinalizeWeights_SingleDominant)
{
    OnlineSoftmaxState state;
    state.reset();
    state.l = 1000000;

    // One dominant position
    std::vector<int32_t> block_weights = {990000, 5000, 3000, 2000};
    std::vector<int16_t> final_weights(4);

    online_softmax_finalize_weights(
        state, block_weights.data(), final_weights.data(),
        4, 32767);

    // Dominant should get most of the weight
    EXPECT_GT(final_weights[0], 30000);
    EXPECT_LT(final_weights[1], 200);
    EXPECT_LT(final_weights[2], 100);
    EXPECT_LT(final_weights[3], 100);
}

// ============================================================================
// flash_decode_process_kv_block Tests (End-to-End Block Processing)
// ============================================================================

TEST_F(Test__Q16OnlineSoftmax, FlashDecodeProcessBlock_SingleBlock_Block64)
{
    /**
     * Test the complete flash_decode_process_kv_block function.
     * This is the main entry point for FlashDecode attention.
     */
    const int head_dim = 64;
    const int kv_block_size = 4;
    const int blocks_per_row = 1;
    const float alpha = 0.01f;

    // Create Q vector
    Q16_1Block_64 Q;
    Q.d = 1.0f;
    for (int i = 0; i < 64; ++i)
    {
        Q.qs[i] = static_cast<int16_t>(100 + i);
    }

    // Create K and V blocks [kv_block_size, blocks_per_row]
    std::vector<Q16_1Block_64> K(kv_block_size);
    std::vector<Q16_1Block_64> V(kv_block_size);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int16_t> dist(-500, 500);

    for (int k = 0; k < kv_block_size; ++k)
    {
        K[k].d = 1.0f;
        V[k].d = 1.0f;
        for (int i = 0; i < 64; ++i)
        {
            K[k].qs[i] = dist(rng);
            V[k].qs[i] = dist(rng);
        }
    }

    // Initialize context and state
    std::vector<int32_t> context(head_dim, 0);
    OnlineSoftmaxState state;
    state.reset();

    // Scratch buffers
    std::vector<int32_t> scores_scratch(kv_block_size);
    std::vector<int32_t> weights_scratch(kv_block_size);

    // Process block
    flash_decode_process_kv_block<Q16_1Block_64>(
        &Q, K.data(), V.data(),
        context.data(), state,
        scores_scratch.data(), weights_scratch.data(),
        0, kv_block_size,
        head_dim, blocks_per_row, alpha);

    // Verify state was updated
    EXPECT_GT(state.m, std::numeric_limits<int32_t>::min()) << "Max should be set";
    EXPECT_GT(state.l, 0) << "Sum should be positive";
    EXPECT_EQ(state.count, kv_block_size) << "All positions counted";

    // Context should have some non-zero values
    bool has_nonzero = false;
    for (int d = 0; d < head_dim; ++d)
    {
        if (context[d] != 0)
            has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero) << "Context should have accumulated values";
}

TEST_F(Test__Q16OnlineSoftmax, FlashDecodeProcessBlock_MultipleBlocks_Block64)
{
    /**
     * Test processing multiple KV blocks sequentially.
     * This tests the online rescaling mechanism.
     */
    const int head_dim = 64;
    const int kv_block_size = 4;
    const int num_blocks = 4;
    const int blocks_per_row = 1;
    const float alpha = 0.01f;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int16_t> dist(-500, 500);

    // Create Q
    Q16_1Block_64 Q;
    Q.d = 1.0f;
    for (int i = 0; i < 64; ++i)
    {
        Q.qs[i] = static_cast<int16_t>(dist(rng));
    }

    // Create K and V for all blocks
    std::vector<Q16_1Block_64> K(kv_block_size * num_blocks);
    std::vector<Q16_1Block_64> V(kv_block_size * num_blocks);

    for (int k = 0; k < kv_block_size * num_blocks; ++k)
    {
        K[k].d = 1.0f;
        V[k].d = 1.0f;
        for (int i = 0; i < 64; ++i)
        {
            K[k].qs[i] = dist(rng);
            V[k].qs[i] = dist(rng);
        }
    }

    // Process blocks sequentially
    std::vector<int32_t> context(head_dim, 0);
    OnlineSoftmaxState state;
    state.reset();

    std::vector<int32_t> scores_scratch(kv_block_size);
    std::vector<int32_t> weights_scratch(kv_block_size);

    for (int b = 0; b < num_blocks; ++b)
    {
        flash_decode_process_kv_block<Q16_1Block_64>(
            &Q, K.data(), V.data(),
            context.data(), state,
            scores_scratch.data(), weights_scratch.data(),
            b * kv_block_size, kv_block_size,
            head_dim, blocks_per_row, alpha);
    }

    // Verify final state
    EXPECT_EQ(state.count, kv_block_size * num_blocks);
    EXPECT_GT(state.l, 0);

    // Context should have accumulated from all blocks
    double context_norm = 0.0;
    for (int d = 0; d < head_dim; ++d)
    {
        context_norm += static_cast<double>(context[d]) * context[d];
    }
    EXPECT_GT(context_norm, 0.0) << "Context should have non-zero values";
}

TEST_F(Test__Q16OnlineSoftmax, FlashDecodeProcessBlock_Block128)
{
    /**
     * Same test for Block128 variant.
     */
    const int head_dim = 128;
    const int kv_block_size = 8;
    const int blocks_per_row = 1;
    const float alpha = 1.0f / std::sqrt(128.0f);

    std::mt19937 rng(99999);
    std::uniform_int_distribution<int16_t> dist(-300, 300);

    Q16_1Block_128 Q;
    Q.d = 1.0f;
    for (int i = 0; i < 128; ++i)
    {
        Q.qs[i] = dist(rng);
    }

    std::vector<Q16_1Block_128> K(kv_block_size);
    std::vector<Q16_1Block_128> V(kv_block_size);

    for (int k = 0; k < kv_block_size; ++k)
    {
        K[k].d = 1.0f;
        V[k].d = 1.0f;
        for (int i = 0; i < 128; ++i)
        {
            K[k].qs[i] = dist(rng);
            V[k].qs[i] = dist(rng);
        }
    }

    std::vector<int32_t> context(head_dim, 0);
    OnlineSoftmaxState state;
    state.reset();

    std::vector<int32_t> scores_scratch(kv_block_size);
    std::vector<int32_t> weights_scratch(kv_block_size);

    flash_decode_process_kv_block<Q16_1Block_128>(
        &Q, K.data(), V.data(),
        context.data(), state,
        scores_scratch.data(), weights_scratch.data(),
        0, kv_block_size,
        head_dim, blocks_per_row, alpha);

    EXPECT_EQ(state.count, kv_block_size);
    EXPECT_GT(state.l, 0);

    bool has_nonzero = false;
    for (int d = 0; d < head_dim; ++d)
    {
        if (context[d] != 0)
            has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(Test__Q16OnlineSoftmax, OnlineSoftmaxState_Empty)
{
    // Test the OnlineSoftmaxState::empty() method
    OnlineSoftmaxState state;
    state.reset();

    EXPECT_TRUE(state.empty()) << "Fresh state should be empty";

    state.count = 1;
    EXPECT_FALSE(state.empty()) << "State with count > 0 should not be empty";

    state.count = 0;
    EXPECT_TRUE(state.empty()) << "State with count = 0 should be empty";
}
