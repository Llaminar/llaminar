/**
 * @file Test__OnlineSoftmaxPerPositionKScales.cpp
 * @brief Unit tests to reproduce the HybridQ16 OnlineSoftmax per-position K scale bug
 *
 * HYPOTHESIS (from handover investigation):
 * =========================================
 * The integration test shows:
 *   - First position (r0) has excellent cosine similarity (0.996)
 *   - Subsequent positions (r1-r8) degrade severely (0.35-0.69)
 *
 * This pattern suggests the OnlineSoftmax running_max rescaling may not handle
 * varying K scales correctly. Specifically:
 *
 * 1. When K scales vary per position, each position's scaled score has a
 *    different "effective range" - the max comparison and rescaling may mix
 *    values from incompatible scales.
 *
 * 2. The first position works because there's no prior state to rescale -
 *    it's simply the initial max. Subsequent positions need to rescale
 *    prior state, and this may be where the bug manifests.
 *
 * 3. The l_processed accumulation may not correctly account for positions
 *    with different alpha values when computing the weighted average.
 *
 * TEST STRATEGY:
 * ==============
 * We design tests that:
 * - Isolate the OnlineSoftmax microkernel from the full attention pipeline
 * - Use synthetic data with known properties
 * - Compare uniform K scales (should work) vs varying K scales (may fail)
 * - Verify the running_max rescaling math is correct for varying alphas
 *
 * @see docs/v2/HANDOVER_HYBRIDQ16_ATTENTION_INVESTIGATION.md
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <random>

#include "kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Exp2Core.h"

using namespace llaminar2::kernels::q16_1::microkernels;

namespace
{

    // ============================================================================
    // Test Configuration
    // ============================================================================

    constexpr int HEAD_DIM = 64;
    constexpr int NUM_POSITIONS = 9;
    constexpr int FRAC_BITS = 11;
    constexpr int LUT_VALUE_BITS = 30;

    // Typical Q scale from Qwen2.5-0.5B
    constexpr float Q_SCALE = 0.00195318f;

    // K scales that vary per position (from actual pipeline data)
    constexpr float K_SCALES_VARYING[] = {
        0.00795628f, 0.00793607f, 0.00797f, 0.00810f, 0.00830f,
        0.00850f, 0.00870f, 0.00890f, 0.00910f};

    // Uniform K scale (what uniform mode would use)
    constexpr float K_SCALE_UNIFORM = 0.00800f;

    // ============================================================================
    // Helper Functions
    // ============================================================================

    /**
     * @brief Compute FP32 softmax for reference
     */
    std::vector<float> reference_softmax(const std::vector<float> &scores)
    {
        float max_score = *std::max_element(scores.begin(), scores.end());

        std::vector<float> weights(scores.size());
        float sum = 0.0f;
        for (size_t i = 0; i < scores.size(); ++i)
        {
            weights[i] = std::exp(scores[i] - max_score);
            sum += weights[i];
        }
        for (size_t i = 0; i < scores.size(); ++i)
        {
            weights[i] /= sum;
        }
        return weights;
    }

    /**
     * @brief Compute FP32 attention output for reference
     */
    std::vector<float> reference_attention(
        const std::vector<float> &weights,
        const std::vector<std::vector<float>> &V,
        int head_dim)
    {
        std::vector<float> output(head_dim, 0.0f);
        for (size_t k = 0; k < weights.size(); ++k)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                output[d] += weights[k] * V[k][d];
            }
        }
        return output;
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        EXPECT_EQ(a.size(), b.size());
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10f);
    }

    /**
     * @brief Compute RMSE between two vectors
     */
    float rmse(const std::vector<float> &a, const std::vector<float> &b)
    {
        EXPECT_EQ(a.size(), b.size());
        float sum_sq = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            float diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / a.size());
    }

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class OnlineSoftmaxPerPositionKScalesTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Ensure LUT is initialized
            ensure_exp2_lut_initialized(LUT_VALUE_BITS);
        }

        /**
         * @brief Generate synthetic integer scores for testing
         *
         * Creates scores that simulate Q×K^T dot products in integer domain.
         * The scores are in INT32 range and need to be scaled by qk_scale.
         */
        std::vector<int32_t> generate_synthetic_scores(int num_positions, int seed = 42)
        {
            std::mt19937 gen(seed);
            // Scores from Q16 dot product typically range from -1M to +1M
            std::uniform_int_distribution<int32_t> dist(-500000, 500000);

            std::vector<int32_t> scores(num_positions);
            for (int i = 0; i < num_positions; ++i)
            {
                scores[i] = dist(gen);
            }
            return scores;
        }

        /**
         * @brief Generate synthetic V data for testing
         */
        std::vector<std::vector<float>> generate_synthetic_V(int num_positions, int head_dim, int seed = 123)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            std::vector<std::vector<float>> V(num_positions, std::vector<float>(head_dim));
            for (int k = 0; k < num_positions; ++k)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    V[k][d] = dist(gen);
                }
            }
            return V;
        }
    };

    // ============================================================================
    // Test: Uniform K scales should work
    // ============================================================================

    TEST_F(OnlineSoftmaxPerPositionKScalesTest, UniformKScales_ShouldProduceCorrectWeights)
    {
        // Setup: Generate synthetic scores
        auto int_scores = generate_synthetic_scores(NUM_POSITIONS);

        // Compute real scores with uniform K scale
        float qk_scale = Q_SCALE * K_SCALE_UNIFORM / std::sqrt(static_cast<float>(HEAD_DIM));
        std::vector<float> real_scores(NUM_POSITIONS);
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            real_scores[i] = static_cast<float>(int_scores[i]) * qk_scale;
        }

        // Reference FP32 softmax
        auto ref_weights = reference_softmax(real_scores);

        // OnlineSoftmax with uniform alpha
        OnlineSoftmaxState state;
        float alpha = qk_scale;
        state.init(alpha, FRAC_BITS, LUT_VALUE_BITS);

        std::vector<int32_t> weights(NUM_POSITIONS);
        int32_t scale_num;
        int scale_shift;

        online_softmax_update_block(state, int_scores.data(), weights.data(),
                                    NUM_POSITIONS, alpha, scale_num, scale_shift);

        // Convert weights to normalized form
        std::vector<float> online_weights(NUM_POSITIONS);
        float sum = 0.0f;
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            online_weights[i] = static_cast<float>(weights[i]);
            sum += online_weights[i];
        }
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            online_weights[i] /= (sum + 1e-10f);
        }

        // Compare
        float cos_sim = cosine_similarity(ref_weights, online_weights);
        std::cout << "Uniform K scales - cosine similarity: " << cos_sim << std::endl;

        // Should be very close
        EXPECT_GT(cos_sim, 0.99f) << "Uniform K scales should produce accurate weights";
    }

    // ============================================================================
    // Test: Varying K scales - simulate the per-position path
    // ============================================================================

    TEST_F(OnlineSoftmaxPerPositionKScalesTest, VaryingKScales_ManualPerPositionAlpha)
    {
        // Setup: Generate synthetic scores
        auto int_scores = generate_synthetic_scores(NUM_POSITIONS);

        // Compute real scores with VARYING K scales (ground truth)
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        std::vector<float> real_scores(NUM_POSITIONS);
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            float qk_scale = Q_SCALE * K_SCALES_VARYING[i] * inv_sqrt_d;
            real_scores[i] = static_cast<float>(int_scores[i]) * qk_scale;
        }

        // Reference FP32 softmax on REAL scores
        auto ref_weights = reference_softmax(real_scores);

        std::cout << "\nReal scores (with varying K scales):" << std::endl;
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            std::cout << "  pos " << i << ": int=" << int_scores[i]
                      << " k_scale=" << K_SCALES_VARYING[i]
                      << " real_score=" << real_scores[i]
                      << " ref_weight=" << ref_weights[i] << std::endl;
        }

        // Now simulate the per-position alpha path
        // This mimics what fa2_prefill_process_kv_tile_with_k_scales does
        float base_alpha = Q_SCALE * inv_sqrt_d; // Without K scale

        // Pre-compute per-position alpha configs
        std::vector<AdaptiveAlphaConfig> alpha_configs(NUM_POSITIONS);
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            float per_pos_alpha = base_alpha * K_SCALES_VARYING[i];
            alpha_configs[i] = AdaptiveAlphaConfig::compute(per_pos_alpha);
        }

        // Compute scaled scores in fixed-point domain
        constexpr int beta_scale_bits = 24;
        std::vector<int64_t> scaled_scores_fixed(NUM_POSITIONS);
        int64_t max_fixed = std::numeric_limits<int64_t>::min();

        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            const auto &cfg = alpha_configs[i];
            double effective_beta = static_cast<double>(cfg.effective_alpha) * LOG2E;
            int64_t M = static_cast<int64_t>(
                std::llround(effective_beta * static_cast<double>(1ULL << beta_scale_bits)));

            int shift_for_t = beta_scale_bits - FRAC_BITS + cfg.alpha_shift;

            int64_t prod = static_cast<int64_t>(int_scores[i]) * M;
            int64_t t_fixed = (shift_for_t >= 0)
                                  ? (prod >> shift_for_t)
                                  : (prod << (-shift_for_t));

            scaled_scores_fixed[i] = t_fixed;
            if (t_fixed > max_fixed)
            {
                max_fixed = t_fixed;
            }
        }

        std::cout << "\nScaled scores (fixed-point):" << std::endl;
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            std::cout << "  pos " << i << ": t_fixed=" << scaled_scores_fixed[i]
                      << " (delta from max=" << (max_fixed - scaled_scores_fixed[i]) << ")"
                      << std::endl;
        }

        // Compute weights via exp2 LUT
        const uint32_t *lut = get_exp2_lut_data();
        const uint32_t one = static_cast<uint32_t>(1U << LUT_VALUE_BITS);

        std::vector<uint32_t> weights(NUM_POSITIONS);
        uint64_t weight_sum = 0;

        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            int64_t delta_64 = scaled_scores_fixed[i] - max_fixed;

            uint32_t w;
            if (delta_64 >= 0)
            {
                w = one;
            }
            else if (delta_64 < -static_cast<int64_t>(32 << FRAC_BITS))
            {
                w = 0;
            }
            else
            {
                int64_t neg_delta = -delta_64;
                int int_part = static_cast<int>(neg_delta >> FRAC_BITS);
                int frac_part = static_cast<int>(neg_delta & ((1 << FRAC_BITS) - 1));

                uint32_t lut_val = lut[frac_part];
                w = (int_part < 32) ? (lut_val >> int_part) : 0;
            }

            weights[i] = w;
            weight_sum += w;
        }

        // Normalize
        std::vector<float> online_weights(NUM_POSITIONS);
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            online_weights[i] = static_cast<float>(weights[i]) / static_cast<float>(weight_sum);
        }

        std::cout << "\nWeight comparison:" << std::endl;
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            std::cout << "  pos " << i << ": ref=" << std::setprecision(6) << ref_weights[i]
                      << " online=" << online_weights[i]
                      << " diff=" << std::abs(ref_weights[i] - online_weights[i])
                      << std::endl;
        }

        float cos_sim = cosine_similarity(ref_weights, online_weights);
        std::cout << "\nVarying K scales - cosine similarity: " << cos_sim << std::endl;

        // This test documents the expected behavior - may or may not pass depending on bug status
        if (cos_sim > 0.99f)
        {
            std::cout << "PASS: Per-position K scales produce correct weights" << std::endl;
        }
        else
        {
            std::cout << "POTENTIAL BUG: Per-position K scales produce divergent weights" << std::endl;
        }

        // For now, we expect this to work at the single-block level
        EXPECT_GT(cos_sim, 0.95f);
    }

    // ============================================================================
    // Test: Multi-tile rescaling with varying K scales (THE SUSPECTED BUG)
    // ============================================================================

    TEST_F(OnlineSoftmaxPerPositionKScalesTest, MultiTileRescaling_VaryingKScales)
    {
        // This test simulates what happens when we process multiple tiles
        // Each tile has different K scales, and we need to rescale when
        // moving from one tile to the next

        constexpr int TILE_SIZE = 4;
        constexpr int NUM_TILES = 2;
        constexpr int TOTAL_POSITIONS = TILE_SIZE * NUM_TILES;

        // Generate scores for two tiles
        // Tile 0: positions 0-3 with K scales ~0.008
        // Tile 1: positions 4-7 with K scales ~0.009 (slightly higher)
        auto int_scores = generate_synthetic_scores(TOTAL_POSITIONS, 42);

        // Make tile 1 have a higher max score to force rescaling
        int_scores[4] = 800000; // This should become the new max when we process tile 1

        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        float k_scales[TOTAL_POSITIONS] = {
            0.0080f, 0.0080f, 0.0080f, 0.0080f, // Tile 0
            0.0090f, 0.0090f, 0.0090f, 0.0090f  // Tile 1: higher scales
        };

        // Compute reference (FP32 softmax on all positions)
        std::vector<float> real_scores(TOTAL_POSITIONS);
        for (int i = 0; i < TOTAL_POSITIONS; ++i)
        {
            float qk_scale = Q_SCALE * k_scales[i] * inv_sqrt_d;
            real_scores[i] = static_cast<float>(int_scores[i]) * qk_scale;
        }
        auto ref_weights = reference_softmax(real_scores);

        std::cout << "\n=== Multi-Tile Rescaling Test ===" << std::endl;
        std::cout << "Real scores:" << std::endl;
        for (int i = 0; i < TOTAL_POSITIONS; ++i)
        {
            std::cout << "  pos " << i << " (tile " << (i / TILE_SIZE) << "): "
                      << "int=" << int_scores[i] << " k_scale=" << k_scales[i]
                      << " real=" << real_scores[i] << std::endl;
        }

        // ===== Simulate tile-by-tile processing =====
        float base_alpha = Q_SCALE * inv_sqrt_d;

        // State variables
        int64_t running_sum = 0;
        int32_t running_max_fixed = std::numeric_limits<int32_t>::min();
        std::vector<uint32_t> all_weights(TOTAL_POSITIONS, 0);
        const uint32_t *lut = get_exp2_lut_data();
        const uint32_t one = static_cast<uint32_t>(1U << LUT_VALUE_BITS);

        // Process each tile
        for (int tile = 0; tile < NUM_TILES; ++tile)
        {
            int tile_start = tile * TILE_SIZE;
            std::cout << "\nProcessing tile " << tile << " (positions " << tile_start << "-" << (tile_start + TILE_SIZE - 1) << ")" << std::endl;

            // Compute scaled scores for this tile
            std::vector<int64_t> tile_scaled_scores(TILE_SIZE);
            int64_t tile_max_fixed = std::numeric_limits<int64_t>::min();

            for (int c = 0; c < TILE_SIZE; ++c)
            {
                int pos = tile_start + c;
                float per_pos_alpha = base_alpha * k_scales[pos];
                auto cfg = AdaptiveAlphaConfig::compute(per_pos_alpha);

                double effective_beta = static_cast<double>(cfg.effective_alpha) * LOG2E;
                int64_t M = static_cast<int64_t>(
                    std::llround(effective_beta * static_cast<double>(1ULL << 24)));

                int shift_for_t = 24 - FRAC_BITS + cfg.alpha_shift;

                int64_t prod = static_cast<int64_t>(int_scores[pos]) * M;
                int64_t t_fixed = (shift_for_t >= 0) ? (prod >> shift_for_t) : (prod << (-shift_for_t));

                tile_scaled_scores[c] = t_fixed;
                if (t_fixed > tile_max_fixed)
                {
                    tile_max_fixed = t_fixed;
                }

                std::cout << "  pos " << pos << ": int_score=" << int_scores[pos]
                          << " alpha=" << per_pos_alpha << " t_fixed=" << t_fixed << std::endl;
            }

            int32_t tile_max = static_cast<int32_t>(
                std::clamp(tile_max_fixed,
                           static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                           static_cast<int64_t>(std::numeric_limits<int32_t>::max())));

            std::cout << "  Tile max (fixed): " << tile_max << ", Running max: " << running_max_fixed << std::endl;

            // Check if we need to rescale
            if (tile == 0 || tile_max > running_max_fixed)
            {
                if (tile > 0)
                {
                    // Need to rescale previous weights
                    int32_t m_old = running_max_fixed;
                    int32_t m_new = tile_max;
                    int64_t delta = static_cast<int64_t>(m_old) - static_cast<int64_t>(m_new);

                    // Compute rescale factor
                    double delta_real = static_cast<double>(delta) / static_cast<double>(1 << FRAC_BITS);
                    double rescale_factor = std::exp2(delta_real);

                    std::cout << "  RESCALING: m_old=" << m_old << " m_new=" << m_new
                              << " delta=" << delta << " factor=" << rescale_factor << std::endl;

                    // Rescale previous weights
                    uint64_t new_sum = 0;
                    for (int i = 0; i < tile_start; ++i)
                    {
                        all_weights[i] = static_cast<uint32_t>(
                            static_cast<double>(all_weights[i]) * rescale_factor);
                        new_sum += all_weights[i];
                    }
                    running_sum = new_sum;
                }
                running_max_fixed = tile_max;
            }

            // Compute weights for this tile
            for (int c = 0; c < TILE_SIZE; ++c)
            {
                int pos = tile_start + c;
                int64_t delta_64 = tile_scaled_scores[c] - static_cast<int64_t>(running_max_fixed);

                uint32_t w;
                if (delta_64 >= 0)
                {
                    w = one;
                }
                else if (delta_64 < -static_cast<int64_t>(32 << FRAC_BITS))
                {
                    w = 0;
                }
                else
                {
                    int64_t neg_delta = -delta_64;
                    int int_part = static_cast<int>(neg_delta >> FRAC_BITS);
                    int frac_part = static_cast<int>(neg_delta & ((1 << FRAC_BITS) - 1));

                    uint32_t lut_val = lut[frac_part];
                    w = (int_part < 32) ? (lut_val >> int_part) : 0;
                }

                all_weights[pos] = w;
                running_sum += w;

                std::cout << "  pos " << pos << ": delta=" << delta_64 << " weight=" << w << std::endl;
            }
        }

        // Normalize final weights
        std::vector<float> online_weights(TOTAL_POSITIONS);
        for (int i = 0; i < TOTAL_POSITIONS; ++i)
        {
            online_weights[i] = static_cast<float>(all_weights[i]) / static_cast<float>(running_sum);
        }

        std::cout << "\nFinal weight comparison:" << std::endl;
        for (int i = 0; i < TOTAL_POSITIONS; ++i)
        {
            std::cout << "  pos " << i << ": ref=" << ref_weights[i]
                      << " online=" << online_weights[i]
                      << " diff=" << std::abs(ref_weights[i] - online_weights[i])
                      << std::endl;
        }

        float cos_sim = cosine_similarity(ref_weights, online_weights);
        float rms = rmse(ref_weights, online_weights);
        std::cout << "\nMulti-tile with varying K scales:" << std::endl;
        std::cout << "  Cosine similarity: " << cos_sim << std::endl;
        std::cout << "  RMSE: " << rms << std::endl;

        // This tests the rescaling path which is suspected to be buggy
        if (cos_sim < 0.95f)
        {
            std::cout << "\n*** BUG DETECTED: Multi-tile rescaling with varying K scales fails ***" << std::endl;
            std::cout << "This confirms the hypothesis from the handover document:" << std::endl;
            std::cout << "  - First tile (like position 0) works fine" << std::endl;
            std::cout << "  - Rescaling when new max appears corrupts the weights" << std::endl;
        }

        EXPECT_GT(cos_sim, 0.90f) << "Multi-tile rescaling should maintain reasonable accuracy";
    }

    // ============================================================================
    // Test: Isolate the rescaling math
    // ============================================================================

    TEST_F(OnlineSoftmaxPerPositionKScalesTest, RescalingMath_DifferentAlphas)
    {
        // This test isolates just the rescaling computation
        // to verify it handles different alpha values correctly

        std::cout << "\n=== Rescaling Math Test ===" << std::endl;

        // Scenario: We have computed weights for positions with alpha_1
        // Now we encounter a new max from positions with alpha_2
        // The rescale factor should correctly downscale the old weights

        float alpha_1 = 0.00024f; // base_alpha * k_scale[0]
        float alpha_2 = 0.00027f; // base_alpha * k_scale[4] (10% higher)

        // Old max was 1000 with alpha_1
        // New max is 1200 with alpha_2

        // The issue: max values are in "scaled domain" where:
        //   scaled_score = int_score * alpha * log2(e) * 2^frac_bits
        //
        // But different alphas mean the scaled_scores are NOT directly comparable!

        int32_t int_score_old = 500000;
        int32_t int_score_new = 550000;

        auto cfg_1 = AdaptiveAlphaConfig::compute(alpha_1);
        auto cfg_2 = AdaptiveAlphaConfig::compute(alpha_2);

        // Compute scaled scores
        auto compute_scaled_score = [](int32_t int_score, const AdaptiveAlphaConfig &cfg) -> int64_t
        {
            double effective_beta = static_cast<double>(cfg.effective_alpha) * LOG2E;
            int64_t M = static_cast<int64_t>(
                std::llround(effective_beta * static_cast<double>(1ULL << 24)));
            int shift_for_t = 24 - FRAC_BITS + cfg.alpha_shift;
            int64_t prod = static_cast<int64_t>(int_score) * M;
            return (shift_for_t >= 0) ? (prod >> shift_for_t) : (prod << (-shift_for_t));
        };

        int64_t scaled_old = compute_scaled_score(int_score_old, cfg_1);
        int64_t scaled_new = compute_scaled_score(int_score_new, cfg_2);

        // Compute what the REAL scores are
        float real_score_old = static_cast<float>(int_score_old) * alpha_1;
        float real_score_new = static_cast<float>(int_score_new) * alpha_2;

        std::cout << "Old score: int=" << int_score_old << " alpha=" << alpha_1
                  << " scaled=" << scaled_old << " real=" << real_score_old << std::endl;
        std::cout << "New score: int=" << int_score_new << " alpha=" << alpha_2
                  << " scaled=" << scaled_new << " real=" << real_score_new << std::endl;

        // The REAL softmax weight for old score after seeing new max:
        //   w_old = exp(real_score_old - real_score_new)
        //         = exp(120 - 148.5) = exp(-28.5) ≈ 4.1e-13

        float true_exp_diff = std::exp(real_score_old - real_score_new);
        std::cout << "True exp(old - new) = " << true_exp_diff << std::endl;

        // What the integer path computes:
        //   delta = scaled_old - scaled_new (in fixed-point domain)
        //   This is WRONG because scaled_old and scaled_new use different alphas!

        int64_t delta_fixed = scaled_old - scaled_new;
        double delta_real = static_cast<double>(delta_fixed) / static_cast<double>(1 << FRAC_BITS);
        double int_path_exp = std::exp2(delta_real);

        std::cout << "Integer path delta (fixed): " << delta_fixed << std::endl;
        std::cout << "Integer path exp2(delta): " << int_path_exp << std::endl;

        // Compare
        float ratio = static_cast<float>(int_path_exp / (true_exp_diff + 1e-20));
        std::cout << "Ratio (int_path / true): " << ratio << std::endl;

        // The key insight: if alphas differ, the ratio will NOT be 1.0
        // because we're comparing apples and oranges in the scaled domain

        if (std::abs(ratio - 1.0f) > 0.1f)
        {
            std::cout << "\n*** BUG CONFIRMED: Mixing different alphas in scaled domain is incorrect ***" << std::endl;
            std::cout << "The scaled_score values are NOT in the same units when alphas differ!" << std::endl;
            std::cout << "This explains why first position works but subsequent positions fail." << std::endl;
        }

        // This test documents the bug - we expect the ratio to NOT be 1.0
        // A fix would need to track max in a common scale or convert back to real domain
    }

    // ============================================================================
    // Test: First position vs subsequent positions
    // ============================================================================

    TEST_F(OnlineSoftmaxPerPositionKScalesTest, FirstPosition_vs_SubsequentPositions)
    {
        // This test verifies the handover observation:
        // - Position 0 (first) has excellent accuracy
        // - Subsequent positions degrade

        std::cout << "\n=== First Position vs Subsequent Positions ===" << std::endl;

        auto int_scores = generate_synthetic_scores(NUM_POSITIONS, 42);
        auto V = generate_synthetic_V(NUM_POSITIONS, HEAD_DIM, 123);

        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        float base_alpha = Q_SCALE * inv_sqrt_d;

        // Test 1: Process only position 0
        {
            float per_pos_alpha = base_alpha * K_SCALES_VARYING[0];
            float real_score = static_cast<float>(int_scores[0]) * Q_SCALE * K_SCALES_VARYING[0] * inv_sqrt_d;

            // With only one position, the weight is always 1.0
            // This should be trivially correct
            std::cout << "Position 0 only: score=" << real_score << " weight=1.0 (trivial)" << std::endl;
        }

        // Test 2: Process positions 0-1 (first rescaling event)
        {
            std::vector<float> real_scores(2);
            for (int i = 0; i < 2; ++i)
            {
                real_scores[i] = static_cast<float>(int_scores[i]) * Q_SCALE * K_SCALES_VARYING[i] * inv_sqrt_d;
            }
            auto ref_weights = reference_softmax(real_scores);

            // Simulate per-position processing
            std::vector<AdaptiveAlphaConfig> cfgs(2);
            std::vector<int64_t> scaled_scores(2);
            int64_t max_fixed = std::numeric_limits<int64_t>::min();

            for (int i = 0; i < 2; ++i)
            {
                cfgs[i] = AdaptiveAlphaConfig::compute(base_alpha * K_SCALES_VARYING[i]);
                double effective_beta = static_cast<double>(cfgs[i].effective_alpha) * LOG2E;
                int64_t M = static_cast<int64_t>(
                    std::llround(effective_beta * static_cast<double>(1ULL << 24)));
                int shift_for_t = 24 - FRAC_BITS + cfgs[i].alpha_shift;
                int64_t prod = static_cast<int64_t>(int_scores[i]) * M;
                scaled_scores[i] = (shift_for_t >= 0) ? (prod >> shift_for_t) : (prod << (-shift_for_t));
                if (scaled_scores[i] > max_fixed)
                    max_fixed = scaled_scores[i];
            }

            const uint32_t *lut = get_exp2_lut_data();
            const uint32_t one = static_cast<uint32_t>(1U << LUT_VALUE_BITS);
            std::vector<uint32_t> weights(2);
            uint64_t sum = 0;

            for (int i = 0; i < 2; ++i)
            {
                int64_t delta = scaled_scores[i] - max_fixed;
                uint32_t w;
                if (delta >= 0)
                {
                    w = one;
                }
                else if (delta < -static_cast<int64_t>(32 << FRAC_BITS))
                {
                    w = 0;
                }
                else
                {
                    int64_t neg_delta = -delta;
                    int int_part = static_cast<int>(neg_delta >> FRAC_BITS);
                    int frac_part = static_cast<int>(neg_delta & ((1 << FRAC_BITS) - 1));
                    w = (int_part < 32) ? (lut[frac_part] >> int_part) : 0;
                }
                weights[i] = w;
                sum += w;
            }

            std::vector<float> online_weights(2);
            for (int i = 0; i < 2; ++i)
            {
                online_weights[i] = static_cast<float>(weights[i]) / static_cast<float>(sum);
            }

            float cos_sim = cosine_similarity(ref_weights, online_weights);
            std::cout << "Positions 0-1: cosine=" << cos_sim << std::endl;
            std::cout << "  ref: " << ref_weights[0] << ", " << ref_weights[1] << std::endl;
            std::cout << "  online: " << online_weights[0] << ", " << online_weights[1] << std::endl;
        }

        // Test 3: All positions
        {
            std::vector<float> real_scores(NUM_POSITIONS);
            for (int i = 0; i < NUM_POSITIONS; ++i)
            {
                real_scores[i] = static_cast<float>(int_scores[i]) * Q_SCALE * K_SCALES_VARYING[i] * inv_sqrt_d;
            }
            auto ref_weights = reference_softmax(real_scores);

            std::vector<AdaptiveAlphaConfig> cfgs(NUM_POSITIONS);
            std::vector<int64_t> scaled_scores(NUM_POSITIONS);
            int64_t max_fixed = std::numeric_limits<int64_t>::min();

            for (int i = 0; i < NUM_POSITIONS; ++i)
            {
                cfgs[i] = AdaptiveAlphaConfig::compute(base_alpha * K_SCALES_VARYING[i]);
                double effective_beta = static_cast<double>(cfgs[i].effective_alpha) * LOG2E;
                int64_t M = static_cast<int64_t>(
                    std::llround(effective_beta * static_cast<double>(1ULL << 24)));
                int shift_for_t = 24 - FRAC_BITS + cfgs[i].alpha_shift;
                int64_t prod = static_cast<int64_t>(int_scores[i]) * M;
                scaled_scores[i] = (shift_for_t >= 0) ? (prod >> shift_for_t) : (prod << (-shift_for_t));
                if (scaled_scores[i] > max_fixed)
                    max_fixed = scaled_scores[i];
            }

            const uint32_t *lut = get_exp2_lut_data();
            const uint32_t one = static_cast<uint32_t>(1U << LUT_VALUE_BITS);
            std::vector<uint32_t> weights(NUM_POSITIONS);
            uint64_t sum = 0;

            for (int i = 0; i < NUM_POSITIONS; ++i)
            {
                int64_t delta = scaled_scores[i] - max_fixed;
                uint32_t w;
                if (delta >= 0)
                {
                    w = one;
                }
                else if (delta < -static_cast<int64_t>(32 << FRAC_BITS))
                {
                    w = 0;
                }
                else
                {
                    int64_t neg_delta = -delta;
                    int int_part = static_cast<int>(neg_delta >> FRAC_BITS);
                    int frac_part = static_cast<int>(neg_delta & ((1 << FRAC_BITS) - 1));
                    w = (int_part < 32) ? (lut[frac_part] >> int_part) : 0;
                }
                weights[i] = w;
                sum += w;
            }

            std::vector<float> online_weights(NUM_POSITIONS);
            for (int i = 0; i < NUM_POSITIONS; ++i)
            {
                online_weights[i] = static_cast<float>(weights[i]) / static_cast<float>(sum);
            }

            float cos_sim = cosine_similarity(ref_weights, online_weights);
            std::cout << "All positions: cosine=" << cos_sim << std::endl;
        }
    }

    // ============================================================================
    // Test: The ROOT CAUSE - scaled_score domain incompatibility
    // ============================================================================

    TEST_F(OnlineSoftmaxPerPositionKScalesTest, RootCause_ScaledScoreDomainIncompatibility)
    {
        // This test demonstrates the fundamental mathematical issue:
        //
        // In the per-position K scale path, we compute:
        //   scaled_score[k] = int_score[k] * alpha[k] * log2(e) * 2^frac_bits
        //
        // Then we find max across positions and compute weights as:
        //   weight[k] = exp2(scaled_score[k] - max_scaled_score)
        //
        // THE BUG: When alpha[k] varies, scaled_score values are NOT in the same
        // mathematical domain. Finding max and computing deltas is like comparing
        // meters and feet without conversion.
        //
        // CORRECT APPROACH: The max and delta computation should be in the
        // REAL score domain (after full dequantization), not the scaled integer domain.

        std::cout << "\n=== ROOT CAUSE DEMONSTRATION ===" << std::endl;

        // Two positions with different K scales
        int32_t int_score_a = 100000;
        int32_t int_score_b = 100000; // Same integer score!

        float k_scale_a = 0.008f;
        float k_scale_b = 0.016f; // 2x different K scale

        float inv_sqrt_d = 1.0f / std::sqrt(64.0f);
        float alpha_a = Q_SCALE * k_scale_a * inv_sqrt_d;
        float alpha_b = Q_SCALE * k_scale_b * inv_sqrt_d;

        // REAL scores (what softmax should see)
        float real_a = static_cast<float>(int_score_a) * alpha_a;
        float real_b = static_cast<float>(int_score_b) * alpha_b;

        std::cout << "Two positions with SAME int_score but DIFFERENT K scales:" << std::endl;
        std::cout << "  Position A: int=" << int_score_a << " k_scale=" << k_scale_a
                  << " alpha=" << alpha_a << " REAL=" << real_a << std::endl;
        std::cout << "  Position B: int=" << int_score_b << " k_scale=" << k_scale_b
                  << " alpha=" << alpha_b << " REAL=" << real_b << std::endl;

        // CORRECT softmax (on real scores)
        float max_real = std::max(real_a, real_b);
        float w_a_correct = std::exp(real_a - max_real);
        float w_b_correct = std::exp(real_b - max_real);
        float sum_correct = w_a_correct + w_b_correct;
        w_a_correct /= sum_correct;
        w_b_correct /= sum_correct;

        std::cout << "\nCORRECT softmax (real domain):" << std::endl;
        std::cout << "  max_real=" << max_real << std::endl;
        std::cout << "  w_a=" << w_a_correct << " w_b=" << w_b_correct << std::endl;

        // BUGGY scaled domain computation
        auto compute_scaled = [](int32_t score, float alpha) -> int64_t
        {
            auto cfg = AdaptiveAlphaConfig::compute(alpha);
            double effective_beta = static_cast<double>(cfg.effective_alpha) * LOG2E;
            int64_t M = static_cast<int64_t>(
                std::llround(effective_beta * static_cast<double>(1ULL << 24)));
            int shift = 24 - FRAC_BITS + cfg.alpha_shift;
            int64_t prod = static_cast<int64_t>(score) * M;
            return (shift >= 0) ? (prod >> shift) : (prod << (-shift));
        };

        int64_t scaled_a = compute_scaled(int_score_a, alpha_a);
        int64_t scaled_b = compute_scaled(int_score_b, alpha_b);

        std::cout << "\nBUGGY scaled domain:" << std::endl;
        std::cout << "  scaled_a=" << scaled_a << std::endl;
        std::cout << "  scaled_b=" << scaled_b << std::endl;

        // The bug: scaled_b is 2x larger even though the INT scores are the same!
        // This is because scaled_score = int_score * alpha, and alpha_b = 2 * alpha_a

        int64_t max_scaled = std::max(scaled_a, scaled_b);
        std::cout << "  max_scaled=" << max_scaled << " (from position " << (scaled_b > scaled_a ? "B" : "A") << ")" << std::endl;

        const uint32_t *lut = get_exp2_lut_data();
        const uint32_t one = static_cast<uint32_t>(1U << LUT_VALUE_BITS);

        auto compute_weight = [&](int64_t scaled, int64_t max_s) -> uint32_t
        {
            int64_t delta = scaled - max_s;
            if (delta >= 0)
                return one;
            if (delta < -static_cast<int64_t>(32 << FRAC_BITS))
                return 0;
            int64_t neg_delta = -delta;
            int int_part = static_cast<int>(neg_delta >> FRAC_BITS);
            int frac_part = static_cast<int>(neg_delta & ((1 << FRAC_BITS) - 1));
            return (int_part < 32) ? (lut[frac_part] >> int_part) : 0;
        };

        uint32_t w_a_buggy = compute_weight(scaled_a, max_scaled);
        uint32_t w_b_buggy = compute_weight(scaled_b, max_scaled);
        float sum_buggy = static_cast<float>(w_a_buggy) + static_cast<float>(w_b_buggy);
        float w_a_norm = static_cast<float>(w_a_buggy) / sum_buggy;
        float w_b_norm = static_cast<float>(w_b_buggy) / sum_buggy;

        std::cout << "\nBUGGY softmax weights:" << std::endl;
        std::cout << "  w_a=" << w_a_norm << " w_b=" << w_b_norm << std::endl;

        std::cout << "\nCOMPARISON:" << std::endl;
        std::cout << "  Correct w_a=" << w_a_correct << " vs Buggy w_a=" << w_a_norm
                  << " (error=" << std::abs(w_a_correct - w_a_norm) << ")" << std::endl;
        std::cout << "  Correct w_b=" << w_b_correct << " vs Buggy w_b=" << w_b_norm
                  << " (error=" << std::abs(w_b_correct - w_b_norm) << ")" << std::endl;

        float error_a = std::abs(w_a_correct - w_a_norm);
        float error_b = std::abs(w_b_correct - w_b_norm);

        if (error_a > 0.01f || error_b > 0.01f)
        {
            std::cout << "\n*** ROOT CAUSE CONFIRMED ***" << std::endl;
            std::cout << "When K scales differ, the scaled_score domain is incompatible." << std::endl;
            std::cout << "Position B has 2x higher K scale, so its scaled_score is 2x larger," << std::endl;
            std::cout << "making it appear to have a much higher score than it really does." << std::endl;
            std::cout << "\nFIX REQUIRED: Track max in REAL score domain, not scaled domain." << std::endl;
        }

        // This test documents the bug
        EXPECT_GT(error_a + error_b, 0.001f)
            << "Expected significant error due to domain incompatibility";
    }

} // namespace

// ============================================================================
// Test: THE ACTUAL BUG - K_block.d vs kv_head_scales mismatch
// ============================================================================

namespace
{

    /**
     * @brief Test the ACTUAL bug from the pipeline:
     *
     * The `.d` field in K cache blocks contains the Q16_1 quantization scale (~0.008),
     * NOT the kv_head_scale (~0.000244). This causes a ~4x alpha mismatch.
     *
     * From handover:
     * - Uniform path: qk_scale = 0.00195 * 0.000244 / 8 ≈ 4.77e-07 ✓
     * - Per-position path: alpha = 0.000244 * 0.00796 = 1.94e-06 (4x too large!) ✗
     */
    TEST_F(OnlineSoftmaxPerPositionKScalesTest, ActualBug_KBlockDotD_vs_KVHeadScales)
    {
        // This test reproduces the EXACT bug from the integration test

        std::cout << "\n=== ACTUAL BUG: K_block.d vs kv_head_scales ===" << std::endl;

        // From the debug output:
        constexpr float Q_SCALE_ACTUAL = 0.00195318f;
        constexpr float KV_HEAD_SCALE = 0.000244148f; // kv_head_scales value
        constexpr float K_BLOCK_D = 0.00795628f;      // K_block.d value (quantization scale)
        constexpr float INV_SQRT_D = 1.0f / 8.0f;     // 1/sqrt(64)

        // CORRECT alpha (what the uniform path uses)
        float alpha_correct = Q_SCALE_ACTUAL * KV_HEAD_SCALE * INV_SQRT_D;

        // BUGGY alpha (what the per-position path computes)
        float base_alpha = Q_SCALE_ACTUAL * INV_SQRT_D;
        float alpha_buggy = base_alpha * K_BLOCK_D;

        std::cout << "Q scale: " << Q_SCALE_ACTUAL << std::endl;
        std::cout << "kv_head_scale: " << KV_HEAD_SCALE << std::endl;
        std::cout << "K_block.d: " << K_BLOCK_D << std::endl;
        std::cout << "inv_sqrt_d: " << INV_SQRT_D << std::endl;
        std::cout << std::endl;
        std::cout << "CORRECT alpha (uniform path): " << alpha_correct << std::endl;
        std::cout << "BUGGY alpha (per-position path): " << alpha_buggy << std::endl;
        std::cout << "Ratio (buggy/correct): " << (alpha_buggy / alpha_correct) << "x" << std::endl;

        // Generate test scores
        auto int_scores = generate_synthetic_scores(NUM_POSITIONS, 42);

        // CORRECT softmax (using correct alpha)
        std::vector<float> real_scores_correct(NUM_POSITIONS);
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            real_scores_correct[i] = static_cast<float>(int_scores[i]) * alpha_correct;
        }
        auto ref_weights = reference_softmax(real_scores_correct);

        // BUGGY softmax (using buggy alpha)
        std::vector<float> real_scores_buggy(NUM_POSITIONS);
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            real_scores_buggy[i] = static_cast<float>(int_scores[i]) * alpha_buggy;
        }
        auto buggy_weights = reference_softmax(real_scores_buggy);

        std::cout << "\nScores and weights:" << std::endl;
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            std::cout << "  pos " << i << ": int=" << int_scores[i]
                      << " correct_real=" << real_scores_correct[i]
                      << " buggy_real=" << real_scores_buggy[i]
                      << " | ref_w=" << ref_weights[i]
                      << " buggy_w=" << buggy_weights[i]
                      << std::endl;
        }

        float cos_sim = cosine_similarity(ref_weights, buggy_weights);
        float rms_err = rmse(ref_weights, buggy_weights);

        std::cout << "\nComparison:" << std::endl;
        std::cout << "  Cosine similarity: " << cos_sim << std::endl;
        std::cout << "  RMSE: " << rms_err << std::endl;

        // With 4x alpha, the softmax becomes much more "peaked" - small differences
        // in scores get amplified. This leads to different weight distributions.
        //
        // When alpha is too large:
        // - exp(-large_alpha * small_delta) -> 0 very quickly
        // - Only the position with max score gets significant weight
        // - All other positions get near-zero weight
        //
        // This explains the severe cosine similarity degradation!

        // Find which position has max in each case
        int max_pos_correct = std::distance(ref_weights.begin(),
                                            std::max_element(ref_weights.begin(), ref_weights.end()));
        int max_pos_buggy = std::distance(buggy_weights.begin(),
                                          std::max_element(buggy_weights.begin(), buggy_weights.end()));

        std::cout << "\nMax position: correct=" << max_pos_correct << " buggy=" << max_pos_buggy << std::endl;
        std::cout << "Max weight: correct=" << ref_weights[max_pos_correct]
                  << " buggy=" << buggy_weights[max_pos_buggy] << std::endl;

        // The buggy path should show much more peaked distribution
        float entropy_correct = 0.0f, entropy_buggy = 0.0f;
        for (int i = 0; i < NUM_POSITIONS; ++i)
        {
            if (ref_weights[i] > 1e-10f)
                entropy_correct -= ref_weights[i] * std::log2(ref_weights[i]);
            if (buggy_weights[i] > 1e-10f)
                entropy_buggy -= buggy_weights[i] * std::log2(buggy_weights[i]);
        }
        std::cout << "\nEntropy (higher = more uniform):" << std::endl;
        std::cout << "  Correct: " << entropy_correct << " bits" << std::endl;
        std::cout << "  Buggy:   " << entropy_buggy << " bits" << std::endl;

        if (entropy_buggy < entropy_correct * 0.9f)
        {
            std::cout << "\n*** BUG CONFIRMED: Buggy alpha makes softmax too peaked! ***" << std::endl;
            std::cout << "The 4x larger alpha causes the softmax to concentrate" << std::endl;
            std::cout << "weight on fewer positions, losing information from other positions." << std::endl;
        }

        // We expect significant degradation
        EXPECT_LT(cos_sim, 0.99f)
            << "4x alpha mismatch should cause significant weight divergence";
    }

    /**
     * @brief Test what happens with EXACTLY the formula from the buggy path
     *
     * The handover shows the per-position path does:
     *   alpha[c] = base_alpha_fp32 * K_block.d
     *
     * Where:
     *   base_alpha_fp32 = q_scale / sqrt(head_dim)  [no kv_scale!]
     *   K_block.d = quantization scale (~1/128 or similar)
     *
     * The CORRECT formula should be:
     *   alpha = q_scale * kv_head_scale / sqrt(head_dim)
     *
     * The bug is that K_block.d != kv_head_scale!
     */
    TEST_F(OnlineSoftmaxPerPositionKScalesTest, ExactBuggyFormula_Analysis)
    {
        std::cout << "\n=== Exact Buggy Formula Analysis ===" << std::endl;

        // Realistic values from actual pipeline
        constexpr float Q_SCALE = 0.00195318f;
        constexpr float KV_CACHE_SCALE = 256.0f;
        constexpr float KV_HEAD_SCALE = 1.0f / (KV_CACHE_SCALE * 16.0f); // ~0.000244
        constexpr int SQRT_HEAD_DIM = 8;                                 // sqrt(64)

        // What the Q16_1Block.d field contains
        // This is derived from how K was quantized:
        // K_int16 = round(K_fp32 / K_block.d)
        // So K_block.d = max_abs(K_fp32) / 32767 roughly
        // From handover: ~0.008 for typical K values

        float k_block_d_typical = 0.00795628f;

        std::cout << "Pipeline values:" << std::endl;
        std::cout << "  Q_SCALE: " << Q_SCALE << std::endl;
        std::cout << "  KV_CACHE_SCALE: " << KV_CACHE_SCALE << std::endl;
        std::cout << "  KV_HEAD_SCALE (derived): " << KV_HEAD_SCALE << std::endl;
        std::cout << "  K_block.d (typical): " << k_block_d_typical << std::endl;
        std::cout << std::endl;

        // CORRECT qk_scale computation
        float qk_scale_correct = Q_SCALE * KV_HEAD_SCALE / SQRT_HEAD_DIM;
        std::cout << "CORRECT qk_scale = Q_SCALE * KV_HEAD_SCALE / sqrt(head_dim)" << std::endl;
        std::cout << "                 = " << Q_SCALE << " * " << KV_HEAD_SCALE << " / " << SQRT_HEAD_DIM << std::endl;
        std::cout << "                 = " << qk_scale_correct << std::endl;

        // BUGGY qk_scale computation (what the per-position path does)
        float base_alpha = Q_SCALE / SQRT_HEAD_DIM;
        float qk_scale_buggy = base_alpha * k_block_d_typical;
        std::cout << "\nBUGGY qk_scale = (Q_SCALE / sqrt(head_dim)) * K_block.d" << std::endl;
        std::cout << "               = " << base_alpha << " * " << k_block_d_typical << std::endl;
        std::cout << "               = " << qk_scale_buggy << std::endl;

        float ratio = qk_scale_buggy / qk_scale_correct;
        std::cout << "\nRatio (buggy/correct): " << ratio << "x" << std::endl;

        // Compute what the K_block.d SHOULD be if it were meant to replace kv_head_scale
        float k_block_d_expected = KV_HEAD_SCALE;
        std::cout << "\nIf K_block.d were meant to replace kv_head_scale:" << std::endl;
        std::cout << "  K_block.d SHOULD be: " << k_block_d_expected << std::endl;
        std::cout << "  K_block.d ACTUAL is: " << k_block_d_typical << std::endl;
        std::cout << "  Mismatch ratio: " << (k_block_d_typical / k_block_d_expected) << "x" << std::endl;

        // The conclusion
        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "CONCLUSION: The bug is using K_block.d (quantization scale)" << std::endl;
        std::cout << "instead of kv_head_scale (normalization scale)." << std::endl;
        std::cout << std::endl;
        std::cout << "K_block.d ≈ max_abs(K) / 32767 ≈ 130 / 32767 ≈ 0.004-0.008" << std::endl;
        std::cout << "kv_head_scale = 1 / (KV_CACHE_SCALE * 16) ≈ 0.000244" << std::endl;
        std::cout << std::endl;
        std::cout << "These are ~30x different, causing alpha to be ~30x too large," << std::endl;
        std::cout << "which makes softmax extremely peaked (low entropy)." << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Verify the ratio is large
        EXPECT_GT(ratio, 3.0f) << "K_block.d should be significantly different from kv_head_scale";
    }

    /**
     * @brief Demonstrate the effect of alpha magnitude on softmax peakedness
     *
     * softmax(alpha * scores) becomes more peaked as alpha increases:
     * - alpha = 1: normal softmax
     * - alpha = 4: much more peaked
     * - alpha = 30: almost one-hot
     */
    TEST_F(OnlineSoftmaxPerPositionKScalesTest, AlphaMagnitude_SoftmaxPeakedness)
    {
        std::cout << "\n=== Alpha Magnitude vs Softmax Peakedness ===" << std::endl;

        // Use simple integer scores with known distribution
        std::vector<float> base_scores = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f, 0.0f, -0.2f, -0.4f, -0.6f};
        int n = base_scores.size();

        std::vector<float> alpha_values = {1.0f, 4.0f, 30.0f, 100.0f};

        for (float alpha : alpha_values)
        {
            // Scale scores by alpha
            std::vector<float> scaled_scores(n);
            for (int i = 0; i < n; ++i)
            {
                scaled_scores[i] = alpha * base_scores[i];
            }

            // Compute softmax
            auto weights = reference_softmax(scaled_scores);

            // Compute entropy
            float entropy = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                if (weights[i] > 1e-10f)
                    entropy -= weights[i] * std::log2(weights[i]);
            }

            // Find max weight and its position
            auto max_it = std::max_element(weights.begin(), weights.end());
            int max_pos = std::distance(weights.begin(), max_it);
            float max_weight = *max_it;

            std::cout << "alpha=" << std::setw(5) << alpha
                      << " | max_weight=" << std::fixed << std::setprecision(4) << max_weight
                      << " (pos " << max_pos << ")"
                      << " | entropy=" << std::setprecision(3) << entropy << " bits"
                      << " | weights: ";
            for (int i = 0; i < std::min(5, n); ++i)
            {
                std::cout << std::setprecision(3) << weights[i] << " ";
            }
            std::cout << "..." << std::endl;
        }

        std::cout << "\nConclusion: As alpha increases, softmax becomes more peaked," << std::endl;
        std::cout << "concentrating weight on fewer positions and losing information." << std::endl;
        std::cout << "With alpha=30, almost all weight goes to position 0." << std::endl;
    }

    /**
     * @brief Simulate the EXACT integration test scenario
     *
     * The integration test shows ATTENTION_CONTEXT cosine of 0.04-0.60.
     * Let's see if our buggy alpha (32x too large) can reproduce this.
     */
    TEST_F(OnlineSoftmaxPerPositionKScalesTest, SimulateIntegrationTestScenario)
    {
        std::cout << "\n=== Simulating Integration Test Scenario ===" << std::endl;

        // Values from actual pipeline
        constexpr float Q_SCALE = 0.00195318f;
        constexpr float KV_HEAD_SCALE = 0.000244148f;
        constexpr float K_BLOCK_D = 0.00795628f;
        constexpr float INV_SQRT_D = 0.125f;

        float alpha_correct = Q_SCALE * KV_HEAD_SCALE * INV_SQRT_D;
        float alpha_buggy = (Q_SCALE * INV_SQRT_D) * K_BLOCK_D;

        // Generate synthetic V values
        auto V = generate_synthetic_V(NUM_POSITIONS, HEAD_DIM, 123);

        // Test multiple random score distributions
        std::vector<int> seeds = {42, 123, 456, 789, 1011};
        std::vector<float> cosine_similarities;

        for (int seed : seeds)
        {
            auto int_scores = generate_synthetic_scores(NUM_POSITIONS, seed);

            // CORRECT attention output
            std::vector<float> correct_scores(NUM_POSITIONS);
            for (int i = 0; i < NUM_POSITIONS; ++i)
            {
                correct_scores[i] = static_cast<float>(int_scores[i]) * alpha_correct;
            }
            auto correct_weights = reference_softmax(correct_scores);
            auto correct_output = reference_attention(correct_weights, V, HEAD_DIM);

            // BUGGY attention output
            std::vector<float> buggy_scores(NUM_POSITIONS);
            for (int i = 0; i < NUM_POSITIONS; ++i)
            {
                buggy_scores[i] = static_cast<float>(int_scores[i]) * alpha_buggy;
            }
            auto buggy_weights = reference_softmax(buggy_scores);
            auto buggy_output = reference_attention(buggy_weights, V, HEAD_DIM);

            float cos_sim = cosine_similarity(correct_output, buggy_output);
            cosine_similarities.push_back(cos_sim);

            std::cout << "Seed " << seed << ": attention output cosine = " << cos_sim << std::endl;
        }

        // Compute statistics
        float sum = std::accumulate(cosine_similarities.begin(), cosine_similarities.end(), 0.0f);
        float mean_cos = sum / cosine_similarities.size();
        float min_cos = *std::min_element(cosine_similarities.begin(), cosine_similarities.end());
        float max_cos = *std::max_element(cosine_similarities.begin(), cosine_similarities.end());

        std::cout << "\nStatistics across " << seeds.size() << " random seeds:" << std::endl;
        std::cout << "  Min cosine: " << min_cos << std::endl;
        std::cout << "  Max cosine: " << max_cos << std::endl;
        std::cout << "  Mean cosine: " << mean_cos << std::endl;

        std::cout << "\nIntegration test showed ATTENTION_CONTEXT cosine of 0.04-0.74." << std::endl;
        std::cout << "Our simulated scenario shows cosine of " << min_cos << "-" << max_cos << std::endl;

        // NOTE: Our simulation shows BETTER results than the integration test because:
        //
        // 1. We use RANDOM V values, not real V values that have structure
        //    Real V has correlated patterns; random V averages out errors
        //
        // 2. We're testing a SINGLE softmax + weighted sum
        //    The integration test has:
        //    - Per-head effects (some heads worse than others)
        //    - Per-row effects (row 0 is correct, rows 1-8 fail)
        //    - Multi-layer compounding through residual connections
        //
        // 3. The integration test's WORST case (cosine 0.04) is from:
        //    - Specific heads (h11, h13, h7 with cosine 0.26-0.33)
        //    - Specific rows (r5, r3 with cosine 0.35-0.37)
        //
        // This unit test confirms the MECHANISM is correct:
        // - 32x alpha causes peaked softmax
        // - Peaked softmax changes weight distribution
        // - Changed weights cause attention output divergence
        //
        // The severity in the integration test is amplified by:
        // - Real data having more extreme score distributions
        // - Per-head scale variations (GQA mapping affects different heads differently)
        // - Causal masking effects (first row sees all positions, last row sees only one)

        // We should still see significant degradation (cosine < 0.95)
        EXPECT_LT(mean_cos, 0.95f)
            << "32x alpha mismatch should cause significant attention output divergence";
    }

} // namespace
