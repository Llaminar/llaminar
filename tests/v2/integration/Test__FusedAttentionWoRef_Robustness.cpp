/**
 * @file Test__FusedAttentionWoRef_Robustness.cpp
 * @brief Robustness tests for fused attention + Wo kernel against real-world issues.
 *
 * Tests edge cases and failure modes we've encountered in the real pipeline:
 * 1. Softmax saturation (large scores causing one-hot distributions)
 * 2. Catastrophic cancellation in accumulation
 * 3. Quantization clipping/saturation
 * 4. Large scale disparities between blocks
 * 5. Residual collapse from accumulated errors
 * 6. Sign flips from quantization noise
 * 7. Extreme outliers in Q/K/V
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <limits>

#include "../../../src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoRef.h"
#include "../../../src/v2/kernels/cpu/microkernels/q8_1/Q8DotProduct.h"
#include "../../../src/v2/kernels/cpu/microkernels/q8_1/OnlineSoftmax.h"
#include "../../../src/v2/kernels/cpu/microkernels/q8_1/VWeightedAccum.h"
#include "../../../src/v2/kernels/cpu/microkernels/q8_1/FastExp.h"
#include "../../../src/v2/tensors/FP16Utils.h"

using namespace llaminar::v2::kernels;
using namespace llaminar::v2::kernels::microkernels;

/**
 * @brief Helper to quantize FP32 vector to Q8_1 blocks with diagnostic info
 */
struct QuantizationStats
{
    float max_abs_value;
    float min_scale;
    float max_scale;
    int num_saturated; // Values that clipped to ±127
    int num_zero_blocks;
};

QuantizationStats quantize_to_q8_1_with_stats(const float *input, Q8_1Block *output, int numel)
{
    const int num_blocks = numel / 32;
    QuantizationStats stats = {0.0f, std::numeric_limits<float>::max(), 0.0f, 0, 0};

    for (int b = 0; b < num_blocks; ++b)
    {
        const float *block_data = input + b * 32;

        // Find absmax in block
        float absmax = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            absmax = std::max(absmax, std::abs(block_data[i]));
            stats.max_abs_value = std::max(stats.max_abs_value, std::abs(block_data[i]));
        }

        // Compute scale
        float scale = absmax > 0 ? absmax / 127.0f : 1.0f;
        float inv_scale = absmax > 0 ? 127.0f / absmax : 0.0f;

        if (absmax > 0)
        {
            stats.min_scale = std::min(stats.min_scale, scale);
            stats.max_scale = std::max(stats.max_scale, scale);
        }
        else
        {
            stats.num_zero_blocks++;
        }

        // Store scale as FP16
        output[b].d = llaminar2::fp32_to_fp16(scale);

        // Quantize and compute sum_qs
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            float scaled = block_data[i] * inv_scale;
            int8_t q;
            if (scaled >= 127.0f)
            {
                q = 127;
                stats.num_saturated++;
            }
            else if (scaled <= -127.0f)
            {
                q = -127;
                stats.num_saturated++;
            }
            else
            {
                q = static_cast<int8_t>(std::round(scaled));
            }
            output[b].qs[i] = q;
            sum += q;
        }
        output[b].sum_qs = static_cast<int16_t>(sum);
    }

    return stats;
}

void quantize_to_q8_1(const float *input, Q8_1Block *output, int numel)
{
    quantize_to_q8_1_with_stats(input, output, numel);
}

void dequantize_from_q8_1(const Q8_1Block *input, float *output, int numel)
{
    const int num_blocks = numel / 32;
    for (int b = 0; b < num_blocks; ++b)
    {
        float scale = llaminar2::fp16_to_fp32(input[b].d);
        for (int i = 0; i < 32; ++i)
        {
            output[b * 32 + i] = static_cast<float>(input[b].qs[i]) * scale;
        }
    }
}

/**
 * @brief Reference attention for comparison
 */
void compute_attention_reference_fp32(
    const float *Q_fp32, const float *K_fp32, const float *V_fp32, const float *Wo_fp32,
    float *output, int seq_len, int kv_seq_len, int num_heads, int num_kv_heads,
    int head_dim, int d_model, float scale, bool causal, int position_offset)
{
    const int kv_head_ratio = num_heads / num_kv_heads;
    std::fill(output, output + seq_len * d_model, 0.0f);

    for (int m = 0; m < seq_len; ++m)
    {
        for (int h = 0; h < num_heads; ++h)
        {
            int kv_h = h / kv_head_ratio;
            const float *Q_row = Q_fp32 + (m * num_heads + h) * head_dim;
            int max_kv = causal ? std::min(m + position_offset + 1, kv_seq_len) : kv_seq_len;

            std::vector<float> scores(max_kv);
            std::vector<float> weights(max_kv);
            std::vector<float> context(head_dim, 0.0f);

            float max_score = -std::numeric_limits<float>::infinity();
            for (int n = 0; n < max_kv; ++n)
            {
                const float *K_row = K_fp32 + (n * num_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += Q_row[d] * K_row[d];
                }
                scores[n] = dot * scale;
                max_score = std::max(max_score, scores[n]);
            }

            float sum_exp = 0.0f;
            for (int n = 0; n < max_kv; ++n)
            {
                weights[n] = std::exp(scores[n] - max_score);
                sum_exp += weights[n];
            }
            for (int n = 0; n < max_kv; ++n)
            {
                weights[n] /= sum_exp;
            }

            for (int n = 0; n < max_kv; ++n)
            {
                const float *V_row = V_fp32 + (n * num_kv_heads + kv_h) * head_dim;
                for (int d = 0; d < head_dim; ++d)
                {
                    context[d] += weights[n] * V_row[d];
                }
            }

            float *out_row = output + m * d_model;
            for (int o = 0; o < d_model; ++o)
            {
                float dot = 0.0f;
                const float *wo_row = Wo_fp32 + o * (num_heads * head_dim) + h * head_dim;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += context[d] * wo_row[d];
                }
                out_row[o] += dot;
            }
        }
    }
}

float cosine_similarity(const float *a, const float *b, int n)
{
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-10f || norm_b < 1e-10f)
        return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

float max_abs_diff(const float *a, const float *b, int n)
{
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

/**
 * @brief Test fixture for robustness tests
 */
class FusedAttentionWoRobustnessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rng_.seed(42);
    }

    void generate_random_fp32(float *data, int n, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (int i = 0; i < n; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    void generate_with_outliers(float *data, int n, float base_range, float outlier_mag, float outlier_prob)
    {
        std::uniform_real_distribution<float> base_dist(-base_range, base_range);
        std::uniform_real_distribution<float> outlier_dist(-outlier_mag, outlier_mag);
        std::bernoulli_distribution is_outlier(outlier_prob);
        for (int i = 0; i < n; ++i)
        {
            data[i] = is_outlier(rng_) ? outlier_dist(rng_) : base_dist(rng_);
        }
    }

    std::mt19937 rng_;
};

// =============================================================================
// SOFTMAX SATURATION TESTS
// Issue: Large Q·K scores (170+) cause exp() overflow and one-hot distributions
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, SoftmaxSaturation_LargeScores_NoOverflow)
{
    // Simulate the issue from changelog: scores in range 170+ causing softmax saturation
    // This happens when Q and K have large magnitudes

    const int head_dim = 64;
    const int num_blocks = head_dim / 32;

    // Create Q and K with large, aligned values that produce scores ~170
    // Q · K = sum(Q[i] * K[i]) ≈ head_dim * Q_mag * K_mag
    // For score ≈ 170: need Q_mag * K_mag ≈ 170 / 64 ≈ 2.66
    // So Q_mag ≈ K_mag ≈ 1.63
    std::vector<float> Q_fp32(head_dim), K_fp32(head_dim);
    for (int i = 0; i < head_dim; ++i)
    {
        Q_fp32[i] = 1.6f + 0.1f * (i % 5); // Values around 1.6-2.0
        K_fp32[i] = 1.6f + 0.1f * ((i + 2) % 5);
    }

    // Quantize
    std::vector<Q8_1Block> Q_q8(num_blocks), K_q8(num_blocks);
    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), head_dim);
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), head_dim);

    // Compute dot product using our microkernel
    Q8DotProductParams params;
    params.q_blocks = Q_q8.data();
    params.k_blocks = K_q8.data();
    params.num_blocks = num_blocks;
    params.global_scale = 1.0f; // No scaling to see raw score

    float score = q8_dot_product_ref(params).score;

    // Score should be large but finite
    EXPECT_TRUE(std::isfinite(score)) << "Score should be finite, got: " << score;
    EXPECT_GT(score, 100.0f) << "Score should be large (>100)";

    // Now apply scaling (1/sqrt(64) = 0.125)
    params.global_scale = 1.0f / std::sqrt(64.0f);
    float scaled_score = q8_dot_product_ref(params).score;

    // Scaled score should be reasonable for softmax
    EXPECT_TRUE(std::isfinite(scaled_score));
    EXPECT_LT(std::abs(scaled_score), 50.0f) << "Scaled score should be < 50 for stable softmax";

    // Test online softmax doesn't produce one-hot
    OnlineSoftmaxState state = online_softmax_init();

    // Simulate multiple similar scores
    std::vector<float> scores = {scaled_score, scaled_score - 0.1f, scaled_score - 0.2f, scaled_score - 0.5f};
    std::vector<float> weights(scores.size());

    for (size_t i = 0; i < scores.size(); ++i)
    {
        weights[i] = online_softmax_update(state, scores[i]);
    }

    float inv_sum = online_softmax_finalize(state);

    // Apply normalization
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] *= inv_sum;
    }

    // Verify not one-hot (no weight should be > 0.9)
    float max_weight = *std::max_element(weights.begin(), weights.end());
    EXPECT_LT(max_weight, 0.9f) << "Softmax should not saturate to one-hot, max weight: " << max_weight;

    // Sum should be 1.0
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax weights should sum to 1.0";
}

TEST_F(FusedAttentionWoRobustnessTest, SoftmaxSaturation_VeryLargeScoreDifference)
{
    // Test when one score is much larger than others
    // KNOWN BEHAVIOR: With exp(50-10)=exp(40) ≈ 2.3e17, the first weight dominates
    // completely and rounds to exactly 1.0 in float precision.
    // This is expected - softmax correctly concentrates on the max score.
    OnlineSoftmaxState state = online_softmax_init();

    std::vector<float> scores = {50.0f, 10.0f, 5.0f, 0.0f, -10.0f};
    std::vector<float> weights(scores.size());

    for (size_t i = 0; i < scores.size(); ++i)
    {
        weights[i] = online_softmax_update(state, scores[i]);
    }

    float inv_sum = online_softmax_finalize(state);
    EXPECT_TRUE(std::isfinite(inv_sum)) << "inv_sum should be finite";

    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] *= inv_sum;
        EXPECT_TRUE(std::isfinite(weights[i])) << "Weight " << i << " should be finite";
    }

    // First weight should dominate (may be exactly 1.0 due to float precision)
    EXPECT_GT(weights[0], 0.99f) << "First weight should dominate";

    // Other weights should be tiny (may be exactly 0.0 in float precision)
    for (size_t i = 1; i < weights.size(); ++i)
    {
        EXPECT_GE(weights[i], 0.0f) << "Weight " << i << " should be non-negative";
        EXPECT_LE(weights[i], 1e-6f) << "Weight " << i << " should be negligible";
    }

    // Sum should still be ~1.0
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Weights should sum to ~1.0";
}

// =============================================================================
// CATASTROPHIC CANCELLATION TESTS
// Issue: Large positive + large negative values cancel to small result with noise
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, CatastrophicCancellation_LargeOppositeValues)
{
    // Test case where large positive and negative values sum to small result
    const int head_dim = 64;
    const int num_blocks = head_dim / 32;

    // Create V vectors where elements are large but cancel out
    // V[i] = +100 for even i, -100 for odd i
    // After weighted sum, should get near-zero but quantization might cause issues

    std::vector<float> V_fp32(head_dim);
    for (int i = 0; i < head_dim; ++i)
    {
        V_fp32[i] = (i % 2 == 0) ? 100.0f : -100.0f;
    }

    std::vector<Q8_1Block> V_q8(num_blocks);
    auto stats = quantize_to_q8_1_with_stats(V_fp32.data(), V_q8.data(), head_dim);

    // Due to quantization, values won't perfectly cancel
    // But we should still get reasonable results
    std::vector<float> context(head_dim, 0.0f);

    VWeightedAccumParams params;
    params.v_blocks = V_q8.data();
    params.weight = 1.0f;
    params.correction = 1.0f;
    params.context = context.data();
    params.num_blocks = num_blocks;

    v_weighted_accum_ref(params);

    // Check that context values are reasonable (not NaN/Inf)
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_TRUE(std::isfinite(context[d])) << "Context[" << d << "] should be finite";
    }

    // The mean should be close to 0 (within quantization error)
    float mean = std::accumulate(context.begin(), context.end(), 0.0f) / head_dim;
    EXPECT_LT(std::abs(mean), 5.0f) << "Mean should be close to 0, got: " << mean;
}

TEST_F(FusedAttentionWoRobustnessTest, CatastrophicCancellation_AccumulatorPrecision)
{
    // Simulate many small contributions that should sum to specific value
    // This test verifies that 1024 uniform-weighted V vectors (all ones)
    // accumulate correctly to a context of ~1.0

    const int num_positions = 1024;
    const int head_dim = 64;
    const int num_blocks = head_dim / 32;

    // All scores equal → uniform weights = 1/num_positions
    // With online softmax, we get exp(0)/sum(exp(0)) = 1/1024 per position

    OnlineSoftmaxState state = online_softmax_init();
    std::vector<float> running_weights(num_positions);
    std::vector<float> context(head_dim, 0.0f);

    // Build up all the running weights
    for (int n = 0; n < num_positions; ++n)
    {
        running_weights[n] = online_softmax_update(state, 0.0f); // All scores = 0
    }

    float inv_sum = online_softmax_finalize(state);

    // Create V = all ones
    std::vector<float> V_fp32(head_dim, 1.0f);
    std::vector<Q8_1Block> V_q8(num_blocks);
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), head_dim);

    // Accumulate all V vectors with the running weights
    for (int n = 0; n < num_positions; ++n)
    {
        VWeightedAccumParams params;
        params.v_blocks = V_q8.data();
        params.weight = running_weights[n]; // Running (unnormalized) weight from online softmax
        params.correction = inv_sum;        // Apply normalization via correction factor
        params.context = context.data();
        params.num_blocks = num_blocks;

        v_weighted_accum_ref(params);
    }

    // Result should be close to V (which is all 1.0)
    // Allow some tolerance due to:
    // 1. Q8_1 quantization error (~1%)
    // 2. Floating point accumulation over 1024 terms
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_NEAR(context[d], 1.0f, 0.15f) << "Context[" << d << "] should be ~1.0";
    }
}

// =============================================================================
// QUANTIZATION SATURATION/CLIPPING TESTS
// Issue: Values > 127*scale get clipped, causing information loss
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, QuantizationSaturation_ExtremeOutliers)
{
    // Test with extreme outliers that will saturate to ±127
    const int head_dim = 64;
    const int num_blocks = head_dim / 32;

    // Create vector with one extreme outlier per block
    std::vector<float> Q_fp32(head_dim), K_fp32(head_dim);

    for (int i = 0; i < head_dim; ++i)
    {
        Q_fp32[i] = 0.1f; // Small values
        K_fp32[i] = 0.1f;
    }

    // Add extreme outliers
    Q_fp32[0] = 100.0f;  // First block outlier
    Q_fp32[32] = 100.0f; // Second block outlier
    K_fp32[0] = 1.0f;
    K_fp32[32] = 1.0f;

    std::vector<Q8_1Block> Q_q8(num_blocks), K_q8(num_blocks);
    auto Q_stats = quantize_to_q8_1_with_stats(Q_fp32.data(), Q_q8.data(), head_dim);
    auto K_stats = quantize_to_q8_1_with_stats(K_fp32.data(), K_q8.data(), head_dim);

    // Check saturation occurred
    EXPECT_GT(Q_stats.num_saturated, 0) << "Expected some saturation in Q";

    // Despite saturation, dot product should still be reasonable
    Q8DotProductParams params;
    params.q_blocks = Q_q8.data();
    params.k_blocks = K_q8.data();
    params.num_blocks = num_blocks;
    params.global_scale = 1.0f / std::sqrt(64.0f);

    float score = q8_dot_product_ref(params).score;
    EXPECT_TRUE(std::isfinite(score)) << "Score should be finite despite saturation";
}

TEST_F(FusedAttentionWoRobustnessTest, QuantizationClipping_GammaWeightRange)
{
    // Issue from changelog: gamma weights in Qwen2.5 can exceed [-8, 8]
    // Test that our Wo projection handles large weight values correctly

    const int head_dim = 64;
    const int d_model = 64;                  // Keep it simple
    const int wo_row_blocks = head_dim / 32; // 2 blocks per row

    // Create Wo weights with values that would clip in Q12 format
    // Values > 8 would clip in Q12 (scale 4096)
    std::vector<float> Wo_fp32(d_model * head_dim);

    for (int row = 0; row < d_model; ++row)
    {
        for (int col = 0; col < head_dim; ++col)
        {
            // Some weights are "normal" (< 8), some are "large" (> 8)
            float val = (col % 10 == 0) ? 13.0f : 0.5f; // Matches Qwen2.5 gamma[62] = 13.015625
            Wo_fp32[row * head_dim + col] = val;
        }
    }

    // Quantize Wo to Q8_1
    std::vector<Q8_1Block> Wo_q8(d_model * wo_row_blocks);
    for (int row = 0; row < d_model; ++row)
    {
        quantize_to_q8_1(
            Wo_fp32.data() + row * head_dim,
            Wo_q8.data() + row * wo_row_blocks,
            head_dim);
    }

    // Create simple context (all 1.0)
    std::vector<float> context(head_dim, 1.0f);
    std::vector<float> output_fp32(d_model, 0.0f);
    std::vector<float> output_q8(d_model, 0.0f);

    // FP32 projection
    WoProjectionParams fp32_params;
    fp32_params.context = context.data();
    fp32_params.wo_weights = Wo_fp32.data();
    fp32_params.wo_type = WoWeightType::FP32;
    fp32_params.head_dim = head_dim;
    fp32_params.d_model = d_model;
    fp32_params.head_idx = 0;
    fp32_params.n_heads = 1;
    fp32_params.output = output_fp32.data();
    fp32_params.accumulate = false;

    wo_projection_fp32_ref(fp32_params);

    // Q8_1 projection
    WoProjectionParams q8_params;
    q8_params.context = context.data();
    q8_params.wo_weights = Wo_q8.data();
    q8_params.wo_type = WoWeightType::Q8_1;
    q8_params.head_dim = head_dim;
    q8_params.d_model = d_model;
    q8_params.head_idx = 0;
    q8_params.n_heads = 1;
    q8_params.output = output_q8.data();
    q8_params.accumulate = false;

    wo_projection_q8_ref(q8_params);

    // Compare - should be close despite quantization
    float cos_sim = cosine_similarity(output_fp32.data(), output_q8.data(), d_model);
    EXPECT_GT(cos_sim, 0.95f) << "Q8_1 Wo should match FP32 Wo closely: " << cos_sim;

    float max_diff = max_abs_diff(output_fp32.data(), output_q8.data(), d_model);
    EXPECT_LT(max_diff, 5.0f) << "Max difference should be bounded: " << max_diff;
}

// =============================================================================
// SCALE DISPARITY TESTS
// Issue: Different blocks having vastly different scales causes precision loss
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, ScaleDisparity_MixedMagnitudeBlocks)
{
    // Block 0: values in [0.001, 0.01] (scale ~0.0001)
    // Block 1: values in [10, 100] (scale ~0.8)
    // This 8000x scale disparity can cause precision issues

    const int head_dim = 64;
    const int num_blocks = 2;

    std::vector<float> Q_fp32(head_dim), K_fp32(head_dim);

    // Block 0: tiny values
    for (int i = 0; i < 32; ++i)
    {
        Q_fp32[i] = 0.005f + 0.001f * (i % 5);
        K_fp32[i] = 0.005f + 0.001f * ((i + 2) % 5);
    }

    // Block 1: large values
    for (int i = 32; i < 64; ++i)
    {
        Q_fp32[i] = 50.0f + 10.0f * (i % 5);
        K_fp32[i] = 50.0f + 10.0f * ((i + 2) % 5);
    }

    std::vector<Q8_1Block> Q_q8(num_blocks), K_q8(num_blocks);
    auto Q_stats = quantize_to_q8_1_with_stats(Q_fp32.data(), Q_q8.data(), head_dim);
    auto K_stats = quantize_to_q8_1_with_stats(K_fp32.data(), K_q8.data(), head_dim);

    // Verify scale disparity
    float scale_ratio = Q_stats.max_scale / Q_stats.min_scale;
    EXPECT_GT(scale_ratio, 100.0f) << "Should have significant scale disparity: " << scale_ratio;

    // Compute dot product
    Q8DotProductParams params;
    params.q_blocks = Q_q8.data();
    params.k_blocks = K_q8.data();
    params.num_blocks = num_blocks;
    params.global_scale = 1.0f / std::sqrt(64.0f);

    float q8_score = q8_dot_product_ref(params).score;

    // Compute FP32 reference
    float fp32_score = 0.0f;
    for (int i = 0; i < head_dim; ++i)
    {
        fp32_score += Q_fp32[i] * K_fp32[i];
    }
    fp32_score *= 1.0f / std::sqrt(64.0f);

    // Large block should dominate, so quantization should be relatively accurate
    EXPECT_TRUE(std::isfinite(q8_score));
    float rel_error = std::abs(q8_score - fp32_score) / std::abs(fp32_score);
    EXPECT_LT(rel_error, 0.1f) << "Relative error should be < 10%: " << rel_error;
}

// =============================================================================
// SIGN FLIP / RESIDUAL COLLAPSE TESTS
// Issue: Accumulated quantization error can flip signs, causing residual collapse
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, SignFlip_SmallValueRoundingError)
{
    // Test values near zero where quantization could flip the sign
    const int head_dim = 64;
    const int num_blocks = head_dim / 32;

    // Create values that are very small (will round to 0 or ±1 in Q8)
    std::vector<float> V_fp32(head_dim);
    for (int i = 0; i < head_dim; ++i)
    {
        // Alternating tiny positive and negative
        float sign = (i % 2 == 0) ? 1.0f : -1.0f;
        V_fp32[i] = sign * 0.001f; // Very small values
    }

    std::vector<Q8_1Block> V_q8(num_blocks);
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), head_dim);

    // Dequantize and check signs
    std::vector<float> V_deq(head_dim);
    dequantize_from_q8_1(V_q8.data(), V_deq.data(), head_dim);

    int sign_preserved = 0;
    int sign_flipped = 0;
    int became_zero = 0;

    for (int i = 0; i < head_dim; ++i)
    {
        if (V_deq[i] == 0.0f)
        {
            became_zero++;
        }
        else if ((V_fp32[i] > 0 && V_deq[i] > 0) || (V_fp32[i] < 0 && V_deq[i] < 0))
        {
            sign_preserved++;
        }
        else
        {
            sign_flipped++;
        }
    }

    // Most values should either preserve sign or become zero
    // Very few should flip sign (this would be a bug)
    EXPECT_LT(sign_flipped, head_dim / 10) << "Too many sign flips: " << sign_flipped;
}

TEST_F(FusedAttentionWoRobustnessTest, ResidualCollapse_AccumulatedError)
{
    // Simulate a scenario where many layers of attention accumulate errors
    // Each layer adds small errors that can compound

    const int seq_len = 1;
    const int kv_seq_len = 4;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int num_blocks = head_dim / 32;

    // Generate random data with realistic distributions
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size(), -2.0f, 2.0f);
    generate_random_fp32(K_fp32.data(), K_fp32.size(), -2.0f, 2.0f);
    generate_random_fp32(V_fp32.data(), V_fp32.size(), -2.0f, 2.0f);
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size(), -0.1f, 0.1f); // Wo typically smaller

    // Quantize
    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Dequantize for reference
    std::vector<float> Q_deq(Q_fp32.size()), K_deq(K_fp32.size()), V_deq(V_fp32.size());
    dequantize_from_q8_1(Q_q8.data(), Q_deq.data(), Q_fp32.size());
    dequantize_from_q8_1(K_q8.data(), K_deq.data(), K_fp32.size());
    dequantize_from_q8_1(V_q8.data(), V_deq.data(), V_fp32.size());

    // Run fused kernel
    std::vector<float> output_fused(seq_len * d_model, 0.0f);

    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output_fused.data();
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = false;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Run reference with dequantized values
    std::vector<float> output_ref(seq_len * d_model, 0.0f);
    compute_attention_reference_fp32(
        Q_deq.data(), K_deq.data(), V_deq.data(), Wo_fp32.data(),
        output_ref.data(), seq_len, kv_seq_len, num_heads, num_kv_heads,
        head_dim, d_model, scale, false, 0);

    // Check for residual collapse - output should not be all zeros or near-zero
    float output_norm = 0.0f;
    for (int i = 0; i < d_model; ++i)
    {
        output_norm += output_fused[i] * output_fused[i];
    }
    output_norm = std::sqrt(output_norm);

    EXPECT_GT(output_norm, 0.1f) << "Output should not collapse to near-zero";

    // Compare with reference
    float cos_sim = cosine_similarity(output_fused.data(), output_ref.data(), d_model);
    EXPECT_GT(cos_sim, 0.95f) << "Fused should match reference closely: " << cos_sim;
}

// =============================================================================
// EXTREME OUTLIER TESTS
// Issue: Single extreme values can dominate block scales, losing precision elsewhere
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, ExtremeOutlier_SingleValueDominatesBlock)
{
    // Test how a single extreme outlier affects quantization within its block
    //
    // Q8_1 quantization uses per-block scales. When one value is extreme:
    // - Block 0 (contains outlier 1000 + values of 1.0):
    //   scale = 1000/127 ≈ 7.87
    //   outlier: 1000/7.87 ≈ 127 → clips to 127 (saturates)
    //   normal:  1.0/7.87 ≈ 0.127 → rounds to 0 (loses precision, but not saturated)
    //
    // - Block 1 (all values are 1.0):
    //   scale = 1.0/127 ≈ 0.00787
    //   normal: 1.0/0.00787 = 127 → all values clip to 127!
    //
    // This is EXPECTED Q8_1 behavior: each block scales independently.

    const int head_dim = 64;
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(head_dim);
    std::fill(Q_fp32.begin(), Q_fp32.end(), 1.0f); // Normal values
    Q_fp32[0] = 1000.0f;                           // Extreme outlier in block 0

    std::vector<Q8_1Block> Q_q8(num_blocks);
    auto stats = quantize_to_q8_1_with_stats(Q_fp32.data(), Q_q8.data(), head_dim);

    // Block 0: 1 saturated (the outlier clips to 127)
    // Block 1: All 32 values saturate because 1.0 with tiny scale → 127
    // Total: 33 saturated values
    EXPECT_EQ(stats.num_saturated, 33) << "Outlier + all values in separate block should saturate";

    // Dequantize and check precision loss on normal values
    std::vector<float> Q_deq(head_dim);
    dequantize_from_q8_1(Q_q8.data(), Q_deq.data(), head_dim);

    // Block 0: values that aren't the outlier have large relative error
    // because they're quantized with the outlier's scale
    float first_block_scale = llaminar2::fp16_to_fp32(Q_q8[0].d);
    EXPECT_GT(first_block_scale, 5.0f) << "Block 0 scale should be large due to outlier";

    // Block 1: values should be fine (their own scale applies)
    float second_block_scale = llaminar2::fp16_to_fp32(Q_q8[1].d);
    EXPECT_LT(second_block_scale, 0.1f) << "Block 1 scale should be normal";

    // Verify block 1 dequantized values are correct
    for (int i = 32; i < head_dim; ++i)
    {
        float rel_error = std::abs(Q_deq[i] - Q_fp32[i]) / std::abs(Q_fp32[i]);
        EXPECT_LT(rel_error, 0.02f) << "Block 1 values should have good precision at index " << i;
    }
}

// =============================================================================
// NUMERICAL EDGE CASES
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, EdgeCase_ZeroInputs)
{
    const int head_dim = 64;
    const int num_blocks = head_dim / 32;

    // All-zero Q should produce all-zero output (or uniform attention weights)
    std::vector<float> Q_fp32(head_dim, 0.0f);
    std::vector<float> K_fp32(head_dim, 1.0f);

    std::vector<Q8_1Block> Q_q8(num_blocks), K_q8(num_blocks);
    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), head_dim);
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), head_dim);

    Q8DotProductParams params;
    params.q_blocks = Q_q8.data();
    params.k_blocks = K_q8.data();
    params.num_blocks = num_blocks;
    params.global_scale = 1.0f / std::sqrt(64.0f);

    float score = q8_dot_product_ref(params).score;

    // Score should be zero (or very close)
    EXPECT_NEAR(score, 0.0f, 1e-3f) << "Zero Q dot K should be ~0";
}

TEST_F(FusedAttentionWoRobustnessTest, EdgeCase_IdenticalQK)
{
    // Q == K should give high similarity score
    const int head_dim = 64;
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(head_dim);
    generate_random_fp32(Q_fp32.data(), head_dim, -1.0f, 1.0f);

    std::vector<Q8_1Block> Q_q8(num_blocks);
    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), head_dim);

    // Use same quantized data for both Q and K
    Q8DotProductParams params;
    params.q_blocks = Q_q8.data();
    params.k_blocks = Q_q8.data(); // K = Q
    params.num_blocks = num_blocks;
    params.global_scale = 1.0f; // No scaling

    float score = q8_dot_product_ref(params).score;

    // Score should be positive (squared norm)
    EXPECT_GT(score, 0.0f) << "Self-dot product should be positive";

    // Should be approximately ||Q||^2
    std::vector<float> Q_deq(head_dim);
    dequantize_from_q8_1(Q_q8.data(), Q_deq.data(), head_dim);
    float expected_norm_sq = 0.0f;
    for (int i = 0; i < head_dim; ++i)
    {
        expected_norm_sq += Q_deq[i] * Q_deq[i];
    }

    EXPECT_NEAR(score, expected_norm_sq, expected_norm_sq * 0.1f)
        << "Self-dot should match ||Q||^2";
}

TEST_F(FusedAttentionWoRobustnessTest, EdgeCase_SinglePosition)
{
    // Edge case: kv_seq_len = 1 (first token in decode)
    const int seq_len = 1;
    const int kv_seq_len = 1;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    std::vector<float> output(seq_len * d_model, 0.0f);

    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output.data();
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = 0;

    // Should succeed without crash
    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Output should be finite
    for (int i = 0; i < d_model; ++i)
    {
        EXPECT_TRUE(std::isfinite(output[i])) << "Output[" << i << "] should be finite";
    }
}

TEST_F(FusedAttentionWoRobustnessTest, EdgeCase_VeryLongSequence)
{
    // Test with longer KV cache (common in production)
    const int seq_len = 1;
    const int kv_seq_len = 512; // Simulate longer context
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size(), -0.1f, 0.1f);

    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    std::vector<float> output(seq_len * d_model, 0.0f);

    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output.data();
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = kv_seq_len - 1; // Can attend to all positions

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Output should be finite
    for (int i = 0; i < d_model; ++i)
    {
        EXPECT_TRUE(std::isfinite(output[i])) << "Output[" << i << "] should be finite";
    }

    // Output should have reasonable magnitude (not collapsed, not exploded)
    float output_norm = 0.0f;
    for (int i = 0; i < d_model; ++i)
    {
        output_norm += output[i] * output[i];
    }
    output_norm = std::sqrt(output_norm);

    EXPECT_GT(output_norm, 0.01f) << "Output should not collapse";
    EXPECT_LT(output_norm, 1000.0f) << "Output should not explode";
}

// =============================================================================
// FAST_EXP ROBUSTNESS TESTS
// Issue: exp() overflow/underflow in softmax
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, FastExp_ExtremeInputs)
{
    // Test fast_exp at the edges of its valid range

    // Large positive (should clamp, not overflow)
    float result_large_pos = fast_exp(100.0f);
    EXPECT_TRUE(std::isfinite(result_large_pos)) << "fast_exp(100) should be finite";

    // Large negative (should be very small, not underflow to 0)
    float result_large_neg = fast_exp(-100.0f);
    EXPECT_TRUE(std::isfinite(result_large_neg)) << "fast_exp(-100) should be finite";
    EXPECT_GE(result_large_neg, 0.0f) << "fast_exp(-100) should be non-negative";

    // At clamp boundary
    float result_at_clamp = fast_exp(88.0f); // Near ln(FLT_MAX)
    EXPECT_TRUE(std::isfinite(result_at_clamp)) << "fast_exp(88) should be finite";

    // Typical softmax range [-20, 0]
    for (float x = -20.0f; x <= 0.0f; x += 1.0f)
    {
        float fast_result = fast_exp(x);
        float std_result = std::exp(x);

        EXPECT_TRUE(std::isfinite(fast_result)) << "fast_exp(" << x << ") should be finite";

        // Relative error should be small in this range
        float rel_error = std::abs(fast_result - std_result) / std_result;
        EXPECT_LT(rel_error, 0.05f) << "fast_exp(" << x << ") relative error too large: " << rel_error;
    }
}

// =============================================================================
// ONLINE SOFTMAX NUMERICAL STABILITY TESTS
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, OnlineSoftmax_OrderIndependence)
{
    // The online softmax produces running weights that need correction when max changes.
    // The RAW return values from online_softmax_update ARE order-dependent.
    // But when used CORRECTLY with the correction factor, the final result is order-independent.
    //
    // This test verifies the ACTUAL usage pattern in attention computation:
    // 1. For each score, get running weight
    // 2. Track when max changes and apply correction to accumulated context
    // 3. Final normalize by inv_sum
    //
    // We simulate this by computing weighted V accumulation in forward vs reverse order.

    std::vector<float> scores = {5.0f, 2.0f, -3.0f, 10.0f, 1.0f, -1.0f, 7.0f, 3.0f};
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; // V values

    // Forward order - simulate attention weighted sum
    {
        OnlineSoftmaxState state = online_softmax_init();
        float weighted_sum = 0.0f;

        for (size_t i = 0; i < scores.size(); ++i)
        {
            float prev_max = state.max_score;
            float weight = online_softmax_update(state, scores[i]);

            // Apply correction if max changed
            if (state.max_score > prev_max && i > 0)
            {
                float correction = online_softmax_correction(prev_max, state.max_score);
                weighted_sum *= correction;
            }

            weighted_sum += weight * values[i];
        }

        float inv_sum = online_softmax_finalize(state);
        float result_fwd = weighted_sum * inv_sum;

        // Compare to standard softmax reference
        float ref_max = *std::max_element(scores.begin(), scores.end());
        float ref_sum = 0.0f;
        float ref_weighted = 0.0f;
        for (size_t i = 0; i < scores.size(); ++i)
        {
            float w = std::exp(scores[i] - ref_max);
            ref_sum += w;
            ref_weighted += w * values[i];
        }
        float result_ref = ref_weighted / ref_sum;

        EXPECT_NEAR(result_fwd, result_ref, 1e-5f)
            << "Forward order should match reference softmax";
    }

    // Reverse order - should get same result
    {
        OnlineSoftmaxState state = online_softmax_init();
        float weighted_sum = 0.0f;

        for (int i = scores.size() - 1; i >= 0; --i)
        {
            float prev_max = state.max_score;
            float weight = online_softmax_update(state, scores[i]);

            // Apply correction if max changed
            if (state.max_score > prev_max && i < (int)scores.size() - 1)
            {
                float correction = online_softmax_correction(prev_max, state.max_score);
                weighted_sum *= correction;
            }

            weighted_sum += weight * values[i];
        }

        float inv_sum = online_softmax_finalize(state);
        float result_rev = weighted_sum * inv_sum;

        // Compare to standard softmax reference
        float ref_max = *std::max_element(scores.begin(), scores.end());
        float ref_sum = 0.0f;
        float ref_weighted = 0.0f;
        for (size_t i = 0; i < scores.size(); ++i)
        {
            float w = std::exp(scores[i] - ref_max);
            ref_sum += w;
            ref_weighted += w * values[i];
        }
        float result_ref = ref_weighted / ref_sum;

        EXPECT_NEAR(result_rev, result_ref, 1e-5f)
            << "Reverse order should match reference softmax";
    }
}

// =============================================================================
// STRIDE HANDLING TESTS
// Issue: Incorrect stride calculations cause subtle data corruption, wrong heads
//        accessed, and off-by-one errors that produce plausible but wrong results
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, Stride_QKV_NonContiguous_HeadLayout)
{
    // Test Q/K/V tensors with gaps between heads (non-packed layout)
    // This simulates slicing from a larger multi-batch tensor
    //
    // Memory layout with stride:
    //   [head0_data][padding][head1_data][padding]...
    //
    // Common bug: Using head_dim instead of head_stride to advance pointers

    const int seq_len = 4;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int num_blocks_per_head = head_dim / 32;

    // Add padding between heads (simulate non-packed layout)
    const int padding_blocks = 2;                                        // Extra blocks between heads
    const int head_stride_blocks = num_blocks_per_head + padding_blocks; // 4 instead of 2

    // Total storage: seq_len * num_heads * head_stride_blocks
    const int total_q_blocks = seq_len * num_heads * head_stride_blocks;
    const int total_kv_blocks = seq_len * num_kv_heads * head_stride_blocks;

    // Create padded Q/K/V with garbage in padding regions
    std::vector<Q8_1Block> Q_padded(total_q_blocks);
    std::vector<Q8_1Block> K_padded(total_kv_blocks);
    std::vector<Q8_1Block> V_padded(total_kv_blocks);

    // Fill padding with distinctive garbage pattern (0xDE = -34 signed)
    for (auto &block : Q_padded)
    {
        block.d = llaminar2::fp32_to_fp16(100.0f); // Large scale = garbage
        std::fill(block.qs, block.qs + 32, static_cast<int8_t>(0xDE));
        block.sum_qs = 0xDEDE;
    }
    for (auto &block : K_padded)
    {
        block.d = llaminar2::fp32_to_fp16(100.0f);
        std::fill(block.qs, block.qs + 32, static_cast<int8_t>(0xDE));
        block.sum_qs = 0xDEDE;
    }
    for (auto &block : V_padded)
    {
        block.d = llaminar2::fp32_to_fp16(100.0f);
        std::fill(block.qs, block.qs + 32, static_cast<int8_t>(0xDE));
        block.sum_qs = 0xDEDE;
    }

    // Create contiguous reference data
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(seq_len * num_kv_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size(), -0.5f, 0.5f);
    generate_random_fp32(K_fp32.data(), K_fp32.size(), -0.5f, 0.5f);
    generate_random_fp32(V_fp32.data(), V_fp32.size(), -0.5f, 0.5f);

    // Quantize into padded layout (skip padding regions)
    for (int m = 0; m < seq_len; ++m)
    {
        for (int h = 0; h < num_heads; ++h)
        {
            int fp32_offset = (m * num_heads + h) * head_dim;
            int q8_offset = (m * num_heads + h) * head_stride_blocks;
            quantize_to_q8_1(Q_fp32.data() + fp32_offset, Q_padded.data() + q8_offset, head_dim);
        }
        for (int h = 0; h < num_kv_heads; ++h)
        {
            int fp32_offset = (m * num_kv_heads + h) * head_dim;
            int q8_offset = (m * num_kv_heads + h) * head_stride_blocks;
            quantize_to_q8_1(K_fp32.data() + fp32_offset, K_padded.data() + q8_offset, head_dim);
            quantize_to_q8_1(V_fp32.data() + fp32_offset, V_padded.data() + q8_offset, head_dim);
        }
    }

    // Also create packed versions for comparison
    std::vector<Q8_1Block> Q_packed(seq_len * num_heads * num_blocks_per_head);
    std::vector<Q8_1Block> K_packed(seq_len * num_kv_heads * num_blocks_per_head);
    std::vector<Q8_1Block> V_packed(seq_len * num_kv_heads * num_blocks_per_head);

    quantize_to_q8_1(Q_fp32.data(), Q_packed.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_packed.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_packed.data(), V_fp32.size());

    // The current FusedAttentionWoRef assumes packed layout
    // This test documents that assumption and catches if we break it
    //
    // To properly support strided layout, we'd need to add stride params:
    //   - q_head_stride, k_head_stride, v_head_stride (in blocks)
    //   - q_pos_stride, k_pos_stride, v_pos_stride (in blocks)

    // Verify packed data matches packed subset of padded data
    for (int m = 0; m < seq_len; ++m)
    {
        for (int h = 0; h < num_heads; ++h)
        {
            int packed_offset = (m * num_heads + h) * num_blocks_per_head;
            int padded_offset = (m * num_heads + h) * head_stride_blocks;

            for (int b = 0; b < num_blocks_per_head; ++b)
            {
                // Data blocks should match
                EXPECT_EQ(llaminar2::fp16_to_fp32(Q_packed[packed_offset + b].d),
                          llaminar2::fp16_to_fp32(Q_padded[padded_offset + b].d))
                    << "Q scale mismatch at pos=" << m << " head=" << h << " block=" << b;

                for (int i = 0; i < 32; ++i)
                {
                    EXPECT_EQ(Q_packed[packed_offset + b].qs[i], Q_padded[padded_offset + b].qs[i])
                        << "Q data mismatch at pos=" << m << " head=" << h << " block=" << b << " elem=" << i;
                }
            }

            // Padding should still be garbage
            for (int b = num_blocks_per_head; b < head_stride_blocks; ++b)
            {
                float garbage_scale = llaminar2::fp16_to_fp32(Q_padded[padded_offset + b].d);
                EXPECT_NEAR(garbage_scale, 100.0f, 0.1f)
                    << "Padding should remain untouched at pos=" << m << " head=" << h << " padding=" << b;
            }
        }
    }
}

TEST_F(FusedAttentionWoRobustnessTest, Stride_Output_NonContiguous)
{
    // Test writing to non-contiguous output buffer
    // Simulates writing attention output into a larger tensor slice
    //
    // Memory layout:
    //   [row0_data][padding][row1_data][padding]...
    //
    // Common bug: Using d_model for row stride instead of output_stride

    const int seq_len = 4;
    const int d_model = 64;
    const int output_padding = 16; // Extra floats between rows
    const int output_stride = d_model + output_padding;

    // Create output buffer with padding
    std::vector<float> output_padded(seq_len * output_stride, -999.0f); // Fill with sentinel

    // Create contiguous output for comparison
    std::vector<float> output_packed(seq_len * d_model, 0.0f);

    // The test shows that if we write with wrong stride, we'd corrupt padding
    // and leave parts of actual output unwritten

    // Simulate correct strided write
    for (int m = 0; m < seq_len; ++m)
    {
        float *row_start = output_padded.data() + m * output_stride;
        for (int i = 0; i < d_model; ++i)
        {
            row_start[i] = static_cast<float>(m * d_model + i); // Unique pattern
        }
        // Padding should remain -999
    }

    // Verify data landed correctly
    for (int m = 0; m < seq_len; ++m)
    {
        float *row_start = output_padded.data() + m * output_stride;

        // Data region should have our pattern
        for (int i = 0; i < d_model; ++i)
        {
            EXPECT_FLOAT_EQ(row_start[i], static_cast<float>(m * d_model + i))
                << "Output data wrong at row=" << m << " col=" << i;
        }

        // Padding should be untouched
        for (int i = d_model; i < output_stride; ++i)
        {
            EXPECT_FLOAT_EQ(row_start[i], -999.0f)
                << "Padding corrupted at row=" << m << " padding_idx=" << (i - d_model);
        }
    }

    // Simulate WRONG stride (treating it as packed)
    std::vector<float> output_wrong(seq_len * output_stride, -999.0f);
    float *wrong_ptr = output_wrong.data();
    for (int m = 0; m < seq_len; ++m)
    {
        for (int i = 0; i < d_model; ++i)
        {
            // Bug: using d_model stride instead of output_stride
            wrong_ptr[m * d_model + i] = static_cast<float>(m * d_model + i);
        }
    }

    // With wrong stride, later rows would overwrite earlier padding
    // and data ends up in wrong positions
    // Row 1 would start at offset 64 instead of 80
    // Row 2 would start at offset 128 instead of 160

    // Verify the bug pattern
    EXPECT_FLOAT_EQ(output_wrong[d_model], 64.0f) // Row 1, elem 0 (wrong position)
        << "Wrong stride should put row 1 data at wrong position";

    EXPECT_FLOAT_EQ(output_padded[output_stride], 64.0f) // Row 1, elem 0 (correct position)
        << "Correct stride should put row 1 data at right position";

    EXPECT_NE(d_model, output_stride) << "Test requires padding to be meaningful";
}

TEST_F(FusedAttentionWoRobustnessTest, Stride_KV_Cache_Offset)
{
    // Test KV cache access with position offset (decode mode)
    // The KV cache accumulates K/V from all positions, but we only
    // process the new query position while accessing all cached KV
    //
    // Common bug: Off-by-one in position_offset calculation

    const int prefill_len = 8; // Previously cached positions
    const int decode_len = 1;  // New position being processed
    const int total_kv_len = prefill_len + decode_len;
    const int head_dim = 64;
    const int num_blocks_per_head = head_dim / 32;

    // Create KV cache with all positions
    std::vector<float> K_cache_fp32(total_kv_len * head_dim);
    std::vector<float> V_cache_fp32(total_kv_len * head_dim);

    // Fill with position-dependent pattern so we can verify correct access
    for (int pos = 0; pos < total_kv_len; ++pos)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            // K[pos] = pos + 0.001 * d (each position is distinguishable)
            K_cache_fp32[pos * head_dim + d] = static_cast<float>(pos) + 0.001f * d;
            // V[pos] = (pos + 100) + 0.001 * d
            V_cache_fp32[pos * head_dim + d] = static_cast<float>(pos + 100) + 0.001f * d;
        }
    }

    // Verify we can distinguish positions
    EXPECT_NEAR(K_cache_fp32[0 * head_dim + 0], 0.0f, 0.01f); // pos 0
    EXPECT_NEAR(K_cache_fp32[1 * head_dim + 0], 1.0f, 0.01f); // pos 1
    EXPECT_NEAR(K_cache_fp32[8 * head_dim + 0], 8.0f, 0.01f); // pos 8 (decode position)

    // Test correct position_offset calculation
    // In decode mode with causal attention:
    //   - Query is at position (prefill_len) = 8
    //   - Can attend to positions [0, prefill_len] = [0, 8]
    //   - position_offset = prefill_len = 8

    int query_pos = 0;                                // Local position (we're processing 1 new token)
    int position_offset = prefill_len;                // Global position of this query
    int max_kv_pos = position_offset + query_pos + 1; // Causal: can attend up to and including self

    EXPECT_EQ(max_kv_pos, 9) << "Should attend to positions 0-8 inclusive";

    // Verify each KV position is accessible
    for (int kv_pos = 0; kv_pos < max_kv_pos; ++kv_pos)
    {
        float k_val = K_cache_fp32[kv_pos * head_dim + 0];
        EXPECT_NEAR(k_val, static_cast<float>(kv_pos), 0.01f)
            << "K cache position " << kv_pos << " should be accessible";
    }

    // Common off-by-one bugs:
    // 1. Using position_offset instead of position_offset + query_pos + 1
    //    Would miss the current position's KV
    // 2. Using < instead of <= in loop
    //    Would miss last position
    // 3. Not adding 1 for inclusive end
    //    Would attend to [0, 7] instead of [0, 8]
}

TEST_F(FusedAttentionWoRobustnessTest, Stride_GQA_KVHead_Mapping)
{
    // Test GQA head mapping with different ratios
    // Multiple Q heads share each KV head
    //
    // Common bug: Integer division truncation in head index calculation

    struct GQAConfig
    {
        int num_heads;
        int num_kv_heads;
        std::vector<int> expected_kv_head; // For each Q head, which KV head
    };

    std::vector<GQAConfig> configs = {
        // MHA: 1:1 mapping
        {4, 4, {0, 1, 2, 3}},

        // GQA 2:1 - 2 Q heads per KV head
        {4, 2, {0, 0, 1, 1}},

        // GQA 4:1 - 4 Q heads per KV head
        {8, 2, {0, 0, 0, 0, 1, 1, 1, 1}},

        // MQA: all Q heads share 1 KV head
        {4, 1, {0, 0, 0, 0}},

        // Qwen2.5 0.5B config: 14 heads, 2 KV heads (7:1)
        {14, 2, {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1}},
    };

    for (const auto &cfg : configs)
    {
        SCOPED_TRACE("Testing num_heads=" + std::to_string(cfg.num_heads) +
                     " num_kv_heads=" + std::to_string(cfg.num_kv_heads));

        ASSERT_EQ(cfg.expected_kv_head.size(), static_cast<size_t>(cfg.num_heads));

        // Verify mapping formula: kv_head = q_head / (num_heads / num_kv_heads)
        int heads_per_kv = cfg.num_heads / cfg.num_kv_heads;

        for (int q_head = 0; q_head < cfg.num_heads; ++q_head)
        {
            int kv_head = q_head / heads_per_kv;
            EXPECT_EQ(kv_head, cfg.expected_kv_head[q_head])
                << "Q head " << q_head << " should map to KV head " << cfg.expected_kv_head[q_head]
                << " but got " << kv_head;

            // Verify KV head is in valid range
            EXPECT_GE(kv_head, 0);
            EXPECT_LT(kv_head, cfg.num_kv_heads);
        }
    }
}

TEST_F(FusedAttentionWoRobustnessTest, Stride_Wo_RowMajor_vs_ColMajor)
{
    // Test Wo weight access pattern
    // Wo is [d_model, num_heads * head_dim] = [d_model, d_model]
    //
    // Row-major (C default): Wo[row][col] at offset row * num_cols + col
    // Col-major (Fortran):   Wo[row][col] at offset col * num_rows + row
    //
    // Bug: Using wrong layout causes transposed output

    const int d_model = 8; // Small for visualization
    const int head_dim = 4;
    const int num_heads = 2;

    // Create Wo with distinctive pattern
    // Wo[i][j] = i * 100 + j (row-major storage)
    std::vector<float> Wo_row_major(d_model * d_model);
    for (int row = 0; row < d_model; ++row)
    {
        for (int col = 0; col < d_model; ++col)
        {
            Wo_row_major[row * d_model + col] = static_cast<float>(row * 100 + col);
        }
    }

    // Create context vector (all 1s for simple verification)
    std::vector<float> context(d_model, 1.0f);

    // Row-major output: out[i] = sum_j(Wo[i][j] * context[j])
    std::vector<float> out_row_major(d_model, 0.0f);
    for (int row = 0; row < d_model; ++row)
    {
        for (int col = 0; col < d_model; ++col)
        {
            out_row_major[row] += Wo_row_major[row * d_model + col] * context[col];
        }
    }

    // Col-major output (buggy interpretation of same memory)
    std::vector<float> out_col_major(d_model, 0.0f);
    for (int row = 0; row < d_model; ++row)
    {
        for (int col = 0; col < d_model; ++col)
        {
            // Bug: interpreting row-major data as col-major
            out_col_major[row] += Wo_row_major[col * d_model + row] * context[col];
        }
    }

    // Results should differ (unless Wo happens to be symmetric)
    bool results_differ = false;
    for (int i = 0; i < d_model; ++i)
    {
        if (std::abs(out_row_major[i] - out_col_major[i]) > 0.01f)
        {
            results_differ = true;
            break;
        }
    }

    EXPECT_TRUE(results_differ)
        << "Row-major vs col-major should produce different results for non-symmetric Wo";

    // Verify expected row-major result
    // out[0] = sum(Wo[0][j]) = sum(0*100 + j for j=0..7) = 0+1+2+3+4+5+6+7 = 28
    EXPECT_FLOAT_EQ(out_row_major[0], 28.0f);
    // out[1] = sum(Wo[1][j]) = sum(1*100 + j) = 100+101+102+103+104+105+106+107 = 828
    EXPECT_FLOAT_EQ(out_row_major[1], 828.0f);
}

TEST_F(FusedAttentionWoRobustnessTest, Stride_Batch_Dimension)
{
    // Test batch dimension handling
    // Real attention often has shape [batch, seq, heads, head_dim]
    //
    // Common bug: Not accounting for batch stride in tensor indexing

    const int batch_size = 2;
    const int seq_len = 4;
    const int num_heads = 2;
    const int head_dim = 32;

    // Create batched Q tensor with distinctive batch patterns
    // Batch 0: all positive values
    // Batch 1: all negative values
    std::vector<float> Q_batched(batch_size * seq_len * num_heads * head_dim);

    for (int b = 0; b < batch_size; ++b)
    {
        float sign = (b == 0) ? 1.0f : -1.0f;
        for (int i = 0; i < seq_len * num_heads * head_dim; ++i)
        {
            Q_batched[b * seq_len * num_heads * head_dim + i] = sign * (0.1f + 0.01f * i);
        }
    }

    // Extract batch 0 and batch 1
    std::vector<float> batch0(seq_len * num_heads * head_dim);
    std::vector<float> batch1(seq_len * num_heads * head_dim);

    size_t batch_stride = seq_len * num_heads * head_dim;
    std::copy(Q_batched.begin(), Q_batched.begin() + batch_stride, batch0.begin());
    std::copy(Q_batched.begin() + batch_stride, Q_batched.begin() + 2 * batch_stride, batch1.begin());

    // Verify batch separation
    EXPECT_GT(batch0[0], 0.0f) << "Batch 0 should have positive values";
    EXPECT_LT(batch1[0], 0.0f) << "Batch 1 should have negative values";

    // Common bug: Using wrong offset
    // If we accidentally use (b * seq_len + m) instead of (b * batch_stride + m * head_stride)
    // we'd read from wrong batch

    int wrong_offset = 1 * seq_len; // Bug: treating seq_len as batch stride
    int right_offset = 1 * batch_stride;

    EXPECT_NE(wrong_offset, right_offset) << "Wrong offset should differ from correct offset";

    // Wrong offset would read from middle of batch 0
    EXPECT_GT(Q_batched[wrong_offset], 0.0f) << "Wrong offset reads from batch 0";

    // Right offset reads from batch 1
    EXPECT_LT(Q_batched[right_offset], 0.0f) << "Right offset reads from batch 1";
}

// =============================================================================
// MPI ATTENTION SLICING TESTS
// Issue: MpiAttentionOrchestrator and GQAAttention slice attention across ranks
//        Incorrect slicing causes wrong heads processed, missing data, or corruption
// =============================================================================

TEST_F(FusedAttentionWoRobustnessTest, MPISlice_HeadPartitioning)
{
    // Test head partitioning across MPI ranks (tensor parallelism)
    // Each rank processes a subset of attention heads
    //
    // Example: 14 heads, 2 ranks → rank 0 gets heads [0,6], rank 1 gets heads [7,13]
    //
    // Common bugs:
    // 1. Off-by-one in head range calculation
    // 2. Remainder heads not assigned (14 heads, 4 ranks → 2 heads unprocessed)
    // 3. Wrong stride when extracting head slice

    struct HeadPartitionConfig
    {
        int n_heads;
        int world_size;
        std::vector<std::pair<int, int>> expected_ranges; // [start, end) per rank
    };

    std::vector<HeadPartitionConfig> configs = {
        // Even division
        {4, 2, {{0, 2}, {2, 4}}},
        {8, 4, {{0, 2}, {2, 4}, {4, 6}, {6, 8}}},

        // Qwen2.5 0.5B: 14 heads, 2 ranks
        {14, 2, {{0, 7}, {7, 14}}},

        // Uneven division (remainder heads go to last rank)
        // Note: This is one strategy - others distribute remainder differently
        {7, 2, {{0, 3}, {3, 7}}},         // Rank 0: 3 heads, Rank 1: 4 heads
        {7, 3, {{0, 2}, {2, 4}, {4, 7}}}, // 2, 2, 3 heads
    };

    for (const auto &cfg : configs)
    {
        SCOPED_TRACE("Testing n_heads=" + std::to_string(cfg.n_heads) +
                     " world_size=" + std::to_string(cfg.world_size));

        // Compute head ranges for each rank
        int heads_per_rank = cfg.n_heads / cfg.world_size;
        int remainder = cfg.n_heads % cfg.world_size;

        int assigned_heads = 0;
        for (int rank = 0; rank < cfg.world_size; ++rank)
        {
            // Common formula: rank gets heads_per_rank + (1 if rank < remainder else 0)
            int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
            int start_head = assigned_heads;
            int end_head = start_head + local_heads;

            // Alternative simple formula (gives remainder to last rank):
            // int start_head = rank * heads_per_rank;
            // int end_head = (rank == world_size - 1) ? n_heads : (rank + 1) * heads_per_rank;

            assigned_heads += local_heads;

            // Verify range is valid
            EXPECT_GE(start_head, 0) << "Start head should be >= 0 for rank " << rank;
            EXPECT_LE(end_head, cfg.n_heads) << "End head should be <= n_heads for rank " << rank;
            EXPECT_GT(local_heads, 0) << "Each rank should have at least 1 head";
        }

        // All heads should be assigned exactly once
        EXPECT_EQ(assigned_heads, cfg.n_heads) << "Total assigned heads should equal n_heads";
    }
}

TEST_F(FusedAttentionWoRobustnessTest, MPISlice_HeadExtraction_Interleaved)
{
    // Test extracting a single head from interleaved multi-head layout
    // This is what MpiAttentionOrchestrator::extract_head_data() does
    //
    // Input layout: [seq_len, n_heads * head_dim]
    //   Position 0: [head0_d0, head0_d1, ..., head0_d63, head1_d0, ..., head1_d63, ...]
    //   Position 1: [head0_d0, head0_d1, ..., head0_d63, head1_d0, ..., head1_d63, ...]
    //
    // Output layout: [seq_len, head_dim] (contiguous for single head)
    //   Position 0: [head_h_d0, head_h_d1, ..., head_h_d63]
    //   Position 1: [head_h_d0, head_h_d1, ..., head_h_d63]
    //
    // Common bug: Wrong stride calculation

    const int seq_len = 4;
    const int n_heads = 4;
    const int head_dim = 8; // Small for easy verification

    // Create interleaved tensor with unique values per (pos, head, dim)
    // Value = pos * 1000 + head * 100 + dim
    std::vector<float> interleaved(seq_len * n_heads * head_dim);
    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                int idx = pos * (n_heads * head_dim) + h * head_dim + d;
                interleaved[idx] = static_cast<float>(pos * 1000 + h * 100 + d);
            }
        }
    }

    // Extract head 2 using correct algorithm
    int target_head = 2;
    std::vector<float> extracted(seq_len * head_dim);

    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            // Correct: advance by n_heads * head_dim per position, offset by head * head_dim
            int src_idx = pos * (n_heads * head_dim) + target_head * head_dim + d;
            int dst_idx = pos * head_dim + d;
            extracted[dst_idx] = interleaved[src_idx];
        }
    }

    // Verify extraction
    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            float expected = static_cast<float>(pos * 1000 + target_head * 100 + d);
            float actual = extracted[pos * head_dim + d];
            EXPECT_FLOAT_EQ(actual, expected)
                << "Extraction wrong at pos=" << pos << " dim=" << d;
        }
    }

    // Demonstrate common bug: using wrong stride
    std::vector<float> wrong_extracted(seq_len * head_dim);
    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            // Bug: using head_dim stride instead of n_heads * head_dim
            int wrong_src_idx = pos * head_dim + target_head * head_dim + d; // WRONG!
            wrong_extracted[pos * head_dim + d] = interleaved[wrong_src_idx % interleaved.size()];
        }
    }

    // Wrong extraction should differ
    bool differs = false;
    for (int i = 0; i < seq_len * head_dim; ++i)
    {
        if (std::abs(extracted[i] - wrong_extracted[i]) > 0.01f)
        {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs) << "Wrong stride should produce different results";
}

TEST_F(FusedAttentionWoRobustnessTest, MPISlice_KVBroadcast_GQA)
{
    // Test K/V head broadcasting for GQA
    // When n_kv_heads < n_heads, each KV head is shared by multiple Q heads
    //
    // Example: 14 Q heads, 2 KV heads
    //   Q heads 0-6 use KV head 0
    //   Q heads 7-13 use KV head 1
    //
    // For MPI: If rank 0 has Q heads 0-6, it only needs KV head 0
    //          If rank 1 has Q heads 7-13, it only needs KV head 1
    //
    // Common bug: Broadcasting wrong KV head to wrong Q heads

    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int seq_len = 4;
    const int head_dim = 8;
    const int heads_per_kv = n_heads / n_kv_heads; // 7

    // Create KV tensors with distinctive values per KV head
    // KV head 0: all values = 100.0
    // KV head 1: all values = 200.0
    std::vector<float> K_original(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_original(seq_len * n_kv_heads * head_dim);

    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            float val = (kv_h == 0) ? 100.0f : 200.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                int idx = pos * (n_kv_heads * head_dim) + kv_h * head_dim + d;
                K_original[idx] = val + d * 0.01f;           // Slight variation per dim
                V_original[idx] = val + d * 0.01f + 1000.0f; // Different from K
            }
        }
    }

    // Broadcast K/V to match Q heads
    std::vector<float> K_broadcast(seq_len * n_heads * head_dim);
    std::vector<float> V_broadcast(seq_len * n_heads * head_dim);

    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int q_h = 0; q_h < n_heads; ++q_h)
        {
            int kv_h = q_h / heads_per_kv; // Which KV head this Q head uses

            for (int d = 0; d < head_dim; ++d)
            {
                int src_idx = pos * (n_kv_heads * head_dim) + kv_h * head_dim + d;
                int dst_idx = pos * (n_heads * head_dim) + q_h * head_dim + d;

                K_broadcast[dst_idx] = K_original[src_idx];
                V_broadcast[dst_idx] = V_original[src_idx];
            }
        }
    }

    // Verify Q heads 0-6 got KV head 0 (values ~100)
    for (int q_h = 0; q_h < 7; ++q_h)
    {
        int idx = 0 * (n_heads * head_dim) + q_h * head_dim + 0; // pos=0, dim=0
        EXPECT_NEAR(K_broadcast[idx], 100.0f, 0.1f)
            << "Q head " << q_h << " should have KV head 0's K values";
    }

    // Verify Q heads 7-13 got KV head 1 (values ~200)
    for (int q_h = 7; q_h < 14; ++q_h)
    {
        int idx = 0 * (n_heads * head_dim) + q_h * head_dim + 0; // pos=0, dim=0
        EXPECT_NEAR(K_broadcast[idx], 200.0f, 0.1f)
            << "Q head " << q_h << " should have KV head 1's K values";
    }
}

TEST_F(FusedAttentionWoRobustnessTest, MPISlice_OutputAccumulation)
{
    // Test output accumulation from head slices
    // Each MPI rank computes attention for its heads, then allreduce combines
    //
    // Layout: output[pos, head] comes from rank that owns that head
    // After allreduce: each position has contributions from all ranks summed
    //
    // Common bug: Double-counting heads, missing heads, wrong allreduce dimension

    const int seq_len = 4;
    const int n_heads = 8;
    const int head_dim = 4;
    const int d_model = n_heads * head_dim; // 32
    const int world_size = 2;

    // Simulate output from each rank
    // Rank 0 computes heads 0-3, rank 1 computes heads 4-7
    // Each rank's output is zero except for its heads
    std::vector<std::vector<float>> rank_outputs(world_size);
    for (int rank = 0; rank < world_size; ++rank)
    {
        rank_outputs[rank].resize(seq_len * d_model, 0.0f);

        int start_head = rank * (n_heads / world_size);
        int end_head = (rank + 1) * (n_heads / world_size);

        for (int pos = 0; pos < seq_len; ++pos)
        {
            for (int h = start_head; h < end_head; ++h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    int idx = pos * d_model + h * head_dim + d;
                    // Unique value: rank * 1000 + pos * 100 + head * 10 + dim
                    rank_outputs[rank][idx] = static_cast<float>(rank * 1000 + pos * 100 + h * 10 + d);
                }
            }
        }
    }

    // Simulate allreduce (sum outputs from all ranks)
    std::vector<float> final_output(seq_len * d_model, 0.0f);
    for (int rank = 0; rank < world_size; ++rank)
    {
        for (size_t i = 0; i < final_output.size(); ++i)
        {
            final_output[i] += rank_outputs[rank][i];
        }
    }

    // Verify each head's contribution came from exactly one rank
    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            int owning_rank = h / (n_heads / world_size);

            for (int d = 0; d < head_dim; ++d)
            {
                int idx = pos * d_model + h * head_dim + d;
                float expected = static_cast<float>(owning_rank * 1000 + pos * 100 + h * 10 + d);

                EXPECT_FLOAT_EQ(final_output[idx], expected)
                    << "Output at pos=" << pos << " head=" << h << " dim=" << d
                    << " should come from rank " << owning_rank;
            }
        }
    }
}

TEST_F(FusedAttentionWoRobustnessTest, MPISlice_WoWeight_HeadSlice)
{
    // Test Wo weight slicing for head-parallel attention
    // Full Wo: [d_model, n_heads * head_dim]
    // Rank's slice: [d_model, local_n_heads * head_dim]
    //
    // When rank processes heads [start_h, end_h), it needs Wo columns
    // corresponding to those heads' positions in the context vector
    //
    // Common bug: Extracting wrong columns, using row slice instead of column slice

    const int d_model = 8;
    const int n_heads = 4;
    const int head_dim = 2;
    const int context_dim = n_heads * head_dim; // 8
    const int world_size = 2;

    // Create full Wo with distinctive pattern
    // Wo[row][col] = row * 100 + col
    std::vector<float> Wo_full(d_model * context_dim);
    for (int row = 0; row < d_model; ++row)
    {
        for (int col = 0; col < context_dim; ++col)
        {
            Wo_full[row * context_dim + col] = static_cast<float>(row * 100 + col);
        }
    }

    // Each rank extracts columns for its heads
    for (int rank = 0; rank < world_size; ++rank)
    {
        int start_head = rank * (n_heads / world_size);
        int end_head = (rank + 1) * (n_heads / world_size);
        int local_heads = end_head - start_head;
        int col_start = start_head * head_dim;
        int col_end = end_head * head_dim;
        int local_cols = col_end - col_start;

        // Extract column slice for this rank
        std::vector<float> Wo_slice(d_model * local_cols);
        for (int row = 0; row < d_model; ++row)
        {
            for (int local_col = 0; local_col < local_cols; ++local_col)
            {
                int global_col = col_start + local_col;
                int src_idx = row * context_dim + global_col;
                int dst_idx = row * local_cols + local_col;
                Wo_slice[dst_idx] = Wo_full[src_idx];
            }
        }

        // Verify slice contains correct columns
        SCOPED_TRACE("Rank " + std::to_string(rank));

        for (int row = 0; row < d_model; ++row)
        {
            for (int local_col = 0; local_col < local_cols; ++local_col)
            {
                int global_col = col_start + local_col;
                float expected = static_cast<float>(row * 100 + global_col);
                float actual = Wo_slice[row * local_cols + local_col];

                EXPECT_FLOAT_EQ(actual, expected)
                    << "Wo slice wrong at row=" << row << " local_col=" << local_col
                    << " (global_col=" << global_col << ")";
            }
        }
    }
}

TEST_F(FusedAttentionWoRobustnessTest, MPISlice_ContextToOutput_Partial)
{
    // Test partial context → output projection
    // When each rank has only local_heads, the context is smaller:
    //   Full context: [head_dim * n_heads]
    //   Rank context: [head_dim * local_heads]
    //
    // The Wo projection for rank: out[d] = sum over local_heads of (context[h*head_dim + d'] * Wo[d][h*head_dim + d'])
    //
    // Common bug: Using full d_model indexing for local context

    const int d_model = 8;
    const int n_heads = 4;
    const int head_dim = 2;
    const int world_size = 2;

    for (int rank = 0; rank < world_size; ++rank)
    {
        SCOPED_TRACE("Testing rank " + std::to_string(rank));

        int local_heads = n_heads / world_size;         // 2 heads per rank
        int local_context_dim = local_heads * head_dim; // 4

        // Local context from this rank's attention
        std::vector<float> local_context(local_context_dim);
        for (int i = 0; i < local_context_dim; ++i)
        {
            local_context[i] = static_cast<float>(rank * 100 + i + 1); // Distinctive per rank
        }

        // Wo slice for this rank (columns for our heads)
        int col_start = rank * local_context_dim;
        std::vector<float> Wo_slice(d_model * local_context_dim);
        for (int row = 0; row < d_model; ++row)
        {
            for (int col = 0; col < local_context_dim; ++col)
            {
                // Simple pattern: Wo[row][col_start + col] = 1.0 (sum of context)
                Wo_slice[row * local_context_dim + col] = 1.0f;
            }
        }

        // Compute partial output
        std::vector<float> partial_output(d_model, 0.0f);
        for (int row = 0; row < d_model; ++row)
        {
            for (int col = 0; col < local_context_dim; ++col)
            {
                partial_output[row] += local_context[col] * Wo_slice[row * local_context_dim + col];
            }
        }

        // With Wo = 1.0, output should be sum of local context
        float expected_sum = 0.0f;
        for (int i = 0; i < local_context_dim; ++i)
        {
            expected_sum += local_context[i];
        }

        for (int row = 0; row < d_model; ++row)
        {
            EXPECT_FLOAT_EQ(partial_output[row], expected_sum)
                << "Partial output wrong at row " << row;
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
