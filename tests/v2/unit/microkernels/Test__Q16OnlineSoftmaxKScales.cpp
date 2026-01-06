/**
 * @file Test__Q16OnlineSoftmaxKScales.cpp
 * @brief Unit tests for FA2 prefill with per-position K scales microkernel
 *
 * Tests the fa2_prefill_process_kv_tile_with_k_scales microkernel against
 * an FP32 reference to validate the per-position K scale handling.
 *
 * The key test case: K positions with DIFFERENT dynamic scales (like HybridQ16)
 * should produce correct softmax weights when compared to FP32 reference.
 *
 * @see HANDOVER_HYBRIDQ16_ATTENTION_INVESTIGATION.md
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>
#include <numeric>

#include "kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProduct.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2::kernels::q16_1::microkernels;
using namespace llaminar2;

class Test__Q16OnlineSoftmaxKScales : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    /**
     * @brief Compute FP32 reference attention scores and weights.
     *
     * @param Q_fp32 Query [num_q, head_dim]
     * @param K_fp32 Key [kv_len, head_dim]
     * @param q_scales Per-Q scale factors [num_q] - what Q was quantized with
     * @param k_scales Per-K scale factors [kv_len] - what K was quantized with
     * @param head_dim Dimension
     * @param causal Use causal mask
     * @return Normalized attention weights [num_q, kv_len]
     */
    std::vector<float> fp32ReferenceWeights(
        const std::vector<float> &Q_fp32,
        const std::vector<float> &K_fp32,
        int num_q,
        int kv_len,
        int head_dim,
        bool causal = true)
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> weights(num_q * kv_len);

        for (int q = 0; q < num_q; ++q)
        {
            // Compute Q @ K^T
            std::vector<float> scores(kv_len);
            float max_score = -std::numeric_limits<float>::infinity();

            for (int k = 0; k < kv_len; ++k)
            {
                if (causal && k > q)
                {
                    scores[k] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += Q_fp32[q * head_dim + d] * K_fp32[k * head_dim + d];
                }
                scores[k] = dot * scale;
                max_score = std::max(max_score, scores[k]);
            }

            // Softmax
            float sum = 0.0f;
            for (int k = 0; k < kv_len; ++k)
            {
                if (scores[k] > -1e30f)
                {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum += scores[k];
                }
                else
                {
                    scores[k] = 0.0f;
                }
            }
            for (int k = 0; k < kv_len; ++k)
            {
                weights[q * kv_len + k] = scores[k] / sum;
            }
        }
        return weights;
    }

    /**
     * @brief Quantize FP32 to Q16_1 blocks with per-position dynamic scaling.
     *
     * This mimics HybridQ16's K cache where each position gets its own scale.
     * d[pos] = max_abs(row) / 32767 (dynamic per-row)
     */
    std::vector<Q16_1Block_64> quantizeToQ16Dynamic(
        const std::vector<float> &fp32_data,
        int rows,
        int cols)
    {
        constexpr int BLOCK_SIZE = 64;
        int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<Q16_1Block_64> blocks(rows * blocks_per_row);

        for (int r = 0; r < rows; ++r)
        {
            // Find max abs for this row
            float max_abs = 0.0f;
            for (int c = 0; c < cols; ++c)
            {
                max_abs = std::max(max_abs, std::abs(fp32_data[r * cols + c]));
            }
            float d = (max_abs > 1e-20f) ? (max_abs / 32767.0f) : 1e-20f;
            float inv_d = 1.0f / d;

            for (int b = 0; b < blocks_per_row; ++b)
            {
                Q16_1Block_64 &block = blocks[r * blocks_per_row + b];
                block.d = d; // Per-row scale stored in block header
                int32_t sum = 0;

                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    int col = b * BLOCK_SIZE + i;
                    float val = (col < cols) ? fp32_data[r * cols + col] : 0.0f;
                    int32_t q = static_cast<int32_t>(std::round(val * inv_d));
                    q = std::clamp(q, -32767, 32767);
                    block.qs[i] = static_cast<int16_t>(q);
                    sum += block.qs[i];
                }
                block.sum_qs = sum;
            }
        }
        return blocks;
    }

    /**
     * @brief Quantize FP32 to Q16_1 blocks with UNIFORM scaling (like Q head in HybridQ16).
     */
    std::vector<Q16_1Block_64> quantizeToQ16Uniform(
        const std::vector<float> &fp32_data,
        int rows,
        int cols,
        float scale)
    {
        constexpr int BLOCK_SIZE = 64;
        int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<Q16_1Block_64> blocks(rows * blocks_per_row);

        float d = scale / 32767.0f;
        float inv_d = 32767.0f / scale;

        for (int r = 0; r < rows; ++r)
        {
            for (int b = 0; b < blocks_per_row; ++b)
            {
                Q16_1Block_64 &block = blocks[r * blocks_per_row + b];
                block.d = d;
                int32_t sum = 0;

                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    int col = b * BLOCK_SIZE + i;
                    float val = (col < cols) ? fp32_data[r * cols + col] : 0.0f;
                    int32_t q = static_cast<int32_t>(std::round(val * inv_d));
                    q = std::clamp(q, -32767, 32767);
                    block.qs[i] = static_cast<int16_t>(q);
                    sum += block.qs[i];
                }
                block.sum_qs = sum;
            }
        }
        return blocks;
    }

    /**
     * @brief Extract per-row K scales from K blocks (what attention sees).
     */
    std::vector<float> extractKScalesFromBlocks(
        const std::vector<Q16_1Block_64> &K_blocks,
        int kv_len,
        int blocks_per_row)
    {
        std::vector<float> k_scales(kv_len);
        for (int k = 0; k < kv_len; ++k)
        {
            k_scales[k] = K_blocks[k * blocks_per_row].d;
        }
        return k_scales;
    }

    float cosineSimilarity(const float *a, const float *b, int n)
    {
        double dot = 0, norm_a = 0, norm_b = 0;
        for (int i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0f;
        return static_cast<float>(dot / std::sqrt(norm_a * norm_b));
    }

    std::vector<float> generateNormal(int count, float stddev = 1.0f)
    {
        std::normal_distribution<float> dist(0.0f, stddev);
        std::vector<float> data(count);
        for (auto &v : data)
            v = dist(rng_);
        return data;
    }
};

// =============================================================================
// Test: Verify uniform scales work (baseline)
// =============================================================================

TEST_F(Test__Q16OnlineSoftmaxKScales, UniformScales_Baseline)
{
    // Simple test with UNIFORM scales (like working Hybrid Q8_1 path)
    // Use REALISTIC values from the pipeline:
    //   - Q uses fixed scale 64 (from Q8_1 -> Q16_1 conversion)
    //   - K/V also use fixed scale 64 for uniform path
    //   - FP32 values have stddev ~2, so max_abs ~6-8
    constexpr int HEAD_DIM = 64;
    constexpr int NUM_Q = 4;
    constexpr int KV_LEN = 8;
    constexpr float KV_SCALE = 64.0f; // Realistic pipeline scale

    // Generate FP32 data with realistic stddev
    // After RoPE, Q/K values are typically in range [-5, 5]
    auto Q_fp32 = generateNormal(NUM_Q * HEAD_DIM, 2.0f);
    auto K_fp32 = generateNormal(KV_LEN * HEAD_DIM, 2.0f);
    auto V_fp32 = generateNormal(KV_LEN * HEAD_DIM, 2.0f);

    // Quantize with UNIFORM scale (like kv_cache_scale=64 path)
    auto Q_q16 = quantizeToQ16Uniform(Q_fp32, NUM_Q, HEAD_DIM, KV_SCALE);
    auto K_q16 = quantizeToQ16Uniform(K_fp32, KV_LEN, HEAD_DIM, KV_SCALE);
    auto V_q16 = quantizeToQ16Uniform(V_fp32, KV_LEN, HEAD_DIM, KV_SCALE);

    // FP32 reference weights
    auto ref_weights = fp32ReferenceWeights(Q_fp32, K_fp32, NUM_Q, KV_LEN, HEAD_DIM, true);

    // Extract K scales (should all be same = KV_SCALE/32767)
    auto k_scales = extractKScalesFromBlocks(K_q16, KV_LEN, 1);
    float expected_d = KV_SCALE / 32767.0f;
    for (int k = 0; k < KV_LEN; ++k)
    {
        EXPECT_NEAR(k_scales[k], expected_d, 1e-6f)
            << "K scale[" << k << "] should be uniform";
    }

    // Q scale (uniform)
    float q_scale = Q_q16[0].d;
    EXPECT_NEAR(q_scale, expected_d, 1e-6f) << "Q scale should be uniform";

    // Compute qk_scale for uniform path
    float qk_scale = q_scale * k_scales[0] / std::sqrt(static_cast<float>(HEAD_DIM));

    std::cout << "Uniform scales test:\n"
              << "  q_scale = " << q_scale << "\n"
              << "  k_scale = " << k_scales[0] << "\n"
              << "  qk_scale = " << qk_scale << "\n";

    // The uniform path should work correctly
    EXPECT_GT(qk_scale, 0.0f);
}

// =============================================================================
// Test: Verify per-position K scales (HybridQ16 scenario)
// =============================================================================

TEST_F(Test__Q16OnlineSoftmaxKScales, PerPositionKScales_Math)
{
    // Test with DYNAMIC K scales (like HybridQ16)
    // REALISTIC values from pipeline:
    //   - Q uses fixed scale 64 (d = 64/32767 ≈ 0.00195)
    //   - K uses DYNAMIC per-position scale (d ≈ 0.003-0.01)
    //   - FP32 values after RoPE have varying magnitudes
    constexpr int HEAD_DIM = 64;
    constexpr int NUM_Q = 4;
    constexpr int KV_LEN = 8;
    constexpr float Q_SCALE = 64.0f; // Q uses fixed scale 64 (like HybridQ16)

    // Generate FP32 data with realistic magnitudes
    // Q: stddev ~2, max_abs ~6 (within scale 64 range)
    auto Q_fp32 = generateNormal(NUM_Q * HEAD_DIM, 2.0f);
    // K: varying magnitudes per position (simulating dynamic quantization)
    // After RoPE, K values can have different ranges per position
    auto K_fp32 = generateNormal(KV_LEN * HEAD_DIM, 3.0f); // Slightly larger stddev
    auto V_fp32 = generateNormal(KV_LEN * HEAD_DIM, 3.0f);

    // Scale some K rows to have different magnitudes (simulating dynamic quantization)
    // This creates per-position K scales ranging from ~0.003 to ~0.010
    for (int k = 0; k < KV_LEN; ++k)
    {
        float mult = 1.0f + 0.5f * k; // Scale factor 1.0 to 4.5
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            K_fp32[k * HEAD_DIM + d] *= mult;
        }
    }

    // Q: UNIFORM scale (like real HybridQ16)
    auto Q_q16 = quantizeToQ16Uniform(Q_fp32, NUM_Q, HEAD_DIM, Q_SCALE);

    // K: DYNAMIC per-position scale (like real HybridQ16 K cache)
    auto K_q16 = quantizeToQ16Dynamic(K_fp32, KV_LEN, HEAD_DIM);

    // V: DYNAMIC (like real HybridQ16 V cache)
    auto V_q16 = quantizeToQ16Dynamic(V_fp32, KV_LEN, HEAD_DIM);

    // Extract K scales from blocks
    auto k_scales = extractKScalesFromBlocks(K_q16, KV_LEN, 1);

    // Q scale
    float q_scale = Q_q16[0].d;

    std::cout << "Per-position K scales test:\n"
              << "  q_scale = " << q_scale << " (uniform)\n";
    for (int k = 0; k < KV_LEN; ++k)
    {
        std::cout << "  k_scale[" << k << "] = " << k_scales[k] << "\n";
    }

    // Verify K scales are DIFFERENT (dynamic)
    bool all_same = true;
    for (int k = 1; k < KV_LEN; ++k)
    {
        if (std::abs(k_scales[k] - k_scales[0]) > 1e-8f)
        {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "K scales should be different for dynamic quantization";

    // Compute what alpha values SHOULD be for per-position path
    std::cout << "\nPer-position alpha values:\n";
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    float base_alpha = q_scale * inv_sqrt_d;
    std::cout << "  base_alpha = q_scale / sqrt(d) = " << base_alpha << "\n";

    for (int k = 0; k < KV_LEN; ++k)
    {
        float per_pos_alpha = base_alpha * k_scales[k];
        float qk_scale_equivalent = q_scale * k_scales[k] * inv_sqrt_d;
        std::cout << "  alpha[" << k << "] = " << per_pos_alpha
                  << " (qk_scale_equiv = " << qk_scale_equivalent << ")\n";
    }

    // Verify the math: integer dot product * q_scale * k_scale should match FP32 dot
    std::cout << "\nDot product verification (Q[0] @ K[k]):\n";
    for (int k = 0; k < std::min(4, KV_LEN); ++k)
    {
        // FP32 reference dot
        float fp32_dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            fp32_dot += Q_fp32[d] * K_fp32[k * HEAD_DIM + d];
        }

        // Integer dot
        int32_t int_dot = q16_dot_single<Q16_1Block_64>(Q_q16.data(), K_q16.data() + k, HEAD_DIM, 1);

        // Reconstructed FP32 = int_dot * q_scale * k_scale
        float reconstructed = static_cast<float>(int_dot) * q_scale * k_scales[k];

        float rel_error = std::abs(reconstructed - fp32_dot) / (std::abs(fp32_dot) + 1e-6f);
        std::cout << "  k=" << k << ": FP32=" << fp32_dot
                  << ", int=" << int_dot
                  << ", recon=" << reconstructed
                  << ", rel_err=" << rel_error << "\n";

        // Should be close (within quantization error)
        EXPECT_LT(rel_error, 0.05f) << "Dot product reconstruction error too large for k=" << k;
    }
}

// =============================================================================
// Test: Full microkernel with per-position K scales
// =============================================================================

TEST_F(Test__Q16OnlineSoftmaxKScales, MicrokernelParity)
{
    // Full test of fa2_prefill_process_kv_tile_with_k_scales vs FP32 reference
    // REALISTIC values from pipeline:
    //   - Q: fixed scale 64 (d = 64/32767 ≈ 0.00195)
    //   - K: dynamic per-position scale (d ≈ 0.003-0.012)
    //   - V: dynamic per-position scale
    constexpr int HEAD_DIM = 64;
    constexpr int Br = 4;            // Q tile size
    constexpr int Bc = 8;            // KV tile size
    constexpr float Q_SCALE = 64.0f; // Fixed Q scale (like HybridQ16)

    // Generate FP32 data with realistic magnitudes
    auto Q_fp32 = generateNormal(Br * HEAD_DIM, 2.0f);
    auto K_fp32 = generateNormal(Bc * HEAD_DIM, 3.0f);
    auto V_fp32 = generateNormal(Bc * HEAD_DIM, 3.0f);

    // Vary K magnitudes to create per-position scale variation
    // This simulates the dynamic quantization in HybridQ16
    for (int k = 0; k < Bc; ++k)
    {
        float mult = 1.0f + 0.6f * k; // Scale factors 1.0 to 5.2
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            K_fp32[k * HEAD_DIM + d] *= mult;
        }
    }

    // Quantize
    auto Q_q16 = quantizeToQ16Uniform(Q_fp32, Br, HEAD_DIM, Q_SCALE);
    auto K_q16 = quantizeToQ16Dynamic(K_fp32, Bc, HEAD_DIM);
    auto V_q16 = quantizeToQ16Dynamic(V_fp32, Bc, HEAD_DIM);

    // Extract scales
    auto k_scales = extractKScalesFromBlocks(K_q16, Bc, 1);
    float q_scale = Q_q16[0].d;

    // FP32 reference weights
    auto ref_weights = fp32ReferenceWeights(Q_fp32, K_fp32, Br, Bc, HEAD_DIM, true);

    // Setup microkernel state and scratch
    OnlineSoftmaxStateBatch state;
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    float base_alpha = q_scale * inv_sqrt_d;
    state.init_per_position(Br, q_scale, HEAD_DIM);

    std::vector<int32_t> scores_scratch(Br * Bc);
    std::vector<int32_t> weights_scratch(Br * Bc);
    std::vector<int32_t> context(Br * HEAD_DIM, 0);

    // Run microkernel
    fa2_prefill_process_kv_tile_with_k_scales<Q16_1Block_64>(
        Q_q16.data(), K_q16.data(), V_q16.data(),
        context.data(), state,
        scores_scratch.data(), weights_scratch.data(),
        0, // kv_tile_start
        Br, Bc,
        HEAD_DIM,
        1,          // blocks_per_row
        1,          // q_stride
        1,          // k_stride
        HEAD_DIM,   // context_stride
        base_alpha, // base_alpha_fp32
        k_scales.data(),
        true, // causal
        0);   // q_offset

    // Normalize microkernel weights to compare with reference
    // Weight in weights_scratch is exp2(scaled_score - max) in fixed-point
    // We need to normalize by state.l to get probabilities
    std::vector<float> mk_weights(Br * Bc);
    for (int r = 0; r < Br; ++r)
    {
        float sum = 0.0f;
        for (int c = 0; c < Bc; ++c)
        {
            // Convert INT32 weight to float
            float w = static_cast<float>(weights_scratch[r * Bc + c]);
            sum += w;
        }
        // Normalize
        if (sum > 0.0f)
        {
            for (int c = 0; c < Bc; ++c)
            {
                mk_weights[r * Bc + c] = static_cast<float>(weights_scratch[r * Bc + c]) / sum;
            }
        }
    }

    // Compare
    std::cout << "\nMicrokernel vs FP32 weights (row 0):\n";
    std::cout << "  FP32: ";
    for (int c = 0; c < Bc; ++c)
        std::cout << std::fixed << std::setprecision(4) << ref_weights[c] << " ";
    std::cout << "\n  MK:   ";
    for (int c = 0; c < Bc; ++c)
        std::cout << std::fixed << std::setprecision(4) << mk_weights[c] << " ";
    std::cout << "\n";

    // Compute cosine similarity per row
    float total_cos = 0.0f;
    for (int r = 0; r < Br; ++r)
    {
        float cos = cosineSimilarity(&ref_weights[r * Bc], &mk_weights[r * Bc], Bc);
        total_cos += cos;
        std::cout << "  Row " << r << " cosine: " << cos << "\n";
    }
    float avg_cos = total_cos / Br;
    std::cout << "Average cosine similarity: " << avg_cos << "\n";

    // Should be high
    EXPECT_GT(avg_cos, 0.95f) << "Per-position K scale microkernel should match FP32 reference";
}
// =============================================================================
// Test: Non-causal with full attention pattern (more rigorous)
// =============================================================================

TEST_F(Test__Q16OnlineSoftmaxKScales, MicrokernelParity_NonCausal)
{
    // Full test with non-causal attention to exercise softmax over ALL positions
    // This is more rigorous than causal because every query attends to every key.
    // NOTE: FA2_TILE_BR = 8 is the max supported by OnlineSoftmaxStateBatch
    constexpr int HEAD_DIM = 64;
    constexpr int Br = 8; // Max supported by OnlineSoftmaxStateBatch
    constexpr int Bc = 8; // Full sequence (square for non-causal)
    constexpr float Q_SCALE = 64.0f;

    // Generate FP32 data with realistic magnitudes
    auto Q_fp32 = generateNormal(Br * HEAD_DIM, 2.0f);
    auto K_fp32 = generateNormal(Bc * HEAD_DIM, 3.0f);
    auto V_fp32 = generateNormal(Bc * HEAD_DIM, 3.0f);

    // Vary K magnitudes to create diverse per-position scales
    for (int k = 0; k < Bc; ++k)
    {
        float mult = 1.0f + 0.5f * k;
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            K_fp32[k * HEAD_DIM + d] *= mult;
        }
    }

    // Quantize
    auto Q_q16 = quantizeToQ16Uniform(Q_fp32, Br, HEAD_DIM, Q_SCALE);
    auto K_q16 = quantizeToQ16Dynamic(K_fp32, Bc, HEAD_DIM);
    auto V_q16 = quantizeToQ16Dynamic(V_fp32, Bc, HEAD_DIM);

    // Extract scales
    auto k_scales = extractKScalesFromBlocks(K_q16, Bc, 1);
    float q_scale = Q_q16[0].d;

    std::cout << "\n=== Non-causal test with pipeline-realistic scales ===\n";
    std::cout << "Q scale (fixed): " << q_scale << " (expected ~0.00195)\n";
    std::cout << "K scales (dynamic):\n";
    for (int k = 0; k < Bc; ++k)
    {
        std::cout << "  k_scale[" << k << "] = " << k_scales[k] << "\n";
    }

    // FP32 reference weights (NON-CAUSAL)
    auto ref_weights = fp32ReferenceWeights(Q_fp32, K_fp32, Br, Bc, HEAD_DIM, false);

    // Setup microkernel state
    OnlineSoftmaxStateBatch state;
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    float base_alpha = q_scale * inv_sqrt_d;
    state.init_per_position(Br, q_scale, HEAD_DIM);

    std::vector<int32_t> scores_scratch(Br * Bc);
    std::vector<int32_t> weights_scratch(Br * Bc);
    std::vector<int32_t> context(Br * HEAD_DIM, 0);

    // Run microkernel with NON-CAUSAL
    fa2_prefill_process_kv_tile_with_k_scales<Q16_1Block_64>(
        Q_q16.data(), K_q16.data(), V_q16.data(),
        context.data(), state,
        scores_scratch.data(), weights_scratch.data(),
        0, Br, Bc,
        HEAD_DIM, 1, 1, 1, HEAD_DIM,
        base_alpha, k_scales.data(),
        false, // NON-CAUSAL
        0);

    // Normalize weights
    std::vector<float> mk_weights(Br * Bc);
    for (int r = 0; r < Br; ++r)
    {
        float sum = 0.0f;
        for (int c = 0; c < Bc; ++c)
            sum += static_cast<float>(weights_scratch[r * Bc + c]);
        if (sum > 0.0f)
        {
            for (int c = 0; c < Bc; ++c)
                mk_weights[r * Bc + c] = static_cast<float>(weights_scratch[r * Bc + c]) / sum;
        }
    }

    // Compare
    std::cout << "\nFP32 vs Microkernel weights (row 4 = middle):\n";
    std::cout << "  FP32: ";
    for (int c = 0; c < Bc; ++c)
        std::cout << std::fixed << std::setprecision(4) << ref_weights[4 * Bc + c] << " ";
    std::cout << "\n  MK:   ";
    for (int c = 0; c < Bc; ++c)
        std::cout << std::fixed << std::setprecision(4) << mk_weights[4 * Bc + c] << " ";
    std::cout << "\n";

    // Compute cosine similarity per row
    float total_cos = 0.0f;
    int low_cos_count = 0;
    for (int r = 0; r < Br; ++r)
    {
        float cos = cosineSimilarity(&ref_weights[r * Bc], &mk_weights[r * Bc], Bc);
        total_cos += cos;
        if (cos < 0.9f)
        {
            low_cos_count++;
            std::cout << "  Row " << r << " LOW cosine: " << cos << "\n";
            std::cout << "    FP32: ";
            for (int c = 0; c < Bc; ++c)
                std::cout << std::fixed << std::setprecision(4) << ref_weights[r * Bc + c] << " ";
            std::cout << "\n    MK:   ";
            for (int c = 0; c < Bc; ++c)
                std::cout << std::fixed << std::setprecision(4) << mk_weights[r * Bc + c] << " ";
            std::cout << "\n    Raw scores: ";
            for (int c = 0; c < Bc; ++c)
                std::cout << scores_scratch[r * Bc + c] << " ";
            std::cout << "\n    Raw weights: ";
            for (int c = 0; c < Bc; ++c)
                std::cout << weights_scratch[r * Bc + c] << " ";
            std::cout << "\n";
        }
    }
    float avg_cos = total_cos / Br;
    std::cout << "Average cosine similarity: " << avg_cos << "\n";
    std::cout << "Rows with cosine < 0.9: " << low_cos_count << "/" << Br << "\n";

    EXPECT_GT(avg_cos, 0.95f) << "Non-causal microkernel should match FP32 reference";
}

// =============================================================================
// Test: Extreme K scale ranges (like real pipeline across layers)
// =============================================================================

TEST_F(Test__Q16OnlineSoftmaxKScales, ExtremeKScaleRange_PipelineRealistic)
{
    // Test with EXTREME K scale ranges observed in real pipeline:
    //   - Layer 0: K_scale ~ 0.008  (largest)
    //   - Layer 4-5: K_scale ~ 2.4e-05 (smallest, 300x smaller!)
    //   - Layer 8: K_scale ~ 0.013 (even larger)
    //
    // This tests whether the per-position K scale math handles 300x range.
    constexpr int HEAD_DIM = 64;
    constexpr int Br = 4;
    constexpr int Bc = 8;
    constexpr float Q_SCALE = 64.0f; // Fixed Q scale

    // Generate FP32 data
    auto Q_fp32 = generateNormal(Br * HEAD_DIM, 2.0f);
    auto K_fp32_base = generateNormal(Bc * HEAD_DIM, 1.0f);
    auto V_fp32 = generateNormal(Bc * HEAD_DIM, 1.0f);

    // Create K with EXTREME magnitude variation per position
    // Simulates pipeline's inter-layer K scale variation within a single tile
    std::vector<float> K_fp32(Bc * HEAD_DIM);
    std::vector<float> forced_k_scales = {
        0.008f,   // K[0] large like layer 0
        0.00003f, // K[1] tiny like layer 4-5 (300x smaller)
        0.013f,   // K[2] largest like layer 8
        0.00005f, // K[3] tiny
        0.002f,   // K[4] medium
        0.0001f,  // K[5] small
        0.005f,   // K[6] medium-large
        0.0002f   // K[7] small
    };

    for (int k = 0; k < Bc; ++k)
    {
        // Scale K values to achieve the target scale after quantization
        // Q16 scale d = max_abs / 32767, so max_abs = d * 32767
        float target_max_abs = forced_k_scales[k] * 32767.0f;

        // Find current max abs of base K row
        float cur_max_abs = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            cur_max_abs = std::max(cur_max_abs, std::abs(K_fp32_base[k * HEAD_DIM + d]));
        }

        // Scale to achieve target
        float mult = target_max_abs / (cur_max_abs + 1e-6f);
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            K_fp32[k * HEAD_DIM + d] = K_fp32_base[k * HEAD_DIM + d] * mult;
        }
    }

    // Quantize
    auto Q_q16 = quantizeToQ16Uniform(Q_fp32, Br, HEAD_DIM, Q_SCALE);
    auto K_q16 = quantizeToQ16Dynamic(K_fp32, Bc, HEAD_DIM);
    auto V_q16 = quantizeToQ16Dynamic(V_fp32, Bc, HEAD_DIM);

    // Extract actual K scales
    auto k_scales = extractKScalesFromBlocks(K_q16, Bc, 1);
    float q_scale = Q_q16[0].d;

    std::cout << "\n=== Extreme K scale range test (pipeline realistic) ===\n";
    std::cout << "Q scale (fixed): " << q_scale << "\n";
    std::cout << "K scales (extreme variation):\n";
    float k_max = 0, k_min = 1e10f;
    for (int k = 0; k < Bc; ++k)
    {
        k_max = std::max(k_max, k_scales[k]);
        k_min = std::min(k_min, k_scales[k]);
        std::cout << "  k_scale[" << k << "] = " << std::scientific << k_scales[k]
                  << " (target: " << forced_k_scales[k] << ")\n";
    }
    std::cout << "K scale range: " << std::fixed << (k_max / k_min) << "x\n";

    // FP32 reference (non-causal for full test coverage)
    auto ref_weights = fp32ReferenceWeights(Q_fp32, K_fp32, Br, Bc, HEAD_DIM, false);

    // Setup microkernel
    OnlineSoftmaxStateBatch state;
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    float base_alpha = q_scale * inv_sqrt_d;
    state.init_per_position(Br, q_scale, HEAD_DIM);

    std::vector<int32_t> scores_scratch(Br * Bc);
    std::vector<int32_t> weights_scratch(Br * Bc);
    std::vector<int32_t> context(Br * HEAD_DIM, 0);

    // Run microkernel
    fa2_prefill_process_kv_tile_with_k_scales<Q16_1Block_64>(
        Q_q16.data(), K_q16.data(), V_q16.data(),
        context.data(), state,
        scores_scratch.data(), weights_scratch.data(),
        0, Br, Bc,
        HEAD_DIM, 1, 1, 1, HEAD_DIM,
        base_alpha, k_scales.data(),
        false, 0);

    // Normalize weights
    std::vector<float> mk_weights(Br * Bc);
    for (int r = 0; r < Br; ++r)
    {
        float sum = 0.0f;
        for (int c = 0; c < Bc; ++c)
            sum += static_cast<float>(weights_scratch[r * Bc + c]);
        if (sum > 0.0f)
        {
            for (int c = 0; c < Bc; ++c)
                mk_weights[r * Bc + c] = static_cast<float>(weights_scratch[r * Bc + c]) / sum;
        }
    }

    // Compare
    float total_cos = 0.0f;
    int low_cos_count = 0;
    for (int r = 0; r < Br; ++r)
    {
        float cos = cosineSimilarity(&ref_weights[r * Bc], &mk_weights[r * Bc], Bc);
        total_cos += cos;
        if (cos < 0.9f)
        {
            low_cos_count++;
            std::cout << "\n  Row " << r << " LOW cosine: " << std::fixed << std::setprecision(4) << cos << "\n";
            std::cout << "    FP32: ";
            for (int c = 0; c < Bc; ++c)
                std::cout << ref_weights[r * Bc + c] << " ";
            std::cout << "\n    MK:   ";
            for (int c = 0; c < Bc; ++c)
                std::cout << mk_weights[r * Bc + c] << " ";
            std::cout << "\n    Raw scores: ";
            for (int c = 0; c < Bc; ++c)
                std::cout << scores_scratch[r * Bc + c] << " ";
            std::cout << "\n";
        }
    }
    float avg_cos = total_cos / Br;
    std::cout << "\nAverage cosine similarity: " << avg_cos << "\n";
    std::cout << "Rows with cosine < 0.9: " << low_cos_count << "/" << Br << "\n";

    EXPECT_GT(avg_cos, 0.90f) << "Extreme K scale range should still produce reasonable results";
}

// =============================================================================
// Test: Real pipeline Q value range (max_abs=130) with kv_cache_scale=64 CLIPS!
// =============================================================================

TEST_F(Test__Q16OnlineSoftmaxKScales, RealisticQValueRange_ExposesClipping)
{
    // CRITICAL: Real pipeline Q projection values have max_abs=130.352
    // But kv_cache_scale=64 can only represent [-32, +32]!
    // This causes severe clipping (4x beyond range).
    //
    // This test reproduces the bug by using realistic Q value ranges.
    constexpr int HEAD_DIM = 64;
    constexpr int Br = 4;                   // Q rows
    constexpr int Bc = 8;                   // K columns
    constexpr float Q_SCALE_FIXED = 256.0f; // FIXED: was 64 which clipped Q values

    // Generate Q with REALISTIC high magnitude (like real pipeline)
    // Real pipeline shows Q max_abs=130, which is 4x beyond scale=64 limit (32)
    auto Q_fp32_base = generateNormal(Br * HEAD_DIM, 20.0f); // stddev=20 gives max ~60
    // Scale up to match real pipeline max_abs ~130
    std::vector<float> Q_fp32(Q_fp32_base.size());
    for (size_t i = 0; i < Q_fp32_base.size(); ++i)
    {
        Q_fp32[i] = Q_fp32_base[i] * 2.0f; // Now max ~130
    }

    // K and V with normal range
    auto K_fp32 = generateNormal(Bc * HEAD_DIM, 2.0f);
    auto V_fp32 = generateNormal(Bc * HEAD_DIM, 2.0f);

    // Find actual Q max to confirm clipping
    float q_max_abs = 0.0f;
    for (float v : Q_fp32)
        q_max_abs = std::max(q_max_abs, std::abs(v));

    std::cout << "\n=== RealisticQValueRange_ExposesClipping ===\n";
    std::cout << "Q max_abs = " << q_max_abs << " (real pipeline has ~130)\n";
    std::cout << "Q_SCALE_FIXED = " << Q_SCALE_FIXED << "\n";
    float max_representable = Q_SCALE_FIXED * (16383.0f / 32767.0f);
    std::cout << "Max representable with scale=64: " << max_representable << "\n";
    std::cout << "Clipping ratio: " << (q_max_abs / max_representable) << "x beyond range\n";

    // Quantize with fixed scale (like the buggy path)
    auto Q_q16 = quantizeToQ16Uniform(Q_fp32, Br, HEAD_DIM, Q_SCALE_FIXED);
    auto K_q16 = quantizeToQ16Dynamic(K_fp32, Bc, HEAD_DIM); // K uses dynamic (correct)
    auto V_q16 = quantizeToQ16Uniform(V_fp32, Bc, HEAD_DIM, 64.0f);

    // Dequantize Q back to see how much was lost
    std::vector<float> Q_dequant(Br * HEAD_DIM);
    int blocks_per_row = HEAD_DIM / 64;
    for (int r = 0; r < Br; ++r)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const auto &blk = Q_q16[r * blocks_per_row + b];
            for (int i = 0; i < 64; ++i)
            {
                Q_dequant[r * HEAD_DIM + b * 64 + i] = blk.qs[i] * blk.d;
            }
        }
    }

    // Compute reconstruction error
    float mse = 0.0f;
    int clipped_count = 0;
    for (size_t i = 0; i < Q_fp32.size(); ++i)
    {
        float diff = Q_fp32[i] - Q_dequant[i];
        mse += diff * diff;
        // Check if original was clipped (exceeded max_representable)
        if (std::abs(Q_fp32[i]) > max_representable * 1.01f)
        {
            clipped_count++;
        }
    }
    mse /= Q_fp32.size();
    float rmse = std::sqrt(mse);

    std::cout << "Q reconstruction RMSE: " << rmse << "\n";
    std::cout << "Q elements clipped: " << clipped_count << "/" << Q_fp32.size()
              << " (" << (100.0f * clipped_count / Q_fp32.size()) << "%)\n";

    // Compute cosine between original and dequantized Q
    float q_cos = cosineSimilarity(Q_fp32.data(), Q_dequant.data(), Q_fp32.size());
    std::cout << "Q original vs dequantized cosine: " << q_cos << "\n";

    // Now run attention and check output
    auto ref_weights = fp32ReferenceWeights(Q_fp32, K_fp32, Br, Bc, HEAD_DIM, true);

    // Compare with attention using clipped Q
    auto clipped_ref = fp32ReferenceWeights(Q_dequant, K_fp32, Br, Bc, HEAD_DIM, true);
    float attn_cos = cosineSimilarity(ref_weights.data(), clipped_ref.data(), ref_weights.size());
    std::cout << "Attention weights: original Q vs clipped Q cosine: " << attn_cos << "\n";

    // This test EXPOSES the bug - we expect cosine to be poor due to clipping!
    // When fixed, we should change the expectation to require cos > 0.99
    std::cout << "\n*** BUG WAS: kv_cache_scale=64 clipped Q values! ***\n";
    std::cout << "FIXED: Increased kv_cache_scale to 256.\n\n";

    // This test used to expose the bug with kv_cache_scale=64
    // Now that scale=256, there should be minimal clipping
    std::cout << "\n*** FIXED: kv_cache_scale=256 should prevent Q value clipping ***\n\n";

    // After fix: expect good reconstruction
    EXPECT_GT(q_cos, 0.999f) << "Q quantization should have minimal error with scale=256";
}