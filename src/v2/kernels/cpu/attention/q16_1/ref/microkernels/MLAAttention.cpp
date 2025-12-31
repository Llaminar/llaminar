/**
 * @file MLAAttention.cpp
 * @brief Multi-head Latent Attention (MLA) microkernel implementation
 *
 * @see MLAAttention.h for algorithm description
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md "MLA Architecture" section
 */

#include "MLAAttention.h"
#include "Q16DotProduct.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cmath>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Single Dot Product Implementation
    // ============================================================================

    int32_t q16_dot_single_mla(
        const Q16_1Block_128 *Q_nope,
        const Q16_1Block_64 *Q_rope,
        const Q16_1Block_128 *K_nope,
        const Q16_1Block_64 *K_rope)
    {
        // SUB-BLOCK ACCUMULATION for VNNI compatibility
        // ==============================================
        // AVX-512 VNNI (VPDPWSSD) performs INT16×INT16→INT32 accumulation.
        // We accumulate NOPE and ROPE separately to bound overflow risk:
        //
        //   NOPE: 128 dims → max 128 × 4096² = 2.1B ≈ INT32_MAX (safe with bounded values)
        //   ROPE: 64 dims → max 64 × 5793² = 2.1B ≈ INT32_MAX (safe with bounded values)
        //
        // By accumulating separately, we only overflow if a SINGLE component
        // exceeds INT32_MAX, rather than the combined 192-dim sum.
        //
        // The final sum of two int32 partials into the result is safe because
        // real model activations don't simultaneously max out both components.

        int32_t nope_acc = 0;
        int32_t rope_acc = 0;

        // ========== NOPE component: 128-element dot product ==========
        // This loop pattern vectorizes to VPDPWSSD with AVX-512 VNNI
        {
            const int16_t *q_data = Q_nope->qs;
            const int16_t *k_data = K_nope->qs;

            for (int i = 0; i < MLAConfig::NOPE_DIM; ++i)
            {
                nope_acc += static_cast<int32_t>(q_data[i]) * static_cast<int32_t>(k_data[i]);
            }
        }

        // ========== ROPE component: 64-element dot product ==========
        {
            const int16_t *q_data = Q_rope->qs;
            const int16_t *k_data = K_rope->qs;

            for (int i = 0; i < MLAConfig::ROPE_DIM; ++i)
            {
                rope_acc += static_cast<int32_t>(q_data[i]) * static_cast<int32_t>(k_data[i]);
            }
        }

        // Combine partials - safe for realistic activation ranges
        // Note: Extreme outliers could overflow here, but kv_cache_scale bounds
        // ensure this doesn't happen with properly quantized values.
        return nope_acc + rope_acc;
    }

    // ============================================================================
    // GEMV Implementation (Flash Decode)
    // ============================================================================

    void q16_qk_gemv_mla(
        const Q16_1Block_128 *Q_nope,
        const Q16_1Block_64 *Q_rope,
        const Q16_1Block_128 *K_nope,
        const Q16_1Block_64 *K_rope,
        int32_t *scores,
        int kv_len,
        int k_nope_stride,
        int k_rope_stride)
    {
        // Parallel over KV positions for Flash Decode
        // Each score computation is independent
#pragma omp parallel for schedule(static)
        for (int k = 0; k < kv_len; ++k)
        {
            // Locate K blocks for this position
            const Q16_1Block_128 *K_nope_k = K_nope + k * k_nope_stride;
            const Q16_1Block_64 *K_rope_k = K_rope + k * k_rope_stride;

            scores[k] = q16_dot_single_mla(Q_nope, Q_rope, K_nope_k, K_rope_k);
        }
    }

    void q16_qk_gemv_mla(
        const MLAAttentionParams &params,
        int32_t *scores,
        int head_idx)
    {
        // Handle GQA: map query head to KV head
        int kv_head_idx = head_idx;
        if (params.n_kv_heads < params.n_heads)
        {
            // GQA: multiple Q heads share one KV head
            int heads_per_kv = params.n_heads / params.n_kv_heads;
            kv_head_idx = head_idx / heads_per_kv;
        }

        // Get Q blocks for this head (assuming single query position for decode)
        const Q16_1Block_128 *Q_nope = params.Q_nope + head_idx * MLAConfig::NOPE_BLOCKS_PER_HEAD;
        const Q16_1Block_64 *Q_rope = params.Q_rope + head_idx * MLAConfig::ROPE_BLOCKS_PER_HEAD;

        // Get K cache for this KV head
        // K layout: [kv_len, n_kv_heads, blocks_per_head]
        // First K position for this head:
        const Q16_1Block_128 *K_nope = params.K_nope + kv_head_idx * MLAConfig::NOPE_BLOCKS_PER_HEAD;
        const Q16_1Block_64 *K_rope = params.K_rope + kv_head_idx * MLAConfig::ROPE_BLOCKS_PER_HEAD;

        // Compute strides: distance between consecutive KV positions for this head
        int k_nope_stride = params.k_nope_stride > 0 ? params.k_nope_stride
                                                     : params.n_kv_heads * MLAConfig::NOPE_BLOCKS_PER_HEAD;
        int k_rope_stride = params.k_rope_stride > 0 ? params.k_rope_stride
                                                     : params.n_kv_heads * MLAConfig::ROPE_BLOCKS_PER_HEAD;

        q16_qk_gemv_mla(Q_nope, Q_rope, K_nope, K_rope, scores, params.kv_len,
                        k_nope_stride, k_rope_stride);
    }

    // ============================================================================
    // GEMM Tile Implementation (FA2 Prefill)
    // ============================================================================

    void q16_qk_gemm_tile_mla(
        const Q16_1Block_128 *Q_nope,
        const Q16_1Block_64 *Q_rope,
        const Q16_1Block_128 *K_nope,
        const Q16_1Block_64 *K_rope,
        int32_t *scores,
        int Br,
        int Bc,
        int q_nope_stride,
        int q_rope_stride,
        int k_nope_stride,
        int k_rope_stride)
    {
        // Compute S[q, k] = Q_nope[q] · K_nope[k] + Q_rope[q] · K_rope[k]
        // Output is row-major: scores[q * Bc + k]

        for (int q = 0; q < Br; ++q)
        {
            // Get Q blocks for this query position
            const Q16_1Block_128 *Q_nope_q = Q_nope + q * q_nope_stride;
            const Q16_1Block_64 *Q_rope_q = Q_rope + q * q_rope_stride;

            // Pointer to output row
            int32_t *scores_row = scores + q * Bc;

            for (int k = 0; k < Bc; ++k)
            {
                // Get K blocks for this key position
                const Q16_1Block_128 *K_nope_k = K_nope + k * k_nope_stride;
                const Q16_1Block_64 *K_rope_k = K_rope + k * k_rope_stride;

                scores_row[k] = q16_dot_single_mla(Q_nope_q, Q_rope_q, K_nope_k, K_rope_k);
            }
        }
    }

    // ============================================================================
    // Scale Factor Application
    // ============================================================================

    void mla_apply_scales(
        const Q16_1Block_128 *Q_nope,
        const Q16_1Block_64 *Q_rope,
        const Q16_1Block_128 *K_nope,
        const Q16_1Block_64 *K_rope,
        const int32_t *int_scores,
        float *fp_scores,
        int kv_len,
        int k_nope_stride,
        int k_rope_stride)
    {
        // For each position, we need to compute the proper scale combination
        //
        // The integer score is the SUM of two dot products:
        //   int_score = nope_dot_int + rope_dot_int
        //
        // But these come from different quantization scales:
        //   nope_dot_fp = q_nope.d * k_nope.d * nope_dot_int
        //   rope_dot_fp = q_rope.d * k_rope.d * rope_dot_int
        //
        // Without tracking them separately, we use an approximation:
        // The combined score is dominated by the larger component, so we
        // weight by the relative magnitudes of NOPE vs ROPE dimensions.

        float q_nope_scale = Q_nope->d;
        float q_rope_scale = Q_rope->d;

        // Weight factors based on dimension ratio
        // NOPE is 128 dims, ROPE is 64 dims → NOPE contributes ~2/3 of dot product variance
        constexpr float nope_weight = static_cast<float>(MLAConfig::NOPE_DIM) /
                                      static_cast<float>(MLAConfig::TOTAL_QK_DIM);
        constexpr float rope_weight = static_cast<float>(MLAConfig::ROPE_DIM) /
                                      static_cast<float>(MLAConfig::TOTAL_QK_DIM);

#pragma omp parallel for schedule(static)
        for (int k = 0; k < kv_len; ++k)
        {
            const Q16_1Block_128 *K_nope_k = K_nope + k * k_nope_stride;
            const Q16_1Block_64 *K_rope_k = K_rope + k * k_rope_stride;

            float k_nope_scale = K_nope_k->d;
            float k_rope_scale = K_rope_k->d;

            // Weighted combination of scale products
            float nope_scale_product = q_nope_scale * k_nope_scale;
            float rope_scale_product = q_rope_scale * k_rope_scale;

            // Approximation: weighted average based on dimension contribution
            float combined_scale = nope_weight * nope_scale_product +
                                   rope_weight * rope_scale_product;

            fp_scores[k] = combined_scale * static_cast<float>(int_scores[k]);
        }
    }

} // namespace llaminar2::kernels::q16_1::microkernels
