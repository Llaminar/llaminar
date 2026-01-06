/**
 * @file OnlineSoftmax.cpp
 * @brief Online softmax microkernel implementation for FlashDecode
 *
 * @see OnlineSoftmax.h for algorithm description
 * @see Exp2Core.h for shared exp2 LUT primitives
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */

#include "OnlineSoftmax.h"
#include "Exp2Core.h"
#include "Exp2FixedSoftmax.h"
#include "Q16DotProduct.h"
#include "PVAccumulate.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Block-wise Online Softmax Operations
    // ============================================================================

    int32_t online_softmax_find_block_max(
        const int32_t *scores,
        int block_size,
        int32_t mask_value)
    {
        // Delegate to shared implementation
        return exp2_find_block_max(scores, block_size, mask_value);
    }

    void online_softmax_compute_rescale_adaptive(
        int32_t m_old,
        int32_t m_new,
        const AdaptiveAlphaConfig &alpha_config,
        int32_t &scale_num,
        int &scale_shift,
        int frac_bits,
        int lut_value_bits)
    {
        // Delegate to shared implementation
        exp2_compute_rescale(m_old, m_new, alpha_config, scale_num, scale_shift, frac_bits, lut_value_bits);
    }

    void online_softmax_compute_rescale(
        int32_t m_old,
        int32_t m_new,
        float alpha,
        int32_t &scale_num,
        int &scale_shift,
        int frac_bits,
        int lut_value_bits)
    {
        // Use adaptive scaling for better precision with small alpha
        AdaptiveAlphaConfig alpha_config = AdaptiveAlphaConfig::compute(alpha);
        exp2_compute_rescale(m_old, m_new, alpha_config, scale_num, scale_shift, frac_bits, lut_value_bits);
    }

    int64_t online_softmax_compute_block_weights_adaptive(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        const AdaptiveAlphaConfig &alpha_config,
        int frac_bits,
        int lut_value_bits)
    {
        // Delegate to shared implementation
        return exp2_compute_block_weights(scores, weights, block_size, running_max, alpha_config, frac_bits, lut_value_bits);
    }

    int64_t online_softmax_compute_block_weights(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        float alpha,
        int frac_bits,
        int lut_value_bits)
    {
        // Use adaptive scaling for better precision with small alpha
        AdaptiveAlphaConfig alpha_config = AdaptiveAlphaConfig::compute(alpha);
        return exp2_compute_block_weights(scores, weights, block_size, running_max, alpha_config, frac_bits, lut_value_bits);
    }

    bool online_softmax_update_block(
        OnlineSoftmaxState &state,
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        float alpha,
        int32_t &scale_num,
        int &scale_shift)
    {
        // Ensure state has alpha_config initialized (lazy init if needed)
        // Note: For best performance, call state.init(alpha) once before processing
        if (state.alpha_config.original_alpha == 0.0f && alpha != 0.0f)
        {
            state.alpha_config = AdaptiveAlphaConfig::compute(alpha);
        }

        // Step 1: Find block max (using shared primitive)
        int32_t block_max = exp2_find_block_max(scores, block_size, MASKED_SCORE);

        // All masked in this block
        if (block_max == MASKED_SCORE)
        {
            std::memset(weights, 0, block_size * sizeof(int32_t));
            scale_num = 1 << state.lut_value_bits;
            scale_shift = state.lut_value_bits;
            return false;
        }

        // Step 2: Check if max needs updating
        bool needs_rescale = false;
        int32_t m_old = state.m;
        int32_t m_new = state.m;

        if (state.empty() || block_max > state.m)
        {
            m_new = block_max;
            needs_rescale = !state.empty(); // Don't rescale on first block
        }

        // Step 3: Compute rescale factor using shared primitive
        if (needs_rescale)
        {
            exp2_compute_rescale(
                m_old, m_new, state.alpha_config,
                scale_num, scale_shift,
                state.frac_bits, state.lut_value_bits);

            // Also rescale running sum l
            int64_t scaled_l = (static_cast<int64_t>(state.l) * scale_num) >> scale_shift;
            state.l = scaled_l;
        }
        else
        {
            scale_num = 1 << state.lut_value_bits;
            scale_shift = state.lut_value_bits;
        }

        // Step 4: Compute weights for this block using shared primitive
        int64_t block_sum = exp2_compute_block_weights(
            scores, weights, block_size,
            m_new, state.alpha_config,
            state.frac_bits, state.lut_value_bits);

        // Step 5: Update state
        state.m = m_new;
        state.l += block_sum;

        // Count unmasked positions
        for (int i = 0; i < block_size; ++i)
        {
            if (scores[i] != MASKED_SCORE)
            {
                ++state.count;
            }
        }

        return needs_rescale;
    }

    int32_t online_softmax_finalize_weights(
        const OnlineSoftmaxState &state,
        const int32_t *block_weights,
        int16_t *final_weights,
        int total_len,
        int16_t weight_max)
    {
        // Delegate to shared implementation
        return exp2_normalize_weights(block_weights, final_weights, total_len, state.l, weight_max);
    }

    // ============================================================================
    // FlashDecode Block Processing
    // ============================================================================

    template <typename BlockType>
    void flash_decode_process_kv_block(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int32_t *context,
        OnlineSoftmaxState &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_block_start,
        int kv_block_size,
        int head_dim,
        int blocks_per_row,
        float alpha)
    {
        // Step 1: Compute Q×K^T scores for this block
        const BlockType *K_block = K + kv_block_start * blocks_per_row;

        for (int k = 0; k < kv_block_size; ++k)
        {
            const BlockType *K_row = K_block + k * blocks_per_row;
            scores_scratch[k] = q16_dot_single<BlockType>(Q, K_row, head_dim, blocks_per_row);
        }

        // Step 2: Online softmax update
        int32_t scale_num;
        int scale_shift;
        bool needs_rescale = online_softmax_update_block(
            state, scores_scratch, weights_scratch,
            kv_block_size, alpha,
            scale_num, scale_shift);

        // Step 3: Rescale l_processed if max changed (using double precision)
        // NOTE: Context does NOT need rescaling!
        // Context stores the normalized weighted average: Σ(w*V) / Σ(w)
        // When weights scale by k: numerator and denominator both scale by k,
        // so the ratio (the running average) is unchanged.
        // We only need to rescale l_processed so the merge formula works correctly.
        if (needs_rescale)
        {
            // Rescale l_processed using exact double arithmetic
            // scale_factor = scale_num / 2^scale_shift
            double scale_factor = static_cast<double>(scale_num) / static_cast<double>(1ULL << scale_shift);
            state.l_processed *= scale_factor;
        }

        // Step 4: Compute weighted V sum for this block and merge using running average
        // Using RUNNING AVERAGE approach to keep context bounded
        //
        // DESIGN INSIGHT:
        // We want context to store the running weighted average: Σ(w*V) / Σ(w)
        // This stays bounded regardless of sequence length.
        //
        // For this block:
        //   block_contribution = Σ(w[k] * V[k]) for k in block
        //   block_sum = Σ(w[k]) for k in block
        //
        // Merge formula:
        //   context_new = (context_old * l_old + block_contribution) / l_new
        //               = (context_old * l_old + block_contribution) / (l_old + block_sum)
        //
        // To avoid overflow, we scale weights down to INT16 range first.
        const BlockType *V_block = V + kv_block_start * blocks_per_row;
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        // Compute block sum of weights at scaled-down precision
        double block_sum_scaled = 0.0;
        for (int k = 0; k < kv_block_size; ++k)
        {
            int32_t w_scaled = weights_scratch[k] >> (state.lut_value_bits - 15);
            block_sum_scaled += w_scaled;
        }

        // Temp buffer for block's weighted V sum (scaled)
        alignas(64) int64_t block_context[256]; // Max head_dim
        std::memset(block_context, 0, head_dim * sizeof(int64_t));

        for (int k = 0; k < kv_block_size; ++k)
        {
            if (weights_scratch[k] == 0)
                continue;

            // Scale weight down to INT16 range (same as before)
            int32_t w_scaled = weights_scratch[k] >> (state.lut_value_bits - 15);
            if (w_scaled == 0 && weights_scratch[k] > 0)
            {
                w_scaled = 1;
            }

            const BlockType *V_row = V_block + k * blocks_per_row;

            // Accumulate w_scaled * V[k] into block_context
            for (int b = 0; b < blocks_per_row; ++b)
            {
                const int16_t *v_data = V_row[b].qs;
                int start = b * BLOCK_SIZE;
                int end = std::min(start + BLOCK_SIZE, head_dim);
                int count = end - start;

                for (int i = 0; i < count; ++i)
                {
                    block_context[start + i] += static_cast<int64_t>(w_scaled) * static_cast<int64_t>(v_data[i]);
                }
            }
        }

        // Step 5: Merge with running context using weighted average (double precision)
        // context stores: (Σ w_scaled * V) / l_processed
        // So we need to un-normalize, add, and re-normalize
        double l_old = state.l_processed;
        double l_new = l_old + block_sum_scaled;

        if (l_new > 0.0)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                // Un-normalize: context * l_old gives the accumulated weighted sum
                // Add new block contribution
                // Re-normalize by dividing by l_new
                double numerator = static_cast<double>(context[d]) * l_old + static_cast<double>(block_context[d]);
                context[d] = static_cast<int32_t>(std::round(numerator / l_new));
            }
        }

        // Update l_processed (in scaled units, matching block_sum_scaled)
        state.l_processed = l_new;
    }

    // ============================================================================
    // FlashDecode Block Processing with Per-Position K Scales (INTEGER-FIRST)
    // ============================================================================

    /**
     * @brief FlashDecode with per-position K scales using PURE INTEGER arithmetic.
     *
     * This is a redesign of the per-position K scale path to stay integer-first,
     * matching the standard path's design principles.
     *
     * @section Algorithm
     *
     * For each position k, the full alpha is:
     *   alpha[k] = q_scale * k_scale[k] * inv_sqrt_d * log2(e)
     *
     * We pre-compute AdaptiveAlphaConfig for each position, which handles the
     * fixed-point scaling to maintain precision with small alphas.
     *
     * The key insight is that we track the MAXIMUM SCALED SCORE across positions,
     * where each position's scaled score is computed with its own alpha_config.
     * This allows us to find the global max and compute weights correctly.
     *
     * @section Complexity
     *
     * The main challenge is that different positions have different alpha values,
     * so we can't directly compare raw scores. We solve this by:
     * 1. Computing scaled_score[k] = score[k] * alpha[k] in integer fixed-point
     * 2. Tracking the global max in this scaled domain
     * 3. Computing weights as exp2(scaled_score[k] - max) in integer domain
     */
    template <typename BlockType>
    void flash_decode_process_kv_block_with_k_scales(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int32_t *context,
        OnlineSoftmaxState &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_block_start,
        int kv_block_size,
        int head_dim,
        int blocks_per_row,
        const float *k_scales)
    {
        // Step 1: Compute Q×K^T scores for this block (PURE INTEGER)
        const BlockType *K_block = K + kv_block_start * blocks_per_row;

        for (int k = 0; k < kv_block_size; ++k)
        {
            const BlockType *K_row = K_block + k * blocks_per_row;
            scores_scratch[k] = q16_dot_single<BlockType>(Q, K_row, head_dim, blocks_per_row);
        }

        // Step 2: Pre-compute per-position AdaptiveAlphaConfig (INTEGER-FIRST)
        // alpha[k] = base_alpha * k_scale[k] where base_alpha = q_scale * inv_sqrt_d * log2(e)
        std::vector<AdaptiveAlphaConfig> alpha_configs(kv_block_size);
        for (int k = 0; k < kv_block_size; ++k)
        {
            float per_pos_alpha = state.base_alpha_fp32 * k_scales[k];
            alpha_configs[k] = AdaptiveAlphaConfig::compute(per_pos_alpha);
        }

        // Step 3: Compute scaled scores in fixed-point domain
        // For each position, t[k] = score[k] * alpha[k] in fixed-point
        // We need to track max in a COMMON scale for comparison
        //
        // IMPORTANT: The alpha_shift in AdaptiveAlphaConfig cancels out in the
        // t_fixed computation, so all t_fixed values ARE in the same scale:
        //   t_fixed = score * per_pos_alpha * LOG2E * 2^frac_bits
        //
        // This is because:
        //   M = effective_alpha * LOG2E * 2^24 = per_pos_alpha * 2^alpha_shift * LOG2E * 2^24
        //   shift_for_t = 24 - frac_bits + alpha_shift
        //   t_fixed = (score * M) >> shift_for_t
        //           = score * per_pos_alpha * 2^alpha_shift * LOG2E * 2^24 / 2^(24 - frac_bits + alpha_shift)
        //           = score * per_pos_alpha * LOG2E * 2^frac_bits
        //
        // So alpha_shift cancels out and t_fixed is directly proportional to the real score.

        constexpr int beta_scale_bits = 24;
        std::vector<int64_t> scaled_scores_fixed(kv_block_size);
        int64_t block_max_fixed = std::numeric_limits<int64_t>::min();

        // DEBUG: Print first few values (ONCE per layer)
        static int debug_count = 0;
        if (debug_count < 1 && kv_block_size > 0)
        {
            std::cout << "DEBUG: flash_decode_process_kv_block_with_k_scales" << std::endl;
            std::cout << "  base_alpha_fp32: " << state.base_alpha_fp32 << std::endl;

            // Print k_scales for all positions
            for (int k = 0; k < kv_block_size && k < 9; ++k)
            {
                float per_pos_alpha = state.base_alpha_fp32 * k_scales[k];
                // Compute the "real" score = score_int * q_scale * k_scale
                float q_scale = Q[0].d; // Use first block's d
                float real_score = static_cast<float>(scores_scratch[k]) * q_scale * k_scales[k];
                std::cout << "  k=" << k
                          << " score_int=" << scores_scratch[k]
                          << " k_scale=" << k_scales[k]
                          << " per_pos_alpha=" << per_pos_alpha
                          << " real_score=" << real_score
                          << std::endl;
            }

            // Print Q and K samples
            std::cout << "  Q[0].qs[0-3]: " << Q[0].qs[0] << " " << Q[0].qs[1] << " " << Q[0].qs[2] << " " << Q[0].qs[3] << std::endl;
            std::cout << "  Q[0].d: " << Q[0].d << std::endl;

            const BlockType *K_row0 = K + kv_block_start * blocks_per_row;
            std::cout << "  K[0].qs[0-3]: " << K_row0[0].qs[0] << " " << K_row0[0].qs[1] << " " << K_row0[0].qs[2] << " " << K_row0[0].qs[3] << std::endl;
            std::cout << "  K[0].d: " << K_row0[0].d << std::endl;
            debug_count++;
        }

        for (int k = 0; k < kv_block_size; ++k)
        {
            if (scores_scratch[k] == MASKED_SCORE)
            {
                scaled_scores_fixed[k] = std::numeric_limits<int64_t>::min();
                continue;
            }

            // Compute t = score * alpha * log2(e) in fixed-point
            // Using the same formula as exp2_compute_block_weights:
            //   M = effective_alpha * LOG2E * 2^beta_scale_bits
            //   t_fixed = (score * M) >> (beta_scale_bits - frac_bits + alpha_shift)
            const auto &cfg = alpha_configs[k];
            double effective_beta = static_cast<double>(cfg.effective_alpha) * LOG2E;
            int64_t M = static_cast<int64_t>(
                std::llround(effective_beta * static_cast<double>(1ULL << beta_scale_bits)));

            int shift_for_t = beta_scale_bits - state.frac_bits + cfg.alpha_shift;

            int64_t prod = static_cast<int64_t>(scores_scratch[k]) * M;
            int64_t t_fixed = (shift_for_t >= 0)
                                  ? (prod >> shift_for_t)
                                  : (prod << (-shift_for_t));

            scaled_scores_fixed[k] = t_fixed;

            if (debug_count < 1 && k < 4)
            {
                std::cout << "  k=" << k << " score=" << scores_scratch[k]
                          << " eff_alpha=" << cfg.effective_alpha
                          << " shift=" << cfg.alpha_shift
                          << " M=" << M
                          << " shift_for_t=" << shift_for_t
                          << " t_fixed=" << t_fixed << std::endl;
            }

            if (t_fixed > block_max_fixed)
            {
                block_max_fixed = t_fixed;
            }
        }

        // All masked in this block
        if (block_max_fixed == std::numeric_limits<int64_t>::min())
        {
            std::memset(weights_scratch, 0, kv_block_size * sizeof(int32_t));
            return;
        }

        // Clamp to int32_t range for state.m
        int32_t block_max = static_cast<int32_t>(
            std::clamp(block_max_fixed,
                       static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                       static_cast<int64_t>(std::numeric_limits<int32_t>::max())));

        // Step 4: Check if max needs updating and compute rescale
        bool needs_rescale = false;
        int32_t m_old = state.m;
        int32_t m_new = state.m;

        if (state.empty() || block_max > state.m)
        {
            m_new = block_max;
            needs_rescale = !state.empty();
        }

        int32_t scale_num = 1 << state.lut_value_bits;
        int scale_shift = state.lut_value_bits;

        if (needs_rescale)
        {
            // Rescale factor: 2^((m_old - m_new) / 2^frac_bits)
            // m_old and m_new are in scaled domain (t * 2^frac_bits)
            int32_t delta = m_new - m_old; // delta >= 0

            int int_part = delta >> state.frac_bits;
            int frac_part = delta & ((1 << state.frac_bits) - 1);

            if (int_part >= 32)
            {
                scale_num = 0;
                scale_shift = 0;
            }
            else
            {
                ensure_exp2_lut_initialized(state.lut_value_bits);
                const uint32_t *lut = get_exp2_lut_data();

                uint32_t lut_val = lut[frac_part];
                scale_num = static_cast<int32_t>(lut_val);
                scale_shift = state.lut_value_bits + int_part;
            }

            // Rescale running sum l
            int64_t scaled_l = (static_cast<int64_t>(state.l) * scale_num) >> scale_shift;
            state.l = scaled_l;

            // Rescale l_processed
            double rescale_factor = static_cast<double>(scale_num) / static_cast<double>(1ULL << scale_shift);
            state.l_processed *= rescale_factor;
        }

        // Step 5: Compute weights using INTEGER arithmetic
        // w[k] = exp2(scaled_score[k] - m_new) via LUT
        ensure_exp2_lut_initialized(state.lut_value_bits);
        const uint32_t *lut = get_exp2_lut_data();
        const uint32_t one = static_cast<uint32_t>(1U << state.lut_value_bits);

        int64_t block_sum = 0;
        for (int k = 0; k < kv_block_size; ++k)
        {
            if (scores_scratch[k] == MASKED_SCORE)
            {
                weights_scratch[k] = 0;
                continue;
            }

            // Delta in fixed-point (frac_bits precision)
            int64_t delta_64 = scaled_scores_fixed[k] - static_cast<int64_t>(m_new);

            // exp2(delta) via LUT
            uint32_t w;
            if (delta_64 >= 0)
            {
                w = one; // exp2(0) = 1
            }
            else if (delta_64 < -static_cast<int64_t>(32 << state.frac_bits))
            {
                w = 0; // Underflow
            }
            else
            {
                int64_t neg_delta = -delta_64;
                int int_part = static_cast<int>(neg_delta >> state.frac_bits);
                int frac_part = static_cast<int>(neg_delta & ((1 << state.frac_bits) - 1));

                uint32_t lut_val = lut[frac_part];
                w = (int_part < 32) ? (lut_val >> int_part) : 0;
            }

            weights_scratch[k] = static_cast<int32_t>(w);
            block_sum += w;
            if (scores_scratch[k] != MASKED_SCORE)
            {
                ++state.count;
            }
        }

        // Step 6: Update state
        state.m = m_new;
        state.l += block_sum;

        // Step 7: Compute weighted V sum and merge using running average
        const BlockType *V_block = V + kv_block_start * blocks_per_row;
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        double block_sum_scaled = 0.0;
        for (int k = 0; k < kv_block_size; ++k)
        {
            int32_t w_scaled = weights_scratch[k] >> (state.lut_value_bits - 15);
            block_sum_scaled += w_scaled;
        }

        alignas(64) int64_t block_context[256];
        std::memset(block_context, 0, head_dim * sizeof(int64_t));

        for (int k = 0; k < kv_block_size; ++k)
        {
            if (weights_scratch[k] == 0)
                continue;

            int32_t w_scaled = weights_scratch[k] >> (state.lut_value_bits - 15);
            if (w_scaled == 0 && weights_scratch[k] > 0)
            {
                w_scaled = 1;
            }

            const BlockType *V_row = V_block + k * blocks_per_row;

            for (int b = 0; b < blocks_per_row; ++b)
            {
                const int16_t *v_data = V_row[b].qs;
                int start = b * BLOCK_SIZE;
                int end = std::min(start + BLOCK_SIZE, head_dim);
                int count = end - start;

                for (int i = 0; i < count; ++i)
                {
                    block_context[start + i] += static_cast<int64_t>(w_scaled) * static_cast<int64_t>(v_data[i]);
                }
            }
        }

        // Step 8: Merge with running context
        double l_old = state.l_processed;
        double l_new = l_old + block_sum_scaled;

        if (l_new > 0.0)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                double numerator = static_cast<double>(context[d]) * l_old + static_cast<double>(block_context[d]);
                context[d] = static_cast<int32_t>(std::round(numerator / l_new));
            }
        }

        state.l_processed = l_new;
    }

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template void flash_decode_process_kv_block<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, const Q16_1Block_64 *,
        int32_t *, OnlineSoftmaxState &, int32_t *, int32_t *,
        int, int, int, int, float);

    template void flash_decode_process_kv_block<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, const Q16_1Block_128 *,
        int32_t *, OnlineSoftmaxState &, int32_t *, int32_t *,
        int, int, int, int, float);

    template void flash_decode_process_kv_block_with_k_scales<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, const Q16_1Block_64 *,
        int32_t *, OnlineSoftmaxState &, int32_t *, int32_t *,
        int, int, int, int, const float *);

    template void flash_decode_process_kv_block_with_k_scales<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, const Q16_1Block_128 *,
        int32_t *, OnlineSoftmaxState &, int32_t *, int32_t *,
        int, int, int, int, const float *);

    // ============================================================================
    // V2: Deferred Normalization Block Processing (PURE INTEGER)
    // ============================================================================

    template <typename BlockType>
    void flash_decode_process_kv_block_v2(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int64_t *context_accum,
        OnlineSoftmaxStateV2 &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_block_start,
        int kv_block_size,
        int head_dim,
        int blocks_per_row,
        float alpha)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        // ─────────────────────────────────────────────────────────────────
        // Step 1: Compute Q×K^T scores (pure integer)
        // ─────────────────────────────────────────────────────────────────
        const BlockType *K_block = K + kv_block_start * blocks_per_row;
        for (int k = 0; k < kv_block_size; ++k)
        {
            scores_scratch[k] = q16_dot_single<BlockType>(
                Q, K_block + k * blocks_per_row, head_dim, blocks_per_row);
        }

        // ─────────────────────────────────────────────────────────────────
        // Step 2: Online softmax - find max, check rescale
        // ─────────────────────────────────────────────────────────────────
        int32_t block_max = online_softmax_find_block_max(scores_scratch, kv_block_size);

        int32_t scale_num = 1;
        int scale_shift = 0;
        bool needs_rescale = false;

        if (block_max > state.m)
        {
            needs_rescale = (state.count > 0); // Only if we have prior accumulation
            if (needs_rescale)
            {
                online_softmax_compute_rescale_adaptive(
                    state.m, block_max, state.alpha_config,
                    scale_num, scale_shift, state.frac_bits, state.lut_value_bits);
            }
            state.m = block_max;
        }

        // ─────────────────────────────────────────────────────────────────
        // Step 3: Compute weights via exp2 LUT
        // ─────────────────────────────────────────────────────────────────
        for (int k = 0; k < kv_block_size; ++k)
        {
            int32_t delta = state.m - scores_scratch[k];
            weights_scratch[k] = exp2_compute_weight(delta, state.alpha_config,
                                                     state.frac_bits, state.lut_value_bits);
        }

        // ─────────────────────────────────────────────────────────────────
        // Step 4: Rescale prior accumulation if max changed (128-bit math)
        // ─────────────────────────────────────────────────────────────────
        if (needs_rescale)
        {
            state.sum_w_unscaled = rescale_int64_v2(state.sum_w_unscaled, scale_num, scale_shift);
            state.sum_w_scaled = rescale_int64_v2(state.sum_w_scaled, scale_num, scale_shift);
            for (int d = 0; d < head_dim; ++d)
            {
                context_accum[d] = rescale_int64_v2(context_accum[d], scale_num, scale_shift);
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Step 5: P×V accumulation with VNNI-safe chunking
        // ─────────────────────────────────────────────────────────────────
        const BlockType *V_block = V + kv_block_start * blocks_per_row;

        // Process in chunks to stay within INT32 accumulation safety
        alignas(64) int32_t chunk_accum[256]; // Max head_dim, stack-allocated
        int32_t chunk_sum_w_scaled = 0;

        for (int chunk_start = 0; chunk_start < kv_block_size; chunk_start += state.chunk_size)
        {
            int chunk_end = std::min(chunk_start + state.chunk_size, kv_block_size);

            // Zero chunk accumulator
            std::memset(chunk_accum, 0, head_dim * sizeof(int32_t));
            chunk_sum_w_scaled = 0;

            // Inner loop: VNNI-friendly INT32 accumulation
            for (int k = chunk_start; k < chunk_end; ++k)
            {
                int32_t w_raw = weights_scratch[k];
                int32_t w_scaled = w_raw >> state.weight_shift;

                // Track both sums
                state.sum_w_unscaled += w_raw;
                chunk_sum_w_scaled += w_scaled;

                if (w_scaled == 0)
                    continue;

                // Accumulate w_scaled × V[k]
                const BlockType *V_row = V_block + k * blocks_per_row;
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    const int16_t *v_data = V_row[b].qs;
                    int start = b * BLOCK_SIZE;
                    int end = std::min(start + BLOCK_SIZE, head_dim);
                    int count_d = end - start;

                    for (int i = 0; i < count_d; ++i)
                    {
                        chunk_accum[start + i] += w_scaled * static_cast<int32_t>(v_data[i]);
                    }
                }
            }

            // Dump chunk to INT64
            for (int d = 0; d < head_dim; ++d)
            {
                context_accum[d] += static_cast<int64_t>(chunk_accum[d]);
            }
            state.sum_w_scaled += static_cast<int64_t>(chunk_sum_w_scaled);
        }

        state.count += kv_block_size;
        // NO DIVISION HERE! Context remains unnormalized.
    }

    // V2 Finalization: Single division pass
    void flash_decode_finalize_v2(
        const int64_t *context_accum,
        int32_t *context_out,
        const OnlineSoftmaxStateV2 &state,
        int head_dim)
    {
        if (state.sum_w_scaled > 0)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                context_out[d] = static_cast<int32_t>(context_accum[d] / state.sum_w_scaled);
            }
        }
        else
        {
            std::memset(context_out, 0, head_dim * sizeof(int32_t));
        }
    }

    // V2 Template instantiations
    template void flash_decode_process_kv_block_v2<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, const Q16_1Block_64 *,
        int64_t *, OnlineSoftmaxStateV2 &, int32_t *, int32_t *,
        int, int, int, int, float);

    template void flash_decode_process_kv_block_v2<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, const Q16_1Block_128 *,
        int64_t *, OnlineSoftmaxStateV2 &, int32_t *, int32_t *,
        int, int, int, int, float);

    // ============================================================================
    // FA2 Prefill Tile Processing
    // ============================================================================

    template <typename BlockType>
    void fa2_prefill_process_kv_tile(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int32_t *context,
        OnlineSoftmaxStateBatch &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_tile_start,
        int Br,
        int Bc,
        int head_dim,
        int blocks_per_row,
        int q_stride,
        int k_stride,
        int context_stride,
        float alpha,
        bool causal,
        int q_offset)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        // Step 1: Compute Q×K^T scores for this tile [Br × Bc]
        // Use the tiled GEMM microkernel
        q16_qk_gemm_tile<BlockType>(
            Q, K + kv_tile_start * k_stride,
            scores_scratch,
            Br, Bc, head_dim, blocks_per_row,
            q_stride, k_stride);

        // Step 2: Apply causal mask if needed
        if (causal)
        {
            for (int r = 0; r < Br; ++r)
            {
                int q_pos = q_offset + r;
                for (int c = 0; c < Bc; ++c)
                {
                    int k_pos = kv_tile_start + c;
                    // Mask out future positions (k > q)
                    if (k_pos > q_pos)
                    {
                        scores_scratch[r * Bc + c] = MASKED_SCORE;
                    }
                }
            }
        }

        // Step 3: Per-row online softmax update with running average
        // Process each query row independently

        // Temp buffer for tile's weighted V sum - OUTSIDE row loop to avoid repeated stack alloc
        alignas(64) int64_t tile_context[256]; // Max head_dim

        for (int r = 0; r < Br; ++r)
        {
            int32_t *row_scores = scores_scratch + r * Bc;
            int32_t *row_weights = weights_scratch + r * Bc;

            // Find max in this row's scores (using shared primitive)
            int32_t row_max = exp2_find_block_max(row_scores, Bc, MASKED_SCORE);

            // Skip if all masked
            if (row_max == MASKED_SCORE)
            {
                std::memset(row_weights, 0, Bc * sizeof(int32_t));
                continue;
            }

            // Check if we need to rescale
            bool needs_rescale = false;
            int32_t m_old = state.m[r];
            int32_t m_new = state.m[r];

            if (state.empty(r) || row_max > state.m[r])
            {
                m_new = row_max;
                needs_rescale = !state.empty(r);
            }

            // Compute rescale factors if needed
            int32_t scale_num = 1 << state.lut_value_bits;
            int scale_shift = state.lut_value_bits;

            // Lazy init alpha_config if not already set
            if (state.alpha_config.original_alpha == 0.0f && alpha != 0.0f)
            {
                state.alpha_config = AdaptiveAlphaConfig::compute(alpha);
            }

            if (needs_rescale)
            {
                exp2_compute_rescale(
                    m_old, m_new, state.alpha_config,
                    scale_num, scale_shift,
                    state.frac_bits, state.lut_value_bits);

                // Rescale running sum l
                int64_t scaled_l = (static_cast<int64_t>(state.l[r]) * scale_num) >> scale_shift;
                state.l[r] = scaled_l;

                // Rescale l_processed using exact double arithmetic
                // NOTE: Context does NOT need rescaling - it's a normalized average
                double scale_factor = static_cast<double>(scale_num) / static_cast<double>(1ULL << scale_shift);
                state.l_processed[r] *= scale_factor;
            }

            // Compute weights for this row using shared primitive
            int64_t row_sum = exp2_compute_block_weights(
                row_scores, row_weights, Bc,
                m_new, state.alpha_config,
                state.frac_bits, state.lut_value_bits);

            // Update state for this row
            state.m[r] = m_new;
            state.l[r] += row_sum;

            // Count unmasked positions
            for (int c = 0; c < Bc; ++c)
            {
                if (row_scores[c] != MASKED_SCORE)
                {
                    ++state.count[r];
                }
            }

            // Step 4: Compute weighted V sum for this tile into temp buffer
            // Using RUNNING AVERAGE approach to keep context bounded
            int32_t *row_context = context + r * context_stride;
            const BlockType *V_tile = V + kv_tile_start * k_stride;

            // Compute tile sum of weights at scaled-down precision (double for accuracy)
            double tile_sum_scaled = 0.0;
            for (int c = 0; c < Bc; ++c)
            {
                int32_t w_scaled = row_weights[c] >> (state.lut_value_bits - 15);
                tile_sum_scaled += w_scaled;
            }

            // Zero the shared tile_context buffer for this row
            std::memset(tile_context, 0, head_dim * sizeof(int64_t));

            for (int k = 0; k < Bc; ++k)
            {
                if (row_weights[k] == 0)
                    continue;

                // Scale weight down to INT16 range
                int32_t w_scaled = row_weights[k] >> (state.lut_value_bits - 15);
                if (w_scaled == 0 && row_weights[k] > 0)
                {
                    w_scaled = 1;
                }

                const BlockType *V_row = V_tile + k * k_stride;

                // Accumulate w_scaled * V[k] into tile_context
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    const int16_t *v_data = V_row[b].qs;
                    int start = b * BLOCK_SIZE;
                    int end = std::min(start + BLOCK_SIZE, head_dim);
                    int count = end - start;

                    for (int i = 0; i < count; ++i)
                    {
                        tile_context[start + i] += static_cast<int64_t>(w_scaled) * static_cast<int64_t>(v_data[i]);
                    }
                }
            }

            // Step 5: Merge with running context using weighted average (double precision)
            double l_old = state.l_processed[r];
            double l_new = l_old + tile_sum_scaled;

            if (l_new > 0.0)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    double numerator = static_cast<double>(row_context[d]) * l_old + static_cast<double>(tile_context[d]);
                    row_context[d] = static_cast<int32_t>(std::round(numerator / l_new));
                }
            }

            // Update l_processed for this row (in scaled units)
            state.l_processed[r] = l_new;
        }
    }

    // ============================================================================
    // FA2 Prefill with Per-Position K Scales
    // ============================================================================

    template <typename BlockType>
    void fa2_prefill_process_kv_tile_with_k_scales(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int32_t *context,
        OnlineSoftmaxStateBatch &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_tile_start,
        int Br,
        int Bc,
        int head_dim,
        int blocks_per_row,
        int q_stride,
        int k_stride,
        int context_stride,
        float base_alpha_fp32,
        const float *k_scales,
        bool causal,
        int q_offset)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
        constexpr int beta_scale_bits = 24;
        // Maximum tile size - must match FA2TileConfig max values
        constexpr int MAX_BC = 128;

        // Step 1: Compute Q×K^T scores for this tile [Br × Bc] (PURE INTEGER)
        q16_qk_gemm_tile<BlockType>(
            Q, K + kv_tile_start * k_stride,
            scores_scratch,
            Br, Bc, head_dim, blocks_per_row,
            q_stride, k_stride);

        // Step 2: Apply causal mask if needed
        if (causal)
        {
            for (int r = 0; r < Br; ++r)
            {
                int q_pos = q_offset + r;
                for (int c = 0; c < Bc; ++c)
                {
                    int k_pos = kv_tile_start + c;
                    if (k_pos > q_pos)
                    {
                        scores_scratch[r * Bc + c] = MASKED_SCORE;
                    }
                }
            }
        }

        // Step 3: Pre-compute per-column AdaptiveAlphaConfig (like decode path)
        // alpha[c] = base_alpha_fp32 * k_scales[c]
        // Stack-allocated to avoid heap allocation in hot path
        alignas(64) AdaptiveAlphaConfig alpha_configs[MAX_BC];
        for (int c = 0; c < Bc; ++c)
        {
            float per_pos_alpha = base_alpha_fp32 * k_scales[c];
            alpha_configs[c] = AdaptiveAlphaConfig::compute(per_pos_alpha);
        }

        // Step 4: Per-row online softmax update
        // Stack-allocated buffers (outside row loop to avoid repeated stack alloc)
        alignas(64) int64_t scaled_scores_fixed[MAX_BC];
        alignas(64) int64_t tile_context[256]; // Max head_dim

        for (int r = 0; r < Br; ++r)
        {
            int32_t *row_scores = scores_scratch + r * Bc;
            int32_t *row_weights = weights_scratch + r * Bc;

            // Step 4a: Compute scaled scores in fixed-point domain (per-column alpha)
            // t_fixed[c] = score[c] * alpha[c] * log2(e) * 2^frac_bits
            int64_t row_max_fixed = std::numeric_limits<int64_t>::min();

            for (int c = 0; c < Bc; ++c)
            {
                if (row_scores[c] == MASKED_SCORE)
                {
                    scaled_scores_fixed[c] = std::numeric_limits<int64_t>::min();
                    continue;
                }

                const auto &cfg = alpha_configs[c];
                double effective_beta = static_cast<double>(cfg.effective_alpha) * LOG2E;
                int64_t M = static_cast<int64_t>(
                    std::llround(effective_beta * static_cast<double>(1ULL << beta_scale_bits)));

                int shift_for_t = beta_scale_bits - state.frac_bits + cfg.alpha_shift;

                int64_t prod = static_cast<int64_t>(row_scores[c]) * M;
                int64_t t_fixed = (shift_for_t >= 0)
                                      ? (prod >> shift_for_t)
                                      : (prod << (-shift_for_t));

                scaled_scores_fixed[c] = t_fixed;
                if (t_fixed > row_max_fixed)
                {
                    row_max_fixed = t_fixed;
                }
            }

            // Skip if all masked
            if (row_max_fixed == std::numeric_limits<int64_t>::min())
            {
                std::memset(row_weights, 0, Bc * sizeof(int32_t));
                continue;
            }

            // Clamp to int32_t range for state.m
            int32_t row_max = static_cast<int32_t>(
                std::clamp(row_max_fixed,
                           static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                           static_cast<int64_t>(std::numeric_limits<int32_t>::max())));

            // Step 4b: Check if max needs updating and compute rescale
            bool needs_rescale = false;
            int32_t m_old = state.m[r];
            int32_t m_new = state.m[r];

            if (state.empty(r) || row_max > state.m[r])
            {
                m_new = row_max;
                needs_rescale = !state.empty(r);
            }

            int32_t scale_num = 1 << state.lut_value_bits;
            int scale_shift = state.lut_value_bits;

            if (needs_rescale)
            {
                // Rescale factor: 2^((m_old - m_new) / 2^frac_bits)
                int64_t delta = static_cast<int64_t>(m_old) - static_cast<int64_t>(m_new);
                double delta_real = static_cast<double>(delta) / static_cast<double>(1ULL << state.frac_bits);
                double rescale_factor = std::exp2(delta_real);

                // Compute scale_num with proper clamping
                double scale_num_double = rescale_factor * static_cast<double>(1ULL << scale_shift);
                scale_num = static_cast<int32_t>(
                    std::clamp(scale_num_double, 1.0, static_cast<double>(1 << 30)));

                // Rescale running sum l
                int64_t scaled_l = (static_cast<int64_t>(state.l[r]) * scale_num) >> scale_shift;
                state.l[r] = scaled_l;

                // Rescale l_processed
                double scale_factor = static_cast<double>(scale_num) / static_cast<double>(1ULL << scale_shift);
                state.l_processed[r] *= scale_factor;
            }

            // Step 4c: Compute weights using per-position scaled scores
            // weight[c] = exp2(t_fixed[c] - m_new) via LUT
            ensure_exp2_lut_initialized(state.lut_value_bits);
            const uint32_t *lut = get_exp2_lut_data();
            const uint32_t one = static_cast<uint32_t>(1U << state.lut_value_bits);

            int64_t row_sum = 0;
            for (int c = 0; c < Bc; ++c)
            {
                if (scaled_scores_fixed[c] == std::numeric_limits<int64_t>::min())
                {
                    row_weights[c] = 0;
                    continue;
                }

                // Delta in fixed-point (frac_bits precision)
                int64_t delta_64 = scaled_scores_fixed[c] - static_cast<int64_t>(m_new);

                // exp2(delta) via LUT
                uint32_t w;
                if (delta_64 >= 0)
                {
                    w = one; // exp2(0) = 1
                }
                else if (delta_64 < -static_cast<int64_t>(32 << state.frac_bits))
                {
                    w = 0; // Underflow
                }
                else
                {
                    int64_t neg_delta = -delta_64;
                    int int_part = static_cast<int>(neg_delta >> state.frac_bits);
                    int frac_part = static_cast<int>(neg_delta & ((1 << state.frac_bits) - 1));

                    uint32_t lut_val = lut[frac_part];
                    w = (int_part < 32) ? (lut_val >> int_part) : 0;
                }

                row_weights[c] = static_cast<int32_t>(w);
                row_sum += w;
            }

            // Step 4d: Update state for this row
            state.m[r] = m_new;
            state.l[r] += row_sum;

            // Count unmasked positions
            for (int c = 0; c < Bc; ++c)
            {
                if (row_scores[c] != MASKED_SCORE)
                {
                    ++state.count[r];
                }
            }

            // Step 5: Compute weighted V sum for this tile
            int32_t *row_context = context + r * context_stride;
            const BlockType *V_tile = V + kv_tile_start * k_stride;

            // Compute tile sum of weights at scaled-down precision
            double tile_sum_scaled = 0.0;
            for (int c = 0; c < Bc; ++c)
            {
                int32_t w_scaled = row_weights[c] >> (state.lut_value_bits - 15);
                tile_sum_scaled += w_scaled;
            }

            // Zero the shared tile_context buffer for this row
            std::memset(tile_context, 0, head_dim * sizeof(int64_t));

            for (int k = 0; k < Bc; ++k)
            {
                if (row_weights[k] == 0)
                    continue;

                // Scale weight down to INT16 range
                int32_t w_scaled = row_weights[k] >> (state.lut_value_bits - 15);
                if (w_scaled == 0 && row_weights[k] > 0)
                {
                    w_scaled = 1;
                }

                const BlockType *V_row = V_tile + k * k_stride;

                // Accumulate w_scaled * V[k] into tile_context
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    const int16_t *v_data = V_row[b].qs;
                    int start = b * BLOCK_SIZE;
                    int end = std::min(start + BLOCK_SIZE, head_dim);
                    int count = end - start;

                    for (int i = 0; i < count; ++i)
                    {
                        tile_context[start + i] += static_cast<int64_t>(w_scaled) * static_cast<int64_t>(v_data[i]);
                    }
                }
            }

            // Step 6: Merge with running context using weighted average
            double l_old = state.l_processed[r];
            double l_new = l_old + tile_sum_scaled;

            if (l_new > 0.0)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    double numerator = static_cast<double>(row_context[d]) * l_old + static_cast<double>(tile_context[d]);
                    row_context[d] = static_cast<int32_t>(std::round(numerator / l_new));
                }
            }

            // Update l_processed for this row
            state.l_processed[r] = l_new;
        }
    }

    // ============================================================================
    // FA2 Prefill Template Instantiations
    // ============================================================================

    template void fa2_prefill_process_kv_tile<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, const Q16_1Block_64 *,
        int32_t *, OnlineSoftmaxStateBatch &, int32_t *, int32_t *,
        int, int, int, int, int, int, int, int, float, bool, int);

    template void fa2_prefill_process_kv_tile<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, const Q16_1Block_128 *,
        int32_t *, OnlineSoftmaxStateBatch &, int32_t *, int32_t *,
        int, int, int, int, int, int, int, int, float, bool, int);

    // FA2 Prefill with K scales template instantiations
    template void fa2_prefill_process_kv_tile_with_k_scales<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, const Q16_1Block_64 *,
        int32_t *, OnlineSoftmaxStateBatch &, int32_t *, int32_t *,
        int, int, int, int, int, int, int, int, float, const float *, bool, int);

    template void fa2_prefill_process_kv_tile_with_k_scales<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, const Q16_1Block_128 *,
        int32_t *, OnlineSoftmaxStateBatch &, int32_t *, int32_t *,
        int, int, int, int, int, int, int, int, float, const float *, bool, int);

} // namespace llaminar2::kernels::q16_1::microkernels
