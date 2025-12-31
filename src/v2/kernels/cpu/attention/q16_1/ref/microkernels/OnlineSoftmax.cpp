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

            // Temp buffer for tile's weighted V sum (per row)
            alignas(64) int64_t tile_context[256]; // Max head_dim
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

} // namespace llaminar2::kernels::q16_1::microkernels
