/**
 * @file OnlineSoftmax.cpp
 * @brief Online softmax microkernel implementation for FlashDecode
 *
 * @see OnlineSoftmax.h for algorithm description
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */

#include "OnlineSoftmax.h"
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
    // Constants (shared with Exp2FixedSoftmax)
    // ============================================================================

    namespace
    {
        /// log₂(e) ≈ 1.4427
        constexpr double LOG2E = 1.4426950408889634073599246810018921;

        /// Mask value for causal attention / padding
        constexpr int32_t MASKED = std::numeric_limits<int32_t>::min();

    } // namespace

    // ============================================================================
    // Block-wise Online Softmax Operations
    // ============================================================================

    int32_t online_softmax_find_block_max(
        const int32_t *scores,
        int block_size,
        int32_t mask_value)
    {
        int32_t block_max = mask_value;

        for (int i = 0; i < block_size; ++i)
        {
            if (scores[i] != mask_value && scores[i] > block_max)
            {
                block_max = scores[i];
            }
        }

        return block_max;
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
        // m_new >= m_old always (running max is non-decreasing)
        // We need: correction = exp(m_old - m_new) = 2^((m_old - m_new) * alpha * log2e)
        //        = 2^(-delta * beta) where delta = m_new - m_old >= 0

        if (m_old == MASKED || m_new <= m_old)
        {
            // No rescaling needed
            scale_num = 1 << lut_value_bits;
            scale_shift = lut_value_bits;
            return;
        }

        int32_t delta = m_new - m_old;
        double beta = static_cast<double>(alpha) * LOG2E;

        // t = delta * beta (how many powers of 2 to scale down)
        double t = static_cast<double>(delta) * beta;

        // Decompose t = ip + frac
        int ip = static_cast<int>(t);
        double frac = t - ip;

        // If ip >= 31, underflow to zero
        if (ip >= 31)
        {
            scale_num = 0;
            scale_shift = 0;
            return;
        }

        // Get exp2 LUT for fractional part
        ensure_exp2_lut_initialized(lut_value_bits);
        const uint32_t *lut = get_exp2_lut_data();

        // Index into LUT: frac * 2^frac_bits
        int lut_idx = static_cast<int>(frac * (1 << frac_bits));
        lut_idx = std::clamp(lut_idx, 0, EXP2_LUT_SIZE - 1);

        // 2^(-t) = 2^(-ip) * 2^(-frac) = lut[idx] >> ip
        scale_num = static_cast<int32_t>(lut[lut_idx]);
        scale_shift = lut_value_bits + ip;
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
        // Initialize LUT
        ensure_exp2_lut_initialized(lut_value_bits);
        const uint32_t *lut = get_exp2_lut_data();

        // Compute beta = alpha * log2(e)
        double beta = static_cast<double>(alpha) * LOG2E;
        int beta_scale_bits = 24;
        int64_t M = static_cast<int64_t>(
            std::llround(beta * static_cast<double>(1ULL << beta_scale_bits)));

        int shift_for_t = beta_scale_bits - frac_bits;
        uint32_t one = static_cast<uint32_t>(1U << lut_value_bits);

        int64_t sum = 0;

        for (int i = 0; i < block_size; ++i)
        {
            if (scores[i] == MASKED)
            {
                weights[i] = 0;
                continue;
            }

            int32_t delta = running_max - scores[i];

            // delta = 0 means this is the max
            if (delta <= 0)
            {
                weights[i] = static_cast<int32_t>(one);
                sum += one;
                continue;
            }

            // t_fixed = delta * M in Q(frac_bits) format
            int64_t prod = static_cast<int64_t>(delta) * M;
            int64_t t_fixed = (shift_for_t >= 0)
                                  ? (prod >> shift_for_t)
                                  : (prod << (-shift_for_t));

            // Decompose: t = ip + frac
            int64_t ip = t_fixed >> frac_bits;
            int frac_idx = static_cast<int>(t_fixed & ((1 << frac_bits) - 1));

            // Underflow check
            if (ip >= 31)
            {
                weights[i] = 0;
                continue;
            }

            // 2^(-t) = lut[frac] >> ip
            uint32_t w = lut[frac_idx] >> static_cast<int>(ip);
            weights[i] = static_cast<int32_t>(w);
            sum += w;
        }

        return sum;
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
        // Step 1: Find block max
        int32_t block_max = online_softmax_find_block_max(scores, block_size, MASKED);

        // All masked in this block
        if (block_max == MASKED)
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

        // Step 3: Compute rescale factor
        if (needs_rescale)
        {
            online_softmax_compute_rescale(
                m_old, m_new, alpha,
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

        // Step 4: Compute weights for this block (using new max)
        int64_t block_sum = online_softmax_compute_block_weights(
            scores, weights, block_size,
            m_new, alpha,
            state.frac_bits, state.lut_value_bits);

        // Step 5: Update state
        state.m = m_new;
        state.l += block_sum;

        // Count unmasked positions
        for (int i = 0; i < block_size; ++i)
        {
            if (scores[i] != MASKED)
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
        if (state.l == 0)
        {
            // No unmasked positions - uniform or zero
            std::memset(final_weights, 0, total_len * sizeof(int16_t));
            return 0;
        }

        // Normalize: w_final = block_weights * weight_max / l
        int64_t half = state.l / 2;
        int64_t sum = 0;

        for (int i = 0; i < total_len; ++i)
        {
            if (block_weights[i] == 0)
            {
                final_weights[i] = 0;
                continue;
            }

            int64_t num = static_cast<int64_t>(block_weights[i]) *
                          static_cast<int64_t>(weight_max);
            int64_t w = (num + half) / state.l;

            w = std::min<int64_t>(w, weight_max);
            final_weights[i] = static_cast<int16_t>(w);
            sum += w;
        }

        return static_cast<int32_t>(std::min<int64_t>(sum, std::numeric_limits<int32_t>::max()));
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

        // Step 3: Rescale existing context if max changed
        if (needs_rescale)
        {
            q16_context_rescale(context, head_dim, scale_num, scale_shift);
        }

        // Step 4: Accumulate P×V for this block
        // We use unnormalized INT32 weights here, final normalization happens at the end
        // Convert INT32 weights to INT16 for accumulation (scale down by lut_value_bits)
        // For now, we'll accumulate with scaled-down weights
        const BlockType *V_block = V + kv_block_start * blocks_per_row;

        for (int k = 0; k < kv_block_size; ++k)
        {
            if (weights_scratch[k] == 0)
                continue;

            // Scale weight from LUT precision to usable range
            // LUT values are in [0, 2^30], scale to [0, 2^15] for INT16 range
            int32_t w_scaled = weights_scratch[k] >> (state.lut_value_bits - 15);
            if (w_scaled == 0 && weights_scratch[k] > 0)
            {
                w_scaled = 1; // Preserve non-zero contribution
            }

            const BlockType *V_row = V_block + k * blocks_per_row;

            // Accumulate w * V[k] into context
            constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
            for (int b = 0; b < blocks_per_row; ++b)
            {
                const int16_t *v_data = V_row[b].qs;
                int start = b * BLOCK_SIZE;
                int end = std::min(start + BLOCK_SIZE, head_dim);
                int count = end - start;

                for (int i = 0; i < count; ++i)
                {
                    context[start + i] += w_scaled * static_cast<int32_t>(v_data[i]);
                }
            }
        }
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

} // namespace llaminar2::kernels::q16_1::microkernels
