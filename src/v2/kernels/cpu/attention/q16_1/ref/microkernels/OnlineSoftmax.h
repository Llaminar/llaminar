/**
 * @file OnlineSoftmax.h
 * @brief Online softmax microkernel for FlashDecode/FlashAttention
 *
 * ALGORITHM (Online Softmax)
 * ==========================
 * Standard softmax requires two passes: find max, then compute exp.
 * Online softmax computes incrementally in a single streaming pass:
 *
 *   Initialize: m = -∞, l = 0, O = 0
 *
 *   For each block of KV positions:
 *     1. Compute scores for block: s[k] = Q · K[k]
 *     2. Find local max: m_block = max(s[k])
 *     3. Update running max: m_new = max(m, m_block)
 *     4. Rescale previous state if max changed:
 *        correction = exp(m - m_new)   // In integer: 2^(-(m_new - m) * scale)
 *        l = l * correction
 *        O = O * correction
 *     5. Accumulate this block:
 *        for k in block:
 *          w = exp(s[k] - m_new)       // In integer: exp2 LUT
 *          l += w
 *          O += w * V[k]
 *     6. m = m_new
 *
 *   Finalize: output = O / l
 *
 * WHY THIS MATTERS
 * ================
 * - Memory efficiency: Never stores full [kv_len] scores array
 * - Cache friendly: Streams through KV cache in blocks
 * - Numerically stable: Running max prevents overflow
 * - JIT-friendly: Clean block-at-a-time pattern
 *
 * INTEGER DOMAIN
 * ==============
 * - m (max score) stored as INT32
 * - l (sum of weights) stored as INT64 for precision
 * - O (context) stored as INT32[head_dim]
 * - Rescaling uses exp2 LUT + integer shifts
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */
#pragma once

#include <cstdint>
#include <limits>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Online Softmax State
    // ============================================================================

    /**
     * @brief Running state for online softmax computation.
     *
     * Maintains the incremental softmax state across KV blocks.
     * All values are in integer domain for JIT-friendly computation.
     */
    struct OnlineSoftmaxState
    {
        /// Running maximum score (INT32)
        int32_t m = std::numeric_limits<int32_t>::min();

        /// Running sum of unnormalized weights (INT64 for precision)
        /// This is Σ exp2_lut(s[k] - m) before final normalization
        int64_t l = 0;

        /// Number of unmasked positions processed so far
        int count = 0;

        /// Configuration: fractional bits for exp2 computation
        int frac_bits = 11;

        /// Configuration: LUT value bits for exp2
        int lut_value_bits = 30;

        /// Reset to initial state
        void reset()
        {
            m = std::numeric_limits<int32_t>::min();
            l = 0;
            count = 0;
        }

        /// Check if any positions have been processed
        bool empty() const { return count == 0; }
    };

    // ============================================================================
    // Block-wise Online Softmax Operations
    // ============================================================================

    /**
     * @brief Find maximum score in a block (for online max update).
     *
     * @param scores INT32 scores for this block [block_size]
     * @param block_size Number of scores in block
     * @param mask_value Value indicating masked position (INT32_MIN)
     * @return Maximum unmasked score, or INT32_MIN if all masked
     */
    int32_t online_softmax_find_block_max(
        const int32_t *scores,
        int block_size,
        int32_t mask_value = std::numeric_limits<int32_t>::min());

    /**
     * @brief Compute exp2 rescaling factor for max update.
     *
     * When running max changes from m_old to m_new, previous state must be
     * rescaled by exp(m_old - m_new) = 2^((m_old - m_new) * beta).
     *
     * Returns integer representation: (scale_num, scale_shift) such that
     *   rescale_factor ≈ scale_num >> scale_shift
     *
     * @param m_old Previous running max (INT32)
     * @param m_new New running max (INT32)
     * @param alpha Score scaling factor (qk_scale)
     * @param scale_num Output: numerator for rescaling
     * @param scale_shift Output: right-shift for rescaling
     * @param config Exp2 configuration
     */
    void online_softmax_compute_rescale(
        int32_t m_old,
        int32_t m_new,
        float alpha,
        int32_t &scale_num,
        int &scale_shift,
        int frac_bits = 11,
        int lut_value_bits = 30);

    /**
     * @brief Compute unnormalized exp2 weights for a block of scores.
     *
     * For each score s[k], computes w[k] = exp2_lut(s[k] - m) where m is
     * the running max. Returns sum of weights for this block.
     *
     * @param scores INT32 scores [block_size]
     * @param weights Output INT32 unnormalized weights [block_size]
     * @param block_size Number of scores
     * @param running_max Current running max (state.m)
     * @param alpha Score scaling factor
     * @param frac_bits Fractional bits for exp2
     * @param lut_value_bits LUT value precision
     * @return Sum of weights for this block (INT64)
     */
    int64_t online_softmax_compute_block_weights(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        float alpha,
        int frac_bits = 11,
        int lut_value_bits = 30);

    /**
     * @brief Update online softmax state with a new block of scores.
     *
     * This is the main online softmax update function:
     * 1. Find block max
     * 2. Update running max
     * 3. Compute rescale factor if max changed
     * 4. Compute weights for this block
     * 5. Update running sum
     *
     * @param state Online softmax state (modified in place)
     * @param scores INT32 scores for this block [block_size]
     * @param weights Output INT32 unnormalized weights [block_size]
     * @param block_size Number of scores in block
     * @param alpha Score scaling factor
     * @param scale_num Output: rescale numerator (1.0 if no rescale needed)
     * @param scale_shift Output: rescale shift (0 if no rescale needed)
     * @return true if rescaling is needed (max changed)
     */
    bool online_softmax_update_block(
        OnlineSoftmaxState &state,
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        float alpha,
        int32_t &scale_num,
        int &scale_shift);

    /**
     * @brief Finalize weights: normalize to INT16 range [0, 32767].
     *
     * After processing all blocks, unnormalized weights in [block_weights]
     * are scaled by weight_max / state.l to produce final INT16 weights.
     *
     * @param state Final online softmax state
     * @param block_weights Unnormalized INT32 weights for all blocks
     * @param final_weights Output INT16 normalized weights
     * @param total_len Total number of KV positions
     * @param weight_max Maximum output weight (32767)
     * @return Sum of final weights (should be close to weight_max * count)
     */
    int32_t online_softmax_finalize_weights(
        const OnlineSoftmaxState &state,
        const int32_t *block_weights,
        int16_t *final_weights,
        int total_len,
        int16_t weight_max = 32767);

    // ============================================================================
    // FlashDecode-Specific Operations
    // ============================================================================

    /**
     * @brief Process one KV block in FlashDecode with online softmax.
     *
     * High-level function that combines:
     * 1. Q×K dot products for block
     * 2. Online softmax state update
     * 3. Context rescaling if needed
     * 4. P×V accumulation for block
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query blocks [blocks_per_row]
     * @param K Key blocks for this block [block_size, blocks_per_row]
     * @param V Value blocks for this block [block_size, blocks_per_row]
     * @param context Running INT32 context [head_dim] (modified in place)
     * @param state Online softmax state (modified in place)
     * @param scores_scratch INT32 scratch for scores [block_size]
     * @param weights_scratch INT32 scratch for unnorm weights [block_size]
     * @param kv_block_start Starting KV position for this block
     * @param kv_block_size Number of positions in this block
     * @param head_dim Head dimension
     * @param blocks_per_row Blocks per head
     * @param alpha QK scale factor
     */
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
        float alpha);

    // ============================================================================
    // Constants
    // ============================================================================

    /// Default KV block size for FlashDecode
    /// 32 is a good balance: small enough for cache, large enough for SIMD
    constexpr int FLASH_DECODE_KV_BLOCK_SIZE = 32;

} // namespace llaminar2::kernels::q16_1::microkernels
