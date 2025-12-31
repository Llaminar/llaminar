/**
 * @file Exp2Core.h
 * @brief Core exp2 LUT primitives for integer softmax implementations
 *
 * This file provides the shared foundation for all exp2-based softmax implementations:
 * - Exp2FixedSoftmax (standalone batch softmax)
 * - OnlineSoftmax (FlashAttention-style online softmax)
 * - Snapshot computation in Q16IntegerAttentionRef
 *
 * @section Design Rationale
 *
 * All Q16 attention softmax variants share the same core algorithm:
 *   weight = exp(-alpha * delta) where delta = max - score
 *
 * Using the identity exp(x) = 2^(x * log2(e)):
 *   weight = 2^(-alpha * delta * log2(e))
 *          = 2^(-t) where t = alpha * delta * log2(e)
 *
 * We decompose t = ip + frac where ip is the integer part:
 *   weight = 2^(-ip) * 2^(-frac)
 *          = (1 >> ip) * LUT[frac]
 *
 * This file provides:
 * 1. Thread-safe LUT initialization and access
 * 2. Fixed-point scaling configuration (AdaptiveAlphaConfig)
 * 3. Core weight computation primitives
 * 4. Rescale factor computation for online softmax
 *
 * @see Exp2FixedSoftmax.h for standalone softmax
 * @see OnlineSoftmax.h for FlashAttention-style processing
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Constants
    // ============================================================================

    /// LUT size: 2^11 = 2048 entries (8KB with uint32_t)
    constexpr int EXP2_LUT_SIZE = 2048;

    /// Default fractional bits for exp2 decomposition
    constexpr int DEFAULT_FRAC_BITS = 11;

    /// Default value bits for LUT entries (2^30 for ~30 bits of precision)
    constexpr int DEFAULT_LUT_VALUE_BITS = 30;

    /// log₂(e) ≈ 1.4427
    constexpr double LOG2E = 1.4426950408889634073599246810018921;

    /// Mask value for causal attention / padding
    constexpr int32_t MASKED_SCORE = std::numeric_limits<int32_t>::min();

    // ============================================================================
    // LUT Management (Thread-Safe Singleton)
    // ============================================================================

    /**
     * @brief Ensure the exp2 LUT is initialized for the given value bits.
     *
     * Thread-safe and idempotent. Call this before any LUT access.
     *
     * @param lut_value_bits Precision bits for LUT values (default 30)
     */
    void ensure_exp2_lut_initialized(int lut_value_bits = DEFAULT_LUT_VALUE_BITS);

    /**
     * @brief Get pointer to the initialized exp2 LUT.
     *
     * @return Pointer to 2048-entry uint32_t array, or nullptr if not initialized
     */
    const uint32_t *get_exp2_lut_data();

    // ============================================================================
    // Adaptive Alpha Configuration
    // ============================================================================

    /**
     * @brief Configuration for handling very small alpha values.
     *
     * When qk_scale is very small (e.g., 7.45e-9 for KV_CACHE_SCALE=8.0), direct
     * integer operations lose precision. This config scales up alpha to maintain
     * precision in the fixed-point computation.
     *
     * @section Algorithm
     *
     * Original: t = delta * alpha * log2(e)
     *
     * With scaling: effective_alpha = alpha * 2^alpha_shift
     *               t = delta * effective_alpha * log2(e) / 2^alpha_shift
     *
     * This keeps effective_alpha in a range where fixed-point multiplication
     * is precise, and we apply the shift when computing the final result.
     */
    struct AdaptiveAlphaConfig
    {
        float original_alpha = 0.0f; ///< Original qk_scale
        int64_t effective_alpha = 0; ///< Scaled up: original_alpha * 2^alpha_shift
        int alpha_shift = 0;         ///< Shift applied to get effective_alpha

        /**
         * @brief Compute adaptive config for a given alpha.
         *
         * Automatically chooses alpha_shift to keep effective_alpha
         * in a precise range for fixed-point math.
         */
        static AdaptiveAlphaConfig compute(float alpha);
    };

    // ============================================================================
    // Core Exp2 Weight Computation
    // ============================================================================

    /**
     * @brief Compute a single exp2 weight from delta.
     *
     * This is the fundamental primitive: computes 2^(-t) where
     * t = delta * alpha * log2(e).
     *
     * @param delta Score difference (max - score), must be >= 0
     * @param alpha_config Adaptive alpha configuration
     * @param frac_bits Fractional bits for decomposition (default 11)
     * @param lut_value_bits LUT precision bits (default 30)
     * @return Weight in LUT precision (0 to 2^lut_value_bits)
     */
    uint32_t exp2_compute_weight(
        int32_t delta,
        const AdaptiveAlphaConfig &alpha_config,
        int frac_bits = DEFAULT_FRAC_BITS,
        int lut_value_bits = DEFAULT_LUT_VALUE_BITS);

    /**
     * @brief Compute exp2 weight using raw alpha (non-adaptive).
     *
     * Convenience overload that computes AdaptiveAlphaConfig internally.
     * For repeated calls, prefer pre-computing the config.
     */
    uint32_t exp2_compute_weight(
        int32_t delta,
        float alpha,
        int frac_bits = DEFAULT_FRAC_BITS,
        int lut_value_bits = DEFAULT_LUT_VALUE_BITS);

    /**
     * @brief Compute rescale factor when running max increases.
     *
     * For online softmax, when max increases from m_old to m_new, all previous
     * weights must be rescaled by exp(m_old - m_new) = 2^(-delta * alpha * log2e).
     *
     * Returns numerator and shift such that: new_weight = (old_weight * num) >> shift
     *
     * @param m_old Previous running maximum
     * @param m_new New running maximum (must be >= m_old)
     * @param alpha_config Adaptive alpha configuration
     * @param[out] scale_num Numerator of scale factor
     * @param[out] scale_shift Right-shift to apply
     * @param frac_bits Fractional bits (default 11)
     * @param lut_value_bits LUT precision (default 30)
     */
    void exp2_compute_rescale(
        int32_t m_old,
        int32_t m_new,
        const AdaptiveAlphaConfig &alpha_config,
        int32_t &scale_num,
        int &scale_shift,
        int frac_bits = DEFAULT_FRAC_BITS,
        int lut_value_bits = DEFAULT_LUT_VALUE_BITS);

    // ============================================================================
    // Batch Weight Computation
    // ============================================================================

    /**
     * @brief Compute weights for a block of scores.
     *
     * This is the workhorse function used by both batch and online softmax.
     * Handles masking and underflow gracefully.
     *
     * @param scores Input scores [block_size]
     * @param weights Output weights [block_size] in LUT precision
     * @param block_size Number of elements
     * @param running_max Maximum score (for computing deltas)
     * @param alpha_config Adaptive alpha configuration
     * @param frac_bits Fractional bits (default 11)
     * @param lut_value_bits LUT precision (default 30)
     * @return Sum of all weights (for normalization)
     */
    int64_t exp2_compute_block_weights(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        const AdaptiveAlphaConfig &alpha_config,
        int frac_bits = DEFAULT_FRAC_BITS,
        int lut_value_bits = DEFAULT_LUT_VALUE_BITS);

    /**
     * @brief Convenience overload with raw alpha.
     */
    int64_t exp2_compute_block_weights(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        float alpha,
        int frac_bits = DEFAULT_FRAC_BITS,
        int lut_value_bits = DEFAULT_LUT_VALUE_BITS);

    // ============================================================================
    // Utility Functions
    // ============================================================================

    /**
     * @brief Find maximum score in a block, ignoring masked values.
     *
     * @param scores Input scores [block_size]
     * @param block_size Number of elements
     * @param mask_value Value indicating masked positions (default MASKED_SCORE)
     * @return Maximum score, or mask_value if all masked
     */
    int32_t exp2_find_block_max(
        const int32_t *scores,
        int block_size,
        int32_t mask_value = MASKED_SCORE);

    /**
     * @brief Normalize weights to INT16 range.
     *
     * Converts LUT-precision weights to INT16 weights that sum to ~weight_max.
     *
     * @param block_weights Input weights in LUT precision [n]
     * @param final_weights Output normalized INT16 weights [n]
     * @param n Number of elements
     * @param weight_sum Sum of block_weights
     * @param weight_max Target maximum weight (default 32767)
     * @return Actual sum of final_weights
     */
    int32_t exp2_normalize_weights(
        const int32_t *block_weights,
        int16_t *final_weights,
        int n,
        int64_t weight_sum,
        int16_t weight_max = 32767);

} // namespace llaminar2::kernels::q16_1::microkernels
