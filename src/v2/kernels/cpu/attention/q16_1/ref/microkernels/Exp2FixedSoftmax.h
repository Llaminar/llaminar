/**
 * @file Exp2FixedSoftmax.h
 * @brief Integer-only softmax via exp2 LUT approximation (v2)
 *
 * ALGORITHM
 * =========
 * Standard softmax: w_i = exp(s_i) / Σ exp(s_j)
 *
 * For numerical stability, subtract max: w_i = exp(s_i - max) / Σ exp(s_j - max)
 *
 * Let δ_i = max - s_i ≥ 0, then: w_i = exp(-δ_i) / Σ exp(-δ_j)
 *
 * Key identity: exp(-x) = 2^(-x × log₂(e))
 *
 * Let t = δ × α × log₂(e), where α is the score scale (e.g., 1/√d).
 * Then: exp(-α×δ) = 2^(-t)
 *
 * Decompose t = ip + frac where ip is integer part, frac ∈ [0,1):
 *   2^(-t) = 2^(-ip) × 2^(-frac)
 *
 * - 2^(-ip) is just a right-shift by ip bits
 * - 2^(-frac) is looked up in a 256-entry LUT indexed by frac×256
 *
 * OUTPUT
 * ======
 * INT16 weights in [0, 32767] suitable for VPDPWSSD accumulation.
 * Weights sum to approximately 32767 (normalized).
 *
 * PRECISION
 * =========
 * - 11 fractional bits → 2048 LUT entries → ~0.05% max relative error on 2^(-frac)
 * - 30-bit internal precision → robust normalization for sequences up to 128K
 * - INT16 output → 15-bit effective precision, sufficient for attention
 * - 4KB LUT fits easily in L1 cache (typically 32-64KB per core)
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 * @see Exp2Core.h for shared LUT primitives
 */
#pragma once

#include "Exp2Core.h" // Shared LUT primitives and constants

#include <cstdint>

namespace llaminar2::kernels::q16_1::microkernels
{

    /**
     * @brief Configuration for exp2 fixed-point softmax.
     *
     * Default values are tuned for attention with typical score ranges.
     * Generally should not need modification.
     */
    struct Exp2SoftmaxConfig
    {
        /// Fractional bits for LUT indexing. 11 = 2048 entries (4KB), excellent precision.
        /// 8× finer granularity than 256-entry LUT, critical for attention distribution quality.
        int frac_bits = DEFAULT_FRAC_BITS;

        /// Internal scaling bits for fixed-point β computation.
        /// Higher = more precision but risk of overflow for large deltas.
        int beta_scale_bits = 24;

        /// LUT value precision bits. 30 gives good normalization headroom.
        int lut_value_bits = DEFAULT_LUT_VALUE_BITS;

        /// Maximum output weight (INT16 range for VNNI compatibility).
        int16_t weight_max = 32767;
    };

    /**
     * @brief Compute softmax over INT32 scores, output INT16 weights.
     *
     * This is the core softmax microkernel for integer attention.
     * NO floating-point operations in the hot path.
     *
     * @param scores   Input INT32 scores [n]. INT32_MIN indicates masked position.
     * @param weights  Output INT16 weights [n] in [0, weight_max].
     * @param n        Number of positions.
     * @param alpha    Score scaling factor in FP (only used to compute fixed-point β).
     *                 For attention: α = s_Q × s_K / √head_dim
     * @param sum_out  Optional output: sum of weights (for debugging/verification).
     * @param config   Algorithm configuration (usually default is fine).
     *
     * @note The only FP operation is computing β = α × log₂(e) once at the start.
     *       All per-element operations are pure integer.
     */
    void exp2_softmax_int32(
        const int32_t *scores,
        int16_t *weights,
        int n,
        float alpha,
        int32_t *sum_out = nullptr,
        const Exp2SoftmaxConfig &config = Exp2SoftmaxConfig{});

    // Note: ensure_exp2_lut_initialized() and get_exp2_lut_data() are now
    // provided by Exp2Core.h for all softmax implementations to share.
    // EXP2_LUT_SIZE constant is also in Exp2Core.h.

} // namespace llaminar2::kernels::q16_1::microkernels
