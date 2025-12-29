/**
 * @file WoProjectionVNNIRef.h
 * @brief Reference microkernel for Wo projection using INT16 VNNI (VPDPWSSD)
 *
 * This microkernel bridges Q16 attention output to Wo projection while
 * staying ENTIRELY in the integer domain with maximum precision.
 *
 * KEY INSIGHT: INT8 context quantization loses too much precision!
 * ═══════════════════════════════════════════════════════════════════════════
 * Instead of: INT32 context → INT8 (Q8_1) → VPDPBUSD (loses 8 bits!)
 * We use:     INT32 context → INT16 → VPDPWSSD (keeps 16 bits!)
 *
 * Model weights arrive as INT8 VNNI-packed, but we SIGN-EXTEND to INT16
 * at runtime. This is essentially free (just load with sign extension).
 *
 * DATA FLOW (ALL INTEGER - MAXIMUM PRECISION):
 * ══════════════════════════════════════════════════════════════════════════════
 *
 *   P×V Output (INT16×INT16→INT32 accumulators, NOT normalized to FP32)
 *       │
 *       │  int32_context[num_heads × head_dim]
 *       │  + weight_sum (INT32, sum of INT16 softmax weights)
 *       │  + v_scale_product (FP32, from V's Q16_1 blocks)
 *       │
 *       ▼
 *   requantize_int32_to_int16() ─── Integer requantization (KEEP 16 bits!)
 *       │
 *       │  Computes: maxabs of INT32 values
 *       │  Scale: combined_scale = maxabs/32767 * v_scale_product / weight_sum
 *       │  Output: INT16 packed context with deferred scale
 *       │
 *       ▼
 *   INT16 context_packed[inner_dim] (16-bit precision preserved!)
 *       │
 *       └───────────────┬────────────────┐
 *                       │                │
 *                       ▼                ▼
 *               Wo_int8[K][N] ──► sign_extend ──► Wo_int16[K][N]
 *                       │
 *                       ▼
 *               VPDPWSSD (INT16 × INT16 → INT32) ← Maximum precision!
 *                       │
 *                       ▼
 *               INT32 GEMM output[d_model]
 *                       │
 *                       ▼
 *               requantize_int32_to_q16_1() ─── Output requantization
 *                       │
 *                       ▼
 *               Q16_1 output[d_model / 32 blocks]
 *                       │
 *                       ▼
 *               q16_1_add_q16_1() ─── Native Q16_1 residual add!
 *
 * ══════════════════════════════════════════════════════════════════════════════
 *
 * WHY VPDPWSSD + SIGN-EXTENDED WEIGHTS:
 * ═════════════════════════════════════
 * - Context: INT32→INT16 keeps 16 bits (vs INT32→INT8 losing 8 bits!)
 * - Weights: INT8→INT16 sign-extend is FREE (vpmovsxbw instruction)
 * - VPDPWSSD: Native INT16×INT16→INT32, same throughput as VPDPBUSD
 * - Net result: 2× context precision at zero computational cost
 *
 * VNNI INSTRUCTION COMPARISON:
 * ════════════════════════════
 * VPDPBUSD: UINT8 × INT8 → INT32  (our weights, but context loses precision)
 * VPDPWSSD: INT16 × INT16 → INT32 (context keeps precision, weights upscaled)
 *
 * @see PROJECT_Q16_INTEGER_ATTENTION.md for overall design
 */
#pragma once

#include "tensors/BlockStructures.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmJit_M1.h"
#include <cstdint>
#include <vector>

namespace llaminar2::kernels::q16_1::microkernels
{

    // Forward declarations
    using Q8_1Block = llaminar2::Q8_1Block;
    using Q16_1Block = llaminar2::Q16_1Block;
    using QuantisedPackedWeights = llaminar2::gemm_v4::QuantisedPackedWeights;

    // ============================================================================
    // Parameters
    // ============================================================================

    /**
     * @brief Parameters for Wo projection with VPDPWSSD (INT16×INT16)
     */
    struct WoProjectionVNNIParams
    {
        /** VNNI-packed Wo weights (INT8, will be sign-extended to INT16) */
        const QuantisedPackedWeights *Wo_packed = nullptr;

        /** Input dimension = num_heads * head_dim */
        int input_dim = 0;

        /** Output dimension = d_model */
        int d_model = 0;

        /** Optional bias vector [d_model] (applied during Q16_1 requantization) */
        const float *bias = nullptr;
    };

    /**
     * @brief Integer context from P×V accumulation (before Wo projection)
     *
     * This struct holds the INT32 accumulators from P×V along with the
     * scale information needed for requantization.
     */
    struct IntegerContext
    {
        /** INT32 P×V accumulators [num_heads × head_dim] */
        const int32_t *int32_data = nullptr;

        /** Per-head sum of INT16 softmax weights (for normalization) [num_heads] */
        const int32_t *weight_sums = nullptr;

        /** Per-element effective V scales used in accumulation [count] */
        const float *v_scales = nullptr;
        int num_heads = 0;
        int head_dim = 0;

        /** Number of elements */
        int count = 0;
    };

    /**
     * @brief INT16 packed context for VPDPWSSD GEMM
     *
     * The context is requantized to INT16 (not INT8!) to preserve precision.
     * This is then multiplied with sign-extended INT16 weights via VPDPWSSD.
     */
    struct Int16PackedContext
    {
        /** INT16 packed context values [count] */
        int16_t *int16_data = nullptr;

        /** Combined scale factor for this context */
        float scale = 1.0f;

        /** Number of elements */
        int count = 0;
    };

    /**
     * @brief Q16_1 output from Wo projection (ready for residual add)
     *
     * The output is directly in Q16_1 format, matching the residual format.
     * This enables native Q16_1 + Q16_1 addition with no FP32 conversion.
     */
    struct Q16_1Projection
    {
        /** Q16_1 output blocks [(d_model + 31) / 32] */
        Q16_1Block *blocks = nullptr;

        /** Number of elements (d_model) */
        int count = 0;
    };

    // ============================================================================
    // Context Requantization (INT32 → INT16 for VPDPWSSD)
    // ============================================================================

    /**
     * @brief Requantize INT32 P×V accumulators to INT16 for VPDPWSSD
     *
     * This function converts INT32 accumulators to INT16 format, preserving
     * 16 bits of precision (vs only 8 bits if we used Q8_1).
     *
     * The scale factor accounts for:
     * - Softmax normalization (1/weight_sum)
     * - V block scales (from Q16_1 V tensor)
     *
     * @param int32_input INT32 P×V accumulators [n]
     * @param n Number of elements
     * @param weight_sum Sum of INT16 softmax weights
     * @param v_scale_product Product of V block scales
     * @param int16_output INT16 output values [n]
     * @param out_combined_scale Combined scale factor for later dequantization
     */
    void requantize_int32_to_int16_context(
        const int32_t *int32_input,
        int n,
        const int32_t *weight_sums,
        const float *v_scales,
        int head_dim,
        int16_t *int16_output,
        float *out_combined_scale);

    // ============================================================================
    // Output Requantization (INT32 → Q16_1)
    // ============================================================================

    /**
     * @brief Requantize INT32 GEMM output to Q16_1 blocks (integer domain)
     *
     * This function converts the INT32 output from Wo GEMM to Q16_1 format,
     * incorporating the accumulated scale factors from the computation.
     *
     * @param int32_output INT32 GEMM output [n]
     * @param n Number of elements (d_model)
     * @param combined_scale Combined scale factor from GEMM (ctx_scale * w_scale)
     * @param bias Optional bias to add (scaled appropriately)
     * @param output Q16_1 output blocks [(n + 31) / 32]
     */
    void requantize_int32_to_q16_1(
        const int32_t *int32_output,
        int n,
        float combined_scale,
        const float *bias,
        Q16_1Block *output);

    // ============================================================================
    // GEMV Variant (Decode - single query) → Q16_1 output using VPDPWSSD
    // ============================================================================

    /**
     * @brief Project INT16 context through Wo using VPDPWSSD (INT16×INT16)
     *
     * Computes: output_q16_1[d_model] = context_int16 × Wo_int16
     *
     * Steps:
     * 1. Requantize INT32 context to INT16 (keeps 16 bits of precision!)
     * 2. Sign-extend INT8 Wo weights to INT16 (lossless)
     * 3. Execute VPDPWSSD (INT16 × INT16 → INT32)
     * 4. Requantize INT32 → Q16_1
     *
     * Output is ready for native Q16_1 + Q16_1 residual add!
     *
     * @param params Projection parameters (includes packed Wo weights)
     * @param context Integer context from P×V (INT32 + scale info)
     * @param output Q16_1 projection result (ready for residual add)
     */
    void wo_projection_vpdpwssd_to_q16_1_gemv(
        const WoProjectionVNNIParams &params,
        const IntegerContext &context,
        Q16_1Projection &output);

    // ============================================================================
    // GEMM Variant (Prefill - multiple queries) → Q16_1 output using VPDPWSSD
    // ============================================================================

    /**
     * @brief Project multiple INT16 contexts through Wo using VPDPWSSD
     *
     * @param params Projection parameters (includes packed Wo weights)
     * @param contexts Array of integer contexts [seq_len]
     * @param seq_len Number of query positions
     * @param outputs Array of Q16_1 projections [seq_len]
     */
    void wo_projection_vpdpwssd_to_q16_1_gemm(
        const WoProjectionVNNIParams &params,
        const IntegerContext *contexts,
        int seq_len,
        Q16_1Projection *outputs);

    // ============================================================================
    // Utility Functions
    // ============================================================================

    /**
     * @brief Find maximum absolute value in INT32 array
     *
     * @param data INT32 array
     * @param n Number of elements
     * @return Maximum absolute value (as INT64 to avoid overflow)
     */
    int64_t int32_maxabs(const int32_t *data, int n);

    /**
     * @brief Sign-extend INT8 weights to INT16 for VPDPWSSD
     *
     * This is the key operation that enables using VPDPWSSD with our INT8 weights.
     * Sign extension is lossless and very cheap (vpmovsxbw instruction).
     *
     * @param int8_weights INT8 weight values [n]
     * @param n Number of elements
     * @param int16_weights INT16 output values [n]
     */
    void sign_extend_int8_to_int16(
        const int8_t *int8_weights,
        int n,
        int16_t *int16_weights);

    /**
     * @brief Quantize INT32 values to INT16 with given scale
     *
     * @param int32_input INT32 input values [n]
     * @param n Number of elements
     * @param scale Scale factor (maxabs / 32767)
     * @param int16_output INT16 output values [n]
     * @return Sum of INT16 values
     */
    int32_t quantize_int32_to_int16(
        const int32_t *int32_input,
        int n,
        float scale,
        int16_t *int16_output);

} // namespace llaminar2::kernels::q16_1::microkernels
