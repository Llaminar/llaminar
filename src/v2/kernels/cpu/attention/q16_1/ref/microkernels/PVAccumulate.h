/**
 * @file PVAccumulate.h
 * @brief Integer P×V weighted accumulation microkernel for Q16 attention
 *
 * ALGORITHM
 * =========
 * Computes weighted sum of value vectors using softmax weights:
 *
 *   context[d] = Σ weights[k] × V[k, d]  for k ∈ [0, kv_len)
 *
 * Where:
 * - weights[k] is INT16 from Exp2FixedSoftmax [0, 32767]
 * - V[k, d] is INT16 from Q16_1 blocks
 * - context[d] is INT32 accumulator
 *
 * KEY DESIGN
 * ==========
 * - Pure INT16×INT16→INT32 accumulation (maps to VPDPWSSD in JIT)
 * - NO floating-point operations during computation
 * - Scale factors are applied ONCE after accumulation completes
 * - Templated on block type for variable block size support
 *
 * NUMERICAL PRECISION
 * ===================
 * - Each weight is at most 32767 (15 bits)
 * - Each V value is at most ±32767 (16 bits signed)
 * - Product is at most ~2^30 (fits INT32 comfortably)
 * - Sum over kv_len values: need INT32 for sequences up to ~65K
 *   (2^30 × 65K = 2^46, but typical weights sum to ~32767 total)
 *
 * For very long sequences (>64K), consider using INT64 accumulators
 * or periodic rescaling. Current implementation uses INT32 which is
 * sufficient for typical inference scenarios.
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */
#pragma once

#include <cstdint>
#include "tensors/BlockStructures.h"

namespace llaminar2::kernels::q16_1::microkernels
{

    // Import block types from BlockStructures.h
    using llaminar2::Q16_1Block;
    using llaminar2::Q16_1Block_128;
    using llaminar2::Q16_1Block_64;
    using llaminar2::Q16BlockSize;

    // ============================================================================
    // Flash Decode: Single Query P×V (Weighted Sum)
    // ============================================================================

    /**
     * @brief Accumulate P×V for single query (Flash Decode path).
     *
     * Pure INT32 accumulation - no FP32 anywhere.
     * Context is initialized to zero before accumulation.
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param weights INT16 softmax weights [kv_len] from Exp2FixedSoftmax
     * @param V Value cache blocks [kv_len, blocks_per_row]
     * @param context Output INT32 context [head_dim] (zeroed on entry)
     * @param kv_len Number of KV cache positions
     * @param head_dim Head dimension
     * @param blocks_per_row Number of blocks per head
     */
    template <typename BlockType>
    void q16_pv_accumulate(
        const int16_t *weights,
        const BlockType *V,
        int32_t *context,
        int kv_len,
        int head_dim,
        int blocks_per_row);

    /**
     * @brief Accumulate P×V for single query, adding to existing context.
     *
     * Same as q16_pv_accumulate but does NOT zero context first.
     * Used for online softmax rescaling where we update existing context.
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param weights INT16 softmax weights [kv_len]
     * @param V Value cache blocks [kv_len, blocks_per_row]
     * @param context INT32 context [head_dim] to accumulate into
     * @param kv_len Number of KV cache positions
     * @param head_dim Head dimension
     * @param blocks_per_row Number of blocks per head
     */
    template <typename BlockType>
    void q16_pv_accumulate_add(
        const int16_t *weights,
        const BlockType *V,
        int32_t *context,
        int kv_len,
        int head_dim,
        int blocks_per_row);

    // ============================================================================
    // FA2 Prefill: Tiled P×V (Multiple Queries)
    // ============================================================================

    /**
     * @brief Accumulate P×V for a tile of queries (FA2 Prefill path).
     *
     * For each query row q in [0, Br):
     *   context[q, d] += Σ P[q, k] × V[k, d]  for k ∈ [0, Bc)
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param P INT16 weight tile [Br, Bc] from Exp2FixedSoftmax
     * @param V Value tile blocks [Bc, blocks_per_row]
     * @param context Output INT32 context tile [Br, head_dim]
     * @param Br Number of queries in tile
     * @param Bc Number of values in tile
     * @param head_dim Head dimension
     * @param blocks_per_row Number of blocks per head
     * @param p_stride Stride between P rows (usually Bc)
     * @param v_stride Stride between V rows in blocks
     * @param context_stride Stride between context rows (usually head_dim)
     */
    template <typename BlockType>
    void q16_pv_gemm_tile(
        const int16_t *P,
        const BlockType *V,
        int32_t *context,
        int Br,
        int Bc,
        int head_dim,
        int blocks_per_row,
        int p_stride,
        int v_stride,
        int context_stride);

    // ============================================================================
    // Online Softmax Support: Context Rescaling
    // ============================================================================

    /**
     * @brief Rescale INT32 context by an integer factor.
     *
     * Used in online softmax to rescale previous context when max changes:
     *   context[d] = (context[d] * scale_num) >> scale_shift
     *
     * This is the integer equivalent of: context *= exp(m_old - m_new)
     *
     * @param context INT32 context [head_dim]
     * @param head_dim Head dimension
     * @param scale_num Numerator of scale factor (from exp2 LUT)
     * @param scale_shift Right-shift to apply (integer part of exp2)
     */
    void q16_context_rescale(
        int32_t *context,
        int head_dim,
        int32_t scale_num,
        int scale_shift);

    // ============================================================================
    // Non-templated dispatch functions
    // ============================================================================

    /**
     * @brief Dispatch P×V accumulate based on block size enum.
     */
    void q16_pv_accumulate_dispatch(
        const int16_t *weights,
        const void *V,
        int32_t *context,
        int kv_len,
        int head_dim,
        Q16BlockSize block_size);

    /**
     * @brief Dispatch P×V GEMM tile based on block size enum.
     */
    void q16_pv_gemm_tile_dispatch(
        const int16_t *P,
        const void *V,
        int32_t *context,
        int Br,
        int Bc,
        int head_dim,
        Q16BlockSize block_size,
        int p_stride,
        int v_stride,
        int context_stride);

    // ============================================================================
    // Explicit Template Instantiations (declared in .cpp)
    // ============================================================================

    extern template void q16_pv_accumulate<Q16_1Block_64>(
        const int16_t *, const Q16_1Block_64 *, int32_t *, int, int, int);
    extern template void q16_pv_accumulate<Q16_1Block_128>(
        const int16_t *, const Q16_1Block_128 *, int32_t *, int, int, int);

    extern template void q16_pv_accumulate_add<Q16_1Block_64>(
        const int16_t *, const Q16_1Block_64 *, int32_t *, int, int, int);
    extern template void q16_pv_accumulate_add<Q16_1Block_128>(
        const int16_t *, const Q16_1Block_128 *, int32_t *, int, int, int);

    extern template void q16_pv_gemm_tile<Q16_1Block_64>(
        const int16_t *, const Q16_1Block_64 *, int32_t *, int, int, int, int, int, int, int);
    extern template void q16_pv_gemm_tile<Q16_1Block_128>(
        const int16_t *, const Q16_1Block_128 *, int32_t *, int, int, int, int, int, int, int);

} // namespace llaminar2::kernels::q16_1::microkernels
