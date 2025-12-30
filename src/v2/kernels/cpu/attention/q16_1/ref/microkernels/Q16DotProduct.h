/**
 * @file Q16DotProduct.h
 * @brief Integer Q×K^T dot product microkernel for Q16 attention
 *
 * ALGORITHM
 * =========
 * Computes dot product between Q16_1 query and key tensors, producing INT32 scores.
 *
 * For single query (Flash Decode):
 *   scores[k] = Σ Q[d] × K[k, d]  for d ∈ [0, head_dim)
 *
 * For multiple queries (FA2 Prefill):
 *   scores[q, k] = Σ Q[q, d] × K[k, d]  for d ∈ [0, head_dim)
 *
 * KEY DESIGN
 * ==========
 * - Pure INT16×INT16→INT32 accumulation (maps to VPDPWSSD in JIT)
 * - NO floating-point operations during computation
 * - Scale factors are NOT applied here - handled post-softmax
 * - Templated on block type for variable block size support
 *
 * BLOCK SIZE OPTIMIZATION
 * =======================
 * With aligned head_dim (64 or 128), each head has exactly 1 Q16_1 block.
 * This gives a single scale factor per head - ideal for integer arithmetic.
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
    // Flash Decode: Single Query GEMV (Q × K^T → scores)
    // ============================================================================

    /**
     * @brief Compute single Q×K dot product for one query-key pair.
     *
     * Pure INT32 accumulation - no FP32 anywhere.
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query block(s) for single head
     * @param K Key block(s) for single head at position k
     * @param head_dim Head dimension
     * @param blocks_per_row Number of blocks per head
     * @return INT32 dot product score
     */
    template <typename BlockType>
    int32_t q16_dot_single(
        const BlockType *Q,
        const BlockType *K,
        int head_dim,
        int blocks_per_row);

    /**
     * @brief Compute Q×K^T for all KV positions (GEMV pattern).
     *
     * Flash Decode path: single query against all cached keys.
     * Output is INT32 scores that can be passed to Exp2FixedSoftmax.
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query blocks for single position [blocks_per_row]
     * @param K Key cache blocks [kv_len, blocks_per_row]
     * @param scores Output INT32 scores [kv_len]
     * @param kv_len Number of KV cache positions
     * @param head_dim Head dimension
     * @param blocks_per_row Number of blocks per head
     */
    template <typename BlockType>
    void q16_qk_gemv(
        const BlockType *Q,
        const BlockType *K,
        int32_t *scores,
        int kv_len,
        int head_dim,
        int blocks_per_row);

    // ============================================================================
    // FA2 Prefill: Multi-Query GEMM (Q_tile × K_tile^T → S_tile)
    // ============================================================================

    /**
     * @brief Compute Q×K^T for a tile of queries and keys (GEMM pattern).
     *
     * FA2 Prefill path: multiple queries against a tile of keys.
     * Produces a score tile [Br, Bc] for online softmax.
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query tile blocks [Br, blocks_per_row]
     * @param K Key tile blocks [Bc, blocks_per_row]
     * @param scores Output score tile [Br, Bc] (row-major)
     * @param Br Number of queries in tile
     * @param Bc Number of keys in tile
     * @param head_dim Head dimension
     * @param blocks_per_row Number of blocks per head
     * @param q_stride Stride between query rows in blocks
     * @param k_stride Stride between key rows in blocks
     */
    template <typename BlockType>
    void q16_qk_gemm_tile(
        const BlockType *Q,
        const BlockType *K,
        int32_t *scores,
        int Br,
        int Bc,
        int head_dim,
        int blocks_per_row,
        int q_stride,
        int k_stride);

    // ============================================================================
    // Non-templated dispatch functions (for use without knowing BlockType)
    // ============================================================================

    /**
     * @brief Dispatch Q×K GEMV based on block size enum.
     */
    void q16_qk_gemv_dispatch(
        const void *Q,
        const void *K,
        int32_t *scores,
        int kv_len,
        int head_dim,
        Q16BlockSize block_size);

    /**
     * @brief Dispatch Q×K GEMM tile based on block size enum.
     */
    void q16_qk_gemm_tile_dispatch(
        const void *Q,
        const void *K,
        int32_t *scores,
        int Br,
        int Bc,
        int head_dim,
        Q16BlockSize block_size,
        int q_stride,
        int k_stride);

    // ============================================================================
    // Explicit Template Instantiations (declared in .cpp)
    // ============================================================================

    extern template int32_t q16_dot_single<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, int, int);
    extern template int32_t q16_dot_single<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, int, int);

    extern template void q16_qk_gemv<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, int32_t *, int, int, int);
    extern template void q16_qk_gemv<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, int32_t *, int, int, int);

    extern template void q16_qk_gemm_tile<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, int32_t *, int, int, int, int, int, int);
    extern template void q16_qk_gemm_tile<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, int32_t *, int, int, int, int, int, int);

} // namespace llaminar2::kernels::q16_1::microkernels
