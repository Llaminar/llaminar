/**
 * @file Q16DotProduct.cpp
 * @brief Integer Q×K^T dot product microkernel implementation
 *
 * @see Q16DotProduct.h for algorithm description
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */

#include "Q16DotProduct.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cstring>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Single Dot Product Implementation
    // ============================================================================

    template <typename BlockType>
    int32_t q16_dot_single(
        const BlockType *Q,
        const BlockType *K,
        int head_dim,
        int blocks_per_row)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        int32_t acc = 0;

        // For aligned head_dim, we have exactly 1 block → single loop iteration
        // For non-aligned, we iterate over multiple blocks
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const int16_t *q_data = Q[b].qs;
            const int16_t *k_data = K[b].qs;

            // Determine how many elements in this block
            int start = b * BLOCK_SIZE;
            int end = std::min(start + BLOCK_SIZE, head_dim);
            int count = end - start;

            // Pure INT32 accumulation
            // NOTE: This loop pattern will vectorize to VPDPWSSD with AVX-512 VNNI
            // when compiled with -march=skylake-avx512 or similar
            for (int i = 0; i < count; ++i)
            {
                acc += static_cast<int32_t>(q_data[i]) * static_cast<int32_t>(k_data[i]);
            }
        }

        return acc;
    }

    // ============================================================================
    // GEMV Implementation (Flash Decode)
    // ============================================================================

    template <typename BlockType>
    void q16_qk_gemv(
        const BlockType *Q,
        const BlockType *K,
        int32_t *scores,
        int kv_len,
        int head_dim,
        int blocks_per_row)
    {
        // Parallel over KV positions for Flash Decode
        // Each score computation is independent
#pragma omp parallel for schedule(static)
        for (int k = 0; k < kv_len; ++k)
        {
            // K is laid out as [kv_len, blocks_per_row]
            const BlockType *K_row = K + k * blocks_per_row;
            scores[k] = q16_dot_single<BlockType>(Q, K_row, head_dim, blocks_per_row);
        }
    }

    // ============================================================================
    // GEMM Tile Implementation (FA2 Prefill)
    // ============================================================================

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
        int k_stride)
    {
        // Compute S[q, k] = Q[q] · K[k] for all (q, k) in tile
        // Output is row-major: scores[q * Bc + k]

        for (int q = 0; q < Br; ++q)
        {
            const BlockType *Q_row = Q + q * q_stride;

            for (int k = 0; k < Bc; ++k)
            {
                const BlockType *K_row = K + k * k_stride;
                scores[q * Bc + k] = q16_dot_single<BlockType>(
                    Q_row, K_row, head_dim, blocks_per_row);
            }
        }
    }

    // ============================================================================
    // Dispatch Functions
    // ============================================================================

    void q16_qk_gemv_dispatch(
        const void *Q,
        const void *K,
        int32_t *scores,
        int kv_len,
        int head_dim,
        Q16BlockSize block_size)
    {
        int blocks_per_row = (head_dim + static_cast<int>(block_size) - 1) /
                             static_cast<int>(block_size);

        switch (block_size)
        {
        case Q16BlockSize::BLOCK_64:
            q16_qk_gemv<Q16_1Block_64>(
                reinterpret_cast<const Q16_1Block_64 *>(Q),
                reinterpret_cast<const Q16_1Block_64 *>(K),
                scores, kv_len, head_dim, blocks_per_row);
            break;

        case Q16BlockSize::BLOCK_128:
            q16_qk_gemv<Q16_1Block_128>(
                reinterpret_cast<const Q16_1Block_128 *>(Q),
                reinterpret_cast<const Q16_1Block_128 *>(K),
                scores, kv_len, head_dim, blocks_per_row);
            break;

        default:
            LOG_ERROR("Q16DotProduct: Unsupported block size: "
                      << static_cast<int>(block_size));
            // Zero output as fallback
            std::memset(scores, 0, kv_len * sizeof(int32_t));
            break;
        }
    }

    void q16_qk_gemm_tile_dispatch(
        const void *Q,
        const void *K,
        int32_t *scores,
        int Br,
        int Bc,
        int head_dim,
        Q16BlockSize block_size,
        int q_stride,
        int k_stride)
    {
        int blocks_per_row = (head_dim + static_cast<int>(block_size) - 1) /
                             static_cast<int>(block_size);

        switch (block_size)
        {
        case Q16BlockSize::BLOCK_64:
            q16_qk_gemm_tile<Q16_1Block_64>(
                reinterpret_cast<const Q16_1Block_64 *>(Q),
                reinterpret_cast<const Q16_1Block_64 *>(K),
                scores, Br, Bc, head_dim, blocks_per_row, q_stride, k_stride);
            break;

        case Q16BlockSize::BLOCK_128:
            q16_qk_gemm_tile<Q16_1Block_128>(
                reinterpret_cast<const Q16_1Block_128 *>(Q),
                reinterpret_cast<const Q16_1Block_128 *>(K),
                scores, Br, Bc, head_dim, blocks_per_row, q_stride, k_stride);
            break;

        default:
            LOG_ERROR("Q16DotProduct: Unsupported block size for GEMM tile: "
                      << static_cast<int>(block_size));
            std::memset(scores, 0, Br * Bc * sizeof(int32_t));
            break;
        }
    }

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template int32_t q16_dot_single<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, int, int);
    template int32_t q16_dot_single<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, int, int);

    template void q16_qk_gemv<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, int32_t *, int, int, int);
    template void q16_qk_gemv<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, int32_t *, int, int, int);

    template void q16_qk_gemm_tile<Q16_1Block_64>(
        const Q16_1Block_64 *, const Q16_1Block_64 *, int32_t *, int, int, int, int, int, int);
    template void q16_qk_gemm_tile<Q16_1Block_128>(
        const Q16_1Block_128 *, const Q16_1Block_128 *, int32_t *, int, int, int, int, int, int);

} // namespace llaminar2::kernels::q16_1::microkernels
