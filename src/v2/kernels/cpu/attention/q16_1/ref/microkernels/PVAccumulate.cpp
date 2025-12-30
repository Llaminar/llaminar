/**
 * @file PVAccumulate.cpp
 * @brief Integer P×V weighted accumulation microkernel implementation
 *
 * @see PVAccumulate.h for algorithm description
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */

#include "PVAccumulate.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cstring>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Core P×V Accumulation (Shared Implementation)
    // ============================================================================

    namespace
    {
        /**
         * @brief Inner loop for P×V accumulation.
         *
         * Accumulates weight × value into context for a range of K positions.
         * This is the hot loop that maps to VPDPWSSD in JIT.
         */
        template <typename BlockType>
        void pv_accumulate_inner(
            const int16_t *weights,
            const BlockType *V,
            int32_t *context,
            int kv_start,
            int kv_end,
            int head_dim,
            int blocks_per_row)
        {
            constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

            for (int k = kv_start; k < kv_end; ++k)
            {
                int16_t w = weights[k];

                // Skip zero weights (masked or negligible)
                if (w == 0)
                    continue;

                // For each block in the head
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    const int16_t *v_data = V[k * blocks_per_row + b].qs;

                    int start = b * BLOCK_SIZE;
                    int end = std::min(start + BLOCK_SIZE, head_dim);
                    int count = end - start;

                    // Pure INT32 accumulation: context[d] += w × v[d]
                    // This loop pattern vectorizes to VPDPWSSD with AVX-512 VNNI
                    for (int i = 0; i < count; ++i)
                    {
                        context[start + i] += static_cast<int32_t>(w) * static_cast<int32_t>(v_data[i]);
                    }
                }
            }
        }

    } // namespace

    // ============================================================================
    // Flash Decode: Single Query P×V
    // ============================================================================

    template <typename BlockType>
    void q16_pv_accumulate(
        const int16_t *weights,
        const BlockType *V,
        int32_t *context,
        int kv_len,
        int head_dim,
        int blocks_per_row)
    {
        // Zero context before accumulation
        std::memset(context, 0, head_dim * sizeof(int32_t));

        // Accumulate all KV positions
        pv_accumulate_inner<BlockType>(
            weights, V, context,
            0, kv_len, head_dim, blocks_per_row);
    }

    template <typename BlockType>
    void q16_pv_accumulate_add(
        const int16_t *weights,
        const BlockType *V,
        int32_t *context,
        int kv_len,
        int head_dim,
        int blocks_per_row)
    {
        // Do NOT zero - add to existing context
        pv_accumulate_inner<BlockType>(
            weights, V, context,
            0, kv_len, head_dim, blocks_per_row);
    }

    // ============================================================================
    // FA2 Prefill: Tiled P×V (GEMM Pattern)
    // ============================================================================

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
        int context_stride)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        // For each query row in tile
        for (int q = 0; q < Br; ++q)
        {
            const int16_t *P_row = P + q * p_stride;
            int32_t *context_row = context + q * context_stride;

            // For each value position in tile
            for (int k = 0; k < Bc; ++k)
            {
                int16_t w = P_row[k];

                if (w == 0)
                    continue;

                // For each block in head
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    // V is laid out as [Bc, blocks_per_row] for the tile
                    const int16_t *v_data = V[k * v_stride + b].qs;

                    int start = b * BLOCK_SIZE;
                    int end = std::min(start + BLOCK_SIZE, head_dim);
                    int count = end - start;

                    for (int i = 0; i < count; ++i)
                    {
                        context_row[start + i] +=
                            static_cast<int32_t>(w) * static_cast<int32_t>(v_data[i]);
                    }
                }
            }
        }
    }

    // ============================================================================
    // Online Softmax: Context Rescaling
    // ============================================================================

    void q16_context_rescale(
        int32_t *context,
        int head_dim,
        int32_t scale_num,
        int scale_shift)
    {
        // Apply integer rescaling: context = (context × scale_num) >> scale_shift
        // This is used when online softmax max changes: context *= exp2(-delta)

        if (scale_shift == 0 && scale_num == 1)
        {
            // No-op: scale factor is 1.0
            return;
        }

        if (scale_num == 0)
        {
            // Scale is 0: zero out context (shouldn't happen in practice)
            std::memset(context, 0, head_dim * sizeof(int32_t));
            return;
        }

        // Apply scaling with proper rounding
        // For positive context values: (context × scale_num + (1 << (shift-1))) >> shift
        // For negative: handle sign carefully
        int32_t round_add = (scale_shift > 0) ? (1 << (scale_shift - 1)) : 0;

        for (int d = 0; d < head_dim; ++d)
        {
            int64_t scaled = static_cast<int64_t>(context[d]) * scale_num;

            if (scaled >= 0)
            {
                context[d] = static_cast<int32_t>((scaled + round_add) >> scale_shift);
            }
            else
            {
                // Round toward zero for negative values
                context[d] = static_cast<int32_t>((scaled - round_add) >> scale_shift);
            }
        }
    }

    // ============================================================================
    // Dispatch Functions
    // ============================================================================

    void q16_pv_accumulate_dispatch(
        const int16_t *weights,
        const void *V,
        int32_t *context,
        int kv_len,
        int head_dim,
        Q16BlockSize block_size)
    {
        int blocks_per_row = (head_dim + static_cast<int>(block_size) - 1) /
                             static_cast<int>(block_size);

        switch (block_size)
        {
        case Q16BlockSize::BLOCK_64:
            q16_pv_accumulate<Q16_1Block_64>(
                weights,
                reinterpret_cast<const Q16_1Block_64 *>(V),
                context, kv_len, head_dim, blocks_per_row);
            break;

        case Q16BlockSize::BLOCK_128:
            q16_pv_accumulate<Q16_1Block_128>(
                weights,
                reinterpret_cast<const Q16_1Block_128 *>(V),
                context, kv_len, head_dim, blocks_per_row);
            break;

        default:
            LOG_ERROR("PVAccumulate: Unsupported block size: "
                      << static_cast<int>(block_size));
            std::memset(context, 0, head_dim * sizeof(int32_t));
            break;
        }
    }

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
        int context_stride)
    {
        int blocks_per_row = (head_dim + static_cast<int>(block_size) - 1) /
                             static_cast<int>(block_size);

        switch (block_size)
        {
        case Q16BlockSize::BLOCK_64:
            q16_pv_gemm_tile<Q16_1Block_64>(
                P,
                reinterpret_cast<const Q16_1Block_64 *>(V),
                context, Br, Bc, head_dim, blocks_per_row,
                p_stride, v_stride, context_stride);
            break;

        case Q16BlockSize::BLOCK_128:
            q16_pv_gemm_tile<Q16_1Block_128>(
                P,
                reinterpret_cast<const Q16_1Block_128 *>(V),
                context, Br, Bc, head_dim, blocks_per_row,
                p_stride, v_stride, context_stride);
            break;

        default:
            LOG_ERROR("PVAccumulate: Unsupported block size for GEMM tile: "
                      << static_cast<int>(block_size));
            break;
        }
    }

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template void q16_pv_accumulate<Q16_1Block_64>(
        const int16_t *, const Q16_1Block_64 *, int32_t *, int, int, int);
    template void q16_pv_accumulate<Q16_1Block_128>(
        const int16_t *, const Q16_1Block_128 *, int32_t *, int, int, int);

    template void q16_pv_accumulate_add<Q16_1Block_64>(
        const int16_t *, const Q16_1Block_64 *, int32_t *, int, int, int);
    template void q16_pv_accumulate_add<Q16_1Block_128>(
        const int16_t *, const Q16_1Block_128 *, int32_t *, int, int, int);

    template void q16_pv_gemm_tile<Q16_1Block_64>(
        const int16_t *, const Q16_1Block_64 *, int32_t *, int, int, int, int, int, int, int);
    template void q16_pv_gemm_tile<Q16_1Block_128>(
        const int16_t *, const Q16_1Block_128 *, int32_t *, int, int, int, int, int, int, int);

} // namespace llaminar2::kernels::q16_1::microkernels
