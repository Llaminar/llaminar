/**
 * @file IntegerGemmKernelTemplate.h
 * @brief Q8_0×Quantized→Q8_0 integer GEMM kernel template
 *
 * This is the new integer-domain GEMM system that operates entirely in quantized space:
 * - Input A: Q8_0 activations (pre-quantized, no FP32)
 * - Input B: Quantized weights (IQ4_NL, Q6_K, Q8_0, etc.)
 * - Computation: INT8×INT8→INT32 (AVX512-VNNI)
 * - Output: Q8_0 activations (requantized from INT32)
 *
 * Benefits over FP32 GEMM:
 * - 4× lower memory bandwidth (Q8_0 vs FP32 activations)
 * - 4× higher compute throughput (INT8 VNNI vs FP32 FMA)
 * - No per-block FP32 dequantization (keep everything in INT8)
 *
 * Algorithm:
 * ```
 * for each M-tile:
 *   for each N-tile:
 *     INT32_acc[TILE_M][TILE_N] = 0
 *     for each K-block:
 *       A_q8[TILE_M] = load Q8_0 activation blocks
 *       B_q8[TILE_N] = decode weight blocks to Q8_0
 *       INT32_acc += VNNI(A_q8, B_q8)  // INT8×INT8→INT32
 *     C_q8[TILE_M][TILE_N] = requantize(INT32_acc)  // INT32→Q8_0
 * ```
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "GemmWeightCache.h"
#include "IntegerRequantization.h"
#include "int8/IntegerGemmMicroKernel.h"
#include "../../../tensors/Tensors.h"
#include <algorithm>
#include <cstring>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Integer GEMM kernel: C_q8 = A_q8 × B_quantized
             *
             * @tparam ISA SIMD ISA tag (must be AVX512VNNITag for INT8 support)
             * @tparam MR Micro-kernel M dimension (rows in register block)
             * @tparam NR Micro-kernel N dimension (cols in register block)
             * @tparam UNROLL_K K-loop unroll factor (1, 2, 4, 8, 16)
             * @tparam PREFETCH_DIST Prefetch distance in iterations (0, 1, 2, 3, 5)
             * @tparam MC M-dimension cache block size (default 256)
             * @tparam KC K-dimension cache block size (default 512, must be multiple of 32)
             * @tparam NC N-dimension cache block size (default 128)
             *
             * Key differences from FP32 GEMM:
             * - Accepts Q8_0Block* inputs (not float*)
             * - Uses Q8_0WeightAccessor (not FP32WeightAccessor)
             * - Outputs Q8_0Block* (not float*)
             * - Uses MicroKernelTemplateINT8 (INT32 accumulation, not float)
             */
            template <
                typename ISA,
                int MR,
                int NR,
                int UNROLL_K = 4,
                int PREFETCH_DIST = 2,
                int MC = 256,
                int KC = 512,
                int NC = 128>
            class IntegerGemmKernel
            {
            public:
                using MicroKernel_t = IntegerGemmMicroKernel<ISA, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC>;

                static constexpr int TILE_M = MR;
                static constexpr int TILE_N = NR;
                static constexpr size_t Q8_0_BLOCK_SIZE = 32; // Q8_0 standard

                // Compile-time validation
                static_assert(KC % 32 == 0, "KC must be divisible by 32 (Q8_0 block size)");
                static_assert(TILE_N == 32, "TILE_N must be 32 for Q8_0 block alignment");

                /**
                 * @brief Execute integer GEMM: C_q8 = A_q8 × B_quantized
                 *
                 * @param A Input Q8_0 activation blocks [m×k] (row-major)
                 * @param B_provider Weight provider (cached or zero-copy Q8_0 access)
                 * @param C Output Q8_0 activation blocks [m×n] (row-major)
                 * @param m Number of rows in A and C
                 * @param n Number of columns in B and C
                 * @param k Number of columns in A and rows in B
                 *
                 * @return true on success, false on error
                 *
                 * Memory layout:
                 * - A: Q8_0Block array, row-major, m rows × k_blocks_a blocks
                 * - C: Q8_0Block array, row-major, m rows × n_blocks_c blocks
                 * - B: Accessed via B_provider (any quantized format)
                 *
                 * where:
                 * - k_blocks_a = (k + 31) / 32
                 * - n_blocks_c = (n + 31) / 32
                 *
                 * Cache behavior:
                 * - Q8_0 weights: Zero-copy direct access (is_zero_copy() == true)
                 * - Other formats: Decoded once per (row, k_block), cached across M-tiles
                 *
                 * Algorithm (mirrors FP32 GEMM pattern):
                 * ```
                 * for ii in range(0, m, TILE_M):
                 *     for jj in range(0, n, TILE_N):
                 *         MicroKernel ukernel;
                 *         ukernel.zero();
                 *         for kb in range(0, num_k_blocks):
                 *             load_panels(A, B, kb, A_panel, B_panel, a_scales, b_scales)
                 *             ukernel.accumulate(A_panel, B_panel, k_panel, a_scales, b_scales)
                 *         ukernel.reduce(C_blocks)
                 * ```
                 */
                static bool multiply(
                    const Q8_0Block *A,
                    Q8_0BlockProvider &B_provider,
                    Q8_0Block *C,
                    int m, int n, int k)
                {
                    if (!A || !C)
                    {
                        return false;
                    }

                    if (m <= 0 || n <= 0 || k <= 0)
                    {
                        return false;
                    }

                    // Validate dimensions
                    const size_t k_blocks_a = (k + 31) / 32;
                    const size_t n_blocks_c = (n + 31) / 32;
                    const size_t k_blocks_b = B_provider.k_blocks();

                    if (k_blocks_a != k_blocks_b)
                    {
                        // K dimension mismatch
                        return false;
                    }

                    if (static_cast<size_t>(n) > B_provider.num_rows() * 32)
                    {
                        // N dimension mismatch
                        return false;
                    }

// Tile loop over M and N dimensions (parallelize over M for load balancing)
#pragma omp parallel for schedule(dynamic, 1)
                    for (int ii = 0; ii < m; ii += TILE_M)
                    {
                        const int actual_tile_m = std::min(TILE_M, m - ii);

                        for (int jj = 0; jj < n; jj += TILE_N)
                        {
                            const int actual_tile_n = std::min(TILE_N, n - jj);

                            // Ensure TILE_N alignment (required for Q8_0 block output)
                            if (actual_tile_n != TILE_N)
                            {
                                // Skip partial tiles at boundary (TODO: handle edge case)
                                continue;
                            }

                            // Allocate packed panels for micro-kernel
                            alignas(64) int8_t A_panel[TILE_M * 32]; // TILE_M rows × 32 elements
                            alignas(64) int8_t B_panel[TILE_N * 32]; // TILE_N columns × 32 elements
                            alignas(64) double a_scales[TILE_M];
                            alignas(64) double b_scales[TILE_N];

                            // Initialize micro-kernel
                            MicroKernel_t ukernel;
                            ukernel.zero();

                            // K-loop over Q8_0 blocks (each block = 32 elements)
                            const size_t num_kb = k_blocks_a;
                            for (size_t kb = 0; kb < num_kb; ++kb)
                            {
                                // Prefetch future K-blocks (L1 cache)
                                if (kb + PREFETCH_DIST < num_kb)
                                {
                                    for (int j = 0; j < actual_tile_n; ++j)
                                    {
                                        const Q8_0Block *future_block = B_provider.get_q8_block(jj + j, kb + PREFETCH_DIST);
                                        __builtin_prefetch(future_block, 0, 1);
                                    }
                                }

                                // Load and pack A panel (Q8_0 → INT8)
                                loadAndPackA(A, ii, actual_tile_m, kb, k_blocks_a, A_panel, a_scales);

                                // Load and pack B panel (quantized → cached Q8_0 → INT8)
                                loadAndPackB(B_provider, jj, actual_tile_n, kb, B_panel, b_scales);

                                // Accumulate INT8×INT8→INT32 with scale tracking
                                ukernel.accumulate(A_panel, B_panel, 32, a_scales, b_scales);
                            }

                            // Reduce INT32 accumulators to Q8_0 output blocks
                            const size_t c_block_row_stride = n_blocks_c;
                            const size_t c_block_col_idx = jj / 32;
                            Q8_0Block *C_blocks = &C[ii * c_block_row_stride + c_block_col_idx];

                            ukernel.reduce(C_blocks);
                        }
                    }

                    return true;
                }

            private:
                /**
                 * @brief Load and pack A panel from Q8_0 blocks to INT8 array
                 *
                 * @param A Source Q8_0 blocks [m × k_blocks]
                 * @param ii Row offset
                 * @param tile_m Number of rows to load
                 * @param kb K-block offset
                 * @param k_blocks Total K-blocks in A
                 * @param A_panel Output INT8 panel [tile_m × 32]
                 * @param a_scales Output scales for each row (FP64 array)
                 */
                static void loadAndPackA(
                    const Q8_0Block *A,
                    int ii, int tile_m,
                    size_t kb, size_t k_blocks,
                    int8_t *A_panel,
                    double *a_scales)
                {
                    for (int i = 0; i < tile_m; ++i)
                    {
                        const Q8_0Block *block = &A[(ii + i) * k_blocks + kb];

                        // Extract INT8 values
                        std::memcpy(A_panel + i * 32, block->qs, 32);

                        // Extract scale (FP16 → FP64)
                        a_scales[i] = static_cast<double>(fp16_to_fp32(block->d));
                    }

                    // Zero-pad remaining rows if tile_m < TILE_M
                    for (int i = tile_m; i < TILE_M; ++i)
                    {
                        std::memset(A_panel + i * 32, 0, 32);
                        a_scales[i] = 1.0;
                    }
                }

                /**
                 * @brief Load and pack B panel from quantized format to INT8 array
                 *
                 * Uses provider to get decoded Q8_0 blocks (from cache or direct).
                 * The cache ensures each weight block is decoded at most once across
                 * all M-tile iterations, fixing the catastrophic redundant-decode bug.
                 *
                 * @param B_provider Weight provider (provides Q8_0 blocks)
                 * @param jj Column offset
                 * @param tile_n Number of columns to load
                 * @param kb K-block offset
                 * @param B_panel Output INT8 panel [tile_n × 32]
                 * @param b_scales Output scales for each column
                 */
                static void loadAndPackB(
                    Q8_0BlockProvider &B_provider,
                    int jj, int tile_n,
                    size_t kb,
                    int8_t *B_panel,
                    double *b_scales)
                {
                    for (int j = 0; j < tile_n; ++j)
                    {
                        // Get cached or zero-copy Q8_0 block
                        // For Q8_0 weights: direct pointer (zero overhead)
                        // For IQ4_NL/Q6_K/FP32: decoded once, cached, reused across M-tiles
                        const Q8_0Block *q8_block = B_provider.get_q8_block(jj + j, kb);

                        // Copy INT8 values to panel
                        std::memcpy(B_panel + j * 32, q8_block->qs, 32);

                        // Extract scale
                        b_scales[j] = static_cast<double>(fp16_to_fp32(q8_block->d));
                    }

                    // Zero-pad remaining columns if tile_n < TILE_N
                    for (int j = tile_n; j < TILE_N; ++j)
                    {
                        std::memset(B_panel + j * 32, 0, 32);
                        b_scales[j] = 1.0;
                    }
                }
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
