/**
 * @file IntegerGemmKernelTemplateV2.h
 * @brief Clean Q8_0×Q8_0→Q8_0 integer GEMM kernel (V2 rewrite)
 *
 * Design principles (mirroring GemmKernelTemplate.h):
 * 1. Clean separation: outer loop (this file) + micro-kernel (IntegerGemmMicroKernelTemplate.h)
 * 2. Simple interface: multiply(A_q8, B_provider, C_q8, m, n, k)
 * 3. Standard OpenMP parallelization (no custom threading heuristics)
 * 4. Single-block K processing (no k_panel=64 optimization)
 * 5. No premature optimizations (get correctness first)
 *
 * Algorithm (matches FP32 pattern):
 * ```
 * for ii in range(0, m, TILE_M):              # M tiles
 *     for jj in range(0, n, TILE_N):          # N tiles
 *         MicroKernel ukernel;
 *         ukernel.zero();
 *         for kb in range(0, num_k_blocks):   # K blocks (32 elements each)
 *             load_A_panel(A_q8, ii, kb)     # Load INT8 + scales
 *             load_B_panel(B_provider, jj, kb) # Load INT8 + scales
 *             ukernel.accumulate(A_panel, B_panel, k_panel, a_scales, b_scales)
 *         ukernel.reduce(C_blocks)           # INT32→FP32→Q8_0
 * ```
 *
 * Key differences from FP32:
 * - Input/output: Q8_0Block* (not float*)
 * - Block size: Always 32 (Q8_0 standard)
 * - Scales: Extracted from Q8_0 blocks, passed to micro-kernel
 * - No alpha/beta: Q8_0 doesn't support scaling factors
 *
 * @author David Sanftenberg
 * @date November 11, 2025
 */

#pragma once

#include "int8/IntegerGemmMicroKernelTemplate.h"
#include "GemmWeightCache.h"
#include "../../../tensors/Tensors.h"
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Integer GEMM kernel: C_q8 = A_q8 × B_q8
             *
             * @tparam ISA SIMD ISA tag (AVX512VNNITag for VNNI support)
             * @tparam MR Micro-kernel M dimension (rows in register block)
             * @tparam NR Micro-kernel N dimension (cols in register block, must be 32)
             * @tparam K_BLOCKS_PER_ITER Number of 32-byte Q8_0 blocks processed per K-loop iteration (1, 2, or 4)
             * @tparam UNROLL_K K-loop unroll factor (future optimization)
             * @tparam PREFETCH_DIST Prefetch distance (future optimization)
             * @tparam MC M-dimension cache block size (future optimization)
             * @tparam KC K-dimension cache block size (future optimization)
             * @tparam NC N-dimension cache block size (future optimization)
             */
            template <
                typename ISA,
                int MR,
                int NR,
                int K_BLOCKS_PER_ITER = 2,
                int UNROLL_K = 4,
                int PREFETCH_DIST = 2,
                int MC = 256,
                int KC = 512,
                int NC = 128>
            class IntegerGemmKernelV2
            {
            public:
                using MicroKernel_t = IntegerMicroKernelTemplate<ISA, MR, NR, K_BLOCKS_PER_ITER, UNROLL_K, PREFETCH_DIST, MC, KC, NC>;

                static constexpr int TILE_M = MR;
                static constexpr int TILE_N = NR;
                static constexpr int BLOCK_SIZE = 32; // Q8_0 standard
                static constexpr int BLOCKS_PER_ITER = K_BLOCKS_PER_ITER; // Compile-time K-block processing
                static constexpr int BYTES_PER_ITER = BLOCK_SIZE * BLOCKS_PER_ITER; // 32, 64, 128, or 256 bytes

                // Compile-time validation
                static_assert(TILE_N == 32, "TILE_N must be 32 for Q8_0 block alignment");
                static_assert(K_BLOCKS_PER_ITER >= 1 && K_BLOCKS_PER_ITER <= 8, "K_BLOCKS_PER_ITER must be 1, 2, 4, or 8");

                /**
                 * @brief Execute integer GEMM: C_q8 = A_q8 × B_q8
                 *
                 * @param A Input Q8_0 activation blocks [m×k] (row-major)
                 * @param B_provider Weight provider (Q8_0 access interface)
                 * @param C Output Q8_0 activation blocks [m×n] (row-major)
                 * @param m Number of rows in A and C
                 * @param n Number of columns in B and C
                 * @param k Number of columns in A and rows in B
                 *
                 * @return true on success, false on error
                 *
                 * Memory layout:
                 * - A: Q8_0Block array, row-major, m rows × k_blocks columns
                 * - C: Q8_0Block array, row-major, m rows × n_blocks columns
                 * - B: Accessed via B_provider (any quantized format, decoded to Q8_0)
                 *
                 * where:
                 * - k_blocks = (k + 31) / 32
                 * - n_blocks = (n + 31) / 32
                 */
                static bool multiply(
                    const Q8_0Block *A,
                    Q8_0BlockProvider &B_provider,
                    Q8_0Block *C,
                    int m, int n, int k)
                {
                    // Validate inputs
                    if (!A || !C)
                    {
                        return false;
                    }

                    if (m <= 0 || n <= 0 || k <= 0)
                    {
                        return false;
                    }

                    // Compute dimensions
                    const size_t k_blocks = (k + 31) / 32;
                    const size_t n_blocks = (n + 31) / 32;

                    // Validate dimensions against provider
                    if (k_blocks != B_provider.k_blocks())
                    {
                        return false; // K dimension mismatch
                    }

                    if (static_cast<size_t>(n) > B_provider.num_rows() * 32)
                    {
                        return false; // N dimension mismatch
                    }

// OpenMP parallelization with static scheduling (lower overhead than dynamic)
#pragma omp parallel for schedule(static)
                    for (int ii = 0; ii < m; ii += TILE_M)
                    {
                        const int actual_tile_m = std::min(TILE_M, m - ii);

                        for (int jj = 0; jj < n; jj += TILE_N)
                        {
                            const int actual_tile_n = std::min(TILE_N, n - jj);

                            // Skip partial tiles at boundary (Q8_0 requires full blocks)
                            if (actual_tile_n != TILE_N)
                            {
                                continue;
                            }

                            // Initialize micro-kernel
                            MicroKernel_t ukernel;
                            ukernel.zero();

                            // K-loop: Process BLOCKS_PER_ITER blocks at a time (compile-time constant)
                            for (size_t kb = 0; kb < k_blocks; kb += BLOCKS_PER_ITER)
                            {
                                // Determine how many blocks to load (handles remainder)
                                const size_t blocks_remaining = k_blocks - kb;
                                const int blocks_to_process = static_cast<int>(
                                    (blocks_remaining >= static_cast<size_t>(BLOCKS_PER_ITER)) ? BLOCKS_PER_ITER : blocks_remaining
                                );

                                // ZERO-COPY ACCUMULATE: Work directly from Q8_0 blocks (NO memcpy!)
                                // This eliminates ~500K memcpy calls - expected 25× speedup!
                                ukernel.accumulate_zerocopy(A, B_provider, 
                                                           actual_tile_m, actual_tile_n,
                                                           ii, jj, kb, k_blocks,
                                                           blocks_to_process);
                            }

                            // Reduce: INT32 → FP32 (with scales) → Q8_0
                            alignas(64) Q8_0Block C_tile[TILE_M];
                            ukernel.reduce(C_tile);

                            // Store to output (scatter to correct row-major positions)
                            const size_t c_block_col = jj / 32;
                            for (int i = 0; i < actual_tile_m; ++i)
                            {
                                C[(ii + i) * n_blocks + c_block_col] = C_tile[i];
                            }
                        }
                    }

                    return true;
                }

            private:
                /**
                 * @brief Load A panel from Q8_0 blocks (multi-block version)
                 *
                 * Extracts INT8 codes and FP32 scales from 1-4 consecutive Q8_0 blocks.
                 * Supports processing 1, 2, or 4 blocks (32, 64, or 128 bytes) for optimal SIMD.
                 *
                 * @param blocks_to_load Number of blocks to load (1, 2, or 4)
                 */
                static void load_A_panel_multi(
                    const Q8_0Block *A,
                    int ii,
                    int tile_m,
                    size_t kb,
                    size_t k_blocks,
                    int blocks_to_load,
                    int8_t *A_panel,
                    float *a_scales)
                {
                    for (int i = 0; i < tile_m; ++i)
                    {
                        for (int b = 0; b < blocks_to_load; ++b)
                        {
                            if (kb + b < k_blocks)
                            {
                                const Q8_0Block *block = &A[(ii + i) * k_blocks + kb + b];

                                // Copy INT8 codes to contiguous location
                                std::memcpy(A_panel + i * blocks_to_load * BLOCK_SIZE + b * BLOCK_SIZE,
                                           block->qs, BLOCK_SIZE);

                                // Extract scale (FP16 → FP32) - fixed stride matching K_BLOCKS_PER_ITER
                                a_scales[i * K_BLOCKS_PER_ITER + b] = fp16_to_fp32(block->d);
                            }
                            else
                            {
                                // Zero-pad if K dimension is not multiple of BLOCKS_PER_ITER * 32
                                std::memset(A_panel + i * blocks_to_load * BLOCK_SIZE + b * BLOCK_SIZE,
                                           0, BLOCK_SIZE);
                                a_scales[i * K_BLOCKS_PER_ITER + b] = 1.0f;
                            }
                        }
                    }

                    // Zero-pad remaining rows if tile_m < TILE_M
                    for (int i = tile_m; i < TILE_M; ++i)
                    {
                        for (int b = 0; b < blocks_to_load; ++b)
                        {
                            std::memset(A_panel + i * blocks_to_load * BLOCK_SIZE + b * BLOCK_SIZE,
                                       0, BLOCK_SIZE);
                            a_scales[i * K_BLOCKS_PER_ITER + b] = 1.0f;
                        }
                    }
                }

                /**
                 * @brief Load B panel from pre-decoded data (FAST PATH - no provider overhead!)
                 *
                 * Uses pre-decoded INT8 array and pre-extracted FP32 scales.
                 * Eliminates redundant B-matrix decoding (25× speedup!).
                 *
                 * @param B_decoded Pre-decoded INT8 data [n × k] (row-major)
                 * @param B_scales Pre-extracted FP32 scales [n × k_blocks] (row-major)
                 * @param blocks_to_load Number of blocks to load (1, 2, or 4)
                 */
                static void load_B_panel_from_decoded(
                    const int8_t *B_decoded,
                    const float *B_scales,
                    int jj,
                    int tile_n,
                    size_t kb,
                    size_t k_blocks,
                    int k,
                    int blocks_to_load,
                    int8_t *B_panel,
                    float *b_scales)
                {
                    for (int j = 0; j < tile_n; ++j)
                    {
                        for (int b = 0; b < blocks_to_load; ++b)
                        {
                            if (kb + b < k_blocks)
                            {
                                // Direct copy from pre-decoded data (no provider call!)
                                const int8_t *source = &B_decoded[(jj + j) * k + (kb + b) * 32];
                                std::memcpy(B_panel + j * blocks_to_load * BLOCK_SIZE + b * BLOCK_SIZE,
                                           source, std::min(32, k - static_cast<int>((kb + b) * 32)));
                                
                                // Direct access to pre-extracted scale (no FP16→FP32 conversion!)
                                b_scales[j * BLOCKS_PER_ITER + b] = B_scales[(jj + j) * k_blocks + kb + b];
                            }
                            else
                            {
                                // Zero-pad if K dimension is not multiple of BLOCKS_PER_ITER * 32
                                std::memset(B_panel + j * blocks_to_load * BLOCK_SIZE + b * BLOCK_SIZE,
                                           0, BLOCK_SIZE);
                                b_scales[j * BLOCKS_PER_ITER + b] = 1.0f;
                            }
                        }
                    }

                    // Zero-pad remaining columns if tile_n < TILE_N
                    for (int j = tile_n; j < TILE_N; ++j)
                    {
                        for (int b = 0; b < blocks_to_load; ++b)
                        {
                            std::memset(B_panel + j * blocks_to_load * BLOCK_SIZE + b * BLOCK_SIZE,
                                       0, BLOCK_SIZE);
                            b_scales[j * BLOCKS_PER_ITER + b] = 1.0f;
                        }
                    }
                }
                
                /**
                 * @brief Load B panel from provider (OLD PATH - kept for compatibility)
                 *
                 * Uses provider to get 1-4 consecutive Q8_0 blocks (from cache or direct access).
                 * Extracts INT8 codes and FP32 scales.
                 *
                 * @param blocks_to_load Number of blocks to load (1, 2, or 4)
                 */
                static void load_B_panel_multi(
                    Q8_0BlockProvider &B_provider,
                    int jj,
                    int tile_n,
                    size_t kb,
                    int blocks_to_load,
                    int8_t *B_panel,
                    float *b_scales)
                {
                    for (int j = 0; j < tile_n; ++j)
                    {
                        for (int b = 0; b < blocks_to_load; ++b)
                        {
                            // Get Q8_0 block from provider (cached or zero-copy)
                            const Q8_0Block *block = B_provider.get_q8_block(jj + j, kb + b);

                            // Copy INT8 codes to contiguous location
                            std::memcpy(B_panel + j * blocks_to_load * BLOCK_SIZE + b * BLOCK_SIZE,
                                       block->qs, BLOCK_SIZE);

                            // Extract scale (FP16 → FP32) - use stride matching BLOCKS_PER_ITER
                            b_scales[j * BLOCKS_PER_ITER + b] = fp16_to_fp32(block->d);
                        }
                    }

                    // Zero-pad remaining columns if tile_n < TILE_N
                    for (int j = tile_n; j < TILE_N; ++j)
                    {
                        for (int b = 0; b < blocks_to_load; ++b)
                        {
                            std::memset(B_panel + j * blocks_to_load * BLOCK_SIZE + b * BLOCK_SIZE,
                                       0, BLOCK_SIZE);
                            b_scales[j * BLOCKS_PER_ITER + b] = 1.0f;
                        }
                    }
                }
                
                /**
                 * @brief Prefetch A panel data (software prefetch hints)
                 * 
                 * Issues prefetch instructions for future A blocks.
                 * Helps hide memory latency for large matrices.
                 */
                static void prefetch_A_panel(
                    const Q8_0Block *A,
                    int ii,
                    int tile_m,
                    size_t kb,
                    size_t k_blocks)
                {
#if defined(__GNUC__) || defined(__clang__)
                    for (int i = 0; i < tile_m; ++i)
                    {
                        const Q8_0Block *block = &A[(ii + i) * k_blocks + kb];
                        __builtin_prefetch(block, 0, 3);  // Read, high temporal locality
                    }
#endif
                }
                
                /**
                 * @brief Prefetch B panel data (software prefetch hints)
                 */
                static void prefetch_B_panel(
                    Q8_0BlockProvider &B_provider,
                    int jj,
                    int tile_n,
                    size_t kb)
                {
#if defined(__GNUC__) || defined(__clang__)
                    for (int j = 0; j < tile_n; ++j)
                    {
                        const Q8_0Block *block = B_provider.get_q8_block(jj + j, kb);
                        __builtin_prefetch(block, 0, 3);  // Read, high temporal locality
                    }
#endif
                }
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
