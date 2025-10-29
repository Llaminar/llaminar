/**
 * @file GemmMicroKernel.h
 * @brief Template-based GEMM micro-kernel with dynamic tile sizes
 *
 * Provides a flexible micro-kernel that supports arbitrary TILE_M × TILE_N sizes,
 * solving the TILE_N=4 limitation of the old macro-based approach.
 *
 * Key features:
 * - ISA-agnostic (works with AVX512, AVX2, Scalar)
 * - Dynamic accumulator array (no hardcoded 8×4 structure)
 * - Loop-based accumulation (compiler unrolls small loops)
 * - Inline-friendly (all methods are small and inline)
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#pragma once

#include "SimdTraits.h"
#include <algorithm>
#include <cstring>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief GEMM micro-kernel: Compute C[TILE_M × TILE_N] += A[TILE_M × K] * B[K × TILE_N]
             *
             * This template class manages TILE_M × TILE_N SIMD accumulators, each accumulating
             * partial products along the K dimension. The final reduction sums each accumulator
             * to produce one scalar output element.
             *
             * @tparam ISA - SIMD ISA tag (AVX512Tag, AVX2Tag, ScalarTag)
             * @tparam TILE_M - Number of output rows (typically 4, 8, 16, 32, 64)
             * @tparam TILE_N - Number of output columns (4, 8, 16, 32)
             *
             * Micro-kernel algorithm:
             * 1. Initialize TILE_M × TILE_N accumulators to zero
             * 2. For each K-block:
             *    - Load TILE_M rows of A (each VECTOR_WIDTH elements)
             *    - Load TILE_N columns of B (each VECTOR_WIDTH elements)
             *    - Accumulate outer product: accumulators[i][j] += a[i] * b[j]
             * 3. Reduce each accumulator to scalar (horizontal sum)
             * 4. Store TILE_M × TILE_N scalar results to C
             *
             * Example (AVX512, 8×4 tile):
             * - 32 __m512 accumulators (8 rows × 4 columns)
             * - Each accumulator holds 16 partial products along K
             * - Reduction: _mm512_reduce_add_ps() on each accumulator
             * - Output: 32 scalar values (8 rows × 4 columns)
             */
            template <typename ISA, int TILE_M, int TILE_N>
            class MicroKernel
            {
            public:
                using Traits = simd::SimdTraits<ISA>;
                using VectorType = typename Traits::VectorType;

                static constexpr int VECTOR_WIDTH = Traits::vector_width;

                // Compile-time validation
                static_assert(TILE_M > 0 && TILE_M <= 256, "TILE_M must be in [1, 256]");
                static_assert(TILE_N > 0 && TILE_N <= 256, "TILE_N must be in [1, 256]");

                /**
                 * @brief Constructor - allocates accumulator storage
                 */
                MicroKernel()
                {
                    // Accumulators allocated on stack or heap (depending on size)
                    // Small tiles (≤256 elements) use stack allocation
                    // Large tiles may benefit from heap allocation (future optimization)
                }

                /**
                 * @brief Zero all accumulators
                 *
                 * Call this once before processing all K-blocks for a tile.
                 */
                __attribute__((always_inline))
                inline void zero()
                {
                    // Force full unrolling for all tile sizes
                    #pragma GCC unroll 256
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        #pragma GCC unroll 256
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            accumulators_[i][j] = Traits::zero();
                        }
                    }
                }

                /**
                 * @brief Accumulate outer product at given offset
                 *
                 * Computes: accumulators[i][j] += A[i, offset:offset+VECTOR_WIDTH] * B[j, offset:offset+VECTOR_WIDTH]
                 *
                 * @param A_panel Pointer to A panel (TILE_M rows, k_panel columns, row-major)
                 * @param B_panel Pointer to B panel (TILE_N columns, k_panel rows, row-major per column)
                 * @param k_panel Number of elements per row in A_panel / per column in B_panel
                 * @param offset Starting position in K dimension (must be multiple of VECTOR_WIDTH)
                 *
                 * Layout assumptions:
                 * - A_panel: A_panel[i * k_panel + k] is element (i, k)
                 * - B_panel: B_panel[j * k_panel + k] is element (j, k)
                 *
                 * Example (TILE_M=8, TILE_N=4, VECTOR_WIDTH=16, offset=32):
                 * - Load a0 = A_panel[0 * k_panel + 32..47] (row 0, columns 32-47)
                 * - Load b0 = B_panel[0 * k_panel + 32..47] (column 0, rows 32-47)
                 * - Accumulate: accumulators[0][0] += a0 * b0 (element-wise FMA)
                 */
                __attribute__((always_inline))
                inline void accumulate(const float * __restrict__ A_panel, 
                                      const float * __restrict__ B_panel,
                                      int k_panel, int offset)
                {
                    // Load A rows into temporary vectors
                    VectorType a_vecs[TILE_M];
                    #pragma GCC unroll 256
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        a_vecs[i] = Traits::load(A_panel + i * k_panel + offset);
                    }

                    // Load B columns into temporary vectors
                    VectorType b_vecs[TILE_N];
                    #pragma GCC unroll 256
                    for (int j = 0; j < TILE_N; ++j)
                    {
                        b_vecs[j] = Traits::load(B_panel + j * k_panel + offset);
                    }

                    // Outer product accumulation (TILE_M × TILE_N FMAs)
                    // Force complete unrolling - this is the critical hot path!
                    #pragma GCC ivdep  // Ignore vector dependencies (false deps from array indexing)
                    #pragma GCC unroll 256
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        #pragma GCC unroll 256
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            // Use register keyword to hint keeping in registers
                            register VectorType acc_reg = accumulators_[i][j];
                            acc_reg = Traits::fmadd(a_vecs[i], b_vecs[j], acc_reg);
                            accumulators_[i][j] = acc_reg;
                        }
                    }
                }

                /**
                 * @brief Accumulate outer product with explicit A/B vector arrays
                 *
                 * Variant that accepts pre-loaded vectors (for manual optimization).
                 *
                 * @param a_vecs Array of TILE_M loaded A row vectors
                 * @param b_vecs Array of TILE_N loaded B column vectors
                 */
                __attribute__((always_inline))
                inline void accumulate_vectors(const VectorType * __restrict__ a_vecs, 
                                              const VectorType * __restrict__ b_vecs)
                {
                    // Critical hot path: force complete unrolling
                    #pragma GCC ivdep  // Ignore vector dependencies
                    #pragma GCC unroll 256
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        #pragma GCC unroll 256
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            // Register hint to avoid memory spills
                            register VectorType acc_reg = accumulators_[i][j];
                            acc_reg = Traits::fmadd(a_vecs[i], b_vecs[j], acc_reg);
                            accumulators_[i][j] = acc_reg;
                        }
                    }
                }

                /**
                 * @brief Reduce accumulators to scalar results
                 *
                 * Performs horizontal reduction on each accumulator:
                 * - AVX512: Sum 16 floats → 1 scalar (_mm512_reduce_add_ps)
                 * - AVX2: Sum 8 floats → 1 scalar (hadd sequence)
                 * - Scalar: Just return the single element
                 *
                 * @param C_tile Output buffer (at least TILE_M * TILE_N floats, row-major)
                 *
                 * Layout: C_tile[i * TILE_N + j] = reduced value for (i, j)
                 */
                __attribute__((always_inline))
                inline void reduce(float *C_tile) const
                {
                    #pragma GCC unroll 256
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        #pragma GCC unroll 256
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            C_tile[i * TILE_N + j] = Traits::reduce_add(accumulators_[i][j]);
                        }
                    }
                }

                /**
                 * @brief Reduce and accumulate into existing C tile
                 *
                 * Computes: C_tile[i][j] = alpha * (reduced accumulator[i][j]) + beta * C_tile[i][j]
                 *
                 * @param C_tile Output buffer (TILE_M * TILE_N floats, row-major)
                 * @param alpha Scaling factor for computed result (typically 1.0)
                 * @param beta Scaling factor for existing C values (typically 0.0 or 1.0)
                 */
                __attribute__((always_inline))
                inline void reduce_accumulate(float *C_tile, float alpha, float beta) const
                {
                    #pragma GCC unroll 256
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        #pragma GCC unroll 256
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int idx = i * TILE_N + j;
                            float reduced = Traits::reduce_add(accumulators_[i][j]);
                            C_tile[idx] = alpha * reduced + beta * C_tile[idx];
                        }
                    }
                }

                /**
                 * @brief Get raw accumulator reference (for advanced use)
                 *
                 * Allows direct manipulation of accumulators (e.g., for custom reductions).
                 */
                inline VectorType &accumulator(int i, int j)
                {
                    return accumulators_[i][j];
                }

                /**
                 * @brief Get raw accumulator reference (const version)
                 */
                inline const VectorType &accumulator(int i, int j) const
                {
                    return accumulators_[i][j];
                }

            private:
                // TILE_M × TILE_N SIMD accumulator array
                // Each accumulator is a SIMD vector (e.g., __m512 for AVX512, __m256 for AVX2)
                // aligned to 64 bytes for optimal performance
                alignas(64) VectorType accumulators_[TILE_M][TILE_N];
            };

            // ========== SPECIALIZATIONS FOR COMMON TILE SIZES ==========

            /**
             * @brief Performance hint: Preferred tile sizes for each ISA
             *
             * These are empirically determined optimal tile sizes for common workloads.
             * The auto-tuner may discover better sizes for specific use cases.
             */
            template <typename ISA>
            struct PreferredTileSizes;

#if defined(__AVX512F__)
            template <>
            struct PreferredTileSizes<simd::AVX512Tag>
            {
                // AVX512 has 32 __m512 registers, can support large tiles
                static constexpr int small_m = 8;
                static constexpr int small_n = 4; // Conservative (existing)
                static constexpr int medium_m = 16;
                static constexpr int medium_n = 8; // NEW: 2× wider
                static constexpr int large_m = 32;
                static constexpr int large_n = 16; // NEW: 4× wider

                // For very large matrices (L3-resident tiles)
                static constexpr int xlarge_m = 64;
                static constexpr int xlarge_n = 32;
            };
#endif

#if defined(__AVX2__)
            template <>
            struct PreferredTileSizes<simd::AVX2Tag>
            {
                // AVX2 has 16 __m256 registers, use smaller tiles
                static constexpr int small_m = 4;
                static constexpr int small_n = 4;
                static constexpr int medium_m = 8;
                static constexpr int medium_n = 4; // Conservative for register pressure
                static constexpr int large_m = 16;
                static constexpr int large_n = 8; // NEW: Wider tiles possible

                // Extra large tiles may cause register spilling
                static constexpr int xlarge_m = 32;
                static constexpr int xlarge_n = 8; // Limit N to avoid spills
            };
#endif

            template <>
            struct PreferredTileSizes<simd::ScalarTag>
            {
                // Scalar fallback: tiny tiles
                static constexpr int small_m = 4;
                static constexpr int small_n = 4;
                static constexpr int medium_m = 4;
                static constexpr int medium_n = 4;
                static constexpr int large_m = 4;
                static constexpr int large_n = 4;
                static constexpr int xlarge_m = 4;
                static constexpr int xlarge_n = 4;
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
