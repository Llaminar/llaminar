/**
 * @file GemmMicroKernelTemplate.h
 * @brief Fully parameterized micro-kernel template for GEMM
 *
 * This template parameterizes ALL search dimensions:
 * - ISA (AVX512, AVX2, etc.)
 * - Tile sizes (MR × NR)
 * - K-loop unroll factor
 * - Prefetch distance
 * - Cache blocking sizes
 *
 * Explicit instantiations are generated in generated/ directory.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "SimdTraits.h"
#include <cstring>
#include <algorithm>

namespace llaminar2 {
namespace kernels {
namespace gemm {

/**
 * @brief Fully parameterized micro-kernel template
 *
 * @tparam ISA Instruction set architecture (AVX512Tag, AVX2Tag)
 * @tparam MR Micro-kernel M dimension (rows in register block)
 * @tparam NR Micro-kernel N dimension (cols in register block)
 * @tparam UNROLL_K K-loop unroll factor (1, 2, 4, 8, 16)
 * @tparam PREFETCH_DIST Prefetch distance in iterations (0, 1, 2, 3, 5)
 * @tparam MC M-dimension cache block size
 * @tparam KC K-dimension cache block size
 * @tparam NC N-dimension cache block size
 */
template<
    typename ISA,
    int MR,
    int NR,
    int UNROLL_K = 4,
    int PREFETCH_DIST = 2,
    int MC = 256,
    int KC = 512,
    int NC = 128
>
class MicroKernelTemplate {
public:
    using Traits = simd::SimdTraits<ISA>;
    using Vec = typename Traits::VectorType;
    static constexpr int kWidth = Traits::vector_width;

    static constexpr int TILE_M = MR;
    static constexpr int TILE_N = NR;

    /**
     * @brief Micro-kernel: Compute C[MR×NR] += A[MR×k_panel] * B[NR×k_panel]^T
     *
     * A_panel: MR rows × k_panel cols (row-major)
     * B_panel: NR rows × k_panel cols (row-major, transposed conceptually)
     * C: MR × NR output tile (row-major, ldc stride)
     *
     * @param A_panel Packed A panel (MR × k_panel)
     * @param B_panel Packed B panel (NR × k_panel)
     * @param C Output tile (MR × NR with stride ldc)
     * @param ldc Leading dimension of C
     * @param k_panel Panel width (K dimension)
     * @param alpha Scaling factor for A*B
     * @param beta Scaling factor for existing C
     * @param mr Actual rows to process (≤ MR)
     * @param nr Actual cols to process (≤ NR)
     */
    static void micro_kernel(
        const float* A_panel,
        const float* B_panel,
        float* C,
        int ldc,
        int k_panel,
        float alpha,
        float beta,
        int mr,
        int nr)
    {
        // Declare MR × NR accumulator registers
        Vec accumulators[MR][NR];

        // Zero accumulators
        for (int i = 0; i < MR; ++i) {
            for (int j = 0; j < NR; ++j) {
                accumulators[i][j] = Traits::zero();
            }
        }

        // Main K-loop with UNROLL_K unrolling
        constexpr int STEP = kWidth * UNROLL_K;
        int p = 0;

        for (; p + STEP <= k_panel; p += STEP) {
            // Prefetch ahead if enabled
            if constexpr (PREFETCH_DIST > 0) {
                constexpr int PREFETCH_OFFSET = kWidth * UNROLL_K * PREFETCH_DIST;
                if (p + PREFETCH_OFFSET < k_panel) {
                    __builtin_prefetch(A_panel + 0 * k_panel + p + PREFETCH_OFFSET, 0, 1);
                    __builtin_prefetch(B_panel + 0 * k_panel + p + PREFETCH_OFFSET, 0, 1);
                }
            }

            // Unrolled K iterations
            #pragma GCC unroll 4
            for (int u = 0; u < UNROLL_K; ++u) {
                int offset = p + u * kWidth;

                // Load A rows and B cols
                Vec a_vecs[MR];
                Vec b_vecs[NR];

                for (int i = 0; i < MR; ++i) {
                    a_vecs[i] = Traits::load(A_panel + i * k_panel + offset);
                }

                for (int j = 0; j < NR; ++j) {
                    b_vecs[j] = Traits::load(B_panel + j * k_panel + offset);
                }

                // Outer product: accumulators[i][j] += a_vecs[i] * b_vecs[j]
                for (int i = 0; i < MR; ++i) {
                    for (int j = 0; j < NR; ++j) {
                        accumulators[i][j] = Traits::fmadd(a_vecs[i], b_vecs[j], accumulators[i][j]);
                    }
                }
            }
        }

        // Cleanup loop: Handle remaining k_panel % STEP
        for (; p + kWidth <= k_panel; p += kWidth) {
            Vec a_vecs[MR];
            Vec b_vecs[NR];

            for (int i = 0; i < MR; ++i) {
                a_vecs[i] = Traits::load(A_panel + i * k_panel + p);
            }

            for (int j = 0; j < NR; ++j) {
                b_vecs[j] = Traits::load(B_panel + j * k_panel + p);
            }

            for (int i = 0; i < MR; ++i) {
                for (int j = 0; j < NR; ++j) {
                    accumulators[i][j] = Traits::fmadd(a_vecs[i], b_vecs[j], accumulators[i][j]);
                }
            }
        }

        // Horizontal reduction to scalars
        float c_scalar[MR][NR];
        for (int i = 0; i < MR; ++i) {
            for (int j = 0; j < NR; ++j) {
                c_scalar[i][j] = Traits::reduce_add(accumulators[i][j]);
            }
        }

        // Tail handling: Process remaining k_panel % kWidth elements with scalar code
        for (int k = (k_panel / kWidth) * kWidth; k < k_panel; ++k) {
            for (int i = 0; i < MR; ++i) {
                float a_val = A_panel[i * k_panel + k];
                for (int j = 0; j < NR; ++j) {
                    c_scalar[i][j] += a_val * B_panel[j * k_panel + k];
                }
            }
        }

        // Write back to C with alpha/beta scaling (only valid mr×nr region)
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nr; ++j) {
                C[i * ldc + j] = alpha * c_scalar[i][j] + beta * C[i * ldc + j];
            }
        }
    }

    /**
     * @brief Pack A panel from row-major to contiguous format
     */
    static void pack_A_panel(
        const float* A,
        float* A_packed,
        int m_panel,
        int k_panel,
        int lda)
    {
        for (int i = 0; i < m_panel; ++i) {
            const float* A_row = A + i * lda;
            float* A_packed_row = A_packed + i * k_panel;

            // Prefetch next row
            if (i + 1 < m_panel) {
                __builtin_prefetch(A + (i + 1) * lda, 0, 3);
            }

            // Copy with unrolling
            int k = 0;
            constexpr int UNROLL = 16;
            for (; k + UNROLL <= k_panel; k += UNROLL) {
                #pragma GCC unroll 16
                for (int kk = 0; kk < UNROLL; ++kk) {
                    A_packed_row[k + kk] = A_row[k + kk];
                }
            }
            for (; k < k_panel; ++k) {
                A_packed_row[k] = A_row[k];
            }
        }
    }

    /**
     * @brief Pack B panel from column-major to row-major contiguous format
     */
    static void pack_B_panel(
        const float* B_decoded,
        float* B_packed,
        int k_panel,
        int n_panel,
        int ldb)
    {
        for (int j = 0; j < n_panel; ++j) {
            const float* B_col = B_decoded + j * ldb;
            float* B_packed_col = B_packed + j * k_panel;

            // Prefetch next column
            if (j + 1 < n_panel) {
                __builtin_prefetch(B_decoded + (j + 1) * ldb, 0, 3);
            }

            // Copy with unrolling
            int k = 0;
            constexpr int UNROLL = 16;
            for (; k + UNROLL <= k_panel; k += UNROLL) {
                #pragma GCC unroll 16
                for (int kk = 0; kk < UNROLL; ++kk) {
                    B_packed_col[k + kk] = B_col[k + kk];
                }
            }
            for (; k < k_panel; ++k) {
                B_packed_col[k] = B_col[k];
            }
        }
    }
};

} // namespace gemm
} // namespace kernels
} // namespace llaminar2
