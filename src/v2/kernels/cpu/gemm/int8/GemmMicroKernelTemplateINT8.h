/**
 * @file GemmMicroKernelTemplateINT8.h
 * @brief INT8×INT8→INT32 micro-kernel template using AVX512 VNNI
 *
 * This template implements high-performance INT8 GEMM using the VNNI instruction:
 *   _mm512_dpbusd_epi32(src, a, b)
 *
 * Key features:
 * - Accumulates in int32 registers (not float)
 * - Uses dpbusd instruction (4-way dot product, unsigned * signed) with a signed→unsigned bias fix-up
 * - Input data is int8 (A: signed original, B: signed). We bias A by +128 to make it unsigned.
 * - Requires VNNI-friendly packing (groups of 4)
 * - Returns INT32 results directly (NO dequantization)
 * - Caller must dequantize INT32→FP32 separately if needed
 *
 * This design allows flexibility: INT32 results can be accumulated across
 * multiple operations before final dequantization, reducing rounding errors
 * and enabling fused operations.
 *
 * Explicit instantiations are generated in generated/ directory.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../SimdTraits.h"
#include <cstring>
#include <algorithm>
#include <iostream>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief INT8 GEMM micro-kernel template for AVX512 VNNI
             *
             * @tparam ISA Must be AVX512VNNITag
             * @tparam MR Micro-kernel M dimension (rows in register block)
             * @tparam NR Micro-kernel N dimension (cols in register block)
             * @tparam UNROLL_K K-loop unroll factor (1, 2, 4, 8, 16)
             * @tparam PREFETCH_DIST Prefetch distance in iterations (0, 1, 2, 3, 5)
             * @tparam MC M-dimension cache block size
             * @tparam KC K-dimension cache block size (must be multiple of 4 for VNNI)
             * @tparam NC N-dimension cache block size
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
            class MicroKernelTemplateINT8
            {
            public:
                using Traits = simd::SimdTraits<ISA>;
                using Vec = typename Traits::VectorType;  // __m512i (64 int8s)
                using Accum = typename Traits::AccumType; // __m512i (16 int32s)

                static constexpr int kVecWidth = Traits::vector_width;       // 64 int8s
                static constexpr int kAccumWidth = Traits::accum_width;      // 16 int32s
                static constexpr int kDotGroupSize = Traits::dot_group_size; // 4 elements per dot product

                static constexpr int TILE_M = MR;
                static constexpr int TILE_N = NR;

                // Ensure KC is divisible by 4 for VNNI efficiency
                static_assert(KC % 4 == 0, "KC must be divisible by 4 for VNNI packing");

                /**
                 * @brief INT8 micro-kernel: Compute C[MR×NR] += A[MR×k_panel] * B[NR×k_panel]^T
                 *
                 * Uses AVX512 VNNI (vpdpbusd) for efficient INT8×INT8→INT32 computation.
                 *
                 * VNNI processes data in groups of 4 INT8 elements:
                 *   dst[lane] += src1[4*lane+0:3] · src2[4*lane+0:3]  (4-way dot product per lane)
                 *
                 * For GEMM C[i,j] = sum_k A[i,k]*B[j,k], we iterate K in steps of 4,
                 * computing one 4-way dot product per iteration and accumulating.
                 *
                 * @param A_panel Packed A panel (MR × k_panel, int8, signed values in [-128,127])
                 * @param B_panel Packed B panel (NR × k_panel, int8, signed values in [-128,127])
                 * @param C Output tile (MR × NR, int32, with stride ldc)
                 * @param ldc Leading dimension of C
                 * @param k_panel Panel width (K dimension, preferably multiple of 4)
                 * @param alpha Scaling factor for A*B (typically 1, applied as int32)
                 * @param beta Scaling factor for existing C (0 or 1)
                 * @param mr Actual rows to process (≤ MR)
                 * @param nr Actual cols to process (≤ NR)
                 */
                static void micro_kernel(
                    const int8_t *A_panel,
                    const int8_t *B_panel,
                    int32_t *C,
                    int ldc,
                    int k_panel,
                    int32_t alpha,
                    int32_t beta,
                    int mr,
                    int nr)
                {
                    // Environment-controlled diagnostic flags
                    bool debug_vnni = (std::getenv("LLAMINAR_INT8_VNNI_DEBUG") != nullptr);
                    bool force_scalar = (std::getenv("LLAMINAR_INT8_VNNI_FORCE_SCALAR") != nullptr);

                    // Fast path: pure scalar reference (for isolating packing vs intrinsic issues)
                    if (force_scalar)
                    {
                        for (int i = 0; i < mr; ++i)
                        {
                            for (int j = 0; j < nr; ++j)
                            {
                                int32_t sum = 0;
                                for (int k = 0; k < k_panel; ++k)
                                {
                                    sum += static_cast<int32_t>(A_panel[i * k_panel + k]) *
                                           static_cast<int32_t>(B_panel[j * k_panel + k]);
                                }
                                if (beta == 0)
                                    C[i * ldc + j] = alpha * sum;
                                else
                                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                            }
                        }
                        return;
                    }

                    // AVX512 VNNI path: dpbusd with signed→unsigned bias correction
                    //
                    // Strategy: VNNI's dpbusd computes unsigned(A) * signed(B), but our inputs are both signed.
                    // We transform: A_signed → A_unsigned = A_signed + 128
                    // Then: dpbusd(acc, A_unsigned, B_signed) = acc + Σ(A_signed * B_signed) + 128 * Σ(B_signed)
                    // Final result: dpbusd_result - 128 * Σ(B_signed) per lane
                    //
                    // Each __m512i holds 64 int8 values → 16 lanes × 4 elements/lane
                    // Each dpbusd produces 16 int32 accumulators (one per 4-element dot product)

                    // Accumulators: one __m512i per (i,j) tile element, each holding 16 int32 lanes
                    // Always zero SIMD accumulators (they accumulate within this K-panel)
                    Accum accumulators[MR][NR];
                    for (int i = 0; i < MR; ++i)
                        for (int j = 0; j < NR; ++j)
                            accumulators[i][j] = Traits::zero_i32();

                    constexpr int STEP = kVecWidth * UNROLL_K; // Process UNROLL_K×64 bytes of K per outer iteration
                    int p = 0;

                    // Main unrolled K-loop (processes 64×UNROLL_K int8 elements per iteration)
                    for (; p + STEP <= k_panel; p += STEP)
                    {
                        if constexpr (PREFETCH_DIST > 0)
                        {
                            constexpr int PREFETCH_OFFSET = kVecWidth * UNROLL_K * PREFETCH_DIST;
                            if (p + PREFETCH_OFFSET < k_panel)
                            {
                                __builtin_prefetch(A_panel + p + PREFETCH_OFFSET, 0, 1);
                                __builtin_prefetch(B_panel + p + PREFETCH_OFFSET, 0, 1);
                            }
                        }

#pragma GCC unroll 4
                        for (int u = 0; u < UNROLL_K; ++u)
                        {
                            int offset = p + u * kVecWidth;
                            Vec a_vecs[MR];
                            Vec b_vecs[NR];
                            for (int i = 0; i < MR; ++i)
                                a_vecs[i] = Traits::load_i8(A_panel + i * k_panel + offset);
                            for (int j = 0; j < NR; ++j)
                                b_vecs[j] = Traits::load_i8(B_panel + j * k_panel + offset);

                            // Apply +128 bias to convert A from signed to unsigned domain
                            const __m512i bias128 = _mm512_set1_epi8((char)0x80);
                            __m512i a_biased[MR];
                            for (int i = 0; i < MR; ++i)
                                a_biased[i] = _mm512_add_epi8(a_vecs[i], bias128);

                            // Compute correction vector: 128 * Σ(B_lane) for each of 16 lanes
                            alignas(64) int8_t b_bytes[64];
                            for (int j = 0; j < NR; ++j)
                            {
                                _mm512_storeu_si512(reinterpret_cast<__m512i *>(b_bytes), b_vecs[j]);

                                // Calculate per-lane sum of B (4 elements per lane)
                                int32_t b_lane_sum[16];
                                for (int lane = 0; lane < 16; ++lane)
                                {
                                    int base = lane * 4;
                                    b_lane_sum[lane] = (int)b_bytes[base + 0] + (int)b_bytes[base + 1] +
                                                       (int)b_bytes[base + 2] + (int)b_bytes[base + 3];
                                }

                                // Build correction vector (128 * sum_b_lane for each lane)
                                alignas(64) int32_t corr_lanes[16];
                                for (int lane = 0; lane < 16; ++lane)
                                    corr_lanes[lane] = 128 * b_lane_sum[lane];
                                __m512i corr_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(corr_lanes));

                                // Accumulate: dpbusd(A_unsigned, B_signed) - correction
                                for (int i = 0; i < MR; ++i)
                                {
                                    accumulators[i][j] = Traits::dpbusd(accumulators[i][j], a_biased[i], b_vecs[j]);
                                    accumulators[i][j] = _mm512_sub_epi32(accumulators[i][j], corr_vec);
                                }
                            }
                        }
                    }

                    // Remainder K-loop (single 64-byte chunk per iteration)
                    for (; p + kVecWidth <= k_panel; p += kVecWidth)
                    {
                        Vec a_vecs[MR];
                        Vec b_vecs[NR];
                        for (int i = 0; i < MR; ++i)
                            a_vecs[i] = Traits::load_i8(A_panel + i * k_panel + p);
                        for (int j = 0; j < NR; ++j)
                            b_vecs[j] = Traits::load_i8(B_panel + j * k_panel + p);

                        const __m512i bias128 = _mm512_set1_epi8((char)0x80);
                        __m512i a_biased[MR];
                        for (int i = 0; i < MR; ++i)
                            a_biased[i] = _mm512_add_epi8(a_vecs[i], bias128);

                        alignas(64) int8_t b_bytes[64];
                        for (int j = 0; j < NR; ++j)
                        {
                            _mm512_storeu_si512(reinterpret_cast<__m512i *>(b_bytes), b_vecs[j]);

                            int32_t b_lane_sum[16];
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                int base = lane * 4;
                                b_lane_sum[lane] = (int)b_bytes[base + 0] + (int)b_bytes[base + 1] +
                                                   (int)b_bytes[base + 2] + (int)b_bytes[base + 3];
                            }

                            alignas(64) int32_t corr_lanes[16];
                            for (int lane = 0; lane < 16; ++lane)
                                corr_lanes[lane] = 128 * b_lane_sum[lane];
                            __m512i corr_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(corr_lanes));

                            for (int i = 0; i < MR; ++i)
                            {
                                accumulators[i][j] = Traits::dpbusd(accumulators[i][j], a_biased[i], b_vecs[j]);
                                accumulators[i][j] = _mm512_sub_epi32(accumulators[i][j], corr_vec);
                            }
                        }
                    }

                    // Reduce accumulators: sum all 16 lanes (each lane = one 4-element dot product)
                    // CRITICAL FIX (Nov 11, 2025): Initialize c_scalar with existing C when beta != 0
                    // This enables true accumulation across K-blocks without double buffering!
                    int32_t c_scalar[MR][NR];
                    for (int i = 0; i < MR; ++i)
                    {
                        for (int j = 0; j < NR; ++j)
                        {
                            // When beta != 0, start with existing C value (accumulation mode)
                            // When beta == 0, start from zero (overwrite mode)
                            int32_t sum = (beta != 0) ? C[i * ldc + j] : 0;

                            // Add horizontal reduction of SIMD lanes
                            for (int lane = 0; lane < kAccumWidth; ++lane)
                                sum += Traits::extract_i32(accumulators[i][j], lane);
                            c_scalar[i][j] = sum;
                        }
                    }

                    // Scalar tail for leftover K elements (< 64)
                    for (int k = p; k < k_panel; ++k)
                        for (int i = 0; i < MR; ++i)
                        {
                            int8_t a_val = A_panel[i * k_panel + k];
                            for (int j = 0; j < NR; ++j)
                                c_scalar[i][j] += static_cast<int32_t>(a_val) *
                                                  static_cast<int32_t>(B_panel[j * k_panel + k]);
                        }

                    // Optional debug: verify C(0,0) against scalar reference
                    if (debug_vnni)
                    {
                        int32_t ref = 0;
                        for (int k = 0; k < k_panel; ++k)
                            ref += static_cast<int32_t>(A_panel[k]) * static_cast<int32_t>(B_panel[k]);

                        int32_t computed = c_scalar[0][0];
                        if (ref == computed)
                            std::cerr << "[INT8_VNNI_DEBUG] ok C(0,0) ref=" << ref
                                      << " lane_sel=0 lane_val=" << Traits::extract_i32(accumulators[0][0], 0) << std::endl;
                        else
                            std::cerr << "[INT8_VNNI_DEBUG] mismatch C(0,0) ref=" << ref
                                      << " computed=" << computed << " diff=" << (computed - ref) << std::endl;
                    }

                    // Write results to output matrix C
                    // CRITICAL FIX (Nov 11, 2025): c_scalar already includes beta*C when beta != 0!
                    // Formula: c_scalar = beta*C_old + horizontal_sum(A*B)
                    // We need: C_new = alpha * horizontal_sum(A*B) + beta * C_old
                    //                = alpha * (c_scalar - beta*C_old) + beta * C_old
                    // But when beta=1 and alpha=1: C_new = c_scalar (simple!)
                    for (int i = 0; i < mr; ++i)
                        for (int j = 0; j < nr; ++j)
                        {
                            if (beta == 0)
                            {
                                // c_scalar = horizontal_sum, no C_old
                                C[i * ldc + j] = alpha * c_scalar[i][j];
                            }
                            else if (alpha == 1 && beta == 1)
                            {
                                // c_scalar = C_old + horizontal_sum, exactly what we want!
                                C[i * ldc + j] = c_scalar[i][j];
                            }
                            else
                            {
                                // General case: extract horizontal_sum from c_scalar
                                int32_t horizontal_sum = c_scalar[i][j] - beta * C[i * ldc + j];
                                C[i * ldc + j] = alpha * horizontal_sum + beta * C[i * ldc + j];
                            }
                        }
                }

                /**
                 * @brief Pack A panel from row-major to VNNI-friendly format
                 *
                 * VNNI requires data in groups of 4 for optimal dpbusd performance.
                 * Input: int8_t A[m][k] (row-major)
                 * Output: int8_t A_packed[m][k] (contiguous, VNNI-friendly)
                 *
                 * @param A Source matrix (row-major, int8)
                 * @param A_packed Destination packed buffer
                 * @param m_panel Number of rows to pack
                 * @param k_panel Number of columns to pack (should be multiple of 4)
                 * @param lda Leading dimension of A
                 */
                static void pack_A_panel(
                    const int8_t *A,
                    int8_t *A_packed,
                    int m_panel,
                    int k_panel,
                    int lda)
                {
                    for (int i = 0; i < m_panel; ++i)
                    {
                        const int8_t *A_row = A + i * lda;
                        int8_t *A_packed_row = A_packed + i * k_panel;

                        // Prefetch next row
                        if (i + 1 < m_panel)
                        {
                            __builtin_prefetch(A + (i + 1) * lda, 0, 3);
                        }

                        // Copy with unrolling (groups of 64 for VNNI efficiency)
                        int k = 0;
                        constexpr int UNROLL = 64;
                        for (; k + UNROLL <= k_panel; k += UNROLL)
                        {
#pragma GCC unroll 64
                            for (int kk = 0; kk < UNROLL; ++kk)
                            {
                                A_packed_row[k + kk] = A_row[k + kk];
                            }
                        }
                        // Remainder
                        for (; k < k_panel; ++k)
                        {
                            A_packed_row[k] = A_row[k];
                        }
                    }
                }

                /**
                 * @brief Pack B panel from column-major to VNNI-friendly row-major format
                 *
                 * Input: int8_t B[k][n] (column-major conceptually, but stored as decoded)
                 * Output: int8_t B_packed[n][k] (row-major, VNNI-friendly)
                 *
                 * @param B_decoded Source matrix (decoded int8)
                 * @param B_packed Destination packed buffer
                 * @param k_panel K dimension
                 * @param n_panel N dimension
                 * @param ldb Leading dimension of B (stride between columns)
                 */
                static void pack_B_panel(
                    const int8_t *B_decoded,
                    int8_t *B_packed,
                    int k_panel,
                    int n_panel,
                    int ldb)
                {
                    for (int j = 0; j < n_panel; ++j)
                    {
                        const int8_t *B_col = B_decoded + j * ldb;
                        int8_t *B_packed_col = B_packed + j * k_panel;

                        // Prefetch next column
                        if (j + 1 < n_panel)
                        {
                            __builtin_prefetch(B_decoded + (j + 1) * ldb, 0, 3);
                        }

                        // Copy with unrolling
                        int k = 0;
                        constexpr int UNROLL = 64;
                        for (; k + UNROLL <= k_panel; k += UNROLL)
                        {
#pragma GCC unroll 64
                            for (int kk = 0; kk < UNROLL; ++kk)
                            {
                                B_packed_col[k + kk] = B_col[k + kk];
                            }
                        }
                        // Remainder
                        for (; k < k_panel; ++k)
                        {
                            B_packed_col[k] = B_col[k];
                        }
                    }
                }
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
