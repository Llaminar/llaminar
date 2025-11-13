/**
 * @file RegisterBlockedInt8Gemm.h
 * @brief Register-blocked INT8 GEMM kernel following OneDNN architecture
 *
 * Phase 1: Implements 6×16 microkernel with register blocking
 * Target: 2,000-3,000 GOPS (4-6× speedup over baseline)
 *
 * Architecture:
 * - 6×16 output tile held in 6 ZMM registers (96 INT32)
 * - AVX512 VNNI instruction for 4-way dot product
 * - Minimal memory traffic: Load C once, accumulate, store once
 *
 * @author David Sanftenberg
 * @date November 12, 2025
 */

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {
            namespace cpu
            {

                /**
                 * @class RegisterBlockedInt8Gemm
                 * @brief INT8 GEMM with 6×16 register blocking (OneDNN-inspired)
                 *
                 * Computes: C[M×N] += A[M×K] * B[K×N]
                 * Where: A is INT8 row-major, B is INT8 column-major, C is INT32 row-major
                 *
                 * Key features:
                 * - 6×16 microkernel: 96 INT32 outputs in 6 ZMM registers
                 * - VNNI optimization: vpdpbusd instruction (4×INT8→INT32 in one op)
                 * - Register reuse: Load C once per tile, accumulate in registers, store once
                 * - Outer tiling: Process M×N in 6×16 tiles
                 * - B-panel packing: Reformat B into cache-friendly [K/4][16][4] layout
                 */
                class RegisterBlockedInt8Gemm
                {
                public:
                    // Microkernel tile dims
                    static constexpr int MR = 6;  // rows per tile
                    static constexpr int NR = 16; // cols per tile

                    /**
                     * @brief Pack B panel for VNNI: layout [K/4][16][4] (64 bytes per K-block).
                     * Each K-block stores, for every column j in 0..15, the 4 consecutive B bytes
                     * B[j, k+0], B[j, k+1], B[j, k+2], B[j, k+3]. This enables a single 64B load
                     * feeding all 16 dpbusd lanes (one 4-byte dot per column) per K-step.
                     *
                     * OneDNN approach: Convert signed→unsigned during packing (XOR with 0x80).
                     * This eliminates the need for compensation during computation.
                     */
                    static void pack_B_panel_vnni(const int8_t *__restrict__ B_src,
                                                  int8_t *__restrict__ B_packed,
                                                  int K,
                                                  int ldb,
                                                  bool convert_s8_to_u8 = true)
                    {
                        // Require K multiple of 4 for fast path
                        for (int k = 0; k < K; k += 4)
                        {
                            int block_index = (k / 4) * 64; // 64 bytes per block
                            int8_t *dst = B_packed + block_index;
                            // For each column j write 4 consecutive bytes
                            for (int j = 0; j < NR; ++j)
                            {
                                for (int kk = 0; kk < 4; ++kk)
                                {
                                    int8_t val = B_src[j * ldb + k + kk];
                                    // OneDNN: XOR with 0x80 to convert signed→unsigned during packing
                                    if (convert_s8_to_u8)
                                    {
                                        val ^= static_cast<int8_t>(0x80);
                                    }
                                    dst[j * 4 + kk] = val;
                                }
                            }
                        }
                    }

                    /**
                     * @brief 6×16 dpbusd microkernel following OneDNN pattern (conversion during packing).
                     *
                     * OneDNN approach: B is converted to unsigned during packing (XOR 0x80).
                     * Result needs compensation: actual = computed - sum(A_row) * 128 * K_elements
                     * OneDNN precomputes these offsets; we compute inline after k-loop.
                     *
                     * Layout assumptions:
                     *  - A rows are contiguous in memory (row-major) so 4 bytes can be loaded as int32.
                     *  - B is packed AND CONVERTED by pack_B_panel_vnni() (64 bytes per K-step, unsigned range).
                     */
                    static void microkernel_6x16_dpbusd(const int8_t *__restrict__ A,
                                                        const int8_t *__restrict__ Bpack,
                                                        int32_t *__restrict__ C,
                                                        int K,
                                                        int lda,
                                                        int ldc)
                    {
                        // 6×16 = 96 int32 outputs in 6 ZMM accumulators
                        __m512i c0 = _mm512_setzero_si512();
                        __m512i c1 = _mm512_setzero_si512();
                        __m512i c2 = _mm512_setzero_si512();
                        __m512i c3 = _mm512_setzero_si512();
                        __m512i c4 = _mm512_setzero_si512();
                        __m512i c5 = _mm512_setzero_si512();

                        // Accumulators for sum(A) per row (for compensation) - vector accumulators
                        __m512i sum_a0 = _mm512_setzero_si512();
                        __m512i sum_a1 = _mm512_setzero_si512();
                        __m512i sum_a2 = _mm512_setzero_si512();
                        __m512i sum_a3 = _mm512_setzero_si512();
                        __m512i sum_a4 = _mm512_setzero_si512();
                        __m512i sum_a5 = _mm512_setzero_si512();

                        // Constant for compensation dpbusd
                        const __m512i ones_u8 = _mm512_set1_epi8(1);

                        // Main loop: process 4 K-blocks (16 elements) per iteration for maximum ILP
                        int k = 0;
                        for (; k + 15 < K; k += 16)
                        {
                            // Load all 4 B blocks
                            __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bpack + (k / 4 + 0) * 64));
                            __m512i b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bpack + (k / 4 + 1) * 64));
                            __m512i b2 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bpack + (k / 4 + 2) * 64));
                            __m512i b3 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bpack + (k / 4 + 3) * 64));

                            // === K-block 0 (k+0 to k+3) ===
                            int32_t a0_dw0 = *reinterpret_cast<const int32_t *>(A + 0 * lda + k);
                            int32_t a1_dw0 = *reinterpret_cast<const int32_t *>(A + 1 * lda + k);
                            int32_t a2_dw0 = *reinterpret_cast<const int32_t *>(A + 2 * lda + k);
                            int32_t a3_dw0 = *reinterpret_cast<const int32_t *>(A + 3 * lda + k);
                            int32_t a4_dw0 = *reinterpret_cast<const int32_t *>(A + 4 * lda + k);
                            int32_t a5_dw0 = *reinterpret_cast<const int32_t *>(A + 5 * lda + k);

                            __m512i a0_v0 = _mm512_set1_epi32(a0_dw0);
                            __m512i a1_v0 = _mm512_set1_epi32(a1_dw0);
                            __m512i a2_v0 = _mm512_set1_epi32(a2_dw0);
                            __m512i a3_v0 = _mm512_set1_epi32(a3_dw0);
                            __m512i a4_v0 = _mm512_set1_epi32(a4_dw0);
                            __m512i a5_v0 = _mm512_set1_epi32(a5_dw0);

                            c0 = _mm512_dpbusd_epi32(c0, b0, a0_v0);
                            c1 = _mm512_dpbusd_epi32(c1, b0, a1_v0);
                            c2 = _mm512_dpbusd_epi32(c2, b0, a2_v0);
                            c3 = _mm512_dpbusd_epi32(c3, b0, a3_v0);
                            c4 = _mm512_dpbusd_epi32(c4, b0, a4_v0);
                            c5 = _mm512_dpbusd_epi32(c5, b0, a5_v0);

                            sum_a0 = _mm512_dpbusd_epi32(sum_a0, ones_u8, a0_v0);
                            sum_a1 = _mm512_dpbusd_epi32(sum_a1, ones_u8, a1_v0);
                            sum_a2 = _mm512_dpbusd_epi32(sum_a2, ones_u8, a2_v0);
                            sum_a3 = _mm512_dpbusd_epi32(sum_a3, ones_u8, a3_v0);
                            sum_a4 = _mm512_dpbusd_epi32(sum_a4, ones_u8, a4_v0);
                            sum_a5 = _mm512_dpbusd_epi32(sum_a5, ones_u8, a5_v0);

                            // === K-block 1 (k+4 to k+7) ===
                            int32_t a0_dw1 = *reinterpret_cast<const int32_t *>(A + 0 * lda + k + 4);
                            int32_t a1_dw1 = *reinterpret_cast<const int32_t *>(A + 1 * lda + k + 4);
                            int32_t a2_dw1 = *reinterpret_cast<const int32_t *>(A + 2 * lda + k + 4);
                            int32_t a3_dw1 = *reinterpret_cast<const int32_t *>(A + 3 * lda + k + 4);
                            int32_t a4_dw1 = *reinterpret_cast<const int32_t *>(A + 4 * lda + k + 4);
                            int32_t a5_dw1 = *reinterpret_cast<const int32_t *>(A + 5 * lda + k + 4);

                            __m512i a0_v1 = _mm512_set1_epi32(a0_dw1);
                            __m512i a1_v1 = _mm512_set1_epi32(a1_dw1);
                            __m512i a2_v1 = _mm512_set1_epi32(a2_dw1);
                            __m512i a3_v1 = _mm512_set1_epi32(a3_dw1);
                            __m512i a4_v1 = _mm512_set1_epi32(a4_dw1);
                            __m512i a5_v1 = _mm512_set1_epi32(a5_dw1);

                            c0 = _mm512_dpbusd_epi32(c0, b1, a0_v1);
                            c1 = _mm512_dpbusd_epi32(c1, b1, a1_v1);
                            c2 = _mm512_dpbusd_epi32(c2, b1, a2_v1);
                            c3 = _mm512_dpbusd_epi32(c3, b1, a3_v1);
                            c4 = _mm512_dpbusd_epi32(c4, b1, a4_v1);
                            c5 = _mm512_dpbusd_epi32(c5, b1, a5_v1);

                            sum_a0 = _mm512_dpbusd_epi32(sum_a0, ones_u8, a0_v1);
                            sum_a1 = _mm512_dpbusd_epi32(sum_a1, ones_u8, a1_v1);
                            sum_a2 = _mm512_dpbusd_epi32(sum_a2, ones_u8, a2_v1);
                            sum_a3 = _mm512_dpbusd_epi32(sum_a3, ones_u8, a3_v1);
                            sum_a4 = _mm512_dpbusd_epi32(sum_a4, ones_u8, a4_v1);
                            sum_a5 = _mm512_dpbusd_epi32(sum_a5, ones_u8, a5_v1);

                            // === K-block 2 (k+8 to k+11) ===
                            int32_t a0_dw2 = *reinterpret_cast<const int32_t *>(A + 0 * lda + k + 8);
                            int32_t a1_dw2 = *reinterpret_cast<const int32_t *>(A + 1 * lda + k + 8);
                            int32_t a2_dw2 = *reinterpret_cast<const int32_t *>(A + 2 * lda + k + 8);
                            int32_t a3_dw2 = *reinterpret_cast<const int32_t *>(A + 3 * lda + k + 8);
                            int32_t a4_dw2 = *reinterpret_cast<const int32_t *>(A + 4 * lda + k + 8);
                            int32_t a5_dw2 = *reinterpret_cast<const int32_t *>(A + 5 * lda + k + 8);

                            __m512i a0_v2 = _mm512_set1_epi32(a0_dw2);
                            __m512i a1_v2 = _mm512_set1_epi32(a1_dw2);
                            __m512i a2_v2 = _mm512_set1_epi32(a2_dw2);
                            __m512i a3_v2 = _mm512_set1_epi32(a3_dw2);
                            __m512i a4_v2 = _mm512_set1_epi32(a4_dw2);
                            __m512i a5_v2 = _mm512_set1_epi32(a5_dw2);

                            c0 = _mm512_dpbusd_epi32(c0, b2, a0_v2);
                            c1 = _mm512_dpbusd_epi32(c1, b2, a1_v2);
                            c2 = _mm512_dpbusd_epi32(c2, b2, a2_v2);
                            c3 = _mm512_dpbusd_epi32(c3, b2, a3_v2);
                            c4 = _mm512_dpbusd_epi32(c4, b2, a4_v2);
                            c5 = _mm512_dpbusd_epi32(c5, b2, a5_v2);

                            sum_a0 = _mm512_dpbusd_epi32(sum_a0, ones_u8, a0_v2);
                            sum_a1 = _mm512_dpbusd_epi32(sum_a1, ones_u8, a1_v2);
                            sum_a2 = _mm512_dpbusd_epi32(sum_a2, ones_u8, a2_v2);
                            sum_a3 = _mm512_dpbusd_epi32(sum_a3, ones_u8, a3_v2);
                            sum_a4 = _mm512_dpbusd_epi32(sum_a4, ones_u8, a4_v2);
                            sum_a5 = _mm512_dpbusd_epi32(sum_a5, ones_u8, a5_v2);

                            // === K-block 3 (k+12 to k+15) ===
                            int32_t a0_dw3 = *reinterpret_cast<const int32_t *>(A + 0 * lda + k + 12);
                            int32_t a1_dw3 = *reinterpret_cast<const int32_t *>(A + 1 * lda + k + 12);
                            int32_t a2_dw3 = *reinterpret_cast<const int32_t *>(A + 2 * lda + k + 12);
                            int32_t a3_dw3 = *reinterpret_cast<const int32_t *>(A + 3 * lda + k + 12);
                            int32_t a4_dw3 = *reinterpret_cast<const int32_t *>(A + 4 * lda + k + 12);
                            int32_t a5_dw3 = *reinterpret_cast<const int32_t *>(A + 5 * lda + k + 12);

                            __m512i a0_v3 = _mm512_set1_epi32(a0_dw3);
                            __m512i a1_v3 = _mm512_set1_epi32(a1_dw3);
                            __m512i a2_v3 = _mm512_set1_epi32(a2_dw3);
                            __m512i a3_v3 = _mm512_set1_epi32(a3_dw3);
                            __m512i a4_v3 = _mm512_set1_epi32(a4_dw3);
                            __m512i a5_v3 = _mm512_set1_epi32(a5_dw3);

                            c0 = _mm512_dpbusd_epi32(c0, b3, a0_v3);
                            c1 = _mm512_dpbusd_epi32(c1, b3, a1_v3);
                            c2 = _mm512_dpbusd_epi32(c2, b3, a2_v3);
                            c3 = _mm512_dpbusd_epi32(c3, b3, a3_v3);
                            c4 = _mm512_dpbusd_epi32(c4, b3, a4_v3);
                            c5 = _mm512_dpbusd_epi32(c5, b3, a5_v3);

                            sum_a0 = _mm512_dpbusd_epi32(sum_a0, ones_u8, a0_v3);
                            sum_a1 = _mm512_dpbusd_epi32(sum_a1, ones_u8, a1_v3);
                            sum_a2 = _mm512_dpbusd_epi32(sum_a2, ones_u8, a2_v3);
                            sum_a3 = _mm512_dpbusd_epi32(sum_a3, ones_u8, a3_v3);
                            sum_a4 = _mm512_dpbusd_epi32(sum_a4, ones_u8, a4_v3);
                            sum_a5 = _mm512_dpbusd_epi32(sum_a5, ones_u8, a5_v3);
                        }

                        // Remainder loop: process remaining K-blocks (4 elements) one at a time
                        for (; k < K; k += 4)
                        {
                            // Load packed B: [K/4][16][4] → single 64B load per K-block
                            // B is already converted (unsigned) from packing
                            __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bpack + (k / 4) * 64));

                            // Load A for all 6 rows (4 bytes per row): A[i, k:k+4]
                            int32_t a0_dword = *reinterpret_cast<const int32_t *>(A + 0 * lda + k);
                            int32_t a1_dword = *reinterpret_cast<const int32_t *>(A + 1 * lda + k);
                            int32_t a2_dword = *reinterpret_cast<const int32_t *>(A + 2 * lda + k);
                            int32_t a3_dword = *reinterpret_cast<const int32_t *>(A + 3 * lda + k);
                            int32_t a4_dword = *reinterpret_cast<const int32_t *>(A + 4 * lda + k);
                            int32_t a5_dword = *reinterpret_cast<const int32_t *>(A + 5 * lda + k);

                            // Broadcast each A dword across all 16 lanes
                            __m512i a0 = _mm512_set1_epi32(a0_dword);
                            __m512i a1 = _mm512_set1_epi32(a1_dword);
                            __m512i a2 = _mm512_set1_epi32(a2_dword);
                            __m512i a3 = _mm512_set1_epi32(a3_dword);
                            __m512i a4 = _mm512_set1_epi32(a4_dword);
                            __m512i a5 = _mm512_set1_epi32(a5_dword);

                            // vpdpbusd(acc, src1_u8, src2_s8): acc += unsigned(src1) * signed(src2)
                            // B is unsigned (converted during packing), A is signed
                            c0 = _mm512_dpbusd_epi32(c0, b, a0);
                            c1 = _mm512_dpbusd_epi32(c1, b, a1);
                            c2 = _mm512_dpbusd_epi32(c2, b, a2);
                            c3 = _mm512_dpbusd_epi32(c3, b, a3);
                            c4 = _mm512_dpbusd_epi32(c4, b, a4);
                            c5 = _mm512_dpbusd_epi32(c5, b, a5);

                            // Accumulate sum(A) using dpbusd with ones vector
                            // This computes sum of the 4 bytes in each A dword
                            sum_a0 = _mm512_dpbusd_epi32(sum_a0, ones_u8, a0);
                            sum_a1 = _mm512_dpbusd_epi32(sum_a1, ones_u8, a1);
                            sum_a2 = _mm512_dpbusd_epi32(sum_a2, ones_u8, a2);
                            sum_a3 = _mm512_dpbusd_epi32(sum_a3, ones_u8, a3);
                            sum_a4 = _mm512_dpbusd_epi32(sum_a4, ones_u8, a4);
                            sum_a5 = _mm512_dpbusd_epi32(sum_a5, ones_u8, a5);
                        }

                        // Reduce sum_a vectors to scalars (horizontal sum across 16 lanes)
                        // Each lane has the same value (from broadcast), so just extract one
                        int32_t total_sum_a0 = _mm512_reduce_add_epi32(sum_a0) / 16; // Divide by 16 lanes
                        int32_t total_sum_a1 = _mm512_reduce_add_epi32(sum_a1) / 16;
                        int32_t total_sum_a2 = _mm512_reduce_add_epi32(sum_a2) / 16;
                        int32_t total_sum_a3 = _mm512_reduce_add_epi32(sum_a3) / 16;
                        int32_t total_sum_a4 = _mm512_reduce_add_epi32(sum_a4) / 16;
                        int32_t total_sum_a5 = _mm512_reduce_add_epi32(sum_a5) / 16;

                        // Apply compensation: subtract sum(A_row) * 128
                        // Since B was converted via XOR 0x80, we computed A × (B + 128) = A×B + A×128
                        __m512i comp0 = _mm512_set1_epi32(total_sum_a0 * 128);
                        __m512i comp1 = _mm512_set1_epi32(total_sum_a1 * 128);
                        __m512i comp2 = _mm512_set1_epi32(total_sum_a2 * 128);
                        __m512i comp3 = _mm512_set1_epi32(total_sum_a3 * 128);
                        __m512i comp4 = _mm512_set1_epi32(total_sum_a4 * 128);
                        __m512i comp5 = _mm512_set1_epi32(total_sum_a5 * 128);

                        c0 = _mm512_sub_epi32(c0, comp0);
                        c1 = _mm512_sub_epi32(c1, comp1);
                        c2 = _mm512_sub_epi32(c2, comp2);
                        c3 = _mm512_sub_epi32(c3, comp3);
                        c4 = _mm512_sub_epi32(c4, comp4);
                        c5 = _mm512_sub_epi32(c5, comp5);

                        // Store results
                        _mm512_storeu_si512(reinterpret_cast<__m512i *>(C + 0 * ldc), c0);
                        _mm512_storeu_si512(reinterpret_cast<__m512i *>(C + 1 * ldc), c1);
                        _mm512_storeu_si512(reinterpret_cast<__m512i *>(C + 2 * ldc), c2);
                        _mm512_storeu_si512(reinterpret_cast<__m512i *>(C + 3 * ldc), c3);
                        _mm512_storeu_si512(reinterpret_cast<__m512i *>(C + 4 * ldc), c4);
                        _mm512_storeu_si512(reinterpret_cast<__m512i *>(C + 5 * ldc), c5);
                    }

                    /**
                     * @brief Top-level GEMM driver using 6×16 dpbusd microkernel.
                     * Fallbacks to scalar computation for edge tiles (partial M/N) or K not multiple of 4.
                     */
                    static void gemm(int M, int N, int K,
                                     const int8_t *__restrict__ A, int lda,
                                     const int8_t *__restrict__ B, int ldb,
                                     int32_t *__restrict__ C, int ldc)
                    {
                        // Two-level tiling (N outer) with per-panel B packing.
                        for (int j = 0; j < N; j += NR)
                        {
                            int n_block = std::min(N - j, NR);
                            bool full_n = (n_block == NR);
                            // Allocate and pack B panel if full width and aligned K
                            std::unique_ptr<int8_t[]> Bpack;
                            if (full_n && (K % 4 == 0))
                            {
                                Bpack.reset(new int8_t[(size_t)K * NR]); // K*16 bytes
                                // Convert during packing (XOR with 0x80)
                                pack_B_panel_vnni(B + j * ldb, Bpack.get(), K, ldb, true);
                            }
                            for (int i = 0; i < M; i += MR)
                            {
                                int m_block = std::min(M - i, MR);
                                bool full_m = (m_block == MR);
                                if (full_m && full_n && (K % 4 == 0))
                                {
                                    microkernel_6x16_dpbusd(A + i * lda, Bpack.get(), C + i * ldc + j, K, lda, ldc);
                                }
                                else
                                {
                                    // Scalar fallback for partial tiles or K misalignment
                                    for (int ii = 0; ii < m_block; ++ii)
                                    {
                                        for (int jj = 0; jj < n_block; ++jj)
                                        {
                                            int32_t sum = 0;
                                            for (int k = 0; k < K; ++k)
                                            {
                                                sum += static_cast<int32_t>(A[(i + ii) * lda + k]) *
                                                       static_cast<int32_t>(B[(j + jj) * ldb + k]);
                                            }
                                            C[(i + ii) * ldc + (j + jj)] = sum;
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

            } // namespace cpu
        } // namespace kernels
    } // namespace v2
} // namespace llaminar
