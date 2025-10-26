/**
 * @file QuantizedGemmL1Opt.cpp
 * @brief L1 cache-optimized quantized GEMM implementation
 *
 * @author David Sanftenberg
 */

#include "QuantizedGemmL1Opt.h"
#include <cstring>
#include <iostream>

namespace llaminar2
{

    bool QuantizedGemmL1Opt::multiply(
        const float *A, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        if (!decoder_)
        {
            return false;
        }

        // Validate dimensions
        int expected_cols = transpose_B ? k : n;
        if (static_cast<int>(decoder_->decoder_cols()) != expected_cols)
        {
            return false;
        }

        const size_t BLOCK_SIZE = decoder_->block_size();
        const int num_k_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Allocate packed buffers (thread-local in parallel region)
        // A_packed: MC × KC panel (contiguous for better cache behavior)
        // B_packed: KC × NC panel (decoded and packed)

#pragma omp parallel
        {
        // Thread-local packed buffers
        float *A_packed = new (std::align_val_t(64)) float[MC * KC];
        float *B_packed = new (std::align_val_t(64)) float[KC * NC];
        float *B_decoded = new (std::align_val_t(64)) float[k * NC]; // Decoded B columns (reduced size)// Outer loop: Iterate over N dimension (columns of C)
#pragma omp for schedule(dynamic)
            for (int jc = 0; jc < n; jc += NC)
            {
                int nc = std::min(NC, n - jc);

                // Decode NC columns of B (reuse across all M rows)
                for (int j_local = 0; j_local < nc; ++j_local)
                {
                    int j = jc + j_local;
                    float *B_col = B_decoded + j_local * k;

                    // Decode all K-blocks for this column
                    for (int kb = 0; kb < num_k_blocks; ++kb)
                    {
                        size_t k_start = kb * BLOCK_SIZE;
                        decoder_->decode_block_at(j, kb, B_col + k_start);
                        
                        // Prefetch next block's output location
                        if (kb + 1 < num_k_blocks) {
                            __builtin_prefetch(B_col + k_start + BLOCK_SIZE, 1, 3);
                        }
                    }
                }

                // Middle loop: Iterate over K dimension (panel blocking)
                for (int kc = 0; kc < k; kc += KC)
                {
                    int kc_size = std::min(KC, k - kc);

                    // Pack B panel: KC × NC (column-major to row-major for better access)
                    pack_B_panel(B_decoded + kc, B_packed, kc_size, nc);

                    // Inner loop: Iterate over M dimension (rows of C)
                    for (int ic = 0; ic < m; ic += MC)
                    {
                        int mc = std::min(MC, m - ic);

                        // Pack A panel: MC × KC (row-major to packed format)
                        pack_A_panel(A + ic * k + kc, A_packed, mc, kc_size, k);

                        // Prefetch next M-panel of A (if exists)
                        if (ic + MC < m && kc == 0) {
                            for (int pf_i = 0; pf_i < std::min(8, m - ic - MC); ++pf_i) {
                                __builtin_prefetch(A + (ic + MC + pf_i) * k, 0, 2);
                            }
                        }

                        // Micro-kernel loop: Process MC × NC in MR × NR chunks
                        for (int ir = 0; ir < mc; ir += MR)
                        {
                            int mr = std::min(MR, mc - ir);

                            for (int jr = 0; jr < nc; jr += NR)
                            {
                                int nr = std::min(NR, nc - jr);

                                // Compute C[ir:ir+mr, jr:jr+nr] += A_packed * B_packed
                                float *C_block = C + (ic + ir) * n + (jc + jr);
                                const float *A_block = A_packed + ir * kc_size;
                                const float *B_block = B_packed + jr * kc_size;

                                // Prefetch next B_block
                                if (jr + NR < nc) {
                                    __builtin_prefetch(B_packed + (jr + NR) * kc_size, 0, 3);
                                }

                                // Use beta only for first K-panel (kc == 0)
                                float beta_use = (kc == 0) ? beta : 1.0f;
                                micro_kernel(A_block, B_block, C_block, n, kc_size,
                                             alpha, beta_use, mr, nr);
                            }
                        }
                    }
                }
            }

            // Free thread-local buffers
            operator delete[](A_packed, std::align_val_t(64));
            operator delete[](B_packed, std::align_val_t(64));
            operator delete[](B_decoded, std::align_val_t(64));
        }

        return true;
    }

    void QuantizedGemmL1Opt::micro_kernel(
        const float *A_panel, const float *B_panel,
        float *C, int ldc, int k_panel,
        float alpha, float beta, int mr, int nr)
    {
#if defined(__AVX512F__)
        // AVX512 micro-kernel: 8 rows × 6 columns (48 accumulators in ZMM registers)
        // Manual unrolling for maximum ILP (instruction-level parallelism)

        __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps(), c02 = _mm512_setzero_ps();
        __m512 c03 = _mm512_setzero_ps(), c04 = _mm512_setzero_ps(), c05 = _mm512_setzero_ps();
        __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps();
        __m512 c13 = _mm512_setzero_ps(), c14 = _mm512_setzero_ps(), c15 = _mm512_setzero_ps();
        __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps();
        __m512 c23 = _mm512_setzero_ps(), c24 = _mm512_setzero_ps(), c25 = _mm512_setzero_ps();
        __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps();
        __m512 c33 = _mm512_setzero_ps(), c34 = _mm512_setzero_ps(), c35 = _mm512_setzero_ps();
        __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps(), c42 = _mm512_setzero_ps();
        __m512 c43 = _mm512_setzero_ps(), c44 = _mm512_setzero_ps(), c45 = _mm512_setzero_ps();
        __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps(), c52 = _mm512_setzero_ps();
        __m512 c53 = _mm512_setzero_ps(), c54 = _mm512_setzero_ps(), c55 = _mm512_setzero_ps();
        __m512 c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps(), c62 = _mm512_setzero_ps();
        __m512 c63 = _mm512_setzero_ps(), c64 = _mm512_setzero_ps(), c65 = _mm512_setzero_ps();
        __m512 c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps(), c72 = _mm512_setzero_ps();
        __m512 c73 = _mm512_setzero_ps(), c74 = _mm512_setzero_ps(), c75 = _mm512_setzero_ps();

        // Process k_panel in chunks of 16 (AVX512 vector width)
        for (int p = 0; p < k_panel; p += 16)
        {
            if (p + 16 > k_panel) break; // Handle tail with scalar code
            
            // Load 8 rows of A
            __m512 a0 = _mm512_loadu_ps(A_panel + 0 * k_panel + p);
            __m512 a1 = _mm512_loadu_ps(A_panel + 1 * k_panel + p);
            __m512 a2 = _mm512_loadu_ps(A_panel + 2 * k_panel + p);
            __m512 a3 = _mm512_loadu_ps(A_panel + 3 * k_panel + p);
            __m512 a4 = _mm512_loadu_ps(A_panel + 4 * k_panel + p);
            __m512 a5 = _mm512_loadu_ps(A_panel + 5 * k_panel + p);
            __m512 a6 = _mm512_loadu_ps(A_panel + 6 * k_panel + p);
            __m512 a7 = _mm512_loadu_ps(A_panel + 7 * k_panel + p);
            
            // Load 6 columns of B
            __m512 b0 = _mm512_loadu_ps(B_panel + 0 * k_panel + p);
            __m512 b1 = _mm512_loadu_ps(B_panel + 1 * k_panel + p);
            __m512 b2 = _mm512_loadu_ps(B_panel + 2 * k_panel + p);
            __m512 b3 = _mm512_loadu_ps(B_panel + 3 * k_panel + p);
            __m512 b4 = _mm512_loadu_ps(B_panel + 4 * k_panel + p);
            __m512 b5 = _mm512_loadu_ps(B_panel + 5 * k_panel + p);
            
            // 48 FMA operations (8 rows × 6 cols)
            c00 = _mm512_fmadd_ps(a0, b0, c00); c01 = _mm512_fmadd_ps(a0, b1, c01);
            c02 = _mm512_fmadd_ps(a0, b2, c02); c03 = _mm512_fmadd_ps(a0, b3, c03);
            c04 = _mm512_fmadd_ps(a0, b4, c04); c05 = _mm512_fmadd_ps(a0, b5, c05);
            
            c10 = _mm512_fmadd_ps(a1, b0, c10); c11 = _mm512_fmadd_ps(a1, b1, c11);
            c12 = _mm512_fmadd_ps(a1, b2, c12); c13 = _mm512_fmadd_ps(a1, b3, c13);
            c14 = _mm512_fmadd_ps(a1, b4, c14); c15 = _mm512_fmadd_ps(a1, b5, c15);
            
            c20 = _mm512_fmadd_ps(a2, b0, c20); c21 = _mm512_fmadd_ps(a2, b1, c21);
            c22 = _mm512_fmadd_ps(a2, b2, c22); c23 = _mm512_fmadd_ps(a2, b3, c23);
            c24 = _mm512_fmadd_ps(a2, b4, c24); c25 = _mm512_fmadd_ps(a2, b5, c25);
            
            c30 = _mm512_fmadd_ps(a3, b0, c30); c31 = _mm512_fmadd_ps(a3, b1, c31);
            c32 = _mm512_fmadd_ps(a3, b2, c32); c33 = _mm512_fmadd_ps(a3, b3, c33);
            c34 = _mm512_fmadd_ps(a3, b4, c34); c35 = _mm512_fmadd_ps(a3, b5, c35);
            
            c40 = _mm512_fmadd_ps(a4, b0, c40); c41 = _mm512_fmadd_ps(a4, b1, c41);
            c42 = _mm512_fmadd_ps(a4, b2, c42); c43 = _mm512_fmadd_ps(a4, b3, c43);
            c44 = _mm512_fmadd_ps(a4, b4, c44); c45 = _mm512_fmadd_ps(a4, b5, c45);
            
            c50 = _mm512_fmadd_ps(a5, b0, c50); c51 = _mm512_fmadd_ps(a5, b1, c51);
            c52 = _mm512_fmadd_ps(a5, b2, c52); c53 = _mm512_fmadd_ps(a5, b3, c53);
            c54 = _mm512_fmadd_ps(a5, b4, c54); c55 = _mm512_fmadd_ps(a5, b5, c55);
            
            c60 = _mm512_fmadd_ps(a6, b0, c60); c61 = _mm512_fmadd_ps(a6, b1, c61);
            c62 = _mm512_fmadd_ps(a6, b2, c62); c63 = _mm512_fmadd_ps(a6, b3, c63);
            c64 = _mm512_fmadd_ps(a6, b4, c64); c65 = _mm512_fmadd_ps(a6, b5, c65);
            
            c70 = _mm512_fmadd_ps(a7, b0, c70); c71 = _mm512_fmadd_ps(a7, b1, c71);
            c72 = _mm512_fmadd_ps(a7, b2, c72); c73 = _mm512_fmadd_ps(a7, b3, c73);
            c74 = _mm512_fmadd_ps(a7, b4, c74); c75 = _mm512_fmadd_ps(a7, b5, c75);
        }

        // Horizontal reduction (sum across vector lanes)
        float c_scalar[8][6];
        c_scalar[0][0] = _mm512_reduce_add_ps(c00); c_scalar[0][1] = _mm512_reduce_add_ps(c01);
        c_scalar[0][2] = _mm512_reduce_add_ps(c02); c_scalar[0][3] = _mm512_reduce_add_ps(c03);
        c_scalar[0][4] = _mm512_reduce_add_ps(c04); c_scalar[0][5] = _mm512_reduce_add_ps(c05);
        
        c_scalar[1][0] = _mm512_reduce_add_ps(c10); c_scalar[1][1] = _mm512_reduce_add_ps(c11);
        c_scalar[1][2] = _mm512_reduce_add_ps(c12); c_scalar[1][3] = _mm512_reduce_add_ps(c13);
        c_scalar[1][4] = _mm512_reduce_add_ps(c14); c_scalar[1][5] = _mm512_reduce_add_ps(c15);
        
        c_scalar[2][0] = _mm512_reduce_add_ps(c20); c_scalar[2][1] = _mm512_reduce_add_ps(c21);
        c_scalar[2][2] = _mm512_reduce_add_ps(c22); c_scalar[2][3] = _mm512_reduce_add_ps(c23);
        c_scalar[2][4] = _mm512_reduce_add_ps(c24); c_scalar[2][5] = _mm512_reduce_add_ps(c25);
        
        c_scalar[3][0] = _mm512_reduce_add_ps(c30); c_scalar[3][1] = _mm512_reduce_add_ps(c31);
        c_scalar[3][2] = _mm512_reduce_add_ps(c32); c_scalar[3][3] = _mm512_reduce_add_ps(c33);
        c_scalar[3][4] = _mm512_reduce_add_ps(c34); c_scalar[3][5] = _mm512_reduce_add_ps(c35);
        
        c_scalar[4][0] = _mm512_reduce_add_ps(c40); c_scalar[4][1] = _mm512_reduce_add_ps(c41);
        c_scalar[4][2] = _mm512_reduce_add_ps(c42); c_scalar[4][3] = _mm512_reduce_add_ps(c43);
        c_scalar[4][4] = _mm512_reduce_add_ps(c44); c_scalar[4][5] = _mm512_reduce_add_ps(c45);
        
        c_scalar[5][0] = _mm512_reduce_add_ps(c50); c_scalar[5][1] = _mm512_reduce_add_ps(c51);
        c_scalar[5][2] = _mm512_reduce_add_ps(c52); c_scalar[5][3] = _mm512_reduce_add_ps(c53);
        c_scalar[5][4] = _mm512_reduce_add_ps(c54); c_scalar[5][5] = _mm512_reduce_add_ps(c55);
        
        c_scalar[6][0] = _mm512_reduce_add_ps(c60); c_scalar[6][1] = _mm512_reduce_add_ps(c61);
        c_scalar[6][2] = _mm512_reduce_add_ps(c62); c_scalar[6][3] = _mm512_reduce_add_ps(c63);
        c_scalar[6][4] = _mm512_reduce_add_ps(c64); c_scalar[6][5] = _mm512_reduce_add_ps(c65);
        
        c_scalar[7][0] = _mm512_reduce_add_ps(c70); c_scalar[7][1] = _mm512_reduce_add_ps(c71);
        c_scalar[7][2] = _mm512_reduce_add_ps(c72); c_scalar[7][3] = _mm512_reduce_add_ps(c73);
        c_scalar[7][4] = _mm512_reduce_add_ps(c74); c_scalar[7][5] = _mm512_reduce_add_ps(c75);
        
        // Handle tail elements (k_panel % 16) with scalar code
        int p_remainder = (k_panel / 16) * 16;
        for (int p = p_remainder; p < k_panel; ++p) {
            for (int i = 0; i < 8; ++i) {
                float a_val = A_panel[i * k_panel + p];
                for (int j = 0; j < 6; ++j) {
                    c_scalar[i][j] += a_val * B_panel[j * k_panel + p];
                }
            }
        }

        // Write back to C (apply alpha/beta) - only write valid mr×nr region
        for (int i = 0; i < mr; ++i)
        {
            for (int j = 0; j < nr; ++j)
            {
                C[i * ldc + j] = alpha * c_scalar[i][j] + beta * C[i * ldc + j];
            }
        }

#else
        // Scalar fallback (no SIMD)
        for (int i = 0; i < mr; ++i)
        {
            for (int j = 0; j < nr; ++j)
            {
                float sum = 0.0f;
                for (int p = 0; p < k_panel; ++p)
                {
                    sum += A_panel[i * k_panel + p] * B_panel[j * k_panel + p];
                }
                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
            }
        }
#endif
    }

    void QuantizedGemmL1Opt::pack_A_panel(
        const float *A, float *A_packed,
        int m_panel, int k_panel, int lda)
    {
        // Pack A from row-major to panel format with prefetching
        for (int i = 0; i < m_panel; ++i)
        {
            const float *A_row = A + i * lda;
            float *A_packed_row = A_packed + i * k_panel;
            
            // Prefetch next row
            if (i + 1 < m_panel) {
                __builtin_prefetch(A + (i + 1) * lda, 0, 3);
            }
            
            // Manual copy with 16-element unrolling for better cache line utilization
            int k = 0;
            for (; k + 16 <= k_panel; k += 16) {
                #pragma GCC unroll 16
                for (int kk = 0; kk < 16; ++kk) {
                    A_packed_row[k + kk] = A_row[k + kk];
                }
            }
            // Handle remainder
            for (; k < k_panel; ++k) {
                A_packed_row[k] = A_row[k];
            }
        }
    }

    void QuantizedGemmL1Opt::pack_B_panel(
        const float *B_decoded, float *B_packed,
        int k_panel, int n_panel)
    {
        // Pack B from column-major to panel format with prefetching
        for (int j = 0; j < n_panel; ++j)
        {
            const float *B_col = B_decoded + j * k_panel;
            float *B_packed_col = B_packed + j * k_panel;
            
            // Prefetch next column
            if (j + 1 < n_panel) {
                __builtin_prefetch(B_decoded + (j + 1) * k_panel, 0, 3);
            }
            
            // Manual copy with 16-element unrolling
            int k = 0;
            for (; k + 16 <= k_panel; k += 16) {
                #pragma GCC unroll 16
                for (int kk = 0; kk < 16; ++kk) {
                    B_packed_col[k + kk] = B_col[k + kk];
                }
            }
            // Handle remainder
            for (; k < k_panel; ++k) {
                B_packed_col[k] = B_col[k];
            }
        }
    }

} // namespace llaminar2
