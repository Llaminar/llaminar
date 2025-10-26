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
            float *B_decoded = new (std::align_val_t(64)) float[k * NC]; // Decoded B columns

// Outer loop: Iterate over N dimension (columns of C)
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

                                // Use beta only for first K-panel (kc == 0)
                                float beta_use = (kc == 0) ? beta : 1.0f;
                                micro_kernel(A_block, B_block, C_block, n, kc_size,
                                             alpha, beta_use);
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
        float alpha, float beta)
    {
#if defined(__AVX512F__)
        // AVX512 micro-kernel: 8 rows × 6 columns (48 accumulators in 48 ZMM registers)
        // We have 32 ZMM registers, so we can hold 6 accumulators per row (6 × 8 = 48)

        __m512 c_00 = _mm512_setzero_ps();
        __m512 c_01 = _mm512_setzero_ps();
        __m512 c_02 = _mm512_setzero_ps();
        __m512 c_03 = _mm512_setzero_ps();
        __m512 c_04 = _mm512_setzero_ps();
        __m512 c_05 = _mm512_setzero_ps();

        __m512 c_10 = _mm512_setzero_ps();
        __m512 c_11 = _mm512_setzero_ps();
        __m512 c_12 = _mm512_setzero_ps();
        __m512 c_13 = _mm512_setzero_ps();
        __m512 c_14 = _mm512_setzero_ps();
        __m512 c_15 = _mm512_setzero_ps();

        // Process k_panel in chunks of 16 (AVX512 vector width)
        for (int p = 0; p < k_panel; p += 16)
        {
            int p_count = std::min(16, k_panel - p);

            // Load A (broadcast each element across vector)
            __m512 a0 = (p_count == 16) ? _mm512_loadu_ps(A_panel + 0 * k_panel + p)
                                        : _mm512_setzero_ps();
            __m512 a1 = (p_count == 16) ? _mm512_loadu_ps(A_panel + 1 * k_panel + p)
                                        : _mm512_setzero_ps();

            // Load B columns (6 columns, each 16-wide)
            __m512 b0 = (p_count == 16) ? _mm512_loadu_ps(B_panel + 0 * k_panel + p)
                                        : _mm512_setzero_ps();
            __m512 b1 = (p_count == 16) ? _mm512_loadu_ps(B_panel + 1 * k_panel + p)
                                        : _mm512_setzero_ps();
            __m512 b2 = (p_count == 16) ? _mm512_loadu_ps(B_panel + 2 * k_panel + p)
                                        : _mm512_setzero_ps();
            __m512 b3 = (p_count == 16) ? _mm512_loadu_ps(B_panel + 3 * k_panel + p)
                                        : _mm512_setzero_ps();
            __m512 b4 = (p_count == 16) ? _mm512_loadu_ps(B_panel + 4 * k_panel + p)
                                        : _mm512_setzero_ps();
            __m512 b5 = (p_count == 16) ? _mm512_loadu_ps(B_panel + 5 * k_panel + p)
                                        : _mm512_setzero_ps();

            // FMA: C[i,j] += A[i,p] * B[p,j]
            c_00 = _mm512_fmadd_ps(a0, b0, c_00);
            c_01 = _mm512_fmadd_ps(a0, b1, c_01);
            c_02 = _mm512_fmadd_ps(a0, b2, c_02);
            c_03 = _mm512_fmadd_ps(a0, b3, c_03);
            c_04 = _mm512_fmadd_ps(a0, b4, c_04);
            c_05 = _mm512_fmadd_ps(a0, b5, c_05);

            c_10 = _mm512_fmadd_ps(a1, b0, c_10);
            c_11 = _mm512_fmadd_ps(a1, b1, c_11);
            c_12 = _mm512_fmadd_ps(a1, b2, c_12);
            c_13 = _mm512_fmadd_ps(a1, b3, c_13);
            c_14 = _mm512_fmadd_ps(a1, b4, c_14);
            c_15 = _mm512_fmadd_ps(a1, b5, c_15);
        }

        // Horizontal reduction (sum across vector lanes)
        float c_scalar[8][6];
        c_scalar[0][0] = _mm512_reduce_add_ps(c_00);
        c_scalar[0][1] = _mm512_reduce_add_ps(c_01);
        c_scalar[0][2] = _mm512_reduce_add_ps(c_02);
        c_scalar[0][3] = _mm512_reduce_add_ps(c_03);
        c_scalar[0][4] = _mm512_reduce_add_ps(c_04);
        c_scalar[0][5] = _mm512_reduce_add_ps(c_05);

        c_scalar[1][0] = _mm512_reduce_add_ps(c_10);
        c_scalar[1][1] = _mm512_reduce_add_ps(c_11);
        c_scalar[1][2] = _mm512_reduce_add_ps(c_12);
        c_scalar[1][3] = _mm512_reduce_add_ps(c_13);
        c_scalar[1][4] = _mm512_reduce_add_ps(c_14);
        c_scalar[1][5] = _mm512_reduce_add_ps(c_15);

        // Write back to C (apply alpha/beta)
        for (int i = 0; i < MR && i < 2; ++i)
        {
            for (int j = 0; j < NR; ++j)
            {
                C[i * ldc + j] = alpha * c_scalar[i][j] + beta * C[i * ldc + j];
            }
        }

#else
        // Scalar fallback (no SIMD)
        for (int i = 0; i < MR; ++i)
        {
            for (int j = 0; j < NR; ++j)
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
        // Pack A from row-major to panel format (contiguous MR × KC blocks)
        for (int i = 0; i < m_panel; ++i)
        {
            const float *A_row = A + i * lda;
            float *A_packed_row = A_packed + i * k_panel;
            std::memcpy(A_packed_row, A_row, k_panel * sizeof(float));
        }
    }

    void QuantizedGemmL1Opt::pack_B_panel(
        const float *B_decoded, float *B_packed,
        int k_panel, int n_panel)
    {
        // Pack B from column-major to panel format (NR columns × KC rows)
        for (int j = 0; j < n_panel; ++j)
        {
            const float *B_col = B_decoded + j * k_panel; // Assuming k stride
            float *B_packed_col = B_packed + j * k_panel;
            std::memcpy(B_packed_col, B_col, k_panel * sizeof(float));
        }
    }

} // namespace llaminar2
