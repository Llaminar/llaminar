/**
 * @file Microbench__WoTranspose.cpp
 * @brief Benchmark comparing row-major vs column-major (transposed) Wo GEMV
 *
 * This benchmark tests the hypothesis that pre-transposing Wo weights
 * before GEMV will significantly improve memory access patterns and
 * thus performance for decode (M=1) operations.
 *
 * Usage:
 *   ./v2_microbench_wo_transpose [--d_model N] [--iterations N]
 */

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>

namespace
{

    // Align to cache line
    constexpr size_t CACHE_LINE = 64;

    template <typename T>
    T *aligned_alloc_array(size_t count)
    {
        // std::aligned_alloc requires size to be a multiple of alignment.
        const size_t bytes = count * sizeof(T);
        const size_t rounded_bytes = ((bytes + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE;
        void *ptr = std::aligned_alloc(CACHE_LINE, rounded_bytes);
        return static_cast<T *>(ptr);
    }

    /**
     * @brief Row-major GEMV: output[i] = sum_k(context[k] * Wo[i, k])
     *
     * Wo is [rows, cols] row-major, so Wo[i, k] = Wo[i * cols + k]
     * This is the CURRENT implementation pattern.
     */
    void gemv_rowmajor_naive(
        const float *context, // [cols]
        const float *Wo,      // [rows, cols] row-major
        float *output,        // [rows]
        int rows,
        int cols)
    {
        for (int i = 0; i < rows; ++i)
        {
            float acc = 0.0f;
            const float *wo_row = Wo + i * cols;
            for (int k = 0; k < cols; ++k)
            {
                acc += context[k] * wo_row[k];
            }
            output[i] = acc;
        }
    }

    /**
     * @brief Row-major GEMV with AVX-512 vectorization
     *
     * Processes 4 output rows at a time (MR=4) to reuse context loads.
     * This is similar to emit_gemv_wox_rowmajor_fp32.
     */
    void gemv_rowmajor_avx512(
        const float *context,
        const float *Wo,
        float *output,
        int rows,
        int cols)
    {
        constexpr int MR = 4; // Process 4 output rows at a time

        int i = 0;
        for (; i + MR <= rows; i += MR)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            const float *wo0 = Wo + (i + 0) * cols;
            const float *wo1 = Wo + (i + 1) * cols;
            const float *wo2 = Wo + (i + 2) * cols;
            const float *wo3 = Wo + (i + 3) * cols;

            int k = 0;
            for (; k + 16 <= cols; k += 16)
            {
                __m512 ctx = _mm512_loadu_ps(context + k);

                // Each row load is from a different cache line (strided access)
                acc0 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo0 + k), acc0);
                acc1 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo1 + k), acc1);
                acc2 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo2 + k), acc2);
                acc3 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo3 + k), acc3);
            }

            // Horizontal sums
            output[i + 0] = _mm512_reduce_add_ps(acc0);
            output[i + 1] = _mm512_reduce_add_ps(acc1);
            output[i + 2] = _mm512_reduce_add_ps(acc2);
            output[i + 3] = _mm512_reduce_add_ps(acc3);

            // Scalar remainder
            for (; k < cols; ++k)
            {
                output[i + 0] += context[k] * wo0[k];
                output[i + 1] += context[k] * wo1[k];
                output[i + 2] += context[k] * wo2[k];
                output[i + 3] += context[k] * wo3[k];
            }
        }

        // Handle remaining rows
        for (; i < rows; ++i)
        {
            __m512 acc = _mm512_setzero_ps();
            const float *wo_row = Wo + i * cols;
            int k = 0;
            for (; k + 16 <= cols; k += 16)
            {
                __m512 ctx = _mm512_loadu_ps(context + k);
                acc = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_row + k), acc);
            }
            float result = _mm512_reduce_add_ps(acc);
            for (; k < cols; ++k)
            {
                result += context[k] * wo_row[k];
            }
            output[i] = result;
        }
    }

    /**
     * @brief Column-major GEMV: output[i] = sum_k(context[k] * Wo_T[k, i])
     *
     * Wo_T is [cols, rows] column-major (transposed), so Wo_T[k, i] = Wo_T[k * rows + i]
     * This gives SEQUENTIAL access through Wo when iterating over k!
     */
    void gemv_colmajor_naive(
        const float *context, // [cols]
        const float *Wo_T,    // [cols, rows] column-major (= transposed row-major)
        float *output,        // [rows]
        int rows,
        int cols)
    {
        // Zero output
        for (int i = 0; i < rows; ++i)
        {
            output[i] = 0.0f;
        }

        // Outer loop over K (reduction dimension)
        // Each iteration accesses ONE contiguous column of Wo_T
        for (int k = 0; k < cols; ++k)
        {
            float ctx_k = context[k];
            const float *wo_col = Wo_T + k * rows; // Column k is contiguous!
            for (int i = 0; i < rows; ++i)
            {
                output[i] += ctx_k * wo_col[i];
            }
        }
    }

    /**
     * @brief Column-major GEMV (compiler-guided)
     *
     * Same math as gemv_colmajor_naive, but provides the compiler stronger aliasing
     * guarantees (restrict) and an explicit SIMD hint for the inner i-loop.
     *
     * This is intended as a "best effort" baseline that often competes with hand
     * intrinsics when scheduling gets tricky.
     */
    void gemv_colmajor_compiler_guided(
        const float *__restrict context, // [cols]
        const float *__restrict Wo_T,    // [cols, rows] column-major
        float *__restrict output,        // [rows]
        int rows,
        int cols)
    {
        for (int i = 0; i < rows; ++i)
        {
            output[i] = 0.0f;
        }

        for (int k = 0; k < cols; ++k)
        {
            const float ctx_k = context[k];
            const float *__restrict wo_col = Wo_T + static_cast<size_t>(k) * rows;

// Encourage vectorization across output rows.
// (OpenMP is already enabled for this TU; this is a SIMD hint only.)
#pragma omp simd
            for (int i = 0; i < rows; ++i)
            {
                output[i] += ctx_k * wo_col[i];
            }
        }
    }

    /**
     * @brief Column-major GEMV with AVX-512 vectorization
     *
     * Key insight: With column-major Wo, we can vectorize over OUTPUT rows
     * while streaming through K sequentially. Each K iteration loads a
     * contiguous column of 16 floats.
     */
    void gemv_colmajor_avx512(
        const float *context,
        const float *Wo_T, // [cols, rows] column-major
        float *output,
        int rows,
        int cols)
    {
        // Process output in chunks of 64 (4 ZMM registers)
        constexpr int NR = 64;

        int i = 0;
        for (; i + NR <= rows; i += NR)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            // Stream through K dimension - SEQUENTIAL memory access!
            for (int k = 0; k < cols; ++k)
            {
                __m512 ctx = _mm512_set1_ps(context[k]); // Broadcast context[k]

                // Load 64 contiguous floats from column k
                const float *wo_col = Wo_T + k * rows + i;
                acc0 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_col + 0), acc0);
                acc1 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_col + 16), acc1);
                acc2 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_col + 32), acc2);
                acc3 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_col + 48), acc3);
            }

            // Store results
            _mm512_storeu_ps(output + i + 0, acc0);
            _mm512_storeu_ps(output + i + 16, acc1);
            _mm512_storeu_ps(output + i + 32, acc2);
            _mm512_storeu_ps(output + i + 48, acc3);
        }

        // Handle remaining rows (scalar fallback)
        for (; i < rows; ++i)
        {
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k)
            {
                acc += context[k] * Wo_T[k * rows + i];
            }
            output[i] = acc;
        }
    }

    /**
     * @brief Column-major GEMV with AVX-512 vectorization (aligned + K unroll)
     *
     * Same math/layout as gemv_colmajor_avx512, but:
     * - Uses aligned loads/stores (safe with our allocator + 16-float alignment)
     * - Unrolls K by 4 to reduce loop overhead and improve ILP
     *
     * Note: Column-major GEMV inherently needs a broadcast per K element.
     */
    void gemv_colmajor_avx512_unroll4_aligned(
        const float *context,
        const float *Wo_T,
        float *output,
        int rows,
        int cols)
    {
        constexpr int NR = 64;

        int i = 0;
        for (; i + NR <= rows; i += NR)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int k = 0;
            for (; k + 4 <= cols; k += 4)
            {
                const float *w0 = Wo_T + (k + 0) * rows + i;
                const float *w1 = Wo_T + (k + 1) * rows + i;
                const float *w2 = Wo_T + (k + 2) * rows + i;
                const float *w3 = Wo_T + (k + 3) * rows + i;

                const __m512 c0 = _mm512_set1_ps(context[k + 0]);
                const __m512 c1 = _mm512_set1_ps(context[k + 1]);
                const __m512 c2 = _mm512_set1_ps(context[k + 2]);
                const __m512 c3 = _mm512_set1_ps(context[k + 3]);

                // 0
                acc0 = _mm512_fmadd_ps(c0, _mm512_load_ps(w0 + 0), acc0);
                acc1 = _mm512_fmadd_ps(c0, _mm512_load_ps(w0 + 16), acc1);
                acc2 = _mm512_fmadd_ps(c0, _mm512_load_ps(w0 + 32), acc2);
                acc3 = _mm512_fmadd_ps(c0, _mm512_load_ps(w0 + 48), acc3);
                // 1
                acc0 = _mm512_fmadd_ps(c1, _mm512_load_ps(w1 + 0), acc0);
                acc1 = _mm512_fmadd_ps(c1, _mm512_load_ps(w1 + 16), acc1);
                acc2 = _mm512_fmadd_ps(c1, _mm512_load_ps(w1 + 32), acc2);
                acc3 = _mm512_fmadd_ps(c1, _mm512_load_ps(w1 + 48), acc3);
                // 2
                acc0 = _mm512_fmadd_ps(c2, _mm512_load_ps(w2 + 0), acc0);
                acc1 = _mm512_fmadd_ps(c2, _mm512_load_ps(w2 + 16), acc1);
                acc2 = _mm512_fmadd_ps(c2, _mm512_load_ps(w2 + 32), acc2);
                acc3 = _mm512_fmadd_ps(c2, _mm512_load_ps(w2 + 48), acc3);
                // 3
                acc0 = _mm512_fmadd_ps(c3, _mm512_load_ps(w3 + 0), acc0);
                acc1 = _mm512_fmadd_ps(c3, _mm512_load_ps(w3 + 16), acc1);
                acc2 = _mm512_fmadd_ps(c3, _mm512_load_ps(w3 + 32), acc2);
                acc3 = _mm512_fmadd_ps(c3, _mm512_load_ps(w3 + 48), acc3);
            }

            for (; k < cols; ++k)
            {
                const __m512 ctx = _mm512_set1_ps(context[k]);
                const float *wo_col = Wo_T + k * rows + i;
                acc0 = _mm512_fmadd_ps(ctx, _mm512_load_ps(wo_col + 0), acc0);
                acc1 = _mm512_fmadd_ps(ctx, _mm512_load_ps(wo_col + 16), acc1);
                acc2 = _mm512_fmadd_ps(ctx, _mm512_load_ps(wo_col + 32), acc2);
                acc3 = _mm512_fmadd_ps(ctx, _mm512_load_ps(wo_col + 48), acc3);
            }

            _mm512_store_ps(output + i + 0, acc0);
            _mm512_store_ps(output + i + 16, acc1);
            _mm512_store_ps(output + i + 32, acc2);
            _mm512_store_ps(output + i + 48, acc3);
        }

        // Remainder
        for (; i < rows; ++i)
        {
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k)
            {
                acc += context[k] * Wo_T[k * rows + i];
            }
            output[i] = acc;
        }
    }

    /**
     * @brief Column-major GEMV with software prefetching
     */
    void gemv_colmajor_avx512_prefetch(
        const float *context,
        const float *Wo_T,
        float *output,
        int rows,
        int cols)
    {
        constexpr int NR = 64;
        constexpr int PREFETCH_DISTANCE = 8; // Prefetch 8 K iterations ahead

        int i = 0;
        for (; i + NR <= rows; i += NR)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int k = 0; k < cols; ++k)
            {
                // Prefetch future K columns
                if (k + PREFETCH_DISTANCE < cols)
                {
                    const float *pf_ptr = Wo_T + (k + PREFETCH_DISTANCE) * rows + i;
                    _mm_prefetch(reinterpret_cast<const char *>(pf_ptr), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char *>(pf_ptr + 64), _MM_HINT_T0);
                }

                __m512 ctx = _mm512_set1_ps(context[k]);
                const float *wo_col = Wo_T + k * rows + i;

                acc0 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_col + 0), acc0);
                acc1 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_col + 16), acc1);
                acc2 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_col + 32), acc2);
                acc3 = _mm512_fmadd_ps(ctx, _mm512_loadu_ps(wo_col + 48), acc3);
            }

            _mm512_storeu_ps(output + i + 0, acc0);
            _mm512_storeu_ps(output + i + 16, acc1);
            _mm512_storeu_ps(output + i + 32, acc2);
            _mm512_storeu_ps(output + i + 48, acc3);
        }

        // Remainder
        for (; i < rows; ++i)
        {
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k)
            {
                acc += context[k] * Wo_T[k * rows + i];
            }
            output[i] = acc;
        }
    }

    /**
     * @brief Transpose row-major matrix to row-major (true transpose)
     *
     * Input:  src[rows, cols] row-major - element [i,j] at src + i*cols + j
     * Output: dst[cols, rows] row-major - element [j,i] at dst + j*rows + i
     *
     * After transpose: row j of dst = column j of src
     * For GEMV y = W @ x where W[M,K], this gives us W^T[K,M]
     * But we still need W[i,:] for output[i], not W^T[i,:]
     */
    void transpose_matrix(
        const float *src, // [rows, cols] row-major
        float *dst,       // [cols, rows] row-major
        int rows,
        int cols)
    {
        // Simple blocked transpose for cache efficiency
        constexpr int BLOCK = 32;

#pragma omp parallel for collapse(2)
        for (int ii = 0; ii < rows; ii += BLOCK)
        {
            for (int jj = 0; jj < cols; jj += BLOCK)
            {
                int i_end = std::min(ii + BLOCK, rows);
                int j_end = std::min(jj + BLOCK, cols);
                for (int i = ii; i < i_end; ++i)
                {
                    for (int j = jj; j < j_end; ++j)
                    {
                        // dst[j,i] = src[i,j]
                        dst[j * rows + i] = src[i * cols + j];
                    }
                }
            }
        }
    }

    /**
     * @brief Create INTERLEAVED layout for better cache utilization
     *
     * Groups rows that will be processed together (MR=4) into contiguous blocks.
     *
     * Original row-major [M,K]:
     *   row 0: W[0,0], W[0,1], ..., W[0,K-1]   (K elements)
     *   row 1: W[1,0], W[1,1], ..., W[1,K-1]   (K elements)
     *   ...
     *
     * Interleaved [M/MR][K][MR]:
     *   block 0: W[0,0], W[1,0], W[2,0], W[3,0], W[0,1], W[1,1], W[2,1], W[3,1], ...
     *   block 1: W[4,0], W[5,0], W[6,0], W[7,0], W[4,1], W[5,1], W[6,1], W[7,1], ...
     *   ...
     *
     * Now accessing 4 rows at same K is CONTIGUOUS!
     */
    void create_interleaved_layout(
        const float *src, // [rows, cols] row-major
        float *dst,       // [rows/MR, cols, MR] interleaved
        int rows,
        int cols,
        int MR = 4)
    {
#pragma omp parallel for collapse(2)
        for (int block = 0; block < rows / MR; ++block)
        {
            for (int k = 0; k < cols; ++k)
            {
                for (int m = 0; m < MR; ++m)
                {
                    int src_row = block * MR + m;
                    int src_idx = src_row * cols + k;
                    int dst_idx = block * cols * MR + k * MR + m;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
        // Handle remainder rows (not multiple of MR) - copy unchanged
        int remainder_start = (rows / MR) * MR;
        for (int i = remainder_start; i < rows; ++i)
        {
            // These go at the end, row-major
            int dst_offset = (rows / MR) * cols * MR;
            for (int k = 0; k < cols; ++k)
            {
                dst[dst_offset + (i - remainder_start) * cols + k] = src[i * cols + k];
            }
        }
    }

    /**
     * @brief GEMV with INTERLEAVED layout - all 4-row accesses are contiguous
     */
    void gemv_interleaved_avx512(
        const float *context,
        const float *Wo_interleaved, // [M/MR, K, MR] interleaved
        float *output,
        int rows,
        int cols)
    {
        constexpr int MR = 4;

        int block = 0;
        for (int i = 0; i + MR <= rows; i += MR, ++block)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            // Base pointer for this block
            const float *block_base = Wo_interleaved + block * cols * MR;

            int k = 0;
            for (; k + 16 <= cols; k += 16)
            {
                __m512 ctx = _mm512_loadu_ps(context + k);

                // Load 4 interleaved rows at 16 consecutive K positions
                // Layout: [k*MR + 0..3] = W[i+0..3, k], [k*MR + 4..7] = W[i+0..3, k+1], ...
                // For AVX-512, we need to gather or restructure

                // Actually, the interleaved format stores MR elements per K, not 16
                // So we need to load [k*MR : (k+16)*MR] and deinterleave
                // That's 64 floats = 4 ZMM registers
                // Then shuffle to get 4 accumulators updated

                // Simpler: just do 16 iterations of scalar loads and accumulate
                // This loses vectorization benefit...

                // Better approach: process K in scalar, vectorize over output within block
                // Since MR=4, we can't vectorize over 16 outputs in one block
            }

            // Actually, let me try a different approach:
            // Vectorize by loading ctx[k] and 4 values W[i+0:4, k] at once
            // Then use FMA with broadcast context

            // Reset and redo with proper approach
            acc0 = _mm512_setzero_ps(); // For output[i+0]
            acc1 = _mm512_setzero_ps(); // For output[i+1]
            acc2 = _mm512_setzero_ps(); // For output[i+2]
            acc3 = _mm512_setzero_ps(); // For output[i+3]

            for (int kk = 0; kk < cols; kk += 16)
            {
                int k_end = std::min(kk + 16, cols);

                // Load context[kk:kk+16]
                __m512 ctx = _mm512_setzero_ps();
                for (int ki = kk; ki < k_end; ++ki)
                {
                    // For each K, load the 4 interleaved weights and accumulate
                    float c = context[ki];
                    float w0 = block_base[ki * MR + 0];
                    float w1 = block_base[ki * MR + 1];
                    float w2 = block_base[ki * MR + 2];
                    float w3 = block_base[ki * MR + 3];

                    // Scalar accumulation (not vectorized!)
                    output[i + 0] += c * w0;
                    output[i + 1] += c * w1;
                    output[i + 2] += c * w2;
                    output[i + 3] += c * w3;
                }
            }

            // Hmm, this isn't vectorizable the way I structured it...
            // Let me reconsider the interleaved layout
        }
    }

    /**
     * @brief GEMV with interleaved layout - CORRECT vectorization
     *
     * Process K in groups of 16, with weight layout:
     * [block][k/16][16][MR] = 64 floats per (block, k_group)
     *
     * This allows loading 4 ZMM registers that contain:
     * zmm0 = W[i+0, k:k+16], zmm1 = W[i+1, k:k+16], etc.
     *
     * But that's NOT what the current interleaved layout provides...
     * The current layout is [block][k][MR] not [block][k/16][16][MR]
     */
    void gemv_interleaved_avx512_v2(
        const float *context,
        const float *Wo_interleaved,
        float *output,
        int rows,
        int cols)
    {
        // Fall back to simple scalar for correctness
        constexpr int MR = 4;

        int i = 0;
        for (int block = 0; i + MR <= rows; i += MR, ++block)
        {
            float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            const float *block_base = Wo_interleaved + block * cols * MR;

            for (int k = 0; k < cols; ++k)
            {
                float c = context[k];
                acc0 += c * block_base[k * MR + 0];
                acc1 += c * block_base[k * MR + 1];
                acc2 += c * block_base[k * MR + 2];
                acc3 += c * block_base[k * MR + 3];
            }

            output[i + 0] = acc0;
            output[i + 1] = acc1;
            output[i + 2] = acc2;
            output[i + 3] = acc3;
        }

        // Remainder
        for (; i < rows; ++i)
        {
            float acc = 0;
            int rem_idx = (rows / MR) * cols * MR + (i - (rows / MR) * MR) * cols;
            for (int k = 0; k < cols; ++k)
            {
                acc += context[k] * Wo_interleaved[rem_idx + k];
            }
            output[i] = acc;
        }
    }

    /**
     * @brief Verify correctness of column-major GEMV
     *
     * Note: Column-major accumulates across K in different order than row-major,
     * causing small FP rounding differences. Tolerance of 2e-3 (0.2%) is appropriate
     * for d_model=3584 (3584 FP32 accumulations).
     */
    bool verify_results(const float *ref, const float *test, int n, float rtol = 2e-3f)
    {
        for (int i = 0; i < n; ++i)
        {
            float diff = std::abs(ref[i] - test[i]);
            float rel = diff / (std::abs(ref[i]) + 1e-10f);
            if (rel > rtol)
            {
                std::cerr << "Mismatch at " << i << ": ref=" << ref[i]
                          << " test=" << test[i] << " rel_diff=" << rel << std::endl;
                return false;
            }
        }
        return true;
    }

    struct BenchmarkResult
    {
        double time_ms;
        double gflops;
    };

    template <typename Func>
    BenchmarkResult benchmark_gemv(
        Func &&func,
        const float *context,
        const float *Wo,
        float *output,
        int rows,
        int cols,
        int warmup_iters,
        int bench_iters)
    {
        // Warmup
        for (int i = 0; i < warmup_iters; ++i)
        {
            func(context, Wo, output, rows, cols);
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i)
        {
            func(context, Wo, output, rows, cols);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = elapsed_ms / bench_iters;

        // GEMV: 2 * rows * cols FLOPs (multiply + add)
        double flops = 2.0 * rows * cols;
        double gflops = (flops / (avg_ms * 1e-3)) / 1e9;

        return {avg_ms, gflops};
    }

} // anonymous namespace

class WoTransposeBenchmark : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use environment variables or defaults
        d_model_ = 3584; // Qwen 7B
        iterations_ = 100;
        warmup_ = 10;

        if (const char *env = std::getenv("D_MODEL"))
        {
            d_model_ = std::atoi(env);
        }
        if (const char *env = std::getenv("ITERATIONS"))
        {
            iterations_ = std::atoi(env);
        }
    }

    int d_model_;
    int iterations_;
    int warmup_;
};

TEST_F(WoTransposeBenchmark, CompareRowMajorVsColumnMajor)
{
    std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  Wo GEMV Benchmark: Row-Major vs Column-Major (Transposed)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  d_model = " << d_model_ << " (Wo size: "
              << (d_model_ * d_model_ * 4 / 1024 / 1024) << " MB)\n";
    std::cout << "  iterations = " << iterations_ << ", warmup = " << warmup_ << "\n";
    std::cout << "───────────────────────────────────────────────────────────────────\n\n";

    // Allocate matrices
    float *context = aligned_alloc_array<float>(d_model_);
    float *Wo_rowmajor = aligned_alloc_array<float>(d_model_ * d_model_);
    float *Wo_colmajor = aligned_alloc_array<float>(d_model_ * d_model_);
    float *output_ref = aligned_alloc_array<float>(d_model_);
    float *output_test = aligned_alloc_array<float>(d_model_);

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < d_model_; ++i)
    {
        context[i] = dist(rng);
    }
    for (int i = 0; i < d_model_ * d_model_; ++i)
    {
        Wo_rowmajor[i] = dist(rng);
    }

    // Time the transpose (one-time cost)
    auto transpose_start = std::chrono::high_resolution_clock::now();
    transpose_matrix(Wo_rowmajor, Wo_colmajor, d_model_, d_model_);
    auto transpose_end = std::chrono::high_resolution_clock::now();
    double transpose_ms = std::chrono::duration<double, std::milli>(transpose_end - transpose_start).count();

    std::cout << "Transpose time (one-time): " << std::fixed << std::setprecision(2)
              << transpose_ms << " ms\n\n";

    // Create interleaved layout
    float *Wo_interleaved = aligned_alloc_array<float>(d_model_ * d_model_);
    auto interleave_start = std::chrono::high_resolution_clock::now();
    create_interleaved_layout(Wo_rowmajor, Wo_interleaved, d_model_, d_model_, 4);
    auto interleave_end = std::chrono::high_resolution_clock::now();
    double interleave_ms = std::chrono::duration<double, std::milli>(interleave_end - interleave_start).count();

    std::cout << "Interleave time (one-time): " << std::fixed << std::setprecision(2)
              << interleave_ms << " ms\n\n";

    // Compute reference
    gemv_rowmajor_naive(context, Wo_rowmajor, output_ref, d_model_, d_model_);

    // Verify column-major naive gives same result
    gemv_colmajor_naive(context, Wo_colmajor, output_test, d_model_, d_model_);
    ASSERT_TRUE(verify_results(output_ref, output_test, d_model_))
        << "Column-major naive verification failed";

    // Verify column-major compiler-guided gives same result
    gemv_colmajor_compiler_guided(context, Wo_colmajor, output_test, d_model_, d_model_);
    ASSERT_TRUE(verify_results(output_ref, output_test, d_model_))
        << "Column-major compiler-guided verification failed";

    // Verify column-major AVX512 gives same result
    gemv_colmajor_avx512(context, Wo_colmajor, output_test, d_model_, d_model_);
    ASSERT_TRUE(verify_results(output_ref, output_test, d_model_))
        << "Column-major AVX512 verification failed";

    // Verify tuned column-major AVX512 gives same result
    gemv_colmajor_avx512_unroll4_aligned(context, Wo_colmajor, output_test, d_model_, d_model_);
    ASSERT_TRUE(verify_results(output_ref, output_test, d_model_))
        << "Column-major tuned (unroll4) verification failed";

    // Verify interleaved gives same result
    for (int i = 0; i < d_model_; ++i)
        output_test[i] = 0.0f;
    gemv_interleaved_avx512_v2(context, Wo_interleaved, output_test, d_model_, d_model_);
    ASSERT_TRUE(verify_results(output_ref, output_test, d_model_))
        << "Interleaved verification failed";

    std::cout << "Correctness verified ✓\n\n";

    // Benchmark all variants
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "┌─────────────────────────────────┬───────────┬───────────┐\n";
    std::cout << "│ Variant                         │ Time (μs) │ GFLOP/s   │\n";
    std::cout << "├─────────────────────────────────┼───────────┼───────────┤\n";

    // Row-major naive
    auto r1 = benchmark_gemv(gemv_rowmajor_naive, context, Wo_rowmajor, output_test,
                             d_model_, d_model_, warmup_, iterations_);
    std::cout << "│ Row-major (naive)               │ " << std::setw(9) << r1.time_ms * 1000
              << " │ " << std::setw(9) << r1.gflops << " │\n";

    // Row-major AVX512
    auto r2 = benchmark_gemv(gemv_rowmajor_avx512, context, Wo_rowmajor, output_test,
                             d_model_, d_model_, warmup_, iterations_);
    std::cout << "│ Row-major (AVX512, MR=4)        │ " << std::setw(9) << r2.time_ms * 1000
              << " │ " << std::setw(9) << r2.gflops << " │\n";

    // Column-major naive
    auto r3 = benchmark_gemv(gemv_colmajor_naive, context, Wo_colmajor, output_test,
                             d_model_, d_model_, warmup_, iterations_);
    std::cout << "│ Column-major (naive)            │ " << std::setw(9) << r3.time_ms * 1000
              << " │ " << std::setw(9) << r3.gflops << " │\n";

    // Column-major compiler-guided
    auto r3b = benchmark_gemv(gemv_colmajor_compiler_guided, context, Wo_colmajor, output_test,
                              d_model_, d_model_, warmup_, iterations_);
    std::cout << "│ Column-major (compiler-guided)  │ " << std::setw(9) << r3b.time_ms * 1000
              << " │ " << std::setw(9) << r3b.gflops << " │\n";

    // Column-major AVX512
    auto r4 = benchmark_gemv(gemv_colmajor_avx512, context, Wo_colmajor, output_test,
                             d_model_, d_model_, warmup_, iterations_);
    std::cout << "│ Column-major (AVX512, NR=64)    │ " << std::setw(9) << r4.time_ms * 1000
              << " │ " << std::setw(9) << r4.gflops << " │\n";

    // Column-major AVX512 tuned (aligned + K unroll)
    auto r4b = benchmark_gemv(gemv_colmajor_avx512_unroll4_aligned, context, Wo_colmajor, output_test,
                              d_model_, d_model_, warmup_, iterations_);
    std::cout << "│ Column-major (AVX512 tuned)     │ " << std::setw(9) << r4b.time_ms * 1000
              << " │ " << std::setw(9) << r4b.gflops << " │\n";

    // Column-major AVX512 with prefetch
    auto r5 = benchmark_gemv(gemv_colmajor_avx512_prefetch, context, Wo_colmajor, output_test,
                             d_model_, d_model_, warmup_, iterations_);
    std::cout << "│ Column-major (AVX512+prefetch)  │ " << std::setw(9) << r5.time_ms * 1000
              << " │ " << std::setw(9) << r5.gflops << " │\n";

    // Interleaved layout (MR=4 rows contiguous)
    auto r6 = benchmark_gemv(gemv_interleaved_avx512_v2, context, Wo_interleaved, output_test,
                             d_model_, d_model_, warmup_, iterations_);
    std::cout << "│ Interleaved (MR=4, scalar K)    │ " << std::setw(9) << r6.time_ms * 1000
              << " │ " << std::setw(9) << r6.gflops << " │\n";

    std::cout << "└─────────────────────────────────┴───────────┴───────────┘\n\n";

    // Summary
    double best_colmajor = std::max({r3.gflops, r3b.gflops, r4.gflops, r4b.gflops, r5.gflops, r6.gflops});
    double speedup = best_colmajor / r2.gflops;
    std::cout << "Summary:\n";
    std::cout << "  Best row-major:    " << r2.gflops << " GFLOP/s\n";
    std::cout << "  Best column-major: " << best_colmajor << " GFLOP/s\n";
    std::cout << "  Speedup:           " << speedup << "x\n";
    std::cout << "  Transpose cost:    " << transpose_ms << " ms\n";
    std::cout << "  Interleave cost:   " << interleave_ms << " ms\n";

    // Cleanup
    std::free(context);
    std::free(Wo_rowmajor);
    std::free(Wo_colmajor);
    std::free(Wo_interleaved);
    std::free(output_ref);
    std::free(output_test);
}

TEST_F(WoTransposeBenchmark, SweepDModelSizes)
{
    std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  d_model Size Sweep: Row-Major vs Column-Major\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    // Model sizes with their d_model values:
    // Qwen 0.5B: 896, Qwen 1.5B: 1536, Qwen 3B: 2048, Qwen 7B: 3584
    // Llama 7B/8B: 4096, Qwen 14B/Llama 13B: 5120
    // Mistral Large: 6144, Llama 30B: 6656
    // Qwen 72B/Llama 70B: 8192, Llama 405B: 16384
    std::vector<std::pair<int, std::string>> sizes = {
        {896, "Qwen 0.5B"},
        {1536, "Qwen 1.5B"},
        {2048, "Qwen 3B"},
        {3584, "Qwen 7B"},
        {4096, "Llama 8B"},
        {5120, "Qwen 14B"},
        {6144, "Mistral Lg"},
        {6656, "Llama 30B"},
        {8192, "Qwen 72B"},
    };

    std::cout << "┌──────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬─────────┬──────────────┐\n";
    std::cout << "│ d_model  │ RowMaj AVX │ ColMaj AVX │ ColMaj Tun │ ColMaj CG  │ ColMaj Nve │ Transpose  │ Speedup │ Model        │\n";
    std::cout << "│          │ (GFLOP/s)  │ (GFLOP/s)  │ (GFLOP/s)  │ (GFLOP/s)  │ (GFLOP/s)  │ (ms)       │         │              │\n";
    std::cout << "├──────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼─────────┼──────────────┤\n";

    for (const auto &[d_model, model_name] : sizes)
    {
        float *context = aligned_alloc_array<float>(d_model);
        float *Wo_rowmajor = aligned_alloc_array<float>(static_cast<size_t>(d_model) * d_model);
        float *Wo_colmajor = aligned_alloc_array<float>(static_cast<size_t>(d_model) * d_model);
        float *output = aligned_alloc_array<float>(d_model);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < d_model; ++i)
            context[i] = dist(rng);
        for (size_t i = 0; i < static_cast<size_t>(d_model) * d_model; ++i)
            Wo_rowmajor[i] = dist(rng);

        auto t_start = std::chrono::high_resolution_clock::now();
        transpose_matrix(Wo_rowmajor, Wo_colmajor, d_model, d_model);
        auto t_end = std::chrono::high_resolution_clock::now();
        double transpose_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        auto r_row = benchmark_gemv(gemv_rowmajor_avx512, context, Wo_rowmajor, output,
                                    d_model, d_model, 5, 50);
        auto r_col_avx = benchmark_gemv(gemv_colmajor_avx512, context, Wo_colmajor, output,
                                        d_model, d_model, 5, 50);
        auto r_col_tuned = benchmark_gemv(gemv_colmajor_avx512_unroll4_aligned, context, Wo_colmajor, output,
                                          d_model, d_model, 5, 50);
        auto r_col_cg = benchmark_gemv(gemv_colmajor_compiler_guided, context, Wo_colmajor, output,
                                       d_model, d_model, 5, 50);
        auto r_col_naive = benchmark_gemv(gemv_colmajor_naive, context, Wo_colmajor, output,
                                          d_model, d_model, 5, 50);

        double best_col = std::max({r_col_avx.gflops, r_col_tuned.gflops, r_col_cg.gflops, r_col_naive.gflops});
        double speedup = best_col / r_row.gflops;

        std::cout << "│ " << std::setw(8) << d_model
                  << " │ " << std::setw(10) << std::fixed << std::setprecision(1) << r_row.gflops
                  << " │ " << std::setw(10) << r_col_avx.gflops
                  << " │ " << std::setw(10) << r_col_tuned.gflops
                  << " │ " << std::setw(10) << r_col_cg.gflops
                  << " │ " << std::setw(10) << r_col_naive.gflops
                  << " │ " << std::setw(10) << std::setprecision(2) << transpose_ms
                  << " │ " << std::setw(7) << std::setprecision(2) << speedup << "x"
                  << " │ " << std::setw(12) << model_name << " │\n";

        std::free(context);
        std::free(Wo_rowmajor);
        std::free(Wo_colmajor);
        std::free(output);
    }

    std::cout << "└──────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴─────────┴──────────────┘\n";

    std::cout << "\nLegend:\n";
    std::cout << "  RowMaj AVX: Row-major with AVX-512, MR=4 (current kernel)\n";
    std::cout << "  ColMaj AVX: Column-major with AVX-512, NR=64\n";
    std::cout << "  ColMaj Tun: Column-major with aligned loads + K unroll4\n";
    std::cout << "  ColMaj CG:  Column-major compiler-guided (restrict + omp simd)\n";
    std::cout << "  ColMaj Nve: Column-major naive (scalar)\n";
    std::cout << "  Speedup: Best column-major / row-major AVX\n";
}
