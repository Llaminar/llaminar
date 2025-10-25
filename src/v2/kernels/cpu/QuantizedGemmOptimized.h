/**
 * @file QuantizedGemmOptimized.h
 * @brief Template-based quantized GEMM kernel eliminating virtual dispatch overhead
 *
 * **CANONICAL PATH**: This template version is the PREFERRED implementation for V2.
 * Use this instead of the interface-based QuantizedGemmKernel whenever possible.
 *
 * This kernel provides the same functionality as QuantizedGemmKernel but uses
 * template instantiation on the concrete tensor type instead of the IBlockDecoder
 * interface. This eliminates virtual function call overhead (~18% speedup) by
 * allowing the compiler to inline decode_block_at() calls.
 *
 * Performance: 33.58 GFLOPS vs 28.55 GFLOPS (virtual dispatch) on Q-Proj 1024.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../utils/DebugEnv.h"
#include <algorithm>
#include <vector>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    /**
     * @brief Template-based quantized GEMM kernel (no virtual dispatch)
     *
     * Identical algorithm to QuantizedGemmKernel but templated on concrete tensor type
     * to eliminate virtual function call overhead on decode_block_at().
     *
     * Usage:
     *   IQ4_NLTensor* weight = ...;
     *   QuantizedGemmOptimized<IQ4_NLTensor> gemm(weight);
     *   gemm.multiply(A, C, m, n, k);
     *
     * Performance: ~10-15% faster than IBlockDecoder version due to:
     * - No virtual dispatch (inlined decode_block_at)
     * - Better compiler optimization (can see through decode logic)
     * - Reduced instruction cache pressure
     */
    template <typename TensorType>
    class QuantizedGemmOptimized
    {
    public:
        explicit QuantizedGemmOptimized(TensorType *tensor)
            : tensor_(tensor) {}

        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f)
        {
            if (!tensor_)
                return false;

            // Validate dimensions
            int expected_cols = transpose_B ? k : n;
            if (static_cast<int>(tensor_->shape()[1]) != expected_cols)
            {
                return false;
            }

            // Strategy selection based on batch size
            if (m >= 2 && m <= 16)
            {
                return multiply_cache_blocked(A, C, m, n, k, alpha, beta);
            }
            else
            {
                return multiply_row_wise(A, C, m, n, k, alpha, beta);
            }
        }

    private:
        TensorType *tensor_;

        // SIMD-optimized dot product (same as QuantizedGemmKernel)
        static inline float dot_product_simd(const float *a, const float *b, size_t count)
        {
#if defined(__AVX512F__)
            __m512 sum = _mm512_setzero_ps();

            size_t i = 0;
            for (; i + 16 <= count; i += 16)
            {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_load_ps(b + i); // Aligned load
                sum = _mm512_fmadd_ps(va, vb, sum);
            }

            float result = _mm512_reduce_add_ps(sum);

            // Scalar tail
            for (; i < count; ++i)
            {
                result += a[i] * b[i];
            }

            return result;

#elif defined(__AVX2__)
            __m256 sum = _mm256_setzero_ps();

            size_t i = 0;
            for (; i + 8 <= count; i += 8)
            {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                sum = _mm256_fmadd_ps(va, vb, sum);
            }

            // Horizontal sum
            __m128 sum_high = _mm256_extractf128_ps(sum, 1);
            __m128 sum_low = _mm256_castps256_ps128(sum);
            __m128 sum128 = _mm_add_ps(sum_low, sum_high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float result = _mm_cvtss_f32(sum128);

            // Scalar tail
            for (; i < count; ++i)
            {
                result += a[i] * b[i];
            }

            return result;

#else
            // Scalar fallback
            float result = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                result += a[i] * b[i];
            }
            return result;
#endif
        }

        bool multiply_cache_blocked(
            const float *A, float *C,
            int m, int n, int k,
            float alpha, float beta)
        {
            const size_t BLOCK_SIZE = 32; // IQ4_NL block size
            const int num_k_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

#pragma omp parallel for schedule(static) if (n > 128)
            for (int j = 0; j < n; ++j)
            {
                float acc[16] = {0};

                for (int kb = 0; kb < num_k_blocks; ++kb)
                {
                    size_t k_start = kb * BLOCK_SIZE;
                    size_t k_count = std::min(BLOCK_SIZE, static_cast<size_t>(k) - k_start);

                    alignas(64) float B_block[64];
                    tensor_->decode_block_at(j, kb, B_block); // DIRECT CALL - no virtual dispatch!

                    for (int i = 0; i < m; ++i)
                    {
                        const float *A_row = A + i * k + k_start;
                        acc[i] += dot_product_simd(A_row, B_block, k_count);
                    }
                }

                for (int i = 0; i < m; ++i)
                {
                    size_t c_idx = i * n + j;
                    C[c_idx] = alpha * acc[i] + beta * C[c_idx];
                }
            }

            return true;
        }

        bool multiply_row_wise(
            const float *A, float *C,
            int m, int n, int k,
            float alpha, float beta)
        {
            const size_t BLOCK_SIZE = 32;
            const int num_k_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Exact V1 adaptive tiling logic
            const auto &env = debugEnv();
            int M_TILE, N_TILE;

            const float aspect_ratio = static_cast<float>(n) / static_cast<float>(m > 0 ? m : 1);
            const bool is_wide_output = aspect_ratio > 2.0f;
            const bool is_square = aspect_ratio >= 0.5f && aspect_ratio <= 2.0f;

            if (env.dequant.iq4_override_m_tile > 0 && env.dequant.iq4_override_n_tile > 0)
            {
                M_TILE = env.dequant.iq4_override_m_tile;
                N_TILE = env.dequant.iq4_override_n_tile;
            }
            else if (is_wide_output)
            {
                M_TILE = 64;
                N_TILE = 32;
            }
            else if (is_square)
            {
                if (m >= 4096 || n >= 4096)
                {
                    M_TILE = 64;
                    N_TILE = 32;
                }
                else if (m >= 2048 || n >= 2048)
                {
                    M_TILE = 64;
                    N_TILE = 32;
                }
                else if (m >= 1024 || n >= 1024)
                {
                    M_TILE = 32;
                    N_TILE = 32; // Empirically optimal for Q-proj-1024 (314 GFLOPS)
                }
                else if (m >= 512 || n >= 512)
                {
                    M_TILE = 96;
                    N_TILE = 96;
                }
                else
                {
                    M_TILE = 128;
                    N_TILE = 128;
                }
            }
            else
            {
                if (m >= 4096)
                {
                    M_TILE = 64;
                    N_TILE = 24;
                }
                else if (m >= 2048)
                {
                    M_TILE = 96;
                    N_TILE = 32;
                }
                else
                {
                    M_TILE = 128;
                    N_TILE = 48;
                }
            }

#pragma omp parallel
            {
                std::vector<float> B_tile(k * N_TILE);

#pragma omp for schedule(dynamic)
                for (int jj = 0; jj < n; jj += N_TILE)
                {
                    int n_block = std::min(N_TILE, n - jj);

                    // 4-column vectorized microkernel (FIXED: k-blocks outer for cache locality)
                    if (env.dequant.iq4_gemm_microkernel && n_block >= 4)
                    {
                        int j_vec = 0;
                        for (; j_vec + 4 <= n_block; j_vec += 4)
                        {
                            // CRITICAL: K-blocks MUST be outer loop for cache locality
                            // Process same k-block across 4 columns before moving to next k-block
                            for (int kb = 0; kb < num_k_blocks; ++kb)
                            {
                                size_t k_start = kb * BLOCK_SIZE;
                                // Unroll 4 columns per k-block
                                for (int jv = 0; jv < 4; ++jv)
                                {
                                    int j = jj + j_vec + jv;
                                    float *B_col = B_tile.data() + (j_vec + jv) * k;
                                    tensor_->decode_block_at(j, kb, B_col + k_start);
                                }
                            }
                        }
                        // Handle remaining columns (< 4) with standard path
                        for (; j_vec < n_block; ++j_vec)
                        {
                            int j = jj + j_vec;
                            float *B_col = B_tile.data() + j_vec * k;
                            for (int kb = 0; kb < num_k_blocks; ++kb)
                            {
                                size_t k_start = kb * BLOCK_SIZE;
                                tensor_->decode_block_at(j, kb, B_col + k_start);
                            }
                        }
                    }
                    else
                    {
                        for (int j_local = 0; j_local < n_block; ++j_local)
                        {
                            int j = jj + j_local;
                            float *B_col = B_tile.data() + j_local * k;

                            for (int kb = 0; kb < num_k_blocks; ++kb)
                            {
                                size_t k_start = kb * BLOCK_SIZE;
                                tensor_->decode_block_at(j, kb, B_col + k_start); // DIRECT CALL!
                            }
                        }
                    }

                    for (int ii = 0; ii < m; ii += M_TILE)
                    {
                        int m_block = std::min(M_TILE, m - ii);

                        // Software prefetching
                        if (ii + M_TILE < m)
                        {
                            const float *next_A = A + (ii + M_TILE) * k;
                            for (int pf = 0; pf < std::min(M_TILE, m - ii - M_TILE); pf += 8)
                            {
                                __builtin_prefetch(next_A + pf * k, 0, 1);
                            }
                        }

                        for (int i_local = 0; i_local < m_block; ++i_local)
                        {
                            int i = ii + i_local;
                            const float *A_row = A + i * k;

                            for (int j_local = 0; j_local < n_block; ++j_local)
                            {
                                int j = jj + j_local;
                                const float *B_col = B_tile.data() + j_local * k;

                                float acc = dot_product_simd(A_row, B_col, k);
                                size_t c_idx = i * n + j;
                                C[c_idx] = alpha * acc + beta * C[c_idx];
                            }
                        }
                    }
                }
            }

            return true;
        }
    };

} // namespace llaminar2
