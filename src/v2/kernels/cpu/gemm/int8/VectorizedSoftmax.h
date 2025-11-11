/**
 * @file VectorizedSoftmax.h
 * @brief Vectorized softmax implementations for AVX512, AVX2, and scalar
 *
 * Provides ISA-specific implementations of stable softmax:
 * ```
 * max_val = max(x)
 * exp_vals[i] = exp(x[i] - max_val)
 * sum_exp = sum(exp_vals)
 * softmax[i] = exp_vals[i] / sum_exp
 * ```
 *
 * Performance targets (32 elements):
 * - AVX512: ~4× faster than scalar (16-wide vectors)
 * - AVX2: ~2× faster than scalar (8-wide vectors)
 * - Scalar: Baseline reference implementation
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "../../SimdTraits.h"
#include <cmath>
#include <algorithm>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// GCC libmvec declarations for vectorized math functions
// These are production-quality implementations from glibc
#if defined(__GNUC__) && (defined(__AVX512F__) || defined(__AVX2__))
extern "C"
{
#if defined(__AVX512F__)
    // AVX-512 vectorized exp (16-wide float)
    __m512 _ZGVeN16v_expf(__m512) __attribute__((const));
#endif
#if defined(__AVX2__)
    // AVX2 vectorized exp (8-wide float)
    __m256 _ZGVdN8v_expf(__m256) __attribute__((const));
#endif
}
#endif

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Vectorized softmax implementations
             *
             * Template specializations for different ISA tags provide
             * optimized SIMD implementations of softmax.
             */
            template <typename ISA>
            struct VectorizedSoftmax;

            // ========================================================================
            // AVX512 Implementation
            // ========================================================================

#if defined(__AVX512F__)
            template <>
            struct VectorizedSoftmax<simd::AVX512Tag>
            {
                /**
                 * @brief AVX512 vectorized softmax
                 *
                 * Uses 16-wide FP32 vectors for max reduction, exp, and normalization.
                 *
                 * @param x Input array (n elements, FP32)
                 * @param softmax_out Output array (n elements, FP32)
                 * @param n Number of elements (must be multiple of 16 for optimal performance)
                 */
                static void apply(const float *x, float *softmax_out, int n)
                {
                    // Step 1: Find max value (16-wide reduction)
                    __m512 max_vec = _mm512_set1_ps(-INFINITY);

                    int i = 0;
                    for (; i + 16 <= n; i += 16)
                    {
                        __m512 x_vec = _mm512_loadu_ps(&x[i]);
                        max_vec = _mm512_max_ps(max_vec, x_vec);
                    }

                    // Horizontal max reduction (16 → 1)
                    float max_val = _mm512_reduce_max_ps(max_vec);

                    // Handle remainder elements
                    for (; i < n; ++i)
                    {
                        if (x[i] > max_val)
                            max_val = x[i];
                    }

                    // Step 2: Compute exp(x - max) and accumulate sum
                    __m512 sum_vec = _mm512_setzero_ps();
                    __m512 max_broadcast = _mm512_set1_ps(max_val);

                    i = 0;
                    for (; i + 16 <= n; i += 16)
                    {
                        __m512 x_vec = _mm512_loadu_ps(&x[i]);
                        __m512 diff = _mm512_sub_ps(x_vec, max_broadcast);

                        // Use GCC libmvec vectorized exp (production quality)
#if defined(__GNUC__)
                        __m512 exp_vec = _ZGVeN16v_expf(diff);
#else
                        // Fallback: scalar exp in loop (should never happen with GCC)
                        alignas(64) float diff_arr[16];
                        _mm512_storeu_ps(diff_arr, diff);
                        float exp_arr[16];
                        for (int j = 0; j < 16; ++j)
                        {
                            exp_arr[j] = std::exp(diff_arr[j]);
                        }
                        __m512 exp_vec = _mm512_loadu_ps(exp_arr);
#endif
                        _mm512_storeu_ps(&softmax_out[i], exp_vec);

                        sum_vec = _mm512_add_ps(sum_vec, exp_vec);
                    }

                    // Horizontal sum reduction (16 → 1)
                    float sum_exp = _mm512_reduce_add_ps(sum_vec);

                    // Handle remainder
                    for (; i < n; ++i)
                    {
                        float exp_val = std::exp(x[i] - max_val);
                        softmax_out[i] = exp_val;
                        sum_exp += exp_val;
                    }

                    // Step 3: Normalize by sum (vectorized division)
                    float inv_sum = 1.0f / (sum_exp + 1e-12f);
                    __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);

                    i = 0;
                    for (; i + 16 <= n; i += 16)
                    {
                        __m512 exp_vec = _mm512_loadu_ps(&softmax_out[i]);
                        __m512 norm_vec = _mm512_mul_ps(exp_vec, inv_sum_vec);
                        _mm512_storeu_ps(&softmax_out[i], norm_vec);
                    }

                    // Handle remainder
                    for (; i < n; ++i)
                    {
                        softmax_out[i] *= inv_sum;
                    }
                }
            };

            /**
             * @brief AVX512VNNI tag delegates to AVX512 implementation
             *
             * AVX512VNNI provides INT8 instructions but shares FP32 SIMD with AVX512F.
             */
            template <>
            struct VectorizedSoftmax<simd::AVX512VNNITag>
            {
                static void apply(const float *x, float *softmax_out, int n)
                {
                    VectorizedSoftmax<simd::AVX512Tag>::apply(x, softmax_out, n);
                }
            };
#endif

            // ========================================================================
            // AVX2 Implementation
            // ========================================================================

#if defined(__AVX2__)
            template <>
            struct VectorizedSoftmax<simd::AVX2Tag>
            {
                /**
                 * @brief AVX2 vectorized softmax
                 *
                 * Uses 8-wide FP32 vectors for max reduction, exp, and normalization.
                 *
                 * @param x Input array (n elements, FP32)
                 * @param softmax_out Output array (n elements, FP32)
                 * @param n Number of elements
                 */
                static void apply(const float *x, float *softmax_out, int n)
                {
                    // Step 1: Find max value (8-wide reduction)
                    __m256 max_vec = _mm256_set1_ps(-INFINITY);

                    int i = 0;
                    for (; i + 8 <= n; i += 8)
                    {
                        __m256 x_vec = _mm256_loadu_ps(&x[i]);
                        max_vec = _mm256_max_ps(max_vec, x_vec);
                    }

                    // Horizontal max reduction (8 → 1)
                    // Extract high and low 128-bit lanes
                    __m128 max_high = _mm256_extractf128_ps(max_vec, 1);
                    __m128 max_low = _mm256_castps256_ps128(max_vec);
                    __m128 max_128 = _mm_max_ps(max_high, max_low);

                    // Horizontal max within 128-bit lane
                    max_128 = _mm_max_ps(max_128, _mm_shuffle_ps(max_128, max_128, _MM_SHUFFLE(2, 3, 0, 1)));
                    max_128 = _mm_max_ps(max_128, _mm_shuffle_ps(max_128, max_128, _MM_SHUFFLE(1, 0, 3, 2)));

                    float max_val = _mm_cvtss_f32(max_128);

                    // Handle remainder elements
                    for (; i < n; ++i)
                    {
                        if (x[i] > max_val)
                            max_val = x[i];
                    }

                    // Step 2: Compute exp(x - max) and accumulate sum
                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 max_broadcast = _mm256_set1_ps(max_val);

                    i = 0;
                    for (; i + 8 <= n; i += 8)
                    {
                        __m256 x_vec = _mm256_loadu_ps(&x[i]);
                        __m256 diff = _mm256_sub_ps(x_vec, max_broadcast);

                        // Use GCC libmvec vectorized exp (production quality)
#if defined(__GNUC__)
                        __m256 exp_vec = _ZGVdN8v_expf(diff);
#else
                        // Fallback: scalar exp in loop
                        alignas(32) float diff_arr[8];
                        _mm256_storeu_ps(diff_arr, diff);
                        float exp_arr[8];
                        for (int j = 0; j < 8; ++j)
                        {
                            exp_arr[j] = std::exp(diff_arr[j]);
                        }
                        __m256 exp_vec = _mm256_loadu_ps(exp_arr);
#endif
                        _mm256_storeu_ps(&softmax_out[i], exp_vec);

                        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
                    }

                    // Horizontal sum reduction (8 → 1)
                    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                    __m128 sum_128 = _mm_add_ps(sum_high, sum_low);

                    sum_128 = _mm_hadd_ps(sum_128, sum_128);
                    sum_128 = _mm_hadd_ps(sum_128, sum_128);

                    float sum_exp = _mm_cvtss_f32(sum_128);

                    // Handle remainder
                    for (; i < n; ++i)
                    {
                        float exp_val = std::exp(x[i] - max_val);
                        softmax_out[i] = exp_val;
                        sum_exp += exp_val;
                    }

                    // Step 3: Normalize by sum (vectorized division)
                    float inv_sum = 1.0f / (sum_exp + 1e-12f);
                    __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);

                    i = 0;
                    for (; i + 8 <= n; i += 8)
                    {
                        __m256 exp_vec = _mm256_loadu_ps(&softmax_out[i]);
                        __m256 norm_vec = _mm256_mul_ps(exp_vec, inv_sum_vec);
                        _mm256_storeu_ps(&softmax_out[i], norm_vec);
                    }

                    // Handle remainder
                    for (; i < n; ++i)
                    {
                        softmax_out[i] *= inv_sum;
                    }
                }
            };
#endif

            // ========================================================================
            // Scalar Fallback Implementation
            // ========================================================================

            template <>
            struct VectorizedSoftmax<simd::ScalarTag>
            {
                /**
                 * @brief Scalar softmax implementation (fallback)
                 *
                 * Pure scalar implementation for systems without SIMD support
                 * or for correctness verification.
                 *
                 * @param x Input array (n elements, FP32)
                 * @param softmax_out Output array (n elements, FP32)
                 * @param n Number of elements
                 */
                static void apply(const float *x, float *softmax_out, int n)
                {
                    // Step 1: Find max for numerical stability
                    float max_val = x[0];
                    for (int i = 1; i < n; ++i)
                    {
                        if (x[i] > max_val)
                            max_val = x[i];
                    }

                    // Step 2: Compute exp(x - max) and sum
                    float sum_exp = 0.0f;
                    for (int i = 0; i < n; ++i)
                    {
                        float exp_val = std::exp(x[i] - max_val);
                        softmax_out[i] = exp_val;
                        sum_exp += exp_val;
                    }

                    // Step 3: Normalize by sum
                    float inv_sum = 1.0f / (sum_exp + 1e-12f);
                    for (int i = 0; i < n; ++i)
                    {
                        softmax_out[i] *= inv_sum;
                    }
                }
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
