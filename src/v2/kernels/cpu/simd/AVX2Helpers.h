/**
 * @file AVX2Helpers.h
 * @brief Common AVX2 SIMD helper functions for FP32 vector operations.
 *
 * Provides reusable building blocks for AVX2 fallback paths:
 *   - Horizontal reductions (sum, max)
 *   - Fast exp/sigmoid/SiLU approximations (8-wide YMM)
 *   - L2 norm + scale
 *   - Vector dot product, saxpy, scale
 *
 * These mirrors their AVX-512 counterparts but process 8 floats per iteration
 * instead of 16. Used by GDN, ShortConvolution, GatedRMSNorm, and
 * AttentionOutputGate stages as AVX2 fallback paths.
 */

#pragma once

#include <cmath>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>

namespace llaminar2
{
    namespace avx2
    {

        // ====================================================================
        // Horizontal reductions
        // ====================================================================

        /** @brief Horizontal sum of 8 floats in a YMM register. */
        static inline float hsum_ps(__m256 v)
        {
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 lo = _mm256_castps256_ps128(v);
            lo = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(lo);
            lo = _mm_add_ps(lo, shuf);
            shuf = _mm_movehl_ps(shuf, lo);
            lo = _mm_add_ss(lo, shuf);
            return _mm_cvtss_f32(lo);
        }

        /** @brief Horizontal max of 8 floats in a YMM register. */
        static inline float hmax_ps(__m256 v)
        {
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 lo = _mm256_castps256_ps128(v);
            lo = _mm_max_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(lo);
            lo = _mm_max_ps(lo, shuf);
            shuf = _mm_movehl_ps(shuf, lo);
            lo = _mm_max_ss(lo, shuf);
            return _mm_cvtss_f32(lo);
        }

        /** @brief Horizontal sum of 8 int32s in a YMM register. */
        static inline int32_t hsum_epi32(__m256i v)
        {
            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);
            lo = _mm_add_epi32(lo, hi);
            lo = _mm_hadd_epi32(lo, lo);
            lo = _mm_hadd_epi32(lo, lo);
            return _mm_extract_epi32(lo, 0);
        }

        // ====================================================================
        // Fast exp/sigmoid/SiLU (8-wide, same polynomial as AVX512 versions)
        // ====================================================================

        /**
         * @brief Fast vectorised exp() for 8 FP32 values using AVX2.
         *
         * Range-reduced: exp(x) = 2^n · P(f) where f = x·log2(e) - n.
         * Degree-5 Horner polynomial for 2^f.
         */
        static inline __m256 fast_exp(__m256 vx)
        {
            vx = _mm256_max_ps(vx, _mm256_set1_ps(-88.0f));
            vx = _mm256_min_ps(vx, _mm256_set1_ps(88.0f));

            const __m256 vlog2e = _mm256_set1_ps(1.4426950408889634f);
            const __m256 vc0 = _mm256_set1_ps(1.0f);
            const __m256 vc1 = _mm256_set1_ps(0.693147180559945f);
            const __m256 vc2 = _mm256_set1_ps(0.240226506959101f);
            const __m256 vc3 = _mm256_set1_ps(0.055504108664822f);
            const __m256 vc4 = _mm256_set1_ps(0.009618129107629f);
            const __m256 vc5 = _mm256_set1_ps(0.001333355814642f);

            __m256 vt = _mm256_mul_ps(vx, vlog2e);
            __m256 vn = _mm256_round_ps(vt, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 vf = _mm256_sub_ps(vt, vn);

            __m256 vpoly = _mm256_fmadd_ps(vc5, vf, vc4);
            vpoly = _mm256_fmadd_ps(vpoly, vf, vc3);
            vpoly = _mm256_fmadd_ps(vpoly, vf, vc2);
            vpoly = _mm256_fmadd_ps(vpoly, vf, vc1);
            vpoly = _mm256_fmadd_ps(vpoly, vf, vc0);

            __m256i vi_n = _mm256_cvtps_epi32(vn);
            vi_n = _mm256_add_epi32(vi_n, _mm256_set1_epi32(127));
            __m256 v2n = _mm256_castsi256_ps(_mm256_slli_epi32(vi_n, 23));
            return _mm256_mul_ps(vpoly, v2n);
        }

        /** @brief Fast sigmoid(x) = 1/(1+exp(-x)) for 8 floats. */
        static inline __m256 fast_sigmoid(__m256 vx)
        {
            __m256 vneg = _mm256_sub_ps(_mm256_setzero_ps(), vx);
            __m256 vexp_neg = fast_exp(vneg);
            __m256 vone = _mm256_set1_ps(1.0f);
            return _mm256_div_ps(vone, _mm256_add_ps(vone, vexp_neg));
        }

        /** @brief Fast SiLU(x) = x * sigmoid(x) for 8 floats. */
        static inline __m256 fast_silu(__m256 vx)
        {
            return _mm256_mul_ps(vx, fast_sigmoid(vx));
        }

        // ====================================================================
        // Vector operations (norm, dot, saxpy, scale)
        // ====================================================================

        /** @brief Compute sum of squares of n floats (AVX2). Returns norm_sq. */
        static inline float norm_sq(const float *data, int n)
        {
            __m256 vsum = _mm256_setzero_ps();
            int i = 0;
            const int n8 = n & ~7;
            for (; i < n8; i += 8)
            {
                __m256 v = _mm256_loadu_ps(data + i);
                vsum = _mm256_fmadd_ps(v, v, vsum);
            }
            float result = hsum_ps(vsum);
            for (; i < n; ++i)
                result += data[i] * data[i];
            return result;
        }

        /**
         * @brief L2-normalize src into dst with optional combined scale.
         *
         * dst[i] = src[i] * (combined_scale / ||src||)
         */
        static inline void l2norm_scale(const float *src, float *dst, int n,
                                        float combined_scale, float eps)
        {
            float nsq = norm_sq(src, n);
            const float inv = combined_scale / std::max(std::sqrt(nsq), eps);
            const __m256 vinv = _mm256_set1_ps(inv);
            int i = 0;
            const int n8 = n & ~7;
            for (; i < n8; i += 8)
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vinv));
            for (; i < n; ++i)
                dst[i] = src[i] * inv;
        }

        /** @brief Scale vector in-place: data[i] *= s */
        static inline void scale(float *data, int n, float s)
        {
            const __m256 vs = _mm256_set1_ps(s);
            int i = 0;
            const int n8 = n & ~7;
            for (; i < n8; i += 8)
                _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), vs));
            for (; i < n; ++i)
                data[i] *= s;
        }

        /** @brief Vector dot product: returns sum(a[i]*b[i]) */
        static inline float dot(const float *a, const float *b, int n)
        {
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            const int n8 = n & ~7;
            for (; i < n8; i += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
            float sum = hsum_ps(acc);
            for (; i < n; ++i)
                sum += a[i] * b[i];
            return sum;
        }

        /**
         * @brief matvec_t: y += a * row, vectorized.
         * Used for transpose matvec accumulation: y[j] += a * row[j]
         */
        static inline void axpy(float *y, const float *x, float a, int n)
        {
            const __m256 va = _mm256_set1_ps(a);
            int i = 0;
            const int n8 = n & ~7;
            for (; i < n8; i += 8)
            {
                __m256 vy = _mm256_loadu_ps(y + i);
                vy = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), vy);
                _mm256_storeu_ps(y + i, vy);
            }
            for (; i < n; ++i)
                y[i] += a * x[i];
        }

        /** @brief Zero a vector: data[i] = 0 */
        static inline void zero(float *data, int n)
        {
            int i = 0;
            const int n8 = n & ~7;
            for (; i < n8; i += 8)
                _mm256_storeu_ps(data + i, _mm256_setzero_ps());
            for (; i < n; ++i)
                data[i] = 0.0f;
        }

        /**
         * @brief dst[i] = (a[i] - b[i]) * s
         */
        static inline void sub_mul(float *dst, const float *a, const float *b,
                                   float s, int n)
        {
            const __m256 vs = _mm256_set1_ps(s);
            int i = 0;
            const int n8 = n & ~7;
            for (; i < n8; i += 8)
            {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_sub_ps(va, vb), vs));
            }
            for (; i < n; ++i)
                dst[i] = (a[i] - b[i]) * s;
        }

        /**
         * @brief Copy src to dst with scaling: dst[i] = src[i] * s
         */
        static inline void copy_scale(float *dst, const float *src, float s, int n)
        {
            const __m256 vs = _mm256_set1_ps(s);
            int i = 0;
            const int n8 = n & ~7;
            for (; i < n8; i += 8)
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vs));
            for (; i < n; ++i)
                dst[i] = src[i] * s;
        }

    } // namespace avx2
} // namespace llaminar2

#endif // __AVX2__
