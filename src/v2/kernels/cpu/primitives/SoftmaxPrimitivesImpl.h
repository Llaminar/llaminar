/**
 * @file SoftmaxPrimitivesImpl.h
 * @brief Inline implementations of softmax primitives (all precisions, all SIMD levels)
 * @author David Sanftenberg
 *
 * Separated implementations allow:
 * - Individual testing of scalar vs AVX2 vs AVX512
 * - Performance benchmarking per SIMD level
 * - Parity validation (scalar as ground truth)
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#define LLAMINAR_HAS_AVX512
#elif defined(__AVX2__)
#include <immintrin.h>
#define LLAMINAR_HAS_AVX2
#endif

#if defined(__F16C__)
#define LLAMINAR_HAS_F16C
#endif

#if defined(__AVX512BF16__)
#define LLAMINAR_HAS_AVX512BF16
#endif

namespace llaminar2::primitives
{

    // ============================================================================
    // Helper Functions
    // ============================================================================

    namespace detail
    {
#if defined(LLAMINAR_HAVE_LIBMVEC)
        extern "C"
        {
#if defined(__AVX512F__)
            __m512 _ZGVeN16v_expf(__m512) __attribute__((weak));
#endif
#if defined(__AVX2__)
            __m256 _ZGVcN8v_expf(__m256) __attribute__((weak));
#endif
#if defined(__SSE2__)
            __m128 _ZGVbN4v_expf(__m128) __attribute__((weak));
#endif
        }
#endif
        // BF16 ↔ FP32 conversion (manual)
        inline float bf16_to_fp32_scalar(uint16_t bf16)
        {
            uint32_t fp32_bits = static_cast<uint32_t>(bf16) << 16;
            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        inline uint16_t fp32_to_bf16_scalar(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(float));
            // Round to nearest even (RNE)
            uint32_t rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1);
            uint32_t rounded = fp32_bits + rounding_bias;
            return static_cast<uint16_t>(rounded >> 16);
        }

#if defined(__SSE2__)
        inline float hsum128(__m128 v)
        {
            __m128 shuf = _mm_movehdup_ps(v);
            __m128 sums = _mm_add_ps(v, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        }

        inline __m128 floor_ps_compat(__m128 x)
        {
            __m128i truncated = _mm_cvttps_epi32(x);
            __m128 floored = _mm_cvtepi32_ps(truncated);
            __m128 mask = _mm_cmpgt_ps(floored, x);
            return _mm_sub_ps(floored, _mm_and_ps(mask, _mm_set1_ps(1.0f)));
        }
#endif

#if defined(__AVX2__)
        inline float hsum256(__m256 v)
        {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            return hsum128(sum);
        }

        inline __m256 floor_ps_compat(__m256 x)
        {
            __m256i truncated = _mm256_cvttps_epi32(x);
            __m256 floored = _mm256_cvtepi32_ps(truncated);
            __m256 mask = _mm256_cmp_ps(floored, x, _CMP_GT_OQ);
            return _mm256_sub_ps(floored, _mm256_and_ps(mask, _mm256_set1_ps(1.0f)));
        }
#endif

#if defined(LLAMINAR_HAS_AVX512)
        inline float hsum512(__m512 v)
        {
            return _mm512_reduce_add_ps(v);
        }

        inline __m512 floor_ps_compat(__m512 x)
        {
            __m512i truncated = _mm512_cvttps_epi32(x);
            __m512 floored = _mm512_cvtepi32_ps(truncated);
            __mmask16 mask = _mm512_cmp_ps_mask(floored, x, _CMP_GT_OQ);
            return _mm512_mask_sub_ps(floored, mask, floored, _mm512_set1_ps(1.0f));
        }
#endif

#if defined(LLAMINAR_HAS_F16C)
        // FP16 ↔ FP32 conversion (using F16C)
        inline float fp16_to_fp32_f16c(uint16_t fp16)
        {
            __m128i vec = _mm_cvtsi32_si128(fp16);
            __m128 fp32_vec = _mm_cvtph_ps(vec);
            return _mm_cvtss_f32(fp32_vec);
        }

        inline uint16_t fp32_to_fp16_f16c(float fp32)
        {
            __m128 fp32_vec = _mm_set_ss(fp32);
            __m128i fp16_vec = _mm_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
            return static_cast<uint16_t>(_mm_cvtsi128_si32(fp16_vec));
        }
#else
        // FP16 ↔ FP32 conversion (manual, fallback)
        inline float fp16_to_fp32_scalar(uint16_t fp16)
        {
            uint32_t sign = (fp16 & 0x8000) << 16;
            uint32_t exp = (fp16 & 0x7C00) >> 10;
            uint32_t mant = (fp16 & 0x03FF);

            uint32_t fp32_bits;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    fp32_bits = sign; // Zero
                }
                else
                {
                    // Denormal
                    exp = 1;
                    while ((mant & 0x0400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x03FF;
                    fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                }
            }
            else if (exp == 0x1F)
            {
                // Inf/NaN
                fp32_bits = sign | 0x7F800000 | (mant << 13);
            }
            else
            {
                // Normal
                fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }

            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        inline uint16_t fp32_to_fp16_scalar(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(float));

            uint32_t sign = (fp32_bits & 0x80000000) >> 16;
            int32_t exp = ((fp32_bits & 0x7F800000) >> 23) - 127 + 15;
            uint32_t mant = (fp32_bits & 0x007FFFFF);

            uint16_t fp16;
            if (exp <= 0)
            {
                if (exp < -10)
                {
                    fp16 = static_cast<uint16_t>(sign); // Underflow to zero
                }
                else
                {
                    // Denormal
                    mant |= 0x00800000;
                    mant >>= (1 - exp);
                    fp16 = static_cast<uint16_t>(sign | (mant >> 13));
                }
            }
            else if (exp >= 0x1F)
            {
                fp16 = static_cast<uint16_t>(sign | 0x7C00); // Overflow to inf
            }
            else
            {
                fp16 = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
            }

            return fp16;
        }
#endif

#if defined(__SSE2__)
        inline __m128 exp128_ps_poly(__m128 x)
        {
            const __m128 max_val = _mm_set1_ps(88.3762626647949f);
            const __m128 min_val = _mm_set1_ps(-88.3762626647949f);
            x = _mm_min_ps(x, max_val);
            x = _mm_max_ps(x, min_val);

            const __m128 log2e = _mm_set1_ps(1.44269504088896341f);
            const __m128 half = _mm_set1_ps(0.5f);
            __m128 fx = _mm_add_ps(_mm_mul_ps(x, log2e), half);
            __m128 floored = floor_ps_compat(fx);

            const __m128 c1 = _mm_set1_ps(0.693359375f);
            const __m128 c2 = _mm_set1_ps(-2.12194440e-4f);
            __m128 temp = _mm_mul_ps(floored, c1);
            __m128 z = _mm_sub_ps(x, temp);
            temp = _mm_mul_ps(floored, c2);
            x = _mm_sub_ps(z, temp);

            const __m128 p0 = _mm_set1_ps(1.9875691500e-4f);
            const __m128 p1 = _mm_set1_ps(1.3981999507e-3f);
            const __m128 p2 = _mm_set1_ps(8.3334519073e-3f);
            const __m128 p3 = _mm_set1_ps(4.1665795894e-2f);
            const __m128 p4 = _mm_set1_ps(1.6666665459e-1f);
            const __m128 p5 = _mm_set1_ps(5.0000001201e-1f);

            __m128 y = p0;
            y = _mm_add_ps(_mm_mul_ps(y, x), p1);
            y = _mm_add_ps(_mm_mul_ps(y, x), p2);
            y = _mm_add_ps(_mm_mul_ps(y, x), p3);
            y = _mm_add_ps(_mm_mul_ps(y, x), p4);
            y = _mm_add_ps(_mm_mul_ps(y, x), p5);
            y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.0f));

            __m128i emm0 = _mm_cvttps_epi32(floored);
            emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(0x7f));
            emm0 = _mm_slli_epi32(emm0, 23);
            __m128 pow2n = _mm_castsi128_ps(emm0);
            return _mm_mul_ps(y, pow2n);
        }

        inline __m128 exp128_ps(__m128 x)
        {
#if defined(LLAMINAR_HAVE_LIBMVEC)
            if (_ZGVbN4v_expf)
            {
                return _ZGVbN4v_expf(x);
            }
#endif
            return exp128_ps_poly(x);
        }
#endif

#if defined(__AVX2__)
        inline __m256 exp256_ps_poly(__m256 x)
        {
            const __m256 max_val = _mm256_set1_ps(88.3762626647949f);
            const __m256 min_val = _mm256_set1_ps(-88.3762626647949f);
            x = _mm256_min_ps(x, max_val);
            x = _mm256_max_ps(x, min_val);

            const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
            const __m256 half = _mm256_set1_ps(0.5f);
            __m256 fx = _mm256_add_ps(_mm256_mul_ps(x, log2e), half);
            __m256 floored = floor_ps_compat(fx);

            const __m256 c1 = _mm256_set1_ps(0.693359375f);
            const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
            __m256 temp = _mm256_mul_ps(floored, c1);
            __m256 z = _mm256_sub_ps(x, temp);
            temp = _mm256_mul_ps(floored, c2);
            x = _mm256_sub_ps(z, temp);

            const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
            const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
            const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
            const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
            const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
            const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);

            __m256 y = p0;
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p1);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p2);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p3);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p4);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p5);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.0f));

            __m256i emm0 = _mm256_cvttps_epi32(floored);
            emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
            emm0 = _mm256_slli_epi32(emm0, 23);
            __m256 pow2n = _mm256_castsi256_ps(emm0);
            return _mm256_mul_ps(y, pow2n);
        }

        inline __m256 exp256_ps(__m256 x)
        {
#if defined(LLAMINAR_HAVE_LIBMVEC)
            if (_ZGVcN8v_expf)
            {
                return _ZGVcN8v_expf(x);
            }
#endif
            return exp256_ps_poly(x);
        }
#endif

#if defined(LLAMINAR_HAS_AVX512)
        inline __m512 exp512_ps_poly(__m512 x)
        {
            const __m512 max_val = _mm512_set1_ps(88.3762626647949f);
            const __m512 min_val = _mm512_set1_ps(-88.3762626647949f);
            x = _mm512_min_ps(x, max_val);
            x = _mm512_max_ps(x, min_val);

            const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
            const __m512 half = _mm512_set1_ps(0.5f);
            __m512 fx = _mm512_add_ps(_mm512_mul_ps(x, log2e), half);
            __m512 floored = floor_ps_compat(fx);

            const __m512 c1 = _mm512_set1_ps(0.693359375f);
            const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);
            __m512 temp = _mm512_mul_ps(floored, c1);
            __m512 z = _mm512_sub_ps(x, temp);
            temp = _mm512_mul_ps(floored, c2);
            x = _mm512_sub_ps(z, temp);

            const __m512 p0 = _mm512_set1_ps(1.9875691500e-4f);
            const __m512 p1 = _mm512_set1_ps(1.3981999507e-3f);
            const __m512 p2 = _mm512_set1_ps(8.3334519073e-3f);
            const __m512 p3 = _mm512_set1_ps(4.1665795894e-2f);
            const __m512 p4 = _mm512_set1_ps(1.6666665459e-1f);
            const __m512 p5 = _mm512_set1_ps(5.0000001201e-1f);

            __m512 y = p0;
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p1);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p2);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p3);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p4);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p5);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), _mm512_set1_ps(1.0f));

            __m512i emm0 = _mm512_cvttps_epi32(floored);
            emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
            emm0 = _mm512_slli_epi32(emm0, 23);
            __m512 pow2n = _mm512_castsi512_ps(emm0);
            return _mm512_mul_ps(y, pow2n);
        }

        inline __m512 exp512_ps(__m512 x)
        {
#if defined(LLAMINAR_HAVE_LIBMVEC)
            if (_ZGVeN16v_expf)
            {
                return _ZGVeN16v_expf(x);
            }
#endif
            return exp512_ps_poly(x);
        }
#endif

#if defined(LLAMINAR_HAS_AVX2)
        // Horizontal max reduction for AVX2
        inline float hmax256(__m256 v)
        {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 m = _mm_max_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(m);
            m = _mm_max_ps(m, shuf);
            shuf = _mm_movehl_ps(shuf, m);
            m = _mm_max_ps(m, shuf);
            return _mm_cvtss_f32(m);
        }
#endif

    } // namespace detail

    // ============================================================================
    // FP32 Softmax - Scalar Implementation
    // ============================================================================

    inline void softmax_row_fp32_scalar(
        float *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // Pass 1: Find max (with causal masking)
        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Compute sum of exp(x - max)
        double sum = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0.0f;
            }
            else
            {
                float v = row[c] * scale;
                row[c] = std::exp(v - row_max) * inv;
            }
        }
    }

    // ============================================================================
    // FP32 Softmax - AVX2 Implementation
    // ============================================================================

    inline void softmax_row_fp32_avx2(
        float *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX2)
        const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        const __m256 scale_ps = _mm256_set1_ps(scale);

        // Pass 1: Find max
        __m256 vmax = neg_inf;
        int c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m256 v = _mm256_loadu_ps(row + c);
            v = _mm256_mul_ps(v, scale_ps);

            if (causal)
            {
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, v);
                for (int lane = 0; lane < 8; ++lane)
                {
                    if (c + lane > row_idx)
                        tmp[lane] = -std::numeric_limits<float>::infinity();
                }
                v = _mm256_load_ps(tmp);
            }

            vmax = _mm256_max_ps(vmax, v);
        }

        float row_max = detail::hmax256(vmax);

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize
        const __m256 inv_ps = _mm256_set1_ps(inv);
        const __m256 max_ps = _mm256_set1_ps(row_max);
        const __m256 zero_ps = _mm256_setzero_ps();

        c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m256 v = _mm256_loadu_ps(row + c);
            v = _mm256_mul_ps(v, scale_ps);

            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, v);

            float out[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                if (causal && c + lane > row_idx)
                {
                    out[lane] = 0.0f;
                }
                else
                {
                    out[lane] = std::exp(tmp[lane] - row_max) * inv;
                }
            }

            _mm256_storeu_ps(row + c, _mm256_loadu_ps(out));
        }

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0.0f;
            }
            else
            {
                float v = row[c] * scale;
                row[c] = std::exp(v - row_max) * inv;
            }
        }
#else
        // Fallback to scalar if AVX2 not available
        softmax_row_fp32_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // FP32 Softmax - AVX512 Implementation
    // ============================================================================

    inline void softmax_row_fp32_avx512(
        float *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512)
        const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        const __m512 scale_ps = _mm512_set1_ps(scale);

        // Pass 1: Find max
        __m512 vmax = neg_inf;
        int c = 0;
        for (; c + 16 <= cols; c += 16)
        {
            __m512 v = _mm512_loadu_ps(row + c);
            v = _mm512_mul_ps(v, scale_ps);

            if (causal)
            {
                // Mask where c+lane > row_idx
                __mmask16 mask = 0;
                for (int lane = 0; lane < 16; ++lane)
                {
                    if (c + lane <= row_idx)
                        mask |= (1 << lane);
                }
                v = _mm512_mask_blend_ps(mask, neg_inf, v);
            }

            vmax = _mm512_max_ps(vmax, v);
        }

        float row_max = _mm512_reduce_max_ps(vmax);

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize
        const __m512 inv_ps = _mm512_set1_ps(inv);
        const __m512 max_ps = _mm512_set1_ps(row_max);
        const __m512 zero_ps = _mm512_setzero_ps();

        c = 0;
        for (; c + 16 <= cols; c += 16)
        {
            __m512 v = _mm512_loadu_ps(row + c);
            v = _mm512_mul_ps(v, scale_ps);

            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, v);

            float out[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                if (causal && c + lane > row_idx)
                {
                    out[lane] = 0.0f;
                }
                else
                {
                    out[lane] = std::exp(tmp[lane] - row_max) * inv;
                }
            }

            _mm512_storeu_ps(row + c, _mm512_loadu_ps(out));
        }

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0.0f;
            }
            else
            {
                float v = row[c] * scale;
                row[c] = std::exp(v - row_max) * inv;
            }
        }
#else
        // Fallback to AVX2 if AVX512 not available
        softmax_row_fp32_avx2(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // FP32 Softmax - Compile-Time Dispatch
    // ============================================================================

    inline void softmax_row_fp32(
        float *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512)
        softmax_row_fp32_avx512(row, cols, causal, scale, row_idx);
#elif defined(LLAMINAR_HAS_AVX2)
        softmax_row_fp32_avx2(row, cols, causal, scale, row_idx);
#else
        softmax_row_fp32_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // BF16 Softmax - Scalar Implementation
    // ============================================================================

    inline void softmax_row_bf16_scalar(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // Pass 1: Find max (convert BF16→FP32 on the fly)
        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = detail::bf16_to_fp32_scalar(row[c]) * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = detail::bf16_to_fp32_scalar(row[c]) * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize and convert FP32→BF16
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0;
            }
            else
            {
                float v = detail::bf16_to_fp32_scalar(row[c]) * scale;
                float result = std::exp(v - row_max) * inv;
                row[c] = detail::fp32_to_bf16_scalar(result);
            }
        }
    }

    // ============================================================================
    // BF16 Softmax - AVX2 Implementation
    // ============================================================================

    inline void softmax_row_bf16_avx2(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // Fallback to scalar for now (TODO: Implement BF16 AVX2 conversion)
        softmax_row_bf16_scalar(row, cols, causal, scale, row_idx);
    }

    // ============================================================================
    // BF16 Softmax - AVX512 Implementation
    // ============================================================================

    inline void softmax_row_bf16_avx512(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512) && defined(LLAMINAR_HAS_AVX512BF16)
        // TODO: Implement using AVX512_BF16 instructions
        // For now, fallback to scalar
        softmax_row_bf16_scalar(row, cols, causal, scale, row_idx);
#else
        softmax_row_bf16_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // BF16 Softmax - Compile-Time Dispatch
    // ============================================================================

    inline void softmax_row_bf16(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512) && defined(LLAMINAR_HAS_AVX512BF16)
        softmax_row_bf16_avx512(row, cols, causal, scale, row_idx);
#elif defined(LLAMINAR_HAS_AVX2)
        softmax_row_bf16_avx2(row, cols, causal, scale, row_idx);
#else
        softmax_row_bf16_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // FP16 Softmax - Scalar Implementation
    // ============================================================================

    inline void softmax_row_fp16_scalar(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // Pass 1: Find max (convert FP16→FP32 on the fly)
        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
#if defined(LLAMINAR_HAS_F16C)
            float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
#else
            float v = detail::fp16_to_fp32_scalar(row[c]) * scale;
#endif
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
#if defined(LLAMINAR_HAS_F16C)
            float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
#else
            float v = detail::fp16_to_fp32_scalar(row[c]) * scale;
#endif
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize and convert FP32→FP16
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0;
            }
            else
            {
#if defined(LLAMINAR_HAS_F16C)
                float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
                float result = std::exp(v - row_max) * inv;
                row[c] = detail::fp32_to_fp16_f16c(result);
#else
                float v = detail::fp16_to_fp32_scalar(row[c]) * scale;
                float result = std::exp(v - row_max) * inv;
                row[c] = detail::fp32_to_fp16_scalar(result);
#endif
            }
        }
    }

    // ============================================================================
    // FP16 Softmax - AVX2 Implementation (uses F16C)
    // ============================================================================

    inline void softmax_row_fp16_avx2(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX2) && defined(LLAMINAR_HAS_F16C)
        const __m256 scale_ps = _mm256_set1_ps(scale);
        const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

        // Pass 1: Find max (convert FP16→FP32)
        __m256 vmax = neg_inf;
        int c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + c));
            __m256 v = _mm256_cvtph_ps(fp16_vec);
            v = _mm256_mul_ps(v, scale_ps);

            if (causal)
            {
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, v);
                for (int lane = 0; lane < 8; ++lane)
                {
                    if (c + lane > row_idx)
                        tmp[lane] = -std::numeric_limits<float>::infinity();
                }
                v = _mm256_load_ps(tmp);
            }

            vmax = _mm256_max_ps(vmax, v);
        }

        float row_max = detail::hmax256(vmax);

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize and convert FP32→FP16
        const __m256 inv_ps = _mm256_set1_ps(inv);
        const __m256 max_ps = _mm256_set1_ps(row_max);

        c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + c));
            __m256 v = _mm256_cvtph_ps(fp16_vec);
            v = _mm256_mul_ps(v, scale_ps);

            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, v);

            float out[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                if (causal && c + lane > row_idx)
                {
                    out[lane] = 0.0f;
                }
                else
                {
                    out[lane] = std::exp(tmp[lane] - row_max) * inv;
                }
            }

            __m256 out_ps = _mm256_load_ps(out);
            __m128i fp16_out = _mm256_cvtps_ph(out_ps, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(row + c), fp16_out);
        }

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0;
            }
            else
            {
                float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
                float result = std::exp(v - row_max) * inv;
                row[c] = detail::fp32_to_fp16_f16c(result);
            }
        }
#else
        // Fallback to scalar if F16C not available
        softmax_row_fp16_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // FP16 Softmax - AVX512 Implementation
    // ============================================================================

    inline void softmax_row_fp16_avx512(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // TODO: Implement using AVX512FP16 if available
        // For now, fallback to AVX2 (F16C)
        softmax_row_fp16_avx2(row, cols, causal, scale, row_idx);
    }

    // ============================================================================
    // FP16 Softmax - Compile-Time Dispatch
    // ============================================================================

    inline void softmax_row_fp16(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512)
        softmax_row_fp16_avx512(row, cols, causal, scale, row_idx);
#elif defined(LLAMINAR_HAS_AVX2) && defined(LLAMINAR_HAS_F16C)
        softmax_row_fp16_avx2(row, cols, causal, scale, row_idx);
#else
        softmax_row_fp16_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // Q8_1 Integer-Aware Softmax - Scalar Implementation (OPTIMIZED)
    // ============================================================================
    // Optimizations applied:
    // 1. ELIMINATED HEAP ALLOCATION: Use fixed-size stack buffer (32 floats = 1 block)
    // 2. FUSED Pass 2+3: Single pass for exp sum AND max_prob finding
    // 3. STREAMING REQUANT: Process block-by-block to stay in cache
    // 4. REDUCED REDUNDANT SCALE CONVERSION: Cache bf16→fp32 per block

    inline void softmax_row_q8_1_scalar(
        Q8_1Block *row,
        int n_blocks,
        bool causal,
        float scale,
        int row_idx)
    {
        // ========================================================================
        // Pass 1: Find Maximum (integer-optimized)
        // ========================================================================
        // Find max(qs[i] * d) across all blocks
        // Key insight: We find integer max per block, then scale and reduce

        float row_max = -std::numeric_limits<float>::infinity();

        for (int b = 0; b < n_blocks; ++b)
        {
            const Q8_1Block &block = row[b];
            const int col_start = b * 32;

            // Convert BF16 scale to FP32 once per block
            const float d = detail::bf16_to_fp32_scalar(block.d);

            // Find max within block (integer comparison)
            int8_t block_max_qs = -128;
            for (int i = 0; i < 32; ++i)
            {
                const int col = col_start + i;
                if (causal && col > row_idx)
                    continue;
                if (block.qs[i] > block_max_qs)
                    block_max_qs = block.qs[i];
            }

            // Scale to FP32 and update row max
            if (block_max_qs > -128)
            { // Only if we found valid elements
                float block_max = static_cast<float>(block_max_qs) * d * scale;
                if (block_max > row_max)
                    row_max = block_max;
            }
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // ========================================================================
        // Pass 2 (FUSED): Compute Sum of Exponentials AND Find Max Probability
        // ========================================================================
        // We compute exp(v - max) and track both:
        // - sum: total for normalization
        // - max_prob_unnorm: max(exp(v - max)) for requantization scale
        // Since exp is monotonic, max_prob_unnorm = exp(0) = 1.0 when v == row_max

        double sum = 0.0;
        // Note: max_prob_unnorm is always 1.0 since exp(row_max - row_max) = exp(0) = 1
        // This is mathematically guaranteed, so we don't need to track it!

        for (int b = 0; b < n_blocks; ++b)
        {
            const Q8_1Block &block = row[b];
            const int col_start = b * 32;
            const float d = detail::bf16_to_fp32_scalar(block.d);
            const float scaled_d = d * scale;

            for (int i = 0; i < 32; ++i)
            {
                const int col = col_start + i;
                if (causal && col > row_idx)
                    continue;
                float v = static_cast<float>(block.qs[i]) * scaled_d;
                sum += std::exp(v - row_max);
            }
        }

        if (sum <= 0.0)
            sum = 1.0;

        const float inv_sum = static_cast<float>(1.0 / sum);

        // max_prob = 1.0 / sum (the maximum softmax value is always exp(0)/sum = 1/sum)
        // Wait, that's not right. max_prob = exp(max - max) * inv_sum = 1.0 * inv_sum = inv_sum
        // But inv_sum = 1/sum, and softmax max is 1/sum only if there's one dominant element.
        // For proper requantization, we need to find actual max prob.
        //
        // OPTIMIZATION: For softmax, max_prob = inv_sum (occurs at the position where v == row_max)
        // This is exact: prob_max = exp(row_max - row_max) / sum = 1 / sum
        const float max_prob = inv_sum;

        // ========================================================================
        // Pass 3 (OPTIMIZED): Streaming Requantize Block-by-Block
        // ========================================================================
        // NO HEAP ALLOCATION: Use stack buffer for one block at a time
        // This keeps working set in L1 cache (32 floats = 128 bytes)

        // Compute requantization scale
        const float new_d = (max_prob > 0.0f) ? max_prob / 127.0f : 1.0f / 127.0f;
        const float inv_d = 1.0f / new_d;
        const uint16_t new_d_bf16 = detail::fp32_to_bf16_scalar(new_d);

        // Stack buffer for one block's probabilities (32 floats = 128 bytes)
        alignas(32) float block_probs[32];

        for (int b = 0; b < n_blocks; ++b)
        {
            Q8_1Block &block = row[b];
            const int col_start = b * 32;
            const float d = detail::bf16_to_fp32_scalar(block.d);
            const float scaled_d = d * scale;

            // Compute probabilities for this block into stack buffer
            for (int i = 0; i < 32; ++i)
            {
                const int col = col_start + i;
                if (causal && col > row_idx)
                {
                    block_probs[i] = 0.0f;
                }
                else
                {
                    float v = static_cast<float>(block.qs[i]) * scaled_d;
                    block_probs[i] = std::exp(v - row_max) * inv_sum;
                }
            }

            // Write new scale
            block.d = new_d_bf16;

            // Quantize and accumulate sum_qs
            int16_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_probs[i] * inv_d));
                q = std::max(-128, std::min(127, q));
                block.qs[i] = static_cast<int8_t>(q);
                sum_qs += block.qs[i];
            }
            block.sum_qs = sum_qs;
        }
    }

    // ============================================================================
    // Q8_1 Integer-Aware Softmax - AVX2 Implementation (OPTIMIZED)
    // ============================================================================
    // Optimizations applied:
    // 1. ELIMINATED HEAP ALLOCATION: Use fixed-size stack buffer (32 floats = 1 block)
    // 2. STREAMING REQUANT: Process block-by-block to stay in L1 cache
    // 3. VECTORIZED SUM_QS: Use AVX2 horizontal sum instead of scalar loop
    // 4. FUSED PROB→QUANT: Compute prob, quantize, accumulate in single pass

    inline void softmax_row_q8_1_avx2(
        Q8_1Block *row,
        int n_blocks,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX2)
        // ========================================================================
        // Pass 1: Find Maximum using AVX2
        // ========================================================================
        // Use vpmaxsb for 32-way INT8 max comparison per block

        float row_max = -std::numeric_limits<float>::infinity();
        const __m256 scale_ps = _mm256_set1_ps(scale);

        for (int b = 0; b < n_blocks; ++b)
        {
            const Q8_1Block &block = row[b];
            const int col_start = b * 32;

            // Load 32 INT8 values as two __m128i (AVX2 vpmaxsb works on 256-bit)
            __m256i qs_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));

            // Find max within 32 bytes using horizontal reduction
            // Split into two 128-bit halves
            __m128i lo = _mm256_castsi256_si128(qs_vec);
            __m128i hi = _mm256_extracti128_si256(qs_vec, 1);
            __m128i max_16 = _mm_max_epi8(lo, hi);

            // Reduce 16 bytes to 8 bytes
            __m128i shifted = _mm_srli_si128(max_16, 8);
            __m128i max_8 = _mm_max_epi8(max_16, shifted);

            // Reduce 8 bytes to 4 bytes
            shifted = _mm_srli_si128(max_8, 4);
            __m128i max_4 = _mm_max_epi8(max_8, shifted);

            // Reduce 4 bytes to 2 bytes
            shifted = _mm_srli_si128(max_4, 2);
            __m128i max_2 = _mm_max_epi8(max_4, shifted);

            // Reduce 2 bytes to 1 byte
            shifted = _mm_srli_si128(max_2, 1);
            __m128i max_1 = _mm_max_epi8(max_2, shifted);

            // Extract scalar max
            int8_t block_max_qs = static_cast<int8_t>(_mm_extract_epi8(max_1, 0));

            // Handle causal masking (may need to recompute if causal cuts this block)
            if (causal)
            {
                const int causal_end = std::min(row_idx + 1, col_start + 32);
                if (causal_end <= col_start)
                {
                    continue; // Entire block is masked
                }
                else if (causal_end < col_start + 32)
                {
                    // Partial block - recompute max for valid range
                    block_max_qs = -128;
                    for (int i = 0; i < causal_end - col_start; ++i)
                    {
                        if (block.qs[i] > block_max_qs)
                            block_max_qs = block.qs[i];
                    }
                }
            }

            // Convert BF16 scale to FP32 and compute block max
            const float d = detail::bf16_to_fp32_scalar(block.d);
            float block_max = static_cast<float>(block_max_qs) * d * scale;
            if (block_max > row_max)
                row_max = block_max;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // ========================================================================
        // Pass 2: Compute Sum of Exponentials using AVX2
        // ========================================================================
        // Process 8 FP32 values at a time after dequantization

        double sum = 0.0;
        const __m256 max_ps = _mm256_set1_ps(row_max);

        for (int b = 0; b < n_blocks; ++b)
        {
            const Q8_1Block &block = row[b];
            const int col_start = b * 32;
            const float d = detail::bf16_to_fp32_scalar(block.d);
            const __m256 d_ps = _mm256_set1_ps(d);
            const __m256 scaled_d_ps = _mm256_mul_ps(d_ps, scale_ps);

            // Process 32 elements in 4 groups of 8
            for (int g = 0; g < 4; ++g)
            {
                // Load 8 INT8 values and convert to FP32
                __m128i qs_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.qs + g * 8));
                __m128i qs_16 = _mm_cvtepi8_epi16(qs_8);
                __m256i qs_32 = _mm256_cvtepi16_epi32(qs_16);
                __m256 v = _mm256_cvtepi32_ps(qs_32);

                // Apply scale: v = v * d * scale
                v = _mm256_mul_ps(v, scaled_d_ps);

                // Subtract max
                v = _mm256_sub_ps(v, max_ps);

                // Handle causal masking
                if (causal)
                {
                    const int group_start = col_start + g * 8;
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        if (group_start + lane > row_idx)
                        {
                            // Mask out this lane (set to large negative)
                            alignas(32) float tmp[8];
                            _mm256_store_ps(tmp, v);
                            tmp[lane] = -100.0f; // Will exp to ~0
                            v = _mm256_load_ps(tmp);
                        }
                    }
                }

                // Compute exp and accumulate (scalar for accuracy)
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, v);
                for (int lane = 0; lane < 8; ++lane)
                {
                    const int col = col_start + g * 8 + lane;
                    if (causal && col > row_idx)
                        continue;
                    sum += std::exp(tmp[lane]);
                }
            }
        }

        if (sum <= 0.0)
            sum = 1.0;

        const float inv_sum = static_cast<float>(1.0 / sum);

        // max_prob = inv_sum (occurs at position where v == row_max)
        const float max_prob = inv_sum;

        // ========================================================================
        // Pass 3 (OPTIMIZED): Streaming Normalize and Requantize using AVX2
        // ========================================================================
        // NO HEAP ALLOCATION: Use stack buffer for one block (128 bytes)
        // VECTORIZED SUM_QS: Use AVX2 horizontal sum

        // Compute requantization scale
        const float new_d = (max_prob > 0.0f) ? max_prob / 127.0f : 1.0f / 127.0f;
        const float inv_d = 1.0f / new_d;
        const uint16_t new_d_bf16 = detail::fp32_to_bf16_scalar(new_d);

        // Stack buffer for one block's probabilities
        alignas(32) float block_probs[32];

        const __m256 inv_d_ps = _mm256_set1_ps(inv_d);
        const __m256 inv_sum_ps = _mm256_set1_ps(inv_sum);

        for (int b = 0; b < n_blocks; ++b)
        {
            Q8_1Block &block = row[b];
            const int col_start = b * 32;
            const float d = detail::bf16_to_fp32_scalar(block.d);
            const __m256 d_ps = _mm256_set1_ps(d);
            const __m256 scaled_d_ps = _mm256_mul_ps(d_ps, scale_ps);

            // Compute probabilities for this block into stack buffer
            for (int g = 0; g < 4; ++g)
            {
                __m128i qs_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.qs + g * 8));
                __m128i qs_16 = _mm_cvtepi8_epi16(qs_8);
                __m256i qs_32 = _mm256_cvtepi16_epi32(qs_16);
                __m256 v = _mm256_cvtepi32_ps(qs_32);
                v = _mm256_mul_ps(v, scaled_d_ps);
                v = _mm256_sub_ps(v, max_ps);

                // Handle causal masking
                if (causal)
                {
                    const int group_start = col_start + g * 8;
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        if (group_start + lane > row_idx)
                        {
                            alignas(32) float tmp[8];
                            _mm256_store_ps(tmp, v);
                            tmp[lane] = -100.0f;
                            v = _mm256_load_ps(tmp);
                        }
                    }
                }

                // Compute exp (scalar for accuracy) and store
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, v);
                for (int lane = 0; lane < 8; ++lane)
                {
                    block_probs[g * 8 + lane] = std::exp(tmp[lane]) * inv_sum;
                }
            }

            block.d = new_d_bf16;

            // OPTIMIZED: Vectorized quantization with AVX2 horizontal sum
            __m256i sum_qs_vec = _mm256_setzero_si256();

            // Process 32 elements in 4 groups of 8
            for (int g = 0; g < 4; ++g)
            {
                // Load 8 probabilities
                __m256 prob_ps = _mm256_load_ps(&block_probs[g * 8]);

                // Scale by inverse of new_d
                __m256 scaled_ps = _mm256_mul_ps(prob_ps, inv_d_ps);

                // Round to nearest
                __m256 rounded_ps = _mm256_round_ps(scaled_ps, _MM_FROUND_TO_NEAREST_INT);

                // Convert to INT32
                __m256i int32_vec = _mm256_cvtps_epi32(rounded_ps);

                // Pack to INT16 (with saturation)
                __m128i lo = _mm256_castsi256_si128(int32_vec);
                __m128i hi = _mm256_extracti128_si256(int32_vec, 1);
                __m128i int16_vec = _mm_packs_epi32(lo, hi);

                // Pack to INT8 (with saturation)
                __m128i int8_vec = _mm_packs_epi16(int16_vec, int16_vec);

                // Store 8 INT8 values
                int64_t packed = _mm_cvtsi128_si64(int8_vec);
                std::memcpy(block.qs + g * 8, &packed, 8);

                // Accumulate sum_qs: extend INT8 to INT32 and add
                // Use stored values for accuracy
                __m128i qs_loaded = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.qs + g * 8));
                __m128i qs_16_ext = _mm_cvtepi8_epi16(qs_loaded);
                __m256i qs_32_ext = _mm256_cvtepi16_epi32(qs_16_ext);
                sum_qs_vec = _mm256_add_epi32(sum_qs_vec, qs_32_ext);
            }

            // Horizontal sum of sum_qs_vec (8x INT32 → 1x INT32)
            __m128i sum_lo = _mm256_castsi256_si128(sum_qs_vec);
            __m128i sum_hi = _mm256_extracti128_si256(sum_qs_vec, 1);
            __m128i sum_4 = _mm_add_epi32(sum_lo, sum_hi);
            __m128i sum_2 = _mm_add_epi32(sum_4, _mm_srli_si128(sum_4, 8));
            __m128i sum_1 = _mm_add_epi32(sum_2, _mm_srli_si128(sum_2, 4));
            block.sum_qs = static_cast<int16_t>(_mm_cvtsi128_si32(sum_1));
        }
#else
        // Fallback to scalar if AVX2 not available
        softmax_row_q8_1_scalar(row, n_blocks, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // Q8_1 Integer-Aware Softmax - AVX512 Implementation (OPTIMIZED)
    // ============================================================================
    // Optimizations applied:
    // 1. ELIMINATED HEAP ALLOCATION: Use fixed-size stack buffer (32 floats = 1 block)
    // 2. STREAMING REQUANT: Process block-by-block to stay in L1 cache
    // 3. VECTORIZED SUM_QS: Use AVX-512 horizontal sum instead of scalar loop
    // 4. OPTIMIZED CAUSAL MASK: Use __mmask16 blend directly

    inline void softmax_row_q8_1_avx512(
        Q8_1Block *row,
        int n_blocks,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512)
        // ========================================================================
        // Pass 1: Find Maximum using AVX-512
        // ========================================================================
        // Use vpmaxsb (zmm) for 64-way INT8 max, but Q8_1 block is 32 bytes

        float row_max = -std::numeric_limits<float>::infinity();
        const __m512 scale_ps = _mm512_set1_ps(scale);

        for (int b = 0; b < n_blocks; ++b)
        {
            const Q8_1Block &block = row[b];
            const int col_start = b * 32;

            // Load 32 INT8 values into lower half of zmm
            __m256i qs_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));

            // Use AVX2 horizontal reduction (32 bytes fits in ymm)
            __m128i lo = _mm256_castsi256_si128(qs_256);
            __m128i hi = _mm256_extracti128_si256(qs_256, 1);
            __m128i max_16 = _mm_max_epi8(lo, hi);
            __m128i shifted = _mm_srli_si128(max_16, 8);
            __m128i max_8 = _mm_max_epi8(max_16, shifted);
            shifted = _mm_srli_si128(max_8, 4);
            __m128i max_4 = _mm_max_epi8(max_8, shifted);
            shifted = _mm_srli_si128(max_4, 2);
            __m128i max_2 = _mm_max_epi8(max_4, shifted);
            shifted = _mm_srli_si128(max_2, 1);
            __m128i max_1 = _mm_max_epi8(max_2, shifted);

            int8_t block_max_qs = static_cast<int8_t>(_mm_extract_epi8(max_1, 0));

            // Handle causal masking
            if (causal)
            {
                const int causal_end = std::min(row_idx + 1, col_start + 32);
                if (causal_end <= col_start)
                {
                    continue;
                }
                else if (causal_end < col_start + 32)
                {
                    block_max_qs = -128;
                    for (int i = 0; i < causal_end - col_start; ++i)
                    {
                        if (block.qs[i] > block_max_qs)
                            block_max_qs = block.qs[i];
                    }
                }
            }

            const float d = detail::bf16_to_fp32_scalar(block.d);
            float block_max = static_cast<float>(block_max_qs) * d * scale;
            if (block_max > row_max)
                row_max = block_max;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // ========================================================================
        // Pass 2: Compute Sum of Exponentials using AVX-512
        // ========================================================================
        // Process 16 FP32 values at a time

        double sum = 0.0;
        const __m512 max_ps = _mm512_set1_ps(row_max);

        for (int b = 0; b < n_blocks; ++b)
        {
            const Q8_1Block &block = row[b];
            const int col_start = b * 32;
            const float d = detail::bf16_to_fp32_scalar(block.d);
            const __m512 d_ps = _mm512_set1_ps(d);
            const __m512 scaled_d_ps = _mm512_mul_ps(d_ps, scale_ps);

            // Process 32 elements in 2 groups of 16
            for (int g = 0; g < 2; ++g)
            {
                // Load 16 INT8 values and convert to FP32
                __m128i qs_8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs + g * 16));
                __m512i qs_32 = _mm512_cvtepi8_epi32(qs_8);
                __m512 v = _mm512_cvtepi32_ps(qs_32);

                // Apply scale: v = v * d * scale
                v = _mm512_mul_ps(v, scaled_d_ps);

                // Subtract max
                v = _mm512_sub_ps(v, max_ps);

                // Handle causal masking using AVX-512 mask (OPTIMIZED)
                if (causal)
                {
                    const int group_start = col_start + g * 16;
                    // Compute mask: valid if position <= row_idx
                    __mmask16 valid_mask = 0xFFFF;
                    const int first_invalid = row_idx - group_start + 1;
                    if (first_invalid < 16)
                    {
                        valid_mask = (first_invalid > 0) ? ((1 << first_invalid) - 1) : 0;
                    }
                    // Set masked lanes to large negative
                    v = _mm512_mask_blend_ps(valid_mask, _mm512_set1_ps(-100.0f), v);
                }

                // Compute exp using AVX-512 polynomial approximation
                __m512 exp_v = detail::exp512_ps(v);

                // Horizontal sum
                sum += static_cast<double>(_mm512_reduce_add_ps(exp_v));
            }
        }

        if (sum <= 0.0)
            sum = 1.0;

        const float inv_sum = static_cast<float>(1.0 / sum);

        // max_prob = inv_sum (occurs at position where v == row_max)
        const float max_prob = inv_sum;

        // ========================================================================
        // Pass 3 (OPTIMIZED): Streaming Normalize and Requantize using AVX-512
        // ========================================================================
        // NO HEAP ALLOCATION: Use stack buffer for one block (128 bytes)
        // VECTORIZED SUM_QS: Use AVX-512 horizontal sum

        // Compute requantization scale
        const float new_d = (max_prob > 0.0f) ? max_prob / 127.0f : 1.0f / 127.0f;
        const float inv_d = 1.0f / new_d;
        const uint16_t new_d_bf16 = detail::fp32_to_bf16_scalar(new_d);

        // Stack buffer for one block's probabilities
        alignas(64) float block_probs[32];

        const __m512 inv_d_ps = _mm512_set1_ps(inv_d);
        const __m512 inv_sum_ps = _mm512_set1_ps(inv_sum);

        for (int b = 0; b < n_blocks; ++b)
        {
            Q8_1Block &block = row[b];
            const int col_start = b * 32;
            const float d = detail::bf16_to_fp32_scalar(block.d);
            const __m512 d_ps = _mm512_set1_ps(d);
            const __m512 scaled_d_ps = _mm512_mul_ps(d_ps, scale_ps);

            // Compute probabilities for this block into stack buffer
            for (int g = 0; g < 2; ++g)
            {
                __m128i qs_8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs + g * 16));
                __m512i qs_32 = _mm512_cvtepi8_epi32(qs_8);
                __m512 v = _mm512_cvtepi32_ps(qs_32);
                v = _mm512_mul_ps(v, scaled_d_ps);
                v = _mm512_sub_ps(v, max_ps);

                // Handle causal masking (OPTIMIZED)
                if (causal)
                {
                    const int group_start = col_start + g * 16;
                    __mmask16 valid_mask = 0xFFFF;
                    const int first_invalid = row_idx - group_start + 1;
                    if (first_invalid < 16)
                    {
                        valid_mask = (first_invalid > 0) ? ((1 << first_invalid) - 1) : 0;
                    }
                    v = _mm512_mask_blend_ps(valid_mask, _mm512_set1_ps(-100.0f), v);
                }

                // Compute exp and normalize
                __m512 exp_v = detail::exp512_ps(v);
                __m512 prob_ps = _mm512_mul_ps(exp_v, inv_sum_ps);

                // Store probabilities to stack buffer
                _mm512_store_ps(&block_probs[g * 16], prob_ps);
            }

            block.d = new_d_bf16;

            // OPTIMIZED: Vectorized quantization with AVX-512 horizontal sum
            __m512i sum_qs_vec = _mm512_setzero_si512();

            // Process 32 elements in 2 groups of 16
            for (int g = 0; g < 2; ++g)
            {
                // Load 16 probabilities
                __m512 prob_ps = _mm512_load_ps(&block_probs[g * 16]);

                // Scale by inverse of new_d
                __m512 scaled_ps = _mm512_mul_ps(prob_ps, inv_d_ps);

                // Round to nearest and convert to INT32
                __m512i int32_vec = _mm512_cvtps_epi32(scaled_ps);

                // Clamp to INT8 range [-128, 127]
                int32_vec = _mm512_max_epi32(int32_vec, _mm512_set1_epi32(-128));
                int32_vec = _mm512_min_epi32(int32_vec, _mm512_set1_epi32(127));

                // Accumulate for sum_qs before packing (INT32 precision)
                sum_qs_vec = _mm512_add_epi32(sum_qs_vec, int32_vec);

                // Pack INT32 to INT8
                // AVX-512 doesn't have direct 32→8 pack, use two-step: 32→16→8
                __m256i int16_lo = _mm512_cvtepi32_epi16(int32_vec);
                __m128i int8_vec = _mm256_cvtepi16_epi8(int16_lo);

                // Store 16 INT8 values
                _mm_storeu_si128(reinterpret_cast<__m128i *>(block.qs + g * 16), int8_vec);
            }

            // Horizontal sum of sum_qs_vec (16x INT32 → 1x INT32)
            block.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(sum_qs_vec));
        }
#else
        // Fallback to AVX2 if AVX-512 not available
        softmax_row_q8_1_avx2(row, n_blocks, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // Q8_1 Integer-Aware Softmax - Compile-Time Dispatch
    // ============================================================================

    inline void softmax_row_q8_1(
        Q8_1Block *row,
        int n_blocks,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512)
        softmax_row_q8_1_avx512(row, n_blocks, causal, scale, row_idx);
#elif defined(LLAMINAR_HAS_AVX2)
        softmax_row_q8_1_avx2(row, n_blocks, causal, scale, row_idx);
#else
        softmax_row_q8_1_scalar(row, n_blocks, causal, scale, row_idx);
#endif
    }

} // namespace llaminar2::primitives
