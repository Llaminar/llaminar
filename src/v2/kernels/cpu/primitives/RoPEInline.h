/**
 * @file RoPEInline.h
 * @brief Inline SIMD RoPE helpers for FP32 tensors
 *
 * Provides apply_rope_to_head_inline() (single-head, precomputed sincos)
 * and apply_rope_to_k_fp32() (multi-row wrapper with inv_freq caching).
 *
 * These are header-only with AVX-512/AVX2/scalar fallback so they can be
 * inlined at call sites in both TQ dequant paths and the attention stage.
 */

#pragma once

#include "RoPEPrimitives.h" // get_inv_freq_cached
#include "utils/OpenMPUtils.h"

#include <cmath>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    /**
     * @brief Apply RoPE rotation to a single head's FP32 data in-place.
     *
     * Uses the standard RoPE formula with precomputed cos/sin:
     *   x[i]        = x[i] * cos - x[i+D/2] * sin
     *   x[i + D/2]  = x[i] * sin + x[i+D/2] * cos
     *
     * @param head_ptr  FP32 data for one head [head_dim]
     * @param cos_cache Precomputed cos values for this position [head_dim/2]
     * @param sin_cache Precomputed sin values for this position [head_dim/2]
     * @param head_dim  Head dimension (must be even)
     */
    inline void apply_rope_to_head_inline_scalar(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        for (int i = 0; i < half_dim; ++i)
        {
            float x0 = head_ptr[i];
            float x1 = head_ptr[i + half_dim];
            head_ptr[i] = x0 * cos_cache[i] - x1 * sin_cache[i];
            head_ptr[i + half_dim] = x0 * sin_cache[i] + x1 * cos_cache[i];
        }
    }

#if defined(__AVX2__)
    inline void apply_rope_to_head_inline_avx2(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;
        for (; i + 8 <= half_dim; i += 8)
        {
            __m256 x_first = _mm256_loadu_ps(head_ptr + i);
            __m256 x_second = _mm256_loadu_ps(head_ptr + i + half_dim);
            __m256 cos_vec = _mm256_loadu_ps(cos_cache + i);
            __m256 sin_vec = _mm256_loadu_ps(sin_cache + i);

            _mm256_storeu_ps(head_ptr + i,
                             _mm256_sub_ps(_mm256_mul_ps(x_first, cos_vec),
                                           _mm256_mul_ps(x_second, sin_vec)));
            _mm256_storeu_ps(head_ptr + i + half_dim,
                             _mm256_add_ps(_mm256_mul_ps(x_first, sin_vec),
                                           _mm256_mul_ps(x_second, cos_vec)));
        }
        for (; i < half_dim; ++i)
        {
            float x0 = head_ptr[i];
            float x1 = head_ptr[i + half_dim];
            head_ptr[i] = x0 * cos_cache[i] - x1 * sin_cache[i];
            head_ptr[i + half_dim] = x0 * sin_cache[i] + x1 * cos_cache[i];
        }
    }
#endif

#if defined(__AVX512F__)
    inline void apply_rope_to_head_inline_avx512(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;
        for (; i + 16 <= half_dim; i += 16)
        {
            __m512 x_first = _mm512_loadu_ps(head_ptr + i);
            __m512 x_second = _mm512_loadu_ps(head_ptr + i + half_dim);
            __m512 cos_vec = _mm512_loadu_ps(cos_cache + i);
            __m512 sin_vec = _mm512_loadu_ps(sin_cache + i);

            _mm512_storeu_ps(head_ptr + i,
                             _mm512_sub_ps(_mm512_mul_ps(x_first, cos_vec),
                                           _mm512_mul_ps(x_second, sin_vec)));
            _mm512_storeu_ps(head_ptr + i + half_dim,
                             _mm512_add_ps(_mm512_mul_ps(x_first, sin_vec),
                                           _mm512_mul_ps(x_second, cos_vec)));
        }
        for (; i < half_dim; ++i)
        {
            float x0 = head_ptr[i];
            float x1 = head_ptr[i + half_dim];
            head_ptr[i] = x0 * cos_cache[i] - x1 * sin_cache[i];
            head_ptr[i + half_dim] = x0 * sin_cache[i] + x1 * cos_cache[i];
        }
    }
#endif

    inline void apply_rope_to_head_inline(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
#if defined(__AVX512F__)
        apply_rope_to_head_inline_avx512(head_ptr, cos_cache, sin_cache, head_dim);
#elif defined(__AVX2__)
        apply_rope_to_head_inline_avx2(head_ptr, cos_cache, sin_cache, head_dim);
#else
        apply_rope_to_head_inline_scalar(head_ptr, cos_cache, sin_cache, head_dim);
#endif
    }

    /**
     * @brief Apply RoPE to an FP32 K tensor in-place.
     *
     * Used for prefill path where K comes from the projection buffer
     * (not from cache). Applies position embeddings to all rows.
     *
     * @param k_fp32         FP32 K data, layout [seq_len, kv_dim]
     * @param seq_len        Number of rows
     * @param head_dim       Head dimension
     * @param n_kv_heads     Number of KV heads
     * @param rope_theta     RoPE frequency base
     * @param position_start Starting position index
     */
    inline void apply_rope_to_k_fp32(
        float *k_fp32,
        int seq_len, int head_dim, int n_kv_heads,
        float rope_theta, int position_start)
    {
        const int kv_dim = n_kv_heads * head_dim;
        const int half_dim = head_dim / 2;
        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, rope_theta);

        auto work = [&]()
        {
            alignas(64) float cos_buf[128];
            alignas(64) float sin_buf[128];

#pragma omp for schedule(static)
            for (int r = 0; r < seq_len; ++r)
            {
                float *k_row = k_fp32 + r * kv_dim;
                const int position = position_start + r;

                for (int i = 0; i < half_dim; ++i)
                {
                    const float angle = static_cast<float>(position) * inv_freq[static_cast<size_t>(i)];
                    cos_buf[i] = std::cos(angle);
                    sin_buf[i] = std::sin(angle);
                }

                for (int h = 0; h < n_kv_heads; ++h)
                {
                    apply_rope_to_head_inline(k_row + h * head_dim,
                                              cos_buf, sin_buf, head_dim);
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work, seq_len >= 4);
    }

} // namespace llaminar2
