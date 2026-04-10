/**
 * @file TurboQuantQuantizeTQ8.h
 * @brief FP32 → TQ8 scalar-full quantization for TurboQuant K-projection cache
 *
 * Quantizes FP32 vectors to 8-bit TQ8 blocks using 256-level Lloyd-Max
 * centroids. Much simpler than TQ4 — no bit packing, direct uint8_t indices.
 *
 * Provides single-vector entry point.
 */

#pragma once

#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"

#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include "../simd/AVX2Helpers.h"
#endif

namespace llaminar2
{

    /**
     * @brief Quantize one FP32 vector to a TQ8 block (scalar-full mode).
     *
     * Path: FP32 → normalize → rotate (Haar Π) → scale (×√D) →
     *       nearest 256-level Lloyd-Max centroid → uint8_t index per element.
     *
     * @tparam D Dimension of the vector (head_dim, typically 64 or 128)
     * @param input  FP32 input vector of length D
     * @param ctx    Pre-generated TurboQuant context (rotation matrix)
     * @param out    Output TQ8 block
     * @param scratch0 Scratch buffer of at least D floats
     * @param scratch1 Unused (kept for API symmetry with TQ4)
     */
    // ========================================================================
    // Named ISA implementations: turboquant_quantize_tq8
    // ========================================================================

    /// Scalar quantize: FP32 → TQ8 (always compiles)
    template <int D>
    inline void turboquant_quantize_tq8_scalar(
        const float *input,
        const TurboQuantContext &ctx,
        TQ8Block<D> &out,
        float *scratch0,
        float *scratch1)
    {
        (void)scratch1;

        float norm_sq = 0.0f;
        for (int i = 0; i < D; ++i)
            norm_sq += input[i] * input[i];
        const float norm = std::sqrt(norm_sq);
        out.norm = norm;
        out.residual_norm = -1.0f;

        if (norm < 1e-30f)
        {
            std::memset(out.indices, 0, D);
            return;
        }

        const float inv_norm = 1.0f / norm;
        alignas(64) float unit_vec[D];
        for (int i = 0; i < D; ++i)
            unit_vec[i] = input[i] * inv_norm;

        apply_rotation(ctx.rotation(), unit_vec, scratch0);

        const float scale = std::sqrt(static_cast<float>(D));
        for (int i = 0; i < D; ++i)
            scratch0[i] *= scale;

        for (int i = 0; i < D; ++i)
            out.indices[i] = tq8_nearest_centroid(scratch0[i]);
    }

#if defined(__AVX2__)
    /// AVX2 quantize: FP32 → TQ8 (8-wide binary search + gather)
    template <int D>
    inline void turboquant_quantize_tq8_avx2(
        const float *input,
        const TurboQuantContext &ctx,
        TQ8Block<D> &out,
        float *scratch0,
        float *scratch1)
    {
        (void)scratch1;
        static_assert(D % 8 == 0, "D must be a multiple of 8 for AVX2");

        __m256 vacc = _mm256_setzero_ps();
        for (int i = 0; i < D; i += 8)
        {
            __m256 v = _mm256_loadu_ps(input + i);
            vacc = _mm256_fmadd_ps(v, v, vacc);
        }
        const float norm_sq = avx2::hsum_ps(vacc);
        const float norm = std::sqrt(norm_sq);
        out.norm = norm;
        out.residual_norm = -1.0f;

        if (norm < 1e-30f)
        {
            std::memset(out.indices, 0, D);
            return;
        }

        const float combined_scale = std::sqrt(static_cast<float>(D)) / norm;
        alignas(32) float scaled_input[D];
        {
            const __m256 vcombined = _mm256_set1_ps(combined_scale);
            for (int i = 0; i < D; i += 8)
            {
                __m256 v = _mm256_loadu_ps(input + i);
                _mm256_store_ps(scaled_input + i, _mm256_mul_ps(v, vcombined));
            }
        }

        apply_rotation(ctx.rotation(), scaled_input, scratch0);

        const float *thresholds = TQ8_THRESHOLDS.data();
        for (int i = 0; i < D; i += 8)
        {
            __m256 vals = _mm256_loadu_ps(scratch0 + i);
            __m256i vidx = _mm256_setzero_si256();

#define TQ8_BSEARCH_STEP_AVX2(STEP, OFFSET)                               \
    {                                                                     \
        __m256i tpos = _mm256_add_epi32(vidx, _mm256_set1_epi32(OFFSET)); \
        __m256 tv = _mm256_i32gather_ps(thresholds, tpos, 4);             \
        __m256 cmp = _mm256_cmp_ps(vals, tv, _CMP_GT_OQ);                 \
        __m256i step_add = _mm256_and_si256(_mm256_castps_si256(cmp),     \
                                            _mm256_set1_epi32(STEP));     \
        vidx = _mm256_add_epi32(vidx, step_add);                          \
    }

            TQ8_BSEARCH_STEP_AVX2(128, 127)
            TQ8_BSEARCH_STEP_AVX2(64, 63)
            TQ8_BSEARCH_STEP_AVX2(32, 31)
            TQ8_BSEARCH_STEP_AVX2(16, 15)
            TQ8_BSEARCH_STEP_AVX2(8, 7)
            TQ8_BSEARCH_STEP_AVX2(4, 3)
            TQ8_BSEARCH_STEP_AVX2(2, 1)

            {
                __m256 tv = _mm256_i32gather_ps(thresholds, vidx, 4);
                __m256 cmp = _mm256_cmp_ps(vals, tv, _CMP_GT_OQ);
                __m256i step_add = _mm256_and_si256(_mm256_castps_si256(cmp),
                                                    _mm256_set1_epi32(1));
                vidx = _mm256_add_epi32(vidx, step_add);
            }

#undef TQ8_BSEARCH_STEP_AVX2

            __m256i shuf_mask = _mm256_setr_epi8(
                0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i shuffled = _mm256_shuffle_epi8(vidx, shuf_mask);
            uint32_t lo4 = static_cast<uint32_t>(_mm256_extract_epi32(shuffled, 0));
            uint32_t hi4 = static_cast<uint32_t>(_mm256_extract_epi32(shuffled, 4));
            std::memcpy(out.indices + i, &lo4, 4);
            std::memcpy(out.indices + i + 4, &hi4, 4);
        }
    }
#endif

#if defined(__AVX512F__)
    /// AVX-512 quantize: FP32 → TQ8 (16-wide binary search + gather)
    template <int D>
    inline void turboquant_quantize_tq8_avx512(
        const float *input,
        const TurboQuantContext &ctx,
        TQ8Block<D> &out,
        float *scratch0,
        float *scratch1)
    {
        (void)scratch1;
        static_assert(D % 16 == 0, "D must be a multiple of 16 for AVX-512");

        __m512 vacc = _mm512_setzero_ps();
        for (int i = 0; i < D; i += 16)
        {
            __m512 v = _mm512_loadu_ps(input + i);
            vacc = _mm512_fmadd_ps(v, v, vacc);
        }
        const float norm_sq = _mm512_reduce_add_ps(vacc);
        const float norm = std::sqrt(norm_sq);
        out.norm = norm;
        out.residual_norm = -1.0f;

        if (norm < 1e-30f)
        {
            std::memset(out.indices, 0, D);
            return;
        }

        const float combined_scale = std::sqrt(static_cast<float>(D)) / norm;
        alignas(64) float scaled_input[D];
        {
            const __m512 vcombined = _mm512_set1_ps(combined_scale);
            for (int i = 0; i < D; i += 16)
            {
                __m512 v = _mm512_loadu_ps(input + i);
                _mm512_store_ps(scaled_input + i, _mm512_mul_ps(v, vcombined));
            }
        }

        apply_rotation(ctx.rotation(), scaled_input, scratch0);

        const float *thresholds = TQ8_THRESHOLDS.data();
        for (int i = 0; i < D; i += 16)
        {
            __m512 vals = _mm512_loadu_ps(scratch0 + i);
            __m512i vidx = _mm512_setzero_si512();

#define TQ8_BSEARCH_STEP(STEP, OFFSET)                                         \
    {                                                                          \
        __m512i tpos = _mm512_add_epi32(vidx, _mm512_set1_epi32(OFFSET));      \
        __m512 tv = _mm512_i32gather_ps(tpos, thresholds, 4);                  \
        __mmask16 gt = _mm512_cmp_ps_mask(vals, tv, _CMP_GT_OQ);               \
        vidx = _mm512_mask_add_epi32(vidx, gt, vidx, _mm512_set1_epi32(STEP)); \
    }

            TQ8_BSEARCH_STEP(128, 127)
            TQ8_BSEARCH_STEP(64, 63)
            TQ8_BSEARCH_STEP(32, 31)
            TQ8_BSEARCH_STEP(16, 15)
            TQ8_BSEARCH_STEP(8, 7)
            TQ8_BSEARCH_STEP(4, 3)
            TQ8_BSEARCH_STEP(2, 1)

            {
                __m512 tv = _mm512_i32gather_ps(vidx, thresholds, 4);
                __mmask16 gt = _mm512_cmp_ps_mask(vals, tv, _CMP_GT_OQ);
                vidx = _mm512_mask_add_epi32(vidx, gt, vidx, _mm512_set1_epi32(1));
            }

#undef TQ8_BSEARCH_STEP

            __m128i packed = _mm512_cvtepi32_epi8(vidx);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out.indices + i), packed);
        }
    }
#endif

    /// Dispatch: picks best available ISA at compile time
    template <int D>
    inline void turboquant_quantize_tq8(
        const float *input,
        const TurboQuantContext &ctx,
        TQ8Block<D> &out,
        float *scratch0,
        float *scratch1)
    {
#if defined(__AVX512F__)
        turboquant_quantize_tq8_avx512(input, ctx, out, scratch0, scratch1);
#elif defined(__AVX2__)
        turboquant_quantize_tq8_avx2(input, ctx, out, scratch0, scratch1);
#else
        turboquant_quantize_tq8_scalar(input, ctx, out, scratch0, scratch1);
#endif
    }

    /**
     * @brief Quantize a row of FP32 data to TQ8 blocks (one block per head).
     *
     * @tparam D Head dimension
     * @param src Source FP32 data: [kv_dim] where kv_dim = n_kv_heads * D
     * @param ctx TurboQuant context
     * @param dst_blocks Destination: array of n_kv_heads TQ8Block<D>
     * @param n_kv_heads Number of KV heads
     * @param scratch0 Per-thread scratch of at least D floats
     * @param scratch1 Unused
     */
    template <int D>
    inline void turboquant_quantize_row_tq8(
        const float *src,
        const TurboQuantContext &ctx,
        TQ8Block<D> *dst_blocks,
        size_t n_kv_heads,
        float *scratch0,
        float *scratch1)
    {
        for (size_t h = 0; h < n_kv_heads; ++h)
        {
            turboquant_quantize_tq8<D>(src + h * D, ctx, dst_blocks[h], scratch0, scratch1);
        }
    }

} // namespace llaminar2
