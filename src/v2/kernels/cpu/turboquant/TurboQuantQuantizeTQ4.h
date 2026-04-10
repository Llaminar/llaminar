/**
 * @file TurboQuantQuantizeTQ4.h
 * @brief FP32 → TQ4 scalar-full quantization for TurboQuant KV cache
 * @author David Sanftenberg
 *
 * Quantizes FP32 vectors to 4-bit TQ4 blocks using scalar-full MSE centroids.
 * All 4 bits encode MSE centroid indices directly.
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
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include "../simd/AVX2Helpers.h"
#endif

namespace llaminar2
{

    inline void pack_bitplane_8(const uint8_t *bits, uint8_t *out)
    {
        uint8_t packed = 0;
        for (int i = 0; i < 8; ++i)
        {
            if (bits[i] & 0x1u)
                packed |= static_cast<uint8_t>(1u << i);
        }
        *out = packed;
    }

    // ========================================================================
    // Single-vector quantize: scalar full-index helpers
    //
    // Used for value vectors, where direct reconstruction fidelity matters more
    // than unbiased inner-product estimation. residual_norm < 0 signals this
    // storage mode at dequantization time.
    // ========================================================================

    // ========================================================================
    // Named ISA implementations: turboquant_quantize_tq4
    // ========================================================================

    /// Scalar quantize: FP32 → TQ4 (always compiles)
    template <int D>
    inline void turboquant_quantize_tq4_scalar(
        const float *input,
        const TurboQuantContext &ctx,
        TQ4Block<D> &out,
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
            std::memset(out.mse_indices, 0, TQ4Block<D>::MSE_BYTES);
            std::memset(out.high_bits, 0, TQ4Block<D>::HIGH_BIT_BYTES);
            return;
        }

        const float inv_norm = 1.0f / norm;
        float unit_vec[D];
        for (int i = 0; i < D; ++i)
            unit_vec[i] = input[i] * inv_norm;

        apply_rotation(ctx.rotation(), unit_vec, scratch0);

        const float scale = std::sqrt(static_cast<float>(D));
        for (int i = 0; i < D; ++i)
            scratch0[i] *= scale;

        for (int i = 0; i < D; i += 8)
        {
            uint8_t idx8[8];
            uint8_t high_bits[8];
            for (int j = 0; j < 8; ++j)
            {
                idx8[j] = tq4_nearest_centroid(scratch0[i + j]);
                high_bits[j] = static_cast<uint8_t>((idx8[j] >> 3) & 0x1u);
                idx8[j] &= 0x7u;
            }
            tq3_pack_8(idx8, out.mse_indices + (i / 8) * 3);
            pack_bitplane_8(high_bits, out.high_bits + (i / 8));
        }
    }

#if defined(__AVX2__)
    /// AVX2 quantize: FP32 → TQ4 (8-wide threshold popcount)
    template <int D>
    inline void turboquant_quantize_tq4_avx2(
        const float *input,
        const TurboQuantContext &ctx,
        TQ4Block<D> &out,
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
            std::memset(out.mse_indices, 0, TQ4Block<D>::MSE_BYTES);
            std::memset(out.high_bits, 0, TQ4Block<D>::HIGH_BIT_BYTES);
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

        for (int i = 0; i < D; i += 8)
        {
            __m256 vals = _mm256_loadu_ps(scratch0 + i);
            __m256i vidx = _mm256_setzero_si256();
            const __m256i one = _mm256_set1_epi32(1);

            for (int t = 0; t < 15; ++t)
            {
                const __m256 thresh = _mm256_set1_ps(TQ4_THRESHOLDS[t]);
                __m256 cmp = _mm256_cmp_ps(vals, thresh, _CMP_GT_OQ);
                __m256i gt_mask = _mm256_and_si256(_mm256_castps_si256(cmp), one);
                vidx = _mm256_add_epi32(vidx, gt_mask);
            }

            alignas(32) int32_t idx_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i *>(idx_arr), vidx);

            const int g = i / 8;
            uint8_t idx8[8];
            uint8_t high_bits[8];
            for (int j = 0; j < 8; ++j)
            {
                const int idx4 = idx_arr[j];
                high_bits[j] = static_cast<uint8_t>((idx4 >> 3) & 0x1u);
                idx8[j] = static_cast<uint8_t>(idx4 & 0x7u);
            }
            tq3_pack_8(idx8, out.mse_indices + g * 3);
            pack_bitplane_8(high_bits, out.high_bits + g);
        }
    }
#endif

#if defined(__AVX512F__)
    /// AVX-512 quantize: FP32 → TQ4 (16-wide threshold popcount)
    template <int D>
    inline void turboquant_quantize_tq4_avx512(
        const float *input,
        const TurboQuantContext &ctx,
        TQ4Block<D> &out,
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
            std::memset(out.mse_indices, 0, TQ4Block<D>::MSE_BYTES);
            std::memset(out.high_bits, 0, TQ4Block<D>::HIGH_BIT_BYTES);
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

        for (int i = 0; i < D; i += 16)
        {
            __m512 vals = _mm512_loadu_ps(scratch0 + i);
            __m512i vidx = _mm512_setzero_si512();
            const __m512i one = _mm512_set1_epi32(1);

            for (int t = 0; t < 15; ++t)
            {
                const __m512 thresh = _mm512_set1_ps(TQ4_THRESHOLDS[t]);
                const __mmask16 gt = _mm512_cmp_ps_mask(vals, thresh, _CMP_GT_OQ);
                vidx = _mm512_mask_add_epi32(vidx, gt, vidx, one);
            }

            alignas(64) int32_t idx_arr[16];
            _mm512_store_epi32(idx_arr, vidx);

            for (int half = 0; half < 2; ++half)
            {
                const int base = half * 8;
                const int g = (i + base) / 8;
                uint8_t idx8[8];
                uint8_t high_bits[8];
                for (int j = 0; j < 8; ++j)
                {
                    const int idx4 = idx_arr[base + j];
                    high_bits[j] = static_cast<uint8_t>((idx4 >> 3) & 0x1u);
                    idx8[j] = static_cast<uint8_t>(idx4 & 0x7u);
                }
                tq3_pack_8(idx8, out.mse_indices + g * 3);
                pack_bitplane_8(high_bits, out.high_bits + g);
            }
        }
    }
#endif

    /// Dispatch: picks best available ISA at compile time
    template <int D>
    inline void turboquant_quantize_tq4(
        const float *input,
        const TurboQuantContext &ctx,
        TQ4Block<D> &out,
        float *scratch0,
        float *scratch1)
    {
#if defined(__AVX512F__)
        turboquant_quantize_tq4_avx512(input, ctx, out, scratch0, scratch1);
#elif defined(__AVX2__)
        turboquant_quantize_tq4_avx2(input, ctx, out, scratch0, scratch1);
#else
        turboquant_quantize_tq4_scalar(input, ctx, out, scratch0, scratch1);
#endif
    }

} // namespace llaminar2
