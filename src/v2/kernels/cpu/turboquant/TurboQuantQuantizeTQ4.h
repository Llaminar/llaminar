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

#if defined(__AVX512F__)
#include <immintrin.h>
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

    template <int D>
    inline void turboquant_quantize_tq4(
        const float *input,
        const TurboQuantContext &ctx,
        TQ4Block<D> &out,
        float *scratch0,
        float *scratch1)
    {
        (void)scratch1;

#if defined(__AVX512F__)
        static_assert(D % 16 == 0, "D must be a multiple of 16 for AVX-512");

        // --- Norm computation (AVX-512 FMA + reduce) ---
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

        // --- Fused normalize + scale: input × (√D / norm) ---
        // Combines unit_vec and sqrt(D) scaling into a single pass.
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

        // --- Rotation (already AVX-512 internally) ---
        apply_rotation(ctx.rotation(), scaled_input, scratch0);

        // --- Nearest centroid (AVX-512 threshold-popcount, 16 elements at a time) ---
        // For each of 16 values, count how many of the 15 thresholds it exceeds.
        // This replaces the scalar binary search with fully parallel comparisons.
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

            // Extract indices and split into 3-bit low + 1-bit high, then pack
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

#else
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
#endif
    }

} // namespace llaminar2
