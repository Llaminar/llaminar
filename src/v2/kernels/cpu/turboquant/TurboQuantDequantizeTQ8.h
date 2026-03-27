/**
 * @file TurboQuantDequantizeTQ8.h
 * @brief TQ8 → FP32 dequantization for TurboQuant K-projection cache
 *
 * Dequantizes TQ8 blocks back to FP32 vectors using 256-level Lloyd-Max
 * centroid lookup. Simpler than TQ4 — direct uint8_t → centroid lookup,
 * no bit unpacking.
 *
 * Provides single-vector entry point.
 */

#pragma once

#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantRotation.h"

#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    /**
     * @brief Dequantize one TQ8 block to FP32 vector.
     *
     * Path: uint8_t index → centroid lookup → ÷√D → inverse rotation → ×norm
     *
     * @tparam D Dimension of the vector (head_dim)
     * @param block   Input TQ8 block
     * @param ctx     Pre-generated TurboQuant context (rotation matrix)
     * @param output  FP32 output vector of length D
     * @param scratch Scratch buffer of at least D floats
     */
    template <int D>
    inline void turboquant_dequantize_tq8(
        const TQ8Block<D> &block,
        const TurboQuantContext &ctx,
        float *output,
        float *scratch)
    {
        if (block.norm < 1e-30f)
        {
            for (int i = 0; i < D; ++i)
                output[i] = 0.0f;
            return;
        }

        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));

#if defined(__AVX512F__)
        static_assert(D % 16 == 0, "D must be a multiple of 16 for AVX-512");

        // Centroid lookup + descale: gather from codebook using indices
        const __m512 vinv_scale = _mm512_set1_ps(inv_scale);
        for (int i = 0; i < D; i += 16)
        {
            // Widen 16 × uint8_t → 16 × int32 via VPMOVZXBD (single instruction)
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.indices + i));
            __m512i vidx = _mm512_cvtepu8_epi32(raw);

            __m512 vcentroids = _mm512_i32gather_ps(vidx, TQ8_CENTROIDS.data(), sizeof(float));
            __m512 vscaled = _mm512_mul_ps(vcentroids, vinv_scale);
            _mm512_storeu_ps(scratch + i, vscaled);
        }
#else
        // Scalar: direct centroid lookup
        for (int i = 0; i < D; ++i)
            scratch[i] = TQ8_CENTROIDS[block.indices[i]] * inv_scale;
#endif

        // Inverse rotation: Πᵀ × scratch → output
        apply_rotation_transpose(ctx.rotation(), scratch, output);

        // Scale by norm
#if defined(__AVX512F__)
        const __m512 vnorm = _mm512_set1_ps(block.norm);
        for (int i = 0; i < D; i += 16)
        {
            __m512 v = _mm512_loadu_ps(output + i);
            _mm512_storeu_ps(output + i, _mm512_mul_ps(v, vnorm));
        }
#else
        for (int i = 0; i < D; ++i)
            output[i] *= block.norm;
#endif
    }

    /**
     * @brief Dequantize a row of TQ8 blocks to FP32 (one block per head).
     *
     * @tparam D Head dimension
     * @param src_blocks Source: array of n_kv_heads TQ8Block<D>
     * @param ctx TurboQuant context
     * @param dst FP32 destination: [kv_dim] where kv_dim = n_kv_heads * D
     * @param n_kv_heads Number of KV heads
     * @param scratch Per-thread scratch of at least D floats
     */
    template <int D>
    inline void turboquant_dequantize_row_tq8(
        const TQ8Block<D> *src_blocks,
        const TurboQuantContext &ctx,
        float *dst,
        size_t n_kv_heads,
        float *scratch)
    {
        for (size_t h = 0; h < n_kv_heads; ++h)
        {
            const auto &head_ctx = ctx.for_layer(static_cast<int>(h));
            turboquant_dequantize_tq8<D>(src_blocks[h], head_ctx, dst + h * D, scratch);
        }
    }

} // namespace llaminar2
