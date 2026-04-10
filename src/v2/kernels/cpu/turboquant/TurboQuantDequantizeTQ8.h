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

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include "../simd/AVX2Helpers.h"
#endif

#include "../../../utils/CPUFeatures.h"

namespace llaminar2
{

    // ========================================================================
    // Named ISA implementations: turboquant_dequantize_tq8
    // ========================================================================

    /// Scalar dequantize: TQ8 → FP32 (always compiles)
    template <int D>
    inline void turboquant_dequantize_tq8_scalar(
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
        for (int i = 0; i < D; ++i)
            scratch[i] = TQ8_CENTROIDS[block.indices[i]] * inv_scale;
        apply_rotation_transpose(ctx.rotation(), scratch, output);
        for (int i = 0; i < D; ++i)
            output[i] *= block.norm;
    }

#if defined(__AVX2__)
    /// AVX2 dequantize: TQ8 → FP32 (8-wide gather + scale)
    template <int D>
    inline void turboquant_dequantize_tq8_avx2(
        const TQ8Block<D> &block,
        const TurboQuantContext &ctx,
        float *output,
        float *scratch)
    {
        static_assert(D % 8 == 0, "D must be a multiple of 8 for AVX2");
        if (block.norm < 1e-30f)
        {
            for (int i = 0; i < D; ++i)
                output[i] = 0.0f;
            return;
        }
        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));
        const __m256 vinv_scale = _mm256_set1_ps(inv_scale);
        for (int i = 0; i < D; i += 8)
        {
            __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.indices + i));
            __m256i vidx = _mm256_cvtepu8_epi32(raw8);
            __m256 vcentroids = _mm256_i32gather_ps(TQ8_CENTROIDS.data(), vidx, sizeof(float));
            _mm256_storeu_ps(scratch + i, _mm256_mul_ps(vcentroids, vinv_scale));
        }
        apply_rotation_transpose(ctx.rotation(), scratch, output);
        const __m256 vnorm = _mm256_set1_ps(block.norm);
        for (int i = 0; i < D; i += 8)
        {
            __m256 v = _mm256_loadu_ps(output + i);
            _mm256_storeu_ps(output + i, _mm256_mul_ps(v, vnorm));
        }
    }
#endif

#if defined(__AVX512F__)
    /// AVX-512 dequantize: TQ8 → FP32 (16-wide gather + scale)
    template <int D>
    inline void turboquant_dequantize_tq8_avx512(
        const TQ8Block<D> &block,
        const TurboQuantContext &ctx,
        float *output,
        float *scratch)
    {
        static_assert(D % 16 == 0, "D must be a multiple of 16 for AVX-512");
        if (block.norm < 1e-30f)
        {
            for (int i = 0; i < D; ++i)
                output[i] = 0.0f;
            return;
        }
        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));
        const __m512 vinv_scale = _mm512_set1_ps(inv_scale);
        for (int i = 0; i < D; i += 16)
        {
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.indices + i));
            __m512i vidx = _mm512_cvtepu8_epi32(raw);
            __m512 vcentroids = _mm512_i32gather_ps(vidx, TQ8_CENTROIDS.data(), sizeof(float));
            _mm512_storeu_ps(scratch + i, _mm512_mul_ps(vcentroids, vinv_scale));
        }
        apply_rotation_transpose(ctx.rotation(), scratch, output);
        const __m512 vnorm = _mm512_set1_ps(block.norm);
        for (int i = 0; i < D; i += 16)
        {
            __m512 v = _mm512_loadu_ps(output + i);
            _mm512_storeu_ps(output + i, _mm512_mul_ps(v, vnorm));
        }
    }
#endif

// Stubs for portability when ISA unavailable at compile time
#if !defined(__AVX2__)
    template <int D>
    inline void turboquant_dequantize_tq8_avx2(
        const TQ8Block<D> &block,
        const TurboQuantContext &ctx,
        float *output,
        float *scratch)
    {
        turboquant_dequantize_tq8_scalar(block, ctx, output, scratch);
    }
#endif
#if !defined(__AVX512F__)
    template <int D>
    inline void turboquant_dequantize_tq8_avx512(
        const TQ8Block<D> &block,
        const TurboQuantContext &ctx,
        float *output,
        float *scratch)
    {
        turboquant_dequantize_tq8_avx2(block, ctx, output, scratch);
    }
#endif

    /// Dispatch: picks best available ISA at runtime
    template <int D>
    inline void turboquant_dequantize_tq8(
        const TQ8Block<D> &block,
        const TurboQuantContext &ctx,
        float *output,
        float *scratch)
    {
        ISA_DISPATCH_VOID(turboquant_dequantize_tq8, block, ctx, output, scratch);
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
