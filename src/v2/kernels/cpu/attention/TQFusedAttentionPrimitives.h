/**
 * @file TQFusedAttentionPrimitives.h
 * @brief SIMD primitives for fused TurboQuant KV attention (zero shadow buffers).
 *
 * ## Key Mathematical Insight
 *
 * TurboQuant dequantization is: dequant(b) = (norm/√D) · Πᵀ · c(b)
 * where c(b) is the centroid vector looked up from block indices.
 *
 * Since Π is orthogonal (Πᵀ = Π⁻¹):
 *   dot(Q, dequant(b)) = (norm/√D) · dot(Π·Q, c(b))
 *
 * This allows us to:
 *   - Pre-rotate Q once per head: Q_rot = Π·Q          [O(D²), amortized]
 *   - Per K position: gather centroids + dot with Q_rot  [O(D) per position]
 *   - Per V position: accumulate weighted centroids       [O(D) per position]
 *   - One final inverse rotation for V output             [O(D²), once]
 *
 * Total: O(D² + kv_len·D) instead of O(kv_len·D²) — up to D× faster.
 *
 * ## Primitives provided
 *
 * - tq8_dot_rotated_q():   dot(Q_rot, centroids(TQ8_block)) · norm
 * - tq4_accum_weighted():  accum += weight · norm · centroids(TQ4_block)
 *
 * Both operate on raw block bytes with runtime head_dim, no templates needed.
 * Attention scale (1/√D) is applied by the caller for flexibility.
 */

#pragma once

#include "../turboquant/TurboQuantCodebook.h"
#include "../../../tensors/BlockStructures.h"

#include <cmath>
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
    // Bit unpacking utility (inline, matches TurboQuantDequantizeTQ4.h)
    // ========================================================================

    /// Unpack one byte into 8 individual bits (LSB first).
    /// Duplicated here (same inline body as TurboQuantDequantizeTQ4.h) to avoid
    /// heavy transitive includes from the TQ dequant header.
    namespace tq_attn_detail
    {
        inline void unpack_bitplane_8_local(const uint8_t *packed, uint8_t *out)
        {
            const uint8_t bits = *packed;
            for (int i = 0; i < 8; ++i)
                out[i] = static_cast<uint8_t>((bits >> i) & 0x1u);
        }
    } // namespace tq_attn_detail

    // ========================================================================
    // TQ8 K dot product: score = dot(Q_rot, centroids) · norm
    // ========================================================================

    /**
     * @brief Compute dot(Q_rot, centroid_vector(K_block)) × norm.
     *
     * The caller provides Q_rot = Π·Q (pre-rotated query). This function
     * gathers TQ8 centroids from block indices and computes the raw dot product
     * scaled by the block's norm. The caller applies the attention scale (1/D).
     *
     * Cost: O(D) — 8 gather+FMA iterations for D=128 on AVX-512.
     *
     * @param Q_rot      Pre-rotated query vector [head_dim], Q_rot = Π·Q
     * @param tq8_block  Raw TQ8 block bytes: [norm:f32, residual:f32, indices:u8×D]
     * @param head_dim   Head dimension (64 or 128)
     * @return dot(Q_rot, centroids) × norm, or 0.0f for zero-norm blocks
     */
    // ========================================================================
    // Named ISA implementations: tq8_dot_rotated_q
    // ========================================================================

    /// Scalar: dot(Q_rot, centroids(TQ8)) × norm
    inline float tq8_dot_rotated_q_scalar(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        float norm;
        std::memcpy(&norm, tq8_block, sizeof(float));
        if (norm < 1e-30f)
            return 0.0f;
        const uint8_t *indices = tq8_block + 8;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i)
            dot += Q_rot[i] * TQ8_CENTROIDS[indices[i]];
        return dot * norm;
    }

#if defined(__AVX2__)
    /// AVX2: dot(Q_rot, centroids(TQ8)) × norm (8-wide gather+FMA)
    inline float tq8_dot_rotated_q_avx2(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        float norm;
        std::memcpy(&norm, tq8_block, sizeof(float));
        if (norm < 1e-30f)
            return 0.0f;
        const uint8_t *indices = tq8_block + 8;

        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= head_dim; i += 8)
        {
            const __m128i vidx_u8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i *>(indices + i));
            const __m256i vidx_i32 = _mm256_cvtepu8_epi32(vidx_u8);
            const __m256 vcentroids = _mm256_i32gather_ps(
                TQ8_CENTROIDS.data(), vidx_i32, sizeof(float));
            const __m256 vq = _mm256_loadu_ps(Q_rot + i);
            acc = _mm256_fmadd_ps(vcentroids, vq, acc);
        }
        float dot = avx2::hsum_ps(acc);
        for (; i < head_dim; ++i)
            dot += Q_rot[i] * TQ8_CENTROIDS[indices[i]];
        return dot * norm;
    }
#endif

#if defined(__AVX512F__)
    /// AVX-512: dot(Q_rot, centroids(TQ8)) × norm (16-wide gather+FMA)
    inline float tq8_dot_rotated_q_avx512(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        float norm;
        std::memcpy(&norm, tq8_block, sizeof(float));
        if (norm < 1e-30f)
            return 0.0f;
        const uint8_t *indices = tq8_block + 8;

        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= head_dim; i += 16)
        {
            const __m128i vidx_u8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(indices + i));
            const __m512i vidx_i32 = _mm512_cvtepu8_epi32(vidx_u8);
            const __m512 vcentroids = _mm512_i32gather_ps(
                vidx_i32, TQ8_CENTROIDS.data(), sizeof(float));
            const __m512 vq = _mm512_loadu_ps(Q_rot + i);
            acc = _mm512_fmadd_ps(vcentroids, vq, acc);
        }
        float dot = _mm512_reduce_add_ps(acc);
        for (; i < head_dim; ++i)
            dot += Q_rot[i] * TQ8_CENTROIDS[indices[i]];
        return dot * norm;
    }
#endif

// Stubs for portability when ISA unavailable at compile time
#if !defined(__AVX2__)
    inline float tq8_dot_rotated_q_avx2(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        return tq8_dot_rotated_q_scalar(Q_rot, tq8_block, head_dim);
    }
#endif
#if !defined(__AVX512F__)
    inline float tq8_dot_rotated_q_avx512(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        return tq8_dot_rotated_q_avx2(Q_rot, tq8_block, head_dim);
    }
#endif

    /// Dispatch: picks best available ISA at runtime
    inline float tq8_dot_rotated_q(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        return ISA_DISPATCH_RETVAL(tq8_dot_rotated_q, Q_rot, tq8_block, head_dim);
    }

    // ========================================================================
    // TQ4 V accumulation in rotated space
    // ========================================================================

    /**
     * @brief Accumulate weighted TQ4 centroid vector into rotated-space accumulator.
     *
     * Computes: accum[i] += weight × norm × TQ4_CENTROIDS[index(block, i)]
     * for all i in [0, head_dim). This accumulates in the "rotated" centroid
     * space — the caller applies Πᵀ once at the end.
     *
     * The combined weight already includes the attention softmax weight.
     * The 1/√D factor is NOT applied here — the caller handles it during
     * the final inverse rotation + normalization.
     *
     * Cost: O(D) — 8 iterations for D=128 on AVX-512, with 3+1 bit unpacking.
     *
     * @param accum      Rotated-space accumulator [head_dim], modified in-place
     * @param tq4_block  Raw TQ4 block bytes: [norm:f32, residual:f32,
     *                   mse_indices:D*3/8 bytes, high_bits:D/8 bytes]
     * @param weight     Softmax attention weight for this position
     * @param head_dim   Head dimension (64 or 128)
     */
    // ========================================================================
    // Named ISA implementations: tq4_accum_weighted
    // ========================================================================

    /// Scalar: accum += weight × norm × centroids(TQ4)
    inline void tq4_accum_weighted_scalar(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        float norm;
        std::memcpy(&norm, tq4_block, sizeof(float));
        if (norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;
        const float combined_weight = weight * norm;
        const uint8_t *mse_indices = tq4_block + 8;
        const uint8_t *high_bits = tq4_block + 8 + head_dim * 3 / 8;

        for (int i = 0; i < head_dim; i += 8)
        {
            const int group = i / 8;
            uint8_t idx8[8];
            uint8_t hb[8];
            tq3_unpack_8(mse_indices + group * 3, idx8);
            tq_attn_detail::unpack_bitplane_8_local(high_bits + group, hb);
            for (int j = 0; j < 8; ++j)
            {
                const uint8_t idx4 = idx8[j] | static_cast<uint8_t>(hb[j] << 3);
                accum[i + j] += combined_weight * TQ4_CENTROIDS[idx4];
            }
        }
    }

#if defined(__AVX2__)
    /// AVX2: accum += weight × norm × centroids(TQ4) (8-wide gather+FMA)
    inline void tq4_accum_weighted_avx2(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        float norm;
        std::memcpy(&norm, tq4_block, sizeof(float));
        if (norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;
        const float combined_weight = weight * norm;
        const uint8_t *mse_indices = tq4_block + 8;
        const uint8_t *high_bits = tq4_block + 8 + head_dim * 3 / 8;

        const __m256 vw = _mm256_set1_ps(combined_weight);
        for (int i = 0; i < head_dim; i += 8)
        {
            const int group = i / 8;
            uint8_t idx8[8];
            uint8_t hb[8];
            tq3_unpack_8(mse_indices + group * 3, idx8);
            tq_attn_detail::unpack_bitplane_8_local(high_bits + group, hb);

            alignas(32) int32_t idx32[8];
            for (int j = 0; j < 8; ++j)
                idx32[j] = idx8[j] | (hb[j] << 3);

            const __m256i vidx = _mm256_load_si256(reinterpret_cast<const __m256i *>(idx32));
            const __m256 vcentroids = _mm256_i32gather_ps(
                TQ4_CENTROIDS.data(), vidx, sizeof(float));
            __m256 vacc = _mm256_loadu_ps(accum + i);
            vacc = _mm256_fmadd_ps(vcentroids, vw, vacc);
            _mm256_storeu_ps(accum + i, vacc);
        }
    }
#endif

#if defined(__AVX512F__)
    /// AVX-512: accum += weight × norm × centroids(TQ4) (16-wide gather+FMA)
    inline void tq4_accum_weighted_avx512(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        float norm;
        std::memcpy(&norm, tq4_block, sizeof(float));
        if (norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;
        const float combined_weight = weight * norm;
        const uint8_t *mse_indices = tq4_block + 8;
        const uint8_t *high_bits = tq4_block + 8 + head_dim * 3 / 8;

        const __m512 vw = _mm512_set1_ps(combined_weight);
        for (int i = 0; i < head_dim; i += 16)
        {
            alignas(64) int32_t idx32[16];
            for (int g = 0; g < 2; ++g)
            {
                const int base = i + g * 8;
                const int group = base / 8;
                uint8_t idx8[8];
                uint8_t hb[8];
                tq3_unpack_8(mse_indices + group * 3, idx8);
                tq_attn_detail::unpack_bitplane_8_local(high_bits + group, hb);
                for (int j = 0; j < 8; ++j)
                    idx32[g * 8 + j] = idx8[j] | (hb[j] << 3);
            }
            const __m512i vidx = _mm512_load_si512(idx32);
            const __m512 vcentroids = _mm512_i32gather_ps(
                vidx, TQ4_CENTROIDS.data(), sizeof(float));
            __m512 vacc = _mm512_loadu_ps(accum + i);
            vacc = _mm512_fmadd_ps(vcentroids, vw, vacc);
            _mm512_storeu_ps(accum + i, vacc);
        }
    }
#endif

// Stubs for portability when ISA unavailable at compile time
#if !defined(__AVX2__)
    inline void tq4_accum_weighted_avx2(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        tq4_accum_weighted_scalar(accum, tq4_block, weight, head_dim);
    }
#endif
#if !defined(__AVX512F__)
    inline void tq4_accum_weighted_avx512(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        tq4_accum_weighted_avx2(accum, tq4_block, weight, head_dim);
    }
#endif

    /// Dispatch: picks best available ISA at runtime
    inline void tq4_accum_weighted(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        ISA_DISPATCH_VOID(tq4_accum_weighted, accum, tq4_block, weight, head_dim);
    }

} // namespace llaminar2
