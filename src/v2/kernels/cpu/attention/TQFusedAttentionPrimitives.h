/**
 * @file TQFusedAttentionPrimitives.h
 * @brief SIMD primitives for fused TurboQuant KV attention (zero shadow buffers).
 *
 * ## Key Mathematical Insight
 *
 * TurboQuant value reconstruction is:
 * dequant(b) = (reconstruction_norm/√D) · Πᵀ · c(b)
 * where c(b) is the centroid vector looked up from block indices.
 *
 * Keys deliberately retain the source norm instead. A least-squares radius
 * minimizes reconstruction MSE, but it also introduces a data-dependent bias
 * into QK scores. The source norm preserves TurboQuant's score estimator:
 *   dot(Q, dequant(K)) = (source_norm/√D) · dot(Π·Q, c(K))
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
 * - tq8_dot_rotated_q():   dot(Q_rot, centroids(TQ8_block)) · source_norm
 * - tq4_accum_weighted():  accum += weight · reconstruction_norm · centroids(TQ4_block)
 * - tq8_accum_weighted():  accum += weight · reconstruction_norm · centroids(TQ8_block)
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
#include "CPUTQAttentionOperands.h"
#if LLAMINAR_COMPILED_WITH_NEON
#include "NEONTQAttentionPrimitives.h"
#endif

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
        /** @brief Expand one original high-bit plane into bounded diagnostic bytes.
         * @param packed Exactly one original byte; @param out Exactly eight private output bytes.
         */
        inline void unpack_bitplane_8_local(const uint8_t *packed, uint8_t *out)
        {
            const uint8_t bits = *packed;
            for (int i = 0; i < 8; ++i)
                out[i] = static_cast<uint8_t>((bits >> i) & 0x1u);
        }
    } // namespace tq_attn_detail

    // ========================================================================
    // TQ8 K dot product: score = dot(Q_rot, centroids) · source norm
    // ========================================================================

    /**
     * @brief Compute dot(Q_rot, centroid_vector(K_block)) × source norm.
     *
     * The caller provides Q_rot = Π·Q (pre-rotated query). This function
     * gathers TQ8 centroids from block indices and computes the raw dot product
     * scaled by the block's source norm. Using the value-oriented fitted
     * reconstruction radius here would minimize K-vector MSE but bias the QK
     * score distribution. The caller applies the combined TurboQuant and
     * attention scale (1/D).
     *
     * Cost: O(D) — 8 gather+FMA iterations for D=128 on AVX-512.
     *
     * @param Q_rot      Pre-rotated query vector [head_dim], Q_rot = Π·Q
     * @param tq8_block  Raw TQ8 block bytes: [norm:f32, reconstruction_norm:f32, indices:u8×D]
     * @param head_dim   Head dimension (64 or 128)
     * @return dot(Q_rot, centroids) × source norm, or 0.0f for zero blocks
     */
    // ========================================================================
    // Named ISA implementations: tq8_dot_rotated_q
    // ========================================================================

    /** @brief Explicit scalar diagnostic TQ8 scoring program.
     * @param Q_rot Original query; @param tq8_block Valid immutable physical block.
     * @param head_dim Complete diagnostic width, with valid original storage.
     * @return Ordered diagnostic dot times the original source norm.
     */
    inline float tq8_dot_rotated_q_scalar(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        float source_norm;
        std::memcpy(&source_norm, tq8_block, sizeof(float));
        if (source_norm < 1e-30f)
            return 0.0f;
        const uint8_t *indices = tq8_block + 8;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i)
            dot += Q_rot[i] * TQ8_CENTROIDS[indices[i]];
        return dot * source_norm;
    }

#if defined(__AVX2__)
    /** @brief Existing AVX2 eight-wide gather/FMA score program.
     * @param Q_rot Checked original query; @param tq8_block Checked immutable block.
     * @param head_dim Complete validated ABI width.
     * @return AVX2 score times the source norm; no alternate ISA is selected here.
     */
    inline float tq8_dot_rotated_q_avx2(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        float source_norm;
        std::memcpy(&source_norm, tq8_block, sizeof(float));
        if (source_norm < 1e-30f)
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
        return dot * source_norm;
    }
#endif

#if defined(__AVX512F__)
    /** @brief Existing AVX-512 sixteen-wide gather/FMA score program.
     * @param Q_rot Checked original query; @param tq8_block Checked immutable block.
     * @param head_dim Complete validated ABI width.
     * @return AVX-512 score times the original source norm.
     */
    inline float tq8_dot_rotated_q_avx512(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq8_block,
        int head_dim)
    {
        float source_norm;
        std::memcpy(&source_norm, tq8_block, sizeof(float));
        if (source_norm < 1e-30f)
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
        return dot * source_norm;
    }
#endif


    // ========================================================================
    // TQ4 K dot product: score = dot(Q_rot, centroids) * source norm
    // ========================================================================

    /**
     * @brief Compute a scalar dot product against one native TQ4 key block.
     *
     * TQ4 stores each centroid index as a three-bit MSE code plus a separate
     * high-bit plane. Decoding those two streams directly inside the score
     * loop avoids constructing an FP32 key row while preserving the exact
     * TurboQuant rotation and scaling used by the TQ8-key path.
     *
     * @param Q_rot Query after the per-head TurboQuant forward rotation.
     * @param tq4_block Raw TQ4 block bytes for one KV head and position.
     * @param head_dim Number of centroid indices in the block; must be a
     *        positive multiple of eight.
     * @return `dot(Q_rot, centroid_vector) * source_norm`.
     */
    inline float tq4_dot_rotated_q_scalar(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq4_block,
        int head_dim)
    {
        float source_norm;
        std::memcpy(&source_norm, tq4_block, sizeof(float));
        if (source_norm < 1e-30f)
            return 0.0f;

        const uint8_t *mse_indices = tq4_block + 8;
        const uint8_t *high_bits = tq4_block + 8 + head_dim * 3 / 8;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; i += 8)
        {
            const int group = i / 8;
            uint8_t low_indices[8];
            uint8_t high_indices[8];
            tq3_unpack_8(mse_indices + group * 3, low_indices);
            tq_attn_detail::unpack_bitplane_8_local(
                high_bits + group,
                high_indices);
            for (int lane = 0; lane < 8; ++lane)
            {
                const uint8_t centroid =
                    low_indices[lane] |
                    static_cast<uint8_t>(high_indices[lane] << 3);
                dot += Q_rot[i + lane] * TQ4_CENTROIDS[centroid];
            }
        }
        return dot * source_norm;
    }

#if defined(__AVX2__)
    /** @brief Existing AVX2 TQ4 score program with original-coordinate centroid gathers.
     * @param Q_rot Checked original query; @param tq4_block Checked immutable physical head.
     * @param head_dim Complete validated ABI width.
     * @return AVX2 ordered dot times the original source norm.
     */
    inline float tq4_dot_rotated_q_avx2(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq4_block,
        int head_dim)
    {
        float source_norm;
        std::memcpy(&source_norm, tq4_block, sizeof(float));
        if (source_norm < 1e-30f)
            return 0.0f;

        const uint8_t *mse_indices = tq4_block + 8;
        const uint8_t *high_bits = tq4_block + 8 + head_dim * 3 / 8;
        __m256 accumulator = _mm256_setzero_ps();
        for (int i = 0; i < head_dim; i += 8)
        {
            const int group = i / 8;
            uint8_t low_indices[8];
            uint8_t high_indices[8];
            alignas(32) int32_t expanded_indices[8];
            tq3_unpack_8(mse_indices + group * 3, low_indices);
            tq_attn_detail::unpack_bitplane_8_local(
                high_bits + group,
                high_indices);
            for (int lane = 0; lane < 8; ++lane)
            {
                expanded_indices[lane] =
                    low_indices[lane] | (high_indices[lane] << 3);
            }

            const __m256i indices = _mm256_load_si256(
                reinterpret_cast<const __m256i *>(expanded_indices));
            const __m256 centroids = _mm256_i32gather_ps(
                TQ4_CENTROIDS.data(),
                indices,
                sizeof(float));
            accumulator = _mm256_fmadd_ps(
                _mm256_loadu_ps(Q_rot + i),
                centroids,
                accumulator);
        }
        return avx2::hsum_ps(accumulator) * source_norm;
    }
#endif

#if defined(__AVX512F__)
    /** @brief Existing AVX-512 TQ4 score program with original-coordinate gathers.
     * @param Q_rot Checked original query; @param tq4_block Checked immutable physical head.
     * @param head_dim Complete validated ABI width.
     * @return AVX-512 ordered dot times the original source norm.
     */
    inline float tq4_dot_rotated_q_avx512(
        const float *__restrict__ Q_rot,
        const uint8_t *__restrict__ tq4_block,
        int head_dim)
    {
        float source_norm;
        std::memcpy(&source_norm, tq4_block, sizeof(float));
        if (source_norm < 1e-30f)
            return 0.0f;

        const uint8_t *mse_indices = tq4_block + 8;
        const uint8_t *high_bits = tq4_block + 8 + head_dim * 3 / 8;
        __m512 accumulator = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= head_dim; i += 16)
        {
            alignas(64) int32_t expanded_indices[16];
            for (int half = 0; half < 2; ++half)
            {
                const int base = i + half * 8;
                const int group = base / 8;
                uint8_t low_indices[8];
                uint8_t high_indices[8];
                tq3_unpack_8(mse_indices + group * 3, low_indices);
                tq_attn_detail::unpack_bitplane_8_local(
                    high_bits + group,
                    high_indices);
                for (int lane = 0; lane < 8; ++lane)
                {
                    expanded_indices[half * 8 + lane] =
                        low_indices[lane] | (high_indices[lane] << 3);
                }
            }

            const __m512i indices = _mm512_load_si512(expanded_indices);
            const __m512 centroids = _mm512_i32gather_ps(
                indices,
                TQ4_CENTROIDS.data(),
                sizeof(float));
            accumulator = _mm512_fmadd_ps(
                _mm512_loadu_ps(Q_rot + i),
                centroids,
                accumulator);
        }

        float dot = _mm512_reduce_add_ps(accumulator);
        for (; i < head_dim; i += 8)
        {
            const int group = i / 8;
            uint8_t low_indices[8];
            uint8_t high_indices[8];
            tq3_unpack_8(mse_indices + group * 3, low_indices);
            tq_attn_detail::unpack_bitplane_8_local(
                high_bits + group,
                high_indices);
            for (int lane = 0; lane < 8; ++lane)
            {
                const uint8_t centroid =
                    low_indices[lane] |
                    static_cast<uint8_t>(high_indices[lane] << 3);
                dot += Q_rot[i + lane] * TQ4_CENTROIDS[centroid];
            }
        }
        return dot * source_norm;
    }
#endif


    // ========================================================================
    // TQ8 V accumulation in rotated space
    // ========================================================================

    /**
     * @brief Accumulate a weighted TQ8 centroid vector in rotated space.
     *
     * Symmetric TQ8-K/TQ8-V caches use exactly the same block encoding for K
     * and V. The attention numerator needs the V centroid vector rather than a
     * dot product, so this primitive gathers each centroid and performs the
     * fixed-order update `accum[i] += weight * reconstruction_norm * centroid[index[i]]`.
     * The caller applies the common `1/sqrt(D)` scale and inverse rotation once
     * after the online-softmax reduction.
     *
     * @param accum Rotated-space attention numerator, updated in place.
     * @param tq8_block Raw `[norm, reconstruction_norm, indices...]` TQ8 block.
     * @param weight Unnormalized online-softmax probability for this K/V row.
     * @param head_dim Number of elements encoded by the block.
     */
    inline void tq8_accum_weighted_scalar(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq8_block,
        float weight,
        int head_dim)
    {
        float reconstruction_norm;
        std::memcpy(&reconstruction_norm, tq8_block + sizeof(float), sizeof(float));
        if (reconstruction_norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;

        const float combined_weight = weight * reconstruction_norm;
        const uint8_t *indices = tq8_block + 8;
        for (int i = 0; i < head_dim; ++i)
            accum[i] += combined_weight * TQ8_CENTROIDS[indices[i]];
    }

#if defined(__AVX2__)
    /** @brief Existing AVX2 TQ8 numerator FMA program.
     * @param accum Checked original numerator; @param tq8_block Immutable disjoint value head.
     * @param weight Original finite softmax weight; @param head_dim Complete validated ABI width.
     */
    inline void tq8_accum_weighted_avx2(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq8_block,
        float weight,
        int head_dim)
    {
        float reconstruction_norm;
        std::memcpy(&reconstruction_norm, tq8_block + sizeof(float), sizeof(float));
        if (reconstruction_norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;

        const __m256 weighted_norm = _mm256_set1_ps(weight * reconstruction_norm);
        const uint8_t *indices = tq8_block + 8;
        int i = 0;
        for (; i + 8 <= head_dim; i += 8)
        {
            const __m128i packed_indices = _mm_loadl_epi64(
                reinterpret_cast<const __m128i *>(indices + i));
            const __m256i expanded_indices =
                _mm256_cvtepu8_epi32(packed_indices);
            const __m256 centroids = _mm256_i32gather_ps(
                TQ8_CENTROIDS.data(), expanded_indices, sizeof(float));
            const __m256 previous = _mm256_loadu_ps(accum + i);
            _mm256_storeu_ps(
                accum + i,
                _mm256_fmadd_ps(centroids, weighted_norm, previous));
        }
        for (; i < head_dim; ++i)
            accum[i] += weight * reconstruction_norm * TQ8_CENTROIDS[indices[i]];
    }
#endif

#if defined(__AVX512F__)
    /** @brief Existing AVX-512 TQ8 numerator FMA program.
     * @param accum Checked original numerator; @param tq8_block Immutable disjoint value head.
     * @param weight Original finite weight; @param head_dim Complete validated ABI width.
     */
    inline void tq8_accum_weighted_avx512(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq8_block,
        float weight,
        int head_dim)
    {
        float reconstruction_norm;
        std::memcpy(&reconstruction_norm, tq8_block + sizeof(float), sizeof(float));
        if (reconstruction_norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;

        const __m512 weighted_norm = _mm512_set1_ps(weight * reconstruction_norm);
        const uint8_t *indices = tq8_block + 8;
        int i = 0;
        for (; i + 16 <= head_dim; i += 16)
        {
            const __m128i packed_indices = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(indices + i));
            const __m512i expanded_indices =
                _mm512_cvtepu8_epi32(packed_indices);
            const __m512 centroids = _mm512_i32gather_ps(
                expanded_indices, TQ8_CENTROIDS.data(), sizeof(float));
            const __m512 previous = _mm512_loadu_ps(accum + i);
            _mm512_storeu_ps(
                accum + i,
                _mm512_fmadd_ps(centroids, weighted_norm, previous));
        }
        for (; i < head_dim; ++i)
            accum[i] += weight * reconstruction_norm * TQ8_CENTROIDS[indices[i]];
    }
#endif


    // ========================================================================
    // TQ4 V accumulation in rotated space
    // ========================================================================

    /**
     * @brief Accumulate weighted TQ4 centroid vector into rotated-space accumulator.
     *
     * Computes: accum[i] += weight × reconstruction_norm ×
     * TQ4_CENTROIDS[index(block, i)]
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
     * @param tq4_block  Raw TQ4 block bytes: [norm:f32, reconstruction_norm:f32,
     *                   mse_indices:D*3/8 bytes, high_bits:D/8 bytes]
     * @param weight     Softmax attention weight for this position
     * @param head_dim   Head dimension (64 or 128)
     */
    // ========================================================================
    // Named ISA implementations: tq4_accum_weighted
    // ========================================================================

    /** @brief Explicit scalar diagnostic TQ4 numerator update using the fitted radius.
     * @param accum Original diagnostic numerator; @param tq4_block Valid disjoint physical head.
     * @param weight Original finite weight; @param head_dim Complete valid diagnostic width.
     */
    inline void tq4_accum_weighted_scalar(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        float reconstruction_norm;
        std::memcpy(
            &reconstruction_norm,
            tq4_block + sizeof(float),
            sizeof(float));
        if (reconstruction_norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;
        const float combined_weight = weight * reconstruction_norm;
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
    /** @brief Existing AVX2 TQ4 numerator update without changing the centroid/FMA program.
     * @param accum Checked original numerator; @param tq4_block Immutable disjoint value head.
     * @param weight Original finite weight; @param head_dim Complete validated ABI width.
     */
    inline void tq4_accum_weighted_avx2(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        float reconstruction_norm;
        std::memcpy(
            &reconstruction_norm,
            tq4_block + sizeof(float),
            sizeof(float));
        if (reconstruction_norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;
        const float combined_weight = weight * reconstruction_norm;
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
    /** @brief Existing AVX-512 TQ4 numerator update with unchanged fitted-radius arithmetic.
     * @param accum Checked original numerator; @param tq4_block Immutable disjoint value head.
     * @param weight Original finite weight; @param head_dim Complete validated ABI width.
     */
    inline void tq4_accum_weighted_avx512(
        float *__restrict__ accum,
        const uint8_t *__restrict__ tq4_block,
        float weight,
        int head_dim)
    {
        float reconstruction_norm;
        std::memcpy(
            &reconstruction_norm,
            tq4_block + sizeof(float),
            sizeof(float));
        if (reconstruction_norm < 1e-30f || std::abs(weight) < 1e-30f)
            return;
        const float combined_weight = weight * reconstruction_norm;
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

    namespace tq_attn_detail
    {
        using cpu::tq_attention::Format;

        /** @brief Execute explicitly selected scalar diagnostic scoring on already checked operands.
         * @tparam Encoding Original physical layout; @param input Original checked query/head borrows.
         * @return Scalar diagnostic result, or positive zero for the original inactive program.
         */
        template <Format Encoding> inline float dot_scalar(const cpu::tq_attention::DotOperand<Encoding> &input)
        {
            if (!input.active()) return 0;
            if constexpr (Encoding == Format::TQ4)
                return tq4_dot_rotated_q_scalar(input.query(), input.head().data(), input.head().dimensions());
            else
                return tq8_dot_rotated_q_scalar(input.query(), input.head().data(), input.head().dimensions());
        }
        /** @brief Execute explicitly selected scalar diagnostic accumulation without a fallback route.
         * @tparam Encoding Original layout; @param input Checked original numerator/head/weight.
         */
        template <Format Encoding> inline void accumulate_scalar(const cpu::tq_attention::AccumOperand<Encoding> &input)
        {
            if (!input.active()) return;
            if constexpr (Encoding == Format::TQ4)
                tq4_accum_weighted_scalar(input.output(), input.head().data(), input.weight(), input.head().dimensions());
            else
                tq8_accum_weighted_scalar(input.output(), input.head().data(), input.weight(), input.head().dimensions());
        }
#if defined(__AVX2__)
        /** @brief Adapt checked original operands to the existing AVX2 score program.
         * @tparam Encoding Original layout; @param input Checked query/head borrows.
         * @return Original AVX2 result, with unchanged arithmetic.
         */
        template <Format Encoding> inline float dot_avx2(const cpu::tq_attention::DotOperand<Encoding> &input)
        {
            if (!input.active()) return 0;
            if constexpr (Encoding == Format::TQ4)
                return tq4_dot_rotated_q_avx2(input.query(), input.head().data(), input.head().dimensions());
            else
                return tq8_dot_rotated_q_avx2(input.query(), input.head().data(), input.head().dimensions());
        }
        /** @brief Adapt checked original storage to the unchanged AVX2 numerator program.
         * @tparam Encoding Original layout; @param input Checked numerator update.
         */
        template <Format Encoding> inline void accumulate_avx2(const cpu::tq_attention::AccumOperand<Encoding> &input)
        {
            if (!input.active()) return;
            if constexpr (Encoding == Format::TQ4)
                tq4_accum_weighted_avx2(input.output(), input.head().data(), input.weight(), input.head().dimensions());
            else
                tq8_accum_weighted_avx2(input.output(), input.head().data(), input.weight(), input.head().dimensions());
        }
#endif
#if defined(__AVX512F__)
        /** @brief Adapt checked original operands to the existing AVX-512 score program.
         * @tparam Encoding Original layout; @param input Checked query/head borrows.
         * @return Original AVX-512 result, with unchanged arithmetic.
         */
        template <Format Encoding> inline float dot_avx512(const cpu::tq_attention::DotOperand<Encoding> &input)
        {
            if (!input.active()) return 0;
            if constexpr (Encoding == Format::TQ4)
                return tq4_dot_rotated_q_avx512(input.query(), input.head().data(), input.head().dimensions());
            else
                return tq8_dot_rotated_q_avx512(input.query(), input.head().data(), input.head().dimensions());
        }
        /** @brief Adapt checked original storage to the unchanged AVX-512 numerator program.
         * @tparam Encoding Original layout; @param input Checked numerator update.
         */
        template <Format Encoding> inline void accumulate_avx512(const cpu::tq_attention::AccumOperand<Encoding> &input)
        {
            if (!input.active()) return;
            if constexpr (Encoding == Format::TQ4)
                tq4_accum_weighted_avx512(input.output(), input.head().data(), input.weight(), input.head().dimensions());
            else
                tq8_accum_weighted_avx512(input.output(), input.head().data(), input.weight(), input.head().dimensions());
        }
#endif
#if LLAMINAR_COMPILED_WITH_NEON
        /** @brief Execute native ASIMD scoring; no raw unchecked view or x86 body can enter this leaf.
         * @tparam Encoding Original layout; @param input Checked immutable original borrows.
         * @return Ordered native score, with one retained source-norm multiply.
         */
        template <Format Encoding> inline float dot_neon(const cpu::tq_attention::DotOperand<Encoding> &input)
        { return cpu::neon_tq_attention::dot(input); }
        /** @brief Execute the native weighted-centroid FMA program on checked disjoint storage.
         * @tparam Encoding Original layout; @param input Checked original numerator update.
         */
        template <Format Encoding> inline void accumulate_neon(const cpu::tq_attention::AccumOperand<Encoding> &input)
        { cpu::neon_tq_attention::accumulate(input); }
#endif
    } // namespace tq_attn_detail

    /** @brief Score one TQ8 head using its original source norm and canonical native ISA.
     * @param Q_rot Original rotated query; @param tq8_block Original complete immutable physical head.
     * @param head_dim Existing TQ tensor ABI width (64/128/256), or explicit zero work.
     * @return Native ordered dot times the source norm; zero for original inactive work.
     * @throws std::invalid_argument For malformed, missing or overlapping original operands.
     * @throws std::domain_error For invalid selected source metadata.
     */
    inline float tq8_dot_rotated_q(const float *Q_rot, const std::uint8_t *tq8_block, int head_dim)
    {
        const auto input = cpu::tq_attention::DotOperand<cpu::tq_attention::Format::TQ8>::bind(Q_rot, tq8_block, head_dim);
        return ISA_DISPATCH_RETVAL(tq_attn_detail::dot, input);
    }
    /** @brief Score one TQ4 head without a decoded key/head mirror.
     * @param Q_rot Original rotated query; @param tq4_block Original complete physical head.
     * @param head_dim Existing ABI width (64/128/256), or explicit zero work.
     * @return Native ordered dot times the original source norm.
     * @throws std::invalid_argument For malformed, missing or overlapping operands.
     * @throws std::domain_error For invalid selected source metadata.
     */
    inline float tq4_dot_rotated_q(const float *Q_rot, const std::uint8_t *tq4_block, int head_dim)
    {
        const auto input = cpu::tq_attention::DotOperand<cpu::tq_attention::Format::TQ4>::bind(Q_rot, tq4_block, head_dim);
        return ISA_DISPATCH_RETVAL(tq_attn_detail::dot, input);
    }
    /** @brief Update a numerator from original compressed TQ8 values and their fitted radius.
     * @param accum Original writable numerator; @param tq8_block Immutable physical value head.
     * @param weight Original finite softmax weight; @param head_dim Existing ABI width or zero work.
     * @throws std::invalid_argument For unsupported, missing or aliased storage.
     * @throws std::domain_error For invalid selected metadata/weight or a nonfinite live product.
     */
    inline void tq8_accum_weighted(float *accum, const std::uint8_t *tq8_block, float weight, int head_dim)
    {
        const auto input = cpu::tq_attention::AccumOperand<cpu::tq_attention::Format::TQ8>::bind(accum, tq8_block, weight, head_dim);
        ISA_DISPATCH_VOID(tq_attn_detail::accumulate, input);
    }
    /** @brief Update a numerator directly from original TQ4 bitplanes and exact centroid words.
     * @param accum Original writable numerator; @param tq4_block Immutable physical value head.
     * @param weight Original finite weight; @param head_dim Existing ABI width or zero work.
     * @throws std::invalid_argument For unsupported, missing or aliased storage.
     * @throws std::domain_error For invalid selected metadata/weight or a nonfinite live product.
     */
    inline void tq4_accum_weighted(float *accum, const std::uint8_t *tq4_block, float weight, int head_dim)
    {
        const auto input = cpu::tq_attention::AccumOperand<cpu::tq_attention::Format::TQ4>::bind(accum, tq4_block, weight, head_dim);
        ISA_DISPATCH_VOID(tq_attn_detail::accumulate, input);
    }
} // namespace llaminar2
