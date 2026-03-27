/**
 * @file TurboQuantDequantizeTQ4.h
 * @brief TQ4 → FP32 dequantization for TurboQuant KV cache
 * @author David Sanftenberg
 *
 * Dequantizes TQ4 scalar-full blocks back to FP32 vectors.
 *
 * Provides both single-vector and batch entry points.
 */

#pragma once

#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantRotation.h"
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "kernels/cpu/primitives/RoPEInline.h"
#include "utils/OpenMPUtils.h"

#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    /// Unpack one byte into 8 individual bits (LSB first).
    inline void unpack_bitplane_8(const uint8_t *packed, uint8_t *out)
    {
        const uint8_t bits = *packed;
        for (int i = 0; i < 8; ++i)
            out[i] = static_cast<uint8_t>((bits >> i) & 0x1u);
    }

    // ========================================================================
    // Single-vector dequantize: TQ4 → FP32
    // ========================================================================

    /**
     * @brief Dequantize one TQ4 block to FP32 vector.
     *
     * Unpacks 4-bit centroid indices (3 low bits from mse_indices +
     * 1 high bit from high_bits), looks up TQ4_CENTROIDS, applies
     * inverse rotation, and scales by the stored norm.
     *
     * @tparam D Dimension of the vector (head_dim)
     * @param block Input TQ4 block
     * @param ctx Pre-generated TurboQuant context
     * @param output FP32 output vector of length D
     * @param scratch Scratch buffer of at least D floats (for centroid vector)
     */
    template <int D>
    inline void turboquant_dequantize_tq4(
        const TQ4Block<D> &block,
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

        // Unpack 3+1 bit indices → centroid gather (16 elements per iteration)
        const __m512 vinv_scale = _mm512_set1_ps(inv_scale);
        for (int i = 0; i < D; i += 16)
        {
            // Unpack 2 groups of 8 indices from packed 3-bit + 1 high-bit format
            alignas(64) int32_t idx32[16];
            for (int g = 0; g < 2; ++g)
            {
                const int base = i + g * 8;
                const int group = base / 8;
                uint8_t idx8[8];
                uint8_t high_bits[8];
                tq3_unpack_8(block.mse_indices + group * 3, idx8);
                unpack_bitplane_8(block.high_bits + group, high_bits);
                for (int j = 0; j < 8; ++j)
                    idx32[g * 8 + j] = idx8[j] | (high_bits[j] << 3);
            }

            // Vectorized centroid gather + descale
            __m512i vidx = _mm512_load_si512(idx32);
            __m512 vcentroids = _mm512_i32gather_ps(vidx, TQ4_CENTROIDS.data(), sizeof(float));
            _mm512_storeu_ps(scratch + i, _mm512_mul_ps(vcentroids, vinv_scale));
        }
#else
        for (int i = 0; i < D; i += 8)
        {
            uint8_t idx8[8];
            uint8_t high_bits[8];
            tq3_unpack_8(block.mse_indices + (i / 8) * 3, idx8);
            unpack_bitplane_8(block.high_bits + (i / 8), high_bits);
            for (int j = 0; j < 8; ++j)
                scratch[i + j] = TQ4_CENTROIDS[idx8[j] | static_cast<uint8_t>(high_bits[j] << 3)] * inv_scale;
        }
#endif

        apply_rotation_transpose(ctx.rotation(), scratch, output);

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

    // ========================================================================
    // Row-range dequantization helpers for AttentionComputeStage
    //
    // These wrap the per-block dequantize in a row/head loop so that
    // the stage class only needs a single call per path.
    // ========================================================================

    /**
     * @brief Dequantize V rows from TQ4 to FP32.
     *
     * @param v_raw      Raw TQ4 block bytes (TQ4Tensor::typed_data())
     * @param ctx        Layer-level TurboQuant context (rotation matrices)
     * @param v_fp32     Output FP32 buffer, layout [kv_len × kv_dim]
     * @param from_row   First row to dequantize (inclusive)
     * @param to_row     Last row to dequantize (exclusive)
     * @param head_dim   Head dimension (must be 128 on this path)
     * @param n_kv_heads Number of KV heads
     * @param row_bytes  Bytes per row (blocks_per_row * block_bytes)
     * @param block_bytes Bytes per single TQ4 block
     */
    inline void turboquant_dequantize_v_rows(
        const uint8_t *v_raw,
        const TurboQuantContext &ctx,
        float *v_fp32,
        int from_row, int to_row,
        int head_dim, int n_kv_heads,
        size_t row_bytes, size_t block_bytes)
    {
        const int kv_dim = n_kv_heads * head_dim;
        const int num_rows = to_row - from_row;
        auto work = [&]()
        {
#pragma omp for schedule(static)
            for (int r = from_row; r < to_row; ++r)
            {
                float *v_dst = v_fp32 + r * kv_dim;
                const uint8_t *v_row = v_raw + static_cast<size_t>(r) * row_bytes;
                alignas(64) float scratch[128];

                for (int h = 0; h < n_kv_heads; ++h)
                {
                    const auto &head_ctx = ctx.for_layer(h);
                    const auto *vb = reinterpret_cast<const TQ4Block_128 *>(
                        v_row + static_cast<size_t>(h) * block_bytes);
                    turboquant_dequantize_tq4(*vb, head_ctx,
                                              v_dst + h * head_dim, scratch);
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work, num_rows >= 4);
    }

    /**
     * @brief Dequantize both K and V rows from TQ4 to FP32.
     *
     * @param k_raw      Raw TQ4 block bytes for K (TQ4Tensor::typed_data())
     * @param v_raw      Raw TQ4 block bytes for V (TQ4Tensor::typed_data())
     * @param ctx        Layer-level TurboQuant context (rotation matrices)
     * @param k_fp32     Output FP32 buffer for K, layout [kv_len × kv_dim]
     * @param v_fp32     Output FP32 buffer for V, layout [kv_len × kv_dim]
     * @param from_row   First row to dequantize (inclusive)
     * @param to_row     Last row to dequantize (exclusive)
     * @param head_dim   Head dimension (64 or 128)
     * @param n_kv_heads Number of KV heads
     * @param k_row_bytes Bytes per K row
     * @param v_row_bytes Bytes per V row
     * @param k_block_bytes Bytes per single K block
     * @param v_block_bytes Bytes per single V block
     */
    inline void turboquant_dequantize_kv_rows(
        const uint8_t *k_raw,
        const uint8_t *v_raw,
        const TurboQuantContext &ctx,
        float *k_fp32, float *v_fp32,
        int from_row, int to_row,
        int head_dim, int n_kv_heads,
        size_t k_row_bytes, size_t v_row_bytes,
        size_t k_block_bytes, size_t v_block_bytes)
    {
        const int kv_dim = n_kv_heads * head_dim;
        const int num_rows = to_row - from_row;
        auto work = [&]()
        {
#pragma omp for schedule(static)
            for (int r = from_row; r < to_row; ++r)
            {
                const uint8_t *k_row = k_raw + static_cast<size_t>(r) * k_row_bytes;
                const uint8_t *v_row = v_raw + static_cast<size_t>(r) * v_row_bytes;
                float *k_dst = k_fp32 + r * kv_dim;
                float *v_dst = v_fp32 + r * kv_dim;
                alignas(64) float scratch[128];

                for (int h = 0; h < n_kv_heads; ++h)
                {
                    const auto &head_ctx = ctx.for_layer(h);
                    if (head_dim == 128)
                    {
                        const auto *kb = reinterpret_cast<const TQ4Block_128 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq4(*kb, head_ctx,
                                                  k_dst + h * head_dim, scratch);
                        const auto *vb = reinterpret_cast<const TQ4Block_128 *>(
                            v_row + static_cast<size_t>(h) * v_block_bytes);
                        turboquant_dequantize_tq4(*vb, head_ctx,
                                                  v_dst + h * head_dim, scratch);
                    }
                    else
                    {
                        const auto *kb = reinterpret_cast<const TQ4Block_64 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq4(*kb, head_ctx,
                                                  k_dst + h * head_dim, scratch);
                        const auto *vb = reinterpret_cast<const TQ4Block_64 *>(
                            v_row + static_cast<size_t>(h) * v_block_bytes);
                        turboquant_dequantize_tq4(*vb, head_ctx,
                                                  v_dst + h * head_dim, scratch);
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work, num_rows >= 4);
    }

    // ========================================================================
    // Fused K dequantize + RoPE: TQ4 → FP32 with position embeddings
    //
    // Applies RoPE in-place immediately after dequantization, fusing the
    // two operations into a single pass over the output buffer. This enables
    // "RoPE-on-read" mode where K is stored pre-RoPE in the KV cache and
    // position embeddings are applied lazily during attention.
    //
    // Benefits:
    //   - RoPE is O(D) per head, dequant rotation is O(D²) → essentially free
    //   - Position-free KV cache enables speculative decoding
    //   - Eliminates separate RoPE stage for K
    // ========================================================================

    // apply_rope_to_head_inline() and apply_rope_to_k_fp32() have been moved
    // to kernels/cpu/primitives/RoPEInline.h (included above).

    /**
     * @brief Dequantize K rows from TQ4 to FP32 with fused RoPE application.
     *
     * For each row r in [from_row, to_row):
     *   1. Dequantize each KV head's TQ4 block → FP32
     *   2. Apply RoPE at position (position_start + r) to each head
     *
     * The position mapping assumes contiguous positions starting from
     * position_start. For ring-buffer caches with wrapping, the caller
     * must handle position mapping externally.
     *
     * @param k_raw          Raw TQ4 block bytes (TQ4Tensor::typed_data())
     * @param ctx            Layer-level TurboQuant context (rotation matrices)
     * @param k_fp32         Output FP32 buffer, layout [kv_len × kv_dim]
     * @param from_row       First row to dequantize (inclusive)
     * @param to_row         Last row to dequantize (exclusive)
     * @param head_dim       Head dimension (must be 128 on this path)
     * @param n_kv_heads     Number of KV heads
     * @param row_bytes      Bytes per row (blocks_per_row * block_bytes)
     * @param block_bytes    Bytes per single TQ4 block
     * @param rope_theta     RoPE frequency base (e.g. 10000.0 or 1000000.0)
     * @param position_start RoPE position of from_row (typically 0 for full cache)
     */
    inline void turboquant_dequantize_k_rows_with_rope(
        const uint8_t *k_raw,
        const TurboQuantContext &ctx,
        float *k_fp32,
        int from_row, int to_row,
        int head_dim, int n_kv_heads,
        size_t row_bytes, size_t block_bytes,
        float rope_theta, int position_start)
    {
        const int kv_dim = n_kv_heads * head_dim;
        const int num_rows = to_row - from_row;
        const int half_dim = head_dim / 2;

        // Get cached inverse frequencies for RoPE computation
        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, rope_theta);

        auto work = [&]()
        {
            // Thread-local cos/sin buffers for RoPE
            alignas(64) float cos_buf[128]; // head_dim/2 max = 64 for 128, fits
            alignas(64) float sin_buf[128];

#pragma omp for schedule(static)
            for (int r = from_row; r < to_row; ++r)
            {
                const uint8_t *k_row = k_raw + static_cast<size_t>(r) * row_bytes;
                float *k_dst = k_fp32 + r * kv_dim;
                alignas(64) float scratch[128];

                // Compute cos/sin for this position (shared across all KV heads)
                const int position = position_start + r;
                for (int i = 0; i < half_dim; ++i)
                {
                    const float angle = static_cast<float>(position) * inv_freq[static_cast<size_t>(i)];
                    cos_buf[i] = std::cos(angle);
                    sin_buf[i] = std::sin(angle);
                }

                // Dequant + RoPE for each KV head
                for (int h = 0; h < n_kv_heads; ++h)
                {
                    const auto &head_ctx = ctx.for_layer(h);
                    const auto *kb = reinterpret_cast<const TQ4Block_128 *>(
                        k_row + static_cast<size_t>(h) * block_bytes);
                    turboquant_dequantize_tq4(*kb, head_ctx,
                                              k_dst + h * head_dim, scratch);

                    // Fuse RoPE: apply position embedding in-place
                    apply_rope_to_head_inline(k_dst + h * head_dim,
                                              cos_buf, sin_buf, head_dim);
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work, num_rows >= 4);
    }

    /**
     * @brief Dequantize K and V rows from TQ4 to FP32, with fused RoPE on K only.
     *
     * V is dequantized without RoPE (RoPE is only applied to K).
     * When rope_theta <= 0, falls back to plain dequant (no RoPE).
     *
     * @param k_raw          Raw TQ4 block bytes for K
     * @param v_raw          Raw TQ4 block bytes for V
     * @param ctx            Layer-level TurboQuant context
     * @param k_fp32         Output FP32 buffer for K [kv_len × kv_dim]
     * @param v_fp32         Output FP32 buffer for V [kv_len × kv_dim]
     * @param from_row       First row (inclusive)
     * @param to_row         Last row (exclusive)
     * @param head_dim       Head dimension
     * @param n_kv_heads     Number of KV heads
     * @param k_row_bytes    Bytes per K row
     * @param v_row_bytes    Bytes per V row
     * @param k_block_bytes  Bytes per K block
     * @param v_block_bytes  Bytes per V block
     * @param rope_theta     RoPE frequency base (<=0 to disable RoPE)
     * @param position_start RoPE position of from_row
     */
    inline void turboquant_dequantize_kv_rows_with_rope(
        const uint8_t *k_raw,
        const uint8_t *v_raw,
        const TurboQuantContext &ctx,
        float *k_fp32, float *v_fp32,
        int from_row, int to_row,
        int head_dim, int n_kv_heads,
        size_t k_row_bytes, size_t v_row_bytes,
        size_t k_block_bytes, size_t v_block_bytes,
        float rope_theta, int position_start)
    {
        const int kv_dim = n_kv_heads * head_dim;
        const int num_rows = to_row - from_row;
        const int half_dim = head_dim / 2;
        const bool apply_rope = (rope_theta > 0.0f);

        // Pre-fetch inverse frequencies if RoPE is active
        const std::vector<float> *inv_freq_ptr = nullptr;
        if (apply_rope)
            inv_freq_ptr = &primitives::get_inv_freq_cached(head_dim, rope_theta);

        auto work = [&]()
        {
            alignas(64) float cos_buf[128];
            alignas(64) float sin_buf[128];

#pragma omp for schedule(static)
            for (int r = from_row; r < to_row; ++r)
            {
                const uint8_t *k_row = k_raw + static_cast<size_t>(r) * k_row_bytes;
                const uint8_t *v_row = v_raw + static_cast<size_t>(r) * v_row_bytes;
                float *k_dst = k_fp32 + r * kv_dim;
                float *v_dst = v_fp32 + r * kv_dim;
                alignas(64) float scratch[128];

                // Compute cos/sin once per position (shared across KV heads)
                if (apply_rope)
                {
                    const int position = position_start + r;
                    for (int i = 0; i < half_dim; ++i)
                    {
                        const float angle = static_cast<float>(position) * (*inv_freq_ptr)[static_cast<size_t>(i)];
                        cos_buf[i] = std::cos(angle);
                        sin_buf[i] = std::sin(angle);
                    }
                }

                for (int h = 0; h < n_kv_heads; ++h)
                {
                    const auto &head_ctx = ctx.for_layer(h);

                    // K: dequant + optional RoPE
                    if (head_dim == 128)
                    {
                        const auto *kb = reinterpret_cast<const TQ4Block_128 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq4(*kb, head_ctx,
                                                  k_dst + h * head_dim, scratch);
                    }
                    else
                    {
                        const auto *kb = reinterpret_cast<const TQ4Block_64 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq4(*kb, head_ctx,
                                                  k_dst + h * head_dim, scratch);
                    }
                    if (apply_rope)
                    {
                        apply_rope_to_head_inline(k_dst + h * head_dim,
                                                  cos_buf, sin_buf, head_dim);
                    }

                    // V: dequant only (no RoPE)
                    if (head_dim == 128)
                    {
                        const auto *vb = reinterpret_cast<const TQ4Block_128 *>(
                            v_row + static_cast<size_t>(h) * v_block_bytes);
                        turboquant_dequantize_tq4(*vb, head_ctx,
                                                  v_dst + h * head_dim, scratch);
                    }
                    else
                    {
                        const auto *vb = reinterpret_cast<const TQ4Block_64 *>(
                            v_row + static_cast<size_t>(h) * v_block_bytes);
                        turboquant_dequantize_tq4(*vb, head_ctx,
                                                  v_dst + h * head_dim, scratch);
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work, num_rows >= 4);
    }

} // namespace llaminar2
