/**
 * @file TurboQuantDequantizeSplitTQ.h
 * @brief Split TQ dequantization: K from TQ8 + V from TQ4 → FP32
 *
 * Used by AttentionComputeStage when the KV cache uses split precision
 * (TQ8 for K, TQ4 for V). Provides row-range helpers that dequantize
 * K rows via TQ8 and V rows via TQ4, with optional fused RoPE on K.
 */

#pragma once

#include "TurboQuantDequantizeTQ4.h"
#include "TurboQuantDequantizeTQ8.h"
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "utils/OpenMPUtils.h"

namespace llaminar2
{

    /**
     * @brief Dequantize split TQ rows: K from TQ8, V from TQ4.
     *
     * @param k_raw         Raw TQ8 bytes (TQ8Tensor::typed_data())
     * @param v_raw         Raw TQ4 bytes (TQ4Tensor::typed_data())
     * @param ctx           Layer-level TurboQuant context
     * @param k_fp32        Output K buffer [kv_len × kv_dim]
     * @param v_fp32        Output V buffer [kv_len × kv_dim]
     * @param from_row      First row (inclusive)
     * @param to_row        Last row (exclusive)
     * @param head_dim      Head dimension (64 or 128)
     * @param n_kv_heads    Number of KV heads
     * @param k_row_bytes   Bytes per K row in TQ8 storage
     * @param v_row_bytes   Bytes per V row in TQ4 storage
     * @param k_block_bytes Bytes per single TQ8 block
     * @param v_block_bytes Bytes per single TQ4 block
     */
    inline void turboquant_dequantize_split_kv_rows(
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

                    // K: TQ8 dequant
                    if (head_dim == 128)
                    {
                        const auto *kb = reinterpret_cast<const TQ8Block_128 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq8<128>(*kb, head_ctx,
                                                       k_dst + h * head_dim, scratch);
                    }
                    else
                    {
                        const auto *kb = reinterpret_cast<const TQ8Block_64 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq8<64>(*kb, head_ctx,
                                                      k_dst + h * head_dim, scratch);
                    }

                    // V: TQ4 dequant
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

    /**
     * @brief Dequantize split TQ rows with fused RoPE on K: K from TQ8, V from TQ4.
     *
     * @param k_raw          Raw TQ8 bytes (TQ8Tensor::typed_data())
     * @param v_raw          Raw TQ4 bytes (TQ4Tensor::typed_data())
     * @param ctx            Layer-level TurboQuant context
     * @param k_fp32         Output K buffer [kv_len × kv_dim]
     * @param v_fp32         Output V buffer [kv_len × kv_dim]
     * @param from_row       First row (inclusive)
     * @param to_row         Last row (exclusive)
     * @param head_dim       Head dimension (64 or 128)
     * @param n_kv_heads     Number of KV heads
     * @param k_row_bytes    Bytes per K row in TQ8 storage
     * @param v_row_bytes    Bytes per V row in TQ4 storage
     * @param k_block_bytes  Bytes per single TQ8 block
     * @param v_block_bytes  Bytes per single TQ4 block
     * @param rope_theta     RoPE frequency base (<=0 to disable RoPE)
     * @param position_start RoPE position of from_row
     */
    inline void turboquant_dequantize_split_kv_rows_with_rope(
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

                    // K: TQ8 dequant + optional RoPE
                    if (head_dim == 128)
                    {
                        const auto *kb = reinterpret_cast<const TQ8Block_128 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq8<128>(*kb, head_ctx,
                                                       k_dst + h * head_dim, scratch);
                    }
                    else
                    {
                        const auto *kb = reinterpret_cast<const TQ8Block_64 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq8<64>(*kb, head_ctx,
                                                      k_dst + h * head_dim, scratch);
                    }
                    if (apply_rope)
                    {
                        apply_rope_to_head_inline(k_dst + h * head_dim,
                                                  cos_buf, sin_buf, head_dim);
                    }

                    // V: TQ4 dequant (no RoPE)
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
