/**
 * @file ActivationPackingAdapters.h
 * @brief Adapters to convert IActivationTensor to packed 4x4-grouped int8 format
 * @author David Sanftenberg
 *
 * Converts various activation tensor formats to the packed A format expected by VNNI GEMM:
 * - 4x4-grouped layout (M_R/4 groups, K_BLK/4 chunks per group)
 * - 16-byte aligned chunks
 * - Zero-padded boundaries
 *
 * Layout per group (4 rows):
 *   For each K chunk (4 elements):
 *     row 0: k+0, k+1, k+2, k+3
 *     row 1: k+0, k+1, k+2, k+3
 *     row 2: k+0, k+1, k+2, k+3
 *     row 3: k+0, k+1, k+2, k+3
 */

#pragma once

#include "tensors/Tensors.h"
#include "tensors/Q8_0Helpers.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace llaminar2
{

    namespace detail
    {
        inline int8_t apply_sign_bias(int8_t value)
        {
            const uint8_t biased = static_cast<uint8_t>(static_cast<uint8_t>(value) ^ 0x80u);
            return static_cast<int8_t>(biased);
        }
    }

    /**
     * @brief Pack FP32 activations to int8 4x4-grouped format with per-row quantization
     *
     * Quantizes FP32 activations to int8 using symmetric per-row quantization,
     * then packs to 4x4-grouped layout for VNNI GEMM.
     *
     * @tparam M_R Micro-kernel M dimension (must be multiple of 4)
     * @tparam K_BLK K block size (must be multiple of 4)
     * @param A_fp32 FP32 activation tensor [M x K], row-major
     * @param M Number of rows
     * @param K Number of columns
     * @param M0 Starting row index in A_fp32
     * @param k0 Starting column index in A_fp32
     * @param mr Actual number of rows to pack (<= M_R)
     * @param kblk Actual K block size to pack (<= K_BLK)
     * @param A_tile_packed Output buffer for packed int8 data
     * @param act_scales Output per-row activation scales [M_R] (dequant = int8 * scale)
     */
    template <int M_R, int K_BLK>
    void pack_fp32_activations_to_4x4_grouped(
        const float *__restrict A_fp32,
        int M, int K,
        int M0, int k0,
        int mr, int kblk,
        int8_t *__restrict A_tile_packed,
        float *__restrict act_scales)
    {
        static_assert(M_R % 4 == 0, "M_R must be multiple of 4");
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        const int K_chunks = kblk / 4;
        const int group_stride = K_chunks * 16;

        // First pass: Quantize FP32 -> INT8 with per-row scaling
        std::vector<int8_t> A_int8(M_R * kblk, 0);

#pragma omp parallel for schedule(dynamic) if (mr * kblk >= 1024)
        for (int m = 0; m < mr; ++m)
        {
            const int global_row = M0 + m;
            if (global_row >= M)
            {
                act_scales[m] = 1.0f;
                continue;
            }

            const float *src_row = A_fp32 + global_row * K + k0;
            float max_abs = simd::find_max_abs_fp32(src_row, static_cast<size_t>(kblk));
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            act_scales[m] = scale;
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 1.0f;
            int8_t *dst_row = A_int8.data() + m * kblk;
            simd::quantize_fp32_to_int8(src_row, dst_row, inv_scale, static_cast<size_t>(kblk));

            const float *next_row = src_row + K;
            _mm_prefetch(reinterpret_cast<const char *>(next_row), _MM_HINT_T0);
        }

        // Zero remaining scales
        for (int m = mr; m < M_R; ++m)
        {
            act_scales[m] = 1.0f;
        }

        // Second pass: Pack INT8 data to 4x4-grouped layout
#pragma omp parallel for schedule(dynamic)
        for (int m_base = 0; m_base < mr; m_base += 4)
        {
            const int group_idx = m_base / 4;
            int8_t *group_ptr = A_tile_packed + group_idx * group_stride;

            for (int kk = 0; kk < K_chunks; ++kk)
            {
                int8_t *dst = group_ptr + kk * 16;

                for (int lane = 0; lane < 4; ++lane)
                {
                    const int row_in_tile = m_base + lane;

                    if (row_in_tile < mr)
                    {
                        const int8_t *src_row = A_int8.data() + row_in_tile * kblk + kk * 4;
                        dst[lane * 4 + 0] = detail::apply_sign_bias(src_row[0]);
                        dst[lane * 4 + 1] = detail::apply_sign_bias(src_row[1]);
                        dst[lane * 4 + 2] = detail::apply_sign_bias(src_row[2]);
                        dst[lane * 4 + 3] = detail::apply_sign_bias(src_row[3]);
                    }
                    else
                    {
                        // Zero-pad partial groups (biased zero = 0x80)
                        const int8_t biased_zero = detail::apply_sign_bias(0);
                        dst[lane * 4 + 0] = biased_zero;
                        dst[lane * 4 + 1] = biased_zero;
                        dst[lane * 4 + 2] = biased_zero;
                        dst[lane * 4 + 3] = biased_zero;
                    }
                }
            }
        }

        // Zero-pad remaining groups if mr < M_R
#pragma omp parallel for schedule(dynamic)
        for (int m_base = mr; m_base < M_R; m_base += 4)
        {
            const int group_idx = m_base / 4;
            int8_t *group_ptr = A_tile_packed + group_idx * group_stride;
            std::memset(group_ptr, 0x80, group_stride);
        }
    }

    /**
     * @brief Pack pre-quantized int8 activations to 4x4-grouped format
     *
     * For activations already quantized to int8 (e.g., INT8Tensor),
     * just repack to 4x4-grouped layout without requantization.
     *
     * @tparam M_R Micro-kernel M dimension (must be multiple of 4)
     * @tparam K_BLK K block size (must be multiple of 4)
     * @param A_int8 Pre-quantized int8 tensor [M x K], row-major
     * @param M Number of rows
     * @param K Number of columns
     * @param M0 Starting row index
     * @param k0 Starting column index
     * @param mr Actual rows to pack
     * @param kblk Actual K block size
     * @param A_tile_packed Output buffer for packed data
     * @param act_scales Pre-computed activation scales [M_R] (must be provided by caller)
     */
    template <int M_R, int K_BLK>
    void pack_int8_activations_to_4x4_grouped(
        const int8_t *__restrict A_int8,
        int M, int K,
        int M0, int k0,
        int mr, int kblk,
        int8_t *__restrict A_tile_packed,
        const float *__restrict act_scales) // Pre-computed, just for documentation
    {
        static_assert(M_R % 4 == 0, "M_R must be multiple of 4");
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        const int K_chunks = kblk / 4;
        const int group_stride = K_chunks * 16;

        // Pack INT8 data directly to 4x4-grouped layout (no quantization needed)
        for (int m_base = 0; m_base < mr; m_base += 4)
        {
            const int group_idx = m_base / 4;
            int8_t *group_ptr = A_tile_packed + group_idx * group_stride;

            for (int kk = 0; kk < K_chunks; ++kk)
            {
                int8_t *dst = group_ptr + kk * 16;

                for (int lane = 0; lane < 4; ++lane)
                {
                    const int row_in_tile = m_base + lane;
                    const int global_row = M0 + row_in_tile;

                    if (row_in_tile < mr && global_row < M)
                    {
                        const int8_t *src_row = A_int8 + global_row * K + k0 + kk * 4;
                        dst[lane * 4 + 0] = detail::apply_sign_bias(src_row[0]);
                        dst[lane * 4 + 1] = detail::apply_sign_bias(src_row[1]);
                        dst[lane * 4 + 2] = detail::apply_sign_bias(src_row[2]);
                        dst[lane * 4 + 3] = detail::apply_sign_bias(src_row[3]);
                    }
                    else
                    {
                        // Zero-pad boundaries (biased zero)
                        const int8_t biased_zero = detail::apply_sign_bias(0);
                        dst[lane * 4 + 0] = biased_zero;
                        dst[lane * 4 + 1] = biased_zero;
                        dst[lane * 4 + 2] = biased_zero;
                        dst[lane * 4 + 3] = biased_zero;
                    }
                }
            }
        }

        // Zero-pad remaining groups
        for (int m_base = mr; m_base < M_R; m_base += 4)
        {
            const int group_idx = m_base / 4;
            int8_t *group_ptr = A_tile_packed + group_idx * group_stride;
            std::memset(group_ptr, 0x80, group_stride);
        }
    }

} // namespace llaminar2
