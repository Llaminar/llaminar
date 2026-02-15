/**
 * @file QuantizationUtils.h
 * @brief Shared quantization utilities for FP32 <-> Q8_1 conversion
 *
 * This file provides common quantization operations used across:
 * - QuantisedGemmKernel (activation quantization)
 * - FusedAttentionWoKernel (Hybrid mode Q tensor conversion)
 * - Any other component needing FP32 → Q8_1 conversion
 *
 * All operations use SIMD-optimized primitives from SIMDHelpers.h
 *
 * @author David Sanftenberg
 * @date December 2025
 */
#pragma once

#include "Tensors.h"
#include "SIMDHelpers.h"
#include "../utils/OpenMPUtils.h"
#include "../utils/DebugEnv.h"
#include <memory>
#include <vector>
#include <immintrin.h>
#include <algorithm>

namespace llaminar2::quantization
{

    /**
     * @brief Quantize FP32 buffer to Q8_1 blocks (in-place to pre-allocated buffer)
     *
     * This is the core quantization routine used by GEMM kernels. It performs
     * parallel SIMD quantization of FP32 activations to Q8_1 format.
     *
     * @param A Input FP32 buffer [m, k] row-major
     * @param q8_1_buffer Output Q8_1 block buffer (must be pre-allocated)
     * @param m Number of rows
     * @param k Number of columns (elements per row)
     * @return true on success
     *
     * Buffer sizing:
     *   k_blocks = (k + 31) / 32
     *   total_blocks = m * k_blocks
     *   buffer_size_bytes = total_blocks * sizeof(Q8_1Block)
     */
    inline bool quantize_fp32_to_q8_1_buffer(
        const float *A,
        void *q8_1_buffer,
        int m, int k)
    {
        if (!A || !q8_1_buffer)
        {
            return false;
        }

        const int k_blocks = (k + 31) / 32;
        Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(q8_1_buffer);
        const bool k_aligned = (k % 32 == 0);
        const int total_blocks = m * k_blocks;

        if (total_blocks < 128)
        {
            for (int i = 0; i < m; ++i)
            {
                const float *a_row = A + i * k;
                Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                if (k_aligned)
                {
                    for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                    {
                        simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                    }
                }
                else
                {
                    for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                    {
                        const int valid_elements = std::min(32, k - k_blk * 32);
                        simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], valid_elements);
                    }
                }
            }

            return true;
        }

        // Use OMP_WORKSHARE_REGION for nested parallelism compatibility
        auto do_quantize = [&]()
        {
            int quant_thresh = debugEnv().gemm.gemm_quant_parallel_threshold;
            if (quant_thresh == 0)
                quant_thresh = omp_get_max_threads();

            // For small m, parallelize across both rows and blocks
            if (m < quant_thresh)
            {
#pragma omp for collapse(2) schedule(static)
                for (int i = 0; i < m; ++i)
                {
                    for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                    {
                        const float *a_row = A + i * k + k_blk * 32;
                        Q8_1Block *row_blocks = all_blocks + i * k_blocks;
                        int valid_elements = std::min(32, k - k_blk * 32);
                        simd::quantize_single_block(a_row, row_blocks[k_blk], valid_elements);
                    }
                }
            }
            else
            {
                // For large m, parallelize only across rows
#pragma omp for schedule(static)
                for (int i = 0; i < m; ++i)
                {
                    const float *a_row = A + i * k;
                    Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                    if (k_aligned)
                    {
                        for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                        {
                            simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                        }
                    }
                    else
                    {
                        // Handle all full blocks
                        for (int k_blk = 0; k_blk < k_blocks - 1; ++k_blk)
                        {
                            simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                        }
                        // Handle partial last block
                        if (k_blocks > 0)
                        {
                            int last_k_blk = k_blocks - 1;
                            int valid_elements = k - last_k_blk * 32;
                            simd::quantize_single_block(a_row + last_k_blk * 32, row_blocks[last_k_blk], valid_elements);
                        }
                    }
                }
            }
        };

        OMP_WORKSHARE_REGION(do_quantize);

        return true;
    }

    /**
     * @brief Calculate required buffer size for Q8_1 quantization
     *
     * @param m Number of rows
     * @param k Number of columns
     * @return Size in bytes needed for Q8_1 block buffer
     */
    inline size_t q8_1_buffer_size(int m, int k)
    {
        const size_t k_blocks = (k + 31) / 32;
        return static_cast<size_t>(m) * k_blocks * sizeof(Q8_1Block);
    }

    /**
     * @brief Calculate number of Q8_1 blocks per row
     */
    inline size_t q8_1_blocks_per_row(int k)
    {
        return (k + 31) / 32;
    }

    /**
     * @brief Quantize FP32 tensor to new Q8_1 tensor
     *
     * Creates and returns a new Q8_1Tensor containing the quantized data.
     * This is a convenience wrapper for cases where a tensor output is needed.
     *
     * @param fp32_tensor Input FP32 tensor [rows, cols]
     * @return New Q8_1Tensor with quantized data, or nullptr on failure
     */
    inline std::unique_ptr<Q8_1Tensor> quantize_fp32_tensor_to_q8_1(
        const FP32Tensor *fp32_tensor)
    {
        if (!fp32_tensor)
        {
            return nullptr;
        }

        const auto &shape = fp32_tensor->shape();
        if (shape.size() < 1)
        {
            return nullptr;
        }

        // Determine rows and cols
        size_t rows = shape[0];
        size_t cols = shape.size() > 1 ? shape[1] : 1;

        // Create output Q8_1 tensor
        auto q8_tensor = std::make_unique<Q8_1Tensor>(shape);

        // Quantize
        if (!quantize_fp32_to_q8_1_buffer(
                fp32_tensor->data(),
                q8_tensor->mutable_typed_data(),
                static_cast<int>(rows),
                static_cast<int>(cols)))
        {
            return nullptr;
        }

        return q8_tensor;
    }

    /**
     * @brief Quantize FP32 data to existing Q8_1 tensor's blocks
     *
     * Useful when you have a pre-allocated Q8_1Tensor and want to
     * quantize new FP32 data into it.
     *
     * @param fp32_data Input FP32 data [rows * cols]
     * @param q8_tensor Output Q8_1 tensor (must be pre-allocated with correct shape)
     * @param rows Number of rows
     * @param cols Number of columns
     * @return true on success
     */
    inline bool quantize_fp32_to_existing_q8_1(
        const float *fp32_data,
        Q8_1Tensor *q8_tensor,
        int rows, int cols)
    {
        if (!fp32_data || !q8_tensor)
        {
            return false;
        }

        return quantize_fp32_to_q8_1_buffer(
            fp32_data,
            q8_tensor->mutable_typed_data(),
            rows, cols);
    }

    // =========================================================================
    // STREAMING DEQUANTIZATION FOR VNNI-PACKED WEIGHTS
    // =========================================================================
    // These functions provide streaming (row-by-row) dequantization of VNNI-packed
    // quantized weights to FP32. This enables Hybrid mode's highest-precision
    // Wo projection: FP32 context × dequant(VNNI-packed Wo) → FP32 output
    //
    // The streaming approach avoids pre-dequantizing the entire weight matrix,
    // which would increase memory bandwidth. Instead, we dequantize one row at
    // a time into a temporary buffer during the outer GEMM loop.
    // =========================================================================

    /**
     * @brief Dequantize a single row from VNNI-packed weights to FP32 (streaming)
     *
     * This function dequantizes one row of VNNI-packed weights to FP32.
     * It's designed for streaming use in GEMM where we process one output
     * row at a time, avoiding the need to dequantize the entire weight matrix.
     *
     * VNNI-packed layout reminder:
     *   packed_data: [K/4][N][4] flattened - groups of 4 K elements for VNNI
     *   scales: [K/32][N] - per-block scales
     *   mins: [K/32][N] - asymmetric offsets (optional, for IQ4_NL)
     *
     * Dequantization formula:
     *   fp32[k] = packed_data[k] * scale[k_block] + min[k_block]
     *
     * @param packed_data VNNI-packed INT8 weight data [K/4 * N * 4]
     * @param scales Per-block scale factors [K/32 * N]
     * @param mins Per-block min values (can be nullptr for symmetric quant) [K/32 * N]
     * @param output_fp32 Output FP32 buffer [K] (must be pre-allocated)
     * @param row_idx Row index in the weight matrix (0 to N-1)
     * @param K Number of input features (columns in original weight matrix)
     * @param N Number of output features (rows in original weight matrix)
     */
    inline void dequantize_vnni_packed_row_to_fp32(
        const int8_t *packed_data,
        const float *scales,
        const float *mins,
        float *output_fp32,
        int row_idx,
        int K,
        int N)
    {
        const int k_blocks = (K + 31) / 32;

        // Process each block of 32 K elements
        for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
        {
            // Get scale and min for this block
            // scales/mins layout: [K/32][N], so index is k_blk * N + row_idx
            const float scale = scales[k_blk * N + row_idx];
            const float min_val = mins ? mins[k_blk * N + row_idx] : 0.0f;

            // Dequantize 32 elements from packed format
            // packed_data layout: [K/4][N][4]
            // For a given k in [0, K), the packed index is:
            //   group = k / 4
            //   offset_in_group = k % 4
            //   packed_idx = group * (N * 4) + row_idx * 4 + offset_in_group
            const int k_start = k_blk * 32;
            const int k_end = std::min(k_start + 32, K);

#ifdef __AVX512F__
            // AVX-512 optimized path: process 16 elements at a time
            const __m512 scale_vec = _mm512_set1_ps(scale);
            const __m512 min_vec = _mm512_set1_ps(min_val);

            for (int k = k_start; k < k_end; k += 16)
            {
                int elements_left = k_end - k;
                if (elements_left >= 16)
                {
                    // Load 16 INT8 values from packed format (scattered)
                    alignas(64) int8_t temp[16];
                    for (int i = 0; i < 16; ++i)
                    {
                        int kk = k + i;
                        int group = kk / 4;
                        int offset = kk % 4;
                        temp[i] = packed_data[group * (N * 4) + row_idx * 4 + offset];
                    }

                    // Convert INT8 -> INT32 -> FP32
                    __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(temp));
                    __m512i i32 = _mm512_cvtepi8_epi32(q8);
                    __m512 f32 = _mm512_cvtepi32_ps(i32);

                    // Dequantize: fp32 = q8 * scale + min
                    __m512 result = _mm512_fmadd_ps(f32, scale_vec, min_vec);
                    _mm512_storeu_ps(output_fp32 + k, result);
                }
                else
                {
                    // Handle remaining elements scalar
                    for (int i = 0; i < elements_left; ++i)
                    {
                        int kk = k + i;
                        int group = kk / 4;
                        int offset = kk % 4;
                        int8_t q = packed_data[group * (N * 4) + row_idx * 4 + offset];
                        output_fp32[kk] = static_cast<float>(q) * scale + min_val;
                    }
                }
            }
#elif defined(__AVX2__)
            // AVX2 optimized path: process 8 elements at a time
            const __m256 scale_vec = _mm256_set1_ps(scale);
            const __m256 min_vec = _mm256_set1_ps(min_val);

            for (int k = k_start; k < k_end; k += 8)
            {
                int elements_left = k_end - k;
                if (elements_left >= 8)
                {
                    // Load 8 INT8 values from packed format
                    alignas(32) int8_t temp[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        int kk = k + i;
                        int group = kk / 4;
                        int offset = kk % 4;
                        temp[i] = packed_data[group * (N * 4) + row_idx * 4 + offset];
                    }

                    // Convert INT8 -> INT32 -> FP32
                    __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(temp));
                    __m256i i32 = _mm256_cvtepi8_epi32(q8);
                    __m256 f32 = _mm256_cvtepi32_ps(i32);

                    // Dequantize: fp32 = q8 * scale + min
                    __m256 result = _mm256_fmadd_ps(f32, scale_vec, min_vec);
                    _mm256_storeu_ps(output_fp32 + k, result);
                }
                else
                {
                    // Handle remaining elements scalar
                    for (int i = 0; i < elements_left; ++i)
                    {
                        int kk = k + i;
                        int group = kk / 4;
                        int offset = kk % 4;
                        int8_t q = packed_data[group * (N * 4) + row_idx * 4 + offset];
                        output_fp32[kk] = static_cast<float>(q) * scale + min_val;
                    }
                }
            }
#else
            // Scalar fallback
            for (int k = k_start; k < k_end; ++k)
            {
                int group = k / 4;
                int offset = k % 4;
                int8_t q = packed_data[group * (N * 4) + row_idx * 4 + offset];
                output_fp32[k] = static_cast<float>(q) * scale + min_val;
            }
#endif
        }
    }

    /**
     * @brief Calculate FP32 buffer size for dequantized row
     *
     * @param K Number of input features
     * @return Size in bytes for FP32 row buffer
     */
    inline size_t dequantized_row_buffer_size(int K)
    {
        // Align to 64 bytes for AVX-512
        return ((K + 15) / 16) * 16 * sizeof(float);
    }

    /**
     * @brief Dequantize entire VNNI-packed weight matrix to FP32 (bulk)
     *
     * This function dequantizes the entire VNNI-packed weight matrix to FP32.
     * Use this when you need the full FP32 weight matrix (e.g., for debugging
     * or when the weight matrix is small enough to fit in cache).
     *
     * For large matrices, prefer streaming dequantization via
     * dequantize_vnni_packed_row_to_fp32() to avoid memory bloat.
     *
     * @param packed_data VNNI-packed INT8 weight data
     * @param scales Per-block scale factors
     * @param mins Per-block min values (can be nullptr)
     * @param output_fp32 Output FP32 buffer [N * K] (must be pre-allocated)
     * @param K Number of input features (columns)
     * @param N Number of output features (rows)
     */
    inline void dequantize_vnni_packed_all_to_fp32(
        const int8_t *packed_data,
        const float *scales,
        const float *mins,
        float *output_fp32,
        int K,
        int N)
    {
        auto do_dequant = [&]()
        {
#pragma omp for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                dequantize_vnni_packed_row_to_fp32(
                    packed_data, scales, mins,
                    output_fp32 + n * K,
                    n, K, N);
            }
        };

        OMP_WORKSHARE_REGION(do_dequant);
    }

} // namespace llaminar2::quantization
