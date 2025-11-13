/**
 * @file IntegerRequantization.h
 * @brief INT32→Q8_0 requantization utilities for integer GEMM
 *
 * After INT8×INT8→INT32 accumulation, we need to convert the INT32 results
 * back to Q8_0 format for storage and subsequent operations. This involves:
 *
 * 1. Scale composition: C_scale = A_scale × B_scale
 * 2. INT32 → FP32 dequantization: C_fp32[i] = C_int32[i] * C_scale
 * 3. FP32 → Q8_0 requantization: Find new scale, quantize to INT8
 *
 * This file provides both per-element and block-wise requantization functions.
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Requantize INT32 accumulator tile to Q8_0 blocks
             *
             * This function handles the final stage of integer GEMM:
             * 1. Dequantize INT32 → FP32 using combined scale (A_scale × B_scale)
             * 2. Requantize FP32 → Q8_0 (find new scale, quantize to INT8)
             *
             * @param int32_tile INT32 accumulator values [tile_m × tile_n]
             * @param tile_m Number of rows in tile
             * @param tile_n Number of columns in tile
             * @param combined_scale Scale factor from A_scale × B_scale
             * @param output Q8_0 output blocks (must have capacity for tile_m × tile_n elements)
             *
             * Note: Output is organized as row-major Q8_0 blocks.
             * Each row is divided into blocks of 32 elements.
             */
            inline void requantizeINT32ToQ8_0(
                const int32_t *int32_tile,
                int tile_m,
                int tile_n,
                double combined_scale,
                Q8_0Block *output)
            {
                // Process each row independently
                for (int i = 0; i < tile_m; ++i)
                {
                    const int32_t *row_int32 = int32_tile + i * tile_n;

                    // Number of Q8_0 blocks in this row
                    const int num_blocks = (tile_n + 31) / 32;

                    for (int b = 0; b < num_blocks; ++b)
                    {
                        Q8_0Block *block = &output[i * num_blocks + b];
                        const int block_start = b * 32;
                        const int block_end = std::min(block_start + 32, tile_n);
                        const int block_len = block_end - block_start;

                        // Step 1: Dequantize INT32 → FP32
                        float fp32_values[32];
                        for (int j = 0; j < block_len; ++j)
                        {
                            fp32_values[j] = static_cast<float>(row_int32[block_start + j] * combined_scale);
                        }
                        for (int j = block_len; j < 32; ++j)
                        {
                            fp32_values[j] = 0.0f; // Pad with zeros
                        }

                        // Step 2: Find max absolute value in block
                        float amax = 0.0f;
                        for (int j = 0; j < block_len; ++j)
                        {
                            amax = std::max(amax, std::fabs(fp32_values[j]));
                        }

                        if (amax == 0.0f)
                        {
                            // All zeros
                            std::memset(block->qs, 0, 32);
                            block->d = fp32_to_fp16(0.0f);
                            continue;
                        }

                        // Step 3: Quantize with scale = amax / 127
                        float scale = amax / 127.0f;
                        float inv_scale = 1.0f / scale;

                        for (int j = 0; j < 32; ++j)
                        {
                            int32_t q = static_cast<int32_t>(std::round(fp32_values[j] * inv_scale));
                            q = std::max(-127, std::min(127, q));
                            block->qs[j] = static_cast<int8_t>(q);
                        }

                        block->d = fp32_to_fp16(scale);
                    }
                }
            }

            /**
             * @brief Requantize single INT32 value to Q8_0 (for small tiles)
             *
             * This is a simplified version for small accumulations that don't justify
             * full block-wise processing.
             *
             * @param int32_val INT32 accumulator value
             * @param combined_scale Scale factor from A_scale × B_scale
             * @return Quantized INT8 value
             *
             * Note: This loses the scale information (assumes caller tracks it separately)
             */
            inline int8_t requantizeScalarINT32(int32_t int32_val, double combined_scale)
            {
                // Dequantize to FP32
                float fp32_val = static_cast<float>(int32_val * combined_scale);

                // Find magnitude-based scale (assuming we want to preserve dynamic range)
                float amax = std::fabs(fp32_val);
                if (amax == 0.0f)
                    return 0;

                float scale = amax / 127.0f;
                float inv_scale = 1.0f / scale;

                // Quantize
                int32_t q = static_cast<int32_t>(std::round(fp32_val * inv_scale));
                q = std::max(-127, std::min(127, q));
                return static_cast<int8_t>(q);
            }

            /**
             * @brief Compute combined scale for INT32 accumulator
             *
             * When computing C = A × B with quantized operands:
             * - A has scale a_scale (from Q8_0 blocks)
             * - B has scale b_scale (from quantized weight blocks)
             * - INT32 accumulator represents sum of (a_quant × b_quant) products
             * - True FP32 value: C_fp32 = INT32_acc × a_scale × b_scale
             *
             * This function computes the combined scale for an entire tile by
             * multiplying corresponding A and B block scales.
             *
             * @param a_scales FP16 scales from A blocks (length: num_a_blocks)
             * @param b_scales FP16 scales from B blocks (length: num_b_blocks)
             * @param num_a_blocks Number of A blocks (typically tile_m)
             * @param num_b_blocks Number of B blocks (typically tile_n)
             * @param output Combined scales (length: num_a_blocks × num_b_blocks)
             *
             * Note: This assumes each (i,j) element accumulated from k blocks,
             * each with scale a_scale[i] × b_scale[j]. For full GEMM, we need
             * to sum over K dimension, but for requantization we only care about
             * the final combined scale.
             */
            inline void computeCombinedScales(
                const uint16_t *a_scales,
                const uint16_t *b_scales,
                int num_a_blocks,
                int num_b_blocks,
                double *output)
            {
                for (int i = 0; i < num_a_blocks; ++i)
                {
                    float a_scale = fp16_to_fp32(a_scales[i]);
                    for (int j = 0; j < num_b_blocks; ++j)
                    {
                        float b_scale = fp16_to_fp32(b_scales[j]);
                        output[i * num_b_blocks + j] = static_cast<double>(a_scale) * static_cast<double>(b_scale);
                    }
                }
            }

            /**
             * @brief Accumulate INT32 tile with scale tracking
             *
             * Helper structure to track INT32 accumulator + combined scale.
             * Used during GEMM K-loop to accumulate contributions from multiple blocks.
             */
            struct INT32Accumulator
            {
                int32_t value;
                double scale;

                INT32Accumulator() : value(0), scale(0.0) {}

                void add(int32_t delta, double delta_scale)
                {
                    // Convert both to FP32, add, then convert back to INT32
                    // This is necessary because INT32 values have different scales
                    double fp32_current = static_cast<double>(value) * scale;
                    double fp32_delta = static_cast<double>(delta) * delta_scale;
                    double fp32_sum = fp32_current + fp32_delta;

                    // Update scale to be the maximum of the two (preserves dynamic range)
                    scale = std::max(scale, delta_scale);

                    // Requantize to INT32 with new scale
                    if (scale > 0.0)
                    {
                        value = static_cast<int32_t>(std::round(fp32_sum / scale));
                    }
                    else
                    {
                        value = 0;
                    }
                }

                void reset()
                {
                    value = 0;
                    scale = 0.0;
                }
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
