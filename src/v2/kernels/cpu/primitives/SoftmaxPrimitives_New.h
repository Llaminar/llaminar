/**
 * @file SoftmaxPrimitives.h
 * @brief Multi-precision Softmax primitives with separated SIMD variants
 * @author David Sanftenberg
 *
 * V2 Softmax implementation with:
 * - Native precision support (FP32, BF16, FP16)
 * - Q8_1 integer-aware softmax (minimizes FP conversions)
 * - Separated scalar/AVX2/AVX512 variants for testing
 * - Causal masking support
 * - Numerical stability (max subtraction)
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include "../../../tensors/BlockStructures.h"

namespace llaminar2::primitives
{

    // ============================================================================
    // FP32 Softmax Primitives
    // ============================================================================

    /**
     * @brief Softmax for single FP32 row (scalar implementation)
     *
     * @param row Input/output [cols] (in-place)
     * @param cols Row length
     * @param causal Apply causal masking (j > i → 0 probability)
     * @param scale Scale factor applied before exp
     * @param row_idx Row index (for causal masking: only needed if causal=true)
     *
     * Algorithm:
     *   1. max_val = max(row) with causal masking
     *   2. sum = Σ exp(row[j] - max_val) for valid j
     *   3. row[j] = exp(row[j] - max_val) / sum
     */
    inline void softmax_row_fp32_scalar(
        float *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP32 row (AVX2 implementation)
     */
    inline void softmax_row_fp32_avx2(
        float *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP32 row (AVX512 implementation)
     */
    inline void softmax_row_fp32_avx512(
        float *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP32 row (compile-time dispatch)
     *
     * Dispatches to best available SIMD implementation.
     */
    inline void softmax_row_fp32(
        float *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    // ============================================================================
    // BF16 Softmax Primitives
    // ============================================================================

    /**
     * @brief Softmax for single BF16 row (scalar implementation)
     *
     * Native BF16 computation:
     *   - Convert BF16 → FP32 for exp/div (unavoidable)
     *   - Accumulation in FP32 for numerical stability
     *   - Convert FP32 → BF16 for output
     *
     * @param row Input/output [cols] (in-place, BF16 as uint16_t)
     * @param cols Row length
     * @param causal Apply causal masking
     * @param scale Scale factor
     * @param row_idx Row index (for causal masking)
     */
    inline void softmax_row_bf16_scalar(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single BF16 row (AVX2 implementation)
     *
     * Uses _mm256_cvtne2ps_pbh (AVX512_BF16) if available, otherwise manual conversion.
     */
    inline void softmax_row_bf16_avx2(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single BF16 row (AVX512 implementation)
     *
     * Uses native AVX512_BF16 instructions for conversion.
     */
    inline void softmax_row_bf16_avx512(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single BF16 row (compile-time dispatch)
     */
    inline void softmax_row_bf16(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    // ============================================================================
    // FP16 Softmax Primitives
    // ============================================================================

    /**
     * @brief Softmax for single FP16 row (scalar implementation)
     *
     * Native FP16 computation:
     *   - Convert FP16 → FP32 for exp/div
     *   - Accumulation in FP32 for numerical stability
     *   - Convert FP32 → FP16 for output
     *
     * @param row Input/output [cols] (in-place, FP16 as uint16_t)
     * @param cols Row length
     * @param causal Apply causal masking
     * @param scale Scale factor
     * @param row_idx Row index (for causal masking)
     */
    inline void softmax_row_fp16_scalar(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP16 row (AVX2 implementation)
     *
     * Uses F16C instructions (_mm256_cvtph_ps, _mm256_cvtps_ph).
     */
    inline void softmax_row_fp16_avx2(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP16 row (AVX512 implementation)
     *
     * Uses AVX512FP16 if available, otherwise falls back to F16C.
     */
    inline void softmax_row_fp16_avx512(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP16 row (compile-time dispatch)
     */
    inline void softmax_row_fp16(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    // ============================================================================
    // Q8_1 Integer-Aware Softmax Primitives
    // ============================================================================

    /**
     * @brief Softmax for single Q8_1 row (scalar implementation)
     *
     * Integer-aware Q8_1 softmax that minimizes floating-point conversions:
     *
     * Q8_1 Block format:
     *   - d: FP16 scale factor
     *   - sum_qs: INT16 sum of quantized values
     *   - qs[32]: INT8 quantized values
     *   - Dequantized value: x[i] = qs[i] * fp16_to_fp32(d)
     *
     * Algorithm (3-pass, optimized for Q8_1):
     *
     * Pass 1 - Find Maximum (integer-optimized):
     *   - Find max(qs[i]) within each block (pure integer comparison)
     *   - Scale block max: block_max_fp32 = max_qs * d
     *   - Reduce across blocks to find row maximum
     *
     * Pass 2 - Sum of Exponentials:
     *   - For each element: exp_val = exp(qs[i] * d - row_max)
     *   - Factor out scale: exp(d * (qs[i] - row_max/d))
     *   - When d is uniform across blocks, can batch multiply
     *   - Accumulate sum in FP32 for stability
     *
     * Pass 3 - Normalize and Requantize:
     *   - prob[i] = exp_val / sum
     *   - Find new scale d' = max(prob[i]) / 127.0f
     *   - Requantize: qs'[i] = round(prob[i] / d')
     *   - Recompute sum_qs' = Σ qs'[i]
     *
     * Performance benefit vs full dequant:
     *   - Pass 1: Integer max is ~4x faster than FP32 max
     *   - Pass 2: Batched scale multiply (1 mul per block vs per element)
     *   - Pass 3: Direct requantization (no intermediate storage)
     *
     * @param row Input/output Q8_1 blocks [n_blocks] (in-place)
     * @param n_blocks Number of Q8_1 blocks (cols = n_blocks * 32)
     * @param causal Apply causal masking (j > row_idx → 0 probability)
     * @param scale Scale factor applied before exp (typically 1/sqrt(d_k))
     * @param row_idx Row index (for causal masking: only needed if causal=true)
     *
     * @note Output probabilities are in [0, 1] range, requantized to Q8_1.
     *       This may lose precision for very small probabilities.
     */
    inline void softmax_row_q8_1_scalar(
        Q8_1Block *row,
        int n_blocks,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single Q8_1 row (AVX2 implementation)
     *
     * Vectorized Q8_1 softmax using AVX2:
     *   - vpmaxsb: 32-way INT8 max comparison
     *   - vcvtph2ps: FP16 → FP32 scale conversion (F16C)
     *   - vpmovmskb: Extract max from 32 INT8 lanes
     *   - 8-way FP32 exp polynomial (avoid _mm256_exp_ps dependency)
     */
    inline void softmax_row_q8_1_avx2(
        Q8_1Block *row,
        int n_blocks,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single Q8_1 row (AVX512 implementation)
     *
     * Vectorized Q8_1 softmax using AVX-512:
     *   - vpmaxsb (zmm): 64-way INT8 max comparison
     *   - vcvtph2ps (zmm): 16-way FP16 → FP32 conversion
     *   - _mm512_reduce_max_epi8: Horizontal max reduction
     *   - 16-way FP32 exp polynomial
     */
    inline void softmax_row_q8_1_avx512(
        Q8_1Block *row,
        int n_blocks,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single Q8_1 row (compile-time dispatch)
     *
     * Dispatches to best available SIMD implementation.
     */
    inline void softmax_row_q8_1(
        Q8_1Block *row,
        int n_blocks,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    // ============================================================================
    // Row-Major Batch Softmax (Multi-row)
    // ============================================================================

    /**
     * @brief Apply softmax to multiple FP32 rows (row-major layout)
     *
     * @param scores Input/output [rows, cols] (in-place)
     * @param rows Number of rows
     * @param cols Number of columns per row
     * @param causal Apply causal masking (row i: only j <= i valid)
     * @param scale Scale factor applied before exp
     * @param parallel Enable OpenMP parallelization
     */
    inline void softmax_row_major_fp32(
        float *scores,
        int rows,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        bool parallel = true)
    {
#pragma omp parallel for schedule(static) if (parallel)
        for (int r = 0; r < rows; ++r)
        {
            softmax_row_fp32(scores + r * cols, cols, causal, scale, r);
        }
    }

    /**
     * @brief Apply softmax to multiple BF16 rows (row-major layout)
     */
    inline void softmax_row_major_bf16(
        uint16_t *scores,
        int rows,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        bool parallel = true)
    {
#pragma omp parallel for schedule(static) if (parallel)
        for (int r = 0; r < rows; ++r)
        {
            softmax_row_bf16(scores + r * cols, cols, causal, scale, r);
        }
    }

    /**
     * @brief Apply softmax to multiple FP16 rows (row-major layout)
     */
    inline void softmax_row_major_fp16(
        uint16_t *scores,
        int rows,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        bool parallel = true)
    {
#pragma omp parallel for schedule(static) if (parallel)
        for (int r = 0; r < rows; ++r)
        {
            softmax_row_fp16(scores + r * cols, cols, causal, scale, r);
        }
    }

    /**
     * @brief Apply softmax to multiple Q8_1 rows (block-major layout)
     *
     * @param scores Input/output Q8_1 blocks [rows * n_blocks_per_row]
     * @param rows Number of rows
     * @param n_blocks_per_row Number of Q8_1 blocks per row (cols = n_blocks_per_row * 32)
     * @param causal Apply causal masking (row i: only j <= i valid)
     * @param scale Scale factor applied before exp
     * @param parallel Enable OpenMP parallelization
     */
    inline void softmax_row_major_q8_1(
        Q8_1Block *scores,
        int rows,
        int n_blocks_per_row,
        bool causal = false,
        float scale = 1.0f,
        bool parallel = true)
    {
#pragma omp parallel for schedule(static) if (parallel)
        for (int r = 0; r < rows; ++r)
        {
            softmax_row_q8_1(scores + r * n_blocks_per_row, n_blocks_per_row, causal, scale, r);
        }
    }

} // namespace llaminar2::primitives

// ============================================================================
// Inline Implementations
// ============================================================================

#include "SoftmaxPrimitivesImpl.h"
