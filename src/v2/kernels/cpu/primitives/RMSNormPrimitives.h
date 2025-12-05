/**
 * @file RMSNormPrimitives.h
 * @brief Vectorized RMSNorm primitives for V2 (ported from V1)
 * @author David Sanftenberg
 *
 * High-performance RMSNorm implementation with:
 * - AVX512/AVX2 vectorization (4-8× speedup)
 * - Double precision accumulation for accuracy
 * - Thread-local scratch buffers
 * - T5 compatibility mode
 * - INT32→INT8 pipeline support for full INT8 inference
 * - Q8_1 integer-space RMSNorm (avoids FP32 dequant overhead)
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "../../../tensors/BlockStructures.h"

namespace llaminar2::primitives
{

    /**
     * @brief Execution options for RMSNorm
     */
    struct RMSNormExecOptions
    {
        bool allow_parallel = true;                  // Permit OpenMP parallelization
        std::size_t parallel_threshold_elems = 2048; // Threshold for parallelization
        bool t5_compat_mode = false;                 // Use float32 accumulation for T5 parity
        bool force_scalar = false;                   // Force scalar path (for testing)
    };

    /**
     * @brief Reusable scratch buffers to avoid per-call allocations
     */
    struct RMSNormScratch
    {
        std::vector<double> row_sumsq; // Per-row sum of squares
        std::vector<float> inv;        // Per-row inverse RMS

        void ensure(std::size_t rows)
        {
            if (row_sumsq.size() < rows)
                row_sumsq.resize(rows);
            if (inv.size() < rows)
                inv.resize(rows);
        }

        void clear()
        {
            row_sumsq.clear();
            inv.clear();
        }
    };

    /**
     * @brief Compute per-row sum of squares (vectorized)
     *
     * @param src Input tensor [rows, cols]
     * @param rows Number of rows
     * @param cols Number of columns
     * @param row_sumsq Output per-row sum of squares (length = rows)
     * @param opts Execution options
     */
    void rmsnorm_compute_row_sumsq_vectorized(
        const float *src,
        std::size_t rows,
        std::size_t cols,
        double *row_sumsq,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Convert row sum of squares to inverse RMS scaling factors
     *
     * @param row_sumsq Per-row sum of squares
     * @param rows Number of rows
     * @param cols Number of columns (for mean calculation)
     * @param epsilon Epsilon for numerical stability
     * @param inv_out Output inverse RMS values (length = rows)
     */
    void rmsnorm_compute_inv(
        const double *row_sumsq,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        float *inv_out);

    /**
     * @brief Apply RMSNorm scaling and gamma weights (vectorized)
     *
     * @param src Input tensor [rows, cols]
     * @param gamma Gamma weights [cols]
     * @param inv Per-row inverse RMS values
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dst Output tensor [rows, cols]
     * @param opts Execution options
     */
    void rmsnorm_apply_vectorized(
        const float *src,
        const float *gamma,
        const float *inv,
        std::size_t rows,
        std::size_t cols,
        float *dst,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Fused RMSNorm computation (all stages)
     *
     * @param src Input tensor [rows, cols]
     * @param gamma Gamma weights [cols]
     * @param dst Output tensor [rows, cols]
     * @param rows Number of rows
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     * @param opts Execution options
     */
    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Fused RMSNorm with provided scratch buffer
     */
    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        RMSNormScratch &scratch,
        const RMSNormExecOptions &opts = {});

    // ========================================================================
    // Per-Row Primitives (Testable SIMD variants)
    // ========================================================================

    /**
     * @brief RMSNorm per-row FP32 scalar implementation
     *
     * Computes RMSNorm for a single row using scalar arithmetic.
     *
     * @param src Input row [cols]
     * @param gamma Gamma weights [cols]
     * @param dst Output row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_scalar(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm per-row FP32 AVX2 implementation
     *
     * Computes RMSNorm for a single row using AVX2 vectorization.
     * Available only when AVX2 is supported at compile time.
     *
     * @param src Input row [cols]
     * @param gamma Gamma weights [cols]
     * @param dst Output row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_avx2(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm per-row FP32 AVX512 implementation
     *
     * Computes RMSNorm for a single row using AVX512 vectorization.
     * Available only when AVX512F is supported at compile time.
     *
     * @param src Input row [cols]
     * @param gamma Gamma weights [cols]
     * @param dst Output row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_avx512(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm per-row BF16 scalar implementation
     *
     * Computes RMSNorm for a single row using scalar arithmetic.
     * Input/output are BF16 (stored as uint16_t), gamma is FP32.
     *
     * @param src Input BF16 row [cols]
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst Output BF16 row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_bf16_scalar(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm per-row BF16 AVX2 implementation
     *
     * Computes RMSNorm for a single row using AVX2 vectorization.
     * Input/output are BF16 (stored as uint16_t), gamma is FP32.
     *
     * @param src Input BF16 row [cols]
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst Output BF16 row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_bf16_avx2(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm per-row BF16 AVX512 implementation
     *
     * Computes RMSNorm for a single row using AVX512 vectorization.
     * Input/output are BF16 (stored as uint16_t), gamma is FP32.
     *
     * @param src Input BF16 row [cols]
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst Output BF16 row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_bf16_avx512(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm per-row FP16 scalar implementation
     *
     * Computes RMSNorm for a single row using scalar arithmetic.
     * Input/output are FP16 (stored as uint16_t), gamma is FP32.
     *
     * @param src Input FP16 row [cols]
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst Output FP16 row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_fp16_scalar(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm per-row FP16 AVX2 implementation
     *
     * Computes RMSNorm for a single row using AVX2 vectorization (F16C).
     * Input/output are FP16 (stored as uint16_t), gamma is FP32.
     *
     * @param src Input FP16 row [cols]
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst Output FP16 row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_fp16_avx2(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm per-row FP16 AVX512 implementation
     *
     * Computes RMSNorm for a single row using AVX512 vectorization.
     * Input/output are FP16 (stored as uint16_t), gamma is FP32.
     *
     * @param src Input FP16 row [cols]
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst Output FP16 row [cols]
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_fused_row_fp16_avx512(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon);

    // ========================================================================
    // INT32 RMSNorm (for full INT8 pipelines)
    // ========================================================================

    /**
     * @brief Compute per-row sum of squares from INT32 input (vectorized)
     *
     * Optimized for INT32 accumulator tensors in full INT8 pipelines.
     * Converts INT32 to double for accumulation, maintaining high precision.
     *
     * @param src Input INT32 tensor [rows, cols]
     * @param rows Number of rows
     * @param cols Number of columns
     * @param row_sumsq Output per-row sum of squares (length = rows)
     * @param opts Execution options
     */
    void rmsnorm_compute_row_sumsq_int32_vectorized(
        const int32_t *src,
        std::size_t rows,
        std::size_t cols,
        double *row_sumsq,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Apply RMSNorm scaling, gamma weights, and requantize to INT8
     *
     * Fused operation for INT32→FP32 normalization→INT8 quantization.
     * Uses per-row dynamic scaling to fit normalized values into INT8 range.
     *
     * @param src Input INT32 tensor [rows, cols]
     * @param gamma Gamma weights [cols] (FP32)
     * @param inv Per-row inverse RMS values (FP32)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dst_int8 Output INT8 tensor [rows, cols]
     * @param dst_row_scales Output per-row INT8 scales [rows]
     * @param opts Execution options
     */
    void rmsnorm_apply_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        const float *inv,
        std::size_t rows,
        std::size_t cols,
        int8_t *dst_int8,
        float *dst_row_scales,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Fused INT32 RMSNorm: INT32 input → INT8 output with requantization
     *
     * Complete pipeline for INT32 accumulator normalization:
     * 1. Compute sum of squares from INT32 input
     * 2. Calculate inverse RMS per row
     * 3. Normalize and apply gamma weights
     * 4. Requantize to INT8 with per-row dynamic scaling
     *
     * @param src Input INT32 tensor [rows, cols]
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst_int8 Output INT8 tensor [rows, cols]
     * @param dst_row_scales Output per-row INT8 scales [rows]
     * @param rows Number of rows
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     * @param opts Execution options
     */
    void rmsnorm_fused_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        int8_t *dst_int8,
        float *dst_row_scales,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Fused INT32 RMSNorm with provided scratch buffer
     */
    void rmsnorm_fused_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        int8_t *dst_int8,
        float *dst_row_scales,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        RMSNormScratch &scratch,
        const RMSNormExecOptions &opts = {});

    // ========================================================================
    // BF16 RMSNorm (native bfloat16 operations)
    // ========================================================================

    /**
     * @brief Fused RMSNorm for BF16 tensors (native precision)
     *
     * Performs RMSNorm directly on BF16 data without conversion to FP32:
     * 1. Convert BF16 input to FP32 for high-precision accumulation
     * 2. Compute RMS normalization
     * 3. Apply gamma weights
     * 4. Convert result back to BF16
     *
     * @param src Input BF16 tensor [rows, cols] (stored as uint16_t)
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst Output BF16 tensor [rows, cols] (stored as uint16_t)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     * @param opts Execution options
     */
    void rmsnorm_fused_bf16_vectorized(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts = {});

    // ========================================================================
    // FP16 RMSNorm (native float16 operations)
    // ========================================================================

    /**
     * @brief Fused RMSNorm for FP16 tensors (native precision)
     *
     * Performs RMSNorm directly on FP16 data without conversion to FP32:
     * 1. Convert FP16 input to FP32 for high-precision accumulation
     * 2. Compute RMS normalization
     * 3. Apply gamma weights
     * 4. Convert result back to FP16
     *
     * @param src Input FP16 tensor [rows, cols] (stored as uint16_t)
     * @param gamma Gamma weights [cols] (FP32)
     * @param dst Output FP16 tensor [rows, cols] (stored as uint16_t)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     * @param opts Execution options
     */
    void rmsnorm_fused_fp16_vectorized(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts = {});

    // ========================================================================
    // Fused RMSNorm + INT8 Quantization (per-row, for FusedRMSNormQuantize kernel)
    // ========================================================================

    /**
     * @brief Fused RMSNorm + INT8 quantization for a single row (scalar fallback)
     *
     * Computes: output = quantize_int8((input / rms) * weight)
     * where rms = sqrt(mean(input²) + eps)
     *
     * @param input_row Input row [cols] FP32
     * @param weight RMSNorm gamma weights [cols] FP32
     * @param output_row Output row [cols] INT8
     * @param out_scale Output per-row scale (for dequantization)
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_quantize_row_scalar(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon);

    /**
     * @brief Fused RMSNorm + INT8 quantization for a single row (AVX2)
     */
    void rmsnorm_quantize_row_avx2(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon);

    /**
     * @brief Fused RMSNorm + INT8 quantization for a single row (AVX512)
     */
    void rmsnorm_quantize_row_avx512(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon);

    // ========================================================================
    // Q8_1 Integer-Space RMSNorm (avoids FP32 dequantization for sum-of-squares)
    // ========================================================================

    /**
     * @brief Q8_1 RMSNorm computed mostly in integer space
     *
     * Key insight: For Q8_1, the dequantized value is: x_i = d * qs[i]
     * So sum(x_i²) = d² * sum(qs[i]²), where sum(qs[i]²) is pure integer!
     *
     * This avoids expensive FP32 dequantization in the sum-of-squares phase.
     * Only the final normalization requires FP32 operations.
     *
     * Algorithm:
     * 1. Integer phase: Compute sum(qs[i]²) using SIMD integer ops (AVX512-VNNI)
     * 2. Scale phase: total_sum_sq = Σ(d_b² × int_sum_sq_b) over all blocks
     * 3. FP phase: inv_rms = 1 / sqrt(total_sum_sq / n + eps)
     * 4. Output: Re-quantize normalized values to Q8_1
     *
     * @param input Input Q8_1 blocks (blocks_per_row blocks)
     * @param gamma Gamma weights [cols] (FP32)
     * @param output Output Q8_1 blocks (blocks_per_row blocks)
     * @param blocks_per_row Number of Q8_1 blocks per row (cols / 32)
     * @param epsilon Epsilon for numerical stability
     */
    void rmsnorm_q8_1_integer_row(
        const Q8_1Block *input,
        const float *gamma,
        Q8_1Block *output,
        std::size_t blocks_per_row,
        float epsilon);

    /**
     * @brief Q8_1 RMSNorm for multiple rows (parallelized)
     *
     * @param input Input Q8_1 blocks [rows * blocks_per_row]
     * @param gamma Gamma weights [cols] (FP32)
     * @param output Output Q8_1 blocks [rows * blocks_per_row]
     * @param rows Number of rows
     * @param blocks_per_row Number of Q8_1 blocks per row (cols / 32)
     * @param epsilon Epsilon for numerical stability
     * @param opts Execution options
     */
    void rmsnorm_q8_1_integer(
        const Q8_1Block *input,
        const float *gamma,
        Q8_1Block *output,
        std::size_t rows,
        std::size_t blocks_per_row,
        float epsilon,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Pure-integer Q8_1 RMSNorm (EXPERIMENTAL)
     *
     * Performs RMSNorm entirely in the integer domain without any FP32 operations.
     * Requires pre-quantized gamma weights. Uses AVX512 vectorization.
     *
     * Key simplification: Ignores the per-block FP16 scale 'd' during normalization,
     * computing inv_rms based only on the integer qs values. This works because
     * RMSNorm normalizes the distribution, and the relative magnitudes within
     * the quantized representation are what matter.
     *
     * @param input Input Q8_1 blocks (blocks_per_row blocks)
     * @param gamma_q8 Pre-quantized gamma weights [cols] in Q8 format (gamma * 256)
     * @param output Output Q8_1 blocks (blocks_per_row blocks)
     * @param blocks_per_row Number of Q8_1 blocks per row (cols / 32)
     * @param epsilon_scaled Epsilon pre-scaled (typically 1-100 for stability)
     */
    void rmsnorm_q8_1_pure_integer_row(
        const Q8_1Block *input,
        const int16_t *gamma_q8,
        Q8_1Block *output,
        std::size_t blocks_per_row,
        int32_t epsilon_scaled);

} // namespace llaminar2::primitives
