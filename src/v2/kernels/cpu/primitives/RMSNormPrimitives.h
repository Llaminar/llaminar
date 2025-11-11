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
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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

} // namespace llaminar2::primitives
