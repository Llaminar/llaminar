/**
 * @file RMSNormPrimitives.h
 * @brief Vectorized RMSNorm primitives for V2 (ported from V1)
 * @author David Sanftenberg
 *
 * FP32 matrix normalization retains complete original rows within one worker
 * transaction: summary, inverse and publication have the same lexical owner.
 * Native ARM floating and original-Q16 tiles share gamma across1..4 rows
 * without decoded row scratch. Q16 retains FP32 squares before FP64 accumulation,
 * unlike floating RMSNorm's widen-before-square program. Their fixed arithmetic
 * is independent of M or worker count. Explicit T5
 * accumulation remains a distinct program. Original tensors/gamma retain their
 * caller-owned storage authority. Other typed publication APIs keep their own
 * numerical programs:
 * - INT32→INT8 original-row summary/inverse and native two-pass publication,
 *   without scratch allocation or decoded-row/TLS mirrors
 * - Q8_1 integer-space RMSNorm (avoids FP32 dequant overhead)
 */
#pragma once

#include <cstddef>
#include <cstdint>
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



    // ========================================================================
    // Per-Row Primitives (Testable SIMD variants)
    // ========================================================================

#if defined(__aarch64__) && defined(__ARM_NEON)
    /**
     * @brief Native noncollective FP32 row leaf with the per-row FP32-mean inverse.
     * @param src Original input row; @param gamma Original immutable weights or identity.
     * @param dst Original output row; @param cols Logical elements; @param epsilon Original policy.
     * @details The caller owns row worksharing. This leaf creates no team or decoded row bank.
     */
    void rmsnorm_fused_row_neon(const float *src,const float *gamma,float *dst,
                               std::size_t cols,float epsilon);
#endif

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
     * @brief Fused residual-add + RMSNorm for a single row (AVX-512)
     *
     * Computes in two passes over L1-resident data:
     *   residual[i] += input[i]   (in-place update)
     *   output[i] = gamma[i] * residual[i] / rms(residual)
     *
     * Optimized for M=1 decode where hidden_dim fits in L1 cache.
     * Eliminates separate kernel dispatch and OMP overhead vs two calls.
     *
     * @param input   Input values to add to residual [cols]
     * @param residual Residual stream, updated in-place [cols]
     * @param gamma   RMSNorm scale weights [cols]
     * @param output  Normalized output [cols]
     * @param cols    Hidden dimension
     * @param epsilon Numerical stability epsilon
     */
    void fused_residual_rmsnorm_row_avx512(
        const float *input,
        float *residual,
        const float *gamma,
        float *output,
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
     * @param row_sumsq Disjoint caller-owned per-row FP64 summaries (length = rows).
     * @param opts Native execution or explicit increasing-column FP64-FMA scalar intent.
     * @throws std::invalid_argument For malformed/wrapping geometry, original bank/options
     * intersections, foreign/uncompiled ISA or undeclared T5 accumulation intent.
     * @details Original integers widen directly to FP64 before squares. Native
     * per-ISA trees need not equal each other, but M/worker ownership cannot
     * change one row's program. All retained members call this collective;
     * valid empty calls complete their team without allocating a standalone one.
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
     * @param opts Native ISA or explicit scalar diagnostic; retained calls are collective.
     * @throws std::invalid_argument For malformed/wrapping geometry, aliases with
     * original source/gamma/inverses/policy, or undeclared T5/native ISA intent.
     * @details Two original-storage passes retain (float(x)*inv)*gamma FP32
     * edges. Scale is A>0?A/127:1, without a foreign quantizer floor. Native
     * complete8/16-coordinate fragments use RNE; scalar tails use ties-away.
     * Inverse0 publishes zero codes/scale1 without reading gamma values.
     * Nonfinite/unrepresentable arithmetic is fatal before that row's stores.
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
     * @param opts Native program or explicit scalar diagnostic; retained calls are collective.
     * @throws std::invalid_argument For malformed geometry, nonfinite/negative
     * epsilon, original source/gamma/policy intersections or undeclared T5/ISA.
     * @details One tile owner retains summary, staged FP64-mean inverse, native
     * maximum and code/scale publication. ARM shares1..4 original summaries;
     * no decoded row, scratch resizing, TLS inverse bank or complete M1 replay
     * is used. Empty retained calls still complete their team.
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
    // Q8_1 Pure-Integer RMSNorm
    // ========================================================================

    /**
     * @brief Pure-integer Q8_1 RMSNorm
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
     * @param gamma_q10 Pre-quantized gamma weights [cols] in Q10 format (gamma * 1024)
     * @param output Output Q8_1 blocks (blocks_per_row blocks)
     * @param blocks_per_row Number of Q8_1 blocks per row (cols / 32)
     * @param epsilon_scaled Epsilon pre-scaled (typically 1-100 for stability)
     */
    void rmsnorm_q8_1_pure_integer_row(
        const Q8_1Block *input,
        const int16_t *gamma_q10,
        Q8_1Block *output,
        std::size_t blocks_per_row,
        int32_t epsilon_scaled);

    /**
     * @brief Pure-integer Q8_1 RMSNorm for multiple rows (parallelized)
     *
     * Wrapper that handles gamma quantization and row dispatch.
     * Converts FP32 gamma to Q10 format internally (10-bit fractional precision).
     *
     * @param input Input Q8_1 blocks [rows * blocks_per_row]
     * @param gamma Gamma weights [cols] (FP32, will be quantized to Q10)
     * @param output Output Q8_1 blocks [rows * blocks_per_row]
     * @param rows Number of rows
     * @param blocks_per_row Number of Q8_1 blocks per row (cols / 32)
     * @param epsilon Epsilon for numerical stability
     * @param opts Execution options
     */
    void rmsnorm_q8_1_pure_integer(
        const Q8_1Block *input,
        const float *gamma,
        Q8_1Block *output,
        std::size_t rows,
        std::size_t blocks_per_row,
        float epsilon,
        const RMSNormExecOptions &opts = {});

    // ========================================================================
    // Q16_1 RMSNorm (Q16_1 input → FP32 output)
    // ========================================================================

    /**
     * @brief RMSNorm with Q16_1 input, FP32 output (scalar per-row)
     *
     * Algorithm:
     * 1. Dequantize Q16_1 block → FP32 in registers
     * 2. Compute sum of squares (for RMS)
     * 3. Apply normalization and gamma scaling
     * 4. Write FP32 output
     *
     * Two original-block traversals: FP32 squares widen to a linear FP64
     * diagnostic summary; publication retains the FP32 mean/sqrt/inverse program.
     * Complete32-value geometry and disjoint expansion/gamma banks are required.
     *
     * @param input Input Q16_1 blocks [blocks_per_row blocks]
     * @param gamma Gamma weights [cols] (FP32)
     * @param output Output FP32 row [cols]
     * @param cols Number of columns (hidden_dim)
     * @param epsilon Epsilon for numerical stability
     * @throws std::invalid_argument For malformed complete-row geometry, policy or aliases.
     * @throws std::domain_error For nonfinite arithmetic or an undefined inverse.
     */
    void rmsnorm_q16_1_fp32_row_scalar(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon);

    /**
     * @brief RMSNorm with Q16_1 input, FP32 output (AVX512 per-row)
     *
     * Uses the actual AVX512 program and original-block register decoding.
     * A non-AVX512 compiler envelope rejects this named entry; it never
     * substitutes a scalar or another ISA implementation.
     *
     * @param input Input Q16_1 blocks [blocks_per_row blocks]
     * @param gamma Gamma weights [cols] (FP32)
     * @param output Output FP32 row [cols]
     * @param cols Number of columns (hidden_dim)
     * @param epsilon Epsilon for numerical stability
     * @throws std::invalid_argument For malformed complete-row geometry, policy or aliases.
     * @throws std::domain_error For nonfinite arithmetic or an undefined inverse.
     * @throws std::runtime_error If the compiler envelope does not contain AVX512.
     */
    void rmsnorm_q16_1_fp32_row_avx512(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon);

    /**
     * @brief Fused Q16_1 RMSNorm for multiple rows (parallelized)
     *
     * Native ARM processes1..4 original compressed rows per shared-gamma tile;
     * AVX2/AVX512 select their genuine arithmetic programs. Scalar execution is
     * explicitly requested diagnostic/backend intent, never an ISA substitute.
     * Complete source/output/gamma/options spans are checked before worksharing.
     * Retained teams complete even for small or empty invocations. No decoded
     * row, invocation workspace or persistent scale/gamma bank is constructed.
     * T5 FP32 accumulation is a distinct program and is not admitted here.
     *
     * @param input Input Q16_1 blocks [rows * blocks_per_row]
     * @param gamma Gamma weights [cols] (FP32)
     * @param output Output FP32 tensor [rows * cols]
     * @param rows Number of rows
     * @param cols Number of columns (hidden_dim, must be multiple of 32)
     * @param epsilon Epsilon for numerical stability
     * @param opts Original immutable execution options; retained by the caller through completion.
     * @throws std::invalid_argument For malformed geometry, policy or original-bank/metadata aliases.
     * @throws std::runtime_error For an active ISA without a matching compiled implementation.
     * @note Arithmetic corruption inside the collective is fatal: exceptions
     * cannot cross an OpenMP parallel boundary. No zero-output repair is used.
     */
    void rmsnorm_q16_1_fp32_fused(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts = {});

} // namespace llaminar2::primitives
