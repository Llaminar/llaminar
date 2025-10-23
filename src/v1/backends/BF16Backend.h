/**
 * @file BF16Backend.h
 * @brief BF16 matrix multiplication backend for Phase 5 activation storage
 *
 * Provides optimized BF16×BF16 GEMM operations using OpenBLAS or Intel MKL backends.
 * Supports two output modes:
 *   1. BF16×BF16→FP32: Activations in BF16, output expanded to FP32 (memory bandwidth optimized)
 *   2. BF16×BF16→BF16: Full BF16 path with FP32 accumulation (maximum memory reduction)
 *
 * Backend selection priority:
 *   1. Intel MKL (hardware-accelerated on Ice Lake+, software emulation on older CPUs)
 *   2. OpenBLAS v0.3.26+ (software BF16 emulation, verified working)
 *   3. FP32 expansion fallback (if BF16 GEMM unavailable)
 *
 * @author David Sanftenberg
 * @date October 20, 2025
 */

#pragma once

#include <cstddef>
#include <string>
#include "../utils/BFloat16.h"

namespace llaminar
{

    /**
     * @enum BF16BackendType
     * @brief Available BF16 GEMM backend implementations
     */
    enum class BF16BackendType
    {
        NONE = 0,         ///< No BF16 backend available (use FP32 fallback)
        OPENBLAS = 1,     ///< OpenBLAS cblas_sbgemm (software emulation)
        INTEL_MKL = 2,    ///< Intel MKL cblas_gemm_bf16bf16f32 (HW-accelerated on Ice Lake+)
        FP32_FALLBACK = 3 ///< Expand to FP32 and use standard GEMM
    };

    /**
     * @class BF16Backend
     * @brief Static class providing BF16 matrix multiplication operations
     *
     * This backend handles BF16×BF16 GEMM operations with two output modes:
     *
     * **Mode 1: BF16×BF16→FP32 (multiply_bf16_to_fp32)**
     *   - Input A: BF16 activations (m×k)
     *   - Input B: BF16 weights (k×n)
     *   - Output C: FP32 result (m×n)
     *   - Use case: Intermediate activations where FP32 precision needed downstream
     *   - Memory: 50% bandwidth reduction on inputs, FP32 output
     *
     * **Mode 2: BF16×BF16→BF16 (multiply_bf16_to_bf16)**
     *   - Input A: BF16 activations (m×k)
     *   - Input B: BF16 weights (k×n)
     *   - Output C: BF16 result (m×n) with FP32 accumulation
     *   - Use case: Maximum memory reduction, can tolerate BF16 precision
     *   - Memory: 50% bandwidth reduction on inputs AND outputs
     *
     * **Backend Selection:**
     * - Intel MKL (if available): Default, hardware-accelerated on Ice Lake+
     * - OpenBLAS v0.3.26+: Fallback, software emulation (verified working)
     * - FP32 expansion: Final fallback if BF16 GEMM unavailable
     *
     * **Environment Variables:**
     * - `LLAMINAR_QUANT_BF16_PREFER_MKL=0`: Force OpenBLAS even if MKL available
     * - `LLAMINAR_FORCE_FP32_BF16_GEMM=1`: Disable BF16 GEMM, use FP32 expansion
     *
     * **Threading:**
     * - Inherits OpenMP thread count from caller
     * - For small ops (<8K elements), single-threaded may be faster
     */
    class BF16Backend
    {
    public:
        /**
         * @brief Initialize BF16 backend and detect available implementations
         *
         * Detects and initializes the best available BF16 GEMM backend:
         * 1. Intel MKL (if HAVE_MKL defined and not disabled by env var)
         * 2. OpenBLAS (if cblas_sbgemm available)
         * 3. FP32 fallback (always available)
         *
         * Should be called once at startup (thread-safe after initialization).
         */
        static void initialize();

        /**
         * @brief Get the currently active BF16 backend type
         *
         * @return BF16BackendType indicating which backend is in use
         */
        static BF16BackendType get_backend_type();

        /**
         * @brief Get human-readable name of active backend
         *
         * @return std::string like "Intel MKL", "OpenBLAS", "FP32 Fallback"
         */
        static std::string get_backend_name();

        /**
         * @brief Check if hardware-accelerated BF16 is available (Ice Lake+ AMX)
         *
         * @return true if running on Ice Lake+ with AMX BF16 instructions, false otherwise
         */
        static bool has_hardware_bf16();

        /**
         * @brief BF16×BF16 matrix multiplication with FP32 output
         *
         * Computes: C = alpha * A * B + beta * C
         *
         * @param transa 'N' or 'T' - transpose flag for A
         * @param transb 'N' or 'T' - transpose flag for B
         * @param m Number of rows in op(A) and C
         * @param n Number of columns in op(B) and C
         * @param k Number of columns in op(A) / rows in op(B)
         * @param alpha Scalar multiplier for A*B
         * @param A Input matrix A in BF16 format (row-major)
         * @param lda Leading dimension of A
         * @param B Input matrix B in BF16 format (row-major)
         * @param ldb Leading dimension of B
         * @param beta Scalar multiplier for C (use 0.0 to ignore existing C values)
         * @param C Output matrix C in FP32 format (row-major)
         * @param ldc Leading dimension of C
         *
         * @return true on success, false if operation failed
         *
         * @note This is the primary function for BF16 activations with FP32 precision downstream.
         *       Uses FP32 accumulation internally for numerical stability.
         */
        static bool multiply_bf16_to_fp32(
            char transa, char transb,
            int m, int n, int k,
            float alpha,
            const bfloat16 *A, int lda,
            const bfloat16 *B, int ldb,
            float beta,
            float *C, int ldc);

        /**
         * @brief BF16×BF16 matrix multiplication with BF16 output
         *
         * Computes: C = alpha * A * B + beta * C (with FP32 accumulation, result converted to BF16)
         *
         * @param transa 'N' or 'T' - transpose flag for A
         * @param transb 'N' or 'T' - transpose flag for B
         * @param m Number of rows in op(A) and C
         * @param n Number of columns in op(B) and C
         * @param k Number of columns in op(A) / rows in op(B)
         * @param alpha Scalar multiplier for A*B
         * @param A Input matrix A in BF16 format (row-major)
         * @param lda Leading dimension of A
         * @param B Input matrix B in BF16 format (row-major)
         * @param ldb Leading dimension of B
         * @param beta Scalar multiplier for C (use 0.0 to ignore existing C values)
         * @param C Output matrix C in BF16 format (row-major)
         * @param ldc Leading dimension of C
         *
         * @return true on success, false if operation failed
         *
         * @note Maximum memory reduction mode. Uses FP32 accumulation internally,
         *       then converts result to BF16. Use when downstream operations can
         *       tolerate BF16 precision (e.g., attention context, FFN intermediates).
         */
        static bool multiply_bf16_to_bf16(
            char transa, char transb,
            int m, int n, int k,
            float alpha,
            const bfloat16 *A, int lda,
            const bfloat16 *B, int ldb,
            float beta,
            bfloat16 *C, int ldc);

        /**
         * @brief Check if BF16 GEMM is supported (not just FP32 fallback)
         *
         * @return true if OpenBLAS or MKL BF16 GEMM available, false if FP32 fallback only
         */
        static bool is_native_bf16_supported();

    private:
        static BF16BackendType backend_type_;
        static bool initialized_;
        static bool has_hardware_bf16_;

        // Helper functions for specific backends
        static bool multiply_mkl_bf16_to_fp32(
            char transa, char transb, int m, int n, int k,
            float alpha, const bfloat16 *A, int lda,
            const bfloat16 *B, int ldb,
            float beta, float *C, int ldc);

        static bool multiply_openblas_bf16_to_fp32(
            char transa, char transb, int m, int n, int k,
            float alpha, const bfloat16 *A, int lda,
            const bfloat16 *B, int ldb,
            float beta, float *C, int ldc);

        static bool multiply_fp32_fallback(
            char transa, char transb, int m, int n, int k,
            float alpha, const bfloat16 *A, int lda,
            const bfloat16 *B, int ldb,
            float beta, float *C, int ldc);

        // Detect hardware BF16 support (AMX on Ice Lake+)
        static bool detect_hardware_bf16();
    };

} // namespace llaminar
