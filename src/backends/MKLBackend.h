/**
 * @file MKLBackend.h
 * @brief Intel MKL backend for BF16 GEMM operations (header only)
 * @author David Sanftenberg
 * @date October 19, 2025
 *
 * Forward declarations for MKL BF16 GEMM wrapper.
 * Implementation is in MKLBackend.cpp to avoid header conflicts with OpenBLAS.
 *
 * Requirements:
 *   - Intel oneAPI MKL 2023.0 or later
 *   - CMake build with -DUSE_MKL=ON
 *   - GCC 4.8+ or compatible compiler (Intel compiler NOT required)
 */

#pragma once

#ifdef HAVE_MKL

#include <string>

// Forward declarations only - no MKL headers here to avoid conflicts with OpenBLAS
namespace llaminar
{

    // Forward declaration of bfloat16 (defined in ../utils/BFloat16.h)
    struct bfloat16;

    /**
     * @brief Perform BF16×BF16→FP32 matrix multiplication using Intel MKL
     *
     * Implemented in MKLBackend.cpp to avoid header conflicts with OpenBLAS.
     *
     * @param A Input matrix A (FP32) [m×k or k×m if transposed]
     * @param B_bf16 Input matrix B (BF16) [k×n or n×k if transposed]
     * @param C Output matrix C (FP32) [m×n]
     * @param m Number of rows in op(A) and C
     * @param n Number of columns in op(B) and C
     * @param k Number of columns in op(A) and rows in op(B)
     * @param alpha Scalar multiplier for A*B
     * @param beta Scalar multiplier for C (use 0.0 to overwrite C)
     * @param transpose_A If true, use A^T instead of A
     * @param transpose_B If true, use B^T instead of B
     * @param validate_inputs If true, check for NaN/Inf in inputs (debug only)
     *
     * @return true if operation succeeded, false on error
     */
    bool mkl_multiply_bf16(
        const float *A,
        const bfloat16 *B_bf16,
        float *C,
        int m, int n, int k,
        float alpha = 1.0f,
        float beta = 0.0f,
        bool transpose_A = false,
        bool transpose_B = false,
        bool validate_inputs = false);

    /**
     * @brief Query MKL version information
     * @return Version string (e.g., "Intel(R) oneAPI Math Kernel Library 2024.0")
     */
    std::string mkl_get_version();

} // namespace llaminar

#endif // HAVE_MKL
