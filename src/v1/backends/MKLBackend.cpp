/**
 * @file MKLBackend.cpp
 * @brief Implementation of Intel MKL backend for BF16 GEMM operations
 * @author David Sanftenberg
 * @date October 19, 2025
 *
 * This file is compiled separately to avoid header conflicts between MKL and OpenBLAS.
 * MKL headers must be included ONLY here, never mixed with OpenBLAS headers.
 */

#ifdef HAVE_MKL

#define MKL_INT int
#include <mkl.h>
#include <mkl_cblas.h>

#include <vector>
#include <cstddef>
#include <cmath>
#include <sstream>
#include "../utils/BFloat16.h"
#include "../Logger.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar
{

    /**
     * @brief Perform BF16×BF16→FP32 matrix multiplication using Intel MKL
     *
     * This function is in a separate compilation unit to avoid header conflicts
     * between MKL and OpenBLAS (both define incompatible CBLAS types).
     */
    bool mkl_multiply_bf16(
        const float *A,         // FP32 activations [m×k]
        const bfloat16 *B_bf16, // BF16 weights [k×n]
        float *C,               // FP32 output [m×n]
        int m, int n, int k,
        float alpha,
        float beta,
        bool transpose_A,
        bool transpose_B,
        bool validate_inputs)
    {
        // Input validation
        if (m <= 0 || n <= 0 || k <= 0)
        {
            LOG_ERROR("[MKL] Invalid matrix dimensions: " << m << "×" << n << "×" << k);
            return false;
        }

        if (!A || !B_bf16 || !C)
        {
            LOG_ERROR("[MKL] Null pointer in input/output matrices");
            return false;
        }

        // Optional input validation (debug builds only)
        if (validate_inputs)
        {
            size_t A_size = static_cast<size_t>(m) * k;
            for (size_t i = 0; i < A_size; ++i)
            {
                if (!std::isfinite(A[i]))
                {
                    LOG_ERROR("[MKL] Non-finite value in A at index " << i << ": " << A[i]);
                    return false;
                }
            }
        }

        // Convert FP32 activations (A) to BF16
        // MKL requires both inputs as BF16 (MKL_BF16 type)
        size_t A_size = transpose_A ? static_cast<size_t>(k) * m : static_cast<size_t>(m) * k;
        std::vector<bfloat16> A_bf16(A_size);

        // Parallel conversion using OpenMP (threshold at 32KB for parallelization)
        constexpr size_t PARALLEL_THRESHOLD = 32768 / sizeof(float);

#pragma omp parallel for if (A_size > PARALLEL_THRESHOLD) schedule(static)
        for (size_t i = 0; i < A_size; ++i)
        {
            A_bf16[i] = bfloat16::from_float(A[i]);
        }

        // Cast to MKL's BF16 type (same binary layout as bfloat16)
        const MKL_BF16 *A_mkl = reinterpret_cast<const MKL_BF16 *>(A_bf16.data());
        const MKL_BF16 *B_mkl = reinterpret_cast<const MKL_BF16 *>(B_bf16);

        // Configure operation
        CBLAS_TRANSPOSE transA = transpose_A ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE transB = transpose_B ? CblasTrans : CblasNoTrans;

        // Leading dimensions (stride between rows)
        int lda = transpose_A ? m : k; // Columns in storage layout of A
        int ldb = transpose_B ? k : n; // Columns in storage layout of B
        int ldc = n;                   // Columns in storage layout of C

        try
        {
            // Call MKL BF16 GEMM
            // C = alpha * op(A) * op(B) + beta * C
            cblas_gemm_bf16bf16f32(
                CblasRowMajor, // Row-major storage (C convention)
                transA,        // Transpose operation for A
                transB,        // Transpose operation for B
                m,             // Rows in op(A) and C
                n,             // Columns in op(B) and C
                k,             // Columns in op(A), rows in op(B)
                alpha,         // Scalar for A*B
                A_mkl,         // Input matrix A (BF16)
                lda,           // Leading dimension of A
                B_mkl,         // Input matrix B (BF16)
                ldb,           // Leading dimension of B
                beta,          // Scalar for C
                C,             // Output matrix C (FP32)
                ldc            // Leading dimension of C
            );

            // Optional output validation (debug builds only)
            if (validate_inputs)
            {
                size_t C_size = static_cast<size_t>(m) * n;
                for (size_t i = 0; i < C_size; ++i)
                {
                    if (!std::isfinite(C[i]))
                    {
                        LOG_ERROR("[MKL] Non-finite value in C at index " << i << ": " << C[i]);
                        return false;
                    }
                }
            }

            LOG_DEBUG("[MKL] BF16 GEMM succeeded: " << m << "×" << k
                                                    << " × " << k << "×" << n << " → " << m << "×" << n
                                                    << " (alpha=" << alpha << ", beta=" << beta << ")");
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[MKL] BF16 GEMM exception: " << e.what());
            return false;
        }
        catch (...)
        {
            LOG_ERROR("[MKL] BF16 GEMM unknown exception");
            return false;
        }
    }

    /**
     * @brief Query MKL version information
     */
    std::string mkl_get_version()
    {
        MKLVersion version;
        mkl_get_version(&version);

        std::ostringstream oss;
        oss << version.ProductStatus << " "
            << version.MajorVersion << "."
            << version.MinorVersion << "."
            << version.UpdateVersion
            << " Build " << version.Build;
        return oss.str();
    }

} // namespace llaminar

#endif // HAVE_MKL
