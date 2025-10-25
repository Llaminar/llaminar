/**
 * @file BF16GemmKernel.cpp
 * @brief CPU BF16 GEMM kernel implementation
 *
 * Supports:
 * - Intel MKL cblas_gemm_bf16bf16f32 (native BF16, Ice Lake+)
 * - Fallback: BF16→FP32 expansion + cblas_sgemm (OpenBLAS, older CPUs)
 *
 * @author David Sanftenberg
 */

#include "BF16GemmKernel.h"
#include "../../utils/BFloat16.h"
#include "../../tensors/SIMDHelpers.h"

#ifdef HAVE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

#include <vector>

namespace llaminar2
{

    // Helper: Batch convert FP32 to BF16
    static void fp32_to_bf16(const float *src, uint16_t *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(src[i]);
        }
    }

    // Helper: Batch convert BF16 to FP32
    static void bf16_to_fp32(const uint16_t *src, float *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::bf16_to_fp32(src[i]);
        }
    }

    bool BF16GemmKernel::multiply(
        const float *A, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        if (!weight_tensor_)
        {
            return false;
        }

        const uint16_t *B_bf16 = weight_tensor_->bf16_data();
        auto shape = weight_tensor_->shape();

        // Validate dimensions
        if (shape.size() != 2)
        {
            return false;
        }

        int B_rows = shape[0];
        int B_cols = shape[1];

        // Check dimension compatibility
        if (transpose_B)
        {
            // B is [n, k], transposed to [k, n] for multiplication
            if (B_rows != n || B_cols != k)
            {
                return false;
            }
        }
        else
        {
            // B is [k, n], no transpose
            if (B_rows != k || B_cols != n)
            {
                return false;
            }
        }

#ifdef HAVE_MKL
        // Intel MKL: Use native BF16 GEMM (cblas_gemm_bf16bf16f32)
        // This is hardware-accelerated on Ice Lake (AVX-512 BF16) and newer

        // Convert A from FP32 to BF16
        std::vector<uint16_t> A_bf16(m * k);
        fp32_to_bf16(A, A_bf16.data(), m * k);

        // MKL BF16 GEMM: C (FP32) = alpha * A (BF16) @ B (BF16) + beta * C (FP32)
        // Note: MKL uses MKL_BF16 type, which is compatible with uint16_t representation

        if (transpose_B)
        {
            // B stored as [n, k], we want to compute A @ B^T
            cblas_gemm_bf16bf16f32(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                m, n, k,
                alpha,
                reinterpret_cast<const MKL_BF16 *>(A_bf16.data()), k, // A is [m, k]
                reinterpret_cast<const MKL_BF16 *>(B_bf16), k,        // B is [n, k]
                beta,
                C, n // C is [m, n]
            );
        }
        else
        {
            // B stored as [k, n], we want to compute A @ B
            cblas_gemm_bf16bf16f32(
                CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha,
                reinterpret_cast<const MKL_BF16 *>(A_bf16.data()), k, // A is [m, k]
                reinterpret_cast<const MKL_BF16 *>(B_bf16), n,        // B is [k, n]
                beta,
                C, n // C is [m, n]
            );
        }
#else
        // OpenBLAS or non-MKL: Expand BF16 to FP32 then use standard SGEMM
        // This is slower but works on any CPU

        size_t B_size = static_cast<size_t>(B_rows) * B_cols;
        std::vector<float> B_fp32(B_size);
        bf16_to_fp32(B_bf16, B_fp32.data(), B_size);

        // Standard FP32 GEMM
        if (transpose_B)
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                m, n, k,
                alpha,
                A, k,             // A is [m, k]
                B_fp32.data(), k, // B is [n, k]
                beta,
                C, n // C is [m, n]
            );
        }
        else
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha,
                A, k,             // A is [m, k]
                B_fp32.data(), n, // B is [k, n]
                beta,
                C, n // C is [m, n]
            );
        }
#endif

        return true;
    }

} // namespace llaminar2
