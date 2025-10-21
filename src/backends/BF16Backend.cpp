/**
 * @file BF16Backend.cpp
 * @brief Implementation of BF16 matrix multiplication backend
 *
 * @author David Sanftenberg
 * @date October 20, 2025
 */

#include "BF16Backend.h"
#include "utils/DebugEnv.h"
#include "utils/CpuFeatures.h"
#include "Logger.h"

#include <cblas.h>
#include <cstring>
#include <algorithm>

#ifdef HAVE_MKL
#include <mkl.h>
#endif

namespace llaminar
{

    // Static member initialization
    BF16BackendType BF16Backend::backend_type_ = BF16BackendType::NONE;
    bool BF16Backend::initialized_ = false;
    bool BF16Backend::has_hardware_bf16_ = false;

    void BF16Backend::initialize()
    {
        if (initialized_)
            return;

        const auto &env = debugEnv();

        // Detect hardware BF16 support
        has_hardware_bf16_ = detect_hardware_bf16();

        // Check for forced FP32 fallback
        if (env.quant.force_fp32_bf16_gemm)
        {
            LOG_INFO("BF16Backend: FP32 fallback forced via LLAMINAR_FORCE_FP32_BF16_GEMM");
            backend_type_ = BF16BackendType::FP32_FALLBACK;
            initialized_ = true;
            return;
        }

#ifdef HAVE_MKL
        // Try Intel MKL first (unless disabled)
        if (env.quant.bf16_prefer_mkl != 0)
        {
            backend_type_ = BF16BackendType::INTEL_MKL;
            LOG_INFO("BF16Backend: Using Intel MKL (HW-accelerated: "
                     << (has_hardware_bf16_ ? "YES" : "NO, software emulation") << ")");
            initialized_ = true;
            return;
        }
        else
        {
            LOG_DEBUG("BF16Backend: Intel MKL disabled via LLAMINAR_QUANT_BF16_PREFER_MKL=0");
        }
#endif

        // Try OpenBLAS cblas_sbgemm
        // OpenBLAS v0.3.26+ provides cblas_sbgemm (verified working on Cascade Lake)
        backend_type_ = BF16BackendType::OPENBLAS;
        LOG_INFO("BF16Backend: Using OpenBLAS cblas_sbgemm (software BF16 emulation)");
        initialized_ = true;
    }

    BF16BackendType BF16Backend::get_backend_type()
    {
        if (!initialized_)
            initialize();
        return backend_type_;
    }

    std::string BF16Backend::get_backend_name()
    {
        if (!initialized_)
            initialize();

        switch (backend_type_)
        {
        case BF16BackendType::INTEL_MKL:
            return has_hardware_bf16_ ? "Intel MKL (HW-accelerated)" : "Intel MKL (SW emulation)";
        case BF16BackendType::OPENBLAS:
            return "OpenBLAS (SW emulation)";
        case BF16BackendType::FP32_FALLBACK:
            return "FP32 Fallback";
        default:
            return "Unknown";
        }
    }

    bool BF16Backend::has_hardware_bf16()
    {
        if (!initialized_)
            initialize();
        return has_hardware_bf16_;
    }

    bool BF16Backend::is_native_bf16_supported()
    {
        if (!initialized_)
            initialize();
        return backend_type_ == BF16BackendType::INTEL_MKL ||
               backend_type_ == BF16BackendType::OPENBLAS;
    }

    bool BF16Backend::detect_hardware_bf16()
    {
        // Check for Intel AMX BF16 support (Ice Lake+ server CPUs)
        // AMX requires both AMX-BF16 and AMX-TILE CPUID flags
        return CpuFeatures::instance().has_amx_bf16();
    }

    bool BF16Backend::multiply_bf16_to_fp32(
        char transa, char transb,
        int m, int n, int k,
        float alpha,
        const bfloat16 *A, int lda,
        const bfloat16 *B, int ldb,
        float beta,
        float *C, int ldc)
    {
        if (!initialized_)
            initialize();

        switch (backend_type_)
        {
        case BF16BackendType::INTEL_MKL:
            return multiply_mkl_bf16_to_fp32(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        case BF16BackendType::OPENBLAS:
            return multiply_openblas_bf16_to_fp32(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        case BF16BackendType::FP32_FALLBACK:
        default:
            return multiply_fp32_fallback(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }

    bool BF16Backend::multiply_bf16_to_bf16(
        char transa, char transb,
        int m, int n, int k,
        float alpha,
        const bfloat16 *A, int lda,
        const bfloat16 *B, int ldb,
        float beta,
        bfloat16 *C, int ldc)
    {
        // Strategy: Use BF16→FP32 GEMM, then convert result to BF16
        // This maintains FP32 accumulation precision while providing BF16 output

        // Allocate temporary FP32 output buffer
        std::vector<float> C_fp32(m * n);

        // If beta != 0, need to convert existing C values to FP32 first
        if (beta != 0.0f)
        {
#pragma omp parallel for
            for (int i = 0; i < m * n; ++i)
            {
                C_fp32[i] = static_cast<float>(C[i]); // Use operator float()
            }
        }

        // Perform BF16×BF16→FP32 GEMM
        bool success = multiply_bf16_to_fp32(
            transa, transb, m, n, k,
            alpha, A, lda, B, ldb,
            beta, C_fp32.data(), n // Leading dimension = n for row-major
        );

        if (!success)
            return false;

// Convert FP32 result to BF16
#pragma omp parallel for
        for (int i = 0; i < m * n; ++i)
        {
            C[i] = bfloat16::from_float(C_fp32[i]); // Use static method
        }

        return true;
    }

    bool BF16Backend::multiply_mkl_bf16_to_fp32(
        char transa, char transb,
        int m, int n, int k,
        float alpha,
        const bfloat16 *A, int lda,
        const bfloat16 *B, int ldb,
        float beta,
        float *C, int ldc)
    {
#ifdef HAVE_MKL
        try
        {
            // Intel MKL uses CBLAS_LAYOUT enum
            CBLAS_LAYOUT layout = CblasRowMajor;
            CBLAS_TRANSPOSE transA = (transa == 'N' || transa == 'n') ? CblasNoTrans : CblasTrans;
            CBLAS_TRANSPOSE transB = (transb == 'N' || transb == 'n') ? CblasNoTrans : CblasTrans;

            // Use cblas_gemm_bf16bf16f32: BF16×BF16→FP32
            cblas_gemm_bf16bf16f32(
                layout, transA, transB,
                m, n, k,
                alpha,
                reinterpret_cast<const MKL_BF16 *>(A), lda,
                reinterpret_cast<const MKL_BF16 *>(B), ldb,
                beta,
                C, ldc);

            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("BF16Backend: MKL GEMM failed: " << e.what());
            return false;
        }
#else
        LOG_ERROR("BF16Backend: MKL not available but multiply_mkl_bf16_to_fp32 called");
        return false;
#endif
    }

    bool BF16Backend::multiply_openblas_bf16_to_fp32(
        char transa, char transb,
        int m, int n, int k,
        float alpha,
        const bfloat16 *A, int lda,
        const bfloat16 *B, int ldb,
        float beta,
        float *C, int ldc)
    {
        try
        {
            // OpenBLAS cblas_sbgemm: BF16×BF16→FP32
            // OpenBLAS bfloat16 is typedef uint16_t, our bfloat16 has uint16_t data member

            CBLAS_LAYOUT layout = CblasRowMajor;
            CBLAS_TRANSPOSE transA = (transa == 'N' || transa == 'n') ? CblasNoTrans : CblasTrans;
            CBLAS_TRANSPOSE transB = (transb == 'N' || transb == 'n') ? CblasNoTrans : CblasTrans;

            // Verify our bfloat16 struct has same memory layout as OpenBLAS bfloat16 (uint16_t)
            static_assert(sizeof(bfloat16) == sizeof(::bfloat16), "bfloat16 size mismatch");

            cblas_sbgemm(
                layout, transA, transB,
                m, n, k,
                alpha,
                reinterpret_cast<const ::bfloat16 *>(A), lda,
                reinterpret_cast<const ::bfloat16 *>(B), ldb,
                beta,
                C, ldc);

            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("BF16Backend: OpenBLAS GEMM failed: " << e.what());
            return false;
        }
    }

    bool BF16Backend::multiply_fp32_fallback(
        char transa, char transb,
        int m, int n, int k,
        float alpha,
        const bfloat16 *A, int lda,
        const bfloat16 *B, int ldb,
        float beta,
        float *C, int ldc)
    {
        try
        {
            // Expand BF16 inputs to FP32
            std::vector<float> A_fp32(m * k);
            std::vector<float> B_fp32(k * n);

#pragma omp parallel for
            for (int i = 0; i < m * k; ++i)
            {
                A_fp32[i] = static_cast<float>(A[i]); // Use operator float()
            }

#pragma omp parallel for
            for (int i = 0; i < k * n; ++i)
            {
                B_fp32[i] = static_cast<float>(B[i]); // Use operator float()
            }

            // Use standard FP32 GEMM
            CBLAS_LAYOUT layout = CblasRowMajor;
            CBLAS_TRANSPOSE transA = (transa == 'N' || transa == 'n') ? CblasNoTrans : CblasTrans;
            CBLAS_TRANSPOSE transB = (transb == 'N' || transb == 'n') ? CblasNoTrans : CblasTrans;

            cblas_sgemm(
                layout, transA, transB,
                m, n, k,
                alpha,
                A_fp32.data(), lda,
                B_fp32.data(), ldb,
                beta,
                C, ldc);

            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("BF16Backend: FP32 fallback GEMM failed: " << e.what());
            return false;
        }
    }

} // namespace llaminar
