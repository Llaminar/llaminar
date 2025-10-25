/**
 * @file BlasWrapper.h
 * @brief Runtime BLAS backend wrapper
 *
 * Provides unified BLAS interface that selects between MKL and OpenBLAS at runtime
 * based on CPU vendor (Intel -> MKL, others -> OpenBLAS).
 *
 * Note: We can't link both MKL and OpenBLAS simultaneously due to symbol conflicts.
 * Instead, we compile separate kernels for each backend and select at runtime.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "CPUFeatures.h"

// Include appropriate BLAS headers based on compile-time flags
#ifdef HAVE_MKL
#include <mkl_cblas.h>
#endif

#ifdef HAVE_OPENBLAS
#include <cblas.h>
#endif

namespace llaminar2
{

    /**
     * @brief Wrapper for cblas_sgemm that uses the appropriate backend
     *
     * This function uses compile-time conditionals to select the best backend.
     * When both MKL and OpenBLAS are available (via separate builds), this
     * provides a unified interface.
     */
    inline void blas_sgemm(
        CBLAS_LAYOUT layout,
        CBLAS_TRANSPOSE trans_a,
        CBLAS_TRANSPOSE trans_b,
        int m, int n, int k,
        float alpha,
        const float *A, int lda,
        const float *B, int ldb,
        float beta,
        float *C, int ldc)
    {
#if defined(HAVE_MKL)
        // Use MKL implementation
        cblas_sgemm(layout, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#elif defined(HAVE_OPENBLAS)
        // Use OpenBLAS implementation
        cblas_sgemm(layout, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
#error "At least one BLAS backend (HAVE_MKL or HAVE_OPENBLAS) must be enabled"
#endif
    }

    /**
     * @brief Get the name of the active BLAS backend
     */
    inline const char *blas_backend_name()
    {
#if defined(HAVE_MKL)
        return "Intel MKL";
#elif defined(HAVE_OPENBLAS)
        return "OpenBLAS";
#else
        return "None";
#endif
    }

    /**
     * @brief Check if MKL is the active backend
     */
    inline bool blas_is_mkl()
    {
#ifdef HAVE_MKL
        return true;
#else
        return false;
#endif
    }

} // namespace llaminar2
