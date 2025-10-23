/**
 * @file QuantizedGemm.h
 * @brief Generic quantized GEMM kernel for block-quantized weight tensors
 *
 * Supports any block-quantized format (IQ4_NL, Q6_K, Q8_0, etc.) via IBlockDecoder strategy.
 * Reuses optimized tiling/threading logic across all quantized formats.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/SIMDHelpers.h"
#include <algorithm>
#include <vector>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    /**
     * @brief Quantized GEMM kernel: A (FP32) × B (quantized) → C (FP32)
     *
     * Implements fused dequant+GEMM with adaptive tiling for optimal performance.
     * Decoder strategy provides format-specific block dequantization (IQ4_NL, Q6_K, etc.).
     *
     * Performance characteristics:
     * - Cache-blocked for small batches (m ∈ [2,16])
     * - Row-wise with adaptive tiling for large batches (m > 16)
     * - SIMD-optimized dot products (AVX512/AVX2/scalar fallback)
     * - Empirically tuned tile sizes (64×32 optimal for most workloads)
     *
     * Target: 335-451 GFLOPS on modern CPUs (competitive with FP32 BLAS)
     */
    class QuantizedGemmKernel : public ITensorGemm
    {
    public:
        /**
         * @brief Construct quantized GEMM kernel
         *
         * @param decoder Block decoder strategy (typically the quantized tensor itself)
         */
        explicit QuantizedGemmKernel(const IBlockDecoder *decoder)
            : decoder_(decoder) {}

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only for now
        }

        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

    private:
        const IBlockDecoder *decoder_;

        // SIMD-optimized dot product
        static inline float dot_product_simd(const float *a, const float *b, size_t count);

        // Helper methods
        bool multiply_cache_blocked(const float *A, float *C, int m, int n, int k, float alpha, float beta);
        bool multiply_row_wise(const float *A, float *C, int m, int n, int k, float alpha, float beta);
    };

    // ========== Inline SIMD Helpers ==========

    inline float QuantizedGemmKernel::dot_product_simd(const float *a, const float *b, size_t count)
    {
#if defined(__AVX512F__)
        __m512 sum = _mm512_setzero_ps();

        size_t i = 0;
        for (; i + 16 <= count; i += 16)
        {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_load_ps(b + i); // Aligned load
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        float result = _mm512_reduce_add_ps(sum);

        // Scalar tail
        for (; i < count; ++i)
        {
            result += a[i] * b[i];
        }

        return result;

#elif defined(__AVX2__)
        __m256 sum = _mm256_setzero_ps();

        size_t i = 0;
        for (; i + 8 <= count; i += 8)
        {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum
        __m128 sum_high = _mm256_extractf128_ps(sum, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(sum_low, sum_high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);

        // Scalar tail
        for (; i < count; ++i)
        {
            result += a[i] * b[i];
        }

        return result;

#else
        // Scalar fallback
        float result = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            result += a[i] * b[i];
        }
        return result;
#endif
    }

} // namespace llaminar2
