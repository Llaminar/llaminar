/**
 * @file CPUNativeVNNIContributionContract.h
 * @brief SIMD publication contract for CPU-resident GPU-aligned NativeVNNI experts.
 *
 * ExpertOverlay may execute one prepared expert on CPU, CUDA, or ROCm in
 * adjacent epochs. The integer dot products are exact, but the surrounding
 * FP32 scale and asymmetric-minimum program must also retain the same rounding
 * boundaries or placement can change a sampled token. This file expresses the
 * device contribution program with AVX2 and AVX-512 intrinsics while retaining
 * the established fused CPU program for non-expert, backend-native matrices.
 */

#pragma once

#include "kernels/common/MoEProjectionNumericalContract.h"

#include <cstdint>
#include <immintrin.h>

namespace llaminar2::cpu::native_vnni
{
#if defined(__GNUC__) && !defined(__clang__)
    /**
     * Prevent contraction while requiring the per-block primitive to inline.
     *
     * These functions execute once per output vector and Q8 block. An
     * out-of-line call dominates the narrow AVX2 segment primitive, while
     * forcing the same hint into AVX-512's dense multi-row body increases its
     * register pressure. Keep the numerical controls common and make only the
     * AVX2 leaf structurally inline.
     */
#define LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP                                      \
    __attribute__((optimize("fp-contract=off", "no-associative-math")))
#define LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP_AVX2                                 \
    __attribute__((always_inline, optimize("fp-contract=off", "no-associative-math")))
#elif defined(__clang__)
#define LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP
#define LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP_AVX2 __attribute__((always_inline))
#else
#define LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP
#define LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP_AVX2
#endif

    /**
     * @brief Retain one rounded AVX2 FP32 vector without memory traffic.
     *
     * An intrinsic names an instruction but does not, by itself, establish a
     * C++ dependency boundary.  GCC can still contract an immediately consumed
     * `_mm256_mul_ps` with the following `_mm256_add_ps` after the helper is
     * inlined.  The empty register dependency emits no instruction; it merely
     * makes the already-rounded product an observable compiler input/output so
     * the next arithmetic operation cannot absorb it.
     *
     * @param value Eight independently rounded binary32 lanes.
     * @return The identical lanes with the rounding boundary retained.
     */
    inline __m256 persistRoundedAVX2(__m256 value) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : "+x"(value));
        return value;
#else
        /*
         * The project builds with GCC or Clang on x86.  Keep a correctness-
         * first implementation for another compiler instead of silently
         * dropping the numerical contract; this branch is intentionally not
         * part of the tuned production binaries.
         */
        alignas(32) volatile float rounded[8];
        _mm256_store_ps(const_cast<float *>(rounded), value);
        return _mm256_load_ps(const_cast<const float *>(rounded));
#endif
    }

#if defined(__AVX512F__)
    /**
     * @brief Retain one rounded AVX-512 FP32 vector without memory traffic.
     * @param value Sixteen independently rounded binary32 lanes.
     * @return The identical lanes with the rounding boundary retained.
     */
    inline __m512 persistRoundedAVX512(__m512 value) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : "+x"(value));
        return value;
#else
        alignas(64) volatile float rounded[16];
        _mm512_store_ps(const_cast<float *>(rounded), value);
        return _mm512_load_ps(const_cast<const float *>(rounded));
#endif
    }
#endif

    /**
     * @brief Append one symmetric 32-value block to eight AVX2 output lanes.
     * @param accumulator Running FP32 output lanes.
     * @param dot Exact signed INT32 dot products.
     * @param weight_scale FP16-derived weight scales.
     * @param activation_scale FP16-derived activation scale.
     * @param policy Backend-native or placement-invariant arithmetic policy.
     * @return Updated output lanes.
     */
    LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP_AVX2
    inline __m256 accumulateSingleScaleBlockAVX2(
        __m256 accumulator,
        __m256i dot,
        __m256 weight_scale,
        float activation_scale,
        CPUProjectionNumericalPolicy policy) noexcept
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const __m256 activation = _mm256_set1_ps(activation_scale);
        const __m256 dot_fp32 = _mm256_cvtepi32_ps(dot);
        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
        {
            /*
             * DeviceNativeVNNIContributionContract rounds the complete block
             * before it enters the running K accumulator. Keeping the product
             * in a named intrinsic also prevents an AVX compiler from silently
             * recovering the backend-native FMA below.
             */
            const __m256 combined_scale = persistRoundedAVX2(
                _mm256_mul_ps(activation, weight_scale));
            const __m256 contribution = persistRoundedAVX2(
                _mm256_mul_ps(combined_scale, dot_fp32));
            return _mm256_add_ps(accumulator, contribution);
        }
        const __m256 combined_scale =
            _mm256_mul_ps(activation, weight_scale);
        return _mm256_fmadd_ps(dot_fp32, combined_scale, accumulator);
    }

    /**
     * @brief Append one asymmetric 32-value block to eight AVX2 output lanes.
     * @param accumulator Running FP32 output lanes.
     * @param dot Exact signed INT32 dot products.
     * @param weight_scale FP16-derived weight scales.
     * @param activation_sum Exact sum of the signed Q8 activation bytes.
     * @param weight_min FP16-derived additive weight minima.
     * @param activation_scale FP16-derived activation scale.
     * @param policy Backend-native or placement-invariant arithmetic policy.
     * @return Updated output lanes.
     */
    LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP_AVX2
    inline __m256 accumulateCorrectedSingleScaleBlockAVX2(
        __m256 accumulator,
        __m256i dot,
        __m256 weight_scale,
        std::int32_t activation_sum,
        __m256 weight_min,
        float activation_scale,
        CPUProjectionNumericalPolicy policy) noexcept
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const __m256 activation = _mm256_set1_ps(activation_scale);
        const __m256 dot_fp32 = _mm256_cvtepi32_ps(dot);
        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
        {
            const __m256 scaled_weight = persistRoundedAVX2(
                _mm256_mul_ps(activation, weight_scale));
            const __m256 dot_contribution = persistRoundedAVX2(
                _mm256_mul_ps(scaled_weight, dot_fp32));
            const __m256 scaled_min = persistRoundedAVX2(
                _mm256_mul_ps(activation, weight_min));
            const __m256 correction = persistRoundedAVX2(
                _mm256_mul_ps(
                    scaled_min,
                    _mm256_set1_ps(static_cast<float>(activation_sum))));
            const __m256 completed_block = persistRoundedAVX2(
                _mm256_add_ps(dot_contribution, correction));
            return _mm256_add_ps(accumulator, completed_block);
        }

        accumulator = _mm256_fmadd_ps(
            dot_fp32,
            _mm256_mul_ps(activation, weight_scale),
            accumulator);
        const __m256 activation_correction = _mm256_set1_ps(
            static_cast<float>(activation_sum) * activation_scale);
        return _mm256_fmadd_ps(
            activation_correction, weight_min, accumulator);
    }

#if defined(__AVX512F__)
    /**
     * @brief Append one symmetric 32-value block to sixteen AVX-512 lanes.
     * @param accumulator Running FP32 output lanes.
     * @param dot Exact signed INT32 dot products.
     * @param weight_scale FP16-derived weight scales.
     * @param activation_scale FP16-derived activation scale.
     * @param policy Backend-native or placement-invariant arithmetic policy.
     * @return Updated output lanes.
     */
    LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP
    inline __m512 accumulateSingleScaleBlockAVX512(
        __m512 accumulator,
        __m512i dot,
        __m512 weight_scale,
        float activation_scale,
        CPUProjectionNumericalPolicy policy) noexcept
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const __m512 activation = _mm512_set1_ps(activation_scale);
        const __m512 dot_fp32 = _mm512_cvtepi32_ps(dot);
        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
        {
            const __m512 combined_scale = persistRoundedAVX512(
                _mm512_mul_ps(activation, weight_scale));
            const __m512 contribution = persistRoundedAVX512(
                _mm512_mul_ps(combined_scale, dot_fp32));
            return _mm512_add_ps(accumulator, contribution);
        }
        const __m512 combined_scale =
            _mm512_mul_ps(activation, weight_scale);
        return _mm512_fmadd_ps(dot_fp32, combined_scale, accumulator);
    }

    /**
     * @brief Append one asymmetric 32-value block to sixteen AVX-512 lanes.
     * @param accumulator Running FP32 output lanes.
     * @param dot Exact signed INT32 dot products.
     * @param weight_scale FP16-derived weight scales.
     * @param activation_sum Exact sum of the signed Q8 activation bytes.
     * @param weight_min FP16-derived additive weight minima.
     * @param activation_scale FP16-derived activation scale.
     * @param policy Backend-native or placement-invariant arithmetic policy.
     * @return Updated output lanes.
     */
    LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP
    inline __m512 accumulateCorrectedSingleScaleBlockAVX512(
        __m512 accumulator,
        __m512i dot,
        __m512 weight_scale,
        std::int32_t activation_sum,
        __m512 weight_min,
        float activation_scale,
        CPUProjectionNumericalPolicy policy) noexcept
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const __m512 activation = _mm512_set1_ps(activation_scale);
        const __m512 dot_fp32 = _mm512_cvtepi32_ps(dot);
        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
        {
            const __m512 scaled_weight = persistRoundedAVX512(
                _mm512_mul_ps(activation, weight_scale));
            const __m512 dot_contribution = persistRoundedAVX512(
                _mm512_mul_ps(scaled_weight, dot_fp32));
            const __m512 scaled_min = persistRoundedAVX512(
                _mm512_mul_ps(activation, weight_min));
            const __m512 correction = persistRoundedAVX512(
                _mm512_mul_ps(
                    scaled_min,
                    _mm512_set1_ps(static_cast<float>(activation_sum))));
            const __m512 completed_block = persistRoundedAVX512(
                _mm512_add_ps(dot_contribution, correction));
            return _mm512_add_ps(accumulator, completed_block);
        }

        accumulator = _mm512_fmadd_ps(
            dot_fp32,
            _mm512_mul_ps(activation, weight_scale),
            accumulator);
        const __m512 activation_correction = _mm512_set1_ps(
            static_cast<float>(activation_sum) * activation_scale);
        return _mm512_fmadd_ps(
            activation_correction, weight_min, accumulator);
    }
#endif

#undef LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP
#undef LLAMINAR_GPU_ALIGNED_EXPERT_EXACT_FP_AVX2
} // namespace llaminar2::cpu::native_vnni
