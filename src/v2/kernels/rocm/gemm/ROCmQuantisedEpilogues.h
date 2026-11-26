/**
 * @file ROCmQuantisedEpilogues.h
 * @brief Canonical count-free and count-owned FP32 projection epilogue ABI.
 *
 * The retained ordinary ABI executes a compile-time count-free specialization.
 * Explicit counted callers copy one immutable row descriptor into kernel args;
 * its device owner supplies the live prefix on the exact execution stream.
 * Inactive rows are rejected before output reads, bias reads or output writes.
 * Both specializations retain the original multiply-then-bias arithmetic.
 */
#pragma once
#include "kernels/common/DeviceRowRange.h"
#include <cstddef>

namespace llaminar2::rocm
{
    /**
     * @brief A terminal FP32 store owns the prior value of its own output lane.
     * The multipliers and optional bias are sealed before capture. Split-K
     * producers retain their raw partials and the ordered reducer owns this fold.
     */
    struct ROCmFP32DestinationFold
    {
        const float *bias = nullptr; ///< Optional original bias by column.
        float alpha = 1.0f;           ///< Raw projection multiplier.
        float beta = 0.0f;            ///< Prior destination multiplier.

        /** @return Whether capture can select the existing raw FP32 destination leaf. */
        bool isIdentity() const noexcept { return alpha == 1.0f && beta == 0.0f && bias == nullptr; }

#if defined(__HIPCC__)
        /**
         * @brief Combine a completed projection with the original output lane.
         * @param value Completed FP32 product under its original arithmetic.
         * @param prior Previous output bank, read only for a nonzero beta.
         * @param index Physical row-major index owned by this lane.
         * @param column Physical column in the original bias plane.
         * @return The value that must be published at the final destination.
         */
        __device__ __forceinline__ float apply(
            float value, const float *prior, std::size_t index, int column) const
        {
            if (alpha == 1.0f && beta == 0.0f && bias == nullptr)
                return value;
            return applyUpdate(value, prior, index, column);
        }

        /**
         * @brief Fold a capture-admitted nonidentity store using the original FP32 arithmetic.
         * @param value Completed original product. @param prior Output lane storage.
         * @param index Exact row-major destination. @param column Original bias column.
         * @return Updated final value; only a nonzero beta reads prior storage.
         */
        __device__ __forceinline__ float applyUpdate(
            float value, const float *prior, std::size_t index, int column) const
        {
            float result = alpha * value;
            if (beta != 0.0f)
                result += beta * prior[index];
            if (bias)
                result += bias[column];
            return result;
        }
#endif
    };
}

extern "C" {
/**
 * @brief Apply the original full-active FP32 epilogue.
 * @param output Borrowed row-major persistent FP32 destination [M,N].
 * @param bias Optional borrowed FP32 bias [N].
 * @param M Positive retained physical row count.
 * @param N Positive output width; M*N must fit the native INT32 index program.
 * @param alpha Original output multiplier, evaluated before optional bias.
 * @param device Exact owning ROCm device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @return Original native submission result, with no alternate arithmetic route.
 * @throws std::invalid_argument For invalid immutable geometry or stream metadata.
 */
bool rocmQuantGemm_applyFp32Epilogue(float* output,const float* bias,int M,int N,
    float alpha,int device,void* stream);
/**
 * @brief Apply the same arithmetic to device-count-owned live rows only.
 * @param output Borrowed row-major persistent FP32 destination [M,N].
 * @param bias Optional borrowed FP32 bias [N].
 * @param M Positive retained physical row count, never a downloaded live count.
 * @param N Positive output width; M*N must fit the native INT32 index program.
 * @param alpha Original output multiplier, evaluated before optional bias.
 * @param device Exact owning ROCm device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @param rows Optional immutable descriptor with exactly M physical rows.
 * @return Native submission result. Null/count-free rows retain ordinary specialization.
 * @throws std::invalid_argument For invalid immutable geometry, stream or row metadata.
 */
bool rocmQuantGemm_applyFp32Epilogue_rows(float* output,const float* bias,int M,int N,
    float alpha,int device,void* stream,const llaminar2::DeviceRowRange* rows);
/**
 * @brief Combine a persistent computed matrix with its original output rows.
 * @param computed Borrowed computed FP32 matrix [M,N], distinct from existing for beta.
 * @param output Final destination [M,N], which may alias computed or existing.
 * @param existing Original output [M,N], required when beta is nonzero.
 * @param bias Optional original column bias [N], added after the beta contribution.
 * @param M Positive physical row extent; never a downloaded live count.
 * @param N Positive output width; M*N must fit the native INT32 index program.
 * @param alpha Computed-result multiplier, preceding beta and bias arithmetic.
 * @param beta Existing-output multiplier; zero does not read existing.
 * @param device Exact owning ROCm device ordinal.
 * @param stream Exact non-default stream ordered after the computed producer.
 * @param rows Optional original prefix authority, also inherited by explicit request work.
 * @return Native submission result; inactive rows have no reads or writes.
 * @throws std::invalid_argument For invalid geometry, stream or prefix extent.
 */
bool rocmQuantGemm_applyFp32CombineEpilogue(const float* computed,float* output,
    const float* existing,const float* bias,int M,int N,float alpha,float beta,
    int device,void* stream,const llaminar2::DeviceRowRange* rows=nullptr);
}
