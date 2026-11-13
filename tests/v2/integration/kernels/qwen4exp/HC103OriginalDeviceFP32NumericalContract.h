/**
 * @file HC103OriginalDeviceFP32NumericalContract.h
 * @brief Cross-backend explicit-rounding primitives for GPU-aligned expert arithmetic.
 *
 * CUDA and HIP expose compatible round-to-nearest intrinsics, but the HIP
 * optimizer may still reassociate a completed result with its consumer. CPU
 * experts must execute the same program when ExpertOverlay moves a logical
 * expert between host and device tiers. These primitives therefore make every
 * published binary32 dependency edge explicit on all three backends. CUDA
 * primitives also retain IEEE subnormal operands and results explicitly: the
 * translation unit's fast-math FTZ policy cannot override this byte contract.
 */

#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HC103_ORIGINAL_LLAMINAR_FP32_CONTRACT_INLINE __device__ __forceinline__
#else
#define HC103_ORIGINAL_LLAMINAR_FP32_CONTRACT_INLINE inline
#endif

namespace llaminar2::test103_original_fp32
{
    /**
     * @brief Reinterpret one IEEE-754 binary32 word without a library call.
     * @param bits Exact binary representation.
     * @return Floating-point value carrying exactly @p bits.
     */
    HC103_ORIGINAL_LLAMINAR_FP32_CONTRACT_INLINE float floatFromBits(
        std::uint32_t bits) noexcept
    {
        union Word
        {
            std::uint32_t bits;
            float value;
        } word{};
        word.bits = bits;
        return word.value;
    }

    /**
     * @brief Reinterpret one binary32 value as its exact IEEE-754 word.
     * @param value Floating-point value to inspect.
     * @return Exact binary representation of @p value.
     */
    HC103_ORIGINAL_LLAMINAR_FP32_CONTRACT_INLINE std::uint32_t bitsFromFloat(
        float value) noexcept
    {
        union Word
        {
            float value;
            std::uint32_t bits;
        } word{};
        word.value = value;
        return word.bits;
    }

    /**
     * @brief Preserve one explicitly rounded binary32 intermediate.
     *
     * The empty GPU/CPU register dependency is a compiler boundary only; it
     * emits no memory traffic or synchronization. CUDA intrinsics already
     * retain this boundary, so CUDA returns the word directly. On x86 SSE,
     * scalar arithmetic has already rounded to binary32 in its XMM register;
     * making the value an opaque register output prevents the next operation
     * from contracting it without forcing every Newton step through memory.
     *
     * @param value Result of one explicit binary32 arithmetic operation.
     * @return The identical binary32 word with its dependency edge retained.
     */
    HC103_ORIGINAL_LLAMINAR_FP32_CONTRACT_INLINE float persistRounded(float value) noexcept
    {
#if defined(__HIP_DEVICE_COMPILE__)
        asm volatile("" : "+v"(value));
#elif !defined(__CUDA_ARCH__) && defined(__SSE2__) && \
    (defined(__GNUC__) || defined(__clang__))
        asm volatile("" : "+x"(value));
#elif !defined(__CUDA_ARCH__)
        /*
         * Hosts without the x86 register constraint still require an explicit
         * binary32 storage boundary (in particular, extended x87 arithmetic).
         * This is the same arithmetic contract, not a different approximation.
         */
        volatile float rounded = value;
        return rounded;
#endif
        return value;
    }

    /**
     * @brief Multiply two values with one retained round-to-nearest boundary.
     * @details CUDA explicitly retains subnormal inputs and results under the
     * translation unit fast-math policy; HIP and CPU keep their existing RNE paths.
     * @param lhs Left operand.
     * @param rhs Right operand.
     * @return Explicitly rounded binary32 product.
     */
    HC103_ORIGINAL_LLAMINAR_FP32_CONTRACT_INLINE float multiply(
        float lhs,
        float rhs) noexcept
    {
#if defined(__CUDA_ARCH__)
        // CUDA's --ftz=true also changes the rounded intrinsic to a flushing
        // instruction. The explicit RNE PTX form deliberately omits .ftz,
        // retaining subnormal inputs/results and forbidding contraction.
        float value;
        asm("mul.rn.f32 %0, %1, %2;"
            : "=f"(value) : "f"(lhs), "f"(rhs));
        return persistRounded(value);
#elif defined(__HIP_DEVICE_COMPILE__)
        return persistRounded(__fmul_rn(lhs, rhs));
#else
        return persistRounded(lhs * rhs);
#endif
    }

    /**
     * @brief Add two values with one retained round-to-nearest boundary.
     * @details CUDA explicitly retains subnormal inputs and results under the
     * translation unit fast-math policy; HIP and CPU keep their existing RNE paths.
     * @param lhs Left operand.
     * @param rhs Right operand.
     * @return Explicitly rounded binary32 sum.
     */
    HC103_ORIGINAL_LLAMINAR_FP32_CONTRACT_INLINE float add(
        float lhs,
        float rhs) noexcept
    {
#if defined(__CUDA_ARCH__)
        // CUDA's --ftz=true also changes the rounded intrinsic to a flushing
        // instruction. The explicit RNE PTX form deliberately omits .ftz,
        // retaining subnormal inputs/results and forbidding contraction.
        float value;
        asm("add.rn.f32 %0, %1, %2;"
            : "=f"(value) : "f"(lhs), "f"(rhs));
        return persistRounded(value);
#elif defined(__HIP_DEVICE_COMPILE__)
        return persistRounded(__fadd_rn(lhs, rhs));
#else
        return persistRounded(lhs + rhs);
#endif
    }

}
#undef HC103_ORIGINAL_LLAMINAR_FP32_CONTRACT_INLINE
