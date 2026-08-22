/**
 * @file DeviceFP32NumericalContract.h
 * @brief Cross-backend explicit-rounding primitives for device arithmetic.
 *
 * CUDA and HIP expose compatible round-to-nearest intrinsics, but the HIP
 * optimizer may still reassociate a completed result with its consumer. These
 * primitives make each published binary32 dependency edge explicit so higher
 * level numerical contracts can describe one program for both device types.
 */

#pragma once

#include <cstdint>

namespace llaminar2::device_fp32_contract
{
    /**
     * @brief Reinterpret one IEEE-754 binary32 word without a library call.
     * @param bits Exact binary representation.
     * @return Floating-point value carrying exactly @p bits.
     */
    __device__ __forceinline__ float floatFromBits(
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
    __device__ __forceinline__ std::uint32_t bitsFromFloat(
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
     * The empty VGPR dependency is a compiler boundary only; it emits no
     * memory traffic or synchronization. CUDA intrinsics already retain this
     * boundary, so CUDA returns the word directly.
     *
     * @param value Result of one explicit binary32 arithmetic operation.
     * @return The identical binary32 word with its dependency edge retained.
     */
    __device__ __forceinline__ float persistRounded(float value) noexcept
    {
#if defined(__HIP_DEVICE_COMPILE__)
        asm volatile("" : "+v"(value));
#endif
        return value;
    }

    /**
     * @brief Multiply two values with one retained round-to-nearest boundary.
     * @param lhs Left operand.
     * @param rhs Right operand.
     * @return Explicitly rounded binary32 product.
     */
    __device__ __forceinline__ float multiply(
        float lhs,
        float rhs) noexcept
    {
        return persistRounded(__fmul_rn(lhs, rhs));
    }

    /**
     * @brief Add two values with one retained round-to-nearest boundary.
     * @param lhs Left operand.
     * @param rhs Right operand.
     * @return Explicitly rounded binary32 sum.
     */
    __device__ __forceinline__ float add(
        float lhs,
        float rhs) noexcept
    {
        return persistRounded(__fadd_rn(lhs, rhs));
    }

    /**
     * @brief Compute a deterministic reciprocal for a value in [1, 2].
     *
     * Five fixed Newton iterations square the initial error below binary32
     * precision. Every dependency is explicitly rounded, so CUDA and ROCm
     * publish the same word instead of selecting vendor reciprocal estimates.
     *
     * @param denominator Finite value in the inclusive interval [1, 2].
     * @return Deterministic binary32 reciprocal.
     */
    __device__ __forceinline__ float reciprocalOneToTwo(
        float denominator) noexcept
    {
        float reciprocal = 0.75f;
#pragma unroll
        for (int iteration = 0; iteration < 5; ++iteration)
        {
            const float product = multiply(denominator, reciprocal);
            const float correction = add(2.0f, -product);
            reciprocal = multiply(reciprocal, correction);
        }
        return reciprocal;
    }

    /**
     * @brief Compute a deterministic reciprocal for a positive binary32 value.
     *
     * The denominator is split into an exact power-of-two exponent and a
     * normalized mantissa in [1, 2). The mantissa uses the shared Newton tree;
     * the reciprocal exponent is then installed with integer bit arithmetic,
     * avoiding CUDA/HIP approximate division instructions entirely.
     *
     * Zero returns positive infinity and positive infinity returns zero, which
     * are their IEEE reciprocal results. Callers own any NaN or negative-input
     * rejection because those values are invalid for activation scales.
     *
     * @param denominator Positive finite value, zero, or positive infinity.
     * @return Deterministic reciprocal word.
     */
    __device__ __forceinline__ float reciprocalPositive(
        float denominator) noexcept
    {
        constexpr std::uint32_t kMantissaMask = 0x007fffffu;
        constexpr std::uint32_t kExponentMask = 0x7f800000u;
        constexpr std::uint32_t kPositiveInfinity = 0x7f800000u;

        const std::uint32_t denominator_bits = bitsFromFloat(denominator);
        std::uint32_t mantissa = denominator_bits & kMantissaMask;
        const int stored_exponent =
            static_cast<int>((denominator_bits & kExponentMask) >> 23u);
        if ((denominator_bits & 0x80000000u) != 0u)
            return floatFromBits(0x7fc00000u);
        if (stored_exponent == 255)
            return mantissa == 0u
                       ? 0.0f
                       : floatFromBits(0x7fc00000u);
        if (stored_exponent == 0 && mantissa == 0u)
            return floatFromBits(kPositiveInfinity);

        int unbiased_exponent = stored_exponent - 127;
        if (stored_exponent == 0)
        {
            // Normalize a positive subnormal by moving its highest set bit to
            // the implicit-one position. `__clz` is an exact integer primitive
            // on both device ISAs.
            const int shift = __clz(mantissa) - 8;
            mantissa = (mantissa << shift) & kMantissaMask;
            unbiased_exponent = -126 - shift;
        }

        const float normalized = floatFromBits(
            (127u << 23u) | mantissa);
        const float normalized_reciprocal =
            reciprocalOneToTwo(normalized);
        const std::uint32_t reciprocal_bits =
            bitsFromFloat(normalized_reciprocal);
        const int target_exponent =
            static_cast<int>((reciprocal_bits & kExponentMask) >> 23u) -
            unbiased_exponent;
        const std::uint32_t reciprocal_mantissa =
            reciprocal_bits & kMantissaMask;
        if (target_exponent >= 255)
            return floatFromBits(kPositiveInfinity);
        if (target_exponent > 0)
        {
            return floatFromBits(
                (static_cast<std::uint32_t>(target_exponent) << 23u) |
                reciprocal_mantissa);
        }

        // Scale a tiny reciprocal into the subnormal range with one explicit
        // round-to-nearest-even integer shift.
        const int shift = 1 - target_exponent;
        if (shift >= 32)
            return 0.0f;
        const std::uint32_t significand =
            0x00800000u | reciprocal_mantissa;
        const std::uint32_t truncated = significand >> shift;
        const std::uint32_t remainder_mask =
            (std::uint32_t{1} << shift) - 1u;
        const std::uint32_t remainder = significand & remainder_mask;
        const std::uint32_t halfway =
            std::uint32_t{1} << (shift - 1);
        const std::uint32_t rounded =
            truncated +
            static_cast<std::uint32_t>(
                remainder > halfway ||
                (remainder == halfway && (truncated & 1u) != 0u));
        return floatFromBits(rounded);
    }
} // namespace llaminar2::device_fp32_contract
