/**
 * @file DeviceSwiGLUNumericalContract.h
 * @brief Cross-backend FP32 arithmetic contract for movable-expert SwiGLU.
 *
 * A heterogeneous ExpertOverlay request may execute the same logical expert
 * on CPU, CUDA, or ROCm in adjacent epochs. Vendor `expf()` implementations
 * are individually accurate, but they are not required to return the same
 * binary32 word. A one-ULP sigmoid difference can change every downstream
 * projection result. This header defines one backend-neutral exponential
 * range reduction, polynomial, and reciprocal, with explicit round-to-nearest
 * operations, so placement never selects a different numerical program.
 *
 * The approximation evaluates exp only on the non-positive half-line used by
 * the stable sigmoid identity. It has less than one binary32 ULP relative error
 * over the normal range. Values below -87 are deliberately returned as exact
 * zero: their sigmoid contribution is below normal binary32 precision and the
 * stable formulation avoids overflow for large negative gates.
 */

#pragma once

#include "kernels/common/DeviceFP32NumericalContract.h"

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_SWIGLU_CONTRACT_INLINE __device__ __forceinline__
#else
#define LLAMINAR_SWIGLU_CONTRACT_INLINE inline
#endif

namespace llaminar2::device_swiglu_contract
{
    using device_fp32_contract::floatFromBits;
    using device_fp32_contract::add;
    using device_fp32_contract::fusedMultiplyAdd;
    using device_fp32_contract::multiply;
    using device_fp32_contract::persistRounded;
    using device_fp32_contract::reciprocalOneToTwo;

    /**
     * @brief Evaluate exp(x) deterministically for x in [-87, 0].
     *
     * The input is reduced as `x = n*ln(2) + r`, with r near zero. Splitting
     * ln(2) into high and low binary32 words prevents the exponent index from
     * amplifying range-reduction error. An eighth-order polynomial then
     * evaluates exp(r) entirely through round-to-nearest FMAs before one exact
     * power-of-two scaling multiplication.
     *
     * @param x Non-positive finite binary32 input.
     * @return Deterministic binary32 approximation to exp(@p x).
     */
    LLAMINAR_SWIGLU_CONTRACT_INLINE float expNonPositive(float x)
    {
        if (x != x)
            return x;
        if (x >= 0.0f)
            return 1.0f;
        if (x <= -87.0f)
            return 0.0f;

        constexpr float kInverseLn2 = 0x1.715476p+0f;
        constexpr float kLn2High = 0x1.62e400p-1f;
        constexpr float kLn2Low = 0x1.7f7d1cp-20f;

        const float scaled = multiply(x, kInverseLn2);
        const int exponent = static_cast<int>(
            add(scaled, scaled >= 0.0f ? 0.5f : -0.5f));
        const float exponent_fp32 = static_cast<float>(exponent);
        float remainder = fusedMultiplyAdd(-exponent_fp32, kLn2High, x);
        remainder = fusedMultiplyAdd(-exponent_fp32, kLn2Low, remainder);

        // Horner evaluation of sum(r^k/k!, k=0..8). Each dependency has one
        // explicit rounding boundary and therefore the same word on both ISAs.
        float polynomial = 0x1.a01a02p-16f;
        polynomial = fusedMultiplyAdd(
            polynomial, remainder, 0x1.a01a02p-13f);
        polynomial = fusedMultiplyAdd(
            polynomial, remainder, 0x1.6c16c2p-10f);
        polynomial = fusedMultiplyAdd(
            polynomial, remainder, 0x1.111112p-7f);
        polynomial = fusedMultiplyAdd(
            polynomial, remainder, 0x1.555556p-5f);
        polynomial = fusedMultiplyAdd(
            polynomial, remainder, 0x1.555556p-3f);
        polynomial = fusedMultiplyAdd(
            polynomial, remainder, 0x1.000000p-1f);
        polynomial = fusedMultiplyAdd(polynomial, remainder, 1.0f);
        polynomial = fusedMultiplyAdd(polynomial, remainder, 1.0f);

        const std::uint32_t exponent_bits =
            static_cast<std::uint32_t>(exponent + 127) << 23u;
        return multiply(polynomial, floatFromBits(exponent_bits));
    }

    /**
     * @brief Evaluate `silu(gate) * up` with cross-backend byte semantics.
     *
     * Positive gates use `1 / (1 + exp(-gate))`. Negative gates use the stable
     * equivalent `exp(gate) / (1 + exp(gate))`, preventing an overflowing
     * intermediate while retaining an accurate result. Every multiplication
     * and addition is explicitly rounded; ordinary division is IEEE binary32
     * because neither GPU translation unit is compiled with fast-math.
     *
     * @param gate Gate-projection value.
     * @param up Up-projection value.
     * @return Deterministic binary32 SwiGLU value.
     */
    LLAMINAR_SWIGLU_CONTRACT_INLINE float swigluValue(float gate, float up)
    {
        float sigmoid = 0.0f;
        if (gate >= 0.0f)
        {
            const float exponential = expNonPositive(-gate);
            const float denominator = add(1.0f, exponential);
            sigmoid = reciprocalOneToTwo(denominator);
        }
        else
        {
            const float exponential = expNonPositive(gate);
            const float denominator = add(1.0f, exponential);
            sigmoid = multiply(
                exponential, reciprocalOneToTwo(denominator));
        }
        const float silu = multiply(gate, sigmoid);
        return multiply(silu, up);
    }

    /**
     * @brief Evaluate SiLU alone through the same shared dependency chain.
     * @param value Input activation.
     * @return Deterministic binary32 SiLU value.
     */
    LLAMINAR_SWIGLU_CONTRACT_INLINE float siluValue(float value)
    {
        return swigluValue(value, 1.0f);
    }
} // namespace llaminar2::device_swiglu_contract

#undef LLAMINAR_SWIGLU_CONTRACT_INLINE
