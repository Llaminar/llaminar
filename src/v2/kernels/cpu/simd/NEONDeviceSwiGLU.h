/**
 * @file NEONDeviceSwiGLU.h
 * @brief Four independent AArch64 programs for placement-invariant expert SwiGLU.
 *
 * Each lane implements DeviceSwiGLUNumericalContract, including the split-ln2
 * range reduction, nine-coefficient Horner tree and five separately rounded
 * reciprocal iterations. There is no approximate reciprocal instruction, row
 * replay, allocation or shared state. Lane-local masks also preserve the scalar
 * contract's early-result domains without changing any other lane's arithmetic.
 * Separate operations retain opaque register edges after inlining; a GCC
 * function optimization attribute alone does not preserve the Newton tree.
 */
#pragma once

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native expert SwiGLU requires AArch64 ASIMD"
#endif
#include "tensors/NEONFP32.h"

#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_DEVICE_SWIGLU_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_NEON_DEVICE_SWIGLU_EXACT
#endif

namespace llaminar2::neon
{
    /**
     * @brief Evaluate the device exponential program in four independent lanes.
     * @param value Nonpositive inputs; the scalar contract also defines NaN and nonnegative results.
     * @return Original-order binary32 words with the scalar contract's rounding boundaries.
     */
    [[gnu::always_inline]] LLAMINAR_NEON_DEVICE_SWIGLU_EXACT inline float32x4_t
    device_exp_non_positive(float32x4_t value) noexcept
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const auto zero = vdupq_n_f32(0.0f);
        const auto scaled = neon_fp32::multiply(value, vdupq_n_f32(0x1.715476p+0f));
        const auto offset = vbslq_f32(vcgeq_f32(scaled, zero), vdupq_n_f32(0.5f), vdupq_n_f32(-0.5f));
        const auto exponent = vcvtq_s32_f32(neon_fp32::add(scaled, offset));
        const auto exponent_float = vcvtq_f32_s32(exponent);
        auto remainder = vfmsq_n_f32(value, exponent_float, 0x1.62e400p-1f);
        remainder = vfmsq_n_f32(remainder, exponent_float, 0x1.7f7d1cp-20f);
        auto polynomial = vfmaq_f32(vdupq_n_f32(0x1.a01a02p-13f), vdupq_n_f32(0x1.a01a02p-16f), remainder);
        polynomial = vfmaq_f32(vdupq_n_f32(0x1.6c16c2p-10f), polynomial, remainder);
        polynomial = vfmaq_f32(vdupq_n_f32(0x1.111112p-7f), polynomial, remainder);
        polynomial = vfmaq_f32(vdupq_n_f32(0x1.555556p-5f), polynomial, remainder);
        polynomial = vfmaq_f32(vdupq_n_f32(0x1.555556p-3f), polynomial, remainder);
        polynomial = vfmaq_f32(vdupq_n_f32(0x1.000000p-1f), polynomial, remainder);
        polynomial = vfmaq_f32(vdupq_n_f32(1.0f), polynomial, remainder);
        polynomial = vfmaq_f32(vdupq_n_f32(1.0f), polynomial, remainder);
        const auto power = vshlq_n_s32(vaddq_s32(exponent, vdupq_n_s32(127)), 23);
        auto result = neon_fp32::multiply(polynomial, vreinterpretq_f32_s32(power));
        result = vbslq_f32(vcleq_f32(value, vdupq_n_f32(-87.0f)), zero, result);
        result = vbslq_f32(vcgeq_f32(value, zero), vdupq_n_f32(1.0f), result);
        return vbslq_f32(vmvnq_u32(vceqq_f32(value, value)), value, result);
    }

    /**
     * @brief Execute four exact heterogeneous expert SwiGLU dependency chains.
     * @param gate Original gate-projection lanes.
     * @param up Original up-projection lanes.
     * @return Placement-invariant lane results, independent of row/tile geometry.
     */
    [[gnu::always_inline]] LLAMINAR_NEON_DEVICE_SWIGLU_EXACT inline float32x4_t
    device_swiglu(float32x4_t gate, float32x4_t up) noexcept
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const auto positive = vcgeq_f32(gate, vdupq_n_f32(0.0f));
        const auto argument = vbslq_f32(positive, vnegq_f32(gate), gate);
        const auto exponential = device_exp_non_positive(argument);
        const auto denominator = neon_fp32::add(vdupq_n_f32(1.0f), exponential);
        auto reciprocal = vdupq_n_f32(0.75f);
#pragma GCC unroll 5
        for (int iteration = 0; iteration < 5; ++iteration)
        {
            // The canonical Newton step has three rounded operations. A
            // fused 2-denominator*reciprocal changes placement-visible words.
            const auto product = neon_fp32::multiply(denominator, reciprocal);
            const auto correction = neon_fp32::subtract(vdupq_n_f32(2.0f), product);
            reciprocal = neon_fp32::multiply(reciprocal, correction);
        }
        const auto sigmoid = vbslq_f32(positive, reciprocal, neon_fp32::multiply(exponential, reciprocal));
        return neon_fp32::multiply(neon_fp32::multiply(gate, sigmoid), up);
    }
} // namespace llaminar2::neon
#undef LLAMINAR_NEON_DEVICE_SWIGLU_EXACT
