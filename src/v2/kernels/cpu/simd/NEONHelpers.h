/**
 * @file NEONHelpers.h
 * @brief Native four-lane AArch64 nonlinear arithmetic for CPU channel kernels.
 *
 * Range reduction, ties-even integer selection and explicit FMA polynomial
 * order match the established CPU exponential program. Hardware division is
 * retained for sigmoid rather than substituting a reciprocal estimate. Every
 * operation is lane-local, independent of row count and workshare ownership.
 */
#pragma once
#include <arm_neon.h>

#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_NONLINEAR_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_NEON_NONLINEAR_EXACT
#endif

namespace llaminar2::neon
{
    /** @brief Evaluate the established finite-input CPU exponential polynomial.
     * @param value Four original FP32 values, clamped to the published [-88,88] range.
     * @return Four range-reduced polynomial/exponent products in original lane order.
     */
    [[gnu::always_inline]] LLAMINAR_NEON_NONLINEAR_EXACT inline float32x4_t fast_exp(float32x4_t value) noexcept
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const auto bounded = vminq_f32(vmaxq_f32(value, vdupq_n_f32(-88.0f)), vdupq_n_f32(88.0f));
        const auto scaled = vmulq_n_f32(bounded, 1.4426950408889634f);
        const auto integer = vrndnq_f32(scaled);
        const auto fraction = vsubq_f32(scaled, integer);
        auto polynomial = vfmaq_n_f32(vdupq_n_f32(0.009618129107629f), fraction, 0.001333355814642f);
        polynomial = vfmaq_f32(vdupq_n_f32(0.055504108664822f), polynomial, fraction);
        polynomial = vfmaq_f32(vdupq_n_f32(0.240226506959101f), polynomial, fraction);
        polynomial = vfmaq_f32(vdupq_n_f32(0.693147180559945f), polynomial, fraction);
        polynomial = vfmaq_f32(vdupq_n_f32(1.0f), polynomial, fraction);
        const auto exponent = vshlq_n_s32(vaddq_s32(vcvtnq_s32_f32(integer), vdupq_n_s32(127)), 23);
        return vmulq_f32(polynomial, vreinterpretq_f32_s32(exponent));
    }

    /** @brief Evaluate four CPU sigmoid lanes with the existing polynomial and exact division.
     * @param value Four finite original FP32 inputs.
     * @return 1/(1+exp(-value)) under the declared CPU polynomial program.
     */
    [[gnu::always_inline]] LLAMINAR_NEON_NONLINEAR_EXACT inline float32x4_t fast_sigmoid(float32x4_t value) noexcept
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        const auto one = vdupq_n_f32(1.0f);
        return vdivq_f32(one, vaddq_f32(one, fast_exp(vsubq_f32(vdupq_n_f32(0.0f), value))));
    }

    /** @brief Evaluate four independent CPU SiLU lanes.
     * @param value Four finite original FP32 inputs.
     * @return value*sigmoid(value), with no cross-lane arithmetic.
     */
    [[gnu::always_inline]] LLAMINAR_NEON_NONLINEAR_EXACT inline float32x4_t fast_silu(float32x4_t value) noexcept
    { return vmulq_f32(value, fast_sigmoid(value)); }
} // namespace llaminar2::neon
#undef LLAMINAR_NEON_NONLINEAR_EXACT
