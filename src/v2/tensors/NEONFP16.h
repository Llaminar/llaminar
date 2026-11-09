/**
 * @file NEONFP16.h
 * @brief Native four-lane half conversion with the portable storage bit contract.
 *
 * Expansion uses the mandatory AArch64 half-to-single conversion while restoring
 * nonfinite sign/payload words. Narrowing retains FP16Utils' normal ties-even and
 * truncated-subnormal programs, rather than silently selecting hardware's fully
 * rounded subnormal conversion. Tensor readers, nonlinear leaves and prepared
 * weights share these register-only operations; they own no storage or policy.
 */
#pragma once

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native half expansion requires AArch64 ASIMD"
#endif
#include <arm_neon.h>

namespace llaminar2::neon_fp16
{
    /**
     * @brief Expand four original half words without narrowing or payload changes.
     * @param half Four binary16 storage words.
     * @return Four binary32 words with the same portable sign/exponent/payload.
     */
    [[gnu::always_inline]] inline float32x4_t widen(uint16x4_t half) noexcept
    {
        const auto raw = vmovl_u16(half);
        const auto converted = vreinterpretq_u32_f32(vcvt_f32_f16(vreinterpret_f16_u16(half)));
        const auto nonfinite = vceqq_u32(vandq_u32(raw, vdupq_n_u32(0x7c00)), vdupq_n_u32(0x7c00));
        const auto source_nonfinite = vorrq_u32(vdupq_n_u32(0x7f800000),
            vorrq_u32(vshlq_n_u32(vandq_u32(raw, vdupq_n_u32(0x8000)), 16),
                      vshlq_n_u32(vandq_u32(raw, vdupq_n_u32(0x03ff)), 13)));
        return vreinterpretq_f32_u32(vbslq_u32(nonfinite, source_nonfinite, converted));
    }
    /**
     * @brief Narrow four words through the exact FP16Utils storage program.
     *
     * Normal mantissas use ties-even, including exponent carry. Subnormals are
     * truncated after installing the implicit bit; nonfinite payloads retain the
     * original upper ten bits without adding a hardware quiet-NaN bit. Native
     * variable shifts handle each lane's underflow without scalar replay.
     *
     * @param value Four original binary32 storage values.
     * @return Four binary16 words, byte-identical to fp32_to_fp16().
     */
    [[gnu::always_inline]] inline uint16x4_t narrow(float32x4_t value) noexcept
    {
        const auto bits = vreinterpretq_u32_f32(value);
        const auto sign = vshrq_n_u32(vandq_u32(bits, vdupq_n_u32(0x80000000u)), 16);
        const auto stored_exponent = vandq_u32(vshrq_n_u32(bits, 23), vdupq_n_u32(255));
        const auto exponent = vsubq_s32(vreinterpretq_s32_u32(stored_exponent), vdupq_n_s32(112));
        const auto mantissa = vandq_u32(bits, vdupq_n_u32(0x007fffffu));
        const auto rounding = vandq_u32(vshrq_n_u32(mantissa, 13), vdupq_n_u32(1));
        const auto rounded = vaddq_u32(mantissa, vaddq_u32(vdupq_n_u32(0x0fffu), rounding));
        const auto carry = vcgeq_u32(rounded, vdupq_n_u32(0x00800000u));
        const auto rounded_exponent = vaddq_s32(exponent, vreinterpretq_s32_u32(vandq_u32(carry, vdupq_n_u32(1))));
        const auto normal = vorrq_u32(sign,
            vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(rounded_exponent), 10),
                      vshrq_n_u32(vbicq_u32(rounded, carry), 13)));
        const auto subnormal_mantissa = vshlq_u32(
            vorrq_u32(mantissa, vdupq_n_u32(0x00800000u)),
            vsubq_s32(exponent, vdupq_n_s32(1)));
        const auto subnormal = vorrq_u32(sign, vshrq_n_u32(subnormal_mantissa, 13));
        const auto nan_payload = vbslq_u32(vceqq_u32(stored_exponent, vdupq_n_u32(255)),
                                           vshrq_n_u32(mantissa, 13), vdupq_n_u32(0));
        const auto overflow = vorrq_u32(sign, vorrq_u32(vdupq_n_u32(0x7c00), nan_payload));
        const auto nonunderflow = vbslq_u32(vcgeq_s32(exponent, vdupq_n_s32(31)), overflow, normal);
        return vmovn_u32(vbslq_u32(vcleq_s32(exponent, vdupq_n_s32(0)), subnormal, nonunderflow));
    }
} // namespace llaminar2::neon_fp16
