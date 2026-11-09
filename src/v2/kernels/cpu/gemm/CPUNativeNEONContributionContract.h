/**
 * @file CPUNativeNEONContributionContract.h
 * @brief Native AArch64 integer dots and rounded NativeQuantised block contributions.
 *
 * The existing packed stream stores four consecutive signed bytes per output
 * column. A 128-bit register therefore holds four independent column dots.
 * Integer products widen before pair addition, avoiding PMADDUBSW-style
 * saturation at signed-byte extrema. The FP32 helpers preserve the existing
 * BackendNative FMA program and placement-invariant GPUAlignedExpert rounding
 * edges independently of physical SIMD width. No storage, workspace or policy
 * state is owned here; callers retain the ordered K-partition/fold plan.
 */
#pragma once

#include "kernels/common/MoEProjectionNumericalContract.h"

#include <arm_neon.h>
#include <cstdint>
#include <stdexcept>

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native NEON contributions require the AArch64 ASIMD compiler envelope"
#endif

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief Retain a rounded FP32 vector through an instruction-free register edge.
     * @param value Four completed binary32 words.
     * @return Identical words whose following consumer cannot absorb their rounding.
     */
    [[gnu::always_inline]] inline float32x4_t persistRoundedNEON(float32x4_t value) noexcept
    {
        asm volatile("" : "+w"(value));
        return value;
    }

    /**
     * @brief Add four independent four-byte signed dots without saturation.
     * @param accumulator Four existing INT32 column sums with sufficient headroom.
     * @param activation Four consecutive activation bytes repeated in each column group.
     * @param weights Four consecutive signed weight bytes per output column.
     * @return The four updated column sums; no horizontal FP reduction occurs.
     */
    [[gnu::always_inline]] inline int32x4_t dotFourColumnsASIMD(
        int32x4_t accumulator, int8x16_t activation, int8x16_t weights) noexcept
    {
        const int16x8_t low = vmull_s8(vget_low_s8(activation), vget_low_s8(weights));
        const int16x8_t high = vmull_s8(vget_high_s8(activation), vget_high_s8(weights));
        // Widen pair sums before addition: (-128*-128)*2 is 32768, which
        // does not fit signed INT16. Each resulting INT32 lane owns one column.
        const int32x4_t low_pairs = vpaddlq_s16(low);
        const int32x4_t high_pairs = vpaddlq_s16(high);
        return vaddq_s32(accumulator, vpaddq_s32(low_pairs, high_pairs));
    }

#if defined(__ARM_FEATURE_DOTPROD)
    /**
     * @brief Execute the same four signed column dots with native SDOT.
     * @param accumulator Four existing INT32 column sums with sufficient headroom.
     * @param activation Repeated four-byte activation word in original K order.
     * @param weights Four signed bytes for each of four output columns.
     * @return Exact INT32 sums, identical to dotFourColumnsASIMD.
     * @details Only DotProd-targeted translation units expose this leaf. Their
     * caller must admit that compiler envelope against native CPU capabilities.
     */
    [[gnu::always_inline]] inline int32x4_t dotFourColumnsDotProd(
        int32x4_t accumulator, int8x16_t activation, int8x16_t weights) noexcept
    {
        return vdotq_s32(accumulator, activation, weights);
    }
#endif

    /**
     * @brief Append one symmetric 32-value block to four output columns.
     * @param accumulator Running FP32 column sums.
     * @param dot Exact signed INT32 block dots.
     * @param weight_scale Four decoded weight scales, retaining source-factor precision.
     * @param activation_scale Exact widened persisted activation half scale.
     * @param policy Backend-native FMA or canonical placement-invariant program.
     * @return Updated column sums in the selected numerical program.
     * @throws std::invalid_argument For an unknown numerical policy.
     */
    [[gnu::always_inline]] inline float32x4_t accumulateSingleScaleBlockNEON(
        float32x4_t accumulator, int32x4_t dot, float32x4_t weight_scale,
        float activation_scale, CPUProjectionNumericalPolicy policy)
    {
        const float32x4_t combined_scale = persistRoundedNEON(
            vmulq_f32(vdupq_n_f32(activation_scale), weight_scale));
        const float32x4_t dot_fp32 = vcvtq_f32_s32(dot);
        switch (policy)
        {
        case CPUProjectionNumericalPolicy::GPUAlignedExpert:
        {
            const float32x4_t contribution = persistRoundedNEON(
                vmulq_f32(combined_scale, dot_fp32));
            return vaddq_f32(accumulator, contribution);
        }
        case CPUProjectionNumericalPolicy::BackendNative:
            return vfmaq_f32(accumulator, dot_fp32, combined_scale);
        }
        throw std::invalid_argument("Native NEON contribution requires a known CPU numerical policy");
    }

    /**
     * @brief Append one asymmetric block with its independently rounded minimum correction.
     * @param accumulator Running FP32 column sums.
     * @param dot Exact signed INT32 block dots.
     * @param weight_scale Four decoded weight scales.
     * @param activation_sum Exact signed-byte sum for this activation block.
     * @param weight_min Four decoded additive minima.
     * @param activation_scale Exact widened persisted activation half scale.
     * @param policy Backend-native or placement-invariant numerical program.
     * @return Updated column sums with the same block boundaries as serial decode.
     * @throws std::invalid_argument For an unknown numerical policy.
     */
    [[gnu::always_inline]] inline float32x4_t accumulateCorrectedSingleScaleBlockNEON(
        float32x4_t accumulator, int32x4_t dot, float32x4_t weight_scale,
        std::int32_t activation_sum, float32x4_t weight_min,
        float activation_scale, CPUProjectionNumericalPolicy policy)
    {
        const float32x4_t activation = vdupq_n_f32(activation_scale);
        const float32x4_t dot_fp32 = vcvtq_f32_s32(dot);
        const float32x4_t scaled_weight = persistRoundedNEON(vmulq_f32(activation, weight_scale));
        switch (policy)
        {
        case CPUProjectionNumericalPolicy::GPUAlignedExpert:
        {
            const float32x4_t dot_contribution = persistRoundedNEON(vmulq_f32(scaled_weight, dot_fp32));
            const float32x4_t scaled_min = persistRoundedNEON(vmulq_f32(activation, weight_min));
            const float32x4_t correction = persistRoundedNEON(vmulq_f32(
                scaled_min, vdupq_n_f32(static_cast<float>(activation_sum))));
            const float32x4_t completed_block = persistRoundedNEON(vaddq_f32(dot_contribution, correction));
            return vaddq_f32(accumulator, completed_block);
        }
        case CPUProjectionNumericalPolicy::BackendNative:
        {
            const float32x4_t value = vfmaq_f32(accumulator, dot_fp32, scaled_weight);
            const float32x4_t activation_correction = persistRoundedNEON(vmulq_f32(
                activation, vdupq_n_f32(static_cast<float>(activation_sum))));
            return vfmaq_f32(value, activation_correction, weight_min);
        }
        }
        throw std::invalid_argument("Native NEON contribution requires a known CPU numerical policy");
    }
} // namespace llaminar2::cpu::native_quantised
