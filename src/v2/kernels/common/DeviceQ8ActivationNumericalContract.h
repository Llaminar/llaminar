/**
 * @file DeviceQ8ActivationNumericalContract.h
 * @brief Cross-backend byte contract for GPU-aligned expert Q8 activation publication.
 *
 * ExpertOverlay can move one expert between CUDA and ROCm while retaining the
 * same prepared weights. Its FP32 activation row must therefore quantize to the
 * same scale and signed byte on either device. CUDA and HIP fast reciprocal
 * instructions are not byte-equivalent near half-integer boundaries, so this
 * contract derives the reciprocal through the shared deterministic FP32 tree.
 */

#pragma once

#include "kernels/common/DeviceHalfMetadataContract.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"

#include <cmath>
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

namespace llaminar2::device_q8_activation_contract
{
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_Q8_CONTRACT_INLINE __device__ __forceinline__
#else
#define LLAMINAR_Q8_CONTRACT_INLINE inline
#endif

    /**
     * @brief Convert a bounded positive Q8 scale with the qualified native half evaluator.
     * @param value Finite positive FP32 value in [2^-24,65504].
     * @return The canonical binary16 RNE scale widened exactly to FP32.
     * @pre HIP callers must qualify their actual kernel descriptor for RNE and
     * binary16 source/result denormal preservation. The general scale() entry
     * deliberately retains its descriptor-independent canonical conversion.
     * @details Only a caller with the bounded-scale and descriptor contracts
     * may use this leaf. It changes neither reciprocal policy nor rounding of
     * activation values. CPU/CUDA use the existing canonical conversion.
     */
    LLAMINAR_Q8_CONTRACT_INLINE float publishPositiveFiniteHalfScale(
        float value) noexcept
    {
#if defined(__HIP_DEVICE_COMPILE__)
        return __half2float(__float2half_rn(value));
#else
        return canonicalPreparedHalfValue(value);
#endif
    }

    /**
     * @brief Select the larger finite magnitude without floating-point comparisons.
     * @param lhs First finite source value or previously selected magnitude.
     * @param rhs Second finite source value or previously selected magnitude.
     * @return Nonnegative FP32 value with the larger exact magnitude word.
     * @details Positive IEEE-754 finite words have unsigned numeric order.
     * Integer selection preserves subnormal input bits under device FTZ modes;
     * a floating maximum could otherwise turn a nonzero block into a zero block.
     * This changes no reduction topology or arithmetic on normal magnitudes.
     */
    LLAMINAR_Q8_CONTRACT_INLINE float maximumAbsoluteValue(
        float lhs, float rhs) noexcept
    {
        const std::uint32_t lhs_bits =
            device_fp32_contract::bitsFromFloat(lhs) & 0x7fffffffu;
        const std::uint32_t rhs_bits =
            device_fp32_contract::bitsFromFloat(rhs) & 0x7fffffffu;
        return device_fp32_contract::floatFromBits(
            lhs_bits > rhs_bits ? lhs_bits : rhs_bits);
    }

    /**
     * @brief Derive the clamped FP32 precursor of a canonical Q8 scale.
     * @param maximum_absolute_value Maximum absolute finite FP32 value.
     * @return Positive finite FP32 scale in the binary16 representable range.
     * @details Zero blocks retain scale 1. Positive derived scales saturate to
     * the complete positive finite half range before RNE publication. Tiny
     * inputs therefore quantize with the smallest half scale; very large
     * inputs use the largest finite half scale and signed-byte saturation.
     * Existing positive finite half publications retain identical bits.
     */
    LLAMINAR_Q8_CONTRACT_INLINE float boundedScale(
        float maximum_absolute_value) noexcept
    {
        // A subnormal maximum may compare equal to zero under CUDA FTZ.
        // Distinguish exact zero by its representation before any arithmetic;
        // a tiny nonzero product may then flush to zero safely at the half floor.
        const bool nonzero =
            (device_fp32_contract::bitsFromFloat(maximum_absolute_value) &
             0x7fffffffu) != 0u;
        const float binary32_scale = nonzero
                                         ? device_fp32_contract::multiply(
                                               maximum_absolute_value,
                                               MoEProjectionNumericalContract::q8_scale_multiplier)
                                         : 1.0f;
        // CPU Q8_1 persists `d` as binary16.  GPU sidecars deliberately widen
        // that same word back to binary32 rather than retaining extra precision
        // that would make residency select a different dequantization scale.
        // This is the activation format's explicit representable range, not
        // a precision switch. Never publish zero/Inf and then divide by it.
        return fminf(65504.0f, fmaxf(0x1p-24f, binary32_scale));
    }

    /**
     * @brief Publish the canonical symmetric Q8 scale for a 32-value block.
     * @param maximum_absolute_value Maximum absolute finite FP32 value.
     * @return Positive finite binary16 scale widened to FP32.
     * @details The integer half conversion is valid independently of the
     * consuming GPU kernel's floating-point descriptor modes.
     */
    LLAMINAR_Q8_CONTRACT_INLINE float scale(float maximum_absolute_value) noexcept
    {
        return canonicalPreparedHalfValue(boundedScale(maximum_absolute_value));
    }

    /**
     * @brief Apply the canonical Q8 conversion with an already prepared reciprocal.
     * @param value Original finite activation value.
     * @param inverse_scale Exact reciprocalPositive(scale(block_maximum)) result.
     * @return Original RNE-before-saturation integer in [-127,127].
     * @details One immutable reciprocal may serve all values of the same block.
     * Multiplication, rounding and clamp retain the original scalar dependencies.
     */
    LLAMINAR_Q8_CONTRACT_INLINE std::int32_t quantizePreparedInverse(
        float value, float inverse_scale) noexcept
    {
        const float scaled = device_fp32_contract::multiply(value, inverse_scale);
        // Round before saturation and clamp before integer conversion, including
        // finite activations whose scaled magnitude exceeds the INT32 range.
        const float rounded = rintf(scaled);
        const float bounded = fminf(127.0f, fmaxf(-127.0f, rounded));
        return static_cast<std::int32_t>(bounded);
    }

    /**
     * @brief Quantize one FP32 value through the heterogeneous Q8 byte contract.
     *
     * The caller publishes @p block_scale beside the returned byte. A single
     * deterministic reciprocal is multiplied by every value in that block;
     * round-to-nearest-even precedes signed-byte saturation exactly as in serial
     * NativeQuantised decode.
     *
     * @param value Finite source activation.
     * @param block_scale Positive scale returned by scale().
     * @return Signed integer in the inclusive range [-127, 127].
     */
    LLAMINAR_Q8_CONTRACT_INLINE std::int32_t quantize(
        float value,
        float block_scale) noexcept
    {
        const float inverse_scale =
            device_fp32_contract::reciprocalPositive(block_scale);
        return quantizePreparedInverse(value, inverse_scale);
    }

#undef LLAMINAR_Q8_CONTRACT_INLINE
} // namespace llaminar2::device_q8_activation_contract
