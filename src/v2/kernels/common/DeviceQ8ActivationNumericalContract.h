/**
 * @file DeviceQ8ActivationNumericalContract.h
 * @brief Cross-backend byte contract for movable Q8 activation publication.
 *
 * ExpertOverlay can move one expert between CUDA and ROCm while retaining the
 * same prepared weights. Its FP32 activation row must therefore quantize to the
 * same scale and signed byte on either device. CUDA and HIP fast reciprocal
 * instructions are not byte-equivalent near half-integer boundaries, so this
 * contract derives the reciprocal through the shared deterministic FP32 tree.
 */

#pragma once

#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"

#include <cmath>
#include <cstdint>

namespace llaminar2::device_q8_activation_contract
{
    /**
     * @brief Derive the canonical symmetric Q8 scale for a 32-value block.
     * @param maximum_absolute_value Maximum absolute finite FP32 value.
     * @return Positive scale mapping the maximum magnitude to 127.
     */
    __device__ __forceinline__ float scale(
        float maximum_absolute_value) noexcept
    {
        return maximum_absolute_value > 0.0f
                   ? device_fp32_contract::multiply(
                         maximum_absolute_value,
                         MoEProjectionNumericalContract::q8_scale_multiplier)
                   : 1.0f;
    }

    /**
     * @brief Quantize one FP32 value through the heterogeneous Q8 byte contract.
     *
     * The caller publishes @p block_scale beside the returned byte. A single
     * deterministic reciprocal is multiplied by every value in that block;
     * round-to-nearest-even precedes integer saturation exactly as in serial
     * NativeVNNI decode.
     *
     * @param value Finite source activation.
     * @param block_scale Positive scale returned by scale().
     * @return Signed integer in the inclusive range [-127, 127].
     */
    __device__ __forceinline__ std::int32_t quantize(
        float value,
        float block_scale) noexcept
    {
        const float inverse_scale =
            device_fp32_contract::reciprocalPositive(block_scale);
        const float scaled =
            device_fp32_contract::multiply(value, inverse_scale);
        std::int32_t quantized = static_cast<std::int32_t>(rintf(scaled));
        if (quantized < -127)
            quantized = -127;
        if (quantized > 127)
            quantized = 127;
        return quantized;
    }
} // namespace llaminar2::device_q8_activation_contract
