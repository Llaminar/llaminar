/**
 * @file ROCmSwiGLUDecodeEquivalentMath.h
 * @brief Canonical ROCm SwiGLU and Q8 publication arithmetic for decode.
 *
 * Serial M=1 decode computes SwiGLU into an FP32 row and then publishes that
 * row as 32-value Q8 blocks. Grouped MTP decode fuses those two operations so
 * several routed expert rows can be published economically in one launch.
 * The fused implementation must preserve every FP32 rounding boundary from
 * the serial implementation: changing the exponential, reciprocal, clamp, or
 * conversion sequence can change a Q8 byte or its scale and therefore change
 * all downstream verifier state.
 *
 * Keep the scalar operations in this header shared by standalone SwiGLU,
 * serial Q8 publication, fused prefill publication, and grouped MoE decode.
 * Kernel launch geometry and max reductions may differ, but no caller should
 * spell out an alternative mathematically-equivalent expression.
 */

#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace llaminar2::rocm::decode_equivalent
{

/**
 * @brief Evaluate one FP32 SwiGLU element using serial-decode arithmetic.
 *
 * gfx906 serial decode deliberately uses the fast hardware exponential and
 * reciprocal instructions. Besides being substantially cheaper than a full
 * IEEE division, this sequence is part of the model's established decode
 * byte stream. Grouped kernels must call this helper instead of using
 * `expf()` or `/`, even though those expressions are mathematically similar.
 *
 * @param gate One gate-projection value.
 * @param up The corresponding up-projection value.
 * @return `silu(gate) * up` with the serial-decode operation sequence.
 */
__device__ __forceinline__ float swigluValue(float gate, float up)
{
    const float exponential = __expf(-gate);
    const float reciprocal =
        __builtin_amdgcn_rcpf(1.0f + exponential);
    return (gate * reciprocal) * up;
}

/**
 * @brief Derive the canonical symmetric Q8 scale for one 32-value block.
 *
 * @param maximum_absolute_value Maximum absolute FP32 value in the block.
 * @return FP32 scale used to map the block to the signed [-127, 127] range.
 */
__device__ __forceinline__ float q8Scale(float maximum_absolute_value)
{
    return maximum_absolute_value > 0.0f
        ? maximum_absolute_value / 127.0f
        : 1.0f;
}

/**
 * @brief Quantize one SwiGLU value with serial M=1 Q8 rounding semantics.
 *
 * Serial decode rounds the scaled FP32 value to an integer first and clamps
 * that integer second. Keeping this order shared avoids a fused grouped kernel
 * silently adopting a float-clamp-before-round expression.
 *
 * @param value FP32 activation value to quantize.
 * @param inverse_scale Reciprocal of the block's canonical Q8 scale.
 * @return Signed integer in the inclusive [-127, 127] range.
 */
__device__ __forceinline__ int32_t quantizeQ8(
    float value,
    float inverse_scale)
{
    int32_t quantized =
        static_cast<int32_t>(rintf(value * inverse_scale));
    if (quantized < -127)
        quantized = -127;
    if (quantized > 127)
        quantized = 127;
    return quantized;
}

} // namespace llaminar2::rocm::decode_equivalent
