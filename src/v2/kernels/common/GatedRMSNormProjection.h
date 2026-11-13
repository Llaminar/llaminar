/**
 * @file GatedRMSNormProjection.h
 * @brief Typed retained FP32 normalization input for one prepared SingleQ8 projection.
 *
 * This declaration owns no bytes or runtime registry. The original prepared
 * GEMM invocation owns quantization scratch and immediately consumes it. The
 * values tensor retains the full rounded FP32 normalization result, including
 * diagnostic observations. Ordinary CPU/CUDA/grouped and other geometry
 * programs keep their existing explicit graph operation.
 */
#pragma once
#include "GatedNormActivation.h"
#include <bit>
#include <cstdint>
namespace llaminar2
{
class TensorBase;
/** @brief Immutable normalization arithmetic, independent of projection weights. */
struct GatedRMSNormProjectionParameters
{
    int norm_dim=0; ///< Original per-head width; the selected program admits128.
    int gamma_period=0; ///< Original repeated-gain period, also128.
    float eps=1e-6f; ///< Original FP32 epsilon, without reassociation.
    bool subtract_one=false; ///< Original gamma-plus-one convention.
    GatedNormActivation activation=GatedNormActivation::Identity; ///< Original gate transform.
    /** @brief Compare all policy bits, including epsilon signed zero/NaN payload. */
    friend constexpr bool operator==(const GatedRMSNormProjectionParameters& a,
                                    const GatedRMSNormProjectionParameters& b) noexcept
    {
        return a.norm_dim==b.norm_dim&&a.gamma_period==b.gamma_period&&
            std::bit_cast<std::uint32_t>(a.eps)==std::bit_cast<std::uint32_t>(b.eps)&&
            a.subtract_one==b.subtract_one&&a.activation==b.activation;
    }
};
/**
 * @brief Prove the closed ordinary scalar geometry using immutable metadata.
 * @param rows Physical rows, not an observed device count.
 * @param columns Original local FP32/quantized feature width.
 * @param parameters Exact original normalization policy.
 * @return Whether a complete128 head owns all of its original32-value Q8 blocks.
 */
inline constexpr bool gatedNormSingleQ8Geometry(int rows,int columns,
    const GatedRMSNormProjectionParameters& parameters) noexcept
{
    return rows==1&&columns>0&&columns%128==0&&parameters.norm_dim==128&&
        parameters.gamma_period==128&&validGatedNormActivation(parameters.activation);
}
/**
 * @brief Borrow the original tensors for one immediate, in-place composite.
 * @details Values is both raw recurrence input and retained normalized output;
 * gate/gamma are immutable. No quantized buffer or ownership escapes the call.
 */
struct GatedRMSNormProjectionInput
{
    TensorBase& values; ///< Exact existing arena InOut activation.
    const TensorBase& gate; ///< Original learned gate output, read-only.
    const TensorBase& gamma; ///< Original gain tensor, read-only.
    GatedRMSNormProjectionParameters parameters; ///< Frozen numeric policy.
};
} // namespace llaminar2
