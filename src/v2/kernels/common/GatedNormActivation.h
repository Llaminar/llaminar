/**
 * @file GatedNormActivation.h
 * @brief Immutable arithmetic policy for a gate applied after RMS normalization.
 *
 * Qwen3.5 uses SiLU, while Flash Next uses sigmoid. These policies share the
 * same norm reduction and native CPU/CUDA/ROCm kernel, and are part of each
 * captured stage's identity. Selecting one never changes at replay time.
 */
#pragma once
#include <cstdint>

namespace llaminar2
{
/** @brief Complete gate operations supported by the fused normalization kernel. */
enum class GatedNormActivation : std::uint8_t
{
    Identity, ///< Multiply by the raw gate.
    SiLU, ///< Multiply by z/(1+exp(-z)).
    Sigmoid, ///< Multiply by 1/(1+exp(-z)).
};
/** @brief Reject an invalid serialized/cast policy before capturing a kernel. */
constexpr bool validGatedNormActivation(GatedNormActivation value) noexcept
{
    return value == GatedNormActivation::Identity || value == GatedNormActivation::SiLU ||
           value == GatedNormActivation::Sigmoid;
}
} // namespace llaminar2
