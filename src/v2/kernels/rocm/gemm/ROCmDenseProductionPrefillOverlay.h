/**
 * @file ROCmDenseProductionPrefillOverlay.h
 * @brief Typed admission of common-generated ordinary-prefill exact launches.
 *
 * Immutable measured entries precede the existing total generic policy. Native
 * low-bit and Q8 INT8 weights have separate physical producer families; a key
 * cannot cross those families. Diagnostic tournaments explicitly suspend exact
 * lookup so their Auto baseline remains the generic policy. Neither lookup nor
 * that setup-only diagnostic switch owns device inference state or storage.
 */
#pragma once
#include "ROCmDenseProductionPrefillOverlayGenerated.inc"
#include <cstdint>
#include <optional>
namespace llaminar2::rocm
{
/** @brief Prepared physical weight families with disjoint producer ABIs. */
enum class DensePrefillWeightFamily : std::uint8_t { NativeLowBit, Int8 };
/** @brief Launch value owned by the authenticated common policy generator. */
using DensePrefillExactConfig = generated::ROCmDensePrefillOverlayConfig;
/**
 * @brief Resolve one measured key before graph recording.
 * @param codebook Canonical execution codebook, including Q8 alias pooling.
 * @param m Exact physical row extent; unseen positive rows use generic policy.
 * @param n Actual participant-owned output columns.
 * @param k Actual contraction geometry, independent of full-model dimensions.
 * @param family Prepared weight ABI; a matching key from another ABI is fatal.
 * @return The immutable exact launch, absent only for unmeasured keys or a
 *         deliberately suspended tournament overlay.
 * @throws std::invalid_argument For nonpositive geometry.
 * @throws std::logic_error For a producer/weight ABI mismatch.
 */
[[nodiscard]] std::optional<DensePrefillExactConfig> selectDensePrefillExactConfig(
    std::uint8_t codebook, int m, int n, int k, DensePrefillWeightFamily family);
/** @brief Read the calling construction thread's diagnostic lookup policy. */
[[nodiscard]] bool densePrefillExactOverlayEnabled() noexcept;
/**
 * @brief Set setup-only diagnostic lookup policy on the calling host thread.
 * @param enabled True for installed production lookup, false for generic Auto
 *                tournament evidence; callers retain and restore prior policy.
 * This changes subsequent graph construction, never an existing captured graph.
 */
void setDensePrefillExactOverlayEnabled(bool enabled) noexcept;
}
