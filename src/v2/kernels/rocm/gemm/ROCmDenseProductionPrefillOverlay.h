/**
 * @file ROCmDenseProductionPrefillOverlay.h
 * @brief Typed admission of common-generated ordinary-prefill exact launches.
 *
 * Immutable measured entries precede the existing total generic policy. Native
 * VNNI (including signed Q8) and expanded INT8 have separate prepared families; a key
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
/** @brief Launch value owned by the authenticated common policy generator. */
using DensePrefillExactConfig = generated::ROCmDensePrefillOverlayConfig;
/**
 * @brief Resolve one measured key before graph recording.
 * @param codebook Canonical execution codebook, including Q8 alias pooling.
 * @param m Exact physical row extent; unseen positive rows use generic policy.
 * @param n Actual participant-owned output columns.
 * @param k Actual contraction geometry, independent of full-model dimensions.
 * @param family Explicit prepared ABI, independent of the source codebook.
 *         A matching key with an incompatible configured producer is fatal.
 * @return The immutable exact launch, absent only for unmeasured keys or a
 *         deliberately suspended tournament overlay.
 * @throws std::invalid_argument For nonpositive geometry or an unknown family.
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
