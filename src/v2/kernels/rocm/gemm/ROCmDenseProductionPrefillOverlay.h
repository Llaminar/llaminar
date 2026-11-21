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
#include "ROCmDenseProductionPrefillOverlayTypes.h"
#include <cstdint>
#include <optional>
#include <span>
#include "kernels/common/NativeQuantisedArchitecture.h"
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
 * @param actual_hardware Original participant metadata, required for every new
 *        architecture-qualified native row. No device query is performed.
 *        Missing actual metadata cannot consume qualified rows.
 * @return The immutable exact/nearest-compatible launch. Unmeasured keys,
 *         absent actual facts, incompatible resources, and a deliberately
 *         suspended tournament overlay retain the complete generic dispatcher.
 * @throws std::invalid_argument For nonpositive geometry or an unknown family.
 * @throws std::logic_error For a producer/weight ABI mismatch.
 */
[[nodiscard]] std::optional<DensePrefillExactConfig> selectDensePrefillExactConfig(
    std::uint8_t codebook, int m, int n, int k, DensePrefillWeightFamily family,
    const native_quantised::NativeQuantisedGPUHardware *actual_hardware = nullptr);
/**
 * @brief Resolve the same policy over an immutable admitted inventory.
 * @param entries Geometry/family/architecture-sorted generated entries, or an
 *        authored device-free replay of that exact immutable schema.
 * @param codebook Actual canonical execution decoder.
 * @param m Actual live rows; architecture selection never changes this extent.
 * @param n Actual output columns.
 * @param k Actual contraction columns.
 * @param family Original prepared weight ABI.
 * @param actual_hardware Original device facts, never selected tuning facts.
 * @return The same exact/nearest-compatible launch used by installed lookup.
 * @throws std::logic_error For a malformed matching generated entry.
 * @throws std::invalid_argument For invalid geometry or participant metadata.
 */
[[nodiscard]] std::optional<DensePrefillExactConfig> selectDensePrefillExactConfigFromEntries(
    std::span<const generated::ROCmDensePrefillOverlayEntry> entries,
    std::uint8_t codebook, int m, int n, int k, DensePrefillWeightFamily family,
    const native_quantised::NativeQuantisedGPUHardware *actual_hardware = nullptr);
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
