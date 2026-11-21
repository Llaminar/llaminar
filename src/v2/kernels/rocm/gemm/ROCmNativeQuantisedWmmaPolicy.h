/**
 * @file ROCmNativeQuantisedWmmaPolicy.h
 * @brief Explicit gfx1100 INT8 WMMA candidate identities for native prefill.
 *
 * The construction-thread override selects a production kernel before capture.
 * It owns no inference state and never changes an already recorded graph. Tile
 * selection cannot change source decoding, block scaling or the serial K fold.
 */
#pragma once
#include <optional>

namespace llaminar2::rocm
{
/** @brief Compiled wave64 tiles; four waves partition a two-by-two tile grid. */
enum class NativeQuantisedWmmaTile { M32N32, M32N64, M64N64, M64N128 };
/** @brief Read the current construction thread's optional diagnostic selection. */
[[nodiscard]] std::optional<NativeQuantisedWmmaTile> nativeQuantisedWmmaOverride() noexcept;
/**
 * @brief Set the production candidate used by subsequent graph construction.
 * @param tile Explicit tile, or nullopt to restore normal policy selection.
 */
void setNativeQuantisedWmmaOverride(std::optional<NativeQuantisedWmmaTile> tile) noexcept;
}
