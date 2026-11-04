/**
 * @file NativeWeightSourceWindow.h
 * @brief Original-coordinate metadata for one compact native weight read window.
 *
 * A source owner retains the complete original tensor and selected logical IDs.
 * This value describes only the bytes present in one bounded read; it grants no
 * source, allocation, execution or native-readiness authority. Prepared writers
 * retain their original matrix/source identity and validate these coordinates
 * before writing final admitted storage. No whole-parent pointer is invented.
 */
#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2 {
/** @brief One original matrix rectangle, packed contiguously by native source rows. */
struct NativeWeightSourceWindow final {
    std::size_t original_expert = 0; ///< Original source E coordinate; ordinary matrices use their sole matrix0.
    std::size_t selected_expert_ordinal = 0; ///< Ordinal within the complete original selected-ID sequence.
    std::size_t first_row = 0; ///< Original N coordinate (GGUF axis1), never a window-relative destination.
    std::size_t rows = 0; ///< Actual source rows present in this window.
    std::size_t first_column = 0; ///< Original K coordinate (GGUF axis0), aligned to the source native group.
    std::size_t columns = 0; ///< Actual source elements per row, preserving valid native group boundaries.
    std::size_t native_row_bytes = 0; ///< Actual bytes for window columns in one compact row; excludes source gaps.

    /** @return Exact immutable source-coordinate equality, independent of payload lifetime. */
    bool operator==(const NativeWeightSourceWindow &) const = default;

    /**
     * @brief Return the exact useful native payload extent, never allocation capacity.
     * @return Positive compact row count multiplied by its actual row bytes.
     * @throws std::invalid_argument For empty or overflowing window geometry.
     * @note Format/group/source bounds remain the original reader and writer's
     * responsibility; this arithmetic check does not authenticate a source.
     */
    std::size_t nativeBytes() const {
        if (!rows || !columns || !native_row_bytes ||
            rows > std::numeric_limits<std::size_t>::max() / native_row_bytes)
            throw std::invalid_argument("Native weight source window has invalid useful byte geometry");
        return rows * native_row_bytes;
    }
};
} // namespace llaminar2
