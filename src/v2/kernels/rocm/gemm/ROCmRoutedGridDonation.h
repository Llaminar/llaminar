/**
 * @file ROCmRoutedGridDonation.h
 * @brief Describe the host view of lookup storage donated by the admitted HC prefix.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace llaminar2::rocm {
/** @brief Whole native lookup entry counts in the retired prefix arena. */
struct RoutedGridDonation {
    unsigned iq2_entries; ///< 64-bit IQ2_S entries preceding all 32-bit tables.
    unsigned iq3_small_entries; ///< 32-bit IQ3_S entries.
    unsigned iq3_extra_small_entries; ///< 32-bit IQ3_XXS entries.
};

/**
 * @brief Inspect routed lookup placement using the original native geometry calculation.
 * @param routed_formats Sealed routed gate/up physical decoder mask.
 * @param prefix_bytes Bytes already admitted to the earlier HC phase.
 * @param hidden_bytes Existing live hidden Q8 prefix.
 * @param panel_and_records_bytes Existing live coefficient panel and role records.
 * @return Entries that fit without expanding the admitted prefix or shrinking live storage.
 */
RoutedGridDonation inspectRoutedGridDonation(std::uint32_t routed_formats, unsigned prefix_bytes,
    std::size_t hidden_bytes, std::size_t panel_and_records_bytes) noexcept;
}
