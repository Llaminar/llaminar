/**
 * @file GDNRecurrenceStateLayout.h
 * @brief One immutable physical-format authority for persistent GDN matrices.
 *
 * Request/head order, FP32 representation and byte capacity do not distinguish
 * these layouts. A layout is fixed for a bank's lifetime and participates in
 * cache/request compatibility. This type allocates no storage and performs no
 * conversion. Short-convolution history is not a recurrence matrix.
 */
#pragma once
#include <cstdint>
#include <string_view>
namespace llaminar2
{
/** @brief Physical inner-axis order of each complete persistent GDN head. */
enum class GDNRecurrenceStateLayout : std::uint8_t
{
    KeyMajorValueContiguous = 0, ///< Existing state[key * DV + value].
    ValueMajorKeyContiguous = 1 ///< Explicit transposed state[value * DK + key].
};
/** @brief Reject unknown/corrupt layout words before storage or payload access. */
[[nodiscard]] constexpr bool validGDNRecurrenceStateLayout(GDNRecurrenceStateLayout layout) noexcept
{
    return layout == GDNRecurrenceStateLayout::KeyMajorValueContiguous ||
        layout == GDNRecurrenceStateLayout::ValueMajorKeyContiguous;
}
/** @brief Stable compatibility spelling, never a user-selectable runtime mode. */
[[nodiscard]] constexpr std::string_view gdnRecurrenceStateLayoutName(GDNRecurrenceStateLayout layout) noexcept
{
    switch (layout)
    {
    case GDNRecurrenceStateLayout::KeyMajorValueContiguous: return "key_major_value_contiguous";
    case GDNRecurrenceStateLayout::ValueMajorKeyContiguous: return "value_major_key_contiguous";
    }
    return {};
}
}
