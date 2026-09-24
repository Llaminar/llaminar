/**
 * @file GDNSpeculativeWorkspaceContract.h
 * @brief Exact byte contract for retained GDN speculative-state workspace.
 *
 * Grouped MTP verification snapshots both short-convolution history and GDN
 * recurrence state after every candidate row.  Those buffers are persistent
 * members of the captured serial graph family: the graph stages allocate them
 * later, while model admission must price them before expert weights consume
 * VRAM.  This header is the single arithmetic authority used at both lifecycle
 * points so their byte counts cannot drift.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2::gdn_workspace
{

/**
 * @brief Raw byte footprint of one speculative-state buffer pair.
 *
 * The slot buffer retains one state image per verifier row.  The work buffer
 * holds one mutable state image per concurrently represented request.  Buffer
 * placement applies its own alignment to each member after this exact payload
 * calculation.
 */
struct SpeculativeStateFootprint
{
    std::size_t slot_bytes = 0; ///< Persistent row-indexed rollback images.
    std::size_t work_bytes = 0; ///< Mutable request-local working images.

    /** @brief Return the unaligned sum while rejecting size_t overflow. */
    [[nodiscard]] std::size_t totalBytes() const
    {
        if (work_bytes > std::numeric_limits<std::size_t>::max() - slot_bytes)
        {
            throw std::overflow_error(
                "GDN speculative workspace footprint exceeds size_t");
        }
        return slot_bytes + work_bytes;
    }
};

namespace detail
{

/** @brief Convert one validated positive geometry scalar to size_t. */
[[nodiscard]] inline std::size_t positiveSize(
    int value,
    std::string_view field)
{
    if (value <= 0)
    {
        throw std::invalid_argument(
            "GDN speculative workspace requires positive " +
            std::string(field));
    }
    return static_cast<std::size_t>(value);
}

/** @brief Checked multiplication used by every shared byte formula. */
[[nodiscard]] inline std::size_t checkedMultiply(
    std::size_t left,
    std::size_t right,
    std::string_view contribution)
{
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(
            "GDN speculative workspace overflow while multiplying " +
            std::string(contribution));
    }
    return left * right;
}

/**
 * @brief Expand one state image into row slots and request work buffers.
 * @param state_floats Number of FP32 values in one state image.
 * @param slot_rows Flattened verifier rows retained for rollback.
 * @param request_count Maximum request count represented concurrently.
 */
[[nodiscard]] inline SpeculativeStateFootprint footprintFromStateFloats(
    std::size_t state_floats,
    int slot_rows,
    int request_count)
{
    const std::size_t rows = positiveSize(slot_rows, "slot_rows");
    const std::size_t requests = positiveSize(request_count, "request_count");
    const std::size_t state_bytes = checkedMultiply(
        state_floats, sizeof(float), "state elements and FP32 bytes");
    return {
        .slot_bytes = checkedMultiply(
            rows, state_bytes, "verifier rows and state bytes"),
        .work_bytes = checkedMultiply(
            requests, state_bytes, "requests and state bytes"),
    };
}

} // namespace detail

/**
 * @brief Compute the exact GDN recurrence rollback footprint.
 * @param slot_rows Flattened verifier rows retained for rollback.
 * @param request_count Maximum request count represented concurrently.
 * @param value_heads Participant-local GDN value-head count.
 * @param key_width Per-head key-state width.
 * @param value_width Per-head value-state width.
 * @return Exact raw bytes for the persistent slot and mutable work buffers.
 */
[[nodiscard]] inline SpeculativeStateFootprint recurrenceStateFootprint(
    int slot_rows,
    int request_count,
    int value_heads,
    int key_width,
    int value_width)
{
    std::size_t state_floats = detail::positiveSize(
        value_heads, "value_heads");
    state_floats = detail::checkedMultiply(
        state_floats,
        detail::positiveSize(key_width, "key_width"),
        "value heads and key width");
    state_floats = detail::checkedMultiply(
        state_floats,
        detail::positiveSize(value_width, "value_width"),
        "recurrence key and value widths");
    return detail::footprintFromStateFloats(
        state_floats, slot_rows, request_count);
}

/**
 * @brief Compute the exact short-convolution rollback footprint.
 * @param slot_rows Flattened verifier rows retained for rollback.
 * @param request_count Maximum request count represented concurrently.
 * @param channels Participant-local fused Q/K/V channel count.
 * @param kernel_size Causal convolution kernel width.
 * @return Exact raw bytes for the persistent slot and mutable work buffers.
 */
[[nodiscard]] inline SpeculativeStateFootprint shortConvStateFootprint(
    int slot_rows,
    int request_count,
    int channels,
    int kernel_size)
{
    if (kernel_size <= 1)
    {
        throw std::invalid_argument(
            "GDN short-convolution speculative workspace requires kernel_size > 1");
    }
    const std::size_t state_floats = detail::checkedMultiply(
        detail::positiveSize(channels, "channels"),
        static_cast<std::size_t>(kernel_size - 1),
        "short-convolution channels and history");
    return detail::footprintFromStateFloats(
        state_floats, slot_rows, request_count);
}

} // namespace llaminar2::gdn_workspace
