/**
 * @file CanonicalMoERouteRecord.h
 * @brief Allocation-free packed records for ownership-invariant CPU MoE publication.
 *
 * Apportioned CPU experts produce a sparse subset of the router's original
 * `[row, route]` slots on each tensor-parallel participant.  Transporting a
 * dense zero-filled `[M, top_k, d_model]` tensor through an allreduce preserves
 * arithmetic identity, but wastes bandwidth in direct proportion to `top_k`.
 * This file defines the compact wire layout used by the economical path:
 *
 * ```text
 * record 0: expert_row[d_model], original_flat_route_slot
 * record 1: expert_row[d_model], original_flat_route_slot
 * ...
 * buffer trailer: local or gathered record count
 * ```
 *
 * The slot and count are stored as bit-exact `uint32_t` payloads inside FP32
 * storage.  They are metadata, never floating-point values.  Keeping records in
 * the graph-owned canonical publication tensor avoids hot-path allocation and
 * gives MPI one contiguous payload per participant.  The root reconstructs the
 * original route order from the slot metadata before applying router weights.
 */

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2::canonical_moe_route_record
{
    /** Number of FP32 elements reserved at the end of the publication buffer. */
    inline constexpr size_t kTrailerElements = 1;

    /**
     * @brief Return the packed width of one route record in FP32 elements.
     * @param d_model Number of expert-output elements in one route row.
     * @return `d_model + 1`, or zero for invalid geometry.
     */
    [[nodiscard]] constexpr size_t recordWidth(int d_model) noexcept
    {
        return d_model > 0 ? static_cast<size_t>(d_model) + 1u : 0u;
    }

    /**
     * @brief Return how many complete records fit while preserving the trailer.
     * @param buffer_elements Total FP32 capacity of the publication tensor.
     * @param d_model Number of expert-output elements in one route row.
     * @return Complete record capacity, excluding the count trailer.
     */
    [[nodiscard]] constexpr size_t recordCapacity(
        size_t buffer_elements,
        int d_model) noexcept
    {
        const size_t width = recordWidth(d_model);
        return width > 0 && buffer_elements > kTrailerElements
                   ? (buffer_elements - kTrailerElements) / width
                   : 0u;
    }

    /**
     * @brief Return the beginning of one mutable record.
     * @param buffer Packed publication storage.
     * @param record_index Zero-based record index.
     * @param d_model Expert-output width.
     * @return Pointer to the record's first expert-output element.
     */
    [[nodiscard]] inline float *record(
        float *buffer,
        size_t record_index,
        int d_model) noexcept
    {
        return buffer + record_index * recordWidth(d_model);
    }

    /** Const overload of @ref record. */
    [[nodiscard]] inline const float *record(
        const float *buffer,
        size_t record_index,
        int d_model) noexcept
    {
        return buffer + record_index * recordWidth(d_model);
    }

    /**
     * @brief Publish one original flat route-slot identity in a record trailer.
     * @param route_record Mutable packed route record.
     * @param d_model Expert-output width.
     * @param flat_route_slot Original `row * top_k + route` identity.
     * @return false if the identity cannot be represented by the wire format.
     */
    inline bool writeFlatRouteSlot(
        float *route_record,
        int d_model,
        size_t flat_route_slot) noexcept
    {
        if (!route_record || d_model <= 0 ||
            flat_route_slot >
                static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        {
            return false;
        }
        route_record[static_cast<size_t>(d_model)] =
            std::bit_cast<float>(static_cast<uint32_t>(flat_route_slot));
        return true;
    }

    /**
     * @brief Decode the original flat route-slot identity from one record.
     * @param route_record Packed route record.
     * @param d_model Expert-output width.
     * @return Exact unsigned route-slot identity.
     */
    [[nodiscard]] inline uint32_t readFlatRouteSlot(
        const float *route_record,
        int d_model) noexcept
    {
        return std::bit_cast<uint32_t>(
            route_record[static_cast<size_t>(d_model)]);
    }

    /**
     * @brief Publish the active record count into the fixed buffer trailer.
     * @param buffer Packed publication storage.
     * @param buffer_elements Total FP32 capacity of the publication tensor.
     * @param count Local or gathered record count.
     * @return false for invalid storage or a count outside the wire range.
     */
    inline bool writeRecordCount(
        float *buffer,
        size_t buffer_elements,
        size_t count) noexcept
    {
        if (!buffer || buffer_elements < kTrailerElements ||
            count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        {
            return false;
        }
        buffer[buffer_elements - 1u] =
            std::bit_cast<float>(static_cast<uint32_t>(count));
        return true;
    }

    /**
     * @brief Read the active record count from the fixed buffer trailer.
     * @param buffer Packed publication storage.
     * @param buffer_elements Total FP32 capacity of the publication tensor.
     * @return Published count, or zero for invalid storage.
     */
    [[nodiscard]] inline size_t readRecordCount(
        const float *buffer,
        size_t buffer_elements) noexcept
    {
        if (!buffer || buffer_elements < kTrailerElements)
            return 0u;
        return static_cast<size_t>(std::bit_cast<uint32_t>(
            buffer[buffer_elements - 1u]));
    }
} // namespace llaminar2::canonical_moe_route_record
