/**
 * @file MoEOverlayNodeLocalActivationLayout.h
 * @brief Canonical physical layout for one node-local ExpertOverlay channel.
 *
 * The shared activation transport and physical-memory admission consume this
 * same immutable value.  It is the sole authority for mapping size, every
 * captured byte offset, and the NUMA first-touch ownership of each page.  The
 * layout contains no MPI or device state and may therefore be constructed
 * before any allocation or graph capture occurs.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace llaminar2
{
    /** @brief One half-open byte range within a shared channel mapping. */
    struct MoEOverlayNodeLocalActivationByteRange
    {
        std::size_t offset = 0;
        std::size_t bytes = 0;

        /** @return Exclusive end offset, valid after planner overflow checks. */
        [[nodiscard]] std::size_t end() const noexcept
        {
            return offset + bytes;
        }

        /** @return Whether this range is non-empty and inside @p mapping_bytes. */
        [[nodiscard]] bool validFor(std::size_t mapping_bytes) const noexcept
        {
            return bytes != 0u && offset <= mapping_bytes &&
                   bytes <= mapping_bytes - offset;
        }

        bool operator==(
            const MoEOverlayNodeLocalActivationByteRange &) const = default;
    };

    /** @brief Captured offsets for one participant's dispatch/return payloads. */
    struct MoEOverlayNodeLocalActivationParticipantLayout
    {
        MoEOverlayNodeLocalActivationByteRange dispatch_pages;
        std::size_t row_ids = 0;
        std::size_t entry_offsets = 0;
        std::size_t expert_ids = 0;
        std::size_t route_weights = 0;
        std::size_t original_route_slots = 0;
        std::size_t compact_route_slots = 0;
        std::size_t hidden_rows = 0;
        MoEOverlayNodeLocalActivationByteRange return_pages;
        std::size_t return_row_ids = 0;
        std::size_t output_rows = 0;

        bool operator==(
            const MoEOverlayNodeLocalActivationParticipantLayout &) const =
            default;
    };

    /** @brief Geometry from which the exact shared mapping is derived. */
    struct MoEOverlayNodeLocalActivationLayoutGeometry
    {
        std::size_t participant_count = 0;
        std::size_t max_rows_per_participant = 0;
        std::size_t max_entries_per_participant = 0;
        int d_model = 0;
        std::size_t activation_graph_family_count = 0;

        /** @return Whether all dimensions can describe a production channel. */
        [[nodiscard]] bool valid() const noexcept
        {
            return participant_count != 0u &&
                   max_rows_per_participant != 0u &&
                   max_entries_per_participant != 0u && d_model > 0 &&
                   activation_graph_family_count != 0u;
        }


        bool operator==(
            const MoEOverlayNodeLocalActivationLayoutGeometry &) const =
            default;
    };

    /**
     * @brief Exact immutable shared-memory layout and page ownership contract.
     *
     * The ranges form a disjoint partition of @ref mapping_bytes. Metadata and
     * return pages are first-touched by the continuation/source rank; dispatch
     * pages are first-touched by the follower/target rank. This deterministic
     * ownership is important on hosts where NUMA placement is controlled only
     * by first touch.
     */
    struct MoEOverlayNodeLocalActivationLayout
    {
        MoEOverlayNodeLocalActivationLayoutGeometry geometry;
        std::size_t page_size = 0;
        std::size_t mapping_bytes = 0;
        MoEOverlayNodeLocalActivationByteRange metadata_pages;
        std::size_t dispatch_publication = 0;
        std::size_t return_publication = 0;
        std::size_t participant_controls = 0;
        std::size_t activation_controls = 0;
        MoEOverlayNodeLocalActivationByteRange shared_dispatch_pages;
        std::size_t shared_dispatch_hidden_rows = 0;
        MoEOverlayNodeLocalActivationByteRange shared_return_pages;
        std::size_t shared_return_route_rows = 0;
        std::vector<MoEOverlayNodeLocalActivationParticipantLayout>
            participants;

        /** @return Physical pages owned by the continuation/source NUMA node. */
        [[nodiscard]] std::size_t sourceOwnedBytes() const noexcept;

        /** @return Physical pages owned by the follower/target NUMA node. */
        [[nodiscard]] std::size_t targetOwnedBytes() const noexcept;

        /** @return Whether all offsets and ownership ranges are self-consistent. */
        [[nodiscard]] bool valid() const noexcept;

        bool operator==(
            const MoEOverlayNodeLocalActivationLayout &) const = default;
    };

    /**
     * @brief Resolve the sole physical layout for one node-local channel.
     * @param geometry Complete participant, row, entry, and family dimensions.
     * @return Page-partitioned mapping layout consumed by admission and runtime.
     * @throws std::invalid_argument for incomplete geometry or page properties.
     * @throws std::overflow_error when any allocation arithmetic exceeds size_t.
     */
    [[nodiscard]] MoEOverlayNodeLocalActivationLayout
    planMoEOverlayNodeLocalActivationLayout(
        const MoEOverlayNodeLocalActivationLayoutGeometry &geometry);
} // namespace llaminar2
