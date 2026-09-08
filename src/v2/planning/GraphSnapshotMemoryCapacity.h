/**
 * @file GraphSnapshotMemoryCapacity.h
 * @brief Typed setup-time capacity for graph-resident diagnostic snapshots.
 *
 * Snapshot checkpoints are ordinary physical allocations even though they are
 * used only for diagnostics. This value lets a diagnostic caller declare the
 * maximum simultaneously live device storage before model admission. The
 * declaration is charged to @ref PhysicalMemoryOwner::GraphSnapshotArena and
 * later enforced by the live physical-memory ledger; it is never an anonymous
 * safety reserve or a substitute for runtime allocation attestation.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Maximum graph-snapshot backing retained on each accelerator.
     *
     * A capture policy may retain several native graph alternatives, but only
     * the arenas whose values can be live concurrently contribute to this
     * number. The policy owner must derive the bound from its complete
     * checkpoint inventory and maximum graph geometry. Every actual arena
     * obtains a suballocation lease from this capacity, so an incomplete
     * declaration fails before an unaccounted allocation can occur.
     */
    struct GraphSnapshotMemoryCapacity
    {
        std::size_t per_accelerator_bytes = 0u;

        /** @return Whether the declaration names a usable non-empty capacity. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return per_accelerator_bytes != 0u;
        }

        friend constexpr bool operator==(
            const GraphSnapshotMemoryCapacity &,
            const GraphSnapshotMemoryCapacity &) = default;
    };
} // namespace llaminar2
