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

#include "GraphSnapshotSourceOutput.h"
#include "ServingSnapshotArenaInventory.h"
#include <cstddef>
#include <vector>
#include <optional>

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
        /** Ordinary reference payload capacity; explicitly selected live state prefixes are excluded. */
        std::size_t per_accelerator_bytes = 0u;

        /** Optional main phase payloads projected over exact retained snapshot owners. */
        std::optional<MainSnapshotPhasePayload> main_phase_payload;
        /** Actual sharing/dedicated serving lanes; distinct from native executable counts. */
        std::optional<ServingSnapshotArenaInventory> serving_arenas;

        /** Source-owned authored outputs; live state payload is excluded from the reference envelope. */
        std::vector<GraphSnapshotSourceOutput> authored_source_outputs;
        /** Existing retained forward arena bound; setup resolves it before native state pricing. */
        std::size_t retained_forward_arena_count = 0u;
        /** Main forward owners only: Dense0 sidecar graphs never own main QSA state snapshots. */
        std::size_t retained_main_forward_arena_count = 0u;

        /**
         * @brief Additional cache-read diagnostics absent from reference files.
         *
         * The planner resolves this selection against the actual FA layer,
         * TP-head, request, context and retained-graph inventories. These are
         * full-capacity banks, not prompt-row tensors. Their bytes are charged
         * to the same GraphSnapshotArena owner as the declared checkpoints.
         */
        struct EffectiveKV
        {
            /** Global model layer, or all full-attention layers when absent. */
            std::optional<int> layer;
            /**
             * Upper bound on independently retained forward snapshot arenas.
             * Supplied by the serving-family inventory, not by token counts.
             * Zero is incomplete and rejected by memory planning.
             */
            std::size_t retained_arena_count = 0u;

            friend constexpr bool operator==(
                const EffectiveKV &, const EffectiveKV &) = default;
        };

        /** Absent means no effective-K/V payloads enter the captured topology. */
        std::optional<EffectiveKV> effective_kv;

        /** @return Whether the declaration names a usable non-empty capacity. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return per_accelerator_bytes != 0u || !authored_source_outputs.empty() ||
                (main_phase_payload && (main_phase_payload->wide_main_bytes != 0u ||
                                        main_phase_payload->scalar_main_bytes != 0u));
        }

        friend constexpr bool operator==(
            const GraphSnapshotMemoryCapacity &,
            const GraphSnapshotMemoryCapacity &) = default;
    };
} // namespace llaminar2
