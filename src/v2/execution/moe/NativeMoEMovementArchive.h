/**
 * @file NativeMoEMovementArchive.h
 * @brief Terminal, read-only archive of native GPU placement publications.
 *
 * The device journal is the sole writer of movement facts. This archive only
 * authenticates completed request snapshots and projects their immutable
 * identities into the public diagnostic ledger. It never plans placement,
 * reads live GPU memory, or decides when inference/maintenance may execute.
 */
#pragma once

#include "DeviceMoERebalanceMovementJournal.h"
#include "MoEOptimizationStatus.h"
#include <span>

namespace llaminar2
{
    /** Frozen participant identity, resolved from the admitted model plan. */
    struct NativeMoEMovementParticipant
    {
        DeviceId device;
        int world_rank = -1;
        int priority = 0;
    };

    /** Exact immutable geometry of one native homogeneous publication owner. */
    struct NativeMoEMovementArchiveConfig
    {
        std::uint64_t workspace_generation = 0;
        std::uint32_t layers = 0;
        std::uint32_t experts = 0;
        std::uint32_t wave_capacity = 0;
        std::uint32_t edge_capacity = 0;
        std::vector<NativeMoEMovementParticipant> participants;
    };

    /**
     * @brief Bounded model-lifetime evidence, independent of optional telemetry.
     *
     * The terminal reader must join the device publisher before calling
     * observe(). A newer request may reset the device prefix but cannot reuse
     * a model-lifetime candidate epoch. A repeated observation must retain every
     * earlier byte-semantic record. Exhaustion is explicit and never overwrites
     * old evidence. Callers serialize access at the existing request boundary.
     */
    class NativeMoEMovementArchive final
    {
    public:
        /**
         * @brief Freeze the canonical arena and participant geometry.
         * @param config Admitted native domain; no inferred ranks or capacities.
         * @throws std::invalid_argument for incomplete/non-homogeneous geometry.
         */
        explicit NativeMoEMovementArchive(NativeMoEMovementArchiveConfig config);

        /**
         * @brief Authenticate and archive one completed device snapshot atomically.
         * @param session_epoch Existing request/ticket lifetime, not a new counter.
         * @param workspace_generation Exact generation of the captured buffers.
         * @param state Completed journal prefix copied after publisher completion.
         * @param waves Exactly the populated immutable wave range.
         * @param edges Exactly the populated immutable edge range.
         * @return First newly retained public edge index (for optional mirrors).
         * @throws std::invalid_argument on stale, malformed or mutated evidence.
         *
         * Rejection leaves all accepted history intact. Zero-capacity or corrupt
         * observations cannot masquerade as a Static journal. Allocation is
         * terminal diagnostic work, never part of a captured inference path.
         */
        std::size_t observe(std::uint64_t session_epoch, std::uint64_t workspace_generation,
            const DeviceMoERebalanceMovementJournalState &state,
            std::span<const DeviceMoERebalanceMovementWave> waves,
            std::span<const DeviceMoERebalanceMovementEdge> edges);

        /** @return Immutable diagnostic history; no device access or progress. */
        [[nodiscard]] const MoEOptimizationMovementLedger &ledger() const noexcept { return ledger_; }
        /** @return Completed facts from retained device receipts, never PerfStats. */
        [[nodiscard]] const MoEOptimizationMovementTotals &totals() const noexcept { return totals_; }

    private:
        /**
         * @brief Validate a complete snapshot before allocating or accepting it.
         * @param session_epoch Existing request owner generation.
         * @param workspace_generation Captured arena generation, frozen at construction.
         * @param state Completed prefix and explicit loss counts.
         * @param waves Exactly the committed wave range.
         * @param edges Exactly the committed edge range.
         * @throws std::invalid_argument on malformed, stale or changed history.
         */
        void validate(std::uint64_t session_epoch, std::uint64_t workspace_generation,
            const DeviceMoERebalanceMovementJournalState &state,
            std::span<const DeviceMoERebalanceMovementWave> waves,
            std::span<const DeviceMoERebalanceMovementEdge> edges) const;

        NativeMoEMovementArchiveConfig config_;
        std::uint64_t session_epoch_ = 0;
        std::uint64_t last_model_epoch_ = 0;
        DeviceMoERebalanceMovementJournalState previous_state_{};
        std::vector<DeviceMoERebalanceMovementWave> previous_waves_;
        std::vector<DeviceMoERebalanceMovementEdge> previous_edges_;
        MoEOptimizationMovementLedger ledger_;
        MoEOptimizationMovementTotals totals_;
    };
}
