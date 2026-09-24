/**
 * @file ExecutionRankMembership.h
 * @brief Exact discovery-to-execution rank mapping, without device discovery.
 *
 * Rank and node ordinals belong to a communicator; physical NUMA IDs, GPU
 * ordinals and UUIDs do not. Admission projects one immutable observation into
 * the selected namespace while preserving all physical endpoint facts. This
 * is topology, never a physical-memory reservation or capacity ledger.
 */
#pragma once
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "config/OrchestrationPlanningPolicy.h"
#include <optional>

namespace llaminar2
{
    /** @brief Validated ordered subset of one discovery rank namespace. */
    class ExecutionRankMembership final
    {
    public:
        /**
         * @brief Validate membership and project the existing observation.
         * @param discovery Exact rank-indexed discovery inventory.
         * @param selected Discovery ranks in desired execution rank order.
         * @throws std::invalid_argument for empty, duplicate or invalid membership.
         */
        ExecutionRankMembership(const ClusterInventory &discovery, std::vector<int> selected);

        /** @return Discovery ranks indexed by execution rank. */
        const std::vector<int> &discoveryRanks() const noexcept { return selection_.discoveryRanks(); }
        /** @return Saveable selection; applying it still needs fresh inventory admission. */
        const ExecutionRankSelection &selection() const noexcept { return selection_; }
        /** @return Enclosing discovery size, retained when sealing a reusable launch. */
        int discoverySize() const noexcept { return static_cast<int>(execution_ranks_.size()); }
        /** @return Projected observation; no hardware query takes place. */
        const ClusterInventory &inventory() const noexcept { return inventory_; }
        /**
         * @brief Map a discovery rank to active membership, or explicit absence.
         * @param discovery_rank Rank in the original discovery namespace.
         * @throws std::out_of_range for an invalid discovery rank.
         */
        std::optional<int> executionRank(int discovery_rank) const;

    private:
        ExecutionRankSelection selection_; ///< One ordered membership value.
        std::vector<std::optional<int>> execution_ranks_; ///< Inverse membership.
        ClusterInventory inventory_; ///< Same observations, selected namespace.
    };
}
