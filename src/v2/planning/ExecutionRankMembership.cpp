/**
 * @file ExecutionRankMembership.cpp
 * @brief Pure namespace projection for selected execution participants.
 *
 * Compact communicator-local rank/node ordinals are recomputed together.
 * Physical device identity and affinity are copied unchanged, including sparse
 * NUMA IDs and nonzero accelerator ordinals. Node summaries use the existing
 * canonical aggregation routine, not a second capacity arithmetic path.
 */
#include "planning/ExecutionRankMembership.h"
#include <map>
#include <stdexcept>

namespace llaminar2
{
    ExecutionRankMembership::ExecutionRankMembership(
        const ClusterInventory &discovery, std::vector<int> selected)
        : selection_(std::move(selected))
    {
        if (discovery.world_size <= 0 ||
            discovery.ranks.size() != static_cast<size_t>(discovery.world_size))
            throw std::invalid_argument("Execution membership requires exact discovery inventory");
        for (int rank = 0; rank < discovery.world_size; ++rank)
            if (discovery.ranks[rank].rank != rank || discovery.ranks[rank].node_id < 0)
                throw std::invalid_argument("Execution membership has invalid discovery rank/node identity");
        const auto &discovery_ranks = selection_.discoveryRanks();
        if (discovery_ranks.size() > discovery.ranks.size())
            throw std::invalid_argument("Execution membership requires a nonempty rank subset");
        execution_ranks_.resize(discovery.world_size);
        inventory_.world_size = selection_.size();
        std::map<int, int> nodes;
        std::map<int, int> local_ranks;
        for (int rank = 0; rank < inventory_.world_size; ++rank)
        {
            const int original = discovery_ranks[rank];
            if (original < 0 || original >= discovery.world_size || execution_ranks_[original])
                throw std::invalid_argument("Execution membership contains an absent or repeated discovery rank");
            execution_ranks_[original] = rank;
            auto observed = discovery.ranks[original];
            // Node IDs are logical membership labels, not OS NUMA IDs. Removing
            // a node cannot leave empty phantom nodes in the selected topology.
            const auto [node, inserted] = nodes.try_emplace(observed.node_id, static_cast<int>(nodes.size()));
            (void)inserted;
            observed.rank = rank;
            observed.node_id = node->second;
            observed.local_rank = local_ranks[observed.node_id]++;
            inventory_.ranks.push_back(std::move(observed));
        }
        inventory_.buildNodeAggregations();
    }

    std::optional<int> ExecutionRankMembership::executionRank(int discovery_rank) const
    {
        return execution_ranks_.at(discovery_rank);
    }
}
