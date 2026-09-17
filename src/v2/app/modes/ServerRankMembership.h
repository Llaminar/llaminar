/**
 * @file ServerRankMembership.h
 * @brief Startup-only projection of canonical MPI membership into server evidence.
 *
 * A hostname or rank count cannot prove cross-host inference: multiple ranks
 * may share one physical host, and containers may reuse hostnames. The context's
 * immutable cluster inventory owns physical shared-memory membership. This
 * adapter publishes that observation without discovering hardware again,
 * performing a collective, or becoming an execution/placement authority.
 */
#pragma once

#include "execution/mpi_orchestration/DeviceInventory.h"
#include "utils/PerfStatsCollector.h"
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Describe one server participant in its actual execution communicator.
     * @param inventory Immutable inventory owned by that exact MPI context.
     * @param rank Communicator-local observing rank, not a discovery-world alias.
     * @param authority_rank Communicator-local HTTP/request authority.
     * @return Bounded tags for the existing once-per-rank startup record.
     * @throws std::invalid_argument If membership is absent or inconsistent.
     *
     * Node IDs come from MPI shared-memory groups, not hostname comparison or
     * NUMA numbering. Hostnames are retained only to join diagnostic artifacts
     * to independently observed cloud endpoints. Consumers still require live
     * work and transport witnesses; this record alone proves no computation.
     */
    inline PerfStatsCollector::Tags serverRankMembershipTags(
        const ClusterInventory &inventory, int rank, int authority_rank)
    {
        if (inventory.world_size <= 0 ||
            inventory.ranks.size() != static_cast<size_t>(inventory.world_size) ||
            rank < 0 || rank >= inventory.world_size ||
            authority_rank < 0 || authority_rank >= inventory.world_size)
            throw std::invalid_argument("Server evidence requires exact execution membership");
        const auto &participant = inventory.ranks.at(rank);
        if (participant.rank != rank || participant.node_id < 0 ||
            participant.node_id >= inventory.node_count || participant.local_rank < 0 ||
            participant.hostname.empty() ||
            participant.hostname.find_first_of("\r\n") != std::string::npos ||
            participant.hostname.find('\0') != std::string::npos)
            throw std::invalid_argument("Server evidence requires physical MPI rank identity");
        return {{"rank", std::to_string(rank)},
                {"world_size", std::to_string(inventory.world_size)},
                {"authority_rank", std::to_string(authority_rank)},
                {"node_id", std::to_string(participant.node_id)},
                {"local_rank", std::to_string(participant.local_rank)},
                {"hostname", participant.hostname},
                {"identity_source", "communicator_cluster_inventory"}};
    }
}
