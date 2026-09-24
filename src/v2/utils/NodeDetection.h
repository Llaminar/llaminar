/**
 * @file NodeDetection.h
 * @brief Canonical physical-node identity for MPI initialization.
 *
 * MPI shared-memory membership determines physical reachability. Hostfile
 * aliases and processor names remain labels only. This contract is shared by
 * cluster inventory, MPI topology and global collective domain construction.
 */
#pragma once
#include <mpi.h>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Immutable rank-to-node projection for one communicator. */
    struct NodeDetectionResult
    {
        std::vector<int> node_ids; ///< Compact physical node ID for each rank.
        int node_count = 0; ///< Number of participating physical nodes only.
        std::vector<std::string> hostnames; ///< Diagnostic processor labels.
    };

    /** @brief One physical-node identity authority for inventory and collectives. */
    class NodeDetection
    {
    public:
        /**
         * @brief Collectively discover actual shared-memory rank groups.
         * @param comm Communicator whose every rank enters this operation.
         * @param hostfile_path Launch provenance only; never a second node map.
         * @return Identical node IDs and diagnostic hostnames on every rank.
         * @throws std::runtime_error for an MPI discovery failure.
         */
        static NodeDetectionResult detect(MPI_Comm comm, const std::string &hostfile_path = "");

        /**
         * @brief Validate and compact an authenticated shared-memory leader map.
         * @param leaders Lowest communicator rank in each rank's shared group.
         * @return Compact node IDs ordered by first participating rank.
         * @throws std::invalid_argument for negative, forward or inconsistent leaders.
         */
        static NodeDetectionResult fromSharedNodeLeaders(const std::vector<int> &leaders);

        /**
         * @brief Build synthetic fixture node IDs without touching MPI.
         * @param hostnames Fixture-owned labels in communicator rank order.
         * @return First-appearance grouping; not proof of physical reachability.
         */
        static NodeDetectionResult fromHostnames(const std::vector<std::string> &hostnames);
    };
} // namespace llaminar2
