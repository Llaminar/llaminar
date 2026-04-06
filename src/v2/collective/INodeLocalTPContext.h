/**
 * @file INodeLocalTPContext.h
 * @brief Interface for NODE_LOCAL tensor parallelism (cross-rank, same node)
 *
 * NODE_LOCAL TP = multiple MPI ranks on the same physical machine participating
 * in tensor parallelism. This sits between LOCAL TP (intra-rank) and GLOBAL TP
 * (cross-node) in the scope hierarchy.
 *
 * Communication uses high-bandwidth intra-node interconnects:
 * - UPI (Intel Ultra Path Interconnect) for CPU sockets
 * - Cross-process NCCL/RCCL for GPUs on the same node
 * - Shared memory for low-latency small transfers
 *
 * Example topology using NODE_LOCAL TP:
 *
 *   NodeLocalPipelineParallel(
 *       LocalTP(0:cuda:0, 0:cuda:1, 0:cuda:2, 0:cuda:3),   // rank 0's GPUs
 *       LocalTP(1:cuda:0, 1:cuda:1, 1:cuda:2, 1:cuda:3),   // rank 1's GPUs
 *       NodeLocalTP(0:cpu, 1:cpu)                            // both ranks' CPUs
 *   )
 *
 * The NodeLocalTP domain connects rank 0's CPU and rank 1's CPU into a single
 * TP domain on the same physical machine, communicating via UPI or MPI locally.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "ITPContext.h"
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Interface for NODE_LOCAL tensor parallelism across MPI ranks on the same node
     *
     * Extends ITPContext for cross-rank TP within a single physical machine.
     * Participants are identified by their MPI rank, but communication stays
     * node-local, enabling higher bandwidth than cross-node GlobalTP.
     *
     * This is an interface stub — concrete implementation will be added when
     * the node-local TP backend (UPI/shared-memory/cross-process NCCL) is built.
     *
     * Thread safety: Single thread should call methods on a given instance.
     * Multiple instances (different domains) can be used concurrently.
     */
    class INodeLocalTPContext : public ITPContext
    {
    public:
        ~INodeLocalTPContext() override = default;

        /**
         * @brief NODE_LOCAL TP is cross-rank but same-node
         * @return Always TPScope::NODE_LOCAL
         */
        TPScope scope() const override { return TPScope::NODE_LOCAL; }

        // =====================================================================
        // Node-Local Specific Configuration (stub)
        // =====================================================================

        /**
         * @brief Get the node identifier for this TP domain
         *
         * All participants in a NODE_LOCAL TP domain are on the same node.
         * This returns the node hostname or a numeric node ID.
         *
         * @return Node identifier string
         */
        virtual const std::string &nodeId() const = 0;

        /**
         * @brief Get the MPI world ranks participating in this domain
         *
         * These are the global MPI ranks (in MPI_COMM_WORLD) of the
         * participants in this NODE_LOCAL TP domain.
         *
         * @return Vector of world ranks, ordered by domain index
         */
        virtual const std::vector<int> &worldRanks() const = 0;
    };

} // namespace llaminar2
