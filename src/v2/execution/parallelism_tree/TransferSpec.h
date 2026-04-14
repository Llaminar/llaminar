/**
 * @file TransferSpec.h
 * @brief Transfer mechanism derivation for adjacent PP stage subtrees
 *
 * TransferSpec describes how activation data should be transferred between
 * two adjacent pipeline-parallel stages. The mechanism is derived automatically
 * from the subtree topology:
 *
 * - **LOCAL_PP**: Same MPI rank, different devices. Uses NCCL/RCCL/HOST
 *   backend depending on device types.
 *
 * - **MPI_INTRAHOST**: Different MPI ranks on the same machine. Uses shared
 *   memory or UPI fabric for fast intra-node transfer.
 *
 * - **MPI_INTERHOST**: Different machines. Uses InfiniBand/Ethernet via MPI.
 *
 * The derive() function inspects the leaf devices of two subtrees (the "from"
 * and "to" nodes) and determines the appropriate mechanism.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "ParallelismTree.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Transfer specification between two adjacent PP stage subtrees
     *
     * Describes the mechanism, ranks, and tag for transferring activation
     * data between pipeline stages. Used by PipelineRunner to set up the
     * appropriate send/recv or copy operations.
     */
    struct TransferSpec
    {
        /**
         * @brief Transfer mechanism based on topology
         */
        enum class Mechanism
        {
            LOCAL_PP,      ///< Same MPI rank, different devices (NCCL/RCCL/HOST)
            MPI_INTRAHOST, ///< Same machine, different MPI ranks (UPI/shared memory)
            MPI_INTERHOST  ///< Different machines (InfiniBand/Ethernet)
        };

        Mechanism mechanism = Mechanism::LOCAL_PP;

        int sender_rank = -1;   ///< MPI rank that sends (-1 if same-rank local)
        int receiver_rank = -1; ///< MPI rank that receives (-1 if same-rank local)
        int mpi_tag = 0;        ///< Unique tag for MPI transfers

        CollectiveBackendType local_backend = CollectiveBackendType::AUTO; ///< For LOCAL_PP: which backend

        // =====================================================================
        // Factory Method
        // =====================================================================

        /**
         * @brief Derive transfer mechanism from two subtrees
         *
         * Inspects the leaf devices of `from` and `to` nodes to determine:
         * 1. If any ranks overlap → LOCAL_PP (same-rank transfer)
         * 2. Else if any hostnames match → MPI_INTRAHOST
         * 3. Else → MPI_INTERHOST
         *
         * For cross-rank transfers:
         * - sender_rank = max of from's leafRanks (any would work, max is deterministic)
         * - receiver_rank = min of to's leafRanks
         *
         * @param from The source subtree (sends activation)
         * @param to The destination subtree (receives activation)
         * @param tag_base Base tag for MPI (used as-is)
         * @return TransferSpec with mechanism and rank/tag info
         */
        static TransferSpec derive(const ParallelismNode &from,
                                   const ParallelismNode &to,
                                   int tag_base);

        // =====================================================================
        // Serialization
        // =====================================================================

        /**
         * @brief Convert mechanism enum to string
         * @return Human-readable mechanism name
         */
        static const char *mechanismName(Mechanism m);

        /**
         * @brief Human-readable string representation
         * @return Multi-line description of the transfer spec
         */
        std::string toString() const;

        // =====================================================================
        // Comparison
        // =====================================================================

        /**
         * @brief Equality comparison for testing
         */
        bool operator==(const TransferSpec &other) const;
        bool operator!=(const TransferSpec &other) const { return !(*this == other); }
    };

} // namespace llaminar2
