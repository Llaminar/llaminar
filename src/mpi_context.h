/**
 * @file mpi_context.h
 * @brief MPI runtime context capturing rank, size, and communicator.
 * @author David Sanftenberg
 *
 * Eliminates repeated MPI_Comm_rank/size queries in hot paths by capturing
 * MPI state once at initialization and passing through pipeline constructors.
 */
#pragma once

#include <mpi.h>
#include <string>

namespace llaminar
{
    /**
     * @brief MPI runtime context.
     *
     * Captures MPI rank, size, and communicator to avoid repeated queries.
     * Future extensions may include NUMA node mapping.
     */
    struct MPIContext
    {
        int rank = 0;                   ///< MPI rank (0-based)
        int size = 1;                   ///< Total number of MPI processes
        MPI_Comm comm = MPI_COMM_WORLD; ///< MPI communicator
        // Future: int numa_node = -1;  ///< NUMA node for this rank (-1 if not bound)

        /**
         * @brief Default constructor (single-process context).
         */
        MPIContext() = default;

        /**
         * @brief Explicit constructor.
         *
         * @param r MPI rank
         * @param s MPI size
         * @param c MPI communicator (defaults to MPI_COMM_WORLD)
         */
        explicit MPIContext(int r, int s, MPI_Comm c = MPI_COMM_WORLD)
            : rank(r), size(s), comm(c) {}

        /**
         * @brief Capture current MPI context from the given communicator.
         *
         * @param comm MPI communicator to query (defaults to MPI_COMM_WORLD)
         * @return MPIContext with populated rank and size
         *
         * This is the primary factory method for creating an MPIContext.
         * Call once during initialization and pass to pipeline constructors.
         */
        static MPIContext capture(MPI_Comm comm = MPI_COMM_WORLD)
        {
            MPIContext ctx;
            ctx.comm = comm;
            MPI_Comm_rank(comm, &ctx.rank);
            MPI_Comm_size(comm, &ctx.size);
            return ctx;
        }

        /**
         * @brief Check if running in distributed mode (size > 1).
         */
        bool isDistributed() const { return size > 1; }

        /**
         * @brief Check if this is the root rank.
         */
        bool isRoot() const { return rank == 0; }

        /**
         * @brief String representation for logging.
         */
        std::string toString() const
        {
            return "MPIContext(rank=" + std::to_string(rank) +
                   ", size=" + std::to_string(size) + ")";
        }
    };

} // namespace llaminar
