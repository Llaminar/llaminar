/**
 * @file CommandMPI.h
 * @brief Shared MPI lifecycle helper for subcommands (describe, plan).
 *
 * Encapsulates the two-path pattern used by lightweight subcommands:
 *   - Local-only (--no-mpi-bootstrap): enumerate local devices, no MPI.
 *   - Full MPI: bootstrap → MPI_Init → allgather → MPI_Finalize.
 *
 * RAII wrapper calls MPI_Finalize in its destructor so callers don't need
 * to track it on every error path.
 */

#pragma once

#include "planning/ClusterInventoryGatherer.h"
#include "interfaces/IMPIContext.h"

#include <memory>
#include <optional>
#include <string>

namespace llaminar2
{
    class MPIContext;

    /**
     * @brief RAII result of CommandMPI::bootstrap().
     *
     * Owns the MPI lifecycle: if MPI was initialized, the destructor
     * calls MPI_Finalize(). Callers just use inventory/is_output_rank
     * and return — cleanup is automatic.
     */
    struct CommandMPISession
    {
        ClusterInventory inventory;
        bool is_output_rank = true;

        /// MPI communicator (MPI_COMM_WORLD if MPI active, MPI_COMM_NULL otherwise).
        /// Use for collective operations (broadcast, etc.) after bootstrap.
        MPI_Comm communicator() const;

        /// Whether MPI was initialized (multi-rank mode).
        bool is_mpi() const { return mpi_initialized_; }

        ~CommandMPISession();

        // Move-only (MPI_Finalize must run exactly once)
        CommandMPISession() = default;
        CommandMPISession(CommandMPISession &&) noexcept;
        CommandMPISession &operator=(CommandMPISession &&) noexcept;
        CommandMPISession(const CommandMPISession &) = delete;
        CommandMPISession &operator=(const CommandMPISession &) = delete;

    private:
        friend struct CommandMPI;
        bool mpi_initialized_ = false;
        std::shared_ptr<MPIContext> mpi_ctx_;
    };

    /**
     * @brief Shared MPI lifecycle for lightweight subcommands.
     *
     * Two modes:
     *   1. no_mpi_bootstrap=true  → local device enumeration, no MPI.
     *   2. no_mpi_bootstrap=false → MPIBootstrapPhase (may exec into mpirun),
     *      MPI_Init_thread, gatherClusterInventory via MPI_Allgatherv.
     *
     * Usage:
     * @code
     *   auto [session, rc] = CommandMPI::bootstrap({
     *       .subcommand = name(),
     *       .argc = argc, .argv = argv,
     *       .no_mpi_bootstrap = cfg.no_mpi_bootstrap,
     *       .hostfile = cfg.hostfile,
     *   });
     *   if (rc.has_value()) return *rc;  // bootstrap exited early
     *   // session.inventory, session.is_output_rank available
     *   // MPI_Finalize happens when session goes out of scope
     * @endcode
     */
    struct CommandMPI
    {
        struct Params
        {
            const char *subcommand; ///< e.g. "describe", "plan"
            int argc;
            char **argv;
            bool no_mpi_bootstrap;
            std::string hostfile;
        };

        /**
         * @brief Bootstrap MPI (or skip) and gather cluster inventory.
         *
         * @return pair of (session, optional exit code).
         *   If the optional has a value, the caller should return it immediately.
         *   Otherwise, the session is valid and MPI (if used) is initialized.
         */
        static std::pair<CommandMPISession, std::optional<int>>
        bootstrap(const Params &params);
    };

} // namespace llaminar2
