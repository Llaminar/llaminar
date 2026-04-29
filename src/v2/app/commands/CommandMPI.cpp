/**
 * @file CommandMPI.cpp
 * @brief Shared MPI lifecycle helper — implementation.
 */

#include "app/commands/CommandMPI.h"
#include "app/MPIBootstrapPhase.h"
#include "app/MPIShutdown.h"
#include "backends/ComputeBackend.h"
#include "config/OrchestrationConfig.h"
#include "utils/Logger.h"
#include "utils/MPIContext.h"

#include <mpi.h>
#include <string>
#include <vector>

namespace llaminar2
{

    // ================================================================
    // CommandMPISession — RAII MPI lifecycle
    // ================================================================

    CommandMPISession::~CommandMPISession()
    {
        if (mpi_initialized_)
        {
            mpiShutdown();
        }
    }

    MPI_Comm CommandMPISession::communicator() const
    {
        return mpi_initialized_ ? mpi_ctx_->communicator() : MPI_COMM_NULL;
    }

    CommandMPISession::CommandMPISession(CommandMPISession &&other) noexcept
        : inventory(std::move(other.inventory)),
          is_output_rank(other.is_output_rank),
          mpi_initialized_(other.mpi_initialized_),
          mpi_ctx_(std::move(other.mpi_ctx_))
    {
        other.mpi_initialized_ = false;
    }

    CommandMPISession &CommandMPISession::operator=(CommandMPISession &&other) noexcept
    {
        if (this != &other)
        {
            if (mpi_initialized_)
                mpiShutdown();

            inventory = std::move(other.inventory);
            is_output_rank = other.is_output_rank;
            mpi_initialized_ = other.mpi_initialized_;
            mpi_ctx_ = std::move(other.mpi_ctx_);
            other.mpi_initialized_ = false;
        }
        return *this;
    }

    // ================================================================
    // CommandMPI::bootstrap — two-path lifecycle
    // ================================================================

    std::pair<CommandMPISession, std::optional<int>>
    CommandMPI::bootstrap(const Params &params)
    {
        CommandMPISession session;

        // Initialize DeviceManager with no NUMA filter so we see all devices
        DeviceManager::instance().initialize(-1, false);

        if (params.no_mpi_bootstrap)
        {
            // ==========================================================
            // Local-only path: skip MPI entirely, enumerate local devices
            // ==========================================================
            session.inventory = gatherClusterInventory(nullptr, {}, params.hostfile);
            session.is_output_rank = true;
            return {std::move(session), std::nullopt};
        }

        // ==============================================================
        // Full MPI path: bootstrap → init → allgather
        //
        // SubcommandRouter strips the subcommand from argv before calling
        // the command, so we re-inject it so the self-launched mpirun
        // child process routes back to the correct command.
        // ==============================================================
        OrchestrationConfig orch_config;
        orch_config.mpi_no_bootstrap = false;
        orch_config.hostfile = params.hostfile;

        // Rebuild full argv: [binary, subcommand, ...remaining flags]
        std::string subcmd_str(params.subcommand);
        std::vector<char *> full_argv;
        full_argv.push_back(params.argv[0]);
        full_argv.push_back(subcmd_str.data());
        for (int i = 1; i < params.argc; ++i)
            full_argv.push_back(params.argv[i]);
        int full_argc = static_cast<int>(full_argv.size());

        MPIBootstrapPhase bootstrap_phase;
        auto bs_result = bootstrap_phase.execute(orch_config, full_argc, full_argv.data());
        if (bs_result.action == BootstrapResult::Action::EXIT)
            return {std::move(session), bs_result.exit_code};

        // MPI_Init_thread — session destructor will call MPI_Finalize
        int provided;
        int argc_copy = params.argc;
        char **argv_copy = params.argv;
        MPI_Init_thread(&argc_copy, &argv_copy, MPI_THREAD_MULTIPLE, &provided);
        session.mpi_initialized_ = true;

        session.mpi_ctx_ = MPIContextFactory::global();
        Logger::getInstance().setRank(session.mpi_ctx_->rank());
        session.is_output_rank = (session.mpi_ctx_->rank() == 0);

        // Gather full cluster inventory via MPI_Allgatherv
        session.inventory = gatherClusterInventory(session.mpi_ctx_, {}, params.hostfile);

        return {std::move(session), std::nullopt};
    }

} // namespace llaminar2
