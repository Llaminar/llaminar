/**
 * @file CommandMPI.cpp
 * @brief Shared command MPI lifecycle and canonical inventory handoff.
 *
 * Bootstrap resolves process membership before device discovery. Each child
 * observes its own hardware through ClusterInventoryGatherer; the initiating
 * host never manufactures remote participants or filters them by its NUMA map.
 */

#include "app/commands/CommandMPI.h"
#include "app/MPIBootstrapPhase.h"
#include "app/RuntimeInitPhase.h"
#include "backends/ComputeBackend.h"
#include "config/OrchestrationConfig.h"
#include "execution/runner/RankInitializationLifecycle.h"
#include "utils/Logger.h"
#include "utils/MPIContext.h"

#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{

    // ================================================================
    // CommandMPISession — RAII MPI lifecycle
    // ================================================================

    MPI_Comm CommandMPISession::communicator() const
    {
        return is_mpi() ? mpi_ctx_->communicator() : MPI_COMM_NULL;
    }

    std::shared_ptr<IMPIContext> CommandMPISession::context() const
    {
        return is_mpi() ? mpi_ctx_ : nullptr;
    }

    const ClusterInventory &CommandMPISession::inventory() const
    {
        if (!inventory_) throw std::logic_error("Command inventory is unavailable before discovery completes");
        return *inventory_;
    }

    // ================================================================
    // CommandMPI::bootstrap — two-path lifecycle
    // ================================================================

    std::pair<CommandMPISession, std::optional<int>>
    CommandMPI::bootstrap(const Params &params)
    {
        CommandMPISession session;

        if (params.no_mpi_bootstrap)
        {
            if (!params.hostfile.empty())
                throw std::invalid_argument(
                    "A cluster hostfile requires MPI bootstrap; local-only inventory cannot certify it");
            // ==========================================================
            // Local-only path: skip MPI entirely, enumerate local devices
            // ==========================================================
            const OrchestrationConfig &request = params.request ? *params.request : OrchestrationConfig{};
            // Local planning still materializes backend-owned projection
            // workspaces. Keep its pre-inventory transition identical to the
            // MPI command path rather than silently measuring raw host memory.
            RuntimeInitPhase::initializeDiscoveryRuntime(request, 0, 1);
            session.inventory_ = gatherClusterInventory(nullptr, params.hostfile);
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
        // Planning supplies serving's parsed policy so bootstrap preserves its
        // hard backend constraints and launch geometry. Describe has no model
        // request and deliberately inventories the unrestricted cluster.
        OrchestrationConfig orch_config = params.request ? *params.request : OrchestrationConfig{};
        orch_config.mpi_no_bootstrap = false;
        orch_config.hostfile = params.hostfile;

        // Rebuild full argv: [binary, subcommand, ...remaining flags]
        std::string subcmd_str(params.subcommand);
        std::vector<char *> full_argv;
        full_argv.push_back(params.argv[0]);
        full_argv.push_back(subcmd_str.data());
        for (int i = 1; i < params.argc; ++i)
            full_argv.push_back(params.argv[i]);
        full_argv.push_back(nullptr);
        int full_argc = static_cast<int>(full_argv.size() - 1);

        MPIBootstrapPhase bootstrap_phase;
        auto bs_result = bootstrap_phase.execute(orch_config, full_argc, full_argv.data());
        if (bs_result.action == BootstrapResult::Action::EXIT)
            return {std::move(session), bs_result.exit_code};

        // The same owner used by serving retires MPI after all command owners.
        int argc_copy = params.argc;
        char **argv_copy = params.argv;
        session.mpi_session_.emplace(MPIProcessSession::initialize(argc_copy, argv_copy));

        session.mpi_ctx_ = MPIContextFactory::global();
        Logger::getInstance().setRank(session.mpi_ctx_->rank());
        session.is_output_rank = (session.mpi_ctx_->rank() == 0);

        // Plan and serve share this exact rank-local backend/inventory setup.
        // Publish every local failure before the inventory all-gather so a
        // missing CPU backend cannot strand a peer in planning samples.
        const auto prepared = RankInitializationLifecycle::execute({0, "command_discovery_runtime"}, [&] {
            RuntimeInitPhase::initializeDiscoveryRuntime(
                orch_config, session.mpi_ctx_->rank(), session.mpi_ctx_->world_size());
            return true;
        }, [&](auto identity, auto outcome) {
            return MPIRankInitializationConsensus::reach(session.communicator(), identity, outcome);
        });
        if (!prepared.succeeded())
            throw std::runtime_error(prepared.diagnostic("Command initialization"));

        // Retain the publication, not a detached copy of the same hardware facts.
        session.inventory_ = gatherClusterInventory(session.mpi_ctx_, params.hostfile);

        return {std::move(session), std::nullopt};
    }

} // namespace llaminar2
