/**
 * @file AppContext.h
 * @brief Shared application state passed to execution modes
 */

#pragma once

#include "config/OrchestrationConfig.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "utils/Tokenizer.h"
#include "utils/MPIContext.h"
#include "app/MPIShutdown.h"
#include "utils/Assertions.h"
#include <cstdint>
#include <memory>

namespace llaminar2
{

    /**
     * @brief Process role in the coordinated request-command lifecycle.
     *
     * The authority owns tokenization, request admission, terminal output, and
     * command publication. Every follower remains in the runner's command loop
     * and enters the same graph/collective sequence when admitted by that
     * authority. The role is independent of MPI world-rank numbering.
     */
    enum class CoordinatedRequestRole : std::uint8_t
    {
        Authority, ///< Inventory-selected continuation and command owner.
        Follower,  ///< Participant driven by the authority's command stream.
    };

    /**
     * @brief Shared state produced by RuntimeInitPhase, consumed by execution modes
     */
    struct AppContext
    {
        OrchestrationConfig config;
        std::shared_ptr<IMPIContext> mpi_ctx;
        std::unique_ptr<IOrchestrationRunner> runner;
        std::shared_ptr<ITokenizer> tokenizer;

        /**
         * @brief Resolve this process's immutable request-command role.
         *
         * The orchestration runner is the sole authority for choosing the
         * coordinated root because it owns the resolved execution topology. In
         * particular, a heterogeneous ExpertOverlay continuation domain may
         * live on any MPI rank; application modes must never infer ownership
         * from world rank zero.
         *
         * @return Authority for the selected continuation root, otherwise
         *         Follower.
         */
        [[nodiscard]] CoordinatedRequestRole coordinatedRequestRole() const
        {
            if (!mpi_ctx || !runner)
            {
                LLAMINAR_UNREACHABLE(
                    "Coordinated request role requires initialized MPI and runner authorities");
            }
            const int authority_rank = runner->coordinatedRootRank();
            if (authority_rank < 0 ||
                authority_rank >= mpi_ctx->world_size())
            {
                LLAMINAR_UNREACHABLE(
                    "Coordinated request authority rank " << authority_rank
                    << " is outside MPI world size " << mpi_ctx->world_size());
            }
            return mpi_ctx->rank() == authority_rank
                       ? CoordinatedRequestRole::Authority
                       : CoordinatedRequestRole::Follower;
        }

        /** @brief Shut down the runner and finalize MPI for this process. */
        void finalize()
        {
            if (runner)
                runner->shutdown();
            mpiShutdown();
        }
    };

} // namespace llaminar2
