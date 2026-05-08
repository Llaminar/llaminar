/**
 * @file DomainCommunicatorRegistry.h
 * @brief Registry of domain-specific GlobalTPContext instances for all global TP stages.
 *
 * Phase 3 of Multi-Domain Pipeline Execution Plan.
 *
 * Every MPI rank must call initialize() with the full topology so that
 * MPI_Comm_split is called collectively for every global TP stage in ascending
 * stage_id order.  Non-participants use MPI_UNDEFINED as the color and get no
 * stored context.  Participants get a shared_ptr<IGlobalTPContext> keyed by
 * stage_id.
 *
 * Acceptance guarantee: no stage creates a communicator over ranks outside
 * its domain.  A rank that is not listed in stage.participating_ranks cannot
 * receive a context for that stage.
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#pragma once

#include "../global_pp/GlobalPPTopology.h"
#include "../../collective/IGlobalTPContext.h"

#include <map>
#include <memory>
#include <vector>
#include <mpi.h>

namespace llaminar2
{

    /**
     * @brief Registry of domain-scoped GlobalTPContext instances.
     *
     * One instance per MPI rank.  Contexts are created during initialize() and
     * can be retrieved later by StageRunnerFactory when building global-TP stage
     * runners.
     */
    class DomainCommunicatorRegistry
    {
    public:
        DomainCommunicatorRegistry() = default;

        // Non-copyable (contexts own MPI communicators)
        DomainCommunicatorRegistry(const DomainCommunicatorRegistry &) = delete;
        DomainCommunicatorRegistry &operator=(const DomainCommunicatorRegistry &) = delete;

        DomainCommunicatorRegistry(DomainCommunicatorRegistry &&) = default;
        DomainCommunicatorRegistry &operator=(DomainCommunicatorRegistry &&) = default;

        // =========================================================================
        // Initialization
        // =========================================================================

        /**
         * @brief Initialize the registry from the topology.
         *
         * MUST be called collectively on every rank in world_comm.  Processes all
         * global TP stages in ascending stage_id order; calls MPI_Comm_split for
         * each one.  Participants receive a context; non-participants do not.
         *
         * @param topology   Cluster-wide stage layout
         * @param world_comm Base MPI communicator (typically MPI_COMM_WORLD)
         * @param world_rank This rank's ordinal in world_comm
         */
        void initialize(const GlobalPPTopology &topology,
                        MPI_Comm world_comm,
                        int world_rank);

        // =========================================================================
        // Query
        // =========================================================================

        /** @brief True if this rank has a context for the given stage. */
        bool hasContextForStage(int stage_id) const;

        /**
         * @brief Get the GlobalTPContext for the given stage, or nullptr.
         *
         * Returns nullptr if this rank does not participate in stage_id.
         */
        std::shared_ptr<IGlobalTPContext> globalTPContextForStage(int stage_id) const;

        /** @brief Stage IDs for which this rank has a context. */
        std::vector<int> stageIds() const;

        // =========================================================================
        // Test helpers
        // =========================================================================

        /**
         * @brief Inject a pre-created context for testing.
         *
         * Bypasses MPI_Comm_split so unit tests can verify factory behavior
         * without a real multi-rank MPI session.  Do not use in production.
         */
        void addContextForTest(int stage_id, std::shared_ptr<IGlobalTPContext> ctx);

    private:
        std::map<int, std::shared_ptr<IGlobalTPContext>> contexts_;
    };

} // namespace llaminar2
