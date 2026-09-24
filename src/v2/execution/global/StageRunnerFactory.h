/**
 * @file StageRunnerFactory.h
 * @brief Factory for building per-domain stage runners in a Global PP topology.
 *
 * Phase 3 of Multi-Domain Pipeline Execution Plan.
 *
 * Builds the right IInferenceRunner for a given GlobalPPStageSpec/RankStageAction,
 * dispatching to:
 *   - createPPStageRunner            for concrete single-device PP stages
 *   - createTestableInferenceRunner  for model-light single-device tests
 *   - createRankOrchestrator         for local TP stages (via RankOrchestrator)
 *   - createInferenceRunner          for concrete global/NodeTP stages
 *                                   with an injected IGlobalTPContext
 *   - createTestableInferenceRunner  for model-light global/NodeTP tests
 *
 * Each stage retains its exact layer scope and the model-owned additive
 * PreparedWeightStore. Runtime policies use the same canonical projection as
 * ordinary inference, and a local TP runner owns exactly one TP context.
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#pragma once

#include "GlobalOrchestrator.h"
#include "../global_pp/GlobalPPTopology.h"
#include "../global_pp/GlobalPPRankPlan.h"
#include "../../interfaces/IModelContext.h"
#include "../../interfaces/IMPIContext.h"
#include "../factory/InferenceRunnerFactory.h"
#include "DomainCommunicatorRegistry.h"

#include <memory>

namespace llaminar2
{

    // =========================================================================
    // StageBuildContext
    // =========================================================================

    /**
     * @brief Aggregated context for building one stage runner.
     *
     * Passed to StageRunnerFactory::create().  Fields may be nullptr/empty for
     * topologies that don't require them (e.g., domain_registry is only needed
     * for global TP stages).
     */
    struct StageBuildContext
    {
        /// Model metadata and weights (required)
        std::shared_ptr<IModelContext> model_ctx;

        /// MPI context (optional; nullptr for unit tests without MPI)
        std::shared_ptr<IMPIContext> mpi_ctx;

        /// Canonical policy, projected through the same factories as ordinary
        /// execution. A stage must not invent its own subset of runtime fields.
        RuntimeConfig runtime;

        /// One model-owned additive binding namespace, shared by every stage.
        std::shared_ptr<PreparedWeightStore> prepared_weight_store;

        /// Registry of domain communicators for global TP stages.
        /// May be nullptr; if so, global TP stage building will throw.
        DomainCommunicatorRegistry *domain_registry = nullptr;
    };

    // =========================================================================
    // StageRunnerFactory
    // =========================================================================

    /**
     * @brief Factory for per-domain pipeline stage runners.
     *
     * All methods are static. Callers hold the returned StageRunnerEntry; its
     * runner owns local TP, while the entry retains a borrowed global TP owner.
     */
    class StageRunnerFactory
    {
    public:
        /**
         * @brief Create a StageRunnerEntry for this rank's participation in a stage.
         *
         * Dispatches based on stage.is_global_tp and action.inner_mode:
         *   - SINGLE_DEVICE, !is_global_tp: PP stage runner or testable fallback
         *   - LOCAL_TP / devices.size()>1, !is_global_tp: RankOrchestrator local TP runner
         *   - is_global_tp=true: runner with injected GlobalTPContext
         *
         * @param stage  Cluster-wide stage specification
         * @param action This rank's role and layer range for the stage (must be EXECUTE)
         * @param ctx    Build context
         * @return Populated StageRunnerEntry with non-null runner
         * @throws std::invalid_argument if action.role is IDLE
         * @throws std::runtime_error if a required dependency is missing or runner creation fails
         */
        static StageRunnerEntry create(
            const GlobalPPStageSpec &stage,
            const RankStageAction &action,
            const StageBuildContext &ctx);

    private:
        // -------------------------------------------------------------------------
        // Internal dispatch helpers
        // -------------------------------------------------------------------------

        /** @brief Bind one stage's policy, layer range and shared weight authority. */
        static StageRunnerEntry buildSingleDevice(
            const GlobalPPStageSpec &stage,
            const RankStageAction &action,
            const StageBuildContext &ctx,
            const FactoryPPStageConfig &pp_cfg);

        /** @brief Build a local stage whose rank runner owns its sole TP context. */
        static StageRunnerEntry buildLocalTP(
            const GlobalPPStageSpec &stage,
            const RankStageAction &action,
            const StageBuildContext &ctx,
            const FactoryPPStageConfig &pp_cfg);

        /** @brief Bind the registry's exact cross-rank TP context to a stage. */
        static StageRunnerEntry buildGlobalTP(
            const GlobalPPStageSpec &stage,
            const RankStageAction &action,
            const StageBuildContext &ctx,
            const FactoryPPStageConfig &pp_cfg);

        /// Convert RankStageAction layer range to FactoryPPStageConfig.
        /// Note: GlobalPP last_layer is inclusive; FactoryPPStageConfig last_layer is exclusive.
        static FactoryPPStageConfig makePPStageConfig(const RankStageAction &action);
    };

} // namespace llaminar2
