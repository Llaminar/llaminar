/**
 * @file NamedDomainGraphBuilder.h
 * @brief Physical named-domain graph construction inside the ordinary lifecycle.
 *
 * This builder owns no request, snapshot, model-loading, or admission state.
 * OrchestrationRunner supplies its validated plan and admitted model context;
 * the result enters the same snapshot installation and serving preparation
 * phases as a rank-local graph. Cross-rank transport remains explicit in the
 * GlobalOrchestrator, never hidden inside a participant compute stage.
 */
#pragma once

#include "../mpi_orchestration/IExecutionPlanBuilder.h"
#include "../local_execution/orchestrators/IInferenceRunner.h"

namespace llaminar2
{
    class ModelContext;
    class IMPIContext;

    /**
     * @brief Detect named pipeline ownership that crosses MPI ranks.
     * @param config Complete declarative topology, including stage owners.
     * @return Whether construction needs an explicit global graph.
     */
    bool requiresNamedDomainGlobalGraph(const OrchestrationConfig &config);

    /**
     * @brief Build physical stages after ordinary physical-memory admission.
     * @param config Validated declarative topology.
     * @param plan Canonical rank plan, including resolved runtime policy.
     * @param inventory Same hardware inventory used to build and admit the plan.
     * @param builder Topology authority used by ordinary planning.
     * @param model_ctx Already loaded, rank-owned model authority.
     * @param mpi_ctx Exact world context retained by the enclosing runner.
     * @return Global graph with participant-local stages and explicit transfers.
     * @throws std::invalid_argument for incomplete construction dependencies.
     * @throws std::runtime_error for invalid topology or stage construction.
     */
    std::unique_ptr<IInferenceRunner> buildNamedDomainGlobalGraph(
        const OrchestrationConfig &config,
        const RankExecutionPlan &plan,
        const ClusterInventory &inventory,
        IExecutionPlanBuilder &builder,
        const std::shared_ptr<ModelContext> &model_ctx,
        const std::shared_ptr<IMPIContext> &mpi_ctx);
}
