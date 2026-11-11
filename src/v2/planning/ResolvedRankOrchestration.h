/**
 * @file ResolvedRankOrchestration.h
 * @brief One model-aware topology resolution boundary for planning and execution.
 *
 * The existing ExecutionPlanBuilder compiles participant/shard geometry. This
 * boundary composes it with the existing ExpertOverlay normalizer and role
 * resolver, so frontends cannot install a different topology from the runner.
 * It performs no discovery, MPI, weight loading, allocation, or memory admission.
 * A resolved topology is an input to PhysicalMemoryAuthority, not a reservation.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#pragma once

#include "execution/global_pp/GlobalPPTopology.h"
#include "config/OrchestrationConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/moe/MoEExpertOverlayAuthorityPlan.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include <optional>
#include <utility>
#include <vector>

namespace llaminar2
{
    class IExecutionPlanBuilder;
    class PlanningModelMetadata;
    struct ClusterInventory;
    class ResolvedRankOrchestration;

    /**
     * @brief Immutable topology and BOM projection of one authored local PP stage.
     *
     * The parent remains the rank's execution plan. The child rank view supplies
     * owned layers and exact participants to memory planning; its missing global
     * edge roles describe a local pipeline boundary, never a fabricated MPI peer.
     * Only the production compiler can construct this metadata-only value.
     */
    class ResolvedMoEPipelineStage final
    {
    public:
        /** @return Original global stage ID; local PP uses its canonical ordered index. */
        [[nodiscard]] int stageId() const noexcept { return stage_id_; }
        /** @return Main-model interval and global embedding/output ownership. */
        [[nodiscard]] const FactoryPPStageConfig &scope() const noexcept { return scope_; }
        /** @return Stage-only normalized policy and expert domain declarations. */
        [[nodiscard]] const OrchestrationConfig &config() const noexcept { return config_; }
        /** @return Stage-owned memory-planning view inside the enclosing rank. */
        [[nodiscard]] const RankExecutionPlan &rankPlan() const noexcept { return rank_plan_; }
        /** @return Hardware-resolved expert role of this stage on the parent rank. */
        [[nodiscard]] const MoEExpertOverlayExecutionPlan &overlayExecution() const noexcept { return execution_; }

    private:
        friend class ResolvedRankOrchestration;
        /**
         * @brief Retain one compiler-authenticated scope and its exact projections.
         * @param scope Owned main interval and global component roles.
         * @param config Normalized child expert and serving policy.
         * @param rank_plan Child projection of the retained parent rank plan.
         * @param execution Bound child expert authority role on that rank.
         * @param stage_id Original canonical stage ID retained by the global/local pipeline.
         */
        ResolvedMoEPipelineStage(FactoryPPStageConfig scope, OrchestrationConfig config,
            RankExecutionPlan rank_plan, MoEExpertOverlayExecutionPlan execution, int stage_id)
            : stage_id_(stage_id), scope_(scope), config_(std::move(config)), rank_plan_(std::move(rank_plan)), execution_(std::move(execution)) {}
        int stage_id_ = -1;
        FactoryPPStageConfig scope_;
        OrchestrationConfig config_;
        RankExecutionPlan rank_plan_;
        MoEExpertOverlayExecutionPlan execution_;
    };

    /**
     * @brief Validated configuration and exact rank plan published as one result.
     *
     * Construction is private so a caller cannot publish a preliminary plan
     * with an unbound expert authority or a mismatched hardware MTP profile.
     * This type owns topology only; the existing runtime remains the owner of
     * mutable CPU/device state, events, prepared weights and physical bytes.
     */
    class ResolvedRankOrchestration final
    {
    public:
        /**
         * @brief Resolve model-aware placement through the production compiler.
         * @param requested User configuration, copied before normalization.
         * @param model Root-published real metadata or a device-free test descriptor.
         * @param inventory Exact immutable inventory for this rank namespace.
         * @param builder Canonical rank/shard compiler, also injectable by tests.
         * @param rank Rank in that inventory; negative or absent ranks are errors.
         * @param terminal_projection Ordinary diagnostic output demand, frozen before child projection.
         * @return Validated topology with no resources materialized.
         * @throws std::invalid_argument for invalid topology, inventory or policy.
         * @throws std::logic_error for an unimplemented explicit execution form.
         */
        [[nodiscard]] static ResolvedRankOrchestration resolve(
            const OrchestrationConfig &requested,
            const PlanningModelMetadata &model,
            const ClusterInventory &inventory,
            IExecutionPlanBuilder &builder,
            int rank,
            OrdinaryTerminalProjectionCapacity terminal_projection = {});

        /** @return Configuration after hardware/authority/MTP normalization. */
        [[nodiscard]] const OrchestrationConfig &config() const noexcept { return config_; }
        /** @return Exact continuation or expert-participant plan for this rank. */
        [[nodiscard]] const RankExecutionPlan &rankPlan() const noexcept { return rank_plan_; }
        /** @return Overlay roles when a model-owned overlay authority is installed. */
        [[nodiscard]] const std::optional<MoEExpertOverlayExecutionPlan> &overlayExecution() const noexcept
        { return overlay_execution_; }
        /** @return Explicit origin for passive diagnostics, never an execution decision. */
        [[nodiscard]] MoEExpertOverlayAuthorityPlanDisposition overlayOrigin() const noexcept
        { return overlay_origin_; }
        /** @return Original complete cross-rank topology, absent for rank-local PP. */
        [[nodiscard]] const std::optional<GlobalPPTopology> &globalPipeline() const noexcept
        { return global_pipeline_; }
        /** @return Ordered expert stages physically owned by this rank. */
        [[nodiscard]] const std::vector<ResolvedMoEPipelineStage> &pipelineStages() const noexcept
        { return pipeline_stages_; }

    private:
        /** @brief Only resolve() may seal a complete validated topology. */
        ResolvedRankOrchestration(
            OrchestrationConfig config, RankExecutionPlan rank_plan,
            std::optional<MoEExpertOverlayExecutionPlan> overlay_execution,
            MoEExpertOverlayAuthorityPlanDisposition origin);

        OrchestrationConfig config_;
        RankExecutionPlan rank_plan_;
        std::optional<MoEExpertOverlayExecutionPlan> overlay_execution_;
        MoEExpertOverlayAuthorityPlanDisposition overlay_origin_;
        std::vector<ResolvedMoEPipelineStage> pipeline_stages_;
        std::optional<GlobalPPTopology> global_pipeline_;
    };
}
