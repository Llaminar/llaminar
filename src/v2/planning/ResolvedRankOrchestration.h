/**
 * @file ResolvedRankOrchestration.h
 * @brief One model-aware topology resolution boundary for planning and execution.
 *
 * The existing ExecutionPlanBuilder compiles participant/shard geometry. This
 * boundary composes it with the existing ExpertOverlay normalizer and role
 * resolver, so frontends cannot install a different topology from the runner.
 * It performs no discovery, MPI, weight loading, allocation, or memory admission.
 * A resolved topology is an input to PhysicalMemoryAuthority, not a reservation.
 */
#pragma once

#include "config/OrchestrationConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/moe/MoEExpertOverlayAuthorityPlan.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include <optional>

namespace llaminar2
{
    class IExecutionPlanBuilder;
    class PlanningModelMetadata;
    struct ClusterInventory;

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
         * @return Validated topology with no resources materialized.
         * @throws std::invalid_argument for invalid topology, inventory or policy.
         * @throws std::logic_error for an unimplemented explicit execution form.
         */
        [[nodiscard]] static ResolvedRankOrchestration resolve(
            const OrchestrationConfig &requested,
            const PlanningModelMetadata &model,
            const ClusterInventory &inventory,
            IExecutionPlanBuilder &builder,
            int rank);

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
    };
}
