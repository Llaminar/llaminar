/**
 * @file MoEExpertOverlayAuthorityPlan.h
 * @brief Canonical ExpertOverlay authority plan for implicit multi-device MoE.
 *
 * Explicit routed-expert placement and ordinary LocalTP historically entered
 * different durable-ownership implementations. This normalizer closes that
 * configuration gap: once model metadata proves that a graph contains routed
 * experts, an ordinary multi-device whole-expert domain is represented as a
 * one-tier ExpertOverlay plan before weights, arenas, or graphs are built.
 */

#pragma once

#include "MoERoutedExpertPlacementPlan.h"
#include "execution/factory/FactoryPPStageConfig.h"

#include <memory>
#include <vector>

namespace llaminar2
{
    /** @brief Result category emitted by authority-plan normalization. */
    enum class MoEExpertOverlayAuthorityPlanDisposition
    {
        NotApplicable,
        ExplicitPlan,
        SynthesizedLocalTP,
        SynthesizedPipeline,
    };

    /**
     * @brief Hardware and policy facts needed to normalize one model plan.
     *
     * The caller supplies the already-resolved preliminary execution topology.
     * This prevents the MoE layer from repeating device discovery or guessing
     * which MPI rank owns a local accelerator. Cross-rank and pipeline flags
     * are explicit so unsupported shapes fail before the legacy authority can
     * accidentally remain active.
     */
    struct MoEExpertOverlayAuthorityPlanRequest
    {
        std::shared_ptr<MoERoutedExpertPlacementPlan> requested_plan;
        bool model_has_routed_experts = false;
        int world_rank = 0;
        std::vector<GlobalDeviceAddress> local_tp_participants;
        std::vector<float> local_tp_weights;
        CollectiveBackendType local_tp_backend =
            CollectiveBackendType::AUTO;
        bool has_cross_rank_tensor_parallel = false;
        bool has_pipeline_parallel = false;
        RoutedExpertComputePolicy routed_compute_policy =
            RoutedExpertComputePolicy::Automatic;
        RoutedExpertPhasePolicy routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        /** Independent decode/grouped-verifier and ordinary-prefill scheduling. */
        RoutedExpertWorkloadAssignmentPolicy routed_assignment_policy{
            RoutedExpertAssignmentPolicy::StaticOwner, RoutedExpertAssignmentPolicy::StaticOwner};
        RoutedExpertOwnerOrder owner_order =
            RoutedExpertOwnerOrder::Ordinal;
        MoERebalanceRuntimeMode residency_maintenance =
            MoERebalanceRuntimeMode::Dynamic;
    };

    /** @brief Immutable normalized authority-plan result. */
    struct MoEExpertOverlayAuthorityPlanResult
    {
        std::shared_ptr<MoERoutedExpertPlacementPlan> plan;
        MoEExpertOverlayAuthorityPlanDisposition disposition =
            MoEExpertOverlayAuthorityPlanDisposition::NotApplicable;

        /** @return Whether normalization produced a new implicit plan. */
        [[nodiscard]] bool synthesized() const noexcept
        {
            return disposition ==
                   MoEExpertOverlayAuthorityPlanDisposition::
                       SynthesizedLocalTP;
        }
    };

    /**
     * @brief Normalize explicit or implicit multi-device MoE into ExpertOverlay.
     *
     * An enabled explicit plan is already authoritative, whether it declares
     * `SingleDomain` or `TieredOverlay`. Without one, a routed MoE model using
     * local multi-device MoE receives a one-tier plan whose sole tier has integer
     * priority zero and complete-coverage responsibility. Dynamic/observe modes
     * retain a histogram-capable residency policy, while off remains a static
     * epoch authority that can prove no durable movement.
     * Automatic compute uses movable gate/up and fixed down columns only for
     * homogeneous native GPU LocalTP; CPU/heterogeneous participants retain
     * whole experts. Explicit physical choices are never reinterpreted.
     *
     * @param request Fully resolved topology and user policy.
     * @return Existing, synthesized, or inapplicable authority plan.
     * @throws std::invalid_argument when a multi-device routed model requests a
     *         weight representation that the whole-expert epoch cannot encode.
     * @throws std::logic_error when a cross-rank or PP implicit topology would
     *         otherwise fall through to the replaced legacy authority.
     */
    MoEExpertOverlayAuthorityPlanResult
    normalizeMoEExpertOverlayAuthorityPlan(
        const MoEExpertOverlayAuthorityPlanRequest &request);

    /**
     * @brief Synthesize the expert authority of one already projected PP stage.
     * @param request Exact participants and policy of the child, without parent
     *        pipeline/global-TP flags or a model-wide requested expert plan.
     * @param scope Owned main-model interval and global edge roles, authenticated
     *        against complete model metadata by the enclosing topology compiler.
     * @return One stage-owned plan, including the single-participant case.
     * @throws std::invalid_argument for missing participants, invalid scope,
     *         inherited parent topology or an unsupported compute policy.
     *
     * This performs topology normalization only. It neither allocates memory nor
     * publishes a runnable stage; joint admission and runtime sealing follow.
     */
    [[nodiscard]] std::shared_ptr<MoERoutedExpertPlacementPlan>
    normalizeMoEExpertOverlayPipelineStagePlan(
        const MoEExpertOverlayAuthorityPlanRequest &request,
        const FactoryPPStageConfig &scope);

    /**
     * @brief Normalize one original cross-rank CPU pipeline domain without making it LocalTP.
     * @param request Stage policies and complete original participants; cross-rank TP is explicit.
     * @param scope Exact model interval and global component roles.
     * @param domain Canonical topology compiler's complete original CPU domain/rank declaration.
     * @return Stage-owned host authority preserving every real MPI participant and authored policy.
     * @throws std::invalid_argument for foreign, duplicated, non-CPU or non-node-local ownership.
     * Runtime publication still requires its admitted original CPU collective and stage services.
     */
    [[nodiscard]] std::shared_ptr<MoERoutedExpertPlacementPlan>
    normalizeMoEExpertOverlayPipelineStagePlan(
        const MoEExpertOverlayAuthorityPlanRequest &request,
        const FactoryPPStageConfig &scope, const ExecutionDomainDefinition &domain);

} // namespace llaminar2
