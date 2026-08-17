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
            RoutedExpertComputePolicy::Apportioned;
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
     * local whole-expert TP receives a one-tier plan whose sole tier has integer
     * priority zero and complete-coverage responsibility. Dynamic/observe modes
     * retain a histogram-capable residency policy, while off remains a static
     * epoch authority that can prove no durable movement.
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

} // namespace llaminar2
