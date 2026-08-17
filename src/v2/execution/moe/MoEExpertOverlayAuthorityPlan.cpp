/**
 * @file MoEExpertOverlayAuthorityPlan.cpp
 * @brief Implementation of universal multi-device MoE authority normalization.
 *
 * The generated plan intentionally contains no model-layer placement rows.
 * Model-aware freezing later resolves exact layer/expert geometry and physical
 * capacity, preserving the normal production setup order.
 */

#include "MoEExpertOverlayAuthorityPlan.h"

#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr const char *kImplicitDomainName =
            "implicit_moe_local_tp";
        constexpr const char *kImplicitTierName =
            "priority_0";

        /** @return Residency policy matching the durable maintenance intent. */
        RoutedExpertResidencyPolicy residencyPolicyFor(
            MoERebalanceRuntimeMode mode) noexcept
        {
            return mode == MoERebalanceRuntimeMode::Off
                       ? RoutedExpertResidencyPolicy::StaticById
                       : RoutedExpertResidencyPolicy::
                             RoutedTierRebalanced;
        }

        /** @brief Build the canonical one-domain, one-tier LocalTP plan. */
        std::shared_ptr<MoERoutedExpertPlacementPlan>
        buildImplicitLocalTPPlan(
            const MoEExpertOverlayAuthorityPlanRequest &request)
        {
            auto plan =
                std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology =
                RoutedExpertPlacementTopology::SingleDomain;
            plan->continuation_domain = kImplicitDomainName;
            plan->base_model_domain = kImplicitDomainName;
            plan->shared_expert_domain = kImplicitDomainName;
            plan->residency_policy =
                residencyPolicyFor(request.residency_maintenance);
            plan->owner_order = request.owner_order;

            RoutedExpertDomain domain;
            domain.name = kImplicitDomainName;
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.backend = request.local_tp_backend;
            domain.participants = request.local_tp_participants;
            domain.weights = request.local_tp_weights;
            domain.owner_rank = request.world_rank;
            domain.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            domain.routed_phase_policy =
                RoutedExpertPhasePolicy::Uniform;
            domain.routed_decode_assignment_policy =
                RoutedExpertAssignmentPolicy::StaticOwner;
            domain.routed_prefill_assignment_policy =
                RoutedExpertAssignmentPolicy::StaticOwner;
            plan->domains.push_back(std::move(domain));

            RoutedExpertTier tier;
            tier.name = kImplicitTierName;
            tier.domain = kImplicitDomainName;
            tier.priority = 0;
            tier.fallback = true;
            plan->routed_tiers.push_back(std::move(tier));

            /*
             * The ordinary LocalTP dense/shared trunk remains tensor parallel.
             * Expert apportionment is an independent axis and is represented by
             * the domain above; it must not silently replicate dense weights.
             */
            plan->continuation_domain_spec.domain =
                kImplicitDomainName;
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->continuation_domain_spec.setDensePolicy(
                DenseParallelPolicy::TensorParallel);
            plan->continuation_domain_spec.hidden_layout =
                MoEContinuationActivationLayout::ReplicatedHidden;
            plan->authority_execution =
                resolveMoEOverlayAuthorityExecutionKind(*plan);
            return plan;
        }

        /** @brief Clone and seal a topology-derived authority backend. */
        std::shared_ptr<MoERoutedExpertPlacementPlan>
        freezeAuthorityExecution(
            const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan)
        {
            if (!plan || !plan->usesExpertOverlayAuthority())
                return plan;

            const auto required =
                resolveMoEOverlayAuthorityExecutionKind(*plan);
            if (plan->authority_execution !=
                    MoEOverlayAuthorityExecutionKind::Unresolved &&
                plan->authority_execution != required)
            {
                throw std::invalid_argument(
                    "ExpertOverlay authority execution '" +
                    std::string(toString(plan->authority_execution)) +
                    "' conflicts with topology-required '" +
                    std::string(toString(required)) + "'");
            }
            if (plan->authority_execution == required)
                return plan;

            auto frozen =
                std::make_shared<MoERoutedExpertPlacementPlan>(*plan);
            frozen->authority_execution = required;
            return frozen;
        }
    } // namespace

    MoEExpertOverlayAuthorityPlanResult
    normalizeMoEExpertOverlayAuthorityPlan(
        const MoEExpertOverlayAuthorityPlanRequest &request)
    {
        if (request.requested_plan &&
            request.requested_plan->usesExpertOverlayAuthority())
        {
            return {
                .plan = freezeAuthorityExecution(request.requested_plan),
                .disposition =
                    MoEExpertOverlayAuthorityPlanDisposition::
                        ExplicitPlan,
            };
        }

        if (!request.model_has_routed_experts)
        {
            return {
                .plan = request.requested_plan,
                .disposition =
                    MoEExpertOverlayAuthorityPlanDisposition::
                        NotApplicable,
            };
        }

        const bool local_multi_device =
            request.local_tp_participants.size() > 1u;
        if (!local_multi_device &&
            !request.has_cross_rank_tensor_parallel &&
            !request.has_pipeline_parallel)
        {
            return {
                .plan = request.requested_plan,
                .disposition =
                    MoEExpertOverlayAuthorityPlanDisposition::
                        NotApplicable,
            };
        }

        if (request.has_cross_rank_tensor_parallel ||
            request.has_pipeline_parallel)
        {
            throw std::logic_error(
                "Implicit multi-device MoE authority normalization does not "
                "yet encode cross-rank TP or pipeline participants; provide "
                "one explicit ExpertOverlay domain plan so execution cannot "
                "fall through to the retired legacy residency authority");
        }
        if (request.routed_compute_policy !=
            RoutedExpertComputePolicy::Apportioned)
        {
            throw std::invalid_argument(
                "Implicit ExpertOverlay authority requires "
                "routed_compute=apportioned; replicated and tensor-sharded "
                "residency need a typed multi-owner epoch representation");
        }
        if (!request.local_tp_weights.empty() &&
            request.local_tp_weights.size() !=
                request.local_tp_participants.size())
        {
            throw std::invalid_argument(
                "Implicit ExpertOverlay LocalTP weights must match its "
                "participant count");
        }

        auto plan = buildImplicitLocalTPPlan(request);
        validateMoERoutedExpertPlacementPlanOrThrow(*plan);
        return {
            .plan = std::move(plan),
            .disposition =
                MoEExpertOverlayAuthorityPlanDisposition::
                    SynthesizedLocalTP,
        };
    }

} // namespace llaminar2
