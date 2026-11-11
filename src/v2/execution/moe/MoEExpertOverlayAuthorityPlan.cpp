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
#include <set>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr const char *kImplicitDomainName =
            "implicit_moe_local_tp";
        constexpr const char *kImplicitTierName =
            "priority_0";

        /**
         * @brief Build the canonical one-domain, one-tier authority for its original declaration.
         * @param request Inventory-resolved participants and immutable user intent.
         * @param original_domain Explicit original global CPU stage, or absent for local synthesis.
         * @return One authority with a concrete compute mode and no model placements.
         * @throws std::invalid_argument for unresolved automatic domain geometry.
         */
        std::shared_ptr<MoERoutedExpertPlacementPlan>
        buildImplicitDomainPlan(
            const MoEExpertOverlayAuthorityPlanRequest &request,
            const ExecutionDomainDefinition *original_domain = nullptr)
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
                defaultRoutedExpertResidencyPolicy(
                    request.residency_maintenance);
            plan->owner_order = request.owner_order;

            RoutedExpertDomain domain;
            if (original_domain) domain = RoutedExpertDomain::fromExecutionDomainDefinition(*original_domain);
            else {
                domain.name = kImplicitDomainName;
                domain.scope = request.local_tp_participants.size() == 1u
                    ? ExecutionDomainScope::SINGLE : ExecutionDomainScope::RANK_LOCAL;
                domain.backend = request.local_tp_backend;
                domain.participants = request.local_tp_participants;
                domain.weights = request.local_tp_weights;
                domain.owner_rank = request.world_rank;
            }
            plan->continuation_domain = domain.name;
            plan->base_model_domain = domain.name;
            plan->shared_expert_domain = domain.name;
            domain.routed_compute_policy = request.routed_compute_policy;
            domain.routed_phase_policy = request.routed_phase_policy;
            domain.routed_decode_assignment_policy = request.routed_assignment_policy.decode;
            domain.routed_prefill_assignment_policy = request.routed_assignment_policy.prefill;
            // These are already resolved LocalTP participants, not selectors.
            // Use the same topology authority as authored/inventory-bound tiers.
            domain.routed_compute_policy = domain.toExecutionDomainDefinition()
                .resolveRoutedComputePolicy(/*routed_tier_count=*/1);
            plan->domains.push_back(std::move(domain));

            RoutedExpertTier tier;
            tier.name = kImplicitTierName;
            tier.domain = plan->domains.back().name;
            tier.priority = 0;
            tier.fallback = true;
            plan->routed_tiers.push_back(std::move(tier));

            /* The universal LocalTP authority uses the same phase-specific
             * dense default as an authored rank-local overlay. Physical-memory
             * admission accounts the complete decode replicas before any
             * graph or prepared weight can be materialized. */
            plan->continuation_domain_spec.domain = plan->domains.back().name;
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->continuation_domain_spec.setDensePolicy(
                defaultMoEContinuationDensePolicy(
                    plan->domains.back().scope,
                    request.local_tp_participants.size()));
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

        /**
         * @brief Reject a physical policy that cannot represent this owner epoch.
         * @param request Exact local participant and compute declaration.
         * @throws std::invalid_argument for unsupported compute or workshare geometry.
         */
        void requireImplicitPolicy(const MoEExpertOverlayAuthorityPlanRequest &request)
        {
            if (request.routed_compute_policy != RoutedExpertComputePolicy::Automatic &&
                request.routed_compute_policy != RoutedExpertComputePolicy::Apportioned &&
                request.routed_compute_policy != RoutedExpertComputePolicy::GateUpOwnedDownColumns)
                throw std::invalid_argument(
                    "Implicit ExpertOverlay authority requires expert_compute=auto, apportioned or gate-up-owned-down-columns; "
                    "replicated and tensor-sharded residency need a typed multi-owner epoch representation");
            if (!request.local_tp_weights.empty() &&
                request.local_tp_weights.size() != request.local_tp_participants.size())
                throw std::invalid_argument("Implicit ExpertOverlay LocalTP weights must match its participant count");
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
        requireImplicitPolicy(request);

        auto plan = buildImplicitDomainPlan(request);
        validateMoERoutedExpertPlacementPlanOrThrow(*plan);
        return {
            .plan = std::move(plan),
            .disposition =
                MoEExpertOverlayAuthorityPlanDisposition::
                    SynthesizedLocalTP,
        };
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan>
    normalizeMoEExpertOverlayPipelineStagePlan(
        const MoEExpertOverlayAuthorityPlanRequest &request,
        const FactoryPPStageConfig &scope)
    {
        if (!scope.isValid() || (scope.has_embedding && scope.first_layer != 0) ||
            !request.model_has_routed_experts || request.world_rank < 0 ||
            request.local_tp_participants.empty() || request.requested_plan ||
            request.has_pipeline_parallel || request.has_cross_rank_tensor_parallel)
            throw std::invalid_argument("ExpertOverlay pipeline synthesis requires one projected stage without inherited parent authority");
        requireImplicitPolicy(request);
        auto plan = buildImplicitDomainPlan(request);
        plan->first_model_layer = scope.first_layer;
        validateMoERoutedExpertPlacementPlanOrThrow(*plan);
        return plan;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan>
    normalizeMoEExpertOverlayPipelineStagePlan(
        const MoEExpertOverlayAuthorityPlanRequest &request,
        const FactoryPPStageConfig &scope, const ExecutionDomainDefinition &declaration)
    {
        if (!scope.isValid() || (scope.has_embedding && scope.first_layer != 0) ||
            !request.model_has_routed_experts || request.world_rank < 0 || request.requested_plan ||
            request.has_pipeline_parallel || !request.has_cross_rank_tensor_parallel ||
            declaration.scope != ExecutionDomainScope::NODE_LOCAL || declaration.name.empty() ||
            declaration.participants.size() < 2 || declaration.ranks.size() != declaration.participants.size() ||
            declaration.participants != request.local_tp_participants || declaration.weights != request.local_tp_weights ||
            declaration.backend != request.local_tp_backend)
            throw std::invalid_argument("CPU pipeline authority requires its exact original node-local MPI domain");
        if (request.routed_compute_policy != RoutedExpertComputePolicy::Automatic &&
            request.routed_compute_policy != RoutedExpertComputePolicy::Apportioned &&
            request.routed_compute_policy != RoutedExpertComputePolicy::TensorSharded)
            throw std::invalid_argument("CPU pipeline expert compute requires apportioned or tensor-sharded ownership");
        std::set<int> ranks;
        std::set<std::pair<std::string,int>> endpoints;
        for (std::size_t i = 0; i < declaration.participants.size(); ++i) {
            const auto &address = declaration.participants[i];
            if (!address.isCPU() || address.numa_node < 0 || address.hostname.empty() ||
                declaration.ranks[i] < 0 || !ranks.insert(declaration.ranks[i]).second ||
                !endpoints.emplace(address.hostname,address.numa_node).second)
                throw std::invalid_argument("CPU pipeline authority has foreign or duplicated original CPU ownership");
        }
        if (!ranks.contains(request.world_rank))
            throw std::invalid_argument("CPU pipeline authority excludes this original executing rank");
        // Preserve the original global CPU domain before resolving either
        // compute ownership or dense policy. No local-device representation or
        // temporary alternative compute mode is introduced.
        auto plan = buildImplicitDomainPlan(request, &declaration);
        // The original domain serves both dense TP and routed experts. The
        // graph factory resolves dense and routed declarations independently.
        plan->dense_domains.push_back(plan->domains.front().toExecutionDomainDefinition());
        plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
        plan->first_model_layer = scope.first_layer;
        plan->authority_execution = resolveMoEOverlayAuthorityExecutionKind(*plan);
        validateMoERoutedExpertPlacementPlanOrThrow(*plan);
        return plan;
    }

} // namespace llaminar2
