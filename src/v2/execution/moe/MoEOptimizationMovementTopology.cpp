/**
 * @file MoEOptimizationMovementTopology.cpp
 * @brief Describe movement geometry using already-admitted expert membership.
 *
 * No free-memory query, live ownership lookup, or bandwidth estimate belongs
 * here. The canonical frozen plan supplies membership; the existing authority
 * resolver supplies the required execution location. Policy and completed
 * movement remain owned by the runtime controller and its immutable journal.
 */
#include "execution/moe/MoEOptimizationMovementTopology.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"

#include <optional>
#include <unordered_set>

namespace llaminar2
{
    MoEOptimizationMovementAxes availableMoEOptimizationMovementAxes(const MoERoutedExpertPlacementPlan &plan)
    {
        if (plan.placements.empty() || plan.routed_tiers.empty())
            throw std::invalid_argument("movement topology requires model-frozen tier membership");

        std::vector<const RoutedExpertDomain *> domains;
        domains.reserve(plan.routed_tiers.size());
        for (const auto &tier : plan.routed_tiers)
        {
            const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
                [&](const auto &candidate) { return candidate.name == tier.domain; });
            if (domain == plan.domains.end() || domain->participants.empty())
                throw std::invalid_argument("movement topology cannot resolve a nonempty routed domain: " + tier.domain);
            domains.push_back(&*domain);
        }

        bool tier_axis = false;
        bool participant_axis = false;
        std::unordered_set<int> layers;
        std::vector<std::size_t> counts(plan.routed_tiers.size());
        for (const auto &placement : plan.placements)
        {
            if (placement.layer < 0 || !layers.insert(placement.layer).second || placement.routed_expert_tier.empty())
                throw std::invalid_argument("movement topology has invalid or repeated routed-layer membership");
            std::fill(counts.begin(), counts.end(), 0u);
            for (const int tier : placement.routed_expert_tier)
            {
                if (tier < 0 || static_cast<std::size_t>(tier) >= counts.size())
                    throw std::invalid_argument("movement topology contains an invalid routed-tier index");
                ++counts[static_cast<std::size_t>(tier)];
            }

            // Empty tiers cannot exchange capacity-preserving cycles. Different
            // layers cannot exchange experts, so compare priorities per layer.
            std::optional<int> occupied_priority;
            for (std::size_t tier = 0; tier < counts.size(); ++tier)
            {
                if (counts[tier] == 0u)
                    continue;
                const int priority = plan.routed_tiers[tier].priority;
                tier_axis |= occupied_priority && *occupied_priority != priority;
                occupied_priority = priority;

                // Domains, not equal priority numbers, delimit participant
                // balancing. Tensor-sharded endpoints jointly own an expert.
                const auto &domain = *domains[tier];
                participant_axis |= counts[tier] >= 2u && domain.participants.size() >= 2u &&
                    domain.routed_compute_policy == RoutedExpertComputePolicy::Apportioned;
            }
        }
        return tier_axis
            ? (participant_axis ? MoEOptimizationMovementAxes::Both : MoEOptimizationMovementAxes::TierResidency)
            : (participant_axis ? MoEOptimizationMovementAxes::ParticipantPlacement : MoEOptimizationMovementAxes::None);
    }

    MoEOptimizationMovementTopology describeMoEOptimizationMovementTopology(const MoERoutedExpertPlacementPlan *plan)
    {
        if (!plan || !plan->usesExpertOverlayAuthority())
            return {};
        // Reuse the production resolver rather than constructing a competing
        // vendor/tier heuristic. A frozen plan cannot silently change owners.
        const auto required = resolveMoEOverlayAuthorityExecutionKind(*plan);
        if (plan->authority_execution == MoEOverlayAuthorityExecutionKind::Unresolved ||
            plan->authority_execution != required)
            throw std::invalid_argument("movement topology has unresolved or contradictory authority");
        return {
            .authority = required == MoEOverlayAuthorityExecutionKind::DeviceResident
                ? MoEOptimizationAuthority::Device : MoEOptimizationAuthority::Host,
            .axes = availableMoEOptimizationMovementAxes(*plan),
        };
    }
}
