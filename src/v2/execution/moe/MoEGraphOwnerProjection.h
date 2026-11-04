/**
 * @file MoEGraphOwnerProjection.h
 * @brief Project canonical expert placement into domain tables and joint lanes.
 *
 * Joint CPU fragments have no single whole-expert owner. A continuation graph
 * treats their collective domain as external; the original CPU cohort remains
 * responsible for fragment reduction. Each joint dispatch lane additionally
 * receives its exact source membership; the external scalar alone cannot route
 * its work. These cold projections own no placement state and never choose one
 * shard as a substitute whole owner.
 */
#pragma once

#include "MoEExpertOwnerMap.h"
#include "MoEExpertShardCohort.h"
#include "MoEOverlayStaticJointRouteMembership.h"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Project authored expert domains onto the scalar graph compute lane.
     * @param plan Original placement, retained unchanged for fragment execution.
     * @param error Optional diagnostic for unsupported scalar policy mixtures.
     * @return Uniform scalar policy, or no value for incompatible GPU domains.
     *
     * CPU tensor domains use the ordinary graph lane: degree one owns a full
     * matrix; larger degrees execute source-bound partials through their explicit
     * cohort collective. Their authored TensorSharded identity remains in plan,
     * rather than selecting the GPU tensor lowering through this scalar field.
     */
    inline std::optional<RoutedExpertComputePolicy> routedOverlayComputePolicy(
        const MoERoutedExpertPlacementPlan &plan, std::string *error)
    {
        // CPU-only graphs retain their original policy and validation. This
        // projection applies only alongside an actual non-CPU expert lane.
        const bool has_device_lane = std::any_of(plan.routed_tiers.begin(), plan.routed_tiers.end(),
            [&](const auto &tier) {
                const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
                    [&](const auto &candidate) { return candidate.name == tier.domain; });
                return domain != plan.domains.end() && !domain->participants.empty() &&
                    std::all_of(domain->participants.begin(), domain->participants.end(),
                        [](const auto &address) { return !address.isCPU(); });
            });
        std::optional<RoutedExpertComputePolicy> policy;
        for (const auto &tier : plan.routed_tiers)
        {
            const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
                [&](const auto &candidate) { return candidate.name == tier.domain; });
            if (domain == plan.domains.end()) continue;
            const bool cpu_tensor =
                domain->routed_compute_policy == RoutedExpertComputePolicy::TensorSharded &&
                !domain->participants.empty() &&
                std::all_of(domain->participants.begin(), domain->participants.end(),
                    [](const auto &address) { return address.isCPU(); });
            if (cpu_tensor && has_device_lane) continue;
            if (policy && *policy != domain->routed_compute_policy)
            {
                if (error) *error =
                    "mixed routed expert compute policies are not supported in one graph-native overlay: " +
                    std::string(routedExpertComputePolicyToString(*policy)) + " and " +
                    routedExpertComputePolicyToString(domain->routed_compute_policy);
                return std::nullopt;
            }
            policy = domain->routed_compute_policy;
        }
        return policy.value_or(RoutedExpertComputePolicy::Apportioned);
    }

    /**
     * @param owner_map Original immutable whole/joint expert placement.
     * @param layer_idx Original model-global layer.
     * @param num_experts Original logical expert count.
     * @return Overlay participant IDs for whole experts, or the existing
     *         external sentinel -1 for joint CPU collective work.
     */
    inline std::vector<int> ownerParticipantsFromMap(
        const MoEExpertOwnerMap &owner_map, int layer_idx, int num_experts)
    {
        std::vector<int> owners(static_cast<size_t>(std::max(0, num_experts)), -1);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto *placement = owner_map.placementFor(layer_idx, expert);
            if (placement && !placement->shard_cohort)
                owners[static_cast<size_t>(expert)] = placement->owner_participant;
        }
        return owners;
    }

    /**
     * @brief Project all original joint experts for one captured destination.
     * @param owner_map Original prepared whole/joint source placement authority.
     * @param layer_idx Original model-global layer.
     * @param num_experts Original expert axis.
     * @param target_participant Original global target, independent of member count.
     * @param initial_epoch Original Static registry epoch, required for joint lanes.
     * @return Empty for an ordinary lane; exact immutable membership otherwise.
     * @throws std::invalid_argument For missing target or incomplete joint metadata.
     */
    inline MoEOverlayStaticJointRouteMembership jointRouteMembershipFromMap(
        const MoEExpertOwnerMap &owner_map, int layer_idx, int num_experts,
        int target_participant, std::uint64_t initial_epoch)
    {
        if (!owner_map.participantForId(target_participant) || num_experts <= 0 ||
            num_experts > static_cast<int>(moe_runtime_abi::kMaxExperts))
            throw std::invalid_argument("Joint dispatch projection requires an original target and expert axis");
        std::vector<std::int32_t> expert_ids;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto *placement = owner_map.placementFor(layer_idx, expert);
            if (!placement || !placement->shard_cohort)
                continue;
            const auto members = placement->shard_cohort->participants();
            if (std::any_of(members.begin(), members.end(),
                [target_participant](const auto &member)
                { return member.participant_id == target_participant; }))
                expert_ids.push_back(expert);
        }
        if (expert_ids.empty())
            return {};
        return MoEOverlayStaticJointRouteMembership::fromProjection(
            static_cast<std::uint32_t>(num_experts), target_participant,
            layer_idx, initial_epoch, expert_ids);
    }

    /**
     * @param owner_map Original immutable whole/joint expert placement.
     * @param layer_idx Original model-global layer.
     * @param num_experts Original logical expert count.
     * @param domain_name Domain represented by this scalar device routing table.
     * @return Domain-local whole-owner coordinates, or -1 for external work.
     * @throws std::logic_error If the requested domain itself contains joint
     *         fragments; that domain must use its explicit fragment collective.
     */
    inline std::vector<int> domainLocalOwnerParticipantsFromMap(
        const MoEExpertOwnerMap &owner_map, int layer_idx, int num_experts,
        const std::string &domain_name)
    {
        std::vector<int> owners(static_cast<size_t>(std::max(0, num_experts)), -1);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto *placement = owner_map.placementFor(layer_idx, expert);
            if (!placement)
                continue;
            if (placement->shard_cohort)
            {
                if (placement->domain_name == domain_name)
                    throw std::logic_error(
                        "Joint CPU expert domain requires its fragment collective, not a scalar owner table");
                continue;
            }
            const auto *participant = owner_map.participantForId(placement->owner_participant);
            if (participant && participant->domain_name == domain_name)
                owners[static_cast<size_t>(expert)] = participant->domain_participant_index;
        }
        return owners;
    }
}
