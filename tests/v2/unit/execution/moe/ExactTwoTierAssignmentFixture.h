/**
 * @file ExactTwoTierAssignmentFixture.h
 * @brief Authored, device-free inputs for the original exact placement owner.
 *
 * The third tier is an ordinary validated zero-quota tier, not a solver switch.
 * Its preferred priority increments both active priority ranks as tier count
 * increases: each active expert-id coefficient is therefore unchanged (constant
 * shift zero, positive scale one). Complete assignments can be compared against
 * the original many-tier flow solver without changing the production API.
 */
#pragma once
#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include <array>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace llaminar2::test
{
    /** @brief One immutable authored phase window and its original planner inputs. */
    struct ExactTwoTierAssignmentFixture
    {
        MoERoutedExpertPlacementPlan plan;
        MoERoutedExpertModelMetadata model;
        DecodeExpertHistogramWindow window;
        MoERoutedTierServiceProfile profile;
        std::vector<RoutedExpertLayerPlacement> previous;

        /**
         * @param experts Positive expert count.
         * @param quota Exact first-tier capacity, including zero/full.
         * @param counts Three phase-major count banks for one layer.
         * @param prices Original per-tier phase service costs.
         * @param reverse Declare the fallback first, independently of priorities.
         * @param members Explicit CPU domain participant count; no hardware query.
         * @param layers Number of identical authored layers for bounded timing.
         */
        ExactTwoTierAssignmentFixture(int experts, int quota,
            const std::vector<std::uint64_t> &counts,
            std::array<std::array<std::uint64_t, 3>, 2> prices,
            bool reverse = false, int members = 2, int layers = 1)
        {
            model.num_layers = layers;
            model.first_model_layer = 32;
            model.num_experts = experts;
            model.d_model = 16;
            model.routed_intermediate_size = 32;
            model.routed_quant_type = "Q4_0";
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.residency_policy = RoutedExpertResidencyPolicy::HistogramTieredCache;
            plan.first_model_layer = model.first_model_layer;
            plan.continuation_domain = "first";
            plan.shared_expert_domain = "first";
            RoutedExpertDomain first;
            first.name = "first";
            first.scope = ExecutionDomainScope::SINGLE;
            first.backend = CollectiveBackendType::AUTO;
            first.participants = {GlobalDeviceAddress::cuda(0)};
            RoutedExpertDomain second;
            second.name = "second";
            second.scope = members == 1 ? ExecutionDomainScope::SINGLE : ExecutionDomainScope::NODE_LOCAL;
            second.backend = members == 1 ? CollectiveBackendType::AUTO : CollectiveBackendType::UPI;
            for (int member = 0; member < members; ++member)
                second.participants.push_back(GlobalDeviceAddress::cpu(member));
            plan.domains = {first, second};
            RoutedExpertTier preferred;
            preferred.name = "preferred";
            preferred.domain = "first";
            preferred.priority = -7;
            preferred.resolved_live_experts_per_layer.assign(layers, quota);
            RoutedExpertTier fallback;
            fallback.name = "fallback";
            fallback.domain = "second";
            fallback.priority = 53;
            fallback.fallback = true;
            fallback.resolved_live_experts_per_layer.assign(layers, experts - quota);
            plan.routed_tiers = reverse ? std::vector{fallback, preferred} : std::vector{preferred, fallback};
            window.generation = 7;
            window.token_count = 1;
            window.num_layers = layers;
            window.num_experts = experts;
            window.first_model_layer = model.first_model_layer;
            window.expert_counts.assign(std::size_t(layers) * experts, 0);
            window.source_expert_counts.assign(3 * std::size_t(layers) * experts, 0);
            for (int source = 0; source < 3; ++source)
                for (int layer = 0; layer < layers; ++layer)
                    for (int expert = 0; expert < experts; ++expert)
                    {
                        const auto count = counts.at(std::size_t(source) * experts + expert);
                        window.source_expert_counts[(std::size_t(source) * layers + layer) * experts + expert] = count;
                        window.expert_counts[std::size_t(layer) * experts + expert] += count;
                    }
            profile.identity = "authored-exact-two-tier-assignment";
            profile.production_topology = ExpertHistogramProductionTopology::uniform(
                layers, kAllExpertHistogramProductionSources, model.first_model_layer);
            for (int layer = 0; layer < layers; ++layer)
                for (int tier = 0; tier < 2; ++tier)
                    profile.costs.push_back({tier, model.first_model_layer + layer,
                        prices[reverse ? 1 - tier : tier]});
        }

        /** @return Same assignment problem, invoking the original >2-tier solver. */
        ExactTwoTierAssignmentFixture flowOracle() const
        {
            auto result = *this;
            auto unused_domain = plan.domains.front();
            unused_domain.name = "zero-quota-domain";
            unused_domain.participants = {GlobalDeviceAddress::cuda(17)};
            result.plan.domains.push_back(unused_domain);
            RoutedExpertTier unused;
            unused.name = "zero-quota-tier";
            unused.domain = unused_domain.name;
            unused.priority = -99;
            unused.resolved_live_experts_per_layer.assign(model.num_layers, 0);
            result.plan.routed_tiers.push_back(unused);
            for (int layer = 0; layer < model.num_layers; ++layer)
            {
                const auto source = std::find_if(profile.costs.begin(), profile.costs.end(),
                    [&](const auto &row) { return row.layer == model.first_model_layer + layer; });
                result.profile.costs.push_back({2, model.first_model_layer + layer,
                    source->nanoseconds_per_activation});
            }
            return result;
        }

        /** @return Complete model-global placements from the public production planner. */
        std::vector<RoutedExpertLayerPlacement> solve() const
        {
            MoERoutedExpertPlacementPlannerOptions options;
            options.decode_histogram_window = &window;
            options.phase_service_profile = &profile;
            options.rebalancer.previous_placements = previous;
            return MoERoutedExpertPlacementPlanner::plan(plan, model, options).planned_plan.placements;
        }
    };
}
