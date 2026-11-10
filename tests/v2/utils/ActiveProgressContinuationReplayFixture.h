/**
 * @file ActiveProgressContinuationReplayFixture.h
 * @brief Small authored observation through the original residency authority.
 *
 * Four independent two-target exchanges require several original bounded waves.
 * The routes and prices are explicit test inputs, not reconstructed model heat.
 * Unit lifecycle and original real-MPI publisher owners share this same input.
 */
#pragma once
#include "ObservedExpertDemandFixture.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEOverlayDistributedResidencyProtocol.h"

namespace llaminar2::test::continuation_replay
{
    /** @return One complete270-row invocation and its original PMA demand banks/mailbox.
     * @param rank Host allocation owner. @param mailbox Admit the original MPI mailbox. */
    inline std::shared_ptr<DecodeExpertHistogram> histogram(int rank = 0, bool mailbox = false)
    {
        DecodeExpertHistogramConfig config;
        config.num_layers = 1;
        config.num_experts = 8;
        config.top_k = 1;
        config.window_size = 270;
        config.token_boundary_layer_idx = 0;
        config.sockets = {DeviceId::cuda(0), DeviceId::cpu()};
        config.ownership = MoELayeredExpertOwnership::uniform(1, 2, {0, 0, 0, 0, 1, 1, 1, 1});
        constexpr moe_overlay_economy::TransactionDemandCapacity capacity{270, 270, 1};
        admitObservedExpertDemand(config, capacity, 3, rank,
            mailbox ? moeOverlayDistributedResidencyProposalWireBytes(1, 8, &capacity) : 0);
        auto result = std::make_shared<DecodeExpertHistogram>(std::move(config));
        const std::vector<uint64_t> counts{1, 2, 3, 4, 50, 60, 70, 80};
        recordObservedPrefillBatch(*result, 0, counts);
        return result;
    }

    /** @return The actual two-tier solver, coherent objective and one-cycle physical limit.
     * @param demand Original mutable route authority.
     * @param residency Original policy value, including its legal zero case.
     * @param root Original whole-GPU owner rank; CPU belongs to the other rank. */
    inline std::shared_ptr<MoEOverlayResidencyAuthority> authority(
        DecodeExpertHistogram &demand, uint64_t residency = 0, int root = 0)
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = plan.shared_expert_domain = "hot";
        plan.residency_policy = RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        RoutedExpertDomain hot, cold;
        hot.name = "hot";
        hot.participants = {GlobalDeviceAddress::cuda(0)};
        hot.world_ranks = {root};
        hot.owner_rank = root;
        cold.name = "cold";
        cold.participants = {GlobalDeviceAddress::cpu(0)};
        cold.world_ranks = {1 - root};
        cold.owner_rank = 1 - root;
        plan.domains = {hot, cold};
        plan.routed_tiers = {{.name = "hot", .domain = "hot", .priority = 0, .max_experts_per_layer = 4},
                            {.name = "cold", .domain = "cold", .priority = 1, .max_experts_per_layer = 4, .fallback = true}};
        MoERoutedExpertModelMetadata model;
        model.num_layers = 1;
        model.num_experts = 8;
        model.d_model = 16;
        model.routed_intermediate_size = 8;
        model.routed_quant_type = "F32";
        auto prices = std::make_shared<MoERoutedTierServiceProfile>();
        prices->identity = "authored-continuation-service";
        prices->production_topology = ExpertHistogramProductionTopology::uniform(1, kAllExpertHistogramProductionSources);
        prices->costs = {{0, 0, {10, 20, 30}}, {1, 0, {100, 200, 300}}};
        prices->setParticipantCosts(std::vector<MoERoutedParticipantLayerPhaseServiceCost>{
            {0, 0, {10, 20, 30}}, {1, 0, {100, 200, 300}}});
        auto moves = std::make_shared<MoEOverlayMigrationCostProfile>();
        moves->identity = "authored-continuation-movement";
        for (int source = 0; source < 2; ++source)
            moves->costs.push_back({.coordinate = {MoEOverlayRouteTarget::whole(source),
                MoEOverlayRouteTarget::whole(1 - source), 0}, .transfer_and_repack_ns = 1,
                .inference_interference_ns = 0});
        return std::make_shared<MoEOverlayResidencyAuthority>(MoEOverlayResidencyAuthority::Config{
            .initial_plan = std::move(plan), .model_metadata = model,
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic, .histogram = &demand,
            .histogram_max_window_tokens = 540, .histogram_window_growth_factor = 2.0,
            .phase_service_profile = std::move(prices), .migration_cost_profile = std::move(moves),
            .migration_economy_policy = MoEOverlayMigrationEconomyPolicy{
                .historical_window_weight = 0, .current_window_weight = 1,
                .payoff_horizon_tokens = 270, .minimum_net_benefit_ns = 0,
                .minimum_residency_generations = residency},
            .participant_rebalance_policy = {.enabled = false},
            .max_concurrent_cycles = 1, .perf_device = "authored-continuation-replay"});
    }
}
