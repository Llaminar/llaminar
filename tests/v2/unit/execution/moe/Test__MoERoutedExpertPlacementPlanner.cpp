/**
 * @file Test__MoERoutedExpertPlacementPlanner.cpp
 * @brief Device-free proofs of exact quotas, demand costs and compact stage plans.
 *
 * Stage regressions preserve global expert/weight identities while proving that
 * every storage row and memory estimate covers only the owning pipeline stage.
 */

#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "execution/moe/DecodeExpertHistogram.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {

        RoutedExpertDomain cudaSingleDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::AUTO;
            domain.participants = {GlobalDeviceAddress::cuda(0)};
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain rocmLocalTPDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain cpuNodeTPDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::NODE_LOCAL;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertTier tier(const std::string &name, const std::string &domain, int priority, bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        RoutedExpertLayerPlacement placement(int layer, std::vector<int> routed_expert_tier)
        {
            RoutedExpertLayerPlacement result;
            result.layer = layer;
            result.routed_expert_tier = std::move(routed_expert_tier);
            return result;
        }

        MoERoutedExpertModelMetadata metadata(int num_layers = 2, int num_experts = 6)
        {
            MoERoutedExpertModelMetadata result;
            result.num_layers = num_layers;
            result.num_experts = num_experts;
            result.d_model = 16;
            result.routed_intermediate_size = 32;
            result.shared_intermediate_size = 64;
            result.has_shared_expert = true;
            result.routed_quant_type = "Q4_0";
            result.shared_quant_type = "F16";
            return result;
        }

        /** Bind histogram load accounting to one inert test participant. */
        void bindSingleParticipantOwnership(
            DecodeExpertHistogramConfig &config)
        {
            config.sockets = {DeviceId::cpu()};
            config.ownership = MoELayeredExpertOwnership::uniform(
                config.num_layers,
                /*participant_count=*/1,
                std::vector<int>(
                    static_cast<size_t>(config.num_experts),
                    /*owner=*/0));
        }

        MoERoutedExpertPlacementPlan twoTierRocmCpuPlan(RoutedExpertResidencyPolicy policy = RoutedExpertResidencyPolicy::StaticById)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "rocm_hot";
            plan.shared_expert_domain = "rocm_hot";
            plan.residency_policy = policy;
            plan.domains = {
                rocmLocalTPDomain("rocm_hot"),
                cpuNodeTPDomain("cpu_cold"),
            };
            plan.routed_tiers = {
                tier("hot", "rocm_hot", 0),
                tier("cold", "cpu_cold", 1, true),
            };
            return plan;
        }

        MoERoutedExpertPlacementPlan threeTierCudaRocmCpuPlan(RoutedExpertResidencyPolicy policy = RoutedExpertResidencyPolicy::StaticById)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_fast";
            plan.shared_expert_domain = "cuda_fast";
            plan.residency_policy = policy;
            plan.domains = {
                cudaSingleDomain("cuda_fast"),
                rocmLocalTPDomain("rocm_warm"),
                cpuNodeTPDomain("cpu_cold"),
            };
            plan.routed_tiers = {
                tier("hottest", "cuda_fast", 0),
                tier("warm", "rocm_warm", 1),
                tier("cold", "cpu_cold", 2, true),
            };
            return plan;
        }

    } // namespace

    TEST(Test__MoERoutedExpertPlacementPlanner, AssignsAndAccountsSharedExpertsToConfiguredDomainFirst)
    {
        auto plan = twoTierRocmCpuPlan();
        plan.routed_tiers[0].max_experts_per_layer = 1;
        const auto model = metadata();

        const auto result = MoERoutedExpertPlacementPlanner::plan(plan, model);

        EXPECT_EQ(result.planned_plan.shared_expert_domain, "rocm_hot");
        EXPECT_EQ(result.memory.shared_expert_domain, "rocm_hot");
        EXPECT_EQ(result.memory.shared_expert_bytes_per_layer,
                  MoERoutedExpertPlacementPlanner::estimateSharedExpertBytesPerLayer(model));
        EXPECT_EQ(result.memory.total_shared_expert_bytes,
                  MoERoutedExpertPlacementPlanner::estimateTotalSharedExpertBytes(model));
        ASSERT_FALSE(result.memory.domains.empty());
        EXPECT_EQ(result.memory.domains.front().domain, "rocm_hot");
        EXPECT_EQ(result.memory.domains.front().shared_expert_bytes,
                  result.memory.total_shared_expert_bytes);
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, StaticByIdFillsPriorityTiersBeforeFallback)
    {
        auto plan = threeTierCudaRocmCpuPlan();
        plan.routed_tiers[0].max_experts_per_layer = 3;
        plan.routed_tiers[0].memory_budget_bytes = 2 * MoERoutedExpertPlacementPlanner::estimateRoutedExpertBytesPerExpert(metadata());
        plan.routed_tiers[1].max_experts_per_layer = 2;

        const auto result = MoERoutedExpertPlacementPlanner::plan(plan, metadata());

        ASSERT_EQ(result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(result.planned_plan.placements[0].routed_expert_tier,
                  (std::vector<int>{0, 0, 1, 1, 2, 2}));
        EXPECT_EQ(result.planned_plan.placements[1].routed_expert_tier,
                  (std::vector<int>{0, 0, 1, 1, 2, 2}));
    }

    /**
     * @brief Setup ordering is applied only after exact live quotas exist.
     *
     * This models the production sequence used by automatic ExpertOverlay
     * capacity admission: physical memory resolves two priority-zero slots,
     * then the declarative permutation chooses their expert identities.
     */
    TEST(Test__MoERoutedExpertPlacementPlanner,
         DeferredInitialOrderUsesResolvedPhysicalQuotasAndDefaultsOtherLayers)
    {
        auto plan = twoTierRocmCpuPlan(
            RoutedExpertResidencyPolicy::RoutedTierRebalanced);
        plan.routed_tiers[0].resolved_live_experts_per_layer = {2, 2};
        plan.routed_tiers[1].resolved_live_experts_per_layer = {4, 4};
        plan.initial_layer_order_overrides = {
            RoutedExpertInitialLayerOrder{
                .layer = 0,
                .expert_ids = {5, 4, 3, 2, 1, 0},
            },
        };

        const auto result = MoERoutedExpertPlacementPlanner::plan(
            plan, metadata(/*num_layers=*/2, /*num_experts=*/6));

        ASSERT_EQ(result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(
            result.planned_plan.placements[0].routed_expert_tier,
            (std::vector<int>{1, 1, 1, 1, 0, 0}));
        EXPECT_EQ(
            result.planned_plan.placements[1].routed_expert_tier,
            (std::vector<int>{0, 0, 1, 1, 1, 1}))
            << "the retained MTP source layer must use deterministic default ordering";
        EXPECT_TRUE(result.planned_plan.initial_layer_order_overrides.empty())
            << "the frozen plan must retain one concrete epoch-one authority";
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, HistogramTieredCacheChoosesHottestExpertsWithDeterministicTieBreak)
    {
        auto plan = threeTierCudaRocmCpuPlan(RoutedExpertResidencyPolicy::HistogramTieredCache);
        plan.routed_tiers[0].max_experts_per_layer = 1;
        plan.routed_tiers[1].max_experts_per_layer = 1;

        DecodeExpertHistogramConfig config;
        config.num_layers = 1;
        config.num_experts = 6;
        config.top_k = 2;
        bindSingleParticipantOwnership(config);
        DecodeExpertHistogram histogram(config);
        const int first_route[] = {4, 2};
        const int second_route[] = {2, 4};
        const float weights[] = {0.5f, 0.5f};
        histogram.record(0, first_route, weights, 2);
        histogram.record(0, second_route, weights, 2);

        MoERoutedExpertPlacementPlannerOptions options;
        options.decode_histogram = &histogram;

        const auto result = MoERoutedExpertPlacementPlanner::plan(plan, metadata(1, 6), options);

        ASSERT_EQ(result.planned_plan.placements.size(), 1u);
        EXPECT_EQ(result.planned_plan.placements[0].routed_expert_tier,
                  (std::vector<int>{2, 2, 0, 2, 1, 2}));
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, HistogramTieredCacheFallsBackToByIdWhenHistogramAbsentOrLayerCountsAreZero)
    {
        auto plan = twoTierRocmCpuPlan(RoutedExpertResidencyPolicy::HistogramTieredCache);
        plan.routed_tiers[0].max_experts_per_layer = 2;
        const auto expected_by_id = std::vector<int>{0, 0, 1, 1, 1, 1};

        const auto absent_result = MoERoutedExpertPlacementPlanner::plan(plan, metadata());
        ASSERT_EQ(absent_result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(absent_result.planned_plan.placements[0].routed_expert_tier, expected_by_id);
        EXPECT_EQ(absent_result.planned_plan.placements[1].routed_expert_tier, expected_by_id);

        DecodeExpertHistogramConfig config;
        config.num_layers = 2;
        config.num_experts = 6;
        config.top_k = 1;
        bindSingleParticipantOwnership(config);
        DecodeExpertHistogram zero_histogram(config);
        MoERoutedExpertPlacementPlannerOptions options;
        options.decode_histogram = &zero_histogram;

        const auto zero_result = MoERoutedExpertPlacementPlanner::plan(plan, metadata(), options);
        ASSERT_EQ(zero_result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(zero_result.planned_plan.placements[0].routed_expert_tier, expected_by_id);
        EXPECT_EQ(zero_result.planned_plan.placements[1].routed_expert_tier, expected_by_id);
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, ExplicitPlacementsAndMasksAreAcceptedAndMissingExplicitPlacementIsRejected)
    {
        auto plan = twoTierRocmCpuPlan(RoutedExpertResidencyPolicy::ExplicitMasks);
        plan.placements = {
            placement(0, {0, 1, 0, 1, 0, 1}),
            placement(1, {1, 0, 1, 0, 1, 0}),
        };

        const auto placement_result = MoERoutedExpertPlacementPlanner::plan(plan, metadata());
        ASSERT_EQ(placement_result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(placement_result.planned_plan.placements[0].routed_expert_tier,
                  (std::vector<int>{0, 1, 0, 1, 0, 1}));

        plan.placements.clear();
        MoERoutedExpertPlacementPlannerOptions options;
        options.explicit_masks = {
            {.layer = 0, .tier_index = 0, .expert_ids = {0, 2, 4}},
            {.layer = 0, .tier_index = 1, .expert_ids = {1, 3, 5}},
            {.layer = 1, .tier_index = 0, .expert_ids = {1, 3, 5}},
            {.layer = 1, .tier_index = 1, .expert_ids = {0, 2, 4}},
        };
        const auto mask_result = MoERoutedExpertPlacementPlanner::plan(plan, metadata(), options);
        EXPECT_EQ(mask_result.planned_plan.placements[0].routed_expert_tier,
                  (std::vector<int>{0, 1, 0, 1, 0, 1}));
        EXPECT_EQ(mask_result.planned_plan.placements[1].routed_expert_tier,
                  (std::vector<int>{1, 0, 1, 0, 1, 0}));

        EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, metadata()), std::invalid_argument);
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, PlannedTopologiesSatisfyPhaseOneValidation)
    {
        auto two_tier = twoTierRocmCpuPlan();
        two_tier.routed_tiers[0].max_experts_per_layer = 3;
        const auto two_tier_result = MoERoutedExpertPlacementPlanner::plan(two_tier, metadata());
        EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(
                        two_tier_result.planned_plan,
                        {.layer_count = 2, .routed_expert_count = 6})
                        .ok());

        auto three_tier = threeTierCudaRocmCpuPlan();
        three_tier.routed_tiers[0].max_experts_per_layer = 2;
        three_tier.routed_tiers[1].max_experts_per_layer = 2;
        const auto three_tier_result = MoERoutedExpertPlacementPlanner::plan(three_tier, metadata());
        EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(
                        three_tier_result.planned_plan,
                        {.layer_count = 2, .routed_expert_count = 6})
                        .ok());
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, NoFallbackTierCapacityMustCoverEveryExpert)
    {
        auto plan = twoTierRocmCpuPlan();
        plan.routed_tiers.pop_back();
        plan.routed_tiers[0].max_experts_per_layer = 5;

        try
        {
            (void)MoERoutedExpertPlacementPlanner::plan(plan, metadata());
            FAIL() << "Expected no-fallback capacity validation to throw";
        }
        catch (const std::invalid_argument &error)
        {
            const std::string message = error.what();
            EXPECT_NE(
                message.find("capacity covers only 5 of 6"),
                std::string::npos);
        }
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, PipelineStagePlanningHasNoPrecedingLayerPadding)
    {
        for (const int origin : {0, 20, 40})
        for (const int layers : {1, 2, 21})
        for (const auto policy : {RoutedExpertResidencyPolicy::StaticById,
                                  RoutedExpertResidencyPolicy::HistogramTieredCache,
                                  RoutedExpertResidencyPolicy::RoutedTierRebalanced})
        {
            auto plan = twoTierRocmCpuPlan(policy);
            plan.first_model_layer = origin;
            plan.routed_tiers[0].resolved_live_experts_per_layer.assign(layers, 2);
            plan.routed_tiers[1].resolved_live_experts_per_layer.assign(layers, 4);
            auto model = metadata(layers);
            model.first_model_layer = origin;
            const auto result = MoERoutedExpertPlacementPlanner::plan(plan, model);
            ASSERT_EQ(result.planned_plan.placements.size(), static_cast<size_t>(layers));
            EXPECT_EQ(result.planned_plan.placementLayerCapacity(), layers);
            EXPECT_EQ(result.memory.total_shared_expert_bytes,
                      layers * result.memory.shared_expert_bytes_per_layer);
            EXPECT_EQ(result.memory.total_routed_expert_bytes,
                      layers * 6u * result.memory.routed_expert_bytes_per_expert);
            for (int row = 0; row < layers; ++row)
            {
                EXPECT_EQ(result.planned_plan.placements[row].layer, origin + row);
                EXPECT_EQ(result.planned_plan.placements[row].routed_expert_tier,
                          (std::vector<int>{0, 0, 1, 1, 1, 1}));
            }
        }
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, PipelineStageDynamicCostsUseGlobalEvidenceAndLocalQuotas)
    {
        for (const auto policy : {RoutedExpertResidencyPolicy::HistogramTieredCache,
                                  RoutedExpertResidencyPolicy::RoutedTierRebalanced})
        {
            auto plan = twoTierRocmCpuPlan(policy);
            plan.first_model_layer = 32;
            plan.routed_tiers[0].resolved_live_experts_per_layer = {2, 4};
            plan.routed_tiers[1].resolved_live_experts_per_layer = {4, 2};
            auto model = metadata();
            model.first_model_layer = 32;
            DecodeExpertHistogramConfig cfg;
            cfg.num_layers = 2;
            cfg.num_experts = 6;
            cfg.top_k = 1;
            cfg.sockets = {DeviceId::cpu()};
            cfg.ownership = MoELayeredExpertOwnership::uniform(2, 1, {0, 0, 0, 0, 0, 0}, 32);
            DecodeExpertHistogram histogram(cfg);
            const uint64_t counts[]{1, 2, 3, 4, 5, 6};
            histogram.mergeLayerCounts(32, counts, 6, true, ExpertHistogramSource::PrefillChunk);
            histogram.mergeLayerCounts(33, counts, 6, true, ExpertHistogramSource::PrefillChunk);
            MoERoutedTierServiceProfile profile;
            profile.identity = "stage32-costs";
            profile.production_topology = ExpertHistogramProductionTopology::uniform(
                2, kAllExpertHistogramProductionSources, 32);
            for (int layer = 32; layer < 34; ++layer)
            for (int tier_id = 0; tier_id < 2; ++tier_id)
            {
                const uint64_t cost = tier_id == 0 ? 1 : 9;
                profile.costs.push_back({tier_id, layer, {cost, cost, cost}});
            }
            const auto live = MoERoutedExpertPlacementPlanner::plan(
                plan, model, {.decode_histogram = &histogram, .phase_service_profile = &profile});
            EXPECT_EQ(live.planned_plan.placements[0].routed_expert_tier,
                      (std::vector<int>{1, 1, 1, 1, 0, 0}));
            EXPECT_EQ(live.planned_plan.placements[1].routed_expert_tier,
                      (std::vector<int>{1, 1, 0, 0, 0, 0}));
            const auto window = histogram.freezeAndRotateWindow();
            const auto frozen = MoERoutedExpertPlacementPlanner::plan(plan, model,
                {.decode_histogram_window = &window, .phase_service_profile = &profile});
            EXPECT_EQ(frozen.planned_plan.placements[0].routed_expert_tier,
                      live.planned_plan.placements[0].routed_expert_tier);
            auto empty = histogram.freezeAndRotateWindow();
            MoERoutedExpertPlacementPlannerOptions incumbent;
            incumbent.decode_histogram_window = &empty;
            incumbent.rebalancer.previous_placements = live.planned_plan.placements;
            const auto retained = MoERoutedExpertPlacementPlanner::plan(plan, model, incumbent);
            EXPECT_EQ(retained.planned_plan.placements[1].routed_expert_tier,
                      live.planned_plan.placements[1].routed_expert_tier);
            --incumbent.rebalancer.previous_placements.front().layer;
            EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model, incumbent),
                         std::invalid_argument);
            auto foreign = window;
            --foreign.first_model_layer;
            EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model,
                {.decode_histogram_window = &foreign}), std::invalid_argument);
            profile.production_topology = ExpertHistogramProductionTopology::uniform(
                2, kAllExpertHistogramProductionSources, 31);
            EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model,
                {.decode_histogram_window = &window, .phase_service_profile = &profile}),
                std::invalid_argument);
        }
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, PipelineStageExplicitMasksRejectMissingAndForeignRows)
    {
        auto plan = twoTierRocmCpuPlan(RoutedExpertResidencyPolicy::ExplicitMasks);
        plan.first_model_layer = 32;
        auto model = metadata();
        model.first_model_layer = 32;
        MoERoutedExpertPlacementPlannerOptions options;
        options.explicit_masks = {{32, 0, {0, 1, 2}}, {32, 1, {3, 4, 5}},
                                  {33, 0, {0, 1, 2}}, {33, 1, {3, 4, 5}}};
        const auto complete = MoERoutedExpertPlacementPlanner::plan(plan, model, options);
        EXPECT_EQ(complete.planned_plan.placementLayerCapacity(), 2);
        for (const int foreign : {0, 31, 34})
        {
            auto invalid = options;
            invalid.explicit_masks.front().layer = foreign;
            EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model, invalid),
                         std::invalid_argument);
        }
        options.explicit_masks.pop_back();
        EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model, options),
                     std::invalid_argument);
        --model.first_model_layer;
        EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model), std::invalid_argument);
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, PipelineStageIntervalsRejectOverflowBeforeAllocation)
    {
        auto plan = twoTierRocmCpuPlan();
        auto model = metadata();
        for (const int origin : {-1, std::numeric_limits<int>::max()})
        {
            plan.first_model_layer = origin;
            model.first_model_layer = origin;
            EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model), std::invalid_argument);
            EXPECT_FALSE(validateMoERoutedExpertPlacementPlan(plan, {.layer_count = 2}).ok());
        }
    }

    TEST(Test__MoERoutedExpertPlacementPlanner, PipelineStageRejectsRaggedQuotasBeforeAccounting)
    {
        for (const int origin : {0, 32})
        {
            auto plan = twoTierRocmCpuPlan();
            plan.first_model_layer = origin;
            plan.routed_tiers[0].resolved_live_experts_per_layer = {2, 2};
            plan.routed_tiers[1].resolved_live_experts_per_layer = {4};
            auto model = metadata();
            model.first_model_layer = origin;
            EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model), std::invalid_argument);
            plan.routed_tiers[1].resolved_live_experts_per_layer.clear();
            EXPECT_THROW((void)MoERoutedExpertPlacementPlanner::plan(plan, model), std::invalid_argument);
        }
    }

} // namespace llaminar2::test
