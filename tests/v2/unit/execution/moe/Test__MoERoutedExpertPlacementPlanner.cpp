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
                message.find("no-fallback tier capacity cannot cover every expert"),
                std::string::npos);
        }
    }

} // namespace llaminar2::test
