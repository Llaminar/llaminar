#include "execution/moe/MoEExpertParallelPlanner.h"
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

        ExpertComputeDomain cudaSingleDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::AUTO;
            domain.participants = {GlobalDeviceAddress::cuda(0)};
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain rocmLocalTPDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::LocalTP;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain cpuNodeLocalTPDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::NodeLocalTP;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier tier(const std::string &name, const std::string &domain, int priority, bool fallback = false)
        {
            ExpertRoutedTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        ExpertLayerPlacement placement(int layer, std::vector<int> routed_expert_tier)
        {
            ExpertLayerPlacement result;
            result.layer = layer;
            result.routed_expert_tier = std::move(routed_expert_tier);
            return result;
        }

        MoEExpertModelMetadata metadata(int num_layers = 2, int num_experts = 6)
        {
            MoEExpertModelMetadata result;
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

        MoEExpertParallelPlan twoTierRocmCpuPlan(ExpertResidencyPolicy policy = ExpertResidencyPolicy::StaticById)
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "rocm_hot";
            plan.shared_expert_domain = "rocm_hot";
            plan.residency_policy = policy;
            plan.domains = {
                rocmLocalTPDomain("rocm_hot"),
                cpuNodeLocalTPDomain("cpu_cold"),
            };
            plan.routed_tiers = {
                tier("hot", "rocm_hot", 0),
                tier("cold", "cpu_cold", 1, true),
            };
            return plan;
        }

        MoEExpertParallelPlan threeTierCudaRocmCpuPlan(ExpertResidencyPolicy policy = ExpertResidencyPolicy::StaticById)
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "cuda_fast";
            plan.shared_expert_domain = "cuda_fast";
            plan.residency_policy = policy;
            plan.domains = {
                cudaSingleDomain("cuda_fast"),
                rocmLocalTPDomain("rocm_warm"),
                cpuNodeLocalTPDomain("cpu_cold"),
            };
            plan.routed_tiers = {
                tier("hottest", "cuda_fast", 0),
                tier("warm", "rocm_warm", 1),
                tier("cold", "cpu_cold", 2, true),
            };
            return plan;
        }

    } // namespace

    TEST(Test__MoEExpertParallelPlanner, AssignsAndAccountsSharedExpertsToConfiguredDomainFirst)
    {
        auto plan = twoTierRocmCpuPlan();
        plan.routed_tiers[0].max_experts_per_layer = 1;
        const auto model = metadata();

        const auto result = MoEExpertParallelPlanner::plan(plan, model);

        EXPECT_EQ(result.planned_plan.shared_expert_domain, "rocm_hot");
        EXPECT_EQ(result.memory.shared_expert_domain, "rocm_hot");
        EXPECT_EQ(result.memory.shared_expert_bytes_per_layer,
                  MoEExpertParallelPlanner::estimateSharedExpertBytesPerLayer(model));
        EXPECT_EQ(result.memory.total_shared_expert_bytes,
                  MoEExpertParallelPlanner::estimateTotalSharedExpertBytes(model));
        ASSERT_FALSE(result.memory.domains.empty());
        EXPECT_EQ(result.memory.domains.front().domain, "rocm_hot");
        EXPECT_EQ(result.memory.domains.front().shared_expert_bytes,
                  result.memory.total_shared_expert_bytes);
    }

    TEST(Test__MoEExpertParallelPlanner, StaticByIdFillsPriorityTiersBeforeFallback)
    {
        auto plan = threeTierCudaRocmCpuPlan();
        plan.routed_tiers[0].max_experts_per_layer = 3;
        plan.routed_tiers[0].memory_budget_bytes = 2 * MoEExpertParallelPlanner::estimateRoutedExpertBytesPerExpert(metadata());
        plan.routed_tiers[1].max_experts_per_layer = 2;

        const auto result = MoEExpertParallelPlanner::plan(plan, metadata());

        ASSERT_EQ(result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(result.planned_plan.placements[0].routed_expert_tier,
                  (std::vector<int>{0, 0, 1, 1, 2, 2}));
        EXPECT_EQ(result.planned_plan.placements[1].routed_expert_tier,
                  (std::vector<int>{0, 0, 1, 1, 2, 2}));
    }

    TEST(Test__MoEExpertParallelPlanner, HistogramTieredCacheChoosesHottestExpertsWithDeterministicTieBreak)
    {
        auto plan = threeTierCudaRocmCpuPlan(ExpertResidencyPolicy::HistogramTieredCache);
        plan.routed_tiers[0].max_experts_per_layer = 1;
        plan.routed_tiers[1].max_experts_per_layer = 1;

        DecodeExpertHistogramConfig config;
        config.num_layers = 1;
        config.num_experts = 6;
        config.top_k = 2;
        DecodeExpertHistogram histogram(config);
        const int first_route[] = {4, 2};
        const int second_route[] = {2, 4};
        const float weights[] = {0.5f, 0.5f};
        histogram.record(0, first_route, weights, 2);
        histogram.record(0, second_route, weights, 2);

        MoEExpertParallelPlannerOptions options;
        options.decode_histogram = &histogram;

        const auto result = MoEExpertParallelPlanner::plan(plan, metadata(1, 6), options);

        ASSERT_EQ(result.planned_plan.placements.size(), 1u);
        EXPECT_EQ(result.planned_plan.placements[0].routed_expert_tier,
                  (std::vector<int>{2, 2, 0, 2, 1, 2}));
    }

    TEST(Test__MoEExpertParallelPlanner, HistogramTieredCacheFallsBackToByIdWhenHistogramAbsentOrLayerCountsAreZero)
    {
        auto plan = twoTierRocmCpuPlan(ExpertResidencyPolicy::HistogramTieredCache);
        plan.routed_tiers[0].max_experts_per_layer = 2;
        const auto expected_by_id = std::vector<int>{0, 0, 1, 1, 1, 1};

        const auto absent_result = MoEExpertParallelPlanner::plan(plan, metadata());
        ASSERT_EQ(absent_result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(absent_result.planned_plan.placements[0].routed_expert_tier, expected_by_id);
        EXPECT_EQ(absent_result.planned_plan.placements[1].routed_expert_tier, expected_by_id);

        DecodeExpertHistogramConfig config;
        config.num_layers = 2;
        config.num_experts = 6;
        config.top_k = 1;
        DecodeExpertHistogram zero_histogram(config);
        MoEExpertParallelPlannerOptions options;
        options.decode_histogram = &zero_histogram;

        const auto zero_result = MoEExpertParallelPlanner::plan(plan, metadata(), options);
        ASSERT_EQ(zero_result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(zero_result.planned_plan.placements[0].routed_expert_tier, expected_by_id);
        EXPECT_EQ(zero_result.planned_plan.placements[1].routed_expert_tier, expected_by_id);
    }

    TEST(Test__MoEExpertParallelPlanner, ExplicitPlacementsAndMasksAreAcceptedAndMissingExplicitPlacementIsRejected)
    {
        auto plan = twoTierRocmCpuPlan(ExpertResidencyPolicy::ExplicitMasks);
        plan.placements = {
            placement(0, {0, 1, 0, 1, 0, 1}),
            placement(1, {1, 0, 1, 0, 1, 0}),
        };

        const auto placement_result = MoEExpertParallelPlanner::plan(plan, metadata());
        ASSERT_EQ(placement_result.planned_plan.placements.size(), 2u);
        EXPECT_EQ(placement_result.planned_plan.placements[0].routed_expert_tier,
                  (std::vector<int>{0, 1, 0, 1, 0, 1}));

        plan.placements.clear();
        MoEExpertParallelPlannerOptions options;
        options.explicit_masks = {
            {.layer = 0, .tier_index = 0, .expert_ids = {0, 2, 4}},
            {.layer = 0, .tier_index = 1, .expert_ids = {1, 3, 5}},
            {.layer = 1, .tier_index = 0, .expert_ids = {1, 3, 5}},
            {.layer = 1, .tier_index = 1, .expert_ids = {0, 2, 4}},
        };
        const auto mask_result = MoEExpertParallelPlanner::plan(plan, metadata(), options);
        EXPECT_EQ(mask_result.planned_plan.placements[0].routed_expert_tier,
                  (std::vector<int>{0, 1, 0, 1, 0, 1}));
        EXPECT_EQ(mask_result.planned_plan.placements[1].routed_expert_tier,
                  (std::vector<int>{1, 0, 1, 0, 1, 0}));

        EXPECT_THROW((void)MoEExpertParallelPlanner::plan(plan, metadata()), std::invalid_argument);
    }

    TEST(Test__MoEExpertParallelPlanner, PlannedTopologiesSatisfyPhaseOneValidation)
    {
        auto two_tier = twoTierRocmCpuPlan();
        two_tier.routed_tiers[0].max_experts_per_layer = 3;
        const auto two_tier_result = MoEExpertParallelPlanner::plan(two_tier, metadata());
        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        two_tier_result.planned_plan,
                        {.layer_count = 2, .routed_expert_count = 6})
                        .ok());

        auto three_tier = threeTierCudaRocmCpuPlan();
        three_tier.routed_tiers[0].max_experts_per_layer = 2;
        three_tier.routed_tiers[1].max_experts_per_layer = 2;
        const auto three_tier_result = MoEExpertParallelPlanner::plan(three_tier, metadata());
        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        three_tier_result.planned_plan,
                        {.layer_count = 2, .routed_expert_count = 6})
                        .ok());
    }

    TEST(Test__MoEExpertParallelPlanner, NoFallbackTierCapacityMustCoverEveryExpert)
    {
        auto plan = twoTierRocmCpuPlan();
        plan.routed_tiers.pop_back();
        plan.routed_tiers[0].max_experts_per_layer = 5;

        try
        {
            (void)MoEExpertParallelPlanner::plan(plan, metadata());
            FAIL() << "Expected no-fallback capacity validation to throw";
        }
        catch (const std::invalid_argument &error)
        {
            const std::string message = error.what();
            EXPECT_NE(
                message.find("no-fallback routed tier capacity cannot cover every expert"),
                std::string::npos);
        }
    }

} // namespace llaminar2::test
