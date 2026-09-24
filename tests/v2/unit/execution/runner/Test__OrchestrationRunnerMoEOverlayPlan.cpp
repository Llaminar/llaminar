#include "execution/runner/OrchestrationRunner.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "mocks/MockModelContext.h"

#include <gtest/gtest.h>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    constexpr int kMoELayers = 3;
    constexpr int kMoEExperts = 6;
    constexpr int kMoEDModel = 16;
    constexpr int kMoEIntermediate = 8;
    constexpr size_t kF32RoutedExpertBytes =
        3u * static_cast<size_t>(kMoEDModel) * static_cast<size_t>(kMoEIntermediate) * sizeof(float);

    std::shared_ptr<MockModelContext> makeMoEModelContext()
    {
        auto model_ctx = MockModelContextBuilder()
                             .setArchitecture("qwen3moe")
                             .setBlockCount(kMoELayers)
                             .setEmbeddingLength(kMoEDModel)
                             .setHeadCount(4)
                             .setHeadCountKV(2)
                             .setVocabSize(128)
                             .setContextLength(256)
                             .setFeedForwardLength(kMoEIntermediate)
                             .build();

        model_ctx->mockLoader().setIntParam("qwen3moe.expert_count", kMoEExperts);
        model_ctx->mockLoader().setIntParam("qwen3moe.expert_feed_forward_length", kMoEIntermediate);
        model_ctx->mockLoader().setIntParam("qwen3moe.expert_shared_count", 1);
        return model_ctx;
    }

    RoutedExpertDomain overlayDomain(
        const std::string &name,
        GlobalDeviceAddress participant)
    {
        RoutedExpertDomain domain;
        domain.name = name;
        domain.scope = ExecutionDomainScope::SINGLE;
        domain.backend = CollectiveBackendType::AUTO;
        domain.participants = {std::move(participant)};
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    RoutedExpertTier overlayTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        bool fallback = false)
    {
        RoutedExpertTier tier;
        tier.name = name;
        tier.domain = domain;
        tier.priority = priority;
        tier.fallback = fallback;
        return tier;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> makeRequestedOverlayPlan(
        RoutedExpertResidencyPolicy policy = RoutedExpertResidencyPolicy::StaticById)
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain = "gpu_hot";
        plan->shared_expert_domain = "gpu_hot";
        plan->residency_policy = policy;
        plan->domains = {
            overlayDomain("gpu_hot", GlobalDeviceAddress::cuda(0)),
            overlayDomain("cpu_cold", GlobalDeviceAddress::cpu()),
        };
        plan->routed_tiers = {
            overlayTier("hot", "gpu_hot", 0),
            overlayTier("cold", "cpu_cold", 1, true),
        };
        plan->routed_tiers[0].max_experts_per_layer = kMoEExperts;
        plan->routed_tiers[0].memory_budget_bytes = 2u * kF32RoutedExpertBytes;
        return plan;
    }

    RoutedExpertDomain rocmLocalTPReplicatedDomain(
        const std::string &name,
        int owner_rank)
    {
        RoutedExpertDomain domain;
        domain.name = name;
        domain.scope = ExecutionDomainScope::RANK_LOCAL;
        domain.backend = CollectiveBackendType::RCCL;
        domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        domain.owner_rank = owner_rank;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    RoutedExpertDomain cpuLocalTPReplicatedDomain(
        const std::string &name,
        int owner_rank)
    {
        RoutedExpertDomain domain;
        domain.name = name;
        domain.scope = ExecutionDomainScope::RANK_LOCAL;
        domain.backend = CollectiveBackendType::HOST;
        domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
        domain.owner_rank = owner_rank;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> makeHeterogeneousOverlayPlan()
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain = "rocm_hot";
        plan->base_model_domain = "rocm_hot";
        plan->shared_expert_domain = "rocm_hot";
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->domains = {
            rocmLocalTPReplicatedDomain("rocm_hot", 0),
            cpuLocalTPReplicatedDomain("cpu_cold", 1),
        };
        plan->routed_tiers = {
            overlayTier("hot", "rocm_hot", 0),
            overlayTier("cold", "cpu_cold", 1, true),
        };
        plan->routed_tiers[0].max_experts_per_layer = 128;
        return plan;
    }
}

TEST(Test__OrchestrationRunnerMoEOverlayPlan, FreezesMissingPlacementsFromLoadedModelMetadata)
{
    auto model_ctx = makeMoEModelContext();
    auto requested_plan = makeRequestedOverlayPlan();

    auto frozen_plan = freezeMoEExpertOverlayPlanForModel(
        *model_ctx,
        requested_plan,
        MTPRuntimeConfig{});

    ASSERT_NE(frozen_plan, nullptr);
    EXPECT_NE(frozen_plan.get(), requested_plan.get());
    EXPECT_TRUE(requested_plan->placements.empty());
    ASSERT_EQ(frozen_plan->placements.size(), static_cast<size_t>(kMoELayers));
    for (int layer = 0; layer < kMoELayers; ++layer)
    {
        const auto &placement = frozen_plan->placements[static_cast<size_t>(layer)];
        EXPECT_EQ(placement.layer, layer);
        EXPECT_EQ(placement.routed_expert_tier,
                  (std::vector<int>{0, 0, 1, 1, 1, 1}));
    }
}

TEST(Test__OrchestrationRunnerMoEOverlayPlan, KeepsExplicitPlacementsFrozen)
{
    auto model_ctx = makeMoEModelContext();
    auto explicit_plan = makeRequestedOverlayPlan(RoutedExpertResidencyPolicy::ExplicitMasks);
    explicit_plan->placements = {
        RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 0, 1, 0, 1}},
        RoutedExpertLayerPlacement{.layer = 1, .routed_expert_tier = {1, 0, 1, 0, 1, 0}},
        RoutedExpertLayerPlacement{.layer = 2, .routed_expert_tier = {0, 0, 1, 1, 0, 1}},
    };

    auto frozen_plan = freezeMoEExpertOverlayPlanForModel(
        *model_ctx,
        explicit_plan,
        MTPRuntimeConfig{});

    EXPECT_NE(frozen_plan, explicit_plan)
        << "topology-derived authority execution is sealed in an immutable clone";
    EXPECT_EQ(
        explicit_plan->authority_execution,
        MoEOverlayAuthorityExecutionKind::Unresolved);
    EXPECT_EQ(
        frozen_plan->authority_execution,
        MoEOverlayAuthorityExecutionKind::HostResident);
    ASSERT_EQ(frozen_plan->placements.size(), 3u);
    EXPECT_EQ(frozen_plan->placements[0].routed_expert_tier,
              (std::vector<int>{0, 1, 0, 1, 0, 1}));
}

TEST(Test__OrchestrationRunnerMoEOverlayPlan,
     CpuFallbackParticipantIsAnExecutableAuxiliaryRankRole)
{
    const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(
        makeHeterogeneousOverlayPlan(),
        MoEExpertOverlayExecutionPlanResolverOptions{
            .current_world_rank = 1,
            .world_size = 2,
        });

    const auto &rank = execution_plan.currentRankPlan();
    ASSERT_EQ(rank.world_rank, 1);
    EXPECT_EQ(rank.role, OverlayRankRole::CpuFallbackParticipant);
    EXPECT_TRUE(rank.hasRole(OverlayRankRole::CpuFallbackParticipant));
    EXPECT_FALSE(rank.ownsContinuationGraph());

    EXPECT_TRUE(rank.loads_expert_weights);
    EXPECT_FALSE(rank.local_devices.empty());
}
