#include "execution/moe/MoEExpertParallelPlan.h"
#include "models/GraphTypes.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>

namespace llaminar2::test
{
namespace
{

ExpertComputeDomain singleGpuDomain(const std::string &name)
{
    ExpertComputeDomain domain;
    domain.name = name;
    domain.kind = ExpertDomainKind::SingleDevice;
    domain.backend = CollectiveBackendType::AUTO;
    domain.participants = {GlobalDeviceAddress::cuda(0)};
    domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
    return domain;
}

ExpertComputeDomain localGpuTPDomain(const std::string &name)
{
    ExpertComputeDomain domain;
    domain.name = name;
    domain.kind = ExpertDomainKind::LocalTP;
    domain.backend = CollectiveBackendType::RCCL;
    domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
    domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
    return domain;
}

ExpertComputeDomain cpuNodeLocalTPDomain(const std::string &name)
{
    ExpertComputeDomain domain;
    domain.name = name;
    domain.kind = ExpertDomainKind::NodeLocalTP;
    domain.backend = CollectiveBackendType::UPI;
    domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
    domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
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

MoEExpertParallelPlan validTwoTierPlan()
{
    MoEExpertParallelPlan plan;
    plan.enabled = true;
    plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
    plan.continuation_domain = "gpu_hot";
    plan.shared_expert_domain = "gpu_hot";
    plan.residency_policy = ExpertResidencyPolicy::HistogramTieredCache;
    plan.domains = {
        localGpuTPDomain("gpu_hot"),
        cpuNodeLocalTPDomain("cpu_cold"),
    };
    plan.routed_tiers = {
        tier("hot", "gpu_hot", 0),
        tier("cold", "cpu_cold", 1, true),
    };
    plan.placements = {
        placement(0, {0, 0, 1, 1}),
        placement(1, {0, 1, 0, 1}),
    };
    return plan;
}

MoEExpertParallelValidationOptions twoLayerFourExpertOptions()
{
    MoEExpertParallelValidationOptions options;
    options.layer_count = 2;
    options.routed_expert_count = 4;
    return options;
}

bool hasErrorContaining(const MoEExpertParallelValidationResult &result, const std::string &needle)
{
    return std::any_of(result.errors.begin(), result.errors.end(), [&](const std::string &error) {
        return error.find(needle) != std::string::npos;
    });
}

} // namespace

TEST(Test__MoEExpertParallelPlan, AcceptsValidTwoTierTieredExpertOverlayPlan)
{
    const auto plan = validTwoTierPlan();
    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(plan.isTieredOverlay());
}

TEST(Test__MoEExpertParallelPlan, AcceptsValidThreeTierTieredExpertOverlayPlan)
{
    auto plan = validTwoTierPlan();
    plan.domains = {
        localGpuTPDomain("nvidia_fast"),
        localGpuTPDomain("amd_warm"),
        cpuNodeLocalTPDomain("cpu_cold"),
    };
    plan.domains[0].backend = CollectiveBackendType::NCCL;
    plan.continuation_domain = "nvidia_fast";
    plan.shared_expert_domain = "nvidia_fast";
    plan.routed_tiers = {
        tier("hottest", "nvidia_fast", 0),
        tier("warm", "amd_warm", 1),
        tier("cold", "cpu_cold", 2, true),
    };
    plan.placements = {
        placement(0, {0, 1, 2, 2}),
        placement(1, {1, 0, 2, 1}),
    };

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_TRUE(result.ok());
}

TEST(Test__MoEExpertParallelPlan, EmbedsOptionalPlanInGraphConfigMoEConfig)
{
    GraphConfig config;
    EXPECT_EQ(config.moe.expert_parallel_plan, nullptr);

    config.moe.expert_parallel_plan = std::make_shared<MoEExpertParallelPlan>(validTwoTierPlan());

    ASSERT_NE(config.moe.expert_parallel_plan, nullptr);
    EXPECT_EQ(config.moe.expert_parallel_plan->execution_kind, MoEExpertExecutionKind::TieredExpertOverlay);
}

TEST(Test__MoEExpertParallelPlan, RejectsMissingExpertCoverage)
{
    auto plan = validTwoTierPlan();
    plan.placements[0].routed_expert_tier = {0, 1, 1};

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "does not cover every routed expert"));
}

TEST(Test__MoEExpertParallelPlan, RejectsUnassignedExpertCoverage)
{
    auto plan = validTwoTierPlan();
    plan.placements[0].routed_expert_tier = {0, -1, 1, 1};

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "without a tier"));
}

TEST(Test__MoEExpertParallelPlan, RejectsMissingLayerPlacementCoverageWhenLayerCountProvided)
{
    auto plan = validTwoTierPlan();
    plan.placements.pop_back();

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "missing expert layer placement for layer: 1"));
}

TEST(Test__MoEExpertParallelPlan, RejectsInvalidTierReferencesInLayerPlacement)
{
    auto plan = validTwoTierPlan();
    plan.placements[0].routed_expert_tier = {0, 1, 2, 1};

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "unknown tier index 2"));
}

TEST(Test__MoEExpertParallelPlan, RejectsMissingDomainReference)
{
    auto plan = validTwoTierPlan();
    plan.routed_tiers[1].domain = "missing_cpu_domain";

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "unknown expert compute domain: missing_cpu_domain"));
}

TEST(Test__MoEExpertParallelPlan, RejectsDuplicateDomainNames)
{
    auto plan = validTwoTierPlan();
    plan.domains[1].name = "gpu_hot";

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "duplicate expert compute domain name: gpu_hot"));
}

TEST(Test__MoEExpertParallelPlan, RejectsDuplicateTierNames)
{
    auto plan = validTwoTierPlan();
    plan.routed_tiers[1].name = "hot";

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "duplicate routed tier name: hot"));
}

TEST(Test__MoEExpertParallelPlan, RejectsMissingFallbackTier)
{
    auto plan = validTwoTierPlan();
    for (auto &routed_tier : plan.routed_tiers)
        routed_tier.fallback = false;

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "exactly one fallback tier"));
}

TEST(Test__MoEExpertParallelPlan, RejectsMultipleFallbackTiers)
{
    auto plan = validTwoTierPlan();
    for (auto &routed_tier : plan.routed_tiers)
        routed_tier.fallback = true;

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "exactly one fallback tier"));
}

TEST(Test__MoEExpertParallelPlan, RejectsTensorParallelExpertsWithoutMultiParticipantDomainScopedTP)
{
    auto plan = validTwoTierPlan();
    plan.domains[0] = singleGpuDomain("gpu_hot");
    plan.domains[0].compute_kind = ExpertDomainComputeKind::TensorParallelExperts;

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "TensorParallelExperts"));
    EXPECT_TRUE(hasErrorContaining(result, "multi-participant domain-scoped TP domain"));
}

TEST(Test__MoEExpertParallelPlan, RejectsExpertIdShardedWithoutExpertParallelCapableTPDomain)
{
    auto plan = validTwoTierPlan();
    plan.domains[1] = singleGpuDomain("cpu_cold");
    plan.domains[1].compute_kind = ExpertDomainComputeKind::ExpertIdSharded;

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "TPMode::ExpertParallel"));
}

TEST(Test__MoEExpertParallelPlan, RejectsSingleDomainExpertShardedAcrossMultipleDomains)
{
    auto plan = validTwoTierPlan();
    plan.execution_kind = MoEExpertExecutionKind::SingleDomainExpertSharded;

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasErrorContaining(result, "one compute domain"));
}

TEST(Test__MoEExpertParallelPlan, AcceptsSingleDomainExpertShardedWithExpertIdShardedDomain)
{
    MoEExpertParallelPlan plan;
    plan.enabled = true;
    plan.execution_kind = MoEExpertExecutionKind::SingleDomainExpertSharded;
    plan.continuation_domain = "cpu_sockets";
    plan.shared_expert_domain = "cpu_sockets";
    plan.residency_policy = ExpertResidencyPolicy::StaticById;
    plan.domains = {cpuNodeLocalTPDomain("cpu_sockets")};
    plan.domains[0].compute_kind = ExpertDomainComputeKind::ExpertIdSharded;
    plan.routed_tiers = {tier("routed", "cpu_sockets", 0, true)};
    plan.placements = {
        placement(0, {0, 0, 0, 0}),
        placement(1, {0, 0, 0, 0}),
    };

    const auto result = validateMoEExpertParallelPlan(plan, twoLayerFourExpertOptions());

    EXPECT_TRUE(result.ok());
}

TEST(Test__MoEExpertParallelPlan, TieredOverlayNamesSameLayerOverlayNotSequentialPipelineOwnership)
{
    const auto plan = validTwoTierPlan();

    EXPECT_EQ(plan.execution_kind, MoEExpertExecutionKind::TieredExpertOverlay);
    EXPECT_STREQ(toString(plan.execution_kind), "TieredExpertOverlay");
    EXPECT_TRUE(plan.isTieredOverlay());
}

} // namespace llaminar2::test
