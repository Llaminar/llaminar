#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "models/GraphTypes.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>

namespace llaminar2::test
{
    namespace
    {

        RoutedExpertDomain singleGpuDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::AUTO;
            domain.participants = {GlobalDeviceAddress::cuda(0)};
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain localGpuTPDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::LOCAL;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain cpuNodeLocalTPDomain(const std::string &name)
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

        MoERoutedExpertPlacementPlan validTwoTierPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "gpu_hot";
            plan.shared_expert_domain = "gpu_hot";
            plan.residency_policy = RoutedExpertResidencyPolicy::HistogramTieredCache;
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

        MoERoutedExpertPlacementValidationOptions twoLayerFourExpertOptions()
        {
            MoERoutedExpertPlacementValidationOptions options;
            options.layer_count = 2;
            options.routed_expert_count = 4;
            return options;
        }

        bool hasErrorContaining(const MoERoutedExpertPlacementValidationResult &result, const std::string &needle)
        {
            return std::any_of(result.errors.begin(), result.errors.end(), [&](const std::string &error)
                               { return error.find(needle) != std::string::npos; });
        }

    } // namespace

    TEST(Test__MoERoutedExpertPlacementPlan, AcceptsValidTwoTierTieredOverlayPlan)
    {
        const auto plan = validTwoTierPlan();
        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_TRUE(result.ok());
        EXPECT_TRUE(plan.isTieredOverlay());
    }

    TEST(Test__MoERoutedExpertPlacementPlan, AcceptsLeastLoadedAssignmentOnApportionedDomainScopedTPDomains)
    {
        auto plan = validTwoTierPlan();
        for (auto &domain : plan.domains)
            domain.routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_TRUE(result.ok()) << (result.errors.empty() ? "" : result.errors.front());
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsLeastLoadedAssignmentOnSingleParticipantDomains)
    {
        auto plan = validTwoTierPlan();
        plan.domains[0] = singleGpuDomain("gpu_hot");
        plan.domains[0].routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "routed_assignment=least-loaded-resident but is not a multi-participant collective domain"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsLeastLoadedAssignmentOnNonApportionedDomains)
    {
        auto plan = validTwoTierPlan();
        plan.domains[0].routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
        plan.domains[0].routed_assignment_policy = RoutedExpertAssignmentPolicy::LeastLoadedResident;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "routed_assignment=least-loaded-resident but does not use routed_compute=apportioned"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, AcceptsValidThreeTierTieredOverlayPlan)
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

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_TRUE(result.ok());
    }

    TEST(Test__MoERoutedExpertPlacementPlan, EmbedsOptionalPlanInGraphConfigMoEConfig)
    {
        GraphConfig config;
        EXPECT_EQ(config.moe.routed_expert_plan, nullptr);

        config.moe.routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(validTwoTierPlan());

        ASSERT_NE(config.moe.routed_expert_plan, nullptr);
        EXPECT_EQ(config.moe.routed_expert_plan->topology, RoutedExpertPlacementTopology::TieredOverlay);
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsMissingExpertCoverage)
    {
        auto plan = validTwoTierPlan();
        plan.placements[0].routed_expert_tier = {0, 1, 1};

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "does not cover every routed expert"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsUnassignedExpertCoverage)
    {
        auto plan = validTwoTierPlan();
        plan.placements[0].routed_expert_tier = {0, -1, 1, 1};

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "without a tier"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsMissingLayerPlacementCoverageWhenLayerCountProvided)
    {
        auto plan = validTwoTierPlan();
        plan.placements.pop_back();

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "missing expert layer placement for layer: 1"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsInvalidTierReferencesInLayerPlacement)
    {
        auto plan = validTwoTierPlan();
        plan.placements[0].routed_expert_tier = {0, 1, 2, 1};

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "unknown tier index 2"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsMissingDomainReference)
    {
        auto plan = validTwoTierPlan();
        plan.routed_tiers[1].domain = "missing_cpu_domain";

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "unknown routed expert domain: missing_cpu_domain"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsDuplicateDomainNames)
    {
        auto plan = validTwoTierPlan();
        plan.domains[1].name = "gpu_hot";

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "duplicate expert compute domain name: gpu_hot"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsDuplicateTierNames)
    {
        auto plan = validTwoTierPlan();
        plan.routed_tiers[1].name = "hot";

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "duplicate routed tier name: hot"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, AcceptsMissingFallbackTier)
    {
        auto plan = validTwoTierPlan();
        for (auto &routed_tier : plan.routed_tiers)
            routed_tier.fallback = false;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_TRUE(result.ok());
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsMultipleFallbackTiers)
    {
        auto plan = validTwoTierPlan();
        for (auto &routed_tier : plan.routed_tiers)
            routed_tier.fallback = true;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "at most one fallback tier"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsTensorShardingWithoutCollectiveParticipants)
    {
        auto plan = validTwoTierPlan();
        plan.domains[0] = singleGpuDomain("gpu_hot");
        plan.domains[0].routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "routed_compute=tensor-sharded"));
        EXPECT_TRUE(hasErrorContaining(result, "multi-participant collective domain"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, AcceptsSingleParticipantApportionmentAsDegenerateOwnership)
    {
        auto plan = validTwoTierPlan();
        plan.domains[1] = singleGpuDomain("cpu_cold");
        plan.domains[1].routed_compute_policy = RoutedExpertComputePolicy::Apportioned;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_TRUE(result.ok()) << "Apportionment on one participant assigns every whole expert to that participant";
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsSingleDomainTopologyAcrossMultipleDomains)
    {
        auto plan = validTwoTierPlan();
        plan.topology = RoutedExpertPlacementTopology::SingleDomain;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "one routed compute domain"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, AcceptsSingleDomainWithApportionedDomain)
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::SingleDomain;
        plan.continuation_domain = "cpu_sockets";
        plan.shared_expert_domain = "cpu_sockets";
        plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan.domains = {cpuNodeLocalTPDomain("cpu_sockets")};
        plan.domains[0].routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        plan.routed_tiers = {tier("routed", "cpu_sockets", 0, true)};
        plan.placements = {
            placement(0, {0, 0, 0, 0}),
            placement(1, {0, 0, 0, 0}),
        };

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_TRUE(result.ok());
    }

    TEST(Test__MoERoutedExpertPlacementPlan, TieredOverlayNamesSameLayerOverlayNotSequentialPipelineOwnership)
    {
        const auto plan = validTwoTierPlan();

        EXPECT_EQ(plan.topology, RoutedExpertPlacementTopology::TieredOverlay);
        EXPECT_STREQ(toString(plan.topology), "tiered-overlay");
        EXPECT_TRUE(plan.isTieredOverlay());
    }

} // namespace llaminar2::test
