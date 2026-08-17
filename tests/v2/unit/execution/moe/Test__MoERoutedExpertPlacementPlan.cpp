/**
 * @file Test__MoERoutedExpertPlacementPlan.cpp
 * @brief Unit coverage for declarative routed-expert placement policy.
 *
 * These tests keep routed weight placement, phase scheduling, decode row
 * assignment, and prefill row assignment independent.  That distinction is
 * essential for the economical LLEP lane: tiny serial/grouped decode stays on
 * canonical expert owners while ordinary prefill may distribute the current
 * batch by least-loaded complete-expert residency.
 */

#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "execution/moe/MoEExpertOverlayAuthorityPlan.h"
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
                cpuNodeTPDomain("cpu_cold"),
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

    /**
     * @brief Canonical placement storage must include routed MTP source layers.
     *
     * Qwen NextN weights can live immediately after the ordinary transformer
     * interval. The durable main table is the placement authority for both the
     * main graph and its depth-scoped child, regardless of which graph is built
     * first, so its capacity follows the complete explicit placement plan.
     */
    TEST(Test__MoERoutedExpertPlacementPlan,
         PlacementLayerCapacityIncludesTrailingMTPSourceLayer)
    {
        auto plan = validTwoTierPlan();
        plan.placements.push_back(placement(40, {0, 1, 0, 1}));

        EXPECT_EQ(plan.placementLayerCapacity(/*minimum_layers=*/40), 41);
        EXPECT_EQ(plan.placementLayerCapacity(/*minimum_layers=*/48), 48);
    }

    TEST(Test__MoERoutedExpertPlacementPlan, AcceptsIndependentLeastLoadedAssignmentsOnApportionedDomainScopedTPDomains)
    {
        auto plan = validTwoTierPlan();
        for (auto &domain : plan.domains)
        {
            domain.routed_decode_assignment_policy =
                RoutedExpertAssignmentPolicy::LeastLoadedResident;
            domain.routed_prefill_assignment_policy =
                RoutedExpertAssignmentPolicy::LeastLoadedResident;
        }

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_TRUE(result.ok()) << (result.errors.empty() ? "" : result.errors.front());
    }

    TEST(Test__MoERoutedExpertPlacementPlan, PhaseSplitReplicasUseParticipantAssignedPrefill)
    {
        auto domain = localGpuTPDomain("gpu_hot");
        domain.routed_compute_policy = RoutedExpertComputePolicy::Replicated;
        domain.routed_phase_policy =
            RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        EXPECT_TRUE(domain.usesParticipantAssignedPrefill());
        EXPECT_TRUE(domain.supportsLeastLoadedResidentAssignment());
    }

    TEST(Test__MoERoutedExpertPlacementPlan, UniformReplicasDoNotClaimParticipantAssignedPrefill)
    {
        auto domain = localGpuTPDomain("gpu_hot");
        domain.routed_compute_policy = RoutedExpertComputePolicy::Replicated;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;

        EXPECT_FALSE(domain.usesParticipantAssignedPrefill());
        EXPECT_FALSE(domain.supportsLeastLoadedResidentAssignment());
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsLeastLoadedPrefillAssignmentOnSingleParticipantDomains)
    {
        auto plan = validTwoTierPlan();
        plan.domains[0] = singleGpuDomain("gpu_hot");
        plan.domains[0].routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "routed_prefill_assignment=least-loaded-resident but is not a multi-participant collective domain"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsLeastLoadedDecodeAssignmentOnSingleParticipantDomains)
    {
        auto plan = validTwoTierPlan();
        plan.domains[0] = singleGpuDomain("gpu_hot");
        plan.domains[0].routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        const auto result = validateMoERoutedExpertPlacementPlan(
            plan,
            twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "routed_decode_assignment=least-loaded-resident but is not a multi-participant collective domain"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsLeastLoadedPrefillAssignmentWhenWorkIsNotParticipantAssigned)
    {
        auto plan = validTwoTierPlan();
        plan.domains[0].routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
        plan.domains[0].routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "routed_prefill_assignment=least-loaded-resident but that workload does not use participant-assigned complete experts"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, RejectsLeastLoadedDecodeAssignmentWhenWorkIsNotParticipantAssigned)
    {
        auto plan = validTwoTierPlan();
        plan.domains[0].routed_compute_policy =
            RoutedExpertComputePolicy::TensorSharded;
        plan.domains[0].routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;

        const auto result = validateMoERoutedExpertPlacementPlan(
            plan,
            twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "routed_decode_assignment=least-loaded-resident but that workload does not use participant-assigned complete experts"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, AcceptsValidThreeTierTieredOverlayPlan)
    {
        auto plan = validTwoTierPlan();
        plan.domains = {
            localGpuTPDomain("nvidia_fast"),
            localGpuTPDomain("amd_warm"),
            cpuNodeTPDomain("cpu_cold"),
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

    TEST(Test__MoERoutedExpertPlacementPlan, RequiresUniqueIntegerPriorities)
    {
        auto plan = validTwoTierPlan();
        plan.routed_tiers[0].name = "opaque-alpha";
        plan.routed_tiers[1].name = "opaque-beta";
        plan.routed_tiers[1].priority = plan.routed_tiers[0].priority;

        const auto result = validateMoERoutedExpertPlacementPlan(
            plan,
            twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "priorities must be unique because tier names and declaration order carry no preference semantics"));
    }

    TEST(Test__MoERoutedExpertPlacementPlan, CoverageRoleCannotOverridePriorityOrder)
    {
        auto plan = validTwoTierPlan();
        plan.routed_tiers[0].name = "opaque-alpha";
        plan.routed_tiers[0].priority = 41;
        plan.routed_tiers[1].name = "opaque-beta";
        plan.routed_tiers[1].priority = -9;

        const auto result = validateMoERoutedExpertPlacementPlan(
            plan,
            twoLayerFourExpertOptions());

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(
            result,
            "fallback tier must have the greatest numeric priority"));
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
        plan.domains = {cpuNodeTPDomain("cpu_sockets")};
        plan.domains[0].routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        plan.routed_tiers = {tier("routed", "cpu_sockets", 0, true)};
        plan.placements = {
            placement(0, {0, 0, 0, 0}),
            placement(1, {0, 0, 0, 0}),
        };

        const auto result = validateMoERoutedExpertPlacementPlan(plan, twoLayerFourExpertOptions());

        EXPECT_TRUE(result.ok());
        EXPECT_TRUE(plan.usesExpertOverlayAuthority())
            << "single-domain is a topology cardinality, not a legacy authority";
    }

    TEST(Test__MoERoutedExpertPlacementPlan,
         SynthesizesDynamicOneTierAuthorityForImplicitLocalTPMoE)
    {
        const auto normalized =
            normalizeMoEExpertOverlayAuthorityPlan({
                .model_has_routed_experts = true,
                .world_rank = 3,
                .local_tp_participants = {
                    GlobalDeviceAddress::cuda(0),
                    GlobalDeviceAddress::cuda(1),
                },
                .local_tp_weights = {0.5f, 0.5f},
                .local_tp_backend = CollectiveBackendType::NCCL,
                .routed_compute_policy =
                    RoutedExpertComputePolicy::Apportioned,
                .owner_order = RoutedExpertOwnerOrder::Random,
                .residency_maintenance =
                    MoERebalanceRuntimeMode::Dynamic,
            });

        ASSERT_TRUE(normalized.synthesized());
        ASSERT_NE(normalized.plan, nullptr);
        EXPECT_TRUE(normalized.plan->usesExpertOverlayAuthority());
        EXPECT_FALSE(normalized.plan->isTieredOverlay());
        EXPECT_EQ(
            normalized.plan->topology,
            RoutedExpertPlacementTopology::SingleDomain);
        EXPECT_EQ(
            normalized.plan->residency_policy,
            RoutedExpertResidencyPolicy::RoutedTierRebalanced);
        EXPECT_EQ(
            normalized.plan->authority_execution,
            MoEOverlayAuthorityExecutionKind::HomogeneousDeviceResident);
        EXPECT_EQ(
            normalized.plan->owner_order,
            RoutedExpertOwnerOrder::Random);
        ASSERT_EQ(normalized.plan->domains.size(), 1u);
        EXPECT_EQ(normalized.plan->domains[0].scope,
                  ExecutionDomainScope::RANK_LOCAL);
        EXPECT_EQ(normalized.plan->domains[0].owner_rank, 3);
        EXPECT_EQ(normalized.plan->domains[0].participants.size(), 2u);
        ASSERT_EQ(normalized.plan->routed_tiers.size(), 1u);
        EXPECT_EQ(normalized.plan->routed_tiers[0].priority, 0);
        EXPECT_TRUE(normalized.plan->routed_tiers[0].fallback);
        EXPECT_EQ(
            normalized.plan->continuation_domain_spec.effectiveDensePolicy(),
            DenseParallelPolicy::TensorParallel);
        EXPECT_TRUE(
            validateMoERoutedExpertPlacementPlan(*normalized.plan).ok());
    }

    TEST(Test__MoERoutedExpertPlacementPlan,
         SynthesizesStaticAuthorityWhenMaintenanceIsOff)
    {
        const auto normalized =
            normalizeMoEExpertOverlayAuthorityPlan({
                .model_has_routed_experts = true,
                .local_tp_participants = {
                    GlobalDeviceAddress::rocm(0),
                    GlobalDeviceAddress::rocm(1),
                },
                .local_tp_backend = CollectiveBackendType::RCCL,
                .residency_maintenance =
                    MoERebalanceRuntimeMode::Off,
            });

        ASSERT_TRUE(normalized.synthesized());
        ASSERT_NE(normalized.plan, nullptr);
        EXPECT_EQ(
            normalized.plan->residency_policy,
            RoutedExpertResidencyPolicy::StaticById);
        EXPECT_EQ(
            normalized.plan->authority_execution,
            MoEOverlayAuthorityExecutionKind::HomogeneousDeviceResident);
    }

    TEST(Test__MoERoutedExpertPlacementPlan,
         MultiTierAuthorityIsHostCoordinatedEvenForOneGpuVendor)
    {
        auto plan = validTwoTierPlan();
        plan.domains[1] = localGpuTPDomain("gpu_second_priority");
        plan.routed_tiers[1].domain = "gpu_second_priority";

        EXPECT_EQ(
            resolveMoEOverlayAuthorityExecutionKind(plan),
            MoEOverlayAuthorityExecutionKind::HostCoordinated);
    }

    TEST(Test__MoERoutedExpertPlacementPlan,
         HeterogeneousSingleTierAuthorityIsHostCoordinated)
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::SingleDomain;
        RoutedExpertDomain domain;
        domain.name = "mixed_accelerators";
        domain.scope = ExecutionDomainScope::RANK_LOCAL;
        domain.backend = CollectiveBackendType::HETEROGENEOUS;
        domain.participants = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::rocm(0),
        };
        plan.domains = {domain};
        plan.routed_tiers = {
            tier("priority_0", domain.name, 0, true),
        };

        EXPECT_EQ(
            resolveMoEOverlayAuthorityExecutionKind(plan),
            MoEOverlayAuthorityExecutionKind::HostCoordinated);
    }

    TEST(Test__MoERoutedExpertPlacementPlan,
         ExplicitHostStagingCannotMasqueradeAsDeviceResidentAuthority)
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::SingleDomain;
        RoutedExpertDomain domain;
        domain.name = "cuda_host_staged";
        domain.scope = ExecutionDomainScope::RANK_LOCAL;
        domain.backend = CollectiveBackendType::HOST;
        domain.participants = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
        };
        plan.domains = {domain};
        plan.routed_tiers = {
            tier("priority_0", domain.name, 0, true),
        };
        plan.authority_execution =
            MoEOverlayAuthorityExecutionKind::HomogeneousDeviceResident;

        const auto validation =
            validateMoERoutedExpertPlacementPlan(plan);
        EXPECT_FALSE(validation.ok());
        EXPECT_TRUE(hasErrorContaining(
            validation,
            "does not match topology-required 'host-coordinated'"));
        EXPECT_THROW(
            (void)normalizeMoEExpertOverlayAuthorityPlan({
                .requested_plan =
                    std::make_shared<MoERoutedExpertPlacementPlan>(plan),
                .model_has_routed_experts = true,
            }),
            std::invalid_argument);
    }

    TEST(Test__MoERoutedExpertPlacementPlan,
         DoesNotCreateAuthorityForDenseOrSingleDeviceExecution)
    {
        const auto dense = normalizeMoEExpertOverlayAuthorityPlan({
            .model_has_routed_experts = false,
            .local_tp_participants = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1),
            },
        });
        EXPECT_EQ(dense.plan, nullptr);
        EXPECT_FALSE(dense.synthesized());

        const auto serial = normalizeMoEExpertOverlayAuthorityPlan({
            .model_has_routed_experts = true,
            .local_tp_participants = {
                GlobalDeviceAddress::cuda(0),
            },
        });
        EXPECT_EQ(serial.plan, nullptr);
        EXPECT_FALSE(serial.synthesized());
    }

    TEST(Test__MoERoutedExpertPlacementPlan,
         FailsClosedBeforeLegacyAuthorityForUnrepresentedImplicitTopologies)
    {
        EXPECT_THROW(
            (void)normalizeMoEExpertOverlayAuthorityPlan({
                .model_has_routed_experts = true,
                .local_tp_participants = {
                    GlobalDeviceAddress::cuda(0),
                    GlobalDeviceAddress::cuda(1),
                },
                .routed_compute_policy =
                    RoutedExpertComputePolicy::TensorSharded,
            }),
            std::invalid_argument);

        EXPECT_THROW(
            (void)normalizeMoEExpertOverlayAuthorityPlan({
                .model_has_routed_experts = true,
                .has_cross_rank_tensor_parallel = true,
            }),
            std::logic_error);
    }

    TEST(Test__MoERoutedExpertPlacementPlan, TieredOverlayNamesSameLayerOverlayNotSequentialPipelineOwnership)
    {
        const auto plan = validTwoTierPlan();

        EXPECT_EQ(plan.topology, RoutedExpertPlacementTopology::TieredOverlay);
        EXPECT_STREQ(toString(plan.topology), "tiered-overlay");
        EXPECT_TRUE(plan.isTieredOverlay());
    }

} // namespace llaminar2::test
