/**
 * @file Test__MoEOverlayDeviceControllerTopology.cpp
 * @brief Device-free proof of topology-generic all-GPU authority resolution.
 *
 * These tests deliberately reverse accelerator vendors, rank placement, tier
 * declaration order, continuation roots, and integer priorities. They prove
 * that controller ownership follows the declared continuation root and live
 * device state rather than the words hot/warm/cold or one machine-specific
 * CUDA/ROCm/socket arrangement.
 */

#include "execution/moe/MoEOverlayDeviceControllerTopology.h"
#include "execution/moe/MoEOverlayNodeLocalDeviceControllerFabric.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** Build one homogeneous rank-local GPU domain with explicit ownership. */
        RoutedExpertDomain gpuDomain(
            std::string name,
            DeviceType type,
            int count,
            int world_rank)
        {
            RoutedExpertDomain domain;
            domain.name = std::move(name);
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.backend = type == DeviceType::CUDA
                                 ? CollectiveBackendType::NCCL
                                 : CollectiveBackendType::RCCL;
            domain.owner_rank = world_rank;
            domain.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            for (int ordinal = 0; ordinal < count; ++ordinal)
            {
                domain.participants.push_back(
                    type == DeviceType::CUDA
                        ? GlobalDeviceAddress::cuda(ordinal)
                        : GlobalDeviceAddress::rocm(ordinal));
            }
            return domain;
        }

        /** Build one tier without interpreting its integer priority. */
        RoutedExpertTier tier(
            std::string name,
            std::string domain,
            int priority,
            bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = std::move(name);
            result.domain = std::move(domain);
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        /** Build the target two-domain 2xCUDA/4xROCm node-local topology. */
        MoERoutedExpertPlacementPlan cudaContinuationPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "generation";
            plan.continuation_domain_spec.domain = "generation";
            plan.continuation_domain_spec.logical_root_participant = 1;
            plan.shared_expert_domain = "generation";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.authority_execution =
                MoEOverlayAuthorityExecutionKind::DeviceResident;
            plan.domains = {
                gpuDomain("generation", DeviceType::CUDA, 2, 0),
                gpuDomain("capacity_pool", DeviceType::ROCm, 4, 1),
            };
            plan.routed_tiers = {
                tier("opaque_a", "generation", -7),
                tier("opaque_b", "capacity_pool", 41, true),
            };
            plan.placements = {{
                .layer = 0,
                .routed_expert_tier =
                    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
            }};
            return plan;
        }

        /** Reverse vendor roles and declaration order to expose hard-coding. */
        MoERoutedExpertPlacementPlan rocmContinuationPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "continuation_any_vendor";
            plan.continuation_domain_spec.domain =
                "continuation_any_vendor";
            plan.continuation_domain_spec.logical_root_participant = 0;
            plan.shared_expert_domain = "continuation_any_vendor";
            plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan.authority_execution =
                MoEOverlayAuthorityExecutionKind::DeviceResident;
            plan.domains = {
                gpuDomain("secondary_any_vendor", DeviceType::CUDA, 4, 1),
                gpuDomain("continuation_any_vendor", DeviceType::ROCm, 2, 0),
            };
            plan.routed_tiers = {
                tier("integer_priority_only_0", "secondary_any_vendor", 3),
                tier("integer_priority_only_1", "continuation_any_vendor", 99, true),
            };
            plan.placements = {{
                .layer = 0,
                .routed_expert_tier = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
            }};
            return plan;
        }

        /** Build a single heterogeneous tier whose members cannot share NCCL/RCCL. */
        MoERoutedExpertPlacementPlan mixedSingleTierPlan(
            CollectiveBackendType backend =
                CollectiveBackendType::HETEROGENEOUS)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::SingleDomain;
            plan.continuation_domain = "mixed";
            plan.continuation_domain_spec.domain = "mixed";
            plan.shared_expert_domain = "mixed";
            plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan.authority_execution =
                MoEOverlayAuthorityExecutionKind::DeviceResident;
            RoutedExpertDomain domain;
            domain.name = "mixed";
            domain.scope = ExecutionDomainScope::NODE_LOCAL;
            domain.backend = backend;
            domain.participants = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::rocm(0),
            };
            domain.world_ranks = {0, 1};
            domain.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            plan.domains = {std::move(domain)};
            plan.routed_tiers = {tier("priority_17", "mixed", 17, true)};
            plan.placements = {{
                .layer = 0,
                .routed_expert_tier = {0, 0, 0, 0},
            }};
            return plan;
        }

        /** Resolve from one plan after constructing its immutable first owner map. */
        MoEOverlayDeviceControllerTopology resolve(
            const MoERoutedExpertPlacementPlan &plan,
            const std::vector<int> &node_ids)
        {
            return resolveMoEOverlayDeviceControllerTopology(
                plan,
                MoEExpertOwnerMap::build(plan),
                {.world_rank_node_ids = node_ids});
        }
    } // namespace

    TEST(Test__MoEOverlayDeviceControllerTopology,
         TwoCudaContinuationAndFourRocmFollowersUseTwoNativeGroups)
    {
        const auto topology = resolve(
            cudaContinuationPlan(), std::vector<int>{9, 9});

        ASSERT_TRUE(topology.valid());
        EXPECT_EQ(topology.leader_participant_id, 1);
        EXPECT_EQ(topology.leader_world_rank, 0);
        EXPECT_TRUE(topology.leader_device.is_cuda());
        ASSERT_EQ(topology.groups.size(), 2u);
        EXPECT_EQ(topology.groups[0].participant_ids,
                  (std::vector<int>{0, 1}));
        EXPECT_EQ(topology.groups[0].tier_priority, -7);
        EXPECT_EQ(
            topology.groups[0].intra_group_transport,
            MoEOverlayDeviceControllerIntraGroupTransport::NativeCollective);
        EXPECT_EQ(
            topology.groups[0].inter_group_transport,
            MoEOverlayDeviceControllerInterGroupTransport::LeaderLocal);
        EXPECT_EQ(topology.groups[1].participant_ids,
                  (std::vector<int>{2, 3, 4, 5}));
        EXPECT_EQ(topology.groups[1].tier_priority, 41);
        EXPECT_EQ(
            topology.groups[1].inter_group_transport,
            MoEOverlayDeviceControllerInterGroupTransport::NodeLocalMapped);
        EXPECT_TRUE(topology.usesNodeLocalMappedControl());
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         ReversedVendorAndRankPlacementStillSelectsDeclaredContinuationRoot)
    {
        const auto topology = resolve(
            rocmContinuationPlan(), std::vector<int>{4, 4});

        ASSERT_TRUE(topology.valid());
        EXPECT_EQ(topology.leader_participant_id, 4);
        EXPECT_EQ(topology.leader_world_rank, 0);
        EXPECT_TRUE(topology.leader_device.is_rocm());
        ASSERT_EQ(topology.groups.size(), 2u);
        EXPECT_EQ(topology.leader_group_id, 1)
            << "leader identity must not depend on tier/group declaration order";
        EXPECT_EQ(topology.groups[0].participant_ids,
                  (std::vector<int>{0, 1, 2, 3}));
        EXPECT_EQ(topology.groups[1].participant_ids,
                  (std::vector<int>{4, 5}));
        EXPECT_EQ(topology.groups[1].tier_priority, 99);
        EXPECT_EQ(
            topology.groups[1].inter_group_transport,
            MoEOverlayDeviceControllerInterGroupTransport::LeaderLocal);
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         HeterogeneousSingleTierUsesMappedSingletonFollowers)
    {
        const auto topology = resolve(
            mixedSingleTierPlan(), std::vector<int>{2, 2});

        ASSERT_TRUE(topology.valid());
        ASSERT_EQ(topology.groups.size(), 2u);
        for (const auto &group : topology.groups)
        {
            EXPECT_EQ(group.participant_ids.size(), 1u);
            EXPECT_EQ(
                group.intra_group_transport,
                MoEOverlayDeviceControllerIntraGroupTransport::SingleParticipant);
        }
        EXPECT_TRUE(topology.usesNodeLocalMappedControl());
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         HostStagedGpuDomainStillHasDeviceOwnedControllerGroups)
    {
        auto plan = mixedSingleTierPlan(CollectiveBackendType::HOST);
        plan.domains.front().participants = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
        };
        plan.domains.front().scope = ExecutionDomainScope::RANK_LOCAL;
        plan.domains.front().owner_rank = 0;
        plan.domains.front().world_ranks.clear();
        const auto topology = resolve(plan, std::vector<int>{1});

        ASSERT_TRUE(topology.valid());
        ASSERT_EQ(topology.groups.size(), 2u);
        EXPECT_TRUE(topology.leader_device.is_cuda());
        EXPECT_TRUE(topology.usesNodeLocalMappedControl());
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         CrossNodeAndUnresolvedRankLocalityFailInsteadOfUsingHostPolicy)
    {
        const auto plan = cudaContinuationPlan();
        const auto owner_map = MoEExpertOwnerMap::build(plan);
        const std::vector<int> cross_node_ids{0, 1};
        const std::vector<int> no_node_ids;
        EXPECT_THROW(
            (void)resolveMoEOverlayDeviceControllerTopology(
                plan, owner_map, {.world_rank_node_ids = cross_node_ids}),
            std::logic_error);
        EXPECT_THROW(
            (void)resolveMoEOverlayDeviceControllerTopology(
                plan, owner_map, {.world_rank_node_ids = no_node_ids}),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         CpuOrHostResidentPlansCannotEnterTheDeviceController)
    {
        auto plan = cudaContinuationPlan();
        plan.domains[1].participants[0] = GlobalDeviceAddress::cpu();
        plan.authority_execution =
            MoEOverlayAuthorityExecutionKind::HostResident;
        EXPECT_THROW(
            (void)resolve(
                plan, std::vector<int>{0, 0}),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         IntegerPriorityChangesAuthenticatedTopologyButNotLeaderSelection)
    {
        auto first = cudaContinuationPlan();
        auto second = first;
        second.routed_tiers[1].priority = 700;
        const auto first_topology = resolve(first, std::vector<int>{3, 3});
        const auto second_topology = resolve(second, std::vector<int>{3, 3});

        EXPECT_EQ(
            first_topology.leader_participant_id,
            second_topology.leader_participant_id);
        EXPECT_NE(
            first_topology.topology_fingerprint,
            second_topology.topology_fingerprint);
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         FabricLayoutPageIsolatesLeaderAndVariableGroupSnapshotWriters)
    {
        const auto topology = resolve(
            cudaContinuationPlan(), std::vector<int>{9, 9});
        const auto layout =
            MoEOverlayNodeLocalDeviceControllerFabric::planLayout(
                topology,
                /*num_layers=*/48u,
                /*num_experts=*/256u,
                /*command_capacity=*/32u);

        ASSERT_TRUE(layout.valid());
        EXPECT_EQ(layout.page_bytes, 4096u);
        EXPECT_EQ(layout.leader_owned_begin, layout.page_bytes);
        EXPECT_EQ(layout.leader_owned_end % layout.page_bytes, 0u);
        EXPECT_EQ(
            layout.header.inference_epoch_record_offset %
                alignof(
                    MoEOverlayDeviceControllerInferenceEpochRecord),
            0u);
        EXPECT_GE(
            layout.header.inference_epoch_record_offset,
            layout.header.controller_header_offset +
                sizeof(MoEOverlayDeviceControllerSharedHeader));
        EXPECT_GE(
            layout.header.command_header_offset,
            layout.header.inference_epoch_record_offset +
                sizeof(
                    MoEOverlayDeviceControllerInferenceEpochRecord));
        ASSERT_EQ(layout.groups.size(), 2u);
        EXPECT_EQ(
            layout.groups[0].collected_state_words,
            2u * 48u * 256u);
        EXPECT_EQ(
            layout.groups[1].collected_state_words,
            4u * 48u * 256u);
        EXPECT_EQ(
            layout.groups[0].owned_page_begin,
            layout.leader_owned_end);
        EXPECT_EQ(
            layout.groups[1].owned_page_begin,
            layout.groups[0].owned_page_end);
        EXPECT_EQ(
            layout.groups.back().owned_page_end,
            layout.mapping_bytes);
        EXPECT_EQ(
            layout.header.topology_fingerprint,
            topology.topology_fingerprint);
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         FabricLayoutHasNoFixedTierOrContinuationVendorCardinality)
    {
        auto plan = cudaContinuationPlan();
        plan.domains.push_back(
            gpuDomain("additional_integer_tier", DeviceType::CUDA, 1, 0));
        plan.domains.back().participants.front() =
            GlobalDeviceAddress::cuda(2);
        plan.routed_tiers.push_back(
            tier("opaque_c", "additional_integer_tier", 6));
        plan.placements.front().routed_expert_tier =
            {0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1};
        const auto topology = resolve(plan, std::vector<int>{5, 5});
        const auto layout =
            MoEOverlayNodeLocalDeviceControllerFabric::planLayout(
                topology, 4u, 12u, 7u);

        ASSERT_TRUE(layout.valid());
        ASSERT_EQ(layout.groups.size(), 3u);
        EXPECT_EQ(layout.header.group_count, 3u);
        EXPECT_EQ(layout.header.participant_count, 7u);
        EXPECT_EQ(layout.header.command_capacity, 7u);
        EXPECT_EQ(layout.groups[0].group_id, 0u);
        EXPECT_EQ(layout.groups[1].group_id, 1u);
        EXPECT_EQ(layout.groups[2].group_id, 2u);
    }

    TEST(Test__MoEOverlayDeviceControllerTopology,
         FabricLayoutRejectsGeometryOutsideTheSharedRuntimeABI)
    {
        const auto topology = resolve(
            cudaContinuationPlan(), std::vector<int>{9, 9});
        EXPECT_THROW(
            (void)MoEOverlayNodeLocalDeviceControllerFabric::planLayout(
                topology, 0u, 256u, 1u),
            std::invalid_argument);
        EXPECT_THROW(
            (void)MoEOverlayNodeLocalDeviceControllerFabric::planLayout(
                topology, 48u, 257u, 1u),
            std::invalid_argument);
        EXPECT_THROW(
            (void)MoEOverlayNodeLocalDeviceControllerFabric::planLayout(
                topology, 48u, 256u, 0u),
            std::invalid_argument);
    }
} // namespace llaminar2::test
