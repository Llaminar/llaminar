/**
 * @file Test__ServerRankMembership.cpp
 * @brief Device-free proofs for the server's physical membership witness.
 *
 * These tests use immutable inventory values, never initialize MPI or inspect
 * hardware, and ensure observation cannot infer physical nodes from hostnames,
 * NUMA IDs, or an assumed rank-zero request authority.
 */
#include "app/modes/ServerRankMembership.h"
#include "app/modes/ServerExecutionEvidence.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
    /** @return Three execution ranks on two physical hosts with reused labels. */
    ClusterInventory inventory()
    {
        ClusterInventory result;
        result.world_size = 3;
        result.node_count = 2;
        result.ranks = {{.rank = 0, .node_id = 0, .local_rank = 0, .hostname = "container"},
                        {.rank = 1, .node_id = 1, .local_rank = 0, .hostname = "container"},
                        {.rank = 2, .node_id = 1, .local_rank = 1, .hostname = "container"}};
        return result;
    }
}

TEST(ServerRankMembership, PreservesPhysicalGroupsAndNonzeroAuthority)
{
    const auto topology = inventory();
    for (int rank = 0; rank < topology.world_size; ++rank)
    {
        const auto tags = serverRankMembershipTags(topology, rank, 1);
        EXPECT_EQ(tags.at("rank"), std::to_string(rank));
        EXPECT_EQ(tags.at("authority_rank"), "1");
        EXPECT_EQ(tags.at("world_size"), "3");
        EXPECT_EQ(tags.at("node_id"), rank == 0 ? "0" : "1");
        EXPECT_EQ(tags.at("local_rank"), rank == 2 ? "1" : "0");
        EXPECT_EQ(tags.at("hostname"), "container");
        EXPECT_EQ(tags.at("identity_source"), "communicator_cluster_inventory");
    }
}

TEST(ServerRankMembership, RejectsDifferentCommunicatorAndMissingPhysicalIdentity)
{
    const auto check = [](auto mutate)
    {
        auto topology = inventory();
        mutate(topology);
        EXPECT_THROW(serverRankMembershipTags(topology, 1, 1), std::invalid_argument);
    };
    check([](auto &v) { v.world_size = 0; });
    check([](auto &v) { v.ranks.pop_back(); });
    check([](auto &v) { v.ranks[1].rank = 5; });
    check([](auto &v) { v.ranks[1].node_id = -1; });
    check([](auto &v) { v.ranks[1].node_id = 2; });
    check([](auto &v) { v.ranks[1].local_rank = -1; });
    check([](auto &v) { v.ranks[1].hostname.clear(); });
    check([](auto &v) { v.ranks[1].hostname = "host\nforeign"; });
    check([](auto &v) { v.ranks[1].hostname = std::string("host\0foreign", 12); });
    const auto topology = inventory();
    for (const int rank : {-1, 3})
        EXPECT_THROW(serverRankMembershipTags(topology, rank, 1), std::invalid_argument);
    for (const int authority : {-1, 3})
        EXPECT_THROW(serverRankMembershipTags(topology, 1, authority), std::invalid_argument);
}

TEST(ServerExecutionEvidence, OrdinaryCPUAndBothGPUsUseSelectedParticipants)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(2), DeviceId::rocm(3)})
    {
        RankExecutionPlan plan;
        plan.primary_device = GlobalDeviceAddress::fromLocalDeviceId(device);
        const auto selected = serverExecutionParticipants(plan, {}, 1);
        ASSERT_EQ(selected.size(), 1);
        EXPECT_EQ(selected.at(device), ServerParticipantRole::ModelGraph);
        const auto tags = serverExecutionTopologyTags(selected);
        EXPECT_EQ(tags.at("devices"), device.toString());
        EXPECT_EQ(tags.at("attention_devices"), device.toString());
    }
}

TEST(ServerExecutionEvidence, PreservesTPInsidePPAndRejectsPartialStageMembership)
{
    RankExecutionPlan plan;
    plan.local_pp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)};
    plan.local_pp_stage_tp_info = {
        {.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}},
        {.devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}}};
    const auto selected = serverExecutionParticipants(plan, {}, 1);
    EXPECT_EQ(selected.size(), 4);
    EXPECT_EQ(selected.at(DeviceId::cuda(1)), ServerParticipantRole::ModelGraph);
    EXPECT_EQ(selected.at(DeviceId::rocm(1)), ServerParticipantRole::ModelGraph);
    plan.local_pp_stage_tp_info.pop_back();
    EXPECT_THROW(serverExecutionParticipants(plan, {}, 1), std::invalid_argument);
    plan.local_pp_devices.clear();
    plan.local_pp_stage_tp_info.clear();
    plan.local_tp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(2)};
    EXPECT_EQ(serverExecutionParticipants(plan, {}, 1).size(), 2);
}

TEST(ServerExecutionEvidence, RemoteExpertsAreComputeButNotAttentionOrControlEndpoints)
{
    for (const auto accelerator : {DeviceId::cuda(0), DeviceId::rocm(1)})
    {
        OrchestrationConfig config;
        auto overlay = std::make_shared<MoERoutedExpertPlacementPlan>();
        overlay->enabled = true;
        overlay->topology = RoutedExpertPlacementTopology::TieredOverlay;
        overlay->continuation_domain = "model";
        overlay->shared_expert_domain = "model";
        RoutedExpertDomain model;
        model.name = "model";
        model.scope = ExecutionDomainScope::SINGLE;
        model.participants = {GlobalDeviceAddress::fromLocalDeviceId(accelerator)};
        model.world_ranks = {1};
        model.owner_rank = 1;
        RoutedExpertDomain experts;
        experts.name = "experts";
        experts.scope = ExecutionDomainScope::SINGLE;
        experts.participants = {GlobalDeviceAddress::cpu(0)};
        experts.world_ranks = {0};
        experts.owner_rank = 0;
        overlay->domains = {model, experts};
        overlay->routed_tiers = {{.name = "first", .domain = "model", .priority = 0},
                                 {.name = "last", .domain = "experts", .priority = 1, .fallback = true}};
        config.moe_routed_expert_plan = overlay;
        RankExecutionPlan plan;
        // Deliberately misleading control primary: the overlay authority, not
        // this ordinary-plan field or rank zero, owns expert participation.
        plan.primary_device = GlobalDeviceAddress::cuda(7);
        const auto remote = serverExecutionParticipants(plan, config, 3);
        ASSERT_EQ(remote.size(), 1);
        EXPECT_EQ(remote.at(DeviceId::cpu()), ServerParticipantRole::ExpertOnly);
        EXPECT_EQ(serverExecutionTopologyTags(remote).at("attention_devices"), "");
        plan.rank = 1;
        const auto root = serverExecutionParticipants(plan, config, 3);
        ASSERT_EQ(root.size(), 1);
        EXPECT_EQ(root.at(accelerator), ServerParticipantRole::ModelGraph);
        plan.rank = 2;
        EXPECT_TRUE(serverExecutionParticipants(plan, config, 3).empty());
    }
}

TEST(ServerExecutionEvidence, PolicyUsesAdmittedDefaultsWithoutCLIFlags)
{
    RankExecutionPlan plan;
    plan.runtime.prefix_cache.enabled = true;
    plan.runtime.mtp.enabled = true;
    plan.runtime.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
    plan.runtime.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    plan.runtime.mtp.depth_policy.max_depth = 15;
    plan.runtime.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
    const auto tags = serverExecutionPolicyTags(plan, {});
    EXPECT_EQ(tags.at("prefix_cache"), "true");
    EXPECT_EQ(tags.at("mtp"), "true");
    EXPECT_EQ(tags.at("mtp_max_depth"), "15");
    EXPECT_EQ(tags.at("mtp_verify_mode"), "speculative-sampling");
    EXPECT_EQ(tags.at("residency_maintenance"), "dynamic");
}
