/**
 * @file Test__ServerRankMembership.cpp
 * @brief Device-free proofs for the server's physical membership witness.
 *
 * These tests use immutable inventory values, never initialize MPI or inspect
 * hardware, and ensure observation cannot infer physical nodes from hostnames,
 * NUMA IDs, or an assumed rank-zero request authority. Attention obligations
 * follow actual stage/tensor ownership, including recurrent-only PP slices and
 * the separately owned MTP predictor, without removing compute participants.
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
        const auto tags = serverExecutionTopologyTags(selected, plan, {.n_layers = 4});
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

TEST(ServerExecutionEvidence, PipelineDomainsPreserveMembershipAndFreeLayerSplit)
{
    for (const bool reverse : {false, true})
    {
        RankExecutionPlan plan;
        plan.local_pp_devices = {GlobalDeviceAddress::cuda(1), GlobalDeviceAddress::rocm(3)};
        plan.local_pp_stage_tp_info = {
            {.devices = {GlobalDeviceAddress::cuda(1), GlobalDeviceAddress::cuda(0)}},
            {.devices = {GlobalDeviceAddress::rocm(3), GlobalDeviceAddress::rocm(2)}}};
        if (reverse)
        {
            std::reverse(plan.local_pp_devices.begin(), plan.local_pp_devices.end());
            std::reverse(plan.local_pp_stage_tp_info.begin(), plan.local_pp_stage_tp_info.end());
        }
        for (const int split : {1, 17, 63})
        {
            plan.local_pp_layer_boundaries = {0, split, 64};
            const auto records = serverPipelineDomainTags(plan);
            ASSERT_EQ(records.size(), 2);
            EXPECT_EQ(records[0].at("devices"), reverse ? "ROCm:3,ROCm:2" : "CUDA:1,CUDA:0");
            EXPECT_EQ(records[1].at("devices"), reverse ? "CUDA:1,CUDA:0" : "ROCm:3,ROCm:2");
            EXPECT_EQ(records[0].at("stage"), "0");
            EXPECT_EQ(records[1].at("stage"), "1");
            EXPECT_EQ(records[0].at("end_layer"), std::to_string(split));
            EXPECT_EQ(records[1].at("first_layer"), std::to_string(split));
            EXPECT_EQ(records[1].at("end_layer"), "64");
        }
        const auto reject = [&](auto mutate) {
            auto invalid = plan;
            mutate(invalid);
            EXPECT_THROW(serverPipelineDomainTags(invalid), std::invalid_argument);
        };
        reject([](auto &v) { v.local_pp_layer_boundaries.pop_back(); });
        reject([](auto &v) { v.local_pp_layer_boundaries = {0, 0, 64}; });
        reject([](auto &v) { v.local_pp_layer_boundaries = {0, 64, 63}; });
        reject([](auto &v) { v.local_pp_stage_tp_info.pop_back(); });
        reject([](auto &v) { v.local_pp_stage_tp_info[1].devices[0] = v.local_pp_stage_tp_info[0].devices[0]; });
        reject([](auto &v) { v.local_pp_stage_tp_info[0].devices.pop_back();
                            v.local_pp_devices[0] = GlobalDeviceAddress::cpu(); });
        reject([](auto &v) { v.local_pp_stage_tp_info[0].devices.push_back(GlobalDeviceAddress::cpu()); });
    }
    EXPECT_TRUE(serverPipelineDomainTags(RankExecutionPlan{}).empty());
}

TEST(ServerExecutionEvidence, RecurrentOnlyPipelineStagesStillOwnComputeAndCapture)
{
    const ModelMemoryProfile model{.n_layers = 8, .full_attention_interval = 4};
    for (const auto recurrent : {DeviceId::cuda(2), DeviceId::rocm(1), DeviceId::cpu()})
        for (const auto attention : {DeviceId::cuda(2), DeviceId::rocm(1), DeviceId::cpu()})
        {
            if (recurrent.type == attention.type) continue;
            for (const int width : {1, 2})
            {
                RankExecutionPlan plan;
                for (const auto device : {recurrent, attention})
                {
                    // Frozen addresses carry host/NUMA identity. Rank-local
                    // membership must not require a synthetic localhost alias.
                    auto primary = GlobalDeviceAddress::fromLocalDeviceId(device, "physical-node", 1);
                    plan.local_pp_devices.push_back(primary);
                    RankExecutionPlan::LocalPPStageTPInfo domain{.devices = {primary}};
                    if (width == 2 && device.is_gpu())
                    {
                        ++primary.device_ordinal;
                        domain.devices.push_back(primary);
                    }
                    plan.local_pp_stage_tp_info.push_back(std::move(domain));
                }
                plan.local_pp_layer_boundaries = {0, 1, 8};
                const auto selected = serverExecutionParticipants(plan, {}, 1);
                const auto tags = serverExecutionTopologyTags(selected, plan, model);
                EXPECT_NE(tags.at("devices").find(recurrent.toString()), std::string::npos);
                EXPECT_EQ(tags.at("attention_devices").find(recurrent.toString()), std::string::npos);
                std::string expected = attention.toString();
                if (width == 2 && attention.is_gpu())
                    expected += ',' + plan.local_pp_stage_tp_info[1].devices[1].toLocalDeviceId().toString();
                EXPECT_EQ(tags.at("attention_devices"), expected);
                // Changing the admitted split, not a backend exemption, makes
                // both domains responsible for their full-attention capture.
                plan.local_pp_layer_boundaries = {0, 4, 8};
                EXPECT_EQ(serverExecutionTopologyTags(selected, plan, model).at("attention_devices"),
                          tags.at("devices"));
            }
        }
}

TEST(ServerExecutionEvidence, ActualLayerTensorsAndTerminalPredictorOwnAttention)
{
    ModelMemoryProfile model{.n_layers = 9, .mtp_layer_count = 1, .full_attention_interval = 4};
    // Deliberately override the periodic geometry: layer 7 is recurrent and
    // the trailing predictor is attention even though (8 + 1) % 4 != 0.
    model.tensors = {{.name = "blk.7.attn_qkv.weight", .layer_index = 7},
                     {.name = "blk.8.attn_q.weight", .layer_index = 8}};
    for (const auto terminal : {DeviceId::cuda(1), DeviceId::rocm(2), DeviceId::cpu()})
    {
        const auto first = terminal.is_cuda() ? DeviceId::rocm(0) : DeviceId::cuda(0);
        RankExecutionPlan plan;
        plan.local_pp_devices = {GlobalDeviceAddress::fromLocalDeviceId(first),
                                 GlobalDeviceAddress::fromLocalDeviceId(terminal)};
        plan.local_pp_layer_boundaries = {0, 7, 8};
        const auto selected = serverExecutionParticipants(plan, {}, 1);
        EXPECT_EQ(serverExecutionTopologyTags(selected, plan, model).at("attention_devices"), first.toString());
        plan.runtime.mtp.enabled = true;
        const auto enabled = serverExecutionTopologyTags(selected, plan, model);
        EXPECT_EQ(enabled.at("attention_devices"), enabled.at("devices"));
        plan.has_lm_head = false;
        EXPECT_EQ(serverExecutionTopologyTags(selected, plan, model).at("attention_devices"), first.toString());

        // A global PP follower owns only its own interval, not another rank's
        // attention layers or predictor. The same rule applies to native TP.
        plan.local_pp_devices.clear();
        plan.local_pp_layer_boundaries.clear();
        plan.primary_device = GlobalDeviceAddress::fromLocalDeviceId(terminal);
        plan.first_layer = 7;
        plan.last_layer = 7;
        const auto follower = serverExecutionParticipants(plan, {}, 1);
        EXPECT_EQ(serverExecutionTopologyTags(follower, plan, model).at("attention_devices"), "");
        plan.has_lm_head = true;
        EXPECT_EQ(serverExecutionTopologyTags(follower, plan, model).at("attention_devices"), terminal.toString());
    }
}

TEST(ServerExecutionEvidence, AttentionProjectionRejectsIncompleteOwnership)
{
    const ModelMemoryProfile model{.n_layers = 8, .full_attention_interval = 4};
    RankExecutionPlan plan;
    plan.local_pp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)};
    plan.local_pp_layer_boundaries = {0, 1, 8};
    const auto selected = serverExecutionParticipants(plan, {}, 1);
    const auto reject = [&](auto mutate) {
        auto invalid = plan;
        mutate(invalid);
        EXPECT_THROW(serverExecutionTopologyTags(selected, invalid, model), std::invalid_argument);
    };
    reject([](auto &v) { v.local_pp_layer_boundaries.pop_back(); });
    reject([](auto &v) { v.local_pp_layer_boundaries = {0, 0, 8}; });
    reject([](auto &v) { v.local_pp_layer_boundaries = {0, 1, 9}; });
    reject([](auto &v) { v.local_pp_devices[1] = GlobalDeviceAddress::rocm(7); });
    reject([](auto &v) { v.local_pp_devices[1] = v.local_pp_devices[0]; });
    reject([](auto &v) { v.local_pp_stage_tp_info.resize(1); });
    EXPECT_THROW(serverExecutionTopologyTags(selected, plan, {}), std::invalid_argument);
    EXPECT_THROW(serverExecutionTopologyTags(selected, plan, (ModelMemoryProfile{.n_layers = 8, .mtp_layer_count = 8})),
                 std::invalid_argument);
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
        EXPECT_EQ(serverExecutionTopologyTags(remote, plan, {.n_layers = 4}).at("attention_devices"), "");
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

TEST(ServerExecutionEvidence, PhysicalParticipantsAreNotVisibilityOrRankCounts)
{
    RankInventory first{.rank = 0, .node_id = 0};
    first.gpus = {{.type = DeviceType::CUDA, .local_device_id = 2, .uuid = "cuda-physical"},
                  {.type = DeviceType::ROCm, .local_device_id = 3, .uuid = "rocm-physical"}};
    auto second = first;
    second.rank = 1;
    second.gpus[0].local_device_id = 0;
    EXPECT_EQ(serverExecutionParticipantTags(DeviceId::cuda(2), first, 0),
              serverExecutionParticipantTags(DeviceId::cuda(0), second, 1));
    EXPECT_EQ(serverExecutionParticipantTags(DeviceId::rocm(3), first, 0).at("physical_id"), "rocm-physical");
    EXPECT_NE(serverExecutionParticipantTags(DeviceId::cpu(), first, 0),
              serverExecutionParticipantTags(DeviceId::cpu(), second, 1));
    second.node_id = 1;
    EXPECT_NE(serverExecutionParticipantTags(DeviceId::cuda(2), first, 0),
              serverExecutionParticipantTags(DeviceId::cuda(0), second, 0));
    EXPECT_THROW(serverExecutionParticipantTags(DeviceId::cuda(7), first, 0), std::invalid_argument);
    first.gpus[0].uuid.clear();
    EXPECT_THROW(serverExecutionParticipantTags(DeviceId::cuda(2), first, 0), std::invalid_argument);
    first.gpus.push_back(first.gpus[1]);
    EXPECT_THROW(serverExecutionParticipantTags(DeviceId::rocm(3), first, 0), std::invalid_argument);
    first.cpu.numa_node = 1;
    EXPECT_THROW(serverExecutionParticipantTags(DeviceId::cpu(), first, 0), std::invalid_argument);
}

TEST(ServerExecutionEvidence, ReportsResolvedParallelismNotRequestedFlagSpelling)
{
    RankExecutionPlan plan;
    OrchestrationConfig config;
    EXPECT_EQ(serverExecutionPolicyTags(plan, config).at("execution_strategy"), "single");
    plan.local_tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    EXPECT_EQ(serverExecutionPolicyTags(plan, config).at("execution_strategy"), "tp");
    plan.local_pp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)};
    EXPECT_EQ(serverExecutionPolicyTags(plan, config).at("execution_strategy"), "pp");
    auto overlay = std::make_shared<MoERoutedExpertPlacementPlan>();
    overlay->enabled = true;
    overlay->topology = RoutedExpertPlacementTopology::TieredOverlay;
    overlay->routed_tiers = {{.name = "one", .domain = "first", .priority = 0}};
    config.moe_routed_expert_plan = overlay;
    EXPECT_EQ(serverExecutionPolicyTags(plan, config).at("execution_strategy"), "tp");
    overlay->routed_tiers.push_back({.name = "two", .domain = "second", .priority = 10});
    EXPECT_EQ(serverExecutionPolicyTags(plan, config).at("execution_strategy"), "expert-overlay");
}
