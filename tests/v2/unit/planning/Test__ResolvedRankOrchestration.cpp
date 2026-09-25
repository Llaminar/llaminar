/**
 * @file Test__ResolvedRankOrchestration.cpp
 * @brief Device-free proofs of the shared model-aware topology compiler.
 *
 * Public configuration parsing and the real rank compiler run against observed
 * inventory fixtures, without loading weights or entering MPI. The tests lock
 * down continuation/follower ownership, exact model geometry, policy retention,
 * input immutability and idempotent normalization before physical admission.
 */
#include "planning/ResolvedRankOrchestration.h"
#include "planning/PlanningModelMetadata.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
    /** @return Four main layers and one appended MTP layer, without tensor data. */
    PlanningModelMetadata metadata(bool moe)
    {
        ModelMemoryProfile profile;
        profile.n_layers = 5;
        profile.d_model = 512;
        profile.d_ff = 1536;
        profile.n_heads = 32;
        profile.n_kv_heads = 2;
        profile.head_dim = 16;
        profile.vocab_size = 320;
        profile.max_seq_len = 8192;
        profile.expert_count = moe ? 32 : 0;
        return PlanningModelMetadata(std::move(profile), 4);
    }

    /** @return Physical observation, never a discovered backend or allocation. */
    DeviceInfo card(DeviceType backend, int ordinal, int numa)
    {
        DeviceInfo result;
        result.type = backend;
        result.local_device_id = ordinal;
        result.numa_node = numa;
        result.uuid = std::string(deviceTypeToString(backend)) + std::to_string(ordinal);
        result.name = backend == DeviceType::CUDA ? "NVIDIA GeForce RTX 3090" : "AMD Instinct MI60 / MI50";
        result.compute_units = backend == DeviceType::CUDA ? 82 : 60;
        return result;
    }

    /** @return Sparse, reverse-NUMA rank identity, deliberately unlike rank IDs. */
    ClusterInventory inventory(int ranks, std::vector<DeviceInfo> cards = {}, int gpu_rank = 0)
    {
        ClusterInventory cluster;
        cluster.world_size = ranks;
        for (int rank = 0; rank < ranks; ++rank)
        {
            RankInventory entry;
            entry.rank = entry.local_rank = rank;
            entry.node_id = 0;
            entry.hostname = "node";
            entry.numa_nodes = ranks;
            entry.cpu.numa_node = 7 - rank * 3;
            entry.cpu_cores = 8;
            entry.cpu_worker_threads = 8;
            if (rank == gpu_rank) entry.gpus = cards;
            cluster.ranks.push_back(std::move(entry));
        }
        cluster.buildNodeAggregations();
        return cluster;
    }

    /** @return Parsed real user input with stable argv backing storage. */
    OrchestrationConfig parse(std::vector<std::string> arguments)
    {
        arguments.insert(arguments.begin(), "llaminar2");
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        return OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
    }

    /** @return Model-aware production resolution with no test compiler substitute. */
    ResolvedRankOrchestration resolve(const OrchestrationConfig &config,
        const ClusterInventory &cluster, int rank, bool moe = false)
    {
        ExecutionPlanBuilder builder;
        return ResolvedRankOrchestration::resolve(config, metadata(moe), cluster, builder, rank);
    }
}

TEST(ResolvedRankOrchestration, SingleDevicePreservesRealMainIntervalAndServingPolicy)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(backend));
        const auto cluster = inventory(1, backend == DeviceType::CPU ? std::vector<DeviceInfo>{}
            : std::vector<DeviceInfo>{card(backend, 0, 7)});
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::fromLocalDeviceId(
            DeviceId(backend, backend == DeviceType::CPU ? 7 : 0), "node", 7);
        config.max_seq_len = 8192;
        config.kv_cache_precision = "fp32";
        config.mtp.enabled = true;
        config.mtp.graph_capacity_draft_tokens = 15;
        config.prefix_cache.enabled = true;
        const auto resolved = resolve(config, cluster, 0);
        const auto &plan = resolved.rankPlan();
        EXPECT_EQ(plan.first_layer, 0);
        EXPECT_EQ(plan.last_layer, 3) << "The appended MTP layer is not a main forward layer";
        EXPECT_EQ(plan.primary_device.device_type, backend);
        EXPECT_EQ(plan.numa_node, 7);
        EXPECT_EQ(plan.runtime.max_seq_len, 8192);
        EXPECT_EQ(plan.runtime.kv_cache_precision, KVCachePrecision::FP32);
        EXPECT_TRUE(plan.runtime.prefix_cache.enabled);
        EXPECT_TRUE(plan.runtime.mtp.enabled);
        EXPECT_EQ(plan.runtime.mtp.graph_capacity_draft_tokens, 15);
        EXPECT_EQ(resolved.config().mtp.depth_defaults_profile, plan.runtime.mtp.depth_defaults_profile);
        EXPECT_EQ(resolved.config().mtp.terminal_head_policy, plan.runtime.mtp.terminal_head_policy);
        EXPECT_EQ(plan.runtime.mtp.terminal_head_policy, backend == DeviceType::CPU
            ? MTPTerminalHeadPolicy::VocabularySharded : MTPTerminalHeadPolicy::MirroredFullVocabulary);
        EXPECT_FALSE(resolved.overlayExecution());
        EXPECT_EQ(resolved.overlayOrigin(), MoEExpertOverlayAuthorityPlanDisposition::NotApplicable);
    }
}

TEST(ResolvedRankOrchestration, ImplicitMoELocalTPSealsOneAuthorityWithoutMutatingTheRequest)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Dynamic})
            for (const auto order : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
            {
                SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend)
                    << " mode=" << static_cast<int>(mode) << " order=" << static_cast<int>(order));
                auto cluster = inventory(1);
                OrchestrationConfig config;
                config.moe_rebalance.mode = mode;
                config.routed_expert_owner_order = order;
                for (int ordinal = 0; ordinal < 2; ++ordinal)
                {
                    // Multiple CPU endpoints must be observed, not inferred
                    // from an integer rank. A whole-node CPU record owns both.
                    const int index = backend == DeviceType::CPU ? 7 - ordinal * 3 : ordinal;
                    config.tp_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                        DeviceId(backend, index), "node", backend == DeviceType::CPU ? index : 7));
                    if (backend != DeviceType::CPU) cluster.ranks[0].gpus.push_back(card(backend, ordinal, 7));
                }
                if (backend == DeviceType::CPU)
                {
                    cluster.ranks[0].cpu.numa_node = -1;
                    cluster.ranks[0].numa_nodes = 2;
                    for (int node : {7, 4})
                    {
                        CPUSocketInfo socket;
                        socket.numa_node = node;
                        cluster.ranks[0].cpu_socket_info.push_back(std::move(socket));
                    }
                }
                cluster.buildNodeAggregations();
                const auto resolved = resolve(config, cluster, 0, true);
                EXPECT_FALSE(config.moe_routed_expert_plan);
                EXPECT_TRUE(config.domain_definitions.empty());
                ASSERT_TRUE(resolved.overlayExecution());
                ASSERT_TRUE(resolved.config().moe_routed_expert_plan);
                const auto &overlay = *resolved.config().moe_routed_expert_plan;
                EXPECT_EQ(overlay.owner_order, order);
                EXPECT_EQ(overlay.residency_policy, mode == MoERebalanceRuntimeMode::Off
                    ? RoutedExpertResidencyPolicy::StaticById : RoutedExpertResidencyPolicy::RoutedTierRebalanced);
                EXPECT_EQ(overlay.domains.size(), 1u);
                EXPECT_EQ(overlay.routed_tiers.size(), 1u);
                EXPECT_EQ(resolved.rankPlan().local_tp_devices.size(), 2u);
                EXPECT_TRUE(resolved.overlayExecution()->ownsContinuationGraph());
                EXPECT_EQ(resolved.config().mtp.terminal_head_policy, backend == DeviceType::CPU
                    ? MTPTerminalHeadPolicy::VocabularySharded : MTPTerminalHeadPolicy::MirroredFullVocabulary);
                EXPECT_EQ(resolved.config().mtp.terminal_head_policy,
                    resolved.rankPlan().runtime.mtp.terminal_head_policy);
                EXPECT_EQ(config.mtp.terminal_head_policy, MTPTerminalHeadPolicy::Automatic);
                EXPECT_EQ(resolved.overlayOrigin(), MoEExpertOverlayAuthorityPlanDisposition::SynthesizedLocalTP);
                const auto repeated = resolve(resolved.config(), cluster, 0, true);
                EXPECT_EQ(repeated.overlayOrigin(), MoEExpertOverlayAuthorityPlanDisposition::ExplicitPlan);
                EXPECT_EQ(repeated.rankPlan().toString(), resolved.rankPlan().toString());
                EXPECT_EQ(repeated.config().domain_definitions.size(), 1u);
            }
}

TEST(ResolvedRankOrchestration, CompactOverlayUsesObservedOwnerAndKeepsFollowerOutOfDenseTP)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const int gpu_rank : {0, 1})
        {
            const auto spelling = backend == DeviceType::CUDA ? "cuda" : "rocm";
            SCOPED_TRACE(::testing::Message() << spelling << " owner=" << gpu_rank);
            const auto cluster = inventory(2, {card(backend, 0, 7 - gpu_rank * 3),
                card(backend, 1, 7 - gpu_rank * 3)}, gpu_rank);
            const auto config = parse({"--expert-tier", "storage=cpu:7,cpu:4;priority=12",
                "--expert-tier", std::string("compute=") + spelling + ":0," + spelling + ":1;priority=-2"});
            const auto original = config.moe_routed_expert_plan;
            ASSERT_TRUE(original);
            for (int rank = 0; rank < 2; ++rank)
            {
                const auto resolved = resolve(config, cluster, rank, true);
                ASSERT_TRUE(resolved.overlayExecution());
                const auto &execution = *resolved.overlayExecution();
                EXPECT_EQ(execution.continuation_root_rank, gpu_rank);
                EXPECT_EQ(execution.ownsContinuationGraph(), rank == gpu_rank);
                EXPECT_EQ(resolved.rankPlan().numa_node, 7 - rank * 3);
                EXPECT_EQ(resolved.rankPlan().local_tp_devices.size(), rank == gpu_rank ? 2u : 0u);
                EXPECT_FALSE(resolved.rankPlan().usesLocalPP());
                EXPECT_EQ(resolved.config().mtp.depth_defaults_profile, backend == DeviceType::CUDA
                    ? MTPDepthDefaultsProfile::CUDARTX3090 : MTPDepthDefaultsProfile::ROCmMI50);
                EXPECT_NE(resolved.config().moe_routed_expert_plan.get(), original.get());
                EXPECT_EQ(resolved.config().mtp.terminal_head_policy, MTPTerminalHeadPolicy::MirroredFullVocabulary);
                EXPECT_EQ(resolved.rankPlan().runtime.mtp.terminal_head_policy, MTPTerminalHeadPolicy::MirroredFullVocabulary);
            }
            for (const auto &domain : original->domains)
            {
                EXPECT_EQ(domain.scope, ExecutionDomainScope::AUTO);
                EXPECT_EQ(domain.owner_rank, -1);
                EXPECT_TRUE(domain.world_ranks.empty());
            }
        }
}

TEST(ResolvedRankOrchestration, NodeTPContinuationPreservesShardAndSparseCPUIdentity)
{
    const auto cluster = inventory(2);
    const auto config = parse({"--expert-tier", "compute=cpu:7,cpu:4;priority=5"});
    for (int rank = 0; rank < 2; ++rank)
    {
        const auto resolved = resolve(config, cluster, rank, true);
        ASSERT_TRUE(resolved.overlayExecution());
        EXPECT_TRUE(resolved.overlayExecution()->ownsContinuationGraph());
        EXPECT_TRUE(resolved.rankPlan().usesGlobalTP());
        EXPECT_EQ(resolved.rankPlan().global_tp_domain_size, 2);
        EXPECT_EQ(resolved.rankPlan().weight_shard.total_shards, 2);
        EXPECT_EQ(resolved.rankPlan().weight_shard.shard_index, rank);
        EXPECT_EQ(resolved.rankPlan().numa_node, 7 - rank * 3);
        EXPECT_EQ(resolved.config().mtp.terminal_head_policy, MTPTerminalHeadPolicy::VocabularySharded);
        EXPECT_EQ(resolved.rankPlan().runtime.mtp.terminal_head_policy, MTPTerminalHeadPolicy::VocabularySharded);
    }
}

TEST(ResolvedRankOrchestration, RejectsInvalidInventoryAndMissingExplicitHardware)
{
    auto cluster = inventory(1);
    EXPECT_THROW(resolve({}, cluster, -1), std::invalid_argument);
    EXPECT_THROW(resolve({}, cluster, 1), std::invalid_argument);
    cluster.ranks[0].rank = 2;
    EXPECT_THROW(resolve({}, cluster, 0), std::invalid_argument);
    cluster.ranks[0].rank = 0;
    cluster.world_size = 2;
    EXPECT_THROW(resolve({}, cluster, 0), std::invalid_argument);
    cluster = inventory(2);
    cluster.ranks[1].rank = 0;
    EXPECT_THROW(resolve({}, cluster, 0), std::invalid_argument);
    cluster = inventory(1);
    const auto absent = parse({"--expert-tier", "compute=cuda:0;priority=0"});
    EXPECT_THROW(resolve(absent, cluster, 0, true), std::invalid_argument);
}

TEST(ResolvedRankOrchestration, UnimplementedTreeCannotSilentlySelectAnotherRuntime)
{
    OrchestrationConfig config;
    config.topology_tree.emplace();
    config.topology_tree->root.device = GlobalDeviceAddress::cpu();
    const auto factory = createOrchestrationRunnerFactory();
    // This unit has no MPI or model: rejection precedes both admission steps.
    EXPECT_EQ(factory->createFromOrchestrationConfig(std::move(config)), nullptr);
}
