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
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <functional>

using namespace llaminar2;

namespace
{
    /** @return Requested main layers and one appended MTP layer, without tensor data. */
    PlanningModelMetadata metadata(bool moe, int main_layers = 4)
    {
        ModelMemoryProfile profile;
        profile.n_layers = main_layers + 1;
        profile.d_model = 512;
        profile.d_ff = 1536;
        profile.n_heads = 32;
        profile.n_kv_heads = 2;
        profile.head_dim = 16;
        profile.vocab_size = 320;
        profile.max_seq_len = 8192;
        profile.expert_count = moe ? 32 : 0;
        profile.mtp_layer_count = 1;
        // Use the canonical GGUF head inventory; these tests author main
        // topology geometry independently and never read tensor payloads.
        test::PlanningGGUFFixture file(moe, true);
        const auto fixture_metadata = readPlanningModelMetadata(file.path());
        for (auto tensor : fixture_metadata.memoryProfile().tensors)
            if (tensor.name.starts_with("blk.2."))
            {
                tensor.name.replace(0, 6, "blk." + std::to_string(main_layers) + ".");
                tensor.layer_index = main_layers;
                profile.tensors.push_back(std::move(tensor));
            }
        return PlanningModelMetadata(std::move(profile), main_layers);
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

    /** @return The real hybrid pipeline syntax with a deliberately uneven layer split. */
    OrchestrationConfig pipeline(bool rocm_first, int first_width = 2, int last_width = 2)
    {
        const auto domain = [](std::string name, bool rocm, int width) {
            const std::string device = rocm ? "rocm" : "cuda";
            std::string members;
            for (int ordinal = 0; ordinal < width; ++ordinal)
                members += (ordinal ? "," : "") + device + ":" + std::to_string(ordinal);
            return name + "=" + members + ";scope=" + (width > 1 ? "rank_local" : "single") +
                ";backend=" + (rocm ? "rccl" : "nccl") + ";owner=0";
        };
        return parse({"--define-domain", domain("first", rocm_first, first_width),
            "--define-domain", domain("last", !rocm_first, last_width),
            "--pp-stage", "0=first:0-0", "--pp-stage", "1=last:1-3"});
    }

    /** @return Both observed GPU groups, independent of authored stage order. */
    ClusterInventory hybridInventory(int width = 4)
    {
        std::vector<DeviceInfo> cards;
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            for (int ordinal = 0; ordinal < width; ++ordinal)
                cards.push_back(card(backend, ordinal, 7));
        return inventory(1, std::move(cards));
    }

    /** @brief Adversarial compiler boundary retaining production parsing and compilation. */
    class MutatedPipelineBuilder final : public IExecutionPlanBuilder
    {
    public:
        /** @param mutate Single corruption applied after canonical compilation. */
        explicit MutatedPipelineBuilder(std::function<void(RankExecutionPlan &)> mutate) : mutate_(std::move(mutate)) {}
        /** @return Production plans with the same one-boundary corruption per rank. */
        std::vector<RankExecutionPlan> buildAllPlans(const OrchestrationConfig &config,
            const ModelConfig &model, const ClusterInventory &cluster) override
        {
            auto plans = compiler_.buildAllPlans(config, model, cluster);
            for (auto &plan : plans) mutate_(plan);
            return plans;
        }
        /** @return Production rank plan after the requested invalid-state injection. */
        RankExecutionPlan buildPlanForRank(const OrchestrationConfig &config,
            const ModelConfig &model, const ClusterInventory &cluster, int rank) override
        {
            auto plan = compiler_.buildPlanForRank(config, model, cluster, rank);
            mutate_(plan);
            return plan;
        }
        /** @return Unmodified production configuration diagnostics. */
        std::vector<std::string> validateConfig(const OrchestrationConfig &config,
            const ModelConfig &model, const ClusterInventory &cluster) override
        { return compiler_.validateConfig(config, model, cluster); }
    private:
        ExecutionPlanBuilder compiler_;
        std::function<void(RankExecutionPlan &)> mutate_;
    };
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
                const auto expected_compute = backend == DeviceType::CPU
                    ? RoutedExpertComputePolicy::Apportioned
                    : RoutedExpertComputePolicy::GateUpOwnedDownColumns;
                EXPECT_EQ(overlay.domains.front().routed_compute_policy, expected_compute);
                EXPECT_EQ(resolved.config().routed_expert_compute_policy, expected_compute);
                EXPECT_EQ(resolved.rankPlan().runtime.routed_expert_compute_policy, expected_compute);
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
                for (const auto &domain : resolved.config().moe_routed_expert_plan->domains)
                    EXPECT_EQ(domain.routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
            }
            for (const auto &domain : original->domains)
            {
                EXPECT_EQ(domain.scope, ExecutionDomainScope::AUTO);
                EXPECT_EQ(domain.owner_rank, -1);
                EXPECT_TRUE(domain.world_ranks.empty());
            }
        }
}

/** @test The real compiler seals automatic projection ownership before admission on either rank. */
TEST(ResolvedRankOrchestration, AutomaticHomogeneousMoEComputeDefaultIsSealedAndIdempotent)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const int degree : {2, 3, 4, 8})
            for (const int owner : {0, 1})
                for (const bool explicit_whole : {false, true})
                {
                    const auto spelling = backend == DeviceType::CUDA ? "cuda" : "rocm";
                    std::vector<DeviceInfo> cards;
                    std::string declaration = "native=";
                    for (int index = 0; index < degree; ++index)
                    {
                        cards.push_back(card(backend, index, 7 - owner * 3));
                        if (index) declaration += ",";
                        declaration += std::string(spelling) + ":" + std::to_string(index);
                    }
                    declaration += ";priority=0";
                    std::vector<std::string> args{"--expert-tier", declaration};
                    if (explicit_whole)
                        args.insert(args.end(), {"--moe-routed-expert-compute", "apportioned"});
                    const auto config = parse(args);
                    const auto cluster = inventory(2, std::move(cards), owner);
                    const auto resolved = resolve(config, cluster, owner, true);
                    const auto expected = explicit_whole ? RoutedExpertComputePolicy::Apportioned
                        : RoutedExpertComputePolicy::GateUpOwnedDownColumns;
                    EXPECT_EQ(resolved.config().moe_routed_expert_plan->domains.front().routed_compute_policy, expected);
                    EXPECT_EQ(resolved.config().routed_expert_compute_policy, expected);
                    EXPECT_EQ(resolved.rankPlan().runtime.routed_expert_compute_policy, expected);
                    EXPECT_EQ(config.moe_routed_expert_plan->domains.front().routed_compute_policy,
                        explicit_whole ? RoutedExpertComputePolicy::Apportioned : RoutedExpertComputePolicy::Automatic);
                    const auto repeated = resolve(resolved.config(), cluster, owner, true);
                    EXPECT_EQ(repeated.rankPlan().toString(), resolved.rankPlan().toString());
                }
}

/** @test Tier boundaries and mixed endpoint types retain complete expert ownership. */
TEST(ResolvedRankOrchestration, AutomaticMixedAndMultiTierComputeRemainsWholeExpert)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto spelling = backend == DeviceType::CUDA ? "cuda" : "rocm";
        const auto cluster = inventory(1, {card(backend, 0, 7), card(backend, 1, 7),
            card(backend, 2, 7), card(backend, 3, 7)});
        const auto config = parse({"--expert-tier",
            std::string("first=") + spelling + ":0," + spelling + ":1;priority=-4",
            "--expert-tier", std::string("second=") + spelling + ":2," + spelling + ":3;priority=7"});
        const auto resolved = resolve(config, cluster, 0, true);
        for (const auto &domain : resolved.config().moe_routed_expert_plan->domains)
            EXPECT_EQ(domain.routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
        EXPECT_EQ(resolved.rankPlan().runtime.routed_expert_compute_policy,
            RoutedExpertComputePolicy::Apportioned);

        // A row-assignment contract that needs complete residents is resolved
        // deliberately, rather than entering projection mode and retrying it.
        const auto least_loaded = parse({"--expert-tier", std::string("owners=") +
            spelling + ":0," + spelling + ":1;scope=rank-local;priority=0;routed_prefill_assignment=least-loaded-resident"});
        const auto assigned = resolve(least_loaded, cluster, 0, true);
        EXPECT_EQ(assigned.config().moe_routed_expert_plan->domains.front().routed_compute_policy,
            RoutedExpertComputePolicy::Apportioned);
    }
    const auto cluster = inventory(1, {card(DeviceType::CUDA, 0, 7), card(DeviceType::ROCm, 0, 7)});
    const auto mixed = parse({"--expert-tier", "mixed=cuda:0,rocm:0;priority=0"});
    const auto resolved = resolve(mixed, cluster, 0, true);
    EXPECT_EQ(resolved.config().moe_routed_expert_plan->domains.front().routed_compute_policy,
        RoutedExpertComputePolicy::Apportioned);
    EXPECT_EQ(resolved.rankPlan().runtime.routed_expert_compute_policy,
        RoutedExpertComputePolicy::Apportioned);
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

/** @test Child expert ownership must not erase the authored hybrid pipeline. */
TEST(ResolvedRankOrchestration, MoEPipelinePreservesParentAndScopesEachAuthority)
{
    const auto cluster = hybridInventory();
    for (const bool rocm_first : {false, true})
        for (const int first_width : {1, 2, 4})
            for (const int last_width : {1, 2, 4})
                for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Dynamic})
                {
                    SCOPED_TRACE(::testing::Message() << rocm_first << '/' << first_width << '/' << last_width << '/' << int(mode));
                    auto config = pipeline(rocm_first, first_width, last_width);
                    config.moe_rebalance.mode = mode;
                    config.routed_expert_owner_order = RoutedExpertOwnerOrder::Random;
                    config.max_seq_len = 8192;
                    config.prefix_cache.enabled = true;
                    const auto resolved = resolve(config, cluster, 0, true);
                    EXPECT_EQ(resolved.overlayOrigin(), MoEExpertOverlayAuthorityPlanDisposition::SynthesizedPipeline);
                    EXPECT_FALSE(resolved.overlayExecution());
                    EXPECT_FALSE(resolved.config().moe_routed_expert_plan);
                    EXPECT_TRUE(resolved.rankPlan().usesLocalPP());
                    EXPECT_FALSE(resolved.rankPlan().usesLocalTP());
                    EXPECT_EQ(resolved.config().pp_stage_definitions.size(), 2u);
                    ASSERT_EQ(resolved.pipelineStages().size(), 2u);
                    for (size_t index = 0; index < 2; ++index)
                    {
                        const auto &stage = resolved.pipelineStages()[index];
                        const auto &rank = stage.rankPlan();
                        const auto &child = stage.config();
                        const auto &overlay = *child.moe_routed_expert_plan;
                        const auto width = index == 0 ? first_width : last_width;
                        const auto backend = (index == 0 ? rocm_first : !rocm_first) ? DeviceType::ROCm : DeviceType::CUDA;
                        EXPECT_EQ(stage.scope().first_layer, index == 0 ? 0 : 1);
                        EXPECT_EQ(stage.scope().last_layer, index == 0 ? 1 : 4);
                        EXPECT_EQ(stage.scope().has_embedding, index == 0);
                        EXPECT_EQ(stage.scope().has_lm_head, index == 1);
                        EXPECT_EQ(rank.first_layer, stage.scope().first_layer);
                        EXPECT_EQ(rank.last_layer, stage.scope().last_layer - 1);
                        EXPECT_EQ(overlay.first_model_layer, stage.scope().first_layer);
                        EXPECT_EQ(overlay.domains.size(), 1u);
                        EXPECT_EQ(overlay.domains.front().participants.size(), size_t(width));
                        EXPECT_EQ(rank.primary_device.device_type, backend);
                        EXPECT_EQ(overlay.domains.front().backend,
                            backend == DeviceType::ROCm ? CollectiveBackendType::RCCL : CollectiveBackendType::NCCL);
                        EXPECT_EQ(overlay.owner_order, RoutedExpertOwnerOrder::Random);
                        EXPECT_EQ(child.moe_rebalance.mode, mode);
                        EXPECT_TRUE(overlay.placements.empty()) << "Topology cannot freeze unadmitted placements";
                        EXPECT_TRUE(stage.overlayExecution().ownsContinuationGraph());
                        EXPECT_FALSE(rank.usesLocalPP());
                        EXPECT_FALSE(rank.usesGlobalTP());
                        EXPECT_FALSE(rank.prev_rank);
                        EXPECT_FALSE(rank.next_rank);
                        EXPECT_TRUE(child.pp_stage_definitions.empty());
                        EXPECT_TRUE(child.tp_devices.empty());
                        EXPECT_FALSE(child.topology_tree);
                        EXPECT_TRUE(child.topology_string.empty());
                        EXPECT_EQ(child.domain_definitions.size(), 1u);
                        EXPECT_EQ(rank.runtime.max_seq_len, 8192);
                        EXPECT_TRUE(rank.runtime.prefix_cache.enabled);
                        EXPECT_NE(rank.runtime.routed_expert_compute_policy, RoutedExpertComputePolicy::Automatic);
                        EXPECT_EQ(rank.runtime.routed_expert_compute_policy, child.routed_expert_compute_policy);
                    }
                    EXPECT_FALSE(config.moe_routed_expert_plan);
                    EXPECT_EQ(config.domain_definitions.front().name, "first");
                    EXPECT_EQ(config.pp_stage_definitions.back().first_layer, 1);
                }
}

/** @test Normalizing twice preserves distinct stage compute policies and global scope. */
TEST(ResolvedRankOrchestration, MoEPipelineNormalizationIsIdempotent)
{
    for (const auto compute : {RoutedExpertComputePolicy::Automatic, RoutedExpertComputePolicy::Apportioned,
                              RoutedExpertComputePolicy::GateUpOwnedDownColumns})
    {
        auto config = pipeline(true);
        config.routed_expert_compute_policy = compute;
        const auto first = resolve(config, hybridInventory(), 0, true);
        const auto second = resolve(first.config(), hybridInventory(), 0, true);
        ASSERT_EQ(first.pipelineStages().size(), 2u);
        ASSERT_EQ(second.pipelineStages().size(), 2u);
        EXPECT_EQ(first.rankPlan().toString(), second.rankPlan().toString());
        for (size_t index = 0; index < 2; ++index)
        {
            const auto &a = first.pipelineStages()[index];
            const auto &b = second.pipelineStages()[index];
            EXPECT_EQ(a.rankPlan().toString(), b.rankPlan().toString());
            EXPECT_EQ(a.overlayExecution().diagnostics(), b.overlayExecution().diagnostics());
            EXPECT_EQ(a.config().routed_expert_compute_policy, b.config().routed_expert_compute_policy);
            EXPECT_EQ(a.config().moe_routed_expert_plan->first_model_layer, b.config().moe_routed_expert_plan->first_model_layer);
        }
    }
}

/** @test Automatic, disabled, fixed and dynamic MTP preserve the one parent policy. */
TEST(ResolvedRankOrchestration, MoEPipelineRetainsMTPAndTerminalOwnership)
{
    for (const bool rocm_first : {false, true})
        for (const int width : {2, 4})
        for (const auto activation : {MTPActivationPolicy::Automatic, MTPActivationPolicy::Disabled, MTPActivationPolicy::Enabled})
            for (const auto depth : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Dynamic})
            {
                SCOPED_TRACE(::testing::Message() << rocm_first << '/' << width << '/' << int(activation) << '/' << int(depth));
                auto config = pipeline(rocm_first, width, width);
                config.mtp_activation_policy = activation;
                config.mtp.depth_policy.mode = depth;
                const auto result = resolve(config, hybridInventory(), 0, true);
                ASSERT_EQ(result.pipelineStages().size(), 2u);
                for (const auto &stage : result.pipelineStages())
                {
                    EXPECT_EQ(stage.config().mtp.enabled, activation != MTPActivationPolicy::Disabled);
                    EXPECT_EQ(stage.rankPlan().runtime.mtp.enabled, result.rankPlan().runtime.mtp.enabled);
                    EXPECT_EQ(stage.config().mtp.depth_policy.mode, depth);
                    EXPECT_EQ(stage.config().mtp.depth_defaults_profile, result.config().mtp.depth_defaults_profile);
                    EXPECT_EQ(stage.config().mtp.terminal_head_policy, result.config().mtp.terminal_head_policy);
                    EXPECT_LT(stage.rankPlan().last_layer, metadata(true).mainLayerCount());
                }
                EXPECT_FALSE(result.pipelineStages().front().scope().has_lm_head);
                EXPECT_TRUE(result.pipelineStages().back().scope().has_lm_head);
            }
}

/** @test Corrupted compiler output cannot claim another stage's layers or devices. */
TEST(ResolvedRankOrchestration, MoEPipelineRejectsIncoherentCompilerGeometry)
{
    const std::vector<std::function<void(RankExecutionPlan &)>> mutations{
        [](auto &p) { p.local_pp_layer_boundaries.clear(); },
        [](auto &p) { p.local_pp_layer_boundaries[1] = 0; },
        [](auto &p) { p.local_pp_layer_boundaries.back() = 5; },
        [](auto &p) { p.local_pp_layer_boundaries.front() = 1; },
        [](auto &p) { p.local_pp_stage_tp_info.pop_back(); },
        [](auto &p) { p.local_pp_stage_tp_info[0].devices[0] = p.local_pp_devices[1]; },
        [](auto &p) { p.local_pp_stage_tp_info[0].tp_weights = {1.0f}; },
        [](auto &p) { p.global_tp_domain_id = 4; },
        [](auto &p) { p.next_rank = 1; }};
    for (size_t index = 0; index < mutations.size(); ++index)
    {
        SCOPED_TRACE(index);
        MutatedPipelineBuilder builder(mutations[index]);
        EXPECT_THROW((void)ResolvedRankOrchestration::resolve(pipeline(true), metadata(true),
            hybridInventory(), builder, 0), std::invalid_argument);
    }
}

/** @test Dense pipelines continue through their existing ordinary memory authority. */
TEST(ResolvedRankOrchestration, MoEPipelineProjectionDoesNotInventDenseExpertStages)
{
    const auto result = resolve(pipeline(true), hybridInventory(), 0, false);
    EXPECT_TRUE(result.rankPlan().usesLocalPP());
    EXPECT_TRUE(result.pipelineStages().empty());
    EXPECT_FALSE(result.overlayExecution());
    EXPECT_FALSE(result.config().moe_routed_expert_plan);
}

/** @brief Discoverable four-/eight-device topology coverage without native contexts. */
class MoEPipelineMultiDeviceTopology : public ::testing::TestWithParam<int> {};

/** @test Every child retains its global interval, real device and one terminal MTP owner. */
TEST_P(MoEPipelineMultiDeviceTopology, MiddleStagesAndMTPPolicySurviveCompilation)
{
    const int count = GetParam();
    for (const bool reverse : {false, true})
        for (const int width : {1, 2})
            for (const auto activation : {MTPActivationPolicy::Disabled, MTPActivationPolicy::Enabled})
                for (const auto depth : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Dynamic})
                {
                    SCOPED_TRACE(::testing::Message() << count << '/' << reverse << '/' << width << '/'
                        << int(activation) << '/' << int(depth));
                    const int stages = count / width;
                    std::vector<std::string> args;
                    int next_cuda = 0, next_rocm = 0;
                    for (int stage = 0; stage < stages; ++stage)
                    {
                        const bool rocm = bool(stage % 2) == reverse;
                        const std::string backend = rocm ? "rocm" : "cuda";
                        int &ordinal = rocm ? next_rocm : next_cuda;
                        const auto name = "stage" + std::to_string(stage);
                        std::string declaration = name + "=";
                        for (int member = 0; member < width; ++member)
                            declaration += (member ? "," : "") + backend + ":" + std::to_string(ordinal++);
                        declaration += std::string(";scope=") + (width == 1 ? "single" : "rank_local") +
                            ";backend=" + (rocm ? "rccl" : "nccl") + ";owner=0";
                        args.insert(args.end(), {"--define-domain", declaration, "--pp-stage",
                            std::to_string(stage) + "=" + name + ":" + std::to_string(stage ? 2 * stage - 1 : 0) +
                            "-" + std::to_string(2 * stage)});
                    }
                    auto config = parse(std::move(args));
                    config.mtp_activation_policy = activation;
                    config.mtp.depth_policy.mode = depth;
                    ExecutionPlanBuilder builder;
                    const auto result = ResolvedRankOrchestration::resolve(config, metadata(true, 2 * stages - 1),
                        hybridInventory(count / 2), builder, 0);
                    ASSERT_EQ(result.pipelineStages().size(), size_t(stages));
                    size_t members = 0;
                    for (int stage = 0; stage < stages; ++stage)
                    {
                        const auto &child = result.pipelineStages()[stage];
                        EXPECT_EQ(child.scope().first_layer, stage ? 2 * stage - 1 : 0);
                        EXPECT_EQ(child.scope().last_layer, 2 * stage + 1);
                        EXPECT_EQ(child.scope().has_embedding, stage == 0);
                        EXPECT_EQ(child.scope().has_lm_head, stage + 1 == stages);
                        EXPECT_FALSE(child.rankPlan().prev_rank);
                        EXPECT_FALSE(child.rankPlan().next_rank);
                        EXPECT_EQ(child.config().mtp.enabled, activation == MTPActivationPolicy::Enabled);
                        EXPECT_EQ(child.config().mtp.depth_policy.mode, depth);
                        const auto &domain = child.config().moe_routed_expert_plan->domains.front();
                        ASSERT_EQ(domain.participants.size(), size_t(width));
                        members += domain.participants.size();
                        for (int member = 0; member < width; ++member)
                            EXPECT_EQ(domain.participants[member].device_ordinal, (stage / 2) * width + member);
                    }
                    EXPECT_EQ(members, size_t(count));
                    const auto repeated = ResolvedRankOrchestration::resolve(result.config(), metadata(true, 2 * stages - 1),
                        hybridInventory(count / 2), builder, 0);
                    EXPECT_EQ(repeated.rankPlan().toString(), result.rankPlan().toString());
                }
}

INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEPipelineMultiDeviceTopology, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });
