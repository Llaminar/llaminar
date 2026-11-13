/**
 * @file Test__ResolvedRankOrchestration.cpp
 * @brief Device-free proofs of the shared model-aware topology compiler.
 *
 * Public configuration parsing and the real rank compiler run against observed
 * inventory fixtures, without loading weights or entering MPI. The tests lock
 * down continuation/follower ownership, exact model geometry, policy retention,
 * input immutability and idempotent normalization before physical admission.
 * Runner installation tests exercise the real resolved-config publication
 * boundary and its exact pre-freeze reuse/migration authority identity.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "planning/ResolvedRankOrchestration.h"
#include "execution/global/GlobalStageTPValidation.h"
#include "../../mocks/MockModelContext.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/runner/PreparedWeightAuthorityIdentity.h"
#include "planning/PlanningModelMetadata.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <functional>
#include <array>

using namespace llaminar2;

namespace llaminar2
{
    /** @brief Narrow observation of the production resolved-admission install boundary. */
    class OrchestrationRunnerResolvedAdmissionTestAccess
    {
    public:
        /** @brief Publish the same compiler result consumed by buildExecutionPlan. */
        static void install(OrchestrationRunner &runner, const ResolvedRankOrchestration &resolved)
        {
            runner.installResolvedExecutionPlan(resolved);
        }

        /** @return Exact request identity used by migration-profile/reuse contracts. */
        static const std::string &identity(const OrchestrationRunner &runner)
        {
            return runner.requested_routed_weight_authority_identity_;
        }
    };
}

namespace
{
    /** @return Requested main layers and one appended MTP layer, without tensor data. */
    PlanningModelMetadata metadata(bool moe, int main_layers = 4, int uniform_head_count = 0)
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
        if (uniform_head_count > 0)
        {
            profile.n_heads = profile.n_kv_heads = uniform_head_count;
            profile.d_model = uniform_head_count * profile.head_dim;
            profile.d_ff = 3 * profile.d_model;
        }
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

    /**
     * @brief Declare this metadata fixture's complete observed no-peer topology.
     * @param original Authored rank inventory; no native runtime is queried.
     * @details Sparse ordinals retain discovery order. Every directed edge is
     * explicitly unavailable; missing observations must remain production errors.
     */
    void declareNoPeerTopology(ClusterInventory &original)
    {
        for (auto &rank : original.ranks)
            for (const auto family : {DeviceType::CUDA, DeviceType::ROCm}) {
                const auto count = std::count_if(rank.gpus.begin(), rank.gpus.end(),
                    [family](const auto &gpu) { return gpu.type == family; });
                auto &matrix = family == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
                auto &degree = family == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
                degree = static_cast<int>(count);
                matrix.assign(static_cast<std::size_t>(count * count), false);
            }
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
        declareNoPeerTopology(cluster);
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
                declareNoPeerTopology(cluster);
                cluster.buildNodeAggregations();
                if (backend != DeviceType::CPU) {
                    auto missing = cluster;
                    auto &matrix = backend == DeviceType::CUDA ? missing.ranks[0].p2p_cuda : missing.ranks[0].p2p_rocm;
                    matrix.clear();
                    EXPECT_THROW((void)resolve(config, missing, 0, true), std::invalid_argument);
                }
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
            spelling + ":0," + spelling + ":1;scope=rank-local;priority=0;expert_prefill_assignment=least-loaded-resident"});
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

/** @test Authored output demand reaches immutable child plans before any admission. */
TEST(ResolvedRankOrchestration, OrdinaryProjectionDemandPrecedesPipelineProjection)
{
    const auto demand = OrdinaryTerminalProjectionCapacity::projectedRows(16);
    ExecutionPlanBuilder builder;
    const auto result = ResolvedRankOrchestration::resolve(
        pipeline(true), metadata(true), hybridInventory(), builder, 0, demand);
    EXPECT_EQ(result.rankPlan().runtime.ordinary_terminal_projection, demand);
    ASSERT_EQ(result.pipelineStages().size(), 2u);
    for (const auto &stage : result.pipelineStages())
        EXPECT_EQ(stage.rankPlan().runtime.ordinary_terminal_projection, demand);
}

/** @test Implicit MoE synthesis cannot erase authored demand during canonical recompilation. */
TEST(ResolvedRankOrchestration, OrdinaryProjectionSurvivesImplicitMoERecompile)
{
    OrchestrationConfig config;
    config.tp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
    ExecutionPlanBuilder builder;
    const auto demand = OrdinaryTerminalProjectionCapacity::projectedRows(16);
    const auto result = ResolvedRankOrchestration::resolve(config, metadata(true),
        inventory(1, {card(DeviceType::ROCm, 0, 7), card(DeviceType::ROCm, 1, 7)}), builder, 0, demand);
    EXPECT_EQ(result.overlayOrigin(), MoEExpertOverlayAuthorityPlanDisposition::SynthesizedLocalTP);
    EXPECT_EQ(result.rankPlan().runtime.ordinary_terminal_projection, demand);
}


/**
 * @brief Implicit CPU TP admission installs the synthesized request authority.
 *
 * Constructors see no overlay plan. The actual model-aware compiler introduces
 * it, and the runner's real installation method must publish its identity with
 * config/plan before migration calibration or retained-weight checks use it.
 */
TEST(ResolvedRankOrchestration, RunnerInstallsImplicitCPURoutedAuthorityWithResolvedPlan)
{
    using Access = OrchestrationRunnerResolvedAdmissionTestAccess;
    auto cluster = inventory(1);
    cluster.ranks[0].cpu.numa_node = -1;
    cluster.ranks[0].numa_nodes = 2;
    for (int node : {7, 4})
    {
        CPUSocketInfo socket;
        socket.numa_node = node;
        cluster.ranks[0].cpu_socket_info.push_back(std::move(socket));
    }
    declareNoPeerTopology(cluster);
    cluster.buildNodeAggregations();
    for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Dynamic})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        OrchestrationConfig request;
        request.moe_rebalance.mode = mode;
        request.routed_expert_owner_order = RoutedExpertOwnerOrder::Random;
        request.tp_devices = {GlobalDeviceAddress::cpu(7), GlobalDeviceAddress::cpu(4)};
        OrchestrationRunner runner(request, std::unique_ptr<IExecutionPlanBuilder>{});
        EXPECT_TRUE(Access::identity(runner).empty());
        const auto resolved = resolve(request, cluster, 0, true);
        ASSERT_TRUE(resolved.config().moe_routed_expert_plan);
        const auto expected = routedWeightAuthorityRequestIdentity(
            resolved.config().moe_routed_expert_plan, resolved.config());
        ASSERT_FALSE(expected.empty());
        Access::install(runner, resolved);
        EXPECT_EQ(Access::identity(runner), expected);
        EXPECT_EQ(runner.executionPlan().toString(), resolved.rankPlan().toString());
        EXPECT_EQ(runner.config().moe_rebalance.mode, mode);
        ASSERT_TRUE(runner.config().moe_routed_expert_plan);
        EXPECT_TRUE(runner.config().moe_routed_expert_plan->placements.empty());
        EXPECT_FALSE(request.moe_routed_expert_plan);

        // A repeated explicit declaration and a prebuilt-plan constructor must
        // agree with the newly synthesized original authority. Runtime dynamic
        // intent is preserved; no physical capacity freeze is rerun for reuse.
        const auto explicit_resolution = resolve(resolved.config(), cluster, 0, true);
        Access::install(runner, explicit_resolution);
        EXPECT_EQ(Access::identity(runner), expected);
        OrchestrationRunner prebuilt(resolved.config(), resolved.rankPlan());
        EXPECT_EQ(Access::identity(prebuilt), expected);

        auto different = resolved.config();
        different.routed_expert_owner_order = RoutedExpertOwnerOrder::Ordinal;
        different.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(
            *different.moe_routed_expert_plan);
        different.moe_routed_expert_plan->owner_order = RoutedExpertOwnerOrder::Ordinal;
        const auto foreign = resolve(different, cluster, 0, true);
        Access::install(runner, foreign);
        EXPECT_NE(Access::identity(runner), expected)
            << "Exact retained reuse must distinguish changed requested ownership";
    }
}

/** @brief Pipeline aggregate and no-overlay requests keep their distinct canonical authority. */
TEST(ResolvedRankOrchestration, RunnerResolvedInstallPreservesPipelineAndDenseAuthority)
{
    using Access = OrchestrationRunnerResolvedAdmissionTestAccess;
    const auto request = pipeline(false, 2, 2);
    const auto resolved = resolve(request, hybridInventory(), 0, true);
    ASSERT_FALSE(resolved.pipelineStages().empty());
    OrchestrationRunner runner(request, std::unique_ptr<IExecutionPlanBuilder>{});
    Access::install(runner, resolved);
    EXPECT_EQ(Access::identity(runner), pipelineRoutedWeightAuthorityRequestIdentity(resolved));
    EXPECT_FALSE(Access::identity(runner).empty());
    EXPECT_EQ(runner.executionPlan().toString(), resolved.rankPlan().toString());

    OrchestrationConfig dense;
    dense.tp_devices = {GlobalDeviceAddress::cpu(7)};
    const auto dense_resolved = resolve(dense, inventory(1), 0, false);
    ASSERT_FALSE(dense_resolved.config().moe_routed_expert_plan);
    ASSERT_TRUE(dense_resolved.pipelineStages().empty());
    OrchestrationRunner dense_runner(dense, std::unique_ptr<IExecutionPlanBuilder>{});
    Access::install(dense_runner, dense_resolved);
    EXPECT_TRUE(Access::identity(dense_runner).empty());
}

/** @test Cross-rank preparation retains original global IDs and claims only actual local stages. */
TEST(ResolvedRankOrchestration, MoEGlobalPipelineOwnsOnlyObservedRankStages)
{
    for (const int count : {2, 3})
    {
        auto observed = inventory(count);
        std::vector<std::string> arguments{"--mpi-procs", std::to_string(count), "--no-mtp"};
        for (int rank = 0; rank < count; ++rank)
        {
            auto &owner = observed.ranks[rank];
            const bool cuda = rank % 2 == 0;
            const auto backend = cuda ? DeviceType::CUDA : DeviceType::ROCm;
            const int first = rank * 4 + 1;
            owner.gpus = {card(backend, first, owner.cpu.numa_node), card(backend, first + 2, owner.cpu.numa_node)};
            const std::string kind = cuda ? "cuda" : "rocm";
            arguments.insert(arguments.end(), {"--define-domain", "stage" + std::to_string(rank) + "=" +
                kind + ":" + std::to_string(first) + "," + kind + ":" + std::to_string(first + 2) +
                ";scope=rank_local;backend=" + (cuda ? "nccl" : "rccl") + ";owner=" + std::to_string(rank),
                "--pp-stage", std::to_string(rank) + "=stage" + std::to_string(rank) + ":" +
                std::to_string(rank) + "-" + std::to_string(rank + 1 == count ? 3 : rank)});
        }
        declareNoPeerTopology(observed);
        observed.buildNodeAggregations();
        const auto config = parse(arguments);
        for (int rank = 0; rank < count; ++rank)
        {
            const auto resolved = resolve(config, observed, rank, true);
            ASSERT_TRUE(resolved.globalPipeline());
            EXPECT_EQ(resolved.globalPipeline()->world_size, count);
            ASSERT_EQ(resolved.pipelineStages().size(), 1u);
            const auto &claim = resolved.pipelineStages().front();
            EXPECT_EQ(claim.stageId(), rank);
            EXPECT_EQ(claim.rankPlan().rank, rank);
            EXPECT_EQ(claim.scope().first_layer, rank);
            EXPECT_EQ(claim.scope().last_layer, rank + 1 == count ? 4 : rank + 1);
            EXPECT_EQ(claim.scope().has_embedding, rank == 0);
            EXPECT_EQ(claim.scope().has_lm_head, rank + 1 == count);
            EXPECT_EQ(claim.scope().generation_domain, DeviceGenerationExecutionDomain::HostedBoundary);
            EXPECT_FALSE(claim.rankPlan().prev_rank);
            EXPECT_FALSE(claim.rankPlan().next_rank);
            EXPECT_FALSE(claim.rankPlan().usesLocalPP());
            const auto &plan = *claim.config().moe_routed_expert_plan;
            ASSERT_EQ(plan.domains.size(), 1u);
            EXPECT_EQ(plan.domains.front().owner_rank, rank);
            ASSERT_EQ(plan.domains.front().participants.size(), 2u);
            EXPECT_EQ(plan.domains.front().participants.front().device_ordinal, rank * 4 + 1);
            EXPECT_EQ(plan.domains.front().participants.back().device_ordinal, rank * 4 + 3);
            EXPECT_FALSE(resolved.config().moe_routed_expert_plan);
            const auto repeated = resolve(resolved.config(), observed, rank, true);
            EXPECT_EQ(pipelineRoutedWeightAuthorityRequestIdentity(resolved),
                pipelineRoutedWeightAuthorityRequestIdentity(repeated));
        }
    }
}

/** @test Global stage identity survives reversed endpoint ranks and disjoint local stage ownership. */
TEST(ResolvedRankOrchestration, MoEGlobalPipelineStageIDsDoNotAliasRankOrLocalIndex)
{
    for (const auto &owners : {std::vector<int>{1,0}, std::vector<int>{2,0,2,1}})
    {
        const int count = *std::max_element(owners.begin(), owners.end()) + 1;
        auto observed = inventory(count);
        for (int rank = 0; rank < count; ++rank)
        {
            auto &entry = observed.ranks[rank];
            entry.gpus = {card(rank % 2 ? DeviceType::ROCm : DeviceType::CUDA,
                rank * 4 + 1, entry.cpu.numa_node)};
        }
        declareNoPeerTopology(observed);
        observed.buildNodeAggregations();
        std::vector<std::string> arguments{"--mpi-procs", std::to_string(count), "--no-mtp"};
        for (size_t stage = 0; stage < owners.size(); ++stage)
        {
            const int owner = owners[stage];
            const bool cuda = owner % 2 == 0;
            arguments.insert(arguments.end(), {"--define-domain", "part" + std::to_string(stage) + "=" +
                (cuda ? "cuda:" : "rocm:") + std::to_string(owner * 4 + 1) +
                ";scope=single;backend=" + (cuda ? "nccl" : "rccl") + ";owner=" + std::to_string(owner),
                "--pp-stage", std::to_string(stage) + "=part" + std::to_string(stage) + ":" +
                std::to_string(stage) + "-" + std::to_string(stage + 1 == owners.size() ? 3 : stage)});
        }
        const auto config = parse(arguments);
        for (int rank = 0; rank < count; ++rank)
        {
            const auto resolved = resolve(config, observed, rank, true);
            ASSERT_TRUE(resolved.globalPipeline());
            EXPECT_EQ(resolved.globalPipeline()->stages.front().owning_rank, owners.front());
            EXPECT_EQ(resolved.globalPipeline()->stages.back().owning_rank, owners.back());
            size_t local = 0;
            for (size_t stage = 0; stage < owners.size(); ++stage)
                if (owners[stage] == rank)
                {
                    ASSERT_LT(local, resolved.pipelineStages().size());
                    const auto &claim = resolved.pipelineStages()[local++];
                    EXPECT_EQ(claim.stageId(), stage);
                    EXPECT_EQ(claim.rankPlan().rank, rank);
                    EXPECT_EQ(claim.scope().first_layer, stage);
                    EXPECT_EQ(claim.scope().has_embedding, stage == 0);
                    EXPECT_EQ(claim.scope().has_lm_head, stage + 1 == owners.size());
                }
            EXPECT_EQ(local, resolved.pipelineStages().size());
        }
    }
}

/**
 * @test Replay exact cross-rank stage geometry before context birth without constructing any native owner.
 * Reversed original ownership and sparse device ordinals reject rank/index heuristics;
 * invalid model geometry and changed real birth membership remain fatal.
 */
TEST(ResolvedRankOrchestration, MoEGlobalPipelineTPValidationUsesStageOwnersBeforeContextBirthReplay)
{
    for (const int width : {1, 2})
    for (const bool reverse : {false, true})
    {
        auto observed = inventory(2);
        for (int rank = 0; rank != 2; ++rank)
        {
            auto &owner = observed.ranks[rank];
            const auto family = rank == 0 ? DeviceType::CUDA : DeviceType::ROCm;
            owner.gpus = {card(family, rank * 4 + 1, owner.cpu.numa_node),
                card(family, rank * 4 + 3, owner.cpu.numa_node)};
        }
        declareNoPeerTopology(observed);
        observed.buildNodeAggregations();
        std::vector<std::string> args{"--mpi-procs", "2", "--no-mtp"};
        for (int stage = 0; stage != 2; ++stage)
        {
            const int owner = reverse ? 1 - stage : stage;
            const std::string family = owner == 0 ? "cuda" : "rocm";
            args.insert(args.end(), {"--define-domain", "part" + std::to_string(stage) + "=" +
                family + ":" + std::to_string(owner * 4 + 1) +
                (width == 1 ? std::string{} : "," + family + ":" + std::to_string(owner * 4 + 3)) +
                (width == 1 ? ";scope=single;backend=" : ";scope=rank_local;backend=") + (owner == 0 ? "nccl" : "rccl") + ";owner=" + std::to_string(owner),
                "--pp-stage", std::to_string(stage) + "=part" + std::to_string(stage) + ":" +
                    (stage == 0 ? "0-0" : "1-3")});
        }
        const auto requested = parse(args);
        const auto model = test::MockModelContextBuilder().setArchitecture("qwen35").setBlockCount(4)
            .setEmbeddingLength(512).setHeadCount(32).setHeadCountKV(2)
            .setFeedForwardLength(1536).setVocabSize(320).build();
        const auto invalid_model = test::MockModelContextBuilder().setArchitecture("qwen35").setBlockCount(4)
            .setEmbeddingLength(512).setHeadCount(32).setHeadCountKV(2)
            .setFeedForwardLength(0).setVocabSize(320).build();
        for (int rank = 0; rank != 2; ++rank)
        {
            const auto resolved = resolve(requested, observed, rank, true);
            EXPECT_EQ(resolved.rankPlan().usesLocalTP(), width > 1);
            ASSERT_FALSE(resolved.rankPlan().usesLocalPP());
            for (int replay = 0; replay != 3; ++replay)
            {
                const auto validated = validateResolvedGlobalStageTP(resolved, *model, rank);
                ASSERT_TRUE(validated.valid) << validated.toString();
                EXPECT_FALSE(validateResolvedGlobalStageTP(resolved, *model, 1 - rank).valid);
                const auto &canonical = resolved.pipelineStages().front().config();
                EXPECT_TRUE(canonical.tp_devices.empty());
                EXPECT_TRUE(canonical.tp_weights.empty());
                EXPECT_EQ(canonical.tp_degree, 1);
                EXPECT_EQ(canonical.tp_scope, TPScope::AUTO);
                ASSERT_EQ(canonical.domain_definitions.size(), 1u);
                EXPECT_EQ(canonical.domain_definitions.front().devices.size(), static_cast<size_t>(width));
                auto missing_domain_members = canonical;
                missing_domain_members.domain_definitions.front().devices.clear();
                EXPECT_FALSE(TPPPValidator::validate(missing_domain_members, *model).valid);
                if (width > 1)
                    EXPECT_FALSE(validateResolvedGlobalStageTP(resolved, *invalid_model, rank).valid);
                const auto plan = GlobalPPRankPlanBuilder::build(*resolved.globalPipeline(), rank);
                ASSERT_EQ(plan.executeStages().size(), 1u);
                const auto &action = *plan.executeStages().front();
                EXPECT_NO_THROW(requireOriginalGlobalStageTPBirth(action, action.devices, action.backend));
                auto changed = action.devices;
                std::reverse(changed.begin(), changed.end());
                if (width > 1)
                    EXPECT_THROW(requireOriginalGlobalStageTPBirth(action, changed, action.backend), std::invalid_argument);
                changed.pop_back();
                EXPECT_THROW(requireOriginalGlobalStageTPBirth(action, changed, action.backend), std::invalid_argument);
                EXPECT_THROW(requireOriginalGlobalStageTPBirth(action, action.devices,
                    action.backend == CollectiveBackendType::NCCL ? CollectiveBackendType::RCCL : CollectiveBackendType::NCCL),
                    std::invalid_argument);
                auto changed_action = action;
                changed_action.role = RankStageAction::Role::IDLE;
                EXPECT_THROW(requireOriginalGlobalStageTPBirth(changed_action, action.devices, action.backend), std::invalid_argument);
                changed_action = action;
                changed_action.is_global_tp = true;
                EXPECT_THROW(requireOriginalGlobalStageTPBirth(changed_action, action.devices, action.backend), std::invalid_argument);
            }
        }
    }
}

/**
 * @test Named stage policies override parent intent independently, including
 * unequal widths, three-stage composition, omitted fields and repeated resolve.
 * Before the repair every explicit stage silently reused parent Automatic.
 */
TEST(ResolvedRankOrchestration, AuthoredPipelineRoutedPoliciesSurviveIndependentStagesReplay)
{
    for (int replay = 0; replay < 20; ++replay)
        for (const auto widths : {std::vector<int>{1, 3}, std::vector<int>{3, 1},
                 std::vector<int>{3, 5}, std::vector<int>{1, 3, 4}})
            for (const auto inherited : {RoutedExpertComputePolicy::Automatic,
                     RoutedExpertComputePolicy::Apportioned,
                     RoutedExpertComputePolicy::GateUpOwnedDownColumns})
            {
                std::vector<DeviceInfo> cards;
                std::vector<std::string> args{"--no-mtp"};
                int ordinal = 0;
                for (size_t stage = 0; stage < widths.size(); ++stage)
                {
                    std::string domain = "part" + std::to_string(stage) + "=";
                    for (int member = 0; member < widths[stage]; ++member, ++ordinal)
                    {
                        domain += (member ? "," : "") + std::string("rocm:") + std::to_string(ordinal);
                        cards.push_back(card(DeviceType::ROCm, ordinal, 7));
                    }
                    domain += ";scope=rank_local;backend=rccl;owner=0";
                    if (stage == 0)
                        domain += ";expert_compute=apportioned;expert_phase=uniform;expert_decode_assignment=static-owner;expert_prefill_assignment=" +
                            std::string(widths[stage] > 1 ? "least-loaded-resident" : "static-owner");
                    else if (stage == 1)
                        domain += std::string(";expert_compute=") +
                            (widths[stage] > 1 ? "gate-up-owned-down-columns" : "apportioned") + ";expert_phase=uniform";
                    args.insert(args.end(), {"--define-domain", domain, "--pp-stage",
                        std::to_string(stage) + "=part" + std::to_string(stage) + ":" +
                        std::to_string(stage) + "-" + std::to_string(stage + 1 == widths.size() ? 3 : stage)});
                }
                auto request = parse(std::move(args));
                request.routed_expert_compute_policy = inherited;
                const auto observed = inventory(1, std::move(cards));
                ExecutionPlanBuilder compiler;
                const auto model = metadata(true, 4, 120); // Divisible by all authored widths1/3/4/5.
                const auto result = ResolvedRankOrchestration::resolve(request, model, observed, compiler, 0);
                ASSERT_EQ(result.pipelineStages().size(), widths.size());
                const auto again = ResolvedRankOrchestration::resolve(result.config(), model, observed, compiler, 0);
                for (size_t stage = 0; stage < widths.size(); ++stage)
                {
                    const auto &child = result.pipelineStages()[stage];
                    const auto &domain = child.config().moe_routed_expert_plan->domains.front();
                    const auto expected = stage == 0 ? RoutedExpertComputePolicy::Apportioned :
                        stage == 1 ? (widths[stage] > 1 ? RoutedExpertComputePolicy::GateUpOwnedDownColumns : RoutedExpertComputePolicy::Apportioned) :
                        inherited == RoutedExpertComputePolicy::Automatic
                            ? RoutedExpertComputePolicy::GateUpOwnedDownColumns : inherited;
                    EXPECT_EQ(domain.routed_compute_policy, expected);
                    EXPECT_EQ(child.config().routed_expert_compute_policy, expected);
                    EXPECT_EQ(child.rankPlan().runtime.routed_expert_compute_policy, expected);
                    EXPECT_EQ(domain.routed_phase_policy, RoutedExpertPhasePolicy::Uniform);
                    EXPECT_EQ(domain.routed_decode_assignment_policy, RoutedExpertAssignmentPolicy::StaticOwner);
                    EXPECT_EQ(domain.routed_prefill_assignment_policy, stage == 0 && widths[stage] > 1
                        ? RoutedExpertAssignmentPolicy::LeastLoadedResident : RoutedExpertAssignmentPolicy::StaticOwner);
                    EXPECT_EQ(domain.participants.size(), static_cast<size_t>(widths[stage]));
                    EXPECT_EQ(result.rankPlan().local_pp_stage_tp_info[stage].domain_name,
                              "part" + std::to_string(stage));
                    EXPECT_EQ(domain.routed_compute_policy, again.pipelineStages()[stage].config()
                        .moe_routed_expert_plan->domains.front().routed_compute_policy);
                    EXPECT_EQ(request.routed_expert_compute_policy, inherited);
                    EXPECT_EQ(request.domain_definitions[stage].routed_compute_policy,
                        stage == 0 ? RoutedExpertComputePolicy::Apportioned :
                        stage == 1 ? (widths[stage] > 1 ? RoutedExpertComputePolicy::GateUpOwnedDownColumns : RoutedExpertComputePolicy::Apportioned) :
                        RoutedExpertComputePolicy::Unspecified);
                }
            }
}

/** @test A genuine socket-local CPU singleton keeps its own policy between unequal GPU domains. */
TEST(ResolvedRankOrchestration, AuthoredMixedPipelineRetainsCPUAndUnequalGPUStagePolicies)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        std::vector<DeviceInfo> cards;
        for (int device = 0; device < 8; ++device)
            cards.push_back(card(DeviceType::ROCm, device, 7));
        const auto observed = inventory(1, std::move(cards));
        const auto request = parse({"--no-mtp",
            "--define-domain", "first=rocm:0,rocm:1,rocm:2;scope=rank_local;owner=0;backend=rccl;expert_compute=apportioned;expert_phase=uniform",
            "--define-domain", "middle=cpu:7;scope=single;owner=0;backend=host;expert_compute=apportioned;expert_phase=uniform",
            "--define-domain", "last=rocm:3,rocm:4,rocm:5,rocm:6,rocm:7;scope=rank_local;owner=0;backend=rccl;expert_compute=gate-up-owned-down-columns;expert_phase=uniform",
            "--pp-stage", "0=first:0-0", "--pp-stage", "1=middle:1-1", "--pp-stage", "2=last:2-3"});
        ExecutionPlanBuilder compiler;
        const auto model = metadata(true, 4, 120);
        const auto result = ResolvedRankOrchestration::resolve(request, model, observed, compiler, 0);
        ASSERT_EQ(result.pipelineStages().size(), 3u);
        const auto &first = result.pipelineStages()[0].config().moe_routed_expert_plan->domains.front();
        const auto &middle = result.pipelineStages()[1].config().moe_routed_expert_plan->domains.front();
        const auto &last = result.pipelineStages()[2].config().moe_routed_expert_plan->domains.front();
        EXPECT_EQ(first.participants.size(), 3u);
        EXPECT_EQ(first.routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
        ASSERT_EQ(middle.participants.size(), 1u);
        EXPECT_TRUE(middle.participants.front().isCPU());
        EXPECT_EQ(middle.scope, ExecutionDomainScope::SINGLE);
        EXPECT_EQ(middle.routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
        EXPECT_EQ(middle.backend, CollectiveBackendType::HOST);
        EXPECT_EQ(last.participants.size(), 5u);
        EXPECT_EQ(last.routed_compute_policy, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
        EXPECT_FALSE(result.pipelineStages()[1].rankPlan().usesGlobalTP());
        EXPECT_FALSE(result.pipelineStages()[1].rankPlan().usesLocalTP());
    }
}

/** @test Authored participant counts and policies cannot claim another original NUMA rank's GPUs. */
TEST(ResolvedRankOrchestration, AuthoredPipelineDomainCannotOverrideOriginalRankOwnership)
{
    auto observed = inventory(2);
    for (auto &rank : observed.ranks)
        for (int device = 0; device < 8; ++device)
            rank.gpus.push_back(card(DeviceType::ROCm, device, observed.ranks[1].cpu.numa_node));
    declareNoPeerTopology(observed);
    observed.buildNodeAggregations();
    for (const auto widths : {std::array{1, 2}, std::array{2, 1}, std::array{4, 4}})
    {
        std::vector<std::string> args{"--mpi-procs", "2", "--no-mtp"};
        int offset = 0;
        for (int stage = 0; stage < 2; ++stage)
        {
            std::string domain = "p" + std::to_string(stage) + "=";
            for (int member = 0; member < widths[stage]; ++member)
                domain += (member ? "," : "") + std::string("rocm:") + std::to_string(offset++);
            domain += ";scope=rank_local;backend=rccl;owner=0;expert_compute=apportioned;expert_phase=uniform";
            args.insert(args.end(), {"--define-domain", domain, "--pp-stage", std::to_string(stage) +
                "=p" + std::to_string(stage) + ":" + (stage == 0 ? "0-0" : "1-3")});
        }
        const auto request = parse(std::move(args));
        ExecutionPlanBuilder compiler;
        EXPECT_THROW((void)ResolvedRankOrchestration::resolve(request, metadata(true, 4, 120),
            observed, compiler, 0), std::invalid_argument);
    }
}

namespace {
/** @return One observed node with one real CPU owner per sparse NUMA identity and a rank-local GPU tail. */
ClusterInventory cpuPipelineInventory(int count)
{
    auto result=inventory(count);
    for(int rank=0;rank<count;++rank) {
        result.ranks[rank].cpu.numa_node=3+rank*4;
        result.ranks[rank].numa_nodes=count;
    }
    result.ranks[0].gpus={card(DeviceType::ROCm,41,3)};
    declareNoPeerTopology(result);
    result.buildNodeAggregations();return result;
}
/** @return Authored cross-rank CPU stage plus original singleton GPU stage, without remote GPU aliases. */
OrchestrationConfig cpuPipelineRequest(const ClusterInventory& original,RoutedExpertComputePolicy compute)
{
    OrchestrationConfig config;
    DomainDefinition cpu;
    cpu.name="cpu_stage";cpu.scope=TPScope::NODE_LOCAL;cpu.backend=CollectiveBackendType::UPI;
    cpu.routed_compute_policy=compute;
    for(const auto& rank:original.ranks) {
        cpu.devices.push_back(GlobalDeviceAddress::cpu(rank.cpu.numa_node,rank.hostname));
        cpu.explicit_ranks.push_back(rank.rank);
    }
    DomainDefinition gpu;
    gpu.name="gpu_stage";gpu.scope=TPScope::RANK_LOCAL;gpu.owner_rank=0;
    gpu.devices={GlobalDeviceAddress::rocm(41,3,"node")};
    gpu.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
    config.domain_definitions={cpu,gpu};
    config.pp_stage_definitions={{0,"cpu_stage",0,1},{1,"gpu_stage",2,3}};
    config.kv_cache_precision="fp32";return config;
}
}

/** @test Stage topology preserves each real CPU rank and its exact original dense TP shard. */
TEST(ResolvedRankOrchestration, CPUPipelineStageKeepsOriginalGlobalTPAndPerStageExpertPolicy)
{
    for(int count:{2,3,5}) for(auto compute:{RoutedExpertComputePolicy::TensorSharded,RoutedExpertComputePolicy::Apportioned}) {
        auto original=cpuPipelineInventory(count);const auto config=cpuPipelineRequest(original,compute);
        for(int rank=0;rank<count;++rank) {
            SCOPED_TRACE(::testing::Message()<<count<<"/"<<rank<<"/"<<int(compute));
            ExecutionPlanBuilder builder;
            const auto resolved=ResolvedRankOrchestration::resolve(config,metadata(true,4,120),original,builder,rank);
            ASSERT_TRUE(resolved.globalPipeline());
            EXPECT_EQ(resolved.pipelineStages().size(),rank==0?2u:1u);
            const auto& child=resolved.pipelineStages().front();
            EXPECT_EQ(child.stageId(),0);EXPECT_TRUE(child.rankPlan().usesGlobalTP());
            EXPECT_FALSE(child.rankPlan().usesLocalTP());
            EXPECT_EQ(child.rankPlan().global_tp_domain_size,count);
            EXPECT_EQ(child.rankPlan().global_tp_rank_in_domain,rank);
            EXPECT_EQ(child.rankPlan().weight_shard.total_shards,count);
            EXPECT_EQ(child.rankPlan().weight_shard.shard_index,rank);
            EXPECT_EQ(child.rankPlan().primary_device.numa_node,original.ranks[rank].cpu.numa_node);
            const auto& plan=*child.config().moe_routed_expert_plan;
            ASSERT_EQ(plan.domains.size(),1u);const auto& domain=plan.domains.front();
            EXPECT_EQ(domain.scope,ExecutionDomainScope::NODE_LOCAL);
            EXPECT_EQ(domain.participants.size(),std::size_t(count));
            EXPECT_EQ(domain.routed_compute_policy,compute);
            EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(),DenseParallelPolicy::TensorParallel);
            ASSERT_EQ(domain.world_ranks.size(),std::size_t(count));
            for(int participant=0;participant<count;++participant) {
                EXPECT_EQ(domain.world_ranks[participant],participant);
                EXPECT_EQ(domain.participants[participant].numa_node,original.ranks[participant].cpu.numa_node);
            }
            EXPECT_EQ(child.rankPlan().runtime.routed_expert_compute_policy,compute);
        }
    }
}

/** @test A global CPU stage cannot borrow foreign ranks, change NUMA identity or duplicate physical owners. */
TEST(ResolvedRankOrchestration, CPUPipelineStageRejectsForeignRankNUMAAndDuplicatedPhysicalOwnership)
{
    for(int defect:{0,1,2}) {
        const auto original=cpuPipelineInventory(2);
        auto config=cpuPipelineRequest(original,RoutedExpertComputePolicy::Apportioned);
        if(defect==0) config.domain_definitions[0].explicit_ranks[1]=2;
        if(defect==1) config.domain_definitions[0].devices[1].numa_node=999;
        if(defect==2) config.domain_definitions[0].devices[1]=config.domain_definitions[0].devices[0];
        for(int rank=0;rank<2;++rank) {
            ExecutionPlanBuilder builder;
            EXPECT_THROW(ResolvedRankOrchestration::resolve(config,metadata(true,4,120),original,builder,rank),std::invalid_argument);
        }
    }
}

/** @test The actual global-stage validator authenticates every original CPU member before native birth. */
TEST(ResolvedRankOrchestration, CPUPipelineStageOriginalTPValidationUsesDeclaredRanksAndModelGeometry)
{
    const auto model=test::MockModelContextBuilder().setArchitecture("qwen35").setBlockCount(4)
        .setEmbeddingLength(1920).setHeadCount(120).setHeadCountKV(120)
        .setFeedForwardLength(5760).setVocabSize(320).build();
    const auto absent_ffn=test::MockModelContextBuilder().setArchitecture("qwen35").setBlockCount(4)
        .setEmbeddingLength(1920).setHeadCount(120).setHeadCountKV(120)
        .setFeedForwardLength(0).setVocabSize(320).build();
    const auto incompatible_heads=test::MockModelContextBuilder().setArchitecture("qwen35").setBlockCount(4)
        .setEmbeddingLength(1936).setHeadCount(121).setHeadCountKV(121)
        .setFeedForwardLength(5808).setVocabSize(320).build();
    for(const int count:{2,3,5,6,10,12}) {
        const auto original=cpuPipelineInventory(count);
        const auto source=metadata(true,4,120);
        for(const bool cpu_tail:{false,true})
        for(const auto compute:{RoutedExpertComputePolicy::TensorSharded,RoutedExpertComputePolicy::Apportioned}) {
            auto config=cpuPipelineRequest(original,compute);
            // A CPU sharded domain overrides the supported apportioned parent;
            // an explicit apportioned domain resolves the supported Automatic
            // parent. The ordinary global TensorSharded scalar remains invalid.
            config.routed_expert_compute_policy=compute==RoutedExpertComputePolicy::TensorSharded?
                RoutedExpertComputePolicy::Apportioned:RoutedExpertComputePolicy::Automatic;
            if(cpu_tail) {
                auto& tail=config.domain_definitions[1];
                const auto& owner=original.ranks.back();
                tail.name="cpu_tail";tail.owner_rank=owner.rank;tail.scope=TPScope::AUTO;
                tail.devices={GlobalDeviceAddress::cpu(owner.cpu.numa_node,owner.hostname)};
                tail.backend=CollectiveBackendType::HOST;
                config.pp_stage_definitions[1].domain_name=tail.name;
            }
            for(int rank=0;rank<count;++rank) {
                SCOPED_TRACE(::testing::Message()<<count<<"/"<<rank<<"/"<<cpu_tail<<"/"<<int(compute));
                ExecutionPlanBuilder builder;
                const auto resolved=ResolvedRankOrchestration::resolve(config,source,original,builder,rank);
                for(int replay=0;replay<3;++replay) {
                    const auto checked=validateResolvedGlobalStageTP(resolved,*model,rank);
                    ASSERT_TRUE(checked.valid)<<checked.toString();
                    EXPECT_FALSE(validateResolvedGlobalStageTP(resolved,*model,(rank+1)%count).valid);
                    EXPECT_FALSE(validateResolvedGlobalStageTP(resolved,*absent_ffn,rank).valid);
                    EXPECT_FALSE(validateResolvedGlobalStageTP(resolved,*incompatible_heads,rank).valid);
                }
                const auto& stage=resolved.pipelineStages().front();
                EXPECT_TRUE(stage.rankPlan().local_tp_devices.empty());
                EXPECT_TRUE(stage.rankPlan().local_tp_weights.empty());
                EXPECT_EQ(stage.rankPlan().weight_shard.shard_index,rank);
                EXPECT_EQ(stage.rankPlan().weight_shard.total_shards,count);
                EXPECT_EQ(stage.config().routed_expert_compute_policy,compute);
                const auto& placement = *stage.config().moe_routed_expert_plan;
                const auto dense = std::find_if(placement.dense_domains.begin(), placement.dense_domains.end(),
                    [&](const auto& domain) { return domain.name == placement.effectiveBaseModelDomain(); });
                ASSERT_NE(dense, placement.dense_domains.end());
                EXPECT_EQ(dense->scope, ExecutionDomainScope::NODE_LOCAL);
                EXPECT_EQ(dense->ranks, resolved.globalPipeline()->stages.front().participating_ranks);
                EXPECT_EQ(dense->participants, resolved.globalPipeline()->stages.front().per_rank_devices);
                EXPECT_EQ(dense->backend, resolved.globalPipeline()->stages.front().backend);
                EXPECT_EQ(placement.continuation_domain_spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallel);
                const auto plan=GlobalPPRankPlanBuilder::build(*resolved.globalPipeline(),rank);
                const auto* action=plan.executeStages().front();
                const auto& declared=resolved.globalPipeline()->stages.front();
                EXPECT_NO_THROW((void)OriginalPipelineStageDomain::borrow(declared,*action));
                auto foreign=*action;foreign.device.numa_node=999;
                EXPECT_THROW((void)OriginalPipelineStageDomain::borrow(declared,foreign),std::invalid_argument);
                foreign=*action;foreign.backend=CollectiveBackendType::NCCL;
                EXPECT_THROW((void)OriginalPipelineStageDomain::borrow(declared,foreign),std::invalid_argument);
                foreign=*action;foreign.tp_domain_size--;
                EXPECT_THROW((void)OriginalPipelineStageDomain::borrow(declared,foreign),std::invalid_argument);
            }
        }
    }
}
