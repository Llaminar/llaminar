/**
 * @file Test__MTPHardwareDefaults.cpp
 * @brief Device-free proofs of topology-bound MTP defaults and explicit intent.
 *
 * Synthetic inventory drives the real execution-plan builder. These tests
 * cover complete continuation domains, unrelated expert tiers, repeated local
 * ordinals across ranks, request reuse, parser round trips, and the sealed
 * device policy. No model, GPU context, or performance timing is involved.
 * CPU ownership is an explicit observed NUMA fact, not a rank-ordinal inference.
 * Ordinary QFN vocabulary defaults are constrained independently of MTP to the
 * exact authenticated architecture, native source and full local MI50 TP4 domain.
 */
#include <gtest/gtest.h>

#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationConfigDocument.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/mtp/MTPDeviceGenerationPolicy.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "planning/MemoryPlanner.h"

#include <limits>
#include <utility>

using namespace llaminar2;

namespace
{
    /** @brief Inventory-only measured card, without device discovery. */
    DeviceInfo card(DeviceType type, int ordinal)
    {
        DeviceInfo result;
        result.type = type;
        result.local_device_id = ordinal;
        result.numa_node = 0;
        result.compute_units = type == DeviceType::ROCm ? 60 : 82;
        result.name = type == DeviceType::ROCm
            ? "AMD Instinct MI60 / MI50" : "NVIDIA GeForce RTX 3090";
        return result;
    }

    /** @brief Build canonical rank membership from caller-owned fake devices. */
    ClusterInventory inventory(std::vector<std::vector<DeviceInfo>> devices)
    {
        ClusterInventory result;
        result.world_size = static_cast<int>(devices.size());
        for (int rank = 0; rank < result.world_size; ++rank)
        {
            RankInventory entry;
            entry.rank = rank;
            entry.local_rank = rank;
            entry.hostname = "node";
            entry.node_id = 0;
            entry.numa_nodes = result.world_size;
            entry.cpu.numa_node = rank;
            entry.gpus = std::move(devices[rank]);
            result.ranks.push_back(std::move(entry));
        }
        result.buildNodeAggregations();
        return result;
    }

    /** @brief Physical address sharing the fake inventory's exact locality. */
    GlobalDeviceAddress address(DeviceType type, int ordinal)
    {
        return GlobalDeviceAddress::fromLocalDeviceId(DeviceId(type, ordinal), "node", 0);
    }

    /** @brief Run the real planner without a model file or a backend. */
    std::vector<RankExecutionPlan> plans(
        const OrchestrationConfig &config, const ClusterInventory &cluster)
    {
        ExecutionPlanBuilder builder;
        return builder.buildAllPlans(config, ModelConfig::qwen2_7b(), cluster);
    }

    /** @brief Declared domain with participant-indexed rank ownership. */
    DomainDefinition domain(
        std::string name, std::vector<GlobalDeviceAddress> devices,
        std::vector<int> ranks)
    {
        DomainDefinition result;
        result.name = std::move(name);
        result.devices = std::move(devices);
        result.explicit_ranks = std::move(ranks);
        result.scope = TPScope::NODE_LOCAL;
        return result;
    }
}

TEST(MTPHardwareDefaults, SingleAndHomogeneousLocalTPUseMeasuredCardProfile)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        for (const int count : {1, 2, 3, 4, 8})
        {
            SCOPED_TRACE(::testing::Message() << deviceTypeToString(type) << " count=" << count);
            OrchestrationConfig config;
            std::vector<DeviceInfo> devices;
            for (int ordinal = 0; ordinal < count; ++ordinal)
            {
                devices.push_back(card(type, ordinal));
                config.tp_devices.push_back(address(type, ordinal));
            }
            const auto result = plans(config, inventory({devices}));
            ASSERT_EQ(result.size(), 1u);
            const auto &mtp = result.front().runtime.mtp;
            EXPECT_EQ(mtp.terminal_head_policy, MTPTerminalHeadPolicy::MirroredFullVocabulary);
            EXPECT_EQ(mtp.depth_defaults_profile, type == DeviceType::CUDA
                ? MTPDepthDefaultsProfile::CUDARTX3090 : MTPDepthDefaultsProfile::ROCmMI50);
            EXPECT_DOUBLE_EQ(resolveMTPZeroAcceptDemotionRate(mtp), type == DeviceType::CUDA ? 0.30 : 0.45);
            EXPECT_FALSE(mtp.depth_policy.demote_zero_accept_rate.has_value());
            EXPECT_FALSE(mtp.enabled);
            EXPECT_EQ(mtp.draft_tokens, 1);
            EXPECT_EQ(mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
            EXPECT_EQ(mtp.graph_capacity_draft_tokens, 0);
        }
    }
}

/** @test The measured head choice follows complete model, format and physical membership. */
TEST(MTPHardwareDefaults, DenseQwenGPUHeadDefaultsRequireMeasuredNativeLocalFamily)
{
    ModelConfig model{.name = "qwen35", .n_layers = 64, .n_heads = 24, .n_kv_heads = 4,
        .hidden_size = 5120, .intermediate_size = 17408, .vocab_size = 248320, .head_dim = 256,
        .terminal_head_source = {.codebook_id = native_vnni_formats::Q6_K.codebook_id,
            .is_superblock = true, .present = true}};
    ExecutionPlanBuilder builder;
    for (const auto type : {DeviceType::ROCm, DeviceType::CUDA})
    {
        auto cluster = inventory({{card(type, 0), card(type, 1)}});
        OrchestrationConfig config;
        config.tp_devices = {address(type, 0), address(type, 1)};
        config.mtp.enabled = true;
        config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        const auto expected = MTPTerminalHeadPolicy::VocabularySharded;
        const auto selected = [&] { return builder.buildAllPlans(config, model, cluster)
            .front().runtime.mtp.terminal_head_policy; };
        EXPECT_EQ(selected(), expected);
        config.mtp.depth_policy.max_depth = 2;
        config.mtp.graph_capacity_draft_tokens = 2;
        EXPECT_EQ(selected(), expected);
        config.mtp.graph_capacity_draft_tokens = 15;
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        config.mtp.graph_capacity_draft_tokens = 0;
        config.mtp.depth_policy.max_depth = 3;
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        config.mtp.depth_policy.max_depth = 0;
        auto unmeasured = cluster;
        unmeasured.ranks.front().gpus.back().name = "unmeasured accelerator";
        EXPECT_EQ(builder.buildAllPlans(config, model, unmeasured).front().runtime.mtp.terminal_head_policy,
            MTPTerminalHeadPolicy::MirroredFullVocabulary);
        for (const auto explicit_policy : {MTPTerminalHeadPolicy::VocabularySharded,
                MTPTerminalHeadPolicy::MirroredFullVocabulary})
        {
            config.mtp.terminal_head_policy = explicit_policy;
            EXPECT_EQ(selected(), explicit_policy);
        }
        config.mtp.terminal_head_policy = MTPTerminalHeadPolicy::Automatic;
        config.mtp.enabled = false;
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        config.mtp.enabled = true;
        config.mtp.graph_capacity_draft_tokens = 17;
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        config.mtp.graph_capacity_draft_tokens = 0;
        config.mtp.max_request_batch = 2;
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        config.mtp.max_request_batch = 1;
        model.terminal_head_source.present = false;
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        model.terminal_head_source.present = true;
        model.terminal_head_source.codebook_id = native_vnni_formats::Q8_0.codebook_id;
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        model.terminal_head_source.codebook_id = native_vnni_formats::Q6_K.codebook_id;
        model.name = "qwen35moe";
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        model.name = "qwen35";
        config.tp_devices = {address(type, 0)};
        EXPECT_EQ(selected(), MTPTerminalHeadPolicy::MirroredFullVocabulary);
        config.tp_devices.clear();
        config.domain_definitions = {domain("terminal", {address(type, 0), address(type, 1)}, {0, 0})};
        EXPECT_EQ(selected(), expected);
        config.domain_definitions = {domain("terminal", {address(type, 0), address(type, 0)}, {0, 1})};
        cluster = inventory({{card(type, 0)}, {card(type, 0)}});
        for (const auto &plan : builder.buildAllPlans(config, model, cluster))
            EXPECT_EQ(plan.runtime.mtp.terminal_head_policy, MTPTerminalHeadPolicy::MirroredFullVocabulary);
    }
}

namespace
{
    /** @brief Authenticated published QFN planning geometry, without any model payload. */
    ModelConfig ordinaryQFN()
    {
        return {.name = "qwen4exp", .n_layers = 48, .n_heads = 24, .n_kv_heads = 2,
            .hidden_size = 2560, .intermediate_size = 640, .vocab_size = 248320, .head_dim = 256,
            .terminal_head_source = {.codebook_id = native_vnni_formats::Q8_0.codebook_id,
                .is_superblock = false, .present = true}};
    }

    /**
     * @brief Build one observed MI50 clique with exact local TP addresses.
     * @param count Synthetic physical participants, including unmeasured negative degrees.
     * @return Authored Auto configuration and matching inventory; no backend is created.
     */
    std::pair<OrchestrationConfig, ClusterInventory> ordinaryQFNLocalTP(int count = 4)
    {
        OrchestrationConfig config;
        std::vector<DeviceInfo> devices;
        for (int ordinal = 0; ordinal < count; ++ordinal)
        {
            devices.push_back(card(DeviceType::ROCm, ordinal));
            config.tp_devices.push_back(address(DeviceType::ROCm, ordinal));
        }
        return {config, inventory({devices})};
    }

    /**
     * @brief Assert runtime agreement without mutating authored Auto intent.
     * @param config Caller-owned startup configuration.
     * @param model Authenticated or deliberately unmeasured synthetic geometry.
     * @param cluster Synthetic gathered physical inventory.
     * @param expected Exact resolved physical policy for every returned rank.
     */
    void expectOrdinaryQFNPolicy(const OrchestrationConfig &config, const ModelConfig &model,
        const ClusterInventory &cluster, MTPTerminalHeadPolicy expected)
    {
        const auto result = ExecutionPlanBuilder{}.buildAllPlans(config, model, cluster);
        ASSERT_FALSE(result.empty());
        for (const auto &plan : result)
        {
            EXPECT_EQ(plan.runtime.mtp.terminal_head_policy, expected);
            if (expected == MTPTerminalHeadPolicy::VocabularySharded)
                EXPECT_EQ(plan.local_tp_backend, CollectiveBackendType::RCCL);
        }
        EXPECT_EQ(config.mtp.terminal_head_policy, MTPTerminalHeadPolicy::Automatic);
    }
}

/** @test The measured ordinary default follows both actual simple and named-domain ownership. */
TEST(MTPHardwareDefaults, QFNOrdinaryHeadDefaultsSelectEqualLocalTP4)
{
    const auto model = ordinaryQFN();
    auto [config, cluster] = ordinaryQFNLocalTP();
    for (const bool named : {false, true})
        for (const bool explicit_weights : {false, true})
        {
            auto selected = config;
            if (named)
            {
                selected.domain_definitions = {domain("terminal", selected.tp_devices, {0, 0, 0, 0})};
                selected.tp_devices.clear();
                if (explicit_weights) selected.domain_definitions.front().weights = {.25f, .25f, .25f, .25f};
            }
            else if (explicit_weights) selected.tp_weights = {.25f, .25f, .25f, .25f};
            expectOrdinaryQFNPolicy(selected, model, cluster, MTPTerminalHeadPolicy::VocabularySharded);
            if (named) selected.domain_definitions.front().backend = CollectiveBackendType::RCCL;
            else selected.default_backend = CollectiveBackendType::RCCL;
            expectOrdinaryQFNPolicy(selected, model, cluster, MTPTerminalHeadPolicy::VocabularySharded);
            if (named)
            {
                selected.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>();
                selected.moe_routed_expert_plan->enabled = true;
                selected.moe_routed_expert_plan->continuation_domain = "terminal";
                selected.moe_routed_expert_plan->continuation_domain_spec.domain = "terminal";
                selected.moe_routed_expert_plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
                expectOrdinaryQFNPolicy(selected, model, cluster, MTPTerminalHeadPolicy::VocabularySharded);
            }
        }
}

/** @test A similar architecture or decoder is not provenance for the measured QFN head. */
TEST(MTPHardwareDefaults, QFNOrdinaryHeadDefaultsRejectOtherGeometryAndSource)
{
    const auto [config, cluster] = ordinaryQFNLocalTP();
    const auto expected = MTPTerminalHeadPolicy::MirroredFullVocabulary;
    for (const auto member : {&ModelConfig::n_layers, &ModelConfig::n_heads,
             &ModelConfig::n_kv_heads, &ModelConfig::hidden_size, &ModelConfig::intermediate_size,
             &ModelConfig::vocab_size, &ModelConfig::head_dim})
    {
        auto model = ordinaryQFN();
        ++(model.*member);
        expectOrdinaryQFNPolicy(config, model, cluster, expected);
    }
    auto model = ordinaryQFN();
    model.name = "qwen35moe";
    expectOrdinaryQFNPolicy(config, model, cluster, expected);
    model = ordinaryQFN();
    model.terminal_head_source.present = false;
    expectOrdinaryQFNPolicy(config, model, cluster, expected);
    model = ordinaryQFN();
    model.terminal_head_source.is_superblock = true;
    expectOrdinaryQFNPolicy(config, model, cluster, expected);
    model = ordinaryQFN();
    model.terminal_head_source.codebook_id = native_vnni_formats::Q6_K.codebook_id;
    expectOrdinaryQFNPolicy(config, model, cluster, expected);
}

/** @test Batch demand and true retained capacity stay outside the ordinary M1 default. */
TEST(MTPHardwareDefaults, QFNOrdinaryHeadDefaultsRejectBatchAndRetainedMTP)
{
    const auto model = ordinaryQFN();
    const auto [original, cluster] = ordinaryQFNLocalTP();
    auto config = original;
    config.batch_size = 2;
    expectOrdinaryQFNPolicy(config, model, cluster, MTPTerminalHeadPolicy::MirroredFullVocabulary);
    config = original;
    config.mtp.enabled = true;
    expectOrdinaryQFNPolicy(config, model, cluster, MTPTerminalHeadPolicy::MirroredFullVocabulary);
    config = original;
    config.mtp.graph_capacity_draft_tokens = 1;
    expectOrdinaryQFNPolicy(config, model, cluster, MTPTerminalHeadPolicy::MirroredFullVocabulary);
    // Inactive schema/default draft depth is not retained graph capacity.
    config = original;
    config.mtp.draft_tokens = 15;
    expectOrdinaryQFNPolicy(config, model, cluster, MTPTerminalHeadPolicy::VocabularySharded);
}

/** @test Real hardware, full-model stage and equal shares bound the measured clique. */
TEST(MTPHardwareDefaults, QFNOrdinaryHeadDefaultsRejectOtherTopologyAndOwnership)
{
    const auto model = ordinaryQFN();
    const auto expected = MTPTerminalHeadPolicy::MirroredFullVocabulary;
    for (const int count : {1, 2, 3, 8})
    {
        const auto [config, cluster] = ordinaryQFNLocalTP(count);
        expectOrdinaryQFNPolicy(config, model, cluster, expected);
    }
    const auto [original, observed] = ordinaryQFNLocalTP();
    auto cluster = observed;
    cluster.ranks.front().gpus.back().compute_units = 64;
    expectOrdinaryQFNPolicy(original, model, cluster, expected);
    cluster = observed;
    cluster.ranks.front().gpus.back().name = "AMD Instinct MI100";
    expectOrdinaryQFNPolicy(original, model, cluster, expected);
    for (const bool named : {false, true})
    {
        auto config = original;
        if (named)
        {
            config.domain_definitions = {domain("terminal", config.tp_devices, {0, 0, 0, 0})};
            config.tp_devices.clear();
            config.domain_definitions.front().weights = {.1f, .2f, .3f, .4f};
        }
        else config.tp_weights = {.1f, .2f, .3f, .4f};
        expectOrdinaryQFNPolicy(config, model, observed, expected);
    }
    for (const auto backend : {CollectiveBackendType::MPI, CollectiveBackendType::HOST})
        for (const bool named : {false, true})
        {
            auto config = original;
            if (named)
            {
                config.domain_definitions = {domain("terminal", config.tp_devices, {0, 0, 0, 0})};
                config.domain_definitions.front().backend = backend;
                config.tp_devices.clear();
            }
            else config.default_backend = backend;
            expectOrdinaryQFNPolicy(config, model, observed, expected);
        }
    auto config = original;
    config.domain_definitions = {domain("terminal", config.tp_devices, {0, 0, 0, 0})};
    config.tp_devices.clear();
    config.pp_stage_definitions = {{0, "terminal", 0, 23}, {1, "terminal", 24, 47}};
    expectOrdinaryQFNPolicy(config, model, observed, expected);
    config.pp_stage_definitions = {{0, "terminal", 24, 47}};
    expectOrdinaryQFNPolicy(config, model, observed, expected);
    config.pp_stage_definitions.clear();
    config.domain_definitions.front().devices.back() = config.domain_definitions.front().devices.front();
    expectOrdinaryQFNPolicy(config, model, observed, expected);
    config.domain_definitions = {domain("terminal",
        {address(DeviceType::ROCm, 0), address(DeviceType::ROCm, 0),
         address(DeviceType::ROCm, 0), address(DeviceType::ROCm, 0)}, {0, 1, 2, 3})};
    expectOrdinaryQFNPolicy(config, model,
        inventory({{card(DeviceType::ROCm, 0)}, {card(DeviceType::ROCm, 0)},
                   {card(DeviceType::ROCm, 0)}, {card(DeviceType::ROCm, 0)}}), expected);
    for (const bool disabled_tp : {false, true})
    {
        config = original;
        config.disable_gpu_tp = disabled_tp;
        config.heterogeneous_mode = !disabled_tp;
        expectOrdinaryQFNPolicy(config, model, observed, expected);
    }
    for (const auto policy : {DenseParallelPolicy::Replicated,
             DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
             DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding})
    {
        config = original;
        config.domain_definitions = {domain("terminal", config.tp_devices, {0, 0, 0, 0})};
        config.tp_devices.clear();
        config.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        config.moe_routed_expert_plan->enabled = true;
        config.moe_routed_expert_plan->continuation_domain = "terminal";
        config.moe_routed_expert_plan->continuation_domain_spec.domain = "terminal";
        config.moe_routed_expert_plan->continuation_domain_spec.setDensePolicy(policy);
        expectOrdinaryQFNPolicy(config, model, observed, expected);
    }
    for (const bool mixed : {false, true})
    {
        config = original;
        cluster = observed;
        for (int ordinal = mixed ? 3 : 0; ordinal < 4; ++ordinal)
        {
            config.tp_devices[ordinal] = address(DeviceType::CUDA, ordinal);
            cluster.ranks.front().gpus[ordinal] = card(DeviceType::CUDA, ordinal);
        }
        expectOrdinaryQFNPolicy(config, model, cluster, expected);
    }
}

/** @test An explicit physical policy remains authoritative even outside the Auto evidence cell. */
TEST(MTPHardwareDefaults, QFNOrdinaryHeadDefaultsPreserveExplicitOverrides)
{
    auto [config, cluster] = ordinaryQFNLocalTP();
    for (const auto policy : {MTPTerminalHeadPolicy::VocabularySharded,
             MTPTerminalHeadPolicy::MirroredFullVocabulary})
        for (const bool measured : {true, false})
        {
            auto model = ordinaryQFN();
            config.batch_size = measured ? 1 : 2;
            if (!measured) model.name = "other";
            config.mtp.terminal_head_policy = policy;
            const auto result = ExecutionPlanBuilder{}.buildAllPlans(config, model, cluster);
            for (const auto &plan : result) EXPECT_EQ(plan.runtime.mtp.terminal_head_policy, policy);
            EXPECT_EQ(config.mtp.terminal_head_policy, policy);
        }
}

TEST(MTPHardwareDefaults, ExplicitSingleDeviceIgnoresUnselectedCards)
{
    OrchestrationConfig config;
    config.device_for_this_rank = address(DeviceType::ROCm, 0);
    const auto result = plans(config, inventory({{card(DeviceType::CUDA, 0), card(DeviceType::ROCm, 0)}}));
    EXPECT_EQ(result.front().runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::ROCmMI50);
}

TEST(MTPHardwareDefaults, UnknownOrMixedCardDomainsRemainPortable)
{
    for (const auto other_name : {"AMD Instinct MI100", "", "AMD Instinct MI60 / MI50"})
    {
        auto other = card(DeviceType::ROCm, 1);
        other.name = other_name;
        other.compute_units = 64; // The MI60 shares the marketing name, not the measured card.
        OrchestrationConfig config;
        config.tp_devices = {address(DeviceType::ROCm, 0), address(DeviceType::ROCm, 1)};
        auto result = plans(config, inventory({{card(DeviceType::ROCm, 0), other}}));
        EXPECT_EQ(result.front().runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::Portable);
    }
    OrchestrationConfig mixed;
    mixed.tp_devices = {address(DeviceType::ROCm, 0), address(DeviceType::CUDA, 0)};
    EXPECT_EQ(plans(mixed, inventory({{card(DeviceType::ROCm, 0), card(DeviceType::CUDA, 0)}}))
        .front().runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::Portable);
    mixed.tp_devices = {address(DeviceType::ROCm, 0), GlobalDeviceAddress::cpu(0, "node")};
    EXPECT_EQ(plans(mixed, inventory({{card(DeviceType::ROCm, 0)}}))
        .front().runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::Portable);
    EXPECT_EQ(plans(OrchestrationConfig{}, inventory({{}}))
        .front().runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::Portable);
    auto ti = card(DeviceType::CUDA, 0);
    ti.name = "NVIDIA GeForce RTX 3090 Ti";
    EXPECT_EQ(plans(OrchestrationConfig{}, inventory({{ti}}))
        .front().runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::Portable);
}

TEST(MTPHardwareDefaults, ContinuationProfileDoesNotIncludeOtherExpertTiers)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        OrchestrationConfig config;
        const auto other = type == DeviceType::CUDA ? DeviceType::ROCm : DeviceType::CUDA;
        config.domain_definitions = {
            domain("cpu-tier", {GlobalDeviceAddress::cpu(0, "node")}, {0}),
            domain("other-gpu-tier", {address(other, 0)}, {1}),
            domain("continuation", {address(type, 0), address(type, 1)}, {1, 1}),
        };
        config.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        config.moe_routed_expert_plan->enabled = true;
        config.moe_routed_expert_plan->continuation_domain = "continuation";
        const auto result = plans(config, inventory({{}, {card(other, 0), card(type, 0), card(type, 1)}}));
        ASSERT_EQ(result.size(), 2u);
        for (const auto &plan : result)
        {
            EXPECT_EQ(plan.runtime.mtp.depth_defaults_profile, type == DeviceType::CUDA
                ? MTPDepthDefaultsProfile::CUDARTX3090 : MTPDepthDefaultsProfile::ROCmMI50);
            EXPECT_EQ(plan.runtime.mtp.terminal_head_policy, MTPTerminalHeadPolicy::MirroredFullVocabulary)
                << "An expert-only CPU tier cannot change GPU continuation defaults";
        }
    }
}

TEST(MTPHardwareDefaults, RepeatedLocalOrdinalsHonorDeclaredRankOwners)
{
    OrchestrationConfig config;
    config.domain_definitions = {domain("continuation",
        {address(DeviceType::ROCm, 0), address(DeviceType::ROCm, 0)}, {0, 1})};
    auto cluster = inventory({{card(DeviceType::ROCm, 0)}, {card(DeviceType::ROCm, 0)}});
    auto result = plans(config, cluster);
    ASSERT_EQ(result.size(), 2u);
    for (const auto &plan : result)
    {
        EXPECT_EQ(plan.runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::ROCmMI50);
        EXPECT_TRUE(plan.usesGlobalTP());
        EXPECT_EQ(plan.local_tp_devices.size(), 1u);
    }
    cluster.ranks[1].gpus.front().name = "AMD Instinct MI100";
    result = plans(config, cluster);
    for (const auto &plan : result)
        EXPECT_EQ(plan.runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::Portable);
    config.domain_definitions.front().explicit_ranks = {0, 2};
    EXPECT_THROW(plans(config, cluster), std::invalid_argument);
}

TEST(MTPHardwareDefaults, UnusedNamedDomainsDoNotAffectTheExecutingDomain)
{
    OrchestrationConfig config;
    config.domain_definitions = {
        domain("continuation", {address(DeviceType::ROCm, 0)}, {0}),
        domain("unused", {address(DeviceType::CUDA, 0)}, {0}),
    };
    const auto result = plans(config, inventory({{card(DeviceType::ROCm, 0), card(DeviceType::CUDA, 0)}}));
    EXPECT_EQ(result.front().runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::ROCmMI50);
}

TEST(MTPHardwareDefaults, RankLocalExplicitOwnerSelectsThatRanksCard)
{
    OrchestrationConfig config;
    auto selected = domain("continuation", {address(DeviceType::ROCm, 0)}, {});
    selected.owner_rank = 1;
    selected.scope = TPScope::RANK_LOCAL;
    config.domain_definitions = {selected};
    auto unknown = card(DeviceType::ROCm, 0);
    unknown.name = "AMD Instinct MI100";
    for (const auto &plan : plans(config, inventory({{unknown}, {card(DeviceType::ROCm, 0)}})))
        EXPECT_EQ(plan.runtime.mtp.depth_defaults_profile, MTPDepthDefaultsProfile::ROCmMI50);
}

/** @test CPU terminal shards follow complete membership, not MTP enablement or scope. */
TEST(MTPHardwareDefaults, CPUHeadDefaultsAndExplicitOverridesFollowTerminalOwnership)
{
    for (const int count : {1, 2, 3, 4, 8})
        for (const bool remote : {false, true})
            for (const bool enabled : {false, true})
                for (const auto requested : {MTPTerminalHeadPolicy::Automatic,
                         MTPTerminalHeadPolicy::VocabularySharded,
                         MTPTerminalHeadPolicy::MirroredFullVocabulary})
                {
                    SCOPED_TRACE(::testing::Message() << "count=" << count << " remote=" << remote
                        << " enabled=" << enabled << " policy=" << mtpTerminalHeadPolicyToString(requested));
                    auto cluster = inventory(std::vector<std::vector<DeviceInfo>>(count));
                    std::vector<GlobalDeviceAddress> devices;
                    std::vector<int> owners;
                    for (int rank = 0; rank < count; ++rank)
                    {
                        auto &observed = cluster.ranks[rank];
                        if (remote)
                        {
                            observed.hostname = "host-" + std::to_string(rank);
                            observed.node_id = rank;
                            observed.local_rank = 0;
                        }
                        devices.push_back(GlobalDeviceAddress::cpu(rank, observed.hostname));
                        owners.push_back(rank);
                    }
                    cluster.buildNodeAggregations();
                    OrchestrationConfig config;
                    config.mtp.enabled = enabled;
                    config.mtp.terminal_head_policy = requested;
                    config.domain_definitions = {domain("continuation", devices, owners)};
                    config.domain_definitions.front().scope = remote ? TPScope::GLOBAL : TPScope::NODE_LOCAL;
                    const auto result = plans(config, cluster);
                    ASSERT_EQ(result.size(), static_cast<size_t>(count));
                    const auto expected = requested == MTPTerminalHeadPolicy::Automatic
                        ? MTPTerminalHeadPolicy::VocabularySharded : requested;
                    for (const auto &plan : result)
                    {
                        EXPECT_EQ(plan.runtime.mtp.terminal_head_policy, expected);
                        const auto sets = resolveAdditionalPersistentWeightSets(
                            DenseParallelPolicy::TensorParallel, count, plan.runtime.mtp);
                        EXPECT_EQ(std::count(sets.begin(), sets.end(), AdditionalPersistentWeightSet::MirroredMTPTerminalHead),
                            count > 1 && expected == MTPTerminalHeadPolicy::MirroredFullVocabulary ? 1 : 0);
                        EXPECT_EQ(resolveMTPTerminalLogitsLayout(count > 1, expected),
                            count > 1 && expected == MTPTerminalHeadPolicy::VocabularySharded
                                ? MTPTerminalLogitsLayout::VocabularyShardPerParticipant
                                : MTPTerminalLogitsLayout::FullVocabularyPerParticipant);
                    }
                    EXPECT_EQ(config.mtp.terminal_head_policy, requested);
                }
}

/** @test Single CPU, GPU defaults, and explicit GPU sharding share the same compiler. */
TEST(MTPHardwareDefaults, TerminalHeadExplicitOverridesSurviveSimplePlans)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const auto requested : {MTPTerminalHeadPolicy::Automatic,
                 MTPTerminalHeadPolicy::VocabularySharded, MTPTerminalHeadPolicy::MirroredFullVocabulary})
        {
            OrchestrationConfig config;
            config.device_for_this_rank = backend == DeviceType::CPU
                ? GlobalDeviceAddress::cpu(0, "node") : address(backend, 0);
            config.mtp.terminal_head_policy = requested;
            const auto cluster = inventory({backend == DeviceType::CPU
                ? std::vector<DeviceInfo>{} : std::vector<DeviceInfo>{card(backend, 0)}});
            const auto expected = requested == MTPTerminalHeadPolicy::Automatic
                ? (backend == DeviceType::CPU ? MTPTerminalHeadPolicy::VocabularySharded
                                              : MTPTerminalHeadPolicy::MirroredFullVocabulary)
                : requested;
            EXPECT_EQ(plans(config, cluster).front().runtime.mtp.terminal_head_policy, expected);
        }
}

/** @test A pipeline uses its terminal domain, independent of its first stage's backend. */
TEST(MTPHardwareDefaults, TerminalHeadPipelineDefaultsUseTheFinalDomain)
{
    for (const auto gpu : {DeviceType::CUDA, DeviceType::ROCm})
        for (const bool cpu_last : {false, true})
        {
            OrchestrationConfig config;
            config.domain_definitions = {
                domain("cpu", {GlobalDeviceAddress::cpu(0, "node")}, {0}),
                domain("gpu", {address(gpu, 0)}, {1}),
            };
            config.pp_stage_definitions = {
                {.stage_id = 0, .domain_name = cpu_last ? "gpu" : "cpu", .first_layer = 0, .last_layer = 15},
                {.stage_id = 1, .domain_name = cpu_last ? "cpu" : "gpu", .first_layer = 16, .last_layer = 31},
            };
            for (const auto &plan : plans(config, inventory({{}, {card(gpu, 0)}})))
                EXPECT_EQ(plan.runtime.mtp.terminal_head_policy, cpu_last
                    ? MTPTerminalHeadPolicy::VocabularySharded : MTPTerminalHeadPolicy::MirroredFullVocabulary);
        }
}

/** @test CLI/YAML and saved plans preserve automatic versus explicit head intent. */
TEST(MTPHardwareDefaults, TerminalHeadPolicyRoundTripsWithoutErasingExplicitIntent)
{
    OrchestrationConfigParser parser;
    for (const auto policy : {MTPTerminalHeadPolicy::Automatic,
             MTPTerminalHeadPolicy::VocabularySharded, MTPTerminalHeadPolicy::MirroredFullVocabulary})
    {
        const char *spelling = mtpTerminalHeadPolicyToString(policy);
        const char *argv[] = {"llaminar2", "--mtp-terminal-head-policy", spelling};
        const auto config = parser.parseArgs(3, const_cast<char **>(argv));
        EXPECT_EQ(config.mtp.terminal_head_policy, policy);
        EXPECT_EQ(parser.parseYamlString(std::string("mtp:\n  terminal_head_policy: ") + spelling + "\n")
                      .mtp.terminal_head_policy, policy);
        EXPECT_EQ(deserializeOrchestrationConfig(serializeOrchestrationConfig(config))
                      .mtp.terminal_head_policy, policy);
    }
}

/** @test Uncompiled intent cannot silently choose graph layout or memory charges. */
TEST(MTPHardwareDefaults, TerminalHeadConsumersRejectUnresolvedAutomaticIntent)
{
    const auto pending = OrchestrationConfig{}.mtp;
    EXPECT_EQ(pending.terminal_head_policy, MTPTerminalHeadPolicy::Automatic);
    EXPECT_THROW(mtpTerminalHeadIsMirrored(pending.terminal_head_policy), std::logic_error);
    for (const bool sharded : {false, true})
        EXPECT_THROW(resolveMTPTerminalLogitsLayout(sharded, pending.terminal_head_policy), std::logic_error);
    EXPECT_THROW(resolveAdditionalPersistentWeightSets(DenseParallelPolicy::TensorParallel, 2, pending), std::logic_error);
}

TEST(MTPHardwareDefaults, RequestsRetainTopologyWhileExplicitThresholdsTakePrecedence)
{
    MTPRuntimeConfig retained;
    retained.enabled = true;
    retained.draft_tokens = 15;
    retained.graph_capacity_draft_tokens = 15;
    retained.depth_defaults_profile = MTPDepthDefaultsProfile::ROCmMI50;
    for (const auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Observe, MTPDepthPolicyMode::Dynamic})
    {
        for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
        {
            for (const auto threshold : {std::optional<double>{}, std::optional<double>{0.0},
                                        std::optional<double>{0.30}, std::optional<double>{1.0}})
            {
                MTPRequestPolicy request = makeMTPRequestPolicy(retained);
                request.depth_policy.mode = mode;
                request.depth_policy.demote_zero_accept_rate = threshold;
                request.verify_mode = verify;
                ASSERT_FALSE(validateMTPRequestPolicy(request, retained));
                const auto active = composeMTPRequestConfig(retained, request);
                EXPECT_EQ(active.depth_defaults_profile, retained.depth_defaults_profile);
                EXPECT_EQ(active.graph_capacity_draft_tokens, 15);
                EXPECT_EQ(active.depth_policy.demote_zero_accept_rate, threshold);
                const double expected = threshold.value_or(0.45);
                EXPECT_DOUBLE_EQ(resolveMTPZeroAcceptDemotionRate(active), expected);
                const auto device = resolveMTPDeviceGenerationDepthPolicy(active);
                EXPECT_TRUE(device.valid());
                EXPECT_EQ(device.maximum_depth, 15);
                EXPECT_EQ(device.demote_zero_accept_rate_ppm, static_cast<int>(std::llround(expected * 1'000'000)));
                EXPECT_EQ(resolveMTPDepthPolicyConfig(active).demote_zero_accept_rate, expected);
            }
        }
    }
    EXPECT_FALSE(retained.depth_policy.demote_zero_accept_rate);
}

TEST(MTPHardwareDefaults, ParserPreservesAutoAndExplicitOldDefault)
{
    OrchestrationConfigParser parser;
    auto automatic = parser.parseYamlString("mtp:\n  depth_demote_zero_accept: auto\n");
    EXPECT_FALSE(automatic.mtp.depth_policy.demote_zero_accept_rate);
    EXPECT_NE(automatic.toString().find("depth_demote_zero_accept: auto"), std::string::npos);
    const auto explicit_rate = parser.parseYamlString("mtp:\n  depth_demote_zero_accept: 0.30\n");
    EXPECT_EQ(explicit_rate.mtp.depth_policy.demote_zero_accept_rate, 0.30);
    automatic.mtp.depth_defaults_profile = MTPDepthDefaultsProfile::ROCmMI50;
    EXPECT_DOUBLE_EQ(resolveMTPZeroAcceptDemotionRate(automatic.mtp), 0.45);
    const char *argv[] = {"llaminar2", "--mtp-depth-demote-zero-accept", "0.30",
                         "--mtp-depth-demote-zero-accept", "auto"};
    EXPECT_FALSE(parser.parseArgs(5, const_cast<char **>(argv)).mtp.depth_policy.demote_zero_accept_rate);
    EXPECT_EQ(parser.parseArgs(3, const_cast<char **>(argv)).mtp.depth_policy.demote_zero_accept_rate, 0.30);
    for (const auto invalid : {"nan", "inf", "-0.01", "1.01", "0.3garbage"})
        EXPECT_THROW((void)parseMTPZeroAcceptDemotionRate(invalid), std::invalid_argument);
}

/** @test Hardware profiles never narrow the default adaptive depth range. */
TEST(MTPHardwareDefaults, EveryProfileDefaultsToFullOneThroughFifteenRange)
{
    EXPECT_EQ(defaultMTPAdaptiveMaximumDraftDepth(), 15);
    EXPECT_EQ(
        sampling_math::DeviceGenerationPolicy::kMaximumSupportedDraftDepth,
        defaultMTPAdaptiveMaximumDraftDepth());

    for (const auto profile : {
             MTPDepthDefaultsProfile::Portable,
             MTPDepthDefaultsProfile::CUDARTX3090,
             MTPDepthDefaultsProfile::ROCmMI50})
    {
        MTPRuntimeConfig config;
        config.enabled = true;
        config.depth_defaults_profile = profile;
        config.depth_policy.mode = MTPDepthPolicyMode::Dynamic;

        const auto policy = resolveMTPDeviceGenerationDepthPolicy(config);
        EXPECT_TRUE(policy.valid());
        EXPECT_EQ(policy.minimum_depth, 1);
        EXPECT_EQ(policy.maximum_depth, 15);
    }
}
