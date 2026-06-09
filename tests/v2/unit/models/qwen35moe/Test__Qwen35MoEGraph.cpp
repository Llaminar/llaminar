/**
 * @file Test__Qwen35MoEGraph.cpp
 * @brief Regression tests for Qwen3.5 MoE graph construction.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/prefix_cache/PrefixCacheFingerprint.h"
#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "kernels/KernelFactory.h"
#include "mocks/MockLocalTPContext.h"
#include "mocks/MockMPIContext.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <memory>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    bool hasDependency(const ComputeGraph &graph, const std::string &node_name, const std::string &dependency)
    {
        const auto *node = graph.getNode(node_name);
        if (!node)
            return false;
        return std::find(node->dependencies.begin(), node->dependencies.end(), dependency) != node->dependencies.end();
    }

    bool hasFingerprintField(
        const PrefixFingerprintMaterial &material,
        const std::string &name,
        const std::string &value)
    {
        return std::any_of(material.moe.begin(),
                           material.moe.end(),
                           [&](const PrefixFingerprintField &field)
                           {
                               return field.name == name && field.value == value;
                           });
    }

    bool hasFingerprintFieldWithPrefix(
        const PrefixFingerprintMaterial &material,
        const std::string &prefix)
    {
        return std::any_of(material.moe.begin(),
                           material.moe.end(),
                           [&](const PrefixFingerprintField &field)
                           {
                               return field.name.rfind(prefix, 0) == 0;
                           });
    }

    GraphConfig makeMoEConfig(ITPContext *tp_ctx = nullptr)
    {
        GraphConfig config;
        config.n_layers = 2;
        config.total_n_layers = 2;
        config.d_model = 4;
        config.n_heads = 2;
        config.n_kv_heads = 2;
        config.head_dim = 2;
        config.d_ff = 8;
        config.vocab_size = 16;
        config.rms_norm_eps = 1e-6f;
        config.default_device = DeviceId::cpu();
        config.tp_ctx = tp_ctx;
        config.tp_device_idx = 0;
        config.moe.num_experts = 2;
        config.moe.top_k = 1;
        config.moe.intermediate_size = 3;
        config.moe.has_shared_expert = true;
        config.moe.shared_intermediate_size = 3;
        return config;
    }

    ExecutionDomainDefinition denseDomain(
        std::string name,
        ExecutionDomainScope scope,
        CollectiveBackendType backend,
        std::vector<GlobalDeviceAddress> participants)
    {
        ExecutionDomainDefinition domain;
        domain.name = std::move(name);
        domain.scope = scope;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.compute_kind = ExecutionDomainComputeKind::REPLICATED_EXPERTS;
        domain.owner_rank = 0;
        domain.ranks = {0};
        return domain;
    }

    ExpertComputeDomain expertDomain(
        std::string name,
        ExpertDomainKind kind,
        CollectiveBackendType backend,
        ExpertDomainComputeKind compute_kind,
        std::vector<GlobalDeviceAddress> participants,
        std::vector<int> ranks)
    {
        ExpertComputeDomain domain;
        domain.name = std::move(name);
        domain.kind = kind;
        domain.backend = backend;
        domain.compute_kind = compute_kind;
        domain.participants = std::move(participants);
        domain.world_ranks = std::move(ranks);
        domain.owner_rank = 0;
        return domain;
    }

    std::shared_ptr<MoEExpertParallelPlan> makeOverlayPlan(const std::string &routed_domain)
    {
        auto plan = std::make_shared<MoEExpertParallelPlan>();
        plan->enabled = true;
        plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan->continuation_domain = "continuation";
        plan->base_model_domain = "base";
        plan->shared_expert_domain = "shared";
        plan->residency_policy = ExpertResidencyPolicy::RoutedTierRebalanced;
        plan->continuation_domain_spec.domain = "continuation";
        plan->continuation_domain_spec.logical_root_participant = 0;
        plan->continuation_domain_spec.dense_tp_enabled = false;
        plan->continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
        plan->continuation_domain_spec.shared_expert_uses_dense_tp = true;

        plan->dense_domains.push_back(denseDomain(
            "continuation",
            ExecutionDomainScope::SINGLE,
            CollectiveBackendType::HOST,
            {GlobalDeviceAddress::cpu(0, "node0")}));
        plan->dense_domains.push_back(denseDomain(
            "base",
            ExecutionDomainScope::SINGLE,
            CollectiveBackendType::HOST,
            {GlobalDeviceAddress::cpu(0, "node0")}));
        plan->dense_domains.push_back(denseDomain(
            "shared",
            ExecutionDomainScope::SINGLE,
            CollectiveBackendType::HOST,
            {GlobalDeviceAddress::cpu(0, "node0")}));

        plan->domains.push_back(expertDomain(
            routed_domain,
            ExpertDomainKind::NodeLocalTP,
            CollectiveBackendType::HOST,
            ExpertDomainComputeKind::ExpertIdSharded,
            {GlobalDeviceAddress::cpu(0, "node0"), GlobalDeviceAddress::cpu(1, "node0")},
            {0, 1}));

        plan->routed_tiers.push_back(ExpertRoutedTier{
            .name = "cold",
            .domain = routed_domain,
            .priority = 10,
            .max_experts_per_layer = 2,
            .memory_budget_bytes = 4096,
            .fallback = true,
        });
        plan->placements.push_back(ExpertLayerPlacement{
            .layer = 0,
            .routed_expert_tier = {0, 0},
        });
        return plan;
    }

    class TensorArena
    {
    public:
        FP32Tensor *fp32(std::vector<size_t> shape)
        {
            auto tensor = std::make_shared<FP32Tensor>(std::move(shape));
            auto *ptr = tensor.get();
            tensors_.push_back(std::move(tensor));
            return ptr;
        }

    private:
        std::vector<std::shared_ptr<TensorBase>> tensors_;
    };

    LayerWeights makeMoELayerWeights(TensorArena &arena)
    {
        LayerWeights layer;
        layer.ffn_norm = arena.fp32({4});
        layer.moe_gate = arena.fp32({2, 4});

        // Expert tensor shapes follow GGUF order: [cols, rows, experts].
        layer.moe_gate_exps = arena.fp32({4, 3, 2});
        layer.moe_up_exps = arena.fp32({4, 3, 2});
        layer.moe_down_exps = arena.fp32({3, 4, 2});

        layer.shared_expert_gate = arena.fp32({3, 4});
        layer.shared_expert_up = arena.fp32({3, 4});
        layer.shared_expert_down = arena.fp32({4, 3});
        return layer;
    }

    LayerWeights makeFALayerWeights(TensorArena &arena, const GraphConfig &config)
    {
        LayerWeights layer;
        const int q_width = config.n_heads * config.head_dim * 2;
        const int kv_width = config.n_kv_heads * config.head_dim;

        layer.attn_norm = arena.fp32({static_cast<size_t>(config.d_model)});
        layer.wq = arena.fp32({static_cast<size_t>(q_width), static_cast<size_t>(config.d_model)});
        layer.wk = arena.fp32({static_cast<size_t>(kv_width), static_cast<size_t>(config.d_model)});
        layer.wv = arena.fp32({static_cast<size_t>(kv_width), static_cast<size_t>(config.d_model)});
        layer.wo = arena.fp32({static_cast<size_t>(config.d_model), static_cast<size_t>(config.d_model)});
        layer.q_norm = arena.fp32({static_cast<size_t>(config.head_dim)});
        layer.k_norm = arena.fp32({static_cast<size_t>(config.head_dim)});
        return layer;
    }

    ActivationBuffers makeActivationBuffers(TensorArena &arena, int tokens, int d_model, int num_experts, int top_k)
    {
        ActivationBuffers buffers;
        buffers.attn_proj = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});
        buffers.current_hidden = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});
        buffers.normalized = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});

        buffers.extensions[BufferId::MOE_EXPERT_INDICES] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(top_k)});
        buffers.extensions[BufferId::MOE_EXPERT_WEIGHTS] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(top_k)});
        buffers.extensions[BufferId::MOE_COMBINED_OUTPUT] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});
        buffers.extensions[BufferId::MOE_SHARED_EXPERT_OUTPUT] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(d_model)});
        buffers.extensions[BufferId::MOE_GATE_SCRATCH] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(num_experts)});
        buffers.extensions[BufferId::MOE_UP_SCRATCH] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(num_experts)});
        return buffers;
    }

    ActivationBuffers makeFAActivationBuffers(TensorArena &arena, int tokens, const GraphConfig &config)
    {
        ActivationBuffers buffers;
        const int q_width = config.n_heads * config.head_dim;
        const int q_raw_width = q_width * 2;
        const int kv_width = config.n_kv_heads * config.head_dim;

        buffers.current_hidden = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.normalized = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.Q = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(q_width)});
        buffers.K = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(kv_width)});
        buffers.V = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(kv_width)});
        buffers.attn_output = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(q_width)});
        buffers.attn_proj = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(config.d_model)});
        buffers.extensions[BufferId::FA_Q_RAW] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(q_raw_width)});
        buffers.extensions[BufferId::FA_GATE] = arena.fp32({static_cast<size_t>(tokens), static_cast<size_t>(q_width)});
        return buffers;
    }
}

TEST(Test__Qwen35MoEGraph, ReplicatedRoutedExpertOutputFeedsCombineDirectlyUnderTP)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.moe.expert_mode = MoEExpertMode::Replicated;
    config.moe.local_expert_count = -1;
    Qwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    auto layer = makeMoELayerWeights(arena);
    auto buffers = makeActivationBuffers(arena, /*tokens=*/2, config.d_model, config.moe.num_experts, config.moe.top_k);

    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, /*layer_idx=*/0, /*seq_len=*/2, /*batch_size=*/1, DeviceId::cpu());

    ASSERT_NE(graph.getNode("layer0_moe_expert_ffn"), nullptr);
    ASSERT_EQ(graph.getNode("layer0_moe_expert_allreduce"), nullptr)
        << "Replicated MoE expert weights already produce a full routed expert output per rank";
    ASSERT_NE(graph.getNode("layer0_moe_combine"), nullptr);

    EXPECT_TRUE(hasDependency(graph, "layer0_moe_combine", "layer0_moe_expert_ffn"));
}

TEST(Test__Qwen35MoEGraph, ExpertParallelRoutedExpertOutputAllreducesUnderTP)
{
    auto tp_ctx = std::make_unique<MockLocalTPContext>();
    tp_ctx->setDevices({GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()});
    tp_ctx->setBackend(CollectiveBackendType::HOST);

    GraphConfig config = makeMoEConfig(tp_ctx.get());
    config.moe.expert_mode = MoEExpertMode::ExpertParallel;
    config.moe.local_expert_start = 0;
    config.moe.local_expert_count = 1;
    Qwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    auto layer = makeMoELayerWeights(arena);
    auto buffers = makeActivationBuffers(arena, /*tokens=*/2, config.d_model, config.moe.num_experts, config.moe.top_k);

    ComputeGraph graph = graph_builder.buildFFNGraph(layer, buffers, /*layer_idx=*/0, /*seq_len=*/2, /*batch_size=*/1, DeviceId::cpu());

    ASSERT_NE(graph.getNode("layer0_moe_expert_ffn"), nullptr);
    ASSERT_NE(graph.getNode("layer0_moe_expert_allreduce"), nullptr)
        << "Expert-parallel MoE owns only a local expert range, so routed output is partial until allreduce";
    ASSERT_NE(graph.getNode("layer0_moe_combine"), nullptr);

    EXPECT_TRUE(hasDependency(graph, "layer0_moe_expert_allreduce", "layer0_moe_expert_ffn"));
    EXPECT_TRUE(hasDependency(graph, "layer0_moe_combine", "layer0_moe_expert_allreduce"));
}

TEST(Test__Qwen35MoEGraph, CPUAllPositionMoEVerifierUsesDecodeEquivalentExpertPath)
{
    GraphConfig config = makeMoEConfig();
    config.compute_all_position_logits = true;
    Qwen35MoEGraph graph_builder(config, nullptr);

    TensorArena arena;
    auto layer = makeMoELayerWeights(arena);
    auto buffers = makeActivationBuffers(arena, /*tokens=*/2, config.d_model,
                                         config.moe.num_experts, config.moe.top_k);

    ComputeGraph graph = graph_builder.buildFFNGraph(
        layer, buffers, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, DeviceId::cpu());

    auto *node = graph.getNode("layer0_moe_expert_ffn");
    ASSERT_NE(node, nullptr);
    auto *stage = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
    ASSERT_NE(stage, nullptr);
    EXPECT_TRUE(stage->usesCPUDecodeEquivalentVerifierPrefillForTesting());

    auto *shared_node = graph.getNode("layer0_shared_expert_ffn");
    ASSERT_NE(shared_node, nullptr);
    auto *shared_stage = dynamic_cast<SharedExpertFFNStage *>(shared_node->stage.get());
    ASSERT_NE(shared_stage, nullptr);
    EXPECT_TRUE(shared_stage->usesCPUDecodeEquivalentVerifierPrefillForTesting());
}

TEST(Test__Qwen35MoEGraph, SchemaDefaultsRoutedExpertWeightsToExpertParallel)
{
    Qwen35MoESchemaFactory factory;
    WeightShardingConfig sharding = factory.getWeightShardingConfig();

    EXPECT_EQ(sharding.getMode("blk.0.ffn_gate_exps.weight"), WeightShardingMode::ExpertParallel);
    EXPECT_EQ(sharding.getMode("blk.0.ffn_up_exps.weight"), WeightShardingMode::ExpertParallel);
    EXPECT_EQ(sharding.getMode("blk.0.ffn_down_exps.weight"), WeightShardingMode::ExpertParallel);
}

TEST(Test__Qwen35MoEGraph, FARopeOnReadAppendsNormalizedKToCache)
{
    GraphConfig config = makeMoEConfig();
    config.layer_types = {"full_attention", "gdn"};
    config.partial_rotary_factor = 0.5f;
    config.rope_on_read = true;

    Qwen35MoEGraph graph_builder(config, nullptr);
    TensorArena arena;
    auto layer = makeFALayerWeights(arena, config);
    auto buffers = makeFAActivationBuffers(arena, /*tokens=*/2, config);

    MockMPIContext mpi_ctx;
    llaminar::v2::kernels::KVCacheConfig kv_config;
    kv_config.precision = ActivationPrecision::FP16;
    kv_config.device = DeviceId::cpu();
    kv_config.num_layers = 1;
    kv_config.batch_size = 1;
    kv_config.max_seq_len = 8;
    kv_config.n_kv_heads = config.n_kv_heads;
    kv_config.head_dim = config.head_dim;
    kv_config.mpi_ctx = &mpi_ctx;
    auto kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);
    ASSERT_NE(kv_cache, nullptr);

    ComputeGraph graph = graph_builder.buildAttentionGraph(
        layer, buffers, /*layer_idx=*/0, /*seq_len=*/2, /*batch_size=*/1,
        kv_cache.get(), /*position_ids=*/nullptr, DeviceId::cpu());

    ASSERT_NE(graph.getNode("layer0_kv_append"), nullptr);
    ASSERT_NE(graph.getNode("layer0_rope"), nullptr);
    ASSERT_NE(graph.getNode("layer0_k_norm"), nullptr);

    EXPECT_TRUE(hasDependency(graph, "layer0_rope", "layer0_k_norm"));
    EXPECT_TRUE(hasDependency(graph, "layer0_kv_append", "layer0_rope"))
        << "rope_on_read stores pre-RoPE K, but it must still wait for K norm";
}

TEST(Test__Qwen35MoEGraph, PrefixFingerprintMaterialIncludesExpertOverlayTopology)
{
    GraphConfig config = makeMoEConfig();
    config.moe.expert_parallel_plan = makeOverlayPlan("cold_cpu");
    config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
        config.moe.expert_parallel_plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 0,
            .validate_mvp_root_reachability = false,
        });

    Qwen35MoEGraph graph_builder(config, nullptr);
    PrefixFingerprintMaterial material;
    graph_builder.appendPrefixCacheFingerprintMaterial(material);

    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.enabled", "true"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.continuation_domain", "continuation"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.shared_expert_domain", "shared"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.routed_tier.0.domain", "cold_cpu"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.plan.expert_domain.0.participant.1",
                                    GlobalDeviceAddress::cpu(1, "node0").toString()));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.runtime.enabled", "true"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.runtime.domain.3.name", "cold_cpu"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.runtime.domain.3.participant.1.world_rank", "1"));
    EXPECT_TRUE(hasFingerprintField(material, "expert_overlay.runtime.routed_tier.0.domain_name", "cold_cpu"));

    const uint64_t original_hash = hashPrefixFingerprintFields("moe", material.moe);

    GraphConfig changed_config = makeMoEConfig();
    changed_config.moe.expert_parallel_plan = makeOverlayPlan("warm_rocm");
    changed_config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
        changed_config.moe.expert_parallel_plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 0,
            .validate_mvp_root_reachability = false,
        });

    Qwen35MoEGraph changed_graph_builder(changed_config, nullptr);
    PrefixFingerprintMaterial changed_material;
    changed_graph_builder.appendPrefixCacheFingerprintMaterial(changed_material);
    const uint64_t changed_hash = hashPrefixFingerprintFields("moe", changed_material.moe);

    EXPECT_NE(changed_hash, original_hash)
        << "Changing routed expert overlay domains must invalidate MoE prefix-cache payloads";
}

TEST(Test__Qwen35MoEGraph, PrefixFingerprintMaterialExcludesTransientRuntimeTables)
{
    GraphConfig config = makeMoEConfig();
    Qwen35MoEGraph graph_builder(config, nullptr);

    PrefixFingerprintMaterial material;
    graph_builder.appendPrefixCacheFingerprintMaterial(material);

    EXPECT_FALSE(hasFingerprintFieldWithPrefix(material, "graph.runtime_table"))
        << "MoE runtime tables are lazy graph-execution state and must not drift prefix keys";
    EXPECT_FALSE(hasFingerprintFieldWithPrefix(material, "runtime_table."))
        << "Prefix compatibility is represented by overlay/rebalance placement material instead";
    EXPECT_TRUE(hasFingerprintField(material, "graph.num_experts", "2"));
    EXPECT_TRUE(hasFingerprintField(material, "graph.top_k", "1"));
}
