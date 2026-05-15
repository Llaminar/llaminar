/**
 * @file Test__Qwen35MoEGraph.cpp
 * @brief Regression tests for Qwen3.5 MoE graph construction.
 */

#include <gtest/gtest.h>

#include "models/qwen35moe/Qwen35MoEGraph.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "mocks/MockLocalTPContext.h"
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

TEST(Test__Qwen35MoEGraph, SchemaDefaultsRoutedExpertWeightsToExpertParallel)
{
    Qwen35MoESchemaFactory factory;
    WeightShardingConfig sharding = factory.getWeightShardingConfig();

    EXPECT_EQ(sharding.getMode("blk.0.ffn_gate_exps.weight"), WeightShardingMode::ExpertParallel);
    EXPECT_EQ(sharding.getMode("blk.0.ffn_up_exps.weight"), WeightShardingMode::ExpertParallel);
    EXPECT_EQ(sharding.getMode("blk.0.ffn_down_exps.weight"), WeightShardingMode::ExpertParallel);
}
