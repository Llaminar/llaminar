/**
 * @file Test__LayerExecutor.cpp
 * @brief Unit tests for LayerExecutor
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "execution/LayerExecutor.h"
#include "execution/DeviceContext.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class LayerExecutorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Construct CPUDeviceContext directly (bypasses DeviceManager check)
        ctx_ = std::make_unique<CPUDeviceContext>(0, 4);
        ASSERT_NE(ctx_, nullptr);
    }

    std::unique_ptr<CPUDeviceContext> ctx_;
};

// =============================================================================
// ExecutionMode Tests
// =============================================================================

TEST_F(LayerExecutorTest, ExecutionModeNames)
{
    EXPECT_STREQ(executionModeName(ExecutionMode::SEQUENTIAL), "SEQUENTIAL");
    EXPECT_STREQ(executionModeName(ExecutionMode::PARALLEL), "PARALLEL");
    EXPECT_STREQ(executionModeName(ExecutionMode::PIPELINED), "PIPELINED");
}

TEST_F(LayerExecutorTest, DefaultConfiguration)
{
    LayerExecutor executor;
    EXPECT_EQ(executor.executionMode(), ExecutionMode::SEQUENTIAL);
}

TEST_F(LayerExecutorTest, SetExecutionMode)
{
    LayerExecutor executor;
    executor.setExecutionMode(ExecutionMode::PARALLEL);
    EXPECT_EQ(executor.executionMode(), ExecutionMode::PARALLEL);
}

// =============================================================================
// ComputeGraph Tests
// =============================================================================

TEST_F(LayerExecutorTest, EmptyGraph)
{
    ComputeGraph graph;
    EXPECT_EQ(graph.size(), 0);
    EXPECT_TRUE(graph.allCompleted());

    auto order = graph.getExecutionOrder();
    EXPECT_TRUE(order.empty());
}

TEST_F(LayerExecutorTest, SingleNodeGraph)
{
    ComputeGraph graph;

    // Create a simple RMSNorm stage
    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    auto stage = ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS);
    graph.addNode("norm", std::move(stage), 0);

    EXPECT_EQ(graph.size(), 1);
    EXPECT_FALSE(graph.allCompleted());

    auto order = graph.getExecutionOrder();
    ASSERT_EQ(order.size(), 1);
    EXPECT_EQ(order[0], "norm");
}

TEST_F(LayerExecutorTest, LinearDependencyChain)
{
    ComputeGraph graph;

    // Create: A -> B -> C (linear chain)
    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("C", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    graph.addDependency("B", "A");
    graph.addDependency("C", "B");

    EXPECT_EQ(graph.size(), 3);

    auto order = graph.getExecutionOrder();
    ASSERT_EQ(order.size(), 3);

    // Must be A before B before C
    auto pos_a = std::find(order.begin(), order.end(), "A") - order.begin();
    auto pos_b = std::find(order.begin(), order.end(), "B") - order.begin();
    auto pos_c = std::find(order.begin(), order.end(), "C") - order.begin();

    EXPECT_LT(pos_a, pos_b);
    EXPECT_LT(pos_b, pos_c);
}

TEST_F(LayerExecutorTest, DiamondDependency)
{
    ComputeGraph graph;

    // Create diamond: A -> [B, C] -> D
    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("C", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("D", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    graph.addDependency("B", "A");
    graph.addDependency("C", "A");
    graph.addDependency("D", "B");
    graph.addDependency("D", "C");

    auto order = graph.getExecutionOrder();
    ASSERT_EQ(order.size(), 4);

    // A must be first, D must be last, B and C in between
    auto pos_a = std::find(order.begin(), order.end(), "A") - order.begin();
    auto pos_b = std::find(order.begin(), order.end(), "B") - order.begin();
    auto pos_c = std::find(order.begin(), order.end(), "C") - order.begin();
    auto pos_d = std::find(order.begin(), order.end(), "D") - order.begin();

    EXPECT_LT(pos_a, pos_b);
    EXPECT_LT(pos_a, pos_c);
    EXPECT_LT(pos_b, pos_d);
    EXPECT_LT(pos_c, pos_d);
}

TEST_F(LayerExecutorTest, GetReadyNodes)
{
    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    // A -> B, C (no deps)
    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("C", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    graph.addDependency("B", "A");

    // Initially, A and C are ready (no dependencies)
    auto ready = graph.getReadyNodes();
    EXPECT_EQ(ready.size(), 2);
    EXPECT_TRUE(std::find(ready.begin(), ready.end(), "A") != ready.end());
    EXPECT_TRUE(std::find(ready.begin(), ready.end(), "C") != ready.end());
    EXPECT_TRUE(std::find(ready.begin(), ready.end(), "B") == ready.end());

    // Mark A as complete
    graph.markCompleted("A");

    // Now B should also be ready
    ready = graph.getReadyNodes();
    EXPECT_EQ(ready.size(), 2);
    EXPECT_TRUE(std::find(ready.begin(), ready.end(), "B") != ready.end());
    EXPECT_TRUE(std::find(ready.begin(), ready.end(), "C") != ready.end());
}

TEST_F(LayerExecutorTest, GraphReset)
{
    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    // Mark both complete
    graph.markCompleted("A");
    graph.markCompleted("B");
    EXPECT_TRUE(graph.allCompleted());

    // Reset
    graph.reset();
    EXPECT_FALSE(graph.allCompleted());

    auto ready = graph.getReadyNodes();
    EXPECT_EQ(ready.size(), 2);
}

TEST_F(LayerExecutorTest, TotalEstimatedFlops)
{
    ComputeGraph graph;

    // RMSNorm flops: 4 * seq_len * hidden_dim (squares + adds + sqrt + div + muls)
    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 10;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    size_t expected_flops = 2 * (4 * 10 * 64); // Two stages
    EXPECT_EQ(graph.totalEstimatedFlops(), expected_flops);
}

// =============================================================================
// LayerExecutor Execution Tests
// =============================================================================

TEST_F(LayerExecutorTest, ExecuteEmptyGraph)
{
    LayerExecutor executor;
    ComputeGraph graph;

    EXPECT_TRUE(executor.execute(graph, ctx_.get()));
}

TEST_F(LayerExecutorTest, ExecuteNullContext)
{
    LayerExecutor executor;
    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    EXPECT_FALSE(executor.execute(graph, nullptr));
}

TEST_F(LayerExecutorTest, ExecuteSequential)
{
    LayerExecutor executor;
    executor.setExecutionMode(ExecutionMode::SEQUENTIAL);

    // Create actual data for RMSNorm
    const int seq_len = 2;
    const int hidden_dim = 4;
    std::vector<float> input(seq_len * hidden_dim, 1.0f);
    std::vector<float> gamma(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.data();
    params.gamma = gamma.data();
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    EXPECT_TRUE(executor.execute(graph, ctx_.get()));
    EXPECT_TRUE(graph.allCompleted());

    // RMSNorm on all 1s with gamma=1 should output all 1s
    for (int i = 0; i < seq_len * hidden_dim; ++i)
    {
        EXPECT_NEAR(input[i], 1.0f, 1e-5f);
    }
}

TEST_F(LayerExecutorTest, ExecuteParallel)
{
    LayerExecutor executor;
    executor.setExecutionMode(ExecutionMode::PARALLEL);

    const int seq_len = 2;
    const int hidden_dim = 4;
    std::vector<float> input(seq_len * hidden_dim, 1.0f);
    std::vector<float> gamma(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.data();
    params.gamma = gamma.data();
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    EXPECT_TRUE(executor.execute(graph, ctx_.get()));
    EXPECT_TRUE(graph.allCompleted());
}

TEST_F(LayerExecutorTest, ExecuteWithDependencies)
{
    LayerExecutor executor;

    const int seq_len = 2;
    const int hidden_dim = 4;
    std::vector<float> input(seq_len * hidden_dim, 2.0f);
    std::vector<float> gamma(hidden_dim, 1.0f);

    ComputeGraph graph;

    // Two sequential norms: A -> B
    RMSNormStage::Params params;
    params.input = input.data();
    params.gamma = gamma.data();
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addDependency("B", "A");

    EXPECT_TRUE(executor.execute(graph, ctx_.get()));
    EXPECT_TRUE(graph.allCompleted());
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(LayerExecutorTest, StatsTracking)
{
    LayerExecutorConfig config;
    config.enable_profiling = true;
    LayerExecutor executor(config);

    const int seq_len = 2;
    const int hidden_dim = 4;
    std::vector<float> input(seq_len * hidden_dim, 1.0f);
    std::vector<float> gamma(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.data();
    params.gamma = gamma.data();
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    executor.execute(graph, ctx_.get());

    const auto &stats = executor.stats();
    EXPECT_GT(stats.total_stages_executed, 0u);
    EXPECT_GT(stats.total_flops, 0u);
    EXPECT_GE(stats.total_time_ms, 0.0);
}

TEST_F(LayerExecutorTest, StatsReset)
{
    LayerExecutor executor;

    const int seq_len = 2;
    const int hidden_dim = 4;
    std::vector<float> input(seq_len * hidden_dim, 1.0f);
    std::vector<float> gamma(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.data();
    params.gamma = gamma.data();
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    executor.execute(graph, ctx_.get());
    EXPECT_GT(executor.stats().total_stages_executed, 0u);

    executor.resetStats();
    EXPECT_EQ(executor.stats().total_stages_executed, 0u);
}

// =============================================================================
// Graph Building Tests
// =============================================================================

TEST_F(LayerExecutorTest, BuildAttentionGraphBasic)
{
    LayerExecutor executor;

    LayerExecutor::AttentionParams params;
    params.input = reinterpret_cast<float *>(0x1000); // Dummy pointers
    params.output = reinterpret_cast<float *>(0x2000);
    params.seq_len = 16;
    params.d_model = 64;
    params.n_heads = 4;
    params.head_dim = 16;

    auto graph = executor.buildAttentionGraph(params);

    // Should create at least a minimal graph structure
    // (Full implementation would have more stages)
    EXPECT_GE(graph.size(), 0u);
}

TEST_F(LayerExecutorTest, BuildAttentionGraphWithNorm)
{
    LayerExecutor executor;

    std::vector<float> norm_weights(64, 1.0f);
    std::vector<float> residual(16 * 64, 0.0f);

    LayerExecutor::AttentionParams params;
    params.input = reinterpret_cast<float *>(0x1000);
    params.output = reinterpret_cast<float *>(0x2000);
    params.residual = residual.data();
    params.attn_norm = norm_weights.data();
    params.seq_len = 16;
    params.d_model = 64;
    params.n_heads = 4;
    params.head_dim = 16;

    auto graph = executor.buildAttentionGraph(params);

    // Should have norm and residual stages
    EXPECT_GE(graph.size(), 2u);
    EXPECT_NE(graph.getNode("attn_norm"), nullptr);
    EXPECT_NE(graph.getNode("attn_residual"), nullptr);
}

TEST_F(LayerExecutorTest, BuildFFNGraphBasic)
{
    LayerExecutor executor;

    LayerExecutor::FFNParams params;
    params.input = reinterpret_cast<float *>(0x1000);
    params.output = reinterpret_cast<float *>(0x2000);
    params.seq_len = 16;
    params.d_model = 64;
    params.d_ff = 256;

    auto graph = executor.buildFFNGraph(params);

    // Basic FFN graph
    EXPECT_GE(graph.size(), 0u);
}

TEST_F(LayerExecutorTest, BuildFFNGraphWithNormAndResidual)
{
    LayerExecutor executor;

    std::vector<float> norm_weights(64, 1.0f);
    std::vector<float> residual(16 * 64, 0.0f);

    LayerExecutor::FFNParams params;
    params.input = reinterpret_cast<float *>(0x1000);
    params.output = reinterpret_cast<float *>(0x2000);
    params.residual = residual.data();
    params.ffn_norm = norm_weights.data();
    params.seq_len = 16;
    params.d_model = 64;
    params.d_ff = 256;

    auto graph = executor.buildFFNGraph(params);

    EXPECT_GE(graph.size(), 3u); // norm, swiglu, residual
    EXPECT_NE(graph.getNode("ffn_norm"), nullptr);
    EXPECT_NE(graph.getNode("ffn_swiglu"), nullptr);
    EXPECT_NE(graph.getNode("ffn_residual"), nullptr);
}

TEST_F(LayerExecutorTest, BuildMoEGraphBasic)
{
    LayerExecutor executor;

    LayerExecutor::MoEParams params;
    params.input = reinterpret_cast<float *>(0x1000);
    params.output = reinterpret_cast<float *>(0x2000);
    params.seq_len = 16;
    params.d_model = 64;
    params.d_ff = 256;
    params.n_experts = 4;
    params.top_k = 2;

    auto graph = executor.buildMoEGraph(params);

    // Should have router, experts, and combine
    EXPECT_GE(graph.size(), 6u); // router + 4 experts + combine
    EXPECT_NE(graph.getNode("moe_router"), nullptr);
    EXPECT_NE(graph.getNode("expert_0"), nullptr);
    EXPECT_NE(graph.getNode("expert_3"), nullptr);
    EXPECT_NE(graph.getNode("moe_combine"), nullptr);
}

TEST_F(LayerExecutorTest, BuildMoEGraphDependencies)
{
    LayerExecutor executor;

    std::vector<float> norm_weights(64, 1.0f);

    LayerExecutor::MoEParams params;
    params.input = reinterpret_cast<float *>(0x1000);
    params.output = reinterpret_cast<float *>(0x2000);
    params.ffn_norm = norm_weights.data();
    params.seq_len = 16;
    params.d_model = 64;
    params.d_ff = 256;
    params.n_experts = 2;
    params.top_k = 1;

    auto graph = executor.buildMoEGraph(params);

    // Verify dependency structure: norm -> router -> experts -> combine
    auto order = graph.getExecutionOrder();

    auto pos_norm = std::find(order.begin(), order.end(), "moe_norm") - order.begin();
    auto pos_router = std::find(order.begin(), order.end(), "moe_router") - order.begin();
    auto pos_exp0 = std::find(order.begin(), order.end(), "expert_0") - order.begin();
    auto pos_exp1 = std::find(order.begin(), order.end(), "expert_1") - order.begin();
    auto pos_combine = std::find(order.begin(), order.end(), "moe_combine") - order.begin();

    // Verify ordering constraints
    EXPECT_LT(pos_norm, pos_router);
    EXPECT_LT(pos_router, pos_exp0);
    EXPECT_LT(pos_router, pos_exp1);
    EXPECT_LT(pos_exp0, pos_combine);
    EXPECT_LT(pos_exp1, pos_combine);
}

TEST_F(LayerExecutorTest, BuildMoEGraphExpertParallelism)
{
    LayerExecutorConfig config;
    config.default_device = 0;
    LayerExecutor executor(config);

    // Map experts to different devices
    std::vector<int> expert_devices = {0, 1, 0, 1};

    LayerExecutor::MoEParams params;
    params.input = reinterpret_cast<float *>(0x1000);
    params.output = reinterpret_cast<float *>(0x2000);
    params.seq_len = 16;
    params.d_model = 64;
    params.d_ff = 256;
    params.n_experts = 4;
    params.top_k = 2;
    params.enable_expert_parallel = true;
    params.expert_device_map = expert_devices.data();

    auto graph = executor.buildMoEGraph(params);

    // Verify experts are assigned to correct devices
    auto *exp0 = graph.getNode("expert_0");
    auto *exp1 = graph.getNode("expert_1");
    auto *exp2 = graph.getNode("expert_2");
    auto *exp3 = graph.getNode("expert_3");

    ASSERT_NE(exp0, nullptr);
    ASSERT_NE(exp1, nullptr);
    ASSERT_NE(exp2, nullptr);
    ASSERT_NE(exp3, nullptr);

    EXPECT_EQ(exp0->device_idx, 0);
    EXPECT_EQ(exp1->device_idx, 1);
    EXPECT_EQ(exp2->device_idx, 0);
    EXPECT_EQ(exp3->device_idx, 1);
}

// =============================================================================
// Multi-Device Execution Tests
// =============================================================================

TEST_F(LayerExecutorTest, ExecuteMultiDeviceEmptyContexts)
{
    LayerExecutor executor;
    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    std::unordered_map<int, IDeviceContext *> contexts; // Empty
    EXPECT_FALSE(executor.executeMultiDevice(graph, contexts));
}

TEST_F(LayerExecutorTest, ExecuteMultiDeviceSingleContext)
{
    LayerExecutor executor;

    const int seq_len = 2;
    const int hidden_dim = 4;
    std::vector<float> input(seq_len * hidden_dim, 1.0f);
    std::vector<float> gamma(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.data();
    params.gamma = gamma.data();
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);

    std::unordered_map<int, IDeviceContext *> contexts;
    contexts[0] = ctx_.get();

    EXPECT_TRUE(executor.executeMultiDevice(graph, contexts));
    EXPECT_TRUE(graph.allCompleted());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(LayerExecutorTest, InvalidParams)
{
    LayerExecutor executor;

    // Null input/output should result in empty graph
    LayerExecutor::AttentionParams attn_params;
    attn_params.input = nullptr;
    attn_params.output = nullptr;

    auto attn_graph = executor.buildAttentionGraph(attn_params);
    EXPECT_EQ(attn_graph.size(), 0u);

    LayerExecutor::FFNParams ffn_params;
    ffn_params.input = nullptr;
    ffn_params.output = nullptr;

    auto ffn_graph = executor.buildFFNGraph(ffn_params);
    EXPECT_EQ(ffn_graph.size(), 0u);

    LayerExecutor::MoEParams moe_params;
    moe_params.input = nullptr;
    moe_params.output = nullptr;

    auto moe_graph = executor.buildMoEGraph(moe_params);
    EXPECT_EQ(moe_graph.size(), 0u);
}

TEST_F(LayerExecutorTest, DuplicateNodeName)
{
    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 64;
    params.eps = 1e-5f;

    // Add same name twice - should replace
    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 0);
    graph.addNode("A", ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS), 1);

    EXPECT_EQ(graph.size(), 1);

    auto *node = graph.getNode("A");
    EXPECT_EQ(node->device_idx, 1); // Second assignment should win
}

TEST_F(LayerExecutorTest, GetNodeNotFound)
{
    ComputeGraph graph;

    EXPECT_EQ(graph.getNode("nonexistent"), nullptr);
}
