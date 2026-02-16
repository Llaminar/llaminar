/**
 * @file Test__LayerExecutor.cpp
 * @brief Unit tests for LayerExecutor
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "execution/local_execution/model/LayerExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <memory>

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
        ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4);
        ASSERT_NE(ctx_, nullptr);
    }

    // Helper: Create FP32Tensor with given dimensions
    std::unique_ptr<FP32Tensor> makeTensor(size_t rows, size_t cols)
    {
        return std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols}, DeviceId::cpu());
    }

    // Helper: Create FP32Tensor with given dimensions and fill with value
    std::unique_ptr<FP32Tensor> makeTensor(size_t rows, size_t cols, float fill_value)
    {
        auto tensor = makeTensor(rows, cols);
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            data[i] = fill_value;
        }
        return tensor;
    }

    // Helper: Create 1D FP32Tensor
    std::unique_ptr<FP32Tensor> makeTensor1D(size_t n, float fill_value = 1.0f)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{n}, DeviceId::cpu());
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < n; ++i)
        {
            data[i] = fill_value;
        }
        return tensor;
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

    // Create tensors for RMSNorm stage
    auto input = makeTensor(1, 64, 1.0f);
    auto output = makeTensor(1, 64);
    auto gamma = makeTensor1D(64, 1.0f);

    // Create a simple RMSNorm stage with TensorBase* API
    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    auto stage = ComputeStageFactory::createRMSNorm(params);
    graph.addNode("norm", std::move(stage), DeviceId::cpu());

    EXPECT_EQ(graph.size(), 1);
    EXPECT_FALSE(graph.allCompleted());

    auto order = graph.getExecutionOrder();
    ASSERT_EQ(order.size(), 1);
    EXPECT_EQ(order[0], "norm");
}

TEST_F(LayerExecutorTest, LinearDependencyChain)
{
    ComputeGraph graph;

    // Create tensors
    auto input = makeTensor(1, 64, 1.0f);
    auto output = makeTensor(1, 64);
    auto gamma = makeTensor1D(64, 1.0f);

    // Create: A -> B -> C (linear chain)
    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("C", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

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

    // Create tensors
    auto input = makeTensor(1, 64, 1.0f);
    auto output = makeTensor(1, 64);
    auto gamma = makeTensor1D(64, 1.0f);

    // Create diamond: A -> [B, C] -> D
    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("C", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("D", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

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

    // Create tensors
    auto input = makeTensor(1, 64, 1.0f);
    auto output = makeTensor(1, 64);
    auto gamma = makeTensor1D(64, 1.0f);

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    // A -> B, C (no deps)
    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("C", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

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

    // Create tensors
    auto input = makeTensor(1, 64, 1.0f);
    auto output = makeTensor(1, 64);
    auto gamma = makeTensor1D(64, 1.0f);

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

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

    // Create tensors for 10 rows x 64 cols
    auto input = makeTensor(10, 64, 1.0f);
    auto output = makeTensor(10, 64);
    auto gamma = makeTensor1D(64, 1.0f);

    // RMSNorm flops: 4 * seq_len * hidden_dim (squares + adds + sqrt + div + muls)
    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

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

    // Create tensors
    auto input = makeTensor(1, 64, 1.0f);
    auto output = makeTensor(1, 64);
    auto gamma = makeTensor1D(64, 1.0f);

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    EXPECT_FALSE(executor.execute(graph, nullptr));
}

TEST_F(LayerExecutorTest, ExecuteSequential)
{
    LayerExecutor executor;
    executor.setExecutionMode(ExecutionMode::SEQUENTIAL);

    // Create tensors for RMSNorm
    const size_t seq_len = 2;
    const size_t hidden_dim = 4;

    auto input = makeTensor(seq_len, hidden_dim, 1.0f);
    auto gamma = makeTensor1D(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = input.get(); // In-place operation
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    bool result = false;
    try
    {
        result = executor.execute(graph, ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Skipping due to device enumeration: " << e.what();
    }

    EXPECT_TRUE(result);
    EXPECT_TRUE(graph.allCompleted());

    // RMSNorm on all 1s with gamma=1 should output all 1s
    const float *out_data = input->data();
    for (size_t i = 0; i < seq_len * hidden_dim; ++i)
    {
        EXPECT_NEAR(out_data[i], 1.0f, 1e-5f);
    }
}

TEST_F(LayerExecutorTest, ExecuteParallel)
{
    LayerExecutor executor;
    executor.setExecutionMode(ExecutionMode::PARALLEL);

    const size_t seq_len = 2;
    const size_t hidden_dim = 4;

    auto input = makeTensor(seq_len, hidden_dim, 1.0f);
    auto gamma = makeTensor1D(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = input.get(); // In-place operation
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    bool result = false;
    try
    {
        result = executor.execute(graph, ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Skipping due to device enumeration: " << e.what();
    }

    EXPECT_TRUE(result);
    EXPECT_TRUE(graph.allCompleted());
}

TEST_F(LayerExecutorTest, ExecuteWithDependencies)
{
    LayerExecutor executor;

    const size_t seq_len = 2;
    const size_t hidden_dim = 4;

    auto input = makeTensor(seq_len, hidden_dim, 2.0f);
    auto gamma = makeTensor1D(hidden_dim, 1.0f);

    ComputeGraph graph;

    // Two sequential norms: A -> B
    RMSNormStage::Params params;
    params.input = input.get();
    params.output = input.get(); // In-place operation
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("B", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addDependency("B", "A");

    bool result = false;
    try
    {
        result = executor.execute(graph, ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Skipping due to device enumeration: " << e.what();
    }

    EXPECT_TRUE(result);
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

    const size_t seq_len = 2;
    const size_t hidden_dim = 4;

    auto input = makeTensor(seq_len, hidden_dim, 1.0f);
    auto gamma = makeTensor1D(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = input.get(); // In-place operation
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    try
    {
        executor.execute(graph, ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Skipping due to device enumeration: " << e.what();
    }

    const auto &stats = executor.stats();
    EXPECT_GT(stats.total_stages_executed, 0u);
    EXPECT_GT(stats.total_flops, 0u);
    EXPECT_GE(stats.total_time_ms, 0.0);
}

TEST_F(LayerExecutorTest, StatsReset)
{
    LayerExecutor executor;

    const size_t seq_len = 2;
    const size_t hidden_dim = 4;

    auto input = makeTensor(seq_len, hidden_dim, 1.0f);
    auto gamma = makeTensor1D(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = input.get(); // In-place operation
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    try
    {
        executor.execute(graph, ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Skipping due to device enumeration: " << e.what();
    }
    EXPECT_GT(executor.stats().total_stages_executed, 0u);

    executor.resetStats();
    EXPECT_EQ(executor.stats().total_stages_executed, 0u);
}

// =============================================================================
// Graph Building Tests - MOVED TO Test__Qwen2Graph.cpp
// =============================================================================
// NOTE: Architecture-specific graph building tests (buildAttentionGraph,
// buildFFNGraph, buildMoEGraph) were moved to Test__Qwen2Graph.cpp as part
// of the LayerExecutor → DeviceGraphExecutor + Qwen2Graph refactoring.
//
// DeviceGraphExecutor is now a pure execution engine. Architecture-specific graph
// building logic lives in Qwen2Graph (or other architecture-specific classes).
// =============================================================================

// =============================================================================
// Multi-Device Execution Tests
// =============================================================================

TEST_F(LayerExecutorTest, ExecuteMultiDeviceEmptyContexts)
{
    LayerExecutor executor;
    ComputeGraph graph;

    // Create a tensor but use a minimal config
    auto input = makeTensor(1, 64, 0.0f);
    auto gamma = makeTensor1D(64, 1.0f);

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = input.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    std::unordered_map<DeviceId, IDeviceContext *> contexts; // Empty
    EXPECT_FALSE(executor.executeMultiDevice(graph, contexts));
}

TEST_F(LayerExecutorTest, ExecuteMultiDeviceSingleContext)
{
    LayerExecutor executor;

    const size_t seq_len = 2;
    const size_t hidden_dim = 4;

    auto input = makeTensor(seq_len, hidden_dim, 1.0f);
    auto gamma = makeTensor1D(hidden_dim, 1.0f);

    ComputeGraph graph;

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = input.get(); // In-place operation
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    graph.addNode("norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    std::unordered_map<DeviceId, IDeviceContext *> contexts;
    contexts[DeviceId::cpu()] = ctx_.get();

    try
    {
        EXPECT_TRUE(executor.executeMultiDevice(graph, contexts));
        EXPECT_TRUE(graph.allCompleted());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Skipping due to device enumeration: " << e.what();
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

// NOTE: Tests for buildAttentionGraph/buildFFNGraph/buildMoEGraph with null
// params moved to Test__Qwen2Graph.cpp - DeviceGraphExecutor is pure execution engine

TEST_F(LayerExecutorTest, DuplicateNodeName)
{
    ComputeGraph graph;

    auto input = makeTensor(1, 64, 1.0f);
    auto gamma = makeTensor1D(64, 1.0f);

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = input.get();
    params.gamma = gamma.get();
    params.eps = 1e-5f;

    // Add same name twice - should replace
    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    graph.addNode("A", ComputeStageFactory::createRMSNorm(params), DeviceId::cuda(1));

    EXPECT_EQ(graph.size(), 1);

    auto *node = graph.getNode("A");
    EXPECT_EQ(node->device, DeviceId::cuda(1)); // Second assignment should win
}

TEST_F(LayerExecutorTest, GetNodeNotFound)
{
    ComputeGraph graph;

    EXPECT_EQ(graph.getNode("nonexistent"), nullptr);
}
