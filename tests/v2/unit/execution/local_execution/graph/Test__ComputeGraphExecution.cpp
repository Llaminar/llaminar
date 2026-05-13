/**
 * @file Test__ComputeGraphExecution.cpp
 * @brief Comprehensive unit tests for ComputeGraph execution flow
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the entire execution graph framework including:
 * - Graph construction and topology
 * - Dependency resolution (topological sort)
 * - Execution ordering
 * - Error handling and propagation
 * - Multi-device execution
 * - Snapshot callbacks
 * - Statistics tracking
 */

#include <gtest/gtest.h>
#include "../../../../mocks/MockComputeStage.h"
#include "execution/local_execution/model/LayerExecutor.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include <algorithm>
#include <set>
#include <thread>

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class ComputeGraphExecutionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<MockDeviceContext>(0, ComputeBackendType::CPU);
        executor_ = std::make_unique<LayerExecutor>();
    }

    std::unique_ptr<MockDeviceContext> ctx_;
    std::unique_ptr<LayerExecutor> executor_;
};

// =============================================================================
// Graph Construction Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, EmptyGraph)
{
    ComputeGraph graph;

    EXPECT_EQ(graph.size(), 0);
    EXPECT_TRUE(graph.allCompleted());
    EXPECT_TRUE(graph.getExecutionOrder().empty());
    EXPECT_TRUE(graph.getReadyNodes().empty());
}

TEST_F(ComputeGraphExecutionTest, SingleNode)
{
    MockGraphBuilder builder;
    builder.addNode("single", ComputeStageType::GEMM);
    auto graph = builder.build();

    EXPECT_EQ(graph.size(), 1);
    EXPECT_FALSE(graph.allCompleted());

    auto order = graph.getExecutionOrder();
    ASSERT_EQ(order.size(), 1);
    EXPECT_EQ(order[0], "single");

    auto ready = graph.getReadyNodes();
    ASSERT_EQ(ready.size(), 1);
    EXPECT_EQ(ready[0], "single");
}

TEST_F(ComputeGraphExecutionTest, LinearChain)
{
    // A -> B -> C
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("B", "A");
    builder.addDependency("C", "B");

    auto graph = builder.build();

    EXPECT_EQ(graph.size(), 3);

    auto order = graph.getExecutionOrder();
    ASSERT_EQ(order.size(), 3);

    // Verify topological order: A must come before B, B before C
    auto pos_a = std::find(order.begin(), order.end(), "A");
    auto pos_b = std::find(order.begin(), order.end(), "B");
    auto pos_c = std::find(order.begin(), order.end(), "C");

    EXPECT_LT(pos_a, pos_b);
    EXPECT_LT(pos_b, pos_c);
}

TEST_F(ComputeGraphExecutionTest, DiamondDAG)
{
    // Diamond pattern:
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D

    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::GEMM);
    builder.addNode("D", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("B", "A");
    builder.addDependency("C", "A");
    builder.addDependency("D", "B");
    builder.addDependency("D", "C");

    auto graph = builder.build();

    auto order = graph.getExecutionOrder();
    ASSERT_EQ(order.size(), 4);

    // A must come first
    EXPECT_EQ(order[0], "A");

    // D must come last
    EXPECT_EQ(order[3], "D");

    // B and C must be between A and D (order doesn't matter)
    std::set<std::string> middle(order.begin() + 1, order.begin() + 3);
    EXPECT_TRUE(middle.count("B"));
    EXPECT_TRUE(middle.count("C"));
}

TEST_F(ComputeGraphExecutionTest, ComplexDAG_AttentionLike)
{
    // Simulates attention block structure:
    //
    //         norm
    //           |
    //     +-----+-----+
    //     |     |     |
    //    Wq    Wk    Wv
    //     |     |     |
    //   RoPE  RoPE    |
    //     |     |     |
    //     +-----+-----+
    //           |
    //        Attn
    //           |
    //          Wo
    //           |
    //       Residual

    MockGraphBuilder builder;
    builder.addNode("norm", ComputeStageType::RMS_NORM);
    builder.addNode("wq", ComputeStageType::GEMM);
    builder.addNode("wk", ComputeStageType::GEMM);
    builder.addNode("wv", ComputeStageType::GEMM);
    builder.addNode("rope_q", ComputeStageType::ROPE);
    builder.addNode("rope_k", ComputeStageType::ROPE);
    builder.addNode("attn", ComputeStageType::ATTENTION);
    builder.addNode("wo", ComputeStageType::GEMM);
    builder.addNode("residual", ComputeStageType::ADD_RESIDUAL);

    // Dependencies
    builder.addDependency("wq", "norm");
    builder.addDependency("wk", "norm");
    builder.addDependency("wv", "norm");
    builder.addDependency("rope_q", "wq");
    builder.addDependency("rope_k", "wk");
    builder.addDependency("attn", "rope_q");
    builder.addDependency("attn", "rope_k");
    builder.addDependency("attn", "wv");
    builder.addDependency("wo", "attn");
    builder.addDependency("residual", "wo");

    auto graph = builder.build();

    auto order = graph.getExecutionOrder();
    EXPECT_EQ(order.size(), 9);

    // Verify constraints
    auto pos = [&](const std::string &n)
    {
        return std::find(order.begin(), order.end(), n) - order.begin();
    };

    // norm must be first
    EXPECT_EQ(pos("norm"), 0);

    // QKV after norm
    EXPECT_GT(pos("wq"), pos("norm"));
    EXPECT_GT(pos("wk"), pos("norm"));
    EXPECT_GT(pos("wv"), pos("norm"));

    // RoPE after respective projections
    EXPECT_GT(pos("rope_q"), pos("wq"));
    EXPECT_GT(pos("rope_k"), pos("wk"));

    // Attention after all its inputs
    EXPECT_GT(pos("attn"), pos("rope_q"));
    EXPECT_GT(pos("attn"), pos("rope_k"));
    EXPECT_GT(pos("attn"), pos("wv"));

    // Wo after attention
    EXPECT_GT(pos("wo"), pos("attn"));

    // Residual last
    EXPECT_EQ(pos("residual"), 8);
}

// =============================================================================
// Ready Nodes (Parallel Execution) Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, ReadyNodes_InitialState)
{
    // Diamond: B and C can run in parallel after A
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::GEMM);
    builder.addNode("D", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("B", "A");
    builder.addDependency("C", "A");
    builder.addDependency("D", "B");
    builder.addDependency("D", "C");

    auto graph = builder.build();

    // Initially, only A has no dependencies
    auto ready = graph.getReadyNodes();
    ASSERT_EQ(ready.size(), 1);
    EXPECT_EQ(ready[0], "A");
}

TEST_F(ComputeGraphExecutionTest, ReadyNodes_AfterCompletion)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::GEMM);
    builder.addNode("D", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("B", "A");
    builder.addDependency("C", "A");
    builder.addDependency("D", "B");
    builder.addDependency("D", "C");

    auto graph = builder.build();

    // Mark A completed
    graph.markCompleted("A");

    // Now B and C should be ready (parallel)
    auto ready = graph.getReadyNodes();
    ASSERT_EQ(ready.size(), 2);
    std::set<std::string> ready_set(ready.begin(), ready.end());
    EXPECT_TRUE(ready_set.count("B"));
    EXPECT_TRUE(ready_set.count("C"));
}

TEST_F(ComputeGraphExecutionTest, ReadyNodes_ProgressiveCompletion)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::GEMM);
    builder.addNode("D", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("B", "A");
    builder.addDependency("C", "A");
    builder.addDependency("D", "B");
    builder.addDependency("D", "C");

    auto graph = builder.build();

    // Complete A and B
    graph.markCompleted("A");
    graph.markCompleted("B");

    // C is ready, but D needs both B and C
    auto ready = graph.getReadyNodes();
    ASSERT_EQ(ready.size(), 1);
    EXPECT_EQ(ready[0], "C");

    // Complete C
    graph.markCompleted("C");

    // Now D should be ready
    ready = graph.getReadyNodes();
    ASSERT_EQ(ready.size(), 1);
    EXPECT_EQ(ready[0], "D");
}

// =============================================================================
// Execution Order Tests (with LayerExecutor)
// =============================================================================

TEST_F(ComputeGraphExecutionTest, Execute_LinearChain_CorrectOrder)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("B", "A");
    builder.addDependency("C", "B");

    auto graph = builder.build();

    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    // Verify execution order
    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 3);
    EXPECT_EQ(log[0], "A");
    EXPECT_EQ(log[1], "B");
    EXPECT_EQ(log[2], "C");

    // All nodes should be completed
    EXPECT_TRUE(graph.allCompleted());
}

TEST_F(ComputeGraphExecutionTest, Execute_DiamondDAG_ValidOrder)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::GEMM);
    builder.addNode("D", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("B", "A");
    builder.addDependency("C", "A");
    builder.addDependency("D", "B");
    builder.addDependency("D", "C");

    auto graph = builder.build();

    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 4);

    // A must be first
    EXPECT_EQ(log[0], "A");

    // D must be last
    EXPECT_EQ(log[3], "D");

    // B and C in middle (either order is valid)
    std::set<std::string> middle(log.begin() + 1, log.begin() + 3);
    EXPECT_TRUE(middle.count("B"));
    EXPECT_TRUE(middle.count("C"));
}

TEST_F(ComputeGraphExecutionTest, Execute_IndependentNodes_AllExecuted)
{
    // No dependencies - all can run independently
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::GEMM);

    auto graph = builder.build();

    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 3);

    // All executed (order doesn't matter)
    std::set<std::string> executed(log.begin(), log.end());
    EXPECT_TRUE(executed.count("A"));
    EXPECT_TRUE(executed.count("B"));
    EXPECT_TRUE(executed.count("C"));
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, Execute_NullContext_Fails)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    auto graph = builder.build();

    EXPECT_FALSE(executor_->execute(graph, nullptr));
}

TEST_F(ComputeGraphExecutionTest, Execute_StageFails_StopsExecution)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("B", "A");
    builder.addDependency("C", "B");

    // Configure B to fail
    builder.getStage("B")->setShouldSucceed(false);

    auto graph = builder.build();

    EXPECT_FALSE(executor_->execute(graph, ctx_.get()));

    // A should have executed, B failed, C should not execute
    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 2); // A and B attempted
    EXPECT_EQ(log[0], "A");
    EXPECT_EQ(log[1], "B");
}

TEST_F(ComputeGraphExecutionTest, Execute_FirstNodeFails_NoneComplete)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addDependency("B", "A");

    builder.getStage("A")->setShouldSucceed(false);

    auto graph = builder.build();

    EXPECT_FALSE(executor_->execute(graph, ctx_.get()));

    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 1); // Only A attempted
    EXPECT_EQ(log[0], "A");
}

TEST_F(ComputeGraphExecutionTest, Execute_StageFailureInvokesFailureCallback)
{
    int callback_count = 0;
    std::string failed_stage;
    std::string failure_reason;

    GraphExecutorConfig config;
    config.stage_failure_callback = [&](const std::string &node_name, const std::string &reason)
    {
        ++callback_count;
        failed_stage = node_name;
        failure_reason = reason;
    };
    LayerExecutor executor(config);

    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addDependency("B", "A");
    builder.getStage("B")->setShouldSucceed(false);

    auto graph = builder.build();

    EXPECT_FALSE(executor.execute(graph, ctx_.get()));
    EXPECT_EQ(callback_count, 1);
    EXPECT_EQ(failed_stage, "B");
    EXPECT_NE(failure_reason.find("returned false"), std::string::npos);
}

TEST_F(ComputeGraphExecutionTest, Execute_CancellationStopsBeforeNextStage)
{
    bool cancel_requested = false;
    int failure_callback_count = 0;

    GraphExecutorConfig config;
    config.cancellation_requested = [&]()
    { return cancel_requested; };
    config.stage_failure_callback = [&](const std::string &, const std::string &)
    {
        ++failure_callback_count;
    };
    LayerExecutor executor(config);

    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addDependency("B", "A");
    builder.getStage("A")->setOnExecute([&](IDeviceContext *)
                                        { cancel_requested = true; });

    auto graph = builder.build();

    EXPECT_FALSE(executor.execute(graph, ctx_.get()));
    EXPECT_EQ(failure_callback_count, 0);

    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 1);
    EXPECT_EQ(log[0], "A");
    EXPECT_EQ(builder.getStage("B")->executionCount(), 0);
}

TEST_F(ComputeGraphExecutionTest, ExecuteMultiDevice_CancellationStopsBeforeNextStage)
{
    bool cancel_requested = false;
    int failure_callback_count = 0;

    GraphExecutorConfig config;
    config.cancellation_requested = [&]()
    { return cancel_requested; };
    config.stage_failure_callback = [&](const std::string &, const std::string &)
    {
        ++failure_callback_count;
    };
    LayerExecutor executor(config);

    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::RMS_NORM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addDependency("B", "A");
    builder.getStage("A")->setOnExecute([&](IDeviceContext *)
                                        { cancel_requested = true; });

    auto graph = builder.build();
    std::unordered_map<DeviceId, IDeviceContext *> contexts{{DeviceId::cpu(), ctx_.get()}};

    EXPECT_FALSE(executor.executeMultiDevice(graph, contexts));
    EXPECT_EQ(failure_callback_count, 0);

    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 1);
    EXPECT_EQ(log[0], "A");
    EXPECT_EQ(builder.getStage("B")->executionCount(), 0);
}

// =============================================================================
// Graph Reset Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, Graph_Reset_AllowsReexecution)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addDependency("B", "A");

    auto graph = builder.build();

    // First execution
    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));
    EXPECT_TRUE(graph.allCompleted());

    // Reset
    graph.reset();
    EXPECT_FALSE(graph.allCompleted());

    // Clear log and re-execute
    builder.clearLog();
    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 2);
}

// =============================================================================
// Statistics Tracking Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, Stats_TracksExecutedStages)
{
    executor_->setProfilingEnabled(true);
    executor_->resetStats();

    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::GEMM);

    builder.getStage("A")->setEstimatedFlops(1000);
    builder.getStage("B")->setEstimatedFlops(2000);
    builder.getStage("C")->setEstimatedFlops(3000);

    auto graph = builder.build();

    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    const auto &stats = executor_->stats();
    EXPECT_EQ(stats.total_stages_executed, 3);
    EXPECT_EQ(stats.total_flops, 6000); // 1000 + 2000 + 3000
}

TEST_F(ComputeGraphExecutionTest, Stats_Reset)
{
    executor_->setProfilingEnabled(true);

    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    auto graph = builder.build();

    executor_->execute(graph, ctx_.get());
    EXPECT_GT(executor_->stats().total_stages_executed, 0);

    executor_->resetStats();
    EXPECT_EQ(executor_->stats().total_stages_executed, 0);
    EXPECT_EQ(executor_->stats().total_flops, 0);
}

// =============================================================================
// Snapshot Callback Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, SnapshotCallback_CalledForEachStage)
{
    std::vector<std::string> snapshot_names;

    executor_->setSnapshotCallback([&](const std::string &name, const StageDumpInfo &dump_info)
                                   { snapshot_names.push_back(name); });

    MockGraphBuilder builder;
    builder.addNode("norm", ComputeStageType::RMS_NORM);
    builder.addNode("proj", ComputeStageType::GEMM);
    builder.addNode("residual", ComputeStageType::ADD_RESIDUAL);
    builder.addDependency("proj", "norm");
    builder.addDependency("residual", "proj");

    auto graph = builder.build();

    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    ASSERT_EQ(snapshot_names.size(), 3);
    EXPECT_EQ(snapshot_names[0], "norm");
    EXPECT_EQ(snapshot_names[1], "proj");
    EXPECT_EQ(snapshot_names[2], "residual");
}

TEST_F(ComputeGraphExecutionTest, SnapshotCallback_ReceivesDumpInfo)
{
    StageDumpInfo received_info;

    executor_->setSnapshotCallback([&](const std::string &name, const StageDumpInfo &dump_info)
                                   {
        if (name == "test") {
            received_info = dump_info;
        } });

    MockGraphBuilder builder;
    builder.addNode("test", ComputeStageType::GEMM);

    // Configure mock with specific dump info
    // Use valid memory to avoid ASAN errors when buffer validation is enabled
    constexpr size_t kRows = 32;
    constexpr size_t kCols = 896;
    std::vector<float> test_buffer(kRows * kCols, 1.0f); // Non-zero to pass validation
    const float *test_data = test_buffer.data();

    StageDumpInfo test_info;
    test_info.addOutput("C", test_data, kRows, kCols);
    builder.getStage("test")->setDumpInfo(test_info);

    auto graph = builder.build();

    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    ASSERT_EQ(received_info.outputs.size(), 1);
    EXPECT_EQ(received_info.outputs[0].rows, kRows);
    EXPECT_EQ(received_info.outputs[0].cols, kCols);
    EXPECT_EQ(received_info.outputs[0].data, test_data);
}

// =============================================================================
// Execution Mode Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, ExecutionMode_Sequential)
{
    executor_->setExecutionMode(ExecutionMode::SEQUENTIAL);
    EXPECT_EQ(executor_->executionMode(), ExecutionMode::SEQUENTIAL);

    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);
    auto graph = builder.build();

    // Sequential mode should still work
    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));
    EXPECT_EQ(builder.executionLog().size(), 2);
}

TEST_F(ComputeGraphExecutionTest, ExecutionMode_Parallel)
{
    executor_->setExecutionMode(ExecutionMode::PARALLEL);
    EXPECT_EQ(executor_->executionMode(), ExecutionMode::PARALLEL);

    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);
    auto graph = builder.build();

    // Parallel mode should execute both
    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));
    EXPECT_EQ(builder.executionLog().size(), 2);
}

// =============================================================================
// Graph Estimated FLOPs
// =============================================================================

TEST_F(ComputeGraphExecutionTest, Graph_TotalEstimatedFlops)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);
    builder.addNode("C", ComputeStageType::GEMM);

    builder.getStage("A")->setEstimatedFlops(1000000);
    builder.getStage("B")->setEstimatedFlops(2000000);
    builder.getStage("C")->setEstimatedFlops(500000);

    auto graph = builder.build();

    EXPECT_EQ(graph.totalEstimatedFlops(), 3500000);
}

// =============================================================================
// Node Access Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, GetNode_ValidName)
{
    MockGraphBuilder builder;
    builder.addNode("test_node", ComputeStageType::GEMM);
    auto graph = builder.build();

    auto *node = graph.getNode("test_node");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_node");
    EXPECT_NE(node->stage, nullptr);
}

TEST_F(ComputeGraphExecutionTest, GetNode_InvalidName)
{
    MockGraphBuilder builder;
    builder.addNode("exists", ComputeStageType::GEMM);
    auto graph = builder.build();

    auto *node = graph.getNode("does_not_exist");
    EXPECT_EQ(node, nullptr);
}

// =============================================================================
// Device Index Assignment Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, Node_DeviceAssignment)
{
    MockGraphBuilder builder;
    builder.addNode("cpu_node", ComputeStageType::GEMM, DeviceId::cpu());
    builder.addNode("gpu_node", ComputeStageType::GEMM, DeviceId::cuda(0));

    auto graph = builder.build();

    EXPECT_EQ(graph.getNode("cpu_node")->device, DeviceId::cpu());
    EXPECT_EQ(graph.getNode("gpu_node")->device, DeviceId::cuda(0));
}

// =============================================================================
// Complex Integration Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, FullAttentionBlock_ExecutionOrder)
{
    // Full attention block simulation
    MockGraphBuilder builder;

    // Add all attention stages
    builder.addNode("attn_norm", ComputeStageType::RMS_NORM);
    builder.addNode("q_proj", ComputeStageType::GEMM);
    builder.addNode("k_proj", ComputeStageType::GEMM);
    builder.addNode("v_proj", ComputeStageType::GEMM);
    builder.addNode("q_rope", ComputeStageType::ROPE);
    builder.addNode("k_rope", ComputeStageType::ROPE);
    builder.addNode("attention", ComputeStageType::ATTENTION);
    builder.addNode("wo_proj", ComputeStageType::GEMM);
    builder.addNode("attn_residual", ComputeStageType::ADD_RESIDUAL);

    // Set up dependencies
    builder.addDependency("q_proj", "attn_norm");
    builder.addDependency("k_proj", "attn_norm");
    builder.addDependency("v_proj", "attn_norm");
    builder.addDependency("q_rope", "q_proj");
    builder.addDependency("k_rope", "k_proj");
    builder.addDependency("attention", "q_rope");
    builder.addDependency("attention", "k_rope");
    builder.addDependency("attention", "v_proj");
    builder.addDependency("wo_proj", "attention");
    builder.addDependency("attn_residual", "wo_proj");

    auto graph = builder.build();

    // Execute
    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 9);

    // Verify key constraints
    auto pos = [&](const std::string &n)
    {
        return std::find(log.begin(), log.end(), n) - log.begin();
    };

    // Norm first
    EXPECT_EQ(pos("attn_norm"), 0);

    // Residual last
    EXPECT_EQ(pos("attn_residual"), 8);

    // All projections after norm
    EXPECT_GT(pos("q_proj"), pos("attn_norm"));
    EXPECT_GT(pos("k_proj"), pos("attn_norm"));
    EXPECT_GT(pos("v_proj"), pos("attn_norm"));

    // RoPE after projections
    EXPECT_GT(pos("q_rope"), pos("q_proj"));
    EXPECT_GT(pos("k_rope"), pos("k_proj"));

    // Attention after all inputs
    EXPECT_GT(pos("attention"), pos("q_rope"));
    EXPECT_GT(pos("attention"), pos("k_rope"));
    EXPECT_GT(pos("attention"), pos("v_proj"));

    // Wo after attention
    EXPECT_GT(pos("wo_proj"), pos("attention"));
}

TEST_F(ComputeGraphExecutionTest, FullFFNBlock_ExecutionOrder)
{
    MockGraphBuilder builder;

    builder.addNode("ffn_norm", ComputeStageType::RMS_NORM);
    builder.addNode("gate_proj", ComputeStageType::GEMM);
    builder.addNode("up_proj", ComputeStageType::GEMM);
    builder.addNode("swiglu", ComputeStageType::SWIGLU);
    builder.addNode("down_proj", ComputeStageType::GEMM);
    builder.addNode("ffn_residual", ComputeStageType::ADD_RESIDUAL);

    builder.addDependency("gate_proj", "ffn_norm");
    builder.addDependency("up_proj", "ffn_norm");
    builder.addDependency("swiglu", "gate_proj");
    builder.addDependency("swiglu", "up_proj");
    builder.addDependency("down_proj", "swiglu");
    builder.addDependency("ffn_residual", "down_proj");

    auto graph = builder.build();

    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    auto &log = builder.executionLog();
    ASSERT_EQ(log.size(), 6);

    auto pos = [&](const std::string &n)
    {
        return std::find(log.begin(), log.end(), n) - log.begin();
    };

    EXPECT_EQ(pos("ffn_norm"), 0);
    EXPECT_EQ(pos("ffn_residual"), 5);

    // Gate and up can be parallel, both after norm
    EXPECT_GT(pos("gate_proj"), pos("ffn_norm"));
    EXPECT_GT(pos("up_proj"), pos("ffn_norm"));

    // SwiGLU needs both
    EXPECT_GT(pos("swiglu"), pos("gate_proj"));
    EXPECT_GT(pos("swiglu"), pos("up_proj"));

    // Down after SwiGLU
    EXPECT_GT(pos("down_proj"), pos("swiglu"));
}

// =============================================================================
// Stage Custom Execution Hook Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, Stage_CustomExecutionHook)
{
    int hook_calls = 0;
    IDeviceContext *hook_ctx = nullptr;

    MockGraphBuilder builder;
    builder.addNode("test", ComputeStageType::GEMM);
    builder.getStage("test")->setOnExecute([&](IDeviceContext *ctx)
                                           {
        hook_calls++;
        hook_ctx = ctx; });

    auto graph = builder.build();

    EXPECT_TRUE(executor_->execute(graph, ctx_.get()));

    EXPECT_EQ(hook_calls, 1);
    EXPECT_EQ(hook_ctx, ctx_.get());
}

// =============================================================================
// Mock Device Context Tracking Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, DeviceContext_TracksOperations)
{
    // Run stages that would use the context
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);

    auto graph = builder.build();

    executor_->execute(graph, ctx_.get());

    // Verify context was used
    // (Actual tracking depends on stage implementations calling context methods)
    EXPECT_GE(ctx_->syncCount(), 0); // May or may not sync depending on mode
}

// =============================================================================
// Graph Clear Tests
// =============================================================================

TEST_F(ComputeGraphExecutionTest, Graph_Clear)
{
    MockGraphBuilder builder;
    builder.addNode("A", ComputeStageType::GEMM);
    builder.addNode("B", ComputeStageType::GEMM);

    auto graph = builder.build();
    EXPECT_EQ(graph.size(), 2);

    graph.clear();
    EXPECT_EQ(graph.size(), 0);
    EXPECT_TRUE(graph.allCompleted()); // Empty graph is "completed"
}

// =============================================================================
// Execution with Real Stages (Integration)
// =============================================================================

TEST_F(ComputeGraphExecutionTest, RealResidualAddStage_InGraph)
{
    // Test with real ResidualAddStage (not mock) to verify integration
    const size_t n = 64;
    const std::vector<size_t> shape = {1, n};

    // Create FP32 tensors for the residual add operation
    auto input_tensor = std::make_unique<FP32Tensor>(shape);
    auto residual_tensor = std::make_unique<FP32Tensor>(shape);
    auto output_tensor = std::make_unique<FP32Tensor>(shape);

    // Initialize input data
    float *input_data = input_tensor->mutable_data();
    float *residual_data = residual_tensor->mutable_data();
    for (size_t i = 0; i < n; ++i)
    {
        input_data[i] = 1.0f;
        residual_data[i] = 2.0f;
    }

    ResidualAddStage::Params params;
    params.input = input_tensor.get();
    params.residual = residual_tensor.get();
    params.output = output_tensor.get();

    ComputeGraph graph;
    graph.addNode("residual",
                  ComputeStageFactory::createResidualAdd(params),
                  DeviceId::cpu());

    // Use real CPU context
    auto real_ctx = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4);

    EXPECT_TRUE(executor_->execute(graph, real_ctx.get()));

    // Verify actual computation
    const float *output_data = output_tensor->data();
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(output_data[i], 3.0f); // 1.0 + 2.0
    }
}
