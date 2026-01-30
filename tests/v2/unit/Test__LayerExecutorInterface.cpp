/**
 * @file Test__LayerExecutorInterface.cpp
 * @brief Unit tests for ILayerExecutor interface and MockLayerExecutor
 * @author David Sanftenberg
 * @date January 2025
 *
 * Tests for:
 * - ILayerExecutor interface polymorphism
 * - MockLayerExecutor behavior and tracking
 * - LayerExecutor through the interface
 */

#include <gtest/gtest.h>

#include "../mocks/MockComputeStage.h"
#include "backends/DeviceId.h"
#include "execution/local_execution/model/ILayerExecutor.h"
#include "execution/local_execution/model/LayerExecutor.h"
#include "execution/compute_stages/ComputeStages.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class LayerExecutorInterfaceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<MockDeviceContext>(0, ComputeBackendType::CPU);
    }

    std::unique_ptr<MockDeviceContext> ctx_;
};

// =============================================================================
// ILayerExecutor Interface Tests
// =============================================================================

TEST_F(LayerExecutorInterfaceTest, MockImplementsILayerExecutor)
{
    MockLayerExecutor mock;

    // Verify it's an ILayerExecutor
    ILayerExecutor &interface_ref = mock;

    // Test config access through interface
    EXPECT_EQ(interface_ref.executionMode(), ExecutionMode::SEQUENTIAL);

    interface_ref.setExecutionMode(ExecutionMode::PARALLEL);
    EXPECT_EQ(interface_ref.executionMode(), ExecutionMode::PARALLEL);
}

TEST_F(LayerExecutorInterfaceTest, RealImplementsILayerExecutor)
{
    LayerExecutor real;

    // Verify it's an ILayerExecutor
    ILayerExecutor &interface_ref = real;

    // Test config access through interface
    EXPECT_EQ(interface_ref.executionMode(), ExecutionMode::SEQUENTIAL);

    interface_ref.setExecutionMode(ExecutionMode::PARALLEL);
    EXPECT_EQ(interface_ref.executionMode(), ExecutionMode::PARALLEL);
}

TEST_F(LayerExecutorInterfaceTest, PolymorphicExecute_MockSucceeds)
{
    MockLayerExecutor mock;
    ILayerExecutor &interface_ref = mock;

    ComputeGraph graph;
    graph.addNode("A", std::make_unique<MockComputeStage>(ComputeStageType::GEMM, "A"));

    // Execute through interface
    bool result = interface_ref.execute(graph, ctx_.get());

    EXPECT_TRUE(result);
    EXPECT_EQ(mock.executeCount(), 1);
    EXPECT_EQ(mock.lastContext(), ctx_.get());
}

TEST_F(LayerExecutorInterfaceTest, PolymorphicExecute_MockFails)
{
    MockLayerExecutor mock;
    mock.setShouldSucceed(false);
    ILayerExecutor &interface_ref = mock;

    ComputeGraph graph;
    graph.addNode("A", std::make_unique<MockComputeStage>(ComputeStageType::GEMM, "A"));

    // Execute through interface - should fail
    bool result = interface_ref.execute(graph, ctx_.get());

    EXPECT_FALSE(result);
    EXPECT_EQ(mock.executeCount(), 1);
}

TEST_F(LayerExecutorInterfaceTest, PolymorphicExecute_RealLayerExecutor)
{
    LayerExecutor real;
    ILayerExecutor &interface_ref = real;

    ComputeGraph graph;
    auto mock_stage = std::make_unique<MockComputeStage>(ComputeStageType::GEMM, "A");
    mock_stage->setShouldSucceed(true);
    graph.addNode("A", std::move(mock_stage));

    // Execute through interface
    bool result = interface_ref.execute(graph, ctx_.get());

    EXPECT_TRUE(result);
    EXPECT_EQ(interface_ref.stats().total_stages_executed, 1);
}

TEST_F(LayerExecutorInterfaceTest, InterfaceConfiguration)
{
    MockLayerExecutor mock;
    ILayerExecutor &iface = mock;

    // Test all configuration methods through interface
    iface.setExecutionMode(ExecutionMode::PIPELINED);
    EXPECT_EQ(iface.executionMode(), ExecutionMode::PIPELINED);
    EXPECT_EQ(iface.config().mode, ExecutionMode::PIPELINED);

    iface.setProfilingEnabled(true);
    EXPECT_TRUE(iface.config().enable_profiling);

    iface.setValidationEnabled(true);
    EXPECT_TRUE(iface.config().enable_validation);
}

TEST_F(LayerExecutorInterfaceTest, InterfaceStats)
{
    LayerExecutor real;
    ILayerExecutor &iface = real;

    // Initial stats should be zero
    EXPECT_EQ(iface.stats().total_stages_executed, 0);
    EXPECT_EQ(iface.stats().total_flops, 0);
    EXPECT_DOUBLE_EQ(iface.stats().total_time_ms, 0.0);

    // Execute something
    ComputeGraph graph;
    auto stage = std::make_unique<MockComputeStage>(ComputeStageType::GEMM, "A");
    stage->setEstimatedFlops(1000);
    graph.addNode("A", std::move(stage));

    iface.execute(graph, ctx_.get());

    // Stats should reflect execution
    EXPECT_EQ(iface.stats().total_stages_executed, 1);

    // Reset stats through interface
    iface.resetStats();
    EXPECT_EQ(iface.stats().total_stages_executed, 0);
}

TEST_F(LayerExecutorInterfaceTest, InterfaceSnapshotCallback)
{
    LayerExecutor real;
    ILayerExecutor &iface = real;

    std::vector<std::string> captured_names;

    iface.setSnapshotCallback([&](const std::string &name, const StageDumpInfo &dump_info)
                              { captured_names.push_back(name); });

    ComputeGraph graph;
    auto stage = std::make_unique<MockComputeStage>(ComputeStageType::GEMM, "test_stage");
    graph.addNode("test_stage", std::move(stage));

    iface.execute(graph, ctx_.get());

    // Callback should have been called
    ASSERT_EQ(captured_names.size(), 1);
    EXPECT_EQ(captured_names[0], "test_stage");
}

// =============================================================================
// MockLayerExecutor Specific Tests
// =============================================================================

TEST_F(LayerExecutorInterfaceTest, Mock_TracksExecuteCalls)
{
    MockLayerExecutor mock;

    ComputeGraph graph1, graph2;
    graph1.addNode("A", std::make_unique<MockComputeStage>());
    graph2.addNode("B", std::make_unique<MockComputeStage>());

    EXPECT_EQ(mock.executeCount(), 0);

    mock.execute(graph1, ctx_.get());
    EXPECT_EQ(mock.executeCount(), 1);
    EXPECT_EQ(mock.lastGraph(), &graph1);

    mock.execute(graph2, ctx_.get());
    EXPECT_EQ(mock.executeCount(), 2);
    EXPECT_EQ(mock.lastGraph(), &graph2);
}

TEST_F(LayerExecutorInterfaceTest, Mock_TracksMultiExecuteCalls)
{
    MockLayerExecutor mock;

    ComputeGraph graph;
    std::unordered_map<DeviceId, IDeviceContext *> contexts;
    contexts[DeviceId::cpu()] = ctx_.get();

    EXPECT_EQ(mock.multiExecuteCount(), 0);

    mock.executeMultiDevice(graph, contexts);
    EXPECT_EQ(mock.multiExecuteCount(), 1);
}

TEST_F(LayerExecutorInterfaceTest, Mock_OnExecuteHook)
{
    MockLayerExecutor mock;

    bool hook_called = false;
    ComputeGraph *hook_graph = nullptr;
    IDeviceContext *hook_ctx = nullptr;

    mock.setOnExecute([&](ComputeGraph &g, IDeviceContext *c)
                      {
        hook_called = true;
        hook_graph = &g;
        hook_ctx = c; });

    ComputeGraph graph;
    mock.execute(graph, ctx_.get());

    EXPECT_TRUE(hook_called);
    EXPECT_EQ(hook_graph, &graph);
    EXPECT_EQ(hook_ctx, ctx_.get());
}

TEST_F(LayerExecutorInterfaceTest, Mock_Reset)
{
    MockLayerExecutor mock;

    ComputeGraph graph;
    mock.execute(graph, ctx_.get());
    mock.execute(graph, ctx_.get());

    std::unordered_map<DeviceId, IDeviceContext *> contexts;
    mock.executeMultiDevice(graph, contexts);

    EXPECT_EQ(mock.executeCount(), 2);
    EXPECT_EQ(mock.multiExecuteCount(), 1);

    mock.reset();

    EXPECT_EQ(mock.executeCount(), 0);
    EXPECT_EQ(mock.multiExecuteCount(), 0);
    EXPECT_EQ(mock.lastGraph(), nullptr);
    EXPECT_EQ(mock.lastContext(), nullptr);
}

TEST_F(LayerExecutorInterfaceTest, Mock_SnapshotCallbackWhenEnabled)
{
    MockLayerExecutor mock;
    mock.setCallSnapshotCallback(true);

    std::vector<std::string> captured;
    mock.setSnapshotCallback([&](const std::string &name, const StageDumpInfo &)
                             { captured.push_back(name); });

    ComputeGraph graph;
    graph.addNode("stage1", std::make_unique<MockComputeStage>(ComputeStageType::GEMM, "stage1"));
    graph.addNode("stage2", std::make_unique<MockComputeStage>(ComputeStageType::RMS_NORM, "stage2"));

    mock.execute(graph, ctx_.get());

    EXPECT_EQ(captured.size(), 2);
}

TEST_F(LayerExecutorInterfaceTest, Mock_NoSnapshotCallbackByDefault)
{
    MockLayerExecutor mock;
    // call_snapshot_callback_ defaults to false

    int callback_count = 0;
    mock.setSnapshotCallback([&](const std::string &, const StageDumpInfo &)
                             { callback_count++; });

    ComputeGraph graph;
    graph.addNode("stage1", std::make_unique<MockComputeStage>());

    mock.execute(graph, ctx_.get());

    // Callback should NOT be called by default
    EXPECT_EQ(callback_count, 0);
}

// =============================================================================
// Function Pointer / Lambda Tests with Interface
// =============================================================================

// Helper to demonstrate accepting ILayerExecutor
void processWithExecutor(ILayerExecutor &executor, ComputeGraph &graph, IDeviceContext *ctx)
{
    executor.execute(graph, ctx);
}

TEST_F(LayerExecutorInterfaceTest, AcceptsMockAsInterface)
{
    MockLayerExecutor mock;
    ComputeGraph graph;
    graph.addNode("A", std::make_unique<MockComputeStage>());

    // This demonstrates passing mock through interface
    processWithExecutor(mock, graph, ctx_.get());

    EXPECT_EQ(mock.executeCount(), 1);
}

TEST_F(LayerExecutorInterfaceTest, AcceptsRealAsInterface)
{
    LayerExecutor real;
    ComputeGraph graph;
    auto stage = std::make_unique<MockComputeStage>();
    stage->setShouldSucceed(true);
    graph.addNode("A", std::move(stage));

    // This demonstrates passing real executor through interface
    processWithExecutor(real, graph, ctx_.get());

    EXPECT_EQ(real.stats().total_stages_executed, 1);
}

// =============================================================================
// ExecutionMode Name Helper Tests
// =============================================================================

TEST(ExecutionModeNameTest, Sequential)
{
    EXPECT_STREQ(executionModeName(ExecutionMode::SEQUENTIAL), "SEQUENTIAL");
}

TEST(ExecutionModeNameTest, Parallel)
{
    EXPECT_STREQ(executionModeName(ExecutionMode::PARALLEL), "PARALLEL");
}

TEST(ExecutionModeNameTest, Pipelined)
{
    EXPECT_STREQ(executionModeName(ExecutionMode::PIPELINED), "PIPELINED");
}

// =============================================================================
// LayerExecutorConfig Tests
// =============================================================================

TEST(LayerExecutorConfigTest, DefaultConfig)
{
    LayerExecutorConfig config;

    EXPECT_EQ(config.mode, ExecutionMode::SEQUENTIAL);
    EXPECT_FALSE(config.enable_profiling);
    EXPECT_FALSE(config.enable_validation);
    EXPECT_EQ(config.default_device, DeviceId::cpu());
    EXPECT_EQ(config.snapshot_callback, nullptr);
}

TEST(LayerExecutorConfigTest, CustomConfig)
{
    LayerExecutorConfig config;
    config.mode = ExecutionMode::PARALLEL;
    config.enable_profiling = true;
    config.enable_validation = true;
    config.default_device = DeviceId::cuda(2);

    EXPECT_EQ(config.mode, ExecutionMode::PARALLEL);
    EXPECT_TRUE(config.enable_profiling);
    EXPECT_TRUE(config.enable_validation);
    EXPECT_EQ(config.default_device, DeviceId::cuda(2));
}

// =============================================================================
// LayerExecutorStats Tests
// =============================================================================

TEST(LayerExecutorStatsTest, DefaultStats)
{
    LayerExecutorStats stats;

    EXPECT_EQ(stats.total_stages_executed, 0);
    EXPECT_EQ(stats.total_flops, 0);
    EXPECT_DOUBLE_EQ(stats.total_time_ms, 0.0);
    EXPECT_TRUE(stats.stage_times_ms.empty());
}

TEST(LayerExecutorStatsTest, Reset)
{
    LayerExecutorStats stats;
    stats.total_stages_executed = 10;
    stats.total_flops = 1000000;
    stats.total_time_ms = 123.456;
    stats.stage_times_ms["stage1"] = 50.0;
    stats.stage_times_ms["stage2"] = 73.456;

    stats.reset();

    EXPECT_EQ(stats.total_stages_executed, 0);
    EXPECT_EQ(stats.total_flops, 0);
    EXPECT_DOUBLE_EQ(stats.total_time_ms, 0.0);
    EXPECT_TRUE(stats.stage_times_ms.empty());
}

// =============================================================================
// Multi-Device Execution Tests
// =============================================================================

TEST_F(LayerExecutorInterfaceTest, MultiDeviceExecute_ThroughInterface)
{
    LayerExecutor real;
    ILayerExecutor &iface = real;

    ComputeGraph graph;
    auto stage = std::make_unique<MockComputeStage>();
    stage->setShouldSucceed(true);
    graph.addNode("A", std::move(stage), DeviceId::cpu()); // Device 0

    std::unordered_map<DeviceId, IDeviceContext *> contexts;
    contexts[DeviceId::cpu()] = ctx_.get();

    // Execute multi-device through interface
    bool result = iface.executeMultiDevice(graph, contexts);

    EXPECT_TRUE(result);
}

TEST_F(LayerExecutorInterfaceTest, MockMultiDeviceExecute_ThroughInterface)
{
    MockLayerExecutor mock;
    ILayerExecutor &iface = mock;

    ComputeGraph graph;
    std::unordered_map<DeviceId, IDeviceContext *> contexts;

    bool result = iface.executeMultiDevice(graph, contexts);

    EXPECT_TRUE(result);
    EXPECT_EQ(mock.multiExecuteCount(), 1);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST_F(LayerExecutorInterfaceTest, CreateMockStage_Helper)
{
    auto stage = createMockStage("test_stage", ComputeStageType::ATTENTION, 1000000, true);

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->name(), "test_stage");
    EXPECT_EQ(stage->type(), ComputeStageType::ATTENTION);
    EXPECT_EQ(stage->estimatedFlops(), 1000000);

    // Verify it succeeds
    stage->execute(ctx_.get());
    EXPECT_EQ(stage->executionCount(), 1);
}

TEST_F(LayerExecutorInterfaceTest, CreateMockStage_Failing)
{
    auto stage = createMockStage("fail_stage", ComputeStageType::GEMM, 0, false);

    bool result = stage->execute(ctx_.get());
    EXPECT_FALSE(result);
}
