/**
 * @file Test__StageRunPolicy.cpp
 * @brief Unit tests for StageRunPolicy and the unified runStages/runStage execution paths
 *
 * Tests that:
 * 1. StageRunPolicy factory methods produce correct flag sets
 * 2. Both full() and fastDecode() policies execute stages in correct order
 * 3. Both policies produce the same output for simple graphs
 * 4. Collective intercept is honoured per-policy
 * 5. Stage failures propagate correctly through both policies
 * 6. Profiling stats are collected under full() but not fastDecode()
 * 7. Snapshot callbacks fire under full() but not fastDecode()
 *
 * @author David Sanftenberg
 * @date July 2025
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/AllreduceStage.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "mocks/MockComputeStage.h"
#include "mocks/MockCollectiveContext.h"

using namespace llaminar2;
using namespace llaminar2::testing;
using llaminar2::test::MockCollectiveContext;

// =============================================================================
// StageRunPolicy Factory Tests
// =============================================================================

TEST(Test__StageRunPolicy, FullPolicy_AllFeaturesEnabled)
{
    auto p = StageRunPolicy::full();
    EXPECT_TRUE(p.coherence);
    EXPECT_TRUE(p.weight_coherence);
    EXPECT_TRUE(p.mark_dirty);
    EXPECT_TRUE(p.validation);
    EXPECT_TRUE(p.profiling);
    EXPECT_TRUE(p.collective_intercept);
    EXPECT_TRUE(p.stage_dump);
    EXPECT_TRUE(p.snapshot_callback);
    EXPECT_TRUE(p.timeline);            // On by default in full (prefill profiling)
    EXPECT_FALSE(p.pointer_validation); // Off by default in full
}

TEST(Test__StageRunPolicy, FastDecodePolicy_MinimalOverhead)
{
    auto p = StageRunPolicy::fastDecode();
    EXPECT_FALSE(p.coherence);
    EXPECT_FALSE(p.weight_coherence);
    // mark_dirty must be ON: it's the lightweight mechanism that sets
    // DEVICE_AUTHORITATIVE on arena outputs after GPU kernels execute,
    // preventing stale H2D uploads if a subsequent full() path reads them.
    EXPECT_TRUE(p.mark_dirty);
    EXPECT_FALSE(p.validation);
    EXPECT_FALSE(p.profiling);
    EXPECT_FALSE(p.stage_dump);
    EXPECT_FALSE(p.snapshot_callback);
    EXPECT_FALSE(p.pointer_validation);

    // collective_intercept and timeline should be ON for fast decode
    EXPECT_TRUE(p.collective_intercept);
    EXPECT_TRUE(p.timeline);
}

TEST(Test__StageRunPolicy, DebugPolicy_EverythingOn)
{
    auto p = StageRunPolicy::debug();
    EXPECT_TRUE(p.coherence);
    EXPECT_TRUE(p.weight_coherence);
    EXPECT_TRUE(p.mark_dirty);
    EXPECT_TRUE(p.validation);
    EXPECT_TRUE(p.profiling);
    EXPECT_TRUE(p.collective_intercept);
    EXPECT_TRUE(p.stage_dump);
    EXPECT_TRUE(p.snapshot_callback);
    EXPECT_TRUE(p.timeline);
    EXPECT_TRUE(p.pointer_validation);
}

// =============================================================================
// Unified Execution Tests
// =============================================================================

class Test__UnifiedExecution : public ::testing::Test
{
protected:
    void SetUp() override
    {
        GraphExecutorConfig config;
        config.enable_profiling = true;
        executor_ = std::make_unique<DeviceGraphExecutor>(config);
        cpu_ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu());
    }

    ComputeGraph buildLinearGraph(int n_stages, std::vector<std::string> *exec_log = nullptr)
    {
        ComputeGraph graph;
        for (int i = 0; i < n_stages; ++i)
        {
            std::string name = "stage_" + std::to_string(i);
            auto stage = std::make_unique<MockComputeStage>(
                ComputeStageType::GEMM, name, DeviceId::cpu());
            if (exec_log)
                stage->setExecutionLog(exec_log);
            mock_stages_.push_back(stage.get());
            graph.addNode(name, std::move(stage), DeviceId::cpu());
            if (i > 0)
            {
                graph.addDependency("stage_" + std::to_string(i),
                                    "stage_" + std::to_string(i - 1));
            }
        }
        return graph;
    }

    std::unique_ptr<DeviceGraphExecutor> executor_;
    std::unique_ptr<CPUDeviceContext> cpu_ctx_;
    std::vector<MockComputeStage *> mock_stages_;
};

TEST_F(Test__UnifiedExecution, FullPolicy_ExecutesAllStagesInOrder)
{
    std::vector<std::string> exec_log;
    auto graph = buildLinearGraph(5, &exec_log);

    bool success = executor_->execute(graph, cpu_ctx_.get());

    ASSERT_TRUE(success);
    ASSERT_EQ(exec_log.size(), 5u);
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(exec_log[i], "stage_" + std::to_string(i));
    }

    // All stages should be marked completed
    EXPECT_TRUE(graph.allCompleted());
}

TEST_F(Test__UnifiedExecution, FastDecodePolicy_ExecutesAllStagesInOrder)
{
    std::vector<std::string> exec_log;
    auto graph = buildLinearGraph(5, &exec_log);

    // Fast decode requires the fast schedule to be built (normally done by caller)
    graph.buildFastSchedule();

    bool success = executor_->executeFastDecode(graph, cpu_ctx_.get());

    ASSERT_TRUE(success);
    ASSERT_EQ(exec_log.size(), 5u);
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(exec_log[i], "stage_" + std::to_string(i));
    }
}

TEST_F(Test__UnifiedExecution, BothPolicies_ProduceSameExecutionOrder)
{
    // Full policy
    std::vector<std::string> full_log;
    auto full_graph = buildLinearGraph(4, &full_log);
    mock_stages_.clear();

    executor_->execute(full_graph, cpu_ctx_.get());

    // Fast decode policy
    std::vector<std::string> fast_log;
    auto fast_graph = buildLinearGraph(4, &fast_log);
    mock_stages_.clear();

    fast_graph.buildFastSchedule();
    executor_->executeFastDecode(fast_graph, cpu_ctx_.get());

    // Same order
    ASSERT_EQ(full_log.size(), fast_log.size());
    for (size_t i = 0; i < full_log.size(); ++i)
    {
        EXPECT_EQ(full_log[i], fast_log[i])
            << "Execution order diverged at index " << i;
    }
}

TEST_F(Test__UnifiedExecution, StageFailure_StopsExecution)
{
    std::vector<std::string> exec_log;
    auto graph = buildLinearGraph(5, &exec_log);

    // Make stage 2 fail
    mock_stages_[2]->setShouldSucceed(false);

    bool success = executor_->execute(graph, cpu_ctx_.get());

    EXPECT_FALSE(success);
    // Only stages 0, 1, 2 should have executed (2 fails, 3+ never run)
    ASSERT_EQ(exec_log.size(), 3u);
    EXPECT_EQ(exec_log[0], "stage_0");
    EXPECT_EQ(exec_log[1], "stage_1");
    EXPECT_EQ(exec_log[2], "stage_2");
}

TEST_F(Test__UnifiedExecution, FastDecode_StageFailure_StopsExecution)
{
    std::vector<std::string> exec_log;
    auto graph = buildLinearGraph(5, &exec_log);
    mock_stages_[2]->setShouldSucceed(false);

    graph.buildFastSchedule();
    bool success = executor_->executeFastDecode(graph, cpu_ctx_.get());

    EXPECT_FALSE(success);
    ASSERT_EQ(exec_log.size(), 3u);
}

TEST_F(Test__UnifiedExecution, EmptyGraph_Succeeds)
{
    ComputeGraph graph;
    bool success = executor_->execute(graph, cpu_ctx_.get());
    EXPECT_TRUE(success);
}

TEST_F(Test__UnifiedExecution, EmptyGraph_FastDecode_Succeeds)
{
    ComputeGraph graph;
    graph.buildFastSchedule();
    bool success = executor_->executeFastDecode(graph, cpu_ctx_.get());
    EXPECT_TRUE(success);
}

TEST_F(Test__UnifiedExecution, SingleStage_Executes)
{
    std::vector<std::string> exec_log;
    auto graph = buildLinearGraph(1, &exec_log);

    bool success = executor_->execute(graph, cpu_ctx_.get());

    ASSERT_TRUE(success);
    ASSERT_EQ(exec_log.size(), 1u);
    EXPECT_EQ(exec_log[0], "stage_0");
}

// =============================================================================
// Profiling Stats Tests
// =============================================================================

TEST_F(Test__UnifiedExecution, FullPolicy_CollectsProfilingStats)
{
    auto graph = buildLinearGraph(3);

    executor_->execute(graph, cpu_ctx_.get());

    const auto &stats = executor_->stats();
    EXPECT_EQ(stats.total_stages_executed, 3u);
    EXPECT_GT(stats.total_time_ms, 0.0);
}

TEST_F(Test__UnifiedExecution, FastDecodePolicy_DoesNotCollectDetailedStats)
{
    auto graph = buildLinearGraph(3);

    // Reset stats
    executor_->resetStats();

    graph.buildFastSchedule();
    executor_->executeFastDecode(graph, cpu_ctx_.get());

    const auto &stats = executor_->stats();
    // Fast decode doesn't collect per-stage profiling stats
    EXPECT_EQ(stats.stage_times_ms.size(), 0u);
}

// =============================================================================
// Snapshot Callback Tests
// =============================================================================

TEST_F(Test__UnifiedExecution, FullPolicy_InvokesSnapshotCallback)
{
    int callback_count = 0;
    std::vector<std::string> callback_stages;

    GraphExecutorConfig config;
    config.enable_profiling = true;
    config.snapshot_callback = [&](const std::string &name, const StageDumpInfo &)
    {
        callback_count++;
        callback_stages.push_back(name);
    };
    executor_ = std::make_unique<DeviceGraphExecutor>(config);

    auto graph = buildLinearGraph(3);
    executor_->execute(graph, cpu_ctx_.get());

    EXPECT_EQ(callback_count, 3);
    ASSERT_EQ(callback_stages.size(), 3u);
    EXPECT_EQ(callback_stages[0], "stage_0");
    EXPECT_EQ(callback_stages[1], "stage_1");
    EXPECT_EQ(callback_stages[2], "stage_2");
}

TEST_F(Test__UnifiedExecution, FastDecodePolicy_SkipsSnapshotCallback)
{
    int callback_count = 0;

    GraphExecutorConfig config;
    config.enable_profiling = true;
    config.snapshot_callback = [&](const std::string &, const StageDumpInfo &)
    {
        callback_count++;
    };
    executor_ = std::make_unique<DeviceGraphExecutor>(config);

    auto graph = buildLinearGraph(3);
    graph.buildFastSchedule();
    executor_->executeFastDecode(graph, cpu_ctx_.get());

    // Fast decode policy has snapshot_callback=false, so callback should NOT fire
    EXPECT_EQ(callback_count, 0);
}

// =============================================================================
// Collective Intercept Tests
// =============================================================================

TEST_F(Test__UnifiedExecution, CollectiveIntercept_FullPolicy)
{
    // Create mock collective context
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    // Create buffer for allreduce
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Build graph: mock → allreduce → mock
    ComputeGraph graph;

    auto stage1 = std::make_unique<MockComputeStage>(
        ComputeStageType::GEMM, "compute", DeviceId::cpu());

    AllreduceStage::Params ar_params;
    ar_params.buffer = buffer.get();
    ar_params.count = 16;
    auto stage2 = std::make_unique<AllreduceStage>(ar_params);

    auto stage3 = std::make_unique<MockComputeStage>(
        ComputeStageType::GEMM, "post_compute", DeviceId::cpu());

    graph.addNode("compute", std::move(stage1), DeviceId::cpu());
    graph.addNode("allreduce", std::move(stage2), DeviceId::cpu());
    graph.addNode("post_compute", std::move(stage3), DeviceId::cpu());
    graph.addDependency("allreduce", "compute");
    graph.addDependency("post_compute", "allreduce");

    // Full policy: collective_intercept=true → routes through MockCollectiveContext
    bool success = executor_->execute(graph, cpu_ctx_.get());
    EXPECT_TRUE(success);
    EXPECT_EQ(mock_ctx->allreduce_call_count(), 1);
}

TEST_F(Test__UnifiedExecution, CollectiveIntercept_FastDecodePolicy)
{
    auto mock_ctx = MockCollectiveContext::Builder()
                        .withWorldSize(2)
                        .withRank(0)
                        .withDevice(DeviceId::cpu())
                        .build();

    executor_->setCollectiveContext(mock_ctx.get());

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    ComputeGraph graph;

    auto stage1 = std::make_unique<MockComputeStage>(
        ComputeStageType::GEMM, "compute", DeviceId::cpu());

    AllreduceStage::Params ar_params;
    ar_params.buffer = buffer.get();
    ar_params.count = 16;
    auto stage2 = std::make_unique<AllreduceStage>(ar_params);

    auto stage3 = std::make_unique<MockComputeStage>(
        ComputeStageType::GEMM, "post_compute", DeviceId::cpu());

    graph.addNode("compute", std::move(stage1), DeviceId::cpu());
    graph.addNode("allreduce", std::move(stage2), DeviceId::cpu());
    graph.addNode("post_compute", std::move(stage3), DeviceId::cpu());
    graph.addDependency("allreduce", "compute");
    graph.addDependency("post_compute", "allreduce");

    std::unordered_set<std::string> collective_nodes = {"allreduce"};
    graph.buildFastSchedule(&collective_nodes);

    // Fast decode policy: collective_intercept=true → routes through MockCollectiveContext
    bool success = executor_->executeFastDecode(graph, cpu_ctx_.get(), &collective_nodes);
    EXPECT_TRUE(success);
    EXPECT_EQ(mock_ctx->allreduce_call_count(), 1);
}

// =============================================================================
// Custom Policy Tests
// =============================================================================

TEST_F(Test__UnifiedExecution, CustomPolicy_CanDisableCollectiveIntercept)
{
    // Verify that StageRunPolicy can be customized
    StageRunPolicy custom;
    custom.collective_intercept = false;
    custom.coherence = false;
    custom.profiling = false;

    EXPECT_FALSE(custom.collective_intercept);
    EXPECT_FALSE(custom.coherence);
    EXPECT_FALSE(custom.profiling);

    // Other flags should retain defaults
    EXPECT_TRUE(custom.validation);
    EXPECT_TRUE(custom.mark_dirty);
    EXPECT_TRUE(custom.stage_dump);
}

// =============================================================================
// Execution Count Verification
// =============================================================================

TEST_F(Test__UnifiedExecution, EachStage_ExecutedExactlyOnce)
{
    auto graph = buildLinearGraph(5);

    executor_->execute(graph, cpu_ctx_.get());

    for (size_t i = 0; i < mock_stages_.size(); ++i)
    {
        EXPECT_EQ(mock_stages_[i]->executionCount(), 1)
            << "Stage " << i << " executed " << mock_stages_[i]->executionCount() << " times";
    }
}

TEST_F(Test__UnifiedExecution, FastDecode_EachStage_ExecutedExactlyOnce)
{
    auto graph = buildLinearGraph(5);
    graph.buildFastSchedule();

    executor_->executeFastDecode(graph, cpu_ctx_.get());

    for (size_t i = 0; i < mock_stages_.size(); ++i)
    {
        EXPECT_EQ(mock_stages_[i]->executionCount(), 1)
            << "Stage " << i << " executed " << mock_stages_[i]->executionCount() << " times";
    }
}

TEST_F(Test__UnifiedExecution, CorrectDeviceContext_PassedToStages)
{
    auto graph = buildLinearGraph(3);

    executor_->execute(graph, cpu_ctx_.get());

    for (auto *stage : mock_stages_)
    {
        EXPECT_EQ(stage->lastContext(), cpu_ctx_.get());
    }
}

TEST_F(Test__UnifiedExecution, FastDecode_CorrectDeviceContext_PassedToStages)
{
    auto graph = buildLinearGraph(3);
    graph.buildFastSchedule();

    executor_->executeFastDecode(graph, cpu_ctx_.get());

    for (auto *stage : mock_stages_)
    {
        EXPECT_EQ(stage->lastContext(), cpu_ctx_.get());
    }
}

// =============================================================================
// Coherence Hardening: mark_dirty is always true (P0/P1)
// =============================================================================

TEST(Test__StageRunPolicy, MarkDirtyAlwaysTrue_AllFactories)
{
    // mark_dirty MUST be true in every factory — it's a correctness invariant,
    // not a tunable overhead knob. Without mark_dirty, outputs stay
    // HOST_AUTHORITATIVE after GPU kernel writes, causing stale reads.
    EXPECT_TRUE(StageRunPolicy::full().mark_dirty);
    EXPECT_TRUE(StageRunPolicy::fastDecode().mark_dirty);
    EXPECT_TRUE(StageRunPolicy::debug().mark_dirty);
}

TEST(Test__StageRunPolicy, MarkDirtyFieldExistsButAlwaysTrue)
{
    // Even if someone constructs a policy manually, mark_dirty defaults to true.
    StageRunPolicy p;
    EXPECT_TRUE(p.mark_dirty);
}
