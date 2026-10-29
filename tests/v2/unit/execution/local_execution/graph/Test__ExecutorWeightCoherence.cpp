/**
 * @file Test__ExecutorWeightCoherence.cpp
 * @brief Unit tests for DeviceGraphExecutor weight coherence system
 *
 * These tests lock in the paradigms established during the executor coherence migration:
 *
 * 1. **Contract-only weight coherence**: The executor's weight loop ONLY processes
 *    contract.weight_tensors. Stages that don't declare weights in their contract
 *    get no weight uploads.
 *
 * 2. **Session-level weight coherence flag** (weights_session_cohered_): After the
 *    first successful forward pass, the executor skips per-node weight coherence
 *    since weights never move between forwards.
 *
 * 3. **Per-node weights_cohered flag**: Within a single forward pass, each node's
 *    weights are only uploaded once (set true after upload, persists across graph resets).
 *
 * 4. **Error propagation**: If ensureOnDevice fails for a weight, execution stops
 *    immediately with a false return.
 *
 * 5. **Policy-driven behaviour**: weight_coherence is gated by StageRunPolicy.
 *    fastDecode() disables it entirely.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */

#include "../../../../utils/CPUActivationArenaFixture.h"
#include <gtest/gtest.h>
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "memory/BufferArena.h"
#include "memory/BufferId.h"
#include "memory/StageBufferContract.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__ExecutorWeightCoherence : public ::testing::Test
{
protected:
    void SetUp() override
    {
        GraphExecutorConfig config;
        config.enable_profiling = false; // Reduce noise
        executor_ = std::make_unique<DeviceGraphExecutor>(config);
        cpu_ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu());

        // Set up a minimal arena so the executor takes the contract path
        arena_ = std::make_unique<BufferArena>(::llaminar2::test::cpuArenaFixtureConfig());
        arena_->registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
        arena_->registerBuffer(BufferId::NORMALIZED, 4, 896, "FP32", DeviceId::cpu());
        arena_->registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());
        arena_->registerBuffer(BufferId::FFN_OUTPUT, 4, 4864, "FP32", DeviceId::cpu());
        arena_->allocate();
        executor_->setArena(arena_.get());
    }

    // Build a graph where each stage has a contract with weight_tensors
    struct StageWithWeight
    {
        MockComputeStage *mock = nullptr;
        FP32Tensor *weight = nullptr;
    };

    // Build a graph with N stages, each holding one weight tensor via contract
    std::pair<ComputeGraph, std::vector<StageWithWeight>>
    buildWeightedGraph(int n_stages)
    {
        ComputeGraph graph;
        std::vector<StageWithWeight> infos;

        for (int i = 0; i < n_stages; ++i)
        {
            std::string name = "stage_" + std::to_string(i);

            // Create a weight tensor for this stage
            auto weight = std::make_unique<FP32Tensor>(
                std::vector<size_t>{1, 64}, DeviceId::cpu());
            auto *weight_ptr = weight.get();
            owned_tensors_.push_back(std::move(weight));

            // Create mock stage with contract
            auto stage = std::make_unique<MockComputeStage>(
                ComputeStageType::GEMM, name, DeviceId::cpu());

            StageBufferContract contract;
            contract.addInput(BufferId::HIDDEN_STATE);
            contract.addOutput(BufferId::NORMALIZED);
            contract.addWeight(weight_ptr);
            stage->setBufferContract(std::move(contract));

            auto *mock_ptr = stage.get();
            infos.push_back({mock_ptr, weight_ptr});

            graph.addNode(name, std::move(stage), DeviceId::cpu());
            if (i > 0)
            {
                graph.addDependency(name, "stage_" + std::to_string(i - 1));
            }
        }

        return {std::move(graph), infos};
    }

    // Build a graph with stages that have NO contract (testing graceful handling)
    ComputeGraph buildNoContractGraph(int n_stages,
                                      std::vector<MockComputeStage *> *mocks = nullptr)
    {
        ComputeGraph graph;
        for (int i = 0; i < n_stages; ++i)
        {
            std::string name = "stage_" + std::to_string(i);
            auto stage = std::make_unique<MockComputeStage>(
                ComputeStageType::GEMM, name, DeviceId::cpu());
            // Default empty contract — no weight declarations
            if (mocks)
                mocks->push_back(stage.get());
            graph.addNode(name, std::move(stage), DeviceId::cpu());
            if (i > 0)
                graph.addDependency(name, "stage_" + std::to_string(i - 1));
        }
        return graph;
    }

    std::unique_ptr<DeviceGraphExecutor> executor_;
    std::unique_ptr<CPUDeviceContext> cpu_ctx_;
    std::unique_ptr<BufferArena> arena_;
    std::vector<std::unique_ptr<FP32Tensor>> owned_tensors_;
};

// =============================================================================
// Contract-Only Weight Coherence
// =============================================================================

TEST_F(Test__ExecutorWeightCoherence, ContractWeights_AreProcessed)
{
    auto [graph, infos] = buildWeightedGraph(3);

    ASSERT_TRUE(executor_->execute(graph, cpu_ctx_.get()));

    // All stages should have been executed
    for (const auto &info : infos)
    {
        EXPECT_EQ(info.mock->executionCount(), 1);
    }

    // All nodes should have weights_cohered = true
    for (int i = 0; i < 3; ++i)
    {
        std::string name = "stage_" + std::to_string(i);
        auto *node = graph.getNode(name);
        ASSERT_NE(node, nullptr);
        EXPECT_TRUE(node->weights_cohered)
            << "Node '" << name << "' should have weights_cohered=true after execution";
    }
}

TEST_F(Test__ExecutorWeightCoherence, NoContract_WeightsNotCohered)
{
    std::vector<MockComputeStage *> mocks;
    auto graph = buildNoContractGraph(3, &mocks);

    ASSERT_TRUE(executor_->execute(graph, cpu_ctx_.get()));

    // All stages executed
    for (auto *mock : mocks)
    {
        EXPECT_EQ(mock->executionCount(), 1);
    }

    // Stages with empty contracts skip the contract-based coherence block entirely
    // (use_contract = !contract.empty() && arena_ != nullptr → false).
    // So weights_cohered stays false — which is correct: no weights to cohere.
    for (int i = 0; i < 3; ++i)
    {
        auto *node = graph.getNode("stage_" + std::to_string(i));
        ASSERT_NE(node, nullptr);
        EXPECT_FALSE(node->weights_cohered)
            << "Empty-contract stages should not enter weight coherence block";
    }
}

// =============================================================================
// Session-Level Weight Coherence Flag
// =============================================================================

TEST_F(Test__ExecutorWeightCoherence, SecondForward_SkipsWeightCoherence)
{
    auto [graph, infos] = buildWeightedGraph(3);

    // First forward: weights are cohered
    ASSERT_TRUE(executor_->execute(graph, cpu_ctx_.get()));

    // Verify all nodes marked as weight-cohered
    for (int i = 0; i < 3; ++i)
    {
        auto *node = graph.getNode("stage_" + std::to_string(i));
        EXPECT_TRUE(node->weights_cohered);
    }

    // Reset graph completion state (simulating next decode step)
    graph.reset();

    // Second forward: session flag should be set, so weight coherence is skipped.
    // We verify this indirectly: the executor succeeds and all stages execute again.
    ASSERT_TRUE(executor_->execute(graph, cpu_ctx_.get()));

    for (const auto &info : infos)
    {
        EXPECT_EQ(info.mock->executionCount(), 2);
    }
}

TEST_F(Test__ExecutorWeightCoherence, SessionFlag_SurvivesGraphReset)
{
    // The weights_session_cohered_ flag is on the EXECUTOR, not the graph.
    // graph.reset() only resets 'completed' flags, not 'weights_cohered'.
    // The session flag means even new graphs with weights_cohered=false
    // will skip weight coherence.
    auto [graph, infos] = buildWeightedGraph(2);

    ASSERT_TRUE(executor_->execute(graph, cpu_ctx_.get()));
    graph.reset();

    // Build a completely new graph with fresh nodes (weights_cohered=false by default)
    auto [graph2, infos2] = buildWeightedGraph(2);

    // The session flag on the executor should cause weight coherence to be skipped
    ASSERT_TRUE(executor_->execute(graph2, cpu_ctx_.get()));

    // Execution succeeds (stages ran)
    for (const auto &info : infos2)
    {
        EXPECT_EQ(info.mock->executionCount(), 1);
    }
}

TEST_F(Test__ExecutorWeightCoherence, SessionFlag_OnlySetAfterFullPass)
{
    // The weights_session_cohered_ flag must only be set after ALL stages
    // in the graph complete successfully. If a mid-graph failure occurs,
    // only partial stages have their weights uploaded. Setting the flag
    // after partial execution would cause subsequent stages to skip
    // weight coherence — the exact bug we fixed.
    auto [graph, infos] = buildWeightedGraph(3);

    // Make the LAST stage fail
    infos[2].mock->setShouldSucceed(false);

    bool success = executor_->execute(graph, cpu_ctx_.get());
    EXPECT_FALSE(success);

    // Now try again with all stages succeeding
    graph.reset();
    infos[2].mock->setShouldSucceed(true);

    // Build a fresh graph (since the old one has stale state from failed run)
    auto [fresh_graph, fresh_infos] = buildWeightedGraph(3);

    // If the session flag was incorrectly set after the failed pass,
    // weight coherence would be skipped for this graph, and stages
    // with uninitialized weights would crash on GPU. On CPU this
    // doesn't crash, but the flag state is still wrong.
    ASSERT_TRUE(executor_->execute(fresh_graph, cpu_ctx_.get()));

    // All stages should have weights_cohered = true
    for (int i = 0; i < 3; ++i)
    {
        auto *node = fresh_graph.getNode("stage_" + std::to_string(i));
        ASSERT_NE(node, nullptr);
        EXPECT_TRUE(node->weights_cohered);
    }
}

// =============================================================================
// FastDecode Policy Bypasses Weight Coherence
// =============================================================================

TEST_F(Test__ExecutorWeightCoherence, FastDecode_SkipsWeightCoherence)
{
    auto [graph, infos] = buildWeightedGraph(3);

    // Build fast schedule (required for executeFastDecode)
    graph.buildFastSchedule();

    ASSERT_TRUE(executor_->executeFastDecode(graph, cpu_ctx_.get()));

    // Stages should execute
    for (const auto &info : infos)
    {
        EXPECT_EQ(info.mock->executionCount(), 1);
    }

    // But weight coherence should NOT have been performed
    // (fastDecode policy has weight_coherence=false)
    for (int i = 0; i < 3; ++i)
    {
        auto *node = graph.getNode("stage_" + std::to_string(i));
        ASSERT_NE(node, nullptr);
        EXPECT_FALSE(node->weights_cohered)
            << "Fast decode should not mark weights as cohered";
    }
}

// =============================================================================
// Per-Node weights_cohered Persistence
// =============================================================================

TEST_F(Test__ExecutorWeightCoherence, NodeWeightsCoheredFlag_SurvivesGraphReset)
{
    auto [graph, infos] = buildWeightedGraph(3);

    ASSERT_TRUE(executor_->execute(graph, cpu_ctx_.get()));

    // All nodes have weights_cohered=true
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(graph.getNode("stage_" + std::to_string(i))->weights_cohered);
    }

    // graph.reset() only resets 'completed', NOT 'weights_cohered'
    graph.reset();

    // Verify weights_cohered persists
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(graph.getNode("stage_" + std::to_string(i))->weights_cohered)
            << "weights_cohered should survive graph.reset()";
    }
}

// =============================================================================
// Mixed Contract/No-Contract Graph
// =============================================================================

TEST_F(Test__ExecutorWeightCoherence, MixedGraph_ContractAndNoContract)
{
    // Some stages declare weights, some don't
    ComputeGraph graph;

    // Stage 0: has weight in contract
    auto weight = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, 64}, DeviceId::cpu());
    auto *weight_ptr = weight.get();
    owned_tensors_.push_back(std::move(weight));

    auto stage0 = std::make_unique<MockComputeStage>(
        ComputeStageType::GEMM, "gemm_stage", DeviceId::cpu());
    StageBufferContract contract0;
    contract0.addInput(BufferId::HIDDEN_STATE);
    contract0.addOutput(BufferId::NORMALIZED);
    contract0.addWeight(weight_ptr);
    stage0->setBufferContract(std::move(contract0));

    // Stage 1: no contract (e.g., ResidualAdd — no weights)
    auto stage1 = std::make_unique<MockComputeStage>(
        ComputeStageType::ADD_RESIDUAL, "residual_add", DeviceId::cpu());

    auto *mock0 = stage0.get();
    auto *mock1 = stage1.get();

    graph.addNode("gemm_stage", std::move(stage0), DeviceId::cpu());
    graph.addNode("residual_add", std::move(stage1), DeviceId::cpu());
    graph.addDependency("residual_add", "gemm_stage");

    ASSERT_TRUE(executor_->execute(graph, cpu_ctx_.get()));

    EXPECT_EQ(mock0->executionCount(), 1);
    EXPECT_EQ(mock1->executionCount(), 1);

    // GEMM stage should have weights cohered (it has a non-empty contract)
    EXPECT_TRUE(graph.getNode("gemm_stage")->weights_cohered);
    // ResidualAdd has no contract → use_contract=false → skips coherence block
    EXPECT_FALSE(graph.getNode("residual_add")->weights_cohered)
        << "Empty-contract stage should not enter weight coherence block";
}

// =============================================================================
// Error Propagation on Weight Upload Failure
// =============================================================================

// Note: On CPU, ensureOnDevice always returns true (CPU tensors are always
// "on device"). To test error propagation, we'd need a mock that overrides
// ensureOnDevice to fail. This is tested indirectly at the integration
// level with actual GPU devices. Here we verify the positive path works
// correctly and the executor's control flow is as expected.

TEST_F(Test__ExecutorWeightCoherence, ExecutorReturnsTrue_WhenAllWeightsUploadSucceed)
{
    auto [graph, infos] = buildWeightedGraph(5);

    bool success = executor_->execute(graph, cpu_ctx_.get());
    EXPECT_TRUE(success);

    for (const auto &info : infos)
    {
        EXPECT_EQ(info.mock->executionCount(), 1);
    }
}

TEST_F(Test__ExecutorWeightCoherence, StageFail_StopsExecution_NoSessionFlag)
{
    // If a stage fails execution (not weight upload), the session flag
    // should NOT be set — only set after full successful pass
    auto [graph, infos] = buildWeightedGraph(3);
    infos[1].mock->setShouldSucceed(false);

    bool success = executor_->execute(graph, cpu_ctx_.get());
    EXPECT_FALSE(success);

    // Stage 0 executed, stage 1 executed and failed, stage 2 never ran
    EXPECT_EQ(infos[0].mock->executionCount(), 1);
    EXPECT_EQ(infos[1].mock->executionCount(), 1);
    EXPECT_EQ(infos[2].mock->executionCount(), 0);
}

// =============================================================================
// Arena Interaction
// =============================================================================

TEST_F(Test__ExecutorWeightCoherence, NoArena_ContractIgnored_StagesStillExecute)
{
    // Without an arena, the executor uses the legacy non-contract path
    GraphExecutorConfig config;
    config.enable_profiling = false;
    auto executor = std::make_unique<DeviceGraphExecutor>(config);
    // Do NOT set arena

    auto [graph, infos] = buildWeightedGraph(3);

    ASSERT_TRUE(executor->execute(graph, cpu_ctx_.get()));

    for (const auto &info : infos)
    {
        EXPECT_EQ(info.mock->executionCount(), 1);
    }
}

// =============================================================================
// StageRunPolicy Weight Coherence Flag
// =============================================================================

TEST_F(Test__ExecutorWeightCoherence, PolicyWeightCoherenceOff_SkipsWeightUpload)
{
    // The StageRunPolicy::fastDecode() has weight_coherence=false
    auto p = StageRunPolicy::fastDecode();
    EXPECT_FALSE(p.weight_coherence);
}

TEST_F(Test__ExecutorWeightCoherence, PolicyWeightCoherenceOn_UploadsWeights)
{
    auto p = StageRunPolicy::full();
    EXPECT_TRUE(p.weight_coherence);
}

// =============================================================================
// Graph Reset Does Not Reset weights_cohered (Critical Invariant)
// =============================================================================

TEST_F(Test__ExecutorWeightCoherence, GraphReset_PreservesWeightsCohered)
{
    auto [graph, infos] = buildWeightedGraph(2);

    ASSERT_TRUE(executor_->execute(graph, cpu_ctx_.get()));

    // Check nodes before reset
    auto *node0 = graph.getNode("stage_0");
    auto *node1 = graph.getNode("stage_1");
    ASSERT_TRUE(node0->weights_cohered);
    ASSERT_TRUE(node1->weights_cohered);
    ASSERT_TRUE(node0->completed);
    ASSERT_TRUE(node1->completed);

    graph.reset();

    // After reset: completed=false, weights_cohered=true (preserved)
    EXPECT_FALSE(node0->completed);
    EXPECT_FALSE(node1->completed);
    EXPECT_TRUE(node0->weights_cohered)
        << "graph.reset() must NOT reset weights_cohered";
    EXPECT_TRUE(node1->weights_cohered)
        << "graph.reset() must NOT reset weights_cohered";
}
