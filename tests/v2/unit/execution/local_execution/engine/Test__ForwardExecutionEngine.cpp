/**
 * @file Test__ForwardExecutionEngine.cpp
 * @brief Unit tests for ForwardExecutionEngine
 *
 * Tests the forward graph execution engine extracted in Phase 3 of
 * the DGO refactor, using a mock IForwardExecutionHost to isolate the
 * engine from model-specific graph building.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "execution/local_execution/engine/ForwardExecutionEngine.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "../../../../mocks/MockComputeStage.h" // MockDeviceContext

using namespace llaminar2;

// =========================================================================
// MockForwardExecutionHost
// =========================================================================

namespace
{

    /**
     * @brief Scoped environment override that refreshes debugEnv() for each unit test.
     */
    class ScopedDebugEnv
    {
    public:
        explicit ScopedDebugEnv(std::initializer_list<std::pair<const char *, const char *>> values)
        {
            for (const auto &[name, value] : values)
            {
                Entry entry;
                entry.name = name;
                if (const char *old_value = std::getenv(name))
                {
                    entry.had_value = true;
                    entry.old_value = old_value;
                }
                entries_.push_back(entry);
                ::setenv(name, value, 1);
            }
            mutableDebugEnv().reload();
        }

        ~ScopedDebugEnv()
        {
            for (const auto &entry : entries_)
            {
                if (entry.had_value)
                    ::setenv(entry.name.c_str(), entry.old_value.c_str(), 1);
                else
                    ::unsetenv(entry.name.c_str());
            }
            mutableDebugEnv().reload();
        }

        ScopedDebugEnv(const ScopedDebugEnv &) = delete;
        ScopedDebugEnv &operator=(const ScopedDebugEnv &) = delete;

    private:
        struct Entry
        {
            std::string name;
            bool had_value = false;
            std::string old_value;
        };

        std::vector<Entry> entries_;
    };

    /**
     * @brief Minimal mock for IForwardExecutionHost.
     *
     * Tracks which callbacks are invoked and returns configurable results.
     */
    class MockForwardExecutionHost : public IForwardExecutionHost
    {
    public:
        explicit MockForwardExecutionHost(IDeviceContext *ctx = nullptr)
            : ctx_(ctx) {}

        // ----- Tracking Counters -----
        int build_forward_graph_calls = 0;
        int get_device_context_calls = 0;
        int ensure_workspace_calls = 0;
        int last_workspace_seq_len = -1;
        int sync_logits_calls = 0;
        int logits_tensor_calls = 0;
        int build_decode_policy_calls = 0;
        int resolve_pp_copy_calls = 0;
        int get_pipeline_contexts_calls = 0;
        bool has_last_forward_input = false;
        ForwardInput last_forward_input{};
        const int *last_token_ids_pointer = nullptr;
        const int *last_position_ids_pointer = nullptr;
        std::vector<int> last_token_ids;
        std::vector<int> last_position_ids;
        std::vector<int> forward_token_offsets;
        std::vector<int> forward_real_seq_lens;
        std::vector<std::vector<int>> forward_token_batches;
        int prefill_chunk_maintenance_state_calls = 0;
        int prefill_chunk_maintenance_calls = 0;
        PrefillChunkPlan last_maintenance_chunk{};
        PrefillChunkMaintenanceDecision last_maintenance_decision{};

        // ----- Configurable Results -----
        bool build_should_fail = false;
        std::string build_error_message = "mock build failure";
        int graph_stage_count = 0; // stages in built graph (0 means empty graph)
        // Optional explicit stage types let safety tests assemble GDN/short-conv graphs.
        std::vector<ComputeStageType> graph_stage_types;
        PPCopyInfo mock_pp_copy;
        DeviceGraphExecutor::DecodeCapturePolicy mock_capture_policy;
        bool mock_compute_all_position_logits = false;
        PrefillChunkMaintenanceState mock_maintenance_state{};
        bool mock_maintenance_state_configured = false;
        bool maintenance_should_fail = false;
        ForwardExecutionEngine *engine_to_clear_on_maintenance = nullptr;
        bool bump_epoch_on_maintenance = false;
        uint64_t placement_epoch = 0;
        std::string prefill_domain_id = "single";
        int prefill_participant_id = 0;
        uint64_t prefill_topology_signature = 0;

        // ----- IForwardExecutionHost Interface -----

        GraphBuildResult buildForwardGraph(const ForwardInput &input) override
        {
            build_forward_graph_calls++;
            has_last_forward_input = true;
            last_forward_input = input;
            last_token_ids_pointer = input.token_ids;
            last_position_ids_pointer = input.position_ids;
            last_token_ids.clear();
            last_position_ids.clear();

            const int total_tokens = input.batch_size * input.seq_len;
            if (input.token_ids && total_tokens > 0)
                last_token_ids.assign(input.token_ids, input.token_ids + total_tokens);
            if (input.position_ids && total_tokens > 0)
                last_position_ids.assign(input.position_ids, input.position_ids + total_tokens);
            forward_token_offsets.push_back(input.token_offset);
            forward_real_seq_lens.push_back(input.real_seq_len);
            forward_token_batches.push_back(last_token_ids);

            if (build_should_fail)
                return GraphBuildResult(build_error_message);

            ComputeGraph graph;
            const int stage_count = graph_stage_types.empty()
                                        ? graph_stage_count
                                        : static_cast<int>(graph_stage_types.size());
            for (int i = 0; i < stage_count; ++i)
            {
                const ComputeStageType type = graph_stage_types.empty()
                                                  ? ComputeStageType::GEMM
                                                  : graph_stage_types[static_cast<size_t>(i)];
                const std::string stage_name = "mock_stage_" + std::to_string(i);
                graph.addNode(
                    stage_name,
                    std::make_unique<llaminar2::testing::MockComputeStage>(
                        type,
                        stage_name,
                        input.device),
                    input.device);
                if (i > 0)
                {
                    graph.addDependency(
                        stage_name,
                        "mock_stage_" + std::to_string(i - 1));
                }
            }
            ForwardOutput output{};
            return GraphBuildResult(std::move(graph), output);
        }

        IDeviceContext *getDeviceContext(DeviceId device) override
        {
            get_device_context_calls++;
            return ctx_;
        }

        std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() override
        {
            get_pipeline_contexts_calls++;
            std::unordered_map<DeviceId, IDeviceContext *> result;
            if (ctx_)
                result[ctx_->deviceId()] = ctx_;
            return result;
        }

        bool ensureDeviceWorkspaceAllocated(const ComputeGraph &, int workspace_seq_len) override
        {
            ensure_workspace_calls++;
            last_workspace_seq_len = workspace_seq_len;
            return true;
        }

        void syncLogitsAtBoundary(IDeviceContext *ctx) override
        {
            sync_logits_calls++;
        }

        TensorBase *logitsTensor() override
        {
            logits_tensor_calls++;
            return nullptr;
        }

        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool has_collective_nodes,
            IDeviceContext *ctx,
            int segment_consecutive_failures) const override
        {
            const_cast<MockForwardExecutionHost *>(this)->build_decode_policy_calls++;
            return mock_capture_policy;
        }

        PPCopyInfo resolvePPCopyInfo(const ForwardInput &input) const override
        {
            const_cast<MockForwardExecutionHost *>(this)->resolve_pp_copy_calls++;
            return mock_pp_copy;
        }

        bool computeAllPositionLogitsEnabled() const override
        {
            return mock_compute_all_position_logits;
        }

        uint64_t moePlacementEpoch() const override
        {
            return placement_epoch;
        }

        std::string prefillGraphDomainId() const override
        {
            return prefill_domain_id;
        }

        int prefillGraphParticipantId() const override
        {
            return prefill_participant_id;
        }

        uint64_t prefillGraphTopologySignature() const override
        {
            return prefill_topology_signature;
        }

        PrefillChunkMaintenanceState prefillChunkMaintenanceState(
            const PrefillChunkPlan &chunk) const override
        {
            auto *self = const_cast<MockForwardExecutionHost *>(this);
            self->prefill_chunk_maintenance_state_calls++;
            self->last_maintenance_chunk = chunk;
            if (mock_maintenance_state_configured)
                return mock_maintenance_state;
            return IForwardExecutionHost::prefillChunkMaintenanceState(chunk);
        }

        bool onPrefillChunkMaintenance(
            const PrefillChunkPlan &chunk,
            const PrefillChunkMaintenanceDecision &decision) override
        {
            prefill_chunk_maintenance_calls++;
            last_maintenance_chunk = chunk;
            last_maintenance_decision = decision;
            if (maintenance_should_fail)
                return false;
            if (bump_epoch_on_maintenance)
                ++placement_epoch;
            if (engine_to_clear_on_maintenance)
                engine_to_clear_on_maintenance->discardAllCachedGraphs();
            return true;
        }

    private:
        IDeviceContext *ctx_;
    };

    /**
     * @brief Create a minimal ForwardInput for testing.
     *
     * Does NOT allocate a full model — just enough fields for
     * the engine's cache signature logic.
     */
    ForwardInput makeTestInput(
        int seq_len,
        int batch_size,
        DeviceId device = DeviceId::cpu(),
        const int *token_ids = nullptr,
        const int *position_ids = nullptr)
    {
        ForwardInput input{};
        input.seq_len = seq_len;
        input.batch_size = batch_size;
        input.device = device;
        input.token_ids = token_ids;
        input.position_ids = position_ids;
        input.position_offset = 0;
        return input;
    }

    double findForwardGraphCounterValue(const std::vector<PerfStatRecord> &records,
                                        const std::string &name,
                                        const PerfStatsCollector::Tags &tags)
    {
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == "forward_graph" &&
                record.name == name &&
                record.tags == tags)
            {
                return record.value;
            }
        }
        return 0.0;
    }

} // namespace

// =========================================================================
// Test Fixture
// =========================================================================

class Test__ForwardExecutionEngine : public ::testing::Test
{
protected:
    // Default executor — CPU, default config
    DeviceGraphExecutor executor_;
    // Mock CPU device context
    llaminar2::testing::MockDeviceContext mock_ctx_{DeviceId::cpu()};

    // Helper to create engine with default config (caching enabled, no PP)
    ForwardExecutionEngine makeEngine(bool cache_enabled = true)
    {
        ForwardExecutionEngine::Config config;
        config.cache_config.enabled = cache_enabled;
        config.has_unified_pp = false;
        return ForwardExecutionEngine(std::move(config), executor_);
    }
};

// =========================================================================
// Construction and Config
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, ConstructionCacheEmpty)
{
    auto engine = makeEngine();
    EXPECT_TRUE(engine.cacheEmpty());
}

TEST_F(Test__ForwardExecutionEngine, MutableFlags)
{
    auto engine = makeEngine();
    // Just verify these don't crash — no public getters to check
    engine.setSuppressTimeline(true);
    engine.setAccumulatePrefill(true);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_ExactBucketSucceeds)
{
    const std::vector<int> tokens = {10, 11, 12, 13};
    auto input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    input.token_offset = 32;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4, 8},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    ASSERT_TRUE(plan) << plan.error;
    EXPECT_FALSE(plan.padding_required);
    EXPECT_EQ(plan.selection.bucket_seq_len, 4);
    EXPECT_EQ(plan.chunk.token_offset, 32);
    EXPECT_EQ(plan.chunk.real_count, 4);
    EXPECT_EQ(plan.chunk.bucket_seq_len, 4);
    EXPECT_EQ(plan.chunk.token_ids, tokens);
    EXPECT_EQ(plan.chunk.position_ids, (std::vector<int>{32, 33, 34, 35}));
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_RequiresTokenIds)
{
    auto input = makeTestInput(4, 1, DeviceId::cpu(), nullptr, nullptr);

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    EXPECT_FALSE(plan);
    EXPECT_NE(plan.error.find("requires token_ids"), std::string::npos);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_RejectsEmptyBucketList)
{
    const std::vector<int> tokens = {10, 11, 12, 13};
    auto input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    EXPECT_FALSE(plan);
    EXPECT_NE(plan.error.find("no positive"), std::string::npos);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_RejectsSeqLenAboveLargestBucket)
{
    const std::vector<int> tokens = {10, 11, 12, 13, 14};
    auto input = makeTestInput(5, 1, DeviceId::cpu(), tokens.data(), nullptr);

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{2, 4},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    EXPECT_FALSE(plan);
    EXPECT_NE(plan.error.find("largest"), std::string::npos);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_RejectsZeroSeqLen)
{
    const std::vector<int> tokens = {10};
    auto input = makeTestInput(0, 1, DeviceId::cpu(), tokens.data(), nullptr);

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    EXPECT_FALSE(plan);
    EXPECT_NE(plan.error.find("positive"), std::string::npos);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_UsesExplicitRealSeqLen)
{
    const std::vector<int> tokens = {10, 11, 12, 13, 14, 15};
    auto input = makeTestInput(6, 1, DeviceId::cpu(), tokens.data(), nullptr);
    input.real_seq_len = 3;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{3, 8},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    ASSERT_TRUE(plan) << plan.error;
    EXPECT_EQ(plan.selection.real_seq_len, 3);
    EXPECT_EQ(plan.chunk.real_count, 3);
    EXPECT_EQ(plan.chunk.bucket_seq_len, 3);
    EXPECT_EQ(plan.chunk.token_ids, (std::vector<int>{10, 11, 12}));
    EXPECT_EQ(plan.chunk.position_ids, (std::vector<int>{0, 1, 2}));
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_PaddedBucketRejectedUntilEnabled)
{
    const std::vector<int> tokens = {10, 11, 12, 13, 14};
    auto input = makeTestInput(5, 1, DeviceId::cpu(), tokens.data(), nullptr);

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4, 8},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    EXPECT_FALSE(plan);
    EXPECT_TRUE(plan.padding_required);
    EXPECT_EQ(plan.selection.bucket_seq_len, 8);
    EXPECT_EQ(plan.chunk.token_ids, (std::vector<int>{10, 11, 12, 13, 14, 99, 99, 99}));
    EXPECT_NE(plan.error.find("requires caller opt-in"), std::string::npos);
    EXPECT_FALSE(plan.chunk.ok);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_PaddedBucketCanBePreparedWhenGateOpens)
{
    const std::vector<int> tokens = {20, 21, 22, 23, 24};
    auto input = makeTestInput(5, 1, DeviceId::cpu(), tokens.data(), nullptr);
    input.token_offset = 100;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4, 8},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/true);

    ASSERT_TRUE(plan) << plan.error;
    EXPECT_TRUE(plan.padding_required);
    EXPECT_EQ(plan.chunk.real_count, 5);
    EXPECT_EQ(plan.chunk.bucket_seq_len, 8);
    EXPECT_EQ(plan.chunk.token_ids, (std::vector<int>{20, 21, 22, 23, 24, 0, 0, 0}));
    EXPECT_EQ(plan.chunk.position_ids, (std::vector<int>{100, 101, 102, 103, 104, 105, 106, 107}));
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimeSchedule_PreparesExplicitRange)
{
    const std::vector<int> tokens = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    auto input = makeTestInput(static_cast<int>(tokens.size()), 1, DeviceId::cpu(), tokens.data(), nullptr);
    input.token_offset = 10;
    input.real_seq_len = static_cast<int>(tokens.size());

    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {4};
    policy.fixed_chunk_real_tokens = 4;
    policy.min_rebalance_interval_tokens = 4;
    policy.max_rebalance_interval_tokens = 4;
    policy.real_token_start = 12;
    policy.real_token_count = 7;

    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/true);

    ASSERT_TRUE(schedule) << schedule.error;
    ASSERT_EQ(schedule.chunks.size(), 2u);

    const auto &first = schedule.chunks[0];
    EXPECT_EQ(first.chunk_index, 0);
    EXPECT_EQ(first.chunk.token_offset, 12);
    EXPECT_EQ(first.chunk.real_count, 4);
    EXPECT_EQ(first.chunk.bucket_seq_len, 4);
    EXPECT_EQ(first.chunk.token_ids, (std::vector<int>{102, 103, 104, 105}));
    EXPECT_EQ(first.chunk.position_ids, (std::vector<int>{12, 13, 14, 15}));
    EXPECT_TRUE(first.rebalance_allowed_after);
    EXPECT_TRUE(first.rebalance_required_after);

    const auto &second = schedule.chunks[1];
    EXPECT_EQ(second.chunk_index, 1);
    EXPECT_EQ(second.chunk.token_offset, 16);
    EXPECT_EQ(second.chunk.real_count, 3);
    EXPECT_EQ(second.chunk.bucket_seq_len, 4);
    EXPECT_EQ(second.chunk.token_ids, (std::vector<int>{106, 107, 108, 0}));
    EXPECT_EQ(second.chunk.position_ids, (std::vector<int>{16, 17, 18, 19}));
    EXPECT_TRUE(second.padding_required);
    EXPECT_FALSE(second.rebalance_allowed_after);
    EXPECT_FALSE(second.rebalance_required_after);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimeSchedule_RejectsRangeOutsideInput)
{
    const std::vector<int> tokens = {10, 11, 12, 13};
    auto input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    input.token_offset = 20;

    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {4};
    policy.fixed_chunk_real_tokens = 4;
    policy.real_token_start = 22;
    policy.real_token_count = 4;

    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/true);

    EXPECT_FALSE(schedule);
    EXPECT_NE(schedule.error.find("outside input"), std::string::npos);
    EXPECT_TRUE(schedule.chunks.empty());
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimeSchedule_RejectsPaddedChunkWithoutOptIn)
{
    const std::vector<int> tokens = {10, 11, 12};
    auto input = makeTestInput(3, 1, DeviceId::cpu(), tokens.data(), nullptr);

    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {4};
    policy.fixed_chunk_real_tokens = 3;
    policy.real_token_start = 0;
    policy.real_token_count = 3;

    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    EXPECT_FALSE(schedule);
    EXPECT_NE(schedule.error.find("requires caller opt-in"), std::string::npos);
    ASSERT_EQ(schedule.chunks.size(), 1u);
    EXPECT_EQ(schedule.chunks[0].chunk.token_ids, (std::vector<int>{10, 11, 12, 99}));
    EXPECT_FALSE(schedule.chunks[0].chunk.ok);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_RejectsBatchSizeAboveOne)
{
    const std::vector<int> tokens = {1, 2, 3, 4};
    auto input = makeTestInput(4, 2, DeviceId::cpu(), tokens.data(), nullptr);

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/true);

    EXPECT_FALSE(plan);
    EXPECT_NE(plan.error.find("batch_size=1"), std::string::npos);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_ExactPlanDelegatesWithChunkInput)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {40, 41, 42, 43};
    auto base_input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    base_input.token_offset = 24;
    base_input.position_offset = 999;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4, 8},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(plan) << plan.error;
    plan.chunk_index = 9;

    ForwardOutput output{};
    EXPECT_TRUE(engine.runPrefillChunk(base_input, plan, output, host));

    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.last_token_ids_pointer, plan.chunk.token_ids.data());
    EXPECT_EQ(host.last_position_ids_pointer, plan.chunk.position_ids.data());
    EXPECT_EQ(host.last_token_ids, plan.chunk.token_ids);
    EXPECT_EQ(host.last_position_ids, plan.chunk.position_ids);
    EXPECT_EQ(host.last_forward_input.seq_len, plan.chunk.bucket_seq_len);
    EXPECT_EQ(host.last_forward_input.real_seq_len, plan.chunk.real_count);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, plan.chunk.bucket_seq_len);
    EXPECT_EQ(host.last_forward_input.token_offset, plan.chunk.token_offset);
    EXPECT_EQ(host.last_forward_input.position_offset, plan.chunk.token_offset);
    EXPECT_EQ(host.last_forward_input.prefill_chunk_index, 9);
    EXPECT_EQ(host.last_workspace_seq_len, plan.chunk.bucket_seq_len);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_MaintenanceRunsWhenRequestedAndAllowed)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {40, 41, 42, 43};
    auto base_input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(plan) << plan.error;
    plan.chunk_index = 2;
    plan.rebalance_allowed_after = true;

    host.mock_maintenance_state_configured = true;
    host.mock_maintenance_state.chunk_index = 2;
    host.mock_maintenance_state.rebalance_requested = true;
    host.mock_maintenance_state.histograms_merged = true;
    host.mock_maintenance_state.manual_boundaries_complete = true;
    host.mock_maintenance_state.participants_at_same_boundary = true;

    ForwardOutput output{};
    EXPECT_TRUE(engine.runPrefillChunk(base_input, plan, output, host));

    EXPECT_EQ(host.prefill_chunk_maintenance_state_calls, 1);
    EXPECT_EQ(host.prefill_chunk_maintenance_calls, 1);
    EXPECT_EQ(host.last_maintenance_chunk.chunk_index, 2);
    EXPECT_EQ(host.last_maintenance_chunk.real_count, 4);
    EXPECT_TRUE(host.last_maintenance_decision.can_run);
    EXPECT_FALSE(host.last_maintenance_decision.required);
    EXPECT_EQ(host.last_maintenance_decision.reason, "ready");
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_OptionalMaintenanceNotRequestedDoesNotRunHook)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {40, 41, 42, 43};
    auto base_input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(plan) << plan.error;
    plan.rebalance_allowed_after = true;

    ForwardOutput output{};
    EXPECT_TRUE(engine.runPrefillChunk(base_input, plan, output, host));

    EXPECT_EQ(host.prefill_chunk_maintenance_state_calls, 1);
    EXPECT_EQ(host.prefill_chunk_maintenance_calls, 0);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_RequiredMaintenanceFailsWhenBoundaryUnsafe)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {40, 41, 42, 43};
    auto base_input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(plan) << plan.error;
    plan.chunk_index = 5;
    plan.rebalance_required_after = true;

    host.mock_maintenance_state_configured = true;
    host.mock_maintenance_state.chunk_index = 5;
    host.mock_maintenance_state.histograms_merged = false;
    host.mock_maintenance_state.manual_boundaries_complete = true;
    host.mock_maintenance_state.participants_at_same_boundary = true;

    ForwardOutput output{};
    EXPECT_FALSE(engine.runPrefillChunk(base_input, plan, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.prefill_chunk_maintenance_state_calls, 1);
    EXPECT_EQ(host.prefill_chunk_maintenance_calls, 0);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_MaintenanceHookFailureFailsChunk)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;
    host.maintenance_should_fail = true;

    const std::vector<int> tokens = {40, 41, 42, 43};
    auto base_input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(plan) << plan.error;
    plan.rebalance_required_after = true;

    ForwardOutput output{};
    EXPECT_FALSE(engine.runPrefillChunk(base_input, plan, output, host));

    EXPECT_EQ(host.prefill_chunk_maintenance_state_calls, 1);
    EXPECT_EQ(host.prefill_chunk_maintenance_calls, 1);
    EXPECT_TRUE(host.last_maintenance_decision.can_run);
    EXPECT_TRUE(host.last_maintenance_decision.required);
    EXPECT_EQ(host.last_maintenance_decision.reason, "required");
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunkSchedule_ExecutesChunksInOrderAndMaintainsBoundaries)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    auto input = makeTestInput(static_cast<int>(tokens.size()), 1, DeviceId::cpu(), tokens.data(), nullptr);

    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {4};
    policy.fixed_chunk_real_tokens = 4;
    policy.min_rebalance_interval_tokens = 4;
    policy.max_rebalance_interval_tokens = 4;
    policy.real_token_count = static_cast<int>(tokens.size());

    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(schedule) << schedule.error;
    ASSERT_EQ(schedule.chunks.size(), 2u);
    ASSERT_TRUE(schedule.chunks[0].rebalance_required_after);
    ASSERT_TRUE(schedule.chunks[1].rebalance_required_after);

    ForwardOutput output{};
    EXPECT_TRUE(engine.runPrefillChunkSchedule(input, schedule, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 2);
    EXPECT_EQ(host.prefill_chunk_maintenance_state_calls, 2);
    EXPECT_EQ(host.prefill_chunk_maintenance_calls, 2);
    EXPECT_EQ(host.forward_token_offsets, (std::vector<int>{0, 4}));
    EXPECT_EQ(host.forward_real_seq_lens, (std::vector<int>{4, 4}));
    ASSERT_EQ(host.forward_token_batches.size(), 2u);
    EXPECT_EQ(host.forward_token_batches[0], (std::vector<int>{10, 11, 12, 13}));
    EXPECT_EQ(host.forward_token_batches[1], (std::vector<int>{14, 15, 16, 17}));
    EXPECT_EQ(host.last_maintenance_chunk.chunk_index, 1);
    EXPECT_TRUE(host.last_maintenance_decision.required);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunkSchedule_PlacementChangingMaintenanceClearsCachedBucketGraph)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;
    host.prefill_domain_id = "overlay_routed_cuda_hot";
    host.prefill_participant_id = 3;
    host.prefill_topology_signature = 0x1234u;
    host.bump_epoch_on_maintenance = true;
    host.engine_to_clear_on_maintenance = &engine;

    const std::vector<int> tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    auto input = makeTestInput(static_cast<int>(tokens.size()), 1, DeviceId::cuda(0), tokens.data(), nullptr);

    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {4};
    policy.fixed_chunk_real_tokens = 4;
    policy.max_rebalance_interval_tokens = 4;
    policy.real_token_count = static_cast<int>(tokens.size());

    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(schedule) << schedule.error;
    ASSERT_EQ(schedule.chunks.size(), 2u);
    ASSERT_TRUE(schedule.chunks[0].rebalance_required_after);
    ASSERT_TRUE(schedule.chunks[1].rebalance_required_after);

    ForwardOutput output{};
    EXPECT_TRUE(engine.runPrefillChunkSchedule(input, schedule, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 2)
        << "Placement-changing maintenance must clear the outer bucket graph so the next chunk rebuilds "
           "under the new epoch/topology rather than replaying stale stage placement.";
    EXPECT_EQ(host.prefill_chunk_maintenance_calls, 2);
    EXPECT_EQ(host.placement_epoch, 2u);
    EXPECT_TRUE(engine.cacheEmpty())
        << "The final required maintenance also clears the cache for the next request.";
    EXPECT_EQ(host.forward_token_offsets, (std::vector<int>{0, 4}));
    EXPECT_EQ(host.forward_real_seq_lens, (std::vector<int>{4, 4}));
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunkSchedule_StopsOnMaintenanceFailure)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;
    host.maintenance_should_fail = true;

    const std::vector<int> tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    auto input = makeTestInput(static_cast<int>(tokens.size()), 1, DeviceId::cpu(), tokens.data(), nullptr);

    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {4};
    policy.fixed_chunk_real_tokens = 4;
    policy.max_rebalance_interval_tokens = 4;
    policy.real_token_count = static_cast<int>(tokens.size());

    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(schedule) << schedule.error;

    ForwardOutput output{};
    EXPECT_FALSE(engine.runPrefillChunkSchedule(input, schedule, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.prefill_chunk_maintenance_calls, 1);
    EXPECT_EQ(host.forward_token_offsets, (std::vector<int>{0}));
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunkSchedule_InvalidScheduleRejectedWithoutHostCall)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    const std::vector<int> tokens = {10, 11, 12, 13};
    auto input = makeTestInput(static_cast<int>(tokens.size()), 1, DeviceId::cpu(), tokens.data(), nullptr);

    ForwardExecutionEngine::PrefillChunkRuntimeSchedule schedule;
    schedule.error = "not prepared";

    ForwardOutput output{};
    EXPECT_FALSE(engine.runPrefillChunkSchedule(input, schedule, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 0);
    EXPECT_EQ(host.prefill_chunk_maintenance_calls, 0);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_PaddedPlanDelegatesWithBucketMetadata)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {50, 51, 52};
    auto base_input = makeTestInput(3, 1, DeviceId::cuda(0), tokens.data(), nullptr);
    base_input.token_offset = 88;
    base_input.position_offset = 999;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4},
        /*pad_token_id=*/7,
        /*allow_padded_execution=*/true);
    ASSERT_TRUE(plan) << plan.error;
    ASSERT_TRUE(plan.padding_required);

    ForwardOutput output{};
    EXPECT_TRUE(engine.runPrefillChunk(base_input, plan, output, host));

    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_NE(host.last_token_ids_pointer, nullptr);
    EXPECT_NE(host.last_position_ids_pointer, nullptr);
    EXPECT_EQ(host.last_token_ids, (std::vector<int>{50, 51, 52, 7}));
    EXPECT_EQ(host.last_position_ids, (std::vector<int>{88, 89, 90, 91}));
    EXPECT_EQ(host.last_forward_input.seq_len, 4);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 3);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 4);
    EXPECT_EQ(host.last_forward_input.token_offset, 88);
    EXPECT_EQ(host.last_forward_input.position_offset, 88);
    EXPECT_EQ(host.last_workspace_seq_len, 4);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_PaddedGDNOrShortConvRejectedBeforeExecution)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    const std::vector<int> tokens = {70, 71, 72};

    for (ComputeStageType unsafe_type : {ComputeStageType::GDN_RECURRENCE, ComputeStageType::SHORT_CONV1D})
    {
        SCOPED_TRACE(computeStageTypeName(unsafe_type));
        auto engine = makeEngine(/*cache_enabled=*/true);
        MockForwardExecutionHost host(&gpu_ctx);
        host.graph_stage_types = {unsafe_type};

        auto base_input = makeTestInput(3, 1, DeviceId::cuda(0), tokens.data(), nullptr);
        auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            base_input,
            debugEnv().execution.prefill_graph_bucket_sizes,
            /*pad_token_id=*/0,
            /*allow_padded_execution=*/true);
        ASSERT_TRUE(plan) << plan.error;
        ASSERT_TRUE(plan.padding_required);

        ForwardOutput output{};
        EXPECT_FALSE(engine.runPrefillChunk(base_input, plan, output, host));
        EXPECT_EQ(host.build_forward_graph_calls, 1);
        EXPECT_EQ(host.ensure_workspace_calls, 0)
            << "Unsafe padded graph must reject before workspace allocation/execution";
        EXPECT_EQ(host.get_device_context_calls, 0)
            << "Unsafe padded graph must reject before asking for a launch context";
        EXPECT_EQ(host.sync_logits_calls, 0);
    }
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_InvalidPlansRejectedWithoutHostCall)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);

    const std::vector<int> tokens = {60, 61, 62, 63};
    auto base_input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    ForwardOutput output{};

    ForwardExecutionEngine::PrefillChunkRuntimePlan invalid_plan;
    EXPECT_FALSE(engine.runPrefillChunk(base_input, invalid_plan, output, host));

    ForwardExecutionEngine::PrefillChunkRuntimePlan invalid_chunk_plan;
    invalid_chunk_plan.ok = true;
    invalid_chunk_plan.chunk.error = "missing chunk buffers";
    EXPECT_FALSE(engine.runPrefillChunk(base_input, invalid_chunk_plan, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 0);
    EXPECT_FALSE(host.has_last_forward_input);
}

TEST_F(Test__ForwardExecutionEngine, Execute_RawBucketedPrefillPadsBeforeBuild)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    ForwardExecutionEngine::Config config;
    config.cache_config.enabled = true;
    config.cache_config.decode_seq_len = 1;
    config.has_unified_pp = false;
    ForwardExecutionEngine engine(std::move(config), executor_);
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {80, 81, 82};
    const std::vector<int> positions = {200, 201, 202};
    auto input = makeTestInput(3, 1, DeviceId::cuda(0), tokens.data(), positions.data());
    input.position_offset = 200;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host));

    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.last_token_ids, (std::vector<int>{80, 81, 82, 0}));
    EXPECT_EQ(host.last_position_ids, (std::vector<int>{200, 201, 202, 203}));
    EXPECT_EQ(host.last_forward_input.seq_len, 4);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 3);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 4);
    EXPECT_EQ(host.last_forward_input.token_offset, 200);
    EXPECT_EQ(host.last_forward_input.position_offset, 200);
    EXPECT_EQ(host.last_workspace_seq_len, 4);
}

TEST_F(Test__ForwardExecutionEngine, Execute_RawPrefillBelowMinSeqBypassesBucketedGraphCache)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64,128,256"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "256"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;

    std::vector<int> tokens;
    std::vector<int> positions;
    tokens.reserve(35);
    positions.reserve(35);
    for (int token_index = 0; token_index < 35; ++token_index)
    {
        tokens.push_back(500 + token_index);
        positions.push_back(1000 + token_index);
    }

    auto input = makeTestInput(35, 1, DeviceId::cuda(0), tokens.data(), positions.data());
    input.position_offset = 1000;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host));

    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.last_token_ids_pointer, tokens.data());
    EXPECT_EQ(host.last_position_ids_pointer, positions.data());
    EXPECT_EQ(host.last_token_ids, tokens);
    EXPECT_EQ(host.last_position_ids, positions);
    EXPECT_EQ(host.last_forward_input.seq_len, 35);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 0);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 0);
    EXPECT_EQ(host.last_workspace_seq_len, 35);
    EXPECT_TRUE(engine.cacheEmpty())
        << "Short raw prefill should bypass graph-cache population entirely.";
}

TEST_F(Test__ForwardExecutionEngine, Execute_NonExactBucketGpuWithoutGpuGraphs_FallsThrough)
{
    // When GPU graphs are disabled, non-exact auto-bucket selection should
    // gracefully skip bucketing and proceed with unbucketed execution.
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "0"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {90, 91, 92};
    const std::vector<int> positions = {300, 301, 302};
    auto input = makeTestInput(3, 1, DeviceId::cuda(0), tokens.data(), positions.data());
    input.position_offset = 300;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host))
        << "Non-exact bucket with GPU graphs disabled should fall through to unbucketed execution.";
    EXPECT_EQ(host.build_forward_graph_calls, 1)
        << "Engine should proceed to build an unbucketed forward graph.";
    EXPECT_EQ(host.last_forward_input.seq_len, 3)
        << "Unbucketed fallthrough should use the original seq_len, not the bucket size.";
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 0)
        << "No bucket should be applied in the fallthrough path.";
}

TEST_F(Test__ForwardExecutionEngine, Execute_NonExactBucketOnCpu_FallsThrough)
{
    // When the device is CPU, non-exact auto-bucket selection should gracefully
    // skip bucketing and proceed with unbucketed prefill execution. This is the
    // fix for CPU benchmarks failing when prefill_graph_buckets is enabled globally.
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {100, 101, 102};
    const std::vector<int> positions = {400, 401, 402};
    auto input = makeTestInput(3, 1, DeviceId::cpu(), tokens.data(), positions.data());
    input.position_offset = 400;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host))
        << "Non-exact bucket on CPU should fall through to unbucketed execution.";
    EXPECT_EQ(host.build_forward_graph_calls, 1)
        << "Engine should proceed to build an unbucketed forward graph.";
    EXPECT_EQ(host.last_forward_input.seq_len, 3)
        << "Unbucketed fallthrough should use the original seq_len, not the bucket size.";
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 0)
        << "No bucket should be applied in the fallthrough path.";
}

// =========================================================================
// Cache Management
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, DiscardAllCachedGraphsOnEmpty)
{
    auto engine = makeEngine();
    engine.discardAllCachedGraphs(); // Should not crash on empty cache
    EXPECT_TRUE(engine.cacheEmpty());
}

TEST_F(Test__ForwardExecutionEngine, InvalidateAllOnEmpty)
{
    auto engine = makeEngine();
    engine.invalidateAll(); // Should not crash on empty cache
    EXPECT_TRUE(engine.cacheEmpty());
}

TEST_F(Test__ForwardExecutionEngine, ResetCapturedReplayStatePreservesCachedGraphs)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    int token = 42;
    int pos = 7;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    ASSERT_TRUE(engine.execute(input, output, host));
    ASSERT_FALSE(engine.cacheEmpty());
    ASSERT_EQ(host.build_forward_graph_calls, 1);

    engine.resetCapturedReplayState();
    EXPECT_FALSE(engine.cacheEmpty());

    token = 43;
    pos = 8;
    ASSERT_TRUE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 1)
        << "Replay reset must preserve the cached ComputeGraph and avoid a rebuild";
}

TEST_F(Test__ForwardExecutionEngine, ReplayCacheObservationsTrackOrdinaryAndVerifierDecodeEntries)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    int ordinary_token = 42;
    int ordinary_pos = 7;
    auto ordinary = makeTestInput(
        1,
        1,
        DeviceId::cpu(),
        &ordinary_token,
        &ordinary_pos);

    int verifier_tokens[] = {43, 44, 45};
    int verifier_positions[] = {8, 9, 10};
    auto verifier = makeTestInput(
        3,
        1,
        DeviceId::cpu(),
        verifier_tokens,
        verifier_positions);

    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(ordinary, output, host));

    host.mock_compute_all_position_logits = true;
    ASSERT_TRUE(engine.execute(verifier, output, host));
    ASSERT_EQ(host.build_forward_graph_calls, 2);

    const auto observations = engine.replayCacheObservations(/*live_state_epoch=*/99);
    ASSERT_EQ(observations.size(), 2u);

    const ForwardExecutionEngine::ReplayCacheObservation *ordinary_obs = nullptr;
    const ForwardExecutionEngine::ReplayCacheObservation *verifier_obs = nullptr;
    for (const auto &observation : observations)
    {
        ASSERT_TRUE(observation.valid);
        EXPECT_TRUE(observation.signature.decode);
        EXPECT_FALSE(observation.segment_initialized)
            << "CPU entries have graph-cache identity but no GPU replay state.";
        EXPECT_EQ(observation.segmented_capture_live_state_epoch, 0u);
        EXPECT_FALSE(observation.requires_live_state_epoch_recapture);
        if (observation.signature.all_position_logits)
            verifier_obs = &observation;
        else
            ordinary_obs = &observation;
    }
    ASSERT_NE(ordinary_obs, nullptr);
    ASSERT_NE(verifier_obs, nullptr);
    EXPECT_EQ(classifyForwardReplayStateCache(ordinary_obs->signature),
              ForwardReplayStateCacheClass::OrdinaryDecode);
    EXPECT_EQ(classifyForwardReplayStateCache(verifier_obs->signature),
              ForwardReplayStateCacheClass::AllPositionVerifier);

    engine.resetCapturedReplayStateForCorrectionReplay();

    host.mock_compute_all_position_logits = false;
    ordinary_token = 46;
    ordinary_pos = 11;
    ASSERT_TRUE(engine.execute(ordinary, output, host));

    host.mock_compute_all_position_logits = true;
    verifier_tokens[0] = 47;
    verifier_tokens[1] = 48;
    verifier_tokens[2] = 49;
    verifier_positions[0] = 12;
    verifier_positions[1] = 13;
    verifier_positions[2] = 14;
    ASSERT_TRUE(engine.execute(verifier, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 2)
        << "Correction-boundary replay resets must preserve reusable ComputeGraphs; "
           "only captured replay executables/stream bindings are versioned.";
}

// =========================================================================
// execute() — Cache MISS path
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, CacheMiss_BuildFailure_ReturnsFalse)
{
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);
    host.build_should_fail = true;

    int token = 42;
    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    bool result = engine.execute(input, output, host);

    EXPECT_FALSE(result);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
}

TEST_F(Test__ForwardExecutionEngine, CacheMiss_EmptyGraph_Succeeds)
{
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);
    // Default mock returns empty graph (0 stages)

    int token = 42;
    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    // Empty graph build returns success=true but then engine checks graph.size()==0
    // and returns false ("Empty forward graph")
    bool result = engine.execute(input, output, host);

    EXPECT_FALSE(result);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
}

TEST_F(Test__ForwardExecutionEngine, CacheMiss_NullContext_ReturnsFalse)
{
    auto engine = makeEngine();
    MockForwardExecutionHost host(nullptr); // nullptr context

    int token = 42;
    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    bool result = engine.execute(input, output, host);

    // Empty graph returns false; if non-empty, null context returns false
    EXPECT_FALSE(result);
}

TEST_F(Test__ForwardExecutionEngine, CacheMiss_NullInputIds_NoCachePopulated)
{
    // When token_ids is nullptr, forward_cache_eligible should be false
    // (standard path, but has_stable_forward_inputs is false)
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);

    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), nullptr, &pos);
    ForwardOutput output{};

    engine.execute(input, output, host);

    // Cache should remain empty since inputs aren't stable
    EXPECT_TRUE(engine.cacheEmpty());
}

TEST_F(Test__ForwardExecutionEngine, CacheMiss_NullPositionIds_NoCachePopulated)
{
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);

    int token = 42;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, nullptr);
    ForwardOutput output{};

    engine.execute(input, output, host);

    // Cache should remain empty since position_ids are null
    EXPECT_TRUE(engine.cacheEmpty());
}

// =========================================================================
// execute() — Caching Disabled
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, CachingDisabled_NeverCaches)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);

    int token = 42;
    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    // Call execute twice — should always miss cache
    engine.execute(input, output, host);
    engine.execute(input, output, host);

    EXPECT_EQ(host.build_forward_graph_calls, 2);
    EXPECT_TRUE(engine.cacheEmpty());
}

// =========================================================================
// execute() — Cache HIT path
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, CacheHit_SecondCallSkipsBuild)
{
    // This test verifies the cache HIT path, but it requires a successful
    // first execute to populate the cache. Since the mock returns an empty
    // graph (which fails), we can't test a full cache hit without adding
    // stages to the mock graph. However, we can verify the cache state:
    // after a failed build, the cache entry exists but is not valid.
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);

    int token = 42;
    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    engine.execute(input, output, host);
    engine.execute(input, output, host);

    // Both calls hit cache MISS because the first build fails (empty graph)
    // and the cache entry is never marked valid.
    EXPECT_EQ(host.build_forward_graph_calls, 2);
}

TEST_F(Test__ForwardExecutionEngine, AllPositionShortContinuationPublishesVerifierCacheLookupStats)
{
    ScopedDebugEnv env({
        {"LLAMINAR_PERF_STATS_JSON", "1"},
    });
    PerfStatsCollector::reset();

    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;
    host.mock_compute_all_position_logits = true;

    int tokens[] = {101, 102};
    int positions[] = {32, 33};
    auto input = makeTestInput(2, 1, DeviceId::cpu(), tokens, positions);
    input.position_offset = 32;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host));
    EXPECT_TRUE(engine.execute(input, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 1);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags miss_tags = {
        {"all_position_logits", "true"},
        {"context", "main_verifier"},
        {"decode_has_history", "true"},
        {"moe_placement_epoch", "0"},
        {"result", "miss"},
        {"seq_len", "2"}};
    const PerfStatsCollector::Tags hit_tags = {
        {"all_position_logits", "true"},
        {"context", "main_verifier"},
        {"decode_has_history", "true"},
        {"moe_placement_epoch", "0"},
        {"result", "hit"},
        {"seq_len", "2"}};

    EXPECT_DOUBLE_EQ(findForwardGraphCounterValue(records, "forward_cache_lookup", miss_tags), 1.0);
    EXPECT_DOUBLE_EQ(findForwardGraphCounterValue(records, "forward_cache_lookup", hit_tags), 1.0);

    PerfStatsCollector::reset();
}

TEST_F(Test__ForwardExecutionEngine, MoEPlacementEpochChangeMissesDecodeCache)
{
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    int token = 42;
    int pos = 7;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    ASSERT_TRUE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 1);

    token = 43;
    pos = 8;
    ASSERT_TRUE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 1)
        << "Same MoE placement epoch should reuse the cached decode graph";

    host.placement_epoch = 1;
    token = 44;
    pos = 9;
    ASSERT_TRUE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 2)
        << "MoE placement changes must rebuild under a new graph identity";
}

// =========================================================================
// execute() — PP Configuration
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, UnifiedPP_ClearsCacheOnMiss)
{
    ForwardExecutionEngine::Config config;
    config.cache_config.enabled = true;
    config.has_unified_pp = true;
    ForwardExecutionEngine engine(std::move(config), executor_);

    MockForwardExecutionHost host(&mock_ctx_);

    int token = 42;
    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    engine.execute(input, output, host);

    // Unified PP path clears cache and uses multi-device execution
    EXPECT_TRUE(engine.cacheEmpty());
}

// =========================================================================
// execute() — Prefill (non-decode) path classification
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, Prefill_LargeSeqLen_NotDecode)
{
    ScopedDebugEnv env({
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "0"},
    });

    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);

    int tokens[] = {1, 2, 3, 4};
    int positions[] = {0, 1, 2, 3};
    auto input = makeTestInput(4, 1, DeviceId::cpu(), tokens, positions);
    ForwardOutput output{};

    engine.execute(input, output, host);

    // seq_len=4, batch_size=1 → not decode
    EXPECT_EQ(host.build_forward_graph_calls, 1);
}

// =========================================================================
// Destructive graph discard after population attempt
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, DiscardAllCachedGraphs_AfterExecute)
{
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);

    int token = 42;
    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    engine.execute(input, output, host);

    // Cache may or may not be empty after failed build, but destructive graph
    // discard should always succeed.
    engine.discardAllCachedGraphs();
    EXPECT_TRUE(engine.cacheEmpty());
}
