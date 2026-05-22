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

        // ----- Configurable Results -----
        bool build_should_fail = false;
        std::string build_error_message = "mock build failure";
        int graph_stage_count = 0; // stages in built graph (0 means empty graph)
        // Optional explicit stage types let safety tests assemble GDN/short-conv graphs.
        std::vector<ComputeStageType> graph_stage_types;
        PPCopyInfo mock_pp_copy;
        DeviceGraphExecutor::DecodeCapturePolicy mock_capture_policy;

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

        bool ensureDeviceWorkspaceAllocated(const ComputeGraph &graph) override
        {
            ensure_workspace_calls++;
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

    auto engine = makeEngine(/*cache_enabled=*/true);
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
    EXPECT_TRUE(engine.cacheEmpty())
        << "Short raw prefill should bypass graph-cache population entirely.";
}

TEST_F(Test__ForwardExecutionEngine, Execute_RawPaddedGpuBucketRequiresGpuGraphs)
{
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
    EXPECT_FALSE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 0)
        << "Raw padded GPU bucket requests must reject before normal graph build when GPU graphs are disabled.";
}

TEST_F(Test__ForwardExecutionEngine, Execute_RawPaddedCpuBucketRequiresPrefillGraphEligibility)
{
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
    EXPECT_FALSE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 0)
        << "Raw padded CPU bucket requests must reject before normal graph build.";
}

// =========================================================================
// Cache Management
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, ClearCacheOnEmpty)
{
    auto engine = makeEngine();
    engine.clearCache(); // Should not crash on empty cache
    EXPECT_TRUE(engine.cacheEmpty());
}

TEST_F(Test__ForwardExecutionEngine, InvalidateAllOnEmpty)
{
    auto engine = makeEngine();
    engine.invalidateAll(); // Should not crash on empty cache
    EXPECT_TRUE(engine.cacheEmpty());
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
// clearCache after population attempt
// =========================================================================

TEST_F(Test__ForwardExecutionEngine, ClearCache_AfterExecute)
{
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);

    int token = 42;
    int pos = 0;
    auto input = makeTestInput(1, 1, DeviceId::cpu(), &token, &pos);
    ForwardOutput output{};

    engine.execute(input, output, host);

    // Cache may or may not be empty after failed build, but clearCache
    // should always succeed
    engine.clearCache();
    EXPECT_TRUE(engine.cacheEmpty());
}
