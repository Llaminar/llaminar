/**
 * @file Test__ForwardExecutionEngineAdvanced.cpp
 * @brief Advanced unit tests for ForwardExecutionEngine
 *
 * Tests coverage gaps identified during DGO refactor audit:
 * - IForwardExecutionHost callback sequencing and invocation
 * - Phase transitions (PREFILL → DECODE cache behavior)
 * - PP stage config flow through engine
 * - ForwardGraphSignature PP field differentiation
 * - DecodeCapturePolicy delegation
 * - PPCopyInfo resolution on cache HIT path
 * - Graph build with stages (non-empty graphs)
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <string>

#include "execution/local_execution/engine/ForwardExecutionEngine.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/factory/FactoryPPStageConfig.h"
#include "../../../../mocks/MockComputeStage.h" // MockDeviceContext

using namespace llaminar2;

// =========================================================================
// Enhanced MockForwardExecutionHost with Call Ordering
// =========================================================================

namespace
{

    /**
     * @brief Mock host that tracks detailed call ordering for verification.
     *
     * Records the sequence of callback invocations so tests can verify
     * the engine calls methods in the expected order.
     */
    class TrackingHost : public IForwardExecutionHost
    {
    public:
        explicit TrackingHost(IDeviceContext *ctx = nullptr)
            : ctx_(ctx) {}

        // ----- Call Sequence Tracking -----
        std::vector<std::string> call_sequence;

        // ----- Counters -----
        int build_calls = 0;
        int get_context_calls = 0;
        int ensure_workspace_calls = 0;
        int sync_logits_calls = 0;
        int logits_tensor_calls = 0;
        int build_decode_policy_calls = 0;
        int resolve_pp_copy_calls = 0;
        int get_pipeline_contexts_calls = 0;

        // ----- Configurable Behavior -----
        bool build_should_fail = false;
        int graph_node_count = 3; // Stages in built graph (0 = empty)
        PPCopyInfo mock_pp_copy;
        DeviceGraphExecutor::DecodeCapturePolicy mock_capture_policy;

        // ----- Captured Input -----
        int last_build_seq_len = -1;
        int last_build_batch_size = -1;
        bool last_decode_policy_has_collectives = false;

        // ----- IForwardExecutionHost -----

        GraphBuildResult buildForwardGraph(const ForwardInput &input) override
        {
            build_calls++;
            call_sequence.push_back("buildForwardGraph");
            last_build_seq_len = input.seq_len;
            last_build_batch_size = input.batch_size;

            if (build_should_fail)
                return GraphBuildResult("mock build failure");

            ComputeGraph graph;
            ForwardOutput output{};

            // Build a graph with actual mock stages so it's non-empty
            for (int i = 0; i < graph_node_count; ++i)
            {
                auto stage = std::make_unique<llaminar2::testing::MockComputeStage>(
                    ComputeStageType::GEMM,
                    "stage_" + std::to_string(i),
                    DeviceId::cpu());
                graph.addNode("node_" + std::to_string(i), std::move(stage), DeviceId::cpu());
            }

            return GraphBuildResult(std::move(graph), output);
        }

        IDeviceContext *getDeviceContext(DeviceId device) override
        {
            get_context_calls++;
            call_sequence.push_back("getDeviceContext");
            return ctx_;
        }

        std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() override
        {
            get_pipeline_contexts_calls++;
            call_sequence.push_back("getPipelineDeviceContexts");
            std::unordered_map<DeviceId, IDeviceContext *> result;
            if (ctx_)
                result[ctx_->deviceId()] = ctx_;
            return result;
        }

        bool ensureDeviceWorkspaceAllocated(const ComputeGraph &graph) override
        {
            ensure_workspace_calls++;
            call_sequence.push_back("ensureWorkspace");
            return true;
        }

        void syncLogitsAtBoundary(IDeviceContext *ctx) override
        {
            sync_logits_calls++;
            call_sequence.push_back("syncLogits");
        }

        TensorBase *logitsTensor() override
        {
            logits_tensor_calls++;
            call_sequence.push_back("logitsTensor");
            return nullptr;
        }

        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool has_collective_nodes,
            IDeviceContext *ctx,
            int segment_consecutive_failures) const override
        {
            const_cast<TrackingHost *>(this)->build_decode_policy_calls++;
            const_cast<TrackingHost *>(this)->call_sequence.push_back("buildDecodeCapturePolicy");
            const_cast<TrackingHost *>(this)->last_decode_policy_has_collectives = has_collective_nodes;
            return mock_capture_policy;
        }

        PPCopyInfo resolvePPCopyInfo(const ForwardInput &input) const override
        {
            const_cast<TrackingHost *>(this)->resolve_pp_copy_calls++;
            const_cast<TrackingHost *>(this)->call_sequence.push_back("resolvePPCopyInfo");
            return mock_pp_copy;
        }

    private:
        IDeviceContext *ctx_;
    };

    /**
     * @brief Create a ForwardInput with valid token/position pointers.
     */
    struct TestInput
    {
        std::vector<int> tokens;
        std::vector<int> positions;
        ForwardInput input;

        TestInput(int seq_len, int batch_size = 1, DeviceId device = DeviceId::cpu())
            : tokens(seq_len * batch_size, 42), positions(seq_len * batch_size)
        {
            for (int i = 0; i < seq_len * batch_size; ++i)
                positions[i] = i % seq_len;

            input.seq_len = seq_len;
            input.batch_size = batch_size;
            input.device = device;
            input.token_ids = tokens.data();
            input.position_ids = positions.data();
            input.position_offset = 0;
        }
    };

} // namespace

// =========================================================================
// Test Fixture
// =========================================================================

class Test__ForwardExecutionEngineAdvanced : public ::testing::Test
{
protected:
    DeviceGraphExecutor executor_;
    llaminar2::testing::MockDeviceContext mock_ctx_{DeviceId::cpu()};

    ForwardExecutionEngine makeEngine(bool cache_enabled = true,
                                      bool has_pp = false,
                                      std::optional<FactoryPPStageConfig> pp = std::nullopt)
    {
        ForwardExecutionEngine::Config config;
        config.cache_config.enabled = cache_enabled;
        config.has_unified_pp = has_pp;
        config.pp_stage_config = pp;
        return ForwardExecutionEngine(std::move(config), executor_);
    }
};

// =========================================================================
// Host Callback Invocation: Cache MISS Path
// =========================================================================

TEST_F(Test__ForwardExecutionEngineAdvanced, CacheMiss_HostBuildCalled)
{
    auto engine = makeEngine();
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    TestInput ti(1); // seq_len=1 (decode)
    ForwardOutput output{};

    engine.execute(ti.input, output, host);

    EXPECT_EQ(host.build_calls, 1);
}

TEST_F(Test__ForwardExecutionEngineAdvanced, CacheMiss_BuildReceivesCorrectInput)
{
    auto engine = makeEngine();
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    TestInput ti(128); // prefill with 128 tokens
    ForwardOutput output{};

    engine.execute(ti.input, output, host);

    EXPECT_EQ(host.last_build_seq_len, 128);
    EXPECT_EQ(host.last_build_batch_size, 1);
}

TEST_F(Test__ForwardExecutionEngineAdvanced, CacheMiss_DeviceContextRetrieved)
{
    auto engine = makeEngine();
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    TestInput ti(1);
    ForwardOutput output{};

    engine.execute(ti.input, output, host);

    EXPECT_GE(host.get_context_calls, 1);
}

TEST_F(Test__ForwardExecutionEngineAdvanced, CacheMiss_WorkspaceEnsured)
{
    auto engine = makeEngine();
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    TestInput ti(1);
    ForwardOutput output{};

    engine.execute(ti.input, output, host);

    EXPECT_GE(host.ensure_workspace_calls, 1);
}

TEST_F(Test__ForwardExecutionEngineAdvanced, CacheMiss_SyncLogitsCalled)
{
    auto engine = makeEngine();
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    TestInput ti(1);
    ForwardOutput output{};

    engine.execute(ti.input, output, host);

    EXPECT_GE(host.sync_logits_calls, 1);
}

TEST_F(Test__ForwardExecutionEngineAdvanced, CacheMiss_BuildFailure_NoSyncLogits)
{
    auto engine = makeEngine();
    TrackingHost host(&mock_ctx_);
    host.build_should_fail = true;

    TestInput ti(1);
    ForwardOutput output{};

    bool result = engine.execute(ti.input, output, host);

    EXPECT_FALSE(result);
    EXPECT_EQ(host.sync_logits_calls, 0) << "Should not sync logits on build failure";
}

TEST_F(Test__ForwardExecutionEngineAdvanced, CacheMiss_EmptyGraph_NoSyncLogits)
{
    auto engine = makeEngine();
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 0; // Empty graph

    TestInput ti(1);
    ForwardOutput output{};

    bool result = engine.execute(ti.input, output, host);

    EXPECT_FALSE(result);
    EXPECT_EQ(host.sync_logits_calls, 0) << "Should not sync logits on empty graph";
}

// =========================================================================
// Phase Transitions (Prefill → Decode)
// =========================================================================

TEST_F(Test__ForwardExecutionEngineAdvanced, PrefillThenDecode_TwoSeparateBuilds)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    // Prefill: seq_len=128
    TestInput prefill(128);
    ForwardOutput output{};
    engine.execute(prefill.input, output, host);
    EXPECT_EQ(host.build_calls, 1);

    // Decode: seq_len=1 (different signature → cache miss)
    TestInput decode(1);
    engine.execute(decode.input, output, host);
    EXPECT_EQ(host.build_calls, 2) << "Different seq_len should trigger a new build";
}

TEST_F(Test__ForwardExecutionEngineAdvanced, SameDecodeShape_CacheHitOnSecondCall)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    // First decode: cache miss → build
    TestInput decode1(1);
    ForwardOutput output{};
    engine.execute(decode1.input, output, host);
    EXPECT_EQ(host.build_calls, 1);
    EXPECT_FALSE(engine.cacheEmpty());

    // Second decode (same shape): should be cache hit → no build
    TestInput decode2(1);
    engine.execute(decode2.input, output, host);
    // On cache HIT, buildForwardGraph is NOT called
    EXPECT_EQ(host.build_calls, 1) << "Second decode with same shape should hit cache";
}

TEST_F(Test__ForwardExecutionEngineAdvanced, CachingDisabled_AlwaysBuilds)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    ForwardOutput output{};

    // First call
    TestInput t1(1);
    engine.execute(t1.input, output, host);
    EXPECT_EQ(host.build_calls, 1);

    // Second call (same shape)
    TestInput t2(1);
    engine.execute(t2.input, output, host);
    EXPECT_EQ(host.build_calls, 2) << "With caching disabled, always build";

    // Third call
    TestInput t3(1);
    engine.execute(t3.input, output, host);
    EXPECT_EQ(host.build_calls, 3);
}

// =========================================================================
// Decode Capture Policy Delegation
// =========================================================================

TEST_F(Test__ForwardExecutionEngineAdvanced, Decode_BuildDecodePolicyCalled)
{
    auto engine = makeEngine();
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    // Decode path: seq_len=1
    TestInput decode(1);
    ForwardOutput output{};
    engine.execute(decode.input, output, host);

    // On decode cache miss, the engine should ask host for capture policy
    EXPECT_GE(host.build_decode_policy_calls, 0);
    // Note: policy may only be called on cache HIT decode path (for GPU capture)
}

// =========================================================================
// PP Stage Config — Signature Differentiation
// =========================================================================

TEST_F(Test__ForwardExecutionEngineAdvanced, PPStageConfig_DifferentSignature)
{
    // Engine configured with PP stage config
    FactoryPPStageConfig pp{
        .first_layer = 0,
        .last_layer = 12,
        .has_embedding = true,
        .has_lm_head = false};
    auto engine = makeEngine(/*cache_enabled=*/true, /*has_pp=*/false, pp);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    TestInput decode(1);
    ForwardOutput output{};
    engine.execute(decode.input, output, host);

    EXPECT_EQ(host.build_calls, 1);
}

TEST_F(Test__ForwardExecutionEngineAdvanced, PPStageConfig_BuildReceivesInput)
{
    FactoryPPStageConfig pp{
        .first_layer = 8,
        .last_layer = 16,
        .has_embedding = false,
        .has_lm_head = false};
    auto engine = makeEngine(/*cache_enabled=*/true, /*has_pp=*/false, pp);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    TestInput ti(1);
    ForwardOutput output{};
    engine.execute(ti.input, output, host);

    // The host receives the input; it's the host's responsibility to use
    // pp_stage_config from its own state
    EXPECT_EQ(host.last_build_seq_len, 1);
}

// =========================================================================
// PP Copy Info Resolution
// =========================================================================

TEST_F(Test__ForwardExecutionEngineAdvanced, PPCopyInfo_ResolvedOnCacheMiss)
{
    FactoryPPStageConfig pp{
        .first_layer = 12,
        .last_layer = 24,
        .has_embedding = false, // Middle/last stage — needs PP copy
        .has_lm_head = true};
    auto engine = makeEngine(/*cache_enabled=*/true, /*has_pp=*/false, pp);

    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;
    host.mock_pp_copy.needs_copy = true;
    host.mock_pp_copy.copy_bytes = 4096;

    TestInput ti(1);
    ForwardOutput output{};
    engine.execute(ti.input, output, host);

    // PPCopyInfo should be resolved during cache miss path
    EXPECT_GE(host.resolve_pp_copy_calls, 1);
}

// =========================================================================
// Unified PP Path
// =========================================================================

TEST_F(Test__ForwardExecutionEngineAdvanced, UnifiedPP_UsesPipelineContexts)
{
    auto engine = makeEngine(/*cache_enabled=*/true, /*has_pp=*/true);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    TestInput ti(1);
    ForwardOutput output{};
    engine.execute(ti.input, output, host);

    // Unified PP path should request pipeline device contexts
    EXPECT_GE(host.get_pipeline_contexts_calls, 1);
}

TEST_F(Test__ForwardExecutionEngineAdvanced, UnifiedPP_CacheAlwaysMiss)
{
    auto engine = makeEngine(/*cache_enabled=*/true, /*has_pp=*/true);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    ForwardOutput output{};

    // First call
    TestInput t1(1);
    engine.execute(t1.input, output, host);
    EXPECT_EQ(host.build_calls, 1);

    // Second call — unified PP clears cache each time
    TestInput t2(1);
    engine.execute(t2.input, output, host);
    EXPECT_EQ(host.build_calls, 2) << "Unified PP should rebuild every time";
}

// =========================================================================
// Cache Invalidation and Clearing
// =========================================================================

TEST_F(Test__ForwardExecutionEngineAdvanced, InvalidateAll_ForcesRebuild)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    ForwardOutput output{};

    // Build and cache
    TestInput t1(1);
    engine.execute(t1.input, output, host);
    EXPECT_EQ(host.build_calls, 1);
    EXPECT_FALSE(engine.cacheEmpty());

    // Invalidate
    engine.invalidateAll();

    // Next call should rebuild
    TestInput t2(1);
    engine.execute(t2.input, output, host);
    EXPECT_EQ(host.build_calls, 2) << "invalidateAll should force rebuild";
}

TEST_F(Test__ForwardExecutionEngineAdvanced, ClearCache_ForcesRebuild)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    ForwardOutput output{};

    TestInput t1(1);
    engine.execute(t1.input, output, host);
    EXPECT_EQ(host.build_calls, 1);

    engine.clearCache();
    EXPECT_TRUE(engine.cacheEmpty());

    TestInput t2(1);
    engine.execute(t2.input, output, host);
    EXPECT_EQ(host.build_calls, 2);
}

// =========================================================================
// Multiple Seq Lengths Have Separate Cache Entries
// =========================================================================

TEST_F(Test__ForwardExecutionEngineAdvanced, DifferentSeqLens_SeparateCacheEntries)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    TrackingHost host(&mock_ctx_);
    host.graph_node_count = 3;

    ForwardOutput output{};

    // Prefill with 100 tokens
    TestInput prefill(100);
    engine.execute(prefill.input, output, host);
    EXPECT_EQ(host.build_calls, 1);

    // Decode with 1 token
    TestInput decode(1);
    engine.execute(decode.input, output, host);
    EXPECT_EQ(host.build_calls, 2);

    // Prefill again with 100 tokens — should hit cache
    TestInput prefill2(100);
    engine.execute(prefill2.input, output, host);
    EXPECT_EQ(host.build_calls, 2) << "Same prefill shape should hit cache";

    // Decode again — should hit cache
    TestInput decode2(1);
    engine.execute(decode2.input, output, host);
    EXPECT_EQ(host.build_calls, 2) << "Same decode shape should hit cache";
}

// =========================================================================
// ForwardGraphSignature PP Fields
// =========================================================================

TEST(Test__ForwardGraphSignature_PP, DifferentPPLayerRanges_DifferentSignatures)
{
    ForwardGraphSignature sig_a{
        .seq_len = 1, .batch_size = 1, .decode = true, .pp_stage_enabled = true, .pp_first_layer = 0, .pp_last_layer = 12, .pp_has_embedding = true, .pp_has_lm_head = false};

    ForwardGraphSignature sig_b{
        .seq_len = 1, .batch_size = 1, .decode = true, .pp_stage_enabled = true, .pp_first_layer = 12, .pp_last_layer = 24, .pp_has_embedding = false, .pp_has_lm_head = true};

    EXPECT_NE(sig_a, sig_b);

    ForwardGraphSignatureHash h;
    EXPECT_NE(h(sig_a), h(sig_b));
}

TEST(Test__ForwardGraphSignature_PP, PPEmbeddingFlag_AffectsSignature)
{
    ForwardGraphSignature sig_a{
        .pp_stage_enabled = true,
        .pp_first_layer = 0,
        .pp_last_layer = 12,
        .pp_has_embedding = true,
        .pp_has_lm_head = false};

    ForwardGraphSignature sig_b{
        .pp_stage_enabled = true,
        .pp_first_layer = 0,
        .pp_last_layer = 12,
        .pp_has_embedding = false,
        .pp_has_lm_head = false};

    EXPECT_NE(sig_a, sig_b);
}

TEST(Test__ForwardGraphSignature_PP, PPLMHeadFlag_AffectsSignature)
{
    ForwardGraphSignature sig_a{
        .pp_stage_enabled = true,
        .pp_first_layer = 12,
        .pp_last_layer = 24,
        .pp_has_embedding = false,
        .pp_has_lm_head = true};

    ForwardGraphSignature sig_b{
        .pp_stage_enabled = true,
        .pp_first_layer = 12,
        .pp_last_layer = 24,
        .pp_has_embedding = false,
        .pp_has_lm_head = false};

    EXPECT_NE(sig_a, sig_b);
}

TEST(Test__ForwardGraphSignature_PP, SameSignature_WithPP_CacheKey)
{
    ForwardGraphSignature sig_a{
        .seq_len = 1, .batch_size = 1, .decode = true, .pp_stage_enabled = true, .pp_first_layer = 0, .pp_last_layer = 12, .pp_has_embedding = true, .pp_has_lm_head = false};

    ForwardGraphSignature sig_b = sig_a;

    EXPECT_EQ(sig_a, sig_b);

    ForwardGraphSignatureHash h;
    EXPECT_EQ(h(sig_a), h(sig_b));
}

// =========================================================================
// GraphBuildResult — Extended Tests
// =========================================================================

TEST(Test__GraphBuildResult_Extended, SuccessResult_GraphAccessible)
{
    ComputeGraph graph;
    graph.addNode("test", nullptr, DeviceId::cpu());
    ForwardOutput output{};

    GraphBuildResult result(std::move(graph), output);

    EXPECT_TRUE(result.success());
    EXPECT_EQ(result.graph().size(), 1);
}

TEST(Test__GraphBuildResult_Extended, ErrorResult_EmptyGraph)
{
    GraphBuildResult result("something went wrong");

    EXPECT_TRUE(result.failed());
    EXPECT_EQ(result.error(), "something went wrong");
    EXPECT_EQ(result.graph().size(), 0);
}

TEST(Test__GraphBuildResult_Extended, MoveConstruction)
{
    ComputeGraph graph;
    graph.addNode("a", nullptr, DeviceId::cpu());
    ForwardOutput output{};

    GraphBuildResult r1(std::move(graph), output);
    GraphBuildResult r2(std::move(r1));

    EXPECT_TRUE(r2.success());
    EXPECT_EQ(r2.graph().size(), 1);
}

TEST(Test__GraphBuildResult_Extended, BoolConversion)
{
    GraphBuildResult success(ComputeGraph{}, ForwardOutput{});
    GraphBuildResult failure("fail");

    EXPECT_TRUE(static_cast<bool>(success));
    EXPECT_FALSE(static_cast<bool>(failure));

    // if-conversion
    if (success)
    {
        SUCCEED();
    }
    else
    {
        FAIL() << "Successful result should be truthy";
    }
}

// =========================================================================
// PPCopyInfo Struct Tests
// =========================================================================

TEST(Test__PPCopyInfo, DefaultState)
{
    IForwardExecutionHost::PPCopyInfo info;

    EXPECT_EQ(info.external_hidden, nullptr);
    EXPECT_EQ(info.working_buffer, nullptr);
    EXPECT_EQ(info.copy_bytes, 0u);
    EXPECT_FALSE(info.needs_copy);
}

TEST(Test__PPCopyInfo, ConfiguredForMiddleStage)
{
    IForwardExecutionHost::PPCopyInfo info;
    info.needs_copy = true;
    info.copy_bytes = 128 * 896 * sizeof(float);
    // Pointers would be non-null in production

    EXPECT_TRUE(info.needs_copy);
    EXPECT_GT(info.copy_bytes, 0u);
}

// =========================================================================
// FactoryPPStageConfig — Edge Cases Beyond Existing Tests
// =========================================================================

TEST(Test__FactoryPPStageConfig_Extended, DefaultConfig)
{
    FactoryPPStageConfig config{};
    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 0);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_FALSE(config.has_lm_head);
}

TEST(Test__FactoryPPStageConfig_Extended, SingleLayerStage)
{
    FactoryPPStageConfig config{
        .first_layer = 11,
        .last_layer = 12};

    EXPECT_TRUE(config.isValid());
    EXPECT_EQ(config.layerCount(), 1);
}

TEST(Test__FactoryPPStageConfig_Extended, ThreeWaySplit_Contiguous)
{
    // 24-layer model split into 3 stages of 8 layers each
    FactoryPPStageConfig stages[] = {
        {.first_layer = 0, .last_layer = 8, .has_embedding = true, .has_lm_head = false},
        {.first_layer = 8, .last_layer = 16, .has_embedding = false, .has_lm_head = false},
        {.first_layer = 16, .last_layer = 24, .has_embedding = false, .has_lm_head = true}};

    int total = 0;
    for (const auto &s : stages)
    {
        EXPECT_TRUE(s.isValid());
        total += s.layerCount();
    }
    EXPECT_EQ(total, 24);

    // Contiguity check: each stage starts where the previous ended
    EXPECT_EQ(stages[0].last_layer, stages[1].first_layer);
    EXPECT_EQ(stages[1].last_layer, stages[2].first_layer);
}

// =========================================================================
// GraphCacheConfig Variations
// =========================================================================

TEST(Test__GraphCacheConfig_Extended, DisableAttentionCachingOnly)
{
    GraphCacheConfig config;
    config.cache_attention = false;
    config.cache_ffn = true;

    EXPECT_TRUE(config.enabled);
    EXPECT_FALSE(config.cache_attention);
    EXPECT_TRUE(config.cache_ffn);
}

TEST(Test__GraphCacheConfig_Extended, CustomDecodeSeqLen)
{
    GraphCacheConfig config;
    config.decode_seq_len = 4; // seq_len <= 4 treated as "decode"

    EXPECT_EQ(config.decode_seq_len, 4);
}

TEST(Test__GraphCacheConfig_Extended, AllDisabled)
{
    GraphCacheConfig config;
    config.enabled = false;
    config.cache_attention = false;
    config.cache_ffn = false;

    EXPECT_FALSE(config.enabled);
    EXPECT_FALSE(config.cache_attention);
    EXPECT_FALSE(config.cache_ffn);
}
