/**
 * @file Test__ForwardGraphTypes.cpp
 * @brief Unit tests for ForwardGraphTypes
 *
 * Tests ForwardGraphSignature equality/hashing, GraphBuildResult,
 * GraphCacheConfig defaults, and ForwardGraphCache invalidation.
 */

#include <gtest/gtest.h>
#include <unordered_map>

#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"

using namespace llaminar2;

namespace
{
    class FakeSegmentStage final : public IComputeStage
    {
    public:
        FakeSegmentStage(bool capturable,
                         bool manual_boundary = false,
                         ComputeStageType stage_type = ComputeStageType::COPY)
            : IComputeStage(DeviceId::cpu()),
              capturable_(capturable),
              manual_boundary_(manual_boundary),
              stage_type_(stage_type)
        {
        }

        bool execute(IDeviceContext *) override { return true; }
        ComputeStageType type() const override { return stage_type_; }
        std::string name() const override { return "fake_segment_stage"; }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        bool isGraphCapturable() const override { return capturable_; }
        bool isManualGraphBoundary() const override { return manual_boundary_; }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

    private:
        bool capturable_ = true;
        bool manual_boundary_ = false;
        ComputeStageType stage_type_ = ComputeStageType::COPY;
    };

    void addFakeSegmentStage(ComputeGraph &graph,
                             const std::string &name,
                             bool capturable,
                             bool manual_boundary = false,
                             ComputeStageType stage_type = ComputeStageType::COPY)
    {
        graph.addNode(
            name,
            std::make_unique<FakeSegmentStage>(capturable, manual_boundary, stage_type),
            DeviceId::cpu());
    }
}

// =========================================================================
// ForwardGraphSignature
// =========================================================================

TEST(Test__ForwardGraphSignature, DefaultConstructedEqual)
{
    ForwardGraphSignature a;
    ForwardGraphSignature b;
    EXPECT_EQ(a, b);
}

TEST(Test__ForwardGraphSignature, SameFieldsAreEqual)
{
    ForwardGraphSignature a{.seq_len = 1, .batch_size = 1, .device = DeviceId::cpu(),
                            .decode = true, .standard_path = true, .pp_stage_enabled = false,
                            .pp_first_layer = -1, .pp_last_layer = -1,
                            .pp_has_embedding = false, .pp_has_lm_head = false};
    ForwardGraphSignature b = a;
    EXPECT_EQ(a, b);
}

TEST(Test__ForwardGraphSignature, DifferentSeqLenNotEqual)
{
    ForwardGraphSignature a{.seq_len = 1, .batch_size = 1};
    ForwardGraphSignature b{.seq_len = 128, .batch_size = 1};
    EXPECT_NE(a, b);
}

TEST(Test__ForwardGraphSignature, DifferentBatchSizeNotEqual)
{
    ForwardGraphSignature a{.seq_len = 1, .batch_size = 1};
    ForwardGraphSignature b{.seq_len = 1, .batch_size = 4};
    EXPECT_NE(a, b);
}

TEST(Test__ForwardGraphSignature, DifferentDecodeNotEqual)
{
    ForwardGraphSignature a{.seq_len = 1, .batch_size = 1, .decode = true};
    ForwardGraphSignature b{.seq_len = 1, .batch_size = 1, .decode = false};
    EXPECT_NE(a, b);
}

TEST(Test__ForwardGraphSignature, DifferentAllPositionLogitsNotEqual)
{
    ForwardGraphSignature terminal_only{.seq_len = 2, .batch_size = 1, .decode = true,
                                        .all_position_logits = false};
    ForwardGraphSignature all_positions{.seq_len = 2, .batch_size = 1, .decode = true,
                                        .all_position_logits = true};
    EXPECT_NE(terminal_only, all_positions);
}

TEST(Test__ForwardGraphSignature, DifferentPPFieldsNotEqual)
{
    ForwardGraphSignature a{.pp_stage_enabled = true, .pp_first_layer = 0, .pp_last_layer = 13};
    ForwardGraphSignature b{.pp_stage_enabled = true, .pp_first_layer = 14, .pp_last_layer = 27};
    EXPECT_NE(a, b);
}

TEST(Test__ForwardGraphSignature, PPEnabledVsDisabledNotEqual)
{
    ForwardGraphSignature a{.pp_stage_enabled = false};
    ForwardGraphSignature b{.pp_stage_enabled = true};
    EXPECT_NE(a, b);
}

// =========================================================================
// ForwardGraphSignatureHash
// =========================================================================

TEST(Test__ForwardGraphSignatureHash, EqualSignaturesHaveSameHash)
{
    ForwardGraphSignature a{.seq_len = 1, .batch_size = 1, .decode = true};
    ForwardGraphSignature b{.seq_len = 1, .batch_size = 1, .decode = true};

    ForwardGraphSignatureHash h;
    EXPECT_EQ(h(a), h(b));
}

TEST(Test__ForwardGraphSignatureHash, DifferentSignaturesLikelyDifferentHash)
{
    ForwardGraphSignatureHash h;

    ForwardGraphSignature a{.seq_len = 1, .batch_size = 1, .decode = true};
    ForwardGraphSignature b{.seq_len = 128, .batch_size = 1, .decode = false};
    ForwardGraphSignature c{.seq_len = 1, .batch_size = 4, .decode = true};

    // Not a hard requirement, but extremely likely for different fields
    size_t ha = h(a), hb = h(b), hc = h(c);
    EXPECT_NE(ha, hb);
    EXPECT_NE(ha, hc);
    EXPECT_NE(hb, hc);
}

TEST(Test__ForwardGraphSignatureHash, UsableAsUnorderedMapKey)
{
    std::unordered_map<ForwardGraphSignature, int, ForwardGraphSignatureHash> map;

    ForwardGraphSignature decode{.seq_len = 1, .batch_size = 1, .decode = true};
    ForwardGraphSignature prefill{.seq_len = 128, .batch_size = 1, .decode = false};

    map[decode] = 42;
    map[prefill] = 99;

    EXPECT_EQ(map[decode], 42);
    EXPECT_EQ(map[prefill], 99);
    EXPECT_EQ(map.size(), 2u);

    // Lookup with equivalent key works
    ForwardGraphSignature decode2{.seq_len = 1, .batch_size = 1, .decode = true};
    EXPECT_EQ(map[decode2], 42);
}

// =========================================================================
// GraphBuildResult
// =========================================================================

TEST(Test__GraphBuildResult, DefaultConstructedFails)
{
    GraphBuildResult result;
    EXPECT_FALSE(result.success());
    EXPECT_TRUE(result.failed());
    EXPECT_FALSE(static_cast<bool>(result));
}

TEST(Test__GraphBuildResult, SuccessConstruction)
{
    ComputeGraph graph;
    ForwardOutput output{};
    GraphBuildResult result(std::move(graph), output);

    EXPECT_TRUE(result.success());
    EXPECT_FALSE(result.failed());
    EXPECT_TRUE(static_cast<bool>(result));
}

TEST(Test__GraphBuildResult, ErrorConstruction)
{
    GraphBuildResult result("something went wrong");

    EXPECT_FALSE(result.success());
    EXPECT_TRUE(result.failed());
    EXPECT_EQ(result.error(), "something went wrong");
}

TEST(Test__GraphBuildResult, TakeGraphMovesOwnership)
{
    ComputeGraph graph;
    ForwardOutput output{};
    GraphBuildResult result(std::move(graph), output);

    auto taken = result.takeGraph();
    // After takeGraph(), the original graph is moved-from
    // We can't assert much about the moved-from state, but the taken graph is valid
    (void)taken;
}

// =========================================================================
// GraphCacheConfig
// =========================================================================

TEST(Test__GraphCacheConfig, Defaults)
{
    GraphCacheConfig config;
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.decode_seq_len, 2);
    EXPECT_TRUE(config.cache_attention);
    EXPECT_TRUE(config.cache_ffn);
}

// =========================================================================
// ForwardGraphCache
// =========================================================================

TEST(Test__ForwardGraphCache, DefaultState)
{
    ForwardGraphCache cache;
    EXPECT_FALSE(cache.valid);
    EXPECT_EQ(cache.graph, nullptr);
    EXPECT_TRUE(cache.token_ids.empty());
    EXPECT_TRUE(cache.position_ids.empty());
    EXPECT_TRUE(cache.collective_nodes.empty());
    EXPECT_FALSE(cache.pp_needs_copy);
    EXPECT_FALSE(cache.dynamic_param_stages_cached);
    EXPECT_FALSE(cache.gpu_stream_applied);
    EXPECT_FALSE(cache.phase3_active);
    EXPECT_EQ(cache.gpu_stream, nullptr);
    EXPECT_EQ(cache.gpu_ctx, nullptr);
    EXPECT_EQ(cache.gpu_graph_update_failures, 0);
}

TEST(Test__ForwardGraphCache, InvalidateResetsAllFields)
{
    ForwardGraphCache cache;

    // Set various fields to non-default values
    cache.graph = std::make_unique<ComputeGraph>();
    cache.valid = true;
    cache.token_ids = {1, 2, 3};
    cache.position_ids = {0, 1, 2};
    cache.collective_nodes = {"allreduce_0", "allreduce_1"};
    cache.pp_needs_copy = true;
    cache.pp_copy_bytes = 1024;
    cache.dynamic_param_stages_cached = true;
    cache.gpu_stream_applied = true;
    cache.phase3_active = true;
    cache.gpu_stream = reinterpret_cast<void *>(0xDEAD);
    cache.gpu_graph_update_failures = 3;

    cache.invalidate();

    EXPECT_FALSE(cache.valid);
    EXPECT_EQ(cache.graph, nullptr);
    EXPECT_TRUE(cache.token_ids.empty());
    EXPECT_TRUE(cache.position_ids.empty());
    EXPECT_TRUE(cache.collective_nodes.empty());
    EXPECT_FALSE(cache.pp_needs_copy);
    EXPECT_EQ(cache.pp_copy_bytes, 0u);
    EXPECT_FALSE(cache.dynamic_param_stages_cached);
    EXPECT_FALSE(cache.gpu_stream_applied);
    EXPECT_FALSE(cache.phase3_active);
    EXPECT_EQ(cache.gpu_stream, nullptr);
    EXPECT_EQ(cache.gpu_graph_update_failures, 0);
}

TEST(Test__ForwardGraphCache, InvalidateIdempotent)
{
    ForwardGraphCache cache;
    cache.valid = true;
    cache.token_ids = {1};

    cache.invalidate();
    cache.invalidate(); // Should be safe to call twice

    EXPECT_FALSE(cache.valid);
    EXPECT_TRUE(cache.token_ids.empty());
}

TEST(Test__GraphSegmentCache, ResetCanPreserveCaptureStream)
{
    DeviceGraphExecutor::GraphSegmentCache cache;
    void *stream = reinterpret_cast<void *>(0x1234);

    cache.initialized = true;
    cache.needs_capture = true;
    cache.consecutive_failures = 2;
    cache.decode_step = 17;
    cache.capture_stream = stream;

    cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);

    EXPECT_FALSE(cache.initialized);
    EXPECT_FALSE(cache.needs_capture);
    EXPECT_EQ(cache.consecutive_failures, 0);
    EXPECT_EQ(cache.decode_step, 0u);
    EXPECT_EQ(cache.capture_stream, stream);
}

TEST(Test__GraphSegmentCache, ResetCanDestroyCaptureStream)
{
    DeviceGraphExecutor::GraphSegmentCache cache;
    cache.capture_stream = reinterpret_cast<void *>(0x1234);

    cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);

    EXPECT_EQ(cache.capture_stream, nullptr);
}

TEST(Test__ForwardGraphCache, ReplayResetPreservesSegmentCaptureStream)
{
    ForwardGraphCache cache;
    void *stream = reinterpret_cast<void *>(0x1234);

    cache.segment_cache.capture_stream = stream;
    cache.segment_cache.initialized = true;
    cache.gpu_graph_update_failures = 3;
    cache.phase3_active = true;

    cache.resetReplayState();

    EXPECT_EQ(cache.segment_cache.capture_stream, stream);
    EXPECT_FALSE(cache.segment_cache.initialized);
    EXPECT_EQ(cache.gpu_graph_update_failures, 0);
    EXPECT_FALSE(cache.phase3_active);
}

TEST(Test__ForwardGraphCache, InvalidateDestroysSegmentCaptureStream)
{
    ForwardGraphCache cache;
    cache.segment_cache.capture_stream = reinterpret_cast<void *>(0x1234);

    cache.invalidate();

    EXPECT_EQ(cache.segment_cache.capture_stream, nullptr);
}

TEST(Test__GraphSegmentCache, NamedSparseBoundarySplitsCapturedSegments)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "before", true);
    addFakeSegmentStage(graph, "sparse_dispatch", true, true);
    addFakeSegmentStage(graph, "after", true);
    graph.addDependency("sparse_dispatch", "before");
    graph.addDependency("after", "sparse_dispatch");

    std::unordered_set<std::string> collective_nodes = {"sparse_dispatch"};
    DeviceGraphExecutor::GraphSegmentCache cache;
    DeviceGraphCaptureController::buildWarmupSegments(
        graph,
        cache,
        &collective_nodes,
        /*has_collective_nodes=*/true,
        /*collectives_graph_capturable=*/false);

    ASSERT_EQ(cache.segments.size(), 3u);
    EXPECT_TRUE(cache.segments[0].capturable);
    EXPECT_EQ(cache.segments[0].stage_names, std::vector<std::string>({"before"}));
    EXPECT_FALSE(cache.segments[1].capturable);
    EXPECT_EQ(cache.segments[1].stage_names, std::vector<std::string>({"sparse_dispatch"}));
    EXPECT_TRUE(cache.segments[2].capturable);
    EXPECT_EQ(cache.segments[2].stage_names, std::vector<std::string>({"after"}));
}

TEST(Test__GraphSegmentCache, NonCapturableManualBoundarySplitsCapturedSegmentsWithoutNameSet)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "before", true);
    addFakeSegmentStage(graph, "sparse_return", false, true);
    addFakeSegmentStage(graph, "after", true);
    graph.addDependency("sparse_return", "before");
    graph.addDependency("after", "sparse_return");

    DeviceGraphExecutor::GraphSegmentCache cache;
    DeviceGraphCaptureController::buildWarmupSegments(
        graph,
        cache,
        nullptr,
        /*has_collective_nodes=*/false);

    ASSERT_EQ(cache.segments.size(), 3u);
    EXPECT_TRUE(cache.segments[0].capturable);
    EXPECT_EQ(cache.segments[0].stage_names, std::vector<std::string>({"before"}));
    EXPECT_FALSE(cache.segments[1].capturable);
    EXPECT_EQ(cache.segments[1].stage_names, std::vector<std::string>({"sparse_return"}));
    EXPECT_TRUE(cache.segments[2].capturable);
    EXPECT_EQ(cache.segments[2].stage_names, std::vector<std::string>({"after"}));
}

TEST(Test__GraphSegmentCache, GraphSafeNamedCollectivesRequireExplicitCapturePermission)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "before", true);
    addFakeSegmentStage(graph, "graph_safe_collective", true, true);
    addFakeSegmentStage(graph, "after", true);
    graph.addDependency("graph_safe_collective", "before");
    graph.addDependency("after", "graph_safe_collective");

    std::unordered_set<std::string> collective_nodes = {"graph_safe_collective"};
    DeviceGraphExecutor::GraphSegmentCache cache;
    DeviceGraphCaptureController::buildWarmupSegments(
        graph,
        cache,
        &collective_nodes,
        /*has_collective_nodes=*/true,
        /*collectives_graph_capturable=*/true);

    ASSERT_EQ(cache.segments.size(), 1u);
    EXPECT_TRUE(cache.segments[0].capturable);
    EXPECT_EQ(cache.segments[0].stage_names,
              std::vector<std::string>({"before", "graph_safe_collective", "after"}));
}
