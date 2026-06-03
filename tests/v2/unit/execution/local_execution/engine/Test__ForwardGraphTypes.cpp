/**
 * @file Test__ForwardGraphTypes.cpp
 * @brief Unit tests for ForwardGraphTypes
 *
 * Tests ForwardGraphSignature equality/hashing, GraphBuildResult,
 * GraphCacheConfig defaults, and ForwardGraphCache invalidation.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <future>
#include <unordered_map>

#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "utils/PerfStatsCollector.h"
#include "../../../../mocks/MockComputeStage.h"

using namespace llaminar2;

namespace
{
    class FakeSegmentStage final : public IComputeStage
    {
    public:
        FakeSegmentStage(bool capturable,
                         bool manual_boundary = false,
                         ComputeStageType stage_type = ComputeStageType::COPY,
                         bool warmup_dependent_capture = false,
                         const uint64_t *variant_signature = nullptr)
            : IComputeStage(DeviceId::cpu()),
              capturable_(capturable),
              manual_boundary_(manual_boundary),
              stage_type_(stage_type),
              warmup_dependent_capture_(warmup_dependent_capture),
              variant_signature_(variant_signature)
        {
        }

        bool execute(IDeviceContext *) override { return true; }
        ComputeStageType type() const override { return stage_type_; }
        std::string name() const override { return "fake_segment_stage"; }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        bool isGraphCapturable() const override { return capturable_; }
        uint64_t graphCaptureVariantSignature() const override
        {
            return variant_signature_ ? *variant_signature_ : 0;
        }
        bool supportsWarmupDependentGraphCapture() const override { return warmup_dependent_capture_; }
        bool isManualGraphBoundary() const override { return manual_boundary_; }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

    private:
        bool capturable_ = true;
        bool manual_boundary_ = false;
        ComputeStageType stage_type_ = ComputeStageType::COPY;
        bool warmup_dependent_capture_ = false;
        const uint64_t *variant_signature_ = nullptr;
    };

    class FakeReplayGraphCapture final : public IGPUGraphCapture
    {
    public:
        bool beginCapture() override { return true; }
        bool endCapture() override { return true; }
        bool instantiate() override
        {
            ++instantiate_calls_;
            executable_ = true;
            return true;
        }
        bool launch() override
        {
            ++launch_calls_;
            return executable_;
        }
        GraphUpdateResult tryUpdate() override
        {
            ++try_update_calls_;
            return update_result_;
        }
        bool hasExecutable() const override { return executable_; }
        size_t nodeCount() const override { return 1; }
        void reset() override { executable_ = false; }
        const char *backendName() const override { return "FakeReplay"; }

        int launch_calls_ = 0;
        int instantiate_calls_ = 0;
        int try_update_calls_ = 0;
        GraphUpdateResult update_result_ = GraphUpdateResult::Success;

    private:
        bool executable_ = true;
    };

    class FakeReplayGPUContext final : public IWorkerGPUContext
    {
    public:
        int deviceOrdinal() const override { return 0; }
        std::string deviceName() const override { return "FakeReplayGPU"; }
        bool isInitialized() const override { return true; }

        void submitAndWait(std::function<void()> work) override { work(); }
        std::future<void> submitAsync(std::function<void()> work) override
        {
            work();
            std::promise<void> done;
            done.set_value();
            return done.get_future();
        }

        void *defaultStream() override { return &default_stream_; }
        void *createStream() override { return &capture_stream_; }
        void destroyStream(void *) override {}

        void *createEvent() override { return nullptr; }
        void destroyEvent(void *) override {}
        void recordEvent(void *, void *) override {}
        void waitEvent(void *, void *) override {}
        void synchronizeEvent(void *) override {}
        float eventElapsedTime(void *, void *) override { return 0.0f; }
        void *blasHandle() override { return nullptr; }
        void *blasLtHandle() override { return nullptr; }
        void setCollectiveComm(void *) override {}
        void *collectiveComm() const override { return nullptr; }
        void synchronize() override {}
        void synchronizeStream(void *) override { ++synchronize_stream_calls_; }
        void insertStreamDependency(void *, void *) override {}

        std::unique_ptr<IGPUGraphCapture> createGraphCapture() override
        {
            return std::make_unique<FakeReplayGraphCapture>();
        }

        std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *) override
        {
            return std::make_unique<FakeReplayGraphCapture>();
        }

        int synchronize_stream_calls_ = 0;

    private:
        int default_stream_ = 0;
        int capture_stream_ = 0;
    };

    void addFakeSegmentStage(ComputeGraph &graph,
                             const std::string &name,
                             bool capturable,
                             bool manual_boundary = false,
                             ComputeStageType stage_type = ComputeStageType::COPY,
                             bool warmup_dependent_capture = false,
                             const uint64_t *variant_signature = nullptr)
    {
        graph.addNode(
            name,
            std::make_unique<FakeSegmentStage>(
                capturable,
                manual_boundary,
                stage_type,
                warmup_dependent_capture,
                variant_signature),
            DeviceId::cpu());
    }

    class ScopedEnvVar
    {
    public:
        ScopedEnvVar(const char *name, const char *value)
            : name_(name)
        {
            const char *existing = std::getenv(name);
            if (existing)
            {
                had_existing_ = true;
                existing_ = existing;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnvVar()
        {
            if (had_existing_)
            {
                setenv(name_.c_str(), existing_.c_str(), 1);
            }
            else
            {
                unsetenv(name_.c_str());
            }
        }

        ScopedEnvVar(const ScopedEnvVar &) = delete;
        ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

    private:
        std::string name_;
        bool had_existing_ = false;
        std::string existing_;
    };

    double findCounterValue(const std::vector<PerfStatRecord> &records,
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
        return -1.0;
    }

    uint64_t findTimerCount(const std::vector<PerfStatRecord> &records,
                            const std::string &name,
                            const PerfStatsCollector::Tags &tags)
    {
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Timer &&
                record.domain == "forward_graph" &&
                record.name == name &&
                record.tags == tags)
            {
                return record.count;
            }
        }
        return 0;
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

TEST(Test__ForwardGraphSignature, DifferentMoEPlacementEpochNotEqual)
{
    ForwardGraphSignature epoch0{.seq_len = 1,
                                 .batch_size = 1,
                                 .decode = true,
                                 .moe_placement_epoch = 0};
    ForwardGraphSignature epoch1 = epoch0;
    epoch1.moe_placement_epoch = 1;
    EXPECT_NE(epoch0, epoch1);
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
    ForwardGraphSignature d = a;
    d.moe_placement_epoch = 1;

    // Not a hard requirement, but extremely likely for different fields
    size_t ha = h(a), hb = h(b), hc = h(c), hd = h(d);
    EXPECT_NE(ha, hb);
    EXPECT_NE(ha, hc);
    EXPECT_NE(ha, hd);
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
    EXPECT_EQ(config.decode_seq_len, 4);
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

TEST(Test__GraphSegmentCache, WarmupSegmentsSkipPostWarmupResegmentForStableDenseStages)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "a", true);
    addFakeSegmentStage(graph, "b", true);
    graph.addDependency("b", "a");

    DeviceGraphExecutor::GraphSegmentCache cache;
    DeviceGraphCaptureController::executeWarmupPhase(
        graph,
        cache,
        nullptr,
        false,
        false);

    EXPECT_TRUE(cache.initialized);
    EXPECT_TRUE(cache.needs_capture);
    ASSERT_EQ(cache.segments.size(), 1u);
    EXPECT_TRUE(cache.segments[0].capturable);
}

TEST(Test__GraphSegmentCache, WarmupSegmentsPlanWarmupDependentStagesWithoutResegment)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "before", true);
    addFakeSegmentStage(
        graph,
        "warmup_dependent",
        false,
        false,
        ComputeStageType::MOE_EXPERT_FFN,
        true);
    addFakeSegmentStage(graph, "after", true);
    graph.addDependency("warmup_dependent", "before");
    graph.addDependency("after", "warmup_dependent");

    DeviceGraphExecutor::GraphSegmentCache cache;
    DeviceGraphCaptureController::executeWarmupPhase(
        graph,
        cache,
        nullptr,
        false,
        false);

    EXPECT_TRUE(cache.initialized);
    EXPECT_TRUE(cache.needs_capture);
    ASSERT_EQ(cache.segments.size(), 1u);
    EXPECT_TRUE(cache.segments[0].capturable);
    ASSERT_EQ(cache.segments[0].stage_names.size(), 3u);
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

TEST(Test__GraphSegmentCache, SegmentedPlanPublishesPerfStats)
{
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    ComputeGraph graph;
    addFakeSegmentStage(graph, "gemm", true, false, ComputeStageType::GEMM);
    addFakeSegmentStage(graph, "attention", false, true, ComputeStageType::ATTENTION);
    addFakeSegmentStage(graph, "copy", true, false, ComputeStageType::COPY);
    graph.addDependency("attention", "gemm");
    graph.addDependency("copy", "attention");

    DeviceGraphExecutor::GraphSegmentCache cache;
    DeviceGraphCaptureController::buildWarmupSegments(
        graph,
        cache,
        nullptr,
        /*has_collective_nodes=*/false);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});

    EXPECT_DOUBLE_EQ(findCounterValue(records, "segmented_plan_segments", {{"type", "total"}}), 3.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "segmented_plan_segments", {{"type", "capturable"}}), 2.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "segmented_plan_segments", {{"type", "manual"}}), 1.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "segmented_plan_stages", {{"type", "capturable"}}), 2.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "segmented_plan_stages", {{"type", "manual"}}), 1.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "segmented_plan_max_segment_stages", {{"type", "capturable"}}), 1.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "segmented_plan_max_segment_stages", {{"type", "manual"}}), 1.0);
    EXPECT_DOUBLE_EQ(
        findCounterValue(
            records,
            "segmented_plan_stage_types",
            {{"segment_type", "manual"}, {"stage_type", "ATTENTION"}}),
        1.0);
    EXPECT_DOUBLE_EQ(
        findCounterValue(
            records,
            "segmented_plan_stage_types",
            {{"segment_type", "capturable"}, {"stage_type", "GEMM"}}),
        1.0);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, CapturedReplayPerfStatsIncludeSegmentShapeTags)
{
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    DeviceGraphExecutor::GraphSegment segment;
    segment.capturable = true;
    segment.stage_names = {"gemm", "gdn_projection", "lm_head"};
    segment.capture = std::make_unique<FakeReplayGraphCapture>();

    FakeReplayGPUContext gpu_ctx;
    bool post_launch_called = false;
    int capture_stream = 0;

    ASSERT_TRUE(DeviceGraphCaptureController::executeCapturedReplaySegmentNormal(
        segment,
        &gpu_ctx,
        &capture_stream,
        /*needs_segment_sync=*/true,
        /*perf_context=*/"",
        [&](DeviceGraphExecutor::GraphSegment &, void *)
        {
            post_launch_called = true;
        }));

    EXPECT_TRUE(post_launch_called);
    EXPECT_EQ(gpu_ctx.synchronize_stream_calls_, 1);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags expected_tags = {
        {"type", "capturable"},
        {"stage_count", "3"}};
    EXPECT_EQ(findTimerCount(records, "segmented_replay_graph_launch", expected_tags), 1u);
    EXPECT_EQ(findTimerCount(records, "segmented_replay_post_launch", expected_tags), 1u);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, CapturedReplayPerfStatsIncludeContextTag)
{
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    DeviceGraphExecutor::GraphSegment segment;
    segment.capturable = true;
    segment.stage_names = {"embedding", "attention", "lm_head"};
    segment.capture = std::make_unique<FakeReplayGraphCapture>();

    FakeReplayGPUContext gpu_ctx;
    int capture_stream = 0;

    ASSERT_TRUE(DeviceGraphCaptureController::executeCapturedReplaySegmentNormal(
        segment,
        &gpu_ctx,
        &capture_stream,
        /*needs_segment_sync=*/false,
        /*perf_context=*/"main_verifier",
        [](DeviceGraphExecutor::GraphSegment &, void *) {}));

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags expected_tags = {
        {"context", "main_verifier"},
        {"type", "capturable"},
        {"stage_count", "3"}};
    EXPECT_EQ(findTimerCount(records, "segmented_replay_graph_launch", expected_tags), 1u);
    EXPECT_EQ(findTimerCount(records, "segmented_replay_post_launch", expected_tags), 1u);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, ReplayPhasePerfStatsSplitFinalStreamSync)
{
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    ComputeGraph graph;
    DeviceGraphExecutor::GraphSegmentCache cache;

    FakeReplayGPUContext gpu_ctx;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.perf_context = "main_verifier";
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"verifier_graph"};
    cache.segments.back().capture = std::make_unique<FakeReplayGraphCapture>();

    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/false,
        /*current_step=*/3,
        hooks);

    ASSERT_TRUE(result.success);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags capture_tags = {
        {"context", "main_verifier"},
        {"segment_count", "1"},
        {"stage_count", "1"},
        {"stream", "capture"},
        {"type", "capturable"}};
    const PerfStatsCollector::Tags default_tags = {
        {"context", "main_verifier"},
        {"segment_count", "1"},
        {"stage_count", "1"},
        {"stream", "default"},
        {"type", "capturable"}};
    const PerfStatsCollector::Tags aggregate_tags = {
        {"context", "main_verifier"},
        {"segment_count", "1"},
        {"stage_count", "1"},
        {"type", "capturable"}};

    EXPECT_EQ(findTimerCount(records, "segmented_replay_stream_sync", capture_tags), 1u);
    EXPECT_EQ(findTimerCount(records, "segmented_replay_stream_sync", default_tags), 1u);
    EXPECT_EQ(findTimerCount(records, "segmented_replay_final_sync", aggregate_tags), 1u);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, VariantSignatureChangeRecapturesBeforeReplay)
{
    ComputeGraph graph;
    uint64_t variant = 0x11;
    addFakeSegmentStage(
        graph,
        "bucketed_attention",
        true,
        false,
        ComputeStageType::ATTENTION,
        false,
        &variant);

    DeviceGraphExecutor executor;
    DeviceGraphExecutor::GraphSegmentCache cache;
    FakeReplayGPUContext gpu_ctx;
    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    ASSERT_TRUE(executor.executeWithSegmentedGraphCapture(
        graph,
        &ctx,
        cache,
        gpu_ctx.defaultStream(),
        &gpu_ctx));
    EXPECT_TRUE(cache.initialized);
    EXPECT_TRUE(cache.needs_capture);
    EXPECT_EQ(cache.decode_step, 1u);
    const uint64_t first_signature = cache.capture_variant_signature;
    ASSERT_NE(first_signature, 0u);

    graph.reset();
    ASSERT_TRUE(executor.executeWithSegmentedGraphCapture(
        graph,
        &ctx,
        cache,
        gpu_ctx.defaultStream(),
        &gpu_ctx));
    EXPECT_TRUE(cache.initialized);
    EXPECT_FALSE(cache.needs_capture);
    EXPECT_EQ(cache.decode_step, 2u);

    variant = 0x22;
    graph.reset();
    ASSERT_TRUE(executor.executeWithSegmentedGraphCapture(
        graph,
        &ctx,
        cache,
        gpu_ctx.defaultStream(),
        &gpu_ctx));

    EXPECT_TRUE(cache.initialized);
    EXPECT_TRUE(cache.needs_capture)
        << "variant changes must go through warmup/capture instead of stale replay";
    EXPECT_EQ(cache.decode_step, 1u);
    EXPECT_EQ(cache.variant_recapture_count, 1u);
    EXPECT_NE(cache.capture_variant_signature, first_signature);
}

TEST(Test__GraphSegmentCache, ROCmRecaptureSkipsInPlaceGraphUpdate)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "verifier_graph", true);

    DeviceGraphExecutor::GraphSegment segment;
    segment.capturable = true;
    segment.stage_names = {"verifier_graph"};
    auto capture = std::make_unique<FakeReplayGraphCapture>();
    FakeReplayGraphCapture *capture_ptr = capture.get();
    segment.capture = std::move(capture);

    FakeReplayGPUContext gpu_ctx;
    int capture_stream = 0;
    bool post_launch_called = false;
    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    ASSERT_TRUE(DeviceGraphCaptureController::executeCapturedReplaySegmentRecapture(
        graph,
        segment,
        &ctx,
        &gpu_ctx,
        &capture_stream,
        /*segment_index=*/0,
        [&](DeviceGraphExecutor::GraphSegment &, void *)
        {
            post_launch_called = true;
        }));

    EXPECT_TRUE(post_launch_called);
    EXPECT_EQ(capture_ptr->try_update_calls_, 0);
    EXPECT_EQ(capture_ptr->instantiate_calls_, 1);
    EXPECT_EQ(capture_ptr->launch_calls_, 1);
}

TEST(Test__GraphSegmentCache, CUDARecaptureStillUsesInPlaceGraphUpdate)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "verifier_graph", true);

    DeviceGraphExecutor::GraphSegment segment;
    segment.capturable = true;
    segment.stage_names = {"verifier_graph"};
    auto capture = std::make_unique<FakeReplayGraphCapture>();
    FakeReplayGraphCapture *capture_ptr = capture.get();
    segment.capture = std::move(capture);

    FakeReplayGPUContext gpu_ctx;
    int capture_stream = 0;
    bool post_launch_called = false;
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);

    ASSERT_TRUE(DeviceGraphCaptureController::executeCapturedReplaySegmentRecapture(
        graph,
        segment,
        &ctx,
        &gpu_ctx,
        &capture_stream,
        /*segment_index=*/0,
        [&](DeviceGraphExecutor::GraphSegment &, void *)
        {
            post_launch_called = true;
        }));

    EXPECT_TRUE(post_launch_called);
    EXPECT_EQ(capture_ptr->try_update_calls_, 1);
    EXPECT_EQ(capture_ptr->instantiate_calls_, 0);
    EXPECT_EQ(capture_ptr->launch_calls_, 1);
}
