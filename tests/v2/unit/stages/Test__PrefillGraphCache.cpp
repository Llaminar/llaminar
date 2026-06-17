/**
 * @file Test__PrefillGraphCache.cpp
 * @brief Unit tests for PrefillGraphCache state machine, preflight, and invalidation
 */

#include <gtest/gtest.h>
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "execution/local_execution/engine/PrefillGraphCache.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "backends/DeviceId.h"
#include "backends/GPUDeviceContextPool.h"
#include "mocks/MockComputeStage.h"
#include "utils/DebugEnv.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

// =============================================================================
// Helper: Capturable mock stage for graph capture tests
// =============================================================================

class CapturableMockStage : public MockComputeStage
{
public:
    CapturableMockStage(std::string name, DeviceId dev)
        : MockComputeStage(ComputeStageType::GEMM, std::move(name), dev) {}

    bool isGraphCapturable() const override { return capturable_; }
    void setCapturable(bool v) { capturable_ = v; }

private:
    bool capturable_ = true;
};

// Non-capturable mock stage
class NonCapturableMockStage : public MockComputeStage
{
public:
    NonCapturableMockStage(std::string name, DeviceId dev)
        : MockComputeStage(ComputeStageType::GEMM, std::move(name), dev) {}

    bool isGraphCapturable() const override { return false; }
};

/**
 * @brief Mock stage with cold padded-prefill support but delayed capture readiness.
 */
class ColdPaddedPreflightOnlyMockStage : public MockComputeStage
{
public:
    ColdPaddedPreflightOnlyMockStage(std::string name, DeviceId dev)
        : MockComputeStage(ComputeStageType::GEMM, std::move(name), dev) {}

    bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
    bool isGraphCapturable() const override { return capture_ready_; }
    void setCaptureReady(bool ready) { capture_ready_ = ready; }

private:
    bool capture_ready_ = false;
};

/**
 * @brief Temporarily override the ROCm grouped-prefill flag for preflight tests.
 */
class ScopedRocmGroupedPrefillFlag
{
public:
    explicit ScopedRocmGroupedPrefillFlag(bool enabled)
        : old_prefill_(mutableDebugEnv().rocm.moe_grouped_prefill)
    {
        mutableDebugEnv().rocm.moe_grouped_prefill = enabled;
    }

    ~ScopedRocmGroupedPrefillFlag()
    {
        mutableDebugEnv().rocm.moe_grouped_prefill = old_prefill_;
    }

private:
    bool old_prefill_;
};

/**
 * @brief Mock stateful GDN/short-conv stage that implements padded real-length replay.
 */
class PaddedRealLengthContractMockStage : public MockComputeStage
{
public:
    PaddedRealLengthContractMockStage(ComputeStageType type, std::string name, DeviceId dev)
        : MockComputeStage(type, std::move(name), dev) {}

    bool supportsPaddedPrefillRealLengthContract() const override { return true; }
};

/**
 * @brief Mock stateful stage that has the real-length contract but is not graph-capturable.
 */
class NonCapturablePaddedRealLengthContractMockStage : public PaddedRealLengthContractMockStage
{
public:
    NonCapturablePaddedRealLengthContractMockStage(ComputeStageType type, std::string name, DeviceId dev)
        : PaddedRealLengthContractMockStage(type, std::move(name), dev) {}

    bool isGraphCapturable() const override { return false; }
};

/**
 * @brief Minimal graph capture object for PrefillGraphCache state-machine tests.
 *
 * The cache tests only need to verify lifecycle dispatch and phase transitions,
 * so this fake returns success without touching GPU APIs.
 */
class FakePrefillGraphCapture final : public IGPUGraphCapture
{
public:
    bool beginCapture() override { return true; }
    bool endCapture() override { return true; }
    bool instantiate() override
    {
        executable_ = true;
        return true;
    }
    bool launch() override { return executable_; }
    GraphUpdateResult tryUpdate() override { return GraphUpdateResult::Success; }
    bool hasExecutable() const override { return executable_; }
    size_t nodeCount() const override { return 1; }
    void reset() override { executable_ = false; }
    const char *backendName() const override { return "Fake"; }

private:
    bool executable_ = false;
};

/**
 * @brief Mock worker GPU context that records graph-capture stream selection.
 *
 * This context keeps stream handles as stable in-process addresses and never
 * calls GPU runtime APIs, which lets the prefill graph cache tests verify the
 * explicit-stream overload deterministically.
 */
class PrefillMockGPUContext final : public IWorkerGPUContext
{
public:
    int deviceOrdinal() const override { return 0; }
    std::string deviceName() const override { return "PrefillMock"; }
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
    void *createStream() override { return &created_stream_; }
    void destroyStream(void *stream) override
    {
        destroyed_stream_ = stream;
        destroy_stream_calls_++;
    }

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
    void synchronizeStream(void *) override {}
    void insertStreamDependency(void *, void *) override {}

    std::unique_ptr<IGPUGraphCapture> createGraphCapture() override
    {
        default_capture_calls_++;
        return std::make_unique<FakePrefillGraphCapture>();
    }

    std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *stream) override
    {
        explicit_capture_calls_++;
        last_capture_stream_ = stream;
        return std::make_unique<FakePrefillGraphCapture>();
    }

    int default_capture_calls_ = 0;
    int explicit_capture_calls_ = 0;
    int destroy_stream_calls_ = 0;
    void *last_capture_stream_ = nullptr;
    void *destroyed_stream_ = nullptr;

private:
    int default_stream_ = 0;
    int created_stream_ = 0;
};

// =============================================================================
// Helper: Build a simple graph with capturable stages
// =============================================================================

static ComputeGraph buildCapturableGraph(DeviceId dev, int num_stages = 3)
{
    ComputeGraph graph;
    for (int i = 0; i < num_stages; ++i)
    {
        std::string name = "stage_" + std::to_string(i);
        graph.addNode(name, std::make_unique<CapturableMockStage>(name, dev), dev);
        if (i > 0)
        {
            graph.addDependency(name, "stage_" + std::to_string(i - 1));
        }
    }
    return graph;
}

static PrefillGraphCacheKey makeGPUKey(int seq_len = 512)
{
    PrefillGraphCacheKey key;
    key.seq_len = seq_len;
#ifdef HAVE_ROCM
    key.device_id = DeviceId::rocm(0);
#elif defined(HAVE_CUDA)
    key.device_id = DeviceId::cuda(0);
#else
    key.device_id = DeviceId::cuda(0); // placeholder
#endif
    return key;
}

static MoERoutingStage::Params makeColdRocmRoutingParams(
    TensorBase *input,
    TensorBase *gate_weights,
    TensorBase *output_indices,
    TensorBase *output_weights,
    DeviceId device,
    int seq_len,
    int d_model,
    int num_experts,
    int top_k)
{
    MoERoutingStage::Params params;
    params.device_id = device;
    params.input = input;
    params.gate_weights = gate_weights;
    params.output_indices = output_indices;
    params.output_weights = output_weights;
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.num_experts = num_experts;
    params.top_k = top_k;
    params.norm_topk_prob = true;
    params.layer_idx = 0;
    return params;
}

// =============================================================================
// Test: DefaultConfig
// =============================================================================

TEST(Test__PrefillGraphCache, DefaultConfig_MatchesExpectedDefaults)
{
    PrefillGraphConfig config;
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.min_seq_len, 256);
    EXPECT_FALSE(config.trace);
    EXPECT_FALSE(config.buckets_enabled);
    EXPECT_EQ(config.max_cached_entries, 10u);
}

// =============================================================================
// Test: Phase 6 bucket selection and chunk planning helpers
// =============================================================================

TEST(Test__PrefillGraphCache, BucketSelection_ExactBucket)
{
    auto selection = selectPrefillGraphBucket(512, defaultPrefillGraphBuckets());

    ASSERT_TRUE(selection);
    EXPECT_EQ(selection.real_seq_len, 512);
    EXPECT_EQ(selection.bucket_seq_len, 512);
    EXPECT_TRUE(selection.exact);
}

TEST(Test__PrefillGraphCache, BucketSelection_JustOverBucket)
{
    auto selection = selectPrefillGraphBucket(513, defaultPrefillGraphBuckets());

    ASSERT_TRUE(selection);
    EXPECT_EQ(selection.real_seq_len, 513);
    EXPECT_EQ(selection.bucket_seq_len, 544);
    EXPECT_FALSE(selection.exact);
}

TEST(Test__PrefillGraphCache, BucketSelection_DefaultsIncludeDenseBenchmarkLane)
{
    auto buckets = defaultPrefillGraphBuckets();
    EXPECT_NE(std::find(buckets.begin(), buckets.end(), 600), buckets.end());

    auto selection = selectPrefillGraphBucket(595, buckets);

    ASSERT_TRUE(selection);
    EXPECT_EQ(selection.real_seq_len, 595);
    EXPECT_EQ(selection.bucket_seq_len, 600);
    EXPECT_FALSE(selection.exact);
}

TEST(Test__PrefillGraphCache, BucketSelection_RejectsAboveMaximum)
{
    auto selection = selectPrefillGraphBucket(5000, defaultPrefillGraphBuckets());

    EXPECT_FALSE(selection);
    EXPECT_EQ(selection.bucket_seq_len, 0);
    EXPECT_NE(selection.error.find("largest"), std::string::npos);
}

TEST(Test__PrefillGraphCache, BucketSelection_NormalizesConfiguredBuckets)
{
    const std::vector<int> unsorted = {512, -1, 128, 128, 64, 0, 256};
    auto normalized = normalizePrefillGraphBuckets(unsorted);

    EXPECT_EQ(normalized, (std::vector<int>{64, 128, 256, 512}));

    auto selection = selectPrefillGraphBucket(129, unsorted);
    ASSERT_TRUE(selection);
    EXPECT_EQ(selection.bucket_seq_len, 256);
}

TEST(Test__PrefillGraphCache, BucketPadding_CopiesRealTokensAndPadsTail)
{
    const int tokens[] = {10, 11, 12};
    auto padded = padPrefillTokensToBucket(tokens, 3, 5, 42);

    EXPECT_EQ(padded, (std::vector<int>{10, 11, 12, 42, 42}));
}

TEST(Test__PrefillGraphCache, ChunkPlanning_UsesLargestBucketsThenRemainder)
{
    const std::vector<int> buckets = {64, 128, 256};
    auto chunks = planPrefillChunks(600, buckets);

    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[0].token_offset, 0);
    EXPECT_EQ(chunks[0].real_count, 256);
    EXPECT_EQ(chunks[0].bucket_seq_len, 256);
    EXPECT_EQ(chunks[1].token_offset, 256);
    EXPECT_EQ(chunks[1].real_count, 256);
    EXPECT_EQ(chunks[1].bucket_seq_len, 256);
    EXPECT_EQ(chunks[2].token_offset, 512);
    EXPECT_EQ(chunks[2].real_count, 88);
    EXPECT_EQ(chunks[2].bucket_seq_len, 128);
}

TEST(Test__PrefillGraphCache, ChunkSchedule_UsesFixedIntervalAndRealTokenRange)
{
    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {64, 128};
    policy.fixed_chunk_real_tokens = 96;
    policy.real_token_start = 32;
    policy.real_token_count = 250;

    auto schedule = planPrefillChunkSchedule(policy);

    ASSERT_TRUE(schedule) << schedule.error;
    ASSERT_EQ(schedule.chunks.size(), 3u);
    EXPECT_EQ(schedule.chunks[0].chunk_index, 0);
    EXPECT_EQ(schedule.chunks[0].token_offset, 32);
    EXPECT_EQ(schedule.chunks[0].real_count, 96);
    EXPECT_EQ(schedule.chunks[0].bucket_seq_len, 128);
    EXPECT_EQ(schedule.chunks[1].chunk_index, 1);
    EXPECT_EQ(schedule.chunks[1].token_offset, 128);
    EXPECT_EQ(schedule.chunks[1].real_count, 96);
    EXPECT_EQ(schedule.chunks[1].bucket_seq_len, 128);
    EXPECT_EQ(schedule.chunks[2].chunk_index, 2);
    EXPECT_EQ(schedule.chunks[2].token_offset, 224);
    EXPECT_EQ(schedule.chunks[2].real_count, 58);
    EXPECT_EQ(schedule.chunks[2].bucket_seq_len, 64);
}

TEST(Test__PrefillGraphCache, ChunkSchedule_RebalanceIntervalsCountRealTokensOnly)
{
    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {64};
    policy.fixed_chunk_real_tokens = 32;
    policy.min_rebalance_interval_tokens = 64;
    policy.max_rebalance_interval_tokens = 64;
    policy.real_token_start = 0;
    policy.real_token_count = 96;

    auto schedule = planPrefillChunkSchedule(policy);

    ASSERT_TRUE(schedule) << schedule.error;
    ASSERT_EQ(schedule.chunks.size(), 3u);
    EXPECT_EQ(schedule.chunks[0].bucket_seq_len, 64);
    EXPECT_EQ(schedule.chunks[0].real_count, 32);
    EXPECT_FALSE(schedule.chunks[0].rebalance_allowed_after);
    EXPECT_FALSE(schedule.chunks[0].rebalance_required_after);

    EXPECT_EQ(schedule.chunks[1].bucket_seq_len, 64);
    EXPECT_EQ(schedule.chunks[1].real_count, 32);
    EXPECT_TRUE(schedule.chunks[1].rebalance_allowed_after);
    EXPECT_TRUE(schedule.chunks[1].rebalance_required_after);

    EXPECT_EQ(schedule.chunks[2].bucket_seq_len, 64);
    EXPECT_EQ(schedule.chunks[2].real_count, 32);
    EXPECT_FALSE(schedule.chunks[2].rebalance_allowed_after);
    EXPECT_FALSE(schedule.chunks[2].rebalance_required_after);
}

TEST(Test__PrefillGraphCache, ChunkSchedule_RejectsInvalidPolicy)
{
    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {64};
    policy.fixed_chunk_real_tokens = 128;
    policy.real_token_count = 256;

    auto schedule = planPrefillChunkSchedule(policy);

    EXPECT_FALSE(schedule);
    EXPECT_NE(schedule.error.find("largest"), std::string::npos);

    policy.fixed_chunk_real_tokens = 32;
    policy.min_rebalance_interval_tokens = 128;
    policy.max_rebalance_interval_tokens = 64;

    schedule = planPrefillChunkSchedule(policy);
    EXPECT_FALSE(schedule);
    EXPECT_NE(schedule.error.find("min"), std::string::npos);
}

TEST(Test__PrefillGraphCache, ChunkMaintenanceGate_RequiresMergedHistogramsAndCompletedBoundaries)
{
    PrefillChunkPlan chunk;
    chunk.chunk_index = 3;
    chunk.rebalance_allowed_after = true;

    PrefillChunkMaintenanceState state;
    state.chunk_index = 3;
    state.rebalance_requested = true;

    auto decision = evaluatePrefillChunkMaintenance(chunk, state);
    EXPECT_FALSE(decision);
    EXPECT_FALSE(decision.can_run);
    EXPECT_EQ(decision.reason, "histograms_not_merged");

    state.histograms_merged = true;
    state.manual_boundaries_complete = false;
    decision = evaluatePrefillChunkMaintenance(chunk, state);
    EXPECT_FALSE(decision);
    EXPECT_EQ(decision.reason, "manual_boundary_incomplete");

    state.manual_boundaries_complete = true;
    state.graph_capture_active = true;
    decision = evaluatePrefillChunkMaintenance(chunk, state);
    EXPECT_FALSE(decision);
    EXPECT_EQ(decision.reason, "graph_capture_active");

    state.graph_capture_active = false;
    state.graph_replay_active = true;
    decision = evaluatePrefillChunkMaintenance(chunk, state);
    EXPECT_FALSE(decision);
    EXPECT_EQ(decision.reason, "graph_replay_active");
}

TEST(Test__PrefillGraphCache, ChunkMaintenanceGate_ReadyRequestedRequiredAndParticipantAlignment)
{
    PrefillChunkPlan chunk;
    chunk.chunk_index = 1;
    chunk.rebalance_allowed_after = true;

    PrefillChunkMaintenanceState state;
    state.chunk_index = 1;
    state.histograms_merged = true;

    auto decision = evaluatePrefillChunkMaintenance(chunk, state);
    EXPECT_TRUE(decision);
    EXPECT_FALSE(decision.can_run);
    EXPECT_FALSE(decision.required);
    EXPECT_EQ(decision.reason, "rebalance_not_requested");

    state.rebalance_requested = true;
    decision = evaluatePrefillChunkMaintenance(chunk, state);
    EXPECT_TRUE(decision);
    EXPECT_TRUE(decision.can_run);
    EXPECT_FALSE(decision.required);
    EXPECT_EQ(decision.reason, "ready");

    state.participants_at_same_boundary = false;
    decision = evaluatePrefillChunkMaintenance(chunk, state);
    EXPECT_FALSE(decision);
    EXPECT_EQ(decision.reason, "participants_not_at_same_boundary");

    state.participants_at_same_boundary = true;
    state.rebalance_requested = false;
    chunk.rebalance_required_after = true;
    decision = evaluatePrefillChunkMaintenance(chunk, state);
    EXPECT_TRUE(decision);
    EXPECT_TRUE(decision.can_run);
    EXPECT_TRUE(decision.required);
    EXPECT_EQ(decision.reason, "required");
}

TEST(Test__PrefillGraphCache, ChunkPositionIds_ExactChunk)
{
    auto position_ids = buildPrefillChunkPositionIds(
        /*real_count=*/5,
        /*bucket_seq_len=*/5,
        /*token_offset=*/0);

    EXPECT_EQ(position_ids, (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(Test__PrefillGraphCache, ChunkPositionIds_PaddedFinalChunkUsesAbsolutePositions)
{
    auto position_ids = buildPrefillChunkPositionIds(
        /*real_count=*/3,
        /*bucket_seq_len=*/5,
        /*token_offset=*/512);

    EXPECT_EQ(position_ids, (std::vector<int>{512, 513, 514, 515, 516}));
}

TEST(Test__PrefillGraphCache, ChunkPositionIds_ReplicatesAcrossBatches)
{
    auto position_ids = buildPrefillChunkPositionIds(
        /*real_count=*/3,
        /*bucket_seq_len=*/4,
        /*token_offset=*/7,
        /*batch_size=*/2);

    EXPECT_EQ(position_ids, (std::vector<int>{7, 8, 9, 10, 7, 8, 9, 10}));
}

TEST(Test__PrefillGraphCache, ChunkPositionIds_RejectsInvalidInputs)
{
    EXPECT_TRUE(buildPrefillChunkPositionIds(0, 4, 0).empty());
    EXPECT_TRUE(buildPrefillChunkPositionIds(5, 4, 0).empty());
    EXPECT_TRUE(buildPrefillChunkPositionIds(3, 4, -1).empty());
    EXPECT_TRUE(buildPrefillChunkPositionIds(3, 4, 0, 0).empty());
}

TEST(Test__PrefillGraphCache, ForwardSignature_BucketedPrefillUsesBucketShape)
{
    ForwardGraphSignature first{};
    first.seq_len = 768;
    first.batch_size = 1;
    first.device = makeGPUKey(768).device_id;
    first.decode = false;
    first.standard_path = true;
    first.is_bucketed_prefill = true;
    first.bucket_seq_len = 768;

    ForwardGraphSignature second = first;
    ForwardGraphSignature exact_shape_without_bucket = first;
    exact_shape_without_bucket.is_bucketed_prefill = false;
    exact_shape_without_bucket.bucket_seq_len = 0;
    ForwardGraphSignature different_bucket = first;
    different_bucket.seq_len = 1024;
    different_bucket.bucket_seq_len = 1024;

    EXPECT_EQ(first, second);
    EXPECT_NE(first, exact_shape_without_bucket);
    EXPECT_NE(first, different_bucket);
    EXPECT_EQ(ForwardGraphSignatureHash{}(first), ForwardGraphSignatureHash{}(second));
}

TEST(Test__PrefillGraphCache, ChunkExecutionInput_BuildsExactChunkBuffers)
{
    const std::vector<int> tokens = {10, 11, 12, 13};
    const PrefillChunkPlan chunk{0, 4, 4};

    auto input = buildPrefillChunkExecutionInput(tokens.data(), static_cast<int>(tokens.size()), chunk, 99);

    ASSERT_TRUE(input);
    EXPECT_EQ(input.token_offset, 0);
    EXPECT_EQ(input.real_count, 4);
    EXPECT_EQ(input.bucket_seq_len, 4);
    EXPECT_EQ(input.token_ids, tokens);
    EXPECT_EQ(input.position_ids, (std::vector<int>{0, 1, 2, 3}));
}

TEST(Test__PrefillGraphCache, ChunkExecutionInput_BuildsPaddedFinalChunkBuffers)
{
    const std::vector<int> tokens = {10, 11, 12, 13, 14, 15};
    const PrefillChunkPlan chunk{4, 2, 4};

    auto input = buildPrefillChunkExecutionInput(tokens.data(), static_cast<int>(tokens.size()), chunk, 99);

    ASSERT_TRUE(input) << input.error;
    EXPECT_EQ(input.token_offset, 4);
    EXPECT_EQ(input.real_count, 2);
    EXPECT_EQ(input.bucket_seq_len, 4);
    EXPECT_EQ(input.token_ids, (std::vector<int>{14, 15, 99, 99}));
    EXPECT_EQ(input.position_ids, (std::vector<int>{4, 5, 6, 7}));
}

TEST(Test__PrefillGraphCache, ChunkExecutionInput_RejectsOutOfRangeChunk)
{
    const std::vector<int> tokens = {10, 11, 12};
    const PrefillChunkPlan chunk{2, 2, 4};

    auto input = buildPrefillChunkExecutionInput(tokens.data(), static_cast<int>(tokens.size()), chunk, 99);

    EXPECT_FALSE(input);
    EXPECT_NE(input.error.find("exceeds"), std::string::npos);
}

// =============================================================================
// Test: Phase Transitions
// =============================================================================

TEST(Test__PrefillGraphCache, PhaseTransitions_ColdInitially)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Cold);
    EXPECT_FALSE(cache.hasGraph(key));
}

TEST(Test__PrefillGraphCache, PhaseTransitions_MarkWarmedUp)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    cache.markWarmedUp(key);
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Warmup);
    EXPECT_FALSE(cache.hasGraph(key));
    EXPECT_EQ(cache.size(), 1u);
}

// =============================================================================
// Test: Preflight Rejections
// =============================================================================

TEST(Test__PrefillGraphCache, Preflight_RejectsDisabledConfig)
{
    PrefillGraphConfig config;
    config.enabled = false;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    auto graph = buildCapturableGraph(key.device_id);

    auto reason = cache.preflight(graph, key, nullptr, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::FeatureDisabled);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsLowSeqLen)
{
    PrefillGraphConfig config;
    config.min_seq_len = 256;
    PrefillGraphCache cache(config);

    PrefillGraphCacheKey key;
    key.seq_len = 64;
    key.device_id = DeviceId::rocm(0);
    auto graph = buildCapturableGraph(key.device_id);

    auto reason = cache.preflight(graph, key, nullptr, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::SeqLenBelowMinimum);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsCPUDevice)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = DeviceId::cpu();
    auto graph = buildCapturableGraph(key.device_id);

    auto reason = cache.preflight(graph, key, nullptr, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::NotGPUDevice);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsSnapshots)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    auto graph = buildCapturableGraph(key.device_id);

    auto reason = cache.preflight(graph, key, nullptr, /*snapshots_active=*/true);
    EXPECT_EQ(reason, PrefillGraphRejectReason::SnapshotsActive);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsMoERebalancing)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    auto graph = buildCapturableGraph(key.device_id);

    auto reason = cache.preflight(graph, key, nullptr, false, /*moe_rebalancing_active=*/true);
    EXPECT_EQ(reason, PrefillGraphRejectReason::ActiveMoERebalancing);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsCollectives)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    auto graph = buildCapturableGraph(key.device_id);

    std::unordered_set<std::string> collectives = {"allreduce_0"};
    auto reason = cache.preflight(graph, key, &collectives, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::CollectiveNodesPresent);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsNonCapturableStage)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);

    // Build graph with a non-capturable stage
    ComputeGraph graph;
    graph.addNode("stage_0", std::make_unique<CapturableMockStage>("stage_0", key.device_id), key.device_id);
    graph.addNode("stage_1", std::make_unique<NonCapturableMockStage>("stage_1", key.device_id), key.device_id);
    graph.addDependency("stage_1", "stage_0");

    auto reason = cache.preflight(graph, key, nullptr, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::StageNotCapturable);
}

TEST(Test__PrefillGraphCache, Preflight_ColdPaddedBucketUsesSupportBeforeWarmupReadiness)
{
    PrefillGraphConfig config;
    config.min_seq_len = 1;
    config.buckets_enabled = true;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(608);
    ComputeGraph graph;
    auto stage = std::make_unique<ColdPaddedPreflightOnlyMockStage>("warmup_ready_stage", key.device_id);
    auto *stage_ptr = stage.get();
    graph.addNode("warmup_ready_stage", std::move(stage), key.device_id);

    auto reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/595,
        /*bucket_seq_len=*/608);
    EXPECT_EQ(reason, PrefillGraphRejectReason::None);

    cache.markWarmedUp(key);
    reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/595,
        /*bucket_seq_len=*/608);
    EXPECT_EQ(reason, PrefillGraphRejectReason::StageNotCapturable);

    stage_ptr->setCaptureReady(true);
    reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/595,
        /*bucket_seq_len=*/608);
    EXPECT_EQ(reason, PrefillGraphRejectReason::None);
}

TEST(Test__PrefillGraphCache, Preflight_AcceptsColdPaddedRocmMoERoutingBeforeKernelWarmup)
{
    ScopedRocmGroupedPrefillFlag grouped_prefill(true);

    constexpr int seq_len = 608;
    constexpr int real_seq_len = 595;
    constexpr int d_model = 64;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;

    auto input = TestTensorFactory::createFP32({seq_len, d_model});
    auto gate_weights = TestTensorFactory::createFP32({num_experts, d_model});
    auto output_indices = TestTensorFactory::createFP32({seq_len * top_k, 1});
    auto output_weights = TestTensorFactory::createFP32({seq_len * top_k, 1});

    PrefillGraphConfig config;
    config.min_seq_len = 1;
    config.buckets_enabled = true;
    PrefillGraphCache cache(config);

    PrefillGraphCacheKey key;
    key.seq_len = seq_len;
    key.device_id = DeviceId::rocm(0);

    ComputeGraph graph;
    auto params = makeColdRocmRoutingParams(
        input.get(),
        gate_weights.get(),
        output_indices.get(),
        output_weights.get(),
        key.device_id,
        seq_len,
        d_model,
        num_experts,
        top_k);
    graph.addNode("layer0_moe_routing", std::make_unique<MoERoutingStage>(params), key.device_id);

    const auto reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        real_seq_len,
        seq_len);

#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_EQ(reason, PrefillGraphRejectReason::None);
#else
    EXPECT_EQ(reason, PrefillGraphRejectReason::StageNotCapturable);
#endif
}

TEST(Test__PrefillGraphCache, Preflight_RejectsColdPaddedMoERoutingWhenUnsupported)
{
    constexpr int seq_len = 608;
    constexpr int real_seq_len = 595;
    constexpr int d_model = 64;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;

    auto input = TestTensorFactory::createFP32({seq_len, d_model});
    auto gate_weights = TestTensorFactory::createFP32({num_experts, d_model});
    auto output_indices = TestTensorFactory::createFP32({seq_len * top_k, 1});
    auto output_weights = TestTensorFactory::createFP32({seq_len * top_k, 1});

    PrefillGraphConfig config;
    config.min_seq_len = 1;
    config.buckets_enabled = true;
    PrefillGraphCache cache(config);

    {
        ScopedRocmGroupedPrefillFlag grouped_prefill(false);
        PrefillGraphCacheKey key;
        key.seq_len = seq_len;
        key.device_id = DeviceId::rocm(0);

        ComputeGraph graph;
        auto params = makeColdRocmRoutingParams(
            input.get(), gate_weights.get(), output_indices.get(), output_weights.get(),
            key.device_id, seq_len, d_model, num_experts, top_k);
        graph.addNode("layer0_moe_routing", std::make_unique<MoERoutingStage>(params), key.device_id);

        const auto reason = cache.preflight(
            graph, key, nullptr, false, false, real_seq_len, seq_len);
        EXPECT_EQ(reason, PrefillGraphRejectReason::StageNotCapturable);
    }

    {
        ScopedRocmGroupedPrefillFlag grouped_prefill(true);
        PrefillGraphCacheKey key;
        key.seq_len = seq_len;
        key.device_id = DeviceId::cuda(0);

        ComputeGraph graph;
        auto params = makeColdRocmRoutingParams(
            input.get(), gate_weights.get(), output_indices.get(), output_weights.get(),
            key.device_id, seq_len, d_model, num_experts, top_k);
        graph.addNode("layer0_moe_routing", std::make_unique<MoERoutingStage>(params), key.device_id);

        const auto reason = cache.preflight(
            graph, key, nullptr, false, false, real_seq_len, seq_len);
        EXPECT_EQ(reason, PrefillGraphRejectReason::StageNotCapturable);
    }
}

TEST(Test__PrefillGraphCache, Preflight_AcceptsAllCapturableGraph)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    auto graph = buildCapturableGraph(key.device_id);

    auto reason = cache.preflight(graph, key, nullptr, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::None);
}

TEST(Test__PrefillGraphCache, Preflight_AcceptsEmptyCollectiveSet)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    auto graph = buildCapturableGraph(key.device_id);

    std::unordered_set<std::string> empty_collectives;
    auto reason = cache.preflight(graph, key, &empty_collectives, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::None);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsPaddedBucketWithUnsupportedGDNState)
{
    PrefillGraphConfig config;
    config.min_seq_len = 1;
    config.buckets_enabled = true;
    PrefillGraphCache cache(config);
    auto key = makeGPUKey(768);
    ComputeGraph graph;
    graph.addNode("gdn_recurrence",
                  std::make_unique<MockComputeStage>(
                      ComputeStageType::GDN_RECURRENCE,
                      "gdn_recurrence",
                      key.device_id),
                  key.device_id);

    auto reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/513,
        /*bucket_seq_len=*/768);

    EXPECT_EQ(reason, PrefillGraphRejectReason::GDNWithPaddedBucket);
    EXPECT_STREQ(toString(reason), "GDNWithPaddedBucket");
}

TEST(Test__PrefillGraphCache, Preflight_AcceptsPaddedBucketWithSupportedGDNState)
{
    PrefillGraphConfig config;
    config.min_seq_len = 1;
    config.buckets_enabled = true;
    PrefillGraphCache cache(config);
    auto key = makeGPUKey(768);
    ComputeGraph graph;
    graph.addNode("short_conv",
                  std::make_unique<PaddedRealLengthContractMockStage>(
                      ComputeStageType::SHORT_CONV1D,
                      "short_conv",
                      key.device_id),
                  key.device_id);
    graph.addNode("gdn_recurrence",
                  std::make_unique<PaddedRealLengthContractMockStage>(
                      ComputeStageType::GDN_RECURRENCE,
                      "gdn_recurrence",
                      key.device_id),
                  key.device_id);
    graph.addDependency("gdn_recurrence", "short_conv");

    auto reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/513,
        /*bucket_seq_len=*/768);

    EXPECT_EQ(reason, PrefillGraphRejectReason::None);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsPaddedBucketIfAnyGDNStateLacksContract)
{
    PrefillGraphConfig config;
    config.min_seq_len = 1;
    config.buckets_enabled = true;
    PrefillGraphCache cache(config);
    auto key = makeGPUKey(768);
    ComputeGraph graph;
    graph.addNode("short_conv",
                  std::make_unique<PaddedRealLengthContractMockStage>(
                      ComputeStageType::SHORT_CONV1D,
                      "short_conv",
                      key.device_id),
                  key.device_id);
    graph.addNode("gdn_recurrence",
                  std::make_unique<MockComputeStage>(
                      ComputeStageType::GDN_RECURRENCE,
                      "gdn_recurrence",
                      key.device_id),
                  key.device_id);
    graph.addDependency("gdn_recurrence", "short_conv");

    auto reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/513,
        /*bucket_seq_len=*/768);

    EXPECT_EQ(reason, PrefillGraphRejectReason::GDNWithPaddedBucket);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsSupportedPaddedGDNWhenStageIsNotCapturable)
{
    PrefillGraphConfig config;
    config.min_seq_len = 1;
    config.buckets_enabled = true;
    PrefillGraphCache cache(config);
    auto key = makeGPUKey(768);
    ComputeGraph graph;
    graph.addNode("gdn_recurrence",
                  std::make_unique<NonCapturablePaddedRealLengthContractMockStage>(
                      ComputeStageType::GDN_RECURRENCE,
                      "gdn_recurrence",
                      key.device_id),
                  key.device_id);

    auto reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/513,
        /*bucket_seq_len=*/768);

    EXPECT_EQ(reason, PrefillGraphRejectReason::StageNotCapturable);
}

TEST(Test__PrefillGraphCache, Preflight_RejectsPaddedBucketWhenBucketsDisabled)
{
    PrefillGraphConfig config;
    config.min_seq_len = 1;
    config.buckets_enabled = false;
    PrefillGraphCache cache(config);
    auto key = makeGPUKey(768);
    auto graph = buildCapturableGraph(key.device_id);

    auto reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/513,
        /*bucket_seq_len=*/768);

    EXPECT_EQ(reason, PrefillGraphRejectReason::FeatureDisabled);
}

TEST(Test__PrefillGraphCache, Preflight_AllowsExactBucketWithGDNState)
{
    PrefillGraphConfig config;
    config.min_seq_len = 1;
    PrefillGraphCache cache(config);
    auto key = makeGPUKey(768);
    ComputeGraph graph;
    graph.addNode("gdn_recurrence",
                  std::make_unique<MockComputeStage>(
                      ComputeStageType::GDN_RECURRENCE,
                      "gdn_recurrence",
                      key.device_id),
                  key.device_id);

    auto reason = cache.preflight(
        graph,
        key,
        nullptr,
        /*snapshots_active=*/false,
        /*moe_rebalancing_active=*/false,
        /*real_seq_len=*/768,
        /*bucket_seq_len=*/768);

    EXPECT_EQ(reason, PrefillGraphRejectReason::None);
}

// =============================================================================
// Test: Invalidation
// =============================================================================

TEST(Test__PrefillGraphCache, InvalidateAll_ResetsEntries)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key1 = makeGPUKey(256);
    auto key2 = makeGPUKey(512);
    cache.markWarmedUp(key1);
    cache.markWarmedUp(key2);

    EXPECT_EQ(cache.phase(key1), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(key2), PrefillGraphPhase::Warmup);

    cache.invalidateAll();

    EXPECT_EQ(cache.phase(key1), PrefillGraphPhase::Cold);
    EXPECT_EQ(cache.phase(key2), PrefillGraphPhase::Cold);
    EXPECT_FALSE(cache.hasGraph(key1));
    EXPECT_FALSE(cache.hasGraph(key2));
}

TEST(Test__PrefillGraphCache, Invalidate_SingleEntry)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key1 = makeGPUKey(256);
    auto key2 = makeGPUKey(512);
    cache.markWarmedUp(key1);
    cache.markWarmedUp(key2);

    cache.invalidate(key1);

    EXPECT_EQ(cache.phase(key1), PrefillGraphPhase::Cold);
    EXPECT_EQ(cache.phase(key2), PrefillGraphPhase::Warmup);
}

// =============================================================================
// Test: Launch/Capture Guards
// =============================================================================

TEST(Test__PrefillGraphCache, Launch_FailsIfNotReady)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    EXPECT_FALSE(cache.launch(key));

    cache.markWarmedUp(key);
    EXPECT_FALSE(cache.launch(key));
}

TEST(Test__PrefillGraphCache, BeginCapture_FailsIfNotWarmedUp)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    // Cold state - should fail
    EXPECT_FALSE(cache.beginCapture(key, nullptr, nullptr));
}

TEST(Test__PrefillGraphCache, BeginCapture_FailsWithNullContext)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    cache.markWarmedUp(key);
    // Null GPU context - should fail
    EXPECT_FALSE(cache.beginCapture(key, nullptr, nullptr));
}

TEST(Test__PrefillGraphCache, BeginCapture_FailsWithNullExplicitStream)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);
    PrefillMockGPUContext gpu_ctx;

    auto key = makeGPUKey(512);
    cache.markWarmedUp(key);

    EXPECT_FALSE(cache.beginCapture(key, &gpu_ctx, nullptr));
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Warmup);
    EXPECT_EQ(gpu_ctx.default_capture_calls_, 0);
    EXPECT_EQ(gpu_ctx.explicit_capture_calls_, 0);
}

TEST(Test__PrefillGraphCache, BeginCapture_UsesExplicitStreamOverload)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);
    PrefillMockGPUContext gpu_ctx;

    auto key = makeGPUKey(512);
    cache.markWarmedUp(key);
    void *explicit_stream = gpu_ctx.createStream();

    EXPECT_TRUE(cache.beginCapture(key, &gpu_ctx, explicit_stream));
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Capturing);
    EXPECT_EQ(gpu_ctx.default_capture_calls_, 0);
    EXPECT_EQ(gpu_ctx.explicit_capture_calls_, 1);
    EXPECT_EQ(gpu_ctx.last_capture_stream_, explicit_stream);
}

// =============================================================================
// Test: Cache Key Distinctness
// =============================================================================

TEST(Test__PrefillGraphCache, CacheKey_DifferentSeqLensAreDistinct)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key1 = makeGPUKey(256);
    auto key2 = makeGPUKey(512);

    cache.markWarmedUp(key1);
    EXPECT_EQ(cache.phase(key1), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(key2), PrefillGraphPhase::Cold);
    EXPECT_EQ(cache.size(), 1u);

    cache.markWarmedUp(key2);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(Test__PrefillGraphCache, Capacity_EvictsLeastRecentlyUsedBucket)
{
    PrefillGraphConfig config;
    config.max_cached_entries = 2;
    PrefillGraphCache cache(config);

    auto key64 = makeGPUKey(64);
    auto key128 = makeGPUKey(128);
    auto key256 = makeGPUKey(256);

    cache.markWarmedUp(key64);
    cache.markWarmedUp(key128);
    EXPECT_EQ(cache.size(), 2u);

    // Touch key64 so key128 is the least-recently-used entry when the third
    // bucket is inserted.
    cache.markWarmedUp(key64);
    cache.markWarmedUp(key256);

    EXPECT_EQ(cache.size(), 2u);
    EXPECT_EQ(cache.phase(key64), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(key128), PrefillGraphPhase::Cold);
    EXPECT_EQ(cache.phase(key256), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.evictionCount(), 1u);
}

TEST(Test__PrefillGraphCache, Capacity_EvictedBucketWarmsAndCapturesAgain)
{
    PrefillGraphConfig config;
    config.max_cached_entries = 1;
    PrefillGraphCache cache(config);
    PrefillMockGPUContext gpu_ctx;
    void *explicit_stream = gpu_ctx.createStream();

    auto key256 = makeGPUKey(256);
    auto key512 = makeGPUKey(512);

    // First lifecycle for key256: warm up, capture, instantiate, and become ready.
    cache.markWarmedUp(key256);
    ASSERT_TRUE(cache.beginCapture(key256, &gpu_ctx, explicit_stream));
    ASSERT_TRUE(cache.endCaptureAndInstantiate(key256));
    EXPECT_EQ(cache.phase(key256), PrefillGraphPhase::Ready);
    EXPECT_EQ(cache.warmupCount(key256), 1u);
    EXPECT_EQ(cache.captureCount(key256), 1u);

    // Inserting key512 evicts key256 because the cap is one entry.
    cache.markWarmedUp(key512);
    EXPECT_EQ(cache.phase(key256), PrefillGraphPhase::Cold);
    EXPECT_EQ(cache.phase(key512), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.evictionCount(), 1u);

    // A new key256 request must explicitly warm and capture again. Lifetime
    // counters survive the eviction, so the test distinguishes recapture from
    // any silent normal-path execution that would leave these counts unchanged.
    cache.markWarmedUp(key256);
    ASSERT_TRUE(cache.beginCapture(key256, &gpu_ctx, explicit_stream));
    ASSERT_TRUE(cache.endCaptureAndInstantiate(key256));

    EXPECT_EQ(cache.phase(key256), PrefillGraphPhase::Ready);
    EXPECT_EQ(cache.warmupCount(key256), 2u);
    EXPECT_EQ(cache.captureCount(key256), 2u);
    EXPECT_EQ(cache.evictionCount(), 2u);
}

TEST(Test__PrefillGraphCache, Capacity_ZeroMeansUnlimited)
{
    PrefillGraphConfig config;
    config.max_cached_entries = 0;
    PrefillGraphCache cache(config);

    cache.markWarmedUp(makeGPUKey(64));
    cache.markWarmedUp(makeGPUKey(128));
    cache.markWarmedUp(makeGPUKey(256));

    EXPECT_EQ(cache.size(), 3u);
    EXPECT_EQ(cache.evictionCount(), 0u);
}

TEST(Test__PrefillGraphCache, CacheKey_PlacementEpochDifference)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    PrefillGraphCacheKey key1;
    key1.seq_len = 512;
    key1.device_id = DeviceId::rocm(0);
    key1.placement_epoch = 0;

    PrefillGraphCacheKey key2 = key1;
    key2.placement_epoch = 1;

    cache.markWarmedUp(key1);
    cache.markWarmedUp(key2);

    EXPECT_EQ(cache.size(), 2u);
    cache.invalidate(key1);
    EXPECT_EQ(cache.phase(key1), PrefillGraphPhase::Cold);
    EXPECT_EQ(cache.phase(key2), PrefillGraphPhase::Warmup);
}

TEST(Test__PrefillGraphCache, CacheKey_DomainParticipantAndChunkRangesDoNotAlias)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    PrefillGraphCacheKey base = makeGPUKey(512);
    base.domain_id = "continuation";
    base.participant_id = 0;
    base.real_token_count = 384;
    base.first_layer = 0;
    base.layer_count = 8;
    base.placement_epoch = 7;
    base.topology_signature = 101;

    auto different_domain = base;
    different_domain.domain_id = "expert_hot";

    auto different_participant = base;
    different_participant.participant_id = 1;

    auto different_real_tokens = base;
    different_real_tokens.real_token_count = 256;

    auto different_layer_range = base;
    different_layer_range.first_layer = 8;

    cache.markWarmedUp(base);
    cache.markWarmedUp(different_domain);
    cache.markWarmedUp(different_participant);
    cache.markWarmedUp(different_real_tokens);
    cache.markWarmedUp(different_layer_range);

    EXPECT_EQ(cache.size(), 5u);
    EXPECT_EQ(cache.phase(base), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(different_domain), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(different_participant), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(different_real_tokens), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(different_layer_range), PrefillGraphPhase::Warmup);

    cache.invalidate(base);
    EXPECT_EQ(cache.phase(base), PrefillGraphPhase::Cold);
    EXPECT_EQ(cache.phase(different_domain), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(different_participant), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(different_real_tokens), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(different_layer_range), PrefillGraphPhase::Warmup);
}

// =============================================================================
// Test: toString helper
// =============================================================================

TEST(Test__PrefillGraphCache, ToString_RejectReasons)
{
    EXPECT_STREQ(toString(PrefillGraphRejectReason::None), "None");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::FeatureDisabled), "FeatureDisabled");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::SeqLenBelowMinimum), "SeqLenBelowMinimum");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::NotGPUDevice), "NotGPUDevice");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::SnapshotsActive), "SnapshotsActive");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::ActiveMoERebalancing), "ActiveMoERebalancing");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::CollectiveNodesPresent), "CollectiveNodesPresent");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::StageNotCapturable), "StageNotCapturable");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::NoGPUContext), "NoGPUContext");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::InvalidatedByPlacement), "InvalidatedByPlacement");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::SessionReset), "SessionReset");
    EXPECT_STREQ(toString(PrefillGraphRejectReason::RequestStateReset), "RequestStateReset");
}

// =============================================================================
// Test: Node count and replay count when not ready
// =============================================================================

TEST(Test__PrefillGraphCache, NodeCount_ZeroWhenNotReady)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    EXPECT_EQ(cache.nodeCount(key), 0u);

    cache.markWarmedUp(key);
    EXPECT_EQ(cache.nodeCount(key), 0u);
}

TEST(Test__PrefillGraphCache, ReplayCount_ZeroWhenNeverLaunched)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    EXPECT_EQ(cache.replayCount(key), 0);

    cache.markWarmedUp(key);
    EXPECT_EQ(cache.replayCount(key), 0);
}

// =============================================================================
// GPU-Backend Tests (ROCm or CUDA required)
// These test the full capture/replay lifecycle on real hardware.
// =============================================================================

#if defined(HAVE_ROCM) || defined(HAVE_CUDA)

class PrefillGraphCacheGPUTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
#ifdef HAVE_ROCM
        ensureAMDFactoryRegistered();
        if (!GPUDeviceContextPool::instance().hasAMDSupport())
            GTEST_SKIP() << "ROCm not available";
        gpu_ctx_ = &GPUDeviceContextPool::instance().getAMDContext(0);
#elif defined(HAVE_CUDA)
        ensureNvidiaFactoryRegistered();
        if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
            GTEST_SKIP() << "CUDA not available";
        gpu_ctx_ = &GPUDeviceContextPool::instance().getNvidiaContext(0);
#endif
    }

    IWorkerGPUContext *gpu_ctx_ = nullptr;
};

TEST_F(PrefillGraphCacheGPUTest, FullLifecycle_ColdToReady)
{
    PrefillGraphConfig config;
    config.trace = true;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Cold);

    // Warmup
    cache.markWarmedUp(key);
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Warmup);

    // All GPU operations inside submitAndWait
    gpu_ctx_->submitAndWait([&]
                            {
        void *stream = gpu_ctx_->defaultStream();

        // Begin capture (empty capture is valid — validates state machine)
        ASSERT_TRUE(cache.beginCapture(key, gpu_ctx_, stream));
        EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Capturing);

        // End capture and instantiate (empty graph)
        ASSERT_TRUE(cache.endCaptureAndInstantiate(key));
        EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Ready);
        EXPECT_TRUE(cache.hasGraph(key));

        // Launch (replay)
        ASSERT_TRUE(cache.launch(key));
        EXPECT_EQ(cache.replayCount(key), 1);

        // Launch again
        ASSERT_TRUE(cache.launch(key));
        EXPECT_EQ(cache.replayCount(key), 2); });
}

TEST_F(PrefillGraphCacheGPUTest, InvalidateAll_ResetsReadyEntries)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    cache.markWarmedUp(key);

    gpu_ctx_->submitAndWait([&]
                            {
        void *stream = gpu_ctx_->defaultStream();
        ASSERT_TRUE(cache.beginCapture(key, gpu_ctx_, stream));
        ASSERT_TRUE(cache.endCaptureAndInstantiate(key));
        EXPECT_TRUE(cache.hasGraph(key)); });

    cache.invalidateAll();
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Cold);
    EXPECT_FALSE(cache.hasGraph(key));
    EXPECT_EQ(cache.nodeCount(key), 0u);
}

TEST_F(PrefillGraphCacheGPUTest, ReplayCount_IncrementedOnLaunch)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    cache.markWarmedUp(key);

    gpu_ctx_->submitAndWait([&]
                            {
        void *stream = gpu_ctx_->defaultStream();
        ASSERT_TRUE(cache.beginCapture(key, gpu_ctx_, stream));
        ASSERT_TRUE(cache.endCaptureAndInstantiate(key));

        ASSERT_TRUE(cache.launch(key));
        ASSERT_TRUE(cache.launch(key));
        ASSERT_TRUE(cache.launch(key));
        EXPECT_EQ(cache.replayCount(key), 3); });
}

TEST_F(PrefillGraphCacheGPUTest, NodeCount_ZeroForEmptyCapture)
{
    PrefillGraphConfig config;
    PrefillGraphCache cache(config);

    auto key = makeGPUKey(512);
    cache.markWarmedUp(key);

    gpu_ctx_->submitAndWait([&]
                            {
        void *stream = gpu_ctx_->defaultStream();
        ASSERT_TRUE(cache.beginCapture(key, gpu_ctx_, stream));
        ASSERT_TRUE(cache.endCaptureAndInstantiate(key));
        // Empty capture has 0 nodes
        EXPECT_EQ(cache.nodeCount(key), 0u); });
}

#endif // HAVE_ROCM || HAVE_CUDA
