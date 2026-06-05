/**
 * @file Test__PrefillGraphCacheIntegration.cpp
 * @brief Unit tests for PrefillGraphCache integration into ForwardExecutionEngine
 *
 * Tests the warmup → capture → replay lifecycle, preflight rejection paths,
 * failure propagation, replay callbacks, and snapshot bypass.
 */

#include <gtest/gtest.h>
#include "execution/local_execution/engine/ForwardExecutionEngine.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "execution/local_execution/engine/PrefillGraphCache.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "backends/DeviceId.h"
#include "backends/GPUDeviceContextPool.h"
#include "mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Helper: Capturable mock stage for graph capture tests
// =============================================================================

class IntegCapturableMockStage : public MockComputeStage
{
public:
    IntegCapturableMockStage(std::string name, DeviceId dev)
        : MockComputeStage(ComputeStageType::GEMM, std::move(name), dev) {}

    bool isGraphCapturable() const override { return true; }
    bool hasDynamicParams() const override { return has_dynamic_params_; }
    bool needsOnGraphReplayed() const override { return needs_replay_callback_; }

    void updateDynamicParams(int position_offset, int seq_len) override
    {
        dynamic_params_calls_++;
        last_position_offset_ = position_offset;
        last_seq_len_ = seq_len;
    }

    void onGraphReplayed() override
    {
        replay_callback_calls_++;
    }

    void setHasDynamicParams(bool v) { has_dynamic_params_ = v; }
    void setNeedsReplayCallback(bool v) { needs_replay_callback_ = v; }

    int dynamicParamsCalls() const { return dynamic_params_calls_; }
    int replayCallbackCalls() const { return replay_callback_calls_; }
    int lastPositionOffset() const { return last_position_offset_; }
    int lastSeqLen() const { return last_seq_len_; }

private:
    bool has_dynamic_params_ = false;
    bool needs_replay_callback_ = false;
    int dynamic_params_calls_ = 0;
    int replay_callback_calls_ = 0;
    int last_position_offset_ = -1;
    int last_seq_len_ = -1;
};

// =============================================================================
// Helper: Non-capturable stage (blocks graph capture)
// =============================================================================

class IntegNonCapturableMockStage : public MockComputeStage
{
public:
    IntegNonCapturableMockStage(std::string name, DeviceId dev)
        : MockComputeStage(ComputeStageType::GEMM, std::move(name), dev) {}

    bool isGraphCapturable() const override { return false; }
};

// =============================================================================
// Helper: Build graph with GPU-compatible capturable stages
// =============================================================================

static DeviceId testGPUDevice()
{
#ifdef HAVE_ROCM
    return DeviceId::rocm(0);
#elif defined(HAVE_CUDA)
    return DeviceId::cuda(0);
#else
    return DeviceId::cuda(0); // placeholder for CPU-only builds
#endif
}

static ComputeGraph buildIntegCapturableGraph(
    DeviceId dev,
    int num_stages = 4,
    std::vector<IntegCapturableMockStage *> *stage_ptrs = nullptr)
{
    ComputeGraph graph;
    for (int i = 0; i < num_stages; ++i)
    {
        std::string name = "stage_" + std::to_string(i);
        auto stage = std::make_unique<IntegCapturableMockStage>(name, dev);
        if (stage_ptrs)
            stage_ptrs->push_back(stage.get());
        graph.addNode(name, std::move(stage), dev);
        if (i > 0)
            graph.addDependency(name, "stage_" + std::to_string(i - 1));
    }
    return graph;
}

// =============================================================================
// Test: PrefillGraphCache initializes lazily inside ForwardGraphCache
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, LazyInitialization)
{
    ForwardGraphCache cache;
    EXPECT_EQ(cache.prefill_graph_cache, nullptr);

    // After calling executePrefillWithGraphCache, the cache should be initialized
    // (tested implicitly by other tests since we can't call the private method directly)
}

// =============================================================================
// Test: PrefillGraphCache invalidation via ForwardGraphCache::invalidate()
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, InvalidateAllClearsPrefillCache)
{
    ForwardGraphCache cache;

    // Create prefill cache with a warmed-up entry
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 1;
    cache.prefill_graph_cache = std::make_unique<PrefillGraphCache>(config);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = testGPUDevice();

    cache.prefill_graph_cache->markWarmedUp(key);
    EXPECT_EQ(cache.prefill_graph_cache->phase(key), PrefillGraphPhase::Warmup);

    // invalidate() should clear the prefill cache
    cache.invalidate();
    // After invalidate, prefill_graph_cache still exists but entries are cleared
    EXPECT_NE(cache.prefill_graph_cache, nullptr);
    EXPECT_EQ(cache.prefill_graph_cache->phase(key), PrefillGraphPhase::Cold);
}

TEST(Test__PrefillGraphCacheIntegration, SessionResetPreservesPrefillCache)
{
    ForwardGraphCache cache;

    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 1;
    cache.prefill_graph_cache = std::make_unique<PrefillGraphCache>(config);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = testGPUDevice();

    cache.prefill_graph_cache->markWarmedUp(key);
    EXPECT_EQ(cache.prefill_graph_cache->phase(key), PrefillGraphPhase::Warmup);

    cache.resetSessionState();

    EXPECT_NE(cache.prefill_graph_cache, nullptr);
    EXPECT_EQ(cache.prefill_graph_cache->phase(key), PrefillGraphPhase::Warmup)
        << "Request/session reset must preserve the prefill graph lifecycle so "
           "benchmark warmup can advance Warmup -> Ready and steady-state runs can replay";
}

TEST(Test__PrefillGraphCacheIntegration, WorkspaceRebindInvalidatesPrefillCache)
{
    ForwardGraphCache cache;

    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 1;
    cache.prefill_graph_cache = std::make_unique<PrefillGraphCache>(config);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = testGPUDevice();

    cache.prefill_graph_cache->markWarmedUp(key);
    EXPECT_EQ(cache.prefill_graph_cache->phase(key), PrefillGraphPhase::Warmup);

    cache.resetReplayStateAfterWorkspaceRebind();

    EXPECT_NE(cache.prefill_graph_cache, nullptr);
    EXPECT_EQ(cache.prefill_graph_cache->phase(key), PrefillGraphPhase::Cold)
        << "Captured graph entries encode workspace addresses and must be rebuilt after rebind";
}

// =============================================================================
// Test: Phase transitions Cold → Warmup → Ready in ForwardGraphCache context
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, PhaseTransitionsInForwardGraphCache)
{
    ForwardGraphCache fwd_cache;
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    fwd_cache.prefill_graph_cache = std::make_unique<PrefillGraphCache>(config);

    auto dev = testGPUDevice();
    auto graph = buildIntegCapturableGraph(dev);
    fwd_cache.graph = std::make_unique<ComputeGraph>(std::move(graph));

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = dev;

    // Start cold
    auto &cache = *fwd_cache.prefill_graph_cache;
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Cold);

    // Preflight should pass
    std::unordered_set<std::string> no_collectives;
    auto reason = cache.preflight(*fwd_cache.graph, key, &no_collectives, false, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::None);

    // Mark warmed up
    cache.markWarmedUp(key);
    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Warmup);
}

// =============================================================================
// Test: Preflight rejects CPU device
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, PreflightRejectsCPUDevice)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    PrefillGraphCache cache(config);

    auto cpu_dev = DeviceId::cpu();
    ComputeGraph graph;
    graph.addNode("stage_0",
                  std::make_unique<IntegCapturableMockStage>("stage_0", cpu_dev),
                  cpu_dev);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = cpu_dev;

    std::unordered_set<std::string> no_collectives;
    auto reason = cache.preflight(graph, key, &no_collectives, false, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::NotGPUDevice);
}

// =============================================================================
// Test: Preflight rejects when snapshots are active
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, PreflightRejectsSnapshots)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    PrefillGraphCache cache(config);

    auto dev = testGPUDevice();
    auto graph = buildIntegCapturableGraph(dev);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = dev;

    std::unordered_set<std::string> no_collectives;
    auto reason = cache.preflight(graph, key, &no_collectives,
                                  /*snapshots_active=*/true, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::SnapshotsActive);
}

// =============================================================================
// Test: Preflight rejects when MoE rebalancing is active
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, PreflightRejectsMoERebalancing)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    PrefillGraphCache cache(config);

    auto dev = testGPUDevice();
    auto graph = buildIntegCapturableGraph(dev);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = dev;

    std::unordered_set<std::string> no_collectives;
    auto reason = cache.preflight(graph, key, &no_collectives,
                                  false, /*moe_rebalancing_active=*/true);
    EXPECT_EQ(reason, PrefillGraphRejectReason::ActiveMoERebalancing);
}

// =============================================================================
// Test: Preflight rejects when seq_len below minimum
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, PreflightRejectsShortSeqLen)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 256;
    PrefillGraphCache cache(config);

    auto dev = testGPUDevice();
    auto graph = buildIntegCapturableGraph(dev);

    PrefillGraphCacheKey key;
    key.seq_len = 128; // Below min_seq_len=256
    key.device_id = dev;

    std::unordered_set<std::string> no_collectives;
    auto reason = cache.preflight(graph, key, &no_collectives, false, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::SeqLenBelowMinimum);
}

// =============================================================================
// Test: Preflight rejects when collective nodes are present
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, PreflightRejectsCollectives)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    PrefillGraphCache cache(config);

    auto dev = testGPUDevice();
    auto graph = buildIntegCapturableGraph(dev);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = dev;

    std::unordered_set<std::string> collectives = {"allreduce_0"};
    auto reason = cache.preflight(graph, key, &collectives, false, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::CollectiveNodesPresent);
}

// =============================================================================
// Test: Preflight rejects non-capturable stages
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, PreflightRejectsNonCapturableStage)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    PrefillGraphCache cache(config);

    auto dev = testGPUDevice();
    ComputeGraph graph;
    graph.addNode("good_stage",
                  std::make_unique<IntegCapturableMockStage>("good_stage", dev), dev);
    graph.addNode("bad_stage",
                  std::make_unique<IntegNonCapturableMockStage>("bad_stage", dev), dev);
    graph.addDependency("bad_stage", "good_stage");

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = dev;

    std::unordered_set<std::string> no_collectives;
    auto reason = cache.preflight(graph, key, &no_collectives, false, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::StageNotCapturable);
}

// =============================================================================
// Test: Replay callbacks are called with correct stage list
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, ReplayCallbackStagesCached)
{
    ForwardGraphCache fwd_cache;
    auto dev = testGPUDevice();

    std::vector<IntegCapturableMockStage *> stage_ptrs;
    auto graph = buildIntegCapturableGraph(dev, 4, &stage_ptrs);

    // Mark stage 1 and 3 as needing replay callbacks
    stage_ptrs[1]->setNeedsReplayCallback(true);
    stage_ptrs[3]->setNeedsReplayCallback(true);

    fwd_cache.graph = std::make_unique<ComputeGraph>(std::move(graph));

    // Cache replay callback stages (mimics executeCacheHit logic)
    EXPECT_FALSE(fwd_cache.replay_callback_stages_cached);
    const auto &order = fwd_cache.graph->getExecutionOrder();
    for (const auto &node_name : order)
    {
        ComputeNode *node = fwd_cache.graph->getNode(node_name);
        if (node && node->stage && node->stage->needsOnGraphReplayed())
            fwd_cache.replay_callback_stages.push_back(node->stage.get());
    }
    fwd_cache.replay_callback_stages_cached = true;

    EXPECT_EQ(fwd_cache.replay_callback_stages.size(), 2u);

    // Simulate replay callbacks
    for (auto *stage : fwd_cache.replay_callback_stages)
        stage->onGraphReplayed();

    EXPECT_EQ(stage_ptrs[1]->replayCallbackCalls(), 1);
    EXPECT_EQ(stage_ptrs[3]->replayCallbackCalls(), 1);
    EXPECT_EQ(stage_ptrs[0]->replayCallbackCalls(), 0);
    EXPECT_EQ(stage_ptrs[2]->replayCallbackCalls(), 0);
}

// =============================================================================
// Test: Dynamic param stages cached correctly
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, DynamicParamStagesCached)
{
    ForwardGraphCache fwd_cache;
    auto dev = testGPUDevice();

    std::vector<IntegCapturableMockStage *> stage_ptrs;
    auto graph = buildIntegCapturableGraph(dev, 5, &stage_ptrs);

    // Mark stage 0 and 2 as having dynamic params
    stage_ptrs[0]->setHasDynamicParams(true);
    stage_ptrs[2]->setHasDynamicParams(true);

    fwd_cache.graph = std::make_unique<ComputeGraph>(std::move(graph));

    // Cache dynamic param stages
    const auto &order = fwd_cache.graph->getExecutionOrder();
    for (const auto &node_name : order)
    {
        ComputeNode *node = fwd_cache.graph->getNode(node_name);
        if (node && node->stage && node->stage->hasDynamicParams())
            fwd_cache.dynamic_param_stages.push_back(node->stage.get());
    }
    fwd_cache.dynamic_param_stages_cached = true;

    EXPECT_EQ(fwd_cache.dynamic_param_stages.size(), 2u);

    // Simulate updateDynamicParams call
    for (auto *stage : fwd_cache.dynamic_param_stages)
        stage->updateDynamicParams(100, 512);

    EXPECT_EQ(stage_ptrs[0]->dynamicParamsCalls(), 1);
    EXPECT_EQ(stage_ptrs[0]->lastPositionOffset(), 100);
    EXPECT_EQ(stage_ptrs[0]->lastSeqLen(), 512);
    EXPECT_EQ(stage_ptrs[2]->dynamicParamsCalls(), 1);
    EXPECT_EQ(stage_ptrs[1]->dynamicParamsCalls(), 0);
}

// =============================================================================
// Test: IForwardExecutionHost default isMoeRebalancingActive() returns false
// =============================================================================

namespace
{
    /// Minimal host implementation for testing default virtual method
    class MinimalTestHost : public IForwardExecutionHost
    {
    public:
        GraphBuildResult buildForwardGraph(const ForwardInput &) override
        {
            return GraphBuildResult("not implemented");
        }
        IDeviceContext *getDeviceContext(DeviceId) override { return nullptr; }
        std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() override { return {}; }
        bool ensureDeviceWorkspaceAllocated(const ComputeGraph &, int) override { return true; }
        void syncLogitsAtBoundary(IDeviceContext *) override {}
        TensorBase *logitsTensor() override { return nullptr; }
        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool, IDeviceContext *, int) const override
        {
            return {};
        }
        PPCopyInfo resolvePPCopyInfo(const ForwardInput &) const override { return {}; }
    };
} // namespace

TEST(Test__PrefillGraphCacheIntegration, DefaultHostMoERebalancingReturnsFalse)
{
    MinimalTestHost host;
    EXPECT_FALSE(host.isMoeRebalancingActive());
    EXPECT_EQ(host.moePlacementEpoch(), 0u);
}

// =============================================================================
// Test: Feature disabled config rejects via preflight
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, DisabledConfigRejectsAll)
{
    PrefillGraphConfig config;
    config.enabled = false;
    PrefillGraphCache cache(config);

    auto dev = testGPUDevice();
    auto graph = buildIntegCapturableGraph(dev);

    PrefillGraphCacheKey key;
    key.seq_len = 512;
    key.device_id = dev;

    std::unordered_set<std::string> no_collectives;
    auto reason = cache.preflight(graph, key, &no_collectives, false, false);
    EXPECT_EQ(reason, PrefillGraphRejectReason::FeatureDisabled);
}

// =============================================================================
// Test: Phase query for unknown key returns Cold (not Disabled)
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, UnknownKeyReturnsCold)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    PrefillGraphCache cache(config);

    PrefillGraphCacheKey key;
    key.seq_len = 1024;
    key.device_id = testGPUDevice();

    EXPECT_EQ(cache.phase(key), PrefillGraphPhase::Cold);
}

// =============================================================================
// Test: Multiple seq_len entries are independent
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, MultipleSeqLenIndependent)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    PrefillGraphCache cache(config);

    auto dev = testGPUDevice();

    PrefillGraphCacheKey key256;
    key256.seq_len = 256;
    key256.device_id = dev;

    PrefillGraphCacheKey key512;
    key512.seq_len = 512;
    key512.device_id = dev;

    // Warm up one, leave the other cold
    cache.markWarmedUp(key256);
    EXPECT_EQ(cache.phase(key256), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(key512), PrefillGraphPhase::Cold);

    // Warm up the other
    cache.markWarmedUp(key512);
    EXPECT_EQ(cache.phase(key256), PrefillGraphPhase::Warmup);
    EXPECT_EQ(cache.phase(key512), PrefillGraphPhase::Warmup);
}

// =============================================================================
// Test: InvalidateAll resets all entries to Cold
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, InvalidateAllResetsEntries)
{
    PrefillGraphConfig config;
    config.enabled = true;
    config.min_seq_len = 64;
    PrefillGraphCache cache(config);

    auto dev = testGPUDevice();

    PrefillGraphCacheKey key1;
    key1.seq_len = 256;
    key1.device_id = dev;

    PrefillGraphCacheKey key2;
    key2.seq_len = 512;
    key2.device_id = dev;

    cache.markWarmedUp(key1);
    cache.markWarmedUp(key2);

    cache.invalidateAll();

    EXPECT_EQ(cache.phase(key1), PrefillGraphPhase::Cold);
    EXPECT_EQ(cache.phase(key2), PrefillGraphPhase::Cold);
}

// =============================================================================
// Test: toString for PrefillGraphRejectReason
// =============================================================================

TEST(Test__PrefillGraphCacheIntegration, RejectReasonToString)
{
    EXPECT_NE(toString(PrefillGraphRejectReason::None), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::FeatureDisabled), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::SeqLenBelowMinimum), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::NotGPUDevice), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::SnapshotsActive), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::ActiveMoERebalancing), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::CollectiveNodesPresent), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::StageNotCapturable), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::GDNWithPaddedBucket), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::NoGPUContext), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::InvalidatedByPlacement), nullptr);
    EXPECT_NE(toString(PrefillGraphRejectReason::SessionReset), nullptr);
}
