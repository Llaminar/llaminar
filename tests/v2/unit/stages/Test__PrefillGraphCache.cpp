/**
 * @file Test__PrefillGraphCache.cpp
 * @brief Unit tests for PrefillGraphCache state machine, preflight, and invalidation
 */

#include <gtest/gtest.h>
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
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
