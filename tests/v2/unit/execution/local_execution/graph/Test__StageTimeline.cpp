/**
 * @file Test__StageTimeline.cpp
 * @brief Unit tests for StageTimeline GPU event-based profiler
 *
 * Regression tests for bugs where:
 * 1. Timeline events recorded during GPU graph capture became invalid graph
 *    nodes, causing cudaEventElapsedTime failures when collected later.
 * 2. Stale timeline events persisted across warmup because collectTimeline()
 *    returned early without resetting valid flags.
 * 3. isGraphCaptureActive() was not checked before recording events in
 *    runStages(), allowing events to be recorded on capture streams.
 *
 * These tests validate StageTimeline behavior in isolation using a mock
 * IWorkerGPUContext that tracks event recording calls.
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/StageTimeline.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/compute_stages/IComputeStage.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

using namespace llaminar2;

namespace
{

    // =========================================================================
    // Mock GPU context that tracks event operations
    // =========================================================================

    class TimelineMockGPUContext final : public IWorkerGPUContext
    {
    public:
        TimelineMockGPUContext() = default;

        int deviceOrdinal() const override { return 0; }
        std::string deviceName() const override { return "TimelineMock"; }
        bool isInitialized() const override { return true; }

        void submitAndWait(std::function<void()> work) override { work(); }
        std::future<void> submitAsync(std::function<void()> work) override
        {
            work();
            std::promise<void> done;
            done.set_value();
            return done.get_future();
        }

        void *defaultStream() override { return &fake_stream_; }
        void *createStream() override { return &fake_stream_; }
        void destroyStream(void *) override {}

        void *createEvent() override
        {
            events_created_++;
            // Return distinct fake pointers by using a counter
            return reinterpret_cast<void *>(static_cast<uintptr_t>(0xEE000000 + events_created_));
        }
        void destroyEvent(void *) override {}

        void recordEvent(void *event, void *stream) override
        {
            events_recorded_++;
            last_recorded_event_ = event;
        }

        void waitEvent(void *, void *) override {}

        void synchronizeEvent(void *event) override
        {
            events_synchronized_++;
            if (fail_synchronize_)
            {
                // Simulate the "invalid argument" error behavior
                synchronize_failures_++;
            }
        }

        float eventElapsedTime(void *start, void *stop) override
        {
            elapsed_time_queries_++;
            if (fail_elapsed_time_)
                return -1.0f; // Simulate failure
            return simulated_elapsed_ms_;
        }

        void *blasHandle() override { return nullptr; }
        void *blasLtHandle() override { return nullptr; }

        void setCollectiveComm(void *) override {}
        void *collectiveComm() const override { return nullptr; }

        void synchronize() override {}
        void synchronizeStream(void *) override {}
        void insertStreamDependency(void *, void *) override {}

        std::unique_ptr<IGPUGraphCapture> createGraphCapture() override { return nullptr; }
        std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *) override { return nullptr; }

        // --- Test inspection ---
        int events_created_ = 0;
        int events_recorded_ = 0;
        int events_synchronized_ = 0;
        int synchronize_failures_ = 0;
        int elapsed_time_queries_ = 0;
        void *last_recorded_event_ = nullptr;
        bool fail_synchronize_ = false;
        bool fail_elapsed_time_ = false;
        float simulated_elapsed_ms_ = 1.5f;

    private:
        int fake_stream_ = 0;
    };

    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
                old_value_ = std::string(old);

            if (value)
                setenv(name, value, 1);
            else
                unsetenv(name);
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            if (old_value_.has_value())
                setenv(name_.c_str(), old_value_->c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        std::optional<std::string> old_value_;
    };

} // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class Test__StageTimeline : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gpu_ctx_ = std::make_unique<TimelineMockGPUContext>();
    }

    std::unique_ptr<TimelineMockGPUContext> gpu_ctx_;
};

TEST_F(Test__StageTimeline, DebugEnv_GpuStageTimingDefaultsOff)
{
    ScopedEnv timing("LLAMINAR_GPU_STAGE_TIMING", nullptr);
    ScopedEnv detail("LLAMINAR_GPU_STAGE_TIMING_DETAIL", nullptr);

    EXPECT_FALSE(debugEnv().gpu_stage_timing);
    EXPECT_FALSE(debugEnv().gpu_stage_timing_detail);
}

TEST_F(Test__StageTimeline, DebugEnv_GpuStageTimingEnvEnablesTiming)
{
    ScopedEnv detail("LLAMINAR_GPU_STAGE_TIMING_DETAIL", nullptr);
    ScopedEnv timing("LLAMINAR_GPU_STAGE_TIMING", "1");

    EXPECT_TRUE(debugEnv().gpu_stage_timing);
    EXPECT_FALSE(debugEnv().gpu_stage_timing_detail);
}

TEST_F(Test__StageTimeline, DebugEnv_GpuStageTimingDetailEnablesTiming)
{
    ScopedEnv timing("LLAMINAR_GPU_STAGE_TIMING", nullptr);
    ScopedEnv detail("LLAMINAR_GPU_STAGE_TIMING_DETAIL", "1");

    EXPECT_TRUE(debugEnv().gpu_stage_timing);
    EXPECT_TRUE(debugEnv().gpu_stage_timing_detail);
}

TEST_F(Test__StageTimeline, DebugEnv_GpuStageTimingZeroDisablesTiming)
{
    ScopedEnv detail("LLAMINAR_GPU_STAGE_TIMING_DETAIL", nullptr);
    ScopedEnv timing("LLAMINAR_GPU_STAGE_TIMING", "0");

    EXPECT_FALSE(debugEnv().gpu_stage_timing);
    EXPECT_FALSE(debugEnv().gpu_stage_timing_detail);
}

// =============================================================================
// Basic functionality
// =============================================================================

TEST_F(Test__StageTimeline, Initialize_CreatesEventPairs)
{
    StageTimeline timeline;
    timeline.initialize(gpu_ctx_.get(), 5);

    EXPECT_TRUE(timeline.isInitialized());
    // 5 stages × 2 events (start + stop) = 10 events
    EXPECT_EQ(gpu_ctx_->events_created_, 10);
}

TEST_F(Test__StageTimeline, RecordAndCollect_BasicFlow)
{
    StageTimeline timeline;
    const size_t num_stages = 3;
    timeline.initialize(gpu_ctx_.get(), num_stages);

    // Set stage info
    timeline.setStageInfo(0, "stage_a", ComputeStageType::GEMM);
    timeline.setStageInfo(1, "stage_b", ComputeStageType::ATTENTION);
    timeline.setStageInfo(2, "stage_c", ComputeStageType::RMS_NORM);

    void *stream = gpu_ctx_->defaultStream();

    // Record start/stop for all stages
    for (size_t i = 0; i < num_stages; ++i)
    {
        timeline.recordStart(i, gpu_ctx_.get(), stream);
        timeline.recordStop(i, gpu_ctx_.get(), stream);
    }

    // 3 starts + 3 stops = 6 event records
    EXPECT_EQ(gpu_ctx_->events_recorded_, 6);

    // Collect timings
    timeline.collect(gpu_ctx_.get());

    // Should sync once (on last stop event) and query all 3 pairs
    EXPECT_EQ(gpu_ctx_->events_synchronized_, 1);
    EXPECT_EQ(gpu_ctx_->elapsed_time_queries_, 3);

    // Check total GPU time
    EXPECT_FLOAT_EQ(timeline.totalGpuMs(), 1.5f * 3);
}

TEST_F(Test__StageTimeline, RecordPerfStatsCarriesContextTags)
{
    ScopedEnv json("LLAMINAR_PERF_STATS_JSON", "/tmp/stage_timeline_context_tags.json");
    PerfStatsCollector::reset();

    StageTimeline timeline;
    timeline.initialize(gpu_ctx_.get(), 2);
    timeline.setStageInfo(0, "gemm_stage", ComputeStageType::GEMM);
    timeline.setStageInfo(1, "lm_head_stage", ComputeStageType::LM_HEAD);

    void *stream = gpu_ctx_->defaultStream();
    for (size_t i = 0; i < 2; ++i)
    {
        timeline.recordStart(i, gpu_ctx_.get(), stream);
        timeline.recordStop(i, gpu_ctx_.get(), stream);
    }
    timeline.collect(gpu_ctx_.get());

    timeline.recordPerfStats(
        "decode",
        "CUDA:0",
        "stage_gpu",
        {{"context", "main_verifier"}});

    const auto records = PerfStatsCollector::snapshot({"stage_gpu"});
    auto has_record = [&](const std::string &name, const PerfStatsCollector::Tags &tags) {
        return std::any_of(records.begin(), records.end(), [&](const PerfStatRecord &record) {
            return record.domain == "stage_gpu" &&
                   record.name == name &&
                   record.phase == "decode" &&
                   record.device == "CUDA:0" &&
                   record.tags == tags;
        });
    };

    EXPECT_TRUE(has_record("total", {{"context", "main_verifier"}}));
    EXPECT_TRUE(has_record("type.GEMM", {{"context", "main_verifier"}, {"stage_count", "1"}}));
    EXPECT_TRUE(has_record("gemm_stage", {{"context", "main_verifier"}, {"index", "0"}, {"type", "GEMM"}}));

    PerfStatsCollector::reset();
}

TEST_F(Test__StageTimeline, Collect_NoValidEvents_ReturnsEarly)
{
    StageTimeline timeline;
    timeline.initialize(gpu_ctx_.get(), 5);

    // Don't record any events — all valid flags are false
    EXPECT_FALSE(timeline.hasValidRecords());

    timeline.collect(gpu_ctx_.get());

    // No sync, no queries when no valid events exist
    EXPECT_EQ(gpu_ctx_->events_synchronized_, 0);
    EXPECT_EQ(gpu_ctx_->elapsed_time_queries_, 0);
}

// =============================================================================
// Regression: resetTimings() clears valid flags
// =============================================================================

TEST_F(Test__StageTimeline, ResetTimings_ClearsValidFlags)
{
    StageTimeline timeline;
    timeline.initialize(gpu_ctx_.get(), 3);

    timeline.setStageInfo(0, "s0", ComputeStageType::GEMM);
    timeline.setStageInfo(1, "s1", ComputeStageType::GEMM);
    timeline.setStageInfo(2, "s2", ComputeStageType::GEMM);

    void *stream = gpu_ctx_->defaultStream();

    // Record events → valid=true
    for (size_t i = 0; i < 3; ++i)
    {
        timeline.recordStart(i, gpu_ctx_.get(), stream);
        timeline.recordStop(i, gpu_ctx_.get(), stream);
    }
    EXPECT_TRUE(timeline.hasValidRecords());

    // Reset
    timeline.resetTimings();
    EXPECT_FALSE(timeline.hasValidRecords());

    // Now collect should find no valid events and return early
    timeline.collect(gpu_ctx_.get());

    EXPECT_EQ(gpu_ctx_->events_synchronized_, 0);
    EXPECT_EQ(gpu_ctx_->elapsed_time_queries_, 0);
    EXPECT_FLOAT_EQ(timeline.totalGpuMs(), 0.0f);
}

// =============================================================================
// Regression: Stale events must not persist after resetTimings()
//
// This reproduces the bug where warmup recorded events (valid=true),
// collectTimeline() returned early due to suppress_timeline_ without
// calling resetTimings(), and subsequent collect() tried to use the
// stale events that became invalid after graph capture.
// =============================================================================

TEST_F(Test__StageTimeline, StaleEvents_NotCollectedAfterReset)
{
    StageTimeline timeline;
    timeline.initialize(gpu_ctx_.get(), 4);

    void *stream = gpu_ctx_->defaultStream();

    // Phase 1: Simulate warmup — record events
    for (size_t i = 0; i < 4; ++i)
    {
        timeline.setStageInfo(i, "stage_" + std::to_string(i), ComputeStageType::GEMM);
        timeline.recordStart(i, gpu_ctx_.get(), stream);
        timeline.recordStop(i, gpu_ctx_.get(), stream);
    }
    EXPECT_TRUE(timeline.hasValidRecords());

    // Phase 2: Simulate "suppressed collectTimeline" calling resetTimings()
    // This is the fix — previously this was skipped when suppressed.
    timeline.resetTimings();
    EXPECT_FALSE(timeline.hasValidRecords());

    // Phase 3: Simulate graph capture phase — NO new events recorded
    // (isGraphCaptureActive() prevents recording)

    // Phase 4: Simulate post-capture collectTimeline()
    // Should NOT attempt to sync/query stale events
    gpu_ctx_->events_synchronized_ = 0;
    gpu_ctx_->elapsed_time_queries_ = 0;
    gpu_ctx_->fail_synchronize_ = true; // Would fail if called

    timeline.collect(gpu_ctx_.get());

    EXPECT_EQ(gpu_ctx_->events_synchronized_, 0)
        << "Stale events from warmup should not be synchronized after reset";
    EXPECT_EQ(gpu_ctx_->elapsed_time_queries_, 0)
        << "Stale events from warmup should not be queried after reset";
}

// =============================================================================
// Regression: Without resetTimings(), stale events WOULD be collected
//
// This is the "before fix" scenario — proves the bug existed.
// =============================================================================

TEST_F(Test__StageTimeline, StaleEvents_WouldBeCollectedWithoutReset)
{
    StageTimeline timeline;
    timeline.initialize(gpu_ctx_.get(), 4);

    void *stream = gpu_ctx_->defaultStream();

    // Record events (simulates warmup Cold phase)
    for (size_t i = 0; i < 4; ++i)
    {
        timeline.setStageInfo(i, "stage_" + std::to_string(i), ComputeStageType::GEMM);
        timeline.recordStart(i, gpu_ctx_.get(), stream);
        timeline.recordStop(i, gpu_ctx_.get(), stream);
    }

    // Do NOT reset (simulates the old buggy path where suppress skipped resetTimings)

    // Reset counters to measure only the collect phase
    gpu_ctx_->events_synchronized_ = 0;
    gpu_ctx_->elapsed_time_queries_ = 0;

    // This would trigger the bug: collecting stale events that are invalid
    // after graph capture
    timeline.collect(gpu_ctx_.get());

    // Without reset, it DOES try to sync and query all 4 events
    EXPECT_EQ(gpu_ctx_->events_synchronized_, 1);
    EXPECT_EQ(gpu_ctx_->elapsed_time_queries_, 4);
}

// =============================================================================
// Regression: GraphCaptureGuard prevents timeline recording in runStages
//
// This tests that the isGraphCaptureActive() check works correctly.
// The actual guard is checked in DeviceGraphExecutor::runStages() which
// sets timeline_active = false when isGraphCaptureActive() is true.
// =============================================================================

TEST_F(Test__StageTimeline, GraphCaptureGuard_PreventsRecording)
{
    StageTimeline timeline;
    timeline.initialize(gpu_ctx_.get(), 3);

    timeline.setStageInfo(0, "s0", ComputeStageType::GEMM);
    timeline.setStageInfo(1, "s1", ComputeStageType::GEMM);
    timeline.setStageInfo(2, "s2", ComputeStageType::GEMM);

    void *stream = gpu_ctx_->defaultStream();

    // Simulate what runStages does: check isGraphCaptureActive() before recording
    {
        GraphCaptureGuard guard;

        // With guard active, runStages would set timeline_active=false
        const bool timeline_active = !isGraphCaptureActive(); // simulates the check

        EXPECT_FALSE(timeline_active)
            << "timeline_active should be false during graph capture";

        if (timeline_active)
        {
            for (size_t i = 0; i < 3; ++i)
            {
                timeline.recordStart(i, gpu_ctx_.get(), stream);
                timeline.recordStop(i, gpu_ctx_.get(), stream);
            }
        }
    }

    // No events should have been recorded
    // The initial 6 events were created (createEvent), but no recordEvent calls
    EXPECT_EQ(gpu_ctx_->events_recorded_, 0)
        << "No events should be recorded when GraphCaptureGuard is active";

    // Collect should find nothing valid
    timeline.collect(gpu_ctx_.get());
    EXPECT_EQ(gpu_ctx_->events_synchronized_, 0);
}

TEST_F(Test__StageTimeline, GraphCaptureGuard_NestingRestoresPreviousState)
{
    EXPECT_FALSE(isGraphCaptureActive());

    {
        GraphCaptureGuard outer;
        EXPECT_TRUE(isGraphCaptureActive());

        {
            GraphCaptureGuard inner;
            EXPECT_TRUE(isGraphCaptureActive());
        }

        // Inner guard destroyed, outer still active
        EXPECT_TRUE(isGraphCaptureActive());
    }

    // Both guards destroyed
    EXPECT_FALSE(isGraphCaptureActive());
}

// =============================================================================
// Regression: Full warmup→capture→benchmark lifecycle
//
// Simulates the exact sequence that caused the CUDA benchmark failure:
// 1. Cold phase: events recorded on real stream (valid=true)
// 2. Suppressed collectTimeline: must resetTimings
// 3. Capture phase: GraphCaptureGuard active, no events recorded
// 4. Benchmark iteration: collectTimeline called with no stale events
// =============================================================================

TEST_F(Test__StageTimeline, FullLifecycle_WarmupCaptureAndBenchmark)
{
    StageTimeline timeline;
    const size_t num_stages = 256; // Realistic: 14 layers × ~18 stages
    timeline.initialize(gpu_ctx_.get(), num_stages);

    void *stream = gpu_ctx_->defaultStream();

    // --- Phase 1: Cold phase (warmup) --- Events recorded normally
    for (size_t i = 0; i < num_stages; ++i)
    {
        timeline.setStageInfo(i, "stage_" + std::to_string(i), ComputeStageType::GEMM);
        timeline.recordStart(i, gpu_ctx_.get(), stream);
        timeline.recordStop(i, gpu_ctx_.get(), stream);
    }
    EXPECT_EQ(gpu_ctx_->events_recorded_, num_stages * 2);

    // --- Phase 2: Suppressed collectTimeline (warmup) ---
    // The fix: resetTimings() even when suppressed
    timeline.resetTimings();

    // --- Phase 3: Capture phase --- GraphCaptureGuard active
    {
        GraphCaptureGuard guard;
        const bool timeline_active = !isGraphCaptureActive();
        ASSERT_FALSE(timeline_active);
        // No recordStart/recordStop calls happen
    }

    // --- Phase 4: Benchmark iteration --- collectTimeline called
    gpu_ctx_->events_synchronized_ = 0;
    gpu_ctx_->elapsed_time_queries_ = 0;
    gpu_ctx_->fail_synchronize_ = true;  // CUDA would return "invalid argument" on stale events
    gpu_ctx_->fail_elapsed_time_ = true; // Same

    timeline.collect(gpu_ctx_.get());

    // No events should be synchronized or queried — all were reset in Phase 2
    EXPECT_EQ(gpu_ctx_->events_synchronized_, 0)
        << "After proper resetTimings(), no stale events should be collected";
    EXPECT_EQ(gpu_ctx_->elapsed_time_queries_, 0)
        << "After proper resetTimings(), no stale events should be queried";
}

// =============================================================================
// Edge case: ensureCapacity doesn't resurrect stale events
// =============================================================================

TEST_F(Test__StageTimeline, EnsureCapacity_GrowDoesNotSetValidFlags)
{
    StageTimeline timeline;
    timeline.initialize(gpu_ctx_.get(), 2);

    void *stream = gpu_ctx_->defaultStream();

    // Record 2 events
    timeline.setStageInfo(0, "s0", ComputeStageType::GEMM);
    timeline.setStageInfo(1, "s1", ComputeStageType::GEMM);
    timeline.recordStart(0, gpu_ctx_.get(), stream);
    timeline.recordStop(0, gpu_ctx_.get(), stream);
    timeline.recordStart(1, gpu_ctx_.get(), stream);
    timeline.recordStop(1, gpu_ctx_.get(), stream);

    // Reset
    timeline.resetTimings();

    // Grow to 5 stages
    timeline.ensureCapacity(gpu_ctx_.get(), 5);

    // New slots should NOT be valid
    gpu_ctx_->events_synchronized_ = 0;
    gpu_ctx_->elapsed_time_queries_ = 0;

    timeline.collect(gpu_ctx_.get());

    EXPECT_EQ(gpu_ctx_->events_synchronized_, 0)
        << "Newly allocated event slots should not be marked valid";
    EXPECT_EQ(gpu_ctx_->elapsed_time_queries_, 0);
}
