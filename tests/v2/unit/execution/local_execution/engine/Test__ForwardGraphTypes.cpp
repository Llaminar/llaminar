/**
 * @file Test__ForwardGraphTypes.cpp
 * @brief Unit tests for ForwardGraphTypes
 *
 * Tests ForwardGraphSignature equality/hashing, GraphBuildResult,
 * GraphCacheConfig defaults, and ForwardGraphCache invalidation.
 */

#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <unordered_map>

#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "utils/DebugEnv.h"
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
                         bool segment_boundary_before = false,
                         bool segment_boundary_after = false,
                         const uint64_t *variant_signature = nullptr)
            : IComputeStage(DeviceId::cpu()),
              capturable_(capturable),
              manual_boundary_(manual_boundary),
              stage_type_(stage_type),
              warmup_dependent_capture_(warmup_dependent_capture),
              segment_boundary_before_(segment_boundary_before),
              segment_boundary_after_(segment_boundary_after),
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
        bool requiresGraphCaptureSegmentBoundaryBefore() const override { return segment_boundary_before_; }
        bool requiresGraphCaptureSegmentBoundaryAfter() const override { return segment_boundary_after_; }
        bool isManualGraphBoundary() const override { return manual_boundary_; }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

    private:
        bool capturable_ = true;
        bool manual_boundary_ = false;
        ComputeStageType stage_type_ = ComputeStageType::COPY;
        bool warmup_dependent_capture_ = false;
        bool segment_boundary_before_ = false;
        bool segment_boundary_after_ = false;
        const uint64_t *variant_signature_ = nullptr;
    };

    class FakeGraphLaunchPrepStage final : public IComputeStage
    {
    public:
        explicit FakeGraphLaunchPrepStage(DeviceId device)
            : IComputeStage(device)
        {
        }

        bool execute(IDeviceContext *) override
        {
            ++execute_calls_;
            executed_after_prepare_ = prepare_calls_ > 0;
            return true;
        }

        ComputeStageType type() const override { return ComputeStageType::COPY; }
        std::string name() const override { return "fake_graph_launch_prep_stage"; }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        bool isGraphCapturable() const override { return true; }
        bool needsGraphLaunchPreparation() const override { return true; }

        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            ++prepare_calls_;
            last_ctx_ = ctx;
            last_stream_ = stream;
            if (stream)
                setGPUStream(stream);
            stream_seen_by_stage_ = gpuStream();
            return stream != nullptr;
        }

        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

        int prepare_calls_ = 0;
        int execute_calls_ = 0;
        bool executed_after_prepare_ = false;
        IDeviceContext *last_ctx_ = nullptr;
        void *last_stream_ = nullptr;
        void *stream_seen_by_stage_ = nullptr;
    };

    /**
     * @brief Capturable stage that deliberately rejects its recorded execution.
     *
     * This test double models a production stage discovering an invalid graph
     * capture contract after the native stream has entered capture mode. Its
     * call counter proves the controller does not attempt eager recovery by
     * executing the same stage a second time.
     */
    class FakeCaptureFailureStage final : public IComputeStage
    {
    public:
        explicit FakeCaptureFailureStage(DeviceId device)
            : IComputeStage(device)
        {
        }

        bool execute(IDeviceContext *) override
        {
            ++execute_calls_;
            return false;
        }

        ComputeStageType type() const override { return ComputeStageType::COPY; }
        std::string name() const override { return "fake_capture_failure_stage"; }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        bool isGraphCapturable() const override { return true; }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

        int execute_calls_ = 0;
    };

    class FakeCapturedStateStage final : public IComputeStage
    {
    public:
        FakeCapturedStateStage(DeviceId device,
                               bool has_capture,
                               bool requires_capture = true,
                               bool restore_ok = true)
            : IComputeStage(device),
              has_capture_(has_capture),
              requires_capture_(requires_capture),
              restore_ok_(restore_ok)
        {
        }

        bool execute(IDeviceContext *) override { return true; }
        ComputeStageType type() const override { return ComputeStageType::COPY; }
        std::string name() const override { return "fake_captured_state_stage"; }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        bool hasVerifierStateCapture() const override { return has_capture_; }
        bool requiresVerifierStateCaptureForPublication() const override { return requires_capture_; }
        bool restoreVerifierStateCaptureRow(int row, void *stream) override
        {
            ++restore_calls_;
            last_row_ = row;
            last_stream_ = stream;
            return restore_ok_;
        }
        bool restoreVerifierStateCaptureRows(
            const int *host_row_indices,
            int request_count,
            void *stream) override
        {
            ++restore_rows_calls_;
            last_rows_.assign(
                host_row_indices,
                host_row_indices + request_count);
            last_stream_ = stream;
            return restore_ok_;
        }
        bool restoreVerifierStateCaptureRequestTerminalRows(
            const int *device_request_seq_lens,
            int request_count,
            int request_row_width,
            void *stream) override
        {
            ++restore_device_rows_calls_;
            last_device_lengths_ = device_request_seq_lens;
            last_request_count_ = request_count;
            last_request_row_width_ = request_row_width;
            last_stream_ = stream;
            return restore_ok_;
        }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

        int restore_calls_ = 0;
        int restore_rows_calls_ = 0;
        int restore_device_rows_calls_ = 0;
        int last_row_ = -1;
        int last_request_count_ = 0;
        int last_request_row_width_ = 0;
        const int *last_device_lengths_ = nullptr;
        std::vector<int> last_rows_;
        void *last_stream_ = nullptr;

    private:
        bool has_capture_ = false;
        bool requires_capture_ = true;
        bool restore_ok_ = true;
    };

    class FakeReplayGraphCapture final : public IGPUGraphCapture
    {
    public:
        explicit FakeReplayGraphCapture(void *stream = nullptr) : stream_(stream) {}

        bool beginCapture() override
        {
            ++begin_capture_calls_;
            if (capturing_)
                return false;
            capturing_ = true;
            return true;
        }
        bool endCapture() override
        {
            ++end_capture_calls_;
            if (!capturing_)
                return false;
            capturing_ = false;
            return true;
        }
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
        [[nodiscard]] void *executionStream() const noexcept override { return stream_; }
        GraphUpdateResult tryUpdate() override
        {
            ++try_update_calls_;
            return update_result_;
        }
        [[nodiscard]] bool supportsExecutableUpdate() const noexcept override
        {
            return supports_executable_update_;
        }
        bool hasExecutable() const override { return executable_; }
        size_t nodeCount() const override { return 1; }
        void reset() override
        {
            capturing_ = false;
            executable_ = false;
        }
        const char *backendName() const override { return "FakeReplay"; }

        int begin_capture_calls_ = 0;
        int end_capture_calls_ = 0;
        int launch_calls_ = 0;
        int instantiate_calls_ = 0;
        int try_update_calls_ = 0;
        GraphUpdateResult update_result_ = GraphUpdateResult::Success;
        bool supports_executable_update_ = true;
        bool capturing_ = false;

    private:
        void *stream_ = nullptr;
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
        void destroyStream(void *) override { ++destroy_stream_calls_; }
        void *getOrCreateAuxiliaryStream(const std::string &, bool *created = nullptr) override
        {
            if (created)
                *created = false;
            return &auxiliary_stream_;
        }
        void resetAuxiliaryStreams() override {}

        void *createEvent() override
        {
            ++events_created_;
            return reinterpret_cast<void *>(static_cast<uintptr_t>(0xEF000000 + events_created_));
        }
        void destroyEvent(void *) override { ++events_destroyed_; }
        void recordEvent(void *, void *) override { ++events_recorded_; }
        void waitEvent(void *, void *) override { ++events_waited_; }
        void synchronizeEvent(void *) override { ++events_synchronized_; }
        bool synchronizeEventChecked(void *event) override
        {
            if (!event || !synchronize_event_checked_result_)
                return false;
            synchronizeEvent(event);
            return true;
        }
        float eventElapsedTime(void *, void *) override
        {
            ++event_elapsed_queries_;
            return elapsed_ms_;
        }
        void *blasHandle() override { return nullptr; }
        void *blasLtHandle() override { return nullptr; }
        void setCollectiveComm(void *) override {}
        void *collectiveComm() const override { return nullptr; }
        void synchronize() override { ++device_synchronize_calls_; }
        void synchronizeStream(void *) override { ++synchronize_stream_calls_; }
        bool synchronizeStreamChecked(void *) override
        {
            ++synchronize_stream_checked_calls_;
            return synchronize_stream_checked_result_;
        }
        bool insertStreamDependency(void *, void *) override { return true; }

        std::unique_ptr<IGPUGraphCapture> createGraphCapture() override
        {
            return std::make_unique<FakeReplayGraphCapture>(defaultStream());
        }

        std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *stream) override
        {
            return std::make_unique<FakeReplayGraphCapture>(stream);
        }

        int synchronize_stream_calls_ = 0;
        int synchronize_stream_checked_calls_ = 0;
        int destroy_stream_calls_ = 0;
        bool synchronize_stream_checked_result_ = true;
        int events_created_ = 0;
        int events_destroyed_ = 0;
        int events_recorded_ = 0;
        int events_waited_ = 0;
        int events_synchronized_ = 0;
        int event_elapsed_queries_ = 0;
        int device_synchronize_calls_ = 0;
        bool synchronize_event_checked_result_ = true;
        float elapsed_ms_ = 0.25f;

    private:
        int default_stream_ = 0;
        int capture_stream_ = 0;
        int auxiliary_stream_ = 0;
    };

    void addFakeSegmentStage(ComputeGraph &graph,
                             const std::string &name,
                             bool capturable,
                             bool manual_boundary = false,
                             ComputeStageType stage_type = ComputeStageType::COPY,
                             bool warmup_dependent_capture = false,
                             bool segment_boundary_before = false,
                             bool segment_boundary_after = false,
                             const uint64_t *variant_signature = nullptr)
    {
        graph.addNode(
            name,
            std::make_unique<FakeSegmentStage>(
                capturable,
                manual_boundary,
                stage_type,
                warmup_dependent_capture,
                segment_boundary_before,
                segment_boundary_after,
                variant_signature),
            DeviceId::cpu());
    }

    FakeGraphLaunchPrepStage *addFakeGraphLaunchPrepStage(
        ComputeGraph &graph,
        const std::string &name,
        DeviceId device)
    {
        auto stage = std::make_unique<FakeGraphLaunchPrepStage>(device);
        auto *raw_stage = stage.get();
        graph.addNode(name, std::move(stage), device);
        return raw_stage;
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
            mutableDebugEnv().reload();
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
            mutableDebugEnv().reload();
        }

        ScopedEnvVar(const ScopedEnvVar &) = delete;
        ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

    private:
        std::string name_;
        bool had_existing_ = false;
        std::string existing_;
    };

    double findCounterValue(const std::vector<PerfStatRecord> &records,
                            const std::string &domain,
                            const std::string &name,
                            const PerfStatsCollector::Tags &tags)
    {
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == domain &&
                record.name == name &&
                record.tags == tags)
            {
                return record.value;
            }
        }
        return -1.0;
    }

    double findCounterValue(const std::vector<PerfStatRecord> &records,
                            const std::string &name,
                            const PerfStatsCollector::Tags &tags)
    {
        return findCounterValue(records, "forward_graph", name, tags);
    }

    uint64_t findTimerCount(const std::vector<PerfStatRecord> &records,
                            const std::string &domain,
                            const std::string &name,
                            const PerfStatsCollector::Tags &tags)
    {
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Timer &&
                record.domain == domain &&
                record.name == name &&
                record.tags == tags)
            {
                return record.count;
            }
        }
        return 0;
    }

    uint64_t findTimerCount(const std::vector<PerfStatRecord> &records,
                            const std::string &name,
                            const PerfStatsCollector::Tags &tags)
    {
        return findTimerCount(records, "forward_graph", name, tags);
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

TEST(Test__ForwardGraphSignature, DifferentDeviceTokenSourceNotEqual)
{
    ForwardGraphSignature host_tokens{.seq_len = 1,
                                      .batch_size = 1,
                                      .decode = true,
                                      .uses_device_token_ids = false};
    ForwardGraphSignature device_tokens = host_tokens;
    device_tokens.uses_device_token_ids = true;

    EXPECT_NE(host_tokens, device_tokens);
    EXPECT_NE(ForwardGraphSignatureHash{}(host_tokens),
              ForwardGraphSignatureHash{}(device_tokens));
}

TEST(Test__ForwardGraphSignature, DifferentPositionPolicyNotEqual)
{
    ForwardGraphSignature explicit_rows{
        .seq_len = 256,
        .batch_size = 1,
        .device = DeviceId::cuda(0),
        .position_policy = ForwardPositionPolicy::ExplicitRows};
    ForwardGraphSignature contiguous_offset = explicit_rows;
    contiguous_offset.position_policy = ForwardPositionPolicy::ContiguousOffset;

    EXPECT_NE(explicit_rows, contiguous_offset);
    EXPECT_NE(
        ForwardGraphSignatureHash{}(explicit_rows),
        ForwardGraphSignatureHash{}(contiguous_offset));
}

TEST(Test__ForwardGraphSignature, DifferentDeviceSequenceLengthSourceNotEqual)
{
    ForwardGraphSignature external_rows{
        .seq_len = 16,
        .batch_size = 2,
        .all_position_logits = true,
        .all_position_logit_rows = 2,
        .uses_device_sequence_lengths = false};
    ForwardGraphSignature resident_request_lengths = external_rows;
    resident_request_lengths.uses_device_sequence_lengths = true;

    EXPECT_NE(external_rows, resident_request_lengths)
        << "Verifier-row and request-length-owned graphs must never share a cached executable";
    EXPECT_NE(
        ForwardGraphSignatureHash{}(external_rows),
        ForwardGraphSignatureHash{}(resident_request_lengths));
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

TEST(Test__ForwardGraphSignature, PrefixRuntimeRehydrationHasDedicatedGraphIdentity)
{
    ForwardGraphSignature steady{
        .seq_len = 128,
        .batch_size = 1,
        .decode = false,
        .rehydrate_prefix_runtime_on_device = false};
    ForwardGraphSignature restored = steady;
    restored.rehydrate_prefix_runtime_on_device = true;

    EXPECT_NE(steady, restored);
    EXPECT_NE(ForwardGraphSignatureHash{}(steady),
              ForwardGraphSignatureHash{}(restored))
        << "A restore graph contains an extra captured collective transaction "
           "and must never alias an ordinary prefill executable.";
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

/**
 * @brief Verify replay-time host positions are owned by the current forward input.
 *
 * Bucketed prefill graph replay can reuse the same captured shape for a later
 * prompt suffix chunk.  The cache therefore must not prefer graph-build
 * position rows over fresh replay rows, because RoPE would rotate K with the
 * previous chunk's absolute positions.
 */
TEST(Test__ForwardGraphCache, ReplayHostPositionIdsPreferCurrentInputOverCacheStorage)
{
    ForwardGraphCache cache;
    cache.position_ids = {0, 1, 2, 3};

    const std::vector<int> suffix_positions = {256, 257, 258, 259};
    ForwardInput input;
    input.position_ids = suffix_positions.data();
    input.seq_len = static_cast<int>(suffix_positions.size());

    const int *selected = selectForwardReplayHostPositionIds(cache, input);

    ASSERT_EQ(selected, suffix_positions.data());
    EXPECT_EQ(std::vector<int>(selected, selected + suffix_positions.size()),
              suffix_positions);
}

/**
 * @brief Verify cache-owned host positions are only a no-input compatibility fallback.
 */
TEST(Test__ForwardGraphCache, ReplayHostPositionIdsFallbackToCacheWhenInputHasNoRows)
{
    ForwardGraphCache cache;
    cache.position_ids = {7, 8, 9};

    ForwardInput input;
    input.seq_len = static_cast<int>(cache.position_ids.size());

    const int *selected = selectForwardReplayHostPositionIds(cache, input);

    ASSERT_EQ(selected, cache.position_ids.data());
    EXPECT_EQ(std::vector<int>(selected, selected + cache.position_ids.size()),
              cache.position_ids);
}

/**
 * @brief Verify device-resident replay positions remain the single source of truth.
 */
TEST(Test__ForwardGraphCache, ReplayHostPositionIdsDoNotMaskDeviceResidentRows)
{
    ForwardGraphCache cache;
    cache.position_ids = {0, 1, 2};

    const std::vector<int> host_shadow = {512, 513, 514};
    ForwardInput input;
    input.position_ids = host_shadow.data();
    input.position_ids_device = reinterpret_cast<const void *>(0xCAFE);
    input.seq_len = static_cast<int>(host_shadow.size());

    EXPECT_EQ(selectForwardReplayHostPositionIds(cache, input), nullptr);
}

/**
 * @brief Explicit position tables own every flattened request row.
 */
TEST(Test__ForwardGraphCache, PositionRowCountIncludesBatchDimension)
{
    ForwardInput input;
    input.batch_size = 2;
    input.seq_len = 16;

    EXPECT_EQ(forwardPositionRowCount(input), 32);
}

/**
 * @brief Invalid position geometry cannot wrap into a small RoPE launch.
 */
TEST(Test__ForwardGraphCache, PositionRowCountRejectsInvalidAndOverflowingGeometry)
{
    ForwardInput input;
    input.batch_size = 0;
    input.seq_len = 16;
    EXPECT_EQ(forwardPositionRowCount(input), 0);

    input.batch_size = 2;
    input.seq_len = std::numeric_limits<int>::max();
    EXPECT_EQ(forwardPositionRowCount(input), 0);
}

TEST(Test__DeviceGraphExecutor, CapturedTerminalStatePublishesCapturedStages)
{
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    auto stage = std::make_unique<FakeCapturedStateStage>(
        DeviceId::cpu(),
        /*has_capture=*/true);
    auto *stage_ptr = stage.get();
    graph.addNode("captured", std::move(stage), DeviceId::cpu());

    int stream_token = 0;
    void *stream = &stream_token;
    ASSERT_TRUE(executor.publishCapturedTerminalStateAfterGraphExecution(
        graph,
        /*terminal_row=*/7,
        stream,
        "unit"));

    EXPECT_EQ(stage_ptr->restore_calls_, 1);
    EXPECT_EQ(stage_ptr->last_row_, 7);
    EXPECT_EQ(stage_ptr->last_stream_, stream);
}

/**
 * @brief Capacity metadata must not turn an active scalar request into a batch.
 *
 * Request-batch-capable runners retain graph-stable device metadata at their
 * configured capacity. During a scalar oracle prefill that allocation remains
 * present, while the active request count is one. Publication must therefore
 * restore the scalar terminal row and ignore the inactive capacity entries.
 */
TEST(Test__DeviceGraphExecutor, ScalarPublicationIgnoresCapacitySizedDeviceLengths)
{
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    auto stage = std::make_unique<FakeCapturedStateStage>(
        DeviceId::cpu(),
        /*has_capture=*/true);
    auto *stage_ptr = stage.get();
    graph.addNode("captured", std::move(stage), DeviceId::cpu());

    const int capacity_lengths[2] = {8, 0};
    ASSERT_TRUE(executor.publishCapturedTerminalStateAfterGraphExecution(
        graph,
        /*terminal_row=*/7,
        /*producer_stream_override=*/nullptr,
        "scalar_with_batch_capacity",
        capacity_lengths,
        /*request_count=*/1,
        /*request_row_width=*/8));

    EXPECT_EQ(stage_ptr->restore_calls_, 1);
    EXPECT_EQ(stage_ptr->last_row_, 7);
}

/**
 * @brief A true request batch cannot silently use scalar publication.
 */
TEST(Test__DeviceGraphExecutor, RequestBatchRequiresHostOrDeviceLengths)
{
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    graph.addNode(
        "captured",
        std::make_unique<FakeCapturedStateStage>(
            DeviceId::cpu(),
            /*has_capture=*/true),
        DeviceId::cpu());

    EXPECT_FALSE(executor.publishCapturedTerminalStateAfterGraphExecution(
        graph,
        /*terminal_row=*/7,
        /*producer_stream_override=*/nullptr,
        "missing_request_lengths",
        /*device_request_seq_lens=*/nullptr,
        /*request_count=*/2,
        /*request_row_width=*/8));
}

/**
 * @brief CPU request batches publish one real terminal row per request.
 *
 * CPU grouped recurrence kernels own their request-length metadata on the
 * host. The executor must translate each real length into the flattened padded
 * row domain and invoke the grouped publication primitive exactly once. This
 * protects CPU request batching from accidentally inheriting the GPU-only
 * device-length contract or publishing the scalar request-zero tail.
 */
TEST(Test__DeviceGraphExecutor, CPURequestBatchPublishesHostOwnedTerminalRows)
{
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    auto stage = std::make_unique<FakeCapturedStateStage>(
        DeviceId::cpu(),
        /*has_capture=*/true);
    auto *stage_ptr = stage.get();
    graph.addNode("captured", std::move(stage), DeviceId::cpu());

    const std::vector<int> real_lengths{8, 5};
    ASSERT_TRUE(executor.publishCapturedTerminalStateAfterGraphExecution(
        graph,
        /*terminal_row=*/7,
        /*producer_stream_override=*/nullptr,
        "cpu_request_batch",
        /*device_request_seq_lens=*/nullptr,
        /*request_count=*/2,
        /*request_row_width=*/8,
        &real_lengths));

    EXPECT_EQ(stage_ptr->restore_calls_, 0);
    EXPECT_EQ(stage_ptr->restore_rows_calls_, 1);
    EXPECT_EQ(stage_ptr->last_rows_, (std::vector<int>{7, 12}));
    EXPECT_EQ(stage_ptr->last_stream_, nullptr);
}

/**
 * @brief GPU publication is governed only by resident request metadata.
 *
 * Serving may retain a host shadow for logging, but that shadow is not part of
 * the GPU execution transaction and may lag resident metadata. The executor
 * must neither validate nor dereference it while a GPU stage publishes from
 * the graph-stable device length vector.
 */
TEST(Test__DeviceGraphExecutor, GPURequestBatchIgnoresHostLengthShadow)
{
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    auto stage = std::make_unique<FakeCapturedStateStage>(
        DeviceId::cuda(0),
        /*has_capture=*/true);
    auto *stage_ptr = stage.get();
    graph.addNode("captured", std::move(stage), DeviceId::cuda(0));

    const int resident_lengths[2] = {8, 5};
    const std::vector<int> intentionally_stale_host_shadow{0};
    int stream_token = 0;
    void *stream = &stream_token;
    ASSERT_TRUE(executor.publishCapturedTerminalStateAfterGraphExecution(
        graph,
        /*terminal_row=*/7,
        stream,
        "gpu_request_batch",
        resident_lengths,
        /*request_count=*/2,
        /*request_row_width=*/8,
        &intentionally_stale_host_shadow));

    EXPECT_EQ(stage_ptr->restore_calls_, 0);
    EXPECT_EQ(stage_ptr->restore_rows_calls_, 0);
    EXPECT_EQ(stage_ptr->restore_device_rows_calls_, 1);
    EXPECT_EQ(stage_ptr->last_device_lengths_, resident_lengths);
    EXPECT_EQ(stage_ptr->last_request_count_, 2);
    EXPECT_EQ(stage_ptr->last_request_row_width_, 8);
    EXPECT_EQ(stage_ptr->last_stream_, stream);
}

TEST(Test__DeviceGraphExecutor, CapturedTerminalStateRequiresExplicitGPUStream)
{
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    auto stage = std::make_unique<FakeCapturedStateStage>(
        DeviceId::cuda(0),
        /*has_capture=*/true);
    graph.addNode("captured", std::move(stage), DeviceId::cuda(0));

    EXPECT_THROW(
        (void)executor.publishCapturedTerminalStateAfterGraphExecution(
            graph,
            /*terminal_row=*/3,
            nullptr,
            "unit"),
        std::logic_error);
}

TEST(Test__DeviceGraphExecutor, CapturedTerminalStateFailsWhenRequiredCaptureMissing)
{
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    auto stage = std::make_unique<FakeCapturedStateStage>(
        DeviceId::cpu(),
        /*has_capture=*/false,
        /*requires_capture=*/true);
    graph.addNode("missing_capture", std::move(stage), DeviceId::cpu());

    EXPECT_FALSE(executor.publishCapturedTerminalStateAfterGraphExecution(
        graph,
        /*terminal_row=*/0,
        nullptr,
        "unit"));
}

TEST(Test__GraphSegmentCache, ResetCanPreserveCaptureStream)
{
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    void *stream = cache.capture_stream;

    cache.initialized = true;
    cache.needs_capture = true;
    cache.decode_step = 17;

    cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);

    EXPECT_FALSE(cache.initialized);
    EXPECT_FALSE(cache.needs_capture);
    EXPECT_EQ(cache.decode_step, 0u);
    EXPECT_EQ(cache.capture_stream, stream);
}

/**
 * @brief Prove the worker-to-capture transition is a GPU event edge.
 *
 * This is the lifecycle edge that protects the first graph warmup from reading
 * stale KV/GDN state. It must never regress to a host wait or a full-device
 * synchronization, both of which distort LocalTP collective ordering.
 */
TEST(Test__GraphSegmentCache, CaptureStreamHandoffUsesEventWithoutDeviceSync)
{
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));

    ASSERT_TRUE(cache.orderCaptureStreamAfter(
        &gpu_ctx,
        gpu_ctx.defaultStream()));

    EXPECT_EQ(gpu_ctx.events_created_, 1);
    EXPECT_EQ(gpu_ctx.events_recorded_, 1);
    EXPECT_EQ(gpu_ctx.events_waited_, 1);
    EXPECT_EQ(gpu_ctx.device_synchronize_calls_, 0);
    EXPECT_EQ(gpu_ctx.synchronize_stream_calls_, 0);
    EXPECT_EQ(gpu_ctx.synchronize_stream_checked_calls_, 0);
}

/**
 * @brief Exercise the complete warmup transaction with a LocalTP-style hook.
 *
 * The first cached-decode invocation must queue the event handoff, invoke one
 * deterministic full-graph boundary, and only then execute warmup stages on the
 * capture stream. Actual LocalTP wires this hook to its rank rendezvous.
 */
TEST(Test__GraphSegmentCache, WarmupInvokesBoundaryAfterEventHandoff)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "collective_graph_stage", true);

    DeviceGraphExecutor executor;
    DeviceGraphExecutor::GraphSegmentCache cache;
    cache.perf_context = "main_decode";
    FakeReplayGPUContext gpu_ctx;
    executor.setWorkerGPUContextResolver(
        [&](DeviceId device) -> IWorkerGPUContext *
        {
            return device.is_gpu() ? &gpu_ctx : nullptr;
        });
    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);

    int boundary_calls = 0;
    std::string observed_boundary;
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph,
        &ctx,
        cache,
        gpu_ctx.defaultStream(),
        &gpu_ctx,
        nullptr,
        /*collectives_graph_capturable=*/true,
        /*force_recapture=*/false,
        /*defer_final_sync=*/false,
        [&](const std::string &boundary_name, void *capture_stream)
        {
            ++boundary_calls;
            observed_boundary = boundary_name;
            EXPECT_EQ(capture_stream, gpu_ctx.defaultStream());
            EXPECT_EQ(gpu_ctx.events_recorded_, 1);
            EXPECT_EQ(gpu_ctx.events_waited_, 1);
            return true;
        }));

    EXPECT_EQ(boundary_calls, 1);
    EXPECT_NE(observed_boundary.find("phase=warmup"), std::string::npos);
    EXPECT_NE(observed_boundary.find("scope=full_graph"), std::string::npos);
    EXPECT_NE(observed_boundary.find("context=main_decode"), std::string::npos);
    EXPECT_EQ(gpu_ctx.device_synchronize_calls_, 0);
    EXPECT_TRUE(cache.initialized);
    EXPECT_TRUE(cache.needs_capture);
}

TEST(Test__GraphSegmentCache, GraphLifecyclePhaseNamesAreCanonical)
{
    EXPECT_STREQ(
        DeviceGraphCaptureController::phaseName(
            DeviceGraphCaptureController::Phase::Warmup),
        "warmup");
    EXPECT_STREQ(
        DeviceGraphCaptureController::phaseName(
            DeviceGraphCaptureController::Phase::Capture),
        "capture");
    EXPECT_STREQ(
        DeviceGraphCaptureController::phaseName(
            DeviceGraphCaptureController::Phase::Replay),
        "replay");
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

TEST(Test__GraphSegmentCache, CaptureBoundariesCannotSegmentGraphWithoutHeterogeneousCollectives)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "before", true);
    addFakeSegmentStage(
        graph,
        "rocm_dynamic_attention",
        true,
        false,
        ComputeStageType::ATTENTION,
        false,
        true,
        true);
    addFakeSegmentStage(graph, "attention_gate", true);
    addFakeSegmentStage(graph, "after", true);
    graph.addDependency("rocm_dynamic_attention", "before");
    graph.addDependency("attention_gate", "rocm_dynamic_attention");
    graph.addDependency("after", "attention_gate");

    DeviceGraphExecutor::GraphSegmentCache cache;
    EXPECT_THROW(
        DeviceGraphCaptureController::buildWarmupSegments(
            graph,
            cache,
            nullptr,
            /*has_collective_nodes=*/false),
        std::runtime_error);
}

TEST(Test__GraphSegmentCache, ResetCanDestroyCaptureStream)
{
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));

    cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);

    EXPECT_EQ(cache.capture_stream, nullptr);
    EXPECT_EQ(gpu_ctx.synchronize_stream_checked_calls_, 0);
    EXPECT_EQ(gpu_ctx.events_created_, 1);
    EXPECT_EQ(gpu_ctx.events_recorded_, 1);
    EXPECT_EQ(gpu_ctx.events_synchronized_, 1);
    EXPECT_EQ(gpu_ctx.events_destroyed_, 1);
    EXPECT_EQ(gpu_ctx.destroy_stream_calls_, 1);
}

TEST(Test__GraphSegmentCache, ResetPreserveFencesExplicitCaptureStreamWithCheckedEvent)
{
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    void *stream = cache.capture_stream;
    ASSERT_NE(stream, nullptr);

    cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);

    EXPECT_EQ(cache.capture_stream, stream);
    EXPECT_EQ(cache.gpu_ctx_ref, &gpu_ctx);
    EXPECT_EQ(gpu_ctx.synchronize_stream_checked_calls_, 0);
    EXPECT_EQ(gpu_ctx.synchronize_stream_calls_, 0)
        << "Graph cache reset must never drain an entire stream";
    EXPECT_EQ(gpu_ctx.events_created_, 1);
    EXPECT_EQ(gpu_ctx.events_recorded_, 1);
    EXPECT_EQ(gpu_ctx.events_synchronized_, 1);
    EXPECT_EQ(gpu_ctx.events_destroyed_, 1);
    EXPECT_EQ(gpu_ctx.destroy_stream_calls_, 0);
}

/**
 * @brief Prove a rejected capture-stream event fence is process-fatal.
 *
 * Destroying a graph or stream after the backend rejected the completion-event
 * wait can race queued GPU work. The child process must stop at the failed
 * fence; it may not clear the handle and let inference limp onward.
 */
TEST(Test__GraphSegmentCache, ResetStopsProcessAfterCheckedEventFenceFailure)
{
    EXPECT_DEATH(
        {
            FakeReplayGPUContext gpu_ctx;
            DeviceGraphExecutor::GraphSegmentCache cache;
            gpu_ctx.synchronize_event_checked_result_ = false;
            if (!cache.ensureCaptureStream(&gpu_ctx))
                std::abort();

            cache.reset(
                DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Destroy);
        },
        "Fatal GPU graph resource lifecycle violation");
}

TEST(Test__ForwardGraphCache, ReplayResetPreservesSegmentCaptureStream)
{
    FakeReplayGPUContext gpu_ctx;
    ForwardGraphCache cache;
    ASSERT_TRUE(cache.segment_cache.ensureCaptureStream(&gpu_ctx));
    void *stream = cache.segment_cache.capture_stream;

    cache.segment_cache.initialized = true;
    cache.gpu_graph_update_failures = 3;
    cache.phase3_active = true;
    cache.graph_replay_live_state_epoch = 42;

    cache.resetReplayState();

    EXPECT_EQ(cache.segment_cache.capture_stream, stream);
    EXPECT_FALSE(cache.segment_cache.initialized);
    EXPECT_EQ(cache.gpu_graph_update_failures, 0);
    EXPECT_FALSE(cache.phase3_active);
    EXPECT_EQ(cache.graph_replay_live_state_epoch, 0u);
}

TEST(Test__ForwardGraphCache, MarkGPUStreamBindingsDirtyPreservesReplayState)
{
    FakeReplayGPUContext gpu_ctx;
    ForwardGraphCache cache;
    ASSERT_TRUE(cache.segment_cache.ensureCaptureStream(&gpu_ctx));
    void *stream = cache.segment_cache.capture_stream;

    cache.segment_cache.initialized = true;
    cache.segment_cache.needs_capture = false;
    cache.gpu_stream_applied = true;
    cache.applied_stream = stream;
    cache.phase3_active = true;
    cache.graph_replay_live_state_epoch = 42;

    cache.markGPUStreamBindingsDirty();

    EXPECT_EQ(cache.segment_cache.capture_stream, stream);
    EXPECT_TRUE(cache.segment_cache.initialized);
    EXPECT_FALSE(cache.segment_cache.needs_capture);
    EXPECT_FALSE(cache.gpu_stream_applied);
    EXPECT_EQ(cache.applied_stream, nullptr);
    EXPECT_TRUE(cache.phase3_active);
    EXPECT_EQ(cache.graph_replay_live_state_epoch, 42u);
}

TEST(Test__ForwardGraphCache, MarkReplayStateSafeForLiveEpochStampsPreservedCapture)
{
    ForwardGraphCache cache;
    cache.segment_cache.initialized = true;
    cache.segment_cache.needs_capture = false;
    cache.graph_replay_live_state_epoch = 17;

    cache.markReplayStateSafeForLiveEpoch(23);

    EXPECT_EQ(cache.graph_replay_live_state_epoch, 23u);
    EXPECT_TRUE(cache.segment_cache.initialized);
    EXPECT_FALSE(cache.segment_cache.needs_capture);
}

TEST(Test__ForwardGraphCache, RequestResetPreservesSegmentedReplayAndDemotesWarmupPrefill)
{
    ForwardGraphCache cache;
    cache.segment_cache.initialized = true;
    cache.segment_cache.needs_capture = false;
    cache.segment_cache.decode_step = 9;
    cache.gpu_stream_applied = true;
    cache.applied_stream = reinterpret_cast<void *>(0x4321);
    cache.gpu_graph_update_failures = 2;
    cache.phase3_active = true;
    cache.graph_replay_live_state_epoch = 17;

    PrefillGraphConfig prefill_config;
    prefill_config.enabled = true;
    prefill_config.min_seq_len = 1;
    cache.prefill_graph_cache = std::make_unique<PrefillGraphCache>(prefill_config);
    PrefillGraphCacheKey prefill_key;
    prefill_key.seq_len = 64;
    prefill_key.device_id = DeviceId::cuda(0);
    cache.prefill_graph_cache->markWarmedUp(prefill_key);
    ASSERT_EQ(cache.prefill_graph_cache->phase(prefill_key), PrefillGraphPhase::Warmup);

    cache.resetSessionStatePreservingGraphReplay();

    EXPECT_TRUE(cache.segment_cache.initialized)
        << "Replay-safe decode/verifier cached graph captures should stay hot across request reset.";
    EXPECT_FALSE(cache.segment_cache.needs_capture);
    EXPECT_EQ(cache.segment_cache.decode_step, 9u);
    EXPECT_FALSE(cache.gpu_stream_applied)
        << "Stage stream bindings must be dirtied so dynamic params rebind an explicit capture stream.";
    EXPECT_EQ(cache.applied_stream, nullptr);
    EXPECT_EQ(cache.gpu_graph_update_failures, 0);
    EXPECT_TRUE(cache.phase3_active);
    EXPECT_EQ(cache.graph_replay_live_state_epoch, 0u)
        << "Request reset clears live-state epoch stamps; only version-safe caches may use this path.";
    EXPECT_EQ(cache.prefill_graph_cache->phase(prefill_key), PrefillGraphPhase::Initialized)
        << "A warmed prefill bucket has no executable graph, so request reset must drop request arming "
           "while preserving lazy stage/kernel initialization for strict re-capture preflight.";
    EXPECT_EQ(cache.prefill_graph_cache->initializedCount(prefill_key), 1u);
}

TEST(Test__ForwardGraphCache, ReplayStateEpochClearsOnStateInvalidatingResets)
{
    ForwardGraphCache cache;
    cache.graph_replay_live_state_epoch = 17;
    cache.phase3_active = true;

    cache.resetReplayStateAfterWorkspaceRebind();
    EXPECT_EQ(cache.graph_replay_live_state_epoch, 0u);
    EXPECT_FALSE(cache.phase3_active);

    cache.graph_replay_live_state_epoch = 23;
    cache.valid = true;
    cache.resetSessionState();
    EXPECT_EQ(cache.graph_replay_live_state_epoch, 0u);

    cache.graph_replay_live_state_epoch = 29;
    cache.valid = true;
    cache.invalidate();
    EXPECT_EQ(cache.graph_replay_live_state_epoch, 0u);
    EXPECT_FALSE(cache.valid);
}

TEST(Test__ForwardGraphCache, LiveStateEpochRecaptureAppliesToReadyVersionedDecode)
{
    ForwardGraphCache cache;
    cache.segment_cache.initialized = true;
    cache.segment_cache.needs_capture = false;
    cache.graph_replay_live_state_epoch = 7;

    EXPECT_TRUE(cache.requiresLiveStateEpochRecapture(
        /*live_state_versioned_context=*/true,
        /*graph_replay_allowed=*/true,
        /*live_state_epoch=*/8));
    EXPECT_FALSE(cache.requiresLiveStateEpochRecapture(
        /*live_state_versioned_context=*/true,
        /*graph_replay_allowed=*/true,
        /*live_state_epoch=*/7));
    EXPECT_FALSE(cache.requiresLiveStateEpochRecapture(
        /*live_state_versioned_context=*/false,
        /*graph_replay_allowed=*/true,
        /*live_state_epoch=*/8))
        << "Single-row decode captures are version-safe and only need fresh dynamic metadata.";
    EXPECT_FALSE(cache.requiresLiveStateEpochRecapture(
        /*live_state_versioned_context=*/true,
        /*graph_replay_allowed=*/false,
        /*live_state_epoch=*/8));

    cache.segment_cache.needs_capture = true;
    EXPECT_FALSE(cache.requiresLiveStateEpochRecapture(
        /*live_state_versioned_context=*/true,
        /*graph_replay_allowed=*/true,
        /*live_state_epoch=*/8))
        << "A graph queued for capture does not need an extra recapture reset.";

    cache.segment_cache.needs_capture = false;
    cache.segment_cache.initialized = false;
    EXPECT_FALSE(cache.requiresLiveStateEpochRecapture(
        /*live_state_versioned_context=*/true,
        /*graph_replay_allowed=*/true,
        /*live_state_epoch=*/8));

    cache.segment_cache.initialized = true;
    cache.graph_replay_live_state_epoch = 0;
    EXPECT_FALSE(cache.requiresLiveStateEpochRecapture(
        /*live_state_versioned_context=*/true,
        /*graph_replay_allowed=*/true,
        /*live_state_epoch=*/8))
        << "Unstamped captures are handled by existing reset paths.";
}

TEST(Test__ForwardReplayStatePolicy, CorrectionReplayPreservesSingleTokenDecodeCaches)
{
    ForwardGraphSignature single_token_decode;
    single_token_decode.decode = true;
    single_token_decode.seq_len = 1;
    single_token_decode.batch_size = 1;
    single_token_decode.all_position_logits = false;

    ForwardGraphSignature ordinary_decode = single_token_decode;
    ordinary_decode.seq_len = 2;

    ForwardGraphSignature all_position_verifier = single_token_decode;
    all_position_verifier.all_position_logits = true;
    ForwardGraphSignature multirow_all_position_verifier = all_position_verifier;
    multirow_all_position_verifier.seq_len = 3;

    ForwardGraphSignature prefill;
    prefill.decode = false;
    ForwardGraphSignature bucketed_prefill = prefill;
    bucketed_prefill.is_bucketed_prefill = true;

    EXPECT_EQ(classifyForwardReplayStateCache(single_token_decode),
              ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode);
    EXPECT_EQ(classifyForwardReplayStateCache(ordinary_decode),
              ForwardReplayStateCacheClass::OrdinaryDecode);
    EXPECT_EQ(classifyForwardReplayStateCache(all_position_verifier),
              ForwardReplayStateCacheClass::AllPositionVerifier);
    EXPECT_TRUE(isLiveStateVersionedReplayCache(ordinary_decode));
    EXPECT_FALSE(isLiveStateVersionedReplayCache(single_token_decode));
    EXPECT_FALSE(isLiveStateVersionedReplayCache(all_position_verifier));
    EXPECT_FALSE(isLiveStateVersionedReplayCache(multirow_all_position_verifier))
        << "All-position verifier replay publishes row-local state through stage-owned capture slots "
           "and refreshes row metadata before every launch.";
    EXPECT_EQ(classifyForwardReplayStateCache(prefill),
              ForwardReplayStateCacheClass::ExactPrefill);
    EXPECT_EQ(classifyForwardReplayStateCache(bucketed_prefill),
              ForwardReplayStateCacheClass::BucketedPrefill);

    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                  classifyForwardReplayStateCache(single_token_decode)),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams)
        << "One-token condition decode updates token/position metadata before replay and reads stable live-state buffers.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                  ordinary_decode),
              ForwardReplayStateAction::ResetReplayState)
        << "Multi-token ordinary decode remains conservative until a versioned state contract proves it safe.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                  all_position_verifier),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams)
        << "Single-row verifier captures do not carry row-local multi-token state progression.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                  multirow_all_position_verifier),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams)
        << "Multi-row verifier captures stay warm across publication; stage capture slots and "
           "GPU event handoff carry the freshness contract.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                  classifyForwardReplayStateCache(prefill)),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams);
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                  bucketed_prefill),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams);

    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::GeneralLiveStateMutation,
                  classifyForwardReplayStateCache(all_position_verifier)),
              ForwardReplayStateAction::ResetReplayState)
        << "Only the MTP correction boundary may preserve verifier replay state.";
}

TEST(Test__ForwardReplayStatePolicy, RequestBoundaryPreservesOnlyReplaySafeDecodeClasses)
{
    ForwardGraphSignature single_token_decode;
    single_token_decode.decode = true;
    single_token_decode.seq_len = 1;
    single_token_decode.batch_size = 1;

    ForwardGraphSignature ordinary_decode = single_token_decode;
    ordinary_decode.seq_len = 2;

    ForwardGraphSignature all_position_verifier = ordinary_decode;
    all_position_verifier.all_position_logits = true;

    ForwardGraphSignature prefill;
    prefill.decode = false;
    ForwardGraphSignature bucketed_prefill = prefill;
    bucketed_prefill.is_bucketed_prefill = true;

    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  single_token_decode),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams)
        << "Single-token decode uses stable device buffers and refreshed token/position metadata.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  all_position_verifier),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams)
        << "All-position verifier rows publish through device-owned speculative slots.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  ordinary_decode),
              ForwardReplayStateAction::ResetReplayState)
        << "Multi-token ordinary decode is still live-state-versioned.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  prefill),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams)
        << "Exact prefill keeps the captured parity/serving fast path; monolithic prefill graph-cache executables reset separately.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  bucketed_prefill),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams)
        << "Bucketed prefill replay remains warm while request-local graph-cache entries are demoted.";
}

TEST(Test__ForwardReplayStatePolicy, RequestBoundaryResetsDecodeCachesWithCollectives)
{
    ForwardGraphSignature single_token_decode;
    single_token_decode.decode = true;
    single_token_decode.seq_len = 1;
    single_token_decode.batch_size = 1;

    ForwardGraphSignature all_position_verifier = single_token_decode;
    all_position_verifier.all_position_logits = true;

    ForwardGraphSignature prefill;
    prefill.decode = false;

    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  single_token_decode,
                  /*graph_has_collective_nodes=*/false),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams);
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  single_token_decode,
                  /*graph_has_collective_nodes=*/true),
              ForwardReplayStateAction::ResetReplayState)
        << "LocalTP/MoE decode graphs must not replay a graph executable captured for the previous request.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  all_position_verifier,
                  /*graph_has_collective_nodes=*/true),
              ForwardReplayStateAction::ResetReplayState)
        << "Verifier decode captures with collectives need the same request-boundary recapture contract.";
    EXPECT_EQ(chooseForwardReplayStateAction(
                  ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                  prefill,
                  /*graph_has_collective_nodes=*/true),
              ForwardReplayStateAction::PreserveReplayStateAndRebindStreams)
        << "The collective request-boundary guard applies to decode replay; prefill has a separate "
           "capture/readiness state machine.";
}

TEST(Test__ForwardGraphCache, InvalidateDestroysSegmentCaptureStream)
{
    FakeReplayGPUContext gpu_ctx;
    ForwardGraphCache cache;
    ASSERT_TRUE(cache.segment_cache.ensureCaptureStream(&gpu_ctx));

    cache.invalidate();

    EXPECT_EQ(cache.segment_cache.capture_stream, nullptr);
    EXPECT_EQ(gpu_ctx.destroy_stream_calls_, 1);
}

TEST(Test__GraphSegmentCache, HeterogeneousCollectivePolicyAdmitsNamedSparseBoundary)
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
        /*collectives_graph_capturable=*/false,
        DeviceGraphExecutor::GraphReplayPlanPolicy::
            AllowHeterogeneousCollectiveSegmentation);

    ASSERT_EQ(cache.segments.size(), 3u);
    EXPECT_TRUE(cache.segments[0].capturable);
    EXPECT_EQ(cache.segments[0].stage_names, std::vector<std::string>({"before"}));
    EXPECT_FALSE(cache.segments[1].capturable);
    EXPECT_EQ(cache.segments[1].stage_names, std::vector<std::string>({"sparse_dispatch"}));
    EXPECT_TRUE(cache.segments[2].capturable);
    EXPECT_EQ(cache.segments[2].stage_names, std::vector<std::string>({"after"}));
}

TEST(Test__GraphSegmentCache, NonCollectiveManualBoundaryIsFatal)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "before", true);
    addFakeSegmentStage(graph, "sparse_return", false, true);
    addFakeSegmentStage(graph, "after", true);
    graph.addDependency("sparse_return", "before");
    graph.addDependency("after", "sparse_return");

    DeviceGraphExecutor::GraphSegmentCache cache;
    EXPECT_THROW(
        DeviceGraphCaptureController::buildWarmupSegments(
            graph,
            cache,
            nullptr,
            /*has_collective_nodes=*/false),
        std::runtime_error);
}

TEST(Test__GraphSegmentCache, HomogeneousCollectiveCannotUseSegmentedReplay)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "before", true);
    addFakeSegmentStage(graph, "collective", false, true);
    addFakeSegmentStage(graph, "after", true);
    graph.addDependency("collective", "before");
    graph.addDependency("after", "collective");

    std::unordered_set<std::string> collective_nodes = {"collective"};
    DeviceGraphExecutor::GraphSegmentCache cache;
    EXPECT_THROW(
        DeviceGraphCaptureController::buildWarmupSegments(
            graph,
            cache,
            &collective_nodes,
            /*has_collective_nodes=*/true,
            /*collectives_graph_capturable=*/false),
        std::runtime_error);
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

/**
 * @brief Prove every unconditional collective type shares one canonical policy.
 *
 * Capture planning, fast scheduling, and MTP sidecar discovery all consume the
 * stage contract. This inventory prevents a specialized collective from being
 * added to the graph enum while remaining invisible to one of those consumers.
 */
TEST(Test__GraphSegmentCache, CanonicalCollectiveClassificationCoversSpecializedStages)
{
    constexpr std::array collective_types{
        ComputeStageType::ALLREDUCE,
        ComputeStageType::ALLGATHER,
        ComputeStageType::ALLGATHER_V,
        ComputeStageType::TP_KV_CACHE_STATE_ALLGATHER,
        ComputeStageType::GDN_LIVE_STATE_ALLGATHER,
        ComputeStageType::FUSED_ADD_ALLREDUCE,
    };

    for (const auto stage_type : collective_types)
    {
        EXPECT_TRUE(isCollectiveComputeStageType(stage_type))
            << computeStageTypeName(stage_type);
    }

    EXPECT_FALSE(isCollectiveComputeStageType(ComputeStageType::COPY));
    EXPECT_FALSE(isCollectiveComputeStageType(ComputeStageType::GEMM));
}

/**
 * @brief Lock the LocalTP MTP KV-state handoff into one captured graph.
 *
 * The TP KV-state allgather is an ordinary stream-ordered NCCL/RCCL graph node
 * on homogeneous LocalTP. It must not split the MTP sidecar into captured
 * compute plus a manually executed collective.
 */
TEST(Test__GraphSegmentCache, TPKVStateAllGatherRemainsInsideWholeCapturedGraph)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "sidecar_before", true);
    addFakeSegmentStage(
        graph,
        "tp_kv_state_allgather",
        true,
        false,
        ComputeStageType::TP_KV_CACHE_STATE_ALLGATHER);
    addFakeSegmentStage(graph, "sidecar_after", true);
    graph.addDependency("tp_kv_state_allgather", "sidecar_before");
    graph.addDependency("sidecar_after", "tp_kv_state_allgather");

    std::unordered_set<std::string> collective_nodes{
        "tp_kv_state_allgather"};
    DeviceGraphExecutor::GraphSegmentCache cache;
    DeviceGraphCaptureController::buildWarmupSegments(
        graph,
        cache,
        &collective_nodes,
        /*has_collective_nodes=*/true,
        /*collectives_graph_capturable=*/true);

    ASSERT_EQ(cache.segments.size(), 1u);
    EXPECT_TRUE(cache.segments.front().capturable);
    EXPECT_EQ(
        cache.segments.front().stage_names,
        std::vector<std::string>(
            {"sidecar_before", "tp_kv_state_allgather", "sidecar_after"}));
}

TEST(Test__GraphSegmentCache, UnauthorizedSegmentedPlanPublishesPerfStatsBeforeHardFailure)
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
    EXPECT_THROW(
        DeviceGraphCaptureController::buildWarmupSegments(
            graph,
            cache,
            nullptr,
            /*has_collective_nodes=*/false),
        std::runtime_error);

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

    const auto stage_records = PerfStatsCollector::snapshot({"stage_gpu"});
    EXPECT_DOUBLE_EQ(
        findCounterValue(
            stage_records,
            "stage_gpu",
            "graph_replay_plan_segments",
            {{"attribution", "graph_replay_metadata"},
             {"graph_capture_scope", "segmented_capture_plan"},
             {"source", "segmented_graph_capture"},
             {"type", "total"}}),
        3.0);
    EXPECT_DOUBLE_EQ(
        findCounterValue(
            stage_records,
            "stage_gpu",
            "graph_replay_plan_stage_types",
            {{"attribution", "graph_replay_metadata"},
             {"graph_capture_scope", "segmented_capture_plan"},
             {"segment_type", "manual"},
             {"source", "segmented_graph_capture"},
             {"stage_type", "ATTENTION"}}),
        1.0);
    EXPECT_DOUBLE_EQ(
        findCounterValue(
            stage_records,
            "stage_gpu",
            "graph_replay_plan_stage_types",
            {{"attribution", "graph_replay_metadata"},
             {"graph_capture_scope", "segmented_capture_plan"},
             {"segment_type", "capturable"},
             {"source", "segmented_graph_capture"},
             {"stage_type", "GEMM"}}),
        1.0);

    PerfStatsCollector::reset();
    mutableDebugEnv().execution.gpu_graph_defer_captured_collective_final_sync = false;
}

TEST(Test__GraphSegmentCache, FullGraphPlanPublishesPerfStatsAsGraph)
{
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    ComputeGraph graph;
    addFakeSegmentStage(graph, "gemm", true, false, ComputeStageType::GEMM);
    addFakeSegmentStage(graph, "lm_head", true, false, ComputeStageType::GEMM);
    graph.addDependency("lm_head", "gemm");

    DeviceGraphExecutor::GraphSegmentCache cache;
    DeviceGraphCaptureController::buildWarmupSegments(
        graph,
        cache,
        nullptr,
        /*has_collective_nodes=*/false);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});

    EXPECT_DOUBLE_EQ(findCounterValue(records, "full_graph_plan_graphs", {{"type", "total"}}), 1.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "full_graph_plan_graphs", {{"type", "capturable"}}), 1.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "full_graph_plan_graphs", {{"type", "manual"}}), 0.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "full_graph_plan_stages", {{"type", "capturable"}}), 2.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "full_graph_plan_stages", {{"type", "manual"}}), 0.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "full_graph_plan_max_graph_stages", {{"type", "capturable"}}), 2.0);
    EXPECT_DOUBLE_EQ(findCounterValue(records, "full_graph_plan_max_graph_stages", {{"type", "manual"}}), 0.0);
    EXPECT_DOUBLE_EQ(
        findCounterValue(
            records,
            "full_graph_plan_stage_types",
            {{"graph_type", "capturable"}, {"stage_type", "GEMM"}}),
        2.0);

    const auto stage_records = PerfStatsCollector::snapshot({"stage_gpu"});
    EXPECT_DOUBLE_EQ(
        findCounterValue(
            stage_records,
            "stage_gpu",
            "graph_replay_plan_graphs",
            {{"attribution", "graph_replay_metadata"},
             {"graph_capture_scope", "full_graph_capture_plan"},
             {"source", "full_graph_capture"},
             {"type", "total"}}),
        1.0);
    EXPECT_DOUBLE_EQ(
        findCounterValue(
            stage_records,
            "stage_gpu",
            "graph_replay_plan_stage_types",
            {{"attribution", "graph_replay_metadata"},
             {"graph_capture_scope", "full_graph_capture_plan"},
             {"graph_type", "capturable"},
             {"source", "full_graph_capture"},
             {"stage_type", "GEMM"}}),
        2.0);

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
        /*full_graph_replay=*/true,
        /*perf_context=*/"",
        /*device_name=*/"CUDA:0",
        [&](DeviceGraphExecutor::GraphSegment &, void *)
        {
            post_launch_called = true;
    }));

    EXPECT_TRUE(post_launch_called);
    EXPECT_EQ(gpu_ctx.synchronize_stream_checked_calls_, 1);
    EXPECT_EQ(gpu_ctx.synchronize_stream_calls_, 0);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags expected_tags = {
        {"first_stage", "gemm"},
        {"last_stage", "lm_head"},
        {"type", "capturable"},
        {"stage_count", "3"}};
    EXPECT_EQ(findTimerCount(records, "full_graph_replay_graph_launch", expected_tags), 1u);
    EXPECT_EQ(findTimerCount(records, "full_graph_replay_post_launch", expected_tags), 1u);

    PerfStatsCollector::reset();
    mutableDebugEnv().execution.gpu_graph_defer_captured_collective_final_sync = false;
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
        /*full_graph_replay=*/true,
        /*perf_context=*/"main_verifier",
        /*device_name=*/"CUDA:0",
        [](DeviceGraphExecutor::GraphSegment &, void *) {}));

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags expected_tags = {
        {"context", "main_verifier"},
        {"first_stage", "embedding"},
        {"last_stage", "lm_head"},
        {"type", "capturable"},
        {"stage_count", "3"}};
    EXPECT_EQ(findTimerCount(records, "full_graph_replay_graph_launch", expected_tags), 1u);
    EXPECT_EQ(findTimerCount(records, "full_graph_replay_post_launch", expected_tags), 1u);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, ReplayPhasePerfStatsRecordFinalCaptureEventFence)
{
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvVar enable_stage_timing("LLAMINAR_GPU_STAGE_TIMING", "1");
    PerfStatsCollector::reset();

    ComputeGraph graph;
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
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
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/false,
        /*collectives_graph_capturable=*/false,
        /*current_step=*/3,
        hooks);

    ASSERT_TRUE(result.success);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags capture_tags = {
        {"context", "main_verifier"},
        {"graph_count", "1"},
        {"stage_count", "1"},
        {"stream", "capture_event"},
        {"type", "capturable"}};
    const PerfStatsCollector::Tags default_tags = {
        {"context", "main_verifier"},
        {"graph_count", "1"},
        {"stage_count", "1"},
        {"stream", "context_default"},
        {"type", "capturable"}};
    const PerfStatsCollector::Tags aggregate_tags = {
        {"context", "main_verifier"},
        {"graph_count", "1"},
        {"stage_count", "1"},
        {"type", "capturable"}};
    const PerfStatsCollector::Tags host_aggregate_tags = {
        {"attribution", "host_wall"},
        {"context", "main_verifier"},
        {"graph_capture_scope", "full_graph_replay_host"},
        {"graph_count", "1"},
        {"source", "full_graph_capture"},
        {"stage_count", "1"},
        {"timing_scope", "final_stream_sync_host_wall"},
        {"type", "capturable"}};

    EXPECT_EQ(findTimerCount(records, "full_graph_replay_stream_sync", capture_tags), 1u);
    EXPECT_EQ(findTimerCount(records, "full_graph_replay_stream_sync", default_tags), 0u);
    EXPECT_EQ(findTimerCount(records, "full_graph_replay_final_sync", host_aggregate_tags), 1u);

    const auto stage_records = PerfStatsCollector::snapshot({"stage_gpu"});
    const PerfStatsCollector::Tags stage_total_tags = {
        {"attribution", "gpu_event"},
        {"context", "main_verifier"},
        {"graph_capture_scope", "full_graph_replay_events"},
        {"graph_count", "1"},
        {"source", "full_graph_capture"},
        {"stage_count", "1"},
        {"sync_scope", "stream_synchronized"},
        {"timing_scope", "total_replay_gpu_event"},
        {"type", "capturable"}};
    const PerfStatsCollector::Tags stage_segment_tags = {
        {"attribution", "gpu_event"},
        {"context", "main_verifier"},
        {"first_stage", "verifier_graph"},
        {"graph_capture_scope", "full_graph_replay_events"},
        {"graph_index", "0"},
        {"last_stage", "verifier_graph"},
        {"source", "full_graph_capture"},
        {"stage_count", "1"},
        {"sync_scope", "stream_synchronized"},
        {"timing_scope", "graph_replay_gpu_event"},
        {"type", "capturable"}};

    EXPECT_EQ(findTimerCount(stage_records, "stage_gpu", "graph_replay.total", stage_total_tags), 1u);
    EXPECT_EQ(findTimerCount(stage_records, "stage_gpu", "graph_replay.graph", stage_segment_tags), 1u);
    EXPECT_EQ(findTimerCount(stage_records, "stage_gpu", "graph_replay.final_sync", aggregate_tags), 0u);
    EXPECT_EQ(gpu_ctx.events_created_, 5);
    EXPECT_EQ(gpu_ctx.events_recorded_, 5);
    EXPECT_EQ(gpu_ctx.event_elapsed_queries_, 2);
    EXPECT_EQ(gpu_ctx.events_destroyed_, 4);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, ReplayPhasePreparesGraphLaunchMetadataOnExplicitCaptureStream)
{
    ComputeGraph graph;
    auto *prep_stage = addFakeGraphLaunchPrepStage(graph, "row_select", DeviceId::cuda(0));

    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"row_select"};
    cache.segments.back().capture = std::make_unique<FakeReplayGraphCapture>();

    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/false,
        /*collectives_graph_capturable=*/false,
        /*current_step=*/3,
        hooks);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(prep_stage->prepare_calls_, 1);
    EXPECT_EQ(prep_stage->execute_calls_, 0)
        << "Captured replay must update launch metadata without re-executing the stage body.";
    EXPECT_EQ(prep_stage->last_ctx_, &ctx);
    EXPECT_EQ(prep_stage->last_stream_, cache.capture_stream);
    EXPECT_NE(prep_stage->last_stream_, nullptr);
    EXPECT_EQ(prep_stage->stream_seen_by_stage_, cache.capture_stream);
}

TEST(Test__GraphSegmentCache, CapturePhasePreparesGraphLaunchMetadataBeforeRecording)
{
    ComputeGraph graph;
    auto *prep_stage = addFakeGraphLaunchPrepStage(graph, "row_select", DeviceId::rocm(0));

    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"row_select"};

    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeCapturePhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/false,
        /*current_step=*/2,
        hooks);

    ASSERT_TRUE(result.success);
    EXPECT_FALSE(result.reset_cache);
    EXPECT_EQ(prep_stage->prepare_calls_, 1);
    EXPECT_EQ(prep_stage->execute_calls_, 1);
    EXPECT_TRUE(prep_stage->executed_after_prepare_)
        << "Mutable row metadata must be uploaded before beginCapture records the stage body.";
    EXPECT_EQ(prep_stage->last_ctx_, &ctx);
    EXPECT_EQ(prep_stage->last_stream_, cache.capture_stream);
    EXPECT_NE(prep_stage->last_stream_, nullptr);
    EXPECT_EQ(prep_stage->stream_seen_by_stage_, cache.capture_stream);
}

/**
 * @brief Prove capture failure closes the backend transaction without recovery.
 *
 * The production ROCm long-context regression left a stream capturing after a
 * stage rejected graph recording, then attempted stream synchronization and
 * selective eager re-execution. This hardware-free test locks in the required
 * transaction semantics without initializing a GPU.
 */
TEST(Test__GraphSegmentCache, CaptureStageFailureEndsCaptureAndStopsExecution)
{
    ComputeGraph graph;
    auto failing_stage =
        std::make_unique<FakeCaptureFailureStage>(DeviceId::rocm(0));
    auto *failing_stage_ptr = failing_stage.get();
    graph.addNode(
        "capture_failure",
        std::move(failing_stage),
        DeviceId::rocm(0));

    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"capture_failure"};

    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::rocm(0),
        ComputeBackendType::GPU_ROCM);
    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeCapturePhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/false,
        /*current_step=*/2,
        hooks);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.reset_cache);
    ASSERT_NE(cache.segments.front().capture, nullptr);
    auto *capture = dynamic_cast<FakeReplayGraphCapture *>(
        cache.segments.front().capture.get());
    ASSERT_NE(capture, nullptr);
    EXPECT_EQ(capture->begin_capture_calls_, 1);
    EXPECT_EQ(capture->end_capture_calls_, 1);
    EXPECT_FALSE(capture->capturing_);
    EXPECT_EQ(failing_stage_ptr->execute_calls_, 1)
        << "A capture failure must not re-execute the failed stage eagerly";
    EXPECT_EQ(gpu_ctx.synchronize_stream_checked_calls_, 0)
        << "Capture entry and failure handling must not synchronize the stream";
    EXPECT_EQ(gpu_ctx.events_created_, 1);
    EXPECT_EQ(gpu_ctx.events_recorded_, 1);
    EXPECT_EQ(gpu_ctx.events_synchronized_, 1);
}

TEST(Test__GraphSegmentCache, CaptureManualSegmentRecordsSnapshotsAfterExecuteNodeCallback)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "manual_stage", false);

    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.segments.emplace_back();
    cache.segments.back().capturable = false;
    cache.segments.back().stage_names = {"manual_stage"};

    llaminar2::testing::MockDeviceContext ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);

    int execute_calls = 0;
    int prepare_snapshot_calls = 0;
    int record_snapshot_calls = 0;
    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        [&](ComputeNode &node)
        {
            ++execute_calls;
            return node.stage && node.stage->execute(&ctx);
        },
        [&](ComputeNode &, void *stream)
        {
            ++prepare_snapshot_calls;
            return stream == cache.capture_stream;
        },
        [&](ComputeNode &, void *stream)
        {
            ++record_snapshot_calls;
            return stream == cache.capture_stream;
        },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeCapturePhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/true,
        /*current_step=*/2,
        hooks);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(execute_calls, 1);
    EXPECT_EQ(prepare_snapshot_calls, 0);
    EXPECT_EQ(record_snapshot_calls, 1)
        << "Manual capture segments must snapshot even when they execute through execute_node.";
}

TEST(Test__GraphSegmentCache, ReplayManualSegmentRecordsSnapshotsAfterExecuteNodeCallback)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "manual_stage", false);

    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.perf_context = "manual_snapshot_regression";
    cache.segments.emplace_back();
    cache.segments.back().capturable = false;
    cache.segments.back().stage_names = {"manual_stage"};

    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    int execute_calls = 0;
    int prepare_snapshot_calls = 0;
    int record_snapshot_calls = 0;
    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        [&](ComputeNode &node)
        {
            ++execute_calls;
            return node.stage && node.stage->execute(&ctx);
        },
        [&](ComputeNode &, void *stream)
        {
            ++prepare_snapshot_calls;
            return stream == cache.capture_stream;
        },
        [&](ComputeNode &, void *stream)
        {
            ++record_snapshot_calls;
            return stream == cache.capture_stream;
        },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/true,
        /*collectives_graph_capturable=*/false,
        /*current_step=*/3,
        hooks);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(execute_calls, 1);
    EXPECT_EQ(prepare_snapshot_calls, 0);
    EXPECT_EQ(record_snapshot_calls, 1)
        << "Manual replay segments must snapshot even when they execute through execute_node.";
}

TEST(Test__GraphSegmentCache, ReplayPhaseStageGpuPerfStatsCanRequestGraphCapturedEvents)
{
    ScopedEnvVar disable_legacy_profile("LLAMINAR_PROFILING", "0");
    ScopedEnvVar disable_stage_timing("LLAMINAR_GPU_STAGE_TIMING", "0");
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvVar stage_gpu_filter("LLAMINAR_PERF_STATS_FILTER", "stage_gpu");
    PerfStatsCollector::reset();

    ComputeGraph graph;
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.perf_context = "main_decode";
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"captured_decode_graph"};
    cache.segments.back().capture = std::make_unique<FakeReplayGraphCapture>();

    llaminar2::testing::MockDeviceContext ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);

    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/false,
        /*collectives_graph_capturable=*/false,
        /*current_step=*/5,
        hooks);

    ASSERT_TRUE(result.success);

    const auto stage_records = PerfStatsCollector::snapshot({"stage_gpu"});
    const PerfStatsCollector::Tags total_tags = {
        {"attribution", "gpu_event"},
        {"context", "main_decode"},
        {"graph_capture_scope", "full_graph_replay_events"},
        {"graph_count", "1"},
        {"source", "full_graph_capture"},
        {"stage_count", "1"},
        {"sync_scope", "stream_synchronized"},
        {"timing_scope", "total_replay_gpu_event"},
        {"type", "capturable"}};
    const PerfStatsCollector::Tags segment_tags = {
        {"attribution", "gpu_event"},
        {"context", "main_decode"},
        {"first_stage", "captured_decode_graph"},
        {"graph_capture_scope", "full_graph_replay_events"},
        {"graph_index", "0"},
        {"last_stage", "captured_decode_graph"},
        {"source", "full_graph_capture"},
        {"stage_count", "1"},
        {"sync_scope", "stream_synchronized"},
        {"timing_scope", "graph_replay_gpu_event"},
        {"type", "capturable"}};

    EXPECT_EQ(findTimerCount(stage_records, "stage_gpu", "graph_replay.total", total_tags), 1u);
    EXPECT_EQ(findTimerCount(stage_records, "stage_gpu", "graph_replay.graph", segment_tags), 1u);
    EXPECT_EQ(gpu_ctx.events_created_, 5);
    EXPECT_EQ(gpu_ctx.events_recorded_, 5);
    EXPECT_EQ(gpu_ctx.event_elapsed_queries_, 2);
    EXPECT_EQ(gpu_ctx.events_destroyed_, 4);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, DeferredReplayStageGpuStatsUseSynchronizedGpuEvents)
{
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvVar enable_stage_timing("LLAMINAR_GPU_STAGE_TIMING", "1");
    PerfStatsCollector::reset();

    ComputeGraph graph;
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.perf_context = "mtp_decode_sidecar";
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"sidecar_graph"};
    cache.segments.back().capture = std::make_unique<FakeReplayGraphCapture>();

    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/false,
        /*collectives_graph_capturable=*/false,
        /*current_step=*/7,
        hooks,
        /*force_recapture=*/false,
        /*defer_final_sync=*/true);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(gpu_ctx.synchronize_stream_calls_, 0);
    EXPECT_EQ(gpu_ctx.events_synchronized_, 1);

    const auto stage_records = PerfStatsCollector::snapshot({"stage_gpu"});
    const PerfStatsCollector::Tags total_tags = {
        {"attribution", "gpu_event"},
        {"context", "mtp_decode_sidecar"},
        {"graph_capture_scope", "full_graph_replay_events"},
        {"graph_count", "1"},
        {"source", "full_graph_capture"},
        {"stage_count", "1"},
        {"sync_scope", "profiling_event_synchronized"},
        {"timing_scope", "total_replay_gpu_event"},
        {"type", "capturable"}};
    const PerfStatsCollector::Tags segment_tags = {
        {"attribution", "gpu_event"},
        {"context", "mtp_decode_sidecar"},
        {"first_stage", "sidecar_graph"},
        {"graph_capture_scope", "full_graph_replay_events"},
        {"graph_index", "0"},
        {"last_stage", "sidecar_graph"},
        {"source", "full_graph_capture"},
        {"stage_count", "1"},
        {"sync_scope", "profiling_event_synchronized"},
        {"timing_scope", "graph_replay_gpu_event"},
        {"type", "capturable"}};

    EXPECT_EQ(findTimerCount(stage_records, "stage_gpu", "graph_replay.total", total_tags), 1u);
    EXPECT_EQ(findTimerCount(stage_records, "stage_gpu", "graph_replay.graph", segment_tags), 1u);
    EXPECT_EQ(gpu_ctx.events_created_, 4);
    EXPECT_EQ(gpu_ctx.events_recorded_, 4);
    EXPECT_EQ(gpu_ctx.event_elapsed_queries_, 2);
    EXPECT_EQ(gpu_ctx.events_destroyed_, 4);

    const auto forward_records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags deferred_tags = {
        {"context", "mtp_decode_sidecar"},
        {"graph_count", "1"},
        {"stage_count", "1"},
        {"type", "capturable"}};
    EXPECT_DOUBLE_EQ(findCounterValue(
                         forward_records,
                         "full_graph_replay_final_sync_deferred",
                         deferred_tags),
                     1.0);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, CudaDeferredReplayDoesNotSynchronizeCapturedSegment)
{
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvVar disable_stage_timing("LLAMINAR_GPU_STAGE_TIMING", "0");
    ScopedEnvVar disable_perf_stage_timing("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", "0");
    PerfStatsCollector::reset();

    ComputeGraph graph;
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.perf_context = "moe_rebalance_maintenance";
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"maintenance_graph"};
    cache.segments.back().capture = std::make_unique<FakeReplayGraphCapture>();

    llaminar2::testing::MockDeviceContext ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);

    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/false,
        /*collectives_graph_capturable=*/false,
        /*current_step=*/11,
        hooks,
        /*force_recapture=*/false,
        /*defer_final_sync=*/true);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(gpu_ctx.synchronize_stream_checked_calls_, 0)
        << "Deferred CUDA replay must not synchronize the captured segment; "
           "callers order completion through stream/event dependencies.";
    EXPECT_EQ(gpu_ctx.synchronize_stream_calls_, 0);
    EXPECT_EQ(gpu_ctx.events_synchronized_, 0);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags deferred_tags = {
        {"context", "moe_rebalance_maintenance"},
        {"graph_count", "1"},
        {"stage_count", "1"},
        {"type", "capturable"}};
    EXPECT_DOUBLE_EQ(findCounterValue(
                         records,
                         "full_graph_replay_final_sync_deferred",
                         deferred_tags),
                     1.0);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, CapturedCollectiveReplayDefersFinalFenceWithoutOptIn)
{
    ScopedEnvVar disable_collective_defer("LLAMINAR_GPU_GRAPH_DEFER_CAPTURED_COLLECTIVE_FINAL_SYNC", "0");
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    mutableDebugEnv().execution.reload();
    PerfStatsCollector::reset();
    ASSERT_FALSE(debugEnv().execution.gpu_graph_defer_captured_collective_final_sync);

    ComputeGraph graph;
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.perf_context = "main_decode";
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"captured_collective_graph"};
    cache.segments.back().capture = std::make_unique<FakeReplayGraphCapture>();

    llaminar2::testing::MockDeviceContext ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);

    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/true,
        /*collectives_graph_capturable=*/true,
        /*current_step=*/9,
        hooks,
        /*force_recapture=*/false,
        /*defer_final_sync=*/true);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(gpu_ctx.synchronize_stream_checked_calls_, 0);
    EXPECT_EQ(gpu_ctx.synchronize_stream_calls_, 0)
        << "A fully captured collective graph is one device-owned replay DAG";

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags deferred_tags = {
        {"context", "main_decode"},
        {"graph_count", "1"},
        {"stage_count", "1"},
        {"type", "capturable"}};
    EXPECT_DOUBLE_EQ(findCounterValue(
                         records,
                         "full_graph_replay_final_sync_deferred",
                         deferred_tags),
                     1.0);

    PerfStatsCollector::reset();
}

TEST(Test__GraphSegmentCache, CapturedCollectiveReplayCanDeferFinalSyncWithOptIn)
{
    ScopedEnvVar enable_collective_defer("LLAMINAR_GPU_GRAPH_DEFER_CAPTURED_COLLECTIVE_FINAL_SYNC", "1");
    ScopedEnvVar enable_json("LLAMINAR_PERF_STATS_JSON", "1");
    ScopedEnvVar enable_stage_timing("LLAMINAR_GPU_STAGE_TIMING", "1");
    mutableDebugEnv().execution.reload();
    PerfStatsCollector::reset();

    ComputeGraph graph;
    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.ensureCaptureStream(&gpu_ctx));
    cache.perf_context = "main_decode";
    cache.segments.emplace_back();
    cache.segments.back().capturable = true;
    cache.segments.back().stage_names = {"captured_collective_graph"};
    cache.segments.back().capture = std::make_unique<FakeReplayGraphCapture>();

    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    DeviceGraphCaptureController::ReplayHooks hooks{
        nullptr,
        nullptr,
        [](ComputeNode &, void *) { return true; },
        [](ComputeNode &, void *) { return true; },
        [](DeviceGraphExecutor::GraphSegment &, void *) {}};

    const auto result = DeviceGraphCaptureController::executeReplayPhase(
        graph,
        cache,
        &ctx,
        &gpu_ctx,
        /*has_collective_nodes=*/true,
        /*collectives_graph_capturable=*/true,
        /*current_step=*/10,
        hooks,
        /*force_recapture=*/false,
        /*defer_final_sync=*/true);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(gpu_ctx.synchronize_stream_calls_, 0);
    EXPECT_EQ(gpu_ctx.events_synchronized_, 1);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags deferred_tags = {
        {"context", "main_decode"},
        {"graph_count", "1"},
        {"stage_count", "1"},
        {"type", "capturable"}};
    EXPECT_DOUBLE_EQ(findCounterValue(
                         records,
                         "full_graph_replay_final_sync_deferred",
                         deferred_tags),
                     1.0);

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
        false,
        false,
        &variant);

    FakeReplayGPUContext gpu_ctx;
    DeviceGraphExecutor executor;
    DeviceGraphExecutor::GraphSegmentCache cache;
    int worker_resolver_calls = 0;
    executor.setWorkerGPUContextResolver(
        [&](DeviceId device) -> IWorkerGPUContext *
        {
            ++worker_resolver_calls;
            return device.is_gpu() ? &gpu_ctx : nullptr;
        });
    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
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
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
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
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
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
    EXPECT_EQ(worker_resolver_calls, 0)
        << "Variant recapture already owns an explicit worker and must not resolve a second physical context.";
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
    capture_ptr->supports_executable_update_ = false;
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
        /*current_step=*/0,
        "unit_recapture",
        DeviceGraphExecutor::GraphCaptureBoundaryHook{},
        [](ComputeNode &, void *)
        {
            return true;
        },
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
        /*current_step=*/0,
        "unit_recapture",
        DeviceGraphExecutor::GraphCaptureBoundaryHook{},
        [](ComputeNode &, void *)
        {
            return true;
        },
        [&](DeviceGraphExecutor::GraphSegment &, void *)
        {
            post_launch_called = true;
        }));

    EXPECT_TRUE(post_launch_called);
    EXPECT_EQ(capture_ptr->try_update_calls_, 1);
    EXPECT_EQ(capture_ptr->instantiate_calls_, 0);
    EXPECT_EQ(capture_ptr->launch_calls_, 1);
}
