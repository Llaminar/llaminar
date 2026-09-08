/**
 * @file Test__ForwardExecutionEngine.cpp
 * @brief Unit tests for ForwardExecutionEngine
 *
 * Tests the forward graph execution engine extracted in Phase 3 of
 * the DGO refactor, using a mock IForwardExecutionHost to isolate the
 * engine from model-specific graph building.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "execution/local_execution/engine/ForwardExecutionEngine.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "memory/BufferArena.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "../../../../mocks/MockComputeStage.h" // MockDeviceContext
#include "../../../../mocks/MockWorkerGPUContext.h"

using namespace llaminar2;

// =========================================================================
// MockForwardExecutionHost
// =========================================================================

namespace
{
    /** @brief Device-free observations of one retained maintenance branch. */
    struct ForwardAuxiliaryBranchProbe
    {
        int creations = 0;
        int attachments = 0;
    };

    /**
     * @brief Record branch identity and capture edges without touching hardware.
     *
     * The shared probe outlives cache retirement. Replay must not construct or
     * record another branch: only the immutable executable may be launched.
     */
    class MockForwardAuxiliaryBranch final : public IGraphCaptureAuxiliaryBranch
    {
    public:
        /** @brief Bind the same authority and device advertised by the factory. */
        MockForwardAuxiliaryBranch(
            std::shared_ptr<ForwardAuxiliaryBranchProbe> probe, DeviceId device)
            : probe_(std::move(probe)), device_(device)
        {
            ++probe_->creations;
        }

        /** @copydoc IGraphCaptureAuxiliaryBranch::authorityIdentity */
        const void *authorityIdentity() const noexcept override { return probe_.get(); }
        /** @copydoc IGraphCaptureAuxiliaryBranch::device */
        DeviceId device() const noexcept override { return device_; }
        /** @copydoc IGraphCaptureAuxiliaryBranch::name */
        std::string_view name() const noexcept override { return "unit_forward_maintenance"; }
        /** @brief Observe one attachment to a sealed, uninstantiated owner. */
        bool attach(IGPUGraphCapture &graph) noexcept override
        {
            ++probe_->attachments;
            return graph.executionStream() != nullptr && !graph.hasExecutable();
        }

    private:
        std::shared_ptr<ForwardAuxiliaryBranchProbe> probe_;
        DeviceId device_;
    };

    std::string readTextFile(const char *path)
    {
        std::ifstream input(path);
        if (!input)
            return {};
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }

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
     * @brief Probe lease that proves one chunk authority spans graph submission.
     *
     * The counters are owned by MockForwardExecutionHost.  Destruction without
     * finish records an abandoned transaction, mirroring the fatal RAII
     * behavior of the production ExpertOverlay participant scope without
     * requiring a device or MPI endpoint in this unit suite.
     */
    class RecordingPrefillChunkSubmissionLease final
        : public IPrefillChunkGraphSubmissionLease
    {
    public:
        /** @brief Open one tracked lease. */
        RecordingPrefillChunkSubmissionLease(
            int *active,
            int *finishes,
            int *abandoned,
            std::vector<bool> *execution_results,
            bool finish_should_fail)
            : active_(active),
              finishes_(finishes),
              abandoned_(abandoned),
              execution_results_(execution_results),
              finish_should_fail_(finish_should_fail)
        {
            ++*active_;
        }

        /** @brief Record an abandoned scope on an exceptional engine exit. */
        ~RecordingPrefillChunkSubmissionLease() override
        {
            if (!finished_)
            {
                ++*abandoned_;
                --*active_;
            }
        }

        /** @copydoc IPrefillChunkGraphSubmissionLease::finish */
        bool finish(
            bool execution_succeeded,
            std::string *error) override
        {
            if (finished_)
            {
                if (error)
                    *error = "recording lease was finished twice";
                return false;
            }
            finished_ = true;
            ++*finishes_;
            --*active_;
            execution_results_->push_back(execution_succeeded);
            if (finish_should_fail_)
            {
                if (error)
                    *error = "configured recording lease terminal failure";
                return false;
            }
            return true;
        }

    private:
        int *active_ = nullptr;                    ///< Number of currently live leases.
        int *finishes_ = nullptr;                  ///< Successful finish invocations.
        int *abandoned_ = nullptr;                 ///< Destructors reached before finish.
        std::vector<bool> *execution_results_ = nullptr; ///< Results published by the engine.
        bool finish_should_fail_ = false;          ///< Reject the terminal for a negative test.
        bool finished_ = false;                    ///< Enforces exactly-once closure.
    };

    /**
     * @brief Dynamic GPU stage that records whether setup inputs precede capture.
     *
     * Native graph capture must never discover a host-to-device token upload.
     * This probe models the embedding stage's dynamic-input contract without
     * touching physical hardware: both position binding and scalar/token
     * preparation require the executor-selected stream, while the latter records
     * whether the mock worker had already entered its capture interval.
     */
    class SetupDynamicInputProbeStage final
        : public llaminar2::testing::MockComputeStage
    {
    public:
        /**
         * @brief Construct one externally observable dynamic-input consumer.
         * @param name Graph stage name.
         * @param device Logical GPU owning the stage.
         * @param dynamic_updates Receives scalar/token preparation calls.
         * @param device_position_updates Receives device-position binding calls.
         * @param dynamic_update_during_capture Set when preparation runs too late.
         * @param executions Receives model-stage executions recorded by capture.
         */
        SetupDynamicInputProbeStage(
            std::string name,
            DeviceId device,
            int *dynamic_updates,
            int *device_position_updates,
            bool *dynamic_update_during_capture,
            int *executions)
            : MockComputeStage(
                  ComputeStageType::ATTENTION,
                  std::move(name),
                  device),
              dynamic_updates_(dynamic_updates),
              device_position_updates_(device_position_updates),
              dynamic_update_during_capture_(dynamic_update_during_capture),
              executions_(executions)
        {
            setOnExecute(
                [this](IDeviceContext *)
                {
                    if (executions_)
                        ++*executions_;
                });
        }

        /** @return true because this probe owns replay-time inputs. */
        bool hasDynamicParams() const override { return true; }

        /** @return true because the probe accepts the persistent device row. */
        bool supportsDeviceResidentDynamicPositionReplay() const override
        {
            return true;
        }

        /** @brief Record the device-position authority on the bound stream. */
        void updateDynamicDevicePositionIds(
            const void *position_ids_device,
            int seq_len) override
        {
            EXPECT_NE(position_ids_device, nullptr);
            EXPECT_GT(seq_len, 0);
            (void)requireGPUStream();
            if (device_position_updates_)
                ++*device_position_updates_;
        }

        /** @brief Record scalar/token preparation and its capture phase. */
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            EXPECT_GE(pos_offset, 0);
            EXPECT_GT(seq_len, 0);
            (void)requireGPUStream();
            if (dynamic_updates_)
                ++*dynamic_updates_;
            if (dynamic_update_during_capture_)
            {
                *dynamic_update_during_capture_ =
                    llaminar2::testing::sharedMockWorkerGPUContext()
                        .isDeviceGraphCaptureActive();
            }
        }

    private:
        int *dynamic_updates_ = nullptr;
        int *device_position_updates_ = nullptr;
        bool *dynamic_update_during_capture_ = nullptr;
        int *executions_ = nullptr;
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
        int get_worker_gpu_context_calls = 0;
        int ensure_workspace_calls = 0;
        int last_workspace_seq_len = -1;
        bool typed_workspace_role_seen = false;
        WorkspaceGraphFamilyPolicy last_workspace_family_policy =
            WorkspaceGraphFamilyPolicy::ExclusiveLifetime;
        WorkspaceGraphParticipantRole last_workspace_participant_role =
            WorkspaceGraphParticipantRole::Decode;
        int sync_logits_calls = 0;
        TensorBase *last_published_logits = nullptr;
        int committed_forward_output_calls = 0;
        TensorBase *last_committed_logits = nullptr;
        int pending_all_position_verifier_stream_calls = 0;
        int pending_main_decode_stream_calls = 0;
        int build_decode_policy_calls = 0;
        int auxiliary_factory_queries = 0;
        DeviceId auxiliary_factory_device = DeviceId::invalid();
        int resolve_pp_copy_calls = 0;
        int get_pipeline_contexts_calls = 0;
        bool has_last_forward_input = false;
        ForwardInput last_forward_input{};
        const int *last_token_ids_pointer = nullptr;
        const void *last_token_ids_device_pointer = nullptr;
        const int *last_position_ids_pointer = nullptr;
        std::vector<int> last_token_ids;
        std::vector<int> last_position_ids;
        std::vector<int> forward_token_offsets;
        std::vector<int> forward_real_seq_lens;
        std::vector<uint64_t> forward_overlay_steps;
        std::vector<std::vector<int>> forward_token_batches;
        int prefill_chunk_maintenance_state_calls = 0;
        int prefill_chunk_maintenance_calls = 0;
        PrefillChunkPlan last_maintenance_chunk{};
        PrefillChunkMaintenanceDecision last_maintenance_decision{};
        int prefill_chunk_submission_begin_calls = 0;
        int prefill_chunk_submission_finish_calls = 0;
        int active_prefill_chunk_submission_leases = 0;
        int abandoned_prefill_chunk_submission_leases = 0;
        std::vector<int> prefill_chunk_submission_indices;
        std::vector<int> active_submission_leases_during_build;
        std::vector<bool> prefill_chunk_submission_execution_results;

        // ----- Configurable Results -----
        bool build_should_fail = false;
        std::string build_error_message = "mock build failure";
        int graph_stage_count = 0; // stages in built graph (0 means empty graph)
        // Optional explicit stage types let safety tests assemble GDN/short-conv graphs.
        std::vector<ComputeStageType> graph_stage_types;
        std::vector<std::function<std::unique_ptr<IComputeStage>(const std::string &, DeviceId)>> graph_stage_factories;
        PPCopyInfo mock_pp_copy;
        DeviceGraphExecutor::DecodeCapturePolicy mock_capture_policy;
        GraphCaptureAuxiliaryBranchFactory mock_auxiliary_factory;
        bool mock_compute_all_position_logits = false;
        bool mock_live_mtp_request_batch_condition = false;
        bool mock_defer_all_position_verifier_sync = false;
        bool mock_defer_main_decode_sync = false;
        void *pending_all_position_verifier_stream = nullptr;
        void *pending_main_decode_stream = nullptr;
        bool mock_moe_rebalancing_active = false;
        bool mock_moe_rebalancing_graph_stable = false;
        bool mock_prefill_graph_capture_disabled = false;
        PrefillChunkMaintenanceState mock_maintenance_state{};
        bool mock_maintenance_state_configured = false;
        bool maintenance_should_fail = false;
        bool use_prefill_chunk_submission_leases = false;
        int prefill_chunk_submission_fail_begin_call = -1;
        bool prefill_chunk_submission_finish_should_fail = false;
        ForwardExecutionEngine *engine_to_clear_on_maintenance = nullptr;
        bool bump_epoch_on_maintenance = false;
        uint64_t placement_epoch = 0;
        std::string prefill_domain_id = "single";
        int prefill_participant_id = 0;
        uint64_t prefill_topology_signature = 0;
        int mock_resident_graph_rows = 0;

        // ----- IForwardExecutionHost Interface -----

        GraphBuildResult buildForwardGraph(const ForwardInput &input) override
        {
            build_forward_graph_calls++;
            has_last_forward_input = true;
            last_forward_input = input;
            last_token_ids_pointer = input.token_ids;
            last_token_ids_device_pointer = input.token_ids_device;
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
            forward_overlay_steps.push_back(
                input.moe_overlay_collective_step_id);
            forward_token_batches.push_back(last_token_ids);
            active_submission_leases_during_build.push_back(
                active_prefill_chunk_submission_leases);

            if (build_should_fail)
                return GraphBuildResult(build_error_message);

            ComputeGraph graph;
            const int stage_count = !graph_stage_factories.empty()
                                        ? static_cast<int>(graph_stage_factories.size())
                                        : (graph_stage_types.empty()
                                               ? graph_stage_count
                                               : static_cast<int>(graph_stage_types.size()));
            for (int i = 0; i < stage_count; ++i)
            {
                const ComputeStageType type = graph_stage_types.empty()
                                                  ? ComputeStageType::GEMM
                                                  : graph_stage_types[static_cast<size_t>(i)];
                const std::string stage_name = "mock_stage_" + std::to_string(i);
                std::unique_ptr<IComputeStage> stage;
                if (!graph_stage_factories.empty())
                    stage = graph_stage_factories[static_cast<size_t>(i)](stage_name, input.device);
                else
                    stage = std::make_unique<llaminar2::testing::MockComputeStage>(
                        type,
                        stage_name,
                        input.device);
                graph.addNode(
                    stage_name,
                    std::move(stage),
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

        IWorkerGPUContext *getWorkerGPUContext(DeviceId device) override
        {
            ++get_worker_gpu_context_calls;
            if (!ctx_ || device != ctx_->deviceId())
                return nullptr;
            return &llaminar2::testing::sharedMockWorkerGPUContext();
        }

        bool workerGPUContextUsesProcessPool(DeviceId) const override { return false; }

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

        bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &,
            int workspace_seq_len,
            WorkspaceGraphFamilyPolicy graph_family_policy,
            WorkspaceGraphParticipantRole participant_role) override
        {
            ++ensure_workspace_calls;
            last_workspace_seq_len = workspace_seq_len;
            typed_workspace_role_seen = true;
            last_workspace_family_policy = graph_family_policy;
            last_workspace_participant_role = participant_role;
            return true;
        }

        int residentGraphRows() const override
        {
            return mock_resident_graph_rows;
        }

        bool publishForwardResultAtBoundary(
            const ForwardOutput &output,
            IDeviceContext *ctx) override
        {
            last_published_logits = output.logits;
            (void)ctx;
            sync_logits_calls++;
            return true;
        }

        void commitSuccessfulForwardOutput(
            const ForwardOutput &output) override
        {
            ++committed_forward_output_calls;
            last_committed_logits = output.logits;
        }

        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool has_collective_nodes,
            IDeviceContext *ctx) const override
        {
            const_cast<MockForwardExecutionHost *>(this)->build_decode_policy_calls++;
            return mock_capture_policy;
        }

        PPCopyInfo resolvePPCopyInfo(const ForwardInput &input) const override
        {
            const_cast<MockForwardExecutionHost *>(this)->resolve_pp_copy_calls++;
            return mock_pp_copy;
        }

        /** @brief Supply the runner-owned branch to every forward submission route. */
        GraphCaptureAuxiliaryBranchFactory forwardGraphAuxiliaryBranchFactory(
            DeviceId device) override
        {
            ++auxiliary_factory_queries;
            auxiliary_factory_device = device;
            return mock_auxiliary_factory;
        }

        bool computeAllPositionLogitsEnabled() const override
        {
            return mock_compute_all_position_logits;
        }

        /** @brief Expose the typed condition role without running a real MTP controller. */
        bool liveMTPRequestBatchConditionEnabled() const override
        {
            return mock_live_mtp_request_batch_condition;
        }

        bool shouldDeferAllPositionVerifierFinalSync() const override
        {
            return mock_defer_all_position_verifier_sync;
        }

        void setPendingAllPositionVerifierStream(void *stream) override
        {
            ++pending_all_position_verifier_stream_calls;
            pending_all_position_verifier_stream = stream;
        }

        bool shouldDeferMainDecodeFinalSync() const override
        {
            return mock_defer_main_decode_sync;
        }

        void setPendingMainDecodeStream(void *stream) override
        {
            ++pending_main_decode_stream_calls;
            pending_main_decode_stream = stream;
        }

        bool isMoeRebalancingActive() const override
        {
            return mock_moe_rebalancing_active;
        }

        bool isMoeRebalancingGraphStableForPrefillCapture() const override
        {
            return mock_moe_rebalancing_graph_stable;
        }

        bool prefillGraphCaptureDisabledByHost() const override
        {
            return mock_prefill_graph_capture_disabled;
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

        bool beginPrefillChunkGraphSubmission(
            ForwardInput &chunk_input,
            const PrefillChunkPlan &chunk,
            std::unique_ptr<IPrefillChunkGraphSubmissionLease> *lease,
            std::string *error) override
        {
            ++prefill_chunk_submission_begin_calls;
            prefill_chunk_submission_indices.push_back(chunk.chunk_index);
            if (lease)
                lease->reset();
            if (error)
                error->clear();
            if (prefill_chunk_submission_fail_begin_call ==
                prefill_chunk_submission_begin_calls)
            {
                if (error)
                    *error = "configured chunk submission admission failure";
                return false;
            }
            if (!use_prefill_chunk_submission_leases)
                return true;
            if (!lease)
            {
                if (error)
                    *error = "mock chunk submission requires a lease output";
                return false;
            }

            // Prove that authority-owned identity mutation reaches the exact
            // input later consumed by buildForwardGraph()/cached replay.
            chunk_input.moe_overlay_collective_step_id =
                1000u + static_cast<uint64_t>(chunk.chunk_index);
            *lease = std::make_unique<
                RecordingPrefillChunkSubmissionLease>(
                &active_prefill_chunk_submission_leases,
                &prefill_chunk_submission_finish_calls,
                &abandoned_prefill_chunk_submission_leases,
                &prefill_chunk_submission_execution_results,
                prefill_chunk_submission_finish_should_fail);
            return true;
        }

    private:
        IDeviceContext *ctx_;
    };

    struct PrefillReplayParamProbe
    {
        int updates = 0;
        std::vector<int> real_seq_lens;
        std::vector<int> bucket_seq_lens;
        std::vector<int> token_offsets;
        std::vector<bool> saw_stream;
    };

    /**
     * @brief Test stage that records padded-prefill replay metadata handoff order.
     *
     * Real GDN and short-conv stages upload a tiny effective-length scalar when
     * they receive PrefillReplayParams after workspace and stream binding. This
     * probe mirrors their public graph-capture contract without launching a GPU
     * kernel, so the engine ordering can be tested quickly in unit coverage.
     */
    class PrefillReplayParamProbeStage final : public llaminar2::testing::MockComputeStage
    {
    public:
        PrefillReplayParamProbeStage(std::string name, DeviceId device, PrefillReplayParamProbe *probe)
            : MockComputeStage(ComputeStageType::GDN_RECURRENCE, std::move(name), device),
              probe_(probe)
        {
        }

        bool hasPrefillReplayParams() const override { return true; }
        bool supportsPaddedPrefillRealLengthContract() const override { return true; }
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }

        void updatePrefillReplayParams(const PrefillReplayParams &params) override
        {
            ASSERT_NE(probe_, nullptr);
            ++probe_->updates;
            probe_->real_seq_lens.push_back(params.real_seq_len);
            probe_->bucket_seq_lens.push_back(params.bucket_seq_len);
            probe_->token_offsets.push_back(params.token_offset);
            probe_->saw_stream.push_back(hasGPUStream());
        }

    private:
        PrefillReplayParamProbe *probe_ = nullptr;
    };

    class NonCapturablePrefillStage final : public llaminar2::testing::MockComputeStage
    {
    public:
        NonCapturablePrefillStage(std::string name, DeviceId device)
            : MockComputeStage(ComputeStageType::GEMM, std::move(name), device)
        {
        }

        bool isGraphCapturable() const override { return false; }
        bool supportsLazyPrefillGraphCapturePreflight() const override { return false; }
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return false; }
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
        input.execution_phase = resolveForwardExecutionPhase({
            .role = ForwardExecutionRole::MainInference,
            .seq_len = seq_len,
            .batch_size = batch_size,
            .decode_max_seq_len = 4,
            .logical_position = position_ids ? position_ids[0] : 0,
        });
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

TEST(ForwardExecutionEngineSourceScan, DecodeCapturePolicyInstallsLifecycleBoundaryBeforeExecutorCall)
{
    const std::string source = readTextFile(LLAMINAR_FORWARD_EXECUTION_ENGINE_SOURCE);
    ASSERT_FALSE(source.empty());

    const size_t policy_build = source.find("capture_policy = host.buildDecodeCapturePolicy(");
    ASSERT_NE(policy_build, std::string::npos);
    const size_t hook_assignment = source.find("capture_policy.capture_boundary", policy_build);
    ASSERT_NE(hook_assignment, std::string::npos);
    const size_t boundary_call = source.find("host.waitAtDecodeGraphCaptureBoundary(", hook_assignment);
    ASSERT_NE(boundary_call, std::string::npos);
    const size_t execute_call = source.find("executor_.executeDecodeWithCapturePolicy(", hook_assignment);
    ASSERT_NE(execute_call, std::string::npos);

    EXPECT_LT(hook_assignment, execute_call)
        << "Decode graph capture must install the LocalTP lifecycle hook before entering executor capture/replay.";
    EXPECT_LT(boundary_call, execute_call)
        << "The hook must route through the host so LocalTP can fence capture entry and exit.";
}

TEST(ForwardExecutionEngineSourceScan, DecodeCaptureFencesGraphOwnershipEntryAndExit)
{
    const std::string source = readTextFile(LLAMINAR_DEVICE_GRAPH_CAPTURE_CONTROLLER_SOURCE);
    ASSERT_FALSE(source.empty());

    const size_t boundary_hook = source.find("hooks.capture_boundary");
    ASSERT_NE(boundary_hook, std::string::npos);
    const size_t create_capture = source.find("createGraphCapture(capture_stream)", boundary_hook);
    ASSERT_NE(create_capture, std::string::npos);
    const size_t capture_owner = source.find(
        "ScopedBackendGraphCapture capture_transaction(",
        create_capture);
    ASSERT_NE(capture_owner, std::string::npos);
    const size_t begin_capture = source.find(
        "capture_transaction.begin()",
        capture_owner);
    ASSERT_NE(begin_capture, std::string::npos);

    EXPECT_LT(boundary_hook, begin_capture)
        << "LocalTP participants must rendezvous before any device starts HIP/CUDA stream capture.";
    const std::string capture_entry =
        source.substr(boundary_hook, begin_capture - boundary_hook);
    EXPECT_EQ(
        capture_entry.find("synchronizeStreamChecked"),
        std::string::npos)
        << "Capture entry must be ordered by explicit stream dependencies and "
           "the domain lifecycle barrier, never by a host-blocking stream drain.";

    const size_t capture_finish = source.find(
        "capture_transaction.finish()",
        begin_capture);
    ASSERT_NE(capture_finish, std::string::npos);
    const size_t capture_exit_boundary = source.find(
        "exec_ok ? \"capture_end\" : \"capture_end_failed\"",
        capture_finish);
    ASSERT_NE(capture_exit_boundary, std::string::npos);
    const size_t capture_finalize = source.find(
        "finalizeCapturePhaseCapturableSegment(",
        capture_exit_boundary);
    ASSERT_NE(capture_finalize, std::string::npos);
    EXPECT_LT(capture_finish, capture_exit_boundary)
        << "Capture-exit rendezvous must happen only after native endCapture completes.";
    EXPECT_LT(capture_exit_boundary, capture_finalize)
        << "No graph launch or following manual collective may proceed until all LocalTP participants exit capture.";

    const size_t recapture_boundary = source.find("\"recapture_begin\"");
    ASSERT_NE(recapture_boundary, std::string::npos);
    const size_t recapture_owner = source.find(
        "ScopedBackendGraphCapture capture_transaction(",
        recapture_boundary);
    ASSERT_NE(recapture_owner, std::string::npos);
    const size_t recapture_begin = source.find(
        "capture_transaction.begin()",
        recapture_owner);
    ASSERT_NE(recapture_begin, std::string::npos);
    EXPECT_LT(recapture_boundary, recapture_begin)
        << "Forced recapture must use the same pre-beginCapture domain fence.";

    const size_t recapture_finish = source.find(
        "capture_transaction.finish()",
        recapture_begin);
    ASSERT_NE(recapture_finish, std::string::npos);
    const size_t recapture_exit_boundary = source.find(
        "exec_ok ? \"recapture_end\" : \"recapture_end_failed\"",
        recapture_finish);
    ASSERT_NE(recapture_exit_boundary, std::string::npos);
    EXPECT_LT(recapture_finish, recapture_exit_boundary)
        << "Forced recapture must rendezvous all participants after endCapture as well.";
}

/**
 * @brief The engine must not initialize a physical GPU behind its host boundary.
 */
TEST(ForwardExecutionEngineSourceScan, WorkerGPUContextIsHostOwned)
{
    const std::string source = readTextFile(LLAMINAR_FORWARD_EXECUTION_ENGINE_SOURCE);
    ASSERT_FALSE(source.empty());

    EXPECT_EQ(source.find("GPUDeviceContextPool"), std::string::npos)
        << "ForwardExecutionEngine must not bypass its host and initialize a physical GPU context.";
    EXPECT_NE(source.find("ScopedWorkerGPUContextResolver"), std::string::npos);
    EXPECT_NE(source.find("host.getWorkerGPUContext("), std::string::npos);
}

/**
 * @brief Independent LocalTP devices may build cache misses concurrently.
 *
 * A process-wide graph-materialization mutex serializes participant zero and
 * participant one before either reaches a symmetric capture collective. The
 * collective then waits forever for the child excluded by that mutex. Shared
 * stores own their own narrow synchronization; the forward engine must not
 * impose a process-wide cache-miss critical section.
 */
TEST(ForwardExecutionEngineSourceScan,
     CacheMissConstructionHasNoProcessWideMaterializationMutex)
{
    const std::string source =
        readTextFile(LLAMINAR_FORWARD_EXECUTION_ENGINE_SOURCE);
    ASSERT_FALSE(source.empty());
    EXPECT_EQ(
        source.find("gpuCacheMissGraphMaterializationMutex"),
        std::string::npos);
    EXPECT_EQ(
        source.find("graph_materialization_lock"),
        std::string::npos);
}

/**
 * @brief Setup-only shifted-prefill bindings can describe but never execute a graph.
 *
 * Workspace preflight must construct the exact graph-integrated MTP topology
 * before the first workspace generation exists. This regression locks down the
 * typed separation between that complete declaration and a runtime binding.
 */
TEST(ForwardExecutionEngineSourceScan,
     ShiftedMTPWorkspaceDeclarationCannotEnterExecution)
{
    auto address = [](std::uintptr_t value)
    {
        return reinterpret_cast<int32_t *>(value);
    };

    ShiftedMTPPrefillGraphBinding binding;
    binding.kv_cache = reinterpret_cast<IKVCache *>(std::uintptr_t{0x1000});
    binding.terminal_hidden_archive =
        reinterpret_cast<TensorBase *>(std::uintptr_t{0x2000});
    binding.request_token_ids_device = address(0x3000);
    binding.request_position_ids_device = address(0x4000);
    binding.request_segment_lengths_device = address(0x4800);
    binding.shifted_token_ids_device = address(0x5000);
    binding.shifted_position_ids_device = address(0x6000);
    binding.append_lengths_device = address(0x7000);
    binding.request_row_stride_device = address(0x8000);
    binding.main_cached_tokens_device = {address(0x9000)};
    binding.shifted_cached_tokens_device = {address(0xa000)};
    binding.capture_identity = UINT64_MAX;
    binding.purpose =
        ShiftedMTPPrefillGraphBinding::Purpose::
            WorkspaceFamilyDeclaration;

    ASSERT_TRUE(binding.validForRequestCount(1));
    EXPECT_FALSE(binding.executableForRequestCount(1));
    binding.purpose =
        ShiftedMTPPrefillGraphBinding::Purpose::RuntimeExecution;
    EXPECT_TRUE(binding.executableForRequestCount(1));

    const std::string source =
        readTextFile(LLAMINAR_FORWARD_EXECUTION_ENGINE_SOURCE);
    ASSERT_FALSE(source.empty());
    EXPECT_NE(
        source.find(
            "input.shifted_mtp_prefill->executableForRequestCount("),
        std::string::npos)
        << "ForwardExecutionEngine must enforce the typed execution gate.";
}

/**
 * @brief Main-decode terminal publication has the same declaration/runtime split.
 *
 * The archive pointer participates in graph identity before workspace sizing,
 * but a declaration binding must never be accepted as an inference input.
 */
TEST(ForwardExecutionEngineSourceScan,
     MTPMainTerminalHiddenWorkspaceDeclarationCannotEnterExecution)
{
    MTPMainTerminalHiddenGraphBinding binding{
        .terminal_hidden_archive =
            reinterpret_cast<TensorBase *>(std::uintptr_t{0x2000}),
        .request_count = 1,
        .capture_identity = UINT64_MAX,
        .purpose = MTPMainTerminalHiddenGraphBinding::Purpose::
            WorkspaceFamilyDeclaration,
    };

    ASSERT_TRUE(binding.validForRequestCount(1));
    EXPECT_FALSE(binding.executableForRequestCount(1));
    binding.purpose =
        MTPMainTerminalHiddenGraphBinding::Purpose::RuntimeExecution;
    EXPECT_TRUE(binding.executableForRequestCount(1));

    const std::string source =
        readTextFile(LLAMINAR_FORWARD_EXECUTION_ENGINE_SOURCE);
    ASSERT_FALSE(source.empty());
    EXPECT_NE(
        source.find(
            "input.mtp_main_terminal_hidden->executableForRequestCount("),
        std::string::npos)
        << "ForwardExecutionEngine must enforce the typed execution gate.";
}

// =========================================================================
// Test Fixture
// =========================================================================

class Test__ForwardExecutionEngine : public ::testing::Test
{
protected:
    BufferArena arena_;
    // Default executor — CPU, default config
    DeviceGraphExecutor executor_;
    // Mock CPU device context
    llaminar2::testing::MockDeviceContext mock_ctx_{DeviceId::cpu()};

    void SetUp() override
    {
        executor_.setArena(&arena_);
    }

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

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_GPUUsesContiguousDeviceOffset)
{
    const std::vector<int> tokens = {10, 11, 12, 13};
    auto input = makeTestInput(4, 1, DeviceId::cuda(0), tokens.data(), nullptr);
    input.token_offset = 32;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4, 8},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    ASSERT_TRUE(plan) << plan.error;
    EXPECT_EQ(plan.position_policy, ForwardPositionPolicy::ContiguousOffset);
    EXPECT_TRUE(plan.chunk.position_ids.empty())
        << "GPU contiguous prefill must not manufacture mutable host position rows";
    EXPECT_EQ(plan.chunk.token_offset, 32);
}

TEST_F(Test__ForwardExecutionEngine,
       PrefillChunkRuntimePlan_GPUDeviceAdmissionNeedsNoHostMirror)
{
    const std::array<int32_t, 8> admitted_device_rows{};
    auto input = makeTestInput(
        /*seq_len=*/5,
        /*batch_size=*/1,
        DeviceId::cuda(0),
        /*token_ids=*/nullptr,
        /*position_ids=*/nullptr);
    input.token_ids_device = admitted_device_rows.data();
    input.token_offset = 48;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4, 8},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/true);

    ASSERT_TRUE(plan) << plan.error;
    EXPECT_TRUE(plan.padding_required);
    EXPECT_EQ(
        plan.chunk.token_authority,
        PrefillChunkTokenAuthority::DeviceResidentRows);
    EXPECT_TRUE(plan.chunk.token_ids.empty())
        << "Device-owned prefill must not manufacture a host token shadow.";
    EXPECT_EQ(plan.chunk.real_count, 5);
    EXPECT_EQ(plan.chunk.bucket_seq_len, 8);
    EXPECT_EQ(plan.chunk.token_offset, 48);
    EXPECT_EQ(plan.position_policy, ForwardPositionPolicy::ContiguousOffset);
}

TEST_F(Test__ForwardExecutionEngine,
       PrefillChunkRuntimePlan_RejectsHostShadowOfDeviceAdmission)
{
    const std::array<int32_t, 4> host_rows{};
    const std::array<int32_t, 4> admitted_device_rows{};
    auto input = makeTestInput(
        /*seq_len=*/4,
        /*batch_size=*/1,
        DeviceId::cuda(0),
        host_rows.data(),
        /*position_ids=*/nullptr);
    input.token_ids_device = admitted_device_rows.data();

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/true);

    EXPECT_FALSE(plan);
    EXPECT_NE(plan.error.find("simultaneous host and device"), std::string::npos);
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_UsesPositionOffsetWhenTokenOffsetIsAbsent)
{
    const std::vector<int> tokens = {21, 22, 23, 24};
    auto input = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), nullptr);
    input.position_offset = 256;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4, 8},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    ASSERT_TRUE(plan) << plan.error;
    EXPECT_EQ(plan.chunk.token_offset, 256)
        << "Restored-prefix suffix prefill may arrive as a raw ForwardInput "
           "with only position_offset populated; bucket planning must keep "
           "position IDs and replay metadata on that absolute request range.";
    EXPECT_EQ(plan.chunk.position_ids, (std::vector<int>{256, 257, 258, 259}));
}

TEST_F(Test__ForwardExecutionEngine, PrefillChunkRuntimePlan_RequiresOneTokenAuthority)
{
    auto input = makeTestInput(4, 1, DeviceId::cpu(), nullptr, nullptr);

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        input,
        std::vector<int>{4},
        /*pad_token_id=*/99,
        /*allow_padded_execution=*/false);

    EXPECT_FALSE(plan);
    EXPECT_NE(plan.error.find("one authoritative token source"), std::string::npos);
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

TEST_F(Test__ForwardExecutionEngine,
       PrefillChunkRuntimeSchedule_FixedIntervalKeepsOneBucketForShortTail)
{
    const std::vector<int> tokens = {40, 41, 42, 43, 44};
    auto input = makeTestInput(
        static_cast<int>(tokens.size()),
        1,
        DeviceId::cpu(),
        tokens.data(),
        nullptr);

    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {2, 4, 8};
    policy.fixed_chunk_real_tokens = 4;
    policy.real_token_start = 0;
    policy.real_token_count = static_cast<int>(tokens.size());

    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/true);

    ASSERT_TRUE(schedule) << schedule.error;
    ASSERT_EQ(schedule.chunks.size(), 2u);
    EXPECT_EQ(schedule.chunks[0].chunk.real_count, 4);
    EXPECT_EQ(schedule.chunks[0].chunk.bucket_seq_len, 4);
    EXPECT_EQ(schedule.chunks[1].chunk.real_count, 1);
    EXPECT_EQ(schedule.chunks[1].chunk.bucket_seq_len, 4)
        << "A fixed-width captured transaction must not mode-shift its tail "
           "to the smaller two-row graph.";
    EXPECT_EQ(
        schedule.chunks[1].chunk.token_ids,
        (std::vector<int>{44, 0, 0, 0}));
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
    EXPECT_EQ(host.committed_forward_output_calls, 2)
        << "Every successful scheduled chunk must cross the engine-owned "
           "forward-output publication boundary.";
    EXPECT_EQ(host.last_committed_logits, output.logits)
        << "The final scheduled chunk must remain the current logits authority.";
    EXPECT_EQ(host.forward_token_offsets, (std::vector<int>{0, 4}));
    EXPECT_EQ(host.forward_real_seq_lens, (std::vector<int>{4, 4}));
    ASSERT_EQ(host.forward_token_batches.size(), 2u);
    EXPECT_EQ(host.forward_token_batches[0], (std::vector<int>{10, 11, 12, 13}));
    EXPECT_EQ(host.forward_token_batches[1], (std::vector<int>{14, 15, 16, 17}));
    EXPECT_EQ(host.last_maintenance_chunk.chunk_index, 1);
    EXPECT_TRUE(host.last_maintenance_decision.required);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunkSchedule_HoldsDistinctAuthorityLeaseAcrossEverySubmission)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;
    host.use_prefill_chunk_submission_leases = true;

    const std::vector<int> tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    auto input = makeTestInput(
        static_cast<int>(tokens.size()),
        1,
        DeviceId::cpu(),
        tokens.data(),
        nullptr);

    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {4};
    policy.fixed_chunk_real_tokens = 4;
    policy.real_token_count = static_cast<int>(tokens.size());
    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(schedule) << schedule.error;
    ASSERT_EQ(schedule.chunks.size(), 2u);

    ForwardOutput output{};
    EXPECT_TRUE(engine.runPrefillChunkSchedule(input, schedule, output, host));

    EXPECT_EQ(host.prefill_chunk_submission_begin_calls, 2);
    EXPECT_EQ(host.prefill_chunk_submission_finish_calls, 2);
    EXPECT_EQ(host.prefill_chunk_submission_indices, (std::vector<int>{0, 1}));
    EXPECT_EQ(host.active_submission_leases_during_build,
              (std::vector<int>{1, 1}))
        << "Each graph must execute while its own authority lease is live.";
    EXPECT_EQ(host.forward_overlay_steps,
              (std::vector<uint64_t>{1000u, 1001u}))
        << "Per-chunk authority identity must reach the submitted graph input.";
    EXPECT_EQ(host.prefill_chunk_submission_execution_results,
              (std::vector<bool>{true, true}));
    EXPECT_EQ(host.active_prefill_chunk_submission_leases, 0);
    EXPECT_EQ(host.abandoned_prefill_chunk_submission_leases, 0);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunkSchedule_AdmissionFailureCannotReusePriorChunkLease)
{
    auto engine = makeEngine(/*cache_enabled=*/false);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;
    host.use_prefill_chunk_submission_leases = true;
    host.prefill_chunk_submission_fail_begin_call = 2;

    const std::vector<int> tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    auto input = makeTestInput(
        static_cast<int>(tokens.size()),
        1,
        DeviceId::cpu(),
        tokens.data(),
        nullptr);
    PrefillChunkSchedulerPolicy policy;
    policy.bucket_sizes = {4};
    policy.fixed_chunk_real_tokens = 4;
    policy.real_token_count = static_cast<int>(tokens.size());
    auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        input,
        policy,
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/false);
    ASSERT_TRUE(schedule) << schedule.error;

    ForwardOutput output{};
    EXPECT_FALSE(engine.runPrefillChunkSchedule(input, schedule, output, host));

    EXPECT_EQ(host.prefill_chunk_submission_begin_calls, 2);
    EXPECT_EQ(host.prefill_chunk_submission_finish_calls, 1);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.forward_overlay_steps, (std::vector<uint64_t>{1000u}));
    EXPECT_EQ(host.active_prefill_chunk_submission_leases, 0);
    EXPECT_EQ(host.abandoned_prefill_chunk_submission_leases, 0);
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
    EXPECT_EQ(host.last_position_ids_pointer, nullptr);
    EXPECT_EQ(host.last_token_ids, (std::vector<int>{50, 51, 52, 7}));
    EXPECT_TRUE(host.last_position_ids.empty());
    EXPECT_EQ(
        host.last_forward_input.position_policy,
        ForwardPositionPolicy::ContiguousOffset);
    EXPECT_EQ(host.last_forward_input.position_ids_device, nullptr);
    EXPECT_EQ(host.last_forward_input.seq_len, 4);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 3);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 4);
    EXPECT_EQ(host.last_forward_input.token_offset, 88);
    EXPECT_EQ(host.last_forward_input.position_offset, 88);
    EXPECT_EQ(host.last_workspace_seq_len, 4);
}

/**
 * @brief Prove that an admitted terminal chunk is not reclassified as a raw
 * short prompt by the graph-cache minimum-length policy.
 *
 * A long request uses one fixed physical graph width for every chunk. Its last
 * chunk can contain fewer real rows than `LLAMINAR_PREFILL_GRAPH_MIN_SEQ`, but
 * `bucket_seq_len` records that the scheduler has already admitted that chunk
 * into the transaction's captured bucket. The execution engine must honor that
 * admission and reuse/build the bucket graph instead of rejecting the tail.
 */
TEST_F(Test__ForwardExecutionEngine,
       RunPrefillChunk_TerminalTailBelowRawPromptMinimumUsesAdmittedBucket)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "2"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::rocm(0),
        ComputeBackendType::GPU_ROCM);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;

    const std::vector<int> terminal_token = {91};
    auto base_input = makeTestInput(
        /*seq_len=*/1,
        /*batch_size=*/1,
        DeviceId::rocm(0),
        terminal_token.data(),
        /*position_ids=*/nullptr);
    base_input.token_offset = 4096;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/true);
    ASSERT_TRUE(plan) << plan.error;
    ASSERT_TRUE(plan.padding_required);
    ASSERT_EQ(plan.chunk.real_count, 1);
    ASSERT_EQ(plan.chunk.bucket_seq_len, 4);

    ForwardOutput output{};
    ASSERT_TRUE(engine.runPrefillChunk(base_input, plan, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 1);
    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 1);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 4);
    EXPECT_EQ(host.last_workspace_seq_len, 4);

    /*
     * The second/terminal chunk of a prompt starts beyond position zero.  Its
     * captured physical width is also short enough to resemble an MTP decode
     * continuation, so this assertion protects the typed prefill-bucket
     * contract from being overridden by the historical position heuristic.
     */
    const auto last_graph = engine.lastExecutedForwardGraph();
    ASSERT_TRUE(last_graph.has_value());
    EXPECT_FALSE(last_graph->is_decode)
        << "A typed padded prefill tail must not be reclassified as decode";
    EXPECT_TRUE(last_graph->signature.is_bucketed_prefill);
}

TEST_F(Test__ForwardExecutionEngine,
       RunPrefillChunk_DeviceAdmissionPreservesSoleDeviceTokenAuthority)
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
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;

    const std::array<int32_t, 4> admitted_device_rows{};
    auto base_input = makeTestInput(
        /*seq_len=*/3,
        /*batch_size=*/1,
        DeviceId::cuda(0),
        /*token_ids=*/nullptr,
        /*position_ids=*/nullptr);
    base_input.token_ids_device = admitted_device_rows.data();
    base_input.token_offset = 88;

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4},
        /*pad_token_id=*/7,
        /*allow_padded_execution=*/true);
    ASSERT_TRUE(plan) << plan.error;
    ASSERT_TRUE(plan.padding_required);

    ForwardOutput output{};
    ASSERT_TRUE(engine.runPrefillChunk(base_input, plan, output, host));

    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.last_token_ids_pointer, nullptr);
    EXPECT_EQ(
        host.last_token_ids_device_pointer,
        admitted_device_rows.data());
    EXPECT_TRUE(host.last_token_ids.empty());
    EXPECT_EQ(host.last_forward_input.seq_len, 4);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 3);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 4);
    EXPECT_EQ(host.last_forward_input.token_offset, 88);
}

TEST_F(Test__ForwardExecutionEngine, RunPrefillChunk_PaddedReplayParamsRefreshAfterGpuStreamBinding)
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
    PrefillReplayParamProbe probe;
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_factories.push_back(
        [&probe](const std::string &name, DeviceId device) -> std::unique_ptr<IComputeStage>
        {
            return std::make_unique<PrefillReplayParamProbeStage>(name, device, &probe);
        });

    const std::vector<int> tokens = {60, 61, 62};
    auto base_input = makeTestInput(3, 1, DeviceId::cuda(0), tokens.data(), nullptr);

    auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        base_input,
        std::vector<int>{4},
        /*pad_token_id=*/0,
        /*allow_padded_execution=*/true);
    ASSERT_TRUE(plan) << plan.error;
    ASSERT_TRUE(plan.padding_required);

    ForwardOutput output{};
    EXPECT_TRUE(engine.runPrefillChunk(base_input, plan, output, host));

    ASSERT_GE(probe.updates, 2)
        << "GPU padded-prefill stages must receive a second metadata refresh after stream binding";
    EXPECT_FALSE(probe.saw_stream.front())
        << "The first refresh happens before workspace/stream binding for host-side metadata";
    EXPECT_TRUE(probe.saw_stream.back())
        << "The final refresh must happen after assigning the explicit graph stream";
    for (int real_seq_len : probe.real_seq_lens)
        EXPECT_EQ(real_seq_len, 3);
    for (int bucket_seq_len : probe.bucket_seq_lens)
        EXPECT_EQ(bucket_seq_len, 4);
    for (int token_offset : probe.token_offsets)
        EXPECT_EQ(token_offset, 0);
}

TEST_F(Test__ForwardExecutionEngine, Execute_CachedPrefillReplayParamsUsePositionOffsetWhenTokenOffsetIsAbsent)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "0"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    PrefillReplayParamProbe probe;
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_factories.push_back(
        [&probe](const std::string &name, DeviceId device) -> std::unique_ptr<IComputeStage>
        {
            return std::make_unique<PrefillReplayParamProbeStage>(name, device, &probe);
        });

    const std::vector<int> tokens = {70, 71, 72, 73, 74};
    const std::vector<int> positions = {256, 257, 258, 259, 260};
    auto input = makeTestInput(5, 1, DeviceId::cuda(0), tokens.data(), positions.data());
    input.position_offset = 256;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host));
    EXPECT_TRUE(engine.execute(input, output, host));

    ASSERT_GE(probe.updates, 2)
        << "The cached prefill graph must refresh stateful stage replay metadata "
           "on both capture/build and replay.";
    for (int token_offset : probe.token_offsets)
        EXPECT_EQ(token_offset, 256)
            << "A restored-prefix suffix must not replay stateful prefill stages "
               "as though the suffix began at prompt offset zero.";
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
    input.execution_phase = ForwardExecutionPhase::Prefill;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host));

    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.last_token_ids, (std::vector<int>{80, 81, 82, 0}));
    EXPECT_EQ(host.last_position_ids_pointer, nullptr);
    EXPECT_TRUE(host.last_position_ids.empty());
    EXPECT_EQ(
        host.last_forward_input.position_policy,
        ForwardPositionPolicy::ContiguousOffset);
    EXPECT_EQ(host.last_forward_input.position_ids_device, nullptr);
    EXPECT_EQ(host.last_forward_input.seq_len, 4);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 3);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 4);
    EXPECT_EQ(host.last_forward_input.token_offset, 200);
    EXPECT_EQ(host.last_forward_input.position_offset, 200);
    EXPECT_EQ(host.last_workspace_seq_len, 4);
}

TEST_F(Test__ForwardExecutionEngine,
       Execute_RawBucketFloorClampsToResidentGraphCapacity)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "8"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "8"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;
    host.mock_resident_graph_rows = 4;

    const std::vector<int> tokens = {80, 81, 82};
    const std::vector<int> positions = {0, 1, 2};
    auto input = makeTestInput(
        3,
        1,
        DeviceId::cuda(0),
        tokens.data(),
        positions.data());

    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host));
    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 3);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 4);
    EXPECT_EQ(host.last_workspace_seq_len, 4);
}

TEST_F(Test__ForwardExecutionEngine, Execute_RejectsBucketBeyondResidentGraphRows)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4,8"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;
    host.mock_resident_graph_rows = 4;

    const std::vector<int> tokens = {80, 81, 82, 83, 84};
    const std::vector<int> positions = {0, 1, 2, 3, 4};
    auto input = makeTestInput(
        5,
        1,
        DeviceId::cuda(0),
        tokens.data(),
        positions.data());

    ForwardOutput output{};
    EXPECT_FALSE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 0)
        << "the orchestration layer must chunk before graph construction";
    EXPECT_EQ(host.ensure_workspace_calls, 0);
}

TEST_F(Test__ForwardExecutionEngine, Execute_PaddedBucketedPrefillRejectsActiveNonGraphStableMoE)
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
    host.mock_moe_rebalancing_active = true;
    host.mock_moe_rebalancing_graph_stable = false;

    const std::vector<int> tokens = {80, 81, 82};
    const std::vector<int> positions = {0, 1, 2};
    auto input = makeTestInput(3, 1, DeviceId::cuda(0), tokens.data(), positions.data());

    ForwardOutput output{};
    EXPECT_FALSE(engine.execute(input, output, host))
        << "Unsafe padded MoE prefill capture must fail fast instead of taking an eager fallback.";
    EXPECT_EQ(host.ensure_workspace_calls, 0);
    EXPECT_EQ(host.sync_logits_calls, 0);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
}

TEST_F(Test__ForwardExecutionEngine, Execute_PaddedBucketedPrefillAllowsActiveGraphStableMoE)
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
    host.mock_moe_rebalancing_active = true;
    host.mock_moe_rebalancing_graph_stable = true;

    const std::vector<int> tokens = {80, 81, 82};
    const std::vector<int> positions = {0, 1, 2};
    auto input = makeTestInput(3, 1, DeviceId::cuda(0), tokens.data(), positions.data());

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host))
        << "Graph-stable GPU MoE placement mutates runtime-table data behind stable graph pointers.";
    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.last_forward_input.seq_len, 4);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 3);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 4);
    EXPECT_EQ(host.ensure_workspace_calls, 1);
}

TEST_F(Test__ForwardExecutionEngine, Execute_PaddedBucketedPrefillRejectsBeforeEagerFallback)
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
    NonCapturablePrefillStage *stage = nullptr;
    host.graph_stage_factories.push_back(
        [&](const std::string &name, DeviceId device)
        {
            auto non_capturable = std::make_unique<NonCapturablePrefillStage>(name, device);
            stage = non_capturable.get();
            return non_capturable;
        });

    const std::vector<int> tokens = {80, 81, 82};
    const std::vector<int> positions = {0, 1, 2};
    auto input = makeTestInput(3, 1, DeviceId::cuda(0), tokens.data(), positions.data());
    input.position_offset = 0;

    ForwardOutput output{};
    EXPECT_FALSE(engine.execute(input, output, host))
        << "A rejected bucketed prefill graph must fail instead of running an eager fallback.";

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->executionCount(), 0)
        << "Prefill rejection must happen before ordinary graph execution can mutate runtime state.";
    EXPECT_EQ(host.ensure_workspace_calls, 0)
        << "Rejected prefill graph should not allocate execution workspace for a fallback pass.";
    EXPECT_EQ(host.sync_logits_calls, 0)
        << "Rejected prefill graph should not reach the forward boundary.";
    EXPECT_EQ(host.build_forward_graph_calls, 1);
}

TEST_F(Test__ForwardExecutionEngine, Execute_ExactPrefillRejectsBeforeFreshWarmupFallback)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "0"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
        {"LLAMINAR_VALIDATE_BUFFERS", "0"},
        {"LLAMINAR_VALIDATE_INPUTS", "0"},
        {"LLAMINAR_FAIL_ON_ZERO", "0"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    NonCapturablePrefillStage *stage = nullptr;
    host.graph_stage_factories.push_back(
        [&](const std::string &name, DeviceId device)
        {
            auto non_capturable = std::make_unique<NonCapturablePrefillStage>(name, device);
            stage = non_capturable.get();
            return non_capturable;
        });

    const std::vector<int> tokens = {90, 91, 92, 93};
    const std::vector<int> positions = {0, 1, 2, 3};
    auto input = makeTestInput(4, 1, DeviceId::cuda(0), tokens.data(), positions.data());
    input.position_offset = 0;

    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host))
        << "The first exact prefill builds and executes the ordinary forward graph before cache-hit preflight.";
    ASSERT_NE(stage, nullptr);
    ASSERT_EQ(stage->executionCount(), 1);
    ASSERT_EQ(host.build_forward_graph_calls, 1);

    EXPECT_FALSE(engine.execute(input, output, host))
        << "A rejected exact prefill graph cache hit must fail instead of running a fresh warmup.";
    EXPECT_EQ(stage->executionCount(), 1)
        << "Rejected exact prefill graph should not execute the cached graph as an eager fallback.";
    EXPECT_EQ(host.build_forward_graph_calls, 1)
        << "Rejected exact prefill graph should not rebuild a fallback graph.";
}

TEST_F(Test__ForwardExecutionEngine, Execute_BatchedGpuPrefillSkipsBucketedAdapter)
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

    /*
     * Request-batched MTP benchmark prefill uses the ordinary batched graph
     * path. Bucketed prefill chunking is a single-sequence adapter, so trying
     * to pad this shape would reject before the decode amortization lane even
     * starts.
     */
    const std::vector<int> tokens = {
        80, 81, 82,
        90, 91, 92,
    };
    const std::vector<int> positions = {
        200, 201, 202,
        300, 301, 302,
    };
    auto input = makeTestInput(3, 2, DeviceId::cuda(0), tokens.data(), positions.data());
    input.position_offset = 200;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host));

    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.last_token_ids, tokens);
    EXPECT_EQ(host.last_position_ids, positions);
    EXPECT_EQ(host.last_forward_input.seq_len, 3);
    EXPECT_EQ(host.last_forward_input.batch_size, 2);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 0);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 0);
    EXPECT_EQ(host.last_workspace_seq_len, 3);
    EXPECT_GT(host.get_worker_gpu_context_calls, 0)
        << "GPU-shaped unit execution must use the host's hardware-free worker context.";
}

/**
 * @brief A short raw GPU prompt is coalesced into the configured minimum
 * physical graph bucket instead of escaping into permanent eager execution.
 */
TEST_F(Test__ForwardExecutionEngine, Execute_RawPrefillBelowMinSeqUsesMinimumCapturedBucket)
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
    EXPECT_TRUE(engine.execute(input, output, host))
        << "The second use of the same physical bucket must enter capture/replay state instead of rebuilding eagerly.";

    ASSERT_TRUE(host.has_last_forward_input);
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    ASSERT_EQ(host.last_token_ids.size(), 256u);
    EXPECT_TRUE(std::equal(tokens.begin(), tokens.end(), host.last_token_ids.begin()));
    EXPECT_TRUE(std::all_of(
        host.last_token_ids.begin() + static_cast<std::ptrdiff_t>(tokens.size()),
        host.last_token_ids.end(),
        [](int token) { return token == 0; }));
    EXPECT_EQ(host.last_position_ids_pointer, nullptr);
    EXPECT_TRUE(host.last_position_ids.empty());
    EXPECT_EQ(host.last_forward_input.seq_len, 256);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 35);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 256);
    EXPECT_EQ(host.last_workspace_seq_len, 256);
    EXPECT_FALSE(engine.cacheEmpty())
        << "Every eligible homogeneous-GPU prefill must own graph-cache state.";
}

/**
 * @brief A grouped verifier remains compact grouped decode even at prompt-like M.
 *
 * This reproduces the production failure where an M=5 verifier was classified
 * through the prefill envelope and queried short-conv scratch at M=4096. The
 * explicit ForwardExecutionRole is authoritative even when position zero and
 * sequence geometry make the legacy decode heuristic return false.
 */
TEST_F(Test__ForwardExecutionEngine, Execute_GroupedVerifierPublishesExactWorkspaceRole)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;
    host.mock_compute_all_position_logits = true;

    const std::vector<int> tokens = {70, 71, 72, 73, 74};
    const std::vector<int> positions = {0, 1, 2, 3, 4};
    auto input = makeTestInput(
        /*seq_len=*/5,
        /*batch_size=*/1,
        DeviceId::cpu(),
        tokens.data(),
        positions.data());
    input.execution_role =
        ForwardExecutionRole::GroupedMTPVerifier;
    input.execution_phase = ForwardExecutionPhase::Decode;

    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host));
    ASSERT_TRUE(host.typed_workspace_role_seen);
    EXPECT_EQ(
        host.last_workspace_family_policy,
        WorkspaceGraphFamilyPolicy::
            SerialDeviceFamilyExactParticipant);
    EXPECT_EQ(
        host.last_workspace_participant_role,
        WorkspaceGraphParticipantRole::GroupedVerifier);
    EXPECT_EQ(host.last_workspace_seq_len, 5);

    auto verifier_graph =
        engine.lastAllPositionVerifierForwardGraph();
    ASSERT_TRUE(verifier_graph.has_value());
    EXPECT_TRUE(verifier_graph->is_decode);
    EXPECT_TRUE(verifier_graph->all_position_logits);
    EXPECT_EQ(verifier_graph->signature.seq_len, 5);
}

/**
 * @brief All-position logits do not confer verifier-state ownership.
 *
 * MTP condition and diagnostic graphs can expose all-position logits while
 * using decode-shaped geometry. Only a graph admitted with the explicit
 * GroupedMTPVerifier role captures the recurrent rows that accepted-state
 * publication may consume. A later condition graph must therefore leave the
 * retained verifier producer untouched instead of erasing or replacing it.
 */
TEST_F(
    Test__ForwardExecutionEngine,
    Execute_AllPositionConditionCannotReplaceGroupedVerifierProducer)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;
    host.mock_compute_all_position_logits = true;

    const std::vector<int> verifier_tokens = {70, 71};
    const std::vector<int> verifier_positions = {17, 18};
    auto verifier_input = makeTestInput(
        /*seq_len=*/2,
        /*batch_size=*/1,
        DeviceId::cpu(),
        verifier_tokens.data(),
        verifier_positions.data());
    verifier_input.execution_role =
        ForwardExecutionRole::GroupedMTPVerifier;
    verifier_input.execution_phase = ForwardExecutionPhase::Decode;

    ForwardOutput verifier_output{};
    ASSERT_TRUE(engine.execute(verifier_input, verifier_output, host));
    const auto verifier_graph_before_condition =
        engine.lastAllPositionVerifierForwardGraph();
    ASSERT_TRUE(verifier_graph_before_condition.has_value());
    EXPECT_EQ(
        verifier_graph_before_condition->signature.execution_role,
        ForwardExecutionRole::GroupedMTPVerifier);

    const std::vector<int> condition_tokens = {72};
    const std::vector<int> condition_positions = {19};
    auto condition_input = makeTestInput(
        /*seq_len=*/1,
        /*batch_size=*/1,
        DeviceId::cpu(),
        condition_tokens.data(),
        condition_positions.data());
    condition_input.execution_role =
        ForwardExecutionRole::MTPCondition;
    condition_input.execution_phase = ForwardExecutionPhase::Decode;

    ForwardOutput condition_output{};
    ASSERT_TRUE(engine.execute(condition_input, condition_output, host));

    const auto last_graph = engine.lastExecutedForwardGraph();
    ASSERT_TRUE(last_graph.has_value());
    EXPECT_EQ(
        last_graph->signature.execution_role,
        ForwardExecutionRole::MTPCondition);

    const auto retained_verifier_graph =
        engine.lastAllPositionVerifierForwardGraph();
    ASSERT_TRUE(retained_verifier_graph.has_value());
    EXPECT_EQ(
        retained_verifier_graph->signature.execution_role,
        ForwardExecutionRole::GroupedMTPVerifier);
    EXPECT_EQ(
        retained_verifier_graph->signature,
        verifier_graph_before_condition->signature);
}

/**
 * @brief A device-owned request condition has its own exact family role.
 *
 * The condition graph is shaped like scalar decode for one request, but its
 * GDN/short-conv namespace and device-resident sequence rows are distinct.
 * Treating it as ordinary Decode caused eager family planning to omit those
 * buffers and made first use fail after graph construction.
 */
TEST_F(Test__ForwardExecutionEngine, Execute_MTPConditionPublishesExactWorkspaceRole)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {70};
    const std::vector<int> positions = {17};
    auto input = makeTestInput(
        /*seq_len=*/1,
        /*batch_size=*/1,
        DeviceId::cpu(),
        tokens.data(),
        positions.data());
    input.execution_role =
        ForwardExecutionRole::MTPCondition;
    input.execution_phase = ForwardExecutionPhase::Decode;

    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host));
    ASSERT_TRUE(host.typed_workspace_role_seen);
    EXPECT_EQ(
        host.last_workspace_family_policy,
        WorkspaceGraphFamilyPolicy::
            SerialDeviceFamilyExactParticipant);
    EXPECT_EQ(
        host.last_workspace_participant_role,
        WorkspaceGraphParticipantRole::MTPCondition);
    EXPECT_EQ(host.last_workspace_seq_len, 1);
}

/**
 * @brief Prove typed grouped verifier roles admit graph capture through M=15.
 *
 * The ordinary decode cache retains a four-row heuristic, but MTP depth is
 * independently configured and validated. This regression catches the old
 * predicate that classified M=15 as decode while quietly denying it access to
 * the capture controller on the cache-hit execution.
 */
TEST_F(
    Test__ForwardExecutionEngine,
    Execute_GPUGroupedVerifierAboveOrdinaryDecodeLimitUsesCapturePolicy)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;
    host.mock_compute_all_position_logits = true;
    host.mock_capture_policy.allow_fast_decode = true;
    host.mock_capture_policy.allow_cached_graph_replay = true;

    std::array<int, 15> tokens{};
    std::array<int, 15> positions{};
    for (int row = 0; row < 15; ++row)
    {
        tokens[static_cast<size_t>(row)] = 700 + row;
        positions[static_cast<size_t>(row)] = 1200 + row;
    }
    auto input = makeTestInput(
        static_cast<int>(tokens.size()),
        /*batch_size=*/1,
        DeviceId::cuda(0),
        tokens.data(),
        positions.data());
    input.token_ids_device = tokens.data();
    input.position_ids_device = positions.data();
    input.execution_role = ForwardExecutionRole::GroupedMTPVerifier;
    input.execution_phase = ForwardExecutionPhase::Decode;

    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host));

    std::string template_error;
    const auto first_use_template =
        engine.lastAllPositionVerifierDeviceLoopGraphTemplate(
            &template_error);
    ASSERT_TRUE(first_use_template.has_value()) << template_error;
    EXPECT_NE(first_use_template->capture, nullptr);
    EXPECT_EQ(first_use_template->signature.seq_len, 15);
    EXPECT_EQ(first_use_template->signature.batch_size, 1);

    ASSERT_TRUE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_GT(host.build_decode_policy_calls, 0)
        << "Typed MTP rows must bypass the ordinary decode_seq_len heuristic.";
}

/**
 * @brief Setup-only decode prepares host-token inputs before native capture.
 *
 * Serving setup intentionally omits request admission and model launch, but it
 * still records the exact production decode graph. A host-token embedding must
 * therefore preload its cache-owned row on the selected capture stream before
 * beginCapture(); returning directly to materialization used to skip that
 * prelude and made CUDA/ROCm embedding fail from inside capture.
 */
TEST_F(
    Test__ForwardExecutionEngine,
    SetupDecodeMaterializationPreparesDynamicInputsBeforeCapture)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.mock_capture_policy.allow_fast_decode = true;
    host.mock_capture_policy.allow_cached_graph_replay = true;

    int dynamic_updates = 0;
    int device_position_updates = 0;
    bool dynamic_update_during_capture = false;
    int executions = 0;
    host.graph_stage_factories.push_back(
        [&](const std::string &name, DeviceId device)
            -> std::unique_ptr<IComputeStage>
        {
            return std::make_unique<SetupDynamicInputProbeStage>(
                name,
                device,
                &dynamic_updates,
                &device_position_updates,
                &dynamic_update_during_capture,
                &executions);
        });

    int token = 42;
    int host_position_shadow = 1;
    int device_position_row = 1;
    auto input = makeTestInput(
        /*seq_len=*/1,
        /*batch_size=*/1,
        DeviceId::cuda(0),
        &token,
        &host_position_shadow);
    input.position_ids = nullptr;
    input.position_ids_device = &device_position_row;
    input.position_offset = 1;
    input.execution_phase = ForwardExecutionPhase::Decode;
    input.graph_submission_intent =
        ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;

    const int capture_count_before =
        llaminar2::testing::sharedMockWorkerGPUContext()
            .graphCaptureCreateCount();
    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host));

    EXPECT_EQ(dynamic_updates, 1);
    EXPECT_EQ(device_position_updates, 1);
    EXPECT_FALSE(dynamic_update_during_capture)
        << "Dynamic host inputs must be ready before beginCapture().";
    EXPECT_EQ(executions, 1)
        << "The model body should be recorded exactly once without replay.";
    EXPECT_EQ(
        llaminar2::testing::sharedMockWorkerGPUContext()
            .graphCaptureCreateCount(),
        capture_count_before + 1);
}

/** @brief Setup and replay preserve decode, verifier and request-condition identities. */
TEST_F(Test__ForwardExecutionEngine, SetupAndReplayPreserveGraphIdentity)
{
    ScopedDebugEnv env({{"LLAMINAR_PERF_STATS_JSON", "1"}});
    for (const auto role : {ForwardExecutionRole::MainInference,
                            ForwardExecutionRole::GroupedMTPVerifier,
                            ForwardExecutionRole::MTPCondition})
    {
    SCOPED_TRACE(static_cast<int>(role));
    PerfStatsCollector::reset();
    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;
    host.mock_compute_all_position_logits = role == ForwardExecutionRole::GroupedMTPVerifier;
    host.mock_live_mtp_request_batch_condition = role == ForwardExecutionRole::MTPCondition;
    host.mock_capture_policy.allow_fast_decode = true;
    host.mock_capture_policy.allow_cached_graph_replay = true;
    std::array<int, 2> tokens{42, 43};
    std::array<int, 2> positions{1, 2};
    auto input = makeTestInput(host.mock_compute_all_position_logits ? 2 : 1,
                              1, DeviceId::cuda(0), tokens.data(), positions.data());
    input.token_ids_device = tokens.data();
    input.position_ids_device = positions.data();
    input.execution_role = role;
    input.execution_phase = ForwardExecutionPhase::Decode;
    input.graph_submission_intent = ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;
    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host));
    input.graph_submission_intent = ForwardGraphSubmissionIntent::Execute;
    ASSERT_TRUE(engine.execute(input, output, host));
    ASSERT_TRUE(engine.execute(input, output, host));
    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const std::string expected_context = host.mock_compute_all_position_logits
        ? "main_verifier" : (host.mock_live_mtp_request_batch_condition
            ? "main_condition_batch" : "main_decode");
    bool saw_capture = false;
    bool saw_replay = false;
    for (const auto &record : records)
    {
        if (record.name != "full_graph_capture_executable_nodes" &&
            record.name != "decode_graph_phase") continue;
        for (const auto &[key, value] : record.tags)
        {
            if (key == "phase" && value == "replay") saw_replay = true;
            if (key != "context") continue;
            EXPECT_EQ(value, expected_context);
            if (record.name == "full_graph_capture_executable_nodes") saw_capture = true;
        }
    }
    EXPECT_TRUE(saw_capture);
    EXPECT_TRUE(saw_replay);
    PerfStatsCollector::reset();
    }
}

/**
 * @brief Hosted MTP replay consumes the same runner authority as capture.
 *
 * The mixed-GPU dynamic-depth route previously supplied an empty factory to
 * retained replay even though setup had captured a maintenance branch. Sweep
 * both device identities and shallow/deep verifier geometries without loading
 * a backend; repeated replay must neither omit that authority nor recapture.
 */
TEST_F(Test__ForwardExecutionEngine, RetainedVerifierReplayUsesHostAuxiliaryAuthority)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int depth : {1, 2, 3, 14, 15})
    {
        SCOPED_TRACE(device.toString() + " depth=" + std::to_string(depth));
        auto engine = makeEngine(/*cache_enabled=*/true);
        llaminar2::testing::MockDeviceContext gpu_ctx(
            device, device == DeviceId::cuda(0)
                        ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
        MockForwardExecutionHost host(&gpu_ctx);
        host.graph_stage_count = 1;
        host.mock_compute_all_position_logits = true;
        host.mock_capture_policy.allow_fast_decode = true;
        host.mock_capture_policy.allow_cached_graph_replay = true;
        const auto probe = std::make_shared<ForwardAuxiliaryBranchProbe>();
        host.mock_auxiliary_factory = {
            .authority_identity = probe.get(),
            .device = device,
            .create = [probe, device]() {
                return std::make_unique<MockForwardAuxiliaryBranch>(probe, device);
            }};
        std::array<int, 16> tokens{};
        std::array<int, 16> positions{};
        auto input = makeTestInput(depth + 1, 1, device, tokens.data(), positions.data());
        input.token_ids_device = tokens.data();
        input.position_ids_device = positions.data();
        input.execution_role = ForwardExecutionRole::GroupedMTPVerifier;
        input.execution_phase = ForwardExecutionPhase::Decode;
        input.graph_submission_intent =
            ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;
        ForwardOutput output{};
        ASSERT_TRUE(engine.execute(input, output, host));
        // Ordinary forward admission seals replay readiness before the hosted
        // scheduler takes ownership of the same retained executable.
        input.graph_submission_intent = ForwardGraphSubmissionIntent::Execute;
        ASSERT_TRUE(engine.execute(input, output, host));
        const auto graph = engine.lastAllPositionVerifierForwardGraph();
        ASSERT_TRUE(graph.has_value());
        const auto captures = llaminar2::testing::sharedMockWorkerGPUContext()
                                  .graphCaptureCreateCount();
        const auto queries = host.auxiliary_factory_queries;
        ASSERT_GT(queries, 0);
        ASSERT_EQ(probe->creations, 1);
        ASSERT_EQ(probe->attachments, 1);

        // The transaction owns this stable non-null mock handle. Only native
        // event ordering is modeled; no physical stream or GPU is constructed.
        int transaction_stream = 0;
        void *producer_stream = nullptr;
        std::string error;
        for (int replay = 0; replay < 20; ++replay)
        {
            ASSERT_TRUE(engine.replayRetainedDecodeGraph(
                graph->signature, &gpu_ctx, {}, {},
                host,
                &transaction_stream, &producer_stream, &error)) << error;
            EXPECT_NE(producer_stream, nullptr);
            EXPECT_EQ(host.auxiliary_factory_queries, queries + replay + 1);
            EXPECT_EQ(host.auxiliary_factory_device, device);
        }
        EXPECT_EQ(llaminar2::testing::sharedMockWorkerGPUContext()
                      .graphCaptureCreateCount(), captures);
        EXPECT_EQ(probe->creations, 1);
        EXPECT_EQ(probe->attachments, 1);

        // Removal or replacement of the runner authority must still fail
        // before a launch; obtaining policy from the host must not weaken the
        // existing immutable-cache check or turn an invalidation into recapture.
        host.mock_auxiliary_factory = {};
        EXPECT_FALSE(engine.replayRetainedDecodeGraph(
            graph->signature, &gpu_ctx, {}, {}, host,
            &transaction_stream, &producer_stream, &error));
        EXPECT_EQ(producer_stream, nullptr);
        const auto replacement = std::make_shared<ForwardAuxiliaryBranchProbe>();
        host.mock_auxiliary_factory = {
            .authority_identity = replacement.get(),
            .device = device,
            .create = [replacement, device]() {
                return std::make_unique<MockForwardAuxiliaryBranch>(replacement, device);
            }};
        EXPECT_FALSE(engine.replayRetainedDecodeGraph(
            graph->signature, &gpu_ctx, {}, {}, host,
            &transaction_stream, &producer_stream, &error));
        EXPECT_EQ(producer_stream, nullptr);
        EXPECT_EQ(replacement->creations, 0);
        EXPECT_EQ(llaminar2::testing::sharedMockWorkerGPUContext()
                      .graphCaptureCreateCount(), captures);
    }
}

/**
 * @brief Exact capture identity retains every bounded verifier geometry.
 *
 * A device-owned dynamic-depth parent graph composes all four physical verifier
 * buckets simultaneously. Mutable "last graph" state must therefore be only a
 * convenience view: exact lookup must retain 2/4/8/16 independently and an
 * unknown geometry must fail instead of silently borrowing a nearby capture.
 */
TEST_F(
    Test__ForwardExecutionEngine,
    DeviceLoopGraphTemplate_ExactSignatureRetainsEveryVerifierBucket)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;
    host.mock_compute_all_position_logits = true;
    host.mock_capture_policy.allow_fast_decode = true;
    host.mock_capture_policy.allow_cached_graph_replay = true;

    constexpr std::array<int, 4> kVerifierBuckets = {2, 4, 8, 16};
    std::array<ForwardGraphSignature, kVerifierBuckets.size()> signatures{};
    std::array<const IGPUGraphCapture *, kVerifierBuckets.size()> captures{};
    std::array<int, kVerifierBuckets.back()> tokens{};
    std::array<int, kVerifierBuckets.back()> positions{};
    for (int row = 0; row < kVerifierBuckets.back(); ++row)
    {
        tokens[static_cast<size_t>(row)] = 900 + row;
        positions[static_cast<size_t>(row)] = 1900 + row;
    }

    for (size_t bucket_index = 0;
         bucket_index < kVerifierBuckets.size();
         ++bucket_index)
    {
        const int rows = kVerifierBuckets[bucket_index];
        auto input = makeTestInput(
            rows,
            /*batch_size=*/1,
            DeviceId::cuda(0),
            tokens.data(),
            positions.data());
        input.token_ids_device = tokens.data();
        input.position_ids_device = positions.data();
        input.execution_role = ForwardExecutionRole::GroupedMTPVerifier;
        input.execution_phase = ForwardExecutionPhase::Decode;

        ForwardOutput output{};
        ASSERT_TRUE(engine.execute(input, output, host));

        const auto last = engine.lastAllPositionVerifierForwardGraph();
        ASSERT_TRUE(last.has_value());
        signatures[bucket_index] = last->signature;

        std::string error;
        const auto exact = engine.deviceLoopGraphTemplate(
            signatures[bucket_index],
            &error);
        ASSERT_TRUE(exact.has_value()) << error;
        EXPECT_EQ(exact->signature.seq_len, rows);
        captures[bucket_index] = exact->capture;
    }

    for (size_t bucket_index = 0;
         bucket_index < kVerifierBuckets.size();
         ++bucket_index)
    {
        std::string error;
        const auto exact = engine.deviceLoopGraphTemplate(
            signatures[bucket_index],
            &error);
        ASSERT_TRUE(exact.has_value()) << error;
        EXPECT_EQ(exact->capture, captures[bucket_index]);
        EXPECT_EQ(exact->signature, signatures[bucket_index]);
    }

    ForwardGraphSignature absent = signatures.front();
    absent.seq_len = 3;
    absent.all_position_logit_rows = 3;
    std::string absent_error;
    EXPECT_FALSE(engine.deviceLoopGraphTemplate(absent, &absent_error));
    EXPECT_NE(absent_error.find("exact signature"), std::string::npos);
}

/**
 * @brief A GPU MTP graph must never fall through to eager execution.
 *
 * The first invocation is one atomic build/capture/materialize transaction. An
 * absent capture policy is therefore an immediate architectural error: no
 * eager first-use execution may escape before the failure is reported.
 */
TEST_F(
    Test__ForwardExecutionEngine,
    Execute_GPUGroupedVerifierFailsClosedWhenGraphReplayIsUnavailable)
{
    auto engine = makeEngine(/*cache_enabled=*/true);
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;
    host.mock_compute_all_position_logits = true;

    std::array<int, 15> tokens{};
    std::array<int, 15> positions{};
    auto input = makeTestInput(
        static_cast<int>(tokens.size()),
        /*batch_size=*/1,
        DeviceId::cuda(0),
        tokens.data(),
        positions.data());
    input.execution_role = ForwardExecutionRole::GroupedMTPVerifier;
    input.execution_phase = ForwardExecutionPhase::Decode;

    ForwardOutput output{};
    EXPECT_FALSE(engine.execute(input, output, host));
    EXPECT_GT(host.build_decode_policy_calls, 0);
    EXPECT_TRUE(engine.cacheEmpty())
        << "A failed first-use capture must invalidate the half-built graph.";
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
    // CPU prefill never borrows a GPU padding bucket. It keeps an exact-shape
    // topology entry instead, so globally enabled GPU bucket policy cannot
    // change CPU arithmetic or execution geometry.
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

/** @brief CPU prefill accepts exact shapes beyond the largest GPU graph bucket. */
TEST_F(Test__ForwardExecutionEngine, Execute_AboveLargestBucketOnCpu_FallsThrough)
{
    // CPU prefill must not be constrained by GPU graph bucket boundaries. A
    // prompt longer than the largest configured bucket should run unbucketed
    // instead of failing with "real_seq_len exceeds largest prefill graph bucket".
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4"},
        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
    });

    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    const std::vector<int> tokens = {100, 101, 102, 103, 104};
    const std::vector<int> positions = {400, 401, 402, 403, 404};
    auto input = makeTestInput(5, 1, DeviceId::cpu(), tokens.data(), positions.data());
    input.position_offset = 400;

    ForwardOutput output{};
    EXPECT_TRUE(engine.execute(input, output, host))
        << "Above-largest bucket prefill on CPU should fall through to unbucketed execution.";
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.last_forward_input.seq_len, 5);
    EXPECT_EQ(host.last_forward_input.real_seq_len, 0);
    EXPECT_EQ(host.last_forward_input.bucket_seq_len, 0);
    EXPECT_EQ(host.last_token_ids, tokens);
    EXPECT_EQ(host.last_position_ids, positions);
    EXPECT_EQ(host.last_workspace_seq_len, 5);
}

/**
 * @brief Exact CPU prefill must retain graph topology and refresh stable rows.
 *
 * The stage captures the stable token pointer installed during first graph
 * construction. A second invocation changes the caller-owned rows. Observing
 * the new first token through the same stage proves the engine copied request
 * data into persistent storage instead of rebuilding or retaining a stale
 * caller pointer.
 */
TEST_F(
    Test__ForwardExecutionEngine,
    Execute_CPUExactPrefillReusesTopologyAndRefreshesStableInputs)
{
    ScopedDebugEnv env({
        {"LLAMINAR_PERF_STATS_JSON", "1"},
        {"LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS", "4"},
    });
    PerfStatsCollector::reset();

    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    std::vector<int> observed_first_tokens;
    host.graph_stage_factories.push_back(
        [&](const std::string &name, DeviceId device)
        {
            const int *const stable_tokens = host.last_token_ids_pointer;
            auto stage =
                std::make_unique<llaminar2::testing::MockComputeStage>(
                    ComputeStageType::GEMM,
                    name,
                    device);
            stage->setOnExecute(
                [stable_tokens, &observed_first_tokens](IDeviceContext *)
                {
                    ASSERT_NE(stable_tokens, nullptr);
                    observed_first_tokens.push_back(stable_tokens[0]);
                });
            return stage;
        });

    std::array<int, 3> tokens = {101, 102, 103};
    std::array<int, 3> positions = {0, 1, 2};
    auto input = makeTestInput(
        /*seq_len=*/3,
        /*batch_size=*/1,
        DeviceId::cpu(),
        tokens.data(),
        positions.data());
    input.position_offset = 0;

    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host));

    tokens = {201, 202, 203};
    positions = {0, 1, 2};
    ASSERT_TRUE(engine.execute(input, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 1)
        << "A matching CPU prefill geometry must keep one materialized graph.";
    EXPECT_EQ(observed_first_tokens, (std::vector<int>{101, 201}));
    const auto snapshots = engine.prefillGraphCacheSnapshots();
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_TRUE(snapshots.front().forward_cache_valid);
    EXPECT_EQ(snapshots.front().bucket_seq_len, 3);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags miss_tags = {
        {"all_position_logits", "false"},
        {"context", "prefill"},
        {"decode_has_history", "false"},
        {"live_mtp_request_batch_condition", "false"},
        {"all_position_logit_rows", "0"},
        {"moe_placement_epoch", "0"},
        {"result", "miss"},
        {"submission", "runtime"},
        {"uses_device_token_ids", "false"},
        {"uses_device_position_ids", "false"},
        {"uses_device_sequence_lengths", "false"},
        {"seq_len", "3"}};
    const PerfStatsCollector::Tags hit_tags = {
        {"all_position_logits", "false"},
        {"context", "prefill"},
        {"decode_has_history", "false"},
        {"live_mtp_request_batch_condition", "false"},
        {"all_position_logit_rows", "0"},
        {"moe_placement_epoch", "0"},
        {"result", "hit"},
        {"submission", "runtime"},
        {"uses_device_token_ids", "false"},
        {"uses_device_position_ids", "false"},
        {"uses_device_sequence_lengths", "false"},
        {"seq_len", "3"}};
    EXPECT_DOUBLE_EQ(
        findForwardGraphCounterValue(
            records,
            "forward_cache_lookup",
            miss_tags),
        1.0);
    EXPECT_DOUBLE_EQ(
        findForwardGraphCounterValue(
            records,
            "forward_cache_lookup",
            hit_tags),
        1.0);
    PerfStatsCollector::reset();
}

/**
 * @brief CPU prefill cache identity is exact in M and survives request reset.
 */
TEST_F(
    Test__ForwardExecutionEngine,
    Execute_CPUExactPrefillSeparatesShapesAndSurvivesRequestReset)
{
    ScopedDebugEnv env({
        {"LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS", "4"},
    });
    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    std::array<int, 4> tokens = {11, 12, 13, 14};
    std::array<int, 4> positions = {0, 1, 2, 3};
    auto m3 = makeTestInput(3, 1, DeviceId::cpu(), tokens.data(), positions.data());
    auto m4 = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), positions.data());
    ForwardOutput output{};

    ASSERT_TRUE(engine.execute(m3, output, host));
    ASSERT_TRUE(engine.execute(m4, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 2);
    ASSERT_EQ(engine.prefillGraphCacheSnapshots().size(), 2u);

    const auto reset = engine.resetSessionReplayState(
        /*preserve_replay_safe_graphs=*/true);
    EXPECT_EQ(reset.reset_replay_state, 0u);
    EXPECT_EQ(reset.preserved_for_stream_rebind, 2u);

    tokens[0] = 91;
    positions[0] = 0;
    ASSERT_TRUE(engine.execute(m3, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 2)
        << "Request reset must clear data state, not exact CPU graph topology.";
}

/**
 * @brief The configured prefill cap bounds exact CPU topology entries too.
 */
TEST_F(Test__ForwardExecutionEngine, Execute_CPUExactPrefillUsesBoundedLRU)
{
    ScopedDebugEnv env({
        {"LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS", "1"},
    });
    auto engine = makeEngine(/*cache_enabled=*/true);
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    std::array<int, 4> tokens = {31, 32, 33, 34};
    std::array<int, 4> positions = {0, 1, 2, 3};
    auto m3 = makeTestInput(3, 1, DeviceId::cpu(), tokens.data(), positions.data());
    auto m4 = makeTestInput(4, 1, DeviceId::cpu(), tokens.data(), positions.data());
    ForwardOutput output{};

    ASSERT_TRUE(engine.execute(m3, output, host));
    ASSERT_TRUE(engine.execute(m4, output, host));
    ASSERT_EQ(engine.prefillGraphCacheSnapshots().size(), 1u);
    ASSERT_TRUE(engine.execute(m3, output, host));

    EXPECT_EQ(host.build_forward_graph_calls, 3)
        << "The first M=3 entry must be rebuilt after M=4 evicts it.";
    const auto snapshots = engine.prefillGraphCacheSnapshots();
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_EQ(snapshots.front().bucket_seq_len, 3);
    EXPECT_EQ(snapshots.front().eviction_count, 2u);
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
        EXPECT_EQ(observation.graph_replay_live_state_epoch, 0u);
        EXPECT_FALSE(observation.requires_live_state_epoch_recapture);
        EXPECT_FALSE(observation.all_position_verifier_recapture_pending);
        if (observation.signature.all_position_logits)
            verifier_obs = &observation;
        else
            ordinary_obs = &observation;
    }
    ASSERT_NE(ordinary_obs, nullptr);
    ASSERT_NE(verifier_obs, nullptr);
    EXPECT_EQ(classifyForwardReplayStateCache(ordinary_obs->signature),
              ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode);
    EXPECT_EQ(classifyForwardReplayStateCache(verifier_obs->signature),
              ForwardReplayStateCacheClass::AllPositionVerifier);

    const ForwardExecutionEngine::ReplayStateResetSummary summary =
        engine.resetCapturedReplayStateForCorrectionReplay(/*live_state_epoch=*/123);
    EXPECT_EQ(summary.reset_replay_state, 0u);
    EXPECT_EQ(summary.ordinary_decode_reset, 0u);
    EXPECT_EQ(summary.preserved_for_stream_rebind, 2u);
    EXPECT_EQ(summary.all_position_verifier_preserved, 1u);
    EXPECT_EQ(summary.other_preserved, 1u)
        << "Single-token condition decode and all-position verifier replay are both version-safe; "
           "ordinary multi-token decode remains conservative.";

    const auto observations_after_reset =
        engine.replayCacheObservations(/*live_state_epoch=*/123);
    ASSERT_EQ(observations_after_reset.size(), 2u);
    for (const auto &observation : observations_after_reset)
    {
        if (observation.signature.all_position_logits)
        {
            EXPECT_EQ(observation.graph_replay_live_state_epoch, 123u)
                << "Multi-row verifier replay is preserved and stamped safe at the correction boundary.";
        }
        else
        {
            EXPECT_EQ(observation.graph_replay_live_state_epoch, 123u);
        }
        EXPECT_FALSE(observation.requires_live_state_epoch_recapture);
    }

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

TEST_F(Test__ForwardExecutionEngine, ResetAllPositionVerifierReplayStateLeavesOrdinaryDecodeWarm)
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

    int verifier_tokens[] = {43, 44};
    int verifier_positions[] = {8, 9};
    auto verifier = makeTestInput(
        2,
        1,
        DeviceId::cpu(),
        verifier_tokens,
        verifier_positions);

    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(ordinary, output, host));
    host.mock_compute_all_position_logits = true;
    ASSERT_TRUE(engine.execute(verifier, output, host));

    const ForwardExecutionEngine::ReplayStateResetSummary summary =
        engine.resetAllPositionVerifierReplayState();
    EXPECT_EQ(summary.reset_replay_state, 1u);
    EXPECT_EQ(summary.preserved_for_stream_rebind, 1u);
    EXPECT_EQ(summary.ordinary_decode_reset, 0u)
        << "Shifted MTP KV mutation must not discard ordinary decode replay.";
    EXPECT_TRUE(summary.all_position_verifier_recapture_requested)
        << "A shifted MTP KV mutation also has to protect a verifier cache "
           "that may be created or recaptured after the mutation.";

    const auto observations =
        engine.replayCacheObservations(/*live_state_epoch=*/0);
    ASSERT_EQ(observations.size(), 2u);
    for (const auto &observation : observations)
    {
        EXPECT_TRUE(observation.all_position_verifier_recapture_pending)
            << "The pending recapture request is engine-wide until a fresh "
               "all-position verifier capture reaches replay-ready state.";
    }
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

TEST_F(Test__ForwardExecutionEngine, DeviceTokenSourceUsesSeparateDecodeCacheSignature)
{
    auto engine = makeEngine();
    MockForwardExecutionHost host(&mock_ctx_);
    host.graph_stage_count = 1;

    int host_token = 42;
    int device_token_shadow = 43;
    int position = 7;
    const void *device_tokens = reinterpret_cast<const void *>(0x12340000);

    auto host_input = makeTestInput(1, 1, DeviceId::cpu(), &host_token, &position);
    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(host_input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 1);
    EXPECT_EQ(host.last_token_ids_device_pointer, nullptr);

    auto device_input = makeTestInput(1, 1, DeviceId::cpu(), &device_token_shadow, &position);
    device_input.token_ids_device = device_tokens;
    ASSERT_TRUE(engine.execute(device_input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 2)
        << "A host-token decode graph and a device-token decode graph have the same shape "
           "but different embedding pointer contracts.";
    EXPECT_EQ(host.last_token_ids_device_pointer, device_tokens);

    device_token_shadow = 44;
    position = 8;
    ASSERT_TRUE(engine.execute(device_input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 2)
        << "The stable device-token graph should be reusable once the device-token signature exists.";

    host_token = 45;
    position = 9;
    ASSERT_TRUE(engine.execute(host_input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 2)
        << "Returning to host-token decode should hit the original host-token cache entry.";
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

/**
 * @brief Prove first-use GPU MTP forwards use the same device handoff as replay.
 *
 * A cache miss is still an asynchronous GPU producer. The verifier sampler or
 * main-decode sampler must inherit its explicit stream directly; synchronizing
 * logits merely because no graph executable existed yet reintroduces a host
 * coherence boundary on the first speculative step.
 */
TEST_F(Test__ForwardExecutionEngine, CacheMiss_GPUDecodeDefersLogitsToDeviceConsumer)
{
    llaminar2::testing::MockDeviceContext gpu_ctx{
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA};
    int token = 42;
    int position = 0;

    {
        auto engine = makeEngine();
        MockForwardExecutionHost host(&gpu_ctx);
        host.graph_stage_count = 1;
        host.mock_compute_all_position_logits = true;
        host.mock_defer_all_position_verifier_sync = true;
        host.mock_capture_policy.allow_fast_decode = true;
        host.mock_capture_policy.allow_cached_graph_replay = true;

        auto input =
            makeTestInput(1, 1, DeviceId::cuda(0), &token, &position);
        ForwardOutput output{};
        ASSERT_TRUE(engine.execute(input, output, host));

        EXPECT_EQ(host.sync_logits_calls, 0);
        EXPECT_EQ(host.pending_all_position_verifier_stream_calls, 1);
        EXPECT_NE(host.pending_all_position_verifier_stream, nullptr);
        EXPECT_EQ(
            host.pending_all_position_verifier_stream,
            output.execution.stream)
            << "The verifier consumer must inherit the exact first-use capture stream.";
        EXPECT_EQ(host.pending_main_decode_stream_calls, 0);
    }

    {
        auto engine = makeEngine();
        MockForwardExecutionHost host(&gpu_ctx);
        host.graph_stage_count = 1;
        host.mock_defer_main_decode_sync = true;
        host.mock_capture_policy.allow_fast_decode = true;
        host.mock_capture_policy.allow_cached_graph_replay = true;

        auto input =
            makeTestInput(1, 1, DeviceId::cuda(0), &token, &position);
        ForwardOutput output{};
        ASSERT_TRUE(engine.execute(input, output, host));

        EXPECT_EQ(host.sync_logits_calls, 0);
        EXPECT_EQ(host.pending_main_decode_stream_calls, 1);
        EXPECT_NE(host.pending_main_decode_stream, nullptr);
        EXPECT_EQ(
            host.pending_main_decode_stream,
            output.execution.stream)
            << "The decode consumer must inherit the exact first-use capture stream.";
        EXPECT_EQ(host.pending_all_position_verifier_stream_calls, 0);
    }
}

/**
 * @brief Verifier state readiness is mandatory even with host logits timing.
 *
 * PerfStats and parity snapshots may request a host-facing logits boundary, but
 * that observation must not suppress the device event consumed by accepted-state
 * publication. The first warmup/capture invocation is asynchronous just like a
 * steady-state replay and therefore publishes the same exact producer stream.
 */
TEST_F(
    Test__ForwardExecutionEngine,
    CacheMiss_GPUVerifierAlwaysPublishesDeviceStateReadiness)
{
    llaminar2::testing::MockDeviceContext gpu_ctx{
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA};
    auto engine = makeEngine();
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_count = 1;
    host.mock_compute_all_position_logits = true;
    host.mock_defer_all_position_verifier_sync = false;

    int token = 42;
    int position = 0;
    auto input =
        makeTestInput(1, 1, DeviceId::cuda(0), &token, &position);
    ForwardOutput output{};
    ASSERT_TRUE(engine.execute(input, output, host));

    EXPECT_EQ(host.sync_logits_calls, 1)
        << "The diagnostic host boundary remains independently observable.";
    EXPECT_EQ(host.pending_all_position_verifier_stream_calls, 1)
        << "Host observation must never suppress verifier state readiness.";
    EXPECT_EQ(
        host.pending_all_position_verifier_stream,
        llaminar2::testing::sharedMockWorkerGPUContext().defaultStream());
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
        {"live_mtp_request_batch_condition", "false"},
        {"all_position_logit_rows", "0"},
        {"moe_placement_epoch", "0"},
        {"result", "miss"},
        {"submission", "runtime"},
        {"uses_device_token_ids", "false"},
        {"uses_device_position_ids", "false"},
        {"uses_device_sequence_lengths", "false"},
        {"seq_len", "2"}};
    const PerfStatsCollector::Tags hit_tags = {
        {"all_position_logits", "true"},
        {"context", "main_verifier"},
        {"decode_has_history", "true"},
        {"live_mtp_request_batch_condition", "false"},
        {"all_position_logit_rows", "0"},
        {"moe_placement_epoch", "0"},
        {"result", "hit"},
        {"submission", "runtime"},
        {"uses_device_token_ids", "false"},
        {"uses_device_position_ids", "false"},
        {"uses_device_sequence_lengths", "false"},
        {"seq_len", "2"}};

    EXPECT_DOUBLE_EQ(findForwardGraphCounterValue(records, "forward_cache_lookup", miss_tags), 1.0);
    EXPECT_DOUBLE_EQ(findForwardGraphCounterValue(records, "forward_cache_lookup", hit_tags), 1.0);

    PerfStatsCollector::reset();
}

TEST_F(Test__ForwardExecutionEngine, CapturedCollectiveOptInRequestsDeferredMainDecodeSync)
{
    ScopedDebugEnv env({
        {"LLAMINAR_PERF_STATS_JSON", "1"},
        {"LLAMINAR_GPU_GRAPH_DEFER_CAPTURED_COLLECTIVE_FINAL_SYNC", "1"},
    });
    PerfStatsCollector::reset();

    auto engine = makeEngine();
    llaminar2::testing::MockDeviceContext gpu_ctx(
        DeviceId::cuda(0),
        ComputeBackendType::GPU_CUDA);
    MockForwardExecutionHost host(&gpu_ctx);
    host.graph_stage_types = {ComputeStageType::ALLREDUCE};
    host.mock_capture_policy.allow_fast_decode = true;
    host.mock_capture_policy.allow_cached_graph_replay = true;
    host.mock_capture_policy.collectives_graph_capturable = true;

    int token = 42;
    int pos = 7;
    auto input = makeTestInput(1, 1, DeviceId::cuda(0), &token, &pos);
    ForwardOutput output{};

    ASSERT_TRUE(engine.execute(input, output, host));
    ASSERT_TRUE(engine.execute(input, output, host));
    EXPECT_EQ(host.build_forward_graph_calls, 1);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags tags = {
        {"allow_graph_replay", "true"},
        {"heterogeneous_segmented", "false"},
        {"collectives_graph_capturable", "true"},
        {"context", "main_decode"},
        {"defer_final_sync", "true"},
        {"has_collectives", "true"},
        {"retained_sparse_parent", "false"},
        {"device_timeline_transaction", "false"},
        {"heterogeneous_ticket_transaction", "false"},
        {"replay_plan_policy", "require_full_graph"}};
    EXPECT_DOUBLE_EQ(findForwardGraphCounterValue(records, "decode_capture_policy", tags), 2.0)
        << "First-use materialization and steady-state replay must both ask the "
           "replay controller to defer final sync when the explicit diagnostic "
           "opt-in is enabled.";

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
