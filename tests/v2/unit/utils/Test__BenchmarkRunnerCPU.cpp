/**
 * @file Test__BenchmarkRunnerCPU.cpp
 * @brief Unit tests verifying BenchmarkRunner handles CPU devices correctly
 *
 * Regression tests for:
 * - setSkipLogitsGatherDecode must NOT be enabled on CPU devices
 * - CPU decode samples from host logits as the CPU implementation
 * - GPU decode must fail hard when device sampling fails
 * - Exported hardware defaults and explicit policy intent match the runtime
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <string>
#include <thread>
#include <utility>

#include "utils/BenchmarkRunner.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "app/InferenceRunnerAdapter.h"
#include "config/OrchestrationConfig.h"
#include "backends/DeviceId.h"
#include "mocks/MockOrchestrationRunner.h"
#include "mocks/MockTokenizer.h"
#include "nlohmann/json.hpp"

using namespace llaminar2;
using namespace llaminar2::test;
using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

namespace
{
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name),
              had_previous_(std::getenv(name) != nullptr),
              previous_(had_previous_ ? std::getenv(name) : "")
        {
            if (value)
                setenv(name, value, 1);
            else
                unsetenv(name);
        }

        ~ScopedEnv()
        {
            if (had_previous_)
                setenv(name_.c_str(), previous_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        bool had_previous_ = false;
        std::string previous_;
    };

    /**
     * @brief Mock inference runner that simulates CPU-only execution.
     *
     * - primaryDeviceId() returns CPU
     * - sampleGreedyOnDevice() returns -1 (no GPU argmax)
     * - logits() returns a deterministic distribution
     * - Tracks whether setSkipLogitsGatherDecode was called with true
     */
    class MockCPUInferenceRunner : public IInferenceRunner
    {
    public:
        static constexpr int VOCAB = 100;
        static constexpr int ARGMAX_TOKEN = 42; ///< Token with highest logit

        MockCPUInferenceRunner()
        {
            logits_.assign(VOCAB, -5.0f);
            logits_[ARGMAX_TOKEN] = 10.0f;
        }

        // Core API
        bool forward(const int *tokens, int seq_len) override
        {
            (void)tokens;
            (void)seq_len;
            forward_count_++;
            return true;
        }

        const float *logits() const override { return logits_.data(); }
        int vocab_size() const override { return VOCAB; }
        void clear_cache() override { clear_count_++; }

        // CPU device — no GPU argmax available
        DeviceId primaryDeviceId() const override { return DeviceId::cpu(); }
        int sampleGreedyOnDevice() override { return -1; }

        // Track the skip-logits-gather flag
        void setSkipLogitsGatherDecode(bool skip) override
        {
            skip_logits_gather_decode_ = skip;
            skip_logits_gather_decode_called_ = true;
        }

        void setSkipLogitsGatherPrefill(bool skip) override
        {
            skip_logits_gather_prefill_ = skip;
            skip_logits_gather_prefill_called_ = true;
        }
        void setSuppressTimeline(bool) override {}
        void setAccumulatePrefill(bool) override {}

        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "mock_cpu"; }
        int get_position() const override { return 0; }

        // Test inspection
        bool skipLogitsGatherDecodeWasEnabled() const { return skip_logits_gather_decode_; }
        bool skipLogitsGatherDecodeWasCalled() const { return skip_logits_gather_decode_called_; }
        bool skipLogitsGatherPrefillWasEnabled() const { return skip_logits_gather_prefill_; }
        bool skipLogitsGatherPrefillWasCalled() const { return skip_logits_gather_prefill_called_; }
        int forwardCount() const { return forward_count_; }

    private:
        std::vector<float> logits_;
        int forward_count_ = 0;
        int clear_count_ = 0;
        bool skip_logits_gather_decode_ = false;
        bool skip_logits_gather_decode_called_ = false;
        bool skip_logits_gather_prefill_ = false;
        bool skip_logits_gather_prefill_called_ = false;
    };

    /** @brief Generic application-readiness fixture with no subsystem details. */
    class MockInferenceReadinessRunner : public MockCPUInferenceRunner
    {
    public:
        explicit MockInferenceReadinessRunner(
            InferenceReadinessState state,
            std::string diagnostic = {})
            : state_(state), diagnostic_(std::move(diagnostic))
        {
        }

        InferenceReadiness inferenceReadiness() const override
        {
            return {
                .state = state_,
                .diagnostic = diagnostic_,
            };
        }

    private:
        InferenceReadinessState state_;
        std::string diagnostic_;
    };

    /**
     * @brief Mock inference runner that simulates GPU execution.
     *
     * - primaryDeviceId() returns CUDA:0
     * - sampleGreedyOnDevice() returns a deterministic token
     * - Tracks whether setSkipLogitsGatherDecode was called with true
     */
    class MockGPUInferenceRunner : public IInferenceRunner
    {
    public:
        static constexpr int VOCAB = 100;
        static constexpr int GPU_ARGMAX_TOKEN = 77;

        MockGPUInferenceRunner()
        {
            logits_.assign(VOCAB, -5.0f);
            logits_[GPU_ARGMAX_TOKEN] = 10.0f;
        }

        bool forward(const int *tokens, int seq_len) override
        {
            (void)tokens;
            (void)seq_len;
            ++prefill_forward_count_;
            if (emulate_persistent_prefix_archive_ &&
                persistent_prefix_archive_populated_)
            {
                /*
                 * A complete prefix restore produces a valid asynchronous
                 * request completion but submits no prefill graph. This is the
                 * production failure mode the benchmark regression exercises.
                 */
                forward_pending_ = true;
                return true;
            }
            if (advance_prefill_graph_on_forward_ &&
                !snapshot_.prefill_graphs.empty())
            {
                for (auto &graph : snapshot_.prefill_graphs)
                {
                    /*
                     * A production probe includes inactive forward-cache
                     * signatures. Only the graph selected for this request is
                     * submitted; cold entries remain historical diagnostics.
                     */
                    if (graph.phase == "cold")
                        continue;
                    graph.phase = "ready";
                    if (prefill_forward_count_ == capture_on_forward_)
                    {
                        ++graph.capture_count;
                        graph.capture_phase = "capture";
                    }
                    else
                    {
                        ++graph.replay_count;
                        graph.capture_phase = "replay";
                    }
                }
            }
            persistent_prefix_archive_populated_ =
                emulate_persistent_prefix_archive_;
            forward_pending_ = true;
            return true;
        }

        bool waitForLastInferenceCompletionForBenchmark() override
        {
            if (!forward_pending_)
            {
                completion_order_valid_ = false;
                return false;
            }
            std::this_thread::sleep_for(completion_delay_);
            forward_pending_ = false;
            ++completion_wait_count_;
            return true;
        }

        const float *logits() const override { return logits_.data(); }
        int vocab_size() const override { return VOCAB; }
        void clear_cache() override {}
        bool purgePrefixCache() override
        {
            ++prefix_purge_count_;
            persistent_prefix_archive_populated_ = false;
            return true;
        }

        // GPU device — GPU argmax available
        DeviceId primaryDeviceId() const override { return DeviceId::cuda(0); }
        int sampleGreedyOnDevice() override
        {
            return device_argmax_available_ ? GPU_ARGMAX_TOKEN : -1;
        }

        void setDeviceArgmaxAvailable(bool available)
        {
            device_argmax_available_ = available;
        }

        void setSkipLogitsGatherDecode(bool skip) override
        {
            skip_logits_gather_decode_ = skip;
        }

        void setSkipLogitsGatherPrefill(bool skip) override
        {
            skip_logits_gather_prefill_ = skip;
        }
        void setSuppressTimeline(bool) override {}
        void setAccumulatePrefill(bool) override {}

        bool configureMTPRequestStopTokens(
            const std::vector<int32_t> &stop_tokens) override
        {
            configured_stop_tokens_ = stop_tokens;
            ++stop_policy_publication_count_;
            return true;
        }

        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "mock_gpu"; }
        int get_position() const override { return 0; }
        PrefixRuntimeStateSnapshot prefixStateProbe() const override { return snapshot_; }

        void setPrefixRuntimeState(PrefixRuntimeStateSnapshot snapshot)
        {
            snapshot_ = std::move(snapshot);
        }

        void setAdvancePrefillGraphOnForward(bool enabled)
        {
            advance_prefill_graph_on_forward_ = enabled;
        }

        void setCaptureOnForward(int forward_count)
        {
            capture_on_forward_ = forward_count;
        }

        void setEmulatePersistentPrefixArchive(bool enabled)
        {
            emulate_persistent_prefix_archive_ = enabled;
            persistent_prefix_archive_populated_ = false;
        }

        bool skipLogitsGatherDecodeWasEnabled() const { return skip_logits_gather_decode_; }
        bool skipLogitsGatherPrefillWasEnabled() const { return skip_logits_gather_prefill_; }
        int stopPolicyPublicationCount() const { return stop_policy_publication_count_; }
        const std::vector<int32_t> &configuredStopTokens() const
        {
            return configured_stop_tokens_;
        }
        int completionWaitCount() const { return completion_wait_count_; }
        int prefillForwardCount() const { return prefill_forward_count_; }
        int prefixPurgeCount() const { return prefix_purge_count_; }
        bool completionOrderValid() const { return completion_order_valid_; }
        void setCompletionDelay(std::chrono::milliseconds delay)
        {
            completion_delay_ = delay;
        }

    private:
        std::vector<float> logits_;
        std::vector<int32_t> configured_stop_tokens_;
        PrefixRuntimeStateSnapshot snapshot_;
        bool skip_logits_gather_decode_ = false;
        bool skip_logits_gather_prefill_ = false;
        bool device_argmax_available_ = true;
        bool forward_pending_ = false;
        bool completion_order_valid_ = true;
        int completion_wait_count_ = 0;
        int prefill_forward_count_ = 0;
        int capture_on_forward_ = -1;
        bool advance_prefill_graph_on_forward_ = false;
        bool emulate_persistent_prefix_archive_ = false;
        bool persistent_prefix_archive_populated_ = false;
        std::chrono::milliseconds completion_delay_{0};
        int stop_policy_publication_count_ = 0;
        int prefix_purge_count_ = 0;
    };

    class MockStatsInferenceRunner : public MockCPUInferenceRunner
    {
    public:
        PrefixRuntimeStateSnapshot prefixStateProbe() const override
        {
            return snapshot;
        }

        PrefixRuntimeStateSnapshot snapshot;
    };

    class MockOrchestratedDecodeRunner : public MockCPUInferenceRunner
    {
    public:
        bool supportsDecodeStep() const override { return true; }

        bool configureMTPRequestStopTokens(
            const std::vector<int32_t> &stop_tokens) override
        {
            configured_stop_tokens_ = stop_tokens;
            return true;
        }

        void setDecodeSamplingParams(const SamplingParams &params) override
        {
            sampling_params_set_ = true;
            last_sampling_params_ = params;
        }

        void setDecodeStepTokenBudget(int max_tokens) override
        {
            decode_step_budget_ = max_tokens;
        }

        DecodeStepOutput decodeStepForBenchmark() override
        {
            ++decode_step_calls_;
            const int remaining = decode_step_budget_ > 0 ? decode_step_budget_ : 1;
            const int accepted = std::min(remaining, 2);
            DecodeStepOutput output;
            output.tokens.reserve(static_cast<size_t>(accepted));
            for (int i = 0; i < accepted; ++i)
            {
                output.tokens.push_back(10 + ((emitted_tokens_ + i) % 5));
            }
            emitted_tokens_ += accepted;
            decoded_tokens_returned_ += output.tokens.size();
            return output;
        }

        bool maybeApplyDecodeBoundaryMaintenance(
            uint64_t committed_tokens) override
        {
            maintenance_tokens_ += committed_tokens;
            ++maintenance_calls_;
            return true;
        }

        int sampleGreedyOnDevice() override
        {
            ++sample_greedy_calls_;
            return MockCPUInferenceRunner::sampleGreedyOnDevice();
        }

        int decodeStepCalls() const { return decode_step_calls_; }
        int maintenanceCalls() const { return maintenance_calls_; }
        uint64_t maintenanceTokens() const { return maintenance_tokens_; }
        uint64_t decodedTokensReturned() const
        {
            return decoded_tokens_returned_;
        }
        int sampleGreedyCalls() const { return sample_greedy_calls_; }
        bool samplingParamsSet() const { return sampling_params_set_; }
        float lastTemperature() const { return last_sampling_params_.temperature; }
        const SamplingParams &lastSamplingParams() const { return last_sampling_params_; }
        const std::vector<int32_t> &configuredStopTokens() const
        {
            return configured_stop_tokens_;
        }

    private:
        std::vector<int32_t> configured_stop_tokens_;
        int decode_step_budget_ = 0;
        int decode_step_calls_ = 0;
        int maintenance_calls_ = 0;
        uint64_t maintenance_tokens_ = 0u;
        uint64_t decoded_tokens_returned_ = 0u;
        int sample_greedy_calls_ = 0;
        int emitted_tokens_ = 0;
        bool sampling_params_set_ = false;
        SamplingParams last_sampling_params_;
    };

    class MockBenchmarkEventOrderRunner : public MockOrchestratedDecodeRunner
    {
    public:
        DeviceId primaryDeviceId() const override { return DeviceId::cuda(0); }

        bool forward(const int *tokens, int seq_len) override
        {
            events_.push_back(seq_len > 1 ? "prefill" : "forward");
            return MockOrchestratedDecodeRunner::forward(tokens, seq_len);
        }

        bool waitForLastInferenceCompletionForBenchmark() override
        {
            events_.push_back("prefill_completion");
            return true;
        }

        void clear_cache() override
        {
            events_.push_back("clear");
            MockOrchestratedDecodeRunner::clear_cache();
        }

        DecodeStepOutput decodeStepForBenchmark() override
        {
            events_.push_back("decode");
            return MockOrchestratedDecodeRunner::decodeStepForBenchmark();
        }

        void resetExecutorStats() override
        {
            events_.push_back("reset_stats");
            MockOrchestratedDecodeRunner::resetExecutorStats();
        }

        void recordCallback()
        {
            events_.push_back("callback");
        }

        const std::vector<std::string> &events() const { return events_; }

    private:
        std::vector<std::string> events_;
    };

    class MockPerfStatsMaintenanceRunner : public MockOrchestratedDecodeRunner
    {
    public:
        bool maybeApplyDecodeBoundaryMaintenance(
            uint64_t committed_tokens) override
        {
            maintenance_tokens_ += committed_tokens;
            ++maintenance_calls_;
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "measured_decode_maintenance_marker",
                1.0,
                "decode",
                "cpu");
            return true;
        }

        void drainCompletedDecodeBoundaryMaintenanceDiagnostics() override
        {
            ++drain_calls_;
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "epilogue_drain_marker",
                1.0,
                "decode",
                "cpu");
        }

        int maintenanceCalls() const { return maintenance_calls_; }
        uint64_t maintenanceTokens() const { return maintenance_tokens_; }
        int drainCalls() const { return drain_calls_; }

    private:
        int maintenance_calls_ = 0;
        uint64_t maintenance_tokens_ = 0u;
        int drain_calls_ = 0;
    };

    /** Runtime owner whose typed totals deliberately disagree with telemetry. */
    class MockTypedOptimizationStatusRunner final
        : public MockOrchestratedDecodeRunner
    {
    public:
        bool maybeApplyDecodeBoundaryMaintenance(
            uint64_t committed_tokens) override
        {
            if (!MockOrchestratedDecodeRunner::
                    maybeApplyDecodeBoundaryMaintenance(committed_tokens))
            {
                return false;
            }
            ++status_.published_movement_waves;
            ++status_.completed_movement.transactions;
            status_.completed_movement.commands += 2u;
            status_.completed_movement.physical_bytes += 64u;
            ++status_.completed_movement.promotions;
            ++status_.completed_movement.demotions;

            /* If BenchmarkRunner regresses to ledger authority, this inflated
             * value makes the focused assertion fail unambiguously. */
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "dynamic_movement_transactions",
                100.0,
                "maintenance",
                "cpu");
            return true;
        }

        uint64_t moeRuntimeMovementEpoch() const override
        {
            return status_.published_movement_waves;
        }

        MoEOptimizationStatus moeOptimizationStatus() const override
        {
            return status_;
        }

    private:
        MoEOptimizationStatus status_{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
        };
    };

    class MockBatchedOrchestratedDecodeRunner : public MockCPUInferenceRunner
    {
    public:
        bool supportsDecodeStep() const override { return true; }

        bool supportsPrefillBatchForBenchmark(int request_batch) const override
        {
            return request_batch > 1 && request_batch <= max_request_batch_;
        }

        bool prefillBatchForBenchmark(
            const std::vector<std::vector<int>> &token_batches) override
        {
            ++batch_prefill_calls_;
            last_prefill_batch_ = static_cast<int>(token_batches.size());
            return !token_batches.empty();
        }

        bool supportsDecodeStepBatchForBenchmark(int request_batch) const override
        {
            return request_batch > 1 && request_batch <= max_request_batch_;
        }

        void setDecodeSamplingParams(const SamplingParams &params) override
        {
            sampling_params_set_ = true;
            last_sampling_params_ = params;
        }

        void setDecodeStepTokenBudget(int max_tokens) override
        {
            decode_step_budget_ = max_tokens;
        }

        DecodeStepOutput decodeStepForBenchmark() override
        {
            ++single_decode_step_calls_;
            return DecodeStepOutput{{}, false, "single-request decode should not be used"};
        }

        DecodeBatchStepOutput decodeBatchStepForBenchmark(int request_batch) override
        {
            ++batch_decode_step_calls_;
            last_request_batch_ = request_batch;
            emitted_by_request_.resize(static_cast<size_t>(request_batch), 0);

            const int per_request_emit =
                decode_step_budget_ > 0 ? std::min(decode_step_budget_, 1) : 1;

            DecodeBatchStepOutput output;
            output.tokens_by_request.resize(static_cast<size_t>(request_batch));
            output.is_complete_by_request.assign(static_cast<size_t>(request_batch), false);

            for (int request = 0; request < request_batch; ++request)
            {
                const size_t request_index = static_cast<size_t>(request);
                for (int i = 0; i < per_request_emit; ++i)
                {
                    output.tokens_by_request[request_index].push_back(
                        100 + request * 10 + emitted_by_request_[request_index]);
                    ++emitted_by_request_[request_index];
                }
            }

            for (const auto &request_tokens : output.tokens_by_request)
                decoded_tokens_returned_ += request_tokens.size();

            return output;
        }

        bool maybeApplyDecodeBoundaryMaintenance(
            uint64_t committed_tokens) override
        {
            maintenance_tokens_ += committed_tokens;
            ++maintenance_calls_;
            return true;
        }

        void clear_cache() override
        {
            MockCPUInferenceRunner::clear_cache();
            std::fill(emitted_by_request_.begin(), emitted_by_request_.end(), 0);
        }

        int singleDecodeStepCalls() const { return single_decode_step_calls_; }
        int batchPrefillCalls() const { return batch_prefill_calls_; }
        int batchDecodeStepCalls() const { return batch_decode_step_calls_; }
        int maintenanceCalls() const { return maintenance_calls_; }
        uint64_t maintenanceTokens() const { return maintenance_tokens_; }
        uint64_t decodedTokensReturned() const
        {
            return decoded_tokens_returned_;
        }
        int lastPrefillBatch() const { return last_prefill_batch_; }
        int lastRequestBatch() const { return last_request_batch_; }
        bool samplingParamsSet() const { return sampling_params_set_; }
        const SamplingParams &lastSamplingParams() const { return last_sampling_params_; }

    private:
        static constexpr int max_request_batch_ = 8;
        int decode_step_budget_ = 0;
        int batch_prefill_calls_ = 0;
        int single_decode_step_calls_ = 0;
        int batch_decode_step_calls_ = 0;
        int maintenance_calls_ = 0;
        uint64_t maintenance_tokens_ = 0u;
        uint64_t decoded_tokens_returned_ = 0u;
        int last_prefill_batch_ = 0;
        int last_request_batch_ = 0;
        std::vector<int> emitted_by_request_;
        bool sampling_params_set_ = false;
        SamplingParams last_sampling_params_;
    };

    class MockBatchPrefillOnlyRunner : public MockOrchestratedDecodeRunner
    {
    public:
        bool supportsPrefillBatchForBenchmark(int request_batch) const override
        {
            return request_batch > 1 && request_batch <= max_request_batch_;
        }

        bool prefillBatchForBenchmark(
            const std::vector<std::vector<int>> &token_batches) override
        {
            ++batch_prefill_calls_;
            last_prefill_batch_ = static_cast<int>(token_batches.size());
            return !token_batches.empty();
        }

        int batchPrefillCalls() const { return batch_prefill_calls_; }
        int lastPrefillBatch() const { return last_prefill_batch_; }

    private:
        static constexpr int max_request_batch_ = 8;
        int batch_prefill_calls_ = 0;
        int last_prefill_batch_ = 0;
    };

    class MockMeasuredMTPStatsRunner : public MockOrchestratedDecodeRunner
    {
    public:
        PrefixRuntimeStateSnapshot prefixStateProbe() const override
        {
            ++probe_count_;
            PrefixRuntimeStateSnapshot snapshot;
            snapshot.initialized = true;
            snapshot.architecture = "mock_cpu";
            snapshot.execution_path = "graph";
            snapshot.primary_device = DeviceId::cpu();
            snapshot.mtp_config_enabled = true;
            snapshot.mtp_current_depth = static_cast<int>(probe_count_);
            snapshot.mtp_min_depth = 1;
            snapshot.mtp_max_depth = 3;

            /*
             * Use non-identical per-request counters so the test below proves
             * benchmark reporting is an aggregate over the measured iterations,
             * not a final live snapshot.
             */
            snapshot.mtp_draft_steps = probe_count_;
            snapshot.mtp_accepted_tokens = probe_count_ * 10;
            snapshot.mtp_rejected_tokens = probe_count_;
            snapshot.mtp_rollbacks = probe_count_ * 2;
            snapshot.mtp_bypasses = probe_count_ % 2;
            snapshot.mtp_verifier_runs = probe_count_ * 3;
            snapshot.mtp_verifier_token_count = probe_count_ * 4;
            snapshot.mtp_stochastic_accept_tests = probe_count_ * 5;
            snapshot.mtp_stochastic_accepts = probe_count_ * 4;
            snapshot.mtp_stochastic_residual_samples = probe_count_;
            snapshot.mtp_stochastic_terminal_samples = probe_count_ * 2;
            snapshot.mtp_transaction_commits = probe_count_;
            snapshot.mtp_transaction_rollbacks = probe_count_ + 1;
            snapshot.mtp_transaction_validation_failures = probe_count_ + 2;
            snapshot.mtp_unsafe_verifier_state_rejections = probe_count_ + 3;
            snapshot.mtp_depth_policy_windows = probe_count_ * 6;
            snapshot.mtp_depth_policy_updates = probe_count_;
            snapshot.mtp_depth_policy_promotions = probe_count_ + 4;
            snapshot.mtp_depth_policy_demotions = probe_count_ + 5;
            snapshot.mtp_depth_policy_observe_recommendations = probe_count_ + 6;

            snapshot.mtp_request.enabled = true;
            snapshot.mtp_request.verify_mode = "speculative-sampling";
            snapshot.mtp_request.stochastic_verify = true;
            snapshot.mtp_request.adaptive_depth_enabled = true;
            snapshot.mtp_request.depth_policy_mode = "dynamic";
            snapshot.mtp_request.current_depth = snapshot.mtp_current_depth;
            snapshot.mtp_request.min_depth = snapshot.mtp_min_depth;
            snapshot.mtp_request.max_depth = snapshot.mtp_max_depth;
            snapshot.mtp_request.depth_policy_updates = snapshot.mtp_depth_policy_updates;
            snapshot.mtp_request.last_depth_policy_reason =
                "probe_" + std::to_string(probe_count_);
            snapshot.mtp_request.draft_steps = snapshot.mtp_draft_steps;
            snapshot.mtp_request.accepted_tokens = snapshot.mtp_accepted_tokens;
            snapshot.mtp_request.rejected_tokens = snapshot.mtp_rejected_tokens;
            snapshot.mtp_request.rollbacks = snapshot.mtp_rollbacks;
            snapshot.mtp_request.acceptance_rate =
                static_cast<double>(snapshot.mtp_accepted_tokens) /
                static_cast<double>(snapshot.mtp_accepted_tokens +
                                    snapshot.mtp_rejected_tokens);
            snapshot.mtp_request.stochastic_accept_tests =
                snapshot.mtp_stochastic_accept_tests;
            snapshot.mtp_request.stochastic_accepts = snapshot.mtp_stochastic_accepts;
            snapshot.mtp_request.stochastic_residual_samples =
                snapshot.mtp_stochastic_residual_samples;
            snapshot.mtp_request.stochastic_terminal_samples =
                snapshot.mtp_stochastic_terminal_samples;
            snapshot.mtp_request.stochastic_acceptance_rate =
                static_cast<double>(snapshot.mtp_stochastic_accepts) /
                static_cast<double>(snapshot.mtp_stochastic_accept_tests);
            return snapshot;
        }

        int probeCount() const { return static_cast<int>(probe_count_); }

    private:
        mutable uint64_t probe_count_ = 0;
    };

    /**
     * @brief Helper to create a MockTokenizer with standard expectations.
     *
     * Encodes any prompt to a short token sequence and marks token 99 as stop.
     */
    std::shared_ptr<MockTokenizer> createMockTokenizer()
    {
        auto tok = std::make_shared<MockTokenizer>();

        // encode() returns a short token sequence
        ON_CALL(*tok, encode(_, _, _))
            .WillByDefault(Return(std::vector<int>{1, 2, 3, 4, 5}));

        // decode_token() returns a placeholder string
        ON_CALL(*tok, decode_token(_))
            .WillByDefault(Return(std::string("x")));

        // Stop token detection: token 99 is stop, everything else is not
        ON_CALL(*tok, is_stop_token(_))
            .WillByDefault(Return(false));
        ON_CALL(*tok, is_stop_token(99))
            .WillByDefault(Return(true));
        ON_CALL(*tok, stop_tokens())
            .WillByDefault(Return(std::vector<int>{99}));

        ON_CALL(*tok, vocab_size())
            .WillByDefault(Return(100));

        return tok;
    }

    class ScopedGpuGraphsSetting
    {
    public:
        explicit ScopedGpuGraphsSetting(bool enabled)
            : previous_(mutableDebugEnv().execution.gpu_graphs)
        {
            mutableDebugEnv().execution.gpu_graphs = enabled;
        }

        ~ScopedGpuGraphsSetting()
        {
            mutableDebugEnv().execution.gpu_graphs = previous_;
        }

    private:
        bool previous_ = false;
    };

    class ScopedPrefillGraphRequiredSetting
    {
    public:
        explicit ScopedPrefillGraphRequiredSetting(bool required)
            : previous_(mutableDebugEnv().execution.prefill_graph_required)
        {
            mutableDebugEnv().execution.prefill_graph_required = required;
        }

        ~ScopedPrefillGraphRequiredSetting()
        {
            mutableDebugEnv().execution.prefill_graph_required = previous_;
        }

    private:
        bool previous_ = false;
    };

    std::filesystem::path uniqueBenchmarkPromptPath()
    {
        static std::atomic<uint64_t> sequence{0};
        return std::filesystem::temp_directory_path() /
               ("llaminar_benchmark_prompt_" +
                std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) +
                ".txt");
    }

    /**
     * @brief RAII owner for one exact-byte benchmark prompt fixture.
     */
    class ScopedBenchmarkPromptFile
    {
    public:
        explicit ScopedBenchmarkPromptFile(const std::string &bytes)
            : path_(uniqueBenchmarkPromptPath())
        {
            std::ofstream output(path_, std::ios::binary | std::ios::trunc);
            if (!output)
                throw std::runtime_error("unable to create benchmark prompt fixture");
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            if (!output)
                throw std::runtime_error("unable to write benchmark prompt fixture");
        }

        ~ScopedBenchmarkPromptFile()
        {
            std::error_code error;
            std::filesystem::remove(path_, error);
        }

        const std::filesystem::path &path() const noexcept { return path_; }

    private:
        std::filesystem::path path_;
    };

} // namespace

// =============================================================================
// Tests
// =============================================================================

TEST(Test__BenchmarkRunnerCPU, ResolvesInlinePromptWithoutLegacySentinelSubstitution)
{
    OrchestrationConfig config;
    config.prompt = "Hello, my name is";

    const auto resolved = resolveBenchmarkPrompt(config);

    EXPECT_EQ(resolved.source, BenchmarkPromptSource::Inline);
    EXPECT_EQ(resolved.text, "Hello, my name is");
    EXPECT_TRUE(resolved.file_path.empty());
    EXPECT_EQ(resolved.sha256.size(), 64u);
}

TEST(Test__BenchmarkRunnerCPU, ResolvesPromptFileAsExactBytes)
{
    const std::string exact_prompt = "First line\nSecond line\n";
    ScopedBenchmarkPromptFile prompt_file(exact_prompt);

    OrchestrationConfig file_config;
    file_config.benchmark_prompt_file_path = prompt_file.path().string();
    const auto from_file = resolveBenchmarkPrompt(file_config);

    OrchestrationConfig inline_config;
    inline_config.prompt = exact_prompt;
    const auto from_inline = resolveBenchmarkPrompt(inline_config);

    EXPECT_EQ(from_file.source, BenchmarkPromptSource::File);
    EXPECT_EQ(from_file.text, exact_prompt);
    EXPECT_EQ(from_file.file_path, prompt_file.path().string());
    EXPECT_EQ(from_file.sha256, from_inline.sha256)
        << "Equal prompt bytes must have one source-independent benchmark identity";
}

TEST(Test__BenchmarkRunnerCPU, RejectsAmbiguousOrInvalidPromptFiles)
{
    OrchestrationConfig ambiguous;
    ambiguous.prompt = "inline";
    ambiguous.benchmark_prompt_file_path = "/tmp/prompt.txt";
    EXPECT_THROW((void)resolveBenchmarkPrompt(ambiguous), std::invalid_argument);

    ScopedBenchmarkPromptFile empty_file("");
    OrchestrationConfig empty;
    empty.benchmark_prompt_file_path = empty_file.path().string();
    EXPECT_THROW((void)resolveBenchmarkPrompt(empty), std::invalid_argument);

    OrchestrationConfig missing;
    missing.benchmark_prompt_file_path = uniqueBenchmarkPromptPath().string();
    EXPECT_THROW((void)resolveBenchmarkPrompt(missing), std::runtime_error);
}

TEST(Test__BenchmarkRunnerCPU,
     PreparingRunnerCannotLeakSubsystemWorkIntoBenchmark)
{
    {
        ScopedEnv iterations("LLAMINAR_BENCHMARK_ITERATIONS", "1");
        ScopedEnv warmups(
            "LLAMINAR_BENCHMARK_WARMUP_ITERATIONS", "0");
        mutableDebugEnv().runtime_debug.reload();

        auto runner =
            std::make_shared<MockInferenceReadinessRunner>(
                InferenceReadinessState::Preparing);
        BenchmarkRunner benchmark(runner, createMockTokenizer());

        OrchestrationConfig config;
        config.prompt = "Hello world";
        config.n_predict = 0;

        const BenchmarkResult result = benchmark.run(config);

        EXPECT_FALSE(result.success);
        EXPECT_NE(
            result.failure_reason.find("not ready"),
            std::string::npos)
            << result.failure_reason;
        EXPECT_EQ(runner->forwardCount(), 0)
            << "BenchmarkRunner must never drive preparation workloads";
    }
    mutableDebugEnv().runtime_debug.reload();
}

TEST(Test__BenchmarkRunnerCPU,
     FailedProductionPreparationCannotFallThroughIntoMeasurement)
{
    {
        ScopedEnv iterations("LLAMINAR_BENCHMARK_ITERATIONS", "1");
        ScopedEnv warmups(
            "LLAMINAR_BENCHMARK_WARMUP_ITERATIONS", "0");
        mutableDebugEnv().runtime_debug.reload();

        auto runner = std::make_shared<MockInferenceReadinessRunner>(
            InferenceReadinessState::Failed,
            "injected preparation failure");
        BenchmarkRunner benchmark(runner, createMockTokenizer());

        OrchestrationConfig config;
        config.prompt = "Hello world";
        config.n_predict = 0;

        const BenchmarkResult result = benchmark.run(config);

        EXPECT_FALSE(result.success);
        EXPECT_NE(
            result.failure_reason.find("injected preparation failure"),
            std::string::npos)
            << result.failure_reason;
        EXPECT_EQ(runner->forwardCount(), 0)
            << "A failed readiness contract must abort before timed inference";
    }
    mutableDebugEnv().runtime_debug.reload();
}

/**
 * @brief Verify that CPU benchmark does NOT enable skip-logits-gather.
 *
 * Regression test: BenchmarkRunner previously unconditionally called
 * setSkipLogitsGatherDecode(true), which broke CPU benchmarks because
 * CPU has no GPU-side argmax and logits must be gathered to host.
 */
TEST(Test__BenchmarkRunnerCPU, DoesNotSkipLogitsGatherOnCPU)
{
    auto runner = std::make_shared<MockCPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;

    auto result = bench.run(config);

    // The flag must have been set to false (not true) for CPU
    EXPECT_TRUE(runner->skipLogitsGatherDecodeWasCalled())
        << "BenchmarkRunner must call setSkipLogitsGatherDecode";
    EXPECT_FALSE(runner->skipLogitsGatherDecodeWasEnabled())
        << "CPU device must NOT enable skip-logits-gather (no GPU argmax available)";
    EXPECT_TRUE(runner->skipLogitsGatherPrefillWasCalled())
        << "BenchmarkRunner must configure prefill gather policy";
    EXPECT_FALSE(runner->skipLogitsGatherPrefillWasEnabled())
        << "CPU device must keep prefill logits visible to host-side CPU sampling";
}

/**
 * @brief Verify that GPU benchmark DOES enable skip-logits-gather.
 *
 * Ensures the GPU optimization path is preserved.
 */
TEST(Test__BenchmarkRunnerCPU, EnablesSkipLogitsGatherOnGPU)
{
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;

    auto result = bench.run(config);

    // GPU runner should have skip-logits-gather enabled
    EXPECT_TRUE(runner->skipLogitsGatherDecodeWasEnabled())
        << "GPU device must enable skip-logits-gather for performance";
    EXPECT_TRUE(runner->skipLogitsGatherPrefillWasEnabled())
        << "GPU device must skip prefill logits gather when benchmark sampling stays device-side";
    EXPECT_GT(runner->stopPolicyPublicationCount(), 0)
        << "Every GPU benchmark request must publish its fixed-length stop policy before prefill";
    EXPECT_TRUE(runner->configuredStopTokens().empty())
        << "A fixed-token throughput benchmark must disable stop tokens on the device as well as in the host decode loop";
}

/**
 * @brief GPU prefill timing must include the complete asynchronous transaction.
 *
 * The mock returns immediately from forward() and models the durable event wait
 * with a short delay. MTP is enabled so this contract includes shifted sidecar
 * KV population, not merely the earlier main graph output. A regression that
 * stops the clock at graph submission either omits the wait entirely or reports
 * less than the injected latency.
 */
TEST(Test__BenchmarkRunnerCPU, GPUPrefillTimingWaitsForDurableTerminalEvent)
{
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    runner->setCompletionDelay(std::chrono::milliseconds(5));
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 0;
    config.mtp.enabled = true;

    const auto result = bench.run(config);

    EXPECT_TRUE(result.success) << result.failure_reason;
    EXPECT_TRUE(runner->completionOrderValid());
    EXPECT_GT(runner->completionWaitCount(), 0);
    EXPECT_GE(result.prefill_time_ms, 4.0)
        << "GPU prefill timing stopped before the terminal event wait completed";
}

/**
 * @brief Verify CPU decode uses host-side argmax and succeeds.
 *
 * Regression test: BenchmarkRunner previously treated sampleGreedyOnDevice() == -1
 * as a hard error for CPU. CPU decode should instead use logits() + CPU argmax.
 */
TEST(Test__BenchmarkRunnerCPU, CPUDecodeSucceedsWithHostArgmax)
{
    auto runner = std::make_shared<MockCPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 5;

    auto result = bench.run(config);

    // Benchmark must succeed — CPU decode via host argmax should work
    EXPECT_TRUE(result.success)
        << "CPU benchmark must succeed using host-side argmax";
    EXPECT_TRUE(result.decode_success)
        << "CPU decode phase must succeed";
    EXPECT_EQ(result.decode_tokens, 5)
        << "All requested decode tokens should be generated";
}

/**
 * @brief Verify GPU decode succeeds via sampleGreedyOnDevice().
 */
TEST(Test__BenchmarkRunnerCPU, GPUDecodeSucceedsWithDeviceArgmax)
{
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 5;

    auto result = bench.run(config);

    EXPECT_TRUE(result.success)
        << "GPU benchmark must succeed using device-side argmax";
    EXPECT_TRUE(result.decode_success)
        << "GPU decode phase must succeed";
}

TEST(Test__BenchmarkRunnerCPU, RequiredPrefillGraphCaptureFailsWhenProbeNeverCaptures)
{
    ScopedGpuGraphsSetting force_gpu_graphs(true);
    ScopedPrefillGraphRequiredSetting require_prefill_graph(true);
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 0;

    auto result = bench.run(config);

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.failure_reason.find("required prefill graph capture/replay was not observed"),
              std::string::npos);
}

TEST(Test__BenchmarkRunnerCPU, RequiredPrefillGraphCaptureAcceptsCapturedProbe)
{
    ScopedGpuGraphsSetting force_gpu_graphs(true);
    ScopedPrefillGraphRequiredSetting require_prefill_graph(true);
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    PrefixRuntimeStateSnapshot snapshot;
    PrefillGraphRuntimeProbe graph;
    graph.phase = "ready";
    graph.capture_phase = "capture";
    graph.capture_count = 1;
    graph.node_count = 42;
    graph.domain_id = "mock_tp";
    snapshot.prefill_graphs.push_back(graph);
    runner->setPrefixRuntimeState(snapshot);
    runner->setAdvancePrefillGraphOnForward(true);

    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 0;

    auto result = bench.run(config);

    EXPECT_TRUE(result.success) << result.failure_reason;
    ASSERT_EQ(result.prefix_state.prefill_graphs.size(), 1u);
    EXPECT_EQ(result.prefix_state.prefill_graphs[0].capture_count, 1u);
    EXPECT_EQ(result.prefix_state.prefill_graphs[0].domain_id, "mock_tp");
}

/**
 * @brief Repeated throughput samples must execute prefill despite prefix reuse.
 *
 * Request reset intentionally preserves the production prefix archive. An
 * identical warmup prompt can therefore satisfy the measured request without
 * submitting its retained prefill graph unless benchmark mode also invokes
 * the explicit archive-purge boundary before each full-prefill sample.
 */
TEST(
    Test__BenchmarkRunnerCPU,
    FullPrefillSamplesPurgeReusablePrefixArchiveOutsideMeasurement)
{
    {
        ScopedEnv iterations("LLAMINAR_BENCHMARK_ITERATIONS", "1");
        ScopedEnv warmups(
            "LLAMINAR_BENCHMARK_WARMUP_ITERATIONS", "1");
        mutableDebugEnv().runtime_debug.reload();
        ScopedGpuGraphsSetting force_gpu_graphs(true);
        ScopedPrefillGraphRequiredSetting require_prefill_graph(true);

        auto runner = std::make_shared<MockGPUInferenceRunner>();
        PrefixRuntimeStateSnapshot snapshot;
        PrefillGraphRuntimeProbe graph;
        graph.phase = "ready";
        graph.capture_phase = "capture";
        graph.capture_count = 1;
        graph.node_count = 42;
        graph.domain_id = "persistent_prefix_regression";
        snapshot.prefill_graphs.push_back(graph);
        runner->setPrefixRuntimeState(snapshot);
        runner->setAdvancePrefillGraphOnForward(true);
        runner->setEmulatePersistentPrefixArchive(true);

        BenchmarkRunner bench(runner, createMockTokenizer());
        OrchestrationConfig config;
        config.prompt = "Hello world";
        config.n_predict = 0;

        const auto result = bench.run(config);
        ASSERT_TRUE(result.success) << result.failure_reason;
        EXPECT_EQ(runner->prefillForwardCount(), 3)
            << "One graph-readiness replay, one ordinary warmup, and one measured "
               "prefill are required";
        EXPECT_GE(runner->prefixPurgeCount(), 3)
            << "Graph preparation, warmup, and measurement must each begin with "
               "an empty reusable archive";
        ASSERT_EQ(result.prefix_state.prefill_graphs.size(), 1u);
        EXPECT_EQ(result.prefix_state.prefill_graphs[0].replay_count, 3u)
            << "Every full-prefill submission must replay the retained graph";
    }
    mutableDebugEnv().runtime_debug.reload();
}

TEST(
    Test__BenchmarkRunnerCPU,
    PrefillGraphPreparationStopsAtFirstReplayReadyState)
{
    {
        ScopedEnv iterations("LLAMINAR_BENCHMARK_ITERATIONS", "1");
        ScopedEnv warmups(
            "LLAMINAR_BENCHMARK_WARMUP_ITERATIONS", "0");
        mutableDebugEnv().runtime_debug.reload();
        ScopedGpuGraphsSetting force_gpu_graphs(true);
        ScopedPrefillGraphRequiredSetting require_prefill_graph(true);

        auto runner = std::make_shared<MockGPUInferenceRunner>();
        PrefixRuntimeStateSnapshot snapshot;
        PrefillGraphRuntimeProbe graph;
        graph.phase = "ready";
        graph.capture_phase = "capture";
        graph.capture_count = 1;
        graph.node_count = 42;
        graph.domain_id = "state_driven";
        snapshot.prefill_graphs.push_back(graph);
        runner->setPrefixRuntimeState(snapshot);
        runner->setAdvancePrefillGraphOnForward(true);

        BenchmarkRunner bench(runner, createMockTokenizer());
        OrchestrationConfig config;
        config.prompt = "Hello world";
        config.n_predict = 0;

        const auto result = bench.run(config);
        EXPECT_TRUE(result.success) << result.failure_reason;
        EXPECT_EQ(runner->prefillForwardCount(), 2)
            << "One preparation replay plus one measured replay is sufficient; "
               "fixed repeated full-prompt warmups are forbidden.";
    }
    mutableDebugEnv().runtime_debug.reload();
}

TEST(
    Test__BenchmarkRunnerCPU,
    RequiredPrefillGraphCaptureRejectsStaleColdReplayObservation)
{
    ScopedGpuGraphsSetting force_gpu_graphs(true);
    ScopedPrefillGraphRequiredSetting require_prefill_graph(true);
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    PrefixRuntimeStateSnapshot snapshot;
    PrefillGraphRuntimeProbe graph;
    graph.phase = "cold";
    graph.capture_phase = "replay";
    graph.capture_count = 1;
    graph.replay_count = 4;
    graph.node_count = 42;
    graph.domain_id = "stale_domain";
    snapshot.prefill_graphs.push_back(graph);
    runner->setPrefixRuntimeState(snapshot);

    BenchmarkRunner bench(runner, createMockTokenizer());
    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 0;

    const auto result = bench.run(config);
    EXPECT_FALSE(result.success);
    EXPECT_NE(
        result.failure_reason.find(
            "required prefill graph capture/replay was not observed"),
        std::string::npos)
        << result.failure_reason;
}

TEST(
    Test__BenchmarkRunnerCPU,
    RequiredPrefillGraphReplayIgnoresInactiveColdCacheEntries)
{
    ScopedGpuGraphsSetting force_gpu_graphs(true);
    ScopedPrefillGraphRequiredSetting require_prefill_graph(true);
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    PrefixRuntimeStateSnapshot snapshot;

    PrefillGraphRuntimeProbe active;
    active.phase = "ready";
    active.capture_phase = "capture";
    active.capture_count = 1;
    active.node_count = 42;
    active.domain_id = "active_domain";
    active.participant_id = 0;
    active.bucket_seq_len = 512;
    snapshot.prefill_graphs.push_back(active);

    PrefillGraphRuntimeProbe inactive;
    inactive.phase = "cold";
    inactive.capture_phase = "unknown";
    inactive.domain_id = "";
    inactive.bucket_seq_len = 256;
    snapshot.prefill_graphs.push_back(inactive);

    runner->setPrefixRuntimeState(snapshot);
    runner->setAdvancePrefillGraphOnForward(true);

    BenchmarkRunner bench(runner, createMockTokenizer());
    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 0;

    const auto result = bench.run(config);
    EXPECT_TRUE(result.success) << result.failure_reason;
    ASSERT_EQ(result.prefix_state.prefill_graphs.size(), 2u);
    EXPECT_GT(result.prefix_state.prefill_graphs[0].replay_count, 0);
    EXPECT_EQ(result.prefix_state.prefill_graphs[1].replay_count, 0);
    EXPECT_EQ(result.prefix_state.prefill_graphs[1].phase, "cold");
}

/**
 * @brief A padded active graph must not alias its full-width cache template.
 *
 * Padded prompts and eagerly materialized bucket templates legitimately share
 * domain, participant, chunk, and physical bucket coordinates. The real token
 * count is part of PrefillGraphCacheKey and must therefore participate in the
 * benchmark proof identity as well. Otherwise the first cold template can
 * hide a replay by the active one-token graph.
 */
TEST(
    Test__BenchmarkRunnerCPU,
    RequiredPrefillGraphReplayDistinguishesPaddedRealTokenCount)
{
    ScopedGpuGraphsSetting force_gpu_graphs(true);
    ScopedPrefillGraphRequiredSetting require_prefill_graph(true);
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    PrefixRuntimeStateSnapshot snapshot;

    PrefillGraphRuntimeProbe full_bucket_template;
    full_bucket_template.phase = "cold";
    full_bucket_template.capture_phase = "materialized_without_launch";
    full_bucket_template.capture_count = 1;
    full_bucket_template.node_count = 43;
    full_bucket_template.domain_id = "padded_domain";
    full_bucket_template.participant_id = 0;
    full_bucket_template.chunk_index = 0;
    full_bucket_template.bucket_seq_len = 256;
    full_bucket_template.real_token_count = 256;
    snapshot.prefill_graphs.push_back(full_bucket_template);

    PrefillGraphRuntimeProbe active_padded_graph = full_bucket_template;
    active_padded_graph.phase = "ready";
    active_padded_graph.capture_phase = "replay";
    active_padded_graph.replay_count = 1;
    active_padded_graph.node_count = 42;
    active_padded_graph.real_token_count = 1;
    snapshot.prefill_graphs.push_back(active_padded_graph);

    runner->setPrefixRuntimeState(snapshot);
    runner->setAdvancePrefillGraphOnForward(true);

    BenchmarkRunner bench(runner, createMockTokenizer());
    OrchestrationConfig config;
    config.prompt = "Hello";
    config.n_predict = 0;

    const auto result = bench.run(config);
    EXPECT_TRUE(result.success) << result.failure_reason;
    ASSERT_EQ(result.prefix_state.prefill_graphs.size(), 2u);
    EXPECT_EQ(result.prefix_state.prefill_graphs[0].replay_count, 0);
    EXPECT_GT(result.prefix_state.prefill_graphs[1].replay_count, 1);
}

/**
 * @brief A measured prefill sample must never perform graph capture.
 *
 * State-driven preparation uses forward 1 to reach replay-ready, forward 2 is
 * the configured ordinary warmup, and forward 3 is the first measured sample.
 * The mock deliberately reports a new capture on that sample; accepting it
 * would mix setup work into the production replay speedometer.
 */
TEST(Test__BenchmarkRunnerCPU, RequiredPrefillGraphReplayRejectsMeasuredCapture)
{
    ScopedGpuGraphsSetting force_gpu_graphs(true);
    ScopedPrefillGraphRequiredSetting require_prefill_graph(true);
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    PrefixRuntimeStateSnapshot snapshot;
    PrefillGraphRuntimeProbe graph;
    graph.phase = "ready";
    graph.capture_phase = "capture";
    graph.capture_count = 1;
    graph.node_count = 42;
    graph.domain_id = "mock_tp";
    graph.participant_id = 0;
    graph.bucket_seq_len = 2;
    snapshot.prefill_graphs.push_back(graph);
    runner->setPrefixRuntimeState(snapshot);
    runner->setAdvancePrefillGraphOnForward(true);
    runner->setCaptureOnForward(3);

    auto tokenizer = createMockTokenizer();
    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 0;

    const auto result = bench.run(config);

    EXPECT_FALSE(result.success);
    EXPECT_NE(
        result.failure_reason.find("captured during measurement"),
        std::string::npos)
        << result.failure_reason;
}

TEST(Test__BenchmarkRunnerCPU, GPUDecodeFailsHardWhenDeviceArgmaxFails)
{
    auto runner = std::make_shared<MockGPUInferenceRunner>();
    runner->setDeviceArgmaxAvailable(false);
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 5;

    auto result = bench.run(config);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.decode_success);
    EXPECT_NE(result.failure_reason.find("GPU device sampling failed"),
              std::string::npos);
    EXPECT_NE(result.failure_reason.find("host logits sampling is CPU-only"),
              std::string::npos);
}

TEST(Test__BenchmarkRunnerCPU, FailsBeforePrefillWhenPromptExceedsContext)
{
    auto runner = std::make_shared<MockCPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();
    ON_CALL(*tokenizer, encode(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3, 4, 5, 6}));

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "This prompt intentionally does not fit";
    config.max_seq_len = 4;
    config.n_predict = 1;

    auto result = bench.run(config);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.prefill_success);
    EXPECT_EQ(result.prefill_tokens, 6);
    EXPECT_EQ(runner->forwardCount(), 0)
        << "BenchmarkRunner must reject an oversized prompt before prefill";
    EXPECT_NE(result.failure_reason.find("benchmark prompt has 6 tokens"), std::string::npos);
    EXPECT_NE(result.failure_reason.find("context length is 4"), std::string::npos);
}

TEST(Test__BenchmarkRunnerCPU, FailsBeforePrefillWhenPromptPlusDecodeExceedsContext)
{
    auto runner = std::make_shared<MockCPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Prompt fits, decode does not";
    config.max_seq_len = 8;
    config.n_predict = 4;

    auto result = bench.run(config);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.prefill_success);
    EXPECT_EQ(result.prefill_tokens, 5);
    EXPECT_EQ(runner->forwardCount(), 0)
        << "BenchmarkRunner must reject prompt+decode context overflow before prefill";
    EXPECT_NE(result.failure_reason.find("benchmark request needs 9 total tokens"), std::string::npos);
    EXPECT_NE(result.failure_reason.find("5 prompt + 4 decode"), std::string::npos);
    EXPECT_NE(result.failure_reason.find("context length is 8"), std::string::npos);
}

TEST(Test__BenchmarkRunnerCPU, UsesOrchestratedDecodeStepWhenAvailable)
{
    auto runner = std::make_shared<MockOrchestratedDecodeRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.decode_success);
    EXPECT_EQ(result.decode_tokens, 3);
    EXPECT_EQ(result.decode_after_prefill_tokens, 2)
        << "The first emitted token is sampled from terminal prefill logits";
    EXPECT_GT(result.decode_after_prefill_tokens_per_sec, 0.0);
    EXPECT_LT(
        result.decode_after_prefill_tokens_per_sec,
        result.decode_tokens_per_sec)
        << "Short-run headline throughput must remain distinguishable from "
           "post-prefill decode work";
    EXPECT_THAT(result.generated_token_ids, ::testing::ElementsAre(14, 10, 11))
        << "Benchmark JSON should report the final measured iteration, not warmup output";
    EXPECT_TRUE(runner->samplingParamsSet());
    EXPECT_EQ(runner->lastTemperature(), 0.0f);
    EXPECT_GT(runner->decodeStepCalls(), 0);
    EXPECT_GT(runner->maintenanceCalls(), 0);
    EXPECT_EQ(runner->maintenanceTokens(), runner->decodedTokensReturned())
        << "Every grouped decode transaction must retire exactly the logical "
           "tokens it returned, including warmup and measured iterations.";
    EXPECT_EQ(runner->sampleGreedyCalls(), 0)
        << "BenchmarkRunner must not bypass orchestration decodeStep when it is available";
    EXPECT_TRUE(runner->configuredStopTokens().empty())
        << "Orchestrated MTP must receive the same ignore-stop policy as the benchmark host loop";
}

/**
 * @brief Prove the deprecated unified profiler measures orchestrated MTP through PerfStats.
 *
 * MTP returns from the orchestrated decode-step branch before the legacy
 * token-by-token sampler loop.  This regression prevents the profiling
 * migration from silently reporting no host transaction timing for the
 * production MTP path.
 */
TEST(Test__BenchmarkRunnerCPU, DeprecatedProfilerRecordsOrchestratedDecodeStepPerfStats)
{
    ScopedEnv deprecated_profiling("LLAMINAR_PROFILING", "1");
    PerfStatsCollector::reset();

    auto runner = std::make_shared<MockOrchestratedDecodeRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;

    const auto result = bench.run(config);
    ASSERT_TRUE(result.success) << result.failure_reason;

    const auto records = PerfStatsCollector::snapshot({"decode_loop"});
    const auto has_record = [&records](const std::string &name)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&name](const PerfStatRecord &record)
            {
                return record.domain == "decode_loop" &&
                       record.name == name &&
                       record.kind == PerfStatRecord::Kind::Timer &&
                       record.count > 0;
            });
    };

    EXPECT_TRUE(has_record("orchestrated_step"));
    EXPECT_TRUE(has_record("orchestrated_maintenance"));
    PerfStatsCollector::reset();
}

TEST(Test__BenchmarkRunnerCPU, PostWarmupCallbackSeesDecodeHistogramBeforePrefillCaptureClearsState)
{
    ScopedGpuGraphsSetting force_gpu_graph_warmup(true);
    auto runner = std::make_shared<MockBenchmarkEventOrderRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);
    bench.setPostWarmupCallback([runner]() {
        runner->recordCallback();
    });

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success) << result.failure_reason;

    const auto &events = runner->events();
    const auto callback_it = std::find(events.begin(), events.end(), "callback");
    ASSERT_NE(callback_it, events.end()) << ::testing::PrintToString(events);

    const size_t callback_index = static_cast<size_t>(callback_it - events.begin());
    size_t last_decode_before_callback = events.size();
    size_t last_clear_before_callback = events.size();
    size_t prefill_count_before_callback = 0;

    for (size_t i = 0; i < callback_index; ++i)
    {
        if (events[i] == "decode")
            last_decode_before_callback = i;
        if (events[i] == "clear")
            last_clear_before_callback = i;
        if (events[i] == "prefill")
            ++prefill_count_before_callback;
    }

    ASSERT_NE(last_decode_before_callback, events.size())
        << "Warmup decode must run before the post-warmup callback: "
        << ::testing::PrintToString(events);
    ASSERT_NE(last_clear_before_callback, events.size())
        << "Benchmark should have reset state before warmup: "
        << ::testing::PrintToString(events);
    EXPECT_LT(last_clear_before_callback, last_decode_before_callback)
        << "Prefill graph warmup must not clear the decode histogram before rebalance: "
        << ::testing::PrintToString(events);
    EXPECT_GE(prefill_count_before_callback, 2u)
        << "State-driven graph preparation and the ordinary warmup prefill must "
           "both precede post-warmup rebalance."
        << ::testing::PrintToString(events);
}

TEST(Test__BenchmarkRunnerCPU, PerfStatsResetDropsPostWarmupMoEStatsBeforeMeasuredIterations)
{
    ScopedEnv perf_stats_enabled("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto runner = std::make_shared<MockPerfStatsMaintenanceRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);
    bench.setPostWarmupCallback([]() {
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "post_warmup_setup_marker",
            1.0,
            "warmup",
            "cpu");
    });

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success) << result.failure_reason;
    EXPECT_GT(runner->maintenanceCalls(), 0);
    EXPECT_EQ(runner->maintenanceTokens(), runner->decodedTokensReturned())
        << "The measured maintenance path must preserve exact logical-token "
           "accounting while exporting diagnostics.";
    EXPECT_GT(runner->drainCalls(), 0);

    const auto records = PerfStatsCollector::snapshot({"moe_rebalance"});
    bool saw_post_warmup_marker = false;
    bool saw_measured_maintenance_marker = false;
    bool saw_epilogue_drain_marker = false;
    for (const auto &record : records)
    {
        if (record.name == "post_warmup_setup_marker")
            saw_post_warmup_marker = true;
        if (record.name == "measured_decode_maintenance_marker")
            saw_measured_maintenance_marker = true;
        if (record.name == "epilogue_drain_marker")
            saw_epilogue_drain_marker = true;
    }

    EXPECT_FALSE(saw_post_warmup_marker)
        << "MoE setup/warmup counters should not pollute measured benchmark perfstats.";
    EXPECT_TRUE(saw_measured_maintenance_marker)
        << "Measured decode maintenance counters should remain after the pre-iteration reset.";
    EXPECT_TRUE(saw_epilogue_drain_marker)
        << "Benchmark epilogue should drain completed async maintenance diagnostics before JSON export.";

    PerfStatsCollector::reset();
}

TEST(Test__BenchmarkRunnerCPU,
     MovementAttributionUsesTypedOwnerInsteadOfPerfStatsLedger)
{
    ScopedEnv perf_stats_enabled("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();
    auto runner = std::make_shared<MockTypedOptimizationStatusRunner>();
    BenchmarkRunner benchmark(runner, createMockTokenizer());

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 4;
    config.mtp.enabled = true;

    const auto result = benchmark.run(config);
    ASSERT_TRUE(result.success) << result.failure_reason;
    ASSERT_FALSE(result.iterations.empty());
    for (const auto &iteration : result.iterations)
    {
        EXPECT_GT(iteration.dynamic_movement_transactions, 0u);
        EXPECT_LT(iteration.dynamic_movement_transactions, 100u)
            << "Inflated PerfStats telemetry became benchmark authority";
        EXPECT_EQ(
            iteration.dynamic_movement_commands,
            iteration.dynamic_movement_transactions * 2u);
        EXPECT_EQ(
            iteration.dynamic_physical_bytes,
            iteration.dynamic_movement_transactions * 64u);
        EXPECT_EQ(
            iteration.dynamic_promotions,
            iteration.dynamic_movement_transactions);
        EXPECT_EQ(
            iteration.dynamic_demotions,
            iteration.dynamic_movement_transactions);
        EXPECT_EQ(iteration.dynamic_same_priority_moves, 0u);
        EXPECT_GT(
            iteration.moe_runtime_movement_epoch,
            iteration.moe_runtime_movement_epoch_start);
    }
    PerfStatsCollector::reset();
}

TEST(Test__BenchmarkRunnerCPU, StaticWarmupRearmsPrefillGraphAfterDecodeWorkspaceGrowth)
{
    ScopedGpuGraphsSetting force_gpu_graph_warmup(true);
    auto runner = std::make_shared<MockBenchmarkEventOrderRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success) << result.failure_reason;

    const auto &events = runner->events();
    const auto reset_it = std::find(events.begin(), events.end(), "reset_stats");
    ASSERT_NE(reset_it, events.end()) << ::testing::PrintToString(events);

    size_t last_decode_before_reset = events.size();
    for (auto it = events.begin(); it != reset_it; ++it)
    {
        if (*it == "decode")
            last_decode_before_reset = static_cast<size_t>(it - events.begin());
    }
    ASSERT_NE(last_decode_before_reset, events.size())
        << "Warmup decode must run before executor stats reset: "
        << ::testing::PrintToString(events);

    size_t prefill_after_decode_before_reset = 0;
    for (size_t i = last_decode_before_reset + 1;
         i < static_cast<size_t>(reset_it - events.begin());
         ++i)
    {
        if (events[i] == "prefill")
            ++prefill_after_decode_before_reset;
    }

    EXPECT_GE(prefill_after_decode_before_reset, 1u)
        << "Static benchmark setup must re-check graph readiness after warmup decode "
           "without unconditionally replaying the prompt three times."
        << ::testing::PrintToString(events);
}

TEST(Test__BenchmarkRunnerCPU, PrefillOnlyBenchmarkSkipsDecodeHistogramCallback)
{
    ScopedGpuGraphsSetting force_gpu_graph_warmup(true);
    auto runner = std::make_shared<MockBenchmarkEventOrderRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);
    bench.setPostWarmupCallback([runner]() {
        runner->recordCallback();
    });

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 0;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success) << result.failure_reason;
    EXPECT_TRUE(result.prefill_success);
    EXPECT_EQ(result.decode_tokens, 0);

    const auto &events = runner->events();
    EXPECT_EQ(std::find(events.begin(), events.end(), "callback"), events.end())
        << "MoE post-warmup rebalance callbacks require a decode histogram producer stream; "
           "prefill-only benchmark runs must skip them."
        << ::testing::PrintToString(events);
    EXPECT_EQ(std::find(events.begin(), events.end(), "decode"), events.end())
        << "n_predict=0 should remain prefill-only."
        << ::testing::PrintToString(events);
}

TEST(Test__BenchmarkRunnerCPU, UsesRequestBatchedDecodeStepWhenMTPBatchRequested)
{
    auto runner = std::make_shared<MockBatchedOrchestratedDecodeRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;
    config.mtp.max_request_batch = 2;
    config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
    config.temperature = 0.7f;
    config.top_k = 32;
    config.top_p = 0.9f;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success) << result.failure_reason;
    EXPECT_TRUE(result.decode_success);
    EXPECT_EQ(result.decode_tokens, 6)
        << "Request-batched decode reports aggregate generated tokens across requests";
    EXPECT_THAT(result.generated_token_ids, ::testing::ElementsAre(100, 101, 102))
        << "Human-readable generated tokens stay request-0 only";
    EXPECT_EQ(runner->lastPrefillBatch(), 2);
    EXPECT_EQ(runner->lastRequestBatch(), 2);
    EXPECT_GT(runner->batchPrefillCalls(), 0);
    EXPECT_GT(runner->batchDecodeStepCalls(), 0);
    EXPECT_GT(runner->maintenanceCalls(), 0);
    EXPECT_EQ(runner->maintenanceTokens(), runner->decodedTokensReturned())
        << "Request-batched maintenance must account for every logical token "
           "across all request rows, not merely one host boundary.";
    EXPECT_EQ(runner->forwardCount(), 0)
        << "Request-batched benchmark must not prefill only request 0";
    EXPECT_EQ(runner->singleDecodeStepCalls(), 0)
        << "max_request_batch > 1 must not silently fall through to single-request decode";
    ASSERT_TRUE(runner->samplingParamsSet());
    EXPECT_EQ(runner->lastSamplingParams().temperature, 0.7f);
    EXPECT_EQ(runner->lastSamplingParams().top_k, 32);
    EXPECT_EQ(runner->lastSamplingParams().top_p, 0.9f);
}

/**
 * @brief Prove request-batched MTP exposes its distinct host transaction timing.
 */
TEST(Test__BenchmarkRunnerCPU, DeprecatedProfilerRecordsRequestBatchedDecodePerfStats)
{
    ScopedEnv deprecated_profiling("LLAMINAR_PROFILING", "1");
    PerfStatsCollector::reset();

    auto runner = std::make_shared<MockBatchedOrchestratedDecodeRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;
    config.mtp.max_request_batch = 2;
    config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;

    const auto result = bench.run(config);
    ASSERT_TRUE(result.success) << result.failure_reason;

    const auto records = PerfStatsCollector::snapshot({"decode_loop"});
    const auto has_record = [&records](const std::string &name)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&name](const PerfStatRecord &record)
            {
                return record.domain == "decode_loop" &&
                       record.name == name &&
                       record.kind == PerfStatRecord::Kind::Timer &&
                       record.count > 0;
            });
    };

    EXPECT_TRUE(has_record("request_batch_step"));
    EXPECT_TRUE(has_record("request_batch_maintenance"));
    PerfStatsCollector::reset();
}

TEST(Test__BenchmarkRunnerCPU, FailsRequestBatchedPrefillWhenRunnerDoesNotOptIn)
{
    auto runner = std::make_shared<MockOrchestratedDecodeRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 2;
    config.mtp.enabled = true;
    config.mtp.max_request_batch = 2;

    auto result = bench.run(config);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.prefill_success);
    EXPECT_FALSE(result.decode_success);
    EXPECT_NE(result.failure_reason.find("request-batched benchmark prefill unsupported"),
              std::string::npos);
    EXPECT_EQ(runner->forwardCount(), 0)
        << "Unsupported request batching must hard-fail before single-request prefill";
    EXPECT_EQ(runner->decodeStepCalls(), 0)
        << "Unsupported request batching must fail before decode starts";
}

TEST(Test__BenchmarkRunnerCPU, FailsRequestBatchedDecodeWhenRunnerDoesNotOptIn)
{
    auto runner = std::make_shared<MockBatchPrefillOnlyRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 2;
    config.mtp.enabled = true;
    config.mtp.max_request_batch = 2;

    auto result = bench.run(config);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.decode_success);
    EXPECT_NE(result.failure_reason.find("request-batched benchmark decode unsupported"),
              std::string::npos);
    EXPECT_GT(runner->batchPrefillCalls(), 0);
    EXPECT_EQ(runner->lastPrefillBatch(), 2);
    EXPECT_EQ(runner->decodeStepCalls(), 0)
        << "Unsupported request batching must hard-fail instead of measuring request-count one";
}

TEST(Test__BenchmarkRunnerCPU, AdapterForwardsRequestBatchedDecodeContract)
{
    MockOrchestrationRunner orch;
    InferenceRunnerAdapter adapter(&orch);

    const std::vector<int32_t> stop_tokens = {7, 11};
    EXPECT_CALL(orch, setStopTokens(stop_tokens));
    EXPECT_TRUE(adapter.configureMTPRequestStopTokens(stop_tokens));

    EXPECT_CALL(orch, setSamplingParams(_))
        .WillOnce(Invoke([](const SamplingParams &params) {
            EXPECT_FLOAT_EQ(params.presence_penalty, 0.5f);
            EXPECT_FLOAT_EQ(params.frequency_penalty, 0.25f);
        }));
    EXPECT_TRUE(adapter.configureMTPRequestPenaltyPolicy(
        MTPRequestPenaltyPolicy{
            .presence_penalty = 0.5f,
            .frequency_penalty = 0.25f,
        }));

    EXPECT_CALL(orch, primaryDeviceId())
        .WillOnce(::testing::Return(DeviceId::cuda(1)));
    EXPECT_EQ(adapter.primaryDeviceId(), DeviceId::cuda(1))
        << "BenchmarkRunner relies on the adapter preserving GPU identity to skip full logits gathers";

    EXPECT_CALL(orch, moeRuntimeMovementEpoch())
        .WillOnce(::testing::Return(7u));
    EXPECT_EQ(adapter.moeRuntimeMovementEpoch(), 7u);
    const MoEOptimizationStatus optimization{
        .authority = MoEOptimizationAuthority::Device,
        .state = MoEOptimizationLifecycleState::Active,
        .published_movement_waves = 5u,
        .completed_movement = {
            .transactions = 5u,
            .commands = 11u,
            .physical_bytes = 4096u,
            .promotions = 2u,
            .demotions = 2u,
            .same_priority_moves = 7u,
        },
    };
    EXPECT_CALL(orch, moeOptimizationStatus())
        .WillOnce(::testing::Return(optimization));
    const auto adapted_optimization = adapter.moeOptimizationStatus();
    EXPECT_EQ(
        adapted_optimization.authority,
        MoEOptimizationAuthority::Device);
    EXPECT_EQ(adapted_optimization.published_movement_waves, 5u);
    EXPECT_EQ(adapted_optimization.completed_movement.commands, 11u);

    EXPECT_CALL(orch, supportsDecodeStepBatch(3))
        .WillOnce(::testing::Return(true));
    EXPECT_TRUE(adapter.supportsDecodeStepBatchForBenchmark(3));

    EXPECT_CALL(orch, supportsPrefillBatch(2))
        .WillOnce(::testing::Return(true));
    EXPECT_TRUE(adapter.supportsPrefillBatchForBenchmark(2));

    EXPECT_CALL(orch, prefillBatch(_))
        .WillOnce(Invoke([](const std::vector<std::vector<int32_t>> &token_batches) {
            EXPECT_THAT(token_batches, ::testing::ElementsAre(
                                          ::testing::ElementsAre(1, 2),
                                          ::testing::ElementsAre(3, 4)));
            return true;
        }));
    EXPECT_TRUE(adapter.prefillBatchForBenchmark({{1, 2}, {3, 4}}));

    GenerationResult first;
    first.tokens = {11, 12};
    first.is_complete = false;
    GenerationResult second;
    second.tokens = {21};
    second.is_complete = true;

    GenerationBatchResult batch;
    batch.requests = {first, second};

    EXPECT_CALL(orch, decodeStepBatch(2))
        .WillOnce(::testing::Return(batch));

    DecodeBatchStepOutput output = adapter.decodeBatchStepForBenchmark(2);

    EXPECT_TRUE(output.error.empty());
    ASSERT_EQ(output.tokens_by_request.size(), 2u);
    EXPECT_THAT(output.tokens_by_request[0], ::testing::ElementsAre(11, 12));
    EXPECT_THAT(output.tokens_by_request[1], ::testing::ElementsAre(21));
    EXPECT_THAT(output.is_complete_by_request, ::testing::ElementsAre(false, true));
}

TEST(Test__BenchmarkRunnerCPU, UsesRequestedSamplingParamsForSpeculativeMTPBenchmark)
{
    auto runner = std::make_shared<MockOrchestratedDecodeRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.seed = 123;
    config.temperature = 0.6f;
    config.top_k = 20;
    config.top_p = 0.95f;
    config.mtp.enabled = true;
    config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(runner->samplingParamsSet());
    const SamplingParams &params = runner->lastSamplingParams();
    EXPECT_EQ(params.temperature, 0.6f);
    EXPECT_EQ(params.top_k, 20);
    EXPECT_EQ(params.top_p, 0.95f);
    EXPECT_EQ(params.seed, 123u);
}

TEST(Test__BenchmarkRunnerCPU, AggregatesMeasuredIterationMTPStats)
{
    auto runner = std::make_shared<MockMeasuredMTPStatsRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;
    config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
    config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(runner->probeCount(), 3)
        << "Only measured iterations should be probed for aggregate reporting";

    EXPECT_EQ(result.prefix_state.mtp_draft_steps, 6u);
    EXPECT_EQ(result.prefix_state.mtp_accepted_tokens, 60u);
    EXPECT_EQ(result.prefix_state.mtp_rejected_tokens, 6u);
    EXPECT_EQ(result.prefix_state.mtp_rollbacks, 12u);
    EXPECT_EQ(result.prefix_state.mtp_bypasses, 2u);
    EXPECT_EQ(result.prefix_state.mtp_verifier_runs, 18u);
    EXPECT_EQ(result.prefix_state.mtp_verifier_token_count, 24u);
    EXPECT_EQ(result.prefix_state.mtp_stochastic_accept_tests, 30u);
    EXPECT_EQ(result.prefix_state.mtp_stochastic_accepts, 24u);
    EXPECT_EQ(result.prefix_state.mtp_stochastic_residual_samples, 6u);
    EXPECT_EQ(result.prefix_state.mtp_stochastic_terminal_samples, 12u);
    EXPECT_EQ(result.prefix_state.mtp_transaction_commits, 6u);
    EXPECT_EQ(result.prefix_state.mtp_transaction_rollbacks, 9u);
    EXPECT_EQ(result.prefix_state.mtp_transaction_validation_failures, 12u);
    EXPECT_EQ(result.prefix_state.mtp_unsafe_verifier_state_rejections, 15u);
    EXPECT_EQ(result.prefix_state.mtp_depth_policy_windows, 36u);
    EXPECT_EQ(result.prefix_state.mtp_depth_policy_updates, 6u);
    EXPECT_EQ(result.prefix_state.mtp_depth_policy_promotions, 18u);
    EXPECT_EQ(result.prefix_state.mtp_depth_policy_demotions, 21u);
    EXPECT_EQ(result.prefix_state.mtp_depth_policy_observe_recommendations, 24u);

    EXPECT_EQ(result.prefix_state.mtp_current_depth, 3)
        << "Current depth remains the latest measured request's diagnostic value";
    EXPECT_EQ(result.prefix_state.mtp_request.current_depth, 3);
    EXPECT_EQ(result.prefix_state.mtp_request.last_depth_policy_reason, "probe_3");
    EXPECT_EQ(result.prefix_state.mtp_request.accepted_tokens, 60u);
    EXPECT_EQ(result.prefix_state.mtp_request.rejected_tokens, 6u);
    EXPECT_DOUBLE_EQ(result.prefix_state.mtp_request.acceptance_rate, 60.0 / 66.0);
    EXPECT_EQ(result.prefix_state.mtp_request.stochastic_accept_tests, 30u);
    EXPECT_EQ(result.prefix_state.mtp_request.stochastic_accepts, 24u);
    EXPECT_DOUBLE_EQ(result.prefix_state.mtp_request.stochastic_acceptance_rate, 0.8);

    const auto doc = nlohmann::json::parse(benchmarkResultToJsonString(result, &config));
    EXPECT_EQ(doc.at("mtp").at("accepted_tokens"), 60);
    EXPECT_EQ(doc.at("mtp").at("rejected_tokens"), 6);
    EXPECT_DOUBLE_EQ(doc.at("mtp").at("acceptance_rate").get<double>(), 60.0 / 66.0);
    EXPECT_EQ(doc.at("mtp").at("request").at("stochastic_accepts"), 24);
    EXPECT_DOUBLE_EQ(doc.at("mtp").at("request").at("stochastic_acceptance_rate").get<double>(),
                     0.8);
}

/**
 * @brief Verify benchmark captures prefix-cache and MTP observability.
 */
TEST(Test__BenchmarkRunnerCPU, CapturesPrefixAndMTPStats)
{
    auto runner = std::make_shared<MockStatsInferenceRunner>();
    runner->snapshot.prefix_cache_config_enabled = true;
    runner->snapshot.prefix_cache_ready = true;
    runner->snapshot.prefix_cache_bypassed = true;
    runner->snapshot.prefix_cache_bypass_reason = "RAM budget cannot hold one complete prefix block";
    runner->snapshot.prefix_cache_lookups = 4;
    runner->snapshot.prefix_cache_hits = 2;
    runner->snapshot.prefix_cache_partial_hits = 1;
    runner->snapshot.prefix_cache_misses = 1;
    runner->snapshot.prefix_cache_matched_blocks = 3;
    runner->snapshot.prefix_cache_matched_tokens = 6;
    runner->snapshot.prefix_cache_stores = 5;
    runner->snapshot.prefix_cache_ram_bytes = 4096;
    runner->snapshot.prefix_cache_terminal_state_hits = 2;
    runner->snapshot.prefix_cache_bypasses = 1;
    runner->snapshot.prefix_cache_unsupported_backend_bypasses = 0;
    runner->snapshot.prefix_cache_fingerprint_bypasses = 0;
    runner->snapshot.prefix_cache_terminal_state_bypasses = 0;
    runner->snapshot.prefix_request.enabled = true;
    runner->snapshot.prefix_request.partial_hit = true;
    runner->snapshot.prefix_request.requested_tokens = 10;
    runner->snapshot.prefix_request.matched_tokens = 6;
    runner->snapshot.prefix_request.matched_blocks = 3;
    runner->snapshot.prefix_request.storage_tier = "ram";
    runner->snapshot.mtp_draft_steps = 3;
    runner->snapshot.mtp_accepted_tokens = 2;
    runner->snapshot.mtp_rejected_tokens = 1;
    runner->snapshot.mtp_rollbacks = 3;
    runner->snapshot.mtp_config_enabled = true;
    runner->snapshot.mtp_bypassed = true;
    runner->snapshot.mtp_bypass_reason = "sampling is not greedy";
    runner->snapshot.mtp_bypasses = 1;
    runner->snapshot.mtp_verifier_runs = 4;
    runner->snapshot.mtp_verifier_token_count = 8;
    runner->snapshot.mtp_stochastic_accept_tests = 3;
    runner->snapshot.mtp_stochastic_accepts = 2;
    runner->snapshot.mtp_stochastic_residual_samples = 1;
    runner->snapshot.mtp_stochastic_terminal_samples = 1;
    runner->snapshot.mtp_depth_policy_windows = 2;
    runner->snapshot.mtp_depth_policy_updates = 1;
    runner->snapshot.mtp_depth_policy_demotions = 1;
    runner->snapshot.mtp_current_depth = 1;
    runner->snapshot.mtp_min_depth = 1;
    runner->snapshot.mtp_max_depth = 3;
    runner->snapshot.mtp_request.enabled = true;
    runner->snapshot.mtp_request.bypassed = true;
    runner->snapshot.mtp_request.bypass_reason = "sampling is not greedy";
    runner->snapshot.mtp_request.verify_mode = "speculative-sampling";
    runner->snapshot.mtp_request.stochastic_verify = true;
    runner->snapshot.mtp_request.adaptive_depth_enabled = true;
    runner->snapshot.mtp_request.depth_policy_mode = "dynamic";
    runner->snapshot.mtp_request.current_depth = 1;
    runner->snapshot.mtp_request.min_depth = 1;
    runner->snapshot.mtp_request.max_depth = 3;
    runner->snapshot.mtp_request.depth_policy_updates = 1;
    runner->snapshot.mtp_request.last_depth_policy_reason = "demote_zero_accept_rate";
    runner->snapshot.mtp_request.draft_steps = 3;
    runner->snapshot.mtp_request.accepted_tokens = 2;
    runner->snapshot.mtp_request.rejected_tokens = 1;
    runner->snapshot.mtp_request.rollbacks = 3;
    runner->snapshot.mtp_request.acceptance_rate = 2.0 / 3.0;
    runner->snapshot.mtp_request.stochastic_accept_tests = 3;
    runner->snapshot.mtp_request.stochastic_accepts = 2;
    runner->snapshot.mtp_request.stochastic_residual_samples = 1;
    runner->snapshot.mtp_request.stochastic_terminal_samples = 1;
    runner->snapshot.mtp_request.stochastic_acceptance_rate = 2.0 / 3.0;
    runner->snapshot.prefill_chunk_schedules = 2;
    runner->snapshot.prefill_chunk_successful_schedules = 1;
    runner->snapshot.prefill_chunks = 3;
    runner->snapshot.prefill_chunk_real_tokens = 512;
    runner->snapshot.prefill_chunk_padded_tokens = 32;
    runner->snapshot.prefill_chunk_failures = 1;

    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 1;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.prefix_state.prefix_cache_lookups, 4u);
    EXPECT_EQ(result.prefix_state.prefix_cache_hits, 2u);
    EXPECT_EQ(result.prefix_state.prefix_cache_partial_hits, 1u);
    EXPECT_EQ(result.prefix_state.prefix_cache_matched_tokens, 6u);
    EXPECT_EQ(result.prefix_state.prefix_cache_terminal_state_hits, 2u);
    EXPECT_TRUE(result.prefix_state.prefix_cache_bypassed);
    EXPECT_EQ(result.prefix_state.prefix_cache_bypasses, 1u);
    EXPECT_TRUE(result.prefix_state.prefix_request.partial_hit);
    EXPECT_EQ(result.prefix_state.prefix_request.matched_tokens, 6);
    EXPECT_EQ(result.prefix_state.mtp_draft_steps, 9u);
    EXPECT_EQ(result.prefix_state.mtp_rejected_tokens, 3u);
    EXPECT_TRUE(result.prefix_state.mtp_bypassed);
    EXPECT_EQ(result.prefix_state.mtp_bypasses, 3u);
    EXPECT_EQ(result.prefix_state.mtp_request.accepted_tokens, 6u);
    EXPECT_DOUBLE_EQ(result.prefix_state.mtp_request.acceptance_rate, 2.0 / 3.0);
    EXPECT_EQ(result.prefix_state.mtp_request.verify_mode, "speculative-sampling");
    EXPECT_EQ(result.prefix_state.mtp_request.stochastic_residual_samples, 3u);
    EXPECT_EQ(result.prefix_state.mtp_verifier_runs, 12u);
    EXPECT_EQ(result.prefix_state.mtp_verifier_token_count, 24u);
    EXPECT_EQ(result.prefix_state.mtp_stochastic_accept_tests, 9u);
    EXPECT_EQ(result.prefix_state.mtp_depth_policy_updates, 3u);
    EXPECT_EQ(result.prefix_state.mtp_current_depth, 1);
    EXPECT_EQ(result.prefix_state.mtp_max_depth, 3);
    EXPECT_EQ(result.prefix_state.prefill_chunk_schedules, 2u);
    EXPECT_EQ(result.prefix_state.prefill_chunk_successful_schedules, 1u);
    EXPECT_EQ(result.prefix_state.prefill_chunks, 3u);
    EXPECT_EQ(result.prefix_state.prefill_chunk_real_tokens, 512u);
    EXPECT_EQ(result.prefix_state.prefill_chunk_padded_tokens, 32u);
    EXPECT_EQ(result.prefix_state.prefill_chunk_failures, 1u);

    testing::internal::CaptureStdout();
    bench.printResults(result);
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("PREFIX / MTP STATE"), std::string::npos);
    EXPECT_NE(output.find("Lookup results"), std::string::npos);
    EXPECT_NE(output.find("Prefix request"), std::string::npos);
    EXPECT_NE(output.find("partial-hit"), std::string::npos);
    EXPECT_NE(output.find("Bypasses"), std::string::npos);
    EXPECT_NE(output.find("RAM budget cannot hold one complete prefix block"), std::string::npos);
    EXPECT_NE(output.find("sampling is not greedy"), std::string::npos);
    EXPECT_NE(output.find("MTP request"), std::string::npos);
    EXPECT_NE(output.find("66.67% acceptance"), std::string::npos);
    EXPECT_NE(output.find("depth_policy=dynamic"), std::string::npos);
    EXPECT_NE(output.find("updates=3"), std::string::npos);
    EXPECT_NE(output.find("MTP decode"), std::string::npos);
    EXPECT_NE(output.find("Prefill chunks"), std::string::npos);
    EXPECT_NE(output.find("1/2 schedules"), std::string::npos);
}

/**
 * @brief Verify benchmark JSON carries Phase 14 counters without log parsing.
 */
TEST(Test__BenchmarkRunnerCPU, SerializesMachineReadableBenchmarkJson)
{
    ScopedPrefillGraphRequiredSetting prefill_graph_not_required(false);
    ScopedEnv perf_stats_enabled("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();
    PerfStatsCollector::addCounter(
        "moe_rebalance",
        "gpu_direct_transfer_count",
        4.0,
        "rebalance",
        "cuda:1",
        {{"layer", "3"}, {"src", "cuda:0"}, {"dst", "cuda:1"}});
    PerfStatsCollector::recordTimingNs(
        "forward_graph",
        "full_graph_replay",
        2000,
        "decode",
        "cuda:0",
        {{"sync_scope", "explicit_stream"}});
    PerfStatsCollector::addCounter(
        "kernel",
        "rocm_moe_grouped_prefill_batch_invariant_calls",
        1.0,
        "moe",
        "rocm:0",
        {{"gateup_codebook_mask", "0x00000028"},
         {"down_codebook_mask", "0x00000008"},
         {"policy_source", "generic_mixed_codebooks"}});
    PerfStatsCollector::addCounter(
        "kernel",
        "cuda_moe_grouped_prefill_swiglu_path_calls",
        1.0,
        "moe",
        "cuda:0",
        {{"gateup_geometry_contract", "tensor_core_imma"},
         {"work_scheduler", "compact_directory_grid"},
         {"policy_source", "tensor_core_imma_prefill"}});
    PerfStatsCollector::addCounter("mtp", "draft_steps", 1.0, "decode");
    PerfStatsCollector::addCounter(
        "moe_overlay_controller",
        "dynamic_movement_transactions",
        2.0,
        "maintenance",
        "cuda:0",
        {{"policy_owner", "device"}});

    BenchmarkResult result;
    result.prefill_tokens = 10;
    result.prefill_time_ms = 4.0;
    result.prefill_tokens_per_sec = 2500.0;
    result.prefill_success = true;
    result.decode_tokens = 2;
    result.decode_time_ms = 2.0;
    result.decode_tokens_per_sec = 1000.0;
    result.decode_after_prefill_tokens = 1;
    result.decode_after_prefill_tokens_per_sec = 500.0;
    result.decode_token_latencies_ms = {0.9, 1.1};
    result.decode_latency_mean_ms = 1.0;
    result.decode_latency_p50_ms = 1.0;
    result.decode_latency_p90_ms = 1.08;
    result.decode_success = true;
    result.total_time_ms = 6.0;
    result.success = true;
    result.generated_text = "xy";
    result.generated_token_ids = {77, 88};
    result.prompt_source = BenchmarkPromptSource::File;
    result.prompt_file_path = "/tmp/fixed-prompt.txt";
    result.prompt_bytes = 22;
    result.prompt_sha256 =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    result.iterations.push_back(BenchmarkIterationResult{
        .iteration = 1,
        .prefill_tokens = 10,
        .prefill_time_ms = 4.5,
        .prefill_tokens_per_sec = 2222.222222222222,
        .decode_tokens = 2,
        .decode_time_ms = 2.2,
        .decode_tokens_per_sec = 909.090909090909,
        .decode_after_prefill_tokens = 1,
        .decode_after_prefill_tokens_per_sec = 454.5454545454545,
        .generated_token_ids = {77, 88},
        .moe_runtime_movement_epoch_start = 5,
        .moe_runtime_movement_epoch = 7,
        .dynamic_movement_transactions = 2,
        .dynamic_movement_commands = 9,
        .dynamic_physical_bytes = 4096,
        .dynamic_promotions = 3,
        .dynamic_demotions = 3,
        .dynamic_same_priority_moves = 3,
        .mtp_draft_steps = 4,
        .mtp_accepted_tokens = 3,
        .mtp_rejected_tokens = 1,
        .mtp_verifier_runs = 2,
        .mtp_verifier_token_count = 5,
        .decode_windows = {
            BenchmarkDecodeWindowResult{
                .start_token = 0,
                .token_count = 2,
                .time_ms = 2.2,
                .tokens_per_sec = 909.090909090909,
            }},
    });

    auto &snapshot = result.prefix_state;
    snapshot.initialized = true;
    snapshot.architecture = "mock_cpu";
    snapshot.execution_path = "GRAPH";
    snapshot.primary_device = DeviceId::cpu();
    snapshot.current_position = 12;
    snapshot.moe_runtime_movement_epoch = 7;
    snapshot.prefix_cache_config_enabled = true;
    snapshot.prefix_cache_ready = true;
    snapshot.prefix_cache_lookups = 3;
    snapshot.prefix_cache_hits = 1;
    snapshot.prefix_cache_partial_hits = 1;
    snapshot.prefix_cache_misses = 1;
    snapshot.prefix_cache_matched_blocks = 2;
    snapshot.prefix_cache_matched_tokens = 8;
    snapshot.prefix_cache_stores = 4;
    snapshot.prefix_cache_ram_bytes = 8192;
    snapshot.prefix_cache_device_bytes = 4096;
    snapshot.prefix_cache_disk_bytes = 2048;
    snapshot.prefix_cache_hybrid_state_bytes = 128;
    snapshot.prefix_cache_mtp_state_bytes = 256;
    snapshot.prefix_cache_bypasses = 1;
    snapshot.prefix_cache_unsupported_backend_bypasses = 1;
    snapshot.prefix_request.enabled = true;
    snapshot.prefix_request.partial_hit = true;
    snapshot.prefix_request.requested_tokens = 10;
    snapshot.prefix_request.matched_tokens = 8;
    snapshot.prefix_request.matched_blocks = 2;
    snapshot.prefix_request.terminal_logits_restored = true;
    snapshot.prefix_request.storage_tier = "ram";
    snapshot.mtp_config_enabled = true;
    snapshot.mtp_draft_steps = 4;
    snapshot.mtp_accepted_tokens = 3;
    snapshot.mtp_rejected_tokens = 1;
    snapshot.mtp_rollbacks = 1;
    snapshot.mtp_verifier_runs = 2;
    snapshot.mtp_verifier_token_count = 5;
    snapshot.mtp_depth_policy_windows = 2;
    snapshot.mtp_depth_policy_updates = 1;
    snapshot.mtp_depth_policy_promotions = 1;
    snapshot.mtp_current_depth = 2;
    snapshot.mtp_min_depth = 1;
    snapshot.mtp_max_depth = 3;
    snapshot.mtp_stochastic_accept_tests = 4;
    snapshot.mtp_stochastic_accepts = 3;
    snapshot.mtp_stochastic_residual_samples = 1;
    snapshot.mtp_stochastic_terminal_samples = 2;
    snapshot.mtp_request.enabled = true;
    snapshot.mtp_request.verify_mode = "speculative-sampling";
    snapshot.mtp_request.stochastic_verify = true;
    snapshot.mtp_request.adaptive_depth_enabled = true;
    snapshot.mtp_request.depth_policy_mode = "dynamic";
    snapshot.mtp_request.current_depth = 2;
    snapshot.mtp_request.min_depth = 1;
    snapshot.mtp_request.max_depth = 3;
    snapshot.mtp_request.depth_policy_updates = 1;
    snapshot.mtp_request.last_depth_policy_reason = "promote_full_accept_rate";
    snapshot.mtp_request.draft_steps = 4;
    snapshot.mtp_request.accepted_tokens = 3;
    snapshot.mtp_request.rejected_tokens = 1;
    snapshot.mtp_request.rollbacks = 1;
    snapshot.mtp_request.acceptance_rate = 0.75;
    snapshot.mtp_request.stochastic_accept_tests = 4;
    snapshot.mtp_request.stochastic_accepts = 3;
    snapshot.mtp_request.stochastic_residual_samples = 1;
    snapshot.mtp_request.stochastic_terminal_samples = 2;
    snapshot.mtp_request.stochastic_acceptance_rate = 0.75;
    snapshot.prefill_chunk_schedules = 2;
    snapshot.prefill_chunk_successful_schedules = 2;
    snapshot.prefill_chunks = 5;
    snapshot.prefill_chunk_real_tokens = 1024;
    snapshot.prefill_chunk_padded_tokens = 64;
    PrefillGraphRuntimeProbe prefill_graph;
    prefill_graph.forward_cache_valid = true;
    prefill_graph.prefill_cache_initialized = true;
    prefill_graph.phase = "ready";
    prefill_graph.cache_size = 1;
    prefill_graph.node_count = 1234;
    prefill_graph.replay_count = 2;
    prefill_graph.warmup_count = 3;
    prefill_graph.capture_count = 1;
    prefill_graph.observation_valid = true;
    prefill_graph.bucket_seq_len = 1024;
    prefill_graph.real_token_count = 1000;
    prefill_graph.domain_id = "qwen36_moe_cuda_hot";
    prefill_graph.participant_id = 1;
    prefill_graph.capture_phase = "replay";
    prefill_graph.recapture_reason = "none";
    snapshot.prefill_graphs.push_back(prefill_graph);

    OrchestrationConfig config;
    config.benchmark_mode = true;
    config.model_path = "model.gguf";
    config.n_predict = 2;
    config.prefix_cache.enabled = true;
    config.mtp.enabled = true;
    config.mtp.draft_tokens = 3;
    config.mtp.graph_capacity_draft_tokens = 15;
    config.mtp.max_request_batch = 4;
    config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    config.mtp.depth_policy.max_depth = 3;
    config.mtp.depth_policy.window_size = 8;
    config.benchmark_json_output_path = "/tmp/bench.json";

    const auto doc = nlohmann::json::parse(benchmarkResultToJsonString(result, &config));

    EXPECT_EQ(doc.at("schema"), "llaminar.benchmark.v1");
    EXPECT_TRUE(doc.at("success").get<bool>());
    EXPECT_EQ(doc.at("measurement_iterations"), 3);
    EXPECT_EQ(doc.at("warmup_iterations"), 1);
    EXPECT_EQ(doc.at("prompt").at("source"), "file");
    EXPECT_EQ(doc.at("prompt").at("file_path"), "/tmp/fixed-prompt.txt");
    EXPECT_EQ(doc.at("prompt").at("bytes"), 22);
    EXPECT_EQ(
        doc.at("prompt").at("sha256"),
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    EXPECT_EQ(doc.at("tokens").at("prefill"), 10);
    EXPECT_EQ(doc.at("tokens").at("decode"), 2);
    EXPECT_EQ(doc.at("tokens").at("decode_after_prefill"), 1);
    EXPECT_DOUBLE_EQ(doc.at("timing_ms").at("total").get<double>(), 6.0);
    EXPECT_DOUBLE_EQ(doc.at("throughput_tokens_per_sec").at("overall").get<double>(), 2000.0);
    EXPECT_DOUBLE_EQ(
        doc.at("throughput_tokens_per_sec")
            .at("decode_after_prefill")
            .get<double>(),
        500.0);
    EXPECT_DOUBLE_EQ(doc.at("decode_latency_ms").at("mean").get<double>(), 1.0);
    EXPECT_DOUBLE_EQ(doc.at("decode_latency_ms").at("p50").get<double>(), 1.0);
    EXPECT_DOUBLE_EQ(doc.at("decode_latency_ms").at("p90").get<double>(), 1.08);
    EXPECT_EQ(doc.at("decode_latency_ms").at("samples"), 2);
    EXPECT_EQ(doc.at("generated_text_bytes"), 2);
    EXPECT_EQ(doc.at("generated_token_ids"), nlohmann::json::array({77, 88}));
    ASSERT_EQ(doc.at("iterations").size(), 1u);
    const auto &iteration = doc.at("iterations").front();
    EXPECT_EQ(iteration.at("iteration"), 1);
    EXPECT_DOUBLE_EQ(
        iteration.at("throughput_tokens_per_sec").at("decode").get<double>(),
        909.090909090909);
    EXPECT_DOUBLE_EQ(
        iteration.at("throughput_tokens_per_sec")
            .at("decode_after_prefill")
            .get<double>(),
        454.5454545454545);
    EXPECT_EQ(
        iteration.at("generated_token_ids"),
        nlohmann::json::array({77, 88}));
    EXPECT_EQ(iteration.at("moe_runtime_movement_epoch_start"), 5);
    EXPECT_EQ(iteration.at("moe_runtime_movement_epoch"), 7);
    EXPECT_EQ(
        iteration.at("completed_dynamic_movement").at("transactions"), 2);
    EXPECT_EQ(
        iteration.at("completed_dynamic_movement").at("commands"), 9);
    EXPECT_EQ(
        iteration.at("completed_dynamic_movement").at("promotions"), 3);
    EXPECT_EQ(
        iteration.at("completed_dynamic_movement").at("same_priority_moves"),
        3);
    EXPECT_EQ(iteration.at("mtp").at("draft_steps"), 4);
    EXPECT_EQ(iteration.at("mtp").at("accepted_tokens"), 3);
    EXPECT_EQ(iteration.at("mtp").at("rejected_tokens"), 1);
    EXPECT_EQ(iteration.at("mtp").at("verifier_runs"), 2);
    EXPECT_EQ(iteration.at("mtp").at("verifier_token_count"), 5);
    EXPECT_DOUBLE_EQ(
        iteration.at("mtp").at("acceptance_rate").get<double>(), 0.75);
    ASSERT_EQ(iteration.at("decode_windows").size(), 1u);
    EXPECT_EQ(iteration.at("decode_windows").front().at("start_token"), 0);
    EXPECT_EQ(iteration.at("decode_windows").front().at("token_count"), 2);
    EXPECT_EQ(doc.at("runtime_state").at("moe_runtime_movement_epoch"), 7);
    const auto &perf_records = doc.at("perf_stats").at("records");
    EXPECT_TRUE(std::any_of(
        perf_records.begin(),
        perf_records.end(),
        [](const auto &record)
        {
            return record.at("domain") == "moe_overlay_controller" &&
                   record.at("name") == "dynamic_movement_transactions" &&
                   record.at("value") == 2.0;
        })) << "Machine-readable benchmark evidence omitted completed "
               "device-owned movement";

    const auto &prefix = doc.at("prefix_cache");
    EXPECT_TRUE(prefix.at("config_enabled").get<bool>());
    EXPECT_EQ(prefix.at("lookups"), 3);
    EXPECT_EQ(prefix.at("matched_tokens"), 8);
    EXPECT_EQ(prefix.at("ram_bytes"), 8192);
    EXPECT_EQ(prefix.at("device_bytes"), 4096);
    EXPECT_EQ(prefix.at("request").at("storage_tier"), "ram");
    EXPECT_TRUE(prefix.at("request").at("partial_hit").get<bool>());
    EXPECT_TRUE(prefix.at("request").at("terminal_logits_restored").get<bool>());

    const auto &mtp = doc.at("mtp");
    EXPECT_EQ(mtp.at("draft_steps"), 4);
    EXPECT_EQ(mtp.at("accepted_tokens"), 3);
    EXPECT_DOUBLE_EQ(mtp.at("acceptance_rate").get<double>(), 0.75);
    EXPECT_EQ(mtp.at("stochastic_accept_tests"), 4);
    EXPECT_EQ(mtp.at("stochastic_accepts"), 3);
    EXPECT_EQ(mtp.at("stochastic_residual_samples"), 1);
    EXPECT_EQ(mtp.at("stochastic_terminal_samples"), 2);
    EXPECT_EQ(mtp.at("current_depth"), 2);
    EXPECT_EQ(mtp.at("max_depth"), 3);
    EXPECT_EQ(mtp.at("depth_policy_updates"), 1);
    EXPECT_EQ(mtp.at("depth_policy_promotions"), 1);
    EXPECT_TRUE(mtp.at("request").at("adaptive_depth_enabled").get<bool>());
    EXPECT_EQ(mtp.at("request").at("verify_mode"), "speculative-sampling");
    EXPECT_TRUE(mtp.at("request").at("stochastic_verify").get<bool>());
    EXPECT_EQ(mtp.at("request").at("depth_policy_mode"), "dynamic");
    EXPECT_EQ(mtp.at("request").at("last_depth_policy_reason"), "promote_full_accept_rate");
    EXPECT_EQ(mtp.at("request").at("accepted_tokens"), 3);
    EXPECT_EQ(mtp.at("request").at("stochastic_accept_tests"), 4);
    EXPECT_EQ(mtp.at("request").at("stochastic_residual_samples"), 1);
    EXPECT_DOUBLE_EQ(mtp.at("request").at("stochastic_acceptance_rate").get<double>(), 0.75);

    EXPECT_EQ(doc.at("prefill_chunks").at("chunks"), 5);
    EXPECT_EQ(doc.at("prefill_chunks").at("padded_tokens"), 64);
    const auto &prefill_graphs = doc.at("prefill_graphs");
    EXPECT_FALSE(prefill_graphs.at("required").get<bool>());
    EXPECT_TRUE(prefill_graphs.at("captured_or_replayed").get<bool>());
    EXPECT_EQ(prefill_graphs.at("entries"), 1);
    EXPECT_EQ(prefill_graphs.at("ready_entries"), 1);
    EXPECT_EQ(prefill_graphs.at("warmups"), 3);
    EXPECT_EQ(prefill_graphs.at("captures"), 1);
    EXPECT_EQ(prefill_graphs.at("replays"), 2);
    ASSERT_EQ(prefill_graphs.at("probes").size(), 1u);
    EXPECT_EQ(prefill_graphs.at("probes")[0].at("domain_id"), "qwen36_moe_cuda_hot");
    EXPECT_EQ(prefill_graphs.at("probes")[0].at("capture_phase"), "replay");
    EXPECT_EQ(prefill_graphs.at("probes")[0].at("node_count"), 1234);
    EXPECT_TRUE(doc.at("config").at("prefix_cache_enabled").get<bool>());
    EXPECT_TRUE(doc.at("config").at("mtp_enabled").get<bool>());
    EXPECT_EQ(doc.at("config").at("mtp_draft_tokens"), 3);
    EXPECT_EQ(doc.at("config").at("mtp_graph_capacity_draft_tokens"), 15);
    EXPECT_EQ(doc.at("config").at("mtp_max_request_batch"), 4);
    EXPECT_EQ(doc.at("config").at("mtp_depth_policy"), "dynamic");
    EXPECT_EQ(doc.at("config").at("mtp_max_draft_tokens"), 3);
    EXPECT_EQ(doc.at("config").at("mtp_depth_window"), 8);
    EXPECT_EQ(doc.at("config").at("mtp_depth_promote_windows"), 3);
    EXPECT_EQ(doc.at("config").at("benchmark_json_output_path"), "/tmp/bench.json");

    const auto &perf_stats = doc.at("perf_stats");
    EXPECT_EQ(perf_stats.at("schema"), "llaminar.perf_stats.v1");
    EXPECT_TRUE(perf_stats.at("enabled").get<bool>());
    EXPECT_NE(std::find(perf_stats.at("filters").begin(),
                        perf_stats.at("filters").end(),
                        "moe_rebalance"),
              perf_stats.at("filters").end());
    EXPECT_NE(std::find(perf_stats.at("filters").begin(),
                        perf_stats.at("filters").end(),
                        "forward_graph"),
              perf_stats.at("filters").end());
    EXPECT_NE(std::find(perf_stats.at("filters").begin(),
                        perf_stats.at("filters").end(),
                        "kernel"),
              perf_stats.at("filters").end());

    bool saw_gpu_direct_transfer = false;
    bool saw_full_graph_replay = false;
    bool saw_mixed_moe_kernel_route = false;
    bool saw_filtered_mtp = false;
    for (const auto &record : perf_stats.at("records"))
    {
        const std::string domain = record.at("domain").get<std::string>();
        const std::string name = record.at("name").get<std::string>();
        if (domain == "moe_rebalance" && name == "gpu_direct_transfer_count")
        {
            saw_gpu_direct_transfer = true;
            EXPECT_EQ(record.at("kind"), "counter");
            EXPECT_EQ(record.at("phase"), "rebalance");
            EXPECT_EQ(record.at("device"), "cuda:1");
            EXPECT_EQ(record.at("count"), 1);
            EXPECT_DOUBLE_EQ(record.at("value").get<double>(), 4.0);
            EXPECT_EQ(record.at("tags").at("src"), "cuda:0");
            EXPECT_EQ(record.at("tags").at("dst"), "cuda:1");
        }
        if (domain == "forward_graph" && name == "full_graph_replay")
        {
            saw_full_graph_replay = true;
            EXPECT_EQ(record.at("kind"), "timer");
            EXPECT_EQ(record.at("phase"), "decode");
            EXPECT_EQ(record.at("device"), "cuda:0");
            EXPECT_EQ(record.at("total_ns"), 2000);
            EXPECT_DOUBLE_EQ(record.at("total_ms").get<double>(), 0.002);
            EXPECT_DOUBLE_EQ(record.at("avg_us").get<double>(), 2.0);
        }
        if (domain == "kernel" &&
            name == "rocm_moe_grouped_prefill_batch_invariant_calls")
        {
            saw_mixed_moe_kernel_route = true;
            EXPECT_EQ(record.at("phase"), "moe");
            EXPECT_EQ(record.at("device"), "rocm:0");
            EXPECT_EQ(record.at("tags").at("gateup_codebook_mask"),
                      "0x00000028");
            EXPECT_EQ(record.at("tags").at("down_codebook_mask"),
                      "0x00000008");
            EXPECT_EQ(record.at("tags").at("policy_source"),
                      "generic_mixed_codebooks");
        }
        if (domain == "mtp")
            saw_filtered_mtp = true;
    }

    EXPECT_TRUE(saw_gpu_direct_transfer);
    EXPECT_TRUE(saw_full_graph_replay);
    EXPECT_TRUE(saw_mixed_moe_kernel_route);
    EXPECT_FALSE(saw_filtered_mtp);
    PerfStatsCollector::reset();
}

TEST(Test__BenchmarkRunnerCPU, ExportsResolvedMTPHardwareDefaultsAndOverrideProvenance)
{
    BenchmarkResult result;
    OrchestrationConfig config;
    config.mtp.depth_defaults_profile = MTPDepthDefaultsProfile::ROCmMI50;
    auto doc = nlohmann::json::parse(benchmarkResultToJsonString(result, &config));
    EXPECT_EQ(doc.at("config").at("mtp_depth_defaults_profile"), "rocm-mi50");
    EXPECT_EQ(doc.at("config").at("mtp_depth_demote_zero_accept"), 0.45);
    EXPECT_EQ(doc.at("config").at("mtp_depth_demote_zero_accept_source"), "hardware_default");
    config.mtp.depth_policy.demote_zero_accept_rate = 0.30;
    doc = nlohmann::json::parse(benchmarkResultToJsonString(result, &config));
    EXPECT_EQ(doc.at("config").at("mtp_depth_defaults_profile"), "rocm-mi50");
    EXPECT_EQ(doc.at("config").at("mtp_depth_demote_zero_accept"), 0.30);
    EXPECT_EQ(doc.at("config").at("mtp_depth_demote_zero_accept_source"), "explicit");
}

TEST(Test__BenchmarkRunnerCPU, RuntimeDebugParsesBenchmarkIterationOverrides)
{
    {
        ScopedEnv iterations("LLAMINAR_BENCHMARK_ITERATIONS", "1");
        ScopedEnv warmups("LLAMINAR_BENCHMARK_WARMUP_ITERATIONS", "0");
        ScopedEnv profiler_exit("LLAMINAR_PROFILER_NORMAL_EXIT", "1");
        mutableDebugEnv().runtime_debug.reload();

        EXPECT_EQ(debugEnv().runtime_debug.benchmark_iterations, 1);
        EXPECT_EQ(debugEnv().runtime_debug.benchmark_warmup_iterations, 0);
        EXPECT_TRUE(debugEnv().runtime_debug.profiler_normal_exit);
    }
    mutableDebugEnv().runtime_debug.reload();
    EXPECT_EQ(debugEnv().runtime_debug.benchmark_iterations, 3);
    EXPECT_EQ(debugEnv().runtime_debug.benchmark_warmup_iterations, 1);
    EXPECT_FALSE(debugEnv().runtime_debug.profiler_normal_exit);
}

TEST(Test__BenchmarkRunnerCPU, PreservesImmutableSetupEvidenceAcrossMeasuredReset)
{
    ScopedEnv perf_stats_enabled("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();
    PerfStatsCollector::addCounter(
        "memory",
        "workspace_block_bytes",
        4096.0,
        "allocate",
        "cuda:0",
        {{"buffer_count", "2"}});
    PerfStatsCollector::addCounter(
        "weight_loading",
        "graph_build_ms",
        125.0,
        "model_setup",
        "cuda:0",
        {{"source", "weight_loading_profiler"}});
    PerfStatsCollector::addCounter(
        "gpu_graph_inventory",
        "fragment_kernel_nodes",
        17.0,
        "setup",
        "cuda:0",
        {{"fragment", "all-position verifier forward"}});
    PerfStatsCollector::addCounter(
        "tp_allreduce_bom",
        "stages",
        1.0,
        "graph_setup",
        "cuda:0",
        {{"stage", "layer_0_gdn_wo_allreduce"},
         {"elements", "2048"}});
    PerfStatsCollector::addCounter(
        "tp_rooted_collective_bom",
        "calls",
        1.0,
        "graph_setup",
        "cuda:0",
        {{"stage", "layer_0_moe_canonical_routes_reduce_to_root"},
         {"operation", "reduce_sum"}});
    PerfStatsCollector::addCounter(
        "kernel",
        "rocm_moe_grouped_prefill_batch_invariant_calls",
        1.0,
        "moe",
        "rocm:0",
        {{"seq_len", "512"},
         {"gateup_codebook_mask", "0x00000028"},
         {"down_codebook_mask", "0x00000008"},
         {"policy_source", "generic_mixed_codebooks"}});
    PerfStatsCollector::addCounter(
        "kernel",
        "cuda_moe_grouped_prefill_swiglu_path_calls",
        1.0,
        "moe",
        "cuda:0",
        {{"gateup_geometry_contract", "tensor_core_imma"},
         {"work_scheduler", "compact_directory_grid"},
         {"policy_source", "tensor_core_imma_prefill"}});
    PerfStatsCollector::addCounter(
        "kernel",
        "cuda_fused_projection_stream_pool_calls",
        1.0,
        "gemm",
        "cuda:0",
        {{"mode", "decode"},
         {"m", "1"},
         {"projections", "3"},
         {"streams", "3"}});
    PerfStatsCollector::addCounter(
        "kernel",
        "cuda_native_vnni_small_m_fused_projection_calls",
        1.0,
        "gemm",
        "cuda:0",
        {{"m", "4"},
         {"projections", "2"},
         {"projection_schedule", "concurrent"}});
    PerfStatsCollector::addCounter(
        "kernel",
        "unrelated_warmup_kernel",
        1.0,
        "warmup",
        "rocm:0");
    PerfStatsCollector::addCounter("mtp", "draft_steps", 1.0, "decode");

    auto runner = std::make_shared<MockCPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();

    BenchmarkRunner bench(runner, tokenizer);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 1;

    auto result = bench.run(config);
    ASSERT_TRUE(result.success);

    const auto records = PerfStatsCollector::snapshot();
    const auto has_record = [&](const char *domain, const char *name)
    {
        return std::any_of(records.begin(), records.end(),
                           [&](const PerfStatRecord &record)
                           {
                               return record.domain == domain && record.name == name;
                           });
    };
    EXPECT_TRUE(has_record("memory", "workspace_block_bytes"))
        << "Benchmark JSON diagnostics need the allocation BOM after warmup reset";
    EXPECT_TRUE(has_record("weight_loading", "graph_build_ms"))
        << "Startup and eager graph-family costs must remain attributable in the benchmark artifact";
    EXPECT_TRUE(has_record("gpu_graph_inventory", "fragment_kernel_nodes"))
        << "Captured graph metadata is emitted during warmup and must survive into the benchmark artifact";
    EXPECT_TRUE(has_record("tp_allreduce_bom", "stages"))
        << "Captured allreduce stage/payload evidence must survive the measured reset";
    EXPECT_TRUE(has_record("tp_rooted_collective_bom", "calls"))
        << "Captured rooted-collective stage/payload evidence must survive the measured reset";
    EXPECT_TRUE(has_record(
        "kernel", "rocm_moe_grouped_prefill_batch_invariant_calls"))
        << "A captured MoE graph must retain its exact launch-policy identity";
    EXPECT_TRUE(has_record(
        "kernel", "cuda_moe_grouped_prefill_swiglu_path_calls"))
        << "A captured CUDA MoE graph must retain its persistent IMMA policy identity";
    EXPECT_TRUE(has_record(
        "kernel", "cuda_fused_projection_stream_pool_calls"))
        << "M=1 retained graphs must preserve their fused side-stream schedule evidence";
    EXPECT_TRUE(has_record(
        "kernel", "cuda_native_vnni_small_m_fused_projection_calls"))
        << "Grouped verifier graphs must preserve their fused side-stream schedule evidence";
    EXPECT_FALSE(has_record("kernel", "unrelated_warmup_kernel"))
        << "Retaining one immutable kernel route must not retain the whole noisy domain";
    EXPECT_FALSE(has_record("mtp", "draft_steps"))
        << "Non-preserved warmup counters should still be cleared before measurement";
    PerfStatsCollector::reset();
}
