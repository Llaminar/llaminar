/**
 * @file Test__BenchmarkRunnerCPU.cpp
 * @brief Unit tests verifying BenchmarkRunner handles CPU devices correctly
 *
 * Regression tests for:
 * - setSkipLogitsGatherDecode must NOT be enabled on CPU devices
 * - Decode must fall back to host-side argmax when sampleGreedyOnDevice() returns -1
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <algorithm>

#include "utils/BenchmarkRunner.h"
#include "config/OrchestrationConfig.h"
#include "backends/DeviceId.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockTokenizer.h"
#include "nlohmann/json.hpp"

using namespace llaminar2;
using namespace llaminar2::test;
using ::testing::_;
using ::testing::Return;

namespace
{

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

        void setSkipLogitsGatherPrefill(bool) override {}
        void setSuppressTimeline(bool) override {}
        void setAccumulatePrefill(bool) override {}

        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "mock_cpu"; }
        int get_position() const override { return 0; }

        // Test inspection
        bool skipLogitsGatherDecodeWasEnabled() const { return skip_logits_gather_decode_; }
        bool skipLogitsGatherDecodeWasCalled() const { return skip_logits_gather_decode_called_; }
        int forwardCount() const { return forward_count_; }

    private:
        std::vector<float> logits_;
        int forward_count_ = 0;
        int clear_count_ = 0;
        bool skip_logits_gather_decode_ = false;
        bool skip_logits_gather_decode_called_ = false;
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
            return true;
        }

        const float *logits() const override { return logits_.data(); }
        int vocab_size() const override { return VOCAB; }
        void clear_cache() override {}

        // GPU device — GPU argmax available
        DeviceId primaryDeviceId() const override { return DeviceId::cuda(0); }
        int sampleGreedyOnDevice() override { return GPU_ARGMAX_TOKEN; }

        void setSkipLogitsGatherDecode(bool skip) override
        {
            skip_logits_gather_decode_ = skip;
        }

        void setSkipLogitsGatherPrefill(bool) override {}
        void setSuppressTimeline(bool) override {}
        void setAccumulatePrefill(bool) override {}

        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "mock_gpu"; }
        int get_position() const override { return 0; }

        bool skipLogitsGatherDecodeWasEnabled() const { return skip_logits_gather_decode_; }

    private:
        std::vector<float> logits_;
        bool skip_logits_gather_decode_ = false;
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

        void setDecodeSamplingParams(const SamplingParams &params) override
        {
            sampling_params_set_ = true;
            last_temperature_ = params.temperature;
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
            return output;
        }

        bool maybeApplyDecodeBoundaryMaintenance() override
        {
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
        int sampleGreedyCalls() const { return sample_greedy_calls_; }
        bool samplingParamsSet() const { return sampling_params_set_; }
        float lastTemperature() const { return last_temperature_; }

    private:
        int decode_step_budget_ = 0;
        int decode_step_calls_ = 0;
        int maintenance_calls_ = 0;
        int sample_greedy_calls_ = 0;
        int emitted_tokens_ = 0;
        bool sampling_params_set_ = false;
        float last_temperature_ = 1.0f;
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

        ON_CALL(*tok, vocab_size())
            .WillByDefault(Return(100));

        return tok;
    }

} // namespace

// =============================================================================
// Tests
// =============================================================================

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
    auto mpi = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    BenchmarkRunner bench(runner, tokenizer, mpi);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;

    auto result = bench.run(config);

    // The flag must have been set to false (not true) for CPU
    EXPECT_TRUE(runner->skipLogitsGatherDecodeWasCalled())
        << "BenchmarkRunner must call setSkipLogitsGatherDecode";
    EXPECT_FALSE(runner->skipLogitsGatherDecodeWasEnabled())
        << "CPU device must NOT enable skip-logits-gather (no GPU argmax available)";
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
    auto mpi = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    BenchmarkRunner bench(runner, tokenizer, mpi);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;

    auto result = bench.run(config);

    // GPU runner should have skip-logits-gather enabled
    EXPECT_TRUE(runner->skipLogitsGatherDecodeWasEnabled())
        << "GPU device must enable skip-logits-gather for performance";
}

/**
 * @brief Verify CPU decode falls back to host-side argmax and succeeds.
 *
 * Regression test: BenchmarkRunner previously treated sampleGreedyOnDevice() == -1
 * as a hard error. Now it falls back to logits() + CPU argmax.
 */
TEST(Test__BenchmarkRunnerCPU, CPUDecodeSucceedsWithHostArgmax)
{
    auto runner = std::make_shared<MockCPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();
    auto mpi = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    BenchmarkRunner bench(runner, tokenizer, mpi);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 5;

    auto result = bench.run(config);

    // Benchmark must succeed — CPU decode via host argmax should work
    EXPECT_TRUE(result.success)
        << "CPU benchmark must succeed using host-side argmax fallback";
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
    auto mpi = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    BenchmarkRunner bench(runner, tokenizer, mpi);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 5;

    auto result = bench.run(config);

    EXPECT_TRUE(result.success)
        << "GPU benchmark must succeed using device-side argmax";
    EXPECT_TRUE(result.decode_success)
        << "GPU decode phase must succeed";
}

TEST(Test__BenchmarkRunnerCPU, FailsBeforePrefillWhenPromptExceedsContext)
{
    auto runner = std::make_shared<MockCPUInferenceRunner>();
    auto tokenizer = createMockTokenizer();
    ON_CALL(*tokenizer, encode(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3, 4, 5, 6}));
    auto mpi = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    BenchmarkRunner bench(runner, tokenizer, mpi);

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
    auto mpi = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    BenchmarkRunner bench(runner, tokenizer, mpi);

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
    auto mpi = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    BenchmarkRunner bench(runner, tokenizer, mpi);

    OrchestrationConfig config;
    config.prompt = "Hello world";
    config.n_predict = 3;
    config.mtp.enabled = true;

    auto result = bench.run(config);

    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.decode_success);
    EXPECT_EQ(result.decode_tokens, 3);
    EXPECT_TRUE(runner->samplingParamsSet());
    EXPECT_EQ(runner->lastTemperature(), 0.0f);
    EXPECT_GT(runner->decodeStepCalls(), 0);
    EXPECT_GT(runner->maintenanceCalls(), 0);
    EXPECT_EQ(runner->sampleGreedyCalls(), 0)
        << "BenchmarkRunner must not bypass orchestration decodeStep when it is available";
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
    runner->snapshot.mtp_depth_policy_windows = 2;
    runner->snapshot.mtp_depth_policy_updates = 1;
    runner->snapshot.mtp_depth_policy_demotions = 1;
    runner->snapshot.mtp_current_depth = 1;
    runner->snapshot.mtp_min_depth = 1;
    runner->snapshot.mtp_max_depth = 3;
    runner->snapshot.mtp_request.enabled = true;
    runner->snapshot.mtp_request.bypassed = true;
    runner->snapshot.mtp_request.bypass_reason = "sampling is not greedy";
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
    runner->snapshot.prefill_chunk_schedules = 2;
    runner->snapshot.prefill_chunk_successful_schedules = 1;
    runner->snapshot.prefill_chunks = 3;
    runner->snapshot.prefill_chunk_real_tokens = 512;
    runner->snapshot.prefill_chunk_padded_tokens = 32;
    runner->snapshot.prefill_chunk_failures = 1;

    auto tokenizer = createMockTokenizer();
    auto mpi = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    BenchmarkRunner bench(runner, tokenizer, mpi);

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
    EXPECT_EQ(result.prefix_state.mtp_draft_steps, 3u);
    EXPECT_EQ(result.prefix_state.mtp_rejected_tokens, 1u);
    EXPECT_TRUE(result.prefix_state.mtp_bypassed);
    EXPECT_EQ(result.prefix_state.mtp_bypasses, 1u);
    EXPECT_EQ(result.prefix_state.mtp_request.accepted_tokens, 2u);
    EXPECT_DOUBLE_EQ(result.prefix_state.mtp_request.acceptance_rate, 2.0 / 3.0);
    EXPECT_EQ(result.prefix_state.mtp_verifier_runs, 4u);
    EXPECT_EQ(result.prefix_state.mtp_verifier_token_count, 8u);
    EXPECT_EQ(result.prefix_state.mtp_depth_policy_updates, 1u);
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
    EXPECT_NE(output.find("updates=1"), std::string::npos);
    EXPECT_NE(output.find("MTP decode"), std::string::npos);
    EXPECT_NE(output.find("Prefill chunks"), std::string::npos);
    EXPECT_NE(output.find("1/2 schedules"), std::string::npos);
}

/**
 * @brief Verify benchmark JSON carries Phase 14 counters without log parsing.
 */
TEST(Test__BenchmarkRunnerCPU, SerializesMachineReadableBenchmarkJson)
{
    BenchmarkResult result;
    result.prefill_tokens = 10;
    result.prefill_time_ms = 4.0;
    result.prefill_tokens_per_sec = 2500.0;
    result.prefill_success = true;
    result.decode_tokens = 2;
    result.decode_time_ms = 2.0;
    result.decode_tokens_per_sec = 1000.0;
    result.decode_success = true;
    result.total_time_ms = 6.0;
    result.success = true;
    result.generated_text = "xy";

    auto &snapshot = result.prefix_state;
    snapshot.initialized = true;
    snapshot.architecture = "mock_cpu";
    snapshot.execution_path = "GRAPH";
    snapshot.primary_device = DeviceId::cpu();
    snapshot.current_position = 12;
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
    snapshot.mtp_request.enabled = true;
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
    snapshot.prefill_chunk_schedules = 2;
    snapshot.prefill_chunk_successful_schedules = 2;
    snapshot.prefill_chunks = 5;
    snapshot.prefill_chunk_real_tokens = 1024;
    snapshot.prefill_chunk_padded_tokens = 64;

    OrchestrationConfig config;
    config.benchmark_mode = true;
    config.model_path = "model.gguf";
    config.n_predict = 2;
    config.prefix_cache.enabled = true;
    config.mtp.enabled = true;
    config.mtp.draft_tokens = 3;
    config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    config.mtp.depth_policy.max_depth = 3;
    config.mtp.depth_policy.window_size = 8;
    config.benchmark_json_output_path = "/tmp/bench.json";

    const auto doc = nlohmann::json::parse(benchmarkResultToJsonString(result, &config));

    EXPECT_EQ(doc.at("schema"), "llaminar.benchmark.v1");
    EXPECT_TRUE(doc.at("success").get<bool>());
    EXPECT_EQ(doc.at("tokens").at("prefill"), 10);
    EXPECT_EQ(doc.at("tokens").at("decode"), 2);
    EXPECT_DOUBLE_EQ(doc.at("timing_ms").at("total").get<double>(), 6.0);
    EXPECT_DOUBLE_EQ(doc.at("throughput_tokens_per_sec").at("overall").get<double>(), 2000.0);
    EXPECT_EQ(doc.at("generated_text_bytes"), 2);

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
    EXPECT_EQ(mtp.at("current_depth"), 2);
    EXPECT_EQ(mtp.at("max_depth"), 3);
    EXPECT_EQ(mtp.at("depth_policy_updates"), 1);
    EXPECT_EQ(mtp.at("depth_policy_promotions"), 1);
    EXPECT_TRUE(mtp.at("request").at("adaptive_depth_enabled").get<bool>());
    EXPECT_EQ(mtp.at("request").at("depth_policy_mode"), "dynamic");
    EXPECT_EQ(mtp.at("request").at("last_depth_policy_reason"), "promote_full_accept_rate");
    EXPECT_EQ(mtp.at("request").at("accepted_tokens"), 3);

    EXPECT_EQ(doc.at("prefill_chunks").at("chunks"), 5);
    EXPECT_EQ(doc.at("prefill_chunks").at("padded_tokens"), 64);
    EXPECT_TRUE(doc.at("config").at("prefix_cache_enabled").get<bool>());
    EXPECT_TRUE(doc.at("config").at("mtp_enabled").get<bool>());
    EXPECT_EQ(doc.at("config").at("mtp_draft_tokens"), 3);
    EXPECT_EQ(doc.at("config").at("mtp_depth_policy"), "dynamic");
    EXPECT_EQ(doc.at("config").at("mtp_max_draft_tokens"), 3);
    EXPECT_EQ(doc.at("config").at("mtp_depth_window"), 8);
    EXPECT_EQ(doc.at("config").at("mtp_depth_promote_windows"), 3);
    EXPECT_EQ(doc.at("config").at("benchmark_json_output_path"), "/tmp/bench.json");
}
