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
    EXPECT_EQ(result.prefix_state.mtp_draft_steps, 3u);
    EXPECT_EQ(result.prefix_state.mtp_rejected_tokens, 1u);
    EXPECT_TRUE(result.prefix_state.mtp_bypassed);
    EXPECT_EQ(result.prefix_state.mtp_bypasses, 1u);
    EXPECT_EQ(result.prefix_state.mtp_verifier_runs, 4u);
    EXPECT_EQ(result.prefix_state.mtp_verifier_token_count, 8u);

    testing::internal::CaptureStdout();
    bench.printResults(result);
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("PREFIX / MTP STATE"), std::string::npos);
    EXPECT_NE(output.find("Lookup results"), std::string::npos);
    EXPECT_NE(output.find("Bypasses"), std::string::npos);
    EXPECT_NE(output.find("RAM budget cannot hold one complete prefix block"), std::string::npos);
    EXPECT_NE(output.find("sampling is not greedy"), std::string::npos);
    EXPECT_NE(output.find("MTP decode"), std::string::npos);
}
