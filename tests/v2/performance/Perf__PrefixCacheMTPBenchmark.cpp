/**
 * @file Perf__PrefixCacheMTPBenchmark.cpp
 * @brief Lightweight benchmark JSON smoke tests for prefix-cache/MTP telemetry.
 *
 * These tests intentionally avoid loading a model. They keep the Phase 14
 * benchmark schema covered in the performance suite while real-model benchmark
 * runs remain ordinary benchmark invocations.
 */

#include <gtest/gtest.h>

#include "config/OrchestrationConfig.h"
#include "utils/BenchmarkRunner.h"
#include "nlohmann/json.hpp"

using namespace llaminar2;

namespace
{
    BenchmarkResult makeSuccessfulResult()
    {
        BenchmarkResult result;
        result.prefill_tokens = 128;
        result.prefill_time_ms = 16.0;
        result.prefill_tokens_per_sec = 8000.0;
        result.prefill_success = true;
        result.decode_tokens = 8;
        result.decode_time_ms = 4.0;
        result.decode_tokens_per_sec = 2000.0;
        result.total_time_ms = 20.0;
        result.decode_success = true;
        result.success = true;
        result.generated_text = "abcdefgh";
        return result;
    }
}

TEST(Perf__PrefixCacheMTPBenchmark, JsonSchemaCarriesPrefixAndMTPCounters)
{
    BenchmarkResult result = makeSuccessfulResult();
    auto &state = result.prefix_state;
    state.initialized = true;
    state.architecture = "qwen36";
    state.execution_path = "GRAPH";
    state.primary_device = DeviceId::rocm(0);
    state.current_position = 136;
    state.session_epoch = 7;
    state.prefill_logits_ready = true;
    state.has_hidden = true;
    state.has_logits = true;

    state.prefix_cache_config_enabled = true;
    state.prefix_cache_ready = true;
    state.prefix_cache_lookups = 2;
    state.prefix_cache_hits = 1;
    state.prefix_cache_partial_hits = 1;
    state.prefix_cache_matched_blocks = 4;
    state.prefix_cache_matched_tokens = 128;
    state.prefix_cache_stores = 3;
    state.prefix_cache_inserts = 2;
    state.prefix_cache_promotions = 1;
    state.prefix_cache_terminal_state_hits = 1;
    state.prefix_cache_ram_bytes = 1024;
    state.prefix_cache_device_bytes = 2048;
    state.prefix_cache_disk_bytes = 4096;
    state.prefix_cache_hybrid_state_bytes = 256;
    state.prefix_cache_mtp_state_bytes = 512;
    state.prefix_request.enabled = true;
    state.prefix_request.hit = true;
    state.prefix_request.requested_tokens = 128;
    state.prefix_request.matched_tokens = 128;
    state.prefix_request.matched_blocks = 4;
    state.prefix_request.terminal_logits_restored = true;
    state.prefix_request.terminal_hidden_restored = true;
    state.prefix_request.mtp_state_restored = true;
    state.prefix_request.hybrid_state_restored = true;
    state.prefix_request.storage_tier = "device-hot";

    state.mtp_config_enabled = true;
    state.mtp_draft_steps = 8;
    state.mtp_accepted_tokens = 6;
    state.mtp_rejected_tokens = 2;
    state.mtp_rollbacks = 1;
    state.mtp_verifier_runs = 3;
    state.mtp_verifier_token_count = 8;
    state.mtp_request.enabled = true;
    state.mtp_request.draft_steps = 8;
    state.mtp_request.accepted_tokens = 6;
    state.mtp_request.rejected_tokens = 2;
    state.mtp_request.rollbacks = 1;
    state.mtp_request.acceptance_rate = 0.75;

    state.prefill_chunk_schedules = 1;
    state.prefill_chunk_successful_schedules = 1;
    state.prefill_chunks = 2;
    state.prefill_chunk_real_tokens = 128;
    state.prefill_chunk_padded_tokens = 0;

    OrchestrationConfig config;
    config.benchmark_mode = true;
    config.model_path = "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";
    config.n_predict = 8;
    config.prefix_cache.enabled = true;
    config.mtp.enabled = true;
    config.benchmark_json_output_path = "/tmp/prefix-cache-mtp-benchmark.json";
    config.device_for_this_rank = GlobalDeviceAddress::rocm(0, 0);

    const auto doc = nlohmann::json::parse(benchmarkResultToJsonString(result, &config));

    EXPECT_EQ(doc.at("schema"), "llaminar.benchmark.v1");
    EXPECT_TRUE(doc.at("success").get<bool>());
    EXPECT_EQ(doc.at("tokens").at("prefill"), 128);
    EXPECT_EQ(doc.at("tokens").at("decode"), 8);
    EXPECT_DOUBLE_EQ(doc.at("throughput_tokens_per_sec").at("overall").get<double>(), 6800.0);

    const auto &prefix = doc.at("prefix_cache");
    EXPECT_TRUE(prefix.at("config_enabled").get<bool>());
    EXPECT_TRUE(prefix.at("ready").get<bool>());
    EXPECT_EQ(prefix.at("hits"), 1);
    EXPECT_EQ(prefix.at("partial_hits"), 1);
    EXPECT_EQ(prefix.at("matched_tokens"), 128);
    EXPECT_EQ(prefix.at("terminal_state_hits"), 1);
    EXPECT_EQ(prefix.at("ram_bytes"), 1024);
    EXPECT_EQ(prefix.at("device_bytes"), 2048);
    EXPECT_EQ(prefix.at("disk_bytes"), 4096);
    EXPECT_EQ(prefix.at("request").at("storage_tier"), "device-hot");
    EXPECT_TRUE(prefix.at("request").at("terminal_hidden_restored").get<bool>());
    EXPECT_TRUE(prefix.at("request").at("mtp_state_restored").get<bool>());
    EXPECT_TRUE(prefix.at("request").at("hybrid_state_restored").get<bool>());

    const auto &mtp = doc.at("mtp");
    EXPECT_TRUE(mtp.at("config_enabled").get<bool>());
    EXPECT_EQ(mtp.at("draft_steps"), 8);
    EXPECT_EQ(mtp.at("accepted_tokens"), 6);
    EXPECT_EQ(mtp.at("rejected_tokens"), 2);
    EXPECT_EQ(mtp.at("rollbacks"), 1);
    EXPECT_EQ(mtp.at("verifier_runs"), 3);
    EXPECT_EQ(mtp.at("verifier_token_count"), 8);
    EXPECT_DOUBLE_EQ(mtp.at("acceptance_rate").get<double>(), 0.75);
    EXPECT_DOUBLE_EQ(mtp.at("request").at("acceptance_rate").get<double>(), 0.75);

    EXPECT_EQ(doc.at("prefill_chunks").at("chunks"), 2);
    EXPECT_EQ(doc.at("config").at("device"), "localhost:0:rocm:0");
    EXPECT_TRUE(doc.at("config").at("prefix_cache_enabled").get<bool>());
    EXPECT_TRUE(doc.at("config").at("mtp_enabled").get<bool>());
}

TEST(Perf__PrefixCacheMTPBenchmark, DisabledPathsRemainExplicitlyZero)
{
    BenchmarkResult result = makeSuccessfulResult();

    OrchestrationConfig config;
    config.benchmark_mode = true;
    config.n_predict = 8;
    config.prefix_cache.enabled = false;
    config.mtp.enabled = false;

    const auto doc = nlohmann::json::parse(benchmarkResultToJsonString(result, &config));

    const auto &prefix = doc.at("prefix_cache");
    EXPECT_FALSE(prefix.at("config_enabled").get<bool>());
    EXPECT_FALSE(prefix.at("ready").get<bool>());
    EXPECT_FALSE(prefix.at("bypassed").get<bool>());
    EXPECT_EQ(prefix.at("lookups"), 0);
    EXPECT_EQ(prefix.at("hits"), 0);
    EXPECT_EQ(prefix.at("partial_hits"), 0);
    EXPECT_EQ(prefix.at("matched_tokens"), 0);
    EXPECT_FALSE(prefix.at("request").at("enabled").get<bool>());
    EXPECT_EQ(prefix.at("request").at("storage_tier"), "none");

    const auto &mtp = doc.at("mtp");
    EXPECT_FALSE(mtp.at("config_enabled").get<bool>());
    EXPECT_FALSE(mtp.at("bypassed").get<bool>());
    EXPECT_EQ(mtp.at("draft_steps"), 0);
    EXPECT_EQ(mtp.at("accepted_tokens"), 0);
    EXPECT_EQ(mtp.at("rejected_tokens"), 0);
    EXPECT_EQ(mtp.at("rollbacks"), 0);
    EXPECT_DOUBLE_EQ(mtp.at("acceptance_rate").get<double>(), 0.0);
    EXPECT_FALSE(mtp.at("request").at("enabled").get<bool>());

    EXPECT_FALSE(doc.at("config").at("prefix_cache_enabled").get<bool>());
    EXPECT_FALSE(doc.at("config").at("mtp_enabled").get<bool>());
}

TEST(Perf__PrefixCacheMTPBenchmark, MTPAcceptanceRateUsesAttemptedDraftTokens)
{
    BenchmarkResult result = makeSuccessfulResult();
    auto &state = result.prefix_state;
    state.mtp_config_enabled = true;
    state.mtp_draft_steps = 3;
    state.mtp_accepted_tokens = 6;
    state.mtp_rejected_tokens = 2;
    state.mtp_rollbacks = 2;
    state.mtp_verifier_runs = 3;
    state.mtp_verifier_token_count = 8;

    OrchestrationConfig config;
    config.benchmark_mode = true;
    config.n_predict = 8;
    config.mtp.enabled = true;

    const auto doc = nlohmann::json::parse(benchmarkResultToJsonString(result, &config));
    const auto &mtp = doc.at("mtp");

    EXPECT_EQ(mtp.at("draft_steps"), 3);
    EXPECT_EQ(mtp.at("accepted_tokens"), 6);
    EXPECT_EQ(mtp.at("rejected_tokens"), 2);
    EXPECT_DOUBLE_EQ(mtp.at("acceptance_rate").get<double>(), 0.75);
}
