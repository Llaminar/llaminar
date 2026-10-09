/**
 * @file Test__ServerRuntimeStats.cpp
 * @brief Real HTTP proofs of responsive statistics during serialized inference.
 *
 * The actual production routes execute against device-free runner mocks. Held
 * prefill and queued ordinary/SSE requests cannot starve stats or discovery.
 * Transport status counts remain distinct from errors inside an HTTP 200 SSE
 * response, and every observation rejects live probes and mutable state reads.
 */
#include "app/modes/ServerMode.h"
#include "app/modes/ChatCompletionHandler.h"
#include "app/modes/SerializedInferenceExecutor.h"
#include "mocks/MockOrchestrationRunner.h"
#include "mocks/MockTokenizer.h"
#include "httplib.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <atomic>
#include <future>
#include <thread>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace testing;
using json = nlohmann::json;

namespace
{
    /** @brief The handler enumerates these publishers once; HTTP never calls the runner again. */
    class CacheObservedRunner : public NiceMock<MockOrchestrationRunner>
    {
    public:
        std::shared_ptr<PrefixCacheTierTelemetry> ram = std::make_shared<PrefixCacheTierTelemetry>(100);
        std::shared_ptr<PrefixCacheTierTelemetry> disk = std::make_shared<PrefixCacheTierTelemetry>(200);
        std::shared_ptr<PrefixCacheTelemetry> publisher = std::make_shared<PrefixCacheTelemetry>();
        mutable unsigned enumerations = 0;
        /** @brief Materialize only model-free metadata before constructing the HTTP handler. */
        CacheObservedRunner() { publisher->bind(ram, disk); }
        /** @return Frozen local topology and passive publishers; counted to detect HTTP traversal. */
        PrefixCacheTelemetrySources prefixCacheTelemetrySources() const override
        {
            ++enumerations;
            return {{.participant = "cpu:0", .ram_enabled = true, .disk_enabled = true,
                     .ram_capacity_bytes = 100, .disk_capacity_bytes = 200,
                     .disk_namespace = "fixture", .publisher = publisher}};
        }
    };

    /** @brief Bounded model-free rendezvous, never a production inference deadline. */
    template<class Predicate> bool eventually(Predicate predicate)
    {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!predicate())
        {
            if (std::chrono::steady_clock::now() >= end) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }
}

/** @brief Own all route captures until HTTP and inference work have retired. */
class Test__ServerRuntimeStats : public Test
{
protected:
    /** @brief Configure a real serving route with inert, strictly observed dependencies. */
    void SetUp() override
    {
        runner.simulateInitialized();
        ON_CALL(tokenizer, encodeChat(_, _, _, _)).WillByDefault(Return(std::vector<int>(100, 7)));
        ON_CALL(tokenizer, decode_token(_)).WillByDefault(Return("hello"));
        ON_CALL(runner, prefill(_)).WillByDefault(Return(true));
        ON_CALL(runner, decodeStep()).WillByDefault(Invoke([] {
            GenerationResult result;
            result.tokens = {42};
            return result;
        }));
        EXPECT_CALL(runner, prefixStateProbe(_)).Times(0);
        EXPECT_CALL(runner, lastLogits()).Times(0);
        EXPECT_CALL(runner, currentPosition()).Times(0);
        EXPECT_CALL(runner, moeOptimizationMovementLedger()).Times(0);
        summary.prefix_request.enabled = true;
        summary.prefix_request.requested_tokens = 100;
        summary.prefix_request.matched_tokens = 80;
        summary.prefix_request.partial_hit = true;
        summary.prefix_request.storage_tier = "ram";
        summary.mtp_request.enabled = true;
        summary.mtp_request.accepted_tokens = 3;
        summary.mtp_request.rejected_tokens = 1;
        summary.mtp_request.depth_policy_mode = "dynamic";
        summary.mtp_request.min_depth = 1;
        summary.mtp_request.max_depth = 15;
        summary.mtp_request.current_depth = 3;
        EXPECT_CALL(runner, requestRuntimeSummary()).Times(0);
        configureInferenceHttpServer(server, executor);
        registerChatCompletionEndpoint(server, handler, executor);
        registerOpenAIModelDiscoveryEndpoint(server, "fixture-model");
        auto plan = RankExecutionPlan{};
        plan.primary_device = GlobalDeviceAddress::cpu();
        registerRuntimeStatsEndpoint(server, handler, executor, "fixture-model",
            serverRuntimeDescription(plan, runner.config(), 1));
        port = server.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port, 0);
        listener = std::thread([this] { listenInferenceHttpServer(server); });
    }
    /** @brief Release held work even after an assertion, then join every owner. */
    void TearDown() override
    {
        release();
        server.stop();
        if (listener.joinable()) listener.join();
        executor.shutdown();
    }
    /** @brief Release the first model-free prefill once, including failure cleanup. */
    void release()
    {
        if (!released.exchange(true)) gate.set_value();
    }
    /** @brief Create a client whose finite timeout diagnoses a test deadlock only. */
    std::unique_ptr<httplib::Client> client()
    {
        auto result = std::make_unique<httplib::Client>("127.0.0.1", port);
        result->set_read_timeout(std::chrono::seconds(5));
        return result;
    }
    /** @brief Exercise actual admission with a held request and a second queued response. */
    void expectResponsive(bool streaming)
    {
        EXPECT_CALL(runner, requestRuntimeSummary()).Times(2).WillRepeatedly(Return(summary));
        const auto ready = gate.get_future().share();
        std::atomic<int> prefills{0};
        ON_CALL(runner, prefill(_)).WillByDefault(Invoke([&](const auto &) {
            if (++prefills == 1) ready.wait();
            return true;
        }));
        const std::string request = json{{"messages", {{{"role", "user"}, {"content", "hello"}}}},
            {"stream", streaming}, {"max_tokens", 1}}.dump();
        auto first = std::async(std::launch::async, [&] {
            return client()->Post("/v1/chat/completions", request, "application/json");
        });
        // The scope guard outlives futures, so assertion failure cannot strand
        // std::future's joining destructor behind our intentionally held work.
        struct ReleaseOnExit
        {
            Test__ServerRuntimeStats &fixture;
            ~ReleaseOnExit() { fixture.release(); }
        } release_first{*this};
        ASSERT_TRUE(eventually([&] { return prefills.load() == 1; }));
        auto second = std::async(std::launch::async, [&] {
            return client()->Post("/v1/chat/completions", request, "application/json");
        });
        ReleaseOnExit release_second{*this};
        ASSERT_TRUE(eventually([&] { return executor.snapshot().queued == 1; }));
        const auto stats = client()->Get("/stats");
        ASSERT_TRUE(stats);
        ASSERT_EQ(stats->status, 200);
        EXPECT_EQ(stats->get_header_value("Cache-Control"), "no-store");
        const auto live = json::parse(stats->body);
        EXPECT_EQ(live["model"], "fixture-model");
        EXPECT_EQ(live["requests"]["active_phase"], "prefill");
        EXPECT_EQ(live["requests"]["active"], 1);
        EXPECT_EQ(live["requests"]["completed"], 0);
        EXPECT_EQ(live["inference_queue"]["admitted"], 2);
        EXPECT_EQ(live["inference_queue"]["queued"], 1);
        EXPECT_TRUE(live["timings"]["ttft"]["average_seconds"].is_null());
        EXPECT_TRUE(live["last_request"].is_null());
        EXPECT_EQ(prefills.load(), 1);

        const auto discovery = client()->Get("/v1/models");
        ASSERT_TRUE(discovery);
        EXPECT_EQ(discovery->status, 200);
        const auto overloaded = client()->Post("/v1/chat/completions", request, "application/json");
        ASSERT_TRUE(overloaded);
        EXPECT_EQ(overloaded->status, 503);
        EXPECT_EQ(json::parse(overloaded->body)["error"]["code"], "inference_queue_full");
        const auto malformed = client()->Post("/v1/chat/completions", "{", "application/json");
        ASSERT_TRUE(malformed);
        EXPECT_EQ(malformed->status, 400);
        const auto missing = client()->Get("/missing");
        ASSERT_TRUE(missing);
        EXPECT_EQ(missing->status, 404);
        EXPECT_EQ(first.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
        EXPECT_EQ(second.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
        release();
        const auto first_response = first.get();
        const auto second_response = second.get();
        ASSERT_TRUE(first_response);
        ASSERT_TRUE(second_response);
        EXPECT_EQ(first_response->status, 200);
        EXPECT_EQ(second_response->status, 200);
        ASSERT_TRUE(eventually([&] { return handler.runtimeStats()["requests"]["completed"] == 2; }));
        const auto result = client()->Get("/stats");
        ASSERT_TRUE(result);
        const auto complete = json::parse(result->body);
        EXPECT_EQ(complete["requests"]["active"], 0);
        EXPECT_EQ(complete["tokens"]["prompt"], 200);
        EXPECT_EQ(complete["tokens"]["completion"], 2);
        EXPECT_EQ(complete["prefix_cache"]["token_reuse_rate"], .8);
        EXPECT_EQ(complete["mtp"]["acceptance_rate"], .75);
        EXPECT_EQ(complete["timings"]["ttft"]["samples"], 2);
        EXPECT_GT(complete["last_request"]["timings"]["queue_seconds"].get<double>(), 0);
        EXPECT_GE(complete["last_request"]["timings"]["ttft_seconds"].get<double>(),
                  complete["last_request"]["timings"]["queue_seconds"].get<double>());
        EXPECT_GT(complete["throughput"]["decode_tokens_per_second"].get<double>(), 0);
        EXPECT_EQ(complete["http"]["status_codes"]["503"], 1);
        EXPECT_EQ(complete["http"]["status_codes"]["400"], 1);
        EXPECT_EQ(complete["http"]["status_codes"]["404"], 1);
        EXPECT_EQ(complete["last_request"]["timings"]["first_output_seconds"].is_null(), !streaming);
    }

    /** @brief Reset over HTTP while the real route owns a held prefill or streaming decode. */
    void expectResetWhileActive(bool streaming)
    {
        EXPECT_CALL(runner, requestRuntimeSummary()).Times(2).WillRepeatedly(Return(summary));
        const auto ready = gate.get_future().share();
        std::atomic<bool> entered{false};
        if (streaming)
            ON_CALL(runner, decodeStep()).WillByDefault(Invoke([&] {
                entered = true;
                ready.wait();
                GenerationResult result;
                result.tokens = {42};
                return result;
            }));
        else
            ON_CALL(runner, prefill(_)).WillByDefault(Invoke([&](const auto &) {
                entered = true;
                ready.wait();
                return true;
            }));
        const std::string request = json{{"messages", {{{"role", "user"}, {"content", "hello"}}}},
            {"stream", streaming}, {"max_tokens", 1}}.dump();
        auto old_request = std::async(std::launch::async, [&] {
            return client()->Post("/v1/chat/completions", request, "application/json");
        });
        struct ReleaseOnExit
        {
            Test__ServerRuntimeStats &fixture;
            ~ReleaseOnExit() { fixture.release(); }
        } release_on_exit{*this};
        ASSERT_TRUE(eventually([&] { return entered.load(); }));
        const auto reset = client()->Put("/stats", "", "application/json");
        ASSERT_TRUE(reset);
        EXPECT_EQ(reset->status, 200);
        EXPECT_EQ(json::parse(reset->body)["epoch"], 1);
        const auto active = client()->Get("/stats");
        ASSERT_TRUE(active);
        const auto before = json::parse(active->body);
        EXPECT_EQ(before["requests"]["active"], 1);
        EXPECT_EQ(before["requests"]["active_in_current_epoch"], 0);
        EXPECT_EQ(before["requests"]["active_phase"], streaming ? "decode" : "prefill");
        EXPECT_EQ(before["requests"]["started"], 0);
        EXPECT_TRUE(before["last_request"].is_null());
        release();
        const auto old_response = old_request.get();
        ASSERT_TRUE(old_response);
        EXPECT_EQ(old_response->status, 200);
        ASSERT_TRUE(eventually([&] { return handler.runtimeStats()["requests"]["active"] == 0; }));
        const auto retired = handler.runtimeStats();
        EXPECT_EQ(retired["requests"]["completed"], 0);
        EXPECT_EQ(retired["tokens"]["completion"], 0);
        EXPECT_EQ(retired["timings"]["ttft"]["samples"], 0);
        EXPECT_TRUE(retired["last_request"].is_null());
        const auto fresh = client()->Post("/v1/chat/completions", request, "application/json");
        ASSERT_TRUE(fresh);
        EXPECT_EQ(fresh->status, 200);
        const auto after = handler.runtimeStats();
        EXPECT_EQ(after["requests"]["completed"], 1);
        EXPECT_EQ(after["last_request"]["sequence"], 2);
        EXPECT_EQ(after["tokens"]["prompt"], 100);
        EXPECT_EQ(after["prefix_cache"]["matched_tokens"], 80);
        EXPECT_EQ(after["timings"]["ttft"]["samples"], 1);
    }

    CacheObservedRunner runner;
    NiceMock<MockTokenizer> tokenizer;
    RequestRuntimeSummary summary;
    ChatCompletionHandler handler{runner, tokenizer, "fixture-model"};
    SerializedInferenceExecutor executor{2};
    httplib::Server server;
    std::thread listener;
    int port = 0;
    std::promise<void> gate;
    std::atomic<bool> released{false};
};

/** @brief Real GET/PUT routes observe committed metadata and preserve occupancy across resets. */
TEST_F(Test__ServerRuntimeStats, PrefixStorageHTTPFreshnessAndReset)
{
    EXPECT_EQ(runner.enumerations, 1u);
    runner.ram->publishUsage(40, 2);
    runner.disk->publishUsage(150, 3);
    runner.ram->record(PrefixTierEvent::Demotion, 10);
    runner.disk->record(PrefixTierEvent::Read, 7);
    auto response = client()->Get("/stats");
    ASSERT_TRUE(response);
    ASSERT_EQ(response->status, 200);
    auto storage = json::parse(response->body)["prefix_cache"]["storage"];
    EXPECT_EQ(storage["tiers"]["ram"]["utilization_percent"], 40.0);
    EXPECT_EQ(storage["tiers"]["disk"]["utilization_percent"], 75.0);
    EXPECT_EQ(storage["churn"]["ram_to_disk"]["operations"], 1);
    EXPECT_EQ(storage["churn"]["disk_to_ram"]["bytes"], 7);
    response = client()->Put("/stats", "", "application/json");
    ASSERT_TRUE(response);
    ASSERT_EQ(response->status, 200);
    response = client()->Get("/stats");
    ASSERT_TRUE(response);
    storage = json::parse(response->body)["prefix_cache"]["storage"];
    EXPECT_EQ(storage["tiers"]["ram"]["used_bytes"], 40);
    EXPECT_EQ(storage["tiers"]["disk"]["used_bytes"], 150);
    EXPECT_EQ(storage["churn"]["ram_to_disk"]["operations"], 0);
    EXPECT_EQ(storage["churn"]["disk_to_ram"]["bytes"], 0);
    runner.ram->publishUsage(20, 1);
    runner.disk->record(PrefixTierEvent::Read, 11);
    response = client()->Get("/stats");
    ASSERT_TRUE(response);
    storage = json::parse(response->body)["prefix_cache"]["storage"];
    EXPECT_EQ(storage["tiers"]["ram"]["utilization_percent"], 20.0);
    EXPECT_EQ(storage["churn"]["disk_to_ram"]["bytes"], 11);
    EXPECT_EQ(runner.enumerations, 1u);
}

/** @brief Ordinary requests cannot starve stats or duplicate the sole model owner. */
TEST_F(Test__ServerRuntimeStats, ResponsiveWithActiveAndQueuedOrdinaryRequests)
{
    expectResponsive(false);
}

/** @brief Repeated idle reads and resets consume no inference or terminal probes. */
TEST_F(Test__ServerRuntimeStats, IdleReadsAndResetsNeverTouchTheRunner)
{
    EXPECT_CALL(runner, prefill(_)).Times(0);
    EXPECT_CALL(runner, decodeStep()).Times(0);
    for (unsigned epoch = 1; epoch <= 20; ++epoch) {
        const auto reset = client()->Put("/stats", "", "application/json");
        ASSERT_TRUE(reset);
        ASSERT_EQ(reset->status, 200);
        EXPECT_EQ(json::parse(reset->body)["epoch"], epoch);
        const auto response = client()->Get("/stats");
        ASSERT_TRUE(response);
        ASSERT_EQ(response->status, 200);
        const auto stats = json::parse(response->body);
        EXPECT_EQ(stats["requests"]["started"], 0);
        EXPECT_EQ(stats["requests"]["active"], 0);
        EXPECT_TRUE(stats["last_request"].is_null());
    }
}

/** @brief SSE providers obey the same admission and worker ownership as ordinary calls. */
TEST_F(Test__ServerRuntimeStats, ResponsiveWithActiveAndQueuedStreamingRequests)
{
    expectResponsive(true);
}

/** @brief An SSE error retains its actual HTTP 200 while generation records failure. */
TEST_F(Test__ServerRuntimeStats, SSEErrorCountsActualHeadersSeparatelyFromHandlerFailure)
{
    ON_CALL(runner, prefill(_)).WillByDefault(Return(false));
    const auto response = client()->Post("/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":1})",
        "application/json");
    ASSERT_TRUE(response);
    EXPECT_EQ(response->status, 200);
    EXPECT_NE(response->body.find("Prefill failed"), std::string::npos);
    ASSERT_TRUE(eventually([&] { return handler.runtimeStats()["http"]["responses"] == 1; }));
    const auto stats = handler.runtimeStats();
    EXPECT_EQ(stats["http"]["status_codes"]["200"], 1);
    EXPECT_EQ(stats["requests"]["failed"], 1);
    EXPECT_EQ(stats["last_request"]["handler_status"], 500);
    EXPECT_TRUE(stats["last_request"]["completion_tokens"].is_null());
    EXPECT_TRUE(stats["timings"]["ttft"]["average_seconds"].is_null());
}

/** @brief Resolved TP devices and MTP/cache policy survive without CLI reconstruction. */
TEST(ServerRuntimeDescription, ResolvedTPAndCachePolicy)
{
    RankExecutionPlan plan;
    plan.primary_device = GlobalDeviceAddress::rocm(0);
    plan.local_tp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
    plan.runtime.mtp.enabled = true;
    plan.runtime.prefix_cache.ram_budget_bytes = 16ULL << 30;
    plan.runtime.prefix_cache.disk_budget_bytes = 32ULL << 30;
    OrchestrationConfig config;
    config.max_seq_len = 262144;
    const auto description = serverRuntimeDescription(plan, config, 1);
    EXPECT_EQ(description["topology"]["strategy"], "tp");
    EXPECT_EQ(description["topology"]["devices"].size(), 2);
    EXPECT_EQ(description["topology"]["devices"][1]["id"], "ROCm:1");
    EXPECT_EQ(description["configuration"]["context_tokens"], 262144);
    EXPECT_EQ(description["configuration"]["mtp"]["enabled"], true);
    EXPECT_EQ(description["configuration"]["prefix_cache"]["ram_budget_bytes_per_participant"], 16ULL << 30);
    EXPECT_EQ(description["configuration"]["prefix_cache"]["disk_budget_bytes"], 32ULL << 30);
}


/** @brief A prefill admitted before reset never enters the new measurement period. */
TEST_F(Test__ServerRuntimeStats, ResetDuringPrefillStartsFreshEpoch)
{
    expectResetWhileActive(false);
}

/** @brief HTTP 200 role metadata does not complete TTFT or defeat an in-flight reset. */
TEST_F(Test__ServerRuntimeStats, ResetDuringStreamingDecodeStartsFreshEpoch)
{
    expectResetWhileActive(true);
}
