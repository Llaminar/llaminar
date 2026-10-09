/**
 * @file Test__HttpRuntimeStats.cpp
 * @brief Device-free terminal accounting and stable inference ownership proofs.
 *
 * Unequal prompt sizes distinguish weighted reuse from averaged percentages.
 * Adversarial scope exits, duplicate publication, disabled/bypassed modes and
 * concurrent readers prove one coherent lifetime view. Executor tests hold real
 * host work while checking independent readers, capacity and stable thread IDs.
 */
#include "app/modes/HttpRuntimeStats.h"
#include "app/modes/SerializedInferenceExecutor.h"
#include <gtest/gtest.h>
#include <atomic>
#include <future>
#include <thread>
#include <tuple>

using namespace llaminar2;

namespace
{
    /** @brief Model-free participant fixture with the same lazy publishers as serving. */
    struct PrefixStatsFixture
    {
        std::shared_ptr<PrefixCacheTelemetry> publisher = std::make_shared<PrefixCacheTelemetry>();
        std::shared_ptr<PrefixCacheTierTelemetry> ram;
        PrefixCacheParticipantTelemetry source;
        /** @brief Describe one admitted participant before allocating its cache. */
        PrefixStatsFixture(std::string name, uint64_t capacity, uint64_t disk_capacity = 0,
                           std::string archive = "shared-model-archive")
            : ram(std::make_shared<PrefixCacheTierTelemetry>(capacity)),
              source{.participant = std::move(name), .ram_enabled = capacity > 0,
                     .disk_enabled = disk_capacity > 0, .ram_capacity_bytes = capacity,
                     .disk_capacity_bytes = disk_capacity, .disk_namespace = std::move(archive),
                     .publisher = publisher} {}
    };
}

/** @brief Disabled and lazy tiers cannot manufacture zero-percent utilization. */
TEST(Test__HttpRuntimeStats, PrefixStorageDisabledAndLazyInitialization)
{
    PrefixStatsFixture disabled("cpu:0", 0);
    HttpRuntimeStats off({disabled.source});
    const auto tiers = off.snapshot()["prefix_cache"]["storage"]["tiers"];
    for (const char *name : {"ram", "disk"})
    {
        EXPECT_EQ(tiers[name]["enabled"], false);
        EXPECT_EQ(tiers[name]["capacity_bytes"], 0);
        EXPECT_EQ(tiers[name]["used_bytes"], 0);
        EXPECT_TRUE(tiers[name]["utilization_percent"].is_null());
    }
    PrefixStatsFixture lazy("rocm:0", 100, 200);
    HttpRuntimeStats stats({lazy.source});
    auto storage = stats.snapshot()["prefix_cache"]["storage"];
    EXPECT_EQ(storage["tiers"]["ram"]["enabled"], true);
    EXPECT_EQ(storage["tiers"]["ram"]["initialized"], false);
    EXPECT_TRUE(storage["tiers"]["ram"]["used_bytes"].is_null());
    EXPECT_TRUE(storage["tiers"]["disk"]["utilization_percent"].is_null());
    stats.reset(); // A reset before first inference must also admit future publishers.
    const auto disk = std::make_shared<PrefixCacheTierTelemetry>(200);
    lazy.publisher->bind(lazy.ram, disk);
    lazy.ram->publishUsage(50, 2);
    disk->publishUsage(100, 4);
    disk->record(PrefixTierEvent::Read, 17);
    storage = stats.snapshot()["prefix_cache"]["storage"];
    EXPECT_EQ(storage["tiers"]["ram"]["utilization_percent"], 50.0);
    EXPECT_EQ(storage["tiers"]["disk"]["utilization_percent"], 50.0);
    EXPECT_EQ(storage["churn"]["disk_to_ram"]["bytes"], 17);
    EXPECT_THROW(lazy.publisher->bind(lazy.ram, disk), std::logic_error);
}

/** @brief TP/PP aggregate bytes, while every shared archive contributes capacity and churn once. */
TEST(Test__HttpRuntimeStats, PrefixStorageWeightedOccupancyAndSharedDisk)
{
    for (const char *backend : {"cpu", "cuda", "rocm"})
    {
        PrefixStatsFixture a(std::string(backend) + ":0", 100, 400);
        PrefixStatsFixture b(std::string(backend) + ":1", 300, 400);
        auto disk = std::make_shared<PrefixCacheTierTelemetry>(400);
        HttpRuntimeStats stats({a.source, b.source, a.source}); // Nested enumeration may alias a participant.
        a.publisher->bind(a.ram, disk);
        b.publisher->bind(b.ram, disk);
        a.ram->publishUsage(100, 5);
        b.ram->publishUsage(0, 0);
        disk->publishUsage(200, 10);
        disk->record(PrefixTierEvent::Write, 23);
        disk->record(PrefixTierEvent::Eviction, 7);
        const auto storage = stats.snapshot()["prefix_cache"]["storage"];
        EXPECT_EQ(storage["tiers"]["ram"]["capacity_bytes"], 400);
        EXPECT_EQ(storage["tiers"]["ram"]["utilization_percent"], 25.0);
        EXPECT_EQ(storage["tiers"]["ram"]["instances"].size(), 2);
        EXPECT_EQ(storage["tiers"]["disk"]["capacity_bytes"], 400);
        EXPECT_EQ(storage["tiers"]["disk"]["used_bytes"], 200);
        EXPECT_EQ(storage["tiers"]["disk"]["instances"].size(), 1);
        EXPECT_EQ(storage["tiers"]["disk"]["instances"][0]["participants"].size(), 2);
        EXPECT_EQ(storage["churn"]["disk_payload_writes"]["operations"], 1);
        EXPECT_EQ(storage["churn"]["disk_payload_writes"]["bytes"], 23);
        EXPECT_EQ(storage["churn"]["disk_evictions"]["bytes"], 7);
    }
}

/** @brief Once materialized, path aliases cannot double count one archive owner. */
TEST(Test__HttpRuntimeStats, PrefixStorageDeduplicatesArchivePathAliases)
{
    PrefixStatsFixture a("cpu:0", 100, 400, "/cache/model"), b("cpu:1", 100, 400, "/alias/model");
    const auto disk = std::make_shared<PrefixCacheTierTelemetry>(400);
    a.publisher->bind(a.ram, disk);
    b.publisher->bind(b.ram, disk);
    HttpRuntimeStats stats({a.source, b.source});
    disk->publishUsage(100, 3);
    disk->record(PrefixTierEvent::Read, 30);
    const auto storage = stats.snapshot()["prefix_cache"]["storage"];
    EXPECT_EQ(storage["tiers"]["disk"]["capacity_bytes"], 400);
    EXPECT_EQ(storage["tiers"]["disk"]["utilization_percent"], 25.0);
    EXPECT_EQ(storage["tiers"]["disk"]["instances"].size(), 1);
    EXPECT_EQ(storage["churn"]["disk_to_ram"]["operations"], 1);
}

/** @brief Occupancy and background churn stay fresh during held inference and across resets. */
TEST(Test__HttpRuntimeStats, PrefixStorageResetDuringActiveRequestPreservesOccupancy)
{
    PrefixStatsFixture a("rocm:0", 100, 200);
    auto disk = std::make_shared<PrefixCacheTierTelemetry>(200);
    a.publisher->bind(a.ram, disk);
    HttpRuntimeStats stats({a.source});
    {
        auto request = stats.beginRequest();
        a.ram->publishUsage(60, 3);
        disk->publishUsage(150, 9);
        a.ram->record(PrefixTierEvent::Demotion, 31);
        disk->record(PrefixTierEvent::BackingReuse, 0);
        auto before = stats.snapshot();
        EXPECT_EQ(before["requests"]["active"], 1);
        EXPECT_EQ(before["prefix_cache"]["storage"]["churn"]["ram_to_disk"]["bytes"], 31);
        stats.reset();
        auto reset = stats.snapshot()["prefix_cache"]["storage"];
        for (const char *name : {"ram", "disk"})
        {
            EXPECT_EQ(reset["tiers"][name]["instances"][0]["revision"],
                      before["prefix_cache"]["storage"]["tiers"][name]["instances"][0]["revision"]);
            EXPECT_GE(reset["tiers"][name]["instances"][0]["observation_age_seconds"].get<double>(),
                      before["prefix_cache"]["storage"]["tiers"][name]["instances"][0]["observation_age_seconds"].get<double>());
        }
        EXPECT_EQ(reset["tiers"]["ram"]["used_bytes"], 60);
        EXPECT_EQ(reset["tiers"]["disk"]["utilization_percent"], 75.0);
        for (auto &[name, counter] : reset["churn"].items())
            if (name != "scope") EXPECT_EQ(counter["operations"], 0);
        a.ram->publishUsage(20, 1);
        disk->record(PrefixTierEvent::Read, 11);
        request.fail(); // Old request retirement cannot restore old HTTP counters.
    }
    auto after = stats.snapshot();
    EXPECT_EQ(after["requests"]["failed"], 0);
    EXPECT_EQ(after["prefix_cache"]["storage"]["tiers"]["ram"]["used_bytes"], 20);
    EXPECT_EQ(after["prefix_cache"]["storage"]["churn"]["disk_to_ram"]["bytes"], 11);
    EXPECT_EQ(after["prefix_cache"]["storage"]["churn"]["ram_to_disk"]["operations"], 0);
}

/** @brief Concurrent readers never see torn usage/entry pairs or reset old activity back into view. */
TEST(Test__HttpRuntimeStats, PrefixStorageConcurrentPublicationAndPolling)
{
    PrefixStatsFixture a("cuda:0", 100);
    a.publisher->bind(a.ram, nullptr);
    HttpRuntimeStats stats({a.source});
    std::atomic<bool> done = false;
    auto writer = std::async(std::launch::async, [&] {
        for (unsigned i = 0; i < 2000; ++i)
        {
            const uint64_t entries = i % 11;
            a.ram->publishUsage(entries * 10, entries);
            a.ram->record(PrefixTierEvent::Eviction, 10);
        }
        done = true;
    });
    do
    {
        const auto tier = stats.snapshot()["prefix_cache"]["storage"]["tiers"]["ram"];
        EXPECT_EQ(tier["used_bytes"].get<uint64_t>(), tier["entries"].get<uint64_t>() * 10);
        stats.reset();
    } while (!done);
    writer.get();
    stats.reset();
    EXPECT_EQ(stats.snapshot()["prefix_cache"]["storage"]["churn"]["ram_evictions"]["operations"], 0);
}

/** @brief Conflicting shared-archive authorities/capacities are configuration errors. */
TEST(Test__HttpRuntimeStats, PrefixStorageRejectsAmbiguousAuthority)
{
    PrefixStatsFixture a("cpu:0", 100, 200), b("cpu:1", 100, 300);
    EXPECT_THROW((HttpRuntimeStats({a.source, b.source})), std::logic_error);
    b.source.disk_capacity_bytes = 200;
    a.publisher->bind(a.ram, std::make_shared<PrefixCacheTierTelemetry>(200));
    b.publisher->bind(b.ram, std::make_shared<PrefixCacheTierTelemetry>(200));
    EXPECT_THROW((HttpRuntimeStats({a.source, b.source})), std::logic_error);
}

namespace
{
    /** @brief Build a completed prefix observation, without constructing a runner. */
    RequestRuntimeSummary prefix(int requested, int matched)
    {
        RequestRuntimeSummary result;
        auto &p = result.prefix_request;
        p.enabled = true;
        p.requested_tokens = requested;
        p.matched_tokens = matched;
        p.hit = matched == requested && requested > 0;
        p.partial_hit = matched > 0 && matched < requested;
        p.storage_tier = matched ? "ram" : "none";
        return result;
    }
}

/** @brief No completed request means undefined rates, not NaN or an invented perfect hit. */
TEST(Test__HttpRuntimeStats, EmptySnapshotAndActiveRequest)
{
    HttpRuntimeStats stats;
    auto empty = stats.snapshot();
    EXPECT_TRUE(empty["prefix_cache"]["request_hit_rate"].is_null());
    EXPECT_TRUE(empty["prefix_cache"]["token_reuse_rate"].is_null());
    EXPECT_TRUE(empty["mtp"]["acceptance_rate"].is_null());
    EXPECT_TRUE(empty["last_request"].is_null());
    auto request = stats.beginRequest();
    const auto active = stats.snapshot();
    EXPECT_EQ(active["requests"]["active"], 1);
    EXPECT_EQ(active["requests"]["started"], 1);
    EXPECT_GE(active["requests"]["active_seconds"].get<double>(), 0.0);
    EXPECT_THROW((void)stats.beginRequest(), std::logic_error);
}

/** @brief Aggregate token reuse is weighted by prompt length and includes cold requests. */
TEST(Test__HttpRuntimeStats, WeightedReuseAndRequestHitRates)
{
    HttpRuntimeStats stats;
    for (const auto &[prompt, matched] : {std::pair{100, 0}, {1000, 900}, {100, 100}})
    {
        auto request = stats.beginRequest();
        request.complete(prompt, 17, prefix(prompt, matched), HttpRuntimeStats::Delivery::Completed);
    }
    const auto snapshot = stats.snapshot();
    const auto &cache = snapshot["prefix_cache"];
    EXPECT_EQ(snapshot["requests"]["completed"], 3);
    EXPECT_EQ(snapshot["requests"]["active"], 0);
    EXPECT_EQ(snapshot["tokens"]["prompt"], 1200);
    EXPECT_EQ(snapshot["tokens"]["completion"], 51);
    EXPECT_EQ(cache["full_hits"], 1);
    EXPECT_EQ(cache["partial_hits"], 1);
    EXPECT_EQ(cache["misses"], 1);
    EXPECT_DOUBLE_EQ(cache["token_reuse_rate"].get<double>(), 1000.0 / 1200);
    EXPECT_DOUBLE_EQ(cache["request_hit_rate"].get<double>(), 2.0 / 3);
    EXPECT_EQ(snapshot["last_request"]["prompt_tokens"], 100);
}

/** @brief Failed scopes and disconnects retire once; failed work invents no cache result. */
TEST(Test__HttpRuntimeStats, FailureDisconnectAndDuplicatePublication)
{
    HttpRuntimeStats stats;
    { auto failed = stats.beginRequest(); }
    {
        auto request = stats.beginRequest();
        request.complete(100, 13, prefix(100, 80), HttpRuntimeStats::Delivery::Disconnected);
        EXPECT_THROW(request.complete(100, 13, prefix(100, 80),
                                      HttpRuntimeStats::Delivery::Completed), std::logic_error);
    }
    const auto snapshot = stats.snapshot();
    EXPECT_EQ(snapshot["requests"]["started"], 2);
    EXPECT_EQ(snapshot["requests"]["failed"], 1);
    EXPECT_EQ(snapshot["requests"]["disconnected"], 1);
    EXPECT_EQ(snapshot["requests"]["completed"], 0);
    EXPECT_EQ(snapshot["tokens"]["completion"], 13);
    EXPECT_EQ(snapshot["prefix_cache"]["lookups"], 1);
}

/** @brief Off/bypassed prefix and MTP observations cannot contaminate measured denominators. */
TEST(Test__HttpRuntimeStats, DisabledBypassedAndMTPPolicies)
{
    HttpRuntimeStats stats;
    for (int mode = 0; mode < 4; ++mode)
    {
        auto summary = prefix(100, 50);
        summary.prefix_request.enabled = mode != 0;
        summary.prefix_request.bypassed = mode == 1;
        auto &mtp = summary.mtp_request;
        mtp.enabled = mode != 0;
        mtp.bypassed = mode == 1;
        mtp.depth_policy_mode = mode == 2 ? "fixed" : "dynamic";
        mtp.current_depth = 3;
        mtp.min_depth = mode == 2 ? 3 : 1;
        mtp.max_depth = mode == 2 ? 3 : 15;
        mtp.draft_steps = 20;
        mtp.accepted_tokens = 12;
        mtp.rejected_tokens = 4;
        mtp.depth_policy_updates = mode == 3 ? 5 : 0;
        summary.mtp_verifier_runs = 9;
        auto request = stats.beginRequest();
        request.complete(100, 25, summary, HttpRuntimeStats::Delivery::Completed);
    }
    const auto snapshot = stats.snapshot();
    EXPECT_EQ(snapshot["prefix_cache"]["lookups"], 2);
    EXPECT_EQ(snapshot["prefix_cache"]["bypassed"], 1);
    EXPECT_EQ(snapshot["prefix_cache"]["requested_tokens"], 200);
    EXPECT_EQ(snapshot["mtp"]["requests"], 2);
    EXPECT_EQ(snapshot["mtp"]["draft_tokens"], 40);
    EXPECT_EQ(snapshot["mtp"]["verifier_runs"], 18);
    EXPECT_EQ(snapshot["mtp"]["depth_updates"], 5);
    EXPECT_DOUBLE_EQ(snapshot["mtp"]["acceptance_rate"].get<double>(), .75);
    EXPECT_EQ(snapshot["last_request"]["mtp"]["depth_policy"], "dynamic");
    EXPECT_EQ(snapshot["last_request"]["mtp"]["max_depth"], 15);
}

/** @brief Reject inconsistent terminal counts before publishing any successful totals. */
TEST(Test__HttpRuntimeStats, InvalidTerminalCountsRemainOneFailedScope)
{
    for (const auto &[prompt, requested, matched] :
         {std::tuple{-1, 100, 0}, {100, -1, 0}, {100, 100, -1}, {100, 100, 101}, {10, 100, 5}})
    {
        HttpRuntimeStats stats;
        {
            auto request = stats.beginRequest();
            EXPECT_THROW(request.complete(prompt, 10, prefix(requested, matched),
                                          HttpRuntimeStats::Delivery::Completed), std::invalid_argument);
        }
        const auto snapshot = stats.snapshot();
        EXPECT_EQ(snapshot["requests"]["failed"], 1);
        EXPECT_EQ(snapshot["requests"]["completed"], 0);
        EXPECT_EQ(snapshot["prefix_cache"]["lookups"], 0);
    }
}

/** @brief Readers see coherent terminal totals while requests start and finish. */
TEST(Test__HttpRuntimeStats, ConcurrentReadersSeeWholePublications)
{
    HttpRuntimeStats stats;
    std::atomic<bool> finished{false};
    std::thread publisher([&] {
        for (int n = 0; n < 2000; ++n)
        {
            auto request = stats.beginRequest();
            request.complete(100, 10, prefix(100, 80), HttpRuntimeStats::Delivery::Completed);
        }
        finished = true;
    });
    do
    {
        const auto snapshot = stats.snapshot();
        const auto completed = snapshot["requests"]["completed"].get<uint64_t>();
        EXPECT_EQ(snapshot["prefix_cache"]["requested_tokens"], completed * 100);
        EXPECT_EQ(snapshot["prefix_cache"]["matched_tokens"], completed * 80);
        EXPECT_EQ(snapshot["requests"]["started"].get<uint64_t>(),
                  completed + snapshot["requests"]["active"].get<uint64_t>());
    } while (!finished);
    publisher.join();
    EXPECT_EQ(stats.snapshot()["requests"]["completed"], 2000);
}

/** @brief Releasing a never-submitted stream restores admission capacity immediately. */
TEST(Test__SerializedInferenceExecutor, BoundedAdmissionAndUnpublishedStreamRetirement)
{
    EXPECT_THROW(SerializedInferenceExecutor(0), std::invalid_argument);
    SerializedInferenceExecutor executor(2);
    auto first = executor.tryReserve();
    auto second = executor.tryReserve();
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_FALSE(executor.tryReserve());
    EXPECT_EQ(executor.snapshot().admitted, 2);
    EXPECT_GT(executor.httpWorkerCount(), executor.snapshot().capacity);
    second.reset();
    EXPECT_TRUE(executor.tryReserve());
    EXPECT_EQ(first->run([] { return 42; }), 42);
    EXPECT_THROW(first->run([] { return 0; }), std::logic_error);
}

/** @brief Exceptions and subsequent jobs retain one stable worker and exactly one result. */
TEST(Test__SerializedInferenceExecutor, StableWorkerSurvivesExceptionsAndRejectsReentry)
{
    SerializedInferenceExecutor executor(2);
    const auto worker = executor.tryReserve()->run([] { return std::this_thread::get_id(); });
    EXPECT_NE(worker, std::this_thread::get_id());
    EXPECT_THROW(executor.tryReserve()->run([] { throw std::runtime_error("fixture"); }), std::runtime_error);
    EXPECT_EQ(executor.tryReserve()->run([] { return std::this_thread::get_id(); }), worker);
    executor.tryReserve()->run([&] {
        auto nested = executor.tryReserve();
        ASSERT_TRUE(nested);
        EXPECT_THROW(nested->run([] {}), std::logic_error);
    });
    executor.shutdown();
    EXPECT_FALSE(executor.tryReserve());
}

/** @brief A held generation cannot retain the host snapshot mutex or admit concurrent work. */
TEST(Test__SerializedInferenceExecutor, ActiveWorkDoesNotBlockSnapshotAndQueuedWork)
{
    SerializedInferenceExecutor executor(2);
    std::promise<void> entered, release;
    const auto released = release.get_future().share();
    auto first = std::async(std::launch::async, [&] {
        return executor.tryReserve()->run([&] { entered.set_value(); released.wait(); return 1; });
    });
    entered.get_future().wait();
    auto second = std::async(std::launch::async, [&] { return executor.tryReserve()->run([] { return 2; }); });
    const auto snapshot = executor.snapshot();
    EXPECT_TRUE(snapshot.active);
    EXPECT_GE(snapshot.admitted, 1);
    EXPECT_EQ(first.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    EXPECT_EQ(second.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    release.set_value();
    EXPECT_EQ(first.get(), 1);
    EXPECT_EQ(second.get(), 2);
}

/** @brief Rates use summed work and durations; queue time belongs to TTFT and total latency. */
TEST(Test__HttpRuntimeStats, WeightedThroughputAndTTFTUseExactMonotonicIntervals)
{
    using namespace std::chrono;
    const auto base = HttpRuntimeStats::Clock::now() - hours(1);
    HttpRuntimeStats stats;
    {
        auto request = stats.beginRequest({base}, base + seconds(2));
        request.prefillStarted(base + seconds(3));
        request.prefillFinished(base + seconds(5));
        request.modelTokenObserved(base + seconds(6));
        request.modelTokenObserved(base + seconds(7));
        request.outputPublished(base + milliseconds(6500));
        request.decodeFinished(base + seconds(7));
        request.complete(100, 20, prefix(100, 0), HttpRuntimeStats::Delivery::Completed, base + seconds(8));
    }
    {
        auto request = stats.beginRequest({base + seconds(10)}, base + seconds(11));
        request.prefillStarted(base + seconds(12));
        request.prefillFinished(base + seconds(20));
        request.modelTokenObserved(base + seconds(22));
        request.decodeFinished(base + seconds(24));
        request.complete(1000, 60, prefix(1000, 900), HttpRuntimeStats::Delivery::Completed, base + seconds(25));
    }
    const auto s = stats.snapshot();
    EXPECT_DOUBLE_EQ(s["throughput"]["prefill_tokens_per_second"].get<double>(), 20);
    EXPECT_DOUBLE_EQ(s["throughput"]["effective_prompt_tokens_per_second"].get<double>(), 110);
    EXPECT_DOUBLE_EQ(s["throughput"]["decode_tokens_per_second"].get<double>(), 80.0 / 6);
    EXPECT_EQ(s["timings"]["ttft"]["samples"], 2);
    EXPECT_EQ(s["timings"]["ttft"]["total_seconds"], 18);
    EXPECT_EQ(s["timings"]["ttft"]["average_seconds"], 9);
    EXPECT_EQ(s["timings"]["ttft"]["min_seconds"], 6);
    EXPECT_EQ(s["timings"]["ttft"]["max_seconds"], 12);
    EXPECT_EQ(s["timings"]["latency"]["average_seconds"], 11.5);
    EXPECT_EQ(s["timings"]["queue"]["average_seconds"], 1.5);
    EXPECT_EQ(s["timings"]["first_output"]["samples"], 1);
    EXPECT_EQ(s["timings"]["first_output"]["average_seconds"], 6.5);
    EXPECT_EQ(s["last_request"]["throughput"]["prefill_tokens_per_second"], 12.5);
    EXPECT_EQ(s["last_request"]["throughput"]["decode_tokens_per_second"], 15);
    EXPECT_EQ(s["last_request"]["timings"]["ttft_seconds"], 12);
    EXPECT_EQ(s["last_request"]["timings"]["queue_seconds"], 1);
    EXPECT_EQ(s["last_request"]["mtp"]["acceptance_rate"], nullptr);
}

/** @brief Zero-duration measurements and empty output remain defined JSON with null rates/TTFT. */
TEST(Test__HttpRuntimeStats, EmptyOutputAndZeroDurationNeverCreateInfinity)
{
    const auto now = HttpRuntimeStats::Clock::now();
    HttpRuntimeStats stats;
    {
        auto request = stats.beginRequest({now}, now);
        request.prefillStarted(now);
        request.prefillFinished(now);
        request.decodeFinished(now);
        request.complete(100, 0, prefix(100, 100), HttpRuntimeStats::Delivery::Completed, now);
    }
    const auto s = stats.snapshot();
    EXPECT_TRUE(s["last_request"]["timings"]["ttft_seconds"].is_null());
    EXPECT_TRUE(s["throughput"]["decode_tokens_per_second"].is_null());
    EXPECT_TRUE(s["throughput"]["prefill_tokens_per_second"].is_null());
    EXPECT_EQ(s["throughput"]["prefill_uncached_tokens"], 0);
    EXPECT_EQ(s["timings"]["decode"]["samples"], 1);
    EXPECT_NO_THROW({ const auto round_trip = nlohmann::json::parse(s.dump()); EXPECT_EQ(round_trip, s); });
}

/** @brief Last failure is visible without erasing prior successes or fabricating cache work. */
TEST(Test__HttpRuntimeStats, LastFailureReplacesSuccessWithoutFakeUsage)
{
    HttpRuntimeStats stats;
    {
        auto request = stats.beginRequest();
        request.complete(100, 10, prefix(100, 80), HttpRuntimeStats::Delivery::Completed);
    }
    {
        auto request = stats.beginRequest();
        request.fail(400);
    }
    const auto s = stats.snapshot();
    EXPECT_EQ(s["requests"]["completed"], 1);
    EXPECT_EQ(s["requests"]["failed"], 1);
    EXPECT_EQ(s["last_request"]["outcome"], "failed");
    EXPECT_EQ(s["last_request"]["handler_status"], 400);
    EXPECT_TRUE(s["last_request"]["prompt_tokens"].is_null());
    EXPECT_EQ(s["tokens"]["prompt"], 100);
    EXPECT_EQ(s["requests"]["active_phase"], "idle");
}

/** @brief Every valid HTTP status has one bounded slot; invalid values cannot grow a map. */
TEST(Test__HttpRuntimeStats, HTTPStatusInventoryIncludesEveryResponseClass)
{
    HttpRuntimeStats stats;
    for (int status = 100; status <= 599; ++status)
        stats.recordHttpResponse(status);
    stats.recordHttpResponse(200);
    stats.recordHttpResponse(-1);
    stats.recordHttpResponse(1000000);
    const auto s = stats.snapshot();
    EXPECT_EQ(s["http"]["responses"], 503);
    EXPECT_EQ(s["http"]["status_codes"].size(), 500);
    EXPECT_EQ(s["http"]["status_codes"]["200"], 2);
    EXPECT_EQ(s["http"]["status_codes"]["503"], 1);
    EXPECT_EQ(s["http"]["unclassified"], 2);
    EXPECT_EQ(s["requests"]["started"], 0);
}

/** @brief Unequal speculative windows require token-weighted, not request-averaged acceptance. */
TEST(Test__HttpRuntimeStats, MTPAcceptanceWeightsVerifiedTokens)
{
    HttpRuntimeStats stats;
    for (const auto &[accepted, rejected] : {std::pair{1U, 1U}, {9U, 1U}})
    {
        auto summary = prefix(100, 0);
        summary.mtp_request.enabled = true;
        summary.mtp_request.accepted_tokens = accepted;
        summary.mtp_request.rejected_tokens = rejected;
        summary.mtp_request.stochastic_accept_tests = accepted + rejected;
        summary.mtp_request.stochastic_accepts = accepted;
        auto request = stats.beginRequest();
        request.complete(100, 20, summary, HttpRuntimeStats::Delivery::Completed);
    }
    const auto s = stats.snapshot();
    EXPECT_DOUBLE_EQ(s["mtp"]["acceptance_rate"].get<double>(), 10.0 / 12);
    EXPECT_DOUBLE_EQ(s["mtp"]["stochastic_acceptance_rate"].get<double>(), 10.0 / 12);
    EXPECT_DOUBLE_EQ(s["last_request"]["mtp"]["acceptance_rate"].get<double>(), .9);
}

/** @brief Polling cannot expose a partially published next request or rewrite the previous one. */
TEST(Test__HttpRuntimeStats, LastRequestFreshnessWaitsForCompleteScopeRetirement)
{
    using namespace std::chrono;
    const auto old = HttpRuntimeStats::Clock::now() - seconds(30);
    HttpRuntimeStats stats;
    {
        auto first = stats.beginRequest({old}, old);
        first.complete(100, 10, prefix(100, 80), HttpRuntimeStats::Delivery::Completed, old);
        // Selecting an outcome precedes cache cleanup; it must still be active.
        EXPECT_EQ(stats.snapshot()["requests"]["active"], 1);
        EXPECT_TRUE(stats.snapshot()["last_request"].is_null());
    }
    const auto initial = stats.snapshot();
    EXPECT_EQ(initial["last_request"]["sequence"], 1);
    EXPECT_GE(initial["last_request"]["age_seconds"].get<double>(), 30);
    {
        auto second = stats.beginRequest();
        second.prefillStarted();
        for (int n = 0; n < 20; ++n)
        {
            const auto active = stats.snapshot();
            EXPECT_EQ(active["requests"]["active_sequence"], 2);
            EXPECT_EQ(active["last_request"]["sequence"], 1);
            EXPECT_EQ(active["tokens"]["completion"], 10);
            EXPECT_GE(active["last_request"]["age_seconds"].get<double>(),
                      initial["last_request"]["age_seconds"].get<double>());
        }
        second.prefillFinished();
        second.decodeFinished();
        second.complete(1000, 20, prefix(1000, 0), HttpRuntimeStats::Delivery::Completed);
        EXPECT_EQ(stats.snapshot()["last_request"]["sequence"], 1);
    }
    const auto fresh = stats.snapshot();
    EXPECT_EQ(fresh["last_request"]["sequence"], 2);
    EXPECT_EQ(fresh["tokens"]["completion"], 30);
    EXPECT_LT(fresh["last_request"]["age_seconds"].get<double>(), 30);
    EXPECT_EQ(fresh["requests"]["active"], 0);
    EXPECT_EQ(fresh["prefix_cache"]["matched_tokens"], 80);
}

/** @brief An active old-epoch request cannot resurrect any reset usage, timings or last outcome. */
TEST(Test__HttpRuntimeStats, ResetExcludesActiveOldEpochAndPreservesLiveVisibility)
{
    HttpRuntimeStats stats;
    {
        auto old_request = stats.beginRequest();
        old_request.prefillStarted();
        stats.recordHttpResponse(500);
        EXPECT_EQ(stats.reset(), 1);
        const auto reset = stats.snapshot();
        EXPECT_EQ(reset["epoch"], 1);
        EXPECT_EQ(reset["requests"]["started"], 0);
        EXPECT_EQ(reset["requests"]["active"], 1);
        EXPECT_EQ(reset["requests"]["active_in_current_epoch"], 0);
        EXPECT_EQ(reset["requests"]["active_phase"], "prefill");
        EXPECT_EQ(reset["http"]["responses"], 0);
        EXPECT_TRUE(reset["last_request"].is_null());
        old_request.prefillFinished();
        old_request.modelTokenObserved();
        old_request.decodeFinished();
        old_request.complete(1000, 40, prefix(1000, 900), HttpRuntimeStats::Delivery::Completed);
    }
    const auto retired = stats.snapshot();
    EXPECT_EQ(retired["requests"]["active"], 0);
    EXPECT_EQ(retired["requests"]["completed"], 0);
    EXPECT_EQ(retired["tokens"]["prompt"], 0);
    EXPECT_EQ(retired["timings"]["ttft"]["samples"], 0);
    EXPECT_TRUE(retired["last_request"].is_null());
    {
        auto fresh = stats.beginRequest();
        fresh.complete(100, 10, prefix(100, 80), HttpRuntimeStats::Delivery::Completed);
    }
    EXPECT_EQ(stats.snapshot()["last_request"]["sequence"], 2);
    EXPECT_EQ(stats.snapshot()["requests"]["completed"], 1);
    EXPECT_EQ(stats.reset(), 2);
    EXPECT_TRUE(stats.snapshot()["last_request"].is_null());
    EXPECT_GE(stats.snapshot()["uptime_seconds"].get<double>(), retired["uptime_seconds"].get<double>());
}

/** @brief Concurrent resets and publications preserve epoch accounting and atomic HTTP snapshots. */
TEST(Test__HttpRuntimeStats, ConcurrentResetCannotMixEpochTotals)
{
    HttpRuntimeStats stats;
    std::atomic<bool> finished{false};
    std::thread publisher([&] {
        for (int n = 0; n < 2000; ++n)
        {
            auto request = stats.beginRequest();
            request.complete(100, 10, prefix(100, 80), HttpRuntimeStats::Delivery::Completed);
            stats.recordHttpResponse(200);
        }
        finished = true;
    });
    while (!finished)
    {
        stats.reset();
        const auto snapshot = stats.snapshot();
        const auto completed = snapshot["requests"]["completed"].get<uint64_t>();
        EXPECT_EQ(snapshot["tokens"]["prompt"], completed * 100);
        EXPECT_EQ(snapshot["prefix_cache"]["matched_tokens"], completed * 80);
        EXPECT_EQ(snapshot["requests"]["started"].get<uint64_t>(),
                  completed + snapshot["requests"]["active_in_current_epoch"].get<uint64_t>());
        const auto http_count = snapshot["http"]["responses"].get<uint64_t>();
        EXPECT_EQ(snapshot["http"]["status_codes"].value("200", 0ULL), http_count);
    }
    publisher.join();
}
