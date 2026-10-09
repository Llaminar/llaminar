/**
 * @file Test__HttpProcessStats.cpp
 * @brief Device-free proofs of cached OS metrics, freshness and HTTP flood isolation.
 *
 * Blocked collectors and concurrent readers prove that request frequency cannot
 * amplify OS work. Real procfs/statvfs observations and independent metadata
 * fixtures authenticate units, process identity and partial-error semantics.
 */
#include "app/modes/HttpProcessStats.h"
#include <gtest/gtest.h>

#include <atomic>
#include <fstream>
#include <future>
#include <semaphore>
#include <sys/statvfs.h>
#include <unistd.h>

using namespace llaminar2;
using namespace std::chrono_literals;
using json = nlohmann::json;

namespace
{
    /** @brief Await collector publication with a short deadlock bound, never an inference deadline. */
    json awaitAttempts(const HttpProcessStats &stats, uint64_t attempts)
    {
        const auto deadline = HttpProcessStats::Clock::now() + 2s;
        while (HttpProcessStats::Clock::now() < deadline)
        {
            auto value = stats.snapshot();
            if (value.at("collection").at("attempts").get<uint64_t>() >= attempts) return value;
            std::this_thread::sleep_for(1ms);
        }
        throw std::runtime_error("collector failed to publish");
    }

    /** @brief Release a blocked sample even when a fatal assertion exits its test. */
    struct ReleaseSample
    {
        std::binary_semaphore &gate;
        /** @brief Unblock the worker before its owning sampler is destroyed. */
        ~ReleaseSample() { gate.release(); }
    };

    /** @brief Temporary procfs fixtures with deterministic units and cgroup identity. */
    class ProcessFixture : public testing::Test
    {
    protected:
        std::filesystem::path root;
        /** @brief Create small independent text fixtures, never models or device contexts. */
        void SetUp() override
        {
            char path[] = "/tmp/llaminar-process-stats-XXXXXX";
            root = ::mkdtemp(path);
            std::filesystem::create_directories(root / "proc/self/fd");
            std::filesystem::create_directories(root / "cgroup/work");
            write("proc/self/status", "Pid: 123\nVmRSS: 21 kB\nVmHWM: 30 kB\nVmSize: 80 kB\nRssAnon: 11 kB\nRssFile: 7 kB\nRssShmem: 3 kB\nVmSwap: 2 kB\nThreads: 4\n");
            std::string stat = "123 (a name with ) parentheses) S";
            for (int field = 4; field <= 22; ++field)
                stat += " " + std::to_string(field == 14 ? 200 : field == 15 ? 100 : field == 22 ? 9876 : 0);
            write("proc/self/stat", stat + "\n");
            write("proc/self/io", "read_bytes: 77\nwrite_bytes: 88\nsyscr: 9\nsyscw: 10\n");
            write("proc/meminfo", "MemTotal: 100 kB\nMemAvailable: 60 kB\nMemFree: 20 kB\nCached: 30 kB\nBuffers: 10 kB\nSwapTotal: 10 kB\nSwapFree: 8 kB\nDirty: 1 kB\nWriteback: 0 kB\n");
            write("proc/loadavg", "1.25 2.5 3.75 1/10 123\n");
            write("proc/self/cgroup", "0::/work\n");
            write("cgroup/work/memory.current", "4096\n");
            write("cgroup/work/memory.peak", "8192\n");
            write("cgroup/work/memory.max", "max\n");
            write("cgroup/work/memory.swap.current", "1024\n");
            write("cgroup/work/memory.stat", "anon 100\nfile 200\nshmem 150\n");
            write("cgroup/work/memory.events", "oom 0\noom_kill 0\n");
            write("cgroup/work/cpu.stat", "usage_usec 1000\n");
        }
        /** @brief Retire only this fixture's small metadata directory. */
        void TearDown() override { std::filesystem::remove_all(root); }
        /** @brief Replace one explicit metadata fixture. */
        void write(const std::filesystem::path &name, const std::string &text)
        {
            std::ofstream(root / name) << text;
        }
        /** @brief Bind a parser to the fixture with a real local filesystem probe. */
        HttpProcessStatsReader reader()
        {
            return HttpProcessStatsReader({root}, root / "proc", root / "cgroup");
        }
    };
}

TEST(Test__HttpProcessStats, BlockedFirstProbeDoesNotBlockOrAmplifyConcurrentReads)
{
    std::atomic<unsigned> calls = 0;
    std::binary_semaphore entered{0}, release{0};
    HttpProcessStats stats({.interval = 1h}, [&] {
        ++calls;
        entered.release();
        release.acquire();
        return json{{"fixture", 123}};
    });
    ReleaseSample unblock{release};
    ASSERT_TRUE(entered.try_acquire_for(2s));
    std::atomic<unsigned> invalid = 0;
    std::vector<std::jthread> readers;
    for (unsigned thread = 0; thread < 8; ++thread)
        readers.emplace_back([&] {
            for (unsigned call = 0; call < 1000; ++call)
            {
                const auto value = stats.snapshot();
                if (value["collection"]["available"] != false || value["collection"]["status"] != "warming")
                    ++invalid;
            }
        });
    readers.clear();
    EXPECT_EQ(invalid.load(), 0u);
    EXPECT_EQ(calls.load(), 1u);
}

TEST(Test__HttpProcessStats, FailedRefreshRetainsLastGoodDataWithAgeAndError)
{
    std::atomic<unsigned> calls = 0;
    HttpProcessStats stats({.interval = 2ms}, [&]() -> json {
        if (++calls == 1) return {{"process", {{"rss_bytes", 12345}}}};
        throw std::runtime_error("controlled OS failure");
    });
    const auto value = awaitAttempts(stats, 2);
    EXPECT_EQ(value["process"]["rss_bytes"], 12345);
    EXPECT_EQ(value["collection"]["successful_samples"], 1);
    EXPECT_EQ(value["collection"]["last_error"], "controlled OS failure");
    EXPECT_TRUE(value["collection"]["available"].get<bool>());
    const auto aged = stats.snapshot(HttpProcessStats::Clock::now() + 1h);
    EXPECT_TRUE(aged["collection"]["stale"].get<bool>());
    EXPECT_GT(aged["collection"]["age_seconds"].get<double>(), 3599);
}

TEST(Test__HttpProcessStats, PeriodicRefreshHappensWithoutReaders)
{
    std::promise<void> second;
    std::atomic<unsigned> calls = 0;
    HttpProcessStats stats({.interval = 5ms}, [&] {
        const auto call = ++calls;
        if (call == 2) second.set_value();
        return json{{"value", call}};
    });
    EXPECT_EQ(second.get_future().wait_for(2s), std::future_status::ready);
    EXPECT_GE(awaitAttempts(stats, 2)["value"].get<unsigned>(), 2u);
}

TEST(Test__HttpProcessStats, MissingFirstSampleAndInvalidCollectorAreExplicit)
{
    HttpProcessStats stats({.interval = 1h}, [] { return json::array(); });
    const auto value = awaitAttempts(stats, 1);
    EXPECT_FALSE(value["collection"]["available"].get<bool>());
    EXPECT_EQ(value["collection"]["status"], "unavailable");
    EXPECT_TRUE(value["collection"]["sampled_unix_seconds"].is_null());
    EXPECT_TRUE(value["collection"]["age_seconds"].is_null());
    EXPECT_TRUE(value["collection"]["stale"].get<bool>());
    EXPECT_FALSE(value["collection"]["last_error"].is_null());
}

TEST(Test__HttpProcessStats, InvalidCadenceIsRejectedAndLongCadenceDoesNotDelayDestruction)
{
    EXPECT_THROW(HttpProcessStats stats({.interval = 0ms}), std::invalid_argument);
    const auto begin = HttpProcessStats::Clock::now();
    {
        HttpProcessStats stats({.interval = 1h}, [] { return json::object(); });
        awaitAttempts(stats, 1);
    }
    EXPECT_LT(HttpProcessStats::Clock::now() - begin, 2s);
}

TEST_F(ProcessFixture, UnitsProcessIdentityCpuAndSharedMemoryScopesAreExact)
{
    auto collect = reader();
    const auto first = collect();
    EXPECT_EQ(first["process"]["rss_bytes"], 21 * 1024);
    EXPECT_EQ(first["process"]["threads"], 4);
    EXPECT_EQ(first["process"]["pid"], 123);
    EXPECT_EQ(first["process"]["start_time_ticks"], 9876);
    EXPECT_EQ(first["process"]["io"]["read_bytes"], 77);
    EXPECT_EQ(first["host"]["memory"]["available_bytes"], 60 * 1024);
    EXPECT_EQ(first["host"]["load_average"], json::array({1.25, 2.5, 3.75}));
    EXPECT_EQ(first["cgroup"]["memory_current_bytes"], 4096);
    EXPECT_TRUE(first["cgroup"]["memory_limit_bytes"].is_null());
    EXPECT_EQ(first["cgroup"]["memory_stat"]["file"], 200);
    EXPECT_EQ(first["cgroup"]["memory_stat"]["shmem"], 150);
    EXPECT_TRUE(first["process"]["cpu"]["interval_usage_percent"].is_null());
    EXPECT_EQ(collect()["process"]["cpu"]["interval_usage_percent"], 0.0);
}

TEST_F(ProcessFixture, MissingCgroupOrFilesystemNeverFabricatesZeroUsage)
{
    write("proc/self/cgroup", "2:memory:/legacy\n");
    auto collect = HttpProcessStatsReader({root / "absent"}, root / "proc", root / "cgroup");
    const auto value = collect();
    EXPECT_TRUE(value["process"]["available"].get<bool>());
    EXPECT_FALSE(value["cgroup"]["available"].get<bool>());
    EXPECT_FALSE(value["cgroup"].contains("memory_current_bytes"));
    EXPECT_FALSE(value["filesystems"][0]["available"].get<bool>());
    EXPECT_FALSE(value["filesystems"][0].contains("available_bytes"));
}

TEST_F(ProcessFixture, InvalidIdentityOversizedMetadataAndCounterRegressionFail)
{
    auto collect = reader();
    static_cast<void>(collect());
    write("proc/self/stat", "123 (test) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9876\n");
    EXPECT_THROW(static_cast<void>(collect()), std::runtime_error);
    write("proc/self/stat", "broken\n");
    EXPECT_THROW(static_cast<void>(collect()), std::runtime_error);
    write("proc/self/stat", std::string(65537, 'x'));
    EXPECT_THROW(static_cast<void>(collect()), std::runtime_error);
    write("proc/self/cgroup", "0::/../escape\n");
    write("proc/self/stat", "123 (test) S 0 0 0 0 0 0 0 0 0 0 200 100 0 0 0 0 0 0 9876\n");
    EXPECT_FALSE(collect()["cgroup"]["available"].get<bool>());
}

TEST(Test__HttpProcessStats, ActualLinuxProcessAndFilesystemMetadataAreConsistent)
{
    HttpProcessStatsReader collect({"/tmp", "/tmp"});
    const auto value = collect();
    EXPECT_EQ(value["process"]["pid"], ::getpid());
    EXPECT_GT(value["process"]["rss_bytes"].get<uint64_t>(), 0u);
    EXPECT_GE(value["process"]["peak_rss_bytes"].get<uint64_t>(), value["process"]["rss_bytes"].get<uint64_t>());
    ASSERT_EQ(value["filesystems"].size(), 1u);
    const auto fs = value["filesystems"][0];
    ASSERT_TRUE(fs["available"].get<bool>());
    EXPECT_EQ(fs["capacity_bytes"].get<uint64_t>(), fs["free_bytes"].get<uint64_t>() + fs["used_bytes"].get<uint64_t>());
    EXPECT_LE(fs["available_bytes"].get<uint64_t>(), fs["free_bytes"].get<uint64_t>());
}
