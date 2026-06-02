#include "utils/PerfStatsCollector.h"
#include "utils/KernelProfiler.h"
#include "utils/KVCacheProfiler.h"
#include "utils/WeightLoadingProfiler.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace llaminar2;

namespace
{
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_value_;
    };

    std::filesystem::path uniqueTempPath(const std::string &suffix)
    {
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() /
               ("llaminar_perf_stats_test_" + std::to_string(ticks) + suffix);
    }

    std::string readFile(const std::filesystem::path &path)
    {
        std::ifstream in(path);
        std::stringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
    }
}

TEST(Test__PerfStatsCollector, AggregatesCountersAndTimers)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    PerfStatsCollector::addCounter("mtp", "draft_steps", 1.0, "decode");
    PerfStatsCollector::addCounter("mtp", "draft_steps", 2.0, "decode");
    PerfStatsCollector::recordTimingNs("mtp", "sidecar_forward", 1000, "decode", "rocm:0");
    PerfStatsCollector::recordTimingNs("mtp", "sidecar_forward", 3000, "decode", "rocm:0");

    const auto records = PerfStatsCollector::snapshot();
    ASSERT_EQ(records.size(), 2u);

    const auto counter_it = std::find_if(records.begin(), records.end(), [](const auto &record)
                                         {
                                             return record.kind == PerfStatRecord::Kind::Counter &&
                                                    record.domain == "mtp" &&
                                                    record.name == "draft_steps";
                                         });
    ASSERT_NE(counter_it, records.end());
    EXPECT_EQ(counter_it->count, 2u);
    EXPECT_DOUBLE_EQ(counter_it->value, 3.0);

    const auto timer_it = std::find_if(records.begin(), records.end(), [](const auto &record)
                                       {
                                           return record.kind == PerfStatRecord::Kind::Timer &&
                                                  record.domain == "mtp" &&
                                                  record.name == "sidecar_forward";
                                       });
    ASSERT_NE(timer_it, records.end());
    EXPECT_EQ(timer_it->count, 2u);
    EXPECT_EQ(timer_it->total_ns, 4000u);
    EXPECT_EQ(timer_it->min_ns, 1000u);
    EXPECT_EQ(timer_it->max_ns, 3000u);
    EXPECT_EQ(timer_it->device, "rocm:0");
}

TEST(Test__PerfStatsCollector, ExportsFilteredJsonAndCsv)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    PerfStatsCollector::addCounter("mtp", "accepted_tokens", 2.0, "decode");
    PerfStatsCollector::addCounter("prefix_cache", "hits", 1.0, "prefill");

    const auto json = nlohmann::json::parse(PerfStatsCollector::jsonString({"mtp"}));
    ASSERT_EQ(json.at("schema"), "llaminar.perf_stats.v1");
    ASSERT_EQ(json.at("records").size(), 1u);
    EXPECT_EQ(json.at("records")[0].at("domain"), "mtp");
    EXPECT_EQ(json.at("records")[0].at("name"), "accepted_tokens");
    EXPECT_DOUBLE_EQ(json.at("records")[0].at("value").get<double>(), 2.0);

    const std::string csv = PerfStatsCollector::csvString({"mtp"});
    EXPECT_NE(csv.find("kind,domain,name,phase,device,tags,count,value"), std::string::npos);
    EXPECT_NE(csv.find("counter,mtp,accepted_tokens,decode"), std::string::npos);
    EXPECT_EQ(csv.find("prefix_cache"), std::string::npos);
}

TEST(Test__PerfStatsCollector, SummaryTableCanBeRequestedByEnv)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_TABLE", "1");
    PerfStatsCollector::reset();

    ASSERT_TRUE(PerfStatsCollector::isEnabled());

    PerfStatsCollector::addCounter("mtp", "draft_steps", 3.0, "decode", "rocm:0");
    PerfStatsCollector::recordTimingNs("mtp", "sidecar_forward", 5000, "decode", "rocm:0");

    const std::string summary = PerfStatsCollector::summaryString({"mtp"});
    EXPECT_NE(summary.find("UNIFIED PERF STATS"), std::string::npos);
    EXPECT_NE(summary.find("mtp.draft_steps"), std::string::npos);
    EXPECT_NE(summary.find("mtp.sidecar_forward"), std::string::npos);
}

TEST(Test__PerfStatsCollector, ExistingProfilersPublishStructuredRecords)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();
    KernelProfiler::resetAll();
    KVCacheProfiler::reset();
    WeightLoadingProfiler::reset();

    KernelProfiler::setCurrentPhase(KernelProfiler::Phase::DECODE);
    KernelProfiler::record(KernelType::LM_HEAD, 2000, "rocm:0");

    KVCacheProfiler::setCurrentPhase(KVCacheProfiler::Phase::DECODE);
    KVCacheProfiler::record(KVCacheOpType::APPEND, 3000, 4, 128);

    WeightLoadingProfiler::addDetail("weights.gemm_pack.test", 0.25);

    const auto records = PerfStatsCollector::snapshot();
    auto has_record = [&](const std::string &domain, const std::string &name)
    {
        return std::any_of(records.begin(), records.end(), [&](const auto &record)
                           {
                               return record.domain == domain && record.name == name;
                           });
    };

    EXPECT_TRUE(has_record("kernel", "LM_HEAD"));
    EXPECT_TRUE(has_record("kv_cache", "KV_APPEND"));
    EXPECT_TRUE(has_record("kv_cache", "tokens"));
    EXPECT_TRUE(has_record("kv_cache", "bytes"));
    EXPECT_TRUE(has_record("weight_loading", "weights.gemm_pack.test"));
}

TEST(Test__PerfStatsCollector, FlushFromEnvWritesMachineReadableFiles)
{
    const auto json_path = uniqueTempPath(".json");
    const auto csv_path = uniqueTempPath(".csv");
    ScopedEnv json_env("LLAMINAR_PERF_STATS_JSON", json_path.string().c_str());
    ScopedEnv csv_env("LLAMINAR_PERF_STATS_CSV", csv_path.string().c_str());
    ScopedEnv filter_env("LLAMINAR_PERF_STATS_FILTER", "mtp");
    PerfStatsCollector::reset();

    PerfStatsCollector::recordTimingNs(
        "mtp",
        "verifier_forward",
        123456,
        "decode",
        "cpu",
        {{"depth", "0"}});
    PerfStatsCollector::addCounter("kernel", "gemm_calls", 9.0, "decode");

    ASSERT_TRUE(PerfStatsCollector::flushFromEnv());
    ASSERT_TRUE(std::filesystem::exists(json_path));
    ASSERT_TRUE(std::filesystem::exists(csv_path));

    const auto json = nlohmann::json::parse(readFile(json_path));
    ASSERT_EQ(json.at("records").size(), 1u);
    EXPECT_EQ(json.at("records")[0].at("domain"), "mtp");
    EXPECT_EQ(json.at("records")[0].at("name"), "verifier_forward");
    EXPECT_EQ(json.at("records")[0].at("tags").at("depth"), "0");

    const std::string csv = readFile(csv_path);
    EXPECT_NE(csv.find("timer,mtp,verifier_forward,decode,cpu,depth=0"), std::string::npos);
    EXPECT_EQ(csv.find("kernel"), std::string::npos);

    std::filesystem::remove(json_path);
    std::filesystem::remove(csv_path);
}

TEST(Test__PerfStatsCollector, FlushFromEnvWritesMultipleFilteredDomainsForMTPGraphEvidence)
{
    const auto json_path = uniqueTempPath(".json");
    const auto csv_path = uniqueTempPath(".csv");
    ScopedEnv json_env("LLAMINAR_PERF_STATS_JSON", json_path.string().c_str());
    ScopedEnv csv_env("LLAMINAR_PERF_STATS_CSV", csv_path.string().c_str());
    ScopedEnv filter_env("LLAMINAR_PERF_STATS_FILTER", "mtp,forward_graph");
    PerfStatsCollector::reset();

    PerfStatsCollector::recordTimingNs(
        "mtp",
        "verifier_forward",
        2000000,
        "decode",
        "cuda:0");
    PerfStatsCollector::recordTimingNs(
        "forward_graph",
        "segmented_replay_total",
        750000,
        "decode",
        "cuda:0",
        {{"context", "mtp_decode_sidecar"}});
    PerfStatsCollector::addCounter("kernel", "gemm_calls", 3.0, "decode");

    ASSERT_TRUE(PerfStatsCollector::flushFromEnv());

    const auto json = nlohmann::json::parse(readFile(json_path));
    ASSERT_EQ(json.at("records").size(), 2u);

    bool saw_mtp_verifier = false;
    bool saw_graph_replay = false;
    for (const auto &record : json.at("records"))
    {
        const std::string domain = record.at("domain").get<std::string>();
        const std::string name = record.at("name").get<std::string>();
        EXPECT_NE(domain, "kernel");
        saw_mtp_verifier = saw_mtp_verifier ||
                           (domain == "mtp" && name == "verifier_forward");
        saw_graph_replay = saw_graph_replay ||
                           (domain == "forward_graph" && name == "segmented_replay_total");
    }
    EXPECT_TRUE(saw_mtp_verifier);
    EXPECT_TRUE(saw_graph_replay);

    const std::string csv = readFile(csv_path);
    EXPECT_NE(csv.find("timer,mtp,verifier_forward,decode,cuda:0"), std::string::npos);
    EXPECT_NE(csv.find("timer,forward_graph,segmented_replay_total,decode,cuda:0"), std::string::npos);
    EXPECT_EQ(csv.find("kernel"), std::string::npos);

    std::filesystem::remove(json_path);
    std::filesystem::remove(csv_path);
}
