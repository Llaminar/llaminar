/**
 * @file PerfStatsCollector.h
 * @brief Unified structured performance counter and timer collection.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llaminar2
{
    struct PerfStatRecord
    {
        enum class Kind
        {
            Counter,
            Timer
        };

        Kind kind = Kind::Counter;
        std::string domain;
        std::string name;
        std::string phase;
        std::string device;
        std::map<std::string, std::string> tags;

        uint64_t count = 0;
        double value = 0.0;
        uint64_t total_ns = 0;
        uint64_t min_ns = 0;
        uint64_t max_ns = 0;
    };

    class PerfStatsCollector
    {
    public:
        using Tags = std::map<std::string, std::string>;
        using Clock = std::chrono::steady_clock;

        /**
         * @brief Exact record family retained across a measurement reset.
         *
         * Domain-only retention is appropriate for wholly setup-owned domains
         * such as graph kernel inventory. Mixed domains need a narrower key:
         * graph capture evidence and measured replay counters both live in
         * `forward_graph`, and retaining that complete domain would contaminate
         * the benchmark window with warmup runtime activity.
         */
        struct RecordFamily
        {
            std::string domain; ///< Exact PerfStats domain.
            std::string name;   ///< Exact counter/timer name within the domain.
        };

        static bool isEnabled();
        static bool gpuStageEventTimingEnabled();
        static void reset();
        static void resetPreservingDomains(
            const std::vector<std::string> &domains_to_preserve);

        /**
         * @brief Reset measured evidence while retaining named setup records.
         *
         * A record survives when either its complete domain appears in
         * `domains_to_preserve` or its exact `(domain, name)` pair appears in
         * `record_families_to_preserve`. Tags, phase, device, count, and timing
         * aggregates remain intact for surviving records.
         *
         * @param domains_to_preserve Complete domains owned by immutable setup.
         * @param record_families_to_preserve Exact families from mixed domains.
         */
        static void resetPreserving(
            const std::vector<std::string> &domains_to_preserve,
            const std::vector<RecordFamily> &record_families_to_preserve);

        static void addCounter(
            std::string domain,
            std::string name,
            double value = 1.0,
            std::string phase = {},
            std::string device = {},
            Tags tags = {});

        static void recordTimingNs(
            std::string domain,
            std::string name,
            uint64_t duration_ns,
            std::string phase = {},
            std::string device = {},
            Tags tags = {});

        static std::vector<PerfStatRecord> snapshot(
            const std::vector<std::string> &filters = {});

        static std::string jsonString(
            const std::vector<std::string> &filters = {});

        static std::string csvString(
            const std::vector<std::string> &filters = {});

        static std::string summaryString(
            const std::vector<std::string> &filters = {},
            size_t max_records = 120);

        static bool writeJson(
            const std::string &path,
            const std::vector<std::string> &filters = {});

        static bool writeCsv(
            const std::string &path,
            const std::vector<std::string> &filters = {});

        static void printSummary(
            const std::vector<std::string> &filters = {},
            size_t max_records = 120);

        static bool flushFromEnv();

        class ScopedTimer
        {
        public:
            ScopedTimer(
                std::string domain,
                std::string name,
                std::string phase = {},
                std::string device = {},
                Tags tags = {});

            ~ScopedTimer();

            ScopedTimer(const ScopedTimer &) = delete;
            ScopedTimer &operator=(const ScopedTimer &) = delete;
            ScopedTimer(ScopedTimer &&) = delete;
            ScopedTimer &operator=(ScopedTimer &&) = delete;

        private:
            bool enabled_ = false;
            std::string domain_;
            std::string name_;
            std::string phase_;
            std::string device_;
            Tags tags_;
            Clock::time_point start_{};
        };
    };

} // namespace llaminar2
