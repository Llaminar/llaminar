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

        static bool isEnabled();
        static void reset();

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
