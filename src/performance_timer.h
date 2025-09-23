#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "logger.h"

namespace llaminar
{

    class PerformanceTimer
    {
    public:
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;
        using Duration = std::chrono::duration<double, std::milli>;

        struct TimingResult
        {
            std::string name;
            double time_ms;
            size_t call_count;
            double total_time_ms;
            double avg_time_ms;
            double min_time_ms;
            double max_time_ms;
        };

        static PerformanceTimer &getInstance()
        {
            static PerformanceTimer instance;
            return instance;
        }

        void startTimer(const std::string &name)
        {
            start_times_[name] = Clock::now();
        }

        void endTimer(const std::string &name)
        {
            auto end_time = Clock::now();
            auto it = start_times_.find(name);
            if (it != start_times_.end())
            {
                auto duration = Duration(end_time - it->second).count();
                addTiming(name, duration);
                start_times_.erase(it);
            }
        }

        void addTiming(const std::string &name, double time_ms)
        {
            auto &timing = timings_[name];
            timing.name = name;
            timing.call_count++;
            timing.total_time_ms += time_ms;
            timing.avg_time_ms = timing.total_time_ms / timing.call_count;

            if (timing.call_count == 1)
            {
                timing.min_time_ms = time_ms;
                timing.max_time_ms = time_ms;
            }
            else
            {
                timing.min_time_ms = std::min(timing.min_time_ms, time_ms);
                timing.max_time_ms = std::max(timing.max_time_ms, time_ms);
            }
            timing.time_ms = time_ms; // Last recorded time
        }

        TimingResult getTiming(const std::string &name) const
        {
            auto it = timings_.find(name);
            if (it != timings_.end())
            {
                return it->second;
            }
            return {};
        }

        std::vector<TimingResult> getAllTimings() const
        {
            std::vector<TimingResult> results;
            for (const auto &pair : timings_)
            {
                results.push_back(pair.second);
            }
            return results;
        }

        void printReport(const std::string &context = "") const
        {
            LOG_INFO("=== Performance Report" << (context.empty() ? "" : " - " + context) << " ===");
            for (const auto &pair : timings_)
            {
                const auto &timing = pair.second;
                LOG_INFO(timing.name << ": "
                                     << timing.time_ms << "ms (last) | "
                                     << timing.avg_time_ms << "ms (avg) | "
                                     << timing.min_time_ms << "ms (min) | "
                                     << timing.max_time_ms << "ms (max) | "
                                     << timing.call_count << " calls | "
                                     << timing.total_time_ms << "ms (total)");
            }
            LOG_INFO("=== End Performance Report ===");
        }

        void reset()
        {
            timings_.clear();
            start_times_.clear();
        }

    private:
        std::unordered_map<std::string, TimingResult> timings_;
        std::unordered_map<std::string, TimePoint> start_times_;
    };

    // RAII timer for automatic start/stop
    class ScopedTimer
    {
    public:
        explicit ScopedTimer(const std::string &name) : name_(name)
        {
            PerformanceTimer::getInstance().startTimer(name_);
        }

        ~ScopedTimer()
        {
            PerformanceTimer::getInstance().endTimer(name_);
        }

    private:
        std::string name_;
    };

// Convenience macros
#define PERF_TIMER_START(name) PerformanceTimer::getInstance().startTimer(name)
#define PERF_TIMER_END(name) PerformanceTimer::getInstance().endTimer(name)
#define PERF_SCOPED_TIMER(name) ScopedTimer _timer(name)

} // namespace llaminar