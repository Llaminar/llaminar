/**
 * @file WeightLoadingProfiler.h
 * @brief Profiling for weight loading phases (GGUF parse, tensor load, repack, device upload)
 *
 * Captures timing for each phase of weight loading when LLAMINAR_PROFILING=1 is set.
 * Prints a summary table alongside the other profiling summaries in benchmark mode.
 *
 * Usage:
 *   WeightLoadingProfiler::begin(WeightLoadPhase::GGUF_PARSE);
 *   // ... parse GGUF ...
 *   WeightLoadingProfiler::end(WeightLoadPhase::GGUF_PARSE);
 *
 * Or with RAII:
 *   {
 *       ScopedWeightLoadTimer timer(WeightLoadPhase::GEMM_PACK);
 *       // ... pack weights ...
 *   }
 */
#pragma once

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DebugEnv.h"
#include "PerfStatsCollector.h"
#include "fort.hpp"

namespace llaminar2
{

    enum class WeightLoadPhase : uint8_t
    {
        GGUF_PARSE = 0,    ///< Parse GGUF file header, metadata, tensor directory
        TENSOR_LOAD,       ///< Read tensor data from disk into host memory
        GEMM_PACK,         ///< CPU-side weight repacking for GEMM kernels
        DEVICE_UPLOAD,     ///< Host-to-device transfer (non-GEMM weights)
        GRAPH_BUILD,       ///< Compute graph construction (stage creation, buffer allocation)
        COUNT
    };

    inline const char *weightLoadPhaseName(WeightLoadPhase phase)
    {
        switch (phase)
        {
        case WeightLoadPhase::GGUF_PARSE:
            return "GGUF Parse";
        case WeightLoadPhase::TENSOR_LOAD:
            return "Tensor Load (disk → host)";
        case WeightLoadPhase::GEMM_PACK:
            return "GEMM Repack (host→device)";
        case WeightLoadPhase::DEVICE_UPLOAD:
            return "Non-GEMM Upload (host→device)";
        case WeightLoadPhase::GRAPH_BUILD:
            return "Graph Build";
        default:
            return "UNKNOWN";
        }
    }

    inline const char *weightLoadPhaseKey(WeightLoadPhase phase)
    {
        switch (phase)
        {
        case WeightLoadPhase::GGUF_PARSE:
            return "gguf_parse";
        case WeightLoadPhase::TENSOR_LOAD:
            return "tensor_load";
        case WeightLoadPhase::GEMM_PACK:
            return "gemm_pack";
        case WeightLoadPhase::DEVICE_UPLOAD:
            return "device_upload";
        case WeightLoadPhase::GRAPH_BUILD:
            return "graph_build";
        default:
            return "unknown";
        }
    }

    /**
     * @brief Singleton profiler for weight loading phases
     *
     * Simple wall-clock timing per phase. Not thread-safe (weight loading
     * is single-threaded per rank).
     */
    class WeightLoadingProfiler
    {
    public:
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;

        static constexpr size_t PHASE_COUNT = static_cast<size_t>(WeightLoadPhase::COUNT);

        static bool isEnabled()
        {
            return debugEnv().profile.enabled || PerfStatsCollector::isEnabled();
        }

        static void begin(WeightLoadPhase phase)
        {
            if (!isEnabled())
                return;
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            inst.starts_[static_cast<size_t>(phase)] = Clock::now();
        }

        static void end(WeightLoadPhase phase)
        {
            if (!isEnabled())
                return;
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            auto idx = static_cast<size_t>(phase);
            auto elapsed = Clock::now() - inst.starts_[idx];
            const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
            inst.durations_ms_[idx] += elapsed_ms;
            inst.call_counts_[idx]++;
            PerfStatsCollector::recordTimingNs(
                "weight_loading",
                weightLoadPhaseKey(phase),
                static_cast<uint64_t>(elapsed_ms * 1.0e6),
                "load");
        }

        static void beginDetail(const std::string &label)
        {
            if (!isEnabled())
                return;
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            inst.detail_starts_[label] = Clock::now();
        }

        static void endDetail(const std::string &label)
        {
            if (!isEnabled())
                return;
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            auto it = inst.detail_starts_.find(label);
            if (it == inst.detail_starts_.end())
                return;
            auto elapsed = Clock::now() - it->second;
            const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
            inst.detail_durations_ms_[label] += elapsed_ms;
            inst.detail_call_counts_[label] += 1;
            inst.detail_starts_.erase(it);
            PerfStatsCollector::recordTimingNs(
                "weight_loading",
                label,
                static_cast<uint64_t>(elapsed_ms * 1.0e6),
                "load",
                {},
                {{"detail", "1"}});
        }

        static void addDetail(const std::string &label, double ms)
        {
            if (!isEnabled())
                return;
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            inst.detail_durations_ms_[label] += ms;
            inst.detail_call_counts_[label] += 1;
            PerfStatsCollector::recordTimingNs(
                "weight_loading",
                label,
                static_cast<uint64_t>(ms * 1.0e6),
                "load",
                {},
                {{"detail", "1"}});
        }

        static void reset()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            inst.durations_ms_.fill(0.0);
            inst.call_counts_.fill(0);
            inst.detail_starts_.clear();
            inst.detail_durations_ms_.clear();
            inst.detail_call_counts_.clear();
        }

        /**
         * @brief Get total time across all phases (ms)
         */
        static double getTotalTimeMs()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            double total = 0.0;
            for (size_t i = 0; i < PHASE_COUNT; ++i)
            {
                total += inst.durations_ms_[i];
            }
            return total;
        }

        static bool hasData()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            for (size_t i = 0; i < PHASE_COUNT; ++i)
            {
                if (inst.call_counts_[i] > 0)
                    return true;
            }
            if (!inst.detail_call_counts_.empty())
                return true;
            return false;
        }

        static std::string getSummary()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            bool has_data = false;
            for (size_t i = 0; i < PHASE_COUNT; ++i)
            {
                if (inst.call_counts_[i] > 0)
                {
                    has_data = true;
                    break;
                }
            }
            if (!has_data && !inst.detail_call_counts_.empty())
            {
                has_data = true;
            }
            if (!has_data)
                return "";

            double total_ms = 0.0;
            for (size_t i = 0; i < PHASE_COUNT; ++i)
            {
                total_ms += inst.durations_ms_[i];
            }

            std::ostringstream oss;

            // Title
            fort::utf8_table title;
            title.set_border_style(FT_DOUBLE2_STYLE);
            title << "WEIGHT LOADING PROFILING" << fort::endr;
            title[0][0].set_cell_text_align(fort::text_align::center);
            title.row(0).set_cell_row_type(fort::row_type::header);
            oss << "\n"
                << title.to_string();

            // Data table
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            table << fort::header << "Phase" << "Time (ms)" << "%" << fort::endr;
            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);

            for (size_t i = 0; i < PHASE_COUNT; ++i)
            {
                if (inst.call_counts_[i] == 0)
                    continue;

                double ms = inst.durations_ms_[i];
                double pct = total_ms > 0 ? (ms / total_ms * 100.0) : 0.0;

                std::ostringstream ms_ss, pct_ss;
                ms_ss << std::fixed << std::setprecision(1) << ms;
                pct_ss << std::fixed << std::setprecision(1) << pct << "%";

                table << weightLoadPhaseName(static_cast<WeightLoadPhase>(i))
                      << ms_ss.str() << pct_ss.str() << fort::endr;
            }

            table << fort::separator;
            {
                std::ostringstream total_ss;
                total_ss << std::fixed << std::setprecision(1) << total_ms;
                table << "TOTAL" << total_ss.str() << "" << fort::endr;
            }

            oss << table.to_string();

            if (!inst.detail_durations_ms_.empty())
            {
                fort::utf8_table detail_title;
                detail_title.set_border_style(FT_DOUBLE2_STYLE);
                detail_title << "WEIGHT LOADING DETAIL" << fort::endr;
                detail_title[0][0].set_cell_text_align(fort::text_align::center);
                detail_title.row(0).set_cell_row_type(fort::row_type::header);
                oss << "\n"
                    << detail_title.to_string();

                std::vector<std::pair<std::string, double>> detail_rows;
                detail_rows.reserve(inst.detail_durations_ms_.size());
                for (const auto &entry : inst.detail_durations_ms_)
                {
                    detail_rows.emplace_back(entry.first, entry.second);
                }
                std::sort(detail_rows.begin(), detail_rows.end(), [](const auto &a, const auto &b)
                          { return a.second > b.second; });

                fort::utf8_table detail_table;
                detail_table.set_border_style(FT_DOUBLE2_STYLE);
                detail_table << fort::header << "Detail" << "Time (ms)" << "Calls" << fort::endr;
                detail_table.column(0).set_cell_text_align(fort::text_align::left);
                detail_table.column(1).set_cell_text_align(fort::text_align::right);
                detail_table.column(2).set_cell_text_align(fort::text_align::right);

                for (const auto &[label, ms] : detail_rows)
                {
                    std::ostringstream ms_ss;
                    ms_ss << std::fixed << std::setprecision(2) << ms;
                    const auto call_it = inst.detail_call_counts_.find(label);
                    const uint32_t calls = (call_it == inst.detail_call_counts_.end()) ? 0u : call_it->second;
                    detail_table << label << ms_ss.str() << std::to_string(calls) << fort::endr;
                }

                oss << detail_table.to_string();
            }
            return oss.str();
        }

        static void printSummary()
        {
            std::string summary = getSummary();
            if (!summary.empty())
            {
                fprintf(stderr, "%s", summary.c_str());
            }
        }

    private:
        WeightLoadingProfiler() = default;

        static WeightLoadingProfiler &getInstance()
        {
            static WeightLoadingProfiler instance;
            return instance;
        }

        std::array<TimePoint, PHASE_COUNT> starts_{};
        std::array<double, PHASE_COUNT> durations_ms_{};
        std::array<uint32_t, PHASE_COUNT> call_counts_{};
        std::unordered_map<std::string, TimePoint> detail_starts_{};
        std::unordered_map<std::string, double> detail_durations_ms_{};
        std::unordered_map<std::string, uint32_t> detail_call_counts_{};
        mutable std::mutex mutex_;
    };

    /**
     * @brief RAII scoped timer for weight loading phases
     */
    class ScopedWeightLoadTimer
    {
    public:
        explicit ScopedWeightLoadTimer(WeightLoadPhase phase)
            : phase_(phase), enabled_(WeightLoadingProfiler::isEnabled())
        {
            if (enabled_)
                WeightLoadingProfiler::begin(phase_);
        }

        ~ScopedWeightLoadTimer()
        {
            if (enabled_)
                WeightLoadingProfiler::end(phase_);
        }

        ScopedWeightLoadTimer(const ScopedWeightLoadTimer &) = delete;
        ScopedWeightLoadTimer &operator=(const ScopedWeightLoadTimer &) = delete;

    private:
        WeightLoadPhase phase_;
        bool enabled_;
    };

    class ScopedWeightLoadDetailTimer
    {
    public:
        explicit ScopedWeightLoadDetailTimer(std::string label)
            : label_(std::move(label)), enabled_(WeightLoadingProfiler::isEnabled())
        {
            if (enabled_)
                WeightLoadingProfiler::beginDetail(label_);
        }

        ~ScopedWeightLoadDetailTimer()
        {
            if (enabled_)
                WeightLoadingProfiler::endDetail(label_);
        }

        ScopedWeightLoadDetailTimer(const ScopedWeightLoadDetailTimer &) = delete;
        ScopedWeightLoadDetailTimer &operator=(const ScopedWeightLoadDetailTimer &) = delete;

    private:
        std::string label_;
        bool enabled_;
    };

} // namespace llaminar2
