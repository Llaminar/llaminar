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
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include "DebugEnv.h"
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
            return debugEnv().profile.enabled;
        }

        static void begin(WeightLoadPhase phase)
        {
            if (!isEnabled())
                return;
            auto &inst = getInstance();
            inst.starts_[static_cast<size_t>(phase)] = Clock::now();
        }

        static void end(WeightLoadPhase phase)
        {
            if (!isEnabled())
                return;
            auto &inst = getInstance();
            auto idx = static_cast<size_t>(phase);
            auto elapsed = Clock::now() - inst.starts_[idx];
            inst.durations_ms_[idx] += std::chrono::duration<double, std::milli>(elapsed).count();
            inst.call_counts_[idx]++;
        }

        static void reset()
        {
            auto &inst = getInstance();
            inst.durations_ms_.fill(0.0);
            inst.call_counts_.fill(0);
        }

        /**
         * @brief Get total time across all phases (ms)
         */
        static double getTotalTimeMs()
        {
            auto &inst = getInstance();
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
            for (size_t i = 0; i < PHASE_COUNT; ++i)
            {
                if (inst.call_counts_[i] > 0)
                    return true;
            }
            return false;
        }

        static std::string getSummary()
        {
            auto &inst = getInstance();

            if (!hasData())
                return "";

            double total_ms = getTotalTimeMs();

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

} // namespace llaminar2
