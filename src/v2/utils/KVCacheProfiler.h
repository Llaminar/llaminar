/**
 * @file KVCacheProfiler.h
 * @brief Profiling for KV cache operations (append/gather)
 */
#pragma once

#include <array>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>

#include "DebugEnv.h"
#include "fort.hpp"

namespace llaminar2
{
    enum class KVCacheOpType : uint8_t
    {
        APPEND = 0,
        CONVERT_TO_FP32,
        CONVERT_TO_FP16,
        CONVERT_TO_Q8_1,
        CONVERT_TO_Q16_1,
        GATHER,
        GPU_ALLOC,
        COUNT
    };

    inline const char *kvCacheOpTypeName(KVCacheOpType type)
    {
        switch (type)
        {
        case KVCacheOpType::APPEND:
            return "KV_APPEND";
        case KVCacheOpType::CONVERT_TO_FP32:
            return "KV_CONV_FP32";
        case KVCacheOpType::CONVERT_TO_FP16:
            return "KV_CONV_FP16";
        case KVCacheOpType::CONVERT_TO_Q8_1:
            return "KV_CONV_Q8_1";
        case KVCacheOpType::CONVERT_TO_Q16_1:
            return "KV_CONV_Q16_1";
        case KVCacheOpType::GATHER:
            return "KV_GATHER";
        case KVCacheOpType::GPU_ALLOC:
            return "KV_GPU_ALLOC";
        default:
            return "UNKNOWN";
        }
    }

    class KVCacheProfiler
    {
    public:
        enum class Phase : uint8_t
        {
            UNKNOWN = 0,
            PREFILL,
            DECODE
        };

        struct OpStats
        {
            uint64_t total_ns = 0;
            uint64_t call_count = 0;
            uint64_t total_tokens = 0;
            uint64_t total_bytes = 0;
            uint64_t max_ns = 0;
            uint64_t min_ns = std::numeric_limits<uint64_t>::max();

            void reset()
            {
                total_ns = 0;
                call_count = 0;
                total_tokens = 0;
                total_bytes = 0;
                max_ns = 0;
                min_ns = std::numeric_limits<uint64_t>::max();
            }

            void add(uint64_t duration_ns, uint64_t tokens, uint64_t bytes)
            {
                total_ns += duration_ns;
                call_count += 1;
                total_tokens += tokens;
                total_bytes += bytes;
                if (duration_ns > max_ns)
                {
                    max_ns = duration_ns;
                }
                if (duration_ns < min_ns)
                {
                    min_ns = duration_ns;
                }
            }
        };

        static bool isEnabled()
        {
            return debugEnv().profile.enabled;
        }

        static void setCurrentPhase(Phase phase)
        {
            current_phase() = phase;
        }

        static void clearCurrentPhase()
        {
            current_phase() = Phase::UNKNOWN;
        }

        static void record(KVCacheOpType type, uint64_t duration_ns, uint64_t tokens = 0, uint64_t bytes = 0)
        {
            if (!isEnabled())
            {
                return;
            }

            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            const size_t idx = static_cast<size_t>(type);
            inst.stats_[idx].add(duration_ns, tokens, bytes);

            const Phase phase = current_phase();
            if (phase == Phase::PREFILL)
            {
                inst.prefill_stats_[idx].add(duration_ns, tokens, bytes);
            }
            else if (phase == Phase::DECODE)
            {
                inst.decode_stats_[idx].add(duration_ns, tokens, bytes);
            }
        }

        static void reset()
        {
            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            for (auto &stats : inst.stats_)
            {
                stats.reset();
            }
            for (auto &stats : inst.prefill_stats_)
            {
                stats.reset();
            }
            for (auto &stats : inst.decode_stats_)
            {
                stats.reset();
            }
        }

        static std::string getSummary()
        {
            if (!isEnabled())
            {
                return "";
            }

            auto &inst = getInstance();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            uint64_t total_calls = 0;
            uint64_t total_ns = 0;
            for (size_t i = 0; i < static_cast<size_t>(KVCacheOpType::COUNT); ++i)
            {
                total_calls += inst.stats_[i].call_count;
                total_ns += inst.stats_[i].total_ns;
            }
            if (total_calls == 0)
            {
                return "";
            }

            auto formatBytes = [](uint64_t bytes) -> std::string
            {
                std::ostringstream s;
                s << std::fixed << std::setprecision(2);
                if (bytes >= 1024ULL * 1024ULL * 1024ULL)
                    s << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB";
                else if (bytes >= 1024ULL * 1024ULL)
                    s << (bytes / (1024.0 * 1024.0)) << " MB";
                else if (bytes >= 1024ULL)
                    s << (bytes / 1024.0) << " KB";
                else
                    s << bytes << " B";
                return s.str();
            };

            std::ostringstream oss;

            {
                fort::utf8_table title;
                title.set_border_style(FT_DOUBLE2_STYLE);
                title << "KV CACHE PROFILING SUMMARY" << fort::endr;
                title[0][0].set_cell_text_align(fort::text_align::center);
                title.row(0).set_cell_row_type(fort::row_type::header);
                oss << "\n"
                    << title.to_string();
            }

            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);
                table << fort::header << "Operation" << "Calls" << "Total (ms)" << "Avg (µs)"
                      << "Tokens" << "Bytes" << "BW" << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::right);
                table.column(2).set_cell_text_align(fort::text_align::right);
                table.column(3).set_cell_text_align(fort::text_align::right);
                table.column(4).set_cell_text_align(fort::text_align::right);
                table.column(5).set_cell_text_align(fort::text_align::right);
                table.column(6).set_cell_text_align(fort::text_align::right);

                for (size_t i = 0; i < static_cast<size_t>(KVCacheOpType::COUNT); ++i)
                {
                    const auto &s = inst.stats_[i];
                    if (s.call_count == 0)
                    {
                        continue;
                    }

                    const double total_ms = static_cast<double>(s.total_ns) / 1e6;
                    const double avg_us = (s.call_count > 0) ? (static_cast<double>(s.total_ns) / s.call_count) / 1e3 : 0.0;
                    const double bw_gbps = (s.total_ns > 0) ? (static_cast<double>(s.total_bytes) / 1e9) / (static_cast<double>(s.total_ns) / 1e9) : 0.0;

                    std::ostringstream total_ss, avg_ss, bw_ss;
                    total_ss << std::fixed << std::setprecision(2) << total_ms;
                    avg_ss << std::fixed << std::setprecision(1) << avg_us;
                    bw_ss << std::fixed << std::setprecision(2) << bw_gbps << " GB/s";

                    table << kvCacheOpTypeName(static_cast<KVCacheOpType>(i))
                          << s.call_count
                          << total_ss.str()
                          << avg_ss.str()
                          << s.total_tokens
                          << formatBytes(s.total_bytes)
                          << bw_ss.str()
                          << fort::endr;
                }

                table << fort::separator;
                std::ostringstream total_ms_ss;
                total_ms_ss << std::fixed << std::setprecision(2) << (static_cast<double>(total_ns) / 1e6) << " ms";
                table << "TOTAL" << total_calls << total_ms_ss.str() << "" << "" << "" << "" << fort::endr;

                oss << table.to_string();
            }

            {
                uint64_t prefill_total_ns = 0;
                uint64_t decode_total_ns = 0;
                for (size_t i = 0; i < static_cast<size_t>(KVCacheOpType::COUNT); ++i)
                {
                    prefill_total_ns += inst.prefill_stats_[i].total_ns;
                    decode_total_ns += inst.decode_stats_[i].total_ns;
                }

                if (prefill_total_ns > 0 || decode_total_ns > 0)
                {
                    oss << "\nKV CACHE PHASE BREAKDOWN (Prefill vs Decode):\n";

                    fort::utf8_table phase_table;
                    phase_table.set_border_style(FT_DOUBLE2_STYLE);
                    phase_table << fort::header << "Operation" << "Prefill (ms)" << "Decode (ms)" << "Decode %" << fort::endr;
                    phase_table.column(0).set_cell_text_align(fort::text_align::left);
                    phase_table.column(1).set_cell_text_align(fort::text_align::right);
                    phase_table.column(2).set_cell_text_align(fort::text_align::right);
                    phase_table.column(3).set_cell_text_align(fort::text_align::right);

                    for (size_t i = 0; i < static_cast<size_t>(KVCacheOpType::COUNT); ++i)
                    {
                        const double prefill_ms = static_cast<double>(inst.prefill_stats_[i].total_ns) / 1e6;
                        const double decode_ms = static_cast<double>(inst.decode_stats_[i].total_ns) / 1e6;
                        const double decode_pct = (total_ns > 0) ? (decode_ms / (static_cast<double>(total_ns) / 1e6)) * 100.0 : 0.0;

                        std::ostringstream prefill_ss, decode_ss, pct_ss;
                        prefill_ss << std::fixed << std::setprecision(2) << prefill_ms;
                        decode_ss << std::fixed << std::setprecision(2) << decode_ms;
                        pct_ss << std::fixed << std::setprecision(0) << decode_pct << "%";

                        phase_table << kvCacheOpTypeName(static_cast<KVCacheOpType>(i))
                                    << prefill_ss.str()
                                    << decode_ss.str()
                                    << pct_ss.str()
                                    << fort::endr;
                    }

                    phase_table << fort::separator;
                    std::ostringstream p_total_ss, d_total_ss;
                    p_total_ss << std::fixed << std::setprecision(2) << (static_cast<double>(prefill_total_ns) / 1e6);
                    d_total_ss << std::fixed << std::setprecision(2) << (static_cast<double>(decode_total_ns) / 1e6);
                    phase_table << "PHASE TOTAL" << p_total_ss.str() << d_total_ss.str() << "" << fort::endr;

                    oss << phase_table.to_string();
                }
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
        KVCacheProfiler() = default;

        static KVCacheProfiler &getInstance()
        {
            static KVCacheProfiler instance;
            return instance;
        }

        static Phase &current_phase()
        {
            thread_local Phase phase = Phase::UNKNOWN;
            return phase;
        }

        std::mutex mutex_;
        std::array<OpStats, static_cast<size_t>(KVCacheOpType::COUNT)> stats_{};
        std::array<OpStats, static_cast<size_t>(KVCacheOpType::COUNT)> prefill_stats_{};
        std::array<OpStats, static_cast<size_t>(KVCacheOpType::COUNT)> decode_stats_{};
    };
} // namespace llaminar2
