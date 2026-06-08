/**
 * @file PerfStatsCollector.cpp
 * @brief Unified structured performance counter and timer collection.
 */

#include "PerfStatsCollector.h"

#include "DebugEnv.h"
#include "Logger.h"
#include "ProfilingContext.h"
#include "fort.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <system_error>
#include <tuple>

namespace llaminar2
{
    namespace
    {
        struct PerfStatKey
        {
            PerfStatRecord::Kind kind = PerfStatRecord::Kind::Counter;
            std::string domain;
            std::string name;
            std::string phase;
            std::string device;
            PerfStatsCollector::Tags tags;

            bool operator<(const PerfStatKey &other) const
            {
                return std::tie(kind, domain, name, phase, device, tags) <
                       std::tie(other.kind, other.domain, other.name, other.phase, other.device, other.tags);
            }
        };

        struct PerfStatAccumulator
        {
            PerfStatRecord::Kind kind = PerfStatRecord::Kind::Counter;
            uint64_t count = 0;
            double value = 0.0;
            uint64_t total_ns = 0;
            uint64_t min_ns = std::numeric_limits<uint64_t>::max();
            uint64_t max_ns = 0;
        };

        struct PerfStatsState
        {
            std::mutex mutex;
            std::map<PerfStatKey, PerfStatAccumulator> records;
            size_t version = 0;
            size_t json_version = 0;
            size_t csv_version = 0;
            size_t summary_version = 0;
        };

        PerfStatsState &state()
        {
            static PerfStatsState instance;
            return instance;
        }

        std::string trim(std::string value)
        {
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch)
                                                    { return !std::isspace(ch); }));
            value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch)
                                     { return !std::isspace(ch); })
                            .base(),
                        value.end());
            return value;
        }

        std::string normalizedEnvValue(const char *value)
        {
            if (!value)
                return {};
            std::string normalized = trim(value);
            std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch)
                           { return static_cast<char>(std::tolower(ch)); });
            return normalized;
        }

        bool isTruthyEnvValue(const char *value)
        {
            const std::string normalized = normalizedEnvValue(value);
            if (!normalized.empty() &&
                std::all_of(normalized.begin(), normalized.end(), [](unsigned char ch)
                            { return std::isdigit(ch) != 0; }))
            {
                return std::atoi(normalized.c_str()) != 0;
            }
            return normalized == "1" || normalized == "true" ||
                   normalized == "on" || normalized == "yes";
        }

        bool isFalseyEnvValue(const char *value)
        {
            const std::string normalized = normalizedEnvValue(value);
            return normalized.empty() || normalized == "0" || normalized == "false" ||
                   normalized == "off" || normalized == "no";
        }

        bool isSummaryRequested()
        {
            return !isFalseyEnvValue(DebugEnv::envValue("LLAMINAR_PERF_STATS_TABLE")) ||
                   !isFalseyEnvValue(DebugEnv::envValue("LLAMINAR_PERF_STATS_SUMMARY"));
        }

        size_t summaryLimitFromEnv()
        {
            const char *env = DebugEnv::envValue("LLAMINAR_PERF_STATS_TABLE_LIMIT");
            if (!env)
                return 120;
            const std::string normalized = normalizedEnvValue(env);
            if (normalized.empty() ||
                !std::all_of(normalized.begin(), normalized.end(), [](unsigned char ch)
                             { return std::isdigit(ch) != 0; }))
            {
                return 120;
            }
            const size_t parsed = static_cast<size_t>(std::strtoull(normalized.c_str(), nullptr, 10));
            return parsed == 0 ? 120 : parsed;
        }

        std::string exportPathFromEnv(const char *env_name, const char *default_path)
        {
            const char *value = DebugEnv::envValue(env_name);
            if (!value || isFalseyEnvValue(value))
                return {};
            if (isTruthyEnvValue(value))
                return default_path;
            return value;
        }

        bool isExportRequested()
        {
            return exportPathFromEnv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_perf_stats.json").size() > 0 ||
                   exportPathFromEnv("LLAMINAR_PERF_STATS_CSV", "/tmp/llaminar_perf_stats.csv").size() > 0 ||
                   isSummaryRequested();
        }

        std::vector<std::string> filterListFromEnv()
        {
            std::vector<std::string> filters;
            const char *env = DebugEnv::envValue("LLAMINAR_PERF_STATS_FILTER");
            if (!env)
                return filters;

            std::stringstream stream(env);
            std::string item;
            while (std::getline(stream, item, ','))
            {
                item = trim(item);
                if (!item.empty())
                    filters.push_back(std::move(item));
            }
            return filters;
        }

        bool filterRequestsStageGpuTiming()
        {
            if (!isExportRequested())
                return false;

            const auto filters = filterListFromEnv();
            return std::any_of(filters.begin(), filters.end(), [](const std::string &filter)
                               {
                                   return filter == "stage_gpu" ||
                                          filter == "stage_gpu.*" ||
                                          filter.starts_with("stage_gpu.") ||
                                          filter == "mtp_stage_gpu" ||
                                          filter == "mtp_stage_gpu.*" ||
                                          filter.starts_with("mtp_stage_gpu.");
                               });
        }

        bool perfStatsGpuStageTimingRequested()
        {
            return isTruthyEnvValue(DebugEnv::envValue("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING")) ||
                   filterRequestsStageGpuTiming();
        }

        bool recordMatchesFilters(const PerfStatRecord &record, const std::vector<std::string> &filters)
        {
            if (filters.empty())
                return true;
            const std::string qualified = record.domain + "." + record.name;
            for (const auto &filter : filters)
            {
                if (filter == "*" || filter == "all")
                    return true;
                if (record.domain == filter || qualified == filter)
                    return true;
                if (qualified.starts_with(filter + "."))
                    return true;
            }
            return false;
        }

        const char *kindName(PerfStatRecord::Kind kind)
        {
            switch (kind)
            {
            case PerfStatRecord::Kind::Counter:
                return "counter";
            case PerfStatRecord::Kind::Timer:
                return "timer";
            }
            return "unknown";
        }

        std::string jsonEscape(const std::string &value)
        {
            std::ostringstream out;
            for (char ch : value)
            {
                switch (ch)
                {
                case '\\':
                    out << "\\\\";
                    break;
                case '"':
                    out << "\\\"";
                    break;
                case '\n':
                    out << "\\n";
                    break;
                case '\r':
                    out << "\\r";
                    break;
                case '\t':
                    out << "\\t";
                    break;
                default:
                    out << ch;
                    break;
                }
            }
            return out.str();
        }

        std::string csvEscape(const std::string &value)
        {
            if (value.find_first_of(",\"\n\r") == std::string::npos)
                return value;
            std::string escaped = "\"";
            for (char ch : value)
            {
                if (ch == '"')
                    escaped += "\"\"";
                else
                    escaped += ch;
            }
            escaped += "\"";
            return escaped;
        }

        std::string tagsToCsv(const PerfStatsCollector::Tags &tags)
        {
            std::ostringstream out;
            bool first = true;
            for (const auto &[key, value] : tags)
            {
                if (!first)
                    out << ';';
                out << key << '=' << value;
                first = false;
            }
            return out.str();
        }

        std::string tagsToDisplay(const PerfStatsCollector::Tags &tags)
        {
            const std::string value = tagsToCsv(tags);
            if (value.size() <= 48)
                return value;
            return value.substr(0, 45) + "...";
        }

        bool writeTextFile(const std::string &path, const std::string &contents)
        {
            if (path.empty())
                return false;

            std::error_code ec;
            const std::filesystem::path fs_path(path);
            if (fs_path.has_parent_path())
                std::filesystem::create_directories(fs_path.parent_path(), ec);

            std::ofstream out(path, std::ios::out | std::ios::trunc);
            if (!out.is_open())
            {
                LOG_WARN("[PerfStatsCollector] Failed to open output path '" << path << "'");
                return false;
            }
            out << contents;
            return true;
        }
    } // namespace

    bool PerfStatsCollector::isEnabled()
    {
        return debugEnv().profile.enabled ||
               debugEnv().gpu_stage_timing ||
               perfStatsGpuStageTimingRequested() ||
               isSummaryRequested() ||
               exportPathFromEnv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_perf_stats.json").size() > 0 ||
               exportPathFromEnv("LLAMINAR_PERF_STATS_CSV", "/tmp/llaminar_perf_stats.csv").size() > 0;
    }

    bool PerfStatsCollector::gpuStageEventTimingEnabled()
    {
        return debugEnv().gpu_stage_timing ||
               debugEnv().profile.enabled ||
               perfStatsGpuStageTimingRequested();
    }

    void PerfStatsCollector::reset()
    {
        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.records.clear();
        ++s.version;
        s.json_version = 0;
        s.csv_version = 0;
        s.summary_version = 0;
    }

    void PerfStatsCollector::addCounter(
        std::string domain,
        std::string name,
        double value,
        std::string phase,
        std::string device,
        Tags tags)
    {
        if (!isEnabled())
            return;

        PerfStatKey key;
        key.kind = PerfStatRecord::Kind::Counter;
        key.domain = std::move(domain);
        key.name = std::move(name);
        key.phase = std::move(phase);
        key.device = std::move(device);
        key.tags = std::move(tags);

        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        auto &record = s.records[key];
        record.kind = PerfStatRecord::Kind::Counter;
        record.count += 1;
        record.value += value;
        ++s.version;
    }

    void PerfStatsCollector::recordTimingNs(
        std::string domain,
        std::string name,
        uint64_t duration_ns,
        std::string phase,
        std::string device,
        Tags tags)
    {
        if (!isEnabled())
            return;

        PerfStatKey key;
        key.kind = PerfStatRecord::Kind::Timer;
        key.domain = std::move(domain);
        key.name = std::move(name);
        key.phase = std::move(phase);
        key.device = std::move(device);
        key.tags = std::move(tags);

        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        auto &record = s.records[key];
        record.kind = PerfStatRecord::Kind::Timer;
        record.count += 1;
        record.total_ns += duration_ns;
        record.min_ns = std::min(record.min_ns, duration_ns);
        record.max_ns = std::max(record.max_ns, duration_ns);
        ++s.version;
    }

    std::vector<PerfStatRecord> PerfStatsCollector::snapshot(
        const std::vector<std::string> &filters)
    {
        std::vector<PerfStatRecord> result;
        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        result.reserve(s.records.size());
        for (const auto &[key, acc] : s.records)
        {
            PerfStatRecord record;
            record.kind = key.kind;
            record.domain = key.domain;
            record.name = key.name;
            record.phase = key.phase;
            record.device = key.device;
            record.tags = key.tags;
            record.count = acc.count;
            record.value = acc.value;
            record.total_ns = acc.total_ns;
            record.min_ns = acc.min_ns == std::numeric_limits<uint64_t>::max() ? 0 : acc.min_ns;
            record.max_ns = acc.max_ns;
            if (recordMatchesFilters(record, filters))
                result.push_back(std::move(record));
        }
        return result;
    }

    std::string PerfStatsCollector::jsonString(const std::vector<std::string> &filters)
    {
        const auto records = snapshot(filters);
        std::ostringstream out;
        out << "{\n";
        out << "  \"schema\": \"llaminar.perf_stats.v1\",\n";
        out << "  \"records\": [\n";
        for (size_t i = 0; i < records.size(); ++i)
        {
            const auto &record = records[i];
            const double total_ms = static_cast<double>(record.total_ns) / 1.0e6;
            const double avg_us = record.count > 0
                                      ? (static_cast<double>(record.total_ns) / static_cast<double>(record.count)) / 1.0e3
                                      : 0.0;
            const double min_us = static_cast<double>(record.min_ns) / 1.0e3;
            const double max_us = static_cast<double>(record.max_ns) / 1.0e3;

            out << "    {\n";
            out << "      \"kind\": \"" << kindName(record.kind) << "\",\n";
            out << "      \"domain\": \"" << jsonEscape(record.domain) << "\",\n";
            out << "      \"name\": \"" << jsonEscape(record.name) << "\",\n";
            out << "      \"phase\": \"" << jsonEscape(record.phase) << "\",\n";
            out << "      \"device\": \"" << jsonEscape(record.device) << "\",\n";
            out << "      \"tags\": {";
            bool first_tag = true;
            for (const auto &[key, value] : record.tags)
            {
                if (!first_tag)
                    out << ", ";
                out << "\"" << jsonEscape(key) << "\": \"" << jsonEscape(value) << "\"";
                first_tag = false;
            }
            out << "},\n";
            out << "      \"count\": " << record.count << ",\n";
            out << "      \"value\": " << std::setprecision(17) << record.value << ",\n";
            out << "      \"total_ns\": " << record.total_ns << ",\n";
            out << "      \"total_ms\": " << std::setprecision(17) << total_ms << ",\n";
            out << "      \"avg_us\": " << std::setprecision(17) << avg_us << ",\n";
            out << "      \"min_us\": " << std::setprecision(17) << min_us << ",\n";
            out << "      \"max_us\": " << std::setprecision(17) << max_us << "\n";
            out << "    }" << (i + 1 == records.size() ? "\n" : ",\n");
        }
        out << "  ]\n";
        out << "}\n";
        return out.str();
    }

    std::string PerfStatsCollector::csvString(const std::vector<std::string> &filters)
    {
        const auto records = snapshot(filters);
        std::ostringstream out;
        out << "kind,domain,name,phase,device,tags,count,value,total_ns,total_ms,avg_us,min_us,max_us\n";
        for (const auto &record : records)
        {
            const double total_ms = static_cast<double>(record.total_ns) / 1.0e6;
            const double avg_us = record.count > 0
                                      ? (static_cast<double>(record.total_ns) / static_cast<double>(record.count)) / 1.0e3
                                      : 0.0;
            const double min_us = static_cast<double>(record.min_ns) / 1.0e3;
            const double max_us = static_cast<double>(record.max_ns) / 1.0e3;

            out << kindName(record.kind) << ','
                << csvEscape(record.domain) << ','
                << csvEscape(record.name) << ','
                << csvEscape(record.phase) << ','
                << csvEscape(record.device) << ','
                << csvEscape(tagsToCsv(record.tags)) << ','
                << record.count << ','
                << std::setprecision(17) << record.value << ','
                << record.total_ns << ','
                << std::setprecision(17) << total_ms << ','
                << std::setprecision(17) << avg_us << ','
                << std::setprecision(17) << min_us << ','
                << std::setprecision(17) << max_us << '\n';
        }
        return out.str();
    }

    std::string PerfStatsCollector::summaryString(
        const std::vector<std::string> &filters,
        size_t max_records)
    {
        auto records = snapshot(filters);
        if (records.empty())
            return {};

        std::sort(records.begin(), records.end(), [](const auto &a, const auto &b)
                  {
                      const uint64_t a_weight =
                          a.kind == PerfStatRecord::Kind::Timer ? a.total_ns : static_cast<uint64_t>(a.value);
                      const uint64_t b_weight =
                          b.kind == PerfStatRecord::Kind::Timer ? b.total_ns : static_cast<uint64_t>(b.value);
                      if (a_weight != b_weight)
                          return a_weight > b_weight;
                      return std::tie(a.domain, a.name, a.phase, a.device, a.tags) <
                             std::tie(b.domain, b.name, b.phase, b.device, b.tags);
                  });

        if (max_records == 0)
            max_records = records.size();

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        {
            std::ostringstream title;
            title << "UNIFIED PERF STATS";
            if (!filters.empty())
            {
                title << " [";
                for (size_t i = 0; i < filters.size(); ++i)
                {
                    if (i > 0)
                        title << ",";
                    title << filters[i];
                }
                title << "]";
            }
            table << title.str() << "" << "" << "" << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(8);
            table[0][0].set_cell_text_align(fort::text_align::center);
        }

        table << fort::header
              << "Kind" << "Metric" << "Phase" << "Device" << "Count"
              << "Value/Total" << "Avg" << "Tags" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(2).set_cell_text_align(fort::text_align::left);
        table.column(3).set_cell_text_align(fort::text_align::left);
        table.column(4).set_cell_text_align(fort::text_align::right);
        table.column(5).set_cell_text_align(fort::text_align::right);
        table.column(6).set_cell_text_align(fort::text_align::right);
        table.column(7).set_cell_text_align(fort::text_align::left);

        auto fmt_ms = [](double ms)
        {
            std::ostringstream out;
            out << std::fixed << std::setprecision(3) << ms << " ms";
            return out.str();
        };
        auto fmt_us = [](double us)
        {
            std::ostringstream out;
            out << std::fixed << std::setprecision(2) << us << " us";
            return out.str();
        };
        auto fmt_value = [](double value)
        {
            std::ostringstream out;
            out << std::setprecision(10) << value;
            return out.str();
        };

        const size_t display_count = std::min(records.size(), max_records);
        for (size_t i = 0; i < display_count; ++i)
        {
            const auto &record = records[i];
            const std::string metric = record.domain + "." + record.name;
            if (record.kind == PerfStatRecord::Kind::Timer)
            {
                const double total_ms = static_cast<double>(record.total_ns) / 1.0e6;
                const double avg_us = record.count > 0
                                          ? (static_cast<double>(record.total_ns) / static_cast<double>(record.count)) / 1.0e3
                                          : 0.0;
                table << "timer"
                      << metric
                      << record.phase
                      << record.device
                      << std::to_string(record.count)
                      << fmt_ms(total_ms)
                      << fmt_us(avg_us)
                      << tagsToDisplay(record.tags)
                      << fort::endr;
            }
            else
            {
                const double avg = record.count > 0
                                       ? record.value / static_cast<double>(record.count)
                                       : 0.0;
                table << "counter"
                      << metric
                      << record.phase
                      << record.device
                      << std::to_string(record.count)
                      << fmt_value(record.value)
                      << fmt_value(avg)
                      << tagsToDisplay(record.tags)
                      << fort::endr;
            }
        }

        if (records.size() > display_count)
        {
            table << fort::separator;
            table << "..."
                  << ("and " + std::to_string(records.size() - display_count) + " more records")
                  << "" << "" << "" << "" << "" << "" << fort::endr;
        }

        std::ostringstream out;
        out << table.to_string();
        return out.str();
    }

    bool PerfStatsCollector::writeJson(
        const std::string &path,
        const std::vector<std::string> &filters)
    {
        return writeTextFile(path, jsonString(filters));
    }

    bool PerfStatsCollector::writeCsv(
        const std::string &path,
        const std::vector<std::string> &filters)
    {
        return writeTextFile(path, csvString(filters));
    }

    void PerfStatsCollector::printSummary(
        const std::vector<std::string> &filters,
        size_t max_records)
    {
        const std::string summary = summaryString(filters, max_records);
        if (!summary.empty())
            std::cout << summary << std::flush;
    }

    bool PerfStatsCollector::flushFromEnv()
    {
        if (Logger::getInstance().getRank() > 0)
            return true;

        const std::string json_path =
            exportPathFromEnv("LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_perf_stats.json");
        const std::string csv_path =
            exportPathFromEnv("LLAMINAR_PERF_STATS_CSV", "/tmp/llaminar_perf_stats.csv");
        if (json_path.empty() && csv_path.empty())
            return true;

        const auto filters = filterListFromEnv();
        auto &s = state();
        size_t version = 0;
        size_t json_version = 0;
        size_t csv_version = 0;
        size_t summary_version = 0;
        bool empty = true;
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            version = s.version;
            json_version = s.json_version;
            csv_version = s.csv_version;
            summary_version = s.summary_version;
            empty = s.records.empty();
        }
        if (empty)
            return true;

        bool ok = true;
        if (!json_path.empty() && json_version != version)
        {
            ok = writeJson(json_path, filters) && ok;
            if (ok)
            {
                std::lock_guard<std::mutex> lock(s.mutex);
                s.json_version = s.version;
            }
        }
        if (!csv_path.empty() && csv_version != version)
        {
            ok = writeCsv(csv_path, filters) && ok;
            if (ok)
            {
                std::lock_guard<std::mutex> lock(s.mutex);
                s.csv_version = s.version;
            }
        }
        if (isSummaryRequested() && summary_version != version)
        {
            printSummary(filters, summaryLimitFromEnv());
            std::lock_guard<std::mutex> lock(s.mutex);
            s.summary_version = s.version;
        }
        return ok;
    }

    PerfStatsCollector::ScopedTimer::ScopedTimer(
        std::string domain,
        std::string name,
        std::string phase,
        std::string device,
        Tags tags)
        : enabled_(PerfStatsCollector::isEnabled()),
          domain_(std::move(domain)),
          name_(std::move(name)),
          phase_(std::move(phase)),
          device_(std::move(device)),
          tags_(std::move(tags))
    {
        if (device_.empty() && ProfilingContext::hasDeviceContext())
            device_ = ProfilingContext::getCurrentDeviceKey();
        if (enabled_)
            start_ = Clock::now();
    }

    PerfStatsCollector::ScopedTimer::~ScopedTimer()
    {
        if (!enabled_)
            return;
        const auto end = Clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        PerfStatsCollector::recordTimingNs(
            std::move(domain_),
            std::move(name_),
            ns > 0 ? static_cast<uint64_t>(ns) : 0,
            std::move(phase_),
            std::move(device_),
            std::move(tags_));
    }
} // namespace llaminar2
