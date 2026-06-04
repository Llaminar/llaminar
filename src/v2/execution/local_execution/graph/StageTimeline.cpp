/**
 * @file StageTimeline.cpp
 * @brief Summary table rendering for GPU event-based stage timeline
 */

#include "StageTimeline.h"
#include "../../../utils/PerfStatsCollector.h"
#include "fort.hpp"
#include <iostream>

namespace llaminar2
{

    void StageTimeline::printSummary(const char *phase_name, size_t token_count, double wall_ms,
                                     const char *device_name) const
    {
        auto agg = aggregateByType();
        if (agg.empty())
            return;

        float total_gpu_ms = totalGpuMs();

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Title row
        {
            std::ostringstream title;
            title << "GPU STAGE EVENTS: " << phase_name;
            if (device_name)
                title << " [" << device_name << "]";
            if (token_count > 0)
                title << " (" << token_count << " tokens)";
            table << title.str() << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(5);
            table[0][0].set_cell_text_align(fort::text_align::center);
        }

        // GPU total + wall comparison
        {
            std::ostringstream info;
            info << std::fixed << std::setprecision(2);
            info << "Event Total: " << total_gpu_ms << " ms";
            if (wall_ms > 0.0)
            {
                info << "  |  Wall: " << wall_ms << " ms";
                double gap = wall_ms - total_gpu_ms;
                info << "  |  Non-event wall: " << gap << " ms";
                if (wall_ms > 0.0)
                    info << " (" << std::setprecision(1) << (gap / wall_ms * 100.0) << "%)";
            }
            if (token_count > 0 && total_gpu_ms > 0.0f)
            {
                double tok_per_sec = token_count / (total_gpu_ms / 1000.0);
                info << "  |  Event-limited: " << std::setprecision(1) << tok_per_sec << " tok/s";
            }
            table << info.str() << "" << "" << "" << "" << fort::endr;
            table[1][0].set_cell_span(5);
        }

        // Header
        table << fort::header << "STAGE TYPE" << "COUNT" << "TOTAL (ms)" << "AVG (ms)" << "%" << fort::endr;

        // Column alignments
        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::right);
        table.column(4).set_cell_text_align(fort::text_align::right);

        // Data rows (already sorted by total_ms descending)
        auto fmt_ms = [](float ms) -> std::string
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << ms;
            return oss.str();
        };

        for (const auto &entry : agg)
        {
            float pct = total_gpu_ms > 0.0f ? (entry.total_ms / total_gpu_ms) * 100.0f : 0.0f;
            std::ostringstream pct_str;
            pct_str << std::fixed << std::setprecision(1) << pct << "%";

            table << entry.type_name
                  << std::to_string(entry.count)
                  << fmt_ms(entry.total_ms)
                  << fmt_ms(entry.avg_ms())
                  << pct_str.str()
                  << fort::endr;
        }

        // Separator + total
        table << fort::separator;
        {
            size_t total_count = 0;
            for (const auto &e : agg)
                total_count += e.count;
            table << "TOTAL"
                  << std::to_string(total_count)
                  << fmt_ms(total_gpu_ms)
                  << ""
                  << "100.0%"
                  << fort::endr;
        }

        std::cout << table.to_string() << std::flush;
    }

    bool StageTimeline::printAccumulatedSummary(const char *phase_name,
                                                const char *device_name)
    {
        if (accumulated_iterations_ == 0 || accumulated_.empty())
            return false;

        // Build sorted vector from accumulated map
        std::vector<TypeAggregate> agg;
        agg.reserve(accumulated_.size());
        for (const auto &[_, val] : accumulated_)
            agg.push_back(val);
        std::sort(agg.begin(), agg.end(),
                  [](const TypeAggregate &a, const TypeAggregate &b)
                  { return a.total_ms > b.total_ms; });

        float total_gpu_ms = 0.0f;
        for (const auto &e : agg)
            total_gpu_ms += e.total_ms;

        float avg_gpu_ms = total_gpu_ms / accumulated_iterations_;
        double avg_wall_ms = accumulated_wall_ms_ / accumulated_iterations_;

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Title row
        {
            std::ostringstream title;
            title << "GPU STAGE EVENTS: " << phase_name;
            if (device_name)
                title << " [" << device_name << "]";
            title << " (avg of " << accumulated_iterations_ << " iterations)";
            table << title.str() << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(5);
            table[0][0].set_cell_text_align(fort::text_align::center);
        }

        // Summary info
        {
            std::ostringstream info;
            info << std::fixed << std::setprecision(2);
            info << "Event Avg: " << avg_gpu_ms << " ms";
            info << "  |  Wall Avg: " << avg_wall_ms << " ms";
            double gap = avg_wall_ms - avg_gpu_ms;
            info << "  |  Non-event wall: " << gap << " ms";
            if (avg_wall_ms > 0.0)
                info << " (" << std::setprecision(1) << (gap / avg_wall_ms * 100.0) << "%)";
            double tok_per_sec = 1000.0 / avg_gpu_ms;
            info << "  |  Event-limited: " << std::setprecision(1) << tok_per_sec << " tok/s";
            table << info.str() << "" << "" << "" << "" << fort::endr;
            table[1][0].set_cell_span(5);
        }

        // Header
        table << fort::header << "STAGE TYPE" << "COUNT/iter" << "TOTAL (ms)" << "AVG/iter (ms)" << "%" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::right);
        table.column(4).set_cell_text_align(fort::text_align::right);

        auto fmt_ms = [](float ms) -> std::string
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << ms;
            return oss.str();
        };

        for (const auto &entry : agg)
        {
            float pct = total_gpu_ms > 0.0f ? (entry.total_ms / total_gpu_ms) * 100.0f : 0.0f;
            std::ostringstream pct_str;
            pct_str << std::fixed << std::setprecision(1) << pct << "%";

            size_t count_per_iter = entry.count / accumulated_iterations_;
            float ms_per_iter = entry.total_ms / accumulated_iterations_;

            table << entry.type_name
                  << std::to_string(count_per_iter)
                  << fmt_ms(entry.total_ms)
                  << fmt_ms(ms_per_iter)
                  << pct_str.str()
                  << fort::endr;
        }

        // Separator + total
        table << fort::separator;
        {
            size_t total_count = 0;
            for (const auto &e : agg)
                total_count += e.count;
            size_t count_per_iter = total_count / accumulated_iterations_;
            table << "TOTAL"
                  << std::to_string(count_per_iter)
                  << fmt_ms(total_gpu_ms)
                  << fmt_ms(avg_gpu_ms)
                  << "100.0%"
                  << fort::endr;
        }

        std::cout << table.to_string() << std::flush;

        // Reset accumulated state
        resetAccumulated();
        return true;
    }

    bool StageTimeline::printAccumulatedPrefillSummary(const char *device_name)
    {
        if (prefill_accumulated_iterations_ == 0 || prefill_accumulated_.empty())
            return false;

        // Build sorted vector from prefill accumulated map
        std::vector<TypeAggregate> agg;
        agg.reserve(prefill_accumulated_.size());
        for (const auto &[_, val] : prefill_accumulated_)
            agg.push_back(val);
        std::sort(agg.begin(), agg.end(),
                  [](const TypeAggregate &a, const TypeAggregate &b)
                  { return a.total_ms > b.total_ms; });

        float total_gpu_ms = 0.0f;
        for (const auto &e : agg)
            total_gpu_ms += e.total_ms;

        float avg_gpu_ms = total_gpu_ms / prefill_accumulated_iterations_;
        double avg_wall_ms = prefill_accumulated_wall_ms_ / prefill_accumulated_iterations_;
        size_t avg_tokens = prefill_accumulated_tokens_ / prefill_accumulated_iterations_;

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Title row
        {
            std::ostringstream title;
            title << "GPU STAGE EVENTS: PREFILL";
            if (device_name)
                title << " [" << device_name << "]";
            title << " (avg of " << prefill_accumulated_iterations_ << " iterations, "
                  << avg_tokens << " tokens)";
            table << title.str() << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(5);
            table[0][0].set_cell_text_align(fort::text_align::center);
        }

        // Summary info
        {
            std::ostringstream info;
            info << std::fixed << std::setprecision(2);
            info << "Event Avg: " << avg_gpu_ms << " ms";
            info << "  |  Wall Avg: " << avg_wall_ms << " ms";
            double gap = avg_wall_ms - avg_gpu_ms;
            info << "  |  Non-event wall: " << gap << " ms";
            if (avg_wall_ms > 0.0)
                info << " (" << std::setprecision(1) << (gap / avg_wall_ms * 100.0) << "%)";
            if (avg_tokens > 0 && avg_gpu_ms > 0.0f)
            {
                double tok_per_sec = avg_tokens / (avg_gpu_ms / 1000.0);
                info << "  |  Event-limited: " << std::setprecision(1) << tok_per_sec << " tok/s";
            }
            table << info.str() << "" << "" << "" << "" << fort::endr;
            table[1][0].set_cell_span(5);
        }

        // Header
        table << fort::header << "STAGE TYPE" << "COUNT/iter" << "TOTAL (ms)" << "AVG/iter (ms)" << "%" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::right);
        table.column(4).set_cell_text_align(fort::text_align::right);

        auto fmt_ms = [](float ms) -> std::string
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << ms;
            return oss.str();
        };

        for (const auto &entry : agg)
        {
            float pct = total_gpu_ms > 0.0f ? (entry.total_ms / total_gpu_ms) * 100.0f : 0.0f;
            std::ostringstream pct_str;
            pct_str << std::fixed << std::setprecision(1) << pct << "%";

            size_t count_per_iter = entry.count / prefill_accumulated_iterations_;
            float ms_per_iter = entry.total_ms / prefill_accumulated_iterations_;

            table << entry.type_name
                  << std::to_string(count_per_iter)
                  << fmt_ms(entry.total_ms)
                  << fmt_ms(ms_per_iter)
                  << pct_str.str()
                  << fort::endr;
        }

        // Separator + total
        table << fort::separator;
        {
            size_t total_count = 0;
            for (const auto &e : agg)
                total_count += e.count;
            size_t count_per_iter = total_count / prefill_accumulated_iterations_;
            table << "TOTAL"
                  << std::to_string(count_per_iter)
                  << fmt_ms(total_gpu_ms)
                  << fmt_ms(avg_gpu_ms)
                  << "100.0%"
                  << fort::endr;
        }

        std::cout << table.to_string() << std::flush;

        // Reset accumulated prefill state
        resetAccumulatedPrefill();
        return true;
    }

    void StageTimeline::printDetailedTimeline(const char *phase_name,
                                              const char *device_name) const
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        {
            std::ostringstream title;
            title << "GPU DETAILED STAGE EVENTS: " << phase_name;
            if (device_name)
                title << " [" << device_name << "]";
            table << title.str() << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(4);
            table[0][0].set_cell_text_align(fort::text_align::center);
        }

        table << fort::header << "#" << "STAGE NAME" << "TYPE" << "GPU (ms)" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::right);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(2).set_cell_text_align(fort::text_align::left);
        table.column(3).set_cell_text_align(fort::text_align::right);

        float total_gpu_ms = totalGpuMs();
        size_t idx = 0;

        for (const auto &rec : records_)
        {
            if (!rec.valid)
                continue;

            std::ostringstream ms_str;
            ms_str << std::fixed << std::setprecision(3) << rec.gpu_ms;

            table << std::to_string(idx)
                  << rec.name
                  << computeStageTypeName(rec.type)
                  << ms_str.str()
                  << fort::endr;
            ++idx;
        }

        table << fort::separator;
        {
            std::ostringstream total_str;
            total_str << std::fixed << std::setprecision(3) << total_gpu_ms;
            table << "" << "TOTAL" << "" << total_str.str() << fort::endr;
        }

        std::cout << table.to_string() << std::flush;
    }

    void StageTimeline::recordPerfStats(const char *phase_name,
                                        const char *device_name,
                                        const char *domain,
                                        std::map<std::string, std::string> tags) const
    {
        if (!PerfStatsCollector::isEnabled())
            return;

        const std::string phase = phase_name ? phase_name : "";
        const std::string device = device_name ? device_name : "";
        const std::string perf_domain = domain ? domain : "stage_gpu";
        auto merge_tags = [&](PerfStatsCollector::Tags record_tags = {}) {
            record_tags.emplace("attribution", "gpu_event");
            record_tags.emplace("source", "stage_timeline");
            record_tags.emplace("graph_capture_scope", "eager_stage_events");
            record_tags.insert(tags.begin(), tags.end());
            return record_tags;
        };

        const float total_ms = totalGpuMs();
        if (total_ms > 0.0f)
        {
            PerfStatsCollector::recordTimingNs(
                perf_domain,
                "total",
                static_cast<uint64_t>(static_cast<double>(total_ms) * 1.0e6),
                phase,
                device,
                merge_tags());
        }

        auto agg = aggregateByType();
        for (const auto &entry : agg)
        {
            if (entry.total_ms <= 0.0f)
                continue;
            PerfStatsCollector::recordTimingNs(
                perf_domain,
                std::string("type.") + entry.type_name,
                static_cast<uint64_t>(static_cast<double>(entry.total_ms) * 1.0e6),
                phase,
                device,
                merge_tags({{"stage_count", std::to_string(entry.count)}}));
        }

        for (size_t i = 0; i < records_.size(); ++i)
        {
            const auto &rec = records_[i];
            if (!rec.valid || rec.gpu_ms <= 0.0f)
                continue;
            PerfStatsCollector::recordTimingNs(
                perf_domain,
                rec.name.empty() ? "(unnamed)" : rec.name,
                static_cast<uint64_t>(static_cast<double>(rec.gpu_ms) * 1.0e6),
                phase,
                device,
                merge_tags({{"type", computeStageTypeName(rec.type)},
                            {"index", std::to_string(i)}}));
        }
    }

} // namespace llaminar2
