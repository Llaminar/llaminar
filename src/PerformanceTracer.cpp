/**
 * @file PerformanceTracer.cpp
 * @brief Implementation of hierarchical performance tracing framework
 * @author David Sanftenberg
 * @date 2025-10-15
 */

#include "PerformanceTracer.h"
#include "utils/DebugEnv.h"
#include "Logger.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstring>

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

namespace llaminar
{
    namespace perf
    {

        PerformanceTracer &PerformanceTracer::instance()
        {
            static PerformanceTracer instance;
            return instance;
        }

        PerformanceTracer::PerformanceTracer()
            : enabled_(false), detail_level_(DetailLevel::MEDIUM), filter_(""), mpi_rank_(0)
        {
            // Read configuration from environment
            const auto &env = debugEnv();

            if (env.perf.trace_enabled)
            {
                enabled_ = true;
                LOG_INFO("Performance tracing ENABLED");

                if (!env.perf.trace_filter.empty())
                {
                    filter_ = env.perf.trace_filter;
                    LOG_INFO("Performance trace filter: '" << filter_ << "'");
                }

                // Parse detail level
                if (env.perf.trace_detail == "high")
                {
                    detail_level_ = DetailLevel::HIGH;
                }
                else if (env.perf.trace_detail == "low")
                {
                    detail_level_ = DetailLevel::LOW;
                }
                else
                {
                    detail_level_ = DetailLevel::MEDIUM;
                }

                LOG_INFO("Performance trace detail level: " << env.perf.trace_detail);
            }
        }

        PerformanceTracer::~PerformanceTracer()
        {
            if (enabled_ && debugEnv().perf.trace_dump_on_exit)
            {
                const std::string filename = debugEnv().perf.trace_output_file.empty()
                                                 ? "llaminar_trace.json"
                                                 : debugEnv().perf.trace_output_file;

                dumpResults(filename);
                printSummary(detail_level_ == DetailLevel::HIGH);
            }
        }

        int PerformanceTracer::getCurrentThreadId() const
        {
            std::hash<std::thread::id> hasher;
            return static_cast<int>(hasher(std::this_thread::get_id()) % 10000);
        }

        ThreadTraceStack &PerformanceTracer::getThreadStack()
        {
            int tid = getCurrentThreadId();

            std::lock_guard<std::mutex> lock(mutex_);

            auto it = thread_stacks_.find(tid);
            if (it == thread_stacks_.end())
            {
                auto stack = std::make_unique<ThreadTraceStack>(tid);
                auto *ptr = stack.get();
                thread_stacks_[tid] = std::move(stack);
                return *ptr;
            }

            return *it->second;
        }

        bool PerformanceTracer::matchesFilter(const std::string &name) const
        {
            if (filter_.empty())
                return true;
            return name.find(filter_) != std::string::npos;
        }

        void PerformanceTracer::beginTrace(const std::string &name, const std::string &category)
        {
            if (!enabled_)
                return;

            auto &stack = getThreadStack();

            auto event = std::make_unique<TraceEvent>(
                name,
                stack.thread_id,
                mpi_rank_,
                static_cast<int>(stack.active_stack.size()),
                category);

            event->start_time = std::chrono::high_resolution_clock::now();

            auto *event_ptr = event.get();
            stack.completed_events.push_back(std::move(event));
            stack.active_stack.push_back(event_ptr);
        }

        void PerformanceTracer::endTrace(const std::string &name)
        {
            if (!enabled_)
                return;

            auto end_time = std::chrono::high_resolution_clock::now();

            auto &stack = getThreadStack();

            if (stack.active_stack.empty())
            {
                LOG_WARN("PerformanceTracer: endTrace('" << name << "') called with empty stack");
                return;
            }

            auto *event = stack.active_stack.back();
            stack.active_stack.pop_back();

            if (event->name != name)
            {
                LOG_WARN("PerformanceTracer: endTrace('" << name << "') doesn't match beginTrace('"
                                                         << event->name << "')");
            }

            event->end_time = end_time;
            event->duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                     event->end_time - event->start_time)
                                     .count();
        }

        void PerformanceTracer::printSummary(bool verbose)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            // Aggregate events by name
            std::map<std::string, TraceEvent> aggregated;

            for (const auto &[tid, stack] : thread_stacks_)
            {
                for (const auto &event : stack->completed_events)
                {
                    if (event->duration_us == 0)
                        continue; // Skip incomplete traces

                    const std::string key = event->name;

                    if (aggregated.find(key) == aggregated.end())
                    {
                        aggregated[key] = *event;
                        aggregated[key].call_count = 0;
                        aggregated[key].total_us = 0;
                        aggregated[key].min_us = UINT64_MAX;
                        aggregated[key].max_us = 0;
                    }

                    auto &agg = aggregated[key];
                    agg.call_count++;
                    agg.total_us += event->duration_us;
                    agg.min_us = std::min(agg.min_us, event->duration_us);
                    agg.max_us = std::max(agg.max_us, event->duration_us);
                }
            }

            // Sort by total time
            std::vector<std::pair<std::string, TraceEvent>> sorted;
            for (const auto &[name, event] : aggregated)
            {
                sorted.push_back({name, event});
            }

            std::sort(sorted.begin(), sorted.end(),
                      [](const auto &a, const auto &b)
                      {
                          return a.second.total_us > b.second.total_us;
                      });

            // Print summary
            if (mpi_rank_ == 0)
            {
                std::cout << "\n";
                std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
                std::cout << "║                    PERFORMANCE TRACE SUMMARY                             ║\n";
                std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ " << std::left << std::setw(30) << "Operation"
                          << std::right << std::setw(10) << "Calls"
                          << std::setw(12) << "Total (ms)"
                          << std::setw(12) << "Avg (ms)"
                          << std::setw(12) << "Min (μs)"
                          << std::setw(12) << "Max (μs)" << " ║\n";
                std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";

                for (const auto &[name, event] : sorted)
                {
                    double total_ms = event.total_us / 1000.0;
                    double avg_ms = total_ms / event.call_count;

                    std::cout << "║ " << std::left << std::setw(30) << (name.length() > 30 ? name.substr(0, 27) + "..." : name)
                              << std::right << std::setw(10) << event.call_count
                              << std::setw(12) << std::fixed << std::setprecision(2) << total_ms
                              << std::setw(12) << std::fixed << std::setprecision(3) << avg_ms
                              << std::setw(12) << event.min_us
                              << std::setw(12) << event.max_us << " ║\n";

                    if (verbose)
                    {
                        std::cout << "║   Category: " << std::left << std::setw(20) << event.category
                                  << "  Depth: " << event.depth
                                  << std::setw(48) << "" << "║\n";
                    }
                }

                std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";
                std::cout << "\n";
            }

            aggregated_stats_ = aggregated;
        }

        void PerformanceTracer::dumpResults(const std::string &filename)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            const std::string outfile = filename.empty() ? "llaminar_trace.json" : filename;

            std::ofstream out(outfile);
            if (!out)
            {
                LOG_ERROR("Failed to open trace output file: " << outfile);
                return;
            }

            out << "{\n";
            out << "  \"traceEvents\": [\n";

            bool first = true;
            for (const auto &[tid, stack] : thread_stacks_)
            {
                for (const auto &event : stack->completed_events)
                {
                    if (event->duration_us == 0)
                        continue;

                    if (!first)
                        out << ",\n";
                    first = false;

                    // Chrome trace format
                    out << "    {\n";
                    out << "      \"name\": \"" << event->name << "\",\n";
                    out << "      \"cat\": \"" << event->category << "\",\n";
                    out << "      \"ph\": \"X\",\n"; // Complete event
                    out << "      \"ts\": " << std::chrono::duration_cast<std::chrono::microseconds>(event->start_time.time_since_epoch()).count() << ",\n";
                    out << "      \"dur\": " << event->duration_us << ",\n";
                    out << "      \"tid\": " << event->thread_id << ",\n";
                    out << "      \"pid\": " << event->mpi_rank << ",\n";
                    out << "      \"args\": { \"depth\": " << event->depth << " }\n";
                    out << "    }";
                }
            }

            out << "\n  ],\n";
            out << "  \"displayTimeUnit\": \"ms\",\n";
            out << "  \"otherData\": {\n";
            out << "    \"mpi_rank\": " << mpi_rank_ << "\n";
            out << "  }\n";
            out << "}\n";

            out.close();

            if (mpi_rank_ == 0)
            {
                LOG_INFO("Performance trace written to: " << outfile);
                LOG_INFO("View in Chrome: chrome://tracing");
            }
        }

        void PerformanceTracer::reset()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            thread_stacks_.clear();
            aggregated_stats_.clear();
        }

        void PerformanceTracer::aggregateMPI()
        {
#ifdef MPI_ENABLED
            // TODO: Implement MPI aggregation
            // For now, each rank dumps its own trace
            // Future: gather all events to rank 0
#endif
        }

    } // namespace perf
} // namespace llaminar
