/**
 * @file PerformanceTracer.h
 * @brief Hierarchical performance tracing framework for hot path analysis
 * @author David Sanftenberg
 * @date 2025-10-15
 *
 * Zero-overhead performance instrumentation with hierarchical timing,
 * MPI-aware aggregation, and selective activation via environment variables.
 *
 * Usage:
 *   PERF_TRACE_SCOPE("operation_name");  // RAII-based scoped timer
 *   PERF_TRACE_BEGIN("operation");       // Manual begin/end pair
 *   PERF_TRACE_END("operation");
 *
 * Environment Variables:
 *   LLAMINAR_PERF_TRACE=1               - Enable tracing
 *   LLAMINAR_PERF_TRACE_DETAIL=high     - Set detail level (low/medium/high)
 *   LLAMINAR_PERF_TRACE_DUMP=trace.json - Dump results to JSON
 *   LLAMINAR_PERF_TRACE_FILTER=prefill  - Only trace operations matching filter
 */

#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdint>

namespace llaminar
{
    namespace perf
    {

        // Forward declarations
        class PerformanceTracer;
        class TraceScope;

        /**
         * @brief Single trace event with hierarchical relationship tracking
         */
        struct TraceEvent
        {
            std::string name;
            std::chrono::high_resolution_clock::time_point start_time;
            std::chrono::high_resolution_clock::time_point end_time;
            uint64_t duration_us; // Microseconds
            int thread_id;
            int mpi_rank;
            int depth;            // Nesting depth
            std::string category; // "prefill", "decode", "kernel", "matmul", etc.

            // Statistics for aggregated events
            uint64_t call_count = 1;
            uint64_t total_us = 0;
            uint64_t min_us = UINT64_MAX;
            uint64_t max_us = 0;

            TraceEvent() = default;
            TraceEvent(const std::string &n, int tid, int rank, int d, const std::string &cat)
                : name(n), thread_id(tid), mpi_rank(rank), depth(d), category(cat) {}
        };

        /**
         * @brief Per-thread trace stack for hierarchical timing
         */
        struct ThreadTraceStack
        {
            std::vector<TraceEvent *> active_stack;
            std::vector<std::unique_ptr<TraceEvent>> completed_events;
            int thread_id;

            ThreadTraceStack(int tid) : thread_id(tid) {}
        };

        /**
         * @brief Main performance tracer singleton
         *
         * Thread-safe, MPI-aware hierarchical performance tracing.
         * Maintains separate stacks per thread and aggregates results.
         */
        class PerformanceTracer
        {
        public:
            static PerformanceTracer &instance();

            // Configuration
            bool isEnabled() const { return enabled_; }
            void setEnabled(bool enabled) { enabled_ = enabled; }

            enum class DetailLevel
            {
                LOW,
                MEDIUM,
                HIGH
            };
            DetailLevel getDetailLevel() const { return detail_level_; }
            void setDetailLevel(DetailLevel level) { detail_level_ = level; }

            void setFilter(const std::string &filter) { filter_ = filter; }
            bool matchesFilter(const std::string &name) const;

            // Tracing operations
            void beginTrace(const std::string &name, const std::string &category = "");
            void endTrace(const std::string &name);

            // Results
            void dumpResults(const std::string &filename = "");
            void printSummary(bool verbose = false);
            void reset();

            // MPI integration
            void setMPIRank(int rank) { mpi_rank_ = rank; }
            int getMPIRank() const { return mpi_rank_; }
            void aggregateMPI(); // Gather stats across ranks

        private:
            PerformanceTracer();
            ~PerformanceTracer();

            // Prevent copying
            PerformanceTracer(const PerformanceTracer &) = delete;
            PerformanceTracer &operator=(const PerformanceTracer &) = delete;

            ThreadTraceStack &getThreadStack();
            int getCurrentThreadId() const;

            bool enabled_;
            DetailLevel detail_level_;
            std::string filter_;
            int mpi_rank_;

            std::mutex mutex_;
            std::map<int, std::unique_ptr<ThreadTraceStack>> thread_stacks_;

            // Aggregated statistics (post-processing)
            std::map<std::string, TraceEvent> aggregated_stats_;
        };

        /**
         * @brief RAII scope-based tracing
         *
         * Usage:
         *   {
         *     PERF_TRACE_SCOPE("my_operation");
         *     // ... code to trace ...
         *   } // Automatically ends trace
         */
        class TraceScope
        {
        public:
            TraceScope(const std::string &name, const std::string &category = "")
                : name_(name), active_(false)
            {
                if (PerformanceTracer::instance().isEnabled() &&
                    PerformanceTracer::instance().matchesFilter(name))
                {
                    PerformanceTracer::instance().beginTrace(name, category);
                    active_ = true;
                }
            }

            ~TraceScope()
            {
                if (active_)
                {
                    PerformanceTracer::instance().endTrace(name_);
                }
            }

        private:
            std::string name_;
            bool active_;
        };

// Convenience macros (compile-time elimination when disabled)
#ifdef LLAMINAR_ENABLE_PERF_TRACE
#define PERF_TRACE_SCOPE(name) ::llaminar::perf::TraceScope _perf_trace_##__LINE__(name)
#define PERF_TRACE_SCOPE_CAT(name, category) ::llaminar::perf::TraceScope _perf_trace_##__LINE__(name, category)
#define PERF_TRACE_BEGIN(name) ::llaminar::perf::PerformanceTracer::instance().beginTrace(name)
#define PERF_TRACE_END(name) ::llaminar::perf::PerformanceTracer::instance().endTrace(name)
#define PERF_TRACE_ENABLED() ::llaminar::perf::PerformanceTracer::instance().isEnabled()
#else
#define PERF_TRACE_SCOPE(name) \
    do                         \
    {                          \
    } while (0)
#define PERF_TRACE_SCOPE_CAT(name, category) \
    do                                       \
    {                                        \
    } while (0)
#define PERF_TRACE_BEGIN(name) \
    do                         \
    {                          \
    } while (0)
#define PERF_TRACE_END(name) \
    do                       \
    {                        \
    } while (0)
#define PERF_TRACE_ENABLED() false
#endif

    } // namespace perf
} // namespace llaminar
