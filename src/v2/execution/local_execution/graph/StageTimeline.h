/**
 * @file StageTimeline.h
 * @brief GPU event-based per-stage timeline profiler for the production fast path
 *
 * Records GPU events around each stage in executeFastDecode() to measure
 * true GPU-side kernel execution time with minimal overhead:
 * - Event record: ~1μs CPU overhead per call, zero GPU overhead
 * - Timing collection: single GPU sync at end of forward pass
 *
 * Gated by LLAMINAR_GPU_STAGE_TIMING=1. Zero overhead when disabled.
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>
#include "../../../backends/IWorkerGPUContext.h"
#include "../../compute_stages/IComputeStage.h"

namespace llaminar2
{

    /**
     * @brief GPU event-based stage timeline profiler
     *
     * Pre-allocates event pairs for each stage, records start/stop events
     * around stage execution on the GPU stream, and collects timings via
     * a single GPU sync at the end of the forward pass.
     *
     * Usage:
     *   StageTimeline timeline;
     *   timeline.initialize(gpu_ctx, num_stages);
     *   // For each stage:
     *   timeline.recordStart(i, gpu_ctx, stream);
     *   stage->execute(ctx);
     *   timeline.recordStop(i, gpu_ctx, stream);
     *   // After forward pass:
     *   timeline.collect(gpu_ctx);
     *   timeline.printSummary("PREFILL", token_count);
     */
    class StageTimeline
    {
    public:
        struct StageRecord
        {
            std::string name;
            ComputeStageType type;
            void *event_start = nullptr;
            void *event_stop = nullptr;
            float gpu_ms = 0.0f;
            bool valid = false; ///< True if both events were recorded
        };

        StageTimeline() = default;
        ~StageTimeline() { destroy(); }

        // Non-copyable
        StageTimeline(const StageTimeline &) = delete;
        StageTimeline &operator=(const StageTimeline &) = delete;

        // Move support
        StageTimeline(StageTimeline &&other) noexcept
            : records_(std::move(other.records_)),
              gpu_ctx_(other.gpu_ctx_),
              initialized_(other.initialized_)
        {
            other.gpu_ctx_ = nullptr;
            other.initialized_ = false;
        }

        StageTimeline &operator=(StageTimeline &&other) noexcept
        {
            if (this != &other)
            {
                destroy();
                records_ = std::move(other.records_);
                gpu_ctx_ = other.gpu_ctx_;
                initialized_ = other.initialized_;
                other.gpu_ctx_ = nullptr;
                other.initialized_ = false;
            }
            return *this;
        }

        /**
         * @brief Pre-allocate GPU events for N stages
         *
         * Creates 2*N GPU events (start + stop for each stage).
         * Call once before the first forward pass.
         *
         * @param gpu_ctx GPU context for event creation
         * @param num_stages Number of stages in the execution order
         */
        void initialize(IWorkerGPUContext *gpu_ctx, size_t num_stages)
        {
            if (initialized_)
                destroy();

            gpu_ctx_ = gpu_ctx;
            records_.resize(num_stages);

            for (auto &rec : records_)
            {
                rec.event_start = gpu_ctx->createEvent();
                rec.event_stop = gpu_ctx->createEvent();
            }

            initialized_ = true;
        }

        /**
         * @brief Grow event pool if execution order has more stages than allocated
         */
        void ensureCapacity(IWorkerGPUContext *gpu_ctx, size_t num_stages)
        {
            if (num_stages <= records_.size())
                return;

            size_t old_size = records_.size();
            records_.resize(num_stages);

            for (size_t i = old_size; i < num_stages; ++i)
            {
                records_[i].event_start = gpu_ctx->createEvent();
                records_[i].event_stop = gpu_ctx->createEvent();
            }

            gpu_ctx_ = gpu_ctx;
            initialized_ = true;
        }

        /**
         * @brief Set stage metadata (name + type) for a slot
         */
        void setStageInfo(size_t idx, const std::string &name, ComputeStageType type)
        {
            if (idx < records_.size())
            {
                records_[idx].name = name;
                records_[idx].type = type;
                records_[idx].valid = false;
                records_[idx].gpu_ms = 0.0f;
            }
        }

        /**
         * @brief Record a start event for stage at index
         * Cost: ~1μs CPU, zero GPU overhead
         */
        void recordStart(size_t idx, IWorkerGPUContext *gpu_ctx, void *stream)
        {
            if (idx < records_.size() && records_[idx].event_start)
            {
                gpu_ctx->recordEvent(records_[idx].event_start, stream);
            }
        }

        /**
         * @brief Record a stop event for stage at index
         * Cost: ~1μs CPU, zero GPU overhead
         */
        void recordStop(size_t idx, IWorkerGPUContext *gpu_ctx, void *stream)
        {
            if (idx < records_.size() && records_[idx].event_stop)
            {
                gpu_ctx->recordEvent(records_[idx].event_stop, stream);
                records_[idx].valid = true;
            }
        }

        /**
         * @brief Synchronize and collect all GPU event timings
         *
         * Syncs on the last recorded stop event (single GPU sync), then
         * queries elapsed time for all recorded event pairs.
         *
         * @param gpu_ctx GPU context with eventElapsedTime support
         */
        void collect(IWorkerGPUContext *gpu_ctx)
        {
            // Find the last valid stop event to sync on
            void *last_stop = nullptr;
            for (auto it = records_.rbegin(); it != records_.rend(); ++it)
            {
                if (it->valid && it->event_stop)
                {
                    last_stop = it->event_stop;
                    break;
                }
            }

            if (!last_stop)
                return;

            // Single GPU sync — blocks until all stages complete
            gpu_ctx->synchronizeEvent(last_stop);

            // Query all elapsed times (CPU-only after sync, no GPU waits)
            for (auto &rec : records_)
            {
                if (rec.valid && rec.event_start && rec.event_stop)
                {
                    rec.gpu_ms = gpu_ctx->eventElapsedTime(rec.event_start, rec.event_stop);
                }
            }
        }

        /**
         * @brief Get total GPU time across all recorded stages
         */
        float totalGpuMs() const
        {
            float total = 0.0f;
            for (const auto &rec : records_)
            {
                if (rec.valid && rec.gpu_ms > 0.0f)
                    total += rec.gpu_ms;
            }
            return total;
        }

        /**
         * @brief Get per-type aggregated timing
         *
         * Returns sorted vector of (type_name, total_ms, count, avg_ms)
         */
        struct TypeAggregate
        {
            std::string type_name;
            float total_ms = 0.0f;
            size_t count = 0;
            float avg_ms() const { return count > 0 ? total_ms / count : 0.0f; }
        };

        std::vector<TypeAggregate> aggregateByType() const
        {
            std::unordered_map<std::string, TypeAggregate> agg;

            for (const auto &rec : records_)
            {
                if (!rec.valid || rec.gpu_ms <= 0.0f)
                    continue;

                const char *type_name = computeStageTypeName(rec.type);
                auto &entry = agg[type_name];
                entry.type_name = type_name;
                entry.total_ms += rec.gpu_ms;
                entry.count++;
            }

            std::vector<TypeAggregate> result;
            result.reserve(agg.size());
            for (auto &[_, val] : agg)
                result.push_back(std::move(val));

            // Sort descending by total time
            std::sort(result.begin(), result.end(),
                      [](const TypeAggregate &a, const TypeAggregate &b)
                      { return a.total_ms > b.total_ms; });

            return result;
        }

        /**
         * @brief Get per-stage records (for detailed per-stage output)
         */
        const std::vector<StageRecord> &records() const { return records_; }

        /**
         * @brief Print a formatted summary table to stdout
         *
         * @param phase_name Phase label (e.g., "PREFILL", "DECODE")
         * @param token_count Number of tokens processed (for throughput)
         * @param wall_ms Wall-clock time for comparison (0 = skip)
         */
        void printSummary(const char *phase_name, size_t token_count, double wall_ms = 0.0,
                          const char *device_name = nullptr) const;

        /**
         * @brief Print detailed per-stage timeline (all stages individually)
         *
         * @param phase_name Phase label
         * @param device_name Optional device identifier (e.g., "rocm:0")
         */
        void printDetailedTimeline(const char *phase_name = "STAGES",
                                   const char *device_name = nullptr) const;

        /**
         * @brief Export current per-stage GPU event timings into PerfStatsCollector.
         */
        void recordPerfStats(const char *phase_name,
                             const char *device_name = nullptr,
                             const char *domain = "stage_gpu",
                             std::map<std::string, std::string> tags = {}) const;

        bool isInitialized() const { return initialized_; }

        /**
         * @brief Reset all timing data (keeps events allocated)
         */
        void resetTimings()
        {
            for (auto &rec : records_)
            {
                rec.valid = false;
                rec.gpu_ms = 0.0f;
            }
        }

        /**
         * @brief Accumulate current iteration's timings into running totals
         *
         * Adds per-type totals from the current iteration to accumulated state,
         * increments the iteration counter, and accumulates wall time.
         * Call this instead of printSummary() for decode iterations to avoid
         * printing a table for every single token.
         */
        void accumulateIteration(double wall_ms)
        {
            auto agg = aggregateByType();
            for (const auto &entry : agg)
            {
                auto &acc = accumulated_[entry.type_name];
                acc.type_name = entry.type_name;
                acc.total_ms += entry.total_ms;
                acc.count += entry.count;
            }
            accumulated_wall_ms_ += wall_ms;
            accumulated_iterations_++;
        }

        /**
         * @brief Accumulate current iteration's prefill timings
         *
         * Separate accumulation buffer for prefill (vs decode in accumulateIteration).
         * Used by benchmark mode to avoid per-iteration prefill table printing.
         */
        void accumulatePrefillIteration(double wall_ms, size_t token_count)
        {
            auto agg = aggregateByType();
            for (const auto &entry : agg)
            {
                auto &acc = prefill_accumulated_[entry.type_name];
                acc.type_name = entry.type_name;
                acc.total_ms += entry.total_ms;
                acc.count += entry.count;
            }
            prefill_accumulated_wall_ms_ += wall_ms;
            prefill_accumulated_tokens_ += token_count;
            prefill_accumulated_iterations_++;
        }

        /**
         * @brief Print accumulated summary across multiple decode iterations
         *
         * Prints a single table summarizing all accumulated decode iterations,
         * then resets the accumulated state.
         *
         * @return true if there was data to print
         */
        bool printAccumulatedSummary(const char *phase_name, const char *device_name = nullptr);

        /**
         * @brief Print accumulated prefill summary and reset
         *
         * Prints a single table summarizing all accumulated prefill iterations.
         *
         * @return true if there was data to print
         */
        bool printAccumulatedPrefillSummary(const char *device_name = nullptr);

        /**
         * @brief Check if there is accumulated data pending
         */
        bool hasAccumulatedData() const { return accumulated_iterations_ > 0; }

        /**
         * @brief Check if there is accumulated prefill data pending
         */
        bool hasAccumulatedPrefillData() const { return prefill_accumulated_iterations_ > 0; }

        /**
         * @brief Reset accumulated data without printing
         */
        void resetAccumulated()
        {
            accumulated_.clear();
            accumulated_wall_ms_ = 0.0;
            accumulated_iterations_ = 0;
        }

        /**
         * @brief Reset accumulated prefill data without printing
         */
        void resetAccumulatedPrefill()
        {
            prefill_accumulated_.clear();
            prefill_accumulated_wall_ms_ = 0.0;
            prefill_accumulated_tokens_ = 0;
            prefill_accumulated_iterations_ = 0;
        }

    private:
        void destroy()
        {
            if (gpu_ctx_ && initialized_)
            {
                for (auto &rec : records_)
                {
                    if (rec.event_start)
                        gpu_ctx_->destroyEvent(rec.event_start);
                    if (rec.event_stop)
                        gpu_ctx_->destroyEvent(rec.event_stop);
                    rec.event_start = nullptr;
                    rec.event_stop = nullptr;
                }
            }
            records_.clear();
            initialized_ = false;
            gpu_ctx_ = nullptr;
        }

        std::vector<StageRecord> records_;
        IWorkerGPUContext *gpu_ctx_ = nullptr;
        bool initialized_ = false;

        // Accumulated state for multi-iteration decode summaries
        std::unordered_map<std::string, TypeAggregate> accumulated_;
        double accumulated_wall_ms_ = 0.0;
        size_t accumulated_iterations_ = 0;

        // Accumulated state for multi-iteration prefill summaries (benchmark mode)
        std::unordered_map<std::string, TypeAggregate> prefill_accumulated_;
        double prefill_accumulated_wall_ms_ = 0.0;
        size_t prefill_accumulated_tokens_ = 0;
        size_t prefill_accumulated_iterations_ = 0;
    };

} // namespace llaminar2
