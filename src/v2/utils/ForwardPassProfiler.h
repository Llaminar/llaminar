/**
 * @file ForwardPassProfiler.h
 * @brief Wall-clock decomposition profiler for forward pass phases
 *
 * Tracks the host-side time breakdown of each forward pass step, accumulating
 * across decode iterations and printing a summary table at benchmark end.
 * This complements the GPU Stage Timeline (which shows per-kernel GPU time)
 * by revealing WHERE wall-clock time is spent at the engine/orchestration level.
 *
 * Gated by LLAMINAR_PROFILING=1 (via KernelProfiler::isEnabled()). Zero overhead
 * when disabled — all record*() calls are inlined no-ops.
 *
 * Lifecycle: Owned by ForwardExecutionEngine. Accumulated data is flushed by
 * flushStageTimeline() at benchmark end, alongside GPU stage timing tables.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "fort.hpp"

namespace llaminar2
{

    /**
     * @brief Accumulated wall-clock timing for forward pass phases
     *
     * Each phase is a contiguous time span within a single forward pass call.
     * Phases are non-overlapping and sum to the total forward pass wall time.
     */
    class ForwardPassProfiler
    {
    public:
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;

        /**
         * @brief Per-iteration timing snapshot (nanoseconds)
         *
         * Phases within executeCacheHit():
         *   start → [setup] → exec_t0 → [execute] → exec_t1 → [sync] → end
         *
         * Sub-phases within graph replay (executeReplayPhase):
         *   [graph_launch]: hipGraphLaunch() / cudaGraphLaunch()
         *   [post_launch]:  markOutputsDirty + onGraphReplayed callbacks
         *   [stream_sync]:  synchronizeStream() calls at end of replay
         */
        struct PhaseTimings
        {
            uint64_t setup_ns = 0;       ///< Workspace check, token copy, stream setup, dynamic params, graph reset
            uint64_t execute_ns = 0;     ///< Total graph execution (includes all sub-phases below)
            uint64_t sync_ns = 0;        ///< Post-execution sync + timeline collection

            // Sub-phases of setup
            uint64_t setup_workspace_ns = 0;   ///< Workspace allocation check + generation
            uint64_t setup_token_copy_ns = 0;  ///< Token ID + position ID memcpy
            uint64_t setup_stream_ns = 0;      ///< GPU stream assignment to stages
            uint64_t setup_dynamic_params_ns = 0; ///< updateDynamicParams() calls
            uint64_t setup_graph_reset_ns = 0; ///< graph->reset() call

            // Sub-phases of execute (graph replay path only)
            uint64_t graph_launch_ns = 0;    ///< hipGraphLaunch / cudaGraphLaunch
            uint64_t post_launch_ns = 0;     ///< markOutputsDirty + onGraphReplayed callbacks
            uint64_t stream_sync_ns = 0;     ///< synchronizeStream() at end of replay phase
        };

        /**
         * @brief Thread-local replay phase timing accumulator
         *
         * The graph replay phase runs inside executeReplayPhase() which is called
         * from within executeCacheHit(). Since the replay function doesn't return
         * timing data in its result struct, we use thread-local storage to pass
         * the sub-phase breakdown back to the caller.
         *
         * Usage:
         *   // In executeReplayPhase():
         *   ForwardPassProfiler::resetReplayTimings();
         *   ... (record sub-phases) ...
         *   ForwardPassProfiler::addReplayLaunchNs(ns);
         *
         *   // In executeCacheHit():
         *   auto timings = ForwardPassProfiler::consumeReplayTimings();
         */
        struct ReplayPhaseTimings
        {
            uint64_t graph_launch_ns = 0;
            uint64_t post_launch_ns = 0;
            uint64_t stream_sync_ns = 0;
        };

        /// @brief Reset thread-local replay timings (call before replay phase)
        static void resetReplayTimings()
        {
            auto &t = tls_replay_timings();
            t.graph_launch_ns = 0;
            t.post_launch_ns = 0;
            t.stream_sync_ns = 0;
        }

        /// @brief Add graph launch time (call in replay segment execution)
        static void addReplayLaunchNs(uint64_t ns) { tls_replay_timings().graph_launch_ns += ns; }

        /// @brief Add post-launch callback time
        static void addReplayPostLaunchNs(uint64_t ns) { tls_replay_timings().post_launch_ns += ns; }

        /// @brief Add stream sync time
        static void addReplayStreamSyncNs(uint64_t ns) { tls_replay_timings().stream_sync_ns += ns; }

        /// @brief Consume and return the accumulated replay timings (resets them)
        static ReplayPhaseTimings consumeReplayTimings()
        {
            auto &t = tls_replay_timings();
            ReplayPhaseTimings result = t;
            t = {};
            return result;
        }

        ForwardPassProfiler() = default;

        // ----- Accumulation Interface -----

        /**
         * @brief Record one decode iteration's timing breakdown
         *
         * Called at the end of executeCacheHit() when profiling is enabled.
         * Thread-safety: NOT thread-safe — called from a single orchestrator thread.
         */
        void recordDecodeIteration(const PhaseTimings &timings)
        {
            decode_setup_ns_ += timings.setup_ns;
            decode_execute_ns_ += timings.execute_ns;
            decode_sync_ns_ += timings.sync_ns;
            decode_graph_launch_ns_ += timings.graph_launch_ns;
            decode_post_launch_ns_ += timings.post_launch_ns;
            decode_stream_sync_ns_ += timings.stream_sync_ns;
            decode_setup_workspace_ns_ += timings.setup_workspace_ns;
            decode_setup_token_copy_ns_ += timings.setup_token_copy_ns;
            decode_setup_stream_ns_ += timings.setup_stream_ns;
            decode_setup_dynamic_params_ns_ += timings.setup_dynamic_params_ns;
            decode_setup_graph_reset_ns_ += timings.setup_graph_reset_ns;
            decode_iterations_++;
        }

        /**
         * @brief Record one prefill iteration's timing breakdown
         */
        void recordPrefillIteration(const PhaseTimings &timings)
        {
            prefill_setup_ns_ += timings.setup_ns;
            prefill_execute_ns_ += timings.execute_ns;
            prefill_sync_ns_ += timings.sync_ns;
            prefill_graph_launch_ns_ += timings.graph_launch_ns;
            prefill_post_launch_ns_ += timings.post_launch_ns;
            prefill_stream_sync_ns_ += timings.stream_sync_ns;
            prefill_setup_workspace_ns_ += timings.setup_workspace_ns;
            prefill_setup_token_copy_ns_ += timings.setup_token_copy_ns;
            prefill_setup_stream_ns_ += timings.setup_stream_ns;
            prefill_setup_dynamic_params_ns_ += timings.setup_dynamic_params_ns;
            prefill_setup_graph_reset_ns_ += timings.setup_graph_reset_ns;
            prefill_iterations_++;
        }

        // ----- Reporting -----

        /**
         * @brief Print accumulated summary tables and reset
         *
         * @param device_name Device identifier for table title (e.g., "rocm:0")
         * @return true if there was data to print
         */
        bool printAndReset(const char *device_name = nullptr)
        {
            bool printed = false;

            if (decode_iterations_ > 0)
            {
                printPhaseTable("DECODE", device_name,
                                decode_setup_ns_, decode_execute_ns_, decode_sync_ns_,
                                decode_graph_launch_ns_, decode_post_launch_ns_, decode_stream_sync_ns_,
                                decode_setup_workspace_ns_, decode_setup_token_copy_ns_,
                                decode_setup_stream_ns_, decode_setup_dynamic_params_ns_,
                                decode_setup_graph_reset_ns_,
                                decode_iterations_);
                printed = true;
            }

            if (prefill_iterations_ > 0)
            {
                printPhaseTable("PREFILL", device_name,
                                prefill_setup_ns_, prefill_execute_ns_, prefill_sync_ns_,
                                prefill_graph_launch_ns_, prefill_post_launch_ns_, prefill_stream_sync_ns_,
                                prefill_setup_workspace_ns_, prefill_setup_token_copy_ns_,
                                prefill_setup_stream_ns_, prefill_setup_dynamic_params_ns_,
                                prefill_setup_graph_reset_ns_,
                                prefill_iterations_);
                printed = true;
            }

            reset();
            return printed;
        }

        /**
         * @brief Check if there is accumulated data pending
         */
        bool hasData() const { return decode_iterations_ > 0 || prefill_iterations_ > 0; }

        /**
         * @brief Reset all accumulated data
         */
        void reset()
        {
            decode_setup_ns_ = 0;
            decode_execute_ns_ = 0;
            decode_sync_ns_ = 0;
            decode_graph_launch_ns_ = 0;
            decode_post_launch_ns_ = 0;
            decode_stream_sync_ns_ = 0;
            decode_setup_workspace_ns_ = 0;
            decode_setup_token_copy_ns_ = 0;
            decode_setup_stream_ns_ = 0;
            decode_setup_dynamic_params_ns_ = 0;
            decode_setup_graph_reset_ns_ = 0;
            decode_iterations_ = 0;

            prefill_setup_ns_ = 0;
            prefill_execute_ns_ = 0;
            prefill_sync_ns_ = 0;
            prefill_graph_launch_ns_ = 0;
            prefill_post_launch_ns_ = 0;
            prefill_stream_sync_ns_ = 0;
            prefill_setup_workspace_ns_ = 0;
            prefill_setup_token_copy_ns_ = 0;
            prefill_setup_stream_ns_ = 0;
            prefill_setup_dynamic_params_ns_ = 0;
            prefill_setup_graph_reset_ns_ = 0;
            prefill_iterations_ = 0;
        }

    private:
        /**
         * @brief Render a single phase summary table to stdout
         */
        void printPhaseTable(
            const char *phase_name,
            const char *device_name,
            uint64_t setup_ns,
            uint64_t execute_ns,
            uint64_t sync_ns,
            uint64_t graph_launch_ns,
            uint64_t post_launch_ns,
            uint64_t stream_sync_ns,
            uint64_t setup_workspace_ns,
            uint64_t setup_token_copy_ns,
            uint64_t setup_stream_ns,
            uint64_t setup_dynamic_params_ns,
            uint64_t setup_graph_reset_ns,
            uint64_t iterations)
        {
            // Compute averages in microseconds
            double avg_setup_us = static_cast<double>(setup_ns) / iterations / 1000.0;
            double avg_execute_us = static_cast<double>(execute_ns) / iterations / 1000.0;
            double avg_sync_us = static_cast<double>(sync_ns) / iterations / 1000.0;
            double avg_total_us = avg_setup_us + avg_execute_us + avg_sync_us;

            double avg_launch_us = static_cast<double>(graph_launch_ns) / iterations / 1000.0;
            double avg_post_launch_us = static_cast<double>(post_launch_ns) / iterations / 1000.0;
            double avg_stream_sync_us = static_cast<double>(stream_sync_ns) / iterations / 1000.0;

            // Setup sub-phases
            double avg_workspace_us = static_cast<double>(setup_workspace_ns) / iterations / 1000.0;
            double avg_token_copy_us = static_cast<double>(setup_token_copy_ns) / iterations / 1000.0;
            double avg_stream_setup_us = static_cast<double>(setup_stream_ns) / iterations / 1000.0;
            double avg_dynamic_params_us = static_cast<double>(setup_dynamic_params_ns) / iterations / 1000.0;
            double avg_graph_reset_us = static_cast<double>(setup_graph_reset_ns) / iterations / 1000.0;

            // Total time in ms for the summary line
            double avg_total_ms = avg_total_us / 1000.0;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row
            {
                std::ostringstream title;
                title << "FORWARD PASS WALL-CLOCK: " << phase_name;
                if (device_name)
                    title << " [" << device_name << "]";
                title << " (avg of " << iterations << " iterations)";
                table << title.str() << "" << "" << fort::endr;
                table[0][0].set_cell_span(3);
                table[0][0].set_cell_text_align(fort::text_align::center);
            }

            // Summary line
            {
                std::ostringstream info;
                info << std::fixed << std::setprecision(2);
                info << "Avg Wall: " << avg_total_ms << " ms/iter";
                if (avg_total_ms > 0)
                    info << "  |  Wall-limited: " << std::setprecision(1)
                         << (1000.0 / avg_total_ms) << " tok/s";
                table << info.str() << "" << "" << fort::endr;
                table[1][0].set_cell_span(3);
            }

            // Header
            table << fort::header << "PHASE" << "AVG (μs)" << "%" << fort::endr;

            // Column alignments
            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);

            auto fmt_us = [](double us) -> std::string
            {
                std::ostringstream oss;
                if (us >= 1000.0)
                    oss << std::fixed << std::setprecision(1) << (us / 1000.0) << " ms";
                else
                    oss << std::fixed << std::setprecision(1) << us << " μs";
                return oss.str();
            };

            auto pct = [&](double us) -> std::string
            {
                if (avg_total_us <= 0)
                    return "-";
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(1) << (us / avg_total_us * 100.0) << "%";
                return oss.str();
            };

            // Top-level phases
            table << "Setup (params, stream, reset)" << fmt_us(avg_setup_us) << pct(avg_setup_us) << fort::endr;

            // Setup sub-phases (only if data is available)
            bool has_setup_data = (setup_workspace_ns > 0 || setup_token_copy_ns > 0 ||
                                   setup_stream_ns > 0 || setup_dynamic_params_ns > 0 ||
                                   setup_graph_reset_ns > 0);
            if (has_setup_data)
            {
                double setup_accounted = avg_workspace_us + avg_token_copy_us +
                                         avg_stream_setup_us + avg_dynamic_params_us + avg_graph_reset_us;
                double setup_other = avg_setup_us - setup_accounted;

                if (avg_workspace_us > 0.5)
                    table << "  ├─ Workspace Check" << fmt_us(avg_workspace_us) << pct(avg_workspace_us) << fort::endr;
                if (avg_token_copy_us > 0.5)
                    table << "  ├─ Token/Position Copy" << fmt_us(avg_token_copy_us) << pct(avg_token_copy_us) << fort::endr;
                if (avg_stream_setup_us > 0.5)
                    table << "  ├─ Stream Assignment" << fmt_us(avg_stream_setup_us) << pct(avg_stream_setup_us) << fort::endr;
                if (avg_dynamic_params_us > 0.5)
                    table << "  ├─ Dynamic Params" << fmt_us(avg_dynamic_params_us) << pct(avg_dynamic_params_us) << fort::endr;
                if (avg_graph_reset_us > 0.5)
                    table << "  ├─ Graph Reset" << fmt_us(avg_graph_reset_us) << pct(avg_graph_reset_us) << fort::endr;
                if (setup_other > 1.0)
                    table << "  └─ Other Setup" << fmt_us(setup_other) << pct(setup_other) << fort::endr;
            }

            table << "Execute (graph dispatch)" << fmt_us(avg_execute_us) << pct(avg_execute_us) << fort::endr;

            // Execute sub-phases (only if graph replay data is available)
            bool has_replay_data = (graph_launch_ns > 0 || post_launch_ns > 0 || stream_sync_ns > 0);
            if (has_replay_data)
            {
                // Calculate unaccounted time within execute phase
                double accounted_us = avg_launch_us + avg_post_launch_us + avg_stream_sync_us;
                double other_us = avg_execute_us - accounted_us;

                table << "  ├─ Graph Launch" << fmt_us(avg_launch_us) << pct(avg_launch_us) << fort::endr;
                table << "  ├─ Post-Launch (dirty+cb)" << fmt_us(avg_post_launch_us) << pct(avg_post_launch_us) << fort::endr;
                table << "  ├─ Stream Sync" << fmt_us(avg_stream_sync_us) << pct(avg_stream_sync_us) << fort::endr;
                if (other_us > 1.0)
                {
                    table << "  └─ Other (warmup/capture)" << fmt_us(other_us) << pct(other_us) << fort::endr;
                }
            }

            table << "Sync + Timeline" << fmt_us(avg_sync_us) << pct(avg_sync_us) << fort::endr;

            // Total
            table << fort::separator;
            table << "TOTAL" << fmt_us(avg_total_us) << "100.0%" << fort::endr;

            std::cout << table.to_string() << std::flush;
        }

        // ----- Decode accumulators -----
        uint64_t decode_setup_ns_ = 0;
        uint64_t decode_execute_ns_ = 0;
        uint64_t decode_sync_ns_ = 0;
        uint64_t decode_graph_launch_ns_ = 0;
        uint64_t decode_post_launch_ns_ = 0;
        uint64_t decode_stream_sync_ns_ = 0;
        uint64_t decode_setup_workspace_ns_ = 0;
        uint64_t decode_setup_token_copy_ns_ = 0;
        uint64_t decode_setup_stream_ns_ = 0;
        uint64_t decode_setup_dynamic_params_ns_ = 0;
        uint64_t decode_setup_graph_reset_ns_ = 0;
        uint64_t decode_iterations_ = 0;

        // ----- Prefill accumulators -----
        uint64_t prefill_setup_ns_ = 0;
        uint64_t prefill_execute_ns_ = 0;
        uint64_t prefill_sync_ns_ = 0;
        uint64_t prefill_graph_launch_ns_ = 0;
        uint64_t prefill_post_launch_ns_ = 0;
        uint64_t prefill_stream_sync_ns_ = 0;
        uint64_t prefill_setup_workspace_ns_ = 0;
        uint64_t prefill_setup_token_copy_ns_ = 0;
        uint64_t prefill_setup_stream_ns_ = 0;
        uint64_t prefill_setup_dynamic_params_ns_ = 0;
        uint64_t prefill_setup_graph_reset_ns_ = 0;
        uint64_t prefill_iterations_ = 0;

        /// @brief Thread-local replay timings accessor
        static ReplayPhaseTimings &tls_replay_timings()
        {
            static thread_local ReplayPhaseTimings timings;
            return timings;
        }
    };

} // namespace llaminar2
