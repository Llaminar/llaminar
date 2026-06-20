/**
 * @file IDeviceGraphExecutor.h
 * @brief Interface for compute graph execution orchestration
 * @author David Sanftenberg
 * @date December 2025
 *
 * Defines the abstract interface for executing compute graphs.
 * This interface enables:
 * - Mocking of the executor for testing higher-level components
 * - Alternative executor implementations (e.g., profiling executor)
 * - Plugin-based executor extensions
 *
 * Note: This was previously named ILayerExecutor but renamed to better
 * reflect its purpose - it executes graphs, not layers specifically.
 */

#pragma once

#include "../../compute_stages/ComputeStages.h"
#include "../device/DeviceContext.h"
#include "../../../backends/DeviceId.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class ComputeGraph;
    struct StageDumpInfo;

    /**
     * @brief Execution mode for DeviceGraphExecutor
     */
    enum class ExecutionMode
    {
        SEQUENTIAL, ///< Execute stages one at a time (debugging)
        PARALLEL,   ///< Execute independent stages in parallel
        PIPELINED   ///< Overlap stages where possible (future)
    };

    /**
     * @brief Callback type for capturing snapshots after stage execution
     *
     * Called after each stage completes successfully. The StageDumpInfo contains
     * all input/output/weight buffers from the stage. For simple snapshot capture,
     * use dump_info.outputs[0] to get the primary output.
     *
     * @param node_name Name of the compute node (e.g., "layer0_q_proj")
     * @param dump_info Full dump info from the stage (inputs, outputs, weights, scalars)
     */
    using StageSnapshotCallback = std::function<void(
        const std::string &node_name,
        const StageDumpInfo &dump_info)>;

    using StageFailureCallback = std::function<void(
        const std::string &node_name,
        const std::string &reason)>;

    using ExecutionCancellationCallback = std::function<bool()>;

    /**
     * @brief Configuration for DeviceGraphExecutor
     */
    struct GraphExecutorConfig
    {
        ExecutionMode mode = ExecutionMode::SEQUENTIAL;
        bool enable_profiling = false;             ///< Track per-stage timing
        bool enable_validation = false;            ///< Validate outputs after each stage
        DeviceId default_device = DeviceId::cpu(); ///< Default device for stages

        /// Callback invoked after each stage executes (for snapshot capture)
        StageSnapshotCallback snapshot_callback = nullptr;

        /// Callback invoked immediately when a stage fails. TP runners use this
        /// to abort shared collective contexts before sibling workers enter the
        /// next collective.
        StageFailureCallback stage_failure_callback = nullptr;

        /// Callback queried before each stage. Return true to cancel this graph
        /// pass because a peer worker or rank already failed.
        ExecutionCancellationCallback cancellation_requested = nullptr;

        // Context for stage dumping (set by pipeline before each layer)
        int current_layer_idx = -1; ///< Current layer being executed
        int current_iteration = -1; ///< Current decode iteration (-1 for prefill)
        int mpi_rank = 0;           ///< MPI rank for filtering dumps
    };

    // Backwards compatibility alias
    using LayerExecutorConfig = GraphExecutorConfig;

    /**
     * @brief Overhead breakdown for stage execution
     *
     * Tracks time spent in various non-kernel activities during stage execution.
     * These overheads are cumulative across all stages executed.
     */
    struct ExecutionOverhead
    {
        double input_cohere_ms = 0.0;  ///< Time cohering inputs to target device
        double weight_cohere_ms = 0.0; ///< Time cohering weights to target device
        double output_alloc_ms = 0.0;  ///< Time allocating output buffers on device
        double mark_dirty_ms = 0.0;    ///< Time marking outputs as device-dirty
        double dump_input_ms = 0.0;    ///< Time dumping input tensors (stage dump)
        double dump_output_ms = 0.0;   ///< Time dumping output tensors (stage dump)
        double verify_ms = 0.0;        ///< Time in buffer verification
        double callback_ms = 0.0;      ///< Time in snapshot callbacks
        double get_dump_info_ms = 0.0; ///< Time calling getDumpInfo()

        /// Total overhead (sum of all categories)
        double total() const
        {
            return input_cohere_ms + weight_cohere_ms + output_alloc_ms +
                   mark_dirty_ms + dump_input_ms + dump_output_ms +
                   verify_ms + callback_ms + get_dump_info_ms;
        }

        void reset()
        {
            input_cohere_ms = 0.0;
            weight_cohere_ms = 0.0;
            output_alloc_ms = 0.0;
            mark_dirty_ms = 0.0;
            dump_input_ms = 0.0;
            dump_output_ms = 0.0;
            verify_ms = 0.0;
            callback_ms = 0.0;
            get_dump_info_ms = 0.0;
        }

        ExecutionOverhead &operator+=(const ExecutionOverhead &other)
        {
            input_cohere_ms += other.input_cohere_ms;
            weight_cohere_ms += other.weight_cohere_ms;
            output_alloc_ms += other.output_alloc_ms;
            mark_dirty_ms += other.mark_dirty_ms;
            dump_input_ms += other.dump_input_ms;
            dump_output_ms += other.dump_output_ms;
            verify_ms += other.verify_ms;
            callback_ms += other.callback_ms;
            get_dump_info_ms += other.get_dump_info_ms;
            return *this;
        }
    };

    /**
     * @brief Execution phase for phase-split profiling
     */
    enum class ExecutionPhase : uint8_t
    {
        COMBINED = 0, ///< No phase tracking (legacy)
        PREFILL = 1,
        DECODE = 2
    };

    /**
     * @brief Per-phase stage execution statistics
     *
     * Lightweight struct that tracks stage-type breakdown for a single phase
     * (prefill or decode). The combined/global stats remain in GraphExecutorStats.
     */
    struct PhaseStats
    {
        size_t total_stages_executed = 0;
        double total_execute_ms = 0.0;
        double total_collective_ms = 0.0;
        size_t total_collective_calls = 0;
        std::unordered_map<std::string, double> stage_type_execute_ms;
        std::unordered_map<std::string, size_t> stage_type_counts;
        ExecutionOverhead overhead;

        void reset()
        {
            total_stages_executed = 0;
            total_execute_ms = 0.0;
            total_collective_ms = 0.0;
            total_collective_calls = 0;
            stage_type_execute_ms.clear();
            stage_type_counts.clear();
            overhead.reset();
        }
    };

    /**
     * @brief Graph execution statistics
     */
    struct GraphExecutorStats
    {
        size_t total_stages_executed = 0;
        size_t total_flops = 0;
        double total_time_ms = 0.0;
        double total_execute_ms = 0.0;     ///< Time in actual kernel/stage execution (compute + collective)
        double total_collective_ms = 0.0;  ///< Time in collective operations (allreduce, allgather) — subset of total_execute_ms
        size_t total_collective_calls = 0; ///< Number of collective operations executed
        std::unordered_map<std::string, double> stage_times_ms;
        std::unordered_map<std::string, double> stage_type_execute_ms;
        std::unordered_map<std::string, size_t> stage_type_counts;

        /// Accumulated overhead breakdown
        ExecutionOverhead overhead;

        /// Phase-split stats (prefill vs decode)
        PhaseStats prefill;
        PhaseStats decode;

        /// Thread-local current phase (set by BenchmarkRunner before prefill/decode)
        static void setCurrentPhase(ExecutionPhase phase) { current_phase_ = phase; }
        static ExecutionPhase currentPhase() { return current_phase_; }

        void reset()
        {
            total_stages_executed = 0;
            total_flops = 0;
            total_time_ms = 0.0;
            total_execute_ms = 0.0;
            total_collective_ms = 0.0;
            total_collective_calls = 0;
            stage_times_ms.clear();
            stage_type_execute_ms.clear();
            stage_type_counts.clear();
            overhead.reset();
            prefill.reset();
            decode.reset();
        }

        /// Print a formatted profiling summary to stdout
        void printProfilingSummary(size_t prefill_tokens, size_t decode_tokens) const;

        /// Export host-side executor stage/profile attribution into PerfStatsCollector.
        void recordPerfStats(const std::string &device_name = {}) const;

    private:
        static thread_local ExecutionPhase current_phase_;

        /// Print a single phase table
        void printPhaseTable(const std::string &title, const PhaseStats &phase, size_t tokens) const;
    };

    // Backwards compatibility alias
    using LayerExecutorStats = GraphExecutorStats;

    /**
     * @brief Abstract interface for compute graph execution
     *
     * IGraphExecutor defines the contract for components that execute
     * compute graphs. This enables mocking and alternative implementations.
     *
     * Usage:
     * @code
     * // Use through interface for testability
     * void processGraph(IGraphExecutor& executor, ComputeGraph& graph, IDeviceContext* ctx) {
     *     executor.execute(graph, ctx);
     * }
     *
     * // In production
     * DeviceGraphExecutor executor(config);
     * processGraph(executor, graph, ctx);
     *
     * // In tests
     * MockGraphExecutor mock_executor;
     * processGraph(mock_executor, graph, ctx);
     * @endcode
     */
    class IGraphExecutor
    {
    public:
        virtual ~IGraphExecutor() = default;

        // =========================================================================
        // Configuration
        // =========================================================================

        /**
         * @brief Get current configuration
         */
        virtual const GraphExecutorConfig &config() const = 0;

        /**
         * @brief Set execution mode
         */
        virtual void setExecutionMode(ExecutionMode mode) = 0;

        /**
         * @brief Get current execution mode
         */
        virtual ExecutionMode executionMode() const = 0;

        /**
         * @brief Enable/disable profiling
         */
        virtual void setProfilingEnabled(bool enabled) = 0;

        /**
         * @brief Enable/disable output validation
         */
        virtual void setValidationEnabled(bool enabled) = 0;

        /**
         * @brief Set snapshot callback for debugging
         *
         * When set, this callback is invoked after each compute stage executes,
         * allowing capture of intermediate tensors for comparison.
         *
         * @param callback Function called with (node_name, snapshot_info) after each stage
         */
        virtual void setSnapshotCallback(StageSnapshotCallback callback) = 0;

        /**
         * @brief Set the current layer context for stage dumping
         * @param layer_idx Layer index (0-based, -1 for non-layer operations)
         */
        virtual void setCurrentLayerIdx(int layer_idx) = 0;

        /**
         * @brief Set the current iteration context for stage dumping
         * @param iteration Decode iteration (-1 for prefill)
         */
        virtual void setCurrentIteration(int iteration) = 0;

        /**
         * @brief Set the MPI rank for stage dumping
         * @param rank MPI rank (0 for single-process)
         */
        virtual void setMPIRank(int rank) = 0;

        // =========================================================================
        // Execution
        // =========================================================================

        /**
         * @brief Execute a compute graph
         * @param graph The compute graph to execute
         * @param ctx Device context for execution
         * @return true on success
         */
        virtual bool execute(ComputeGraph &graph, IDeviceContext *ctx) = 0;

        /**
         * @brief Execute a compute graph with multi-device support
         * @param graph The compute graph to execute
         * @param contexts Map of DeviceId -> DeviceContext
         * @return true on success
         */
        virtual bool executeMultiDevice(
            ComputeGraph &graph,
            const std::unordered_map<DeviceId, IDeviceContext *> &contexts) = 0;

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Get execution statistics
         */
        virtual const GraphExecutorStats &stats() const = 0;

        /**
         * @brief Reset statistics
         */
        virtual void resetStats() = 0;
    };

    // Backwards compatibility alias
    using ILayerExecutor = IGraphExecutor;

    /**
     * @brief Get human-readable name for execution mode
     */
    const char *executionModeName(ExecutionMode mode);

} // namespace llaminar2
