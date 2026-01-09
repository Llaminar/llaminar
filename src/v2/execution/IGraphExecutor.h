/**
 * @file IGraphExecutor.h
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

#include "compute_stages/ComputeStages.h"
#include "DeviceContext.h"
#include "../backends/DeviceId.h"
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
     * @brief Execution mode for GraphExecutor
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

    /**
     * @brief Configuration for GraphExecutor
     */
    struct GraphExecutorConfig
    {
        ExecutionMode mode = ExecutionMode::SEQUENTIAL;
        bool enable_profiling = false;             ///< Track per-stage timing
        bool enable_validation = false;            ///< Validate outputs after each stage
        DeviceId default_device = DeviceId::cpu(); ///< Default device for stages

        /// Callback invoked after each stage executes (for snapshot capture)
        StageSnapshotCallback snapshot_callback = nullptr;

        // Context for stage dumping (set by pipeline before each layer)
        int current_layer_idx = -1; ///< Current layer being executed
        int current_iteration = -1; ///< Current decode iteration (-1 for prefill)
        int mpi_rank = 0;           ///< MPI rank for filtering dumps
    };

    // Backwards compatibility alias
    using LayerExecutorConfig = GraphExecutorConfig;

    /**
     * @brief Graph execution statistics
     */
    struct GraphExecutorStats
    {
        size_t total_stages_executed = 0;
        size_t total_flops = 0;
        double total_time_ms = 0.0;
        std::unordered_map<std::string, double> stage_times_ms;

        void reset()
        {
            total_stages_executed = 0;
            total_flops = 0;
            total_time_ms = 0.0;
            stage_times_ms.clear();
        }
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
     * GraphExecutor executor(config);
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
