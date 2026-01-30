/**
 * @file HeterogeneousLayerExecutor.h
 * @brief Layer executor for heterogeneous device execution
 *
 * Executes transformer layers on their assigned devices/domains based on
 * LayerPlacementConfig. Supports GPU and CPU execution with automatic
 * activation transfer at device boundaries.
 *
 * Use cases:
 *   - CPU offload: First/last layers on CPU, middle on GPU
 *   - Memory-constrained: Large layers offloaded to CPU
 *   - Hybrid compute: Mix CPU and GPU for optimal utilization
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../../config/LayerPlacementConfig.h"
#include "../../../config/TPDomain.h"
#include "../../../backends/DeviceId.h"
#include <memory>
#include <chrono>

namespace llaminar2
{

    // Forward declarations
    class ComputeGraph;
    class ICollectiveContext;
    class GraphBufferManager;
    class IDeviceContext;

    /**
     * @brief Executes transformer layers on heterogeneous devices based on LayerPlacementConfig.
     *
     * For each layer:
     * 1. Query LayerPlacementConfig for device assignment
     * 2. Ensure activations are on the correct device
     * 3. Execute layer stages on assigned device
     * 4. Handle cross-domain transfers at boundaries
     *
     * Thread safety: Not thread-safe. Designed for single-threaded pipeline execution.
     *
     * Usage:
     * @code
     * // Configure for first 4 layers on CPU, rest on GPU
     * auto placement = LayerPlacementConfig::cpuFirstLayers(4, 28, DeviceId::cuda(0));
     *
     * HeterogeneousLayerExecutor::Config config;
     * config.placement_config = &placement;
     * config.buffer_manager = buffer_mgr.get();
     *
     * HeterogeneousLayerExecutor executor(config);
     *
     * // Execute layers with automatic device routing
     * for (int i = 0; i < 28; ++i) {
     *     executor.executeLayer(i, &layer_graphs[i]);
     * }
     * @endcode
     */
    class HeterogeneousLayerExecutor
    {
    public:
        /**
         * @brief Configuration for HeterogeneousLayerExecutor
         */
        struct Config
        {
            LayerPlacementConfig *placement_config = nullptr; ///< Layer-to-device mapping (required)
            MultiDomainTPConfig *tp_config = nullptr;         ///< Multi-domain TP config (optional)
            ICollectiveContext *collective_ctx = nullptr;     ///< Collective context for MPI ops (optional)
            GraphBufferManager *buffer_manager = nullptr;     ///< Buffer manager for allocation (optional)
            IDeviceContext *cpu_context = nullptr;            ///< CPU device context (optional)
            IDeviceContext *gpu_context = nullptr;            ///< GPU device context (optional)
            bool enable_profiling = false;                    ///< Track per-layer timing
        };

        /**
         * @brief Construct with configuration
         * @param config Executor configuration
         * @throws std::invalid_argument if placement_config is nullptr
         */
        explicit HeterogeneousLayerExecutor(Config config);

        /**
         * @brief Destructor
         */
        ~HeterogeneousLayerExecutor();

        // Non-copyable
        HeterogeneousLayerExecutor(const HeterogeneousLayerExecutor &) = delete;
        HeterogeneousLayerExecutor &operator=(const HeterogeneousLayerExecutor &) = delete;

        // Movable
        HeterogeneousLayerExecutor(HeterogeneousLayerExecutor &&) noexcept;
        HeterogeneousLayerExecutor &operator=(HeterogeneousLayerExecutor &&) noexcept;

        // =========================================================================
        // Layer Execution
        // =========================================================================

        /**
         * @brief Execute a single transformer layer on its assigned device
         *
         * Queries placement_config to determine the device, then dispatches
         * to executeOnGPU() or executeOnCPU() accordingly.
         *
         * @param layer_idx Layer index (0-based)
         * @param graph Compute graph containing layer stages
         * @return true on success, false on error
         */
        bool executeLayer(int layer_idx, ComputeGraph *graph);

        /**
         * @brief Execute a range of transformer layers with automatic device routing
         *
         * Handles cross-domain transfers at device boundaries automatically.
         *
         * @param start_layer First layer index (inclusive)
         * @param end_layer Last layer index (exclusive)
         * @param graph Compute graph containing all layer stages
         * @return true if all layers executed successfully
         */
        bool executeLayerRange(int start_layer, int end_layer, ComputeGraph *graph);

        // =========================================================================
        // Device/Domain Queries
        // =========================================================================

        /**
         * @brief Get the device assigned to a specific layer
         * @param layer_idx Layer index
         * @return DeviceId for the layer
         * @throws std::out_of_range if layer_idx is invalid
         */
        DeviceId getDeviceForLayer(int layer_idx) const;

        /**
         * @brief Get the TP domain for a specific layer
         *
         * Returns the tensor parallel domain for collective operations.
         *
         * @param layer_idx Layer index
         * @return Pointer to TPDomain, or nullptr if no TP configured
         */
        const TPDomain *getDomainForLayer(int layer_idx) const;

        /**
         * @brief Check if adjacent layers require cross-domain transfer
         *
         * Returns true if from_layer and to_layer are on different devices,
         * requiring explicit data transfer between them.
         *
         * @param from_layer Source layer index
         * @param to_layer Destination layer index
         * @return true if cross-domain transfer is needed
         */
        bool requiresCrossDomainTransfer(int from_layer, int to_layer) const;

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Execution statistics
         */
        struct ExecutionStats
        {
            int gpu_layers_executed = 0;    ///< Number of layers executed on GPU
            int cpu_layers_executed = 0;    ///< Number of layers executed on CPU
            int cross_domain_transfers = 0; ///< Number of cross-device transfers
            double gpu_time_ms = 0.0;       ///< Total GPU execution time (ms)
            double cpu_time_ms = 0.0;       ///< Total CPU execution time (ms)
            double transfer_time_ms = 0.0;  ///< Total transfer time (ms)

            /**
             * @brief Get total execution time
             * @return Total time in milliseconds
             */
            double totalTimeMs() const
            {
                return gpu_time_ms + cpu_time_ms + transfer_time_ms;
            }

            /**
             * @brief Get percentage of time spent on GPU
             * @return GPU time percentage (0-100)
             */
            double gpuTimePercent() const
            {
                double total = totalTimeMs();
                return total > 0.0 ? (gpu_time_ms / total) * 100.0 : 0.0;
            }

            /**
             * @brief Get percentage of time spent on CPU
             * @return CPU time percentage (0-100)
             */
            double cpuTimePercent() const
            {
                double total = totalTimeMs();
                return total > 0.0 ? (cpu_time_ms / total) * 100.0 : 0.0;
            }

            /**
             * @brief Get percentage of time spent on transfers
             * @return Transfer time percentage (0-100)
             */
            double transferTimePercent() const
            {
                double total = totalTimeMs();
                return total > 0.0 ? (transfer_time_ms / total) * 100.0 : 0.0;
            }
        };

        /**
         * @brief Get execution statistics
         * @return Current execution statistics
         */
        ExecutionStats getStats() const { return stats_; }

        /**
         * @brief Reset execution statistics
         */
        void resetStats() { stats_ = ExecutionStats{}; }

        /**
         * @brief Get configuration
         * @return Current configuration
         */
        const Config &config() const { return config_; }

    private:
        /**
         * @brief Execute a layer on GPU
         * @param layer_idx Layer index
         * @param graph Compute graph
         * @return true on success
         */
        bool executeOnGPU(int layer_idx, ComputeGraph *graph);

        /**
         * @brief Execute a layer on CPU
         * @param layer_idx Layer index
         * @param graph Compute graph
         * @return true on success
         */
        bool executeOnCPU(int layer_idx, ComputeGraph *graph);

        /**
         * @brief Transfer activations between devices at layer boundary
         *
         * Copies activation tensors from from_layer's device to to_layer's device.
         * This is a placeholder implementation for Phase 5.1; full implementation
         * will come in Phase 5.3.
         *
         * @param from_layer Source layer index
         * @param to_layer Destination layer index
         * @param graph Compute graph containing activation buffers
         * @return true on success
         */
        bool transferActivations(int from_layer, int to_layer, ComputeGraph *graph);

        /**
         * @brief Execute layer stages on a specific device context
         *
         * Internal helper that runs the actual stage execution.
         *
         * @param layer_idx Layer index
         * @param graph Compute graph
         * @param ctx Device context to use
         * @return true on success
         */
        bool executeLayerStages(int layer_idx, ComputeGraph *graph, IDeviceContext *ctx);

        Config config_;
        ExecutionStats stats_;

        // High-resolution clock for timing
        using Clock = std::chrono::high_resolution_clock;
    };

} // namespace llaminar2
