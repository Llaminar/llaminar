/**
 * @file WorkspaceAllocator.h
 * @brief Standalone workspace allocation for compute graphs
 *
 * Extracted from DeviceGraphBufferManager to decouple workspace management
 * from buffer lifecycle management. Provides model-aware GPU/CPU workspace
 * allocation with per-device budget enforcement.
 *
 * @author David Sanftenberg
 * @date March 2026
 */

#pragma once

#include "DeviceWorkspaceManager.h"
#include "WorkspaceDescriptor.h"
#include "../../../backends/DeviceId.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class ComputeGraph;
    class IComputeStage;
    class IWorkspaceConsumer;
    class IBackend;

    /**
     * @brief Configuration for workspace memory budget calculation
     *
     * Controls how WorkspaceAllocator computes workspace budgets for GPU and CPU
     * devices. The budget is calculated as:
     *   budget = min(max(available * fraction - headroom, min_budget), max_budget)
     */
    struct WorkspaceBudgetConfig
    {
        float gpu_fraction = 0.8f;                     ///< Fraction of free GPU memory to use (0.0-1.0)
        float cpu_fraction = 0.3f;                     ///< Fraction of free CPU memory to use (conservative)
        size_t min_budget = 64 * 1024 * 1024;          ///< Minimum budget (64MB)
        size_t max_budget = 4ULL * 1024 * 1024 * 1024; ///< Maximum budget (4GB)
        size_t headroom = 128 * 1024 * 1024;           ///< Reserved headroom (128MB)
    };

    /**
     * @brief Model-aware sizing hints for workspace consumers
     *
     * Used to derive per-stage workspace dimensions from model metadata.
     */
    struct WorkspaceSizingHints
    {
        int max_seq_len = 4096;
        int n_heads = 0;
        int head_dim = 0;
        int d_model = 0;
        int batch_size = 1;
        int vocab_size = 0;
    };

    /**
     * @brief Explicit workspace request for non-graph consumers
     */
    struct WorkspaceConsumerRequest
    {
        IWorkspaceConsumer *consumer = nullptr;
        DeviceId device;
        int m = 4096;
        int n = 0;
        int k = 0;
    };

    /**
     * @brief Standalone workspace allocator for compute graphs
     *
     * Manages per-device workspace allocation with model-aware budget calculation.
     * Each device gets a single DeviceWorkspaceManager with a computed budget.
     *
     * ## Usage
     *
     * ```cpp
     * WorkspaceAllocator allocator;
     *
     * WorkspaceSizingHints hints;
     * hints.max_seq_len = 4096;
     * hints.n_heads = 32;
     * // ...
     *
     * allocator.allocateForGraph(graph, hints);
     * ```
     *
     * ## Thread Safety
     *
     * NOT thread-safe. Should be used from a single thread.
     */
    class WorkspaceAllocator
    {
    public:
        WorkspaceAllocator() = default;
        ~WorkspaceAllocator() = default;

        // Non-copyable
        WorkspaceAllocator(const WorkspaceAllocator &) = delete;
        WorkspaceAllocator &operator=(const WorkspaceAllocator &) = delete;

        // Movable
        WorkspaceAllocator(WorkspaceAllocator &&) = default;
        WorkspaceAllocator &operator=(WorkspaceAllocator &&) = default;

        // =====================================================================
        // Memory Query
        // =====================================================================

        /**
         * @brief Query available memory for a device
         *
         * Uses the appropriate backend (CPU, CUDA, ROCm) to query free memory.
         *
         * @param device Target device
         * @return Available bytes
         */
        size_t queryAvailableMemory(DeviceId device);

        /**
         * @brief Compute workspace budget for a device
         *
         * Applies budget configuration to compute an appropriate workspace size.
         *
         * @param device Target device
         * @param config Budget configuration
         * @return Computed budget in bytes
         */
        size_t computeWorkspaceBudget(DeviceId device,
                                      const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        // =====================================================================
        // Allocation
        // =====================================================================

        /**
         * @brief Allocate workspace for all GPU consumers in a graph
         *
         * Scans graph stages for IWorkspaceConsumer implementations, derives
         * per-stage dimension hints, allocates per-device workspace, and binds
         * all consumers. Optional extra consumers (e.g., KV cache) can be provided.
         *
         * @param graph The compute graph
         * @param hints Model-aware sizing hints
         * @param extra_consumers Additional workspace consumers outside the graph
         * @param config Budget configuration
         * @return true if all allocations succeeded
         */
        bool allocateForGraph(
            const ComputeGraph &graph,
            const WorkspaceSizingHints &hints,
            const std::vector<WorkspaceConsumerRequest> &extra_consumers = {},
            const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        /**
         * @brief Allocate workspace for a flat list of stages
         *
         * @param stages Stages to scan for IWorkspaceConsumer
         * @param config Budget configuration
         * @return true if all allocations succeeded
         */
        bool allocateForStages(const std::vector<IComputeStage *> &stages,
                               const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        /**
         * @brief Release all workspace allocations
         */
        void releaseAll();

        // =====================================================================
        // Access
        // =====================================================================

        /**
         * @brief Get workspace manager for a device
         * @param device Target device
         * @return Workspace manager (nullptr if not allocated)
         */
        DeviceWorkspaceManager *getDeviceWorkspace(DeviceId device);

        // =====================================================================
        // Metrics
        // =====================================================================

        /**
         * @brief Total workspace allocated across all devices
         */
        size_t totalAllocated() const;

        /**
         * @brief Workspace allocated for a specific device
         */
        size_t deviceAllocated(DeviceId device) const;

        /**
         * @brief Number of devices with workspace allocated
         */
        size_t deviceCount() const { return device_workspaces_.size(); }

    private:
        /// Per-device workspace managers
        std::unordered_map<DeviceId, std::unique_ptr<DeviceWorkspaceManager>> device_workspaces_;

        /// Per-device workspace budgets (for metrics)
        std::unordered_map<DeviceId, size_t> device_workspace_budgets_;

        /**
         * @brief Compute model-aware minimum budget floor
         */
        size_t computeModelAwareBudgetFloor(const WorkspaceSizingHints &hints) const;
    };

} // namespace llaminar2
