/**
 * @file GraphBufferManager.h
 * @brief Centralized buffer management for GraphExecutor
 * @author David Sanftenberg
 * @date December 2025
 *
 * GraphBufferManager handles pre-allocation and lifecycle management of all
 * buffers used by compute stages within a graph. This enables:
 * - Zero-allocation hot paths during inference
 * - Memory planning based on stage requirements
 * - SCRATCH buffer aliasing via liveness analysis
 *
 * ## Design Philosophy
 *
 * Stages declare their buffer requirements via getBufferRequirements().
 * GraphBufferManager collects these requirements, allocates all buffers
 * upfront, then binds them to stages before execution.
 *
 * ## Buffer Aliasing (Phase 4+)
 *
 * SCRATCH buffers with non-overlapping lifetimes can share memory:
 * @code
 * // Allocate with aliasing optimization
 * manager.allocateWithAliasing(graph);  // Uses LivenessAnalyzer
 *
 * // Check memory savings
 * LOG_INFO("Saved " << manager.aliasingSavingsPercent() << "% memory");
 * @endcode
 *
 * ## Typical Usage
 *
 * @code
 * // Create manager with tensor factory
 * GraphBufferManager manager(&tensor_factory, &mpi_ctx);
 *
 * // Allocate all buffers for a graph (with aliasing optimization)
 * manager.allocateWithAliasing(graph);
 *
 * // Retrieve a specific buffer
 * auto* output = manager.getBuffer("attention", "output");
 *
 * // Execute graph (buffers already allocated)
 * executor.execute(graph, ctx);
 *
 * // Release when done
 * manager.releaseAll();
 * @endcode
 *
 * @see BufferRole for buffer classification
 * @see StageBufferRequirements for stage declarations
 * @see LivenessAnalyzer for aliasing analysis
 */

#pragma once

#include "BufferRole.h"
#include "LivenessAnalyzer.h"
#include "CollectiveContext.h"
#include "DeviceWorkspaceManager.h"
#include "../interfaces/IGraphBufferManager.h"
#include "../backends/DeviceId.h"
#include "../tensors/TensorFactory.h"
#include "../utils/MPIContext.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class ComputeGraph;
    class CPUTensorBase;
    class PCIeBARBackend;
    class IComputeStage;
    class IWorkspaceConsumer;
    class IBackend;
    using TensorBase = CPUTensorBase; // Backward compatibility alias

    // =========================================================================
    // Workspace Budget Configuration (Phase 4: Memory Budget Enforcement)
    // =========================================================================

    /**
     * @brief Configuration for workspace memory budget calculation
     *
     * Controls how GraphBufferManager computes workspace budgets for GPU and CPU
     * devices. The budget is calculated as:
     *   budget = min(max(available * fraction - headroom, min_budget), max_budget)
     *
     * @see GraphBufferManager::computeWorkspaceBudget()
     */
    struct WorkspaceBudgetConfig
    {
        float gpu_fraction = 0.8f;                     ///< Fraction of free GPU memory to use (0.0-1.0)
        float cpu_fraction = 0.3f;                     ///< Fraction of free CPU memory to use (conservative)
        size_t min_budget = 64 * 1024 * 1024;          ///< Minimum budget (64MB)
        size_t max_budget = 4ULL * 1024 * 1024 * 1024; ///< Maximum budget (4GB)
        size_t headroom = 128 * 1024 * 1024;           ///< Reserved headroom (128MB)
    };

    // =========================================================================
    // Buffer Allocation Statistics
    // =========================================================================

    /**
     * @brief Statistics about buffer allocations
     */
    struct BufferAllocationStats
    {
        size_t total_buffers = 0;           ///< Number of buffers allocated
        size_t total_bytes = 0;             ///< Total bytes allocated
        size_t input_bytes = 0;             ///< Bytes for INPUT buffers
        size_t output_bytes = 0;            ///< Bytes for OUTPUT buffers
        size_t inout_bytes = 0;             ///< Bytes for INOUT buffers
        size_t scratch_bytes = 0;           ///< Bytes for SCRATCH buffers
        size_t weight_bytes = 0;            ///< Bytes for WEIGHT buffers (rarely managed here)
        size_t scratch_buffers_aliased = 0; ///< SCRATCH buffers sharing memory (future)

        /**
         * @brief Reset statistics
         */
        void reset()
        {
            total_buffers = 0;
            total_bytes = 0;
            input_bytes = 0;
            output_bytes = 0;
            inout_bytes = 0;
            scratch_bytes = 0;
            weight_bytes = 0;
            scratch_buffers_aliased = 0;
        }
    };

    // =========================================================================
    // Buffer Key
    // =========================================================================

    /**
     * @brief Composite key for buffer lookup
     *
     * Buffers are identified by (node_name, buffer_name) pairs.
     */
    struct BufferKey
    {
        std::string node_name;   ///< Graph node name (e.g., "layer0_attention")
        std::string buffer_name; ///< Buffer name within stage (e.g., "output")

        bool operator==(const BufferKey &other) const
        {
            return node_name == other.node_name && buffer_name == other.buffer_name;
        }
    };

    /**
     * @brief Hash function for BufferKey
     */
    struct BufferKeyHash
    {
        size_t operator()(const BufferKey &key) const
        {
            // Simple hash combining node and buffer names
            size_t h1 = std::hash<std::string>{}(key.node_name);
            size_t h2 = std::hash<std::string>{}(key.buffer_name);
            return h1 ^ (h2 << 1);
        }
    };

    // =========================================================================
    // Graph Buffer Manager
    // =========================================================================

    /**
     * @brief Centralized buffer manager for compute graphs
     *
     * Collects buffer requirements from all stages in a graph, allocates
     * tensors via TensorFactory, and tracks buffers for retrieval.
     *
     * ## Thread Safety
     *
     * NOT thread-safe. Should be used from a single thread or with
     * external synchronization.
     *
     * ## Lifecycle
     *
     * 1. Create manager with factory
     * 2. Call allocateForGraph() before execution
     * 3. Use getBuffer() to retrieve buffers
     * 4. Call releaseAll() or let destructor clean up
     *
     * ## Future: Liveness Analysis
     *
     * Phase 4 will add liveness analysis for SCRATCH buffer aliasing.
     * Multiple non-overlapping SCRATCH buffers can share the same
     * physical allocation to reduce memory usage.
     */
    class GraphBufferManager : public IGraphBufferManager
    {
    public:
        /**
         * @brief Construct buffer manager
         * @param factory TensorFactory for allocation (not owned)
         * @param mpi_ctx MPI context for NUMA awareness (not owned)
         */
        explicit GraphBufferManager(TensorFactory *factory, const MPIContext *mpi_ctx = nullptr);

        ~GraphBufferManager();

        // Non-copyable
        GraphBufferManager(const GraphBufferManager &) = delete;
        GraphBufferManager &operator=(const GraphBufferManager &) = delete;

        // Movable
        GraphBufferManager(GraphBufferManager &&) = default;
        GraphBufferManager &operator=(GraphBufferManager &&) = default;

        // =====================================================================
        // Collective Context (Phase 3: Buffer Registration API)
        // =====================================================================

        /**
         * @brief Set the collective context for BAR-aware allocation
         *
         * When set, buffers marked with `participates_in_collective=true`
         * will be allocated from the BAR region if the context has a
         * PCIeBARBackend and the buffer targets a ROCm device.
         *
         * @param ctx Shared pointer to CollectiveContext (may be nullptr)
         */
        void setCollectiveContext(std::shared_ptr<CollectiveContext> ctx) override;

        /**
         * @brief Get the current collective context
         * @return Shared pointer to CollectiveContext (may be nullptr)
         */
        std::shared_ptr<CollectiveContext> collectiveContext() const override
        {
            return collective_ctx_;
        }

        // =====================================================================
        // Allocation
        // =====================================================================

        /**
         * @brief Allocate all buffers for a compute graph
         *
         * Collects requirements from all stages via getBufferRequirements(),
         * then allocates tensors for OUTPUT, INOUT, and SCRATCH buffers.
         *
         * INPUT and WEIGHT buffers are NOT allocated (they come from external
         * sources like previous stages or model weights).
         *
         * @param graph The compute graph to allocate for
         * @return true if all allocations succeeded
         */
        bool allocateForGraph(ComputeGraph &graph) override;

        /**
         * @brief Allocate a single buffer from a descriptor
         *
         * Useful for manual buffer allocation outside of graph context.
         *
         * @param node_name Node/stage name for the buffer key
         * @param desc Buffer descriptor
         * @return true if allocation succeeded
         */
        bool allocateBuffer(const std::string &node_name, const BufferDescriptor &desc) override;

        /**
         * @brief Allocate all buffers with aliasing optimization
         *
         * Uses LivenessAnalyzer to find SCRATCH buffers with non-overlapping
         * lifetimes and allocates them to shared physical memory.
         *
         * This is the recommended allocation method for graphs where memory
         * efficiency matters. Falls back to allocateForGraph() if analysis fails.
         *
         * @param graph The compute graph to allocate for
         * @return true if all allocations succeeded
         */
        bool allocateWithAliasing(ComputeGraph &graph) override;

        /**
         * @brief Release all managed buffers
         */
        void releaseAll() override;

        // =====================================================================
        // Buffer Retrieval
        // =====================================================================

        /**
         * @brief Get a buffer by node and buffer name
         * @param node_name Node name in the graph
         * @param buffer_name Buffer name within the stage
         * @return Pointer to tensor (nullptr if not found)
         */
        TensorBase *getBuffer(const std::string &node_name, const std::string &buffer_name) override;

        /**
         * @brief Get a buffer by composite key
         * @param key Buffer key (node_name, buffer_name)
         * @return Pointer to tensor (nullptr if not found)
         */
        TensorBase *getBuffer(const BufferKey &key) override;

        /**
         * @brief Check if a buffer exists
         * @param node_name Node name
         * @param buffer_name Buffer name
         * @return true if buffer is allocated
         */
        bool hasBuffer(const std::string &node_name, const std::string &buffer_name) const override;

        /**
         * @brief Get all buffer keys
         * @return Vector of all allocated buffer keys
         */
        std::vector<BufferKey> getAllBufferKeys() const override;

        // =====================================================================
        // Binding (for future stage integration)
        // =====================================================================

        /**
         * @brief Bind a buffer to a stage's params
         *
         * This is a hook for future integration where GraphExecutor
         * automatically binds buffers to stage params before execution.
         *
         * Currently a no-op placeholder.
         *
         * @param node_name Node name
         * @param buffer_name Buffer name
         * @param target_ptr Pointer to the Params field to bind
         * @return true if binding succeeded
         */
        bool bindBuffer(const std::string &node_name,
                        const std::string &buffer_name,
                        TensorBase **target_ptr) override;

        // =====================================================================
        // GPU Workspace Management (Phase 4: Memory Budget Enforcement)
        // =====================================================================

        /**
         * @brief Query available memory for a device
         *
         * Uses the appropriate backend (CPU, CUDA, or ROCm) to query the
         * device's free memory. For CPU, this queries the local NUMA node.
         *
         * @param device Target device (CPU, CUDA, ROCm)
         * @return Available bytes (after weights loaded, before workspace)
         */
        size_t queryAvailableMemory(DeviceId device);

        /**
         * @brief Compute workspace budget for a device
         *
         * Applies the budget configuration to compute an appropriate workspace
         * allocation size. The budget calculation considers:
         * - Device-specific fraction (GPU vs CPU)
         * - Headroom reservation
         * - Min/max clamps
         *
         * @param device Target device
         * @param config Budget configuration (defaults to reasonable values)
         * @return Computed budget in bytes
         */
        size_t computeWorkspaceBudget(DeviceId device,
                                      const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        /**
         * @brief Get workspace manager for a device
         *
         * Returns the workspace manager allocated via allocateDeviceWorkspace().
         *
         * @param device Target device
         * @return Workspace manager (nullptr if not allocated)
         */
        DeviceWorkspaceManager *getDeviceWorkspace(DeviceId device);

        /**
         * @brief Allocate GPU workspace for all devices used by stages
         *
         * Workflow:
         * 1. Enumerate unique devices from stages that implement IWorkspaceConsumer
         * 2. Query available memory per device
         * 3. Compute budget per device using config
         * 4. Collect workspace requirements from all consumers on each device
         * 5. Allocate workspace per device
         * 6. Bind workspace to consuming stages
         *
         * @param stages Stages that may need workspace (filtered for IWorkspaceConsumer)
         * @param config Budget configuration
         * @return true if all allocations succeeded
         */
        bool allocateDeviceWorkspace(const std::vector<IComputeStage *> &stages,
                                     const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        /**
         * @brief Release all GPU workspace
         *
         * Releases all workspace managers and clears budget tracking.
         * Workspace consumers are NOT unbound (they will fall back to legacy mode).
         */
        void releaseDeviceWorkspace();

        /**
         * @brief Get total GPU workspace allocated across all devices
         * @return Sum of used bytes from all workspace managers
         */
        size_t totalDeviceWorkspaceAllocated() const;

        /**
         * @brief Get GPU workspace allocated for a specific device
         * @param device Target device
         * @return Used bytes for that device (0 if not allocated)
         */
        size_t deviceWorkspaceAllocated(DeviceId device) const;

        // =====================================================================
        // Statistics
        // =====================================================================

        /**
         * @brief Get allocation statistics
         */
        const BufferAllocationStats &stats() const override { return stats_; }

        /**
         * @brief Reset statistics
         */
        void resetStats() override { stats_.reset(); }

        /**
         * @brief Get number of allocated buffers
         */
        size_t bufferCount() const override { return buffers_.size(); }

        /**
         * @brief Get total allocated bytes
         */
        size_t totalAllocatedBytes() const override { return stats_.total_bytes; }

        /**
         * @brief Get memory savings from aliasing
         * @return Percentage (0-100) of SCRATCH memory saved
         */
        double aliasingSavingsPercent() const override { return aliasing_savings_percent_; }

        /**
         * @brief Get number of aliasing groups
         */
        size_t aliasingGroupCount() const override { return aliasing_groups_.size(); }

        /**
         * @brief Get aliasing groups for inspection
         */
        const std::vector<AliasingGroup> &aliasingGroups() const override { return aliasing_groups_; }

        // =====================================================================
        // Debug
        // =====================================================================

        /**
         * @brief Dump buffer inventory to log
         * @param log_level Log level (DEBUG, INFO, etc.)
         */
        void dumpBufferInventory() const override;

    private:
        TensorFactory *factory_;      ///< Tensor factory (not owned)
        const MPIContext *mpi_ctx_;   ///< MPI context (not owned)
        BufferAllocationStats stats_; ///< Allocation statistics

        /// Buffer storage: key -> owned tensor
        std::unordered_map<BufferKey, std::unique_ptr<TensorBase>, BufferKeyHash> buffers_;

        /// Buffer descriptors for metadata lookup
        std::unordered_map<BufferKey, BufferDescriptor, BufferKeyHash> descriptors_;

        /// Aliasing analysis results (Phase 4+)
        std::vector<AliasingGroup> aliasing_groups_;
        double aliasing_savings_percent_ = 0.0;

        /// Aliased buffer storage: group_idx -> physical tensor
        std::unordered_map<size_t, std::unique_ptr<TensorBase>> aliased_buffers_;

        /// Map from logical buffer key to aliasing group index
        std::unordered_map<BufferKey, size_t, BufferKeyHash> buffer_to_group_;

        /// Collective context for BAR-aware allocation (Phase 3)
        std::shared_ptr<CollectiveContext> collective_ctx_;

        /// GPU workspace managers per device (Phase 4: Memory Budget Enforcement)
        std::unordered_map<DeviceId, std::unique_ptr<DeviceWorkspaceManager>> device_workspaces_;

        /// GPU workspace budgets per device (for metrics)
        std::unordered_map<DeviceId, size_t> device_workspace_budgets_;

        // Internal helpers
        std::unique_ptr<TensorBase> createTensorFromDescriptor(const BufferDescriptor &desc);
        void updateStats(const BufferDescriptor &desc, size_t allocated_bytes);
        bool allocateAliasingGroups(const std::vector<BufferLiveness> &lifetimes);

        /**
         * @brief Check if a buffer should use BAR allocation
         *
         * Returns true if:
         * 1. Buffer has participates_in_collective = true
         * 2. Buffer targets a ROCm device
         * 3. A CollectiveContext is set with a PCIeBARBackend
         *
         * @param desc Buffer descriptor
         * @return true if BAR allocation should be used
         */
        bool shouldUseBarAllocation(const BufferDescriptor &desc) const;

        /**
         * @brief Allocate a buffer from the BAR region
         *
         * Uses PCIeBARBackend::allocateInBarRegion() to get BAR memory,
         * then creates a tensor wrapping the external memory.
         *
         * @param node_name Node name for the buffer
         * @param desc Buffer descriptor
         * @return Allocated tensor, or nullptr on failure
         */
        std::unique_ptr<TensorBase> allocateFromBarRegion(
            const std::string &node_name,
            const BufferDescriptor &desc);
    };

} // namespace llaminar2
