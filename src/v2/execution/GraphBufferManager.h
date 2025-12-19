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
    class TensorBase;

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
    class GraphBufferManager
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
        bool allocateForGraph(ComputeGraph &graph);

        /**
         * @brief Allocate a single buffer from a descriptor
         *
         * Useful for manual buffer allocation outside of graph context.
         *
         * @param node_name Node/stage name for the buffer key
         * @param desc Buffer descriptor
         * @return true if allocation succeeded
         */
        bool allocateBuffer(const std::string &node_name, const BufferDescriptor &desc);

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
        bool allocateWithAliasing(ComputeGraph &graph);

        /**
         * @brief Release all managed buffers
         */
        void releaseAll();

        // =====================================================================
        // Buffer Retrieval
        // =====================================================================

        /**
         * @brief Get a buffer by node and buffer name
         * @param node_name Node name in the graph
         * @param buffer_name Buffer name within the stage
         * @return Pointer to tensor (nullptr if not found)
         */
        TensorBase *getBuffer(const std::string &node_name, const std::string &buffer_name);

        /**
         * @brief Get a buffer by composite key
         * @param key Buffer key (node_name, buffer_name)
         * @return Pointer to tensor (nullptr if not found)
         */
        TensorBase *getBuffer(const BufferKey &key);

        /**
         * @brief Check if a buffer exists
         * @param node_name Node name
         * @param buffer_name Buffer name
         * @return true if buffer is allocated
         */
        bool hasBuffer(const std::string &node_name, const std::string &buffer_name) const;

        /**
         * @brief Get all buffer keys
         * @return Vector of all allocated buffer keys
         */
        std::vector<BufferKey> getAllBufferKeys() const;

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
                        TensorBase **target_ptr);

        // =====================================================================
        // Statistics
        // =====================================================================

        /**
         * @brief Get allocation statistics
         */
        const BufferAllocationStats &stats() const { return stats_; }

        /**
         * @brief Reset statistics
         */
        void resetStats() { stats_.reset(); }

        /**
         * @brief Get number of allocated buffers
         */
        size_t bufferCount() const { return buffers_.size(); }

        /**
         * @brief Get total allocated bytes
         */
        size_t totalAllocatedBytes() const { return stats_.total_bytes; }

        /**
         * @brief Get memory savings from aliasing
         * @return Percentage (0-100) of SCRATCH memory saved
         */
        double aliasingSavingsPercent() const { return aliasing_savings_percent_; }

        /**
         * @brief Get number of aliasing groups
         */
        size_t aliasingGroupCount() const { return aliasing_groups_.size(); }

        /**
         * @brief Get aliasing groups for inspection
         */
        const std::vector<AliasingGroup> &aliasingGroups() const { return aliasing_groups_; }

        // =====================================================================
        // Debug
        // =====================================================================

        /**
         * @brief Dump buffer inventory to log
         * @param log_level Log level (DEBUG, INFO, etc.)
         */
        void dumpBufferInventory() const;

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

        // Internal helpers
        std::unique_ptr<TensorBase> createTensorFromDescriptor(const BufferDescriptor &desc);
        void updateStats(const BufferDescriptor &desc, size_t allocated_bytes);
        bool allocateAliasingGroups(const std::vector<BufferLiveness> &lifetimes);
    };

} // namespace llaminar2
