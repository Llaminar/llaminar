/**
 * @file IGraphBufferManager.h
 * @brief Interface for graph buffer management operations
 *
 * Abstracts buffer allocation and retrieval to enable:
 * 1. Unit testing without actual tensor allocation
 * 2. Mock buffer management for stage testing
 * 3. Testing buffer aliasing logic in isolation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../execution/BufferRole.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class ComputeGraph;
    class TensorBase;
    struct BufferAllocationStats;
    struct BufferKey;
    struct AliasingGroup;
    class CollectiveContext;

    /**
     * @brief Abstract interface for graph buffer management operations
     *
     * This interface abstracts buffer allocation and lifecycle management to enable:
     * - Unit testing of graph execution without actual memory allocation
     * - Mocking buffer availability for stage testing
     * - Testing aliasing algorithms in isolation
     *
     * Implementations:
     * - GraphBufferManager: Real buffer manager with tensor allocation
     * - MockGraphBufferManager: Test implementation with configurable behavior
     */
    class IGraphBufferManager
    {
    public:
        virtual ~IGraphBufferManager() = default;

        // =========================================================================
        // Collective Context (Phase 3: Buffer Registration API)
        // =========================================================================

        /**
         * @brief Set the collective context for BAR-aware allocation
         *
         * When a CollectiveContext is set and it has a backend requiring
         * buffer registration (e.g., PCIeBARBackend), buffers marked with
         * `participates_in_collective=true` will be allocated from the BAR
         * region for cross-vendor P2P transfers.
         *
         * @param ctx Shared pointer to CollectiveContext (may be nullptr to disable)
         */
        virtual void setCollectiveContext(std::shared_ptr<CollectiveContext> ctx) = 0;

        /**
         * @brief Get the current collective context
         * @return Shared pointer to CollectiveContext (may be nullptr)
         */
        virtual std::shared_ptr<CollectiveContext> collectiveContext() const = 0;

        // =========================================================================
        // Allocation
        // =========================================================================

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
        virtual bool allocateForGraph(ComputeGraph &graph) = 0;

        /**
         * @brief Allocate a single buffer from a descriptor
         *
         * Useful for manual buffer allocation outside of graph context.
         *
         * @param node_name Node/stage name for the buffer key
         * @param desc Buffer descriptor
         * @return true if allocation succeeded
         */
        virtual bool allocateBuffer(const std::string &node_name, const BufferDescriptor &desc) = 0;

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
        virtual bool allocateWithAliasing(ComputeGraph &graph) = 0;

        /**
         * @brief Release all managed buffers
         */
        virtual void releaseAll() = 0;

        // =========================================================================
        // Buffer Retrieval
        // =========================================================================

        /**
         * @brief Get a buffer by node and buffer name
         * @param node_name Node name in the graph
         * @param buffer_name Buffer name within the stage
         * @return Pointer to tensor (nullptr if not found)
         */
        virtual TensorBase *getBuffer(const std::string &node_name, const std::string &buffer_name) = 0;

        /**
         * @brief Get a buffer by composite key
         * @param key Buffer key (node_name, buffer_name)
         * @return Pointer to tensor (nullptr if not found)
         */
        virtual TensorBase *getBuffer(const BufferKey &key) = 0;

        /**
         * @brief Check if a buffer exists
         * @param node_name Node name
         * @param buffer_name Buffer name
         * @return true if buffer is allocated
         */
        virtual bool hasBuffer(const std::string &node_name, const std::string &buffer_name) const = 0;

        /**
         * @brief Get all buffer keys
         * @return Vector of all allocated buffer keys
         */
        virtual std::vector<BufferKey> getAllBufferKeys() const = 0;

        // =========================================================================
        // Binding
        // =========================================================================

        /**
         * @brief Bind a buffer to a stage's params
         *
         * This is a hook for future integration where GraphExecutor
         * automatically binds buffers to stage params before execution.
         *
         * @param node_name Node name
         * @param buffer_name Buffer name
         * @param target_ptr Pointer to the Params field to bind
         * @return true if binding succeeded
         */
        virtual bool bindBuffer(const std::string &node_name,
                                const std::string &buffer_name,
                                TensorBase **target_ptr) = 0;

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Get allocation statistics
         */
        virtual const BufferAllocationStats &stats() const = 0;

        /**
         * @brief Reset statistics
         */
        virtual void resetStats() = 0;

        /**
         * @brief Get number of allocated buffers
         */
        virtual size_t bufferCount() const = 0;

        /**
         * @brief Get total allocated bytes
         */
        virtual size_t totalAllocatedBytes() const = 0;

        /**
         * @brief Get memory savings from aliasing
         * @return Percentage (0-100) of SCRATCH memory saved
         */
        virtual double aliasingSavingsPercent() const = 0;

        /**
         * @brief Get number of aliasing groups
         */
        virtual size_t aliasingGroupCount() const = 0;

        /**
         * @brief Get aliasing groups for inspection
         */
        virtual const std::vector<AliasingGroup> &aliasingGroups() const = 0;

        // =========================================================================
        // Debug
        // =========================================================================

        /**
         * @brief Dump buffer inventory to log
         */
        virtual void dumpBufferInventory() const = 0;
    };

} // namespace llaminar2
