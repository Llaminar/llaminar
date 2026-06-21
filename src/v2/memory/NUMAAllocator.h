/**
 * @file NUMAAllocator.h
 * @brief NUMA-aware memory allocation for cross-socket CPU tensor parallelism
 *
 * Provides NUMA-local buffer allocation with first-touch initialization,
 * aligned allocation for SIMD operations, and explicit failure when requested
 * NUMA placement cannot be applied.
 *
 * Key features:
 * - Allocates memory on specific NUMA nodes
 * - First-touch initialization for proper NUMA placement
 * - Aligned allocation for SIMD operations (64-byte cache line)
 * - Fails allocation rather than silently degrading NUMA placement
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace llaminar2
{

    // Forward declaration
    class NodeTopology;

    /**
     * NUMA-aware memory allocation for cross-socket CPU tensor parallelism.
     *
     * Key features:
     * - Allocates memory on specific NUMA nodes
     * - First-touch initialization for proper NUMA placement
     * - Aligned allocation for SIMD operations
     * - Fails allocation rather than silently degrading NUMA placement
     */
    class NUMAAllocator
    {
    public:
        /**
         * Get singleton instance
         */
        static NUMAAllocator &instance();

        /**
         * Allocate memory on a specific NUMA node
         * @param bytes Number of bytes to allocate
         * @param numa_node NUMA node ID (-1 for local node)
         * @param alignment Alignment in bytes (default 64 for cache line)
         * @return Pointer to allocated memory, or nullptr on failure
         */
        void *allocateOnNode(size_t bytes, int numa_node, size_t alignment = 64);

        /**
         * Allocate memory on the local NUMA node (where this thread is running)
         */
        void *allocateLocal(size_t bytes, size_t alignment = 64);

        /**
         * Allocate and initialize with first-touch for NUMA placement
         * @param bytes Number of bytes
         * @param numa_node Target NUMA node
         * @param init_value Value to initialize with (default 0)
         * @return Pointer to allocated and initialized memory
         */
        void *allocateAndTouch(size_t bytes, int numa_node, uint8_t init_value = 0);

        /**
         * Free NUMA-allocated memory
         */
        void free(void *ptr, size_t bytes);

        /**
         * Create a unique_ptr with NUMA allocation and custom deleter
         */
        template <typename T>
        std::unique_ptr<T[], std::function<void(T *)>>
        allocateArray(size_t count, int numa_node)
        {
            if (count == 0)
            {
                return nullptr;
            }

            size_t bytes = count * sizeof(T);
            size_t alignment = alignof(T) < 64 ? 64 : alignof(T);
            T *ptr = static_cast<T *>(allocateOnNode(bytes, numa_node, alignment));
            if (!ptr)
                return nullptr;

            return std::unique_ptr<T[], std::function<void(T *)>>(
                ptr,
                [this, bytes](T *p)
                { this->free(p, bytes); });
        }

        /**
         * Get the NUMA node for the current thread
         * @return NUMA node ID, or 0 if the current CPU cannot be resolved
         */
        int getCurrentNUMANode() const;

        /**
         * Get the NUMA node for a memory address
         * @param ptr Memory address to query
         * @return NUMA node ID, or -1 if cannot be determined
         */
        int getNUMANodeForAddress(const void *ptr) const;

        /**
         * Check if NUMA policy APIs are available at runtime
         */
        bool isNUMAAvailable() const { return numa_available_; }

        /**
         * Get total number of NUMA nodes
         */
        int numNUMANodes() const { return num_numa_nodes_; }

        /**
         * Move memory to a specific NUMA node (expensive, use sparingly)
         * @return true if migration succeeded
         */
        bool migrateToNode(void *ptr, size_t bytes, int numa_node);

        /**
         * Bind current thread to a NUMA node
         * @param numa_node NUMA node to bind to
         * @return true if binding succeeded
         */
        bool bindThreadToNode(int numa_node);

        /**
         * Get memory statistics for a NUMA node
         */
        struct NUMAStats
        {
            size_t total_bytes = 0;
            size_t free_bytes = 0;
            size_t allocated_by_us = 0;
        };
        NUMAStats getNodeStats(int numa_node) const;

    private:
        NUMAAllocator();
        ~NUMAAllocator();

        // Prevent copying
        NUMAAllocator(const NUMAAllocator &) = delete;
        NUMAAllocator &operator=(const NUMAAllocator &) = delete;

        bool numa_available_ = false;
        int num_numa_nodes_ = 1;

        // Track allocations per node for stats
        mutable std::mutex stats_mutex_;
        std::vector<size_t> allocated_per_node_;
        struct AllocationRecord
        {
            int node = -1;
            size_t bytes = 0;
        };
        std::unordered_map<void *, AllocationRecord> allocation_records_;

        // Internal helpers
        void initializeNUMA();
        int resolveNUMANode(int numa_node) const;
    };

    /**
     * RAII wrapper for NUMA-local buffer
     */
    template <typename T>
    class NUMABuffer
    {
    public:
        NUMABuffer() = default;

        NUMABuffer(size_t count, int numa_node = -1)
            : count_(count), numa_node_(numa_node)
        {
            if (count > 0)
            {
                data_ = NUMAAllocator::instance().allocateArray<T>(count, numa_node);
                // Resolve actual NUMA node if -1 was passed
                if (numa_node_ == -1)
                {
                    numa_node_ = NUMAAllocator::instance().getCurrentNUMANode();
                }
            }
        }

        // Move constructor
        NUMABuffer(NUMABuffer &&other) noexcept
            : data_(std::move(other.data_)), count_(other.count_), numa_node_(other.numa_node_)
        {
            other.count_ = 0;
            other.numa_node_ = -1;
        }

        // Move assignment
        NUMABuffer &operator=(NUMABuffer &&other) noexcept
        {
            if (this != &other)
            {
                data_ = std::move(other.data_);
                count_ = other.count_;
                numa_node_ = other.numa_node_;
                other.count_ = 0;
                other.numa_node_ = -1;
            }
            return *this;
        }

        // Prevent copying
        NUMABuffer(const NUMABuffer &) = delete;
        NUMABuffer &operator=(const NUMABuffer &) = delete;

        T *data() { return data_.get(); }
        const T *data() const { return data_.get(); }

        size_t size() const { return count_; }
        int numaNode() const { return numa_node_; }

        bool valid() const { return data_ != nullptr && count_ > 0; }

        T &operator[](size_t i) { return data_[i]; }
        const T &operator[](size_t i) const { return data_[i]; }

        // Iterator support
        T *begin() { return data_.get(); }
        T *end() { return data_.get() + count_; }
        const T *begin() const { return data_.get(); }
        const T *end() const { return data_.get() + count_; }

    private:
        std::unique_ptr<T[], std::function<void(T *)>> data_;
        size_t count_ = 0;
        int numa_node_ = -1;
    };

} // namespace llaminar2
