/**
 * @file NUMAAllocator.cpp
 * @brief NUMA-aware memory allocation implementation
 *
 * Uses libnuma and explicit mbind() so requested NUMA placement either
 * succeeds or allocation fails.
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include "NUMAAllocator.h"
#include "../utils/Logger.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <thread>

#include <numa.h>
#include <numaif.h>

#include <omp.h>

namespace llaminar2
{

    namespace
    {
        // Page size for first-touch initialization
        constexpr size_t PAGE_SIZE = 4096;

        // Minimum alignment for cache line efficiency
        constexpr size_t MIN_ALIGNMENT = 64;

        /**
         * Round up to alignment boundary
         */
        size_t alignUp(size_t value, size_t alignment)
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        /**
         * Check if a value is a power of two
         */
        bool isPowerOfTwo(size_t value)
        {
            return value > 0 && (value & (value - 1)) == 0;
        }
    } // anonymous namespace

    // ============================================================================
    // Singleton Implementation
    // ============================================================================

    NUMAAllocator &NUMAAllocator::instance()
    {
        static NUMAAllocator instance;
        return instance;
    }

    NUMAAllocator::NUMAAllocator()
    {
        initializeNUMA();
    }

    NUMAAllocator::~NUMAAllocator()
    {
        // Nothing to clean up - all memory should be freed by callers
    }

    void NUMAAllocator::initializeNUMA()
    {
        if (numa_available() < 0)
        {
            LOG_ERROR("NUMAAllocator: libnuma is present but NUMA policy APIs are unavailable");
            numa_available_ = false;
            num_numa_nodes_ = 1;
            allocated_per_node_.resize(1, 0);
            return;
        }

        numa_available_ = true;
        num_numa_nodes_ = numa_num_configured_nodes();

        if (num_numa_nodes_ < 1)
        {
            num_numa_nodes_ = 1;
        }

        allocated_per_node_.resize(num_numa_nodes_, 0);

        LOG_DEBUG("NUMAAllocator: NUMA available with " << num_numa_nodes_ << " node(s)");

        // Log memory per node
        for (int i = 0; i < num_numa_nodes_; ++i)
        {
            long long node_size = numa_node_size64(i, nullptr);
            if (node_size > 0)
            {
                LOG_DEBUG("  Node " << i << ": " << (node_size / (1024 * 1024 * 1024)) << " GB");
            }
        }
    }

    // ============================================================================
    // Allocation Methods
    // ============================================================================

    int NUMAAllocator::resolveNUMANode(int numa_node) const
    {
        if (numa_node == -1)
        {
            return getCurrentNUMANode();
        }

        if (!numa_available_)
        {
            LOG_ERROR("NUMAAllocator: NUMA node " << numa_node
                                                  << " requested but NUMA policy APIs are unavailable");
            return -1;
        }

        if (numa_node < 0 || numa_node >= num_numa_nodes_)
        {
            LOG_ERROR("NUMAAllocator: Invalid NUMA node " << numa_node
                                                          << " (valid range: 0-" << (num_numa_nodes_ - 1) << ")");
            return -1;
        }

        return numa_node;
    }

    void *NUMAAllocator::allocateOnNode(size_t bytes, int numa_node, size_t alignment)
    {
        // Handle zero-byte allocation
        if (bytes == 0)
        {
            return nullptr;
        }

        // Validate and fix alignment
        if (!isPowerOfTwo(alignment))
        {
            LOG_WARN("NUMAAllocator: Alignment " << alignment << " is not power of 2, using 64");
            alignment = MIN_ALIGNMENT;
        }
        if (alignment < MIN_ALIGNMENT)
        {
            alignment = MIN_ALIGNMENT;
        }

        // Resolve NUMA node (-1 means local)
        int resolved_node = resolveNUMANode(numa_node);
        if (resolved_node < 0)
        {
            return nullptr;
        }

        size_t allocation_alignment = std::max(alignment, PAGE_SIZE);
        size_t aligned_bytes = alignUp(bytes, allocation_alignment);
        void *ptr = std::aligned_alloc(allocation_alignment, aligned_bytes);
        if (!ptr)
        {
            LOG_ERROR("NUMAAllocator: aligned_alloc failed for " << bytes << " bytes");
            return nullptr;
        }

        struct bitmask *nodemask = numa_allocate_nodemask();
        if (!nodemask)
        {
            LOG_ERROR("NUMAAllocator: Failed to allocate NUMA nodemask for node " << resolved_node);
            std::free(ptr);
            return nullptr;
        }

        numa_bitmask_clearall(nodemask);
        numa_bitmask_setbit(nodemask, resolved_node);

        errno = 0;
        int rc = mbind(ptr, aligned_bytes, MPOL_BIND, nodemask->maskp, nodemask->size, 0);
        int bind_errno = errno;
        numa_free_nodemask(nodemask);

        if (rc != 0)
        {
            LOG_ERROR("NUMAAllocator: mbind failed for " << aligned_bytes
                                                         << " bytes on node " << resolved_node
                                                         << ": " << std::strerror(bind_errno));
            std::free(ptr);
            return nullptr;
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            allocated_per_node_[resolved_node] += bytes;
            allocation_records_[ptr] = AllocationRecord{resolved_node, bytes};
        }

        LOG_TRACE("NUMAAllocator: Allocated " << bytes << " bytes on node " << resolved_node);

        return ptr;
    }

    void *NUMAAllocator::allocateLocal(size_t bytes, size_t alignment)
    {
        if (bytes == 0)
        {
            return nullptr;
        }

        return allocateOnNode(bytes, getCurrentNUMANode(), alignment);
    }

    void *NUMAAllocator::allocateAndTouch(size_t bytes, int numa_node, uint8_t init_value)
    {
        if (bytes == 0)
        {
            return nullptr;
        }

        void *ptr = allocateOnNode(bytes, numa_node, MIN_ALIGNMENT);
        if (!ptr)
        {
            return nullptr;
        }

        // First-touch initialization with OpenMP parallel loop
        // This ensures pages are faulted on the correct NUMA node
        uint8_t *data = static_cast<uint8_t *>(ptr);

        // Parallel first-touch: each thread touches pages in its portion.
        // This works best when threads are already bound to the target NUMA node.
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < bytes; i += PAGE_SIZE)
        {
            // Touch first byte of each page
            size_t touch_offset = std::min(i, bytes - 1);
            data[touch_offset] = init_value;
        }

        // If init_value is non-zero, fill the rest
        if (init_value != 0)
        {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < bytes; ++i)
            {
                data[i] = init_value;
            }
        }
        else
        {
            // For zero init, we already touched pages; memset the rest
            // This is faster than byte-by-byte
            std::memset(ptr, 0, bytes);
        }

        LOG_TRACE("NUMAAllocator: Allocated and touched " << bytes << " bytes on node " << numa_node);
        return ptr;
    }

    void NUMAAllocator::free(void *ptr, size_t bytes)
    {
        if (!ptr || bytes == 0)
        {
            return;
        }

        int node = -1;
        size_t recorded_bytes = bytes;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            auto it = allocation_records_.find(ptr);
            if (it != allocation_records_.end())
            {
                node = it->second.node;
                recorded_bytes = it->second.bytes;
                if (node >= 0 && node < static_cast<int>(allocated_per_node_.size()) &&
                    allocated_per_node_[node] >= recorded_bytes)
                {
                    allocated_per_node_[node] -= recorded_bytes;
                }
                allocation_records_.erase(it);
            }
        }

        std::free(ptr);

        LOG_TRACE("NUMAAllocator: Freed " << recorded_bytes << " bytes from node " << node);
    }

    // ============================================================================
    // Query Methods
    // ============================================================================

    int NUMAAllocator::getCurrentNUMANode() const
    {
        if (numa_available_)
        {
            // Get the NUMA node of the current CPU
            int cpu = sched_getcpu();
            if (cpu >= 0)
            {
                int node = numa_node_of_cpu(cpu);
                if (node >= 0)
                {
                    return node;
                }
            }
            // Fallback: preferred node
            return numa_preferred();
        }
        return 0;
    }

    int NUMAAllocator::getNUMANodeForAddress(const void *ptr) const
    {
        if (!ptr)
        {
            return -1;
        }

        if (numa_available_)
        {
            int node = -1;
            // Use move_pages with NULL destination to query current node
            void *pages[] = {const_cast<void *>(ptr)};
            int status[1] = {-1};

            if (move_pages(0, 1, pages, nullptr, status, 0) == 0)
            {
                node = status[0];
                if (node < 0)
                {
                    // Negative values are errors (e.g., page not mapped)
                    return -1;
                }
                return node;
            }
        }

        return -1; // Cannot determine
    }

    bool NUMAAllocator::migrateToNode(void *ptr, size_t bytes, int numa_node)
    {
        if (!ptr || bytes == 0)
        {
            return false;
        }

        if (!numa_available_)
        {
            LOG_ERROR("NUMAAllocator: Cannot migrate pages because NUMA policy APIs are unavailable");
            return false;
        }

        int resolved_node = resolveNUMANode(numa_node);
        if (resolved_node < 0)
        {
            return false;
        }

        // Create a nodemask for the target node
        struct bitmask *nodemask = numa_allocate_nodemask();
        if (!nodemask)
        {
            LOG_ERROR("NUMAAllocator: Failed to allocate nodemask for migration");
            return false;
        }

        numa_bitmask_clearall(nodemask);
        numa_bitmask_setbit(nodemask, resolved_node);

        // Migrate pages - this is expensive!
        errno = 0;
        int result = mbind(ptr, bytes, MPOL_BIND, nodemask->maskp,
                           nodemask->size, MPOL_MF_MOVE | MPOL_MF_STRICT);
        int bind_errno = errno;

        numa_free_nodemask(nodemask);

        if (result != 0)
        {
            LOG_ERROR("NUMAAllocator: Page migration to node " << resolved_node
                                                               << " failed: " << strerror(bind_errno));
            return false;
        }

        LOG_DEBUG("NUMAAllocator: Migrated " << bytes << " bytes to node " << resolved_node);
        return true;
    }

    bool NUMAAllocator::bindThreadToNode(int numa_node)
    {
        if (!numa_available_)
        {
            LOG_ERROR("NUMAAllocator: Cannot bind thread because NUMA policy APIs are unavailable");
            return false;
        }

        int resolved_node = resolveNUMANode(numa_node);
        if (resolved_node < 0)
        {
            return false;
        }

        // Bind this thread to run only on CPUs of the target NUMA node
        int result = numa_run_on_node(resolved_node);

        if (result != 0)
        {
            LOG_ERROR("NUMAAllocator: Failed to bind thread to node " << resolved_node);
            return false;
        }

        // Also set memory policy for this thread
        numa_set_preferred(resolved_node);

        LOG_DEBUG("NUMAAllocator: Bound thread to NUMA node " << resolved_node);
        return true;
    }

    NUMAAllocator::NUMAStats NUMAAllocator::getNodeStats(int numa_node) const
    {
        NUMAStats stats;

        int resolved_node = resolveNUMANode(numa_node);
        if (resolved_node < 0)
        {
            return stats;
        }

        if (numa_available_)
        {
            long long free_bytes = 0;
            long long total_bytes = numa_node_size64(resolved_node, &free_bytes);

            if (total_bytes > 0)
            {
                stats.total_bytes = static_cast<size_t>(total_bytes);
                stats.free_bytes = static_cast<size_t>(free_bytes);
            }
        }

        // Add our tracked allocation
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (resolved_node >= 0 && resolved_node < static_cast<int>(allocated_per_node_.size()))
            {
                stats.allocated_by_us = allocated_per_node_[resolved_node];
            }
        }

        return stats;
    }

} // namespace llaminar2
