/**
 * @file NUMAAllocator.cpp
 * @brief NUMA-aware memory allocation implementation
 *
 * Uses libnuma if available, with fallback to aligned_alloc for systems
 * without NUMA support or single-socket machines.
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include "NUMAAllocator.h"
#include "../utils/Logger.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <thread>

#ifdef HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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
#ifdef HAVE_NUMA
        // Check if NUMA is available on this system
        if (numa_available() < 0)
        {
            LOG_INFO("NUMAAllocator: libnuma present but NUMA not available on this system");
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

        LOG_INFO("NUMAAllocator: NUMA available with " << num_numa_nodes_ << " node(s)");

        // Log memory per node
        for (int i = 0; i < num_numa_nodes_; ++i)
        {
            long long node_size = numa_node_size64(i, nullptr);
            if (node_size > 0)
            {
                LOG_DEBUG("  Node " << i << ": " << (node_size / (1024 * 1024 * 1024)) << " GB");
            }
        }
#else
        LOG_INFO("NUMAAllocator: libnuma not available, using standard allocation");
        numa_available_ = false;
        num_numa_nodes_ = 1;
        allocated_per_node_.resize(1, 0);
#endif
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

        // Clamp to valid range
        if (numa_node < 0 || numa_node >= num_numa_nodes_)
        {
            LOG_WARN("NUMAAllocator: Invalid NUMA node " << numa_node
                                                         << ", using node 0 (valid range: 0-" << (num_numa_nodes_ - 1) << ")");
            return 0;
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

        void *ptr = nullptr;

#ifdef HAVE_NUMA
        if (numa_available_)
        {
            // Use numa_alloc_onnode for NUMA-aware allocation
            // Note: numa_alloc_onnode returns page-aligned memory (4KB aligned)
            // If we need larger alignment, we need to handle it ourselves

            if (alignment <= PAGE_SIZE)
            {
                ptr = numa_alloc_onnode(bytes, resolved_node);
            }
            else
            {
                // For larger alignments, allocate extra and align manually
                // This is rare (alignment > 4KB)
                size_t padded_bytes = bytes + alignment;
                void *raw = numa_alloc_onnode(padded_bytes, resolved_node);
                if (raw)
                {
                    // Align the pointer
                    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
                    uintptr_t aligned_addr = alignUp(addr, alignment);
                    ptr = reinterpret_cast<void *>(aligned_addr);

                    // Store original pointer for freeing (in a real implementation,
                    // we'd need a map to track this - for now, this is simplified)
                    // TODO: Track original pointers for custom-aligned allocations
                    LOG_WARN("NUMAAllocator: Large alignment requested (" << alignment
                                                                          << "), using simplified allocation (may leak memory on free)");
                }
            }

            if (ptr)
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                allocated_per_node_[resolved_node] += bytes;
                LOG_TRACE("NUMAAllocator: Allocated " << bytes << " bytes on node " << resolved_node);
            }
            else
            {
                LOG_ERROR("NUMAAllocator: numa_alloc_onnode failed for " << bytes << " bytes on node " << resolved_node);
                // Fall back to standard allocation
                ptr = fallbackAlloc(bytes, alignment);
            }
        }
        else
        {
            ptr = fallbackAlloc(bytes, alignment);
        }
#else
        ptr = fallbackAlloc(bytes, alignment);
#endif

        return ptr;
    }

    void *NUMAAllocator::allocateLocal(size_t bytes, size_t alignment)
    {
        if (bytes == 0)
        {
            return nullptr;
        }

#ifdef HAVE_NUMA
        if (numa_available_)
        {
            // Use numa_alloc_local which allocates on the local node
            if (!isPowerOfTwo(alignment))
            {
                alignment = MIN_ALIGNMENT;
            }
            if (alignment < MIN_ALIGNMENT)
            {
                alignment = MIN_ALIGNMENT;
            }

            void *ptr = nullptr;
            if (alignment <= PAGE_SIZE)
            {
                ptr = numa_alloc_local(bytes);
            }
            else
            {
                // Fallback for large alignments
                ptr = fallbackAlloc(bytes, alignment);
            }

            if (ptr)
            {
                int local_node = getCurrentNUMANode();
                std::lock_guard<std::mutex> lock(stats_mutex_);
                allocated_per_node_[local_node] += bytes;
                LOG_TRACE("NUMAAllocator: Allocated " << bytes << " bytes locally (node " << local_node << ")");
            }

            return ptr;
        }
#endif

        return fallbackAlloc(bytes, alignment);
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

#ifdef _OPENMP
// Parallel first-touch: each thread touches pages in its portion
// This works best when threads are already bound to the target NUMA node
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
#else
        // Single-threaded fallback
        std::memset(ptr, init_value, bytes);
#endif

        LOG_TRACE("NUMAAllocator: Allocated and touched " << bytes << " bytes on node " << numa_node);
        return ptr;
    }

    void NUMAAllocator::free(void *ptr, size_t bytes)
    {
        if (!ptr || bytes == 0)
        {
            return;
        }

#ifdef HAVE_NUMA
        if (numa_available_)
        {
            // Get the NUMA node this memory was on (for stats tracking)
            int node = getNUMANodeForAddress(ptr);

            numa_free(ptr, bytes);

            if (node >= 0 && node < num_numa_nodes_)
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                if (allocated_per_node_[node] >= bytes)
                {
                    allocated_per_node_[node] -= bytes;
                }
            }

            LOG_TRACE("NUMAAllocator: Freed " << bytes << " bytes from node " << node);
            return;
        }
#endif

        fallbackFree(ptr, bytes);
    }

    // ============================================================================
    // Query Methods
    // ============================================================================

    int NUMAAllocator::getCurrentNUMANode() const
    {
#ifdef HAVE_NUMA
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
#endif
        return 0;
    }

    int NUMAAllocator::getNUMANodeForAddress(const void *ptr) const
    {
        if (!ptr)
        {
            return -1;
        }

#ifdef HAVE_NUMA
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
#endif

        return -1; // Cannot determine
    }

    bool NUMAAllocator::migrateToNode(void *ptr, size_t bytes, int numa_node)
    {
        if (!ptr || bytes == 0)
        {
            return false;
        }

#ifdef HAVE_NUMA
        if (numa_available_)
        {
            int resolved_node = resolveNUMANode(numa_node);

            // Create a nodemask for the target node
            struct bitmask *nodemask = numa_allocate_nodemask();
            if (!nodemask)
            {
                LOG_ERROR("NUMAAllocator: Failed to allocate nodemask for migration");
                return false;
            }

            numa_bitmask_setbit(nodemask, resolved_node);

            // Migrate pages - this is expensive!
            int result = mbind(ptr, bytes, MPOL_BIND, nodemask->maskp,
                               nodemask->size + 1, MPOL_MF_MOVE | MPOL_MF_STRICT);

            numa_free_nodemask(nodemask);

            if (result != 0)
            {
                LOG_WARN("NUMAAllocator: Page migration to node " << resolved_node
                                                                  << " failed: " << strerror(errno));
                return false;
            }

            LOG_DEBUG("NUMAAllocator: Migrated " << bytes << " bytes to node " << resolved_node);
            return true;
        }
#endif

        return false; // Migration not available
    }

    bool NUMAAllocator::bindThreadToNode(int numa_node)
    {
#ifdef HAVE_NUMA
        if (numa_available_)
        {
            int resolved_node = resolveNUMANode(numa_node);

            // Bind this thread to run only on CPUs of the target NUMA node
            int result = numa_run_on_node(resolved_node);

            if (result != 0)
            {
                LOG_WARN("NUMAAllocator: Failed to bind thread to node " << resolved_node);
                return false;
            }

            // Also set memory policy for this thread
            numa_set_preferred(resolved_node);

            LOG_DEBUG("NUMAAllocator: Bound thread to NUMA node " << resolved_node);
            return true;
        }
#endif

        return false; // Binding not available
    }

    NUMAAllocator::NUMAStats NUMAAllocator::getNodeStats(int numa_node) const
    {
        NUMAStats stats;

        int resolved_node = resolveNUMANode(numa_node);

#ifdef HAVE_NUMA
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
#endif

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

    // ============================================================================
    // Fallback Allocation (when NUMA unavailable)
    // ============================================================================

    void *NUMAAllocator::fallbackAlloc(size_t bytes, size_t alignment)
    {
        if (bytes == 0)
        {
            return nullptr;
        }

        // Ensure alignment is valid for aligned_alloc
        if (!isPowerOfTwo(alignment))
        {
            alignment = MIN_ALIGNMENT;
        }
        if (alignment < sizeof(void *))
        {
            alignment = sizeof(void *);
        }

        // aligned_alloc requires size to be multiple of alignment
        size_t aligned_bytes = alignUp(bytes, alignment);

        void *ptr = std::aligned_alloc(alignment, aligned_bytes);

        if (ptr)
        {
            // Track in node 0 for stats
            std::lock_guard<std::mutex> lock(stats_mutex_);
            allocated_per_node_[0] += bytes;
            LOG_TRACE("NUMAAllocator: Fallback allocated " << bytes << " bytes (aligned to " << alignment << ")");
        }
        else
        {
            LOG_ERROR("NUMAAllocator: Fallback aligned_alloc failed for " << bytes << " bytes");
        }

        return ptr;
    }

    void NUMAAllocator::fallbackFree(void *ptr, size_t bytes)
    {
        if (!ptr)
        {
            return;
        }

        std::free(ptr);

        // Track deallocation
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (allocated_per_node_[0] >= bytes)
            {
                allocated_per_node_[0] -= bytes;
            }
        }

        LOG_TRACE("NUMAAllocator: Fallback freed " << bytes << " bytes");
    }

} // namespace llaminar2
