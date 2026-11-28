/**
 * @file NUMAAllocator.cpp
 * @brief NUMA-aware memory allocation implementation
 *
 * Uses an exact thread-placement scope, page revocation, first touch, and
 * placement certification so requested NUMA placement either succeeds or fails
 * closed. Binding the memory policy during the fault prevents Linux from
 * choosing a remote free node instead of reclaiming local file cache.
 * Batched page-table observations are resolved through a fault-capable lookup
 * when Linux reports a migration/non-present entry; unknown placement is never
 * treated as the requested node. Certification does not move application bytes.
 * The placement owner is armed before native writes; restoration failure is
 * fatal, and allocation storage remains owned until its record is published.
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include "utils/NUMATopology.h"
#include "NUMAAllocator.h"
#include "FirstTouchBoundary.h"
#include "NUMAPageQuery.h"
#include "../utils/Logger.h"

#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <memory>
#include <cstring>
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/mman.h>
#include "utils/HostPageSize.h"

namespace llaminar2
{

    namespace
    {
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
            if (nonNUMAKernelSingleNode(0))
                LOG_INFO("NUMAAllocator: kernel has no NUMA support; node 0 is the whole machine");
            else
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

        if (!numa_available_ && nonNUMAKernelSingleNode(numa_node))
            return 0;
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

        size_t allocation_alignment = std::max(alignment, hostPageBytes());
        size_t aligned_bytes = alignUp(bytes, allocation_alignment);
        std::unique_ptr<void, decltype(&std::free)> allocation(
            std::aligned_alloc(allocation_alignment, aligned_bytes), &std::free);
        void *ptr = allocation.get();
        if (!ptr)
        {
            LOG_ERROR("NUMAAllocator: aligned_alloc failed for " << bytes << " bytes");
            return nullptr;
        }

        // aligned_alloc may recycle already-resident arena pages. Revoke and
        // fault the complete owned capacity while the current thread is scoped
        // to the requested node; a successful policy syscall alone is not a
        // page-placement proof on this platform.
        if (!firstTouchPageRangeOnNode(ptr, aligned_bytes, resolved_node))
        {
            return nullptr;
        }

        // Finish throwing preparation/diagnostics before publication. Record
        // insertion may allocate; its success precedes the nonthrowing counter
        // update, and the local owner releases storage only after both complete.
        LOG_TRACE("NUMAAllocator: Allocated " << bytes << " bytes on node " << resolved_node);
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            allocation_records_[ptr] = AllocationRecord{resolved_node, bytes};
            allocated_per_node_[resolved_node] += bytes;
        }
        return allocation.release();
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

        // allocateOnNode() has already certified every physical page. This
        // initialization changes contents only; it is not another placement
        // authority and therefore does not create an unbound OpenMP team.
        std::memset(ptr, init_value, bytes);

        LOG_TRACE("NUMAAllocator: Allocated and touched " << bytes << " bytes on node " << numa_node);
        return ptr;
    }

    bool NUMAAllocator::prepareExternalReceiveRangeOnNode(
        void *ptr, size_t bytes, int numa_node) const
    {
        if (!ptr || bytes == 0)
            return true;
        // No NUMA kernel: the single implicit node owns every page already.
        if (!numa_available_ && nonNUMAKernelSingleNode(numa_node))
            return true;
        if (!numa_available_)
        {
            LOG_ERROR("NUMAAllocator: Cannot prepare an external receive range because NUMA APIs are unavailable");
            return false;
        }
        if (numa_node < 0 || numa_node >= num_numa_nodes_)
        {
            LOG_ERROR("NUMAAllocator: External receive range requires an exact NUMA node; got "
                      << numa_node);
            return false;
        }

        const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
        if ((address % hostPageBytes()) != 0)
        {
            LOG_ERROR("NUMAAllocator: External receive range " << ptr
                                                               << " is not page aligned");
            return false;
        }
        if (bytes > std::numeric_limits<size_t>::max() - (hostPageBytes() - 1))
        {
            LOG_ERROR("NUMAAllocator: External receive range byte extent overflows page rounding");
            return false;
        }
        const size_t page_bytes = alignUp(bytes, hostPageBytes());
        return firstTouchPageRangeOnNode(ptr, page_bytes, numa_node);
    }

    bool NUMAAllocator::firstTouchPageRangeOnNode(
        void *ptr, size_t page_bytes, int numa_node) const
    {
        // No NUMA kernel: the single implicit node owns every page; first
        // touch needs no policy window or placement certification.
        if (!numa_available_ && nonNUMAKernelSingleNode(numa_node))
        {
            if (ptr && page_bytes)
                std::memset(ptr, 0, page_bytes);
            return ptr != nullptr;
        }
        if (!ptr || page_bytes == 0 ||
            (reinterpret_cast<uintptr_t>(ptr) % hostPageBytes()) != 0 ||
            (page_bytes % hostPageBytes()) != 0 ||
            numa_node < 0 || numa_node >= num_numa_nodes_)
        {
            LOG_ERROR("NUMAAllocator: Invalid exact page range supplied to first-touch authority");
            return false;
        }

        // Recycled aligned_alloc storage can already own pages on another node.
        // The caller promises every byte will be overwritten, so dropping the
        // old contents is both safe and necessary before first touch can be the
        // sole placement authority.
        if (madvise(ptr, page_bytes, MADV_DONTNEED) != 0)
        {
            LOG_ERROR("NUMAAllocator: Could not revoke " << page_bytes
                                                          << " bytes before first touch on node "
                                                          << numa_node << ": "
                                                          << std::strerror(errno));
            return false;
        }

        // The allocator retains range revocation and physical page certification.
        // The boundary borrows only the caller thread and returns after exact restoration.
        const auto populated = firstTouchPopulateRange(
            ptr, page_bytes, hostPageBytes(), numa_node,
            FirstTouchGeometry{numa_max_possible_node()}, firstTouchDefaultOps());
        if (!populated.completed)
        {
            LOG_ERROR("NUMAAllocator: First-touch populate sequence failed in phase "
                      << firstTouchPhaseName(populated.phase)
                      << " errno " << populated.failure_errno << " (node " << numa_node << ")");
            return false;
        }

        constexpr size_t kCertificationBatchPages = 1024;
        std::array<void *, kCertificationBatchPages> page_addresses{};
        std::array<int, kCertificationBatchPages> page_status{};
        const size_t page_count = page_bytes / hostPageBytes();
        for (size_t first_page = 0;
             first_page < page_count;
             first_page += kCertificationBatchPages)
        {
            const size_t batch_pages = std::min(
                kCertificationBatchPages, page_count - first_page);
            for (size_t index = 0; index < batch_pages; ++index)
            {
                page_addresses[index] = static_cast<uint8_t *>(ptr) +
                    ((first_page + index) * hostPageBytes());
                page_status[index] = -1;
            }
            if (move_pages(
                    0,
                    static_cast<unsigned long>(batch_pages),
                    page_addresses.data(),
                    nullptr,
                    page_status.data(),
                    0) != 0)
            {
                LOG_ERROR("NUMAAllocator: Batched first-touch certification failed on node "
                          << numa_node << ": " << std::strerror(errno));
                return false;
            }
            for (size_t index = 0; index < batch_pages; ++index)
            {
                // Linux's batched query ignores migration entries. Resolve only
                // that incomplete observation through get_user_pages-backed
                // MPOL_F_ADDR|MPOL_F_NODE: it joins kernel page migration rather
                // than assuming placement, re-touching bytes, or polling/retrying.
                const int observed_node = resolveObservedNUMANode(
                    page_status[index], page_addresses[index],
                    [this](const void *address) { return getNUMANodeForAddress(address); });
                if (observed_node != numa_node)
                {
                    const size_t offset =
                        (first_page + index) * hostPageBytes();
                    LOG_ERROR("NUMAAllocator: First-touch certification observed node "
                              << observed_node << " (batch status " << page_status[index]
                              << ") instead of "
                              << numa_node << " at page offset " << offset);
                    return false;
                }
            }
        }
        return true;
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
            // Query the physical page, not the VMA's preferred-node policy.
            // The kernel takes a temporary page reference and resolves migration
            // entries, unlike the non-faulting move_pages observation API.
            if (get_mempolicy(&node, nullptr, 0, const_cast<void *>(ptr),
                              MPOL_F_ADDR | MPOL_F_NODE) == 0 && node >= 0)
                return node;
        }

        return -1; // Cannot determine
    }

    bool NUMAAllocator::migrateToNode(void *ptr, size_t bytes, int numa_node)
    {
        if (!ptr || bytes == 0)
        {
            return false;
        }

        if (!numa_available_ && nonNUMAKernelSingleNode(numa_node))
            return true;
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
