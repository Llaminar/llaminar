/**
 * @file CPUSocketInfo.h
 * @brief Per-socket CPU information detected from sysfs
 *
 * Lightweight struct used by both HardwareInventory (local detection)
 * and DeviceInventory (MPI exchange). Kept separate to avoid pulling
 * heavy GPU headers into the MPI serialization layer.
 *
 * @author David Sanftenberg
 * @date 2026-04-11
 */

#pragma once

#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Per-socket CPU information
     *
     * Each physical CPU socket gets one entry with its model name,
     * associated NUMA node, physical core IDs, HT thread IDs, and
     * NUMA-local memory.
     */
    struct CPUSocketInfo
    {
        int socket_id = -1;              ///< Physical socket / package ID
        int numa_node = -1;              ///< Associated NUMA node ID
        std::string model_name;          ///< CPU model from /proc/cpuinfo
        std::vector<int> physical_cores; ///< First thread of each physical core (sorted)
        std::vector<int> ht_threads;     ///< HyperThreading sibling threads (sorted, empty if no HT)
        size_t memory_bytes = 0;         ///< Total NUMA-local memory in bytes.
        /**
         * Memory conservatively allocatable on this NUMA node at discovery.
         *
         * This is deliberately distinct from `memory_bytes`: tmpfs/ramdisk
         * pages and unrelated anonymous allocations consume the latter, while
         * inactive ordinary file cache may be reclaimed. Capacity admission
         * must use this value (or a smaller explicit limit) so two socket-local
         * CPU tiers cannot each claim the machine's nominal RAM.
         */
        size_t available_memory_bytes = 0;

        /// Number of physical cores on this socket
        int num_physical_cores() const { return static_cast<int>(physical_cores.size()); }

        /// Total threads (physical + HT) on this socket
        int num_threads() const { return static_cast<int>(physical_cores.size() + ht_threads.size()); }

        /// Whether this socket has HyperThreading enabled
        bool has_hyperthreading() const { return !ht_threads.empty(); }
    };

} // namespace llaminar2
