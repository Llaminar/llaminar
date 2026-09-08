/**
 * @file HostMemoryCapacity.h
 * @brief Canonical Linux host/NUMA memory accounting for admission decisions.
 *
 * CPU execution, ExpertOverlay capacity planning, and model-load preflight all
 * need the same answer to "how many bytes could be allocated now?" Raw
 * `MemFree` is too pessimistic because ordinary filesystem cache is
 * reclaimable. Linux accounts tmpfs/shmem folios on the anonymous LRU, so
 * `Inactive(file)` already excludes those non-discardable ramdisk bytes. This
 * interface exposes one conservative observation used by every production
 * caller without charging `Shmem` a second time.
 */

#pragma once

#include <cstddef>
#include <string_view>

namespace llaminar2
{
    /**
     * @brief One immutable host-memory observation from a single kernel view.
     *
     * `admission_available_bytes` is `MemFree` plus inactive regular-file cache,
     * capped by physical memory. `Shmem` is retained as diagnostic evidence but
     * is not subtracted: the kernel has already classified those swap-backed
     * folios outside the file LRU. Concrete subsystem allocations are charged
     * after this observation through their typed BOMs; no unnamed reserve is
     * mixed into the kernel observation.
     */
    struct HostMemoryCapacityObservation
    {
        std::size_t total_bytes = 0;               ///< Physical bytes in scope.
        std::size_t free_bytes = 0;                ///< Kernel `MemFree` bytes.
        std::size_t inactive_file_bytes = 0;       ///< Cold file-cache pages.
        std::size_t shared_memory_bytes = 0;       ///< tmpfs/ramdisk pages.
        std::size_t kernel_available_bytes = 0;    ///< Optional `MemAvailable` evidence.
        std::size_t admission_available_bytes = 0; ///< Canonical allocatable authority.

        /** @return Whether the observation contains a usable physical total. */
        [[nodiscard]] bool valid() const noexcept
        {
            return total_bytes > 0;
        }
    };

    /**
     * @brief Parse Linux `/proc` or per-node sysfs memory information.
     *
     * Both files use the same `Field: value kB` grammar; NUMA files merely
     * prefix fields with `Node N`.  Exposing the pure parser lets fast,
     * device-free unit tests lock down tmpfs exclusion and overflow behavior.
     * Unknown fields and malformed lines are ignored.
     *
     * @param text Complete contents of a Linux memory-information file.
     * @return Parsed observation, or an invalid zero observation when no
     *         physical total was present.
     */
    [[nodiscard]] HostMemoryCapacityObservation parseLinuxHostMemoryInfo(
        std::string_view text);

    /**
     * @brief Observe one NUMA node through its sysfs memory-information file.
     *
     * If sysfs is unavailable, libnuma supplies total and raw-free bytes.  The
     * fallback deliberately cannot invent reclaimable-cache evidence.
     *
     * @param numa_node Zero-based configured NUMA node.
     * @return Node-local observation, or an invalid zero observation.
     */
    [[nodiscard]] HostMemoryCapacityObservation observeNUMAMemoryCapacity(
        int numa_node);

    /**
     * @brief Observe machine-wide memory through `/proc/meminfo`.
     * @return System observation, or an invalid zero observation.
     */
    [[nodiscard]] HostMemoryCapacityObservation observeSystemMemoryCapacity();
} // namespace llaminar2
