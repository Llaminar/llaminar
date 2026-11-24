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

    /**
     * @brief Allocatable bytes of an integrated accelerator's shared host pool.
     *
     * An integrated GPU (DGX Spark/GB10, APUs) allocates from host DRAM. Its
     * driver free figure (`cudaMemGetInfo`/`hipMemGetInfo`) tracks `MemFree`
     * and excludes reclaimable page cache, which on a host that just read a
     * model file is routinely tens of GiB (NVIDIA DGX Spark porting guide,
     * "Memory reporting on UMA systems"). The pool's CPU member is already
     * admitted from the canonical host observation, so the accelerator member
     * must read that same observation: both describe one physical pool and
     * must never disagree about its free bytes.
     *
     * @param pool Canonical host observation of the accelerator's memory node
     *        (its NUMA node, or the system view when the node is unknown).
     * @param device_total_bytes Driver-reported accelerator total; an
     *        accelerator can never allocate beyond its own addressable total.
     * @return `min(pool.admission_available_bytes, device_total_bytes)`.
     * @throws std::runtime_error When the pool observation is invalid: an
     *         unobservable pool cannot be replaced by an invented figure.
     */
    [[nodiscard]] std::size_t integratedAcceleratorPoolAvailableBytes(
        const HostMemoryCapacityObservation &pool, std::size_t device_total_bytes);

    /**
     * @brief Observe the host pool that backs an integrated accelerator.
     * @param numa_node Accelerator memory node, or a negative value when unknown.
     * @return The node observation when the node is known and observable,
     *         otherwise the machine-wide `/proc/meminfo` observation.
     */
    [[nodiscard]] HostMemoryCapacityObservation observeIntegratedAcceleratorPool(int numa_node);
} // namespace llaminar2
