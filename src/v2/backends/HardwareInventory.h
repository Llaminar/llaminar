/**
 * @file HardwareInventory.h
 * @brief Startup-policy-scoped hardware inventory for one process
 *
 * Captures CPU sockets (with model, cores, HT threads, NUMA, memory) plus the
 * GPU backends that the process startup policy permits. Excluded vendors are
 * not queried: even a nominally read-only runtime call can materialize primary
 * contexts and consume memory needed by a concurrent inference process.
 *
 * Detected once during DeviceManager::initialize() and available for
 * downstream orchestration decisions without re-detection.
 *
 * @author David Sanftenberg
 * @date 2026-04-11
 */

#pragma once

#include "CPUSocketInfo.h"
#include "ComputeBackend.h"
#include <string>
#include <vector>
#include <map>

namespace llaminar2
{

    /**
     * @brief Policy-scoped hardware inventory for one process on one machine
     *
     * This struct is the single source of truth for hardware detected under
     * `debugEnv().backend_startup`. Consumers must treat an absent accelerator
     * vendor as excluded from this process, not as evidence that the physical
     * host lacks that hardware. Downstream orchestration should reuse this
     * value instead of bypassing the typed startup policy with direct driver
     * discovery.
     *
     * Usage:
     * @code
     *   const auto& hw = DeviceManager::instance().hardware();
     *   for (const auto& sock : hw.cpu_sockets) {
     *       LOG_INFO("Socket " << sock.socket_id << ": " << sock.model_name
     *                << " (" << sock.num_physical_cores() << "c/"
     *                << sock.num_threads() << "t) " << sock.memory_bytes / (1024*1024*1024) << " GB");
     *   }
     *   if (hw.has_cuda()) { ... }
     *   if (hw.rocm_p2p.has_value()) { ... }
     * @endcode
     */
    struct HardwareInventory
    {
        // =====================================================================
        // CPU
        // =====================================================================

        std::vector<CPUSocketInfo> cpu_sockets; ///< Per-socket CPU info (sorted by socket_id)

        /// Total physical cores across all sockets
        int total_physical_cores() const
        {
            int n = 0;
            for (const auto &s : cpu_sockets)
                n += s.num_physical_cores();
            return n;
        }

        /// Total threads across all sockets
        int total_threads() const
        {
            int n = 0;
            for (const auto &s : cpu_sockets)
                n += s.num_threads();
            return n;
        }

        /// Total CPU memory across all NUMA nodes (bytes)
        size_t total_cpu_memory() const
        {
            size_t n = 0;
            for (const auto &s : cpu_sockets)
                n += s.memory_bytes;
            return n;
        }

        /// Number of CPU sockets
        int num_sockets() const { return static_cast<int>(cpu_sockets.size()); }

        /// Whether any socket has HyperThreading
        bool has_hyperthreading() const
        {
            for (const auto &s : cpu_sockets)
                if (s.has_hyperthreading())
                    return true;
            return false;
        }

        // =====================================================================
        // GPUs
        // =====================================================================

        std::vector<ComputeDevice> cuda_devices; ///< CUDA GPUs visible when CUDA startup is enabled.
        std::vector<ComputeDevice> rocm_devices; ///< ROCm GPUs visible when ROCm startup is enabled.

        /// P2P access matrices (populated if >=2 devices of that backend)
        std::optional<P2PMatrix> cuda_p2p;
        std::optional<P2PMatrix> rocm_p2p;

        bool has_cuda() const { return !cuda_devices.empty(); }
        bool has_rocm() const { return !rocm_devices.empty(); }
        bool has_gpu() const { return has_cuda() || has_rocm(); }

        int cuda_device_count() const { return static_cast<int>(cuda_devices.size()); }
        int rocm_device_count() const { return static_cast<int>(rocm_devices.size()); }

        // =====================================================================
        // Detection
        // =====================================================================

        /**
         * @brief Detect all hardware on this machine
         *
         * Reads CPU topology from sysfs and `/proc/cpuinfo`. For each backend
         * permitted by `BackendStartupConfig`, it enumerates visible GPUs and
         * queries NUMA/P2P data. An excluded backend is not entered at all.
         *
         * @return Inventory populated only with process-permitted backends.
         */
        static HardwareInventory detect();

        // =====================================================================
        // Formatting helpers
        // =====================================================================

        /**
         * @brief Format a sorted vector of ints as compact ranges
         * @param cpus Sorted CPU IDs
         * @return e.g., "0-27, 56-83"
         */
        static std::string formatCpuRanges(const std::vector<int> &cpus);
    };

} // namespace llaminar2
