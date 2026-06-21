/**
 * @file NUMATopology.h
 * @brief NUMA topology detection and GPU affinity utilities
 *
 * Provides socket/NUMA node detection for MPI rank-to-device binding.
 * Ensures each MPI rank only uses devices affine to its socket for
 * optimal memory bandwidth and avoiding cross-socket penalties.
 *
 * @author David Sanftenberg
 * @date 2025-10-31
 */

#pragma once

#include <string>
#include <vector>
#include <optional>

namespace llaminar2
{

    /**
     * @brief NUMA topology information for current process
     */
    struct NUMAInfo
    {
        int local_numa_node = 0;          ///< NUMA node this process is running on
        int total_numa_nodes = 1;         ///< Total NUMA nodes in system
        bool detection_succeeded = false; ///< Whether detection was successful
        std::string detection_method;     ///< How node was detected ("hwloc", "procfs", "fallback")
    };

    /**
     * @brief GPU NUMA affinity information
     */
    struct GPUNUMAInfo
    {
        int gpu_id;                     ///< GPU device ID (CUDA/ROCm)
        int numa_node;                  ///< NUMA node GPU is affine to (-1 if unknown)
        std::string pci_bus_id;         ///< PCIe bus ID (e.g., "0000:3b:00.0")
        bool affinity_detected = false; ///< Whether affinity was successfully detected
        std::string detection_method;   ///< How affinity was detected ("sysfs", "fallback")
    };

    /**
     * @brief NUMA topology detection utilities
     *
     * Provides methods to detect:
     * 1. Which NUMA node the current process is bound to
     * 2. Which NUMA node each GPU is affine to
     * 3. Total number of NUMA nodes in the system
     *
     * Detection methods (in priority order):
     * 1. hwloc library (most reliable, optional dependency)
     * 2. /proc/self/status (Linux-specific)
     * 3. /sys/bus/pci/devices/PCIBUSID/numa_node (Linux sysfs)
     * 4. Fallback to node 0 (single-socket assumption)
     */
    class NUMATopology
    {
    public:
        /**
         * @brief Detect NUMA node for current process
         *
         * Tries multiple methods in order:
         * 1. hwloc_get_cpubind() if available
         * 2. Parse /proc/self/status Cpus_allowed_list
         * 3. Fallback to node 0
         *
         * @return NUMA info with detection status
         */
        static NUMAInfo detectLocalNUMANode();

        /**
         * @brief Get total number of NUMA nodes in system
         *
         * @return Number of NUMA nodes (sockets), minimum 1
         */
        static int getNumNUMANodes();

        /**
         * @brief Detect NUMA node for CUDA GPU
         *
         * Uses CUDA PCI IDs plus Linux sysfs to query GPU NUMA affinity.
         * Falls back to node 0 if detection fails.
         *
         * @param cuda_device_id CUDA device ID (0-indexed)
         * @return GPU NUMA info with detection status
         */
        static GPUNUMAInfo getCUDAGPUNUMANode(int cuda_device_id);

#ifdef HAVE_ROCM
        /**
         * @brief Detect NUMA node for ROCm GPU
         *
         * Uses ROCm SMI (rocm_smi_lib) to query GPU's PCIe NUMA affinity.
         * Falls back to node 0 if SMI unavailable or detection fails.
         *
         * @param rocm_device_id ROCm/HIP device ID (0-indexed)
         * @return GPU NUMA info with detection status
         */
        static GPUNUMAInfo getROCmGPUNUMANode(int rocm_device_id);
#endif

        /**
         * @brief Validate GPU is on same NUMA node as process
         *
         * Helper to check if GPU should be used by this MPI rank.
         *
         * @param gpu_numa_node GPU's NUMA node
         * @param process_numa_node Process's NUMA node
         * @return true if GPU is local to process, false if cross-socket
         */
        static bool isGPULocalToProcess(int gpu_numa_node, int process_numa_node)
        {
            // If either is unknown (-1), assume compatible (fallback behavior)
            if (gpu_numa_node < 0 || process_numa_node < 0)
                return true;

            return gpu_numa_node == process_numa_node;
        }

        /**
         * @brief Get CPUs on specific NUMA node
         *
         * @param numa_node NUMA node index
         * @return Vector of CPU IDs on this node (empty if detection fails)
         */
        static std::vector<int> getCPUsOnNode(int numa_node);

    private:
        // Private implementation methods

        /**
         * @brief Detect NUMA node via hwloc library
         *
         * Most reliable method when hwloc is available.
         *
         * @return NUMA node or -1 if detection failed
         */
        static int detectViaHwloc();

        /**
         * @brief Detect NUMA node via /proc/self/status
         *
         * Parses Cpus_allowed_list to infer NUMA node.
         * Works on Linux systems.
         *
         * @return NUMA node or -1 if detection failed
         */
        static int detectViaProcfs();

        /**
         * @brief Get total NUMA nodes via /sys/devices/system/node/
         *
         * Counts node directories in sysfs.
         *
         * @return Number of NUMA nodes or 1 if detection failed
         */
        static int getNumNodesViaSysfs();

        /**
         * @brief Deprecated NVML detection shim; returns -1.
         *
         * @param cuda_device_id CUDA device ID
         * @return NUMA node or -1 if detection failed
         */
        static int detectGPUViaNVML(int cuda_device_id);

        /**
         * @brief Detect GPU NUMA node via sysfs
         *
         * Reads /sys/bus/pci/devices/<bus_id>/numa_node
         *
         * @param pci_bus_id PCIe bus ID (e.g., "0000:3b:00.0")
         * @return NUMA node or -1 if detection failed
         */
        static int detectGPUViaSysfs(const std::string &pci_bus_id);

        /**
         * @brief Get PCIe bus ID for CUDA device
         *
         * @param cuda_device_id CUDA device ID
         * @return PCIe bus ID or empty string if failed
         */
        static std::string getCUDAPCIBusID(int cuda_device_id);

#ifdef HAVE_ROCM
        /**
         * @brief Detect GPU NUMA node via ROCm SMI
         *
         * @param rocm_device_id ROCm device ID
         * @return NUMA node or -1 if detection failed
         */
        static int detectGPUViaROCmSMI(int rocm_device_id);
#endif

    public:
        // =========================================================================
        // CPU Memory Bandwidth Estimation (for phase-aware placement)
        // =========================================================================

        /**
         * @brief Estimated CPU memory bandwidth in GB/s
         *
         * Contains estimated memory bandwidth for decode phase placement.
         * Modern Xeons with 6-8 memory channels should NOT be idle during decode!
         */
        struct CPUBandwidthInfo
        {
            float bandwidth_gbps = 0.0f;       ///< Estimated total bandwidth in GB/s
            int memory_channels = 0;           ///< Number of memory channels per socket
            int num_sockets = 1;               ///< Number of CPU sockets
            std::string detection_method;      ///< How bandwidth was estimated
            bool is_estimate = true;           ///< true if estimated, false if measured
        };

        /**
         * @brief Estimate CPU memory bandwidth for this system
         *
         * Detection methods (in order):
         * 1. Parse DMI/SMBIOS for memory speed and channel count
         * 2. Read /proc/cpuinfo for processor model and estimate
         * 3. Fallback to conservative estimate based on socket count
         *
         * Common estimates:
         * - Xeon Scalable (8 channels DDR5-4800): ~307 GB/s per socket
         * - Xeon Scalable (6 channels DDR4-3200): ~205 GB/s per socket
         * - Consumer DDR5-5600 (2 channels): ~90 GB/s
         * - Consumer DDR4-3200 (2 channels): ~51 GB/s
         *
         * @return CPU bandwidth info with detection status
         */
        static CPUBandwidthInfo estimateCPUBandwidth();

        /**
         * @brief Get DDR bandwidth for common memory types
         * @param ddr_type DDR type string (e.g., "DDR4-3200", "DDR5-4800")
         * @param channels Number of memory channels
         * @return Estimated bandwidth in GB/s
         */
        static float getDDRBandwidth(const std::string& ddr_type, int channels);
    };

} // namespace llaminar2
