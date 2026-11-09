/**
 * @file NUMATopology.cpp
 * @brief NUMA topology detection with GPU identity borrowed from the physical inventory.
 *
 * Native GPU discovery is owned by the backend's explicit startup authority;
 * NUMA placement resolves that admitted PCI identity through the host topology.
 * CPU ownership belongs to the caller's complete binding, never the first
 * textual range match or a union including unrelated runtime/helper threads.
 * Missing or multi-node facts remain unknown for admission to reject.
 * @author David Sanftenberg
 * @date 2025-10-31
 */

#include "NUMATopology.h"
#include "NodeTopology.h"
#include "Logger.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <charconv>
#include <limits>
#include <string_view>

// Conditional includes for optional dependencies
#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

// GPU headers - CANNOT include both CUDA and HIP in same translation unit (type conflicts)
// When both are enabled, use extern declarations from GPUEnumeration.h instead
#if defined(HAVE_CUDA) && !defined(HAVE_ROCM)
#include "backends/cuda/CUDAHardwareMetadata.h"
#endif

#if defined(HAVE_ROCM) && !defined(HAVE_CUDA)
#include <hip/hip_runtime.h>
#endif

// When both are enabled, we use GPUEnumeration.h declarations to access
// NUMA detection functions implemented in separate compilation units
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "../backends/GPUEnumeration.h"
#elif defined(HAVE_ROCM)
#include "../backends/GPUEnumeration.h"
#endif

namespace llaminar2
{
    // Forward declarations of private helper functions
    static int detectViaHwloc();
    static int detectViaProcfs();
    static int getNumNodesViaSysfs();
    static int detectGPUViaNVML(int cuda_device_id);
    static std::string getCUDAPCIBusID(int cuda_device_id);
    static int detectGPUViaSysfs(const std::string &pci_bus_id);

    // ========================================================================
    // Local NUMA Node Detection
    // ========================================================================

    NUMAInfo NUMATopology::detectLocalNUMANode()
    {
        NUMAInfo info;
        info.total_numa_nodes = NUMATopology::getNumNUMANodes();

        // Try hwloc first (most reliable)
        int node = NUMATopology::detectViaHwloc();
        if (node >= 0)
        {
            info.local_numa_node = node;
            info.detection_succeeded = true;
            info.detection_method = "hwloc";
            LOG_DEBUG("[NUMATopology] Detected NUMA node " << node << " via hwloc");
            return info;
        }

        // Try /proc/self/status
        node = NUMATopology::detectViaProcfs();
        if (node >= 0)
        {
            info.local_numa_node = node;
            info.detection_succeeded = true;
            info.detection_method = "procfs";
            LOG_DEBUG("[NUMATopology] Detected NUMA node " << node << " via /proc/thread-self/status");
            return info;
        }

        // Unknown ownership must not publish a physical node. Distributed
        // discovery and explicit CPU admission already reject this result.
        info.local_numa_node = -1;
        info.detection_succeeded = false;
        info.detection_method = "unknown";
        LOG_WARN("[NUMATopology] Caller NUMA ownership could not be proven");
        return info;
    }

    int NUMATopology::getNumNUMANodes()
    {
        int count = NUMATopology::getNumNodesViaSysfs();
        if (count > 0)
        {
            return count;
        }

#ifdef HAVE_HWLOC
        hwloc_topology_t topology;
        if (hwloc_topology_init(&topology) == 0)
        {
            if (hwloc_topology_load(topology) == 0)
            {
                int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
                if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
                {
                    count = hwloc_get_nbobjs_by_depth(topology, depth);
                }
            }
            hwloc_topology_destroy(topology);
            if (count > 0)
                return count;
        }
#endif

        // Fallback
        return 1;
    }

    int NUMATopology::detectViaHwloc()
    {
#ifdef HAVE_HWLOC
        hwloc_topology_t topology;
        if (hwloc_topology_init(&topology) != 0)
        {
            return -1;
        }

        if (hwloc_topology_load(topology) != 0)
        {
            hwloc_topology_destroy(topology);
            return -1;
        }

        // Caller binding excludes unrelated MPI/GPU/library helper masks.
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
        if (!cpuset) { hwloc_topology_destroy(topology); return -1; }
        if (hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD) != 0)
        {
            hwloc_bitmap_free(cpuset);
            hwloc_topology_destroy(topology);
            return -1;
        }

        // Find NUMA node containing this CPU set
        int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
        if (depth == HWLOC_TYPE_DEPTH_UNKNOWN)
        {
            hwloc_bitmap_free(cpuset);
            hwloc_topology_destroy(topology);
            return -1;
        }

        int num_nodes = hwloc_get_nbobjs_by_depth(topology, depth);
        int detected_node = -1;

        for (int i = 0; i < num_nodes; ++i)
        {
            hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, depth, i);
            if (obj && obj->cpuset && !hwloc_bitmap_iszero(cpuset) &&
                hwloc_bitmap_isincluded(cpuset, obj->cpuset) &&
                obj->os_index <= static_cast<unsigned>(std::numeric_limits<int>::max()))
            {
                // Return the physical OS ID; topology order is not NUMA ID.
                if (detected_node >= 0)
                {
                    detected_node = -1;
                    break;
                }
                detected_node = static_cast<int>(obj->os_index);
            }
        }

        hwloc_bitmap_free(cpuset);
        hwloc_topology_destroy(topology);
        return detected_node;
#else
        return -1; // hwloc not available
#endif
    }

    int NUMATopology::detectViaProcfs()
    {
        return detectLocalNUMANodeFromStatus(
            "/proc/thread-self/status", "/sys/devices/system/cpu");
    }

    int NUMATopology::detectLocalNUMANodeFromStatus(
        const std::filesystem::path &status_path,
        const std::filesystem::path &cpu_sysfs_root)
    {
        std::ifstream status(status_path);
        if (!status.is_open()) return -1;
        std::string line;
        while (std::getline(status, line))
        {
            constexpr std::string_view field = "Cpus_allowed_list:";
            if (!line.starts_with(field)) continue;
            const auto allowed = line.substr(field.size());
            const auto report = [&](int node) {
                // Startup-only DEBUG evidence reuses the facts already read;
                // it adds no affinity query, payload read or worker logging.
                LOG_DEBUG("[NUMATopology] Caller allowed CPUs=[" << allowed
                          << "] exact NUMA=" << node);
                return node;
            };
            const auto cpus = NodeTopology::parseCPUListString(allowed);
            if (cpus.empty()) return report(-1);
            std::optional<int> owner;
            for (const int cpu : cpus)
            {
                if (cpu < 0) return report(-1);
                std::error_code error;
                const auto path = cpu_sysfs_root / ("cpu" + std::to_string(cpu));
                std::filesystem::directory_iterator entry(path, error), end;
                if (error) return report(-1);
                std::optional<int> cpu_owner;
                for (; entry != end; entry.increment(error))
                {
                    if (error) return report(-1);
                    const auto name = entry->path().filename().string();
                    if (!name.starts_with("node")) continue;
                    int node = -1;
                    const auto parsed = std::from_chars(name.data() + 4,
                                                        name.data() + name.size(), node);
                    if (parsed.ec != std::errc{} || parsed.ptr != name.data() + name.size() || node < 0)
                        continue;
                    const bool directory = entry->is_directory(error);
                    if (error) return report(-1);
                    if (!directory) continue;
                    if (cpu_owner) return report(-1);
                    cpu_owner = node;
                }
                if (error || !cpu_owner || (owner && *owner != *cpu_owner)) return report(-1);
                owner = cpu_owner;
            }
            return report(owner.value_or(-1));
        }
        return -1;
    }

    int NUMATopology::getNumNodesViaSysfs()
    {
        std::string node_path = "/sys/devices/system/node";
        if (!std::filesystem::exists(node_path))
        {
            return -1;
        }

        int count = 0;
        try
        {
            for (const auto &entry : std::filesystem::directory_iterator(node_path))
            {
                std::string name = entry.path().filename().string();
                if (name.find("node") == 0 && std::isdigit(name[4]))
                {
                    ++count;
                }
            }
        }
        catch (...)
        {
            return -1;
        }

        return count > 0 ? count : -1;
    }

    std::vector<int> NUMATopology::getCPUsOnNode(int numa_node)
    {
        std::vector<int> cpus;

        std::string cpulist_path = "/sys/devices/system/node/node" +
                                   std::to_string(numa_node) + "/cpulist";
        std::ifstream cpulist(cpulist_path);
        if (!cpulist.is_open())
        {
            return cpus;
        }

        std::string cpu_range;
        std::getline(cpulist, cpu_range);

        // Parse ranges like "0-27,56-83" or "0-27"
        std::istringstream ss(cpu_range);
        std::string range;
        while (std::getline(ss, range, ','))
        {
            size_t dash = range.find('-');
            if (dash != std::string::npos)
            {
                try
                {
                    int start = std::stoi(range.substr(0, dash));
                    int end = std::stoi(range.substr(dash + 1));
                    for (int cpu = start; cpu <= end; ++cpu)
                    {
                        cpus.push_back(cpu);
                    }
                }
                catch (...)
                {
                    continue;
                }
            }
            else
            {
                try
                {
                    cpus.push_back(std::stoi(range));
                }
                catch (...)
                {
                    continue;
                }
            }
        }

        return cpus;
    }

    // ========================================================================
    // GPU NUMA Affinity Detection
    // ========================================================================

    GPUNUMAInfo NUMATopology::getCUDAGPUNUMANode(int cuda_device_id)
    {
        GPUNUMAInfo info;
        info.gpu_id = cuda_device_id;
        info.numa_node = -1;

// When both CUDA and ROCm are enabled, we can't include CUDA headers (conflicts with HIP)
// Use sysfs-only detection in that case
#if defined(HAVE_CUDA) && !defined(HAVE_ROCM)
        // Use sysfs via CUDA's PCI IDs. This avoids a load-time dependency on
        // libnvidia-ml.so.1 so CUDA-enabled builds can still start on CPU-only
        // hosts.
        std::string pci_bus = NUMATopology::getCUDAPCIBusID(cuda_device_id);
        if (!pci_bus.empty())
        {
            info.pci_bus_id = pci_bus;
            int node = NUMATopology::detectGPUViaSysfs(pci_bus);
            if (node >= 0)
            {
                info.numa_node = node;
                info.affinity_detected = true;
                info.detection_method = "sysfs";
                LOG_DEBUG("[NUMATopology] CUDA GPU " << cuda_device_id << " on NUMA node " << node << " (via sysfs)");
                return info;
            }
        }
#elif defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Both backends enabled - use the CUDA enumeration helper which reads from sysfs
        // without requiring CUDA headers in this translation unit
        LOG_DEBUG("[NUMATopology] Both CUDA and ROCm enabled, using enumeration helper for CUDA GPU NUMA detection");

        int node = cuda_enumeration::get_cuda_device_numa_node(cuda_device_id);
        if (node >= 0)
        {
            info.numa_node = node;
            info.affinity_detected = true;
            info.detection_method = "sysfs";
            LOG_DEBUG("[NUMATopology] CUDA GPU " << cuda_device_id << " on NUMA node " << node << " (via sysfs, dual-backend)");
            return info;
        }

        // Fallback if sysfs detection failed
        info.numa_node = 0;
        info.affinity_detected = false;
        info.detection_method = "fallback";
        LOG_WARN("[NUMATopology] Could not detect NUMA affinity for CUDA GPU " << cuda_device_id << " (dual-backend), assuming node 0");
        return info;
#endif

        // Fallback to node 0
        info.numa_node = 0;
        info.affinity_detected = false;
        info.detection_method = "fallback";
        LOG_WARN("[NUMATopology] Could not detect NUMA affinity for CUDA GPU " << cuda_device_id << ", assuming node 0");
        return info;
    }

    int NUMATopology::detectGPUViaNVML(int cuda_device_id)
    {
#if defined(HAVE_CUDA) && !defined(HAVE_ROCM)
#endif
        (void)cuda_device_id;
        return -1;
    }

    std::string NUMATopology::getCUDAPCIBusID(int cuda_device_id)
    {
#if defined(HAVE_CUDA) && !defined(HAVE_ROCM)
        // NUMA discovery is an explicit startup boundary. Preserve the actual
        // complete BDF; a PCI function must not be fabricated from an ordinal.
        initializeCUDAHardwareMetadata();
        return cudaHardwareMetadata().at(cuda_device_id).pci_address;
#else
        (void)cuda_device_id;
        return "";
#endif
    }

    int NUMATopology::detectGPUViaSysfs(const std::string &pci_bus_id)
    {
        std::string numa_node_path = "/sys/bus/pci/devices/" + pci_bus_id + "/numa_node";
        std::ifstream numa_file(numa_node_path);
        if (!numa_file.is_open())
        {
            return -1;
        }

        int numa_node;
        numa_file >> numa_node;

        // Some systems return -1 for devices without NUMA affinity
        if (numa_node < 0)
        {
            return -1;
        }

        return numa_node;
    }

#ifdef HAVE_ROCM
    GPUNUMAInfo NUMATopology::getROCmGPUNUMANode(int rocm_device_id)
    {
        GPUNUMAInfo info;
        info.gpu_id = rocm_device_id;
        info.numa_node = -1;

        // Try ROCm SMI
        int node = detectGPUViaROCmSMI(rocm_device_id);
        if (node >= 0)
        {
            info.numa_node = node;
            info.affinity_detected = true;
            info.detection_method = "rocm_smi";
            LOG_DEBUG("[NUMATopology] ROCm GPU " << rocm_device_id << " on NUMA node " << node);
            return info;
        }

        // Fallback
        info.numa_node = 0;
        info.affinity_detected = false;
        info.detection_method = "fallback";
        LOG_WARN("[NUMATopology] Could not detect NUMA affinity for ROCm GPU " << rocm_device_id << ", assuming node 0");
        return info;
    }

    int NUMATopology::detectGPUViaROCmSMI(int rocm_device_id)
    {
        // Use the sysfs-based detection from ROCm enumeration
        // This reads /sys/bus/pci/devices/<pci_addr>/numa_node using HIP properties
        return rocm_enumeration::get_rocm_device_numa_node(rocm_device_id);
    }
#endif

    // ========================================================================
    // CPU Memory Bandwidth Estimation
    // ========================================================================

    float NUMATopology::getDDRBandwidth(const std::string &ddr_type, int channels)
    {
        // DDR bandwidth = transfers_per_second * 8 bytes_per_transfer * channels
        // Example: DDR5-4800 = 4800 MT/s * 8 bytes * 1 channel = 38.4 GB/s per channel

        // Common memory speeds (MT/s)
        float transfers_per_sec = 0.0f;

        // DDR5 variants
        if (ddr_type.find("DDR5-6400") != std::string::npos || ddr_type.find("6400") != std::string::npos)
            transfers_per_sec = 6400.0f;
        else if (ddr_type.find("DDR5-5600") != std::string::npos || ddr_type.find("5600") != std::string::npos)
            transfers_per_sec = 5600.0f;
        else if (ddr_type.find("DDR5-4800") != std::string::npos || ddr_type.find("4800") != std::string::npos)
            transfers_per_sec = 4800.0f;
        // DDR4 variants
        else if (ddr_type.find("DDR4-3600") != std::string::npos || ddr_type.find("3600") != std::string::npos)
            transfers_per_sec = 3600.0f;
        else if (ddr_type.find("DDR4-3200") != std::string::npos || ddr_type.find("3200") != std::string::npos)
            transfers_per_sec = 3200.0f;
        else if (ddr_type.find("DDR4-2933") != std::string::npos || ddr_type.find("2933") != std::string::npos)
            transfers_per_sec = 2933.0f;
        else if (ddr_type.find("DDR4-2666") != std::string::npos || ddr_type.find("2666") != std::string::npos)
            transfers_per_sec = 2666.0f;
        else if (ddr_type.find("DDR4-2400") != std::string::npos || ddr_type.find("2400") != std::string::npos)
            transfers_per_sec = 2400.0f;
        // Default assumption
        else if (ddr_type.find("DDR5") != std::string::npos)
            transfers_per_sec = 4800.0f; // Conservative DDR5 default
        else if (ddr_type.find("DDR4") != std::string::npos)
            transfers_per_sec = 2666.0f; // Conservative DDR4 default
        else
            transfers_per_sec = 2666.0f; // Very conservative fallback

        // Bandwidth = MT/s * 8 bytes * channels / 1000 (to GB/s)
        // Note: DDR is 64-bit wide = 8 bytes per transfer
        return (transfers_per_sec * 8.0f * static_cast<float>(channels)) / 1000.0f;
    }

    NUMATopology::CPUBandwidthInfo NUMATopology::estimateCPUBandwidth()
    {
        CPUBandwidthInfo info;
        info.is_estimate = true;

        // First, get socket count from NUMA nodes (typically 1 NUMA node = 1 socket)
        info.num_sockets = getNumNUMANodes();
        if (info.num_sockets <= 0)
            info.num_sockets = 1;

        // Try to detect processor model from /proc/cpuinfo
        std::string cpu_model;
        int physical_cores = 0;
        try
        {
            std::ifstream cpuinfo("/proc/cpuinfo");
            if (cpuinfo.is_open())
            {
                std::string line;
                while (std::getline(cpuinfo, line))
                {
                    if (line.find("model name") != std::string::npos)
                    {
                        auto pos = line.find(':');
                        if (pos != std::string::npos)
                        {
                            cpu_model = line.substr(pos + 2);
                        }
                    }
                    if (line.find("cpu cores") != std::string::npos)
                    {
                        auto pos = line.find(':');
                        if (pos != std::string::npos)
                        {
                            try
                            {
                                physical_cores = std::stoi(line.substr(pos + 2));
                            }
                            catch (...)
                            {
                            }
                        }
                    }
                }
            }
        }
        catch (...)
        {
            // Ignore errors, use fallback
        }

        LOG_TRACE("[NUMATopology] Detected CPU: " << cpu_model << ", " << physical_cores << " cores, " << info.num_sockets << " sockets");

        // Estimate based on processor model
        // Xeon Scalable (Sapphire Rapids, Emerald Rapids): 8-channel DDR5
        // Xeon Scalable (Ice Lake, Cascade Lake): 6-channel DDR4
        // EPYC (Genoa): 12-channel DDR5
        // EPYC (Milan): 8-channel DDR4
        // Consumer (desktop): 2-channel

        if (cpu_model.find("Xeon") != std::string::npos)
        {
            if (cpu_model.find("Sapphire") != std::string::npos ||
                cpu_model.find("Emerald") != std::string::npos ||
                cpu_model.find("w5") != std::string::npos ||
                cpu_model.find("w7") != std::string::npos ||
                cpu_model.find("w9") != std::string::npos)
            {
                // Sapphire/Emerald Rapids: 8-channel DDR5-4800
                info.memory_channels = 8;
                info.bandwidth_gbps = getDDRBandwidth("DDR5-4800", info.memory_channels) * info.num_sockets;
                info.detection_method = "model_xeon_ddr5_8ch";
            }
            else if (cpu_model.find("Platinum") != std::string::npos ||
                     cpu_model.find("Gold") != std::string::npos ||
                     cpu_model.find("Silver") != std::string::npos ||
                     cpu_model.find("Bronze") != std::string::npos)
            {
                // Cascade Lake / Ice Lake Xeon: 6-channel DDR4-3200
                info.memory_channels = 6;
                info.bandwidth_gbps = getDDRBandwidth("DDR4-3200", info.memory_channels) * info.num_sockets;
                info.detection_method = "model_xeon_ddr4_6ch";
            }
            else
            {
                // Generic Xeon: assume 6-channel DDR4-2933
                info.memory_channels = 6;
                info.bandwidth_gbps = getDDRBandwidth("DDR4-2933", info.memory_channels) * info.num_sockets;
                info.detection_method = "model_xeon_generic";
            }
        }
        else if (cpu_model.find("EPYC") != std::string::npos)
        {
            if (cpu_model.find("9") != std::string::npos && physical_cores >= 64)
            {
                // Genoa/Bergamo: 12-channel DDR5-4800
                info.memory_channels = 12;
                info.bandwidth_gbps = getDDRBandwidth("DDR5-4800", info.memory_channels) * info.num_sockets;
                info.detection_method = "model_epyc_ddr5_12ch";
            }
            else
            {
                // Milan/Rome: 8-channel DDR4-3200
                info.memory_channels = 8;
                info.bandwidth_gbps = getDDRBandwidth("DDR4-3200", info.memory_channels) * info.num_sockets;
                info.detection_method = "model_epyc_ddr4_8ch";
            }
        }
        else if (cpu_model.find("Ryzen") != std::string::npos ||
                 cpu_model.find("Core") != std::string::npos ||
                 cpu_model.find("i7") != std::string::npos ||
                 cpu_model.find("i9") != std::string::npos)
        {
            // Consumer desktop: 2-channel
            info.memory_channels = 2;
            if (cpu_model.find("13") != std::string::npos ||
                cpu_model.find("14") != std::string::npos ||
                cpu_model.find("7") != std::string::npos)
            {
                // Newer desktop with DDR5
                info.bandwidth_gbps = getDDRBandwidth("DDR5-5600", info.memory_channels);
                info.detection_method = "model_desktop_ddr5";
            }
            else
            {
                // Older desktop with DDR4
                info.bandwidth_gbps = getDDRBandwidth("DDR4-3200", info.memory_channels);
                info.detection_method = "model_desktop_ddr4";
            }
        }
        else
        {
            // Unknown: conservative estimate based on socket count
            if (info.num_sockets >= 2)
            {
                // Server-class: assume 6-channel DDR4-2666
                info.memory_channels = 6;
                info.bandwidth_gbps = getDDRBandwidth("DDR4-2666", info.memory_channels) * info.num_sockets;
                info.detection_method = "fallback_server";
            }
            else
            {
                // Single socket: assume consumer 2-channel DDR4-3200
                info.memory_channels = 2;
                info.bandwidth_gbps = getDDRBandwidth("DDR4-3200", info.memory_channels);
                info.detection_method = "fallback_desktop";
            }
        }

        LOG_DEBUG("[NUMATopology] Estimated CPU memory bandwidth: " << info.bandwidth_gbps
                                                                    << " GB/s (" << info.memory_channels << " channels × " << info.num_sockets
                                                                    << " sockets, method=" << info.detection_method << ")");

        return info;
    }

} // namespace llaminar2
