/**
 * @file NUMATopology.cpp
 * @brief NUMA topology detection implementation
 * @author David Sanftenberg
 * @date 2025-10-31
 */

#include "NUMATopology.h"
#include "Logger.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

// Conditional includes for optional dependencies
#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <nvml.h>
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
// TODO: Add ROCm SMI include when available
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
            LOG_DEBUG("[NUMATopology] Detected NUMA node " << node << " via /proc/self/status");
            return info;
        }

        // Fallback to node 0 (single-socket assumption)
        info.local_numa_node = 0;
        info.detection_succeeded = false;
        info.detection_method = "fallback";
        LOG_WARN("[NUMATopology] NUMA detection failed, assuming node 0 (single socket)");
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

        // Get CPU binding
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
        if (hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_PROCESS) != 0)
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
            if (obj && hwloc_bitmap_intersects(cpuset, obj->cpuset))
            {
                detected_node = i;
                break;
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
        // Read /proc/self/status and look for Cpus_allowed_list
        std::ifstream status("/proc/self/status");
        if (!status.is_open())
        {
            return -1;
        }

        std::string line;
        while (std::getline(status, line))
        {
            if (line.find("Cpus_allowed_list:") == 0)
            {
                // Parse CPU range (e.g., "0-27" or "28-55")
                size_t colon = line.find(':');
                if (colon == std::string::npos)
                    continue;

                std::string cpu_list = line.substr(colon + 1);
                // Trim whitespace
                cpu_list.erase(0, cpu_list.find_first_not_of(" \t"));

                // Parse first CPU in range
                size_t dash = cpu_list.find('-');
                int first_cpu = -1;
                try
                {
                    if (dash != std::string::npos)
                    {
                        first_cpu = std::stoi(cpu_list.substr(0, dash));
                    }
                    else
                    {
                        first_cpu = std::stoi(cpu_list);
                    }
                }
                catch (...)
                {
                    return -1;
                }

                // Infer NUMA node from first CPU
                // Read /sys/devices/system/cpu/cpu{N}/node{M}/cpulist
                for (int node = 0; node < 8; ++node) // Try up to 8 NUMA nodes
                {
                    std::string cpulist_path = "/sys/devices/system/node/node" +
                                               std::to_string(node) + "/cpulist";
                    std::ifstream cpulist(cpulist_path);
                    if (!cpulist.is_open())
                        continue;

                    std::string node_cpus;
                    std::getline(cpulist, node_cpus);

                    // Check if first_cpu is in this node's CPU list
                    // Simple check: does the string contain our CPU number?
                    if (node_cpus.find(std::to_string(first_cpu)) != std::string::npos)
                    {
                        return node;
                    }
                }
            }
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

#ifdef HAVE_CUDA
        // Try NVML first (most reliable)
        int node = NUMATopology::detectGPUViaNVML(cuda_device_id);
        if (node >= 0)
        {
            info.numa_node = node;
            info.affinity_detected = true;
            info.detection_method = "nvml";
            LOG_DEBUG("[NUMATopology] CUDA GPU " << cuda_device_id << " on NUMA node " << node << " (via NVML)");
            return info;
        }

        // Try sysfs fallback
        std::string pci_bus = NUMATopology::getCUDAPCIBusID(cuda_device_id);
        if (!pci_bus.empty())
        {
            info.pci_bus_id = pci_bus;
            node = NUMATopology::detectGPUViaSysfs(pci_bus);
            if (node >= 0)
            {
                info.numa_node = node;
                info.affinity_detected = true;
                info.detection_method = "sysfs";
                LOG_DEBUG("[NUMATopology] CUDA GPU " << cuda_device_id << " on NUMA node " << node << " (via sysfs)");
                return info;
            }
        }
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
#ifdef HAVE_CUDA
        // Initialize NVML
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS)
        {
            LOG_DEBUG("[NUMATopology] NVML init failed: " << nvmlErrorString(result));
            return -1;
        }

        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(cuda_device_id, &device);
        if (result != NVML_SUCCESS)
        {
            nvmlShutdown();
            return -1;
        }

        // Get PCIe info
        nvmlPciInfo_t pci_info;
        result = nvmlDeviceGetPciInfo(device, &pci_info);
        if (result != NVML_SUCCESS)
        {
            nvmlShutdown();
            return -1;
        }

        // Get NUMA node from PCIe topology
        unsigned int numa_node;
        result = nvmlDeviceGetNumaNodeId(device, &numa_node); // Correct function name
        nvmlShutdown();

        if (result == NVML_SUCCESS && numa_node != (unsigned int)-1) // Use -1 for unknown
        {
            return static_cast<int>(numa_node);
        }
#endif
        return -1;
    }

    std::string NUMATopology::getCUDAPCIBusID(int cuda_device_id)
    {
#ifdef HAVE_CUDA
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, cuda_device_id) == cudaSuccess)
        {
            // Format: domain:bus:device.function
            char bus_id[32];
            snprintf(bus_id, sizeof(bus_id), "%04x:%02x:%02x.0",
                     prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
            return std::string(bus_id);
        }
#endif
        return "";
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
        // TODO: Implement ROCm SMI detection when library is available
        // For now, return -1 (detection not supported)
        return -1;
    }
#endif

} // namespace llaminar2
