/**
 * @file DeviceRegistry.cpp
 * @brief Singleton device registry implementation
 *
 * Provides centralized device discovery for CPU NUMA nodes, CUDA GPUs,
 * and ROCm GPUs. Uses GlobalDeviceAddress as the canonical identifier.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "DeviceRegistry.h"
#include "BackendManager.h"
#include "GPUEnumeration.h"
#include "../utils/Logger.h"

#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

#ifdef HAVE_CUDA
#include "cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "rocm/ROCmBackend.h"
#endif

namespace llaminar2
{
    namespace
    {
        std::pair<int, int> splitComputeCapability(int capability)
        {
            return {capability / 10, capability % 10};
        }
    }

    // =========================================================================
    // Singleton Instance
    // =========================================================================

    DeviceRegistry &DeviceRegistry::instance()
    {
        static DeviceRegistry instance;
        return instance;
    }

    DeviceRegistry::DeviceRegistry()
    {
        // Don't auto-discover in constructor - let caller decide when
    }

    // =========================================================================
    // Discovery
    // =========================================================================

    void DeviceRegistry::discover()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Clear existing data
        devices_.clear();
        device_index_.clear();
        device_info_.clear();

        // Discover all device types
        discoverCpuDevices();
        discoverCudaDevices();
        discoverRocmDevices();

        discovered_ = true;

        LOG_INFO("[DeviceRegistry] Discovered " << devices_.size() << " devices ("
                                                << deviceCount(DeviceType::CPU) << " CPU, "
                                                << deviceCount(DeviceType::CUDA) << " CUDA, "
                                                << deviceCount(DeviceType::ROCm) << " ROCm)");
    }

    bool DeviceRegistry::isDiscovered() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return discovered_;
    }

    void DeviceRegistry::refresh()
    {
        discover();
    }

    // =========================================================================
    // CPU Discovery
    // =========================================================================

    void DeviceRegistry::discoverCpuDevices()
    {
        // Count NUMA nodes from /sys/devices/system/node/
        std::string numa_path = "/sys/devices/system/node";
        DIR *dir = opendir(numa_path.c_str());

        if (!dir)
        {
            // No NUMA info available - assume single node
            LOG_DEBUG("[DeviceRegistry] No NUMA topology found, assuming single CPU node");

            GlobalDeviceAddress addr = GlobalDeviceAddress::cpu(0);
            std::string key = getKey(addr);

            devices_.push_back(addr);
            device_index_[key] = devices_.size() - 1;

            DeviceInfo info;
            info.name = "CPU:NUMA0";
            info.numa_affinity = 0;

            // Get system memory
            std::ifstream meminfo("/proc/meminfo");
            if (meminfo.is_open())
            {
                std::string line;
                while (std::getline(meminfo, line))
                {
                    if (line.find("MemTotal:") == 0)
                    {
                        std::istringstream iss(line);
                        std::string label;
                        size_t mem_kb;
                        iss >> label >> mem_kb;
                        info.memory_capacity = mem_kb * 1024; // Convert to bytes
                        break;
                    }
                }
            }

            device_info_[key] = info;
            return;
        }

        // Enumerate NUMA nodes
        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            if (entry->d_name[0] == '.')
                continue;

            // Match "node0", "node1", etc.
            if (strncmp(entry->d_name, "node", 4) == 0)
            {
                int numa_node = atoi(entry->d_name + 4);

                GlobalDeviceAddress addr = GlobalDeviceAddress::cpu(numa_node);
                std::string key = getKey(addr);

                devices_.push_back(addr);
                device_index_[key] = devices_.size() - 1;

                DeviceInfo info;
                info.name = "CPU:NUMA" + std::to_string(numa_node);
                info.numa_affinity = numa_node;

                // Read NUMA node memory from sysfs
                // Format: "Node 0 MemTotal:       12345678 kB"
                std::string meminfo_path = numa_path + "/" + entry->d_name + "/meminfo";
                std::ifstream meminfo(meminfo_path);
                if (meminfo.is_open())
                {
                    std::string line;
                    while (std::getline(meminfo, line))
                    {
                        if (line.find("MemTotal:") != std::string::npos)
                        {
                            // Parse "Node X MemTotal:       NNNNN kB"
                            // Find the colon and parse the number after it
                            size_t colon_pos = line.find(':');
                            if (colon_pos != std::string::npos)
                            {
                                std::string value_part = line.substr(colon_pos + 1);
                                std::istringstream iss(value_part);
                                size_t mem_kb;
                                iss >> mem_kb;
                                info.memory_capacity = mem_kb * 1024; // Convert to bytes
                            }
                            break;
                        }
                    }
                }

                device_info_[key] = info;
            }
        }

        closedir(dir);
    }

    // =========================================================================
    // CUDA Discovery
    // =========================================================================

    void DeviceRegistry::discoverCudaDevices()
    {
#ifdef HAVE_CUDA
        auto cuda_devices = cuda_enumeration::enumerate_cuda_devices();

        for (const auto &dev : cuda_devices)
        {
            int numa_node = cuda_enumeration::get_cuda_device_numa_node(dev.device_id);

            GlobalDeviceAddress addr = GlobalDeviceAddress::cuda(dev.device_id, numa_node);
            std::string key = getKey(addr);

            devices_.push_back(addr);
            device_index_[key] = devices_.size() - 1;

            DeviceInfo info;
            info.name = dev.name;
            info.numa_affinity = numa_node;
            info.compute_capability = splitComputeCapability(dev.compute_capability);
            info.arch_info = dev.arch_info;

            // Get memory and compute capability from backend
            IBackend *backend = getCUDABackend();
            if (backend)
            {
                info.memory_capacity = backend->deviceMemoryTotal(dev.device_id);
            }

            // Extract compute capability from ComputeDevice if available
            // For now, store device_id in a way we can query later
            // The actual compute capability would need to come from cudaGetDeviceProperties

            device_info_[key] = info;
        }
#else
        LOG_DEBUG("[DeviceRegistry] CUDA not available (HAVE_CUDA not defined)");
#endif
    }

    // =========================================================================
    // ROCm Discovery
    // =========================================================================

    void DeviceRegistry::discoverRocmDevices()
    {
#ifdef HAVE_ROCM
        auto rocm_devices = rocm_enumeration::enumerate_rocm_devices();

        for (const auto &dev : rocm_devices)
        {
            int numa_node = rocm_enumeration::get_rocm_device_numa_node(dev.device_id);

            GlobalDeviceAddress addr = GlobalDeviceAddress::rocm(dev.device_id, numa_node);
            std::string key = getKey(addr);

            devices_.push_back(addr);
            device_index_[key] = devices_.size() - 1;

            DeviceInfo info;
            info.name = dev.name;
            info.numa_affinity = numa_node;
            info.compute_capability = splitComputeCapability(dev.compute_capability);
            info.arch_info = dev.arch_info;

            // Get memory from backend
            IBackend *backend = getROCmBackend();
            if (backend)
            {
                info.memory_capacity = backend->deviceMemoryTotal(dev.device_id);
            }

            device_info_[key] = info;
        }
#else
        LOG_DEBUG("[DeviceRegistry] ROCm not available (HAVE_ROCM not defined)");
#endif
    }

    // =========================================================================
    // Device Queries
    // =========================================================================

    std::vector<GlobalDeviceAddress> DeviceRegistry::allDevices() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return devices_;
    }

    std::vector<GlobalDeviceAddress> DeviceRegistry::devicesByType(DeviceType type) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<GlobalDeviceAddress> result;
        for (const auto &addr : devices_)
        {
            if (addr.device_type == type)
            {
                result.push_back(addr);
            }
        }
        return result;
    }

    std::vector<GlobalDeviceAddress> DeviceRegistry::devicesByNuma(int numa_node) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<GlobalDeviceAddress> result;
        for (const auto &addr : devices_)
        {
            if (addr.numa_node == numa_node)
            {
                result.push_back(addr);
            }
        }
        return result;
    }

    size_t DeviceRegistry::deviceCount(DeviceType type) const
    {
        // Note: Don't lock here as we're already locked in discover()
        // Caller should lock if needed
        size_t count = 0;
        for (const auto &addr : devices_)
        {
            if (addr.device_type == type)
            {
                count++;
            }
        }
        return count;
    }

    size_t DeviceRegistry::totalDeviceCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return devices_.size();
    }

    // =========================================================================
    // Validation
    // =========================================================================

    bool DeviceRegistry::isValid(const GlobalDeviceAddress &addr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Only validate local devices
        if (!addr.isLocal())
        {
            // Can't validate remote devices
            return false;
        }

        std::string key = getKey(addr);
        return device_index_.find(key) != device_index_.end();
    }

    bool DeviceRegistry::isAvailable(const GlobalDeviceAddress &addr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const DeviceInfo *info = findInfo(addr);
        if (!info)
            return false;

        return info->available;
    }

    // =========================================================================
    // Device Properties
    // =========================================================================

    size_t DeviceRegistry::memoryCapacity(const GlobalDeviceAddress &addr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const DeviceInfo *info = findInfo(addr);
        if (!info)
            return 0;

        return info->memory_capacity;
    }

    size_t DeviceRegistry::memoryAvailable(const GlobalDeviceAddress &addr) const
    {
        // For dynamic memory query, we need to ask the backend
        IBackend *backend = nullptr;

        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (!isValid(addr))
                return 0;
        }

        // Get backend outside lock to avoid potential deadlock
        switch (addr.device_type)
        {
        case DeviceType::CPU:
            backend = getCPUBackend();
            break;
        case DeviceType::CUDA:
            backend = getCUDABackend();
            break;
        case DeviceType::ROCm:
            backend = getROCmBackend();
            break;
        default:
            return 0;
        }

        if (backend)
        {
            return backend->deviceMemoryFree(addr.device_ordinal);
        }

        return 0;
    }

    std::pair<int, int> DeviceRegistry::computeCapability(const GlobalDeviceAddress &addr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const DeviceInfo *info = findInfo(addr);
        if (!info)
            return {0, 0};

        return info->compute_capability;
    }

    std::string DeviceRegistry::deviceName(const GlobalDeviceAddress &addr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const DeviceInfo *info = findInfo(addr);
        if (!info)
            return "";

        return info->name;
    }

    // =========================================================================
    // Backend Access
    // =========================================================================

    IBackend *DeviceRegistry::backendFor(const GlobalDeviceAddress &addr)
    {
        switch (addr.device_type)
        {
        case DeviceType::CPU:
            return getCPUBackend();
        case DeviceType::CUDA:
            return getCUDABackend();
        case DeviceType::ROCm:
            return getROCmBackend();
        default:
            return nullptr;
        }
    }

    // =========================================================================
    // Topology
    // =========================================================================

    int DeviceRegistry::numaAffinity(const GlobalDeviceAddress &addr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const DeviceInfo *info = findInfo(addr);
        if (!info)
            return -1;

        return info->numa_affinity;
    }

    std::string DeviceRegistry::pcieBusId(const GlobalDeviceAddress &addr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const DeviceInfo *info = findInfo(addr);
        if (!info)
            return "";

        return info->pcie_bus_id;
    }

    bool DeviceRegistry::canP2P(const GlobalDeviceAddress &a, const GlobalDeviceAddress &b) const
    {
        // CPU never does P2P
        if (a.isCPU() || b.isCPU())
            return false;

        // Different device types can't do P2P (CUDA ↔ ROCm uses HOST staging)
        if (a.device_type != b.device_type)
            return false;

        // Same device - trivially true
        if (a == b)
            return true;

        // For actual P2P capability, we'd need to query the driver
        // For now, assume same-type GPUs on same NUMA can do P2P
        return a.sameNuma(b);
    }

    // =========================================================================
    // Additional Methods
    // =========================================================================

    std::optional<GlobalDeviceAddress> DeviceRegistry::defaultDevice(DeviceType type) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto &addr : devices_)
        {
            if (addr.device_type == type && addr.device_ordinal == 0)
            {
                return addr;
            }
        }

        return std::nullopt;
    }

    std::optional<DeviceArchInfo> DeviceRegistry::archInfo(DeviceId device) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!device.is_gpu())
        {
            return std::nullopt;
        }

        for (const auto &addr : devices_)
        {
            if (addr.device_type == device.type && addr.device_ordinal == device.ordinal)
            {
                const DeviceInfo *info = findInfo(addr);
                if (info && info->arch_info)
                {
                    return info->arch_info;
                }
                return std::nullopt;
            }
        }

        return std::nullopt;
    }

    DeviceId DeviceRegistry::resolve(DeviceId device) const
    {
        auto info = archInfo(device);
        if (!info)
        {
            return device;
        }
        return device.with_arch_info(*info);
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    std::string DeviceRegistry::getKey(const GlobalDeviceAddress &addr) const
    {
        return addr.toString();
    }

    const DeviceRegistry::DeviceInfo *DeviceRegistry::findInfo(const GlobalDeviceAddress &addr) const
    {
        std::string key = getKey(addr);
        auto it = device_info_.find(key);
        if (it == device_info_.end())
            return nullptr;
        return &it->second;
    }

} // namespace llaminar2
