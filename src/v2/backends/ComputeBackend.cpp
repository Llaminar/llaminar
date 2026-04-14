/**
 * @file ComputeBackend.cpp
 * @brief Device manager and backend implementation
 *
 * Supports:
 * - CPU (OpenBLAS/MKL)
 * - NVIDIA CUDA
 * - AMD ROCm
 * - Vulkan (cross-vendor)
 *
 * Phase 6: Multi-GPU (heterogeneous)
 * GPU enumeration is now in separate compilation units to avoid header conflicts:
 *   - CUDAEnumeration.cu (CUDA only)
 *   - ROCmEnumeration.cpp (ROCm only, compiled with hipcc)
 * This allows CUDA and ROCm to coexist in the same binary.
 *
 * @author David Sanftenberg
 */

#include "ComputeBackend.h"
#include "HardwareInventory.h"
#include "GPUEnumeration.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "../utils/CPUFeatures.h"
#include "../utils/NUMATopology.h"
#include "../kernels/cpu/ops/CPURoPEKernelT.h"
#include "../kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "../kernels/cpu/ops/CPUSoftmaxKernelT.h"
#include "fort.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <numa.h>

// ============================================================================
// GPU Header Includes REMOVED (Phase 6)
// ============================================================================
// GPU enumeration moved to separate compilation units to enable heterogeneous
// multi-GPU (CUDA + ROCm in same binary). See:
//   - backends/CUDAEnumeration.cu
//   - backends/ROCmEnumeration.cpp
//   - backends/GPUEnumeration.h (declarations)
// ============================================================================

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace llaminar2
{

    // ============================================================================
    // Helper: Get backend type name
    // ============================================================================

    static const char *backend_type_name(ComputeBackendType type)
    {
        switch (type)
        {
        case ComputeBackendType::CPU:
            return "CPU";
        case ComputeBackendType::GPU_CUDA:
            return "NVIDIA CUDA";
        case ComputeBackendType::GPU_ROCM:
            return "AMD ROCm";
        case ComputeBackendType::GPU_VULKAN:
            return "Vulkan";
        default:
            return "Unknown";
        }
    }

    // ============================================================================
    // CPU Device Enumeration
    // ============================================================================

    static ComputeDevice enumerate_cpu_device(int numa_node = -1)
    {
        ComputeDevice dev;

        dev.type = ComputeBackendType::CPU;
        dev.name = "CPU";

        dev.device_id = 0;
        dev.compute_capability = 0;
        dev.numa_node = numa_node;

        // Get memory info - prefer NUMA-local memory if node specified
#ifdef __linux__
        if (numa_node >= 0 && numa_available() >= 0)
        {
            // Get NUMA-local memory for this node
            long long numa_size = numa_node_size64(numa_node, nullptr);
            if (numa_size > 0)
            {
                dev.total_memory_bytes = static_cast<size_t>(numa_size);
                // Free memory approximation: assume ~90% available
                long long numa_free = 0;
                numa_node_size64(numa_node, &numa_free);
                dev.free_memory_bytes = (numa_free > 0) ? static_cast<size_t>(numa_free) : dev.total_memory_bytes;
            }
            else
            {
                // Fallback to system memory divided by NUMA nodes
                int num_nodes = numa_num_configured_nodes();
                FILE *meminfo = fopen("/proc/meminfo", "r");
                if (meminfo)
                {
                    char line[256];
                    while (fgets(line, sizeof(line), meminfo))
                    {
                        if (strncmp(line, "MemAvailable:", 13) == 0)
                        {
                            unsigned long kb = 0;
                            if (sscanf(line + 13, "%lu", &kb) == 1)
                            {
                                dev.total_memory_bytes = static_cast<size_t>(kb) * 1024 / (num_nodes > 0 ? num_nodes : 1);
                                dev.free_memory_bytes = dev.total_memory_bytes;
                            }
                            break;
                        }
                    }
                    fclose(meminfo);
                }
            }
        }
        else
        {
            // No NUMA node specified, get total system memory
            FILE *meminfo = fopen("/proc/meminfo", "r");
            if (meminfo)
            {
                char line[256];
                while (fgets(line, sizeof(line), meminfo))
                {
                    if (strncmp(line, "MemAvailable:", 13) == 0)
                    {
                        unsigned long kb = 0;
                        if (sscanf(line + 13, "%lu", &kb) == 1)
                        {
                            dev.total_memory_bytes = static_cast<size_t>(kb) * 1024;
                            dev.free_memory_bytes = dev.total_memory_bytes;
                        }
                        break;
                    }
                }
                fclose(meminfo);
            }
        }
#else
        // Fallback: assume 16 GB
        dev.total_memory_bytes = 16ULL * 1024 * 1024 * 1024;
        dev.free_memory_bytes = dev.total_memory_bytes;
#endif

        dev.supports_fp16 = false; // Depends on CPU features (AVX512-FP16)
        dev.supports_bf16 = true;  // Software emulation always available
        dev.supports_int8 = true;  // VNNI/DP4A support (detect at runtime)

        return dev;
    }

    // ============================================================================
    // CUDA Device Enumeration (DEPRECATED - Phase 3)
    // ============================================================================
    // GPU Device Enumeration (Phase 6: Separate compilation units)
    // ============================================================================
    // CUDA and ROCm enumeration moved to separate files to avoid header conflicts:
    //   - CUDAEnumeration.cu (CUDA runtime headers only)
    //   - ROCmEnumeration.cpp (HIP runtime headers only, compiled with hipcc)
    // This enables heterogeneous multi-GPU (NVIDIA + AMD in same binary).
    // ============================================================================

    // Wrapper functions that call into separate compilation units
    static std::vector<ComputeDevice> enumerate_cuda_devices()
    {
#ifdef HAVE_CUDA
        return cuda_enumeration::enumerate_cuda_devices();
#else
        return {};
#endif
    }

    static std::vector<ComputeDevice> enumerate_rocm_devices()
    {
#ifdef HAVE_ROCM
        return rocm_enumeration::enumerate_rocm_devices();
#else
        return {};
#endif
    }

    // ============================================================================
    // Vulkan Device Enumeration (DEPRECATED - Phase 3)
    // ============================================================================
    // Vulkan support is currently stubbed out.
    // ============================================================================

#if 0 // Vulkan enumeration disabled (Phase 3)
#ifdef HAVE_VULKAN
    static std::vector<ComputeDevice> enumerate_vulkan_devices()
    {
        std::vector<ComputeDevice> devices;

        // Create Vulkan instance
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Llaminar";
        app_info.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
        app_info.pEngineName = "Llaminar";
        app_info.engineVersion = VK_MAKE_VERSION(2, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
        {
            return devices; // Vulkan not available
        }

        // Enumerate physical devices
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

        if (device_count == 0)
        {
            vkDestroyInstance(instance, nullptr);
            return devices;
        }

        std::vector<VkPhysicalDevice> physical_devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());

        for (uint32_t i = 0; i < device_count; ++i)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_devices[i], &props);

            VkPhysicalDeviceMemoryProperties mem_props;
            vkGetPhysicalDeviceMemoryProperties(physical_devices[i], &mem_props);

            ComputeDevice dev;
            dev.type = ComputeBackendType::GPU_VULKAN;
            dev.name = std::string(props.deviceName);
            dev.device_id = i;
            dev.compute_capability = VK_VERSION_MAJOR(props.apiVersion) * 10 +
                                     VK_VERSION_MINOR(props.apiVersion);

            // Sum up device-local memory heaps
            dev.total_memory_bytes = 0;
            for (uint32_t j = 0; j < mem_props.memoryHeapCount; ++j)
            {
                if (mem_props.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                {
                    dev.total_memory_bytes += mem_props.memoryHeaps[j].size;
                }
            }
            dev.free_memory_bytes = dev.total_memory_bytes; // Approximate

            // Vulkan feature support (query via extensions)
            dev.supports_fp16 = true;  // VK_KHR_shader_float16_int8
            dev.supports_bf16 = false; // Limited BF16 support in Vulkan
            dev.supports_int8 = true;  // VK_KHR_shader_float16_int8

            devices.push_back(dev);
        }

        vkDestroyInstance(instance, nullptr);
        return devices;
    }
#else
    static std::vector<ComputeDevice> enumerate_vulkan_devices()
    {
        return {}; // Vulkan not available
    }
#endif
#endif // #if 0 - Vulkan enumeration disabled (Phase 3)

    // Replacement stub (always returns empty)
    static std::vector<ComputeDevice> enumerate_vulkan_devices()
    {
        return {}; // Vulkan enumeration moved to IBackend (Phase 3)
    }

    // ============================================================================
    // DeviceManager Implementation
    // ============================================================================

    namespace
    {
        // Read CPU model name per socket from /proc/cpuinfo
        // Returns socket_id -> model name (different sockets may have different CPUs)
        std::map<int, std::string> read_cpu_model_names_per_socket()
        {
            std::map<int, std::string> result;
            std::ifstream cpuinfo("/proc/cpuinfo");
            if (!cpuinfo.is_open())
                return result;

            int current_physical_id = -1;
            std::string current_model;
            std::string line;
            while (std::getline(cpuinfo, line))
            {
                if (line.compare(0, 11, "physical id") == 0)
                {
                    auto pos = line.find(':');
                    if (pos != std::string::npos)
                        current_physical_id = std::atoi(line.c_str() + pos + 1);
                }
                else if (line.compare(0, 10, "model name") == 0)
                {
                    auto pos = line.find(':');
                    if (pos != std::string::npos)
                    {
                        std::string name = line.substr(pos + 1);
                        auto start = name.find_first_not_of(" \t");
                        if (start != std::string::npos)
                            name = name.substr(start);
                        current_model = name;
                    }
                }
                else if (line.empty() || line[0] == '\n')
                {
                    // End of a CPU block
                    if (current_physical_id >= 0 && !current_model.empty())
                        result[current_physical_id] = current_model;
                    current_physical_id = -1;
                    current_model.clear();
                }
            }
            // Handle last block (file may not end with blank line)
            if (current_physical_id >= 0 && !current_model.empty())
                result[current_physical_id] = current_model;

            return result;
        }

        // Format memory size as "XX GB"
        std::string format_memory_gb(size_t bytes)
        {
            return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
        }

        // Format PCIe link info as "GenX xY ZZ GB/s"
        std::string format_pcie_link(const PCIeLinkInfo &pcie)
        {
            if (pcie.link_speed_gts <= 0)
                return "N/A";
            char buf[64];
            snprintf(buf, sizeof(buf), "Gen%d x%d %.0f GB/s",
                     pcie.pcie_gen, pcie.link_width, pcie.bandwidth_gbps());
            return buf;
        }

        // Format NUMA node(s) as a string
        std::string format_numa(int numa_node)
        {
            if (numa_node < 0)
                return "-";
            return std::to_string(numa_node);
        }

        // Strip vendor prefix from GPU name for cleaner display
        std::string strip_vendor_prefix(const std::string &name)
        {
            // Remove "NVIDIA " prefix
            if (name.compare(0, 7, "NVIDIA ") == 0)
                return name.substr(7);
            // Remove "AMD " prefix
            if (name.compare(0, 4, "AMD ") == 0)
                return name.substr(4);
            return name;
        }

        // Log a libfort table through the Logger (line by line)
        void log_table(const std::string &table_str)
        {
            std::istringstream stream(table_str);
            std::string line;
            while (std::getline(stream, line))
            {
                if (!line.empty())
                    LOG_INFO(line);
            }
        }

        // Build and log a GPU table for a given backend
        void log_gpu_table(const char *title,
                           const std::vector<ComputeDevice> &devices)
        {
            if (devices.empty())
                return;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row spanning all columns
            table << title << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(5);
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.row(0).set_cell_row_type(fort::row_type::header);

            // Header row
            table << "ID" << "Name" << "VRAM" << "PCIe Link" << "NUMA" << fort::endr;
            table.row(1).set_cell_row_type(fort::row_type::header);

            table.column(0).set_cell_text_align(fort::text_align::center);
            table.column(1).set_cell_text_align(fort::text_align::left);
            table.column(2).set_cell_text_align(fort::text_align::right);
            table.column(3).set_cell_text_align(fort::text_align::left);
            table.column(4).set_cell_text_align(fort::text_align::center);

            for (const auto &dev : devices)
            {
                // Build name with arch info embedded
                std::string display_name = strip_vendor_prefix(dev.name);

                // For CUDA devices, append SM version if not already in name
                if (dev.type == ComputeBackendType::GPU_CUDA)
                {
                    std::string sm = "SM " + std::to_string(dev.compute_capability / 10) + "." + std::to_string(dev.compute_capability % 10);
                    if (display_name.find("SM") == std::string::npos)
                        display_name += " (" + sm + ")";
                }

                table << std::to_string(dev.device_id)
                      << display_name
                      << format_memory_gb(dev.total_memory_bytes)
                      << format_pcie_link(dev.pcie)
                      << format_numa(dev.numa_node)
                      << fort::endr;
            }

            log_table(table.to_string());

            // Print degraded link warnings after the table
            for (const auto &dev : devices)
            {
                if (dev.pcie.degraded)
                {
                    int max_gen = (dev.pcie.max_speed_gts >= 64.0)   ? 6
                                  : (dev.pcie.max_speed_gts >= 32.0) ? 5
                                  : (dev.pcie.max_speed_gts >= 16.0) ? 4
                                  : (dev.pcie.max_speed_gts >= 8.0)  ? 3
                                  : (dev.pcie.max_speed_gts >= 5.0)  ? 2
                                                                     : 1;
                    char cap_buf[64];
                    snprintf(cap_buf, sizeof(cap_buf), "Gen%d x%d (%.1f GB/s)",
                             max_gen, dev.pcie.max_width,
                             // Compute max bandwidth
                             dev.pcie.max_speed_gts * dev.pcie.max_width * ((max_gen >= 3) ? (128.0 / 130.0) : 0.8) / 8.0);

                    const char *type_prefix = (dev.type == ComputeBackendType::GPU_CUDA) ? "cuda" : "rocm";
                    if (!dev.pcie.bottleneck_bdf.empty())
                    {
                        LOG_WARN("  ⚠ " << type_prefix << ":" << dev.device_id
                                        << " link degraded: " << format_pcie_link(dev.pcie)
                                        << " — capable of " << cap_buf
                                        << " (bottleneck at upstream bridge " << dev.pcie.bottleneck_bdf << ")");
                    }
                    else
                    {
                        LOG_WARN("  ⚠ " << type_prefix << ":" << dev.device_id
                                        << " link degraded: " << format_pcie_link(dev.pcie)
                                        << " — capable of " << cap_buf);
                    }
                }
            }
        }

        // Build and log a P2P access matrix table
        void log_p2p_table(const char *backend_name, const P2PMatrix &matrix)
        {
            const int n = matrix.device_count();
            if (n < 2)
                return;

            // Count P2P-enabled pairs
            int p2p_pairs = 0;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    if (i != j && matrix.can_access[i][j])
                        ++p2p_pairs;
            const int total_pairs = n * (n - 1);

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row
            std::ostringstream title;
            title << backend_name << " P2P Access (" << p2p_pairs << "/" << total_pairs << " pairs)";
            // First cell + N GPU columns
            table << title.str();
            for (int j = 0; j < n; ++j)
                table << "";
            table << fort::endr;
            table[0][0].set_cell_span(n + 1);
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.row(0).set_cell_row_type(fort::row_type::header);

            // Header row: empty + GPU0, GPU1, ...
            table << "";
            for (int j = 0; j < n; ++j)
                table << ("GPU" + std::to_string(matrix.device_ids[j]));
            table << fort::endr;
            table.row(1).set_cell_row_type(fort::row_type::header);

            // Set all columns to center
            for (int c = 0; c <= n; ++c)
                table.column(c).set_cell_text_align(fort::text_align::center);

            // Data rows
            for (int i = 0; i < n; ++i)
            {
                table << ("GPU" + std::to_string(matrix.device_ids[i]));
                for (int j = 0; j < n; ++j)
                {
                    if (i == j)
                        table << "-";
                    else
                        table << (matrix.can_access[i][j] ? "✓" : "✗");
                }
                table << fort::endr;
            }

            log_table(table.to_string());
        }
    } // anonymous namespace

    DeviceManager::DeviceManager() = default;
    DeviceManager::~DeviceManager() = default;

    void DeviceManager::initialize(int local_numa_node, bool log_inventory)
    {
        devices_.clear();
        contexts_.clear();
        local_numa_node_ = local_numa_node;

        // Log NUMA filtering mode
        if (local_numa_node >= 0)
        {
            LOG_DEBUG("[DeviceManager] Initializing with NUMA node " << local_numa_node << " filtering (MPI rank mode)");
        }
        else
        {
            LOG_DEBUG("[DeviceManager] Initializing without NUMA filtering (all devices visible)");
        }

        // Always enumerate CPU first (device index 0)
        // Pass NUMA node so memory reporting is NUMA-local
        auto cpu_dev = enumerate_cpu_device(local_numa_node >= 0 ? local_numa_node : 0);
        devices_.push_back(cpu_dev);

        const char *cpu_only_env = std::getenv("LLAMINAR_FORCE_CPU_ONLY_STARTUP");
        const bool force_cpu_only_startup = (cpu_only_env && std::atoi(cpu_only_env) != 0);

        // Selective backend skip: when we know the target backend, skip the other(s)
        // to avoid expensive GPU driver initialization (~250ms per CUDA device).
        const char *skip_cuda_env = std::getenv("LLAMINAR_SKIP_CUDA_STARTUP");
        const bool skip_cuda = (skip_cuda_env && std::atoi(skip_cuda_env) != 0);
        const char *skip_rocm_env = std::getenv("LLAMINAR_SKIP_ROCM_STARTUP");
        const bool skip_rocm = (skip_rocm_env && std::atoi(skip_rocm_env) != 0);

        // Enumerate GPUs with optional NUMA filtering
        std::vector<ComputeDevice> cuda_devices;
        std::vector<ComputeDevice> rocm_devices;
        std::vector<ComputeDevice> vulkan_devices;

        if (!force_cpu_only_startup)
        {
            if (!skip_cuda)
                cuda_devices = enumerate_cuda_devices();
            else
                LOG_INFO("[DeviceManager] Skipping CUDA enumeration (LLAMINAR_SKIP_CUDA_STARTUP=1)");

            if (!skip_rocm)
                rocm_devices = enumerate_rocm_devices();
            else
                LOG_INFO("[DeviceManager] Skipping ROCm enumeration (LLAMINAR_SKIP_ROCM_STARTUP=1)");

            vulkan_devices = enumerate_vulkan_devices();
        }
        else
        {
            LOG_INFO("[DeviceManager] CPU-only startup fast-path active: skipping GPU enumeration");
        }

        // Filter CUDA devices by NUMA affinity
        if (local_numa_node >= 0)
        {
            std::vector<ComputeDevice> filtered_cuda;
            for (auto &dev : cuda_devices)
            {
                auto gpu_info = NUMATopology::getCUDAGPUNUMANode(dev.device_id);
                dev.numa_node = gpu_info.numa_node;

                if (NUMATopology::isGPULocalToProcess(gpu_info.numa_node, local_numa_node))
                {
                    filtered_cuda.push_back(dev);
                    LOG_DEBUG("[DeviceManager] Including CUDA GPU " << dev.device_id
                                                                    << " (NUMA node " << gpu_info.numa_node << ", " << gpu_info.detection_method << ")");
                }
                else
                {
                    LOG_DEBUG("[DeviceManager] Filtering out CUDA GPU " << dev.device_id
                                                                        << " (on NUMA node " << gpu_info.numa_node << ", process on node " << local_numa_node << ")");
                }
            }
            cuda_devices = filtered_cuda;
        }
        else
        {
            // No filtering, but still populate NUMA info for logging
            for (auto &dev : cuda_devices)
            {
                auto gpu_info = NUMATopology::getCUDAGPUNUMANode(dev.device_id);
                dev.numa_node = gpu_info.numa_node;
            }
        }

#ifdef HAVE_ROCM
        // Filter ROCm devices by NUMA affinity
        if (local_numa_node >= 0)
        {
            std::vector<ComputeDevice> filtered_rocm;
            for (auto &dev : rocm_devices)
            {
                auto gpu_info = NUMATopology::getROCmGPUNUMANode(dev.device_id);
                dev.numa_node = gpu_info.numa_node;

                if (NUMATopology::isGPULocalToProcess(gpu_info.numa_node, local_numa_node))
                {
                    filtered_rocm.push_back(dev);
                    LOG_DEBUG("[DeviceManager] Including ROCm GPU " << dev.device_id
                                                                    << " (NUMA node " << gpu_info.numa_node << ")");
                }
                else
                {
                    LOG_DEBUG("[DeviceManager] Filtering out ROCm GPU " << dev.device_id
                                                                        << " (on NUMA node " << gpu_info.numa_node << ")");
                }
            }
            rocm_devices = filtered_rocm;
        }
        else
        {
            // No filtering, populate NUMA info
            for (auto &dev : rocm_devices)
            {
                auto gpu_info = NUMATopology::getROCmGPUNUMANode(dev.device_id);
                dev.numa_node = gpu_info.numa_node;
            }
        }
#endif

        // Vulkan devices: not filtered (NUMA affinity unknown)
        for (auto &dev : vulkan_devices)
        {
            dev.numa_node = -1; // Unknown
        }

        devices_.insert(devices_.end(), cuda_devices.begin(), cuda_devices.end());
        devices_.insert(devices_.end(), rocm_devices.begin(), rocm_devices.end());
        devices_.insert(devices_.end(), vulkan_devices.begin(), vulkan_devices.end());

        // Resize contexts_ vector to match devices
        contexts_.resize(devices_.size(), nullptr);

        // ====================================================================
        // Device inventory tables (libfort) — print once on first init
        // ====================================================================
        if (log_inventory && !inventory_logged_)
        {
            inventory_logged_ = true;

            // --- CPU table (per-socket detail) ---
            {
                auto cpu_models = read_cpu_model_names_per_socket();

                // Per-socket info
                struct SocketInfo
                {
                    int socket_id = -1;
                    int numa_node = -1;
                    std::string model_name = "Unknown CPU";
                    std::vector<int> physical_cores; // First thread of each core
                    std::vector<int> ht_threads;     // Sibling threads (HT)
                    size_t memory_bytes = 0;
                };
                std::vector<SocketInfo> sockets;

                int num_numa = (numa_available() >= 0) ? numa_num_configured_nodes() : 1;
                int total_cpus = sysconf(_SC_NPROCESSORS_ONLN);

                // Map each CPU to its socket and detect HT siblings
                // socket_id -> { core_id -> vector<cpu_id> }
                std::map<int, std::map<int, std::vector<int>>> socket_core_map;
                std::map<int, int> socket_to_numa; // socket -> NUMA node

                for (int cpu = 0; cpu < total_cpus; ++cpu)
                {
                    char path[256];
                    int pkg = 0, core = 0;

                    snprintf(path, sizeof(path),
                             "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
                    FILE *f = fopen(path, "r");
                    if (f)
                    {
                        if (fscanf(f, "%d", &pkg) != 1)
                            pkg = 0;
                        fclose(f);
                    }

                    snprintf(path, sizeof(path),
                             "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
                    f = fopen(path, "r");
                    if (f)
                    {
                        if (fscanf(f, "%d", &core) != 1)
                            core = 0;
                        fclose(f);
                    }

                    socket_core_map[pkg][core].push_back(cpu);

                    // Detect NUMA node for this CPU
                    if (socket_to_numa.find(pkg) == socket_to_numa.end())
                    {
                        for (int n = 0; n < num_numa; ++n)
                        {
                            char numa_path[256];
                            snprintf(numa_path, sizeof(numa_path),
                                     "/sys/devices/system/cpu/cpu%d/node%d", cpu, n);
                            struct stat st;
                            if (stat(numa_path, &st) == 0)
                            {
                                socket_to_numa[pkg] = n;
                                break;
                            }
                        }
                    }
                }

                // Build SocketInfo for each socket
                for (auto &[pkg, cores] : socket_core_map)
                {
                    SocketInfo si;
                    si.socket_id = pkg;
                    si.numa_node = (socket_to_numa.count(pkg) > 0) ? socket_to_numa[pkg] : pkg;
                    if (cpu_models.count(pkg) > 0)
                        si.model_name = cpu_models[pkg];

                    for (auto &[core_id, cpus] : cores)
                    {
                        // Sort CPUs: lowest is the physical core, rest are HT siblings
                        std::sort(cpus.begin(), cpus.end());
                        si.physical_cores.push_back(cpus[0]);
                        for (size_t i = 1; i < cpus.size(); ++i)
                            si.ht_threads.push_back(cpus[i]);
                    }
                    std::sort(si.physical_cores.begin(), si.physical_cores.end());
                    std::sort(si.ht_threads.begin(), si.ht_threads.end());

                    // Per-NUMA memory
                    if (numa_available() >= 0 && si.numa_node >= 0)
                    {
                        long long sz = numa_node_size64(si.numa_node, nullptr);
                        if (sz > 0)
                            si.memory_bytes = static_cast<size_t>(sz);
                    }

                    sockets.push_back(std::move(si));
                }

                // Sort sockets by ID
                std::sort(sockets.begin(), sockets.end(),
                          [](const SocketInfo &a, const SocketInfo &b)
                          { return a.socket_id < b.socket_id; });

                // Fallback: if no sockets detected, create a single entry
                if (sockets.empty())
                {
                    SocketInfo si;
                    si.socket_id = 0;
                    si.numa_node = 0;

                    for (const auto &dev : devices_)
                        if (dev.type == ComputeBackendType::CPU)
                        {
                            si.memory_bytes = dev.total_memory_bytes;
                            break;
                        }

                    sockets.push_back(si);
                }

                // Helper: format a sorted vector of ints as compact ranges (e.g., "0-27, 56-83")
                auto format_cpu_ranges = [](const std::vector<int> &cpus) -> std::string
                {
                    if (cpus.empty())
                        return "-";
                    std::ostringstream oss;
                    int start = cpus[0], prev = cpus[0];
                    for (size_t i = 1; i <= cpus.size(); ++i)
                    {
                        if (i < cpus.size() && cpus[i] == prev + 1)
                        {
                            prev = cpus[i];
                        }
                        else
                        {
                            if (oss.tellp() > 0)
                                oss << ", ";
                            if (start == prev)
                                oss << start;
                            else
                                oss << start << "-" << prev;
                            if (i < cpus.size())
                            {
                                start = cpus[i];
                                prev = cpus[i];
                            }
                        }
                    }
                    return oss.str();
                };

                // Compute total memory for title
                size_t total_mem = 0;
                for (const auto &s : sockets)
                    total_mem += s.memory_bytes;

                // Build table
                const bool has_ht = !sockets.empty() && !sockets[0].ht_threads.empty();
                const int num_cols = has_ht ? 7 : 6;

                fort::utf8_table cpu_table;
                cpu_table.set_border_style(FT_DOUBLE2_STYLE);

                // Title row
                cpu_table << "CPU";
                for (int c = 1; c < num_cols; ++c)
                    cpu_table << "";
                cpu_table << fort::endr;
                cpu_table[0][0].set_cell_span(num_cols);
                cpu_table[0][0].set_cell_text_align(fort::text_align::center);
                cpu_table.row(0).set_cell_row_type(fort::row_type::header);

                // Header row
                if (has_ht)
                    cpu_table << "Socket" << "Processor" << "NUMA" << "Physical Cores" << "HT Threads" << "Cores" << "Memory" << fort::endr;
                else
                    cpu_table << "Socket" << "Processor" << "NUMA" << "Cores" << "Core Count" << "Memory" << fort::endr;
                cpu_table.row(1).set_cell_row_type(fort::row_type::header);

                // Column alignments
                cpu_table.column(0).set_cell_text_align(fort::text_align::center);
                cpu_table.column(1).set_cell_text_align(fort::text_align::left);
                cpu_table.column(2).set_cell_text_align(fort::text_align::center);
                if (has_ht)
                {
                    cpu_table.column(3).set_cell_text_align(fort::text_align::left);
                    cpu_table.column(4).set_cell_text_align(fort::text_align::left);
                    cpu_table.column(5).set_cell_text_align(fort::text_align::center);
                    cpu_table.column(6).set_cell_text_align(fort::text_align::right);
                }
                else
                {
                    cpu_table.column(3).set_cell_text_align(fort::text_align::left);
                    cpu_table.column(4).set_cell_text_align(fort::text_align::center);
                    cpu_table.column(5).set_cell_text_align(fort::text_align::right);
                }

                // Data rows
                for (const auto &s : sockets)
                {
                    std::string cores_str = std::to_string(s.physical_cores.size()) + "c/" + std::to_string(s.physical_cores.size() + s.ht_threads.size()) + "t";

                    if (has_ht)
                    {
                        cpu_table << std::to_string(s.socket_id)
                                  << s.model_name
                                  << std::to_string(s.numa_node)
                                  << format_cpu_ranges(s.physical_cores)
                                  << format_cpu_ranges(s.ht_threads)
                                  << cores_str
                                  << format_memory_gb(s.memory_bytes)
                                  << fort::endr;
                    }
                    else
                    {
                        cpu_table << std::to_string(s.socket_id)
                                  << s.model_name
                                  << std::to_string(s.numa_node)
                                  << format_cpu_ranges(s.physical_cores)
                                  << cores_str
                                  << format_memory_gb(s.memory_bytes)
                                  << fort::endr;
                    }
                }

                // Total row
                if (sockets.size() > 1)
                {
                    int total_phys = 0, total_threads = 0;
                    for (const auto &s : sockets)
                    {
                        total_phys += static_cast<int>(s.physical_cores.size());
                        total_threads += static_cast<int>(s.physical_cores.size() + s.ht_threads.size());
                    }
                    std::string total_cores_str = std::to_string(total_phys) + "c/" + std::to_string(total_threads) + "t total";

                    cpu_table << fort::separator;
                    if (has_ht)
                        cpu_table << "" << "Total" << "" << "" << "" << total_cores_str << format_memory_gb(total_mem) << fort::endr;
                    else
                        cpu_table << "" << "Total" << "" << "" << total_cores_str << format_memory_gb(total_mem) << fort::endr;
                }

                log_table(cpu_table.to_string());
            }

            // --- NVIDIA CUDA GPU table ---
            if (!cuda_devices.empty())
            {
                log_gpu_table("NVIDIA CUDA GPUs", cuda_devices);
            }

            // --- AMD ROCm GPU table ---
            if (!rocm_devices.empty())
            {
                log_gpu_table("AMD ROCm GPUs", rocm_devices);
            }

            // --- P2P access matrices ---
            p2p_matrices_.clear();

#ifdef HAVE_CUDA
            if (cuda_devices.size() > 1)
            {
                auto cuda_p2p = cuda_enumeration::query_p2p_matrix(cuda_devices);
                log_p2p_table("CUDA", cuda_p2p);
                p2p_matrices_.push_back(std::move(cuda_p2p));
            }
#endif

#ifdef HAVE_ROCM
            if (rocm_devices.size() > 1)
            {
                auto rocm_p2p = rocm_enumeration::query_p2p_matrix(rocm_devices);
                log_p2p_table("ROCm", rocm_p2p);
                p2p_matrices_.push_back(std::move(rocm_p2p));
            }
#endif

        } // end if (!inventory_logged_)

        // Note: All devices are available for heterogeneous work distribution.
        // CPU may get 0% prefill but significant decode work due to memory bandwidth.
        // GPU kernels are under development; CPU backend is currently primary.
    }

    std::shared_ptr<ComputeContext> DeviceManager::create_context(size_t device_index)
    {
        if (device_index >= devices_.size())
        {
            LOG_ERROR("[DeviceManager] Invalid device index: " << device_index << "");
            return nullptr;
        }

        // Check if context already exists
        if (device_index < contexts_.size() && contexts_[device_index])
        {
            return contexts_[device_index]; // Reuse existing context
        }

        // Ensure contexts_ vector is large enough
        if (device_index >= contexts_.size())
        {
            contexts_.resize(device_index + 1, nullptr);
        }

        // Create concrete context based on backend type
        std::shared_ptr<ComputeContext> ctx;
        const auto &device = devices_[device_index];

        switch (device.type)
        {
        case ComputeBackendType::CPU:
            ctx = std::make_shared<CPUComputeContext>();
            break;

#if 0 // GPU context creation disabled (Phase 3)
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
        {
            auto cuda_ctx = std::make_shared<CUDAComputeContext>();
            if (cudaSetDevice(device.device_id) != cudaSuccess)
            {
                LOG_ERROR("[DeviceManager] Failed to set CUDA device "
                          << device.device_id << "");
                return nullptr;
            }

            cudaStream_t stream;
            if (cudaStreamCreate(&stream) != cudaSuccess)
            {
                LOG_ERROR("[DeviceManager] Failed to create CUDA stream");
                return nullptr;
            }
            cuda_ctx->stream = stream;
            cuda_ctx->device_id = device.device_id;

            cublasHandle_t cublas_handle;
            if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[DeviceManager] Failed to create cuBLAS handle");
                cudaStreamDestroy(stream);
                return nullptr;
            }
            cublasSetStream(cublas_handle, stream);
            cuda_ctx->cublas_handle = cublas_handle;
            ctx = cuda_ctx;
            break;
        }
#endif

#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
        {
            auto rocm_ctx = std::make_shared<ROCmComputeContext>();
            if (hipSetDevice(device.device_id) != hipSuccess)
            {
                LOG_ERROR("[DeviceManager] Failed to set ROCm device "
                          << device.device_id << "");
                return nullptr;
            }

            hipStream_t stream;
            if (hipStreamCreate(&stream) != hipSuccess)
            {
                LOG_ERROR("[DeviceManager] Failed to create HIP stream");
                return nullptr;
            }
            rocm_ctx->stream = stream;
            rocm_ctx->device_id = device.device_id;

            hipblasHandle_t hipblas_handle;
            if (hipblasCreate(&hipblas_handle) != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[DeviceManager] Failed to create hipBLAS handle");
                hipStreamDestroy(stream);
                return nullptr;
            }
            hipblasSetStream(hipblas_handle, stream);
            rocm_ctx->hipblas_handle = hipblas_handle;
            ctx = rocm_ctx;
            break;
        }
#endif

#ifdef HAVE_VULKAN
        case ComputeBackendType::GPU_VULKAN:
            // TODO: Vulkan context initialization
            ctx = std::make_shared<VulkanComputeContext>();
            LOG_ERROR("[DeviceManager] Vulkan context creation not fully implemented");
            break;
#else
        case ComputeBackendType::GPU_VULKAN:
            LOG_ERROR("[DeviceManager] Vulkan not available in this build");
            return nullptr;
#endif
#endif // #if 0 - GPU context creation disabled (Phase 3)

        // GPU context creation now handled by IBackend (Phase 3)
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
        case ComputeBackendType::GPU_VULKAN:
            LOG_ERROR("[DeviceManager] GPU context creation moved to IBackend (Phase 3)");
            return nullptr;

        default:
            LOG_ERROR("[DeviceManager] Unknown backend type");
            return nullptr;
        }

        // Cache context
        contexts_[device_index] = ctx;

        LOG_DEBUG("[DeviceManager] Created context for device " << device_index
                                                                << " (" << backend_type_name(device.type) << ")");

        return ctx;
    }

    int DeviceManager::find_device(ComputeBackendType type, int device_id) const
    {
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == type && devices_[i].device_id == device_id)
            {
                return static_cast<int>(i);
            }
        }
        return -1; // Not found
    }

    bool DeviceManager::deviceExists(const DeviceId &device) const
    {
        if (!device.is_valid())
        {
            return false;
        }

        if (device.is_cpu())
        {
            // CPU is always available
            return true;
        }

        // Map DeviceId type to ComputeBackendType
        ComputeBackendType backend_type;
        switch (device.type)
        {
        case DeviceType::CUDA:
            backend_type = ComputeBackendType::GPU_CUDA;
            break;
        case DeviceType::ROCm:
            backend_type = ComputeBackendType::GPU_ROCM;
            break;
        default:
            return false;
        }

        return find_device(backend_type, device.ordinal) >= 0;
    }

    bool DeviceManager::deviceExists(const GlobalDeviceAddress &device, bool strict_numa) const
    {
        if (device.isCPU())
        {
            if (!strict_numa)
            {
                return true;
            }

            if (local_numa_node_ >= 0)
            {
                return device.numa_node == local_numa_node_;
            }

            const int total_numa = NUMATopology::getNumNUMANodes();
            return device.numa_node >= 0 && device.numa_node < std::max(1, total_numa);
        }

        ComputeBackendType backend_type;
        switch (device.device_type)
        {
        case DeviceType::CUDA:
            backend_type = ComputeBackendType::GPU_CUDA;
            break;
        case DeviceType::ROCm:
            backend_type = ComputeBackendType::GPU_ROCM;
            break;
        default:
            return false;
        }

        for (const auto &dev : devices_)
        {
            if (dev.type != backend_type || dev.device_id != device.device_ordinal)
            {
                continue;
            }

            if (!strict_numa)
            {
                return true;
            }

            if (dev.numa_node < 0 || dev.numa_node == device.numa_node)
            {
                return true;
            }
        }

        return false;
    }

    std::string DeviceManager::availableDevicesString() const
    {
        std::string result;
        for (const auto &dev : devices_)
        {
            if (!result.empty())
            {
                result += ", ";
            }
            switch (dev.type)
            {
            case ComputeBackendType::CPU:
                result += "CPU";
                break;
            case ComputeBackendType::GPU_CUDA:
                result += "CUDA:" + std::to_string(dev.device_id);
                if (!dev.name.empty())
                {
                    result += " (" + dev.name + ")";
                }
                break;
            case ComputeBackendType::GPU_ROCM:
                result += "ROCm:" + std::to_string(dev.device_id);
                if (!dev.name.empty())
                {
                    result += " (" + dev.name + ")";
                }
                break;
            default:
                result += "Unknown:" + std::to_string(dev.device_id);
                break;
            }
        }
        return result.empty() ? "(none)" : result;
    }

    size_t DeviceManager::select_device(size_t estimated_memory_bytes)
    {
        // NOTE: This method selects a PRIMARY device for legacy single-device code paths.
        // For heterogeneous tensor-parallel execution, use all devices via devices() and
        // let the work distributor allocate work based on device capabilities.
        //
        // Current behavior: Returns CPU (index 0) since GPU kernels are not yet implemented.
        // Future behavior: Will be deprecated in favor of multi-device orchestration.

        if (devices_.empty())
        {
            LOG_ERROR("[DeviceManager] No devices available");
            return 0;
        }

        // For now, always use CPU backend since GPU kernels are under development
        // All devices remain available for future heterogeneous work distribution
        LOG_DEBUG("[DeviceManager] Using CPU backend (GPU kernels under development)");
        return 0; // CPU is always device 0
    }

    bool DeviceManager::has_gpu() const
    {
        for (const auto &dev : devices_)
        {
            if (dev.type == ComputeBackendType::GPU_CUDA ||
                dev.type == ComputeBackendType::GPU_ROCM ||
                dev.type == ComputeBackendType::GPU_VULKAN)
            {
                return true;
            }
        }
        return false;
    }

    int DeviceManager::cuda_device_count() const
    {
        int count = 0;
        for (const auto &dev : devices_)
        {
            if (dev.type == ComputeBackendType::GPU_CUDA)
            {
                count++;
            }
        }
        return count;
    }

    int DeviceManager::rocm_device_count() const
    {
        int count = 0;
        for (const auto &dev : devices_)
        {
            if (dev.type == ComputeBackendType::GPU_ROCM)
            {
                count++;
            }
        }
        return count;
    }

    std::vector<size_t> DeviceManager::get_devices_by_type(ComputeBackendType type) const
    {
        std::vector<size_t> result;
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == type)
            {
                result.push_back(i);
            }
        }
        return result;
    }

    int DeviceManager::get_device_id_for_type(ComputeBackendType type, int local_index) const
    {
        int count = 0;
        for (const auto &dev : devices_)
        {
            if (dev.type == type)
            {
                if (count == local_index)
                {
                    return dev.device_id;
                }
                count++;
            }
        }
        return -1; // Not found
    }

    // ============================================================================
    // CPUComputeContext Implementation
    // ============================================================================

    struct CPUComputeContext::Impl
    {
        // Note: The typed kernels don't implement ITensorRoPE/ITensorSwiGLU interfaces,
        // so we use void* and cast when needed. This is a transitional pattern
        // that will be cleaned up when the compute context is refactored.
        std::unique_ptr<CPURoPEKernelT<ActivationPrecision::FP32>> rope_kernel;
        std::unique_ptr<CPUSoftmaxKernelT<ActivationPrecision::FP32>> softmax_kernel;
        std::unique_ptr<CPUSwiGLUKernelT<ActivationPrecision::FP32>> swiglu_kernel;
    };

    CPUComputeContext::CPUComputeContext()
        : pimpl_(std::make_unique<Impl>())
    {
    }

    CPUComputeContext::~CPUComputeContext() = default;

    void *CPUComputeContext::allocate(size_t bytes)
    {
        return std::malloc(bytes);
    }

    void CPUComputeContext::free(void *ptr)
    {
        std::free(ptr);
    }

    void CPUComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        std::memcpy(dst, src, bytes); // CPU-to-CPU copy
    }

    void CPUComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        std::memcpy(dst, src, bytes); // CPU-to-CPU copy
    }

    ITensorRoPE *CPUComputeContext::get_rope_kernel()
    {
        // Note: The typed kernels no longer implement ITensorRoPE interface.
        // This method is deprecated and returns nullptr.
        // Use KernelFactory::createRoPE() instead.
        return nullptr;
    }

    ITensorSoftmax *CPUComputeContext::get_softmax_kernel()
    {
        // Note: The typed kernels no longer implement ITensorSoftmax interface.
        // This method is deprecated and returns nullptr.
        // Use KernelFactory::createSoftmax() instead.
        return nullptr;
    }

    ITensorSwiGLU *CPUComputeContext::get_swiglu_kernel()
    {
        // Note: The typed kernels no longer implement ITensorSwiGLU interface.
        // This method is deprecated and returns nullptr.
        // Use KernelFactory::createSwiGLU() instead.
        return nullptr;
    }

    // ============================================================================
    // CUDAComputeContext Implementation (DEPRECATED - Phase 3)
    // ============================================================================
    // GPU context implementations moved to IBackend interface.
    // See backends/cuda/CUDABackend.cu for new CUDA implementation.
    // ============================================================================

#if 0 // CUDA context methods disabled (Phase 3)
#ifdef HAVE_CUDA
    void *CUDAComputeContext::allocate(size_t bytes)
    {
        void *ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDA] Failed to allocate " << bytes << " bytes: "
                                                   << cudaGetErrorString(err) << "");
            return nullptr;
        }
        return ptr;
    }

    void CUDAComputeContext::free(void *ptr)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    void CUDAComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDA] copy_to_device failed: " << cudaGetErrorString(err) << "");
        }
    }

    void CUDAComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDA] copy_from_device failed: " << cudaGetErrorString(err) << "");
        }
    }

    void CUDAComputeContext::synchronize()
    {
        if (stream)
        {
            cudaStreamSynchronize(stream);
        }
        else
        {
            cudaDeviceSynchronize();
        }
    }
#endif

    // ============================================================================
    // ROCmComputeContext Implementation (DEPRECATED - Phase 3)
    // ============================================================================
    // GPU context implementations moved to IBackend interface.
    // See backends/rocm/ROCmBackend.cpp for new ROCm implementation.
    // ============================================================================

#ifdef HAVE_ROCM
    void *ROCmComputeContext::allocate(size_t bytes)
    {
        void *ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCm] Failed to allocate " << bytes << " bytes");
            return nullptr;
        }
        return ptr;
    }

    void ROCmComputeContext::free(void *ptr)
    {
        if (ptr)
        {
            hipFree(ptr);
        }
    }

    void ROCmComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCm] copy_to_device failed");
        }
    }

    void ROCmComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCm] copy_from_device failed");
        }
    }

    void ROCmComputeContext::synchronize()
    {
        if (stream)
        {
            hipStreamSynchronize(stream);
        }
        else
        {
            hipDeviceSynchronize();
        }
    }
#endif

    // ============================================================================
    // VulkanComputeContext Implementation (Stub)
    // ============================================================================

#ifdef HAVE_VULKAN
    void *VulkanComputeContext::allocate(size_t bytes)
    {
        // TODO: Vulkan buffer allocation
        LOG_ERROR("[Vulkan] allocate() not yet implemented");
        return nullptr;
    }

    void VulkanComputeContext::free(void *ptr)
    {
        // TODO: Vulkan buffer deallocation
    }

    void VulkanComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        // TODO: Vulkan staging buffer upload
        LOG_ERROR("[Vulkan] copy_to_device() not yet implemented");
    }

    void VulkanComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        // TODO: Vulkan staging buffer download
        LOG_ERROR("[Vulkan] copy_from_device() not yet implemented");
    }

    void VulkanComputeContext::synchronize()
    {
        // TODO: Vulkan queue submit + wait
    }
#endif
#endif // #if 0 - GPU context methods disabled (Phase 3)

} // namespace llaminar2
