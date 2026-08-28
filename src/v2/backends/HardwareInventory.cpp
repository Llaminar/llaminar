/**
 * @file HardwareInventory.cpp
 * @brief Hardware inventory detection implementation
 *
 * Detects CPU sockets (model, cores, HT, memory), GPU devices, and P2P
 * access matrices from sysfs, /proc/cpuinfo, and GPU driver APIs.
 *
 * @author David Sanftenberg
 * @date 2026-04-11
 */

#include "HardwareInventory.h"
#include "GPUEnumeration.h"
#include "HostMemoryCapacity.h"
#include "../utils/Logger.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include <sys/stat.h>
#include <unistd.h>
#include <numa.h>

namespace llaminar2
{

    // =========================================================================
    // CPU detection helpers
    // =========================================================================

    namespace
    {
        /// Read CPU model name per socket from /proc/cpuinfo
        /// Returns socket_id -> model name (different sockets may have different CPUs)
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
                    if (current_physical_id >= 0 && !current_model.empty())
                        result[current_physical_id] = current_model;
                    current_physical_id = -1;
                    current_model.clear();
                }
            }
            // Handle last block
            if (current_physical_id >= 0 && !current_model.empty())
                result[current_physical_id] = current_model;

            return result;
        }

        /// Detect CPU sockets from sysfs topology
        std::vector<CPUSocketInfo> detect_cpu_sockets()
        {
            auto cpu_models = read_cpu_model_names_per_socket();

            int num_numa = (numa_available() >= 0) ? numa_num_configured_nodes() : 1;
            int total_cpus = sysconf(_SC_NPROCESSORS_ONLN);

            // socket_id -> { core_id -> vector<cpu_id> }
            std::map<int, std::map<int, std::vector<int>>> socket_core_map;
            std::map<int, int> socket_to_numa;

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

            std::vector<CPUSocketInfo> sockets;

            for (auto &[pkg, cores] : socket_core_map)
            {
                CPUSocketInfo si;
                si.socket_id = pkg;
                si.numa_node = (socket_to_numa.count(pkg) > 0) ? socket_to_numa[pkg] : pkg;
                if (cpu_models.count(pkg) > 0)
                    si.model_name = cpu_models[pkg];
                else
                    si.model_name = "Unknown CPU";

                for (auto &[core_id, cpus] : cores)
                {
                    std::sort(cpus.begin(), cpus.end());
                    si.physical_cores.push_back(cpus[0]);
                    for (size_t i = 1; i < cpus.size(); ++i)
                        si.ht_threads.push_back(cpus[i]);
                }
                std::sort(si.physical_cores.begin(), si.physical_cores.end());
                std::sort(si.ht_threads.begin(), si.ht_threads.end());

                /*
                 * Inventory and runtime preflight consume the same immutable
                 * observation.  It charges tmpfs pages while admitting
                 * reclaimable ordinary file cache, which is essential when a
                 * production parity corpus is staged into a RAM filesystem.
                 */
                if (si.numa_node >= 0)
                {
                    const auto memory =
                        observeNUMAMemoryCapacity(si.numa_node);
                    si.memory_bytes = memory.total_bytes;
                    si.available_memory_bytes =
                        memory.admission_available_bytes;
                }

                sockets.push_back(std::move(si));
            }

            std::sort(sockets.begin(), sockets.end(),
                      [](const CPUSocketInfo &a, const CPUSocketInfo &b)
                      { return a.socket_id < b.socket_id; });

            // Fallback: if nothing detected, create a single entry
            if (sockets.empty())
            {
                CPUSocketInfo si;
                si.socket_id = 0;
                si.numa_node = 0;
                si.model_name = "Unknown CPU";
                sockets.push_back(si);
            }

            return sockets;
        }
    } // anonymous namespace

    // =========================================================================
    // HardwareInventory::detect()
    // =========================================================================

    HardwareInventory HardwareInventory::detect()
    {
        HardwareInventory hw;

        // --- CPU sockets ---
        hw.cpu_sockets = detect_cpu_sockets();

        // --- GPU devices ---
#ifdef HAVE_CUDA
        hw.cuda_devices = cuda_enumeration::enumerate_cuda_devices();
        // Populate NUMA info for each CUDA device
        for (auto &dev : hw.cuda_devices)
            dev.numa_node = cuda_enumeration::get_cuda_device_numa_node(dev.device_id);
        if (hw.cuda_devices.size() >= 2)
            hw.cuda_p2p = cuda_enumeration::query_p2p_matrix(hw.cuda_devices);
#endif

#ifdef HAVE_ROCM
        hw.rocm_devices = rocm_enumeration::enumerate_rocm_devices();
        // Populate NUMA info for each ROCm device
        for (auto &dev : hw.rocm_devices)
            dev.numa_node = rocm_enumeration::get_rocm_device_numa_node(dev.device_id);
        if (hw.rocm_devices.size() >= 2)
            hw.rocm_p2p = rocm_enumeration::query_p2p_matrix(hw.rocm_devices);
#endif

        return hw;
    }

    // =========================================================================
    // Formatting helpers
    // =========================================================================

    std::string HardwareInventory::formatCpuRanges(const std::vector<int> &cpus)
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
    }

} // namespace llaminar2
