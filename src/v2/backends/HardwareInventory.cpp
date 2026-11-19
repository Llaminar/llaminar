/**
 * @file HardwareInventory.cpp
 * @brief Hardware inventory detection implementation
 *
 * Detects CPU sockets (model, cores, HT, memory), GPU devices, and P2P
 * access matrices from sysfs, /proc/cpuinfo, and GPU driver APIs.
 *
 * Native discovery borrows the calling thread; its exact CPU binding must be
 * restored before returning or unwinding, without touching existing workers.
 *
 * @author David Sanftenberg
 * @date 2026-04-11
 *
 * Cached CPU placement keeps OS NUMA/PU identities separate from MPI logical
 * core indices. Discovery publishes exact per-node facts once; launch and
 * admission reject missing or contradictory observations without rediscovery.
 */

#include "HardwareInventory.h"
#include "CPUCacheInventory.h"
#include "GPUEnumeration.h"
#include "HostMemoryCapacity.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "../utils/ScopedCallerThreadAffinity.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <sys/stat.h>
#include <unistd.h>
#include <numa.h>

namespace llaminar2
{

    void assignNonNUMAHostMemory(std::vector<CPUSocketInfo> &sockets,
                                 const HostMemoryCapacityObservation &system)
    {
        const auto with_memory = std::count_if(sockets.begin(), sockets.end(),
            [](const CPUSocketInfo &socket) { return socket.memory_bytes > 0; });
        if (sockets.empty() || with_memory == static_cast<std::ptrdiff_t>(sockets.size()))
            return;
        if (with_memory > 0)
            throw std::runtime_error("Host memory topology is partial: NUMA memory observed for " +
                std::to_string(with_memory) + " of " + std::to_string(sockets.size()) + " CPU packages");
        if (sockets.size() > 1)
            throw std::runtime_error("Host exposes " + std::to_string(sockets.size()) +
                " CPU packages but no NUMA memory topology; per-package memory cannot be attributed");
        if (!system.valid())
            throw std::runtime_error("Host memory capacity is unobservable: no NUMA node memory and no /proc/meminfo total");
        sockets.front().memory_bytes = system.total_bytes;
        sockets.front().available_memory_bytes = system.admission_available_bytes;
        LOG_INFO("[HardwareInventory] No NUMA memory topology; attributing system memory ("
                 << system.total_bytes / (1024 * 1024) << " MB) to the single CPU package");
    }

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

            assignNonNUMAHostMemory(sockets, observeSystemMemoryCapacity());
            return sockets;
        }
    } // anonymous namespace

    // =========================================================================
    // HardwareInventory::detect()
    // =========================================================================

    ComputeDevice HardwareInventory::cpuDevice(int numa_node) const
    {
        if (cpu_numa_devices.empty())
            throw std::invalid_argument("CPU NUMA observation is incomplete; package labels cannot establish ownership");
        for (const auto &socket : cpu_sockets)
            if (socket.available_memory_bytes > socket.memory_bytes)
                throw std::invalid_argument("CPU free memory exceeds observed physical capacity");
        for (const auto &[node, cpu] : cpu_numa_devices)
            if (node < 0 || cpu.type != ComputeBackendType::CPU || cpu.numa_node != node ||
                cpu.total_memory_bytes == 0 || cpu.free_memory_bytes > cpu.total_memory_bytes || cpu.compute_units <= 0)
                throw std::invalid_argument("CPU NUMA observation has contradictory physical geometry");
        if (numa_node >= 0)
        {
            const auto found = cpu_numa_devices.find(numa_node);
            if (found == cpu_numa_devices.end())
                throw std::invalid_argument("CPU NUMA locality absent from hardware observation");
            return found->second;
        }
        if (numa_node != -1)
            throw std::invalid_argument("CPU NUMA locality absent from hardware observation");
        ComputeDevice combined{};
        combined.type = ComputeBackendType::CPU; combined.numa_node = -1;
        combined.supports_bf16 = combined.supports_int8 = true;
        for (const auto &[node, cpu] : cpu_numa_devices)
        {
            if (cpu.free_memory_bytes > cpu.total_memory_bytes ||
                cpu.total_memory_bytes > std::numeric_limits<size_t>::max() - combined.total_memory_bytes ||
                cpu.free_memory_bytes > std::numeric_limits<size_t>::max() - combined.free_memory_bytes ||
                cpu.compute_units > std::numeric_limits<int>::max() - combined.compute_units)
                throw std::invalid_argument("CPU NUMA observations have invalid aggregate geometry");
            combined.total_memory_bytes += cpu.total_memory_bytes;
            combined.free_memory_bytes += cpu.free_memory_bytes;
            combined.compute_units += cpu.compute_units;
            if (combined.name.empty()) combined.name = cpu.name;
        }
        bool complete = true;
        for (const auto &socket : cpu_sockets)
        {
            const auto cache = cpu_last_level_cache_bytes.find(socket.socket_id);
            if (cache == cpu_last_level_cache_bytes.end() || !cache->second) complete = false;
            else
            {
                if (cache->second > std::numeric_limits<size_t>::max() - combined.last_level_cache_bytes)
                    throw std::overflow_error("CPU endpoint aggregate cache overflow");
                combined.last_level_cache_bytes += cache->second;
            }
        }
        if (!complete) combined.last_level_cache_bytes = 0;
        return combined;
    }

    HardwareInventory HardwareInventory::detect()
    {
        const ScopedCallerThreadAffinity preserve_caller;
        HardwareInventory hw;
        const auto &startup = debugEnv().backend_startup;

        // --- CPU sockets ---
        hw.cpu_sockets = detect_cpu_sockets();
        hw.mpi_core_placement = NUMATopology::observeMPICorePlacement();
        hw.cpu_execution = CPUExecutionGeometry::local();
        for (const auto &socket : hw.cpu_sockets)
            hw.cpu_last_level_cache_bytes.emplace(socket.socket_id, observeCPUSharedCacheBytes(socket.physical_cores));
        // Package records remain package records. Exact per-NUMA rank resources
        // come from original CPU membership and the same canonical memory/cache
        // observers, once at discovery; later admission never probes them again.
        std::map<int,std::vector<int>> numa_cpus;
        for (const auto &core : hw.mpi_core_placement)
            numa_cpus[core.os_numa_node].insert(numa_cpus[core.os_numa_node].end(),
                core.os_cpu_ids.begin(), core.os_cpu_ids.end());
        for (const auto &[node, cpus] : numa_cpus)
        {
            const auto memory = observeNUMAMemoryCapacity(node);
            if (!memory.valid() || cpus.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("Canonical CPU NUMA observation has invalid physical geometry");
            ComputeDevice cpu{};
            cpu.type = ComputeBackendType::CPU; cpu.numa_node = node;
            cpu.supports_bf16 = cpu.supports_int8 = true;
            cpu.total_memory_bytes = memory.total_bytes;
            cpu.free_memory_bytes = memory.admission_available_bytes;
            cpu.compute_units = static_cast<int>(cpus.size());
            cpu.last_level_cache_bytes = observeCPUSharedCacheBytes(cpus);
            for (const auto &socket : hw.cpu_sockets)
                if (std::any_of(cpus.begin(), cpus.end(), [&](int id) {
                        return std::find(socket.physical_cores.begin(), socket.physical_cores.end(), id) != socket.physical_cores.end() ||
                            std::find(socket.ht_threads.begin(), socket.ht_threads.end(), id) != socket.ht_threads.end();
                    }))
                { cpu.name = socket.model_name; break; }
            hw.cpu_numa_devices.emplace(node, std::move(cpu));
        }

        // --- GPU devices ---
#ifdef HAVE_CUDA
        if (startup.cudaEnabled())
        {
            hw.cuda_devices = cuda_enumeration::enumerate_cuda_devices();
            // NUMA and P2P queries also enter the CUDA runtime, so they remain
            // inside the same authoritative discovery-policy branch.
            for (auto &dev : hw.cuda_devices)
                dev.numa_node = cuda_enumeration::get_cuda_device_numa_node(dev.device_id);
            if (hw.cuda_devices.size() >= 2)
                hw.cuda_p2p = cuda_enumeration::query_p2p_matrix(hw.cuda_devices);
        }
#endif

#ifdef HAVE_ROCM
        if (startup.rocmEnabled())
        {
            hw.rocm_devices = rocm_enumeration::enumerate_rocm_devices();
            // NUMA and P2P queries can open KFD/runtime state and therefore
            // must never escape the ROCm discovery-policy branch.
            for (auto &dev : hw.rocm_devices)
                dev.numa_node = rocm_enumeration::get_rocm_device_numa_node(dev.device_id);
            if (hw.rocm_devices.size() >= 2)
                hw.rocm_p2p = rocm_enumeration::query_p2p_matrix(hw.rocm_devices);
        }
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
