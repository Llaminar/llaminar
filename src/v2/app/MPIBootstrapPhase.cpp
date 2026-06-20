/**
 * @file MPIBootstrapPhase.cpp
 * @brief Pre-MPI topology planning, NUMA resolution, and MPI self-launch
 */

#include "app/MPIBootstrapPhase.h"
#include "backends/ComputeBackend.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "utils/NUMATopology.h"
#include <omp.h>
#include <sched.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <filesystem>

namespace llaminar2
{

    // =========================================================================
    // Static helpers (extracted from anonymous namespace in Main.cpp)
    // =========================================================================

    std::vector<int> MPIBootstrapPhase::parseCpuList(const std::string &cpulist)
    {
        std::vector<int> cpus;
        std::stringstream ss(cpulist);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            if (token.empty())
            {
                continue;
            }
            auto dash = token.find('-');
            if (dash == std::string::npos)
            {
                cpus.push_back(std::stoi(token));
                continue;
            }
            int start = std::stoi(token.substr(0, dash));
            int end = std::stoi(token.substr(dash + 1));
            if (end < start)
            {
                std::swap(start, end);
            }
            for (int cpu = start; cpu <= end; ++cpu)
            {
                cpus.push_back(cpu);
            }
        }
        std::sort(cpus.begin(), cpus.end());
        cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
        return cpus;
    }

    int MPIBootstrapPhase::detectCpuNumaNode(int cpu)
    {
        const std::string cpu_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/";
        for (int node = 0; node < 256; ++node)
        {
            if (std::filesystem::exists(cpu_path + "node" + std::to_string(node)))
            {
                return node;
            }
        }
        return -1;
    }

    std::set<int> MPIBootstrapPhase::resolveInferenceNUMANodes(
        const OrchestrationConfig &config,
        const DeviceManager &dm,
        const CPUTopology &cpu_topology)
    {
        const auto &devices = dm.devices();
        std::set<int> numa_nodes;

        auto device_numa = [&](const GlobalDeviceAddress &addr) -> int
        {
            if (addr.isCPU())
                return addr.numa_node;

            ComputeBackendType bt = addr.isCUDA() ? ComputeBackendType::GPU_CUDA
                                                  : ComputeBackendType::GPU_ROCM;
            for (const auto &dev : devices)
            {
                if (dev.type == bt && dev.device_id == addr.device_ordinal)
                    return dev.numa_node;
            }
            return -1;
        };

        // CPU modes
        if (config.cpu_global_tp_all_local)
        {
            for (int n = 0; n < cpu_topology.numa_nodes; ++n)
                numa_nodes.insert(n);
            return numa_nodes;
        }

        if (config.device_for_this_rank.has_value() &&
            config.device_for_this_rank->isCPU() &&
            config.device_for_this_rank_numa_explicit)
        {
            numa_nodes.insert(config.device_for_this_rank->numa_node);
            return numa_nodes;
        }

        // Explicit GPU modes
        if (config.device_for_this_rank.has_value() && config.device_for_this_rank->isGPU())
        {
            int n = device_numa(*config.device_for_this_rank);
            if (n >= 0)
                numa_nodes.insert(n);
        }

        for (const auto &[rank_id, addr] : config.device_map)
        {
            int n = device_numa(addr);
            if (n >= 0)
                numa_nodes.insert(n);
        }

        for (const auto &addr : config.tp_devices)
        {
            int n = device_numa(addr);
            if (n >= 0)
                numa_nodes.insert(n);
        }

        for (const auto &dom : config.domain_definitions)
        {
            for (const auto &addr : dom.devices)
            {
                int n = device_numa(addr);
                if (n >= 0)
                    numa_nodes.insert(n);
            }
        }

        if (!numa_nodes.empty())
            return numa_nodes;

        // Simple TP (auto-pick GPUs)
        if (config.tp_degree > 1)
        {
            std::vector<const ComputeDevice *> gpus;
            for (const auto &dev : devices)
            {
                if (dev.type == ComputeBackendType::GPU_CUDA ||
                    dev.type == ComputeBackendType::GPU_ROCM)
                    gpus.push_back(&dev);
            }

            int count = std::min(static_cast<int>(gpus.size()), config.tp_degree);
            for (int i = 0; i < count; ++i)
            {
                if (gpus[i]->numa_node >= 0)
                    numa_nodes.insert(gpus[i]->numa_node);
            }

            if (!numa_nodes.empty())
                return numa_nodes;
        }

        // Default: NUMA 0
        numa_nodes.insert(0);
        return numa_nodes;
    }

    int MPIBootstrapPhase::physicalRepresentativeForCpu(int cpu)
    {
        std::ifstream siblings_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        if (!siblings_file.is_open())
        {
            return cpu;
        }

        std::string siblings;
        std::getline(siblings_file, siblings);
        auto sibling_cpus = parseCpuList(siblings);
        if (sibling_cpus.empty())
        {
            return cpu;
        }
        return *std::min_element(sibling_cpus.begin(), sibling_cpus.end());
    }

    bool MPIBootstrapPhase::verifyStartupThreadAffinity(int required_numa,
                                                        bool require_physical_only,
                                                        std::string &details)
    {
        const int max_threads = std::max(1, omp_get_max_threads());
        std::vector<int> observed_cpu(max_threads, -1);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            if (tid >= 0 && tid < static_cast<int>(observed_cpu.size()))
            {
                observed_cpu[tid] = sched_getcpu();
            }
        }

        std::set<int> numa_nodes;
        std::unordered_map<int, int> physical_core_usage;

        for (int cpu : observed_cpu)
        {
            if (cpu < 0)
            {
                details = "failed to sample CPU for one or more OpenMP threads";
                return false;
            }

            const int numa = detectCpuNumaNode(cpu);
            if (numa >= 0)
            {
                numa_nodes.insert(numa);
            }

            if (require_physical_only)
            {
                const int rep = physicalRepresentativeForCpu(cpu);
                physical_core_usage[rep] += 1;
            }
        }

        if (!numa_nodes.empty())
        {
            if (numa_nodes.size() > 1)
            {
                std::ostringstream oss;
                oss << "threads are spread across NUMA nodes";
                bool first = true;
                oss << " [";
                for (int n : numa_nodes)
                {
                    if (!first)
                    {
                        oss << ",";
                    }
                    first = false;
                    oss << n;
                }
                oss << "]";
                details = oss.str();
                return false;
            }

            if (required_numa >= 0 && *numa_nodes.begin() != required_numa)
            {
                std::ostringstream oss;
                oss << "threads are pinned to NUMA " << *numa_nodes.begin()
                    << " but required NUMA is " << required_numa;
                details = oss.str();
                return false;
            }
        }

        if (require_physical_only)
        {
            for (const auto &[rep, count] : physical_core_usage)
            {
                if (count > 1)
                {
                    std::ostringstream oss;
                    oss << "multiple OpenMP threads share physical core representative CPU " << rep;
                    details = oss.str();
                    return false;
                }
            }
        }

        details = "ok";
        return true;
    }

    void MPIBootstrapPhase::listDevices()
    {
        auto &dm = DeviceManager::instance();
        dm.initialize(-1, false); // No tables — printed post-MPI via InventoryPrinter

        const auto &devices = dm.devices();

        LOG_DEBUG("\n=== Available Devices ===\n\n");
        for (size_t i = 0; i < devices.size(); ++i)
        {
            const auto &dev = devices[i];
            LOG_DEBUG("Device " << i << ": ");

            switch (dev.type)
            {
            case ComputeBackendType::CPU:
                LOG_DEBUG("CPU");
                break;
            case ComputeBackendType::GPU_CUDA:
                LOG_DEBUG("GPU (CUDA) - " << dev.name);
                break;
            case ComputeBackendType::GPU_ROCM:
                LOG_DEBUG("GPU (ROCm) - " << dev.name);
                break;
            case ComputeBackendType::GPU_VULKAN:
                LOG_DEBUG("GPU (Vulkan) - " << dev.name);
                break;
            case ComputeBackendType::GPU_METAL:
                LOG_DEBUG("GPU (Metal) - " << dev.name);
                break;
            }

            if (dev.total_memory_bytes > 0)
            {
                double total_gb = dev.total_memory_bytes / (1024.0 * 1024.0 * 1024.0);
                double free_gb = dev.free_memory_bytes / (1024.0 * 1024.0 * 1024.0);
                LOG_DEBUG(" (" << total_gb << " GB total, " << free_gb << " GB free)");
            }

            LOG_DEBUG("\n");
        }

        LOG_DEBUG("\n");
    }

    // =========================================================================
    // Main execute method
    // =========================================================================

    BootstrapResult MPIBootstrapPhase::execute(const OrchestrationConfig &config,
                                               int argc, char *argv[])
    {
        // Detect CPU topology (needed for both bootstrap and runtime config)
        CPUTopology cpu_topology = MPIBootstrap::detectCPUTopology();

        // Detect if we're already running under MPI
        MPIEnvironmentInfo mpi_env = MPIBootstrap::detectMPIEnvironment();

        // If already in MPI context or bootstrap disabled, continue to runtime init
        if (mpi_env.is_mpi_process || config.mpi_no_bootstrap)
        {
            return {BootstrapResult::Action::CONTINUE, 0};
        }

        // =================================================================
        // Phase 1: Full device enumeration (no NUMA filtering)
        // =================================================================
        auto &dm = DeviceManager::instance();
        dm.initialize(-1, false); // No tables — printed post-MPI via InventoryPrinter

        // =================================================================
        // Phase 2: Determine inference NUMA nodes
        // =================================================================
        const std::set<int> inference_numas =
            resolveInferenceNUMANodes(config, dm, cpu_topology);

        {
            std::string nlist;
            for (auto it = inference_numas.begin(); it != inference_numas.end(); ++it)
            {
                if (it != inference_numas.begin())
                    nlist += ",";
                nlist += std::to_string(*it);
            }
            LOG_DEBUG("[Main] Inference NUMA nodes: {" << nlist << "} ("
                                                       << inference_numas.size() << " of "
                                                       << cpu_topology.numa_nodes << " total)");
        }

        // =================================================================
        // Phase 3: Build MPI launch configuration
        // =================================================================
        MPILaunchConfig launch_config = MPIBootstrap::getDefaultConfig(cpu_topology);

        const bool cpu_intent_bootstrap =
            config.cpu_global_tp_all_local ||
            (config.device_for_this_rank.has_value() && config.device_for_this_rank->isCPU());

        setenv("LLAMINAR_SELF_BOOTSTRAPPED", "1", 1);

        if (cpu_intent_bootstrap)
        {
            setenv("LLAMINAR_FORCE_CPU_ONLY_STARTUP", "1", 1);
        }
        else
        {
            unsetenv("LLAMINAR_FORCE_CPU_ONLY_STARTUP");
        }

        const bool use_tuned_mpi_profile =
            (config.mpi_profile == MPIProfile::TUNED) ||
            (config.mpi_profile == MPIProfile::AUTO && cpu_intent_bootstrap);

        if (use_tuned_mpi_profile)
        {
            if (config.mpi_procs <= 0)
            {
                launch_config.num_procs = std::max(1, cpu_topology.numa_nodes);
            }
            launch_config.omp_threads_per_rank = std::max(1, cpu_topology.cores_per_socket);
            launch_config.omp_places = "cores";
            launch_config.omp_proc_bind = "close";
            launch_config.bind_to_socket = true;
            launch_config.map_by_socket = true;
            launch_config.use_physical_cores = true;

            LOG_DEBUG("[Main] MPI bootstrap profile: tuned"
                      << (config.mpi_profile == MPIProfile::AUTO ? " (auto-selected for CPU intent)" : ""));
        }

        // Override with user-specified values
        if (config.mpi_procs > 0)
        {
            launch_config.num_procs = config.mpi_procs;
        }
        if (!config.hostfile.empty())
        {
            launch_config.hostfile = config.hostfile;
        }
        launch_config.report_bindings = config.mpi_verbose || (config.verbose_level > 0);
        launch_config.verbose = config.mpi_verbose;
        launch_config.oversubscribe = config.mpi_oversubscribe;

        // CPU shorthand semantics
        if (config.cpu_global_tp_all_local)
        {
            if (config.mpi_procs <= 0)
            {
                launch_config.num_procs = std::max(1, cpu_topology.numa_nodes);
            }
            launch_config.omp_threads_per_rank = std::max(1, cpu_topology.cores_per_socket);
            launch_config.omp_places = "cores";
            launch_config.omp_proc_bind = "close";
            LOG_DEBUG("[Main] CPU shorthand detected: launching " << launch_config.num_procs
                                                                  << " rank(s) for CPU GLOBAL TP across local NUMA nodes");
        }
        else if (config.device_for_this_rank.has_value() &&
                 config.device_for_this_rank->isCPU() &&
                 config.device_for_this_rank_numa_explicit)
        {
            if (config.mpi_procs <= 0)
            {
                launch_config.num_procs = 1;
            }

            launch_config.omp_threads_per_rank = std::max(1, cpu_topology.cores_per_socket);
            launch_config.omp_places = "cores";
            launch_config.omp_proc_bind = "close";
            launch_config.bind_to_socket = false;
            launch_config.map_by_socket = false;
            launch_config.cpu_set = MPIBootstrap::getPhysicalCpuSetForNumaNode(config.device_for_this_rank->numa_node);

            if (launch_config.cpu_set.empty())
            {
                launch_config.cpu_set = MPIBootstrap::getCpuSetForNumaNode(config.device_for_this_rank->numa_node);
            }

            if (!launch_config.cpu_set.empty())
            {
                LOG_INFO("[Main] Explicit CPU NUMA target " << config.device_for_this_rank->numa_node
                                                            << " detected; applying MPI cpu-set='" << launch_config.cpu_set << "'");
            }
            else
            {
                LOG_WARN("[Main] Explicit CPU NUMA target " << config.device_for_this_rank->numa_node
                                                            << " requested, but cpu-set lookup failed; relying on launcher defaults");
            }
        }

        // GPU NUMA affinity
        if (launch_config.cpu_set.empty() && !cpu_intent_bootstrap)
        {
            if (config.mpi_procs <= 0)
            {
                launch_config.num_procs = static_cast<int>(inference_numas.size());
            }

            if (inference_numas.size() == 1)
            {
                const int target_numa = *inference_numas.begin();
                std::string cpu_set = MPIBootstrap::getPhysicalCpuSetForNumaNode(target_numa);
                if (cpu_set.empty())
                    cpu_set = MPIBootstrap::getCpuSetForNumaNode(target_numa);

                if (!cpu_set.empty())
                {
                    launch_config.bind_to_socket = false;
                    launch_config.map_by_socket = false;
                    launch_config.cpu_set = cpu_set;
                    launch_config.omp_threads_per_rank = std::max(1, cpu_topology.cores_per_socket);
                    launch_config.omp_places = "cores";
                    launch_config.omp_proc_bind = "close";

                    LOG_DEBUG("[Main] All target devices on NUMA node " << target_numa
                                                                        << "; binding MPI process to cpu-set='" << cpu_set << "'");
                }
            }
            else if (inference_numas.size() > 1)
            {
                launch_config.bind_to_socket = true;
                launch_config.map_by_socket = true;
                launch_config.omp_threads_per_rank = std::max(1, cpu_topology.cores_per_socket);
                launch_config.omp_places = "cores";
                launch_config.omp_proc_bind = "close";

                std::string nodes_str;
                for (auto it = inference_numas.begin(); it != inference_numas.end(); ++it)
                {
                    if (it != inference_numas.begin())
                        nodes_str += ",";
                    nodes_str += std::to_string(*it);
                }
                LOG_INFO("[Main] Inference spans NUMA nodes {" << nodes_str
                                                               << "}; launching " << launch_config.num_procs
                                                               << " process(es) with socket binding");
            }
        }

        // Handle dry-run
        if (config.mpi_dry_run)
        {
            MPIBootstrap::printConfigurationSummary(cpu_topology, launch_config, mpi_env);
            std::cout << "Dry run requested - exiting without launching MPI.\n";
            return {BootstrapResult::Action::EXIT, 0};
        }

        // Print config summary before launch
        MPIBootstrap::printConfigurationSummary(cpu_topology, launch_config, mpi_env);

        // Self-launch via mpirun (replaces current process)
        int result = MPIBootstrap::selfLaunchMPI(argc, argv, launch_config, cpu_topology);

        // If we get here, exec failed
        LOG_ERROR("Failed to self-launch via mpirun");
        return {BootstrapResult::Action::EXIT, result};
    }

} // namespace llaminar2
