/**
 * @file MPIBootstrapPhase.cpp
 * @brief Pre-MPI topology planning, NUMA resolution, and MPI self-launch
 *
 * Parsed device declarations first constrain the canonical startup authority.
 * This precedes every inventory query in both the launcher and its MPI children:
 * naming a CPU domain must not create foreign CUDA/HIP contexts merely to choose
 * a NUMA placement. The same intent then selects the CPU MPI tuning profile.
 * An explicit CPU node requires a real physical-core set before self-launch;
 * unresolved locality never disables the default full-team MPI binding.
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
#include <map>
#include <filesystem>
#include <stdexcept>

namespace llaminar2
{

    BootstrapDeviceIntent MPIBootstrapPhase::classifyDeviceIntent(
        const OrchestrationConfig &config)
    {
        if ((!config.topology_string.empty() ||
             !config.topology_file_path.empty()) && !config.topology_tree)
        {
            // An unparsed tree is not evidence of a CPU-only deployment.
            return BootstrapDeviceIntent::Automatic;
        }
        BootstrapDeviceIntent intent = config.cpu_global_tp_all_local
                                          ? BootstrapDeviceIntent::CpuOnly
                                          : BootstrapDeviceIntent::Automatic;
        const auto include = [&](const GlobalDeviceAddress &address)
        {
            if (!address.isCPU())
                intent = BootstrapDeviceIntent::Accelerator;
            else if (intent == BootstrapDeviceIntent::Automatic)
                intent = BootstrapDeviceIntent::CpuOnly;
        };
        if (config.device_for_this_rank)
            include(*config.device_for_this_rank);
        for (const auto &[rank, address] : config.device_map)
            include(address);
        for (const auto &address : config.tp_devices)
            include(address);
        for (const auto &domain : config.domain_definitions)
            for (const auto &address : domain.devices)
                include(address);
        if (config.topology_tree)
            for (const auto *leaf : config.topology_tree->root.leafDevices())
                include(leaf->device);
        if (const auto &overlay = config.moe_routed_expert_plan;
            overlay && overlay->usesExpertOverlayAuthority())
        {
            // Include both ownership axes. Looking only at the continuation
            // device would accidentally suppress GPU experts in a CPU/GPU tier.
            for (const auto &domain : overlay->domains)
                for (const auto &address : domain.participants)
                    include(address);
            for (const auto &domain : overlay->dense_domains)
                for (const auto &address : domain.participants)
                    include(address);
        }
        if (intent == BootstrapDeviceIntent::Automatic)
        {
            // A CPU-only hard filter is compute intent even before a device
            // has been selected. Honor it before backend discovery so remote
            // CPU admission does not initialize excluded GPU runtimes.
            const AutomaticOrchestrationRequest automatic(config.automatic_planning);
            if (!automatic.allows(DeviceType::CUDA) && !automatic.allows(DeviceType::ROCm))
                return BootstrapDeviceIntent::CpuOnly;
        }
        return intent;
    }

    BootstrapDeviceIntent MPIBootstrapPhase::installDeviceStartupIntent(
        const OrchestrationConfig &config)
    {
        const auto intent = classifyDeviceIntent(config);
        if (intent == BootstrapDeviceIntent::CpuOnly)
        {
            // Export before self-launch and refresh before in-process discovery.
            // Never erase a caller's operational exclusions for another mode.
            if (setenv("LLAMINAR_FORCE_CPU_ONLY_STARTUP", "1", 1) != 0)
                throw std::runtime_error("Could not export CPU-only startup intent");
            mutableDebugEnv().backend_startup.reload();
        }
        return intent;
    }

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
            if (!config.device_for_this_rank->hasValidNuma())
                throw std::invalid_argument(
                    "Explicit CPU NUMA placement requires a non-negative node");
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

    MPILaunchConfig MPIBootstrapPhase::discoveryLaunchConfig(const OrchestrationConfig &config)
    {
        const bool local_auto_world = config.hostfile.empty() &&
            !config.execution_rank_selection && config.mpi_procs > 0 &&
            std::holds_alternative<AutomaticOrchestrationRequest>(
                resolveOrchestrationIntent(config));
        if (config.hostfile.empty() && !config.execution_rank_selection &&
            !local_auto_world)
            throw std::invalid_argument(
                "Discovery launch requires a hostfile, saved execution selection, or an automatic MPI world size");
        if (config.execution_rank_selection)
        {
            if (config.hostfile.empty() && config.mpi_procs <= 0)
                throw std::invalid_argument("Saved local execution selection requires its discovery MPI process count");
            if (config.mpi_procs > 0 && std::any_of(
                    config.execution_rank_selection->discoveryRanks().begin(),
                    config.execution_rank_selection->discoveryRanks().end(),
                    [&](int rank) { return rank >= config.mpi_procs; }))
                throw std::invalid_argument("Saved execution selection exceeds the requested discovery MPI process count");
        }
        MPILaunchConfig launch;
        launch.hostfile = config.hostfile;
        launch.num_procs = config.mpi_procs;
        launch.report_bindings = config.mpi_verbose || config.verbose_level > 0;
        launch.verbose = config.mpi_verbose;
        launch.oversubscribe = config.mpi_oversubscribe;
        // MPI maps/binds sockets using each host's hwloc discovery. Thread
        // counts and CPU indices are deliberately absent from this command.
        return launch;
    }

    void MPIBootstrapPhase::configureRequestedCPUThreads(const OrchestrationConfig &config)
    {
        if (config.n_threads > 0)
        {
            if (setenv("OMP_NUM_THREADS", std::to_string(config.n_threads).c_str(), 1) != 0)
                throw std::runtime_error("Cannot publish requested CPU thread count");
            omp_set_num_threads(config.n_threads);
        }
        // Scratch and service observations require the admitted team, not an
        // OpenMP runtime that silently shrinks it under external load.
        omp_set_dynamic(0);
    }

    void MPIBootstrapPhase::configureClusterRankThreads(const OrchestrationConfig &config)
    {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0)
            throw std::runtime_error("Cannot read MPI rank affinity for cluster worker admission");
        std::map<int, int> physical_cores;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &affinity))
                physical_cores.try_emplace(physicalRepresentativeForCpu(cpu), cpu);
        if (physical_cores.empty() || physical_cores.contains(-1))
            throw std::runtime_error("MPI cluster rank has no valid physical-core binding");
        MPILaunchConfig workers;
        workers.omp_threads_per_rank = config.n_threads > 0
            ? config.n_threads : static_cast<int>(physical_cores.size());
        workers.omp_places = "cores";
        workers.omp_proc_bind = "close";
        MPIBootstrap::configureOpenMPEnvironment(MPIBootstrap::detectCPUTopology(), workers);
        // libgomp may already have parsed its environment before main(). The
        // explicit runtime operation makes the observed team size authoritative.
        omp_set_dynamic(0);
        omp_set_num_threads(workers.omp_threads_per_rank);

        // libgomp has no runtime setter for OMP_PLACES. Bind its persistent
        // worker pool using the rank's own allowed CPU IDs instead of relying
        // on environment strings changed after the runtime was loaded. Select
        // an allowed sibling of each physical core, never an unavailable ID.
        std::vector<int> worker_cpus;
        for (const auto &[physical, allowed] : physical_cores)
            worker_cpus.push_back(allowed);
        int binding_failed = 0;
#pragma omp parallel reduction(| : binding_failed)
        {
            cpu_set_t worker;
            CPU_ZERO(&worker);
            CPU_SET(worker_cpus[static_cast<std::size_t>(omp_get_thread_num()) % worker_cpus.size()], &worker);
            binding_failed |= sched_setaffinity(0, sizeof(worker), &worker) != 0;
        }
        if (binding_failed)
            throw std::runtime_error("Cannot bind MPI cluster rank's OpenMP workers to their physical cores");
    }

    BootstrapResult MPIBootstrapPhase::execute(const OrchestrationConfig &config,
                                               int argc, char *argv[])
    {
        const auto device_intent = installDeviceStartupIntent(config);
        const bool automatic_multi_rank_discovery =
            device_intent == BootstrapDeviceIntent::Automatic &&
            config.mpi_procs > 1;
        // Detect CPU topology (needed for both bootstrap and runtime config)
        CPUTopology cpu_topology = MPIBootstrap::detectCPUTopology();

        // Detect if we're already running under MPI
        MPIEnvironmentInfo mpi_env = MPIBootstrap::detectMPIEnvironment();

        // If already in MPI context or bootstrap disabled, continue to runtime init
        if (mpi_env.is_mpi_process || config.mpi_no_bootstrap)
        {
            if (mpi_env.is_mpi_process &&
                (!config.hostfile.empty() || config.execution_rank_selection ||
                 automatic_multi_rank_discovery))
                configureClusterRankThreads(config);
            return {BootstrapResult::Action::CONTINUE, 0};
        }

        if (!config.hostfile.empty() || config.execution_rank_selection ||
            automatic_multi_rank_discovery)
        {
            const auto launch = discoveryLaunchConfig(config);
            if (config.mpi_dry_run)
            {
                for (const auto &argument : MPIBootstrap::buildMPIRunCommand(
                         argc, argv, launch, cpu_topology))
                    std::cout << argument << ' ';
                std::cout << '\n';
                return {BootstrapResult::Action::EXIT, 0};
            }
            // No unresolved auto plan can narrow the discovery namespace to
            // the launcher's NUMA-0 default. MPI starts the requested ranks;
            // their actual CPU/GPU inventory is gathered after initialization.
            const int result = MPIBootstrap::selfLaunchMPI(argc, argv, launch, cpu_topology);
            return {BootstrapResult::Action::EXIT, result < 0 ? 1 : result};
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
            device_intent == BootstrapDeviceIntent::CpuOnly;

        setenv("LLAMINAR_SELF_BOOTSTRAPPED", "1", 1);

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

            const int target_numa = config.device_for_this_rank->numa_node;
            const auto cpu_set = MPIBootstrap::getPhysicalCpuSetForNumaNode(target_numa);
            if (cpu_set.empty())
            {
                throw std::runtime_error(
                    "Explicit CPU NUMA target " + std::to_string(target_numa) +
                    " has no physical CPU set; refusing to launch with different affinity");
            }

            // Only replace socket mapping once the requested node is proven.
            // A NUMA node may be smaller than a socket: size its worker team
            // from these actual physical cores, never an unrelated socket sum.
            launch_config.cpu_set = cpu_set;
            launch_config.omp_threads_per_rank = static_cast<int>(parseCpuList(cpu_set).size());
            launch_config.omp_places = "cores";
            launch_config.omp_proc_bind = "close";
            launch_config.bind_to_socket = false;
            launch_config.map_by_socket = false;
            LOG_INFO("[Main] Explicit CPU NUMA target " << target_numa
                     << " detected; applying MPI cpu-set='" << cpu_set << "'");
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
