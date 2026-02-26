/**
 * @file Main.cpp
 * @brief Llaminar v2 entry point
 *
 * Clean implementation using OrchestrationRunner for:
 * - Device manager initialization
 * - Multi-GPU heterogeneous support
 * - Pipeline and tensor parallelism
 * - Direct kernel orchestration
 * - Self-bootstrap MPI support (auto-launches mpirun if not in MPI context)
 *
 * @author David Sanftenberg
 */

#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"
#include "utils/MPIBootstrap.h"
#include "utils/NUMATopology.h"
#include "utils/Sampler.h"
#include "utils/ChatUI.h"
#include "utils/ChatTemplate.h"
#include "utils/BenchmarkRunner.h"
#include "backends/ComputeBackend.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <iostream>
#include <climits>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <set>
#include <unordered_map>
#include <filesystem>

using namespace llaminar2;

namespace
{
    std::vector<int> parseCpuList(const std::string &cpulist)
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

    int detectCpuNumaNode(int cpu)
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

    /**
     * @brief Resolve which NUMA nodes need MPI processes for inference.
     *
     * Examines the OrchestrationConfig against the full (NUMA-unfiltered)
     * DeviceManager inventory and returns the set of NUMA nodes that will
     * participate in inference.
     *
     * Rules:
     *   - GPU devices requested via any CLI mode → NUMA nodes of those GPUs
     *   - `-d cpu` (global)  → all NUMA nodes
     *   - `-d cpu:N`         → NUMA node N
     *   - simple `-tp N` (auto-pick)  → NUMA nodes of the first N GPUs
     *   - default (no device flag)    → {0} (local NUMA)
     */
    std::set<int> resolveInferenceNUMANodes(
        const OrchestrationConfig &config,
        const DeviceManager &dm,
        const CPUTopology &cpu_topology)
    {
        const auto &devices = dm.devices();
        std::set<int> numa_nodes;

        // Helper: look up NUMA node for a GlobalDeviceAddress from the
        // already-enumerated DeviceManager inventory (avoids duplicate
        // sysfs / NVML calls).
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
            return -1; // unknown — will be caught at validation time
        };

        // ---- CPU modes ----
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

        // ---- Explicit GPU modes ----

        // --device (single device)
        if (config.device_for_this_rank.has_value() && config.device_for_this_rank->isGPU())
        {
            int n = device_numa(*config.device_for_this_rank);
            if (n >= 0)
                numa_nodes.insert(n);
        }

        // --device-map
        for (const auto &[rank_id, addr] : config.device_map)
        {
            int n = device_numa(addr);
            if (n >= 0)
                numa_nodes.insert(n);
        }

        // --tp-devices
        for (const auto &addr : config.tp_devices)
        {
            int n = device_numa(addr);
            if (n >= 0)
                numa_nodes.insert(n);
        }

        // --define-domain
        for (const auto &dom : config.domain_definitions)
        {
            for (const auto &addr : dom.devices)
            {
                int n = device_numa(addr);
                if (n >= 0)
                    numa_nodes.insert(n);
            }
        }

        // If any of the above populated the set, we're done.
        if (!numa_nodes.empty())
            return numa_nodes;

        // ---- Simple TP (auto-pick GPUs) ----
        if (config.tp_degree > 1)
        {
            // Collect all GPUs sorted by type then ordinal, pick first N.
            // This mirrors what the TP auto-picker would choose.
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

        // ---- Default: NUMA 0 ----
        numa_nodes.insert(0);
        return numa_nodes;
    }

    int physicalRepresentativeForCpu(int cpu)
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

    bool verifyStartupThreadAffinity(int required_numa,
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
}

void list_devices()
{
    auto &dm = DeviceManager::instance();
    // List devices without NUMA filtering (show all available devices)
    dm.initialize(-1);

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

/**
 * @brief Main entry point with MPI self-bootstrap support
 *
 * Execution flow:
 * 1. Parse arguments (lightweight, before MPI)
 * 2. Detect if running under MPI environment
 * 3. If not under MPI: self-launch via mpirun (replaces process)
 * 4. If under MPI: initialize MPI and proceed with inference
 */
int main(int argc, char *argv[])
{
    // Initialize logging from environment (LLAMINAR_LOG_LEVEL)
    // This happens before MPI so we can log bootstrap info
    initializeLogging();

    // Parse command-line arguments using OrchestrationConfigParser
    OrchestrationConfigParser parser;
    OrchestrationConfig config = parser.parseArgs(argc, argv);

    // Handle help early (no MPI needed)
    if (config.show_help)
    {
        std::cout << OrchestrationConfigParser::getHelpText() << std::endl;
        return 0;
    }

    // Handle list-devices early (no MPI needed)
    if (config.list_devices)
    {
        list_devices();
        return 0;
    }

    // ========================================================================
    // MPI Bootstrap Detection and Self-Launch
    // ========================================================================

    // Detect CPU topology (needed for both bootstrap and runtime config)
    CPUTopology cpu_topology = MPIBootstrap::detectCPUTopology();

    // Detect if we're already running under MPI
    MPIEnvironmentInfo mpi_env = MPIBootstrap::detectMPIEnvironment();

    // If NOT running under MPI and bootstrap is not disabled, self-launch via mpirun
    if (!mpi_env.is_mpi_process && !config.mpi_no_bootstrap)
    {
        // =================================================================
        // Phase 1: Full device enumeration (no NUMA filtering)
        //
        // Before MPI bootstrap, initialise DeviceManager with no NUMA
        // filter so we get a complete view of every device in the system.
        // This lets us plan topology and decide which NUMA nodes need
        // MPI processes.  The singleton is destroyed by execvp() when we
        // self-launch, so child processes will re-initialise cleanly.
        // =================================================================
        auto &dm = DeviceManager::instance();
        dm.initialize(-1); // full view — no NUMA filtering

        // =================================================================
        // Phase 2: Determine inference NUMA nodes
        //
        // Examine the user's config against the full device inventory to
        // decide which NUMA nodes will participate in inference.
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
            LOG_INFO("[Main] Inference NUMA nodes: {" << nlist << "} ("
                                                      << inference_numas.size() << " of "
                                                      << cpu_topology.numa_nodes << " total)");
        }

        // =================================================================
        // Phase 3: Build MPI launch configuration
        //
        // One MPI process per NUMA node that participates in inference.
        // Each process is bound to its NUMA node's CPU cores.
        // =================================================================
        MPILaunchConfig launch_config = MPIBootstrap::getDefaultConfig(cpu_topology);

        const bool cpu_intent_bootstrap =
            config.cpu_global_tp_all_local ||
            (config.device_for_this_rank.has_value() && config.device_for_this_rank->isCPU());

        // Signal to child processes that they were self-bootstrapped.
        // The child will skip redundant NUMA-filtered device enumeration
        // because topology planning already happened here.
        setenv("LLAMINAR_SELF_BOOTSTRAPPED", "1", 1);

        // For self-bootstrapped CPU-only runs, instruct child ranks to skip GPU
        // startup enumeration and factory registration. This avoids fixed startup
        // overhead that is irrelevant for CPU execution.
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
            // Tuned profile: one rank per local NUMA/socket by default, core-level OpenMP pinning,
            // and physical-core thread count per rank.
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

            LOG_INFO("[Main] MPI bootstrap profile: tuned"
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

        // CPU shorthand semantics:
        //   -d cpu   => CPU GLOBAL TP across all local NUMA nodes
        //   -d cpu:N => single-rank CPU execution pinned to NUMA node N
        if (config.cpu_global_tp_all_local)
        {
            if (config.mpi_procs <= 0)
            {
                launch_config.num_procs = std::max(1, cpu_topology.numa_nodes);
            }
            launch_config.omp_threads_per_rank = std::max(1, cpu_topology.cores_per_socket);
            launch_config.omp_places = "cores";
            launch_config.omp_proc_bind = "close";
            LOG_INFO("[Main] CPU shorthand detected: launching " << launch_config.num_procs
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

            // Explicit single-socket CPU placement should use physical cores
            // from that socket only (not all system cores / hyperthreads).
            launch_config.omp_threads_per_rank = std::max(1, cpu_topology.cores_per_socket);
            launch_config.omp_places = "cores";
            launch_config.omp_proc_bind = "close";

            // For explicit cpu:N in single-rank self-bootstrap, map-by socket alone
            // always places rank 0 on the first socket. Use an explicit MPI cpu-set
            // so cpu:1 resolves to NUMA node 1 as requested.
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

        // =================================================================
        // GPU NUMA affinity: bind MPI process(es) to the NUMA node(s)
        // where inference will run.  Uses the pre-bootstrap device
        // inventory instead of raw sysfs probing.
        //
        // Only applies when the CPU-specific paths above haven't
        // already configured a cpu-set.
        // =================================================================
        if (launch_config.cpu_set.empty() && !cpu_intent_bootstrap)
        {
            if (config.mpi_procs <= 0)
            {
                launch_config.num_procs = static_cast<int>(inference_numas.size());
            }

            if (inference_numas.size() == 1)
            {
                // All inference on a single NUMA node — pin to it.
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

                    LOG_INFO("[Main] All target devices on NUMA node " << target_numa
                                                                       << "; binding MPI process to cpu-set='" << cpu_set << "'");
                }
            }
            else if (inference_numas.size() > 1)
            {
                // Inference spans multiple NUMA nodes.
                // Use socket binding so mpirun naturally maps rank N → socket N.
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

        // Handle dry-run: print config and exit
        if (config.mpi_dry_run)
        {
            MPIBootstrap::printConfigurationSummary(cpu_topology, launch_config, mpi_env);
            std::cout << "Dry run requested - exiting without launching MPI.\n";
            return 0;
        }

        // Print config summary before launch
        MPIBootstrap::printConfigurationSummary(cpu_topology, launch_config, mpi_env);

        // Self-launch via mpirun (this replaces the current process)
        // If successful, this function does not return
        int result = MPIBootstrap::selfLaunchMPI(argc, argv, launch_config, cpu_topology);

        // If we get here, exec failed
        LOG_ERROR("Failed to self-launch via mpirun");
        return result;
    }

    // ========================================================================
    // MPI Runtime - We are running under mpirun
    // ========================================================================

    // OpenMPI vader's default CMA single-copy path can emit noisy
    // "Read -1, expected ..., errno = 1" warnings in containerized environments
    // where process_vm_readv is restricted. Keep user override if explicitly set.
    if (mpi_env.is_mpi_process && debugEnv().mpi_bootstrap.ompi_mca_btl_vader_single_copy_mechanism.empty())
    {
        setenv("OMPI_MCA_btl_vader_single_copy_mechanism", "none", 0);
    }

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Re-parse arguments (MPI_Init may modify argc/argv)
    config = parser.parseArgs(argc, argv);

    auto mpi_ctx = MPIContextFactory::global();

    // MPI runtime should rely on launcher-provided affinity/thread settings
    // (e.g. run_llaminar.sh + mpirun binding), not retune OpenMP internally.
    if (std::getenv("OMP_NUM_THREADS") == nullptr)
    {
        LOG_WARN("[Main] Running under MPI without OMP_NUM_THREADS set. For strict per-rank core pinning, "
                 "launch with canonical wrapper or set OMP_NUM_THREADS/OMP_PLACES/OMP_PROC_BIND in the launcher.");
    }

    // Detect NUMA node for MPI rank
    auto numa_info = NUMATopology::detectLocalNUMANode();

    const bool cpu_explicit_numa_mode =
        config.device_for_this_rank.has_value() &&
        config.device_for_this_rank->isCPU() &&
        config.device_for_this_rank_numa_explicit;

    const bool cpu_global_mode = config.cpu_global_tp_all_local;

    if (cpu_explicit_numa_mode || cpu_global_mode)
    {
        int required_numa = -1;
        if (cpu_explicit_numa_mode)
        {
            required_numa = config.device_for_this_rank->numa_node;
        }
        else if (numa_info.detection_succeeded)
        {
            required_numa = numa_info.local_numa_node;
        }

        std::string affinity_details;
        const bool affinity_ok = verifyStartupThreadAffinity(required_numa, true, affinity_details);

        const bool strict_assert = []()
        {
            const char *value = std::getenv("LLAMINAR_ASSERT_THREAD_AFFINITY");
            return value != nullptr && std::string(value) == "1";
        }();

        if (!affinity_ok)
        {
            const bool fail_fast = strict_assert || cpu_explicit_numa_mode;
            if (fail_fast)
            {
                LOG_ERROR("[Main] Startup thread affinity verification failed: " << affinity_details);
                LOG_ERROR("[Main] Fix launcher pinning (mpirun binding/cpu-set) or set LLAMINAR_ASSERT_THREAD_AFFINITY=0 to downgrade to warning");
                return 1;
            }

            LOG_WARN("[Main] Startup thread affinity verification warning: " << affinity_details);
        }
        else
        {
            LOG_INFO("[Main] Startup thread affinity verification passed");
        }
    }

    // CPU shorthand runtime semantics (inside MPI runtime):
    // convert `-d cpu` into explicit per-rank CPU mapping and GLOBAL TP config.
    if (config.cpu_global_tp_all_local)
    {
        int local_rank = 0;
        MPI_Comm local_comm = MPI_COMM_NULL;
        if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_ctx->rank(), MPI_INFO_NULL, &local_comm) == MPI_SUCCESS)
        {
            MPI_Comm_rank(local_comm, &local_rank);
            MPI_Comm_free(&local_comm);
        }

        config.device_mode = DeviceAssignmentMode::EXPLICIT;
        config.device_map.clear();
        config.device_map.emplace_back(mpi_ctx->rank(), GlobalDeviceAddress::cpu(local_rank));
        config.device_map_numa_explicit.clear();
        config.device_map_numa_explicit.emplace_back(mpi_ctx->rank(), true);

        if (config.tp_scope == TPScope::AUTO)
        {
            config.tp_scope = TPScope::GLOBAL;
        }
        if (config.tp_degree <= 1)
        {
            config.tp_degree = mpi_ctx->world_size();
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("[Main] CPU shorthand runtime mapping enabled: GLOBAL TP degree="
                     << config.tp_degree << ", world_size=" << mpi_ctx->world_size());
        }
    }

    // Set logger rank for log output
    Logger::getInstance().setRank(mpi_ctx->rank());

    if (mpi_ctx->rank() == 0)
    {
        LOG_DEBUG("=== NUMA Topology Detection ===");
    }

    LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] NUMA node: " << numa_info.local_numa_node
                       << " (detection: " << numa_info.detection_method << ")");

    if (!numa_info.detection_succeeded && mpi_ctx->rank() == 0)
    {
        LOG_WARN("NUMA detection failed, using fallback node 0. This may impact multi-socket performance.");
    }

    // ========================================================================
    // Initialize Device Manager
    //
    // When self-bootstrapped (LLAMINAR_SELF_BOOTSTRAPPED=1), the parent
    // process already did full topology planning and bound this MPI rank
    // to the correct NUMA node.  We initialise without NUMA filtering so
    // all devices remain reachable — the orchestrator picks the right ones.
    //
    // When externally launched (user ran mpirun), we fall back to NUMA
    // filtering (if no GPU is explicitly requested) so each rank only
    // sees devices on its own socket.
    // ========================================================================
    const bool self_bootstrapped = (std::getenv("LLAMINAR_SELF_BOOTSTRAPPED") != nullptr);

    int device_manager_numa_filter;
    if (self_bootstrapped)
    {
        // Parent already planned topology — no filtering needed.
        device_manager_numa_filter = -1;
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Self-bootstrapped: initialising DeviceManager without NUMA filtering");
    }
    else
    {
        // External MPI launch — apply NUMA filtering by default, but
        // disable it when any GPU is explicitly requested so cross-NUMA
        // targets remain reachable.
        device_manager_numa_filter = numa_info.local_numa_node;

        std::optional<GlobalDeviceAddress> mapped_device_for_rank;
        for (const auto &[mapped_rank, mapped_addr] : config.device_map)
        {
            if (mapped_rank == mpi_ctx->rank())
            {
                mapped_device_for_rank = mapped_addr;
                break;
            }
        }

        const bool has_gpu_request =
            (config.device_for_this_rank.has_value() && config.device_for_this_rank->isGPU()) ||
            (mapped_device_for_rank.has_value() && mapped_device_for_rank->isGPU()) ||
            std::any_of(config.tp_devices.begin(), config.tp_devices.end(),
                        [](const GlobalDeviceAddress &d)
                        { return d.isGPU(); }) ||
            std::any_of(config.domain_definitions.begin(), config.domain_definitions.end(),
                        [](const DomainDefinition &dom)
                        {
                            return std::any_of(dom.devices.begin(), dom.devices.end(),
                                               [](const GlobalDeviceAddress &d)
                                               { return d.isGPU(); });
                        }) ||
            (config.tp_degree > 1);

        if (has_gpu_request)
        {
            device_manager_numa_filter = -1;
            LOG_INFO("[Main] GPU device(s) requested; initialising DeviceManager without NUMA filtering");
        }
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(device_manager_numa_filter);

    // Introspection dry-run mode should stop before model loading and inference,
    // even when self-bootstrapped under MPI.
    if (config.dry_run)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("[Main] --dry-run requested: configuration validated, skipping model load/inference");
        }
        MPI_Finalize();
        return 0;
    }

    // Validate required arguments
    if (config.model_path.empty())
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Model path required (-m)\n\n");
            std::cout << OrchestrationConfigParser::getHelpText() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // ========================================================================
    // Create and Initialize OrchestrationRunner
    // ========================================================================

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(config);

    if (!runner)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Failed to create orchestration runner");
        }
        MPI_Finalize();
        return 1;
    }

    if (!runner->initialize())
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Failed to initialize: " << runner->lastError());
        }
        MPI_Finalize();
        return 1;
    }

    // Get tokenizer from runner
    auto tokenizer = runner->tokenizer();
    if (!tokenizer)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Failed to get tokenizer from runner");
        }
        MPI_Finalize();
        return 1;
    }

    // Handle chat template override if specified
    if (!config.chat_template_override.empty())
    {
        // Parse template type from string
        ChatTemplateType override_type = ChatTemplateType::UNKNOWN;
        std::string tmpl_lower = config.chat_template_override;
        std::transform(tmpl_lower.begin(), tmpl_lower.end(), tmpl_lower.begin(), ::tolower);

        if (tmpl_lower == "chatml")
            override_type = ChatTemplateType::CHATML;
        else if (tmpl_lower == "llama3")
            override_type = ChatTemplateType::LLAMA3;
        else if (tmpl_lower == "llama2")
            override_type = ChatTemplateType::LLAMA2;
        else if (tmpl_lower == "mistral" || tmpl_lower == "mistral_v1")
            override_type = ChatTemplateType::MISTRAL_V1;
        else if (tmpl_lower == "mistral_v3")
            override_type = ChatTemplateType::MISTRAL_V3;
        else if (tmpl_lower == "mistral_v7")
            override_type = ChatTemplateType::MISTRAL_V7;
        else if (tmpl_lower == "phi3")
            override_type = ChatTemplateType::PHI3;
        else if (tmpl_lower == "phi4")
            override_type = ChatTemplateType::PHI4;
        else if (tmpl_lower == "gemma")
            override_type = ChatTemplateType::GEMMA;
        else if (tmpl_lower == "deepseek")
            override_type = ChatTemplateType::DEEPSEEK;
        else if (tmpl_lower == "deepseek2")
            override_type = ChatTemplateType::DEEPSEEK2;
        else if (tmpl_lower == "deepseek3")
            override_type = ChatTemplateType::DEEPSEEK3;
        else if (tmpl_lower == "zephyr")
            override_type = ChatTemplateType::ZEPHYR;
        else if (tmpl_lower == "vicuna")
            override_type = ChatTemplateType::VICUNA;
        else if (tmpl_lower == "command_r" || tmpl_lower == "command-r")
            override_type = ChatTemplateType::COMMAND_R;
        else
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_WARN("Unknown chat template '" << config.chat_template_override << "', using model's template");
            }
        }

        if (override_type != ChatTemplateType::UNKNOWN)
        {
            tokenizer->setChatTemplate(ChatTemplate::create(override_type));
            if (mpi_ctx->rank() == 0)
            {
                LOG_DEBUG("Using chat template override: " << config.chat_template_override);
            }
        }
    }

    // ========================================================================
    // Chat Mode Handling
    // ========================================================================

    // Interactive chat mode (--chat)
    if (config.chat_mode)
    {
        if (mpi_ctx->rank() == 0)
        {
            if (!tokenizer->hasChatTemplate())
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
                MPI_Finalize();
                return 1;
            }

            LOG_INFO("Starting interactive chat mode...");

            ChatUIConfig chat_config;
            chat_config.system_prompt = config.system_prompt;
            chat_config.max_tokens = config.n_predict;
            chat_config.temperature = config.temperature;
            chat_config.top_k = config.top_k;
            chat_config.top_p = config.top_p;

            // Create an adapter to use OrchestrationRunner with ChatUI
            // ChatUI needs IInferenceRunner, so we wrap the runner calls
            class OrchestrationRunnerAdapter : public IInferenceRunner
            {
            public:
                OrchestrationRunnerAdapter(IOrchestrationRunner *orch_runner)
                    : orch_runner_(orch_runner), position_(0) {}

                bool forward(const int *tokens, int seq_len) override
                {
                    std::vector<int32_t> token_vec(tokens, tokens + seq_len);
                    bool result = orch_runner_->prefill(token_vec);
                    if (result)
                        position_ += seq_len;
                    return result;
                }

                const float *logits() const override
                {
                    return orch_runner_->lastLogits();
                }

                int vocab_size() const override
                {
                    return orch_runner_->vocabSize();
                }

                void clear_cache() override
                {
                    orch_runner_->clearCache();
                    position_ = 0;
                }

                int get_position() const override
                {
                    return position_;
                }

                ExecutionPath executionPath() const override
                {
                    return ExecutionPath::GRAPH;
                }

                const char *architecture() const override
                {
                    return "orchestrated";
                }

            private:
                IOrchestrationRunner *orch_runner_;
                int position_;
            };

            auto adapter = std::make_shared<OrchestrationRunnerAdapter>(runner.get());
            ChatUI chat_ui(tokenizer, adapter, chat_config);
            int result = chat_ui.run();

            runner->shutdown();
            MPI_Finalize();
            return result;
        }
        else
        {
            // Non-rank-0 processes wait for chat to complete
            // TODO: Implement proper multi-rank chat support
            MPI_Barrier(MPI_COMM_WORLD);
            runner->shutdown();
            MPI_Finalize();
            return 0;
        }
    }

    // Single-shot chat mode (--chat-single)
    // This mode applies the chat template to the prompt and generates a response.
    // All MPI ranks must participate in forward passes to avoid deadlocks.
    if (config.single_shot_chat)
    {
        if (!tokenizer->hasChatTemplate())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
            }
            MPI_Barrier(MPI_COMM_WORLD);
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running single-shot chat...");
        }

        // Build conversation and encode with chat template (rank 0 only, then broadcast)
        std::vector<int32_t> token_ids;
        int token_count = 0;

        if (mpi_ctx->rank() == 0)
        {
            std::vector<ChatMessage> conversation;
            if (!config.system_prompt.empty())
            {
                conversation.push_back(ChatMessage("system", config.system_prompt));
            }
            conversation.push_back(ChatMessage("user", config.prompt));

            auto encoded = tokenizer->encodeChat(conversation, /*add_generation_prompt=*/true);
            token_ids.assign(encoded.begin(), encoded.end());
            token_count = static_cast<int>(token_ids.size());

            if (token_ids.empty())
            {
                LOG_ERROR("Failed to encode conversation with chat template");
                token_count = -1; // Signal error to other ranks
            }
            else
            {
                LOG_DEBUG("Encoded " << token_count << " tokens with chat template");
            }
        }

        // Broadcast token count first (to handle errors and allocate on other ranks)
        MPI_Bcast(&token_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (token_count <= 0)
        {
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        // Resize token_ids on non-rank-0 processes and broadcast the tokens
        if (mpi_ctx->rank() != 0)
        {
            token_ids.resize(token_count);
        }
        MPI_Bcast(token_ids.data(), token_count, MPI_INT, 0, MPI_COMM_WORLD);

        // All ranks participate in prefill
        if (mpi_ctx->rank() == 0)
        {
            LOG_DEBUG("Running prefill (" << token_count << " tokens)...");
        }

        if (!runner->prefill(token_ids))
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat prefill failed: " << runner->lastError());
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        // Determine max tokens: -1 means unlimited (use max_seq_len as practical limit)
        int max_tokens = config.n_predict;
        if (max_tokens < 0)
        {
            max_tokens = config.max_seq_len - token_count;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Generating response (max " << max_tokens << " tokens)...\n");
        }

        // Decode loop - use decodeStep() which returns sampled token
        for (int i = 0; i < max_tokens; ++i)
        {
            // decodeStep() handles sampling internally based on config
            GenerationResult result = runner->decodeStep();

            if (!result.success())
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_ERROR("Decode step failed: " << result.error);
                }
                runner->shutdown();
                MPI_Finalize();
                return 1;
            }

            if (result.tokens.empty())
            {
                // No token generated - shouldn't happen
                break;
            }

            int32_t next_token = result.tokens[0];

            // Output token text (streaming) on rank 0 - don't print stop tokens
            if (mpi_ctx->rank() == 0 && !tokenizer->is_stop_token(next_token))
            {
                std::string token_text = tokenizer->decode_token(next_token);
                std::cout << token_text << std::flush;
            }

            // Check for completion (EOS or stop token)
            if (result.is_complete || tokenizer->is_stop_token(next_token))
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_DEBUG("Stop token encountered (" << next_token << "), stopping generation");
                }
                break;
            }
        }

        if (mpi_ctx->rank() == 0)
        {
            std::cout << std::endl;
            LOG_INFO("Chat generation complete.");
        }

        runner->shutdown();
        MPI_Finalize();
        return 0;
    }

    // ========================================================================
    // Benchmark Mode
    // ========================================================================
    if (config.benchmark_mode)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running benchmark mode...");
        }

        // Create an adapter to use OrchestrationRunner with BenchmarkRunner
        class BenchmarkRunnerAdapter : public IInferenceRunner
        {
        public:
            BenchmarkRunnerAdapter(IOrchestrationRunner *orch_runner)
                : orch_runner_(orch_runner), position_(0) {}

            bool forward(const int *tokens, int seq_len) override
            {
                std::vector<int32_t> token_vec(tokens, tokens + seq_len);
                bool result = orch_runner_->prefill(token_vec);
                if (result)
                    position_ += seq_len;
                return result;
            }

            const float *logits() const override
            {
                return orch_runner_->lastLogits();
            }

            int vocab_size() const override
            {
                return orch_runner_->vocabSize();
            }

            void clear_cache() override
            {
                orch_runner_->clearCache();
                position_ = 0;
            }

            int get_position() const override
            {
                return position_;
            }

            ExecutionPath executionPath() const override
            {
                return ExecutionPath::GRAPH;
            }

            const char *architecture() const override
            {
                return "orchestrated";
            }

            const GraphExecutorStats *executorStats() const override
            {
                return orch_runner_->executorStats();
            }

            void resetExecutorStats() override
            {
                orch_runner_->resetExecutorStats();
            }

        private:
            IOrchestrationRunner *orch_runner_;
            int position_;
        };

        auto adapter = std::make_shared<BenchmarkRunnerAdapter>(runner.get());

        BenchmarkRunner benchmark(adapter, tokenizer, mpi_ctx);
        BenchmarkResult result = benchmark.run(config);
        benchmark.printResults(result);

        runner->shutdown();
        MPI_Finalize();
        return result.success ? 0 : 1;
    }

    // ========================================================================
    // Standard Inference Mode
    // ========================================================================

    // Tokenize prompt
    std::vector<int32_t> tokens;
    try
    {
        // Encode WITHOUT BOS token (E2E tests don't use BOS)
        auto encoded = tokenizer->encode(config.prompt, /*add_bos=*/false, /*add_eos=*/false);
        tokens.assign(encoded.begin(), encoded.end());

        if (tokens.empty())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Tokenization resulted in empty token sequence");
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Tokenized prompt: " << tokens.size() << " tokens");
            std::ostringstream token_ids_str;
            token_ids_str << "Token IDs: [";
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                token_ids_str << tokens[i];
                if (i < tokens.size() - 1)
                    token_ids_str << ", ";
            }
            token_ids_str << "]";
            LOG_INFO(token_ids_str.str());
        }
    }
    catch (const std::exception &e)
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error tokenizing prompt: " << e.what());
        }
        runner->shutdown();
        MPI_Finalize();
        return 1;
    }

    // Set up sampling parameters for logging
    SamplingParams sampling_params;
    sampling_params.temperature = config.temperature;
    sampling_params.top_k = config.top_k;
    sampling_params.top_p = config.top_p;
    sampling_params.seed = config.seed;

    if (mpi_ctx->rank() == 0)
    {
        LOG_DEBUG("Sampling parameters:");
        LOG_DEBUG("  temperature: " << sampling_params.temperature);
        LOG_DEBUG("  top_k: " << sampling_params.top_k);
        LOG_DEBUG("  top_p: " << sampling_params.top_p);
        LOG_DEBUG("  seed: " << sampling_params.seed);
    }

    // Run prefill inference
    if (mpi_ctx->rank() == 0)
    {
        LOG_INFO("Running prefill (" << tokens.size() << " tokens)...");
    }

    if (!runner->prefill(tokens))
    {
        if (mpi_ctx->rank() == 0)
        {
            LOG_ERROR("Error: Prefill forward pass failed: " << runner->lastError());
        }
        runner->shutdown();
        MPI_Finalize();
        return 1;
    }

    if (mpi_ctx->rank() == 0)
    {
        if (config.n_predict == -1)
        {
            LOG_DEBUG("Prefill complete. Generating tokens until EOS...\n");
        }
        else
        {
            LOG_DEBUG("Prefill complete. Generating " << config.n_predict << " tokens...\n");
        }
    }

    // Generate tokens autoregressively using decodeStep()
    // n_predict = -1 means unlimited (generate until EOS)
    int max_tokens = (config.n_predict == -1) ? INT_MAX : config.n_predict;
    for (int i = 0; i < max_tokens; ++i)
    {
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Starting decode iteration " << i);

        // decodeStep() returns the sampled token
        GenerationResult result = runner->decodeStep();

        if (!result.success())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("\nError: Decode step failed at token " << (i + 1) << ": " << result.error);
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        if (result.tokens.empty())
        {
            // No token generated - shouldn't happen
            LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] No token generated at iteration " << i);
            break;
        }

        int32_t next_token = result.tokens[0];
        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Generated token: " << next_token);

        // Output on rank 0 (streaming) - don't print stop tokens
        if (mpi_ctx->rank() == 0 && !tokenizer->is_stop_token(next_token))
        {
            std::string token_text = tokenizer->decode_token(next_token);
            std::cout << token_text << std::flush;
        }

        // Check for stop tokens (EOS, <|im_end|>, etc.)
        if (result.is_complete || tokenizer->is_stop_token(next_token))
        {
            if (mpi_ctx->rank() == 0 && config.verbose_level > 0)
            {
                LOG_DEBUG("\nGeneration stopped: stop token " << next_token << " encountered");
            }
            break;
        }
    }

    if (mpi_ctx->rank() == 0)
    {
        std::cout << "\n"
                  << std::endl; // Final newline
        LOG_DEBUG("Generation complete.");
    }

    if (mpi_ctx->world_size() > 1)
    {
        mpi_ctx->barrier();
    }

    runner->shutdown();

    if (mpi_ctx->world_size() > 1)
    {
        mpi_ctx->barrier();
    }

    MPI_Finalize();
    return 0;
}
