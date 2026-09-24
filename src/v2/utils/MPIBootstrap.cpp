/**
 * @file MPIBootstrap.cpp
 * @brief MPI self-bootstrap with explicit local versus cluster placement.
 *
 * The MPI launcher owns hostfile grammar and slot allocation. Local topology
 * may supply placement only for a local launch; remote ranks size their teams
 * from their own binding before inventory discovery. Never copy this host's
 * CPU set or socket width into a cluster-wide command.
 * Portable diagnostic environment is explicitly exported by name: remote SSH
 * ranks do not inherit the launcher's ordinary environment. Rank/device-local
 * placement and cloud credentials must never cross that boundary implicitly.
 *
 * @author David Sanftenberg
 */

#include "MPIBootstrap.h"
#include "Logger.h"
#include "DebugEnv.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <map>
#include <set>
#include <algorithm>
#include <filesystem>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Export explicitly supplied diagnostic policy to every MPI child.
         * @param command Launcher argv, before the application executable.
         *
         * OpenMPI's name-only -x form preserves values containing spaces and
         * avoids embedding paths or values in the printed launcher command.
         * This is a propagation allowlist, not a second configuration parser:
         * each rank's ordinary diagnostic authority interprets the same values.
         * Never export the whole environment, backend visibility, CPU affinity,
         * allocator policy or orchestration-provider credentials here.
         */
        void appendDiagnosticEnvironment(std::vector<std::string> &command)
        {
            constexpr const char *names[] = {
                "LLAMINAR_LOG_LEVEL",
                "LLAMINAR_PERF_STATS_JSON",
                "LLAMINAR_PERF_STATS_CSV",
                "LLAMINAR_PERF_STATS_FILTER",
                "LLAMINAR_PERF_STATS_SUMMARY",
                "LLAMINAR_PERF_STATS_TABLE",
                "LLAMINAR_PERF_STATS_TABLE_LIMIT",
                "LLAMINAR_PERF_STATS_CPU_STAGE_TIMING",
                "LLAMINAR_PERF_STATS_GPU_STAGE_TIMING",
                "LLAMINAR_GPU_STAGE_TIMING",
                "LLAMINAR_GPU_STAGE_TIMING_DETAIL",
                "LLAMINAR_PROFILING",
            };
            for (const char *name : names)
            {
                // Empty/false values are explicit policy too. Export only
                // variables present in the launcher, never invent defaults.
                if (DebugEnv::envValue(name) != nullptr)
                {
                    command.emplace_back("-x");
                    command.emplace_back(name);
                }
            }
        }

        /** @brief Configure launcher-only hwloc discovery before execing MPI. */
        void configureLauncherEnvironment()
        {
            if (debugEnv().mpi_bootstrap.get("HWLOC_COMPONENTS").has_value())
            {
                return;
            }

            // OpenMPI initializes hwloc in the launcher before ranks start. The
            // hwloc GL discovery plugin can block on stale or inaccessible X
            // sockets, so disable only that component by default.
            if (setenv("HWLOC_COMPONENTS", "-gl", 0) != 0)
            {
                LOG_WARN("[MPIBootstrap] Failed to set HWLOC_COMPONENTS=-gl: " << strerror(errno));
            }
            else
            {
                LOG_DEBUG("[MPIBootstrap] Set HWLOC_COMPONENTS=-gl for mpirun launcher");
            }
        }
    }

    // ========================================================================
    // Environment Variable Helpers
    // ========================================================================

    std::optional<std::string> MPIBootstrap::getEnv(const char *var)
    {
        return debugEnv().mpi_bootstrap.get(var);
    }

    // ========================================================================
    // MPI Environment Detection
    // ========================================================================

    MPIEnvironmentInfo MPIBootstrap::detectMPIEnvironment()
    {
        MPIEnvironmentInfo info;

        // Check OpenMPI environment variables
        auto ompi_size = getEnv("OMPI_COMM_WORLD_SIZE");
        auto ompi_rank = getEnv("OMPI_COMM_WORLD_RANK");
        if (ompi_size && ompi_rank)
        {
            info.is_mpi_process = true;
            info.detected_world_size = std::stoi(*ompi_size);
            info.detected_rank = std::stoi(*ompi_rank);
            info.mpi_implementation = "openmpi";
            info.detection_method = "OMPI_COMM_WORLD_*";
            return info;
        }

        // Check MPICH/Intel MPI environment variables
        auto pmi_size = getEnv("PMI_SIZE");
        auto pmi_rank = getEnv("PMI_RANK");
        if (pmi_size && pmi_rank)
        {
            info.is_mpi_process = true;
            info.detected_world_size = std::stoi(*pmi_size);
            info.detected_rank = std::stoi(*pmi_rank);
            info.mpi_implementation = "mpich";
            info.detection_method = "PMI_*";
            return info;
        }

        // Check SLURM environment
        auto slurm_ntasks = getEnv("SLURM_NTASKS");
        auto slurm_procid = getEnv("SLURM_PROCID");
        if (slurm_ntasks && slurm_procid)
        {
            info.is_mpi_process = true;
            info.detected_world_size = std::stoi(*slurm_ntasks);
            info.detected_rank = std::stoi(*slurm_procid);
            info.mpi_implementation = "slurm";
            info.detection_method = "SLURM_*";
            return info;
        }

        // Check generic MPI local rank (some implementations set this)
        auto mpi_localrank = getEnv("MPI_LOCALRANKID");
        if (mpi_localrank)
        {
            info.is_mpi_process = true;
            info.detected_rank = std::stoi(*mpi_localrank);
            info.mpi_implementation = "unknown";
            info.detection_method = "MPI_LOCALRANKID";
            // World size not available from this variable alone
            return info;
        }

        // Also check OMPI_COMM_WORLD_LOCAL_RANK (OpenMPI per-node rank)
        auto ompi_local = getEnv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (ompi_local)
        {
            info.is_mpi_process = true;
            info.detected_rank = std::stoi(*ompi_local);
            info.mpi_implementation = "openmpi";
            info.detection_method = "OMPI_COMM_WORLD_LOCAL_RANK";
            return info;
        }

        // Not running under MPI
        info.is_mpi_process = false;
        info.detection_method = "none";
        return info;
    }

    // ========================================================================
    // CPU Topology Detection
    // ========================================================================

    int MPIBootstrap::getNumaNoneCount()
    {
        // Count NUMA nodes from sysfs
        const std::string numa_path = "/sys/devices/system/node";
        int count = 0;

        try
        {
            if (std::filesystem::exists(numa_path))
            {
                for (const auto &entry : std::filesystem::directory_iterator(numa_path))
                {
                    std::string name = entry.path().filename().string();
                    if (name.substr(0, 4) == "node" && name.size() > 4)
                    {
                        // Check if remaining part is numeric
                        bool is_node = true;
                        for (size_t i = 4; i < name.size(); ++i)
                        {
                            if (!std::isdigit(name[i]))
                            {
                                is_node = false;
                                break;
                            }
                        }
                        if (is_node)
                        {
                            ++count;
                        }
                    }
                }
            }
        }
        catch (const std::exception &)
        {
            // Fallback
        }

        return (count > 0) ? count : 1;
    }

    CPUTopology MPIBootstrap::parseProcCpuinfo()
    {
        CPUTopology topo;
        topo.detection_method = "procfs";

        std::ifstream cpuinfo("/proc/cpuinfo");
        if (!cpuinfo.is_open())
        {
            topo.detection_method = "fallback";
            return topo;
        }

        // Track unique (physical_id, core_id) pairs to count physical cores
        std::set<std::pair<int, int>> socket_core_pairs;
        std::set<int> physical_ids;
        int processor_count = 0;
        int current_physical_id = -1;
        int current_core_id = -1;

        std::string line;
        while (std::getline(cpuinfo, line))
        {
            if (line.find("processor") == 0)
            {
                ++processor_count;
                current_physical_id = -1;
                current_core_id = -1;
            }
            else if (line.find("physical id") == 0)
            {
                size_t colon = line.find(':');
                if (colon != std::string::npos)
                {
                    current_physical_id = std::stoi(line.substr(colon + 1));
                    physical_ids.insert(current_physical_id);
                }
            }
            else if (line.find("core id") == 0)
            {
                size_t colon = line.find(':');
                if (colon != std::string::npos)
                {
                    current_core_id = std::stoi(line.substr(colon + 1));
                    if (current_physical_id >= 0)
                    {
                        socket_core_pairs.insert({current_physical_id, current_core_id});
                    }
                }
            }
        }

        // Calculate topology
        topo.logical_cores = processor_count;
        topo.num_sockets = static_cast<int>(physical_ids.size());
        if (topo.num_sockets == 0)
        {
            topo.num_sockets = 1;
        }

        topo.physical_cores = static_cast<int>(socket_core_pairs.size());
        if (topo.physical_cores == 0)
        {
            topo.physical_cores = processor_count;
        }

        topo.cores_per_socket = topo.physical_cores / topo.num_sockets;
        if (topo.cores_per_socket == 0)
        {
            topo.cores_per_socket = 1;
        }

        topo.threads_per_core = topo.logical_cores / topo.physical_cores;
        if (topo.threads_per_core == 0)
        {
            topo.threads_per_core = 1;
        }

        topo.hyperthreading = (topo.threads_per_core > 1);
        topo.numa_nodes = getNumaNoneCount();

        return topo;
    }

    CPUTopology MPIBootstrap::detectCPUTopology()
    {
        return parseProcCpuinfo();
    }

    std::string MPIBootstrap::getCpuSetForNumaNode(int numa_node)
    {
        if (numa_node < 0)
        {
            return "";
        }

        std::ifstream cpulist_file("/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist");
        if (!cpulist_file.is_open())
        {
            return "";
        }

        std::string cpulist;
        std::getline(cpulist_file, cpulist);
        return cpulist;
    }

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

        std::string formatCpuList(const std::vector<int> &cpus)
        {
            if (cpus.empty())
            {
                return "";
            }

            std::ostringstream out;
            int range_start = cpus.front();
            int prev = cpus.front();

            auto emit_range = [&](int start, int end)
            {
                if (out.tellp() > 0)
                {
                    out << ",";
                }
                if (start == end)
                {
                    out << start;
                }
                else
                {
                    out << start << "-" << end;
                }
            };

            for (size_t i = 1; i < cpus.size(); ++i)
            {
                int current = cpus[i];
                if (current == prev + 1)
                {
                    prev = current;
                    continue;
                }
                emit_range(range_start, prev);
                range_start = prev = current;
            }
            emit_range(range_start, prev);
            return out.str();
        }

        std::vector<int> readThreadSiblings(int cpu)
        {
            std::ifstream siblings_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
            if (!siblings_file.is_open())
            {
                return {cpu};
            }

            std::string siblings_list;
            std::getline(siblings_file, siblings_list);
            auto siblings = parseCpuList(siblings_list);
            if (siblings.empty())
            {
                siblings.push_back(cpu);
            }
            return siblings;
        }
    }

    std::string MPIBootstrap::getPhysicalCpuSetForNumaNode(int numa_node)
    {
        const std::string full_cpulist = getCpuSetForNumaNode(numa_node);
        if (full_cpulist.empty())
        {
            return "";
        }

        const auto node_cpus = parseCpuList(full_cpulist);
        std::set<int> node_cpu_set(node_cpus.begin(), node_cpus.end());
        std::vector<int> physical_cpus;
        std::set<int> chosen_representatives;

        for (int cpu : node_cpus)
        {
            auto siblings = readThreadSiblings(cpu);

            int representative = INT_MAX;
            for (int sibling : siblings)
            {
                if (node_cpu_set.count(sibling))
                {
                    representative = std::min(representative, sibling);
                }
            }

            if (representative == INT_MAX)
            {
                representative = cpu;
            }

            if (!chosen_representatives.count(representative))
            {
                chosen_representatives.insert(representative);
                physical_cpus.push_back(representative);
            }
        }

        std::sort(physical_cpus.begin(), physical_cpus.end());
        if (physical_cpus.empty())
        {
            return full_cpulist;
        }

        return formatCpuList(physical_cpus);
    }

    // ========================================================================
    // OpenMP Environment Configuration
    // ========================================================================

    void MPIBootstrap::configureOpenMPEnvironment(const CPUTopology &topology,
                                                  const MPILaunchConfig &config)
    {
        // Determine thread count
        int omp_threads = config.omp_threads_per_rank;
        if (omp_threads <= 0)
        {
            // Auto: use physical cores per socket
            omp_threads = topology.cores_per_socket;
        }

        // Set OpenMP environment variables
        setenv("OMP_NUM_THREADS", std::to_string(omp_threads).c_str(), 1);
        setenv("OMP_PLACES", config.omp_places.c_str(), 1);
        setenv("OMP_PROC_BIND", config.omp_proc_bind.c_str(), 1);
        setenv("OMP_NESTED", "false", 1);
        setenv("OMP_DYNAMIC", "false", 1);

        // Intel KMP settings
        setenv("KMP_AFFINITY", "granularity=fine,compact,1,0", 1);
        setenv("KMP_BLOCKTIME", "0", 1);

        // MKL thread settings
        setenv("MKL_NUM_THREADS", std::to_string(omp_threads).c_str(), 1);
        setenv("MKL_DYNAMIC", "false", 1);

        // Llaminar-specific threading policy flags
        if (config.use_physical_cores)
        {
            setenv("LLAMINAR_OMP_USE_PHYSICAL", "1", 1);
        }

        // Log configuration
        LOG_DEBUG("[MPIBootstrap] OpenMP configured: " << omp_threads << " threads, "
                                                       << "places=" << config.omp_places << ", bind=" << config.omp_proc_bind);
    }

    // ========================================================================
    // MPI Launch Configuration
    // ========================================================================

    MPILaunchConfig MPIBootstrap::getDefaultConfig(const CPUTopology &topology)
    {
        MPILaunchConfig config;

        // CONSERVATIVE DEFAULT: Single process to avoid heterogeneous device trap
        // Mixed CUDA+ROCm systems fall back to slow MPI host-staged collectives.
        // Users wanting multi-GPU should use explicit --mpi-procs N or --tp-devices.
        // Future: Add --auto-tp for model-aware automatic scaling.
        config.num_procs = 1;

        // Use physical cores per socket for OpenMP threads (full socket for single rank)
        config.omp_threads_per_rank = topology.cores_per_socket * topology.num_sockets;

        // Default binding
        config.bind_to_socket = true;
        config.map_by_socket = true;
        config.use_physical_cores = true;

        return config;
    }

    // ========================================================================
    // MPI Run Command Building
    // ========================================================================

    std::vector<std::string> MPIBootstrap::buildMPIRunCommand(
        int argc, char *argv[],
        const MPILaunchConfig &config,
        const CPUTopology &topology)
    {
        std::vector<std::string> cmd;

        cmd.push_back("mpirun");

        // Number of processes
        int num_procs = config.num_procs;
        if (num_procs <= 0 && config.hostfile.empty())
        {
            num_procs = topology.num_sockets;
        }

        // Omitting -np delegates cluster slot admission to MPI itself. Parsing
        // a subset of hostfile grammar here would create a second authority
        // and could silently launch only this machine after a parse failure.
        if (num_procs > 0)
        {
            cmd.push_back("-np");
            cmd.push_back(std::to_string(num_procs));
        }

        // Hostfile for multi-machine
        if (!config.hostfile.empty())
        {
            cmd.push_back("--hostfile");
            cmd.push_back(config.hostfile);
        }

        const int pe_threads = config.omp_threads_per_rank > 0
                                   ? config.omp_threads_per_rank
                                   : std::max(1, topology.cores_per_socket);
        const bool use_pe_mapping =
            config.hostfile.empty() &&
            config.bind_to_socket &&
            config.map_by_socket &&
            pe_threads > 1;
        const bool pe_fits_socket =
            pe_threads <= std::max(1, topology.cores_per_socket);

        // Process binding. OpenMPI requires bind-to core when a mapping
        // requests multiple processing elements per rank; bind-to socket with
        // PE=N is rejected and plain bind-to socket may bind only one core plus
        // its SMT sibling on some systems.
        if (config.bind_to_socket)
        {
            cmd.push_back("--bind-to");
            cmd.push_back(use_pe_mapping ? "core" : "socket");
        }

        // Process mapping
        if (config.map_by_socket)
        {
            cmd.push_back("--map-by");
            if (use_pe_mapping)
            {
                std::ostringstream mapping;
                mapping << (pe_fits_socket ? "socket" : "slot")
                        << ":PE=" << pe_threads;
                cmd.push_back(mapping.str());
            }
            else
            {
                cmd.push_back("socket");
            }
        }

        // Optional explicit CPU affinity set
        if (!config.cpu_set.empty())
        {
            if (!config.hostfile.empty())
                throw std::invalid_argument(
                    "A launcher-local CPU set cannot be applied to a hostfile cluster");
            cmd.push_back("--cpu-set");
            cmd.push_back(config.cpu_set);
        }

        // Report bindings
        if (config.report_bindings)
        {
            cmd.push_back("--report-bindings");
        }

        // Oversubscribe (allow more ranks than slots)
        if (config.oversubscribe)
        {
            cmd.push_back("--oversubscribe");
        }

        // MCA parameters for optimization
        cmd.push_back("--mca");
        cmd.push_back("mpi_leave_pinned");
        cmd.push_back("1");

        cmd.push_back("--mca");
        cmd.push_back("btl_vader_single_copy_mechanism");
        cmd.push_back("none");

        appendDiagnosticEnvironment(cmd);

        // Add original program and its arguments
        // argv[0] is the program path
        cmd.push_back(argv[0]);

        // Copy all original arguments, filtering out MPI bootstrap-specific ones
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            // Skip bootstrap-specific arguments (they're already processed)
            if (arg == "--mpi-bootstrap" ||
                arg.rfind("--mpi-procs=", 0) == 0 ||
                arg == "--mpi-dry-run")
            {
                continue;
            }

            // Handle arguments with separate values
            if (arg == "--mpi-procs")
            {
                ++i; // Skip the value too
                continue;
            }

            cmd.push_back(argv[i]);
        }

        return cmd;
    }

    // ========================================================================
    // Self-Launch via mpirun
    // ========================================================================

    int MPIBootstrap::selfLaunchMPI(int argc, char *argv[],
                                    const MPILaunchConfig &config,
                                    const CPUTopology &topology)
    {
        // Configure OpenMP environment before exec (inherited by child processes)
        if (config.hostfile.empty())
            configureOpenMPEnvironment(topology, config);
        configureLauncherEnvironment();

        // Build command
        auto cmd = buildMPIRunCommand(argc, argv, config, topology);

        // Convert to C-style array for execvp
        std::vector<char *> c_argv;
        for (auto &s : cmd)
        {
            c_argv.push_back(&s[0]);
        }
        c_argv.push_back(nullptr);

        // Log what we're executing
        std::ostringstream cmd_str;
        for (size_t i = 0; i < cmd.size(); ++i)
        {
            if (i > 0)
                cmd_str << " ";
            // Quote arguments with spaces
            if (cmd[i].find(' ') != std::string::npos)
            {
                cmd_str << "\"" << cmd[i] << "\"";
            }
            else
            {
                cmd_str << cmd[i];
            }
        }
        LOG_DEBUG("[MPIBootstrap] Launching: " << cmd_str.str());

        // Replace current process with mpirun
        execvp("mpirun", c_argv.data());

        // If we get here, exec failed
        LOG_ERROR("[MPIBootstrap] Failed to exec mpirun: " << strerror(errno));
        return -1;
    }

    // ========================================================================
    // Configuration Summary
    // ========================================================================

    void MPIBootstrap::printConfigurationSummary(const CPUTopology &topology,
                                                 const MPILaunchConfig &config,
                                                 const MPIEnvironmentInfo &env)
    {
        std::cout << "=== Llaminar MPI Configuration ===" << std::endl;

        // System topology
        std::cout << "System: " << topology.num_sockets << " socket(s), "
                  << topology.cores_per_socket << " cores/socket, "
                  << topology.numa_nodes << " NUMA node(s)" << std::endl;
        std::cout << "Topology: " << topology.physical_cores << " physical cores, "
                  << topology.logical_cores << " logical cores" << std::endl;
        std::cout << "Hyperthreading: " << (topology.hyperthreading ? "Yes" : "No")
                  << " (" << topology.threads_per_core << " threads/core)" << std::endl;

        // MPI configuration
        int num_procs = config.num_procs > 0 ? config.num_procs : topology.num_sockets;
        int omp_threads = config.omp_threads_per_rank > 0
                              ? config.omp_threads_per_rank
                              : topology.cores_per_socket;

        std::cout << "MPI: ";
        if (!config.hostfile.empty() && config.num_procs <= 0)
            std::cout << "hostfile-defined slots";
        else
            std::cout << num_procs << " process(es)";
        if (!config.hostfile.empty())
        {
            std::cout << " (hostfile: " << config.hostfile << ")";
        }
        std::cout << std::endl;

        std::cout << "OpenMP: "
                  << (config.hostfile.empty() ? std::to_string(omp_threads)
                                               : "rank-local physical-core count")
                  << " threads/rank, "
                  << "places=" << config.omp_places << ", "
                  << "bind=" << config.omp_proc_bind << std::endl;

        if (!config.cpu_set.empty())
        {
            std::cout << "CPU Set: " << config.cpu_set << std::endl;
        }

        // MPI environment status
        if (env.is_mpi_process)
        {
            std::cout << "Status: Running under MPI (" << env.mpi_implementation
                      << ", detected via " << env.detection_method << ")" << std::endl;
            if (env.detected_rank >= 0)
            {
                std::cout << "Rank: " << env.detected_rank;
                if (env.detected_world_size > 0)
                {
                    std::cout << "/" << env.detected_world_size;
                }
                std::cout << std::endl;
            }
        }
        else
        {
            std::cout << "Status: Not running under MPI (will self-launch)" << std::endl;
        }

        std::cout << std::endl;
    }

} // namespace llaminar2
