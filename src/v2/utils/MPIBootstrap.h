/**
 * @file MPIBootstrap.h
 * @brief MPI self-bootstrap and environment configuration utilities
 *
 * Enables Llaminar to self-launch with mpirun if not already running
 * under an MPI environment. Supports:
 * - MPI environment detection via standard environment variables
 * - CPU topology detection for optimal thread/rank configuration
 * - Self-launch via mpirun with proper bindings
 * - Multi-machine support via hostfile
 *
 * @author David Sanftenberg
 */

#pragma once

#include <string>
#include <vector>
#include <optional>

namespace llaminar2
{

    /**
     * @brief CPU topology information for a single machine
     */
    struct CPUTopology
    {
        int num_sockets = 1;          ///< Number of CPU sockets
        int physical_cores = 1;       ///< Total physical cores
        int logical_cores = 1;        ///< Total logical cores (with HT)
        int cores_per_socket = 1;     ///< Physical cores per socket
        int threads_per_core = 1;     ///< Threads per physical core (1 or 2)
        int numa_nodes = 1;           ///< Number of NUMA nodes
        bool hyperthreading = false;  ///< Hyperthreading detected
        std::string detection_method; ///< How topology was detected
    };

    /**
     * @brief MPI launch configuration
     */
    struct MPILaunchConfig
    {
        int num_procs = 0;            ///< Number of MPI processes (0 = auto)
        std::string hostfile;         ///< Path to hostfile for multi-machine (empty = local only)
        bool bind_to_socket = true;   ///< Bind ranks to sockets
        bool map_by_socket = true;    ///< Map ranks by socket
        bool report_bindings = false; ///< Print MPI binding information
        bool verbose = false;         ///< Verbose mpirun output
        bool oversubscribe = false;   ///< Allow more ranks than available slots

        // Per-rank OpenMP configuration
        int omp_threads_per_rank = 0;   ///< OMP_NUM_THREADS (0 = auto)
        bool use_physical_cores = true; ///< Restrict to physical cores only
        std::string omp_places = "sockets";
        std::string omp_proc_bind = "close";
    };

    /**
     * @brief Result of MPI environment detection
     */
    struct MPIEnvironmentInfo
    {
        bool is_mpi_process = false;    ///< Currently running under MPI
        int detected_rank = -1;         ///< Rank if detected (-1 if not)
        int detected_world_size = -1;   ///< World size if detected (-1 if not)
        std::string mpi_implementation; ///< "openmpi", "mpich", "unknown"
        std::string detection_method;   ///< How MPI was detected
    };

    /**
     * @brief MPI bootstrap utilities
     *
     * Provides functionality to:
     * 1. Detect if running under MPI via environment variables
     * 2. Detect local CPU topology for optimal configuration
     * 3. Self-launch via mpirun if not already in MPI context
     * 4. Configure OpenMP and other threading settings
     */
    class MPIBootstrap
    {
    public:
        /**
         * @brief Detect if currently running under an MPI environment
         *
         * Checks standard environment variables from various MPI implementations:
         * - OpenMPI: OMPI_COMM_WORLD_SIZE, OMPI_COMM_WORLD_RANK
         * - MPICH/Intel MPI: PMI_RANK, PMI_SIZE
         * - SLURM: SLURM_PROCID, SLURM_NTASKS
         * - Generic: MPI_LOCALRANKID
         *
         * @return MPIEnvironmentInfo with detection results
         */
        static MPIEnvironmentInfo detectMPIEnvironment();

        /**
         * @brief Detect CPU topology of local machine
         *
         * Parses /proc/cpuinfo and /sys filesystem to determine:
         * - Number of sockets
         * - Physical vs logical cores
         * - NUMA topology
         * - Hyperthreading status
         *
         * @return CPUTopology with detected values
         */
        static CPUTopology detectCPUTopology();

        /**
         * @brief Configure OpenMP environment for optimal performance
         *
         * Sets environment variables:
         * - OMP_NUM_THREADS: Based on cores per socket (or forced value)
         * - OMP_PLACES: sockets
         * - OMP_PROC_BIND: close
         * - OPENBLAS_NUM_THREADS, MKL_NUM_THREADS: Match OMP
         *
         * @param topology CPU topology info
         * @param config Launch configuration
         */
        static void configureOpenMPEnvironment(const CPUTopology &topology,
                                               const MPILaunchConfig &config);

        /**
         * @brief Build mpirun command line for self-launch
         *
         * Constructs appropriate mpirun invocation based on configuration:
         * - Process binding (--bind-to socket)
         * - Process mapping (--map-by socket)
         * - Number of processes
         * - Hostfile for multi-machine
         * - MCA parameters for optimization
         *
         * @param argc Original argc
         * @param argv Original argv
         * @param config Launch configuration
         * @param topology CPU topology (for auto-detecting num_procs)
         * @return Vector of command-line arguments for execvp
         */
        static std::vector<std::string> buildMPIRunCommand(
            int argc, char *argv[],
            const MPILaunchConfig &config,
            const CPUTopology &topology);

        /**
         * @brief Self-launch via mpirun
         *
         * Replaces current process with mpirun invocation.
         * This function does not return on success.
         *
         * @param argc Original argc
         * @param argv Original argv
         * @param config Launch configuration
         * @param topology CPU topology
         * @return -1 on failure (exec failed)
         */
        static int selfLaunchMPI(int argc, char *argv[],
                                 const MPILaunchConfig &config,
                                 const CPUTopology &topology);

        /**
         * @brief Get default launch configuration based on topology
         *
         * Returns sensible defaults:
         * - num_procs = num_sockets (one rank per socket)
         * - omp_threads_per_rank = cores_per_socket
         * - bind_to_socket, map_by_socket = true
         *
         * @param topology CPU topology
         * @return Default MPILaunchConfig
         */
        static MPILaunchConfig getDefaultConfig(const CPUTopology &topology);

        /**
         * @brief Print configuration summary
         *
         * @param topology CPU topology
         * @param config Launch configuration
         * @param env MPI environment info
         */
        static void printConfigurationSummary(const CPUTopology &topology,
                                              const MPILaunchConfig &config,
                                              const MPIEnvironmentInfo &env);

        /**
         * @brief Parse hostfile to determine total available hosts/slots
         *
         * Supports OpenMPI hostfile format:
         *   hostname1 slots=N
         *   hostname2 slots=M
         *
         * @param hostfile_path Path to hostfile
         * @return Vector of (hostname, slots) pairs, empty if file invalid
         */
        static std::vector<std::pair<std::string, int>> parseHostfile(
            const std::string &hostfile_path);

    private:
        /**
         * @brief Check for specific environment variable
         * @param var Environment variable name
         * @return Value if set, nullopt otherwise
         */
        static std::optional<std::string> getEnv(const char *var);

        /**
         * @brief Parse /proc/cpuinfo for topology
         */
        static CPUTopology parseProcCpuinfo();

        /**
         * @brief Get NUMA node count from sysfs
         */
        static int getNumaNoneCount();
    };

} // namespace llaminar2
