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
 *
 * Cached CPU placement keeps OS NUMA/PU identities separate from MPI logical
 * core indices. Discovery publishes exact per-node facts once; launch and
 * admission reject missing or contradictory observations without rediscovery.
 */

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <span>
#include "NUMATopology.h"

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
     * @brief Immutable cold mapping of one real rank per observed NUMA partition.
     *
     * Node labels are physical observations, never MPI rank IDs. The original
     * launcher writes native rankfile entries from their exact logical core
     * sets. Rank indices enumerate observed partitions; they are never NUMA IDs.
     */
    class MPINUMAMapping final
    {
    public:
        /**
         * @brief Seal distinct observed nodes and their original logical core sets.
         * @param placement Original cached OS NUMA/PU and logical-core observations.
         * @param selected_nodes Exact participating OS NUMA labels, never rank IDs.
         * @return Checked placement containing no worker/thread execution state.
         * @throws std::invalid_argument For unknown nodes, empty lists or overlapping/invalid CPUs.
         */
        static MPINUMAMapping fromObservedPlacement(std::span<const MPICorePlacement> placement,
            std::span<const int> selected_nodes);
        /** @return Original node roster, unrelated to eventual MPI rank numbers. */
        const std::vector<int> &nodes() const noexcept { return nodes_; }
        /** @return Exact CPU union in the original launcher CPU-set grammar. */
        const std::string &cpuSet() const noexcept { return cpu_set_; }
        /** @return Native local OpenMPI rankfile, one complete logical-core set per observed node. */
        const std::string &rankFileContents() const noexcept { return rank_file_contents_; }
    private:
        /** @brief Only the checked observation factory may publish a placement. */
        MPINUMAMapping() = default;
        std::vector<int> nodes_;
        std::string cpu_set_;
        std::string rank_file_contents_;
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
        std::string omp_places = "cores"; ///< One place per physical core, including its allowed SMT siblings.
        std::string omp_proc_bind = "close";
        std::string cpu_set; ///< Optional OpenMPI --cpu-set for explicit CPU NUMA targeting
        std::optional<MPINUMAMapping> numa_mapping; ///< Exact local one-rank-per-observed-NUMA declaration.
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

    /** @brief Original application argv ownership during an MPI self-launch. */
    enum class MPIBootstrapArgumentPolicy
    {
        FilterLaunchArguments, ///< Existing frontends consume launcher-only flags.
        PreserveOriginalArguments, ///< Opaque wrappers retain their exact parser/config contract.
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
         * @brief Publish only portable placement before any child loads libgomp.
         * @param config Existing explicit launch policy; no host-local CPU count is exported.
         * @throws std::runtime_error If the exact selected environment cannot be published.
         * @throws std::invalid_argument For empty placement or binding policy.
         */
        static void configureOpenMPPlacement(const MPILaunchConfig &config);

        /**
         * @brief Configure OpenMP environment for optimal performance
         *
         * Sets environment variables:
         * - OMP_NUM_THREADS: Based on cores per socket (or forced value)
         * - OMP_PLACES: selected placement (physical core places by default)
         * - OMP_PROC_BIND: close
         * - MKL_NUM_THREADS: Match OMP
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
         * @param arguments Whether original application flags remain exact after re-exec.
         * @param observed_numa_rankfile Original guardian-owned rankfile pathname, required only for a typed NUMA mapping.
         * @return Vector of command-line arguments for execvp
         */
        static std::vector<std::string> buildMPIRunCommand(
            int argc, char *argv[],
            const MPILaunchConfig &config,
            const CPUTopology &topology,
            MPIBootstrapArgumentPolicy arguments = MPIBootstrapArgumentPolicy::FilterLaunchArguments,
            const std::string &observed_numa_rankfile = {});

        /**
         * @brief Self-launch via mpirun
         *
         * Replaces this managed child with exact mpirun argv beneath its original ancestor owner.
         * This function does not return on success.
         *
         * @param argc Original argc
         * @param argv Original argv
         * @param config Launch configuration
         * @param topology CPU topology
         * @param arguments Existing frontends filter launch flags; typed wrappers may retain them.
         * @return -1 on failure (owned mpirun exec failed)
         */
        static int selfLaunchMPI(int argc, char *argv[],
                                 const MPILaunchConfig &config,
                                 const CPUTopology &topology,
                                 MPIBootstrapArgumentPolicy arguments = MPIBootstrapArgumentPolicy::FilterLaunchArguments);

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
         * @brief Read OS CPU-list string for a NUMA node (e.g., "28-55,84-111")
         */
        static std::string getCpuSetForNumaNode(int numa_node);

        /**
         * @brief Read physical-core representatives as OS CPU IDs for a NUMA node.
         *
         * Uses thread_siblings topology to keep one logical CPU per physical core
         * and returns a compact OS range list (e.g., "28-55"). These IDs are
         * diagnostics/affinity inputs, never OpenMPI logical cpu-set indices.
         */
        static std::string getPhysicalCpuSetForNumaNode(int numa_node);

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
