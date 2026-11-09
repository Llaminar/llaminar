/**
 * @file MPIBootstrapPhase.h
 * @brief Pre-MPI topology planning, NUMA resolution, and MPI self-launch
 *
 * Extracted from Main.cpp: anonymous namespace helpers and the pre-MPI
 * bootstrap block that detects topology, resolves NUMA nodes, and
 * self-launches via mpirun if not already in an MPI context.
 */

#pragma once

#include "config/OrchestrationConfig.h"
#include "utils/MPIBootstrap.h"
#include <set>
#include <string>
#include <vector>

namespace llaminar2
{

    class DeviceManager;

    /** Discovery requirements derived from parsed device declarations, not names. */
    enum class BootstrapDeviceIntent
    {
        Automatic,   ///< No complete device declaration; inventory is required.
        CpuOnly,     ///< Every declared execution endpoint is a CPU.
        Accelerator, ///< At least one declaration requires accelerator discovery.
    };

    /**
     * @brief Result of the MPI bootstrap phase
     */
    struct BootstrapResult
    {
        enum class Action
        {
            CONTINUE, ///< Proceed to RuntimeInitPhase
            EXIT      ///< Exit with exit_code
        };
        Action action;
        int exit_code = 0;
    };

    /**
     * @brief Pre-MPI topology planning and self-launch
     *
     * Handles everything that happens before MPI_Init_thread:
     * - CPU topology detection
     * - MPI environment detection
     * - NUMA node resolution for inference
     * - MPI launch configuration
     * - Self-launch via mpirun (replaces process)
     */
    class MPIBootstrapPhase
    {
    public:
        /**
         * @brief Execute the bootstrap phase
         *
         * If not running under MPI, this will self-launch via mpirun
         * (replacing the current process — does not return).
         *
         * @param config Orchestration config (parsed from CLI)
         * @param argc Argument count (for MPI self-launch)
         * @param argv Argument vector (for MPI self-launch)
         * @param arguments Default frontend filtering or exact opaque-wrapper argv retention.
         * @return CONTINUE to proceed, or EXIT with an exit code
         */
        BootstrapResult execute(const OrchestrationConfig &config, int argc, char *argv[],
                                MPIBootstrapArgumentPolicy arguments = MPIBootstrapArgumentPolicy::FilterLaunchArguments);

        /**
         * @brief Classify every typed topology surface before opening drivers.
         * @param config Parsed rank devices, TP/PP tree, and dense/routed domains.
         * @return CPU-only only for a nonempty, exclusively CPU declaration.
         *
         * A CPU continuation with GPU experts still requires GPU discovery.
         * MPI degree and collective backend are not compute-device types.
         */
        static BootstrapDeviceIntent classifyDeviceIntent(
            const OrchestrationConfig &config);

        /**
         * @brief Install CLI CPU-only intent in the canonical startup authority.
         * @param config Parsed declarations, including named CPU domains.
         * @return The same typed intent used to choose the MPI launch profile.
         * @throws std::runtime_error if the child-process policy cannot be exported.
         *
         * Runs before discovery and before the already-in-MPI/direct-launch
         * early return. Explicit operational backend exclusions are preserved.
         */
        static BootstrapDeviceIntent installDeviceStartupIntent(
            const OrchestrationConfig &config);

        /** @return Sorted unique CPUs described by a Linux list/range string. */
        static std::vector<int> parseCpuList(const std::string &cpulist);
        /** @return Sysfs NUMA owner of a CPU, or -1 when unavailable. */
        static int detectCpuNumaNode(int cpu);
        /** @return First sibling representing the physical core containing a CPU. */
        static int physicalRepresentativeForCpu(int cpu);
        /**
         * @brief Relative compute capacity of one CPU for worker ordering.
         * @param cpu Logical CPU id.
         * @param sysfs_root Linux CPU sysfs root (a device-free fixture seam).
         * @return Kernel `cpu_capacity` (ARM big.LITTLE/hybrid), otherwise
         *         `cpufreq/cpuinfo_max_freq`, otherwise zero (unknown/uniform).
         */
        static long cpuCapacity(int cpu, const std::string &sysfs_root = "/sys/devices/system/cpu");
        /**
         * @brief Order physical-core representatives fastest first.
         * @param cpus Allowed representative CPUs in any order.
         * @param sysfs_root Linux CPU sysfs root (a device-free fixture seam).
         * @return Descending capacity, CPU id ascending within one capacity.
         *
         * Heterogeneous hosts (GB10: X925 + A725) place OpenMP worker 0, which
         * also runs serial sections, and the low worker ids on the fast cores.
         * Homogeneous hosts keep plain CPU-id order.
         */
        static std::vector<int> capacityOrderedCpus(
            std::vector<int> cpus, const std::string &sysfs_root = "/sys/devices/system/cpu");
        /**
         * @brief Plan OpenMP worker places: one physical core each, fastest first.
         * @param allowed_cpus The rank's admitted CPU mask; never widened.
         * @param sysfs_root Linux CPU sysfs root (a device-free fixture seam).
         * @return One place per physical core with an allowed CPU, holding that
         *         core's allowed siblings. Places follow capacityOrderedCpus()
         *         over the cores' first allowed siblings.
         * @details This is the OMP_PLACES=cores place list in capacity order.
         * Without published capacities it is libgomp's own CPU-id core order.
         */
        static std::vector<std::vector<int>> capacityOrderedCorePlaces(
            const std::vector<int> &allowed_cpus, const std::string &sysfs_root = "/sys/devices/system/cpu");
        /**
         * @brief Bind the persistent OpenMP worker pool to capacity-ordered core places.
         * @throws std::runtime_error If the rank mask is unreadable/empty or a binding fails.
         * @details The single authority for every OMP_PLACES=cores /
         * OMP_PROC_BIND=close rank. libgomp has no runtime place setter and
         * orders places by CPU id, so on GB10 workers 0-4, which include the
         * serial master, would otherwise run on A725 cores. One region of one
         * thread per place binds pool thread t, whose close-binding place is t,
         * to place t. Any later close team of w threads therefore occupies the
         * w fastest cores. The admitted set is this thread's mask joined
         * with libgomp's place list, which libgomp derived from the rank mask
         * before binding the initial thread; it is never widened further.
         */
        static void bindOpenMPWorkersFastestFirst();
        /** @return Whether current thread affinity meets the requested socket/core policy. */
        static bool verifyStartupThreadAffinity(int required_numa,
                                                bool require_physical_only,
                                                std::string &details);
        /** @return NUMA participants resolved from declared devices and actual inventory. */
        static std::set<int> resolveInferenceNUMANodes(
            const OrchestrationConfig &config,
            const DeviceManager &dm,
            const CPUTopology &cpu_topology);

        /**
         * @brief Launch discovery without narrowing it to selected inference endpoints.
         * @param config Hostfile, saved membership, or local automatic rank-count intent.
         * @return Slot-driven MPI launch; workers resolve affinity on each host.
         * @throws std::invalid_argument without a complete discovery launch intent.
         */
        static MPILaunchConfig discoveryLaunchConfig(const OrchestrationConfig &config);

        /**
         * @brief Size a hostfile-launched rank's worker team from actual affinity.
         * @param config User thread override, otherwise discovered physical cores.
         * @throws std::runtime_error when physical rank affinity is unavailable.
         */
        static void configureClusterRankThreads(const OrchestrationConfig &config);

        /**
         * @brief Apply an explicit thread request before canonical inventory publication.
         * @param config Parsed plan/serve request; an absent override retains bootstrap's team.
         * @throws std::runtime_error if the requested environment cannot be published.
         *
         * This does not rediscover cores or change affinity. Both frontends use
         * the same fixed OpenMP budget; inventory observes it separately from
         * physical capacity and workspace admission consumes that observation.
         */
        static void configureRequestedCPUThreads(const OrchestrationConfig &config);

        /** @brief Print the already initialized device inventory for diagnostics. */
        static void listDevices();
    };

} // namespace llaminar2
