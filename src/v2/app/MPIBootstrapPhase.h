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
         * @return CONTINUE to proceed, or EXIT with an exit code
         */
        BootstrapResult execute(const OrchestrationConfig &config, int argc, char *argv[]);

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
        /** @return Whether current thread affinity meets the requested socket/core policy. */
        static bool verifyStartupThreadAffinity(int required_numa,
                                                bool require_physical_only,
                                                std::string &details);
        /** @return NUMA participants resolved from declared devices and actual inventory. */
        static std::set<int> resolveInferenceNUMANodes(
            const OrchestrationConfig &config,
            const DeviceManager &dm,
            const CPUTopology &cpu_topology);

        /** @brief Print the already initialized device inventory for diagnostics. */
        static void listDevices();
    };

} // namespace llaminar2
