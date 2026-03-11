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

        // Exposed for unit testing / reuse

        static std::vector<int> parseCpuList(const std::string &cpulist);
        static int detectCpuNumaNode(int cpu);
        static int physicalRepresentativeForCpu(int cpu);
        static bool verifyStartupThreadAffinity(int required_numa,
                                                bool require_physical_only,
                                                std::string &details);
        static std::set<int> resolveInferenceNUMANodes(
            const OrchestrationConfig &config,
            const DeviceManager &dm,
            const CPUTopology &cpu_topology);

        static void listDevices();
    };

} // namespace llaminar2
