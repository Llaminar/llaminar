/**
 * @file RuntimeInitPhase.h
 * @brief Post-MPI runtime initialization: MPI init, affinity, DeviceManager, runner creation
 */

#pragma once

#include "app/AppContext.h"
#include "config/OrchestrationConfig.h"
#include <iosfwd>
#include <optional>

namespace llaminar2
{
    struct NUMAInfo;

    /**
     * @brief Post-MPI runtime initialization
     *
     * Handles everything from MPI_Init_thread through runner creation:
     * - MPI initialization and arg re-parse
     * - NUMA detection and affinity verification
     * - CPU shorthand runtime config mapping
     * - DeviceManager initialization
     * - OrchestrationRunner creation and initialization
     * - Tokenizer acquisition and chat template override
     */
    class RuntimeInitPhase
    {
    public:
        /**
         * @brief Resolve the immutable CPU backend NUMA identity for one rank.
         *
         * Explicit CPU device-map placement takes precedence and must agree
         * with the process affinity detected after MPI launch. Multi-rank
         * processes without an explicit CPU map inherit their detected local
         * node so host staging remains rank-local. A genuinely aggregate
         * single-process runtime returns `-1`.
         *
         * This pure policy function is public so unit tests can exhaust the
         * placement matrix without starting MPI or mutating the process-global
         * backend singleton.
         *
         * @param config Parsed orchestration configuration for this process.
         * @param mpi_rank World rank being initialized.
         * @param mpi_world_size Number of ranks in the world communicator.
         * @param numa_info Affinity-derived NUMA observation for this process.
         * @return Exact non-negative node for rank-local ownership, or `-1`
         *         only for a legitimate aggregate single-process runtime.
         * @throws std::runtime_error when explicit placement and affinity
         *         disagree or two explicit declarations conflict.
         */
        static int resolveCPUBackendNUMANode(
            const OrchestrationConfig &config,
            int mpi_rank,
            int mpi_world_size,
            const NUMAInfo &numa_info);

        /**
         * @brief Decide whether this rank must enumerate accelerators host-wide.
         *
         * CPU affinity remains an execution preference, not a device
         * visibility boundary. Any explicit accelerator intent—including a
         * rank-agnostic ExpertOverlay domain—requires the complete host view so
         * inventory binding can choose the actual owning rank after cards move
         * between sockets.
         *
         * @param config Parsed orchestration intent.
         * @param mpi_rank World rank being initialized.
         * @return True when DeviceManager must disable NUMA filtering.
         */
        static bool requiresHostWideAcceleratorVisibility(
            const OrchestrationConfig &config,
            int mpi_rank);

        /**
         * @brief Run dry-run preflight on an already-created runner.
         *
         * This helper owns the dry-run-specific lifecycle after MPI/device setup:
         * initialize the runner in validation-only mode, print the resolved plan
         * on rank 0, and shut the runner back down.
         *
         * @return true when dry-run validation succeeds; false when it fails
         */
        static bool runDryRunPreflight(OrchestrationConfig &config,
                                       IOrchestrationRunner &runner,
                                       int mpi_rank,
                                       std::ostream &out);

        /**
         * @brief Execute the runtime init phase
         *
         * On success, returns a fully-initialized AppContext with runner and tokenizer.
         * On failure, returns nullopt (errors already logged, MPI finalized).
         *
         * @param config Orchestration config (may be mutated for CPU shorthand mapping)
         * @param argc Argument count (MPI_Init may modify)
         * @param argv Argument vector (MPI_Init may modify)
         * @return AppContext on success, nullopt on failure
         */
        std::optional<AppContext> execute(OrchestrationConfig &config, int &argc, char **&argv);
    };

} // namespace llaminar2
