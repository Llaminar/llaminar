/**
 * @file RuntimeInitPhase.h
 * @brief Post-MPI runtime initialization: MPI init, affinity, DeviceManager, runner creation
 *
 * Resolve declared placement against observed rank affinity before creating
 * allocator singletons. Pure placement helpers are shared with tests so MPI
 * rank numbering cannot accidentally become physical NUMA identity.
 * Automatic selection publishes root's complete apply configuration on the
 * discovery communicator. Saved and automatic configurations then use the same
 * MPI split authority before this frontend creates any inference runner.
 */

#pragma once

#include "app/AppContext.h"
#include "config/OrchestrationConfig.h"
#include "planning/AutomaticPlanningStartup.h"
#include <iosfwd>
#include <optional>
#include <variant>

namespace llaminar2
{
    struct NUMAInfo;

    /** @brief Terminal startup outcome when no inference context is returned. */
    enum class RuntimeInitExit
    {
        Completed = 0, ///< Successful dry-run or inactive-rank completion.
        Failed = 1,    ///< A precise startup failure has already been reported.
    };

    /** @brief Active runtime or an explicit terminal result, never a config flag. */
    using RuntimeInitResult = std::variant<AppContext, RuntimeInitExit>;

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
        /** @brief Use the same startup cost policy as the public plan command. */
        RuntimeInitPhase();
        /**
         * @brief Inject cost preparation without replacing frontend or admission lifecycle.
         * @param prepare Nonempty evidence policy; every discovery rank participates.
         * @throws std::invalid_argument for an absent preparation policy.
         *
         * This dependency changes only evidence collection/pricing. Model parsing, the
         * production compiler, PMA, publication, rank admission and runner
         * construction remain the ordinary frontend path.
         */
        explicit RuntimeInitPhase(AutomaticPlanningStartup::Prepare prepare);
        /**
         * @brief Resolve CPU shorthand to this rank's observed physical endpoint.
         * @param config Parsed intent; unchanged when CPU shorthand is not active.
         * @param mpi_rank Actual communicator rank, not a socket/NUMA index.
         * @param mpi_world_size Admitted communicator size.
         * @param numa_info Observed rank affinity from the canonical NUMA probe.
         * @throws std::runtime_error if the rank or its CPU locality is unknown.
         *
         * Retire the shorthand's device selector while installing the explicit
         * map. Repeated resolution is idempotent and requires no extra collective.
         */
        static void resolveCPUShorthand(
            OrchestrationConfig &config, int mpi_rank, int mpi_world_size,
            const NUMAInfo &numa_info);

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
         * between sockets. Automatic search also requires that view whenever
         * its hard backend constraints permit a GPU. A CPU-only hard filter
         * keeps the rank-local CPU observation without creating GPU contexts.
         *
         * @param config Parsed orchestration intent.
         * @param mpi_rank World rank being initialized.
         * @return True when DeviceManager must disable NUMA filtering.
         */
        static bool requiresHostWideAcceleratorVisibility(
            const OrchestrationConfig &config,
            int mpi_rank);

        /**
         * @brief Establish one rank's immutable CPU/backend and hardware observation identities.
         *
         * This is the shared pre-inventory transition for the serving frontend
         * and the public `plan` command.  It intentionally performs no MPI
         * collective, model access, planner selection, arena allocation, or
         * runner construction: its caller owns failure publication before a
         * subsequent inventory exchange.  CPU backend identity and
         * DeviceManager's complete hardware observation must agree, otherwise
         * planning could measure a different endpoint from the one that later
         * admits inference.
         *
         * The configuration is borrowed and never rewritten.  Serving resolves
         * the legacy CPU shorthand to an explicit map immediately before this
         * call; `plan` deliberately retains its automatic request unchanged so
         * its apply document is the sole selection authority.
         *
         * @param config Parsed request or already-applied configuration.
         * @param discovery_rank Exact rank in the unsplit discovery communicator.
         * @param discovery_world_size Size of that communicator.
         * @throws std::runtime_error when affinity or CPU ownership cannot be
         *         established for the declared topology.
         */
        static void initializeDiscoveryRuntime(
            const OrchestrationConfig &config,
            int discovery_rank,
            int discovery_world_size);

        /**
         * @brief Run dry-run preflight on an already-created runner.
         *
         * Initialize validation-only state and print the resolved plan on rank
         * zero. The caller owns runner destruction and the outer MPI session;
         * reporting must not encode failure by mutating requested configuration.
         *
         * @return true when dry-run validation succeeds; false when it fails
         * @param runner Caller-owned runner; this function does not shut it down.
         * @param mpi_rank Exact admitted communicator rank for output selection.
         * @param out Destination for the resolved plan on rank zero.
         */
        static bool runDryRunPreflight(IOrchestrationRunner &runner,
                                       int mpi_rank,
                                       std::ostream &out);

        /**
         * @brief Execute the runtime init phase
         *
         * On success, returns a fully-initialized AppContext with runner and tokenizer.
         * Successful dry-run and failure are distinct terminal values. Their
         * local owners unwind before the process session finalizes MPI.
         *
         * @param config Request, replaced by published apply policy before rank admission.
         * @param argc Argument count (MPI_Init may modify)
         * @param argv Argument vector (MPI_Init may modify)
         * @return Active context or typed terminal exit (errors already logged).
         */
        RuntimeInitResult execute(OrchestrationConfig &config, int &argc, char **&argv);

    private:
        AutomaticPlanningStartup::Prepare prepare_; ///< Shared evidence policy before discovery membership narrows.
    };

} // namespace llaminar2
