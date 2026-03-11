/**
 * @file RuntimeInitPhase.h
 * @brief Post-MPI runtime initialization: MPI init, affinity, DeviceManager, runner creation
 */

#pragma once

#include "app/AppContext.h"
#include "config/OrchestrationConfig.h"
#include <optional>

namespace llaminar2
{

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
