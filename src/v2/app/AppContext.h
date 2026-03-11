/**
 * @file AppContext.h
 * @brief Shared application state passed to execution modes
 */

#pragma once

#include "config/OrchestrationConfig.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "utils/Tokenizer.h"
#include "utils/MPIContext.h"
#include <memory>
#include <mpi.h>

namespace llaminar2
{

    /**
     * @brief Shared state produced by RuntimeInitPhase, consumed by execution modes
     */
    struct AppContext
    {
        OrchestrationConfig config;
        std::shared_ptr<MPIContext> mpi_ctx;
        std::unique_ptr<IOrchestrationRunner> runner;
        std::shared_ptr<ITokenizer> tokenizer;

        bool isRootRank() const { return mpi_ctx->rank() == 0; }

        /// Call to clean up MPI when done
        void finalize()
        {
            if (runner)
                runner->shutdown();
            MPI_Finalize();
        }
    };

} // namespace llaminar2
