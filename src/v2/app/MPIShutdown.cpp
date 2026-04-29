/**
 * @file MPIShutdown.cpp
 * @brief Centralized MPI shutdown implementation.
 */

#include "app/MPIShutdown.h"
#include "collective/BackendRouter.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"

#include <mpi.h>

namespace llaminar2
{

    void mpiShutdown()
    {
        // Guard against double-finalize
        int already_finalized = 0;
        MPI_Finalized(&already_finalized);
        if (already_finalized)
        {
            return;
        }

        // 1. Release GlobalBackendRouter — owns shared_ptr<MPIContext> → MPITopology
        //    which holds MPI communicators (intra_node_comm_, inter_node_comm_).
        //    Must be freed while MPI is still active.
        GlobalBackendRouter::shutdown();

        // 2. Release KernelFactory caches — may hold shared_ptr<LoadOrchestrator>
        //    (GPU pipeline lifetime owners) that transitively reference GPU resources.
        //    Clearing here prevents delayed destruction during static cleanup.
        llaminar::v2::kernels::KernelFactory::clearCache();

        // 3. Now safe to finalize MPI
        MPI_Finalize();
    }

} // namespace llaminar2
