/**
 * @file MPIShutdown.cpp
 * @brief Centralized MPI shutdown implementation.
 * GPU admission closes and worker/native resources retire while MPI and vendor
 * runtimes are still live; later process-scope destruction is idempotent.
 */

#include "app/MPIShutdown.h"
#include "collective/BackendRouter.h"
#include "kernels/KernelFactory.h"
#include "backends/GPUDeviceProcessLifetime.h"
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

        // 3. Retire exact native workers after model/cache owners, while their
        // dependency runtimes and MPI are still live. No static teardown callback.
        GPUDeviceProcessLifetime::retireActive();

        // 4. Now safe to finalize MPI
        MPI_Finalize();
    }

} // namespace llaminar2
