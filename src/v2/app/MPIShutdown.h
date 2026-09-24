/**
 * @file MPIShutdown.h
 * @brief Centralized MPI shutdown — cleans up static resources before MPI_Finalize.
 *
 * MPIProcessSession is the sole production caller of this infrastructure hook.
 * Commands and request modes release their scoped owners; they never invoke
 * finalization directly. Focused MPI infrastructure tests may own this edge.
 * This ensures GlobalBackendRouter (which holds MPI communicators via
 * MPITopology) is destroyed before MPI is finalized, preventing
 * "MPI_Comm_free called after MPI_FINALIZE" errors during static destruction.
 */

#pragma once

namespace llaminar2
{

    /**
     * @brief Infrastructure cleanup invoked after scoped context owners retire.
     *
     * Performs, in order:
     *   1. GlobalBackendRouter::shutdown()  — releases MPI communicators
     *   2. KernelFactory::clearCache()      — releases GPU pipeline lifetime owners
     *   3. MPI_Finalize()
     *
     * Safe to call even if GlobalBackendRouter was never initialized.
     * Idempotent with respect to MPI (checks MPI_Finalized before calling).
     */
    void mpiShutdown();

} // namespace llaminar2
