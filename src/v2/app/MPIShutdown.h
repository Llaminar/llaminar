/**
 * @file MPIShutdown.h
 * @brief Centralized MPI shutdown — cleans up static resources before MPI_Finalize.
 *
 * ALL code that needs to call MPI_Finalize should use mpiShutdown() instead.
 * This ensures GlobalBackendRouter (which holds MPI communicators via
 * MPITopology) is destroyed before MPI is finalized, preventing
 * "MPI_Comm_free called after MPI_FINALIZE" errors during static destruction.
 */

#pragma once

namespace llaminar2
{

    /**
     * @brief Clean up global state and finalize MPI.
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
