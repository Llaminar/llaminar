/** @file KernelFactory.h
 * @brief Inert cache teardown witness for actual MPI shutdown replay.
 */
#pragma once
namespace mpi_entry_replay { void shutdownKernelCache(); }
namespace llaminar::v2::kernels {
class KernelFactory {
public:
    /** @brief Observe cache retirement without loading any kernel backend. */
    static void clearCache() { mpi_entry_replay::shutdownKernelCache(); }
};
}
