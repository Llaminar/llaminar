/** @file BackendRouter.h
 * @brief Inert collective teardown witness for the actual shared MPI main.
 */
#pragma once
namespace mpi_entry_replay
{
    void shutdownCollectives();
}
namespace llaminar2
{
    class GlobalBackendRouter
    {
    public:
        static void shutdown() { mpi_entry_replay::shutdownCollectives(); }
    };
}
