/** @file GPUDeviceContextPool.h
 * @brief Inert teardown witness; this replay never loads a GPU runtime.
 */
#pragma once
namespace mpi_entry_replay
{
    void shutdownContexts();
}
namespace llaminar2
{
    class GPUDeviceContextPool
    {
    public:
        static GPUDeviceContextPool &instance()
        {
            static GPUDeviceContextPool pool;
            return pool;
        }
        void shutdown() { mpi_entry_replay::shutdownContexts(); }
    };
}
