/**
 * @file PlanningGPUClockMarker.cu
 * @brief One-thread captured CUDA wall-clock stamp without native timing-event scheduling.
 * The original worker selects the device and owns the exact stream and PMA slot.
 */
#include "backends/PlanningGPUClockMarker.h"
#include <cuda_runtime.h>
namespace llaminar2
{
    namespace
    {
        /** @brief Publish only this slot's device-wide constant nanosecond clock. */
        __global__ void planningClockMarker(std::uint64_t *destination)
        {
            std::uint64_t tick;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick));
            *destination = tick;
        }
    }
    bool prepareCUDAPlanningClock()
    {
        // Function resolution is setup work. Native module/context ownership
        // remains the cache authority; no process-global ready flag survives reset.
        cudaFuncAttributes attributes{};
        return cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(planningClockMarker)) == cudaSuccess;
    }

    bool recordCUDAPlanningClock(std::uint64_t *destination, void *stream)
    {
        if (!destination || !stream) return false;
        planningClockMarker<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(destination);
        return cudaGetLastError() == cudaSuccess;
    }
}
