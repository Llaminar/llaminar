/**
 * @file PlanningGPUClockMarker.hip
 * @brief One-thread captured HIP constant-frequency clock stamp without timing-event scheduling.
 * The original worker selects the device and owns the exact stream and PMA slot.
 */
#include "backends/PlanningGPUClockMarker.h"
#include <hip/hip_runtime.h>
namespace llaminar2
{
    namespace
    {
        /** @brief Publish only this slot's device-wide constant-frequency wall clock. */
        __global__ void planningClockMarker(std::uint64_t *destination)
        {
            *destination = static_cast<std::uint64_t>(wall_clock64());
        }
    }
    bool prepareROCmPlanningClock()
    {
        // Function resolution is setup work. Native module/context ownership
        // remains the cache authority; no process-global ready flag survives reset.
        hipFuncAttributes attributes{};
        return hipFuncGetAttributes(&attributes, reinterpret_cast<const void *>(planningClockMarker)) == hipSuccess;
    }

    bool recordROCmPlanningClock(std::uint64_t *destination, void *stream)
    {
        if (!destination || !stream) return false;
        planningClockMarker<<<1, 1, 0, static_cast<hipStream_t>(stream)>>>(destination);
        return hipGetLastError() == hipSuccess;
    }
}
