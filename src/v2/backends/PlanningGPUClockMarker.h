/**
 * @file PlanningGPUClockMarker.h
 * @brief Exact-stream fixed-frequency device timestamp instrumentation for startup graphs.
 */
#pragma once
#include <cstdint>
namespace llaminar2
{
    /** @brief Resolve CUDA marker function attributes on the original worker before any capture. */
    bool prepareCUDAPlanningClock();
    /** @brief Resolve HIP marker function attributes on the original worker before any capture. */
    bool prepareROCmPlanningClock();
    /** @brief Write one CUDA globaltimer nanosecond tick with one captured thread; false is fatal. */
    bool recordCUDAPlanningClock(std::uint64_t *destination, void *stream);
    /** @brief Write one HIP wall_clock64 tick with one captured thread; false is fatal. */
    bool recordROCmPlanningClock(std::uint64_t *destination, void *stream);
}
