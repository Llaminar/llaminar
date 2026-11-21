/**
 * @file CompiledGPUBackends.h
 * @brief Device identities for the GPU backend families compiled into a test binary.
 *
 * Device-free fixtures often exercise the CUDA and ROCm adapters side by side
 * with fake ordinals. A build compiles only its configured families (the GB10
 * ARM64 image is CUDA-only), and production code correctly rejects a device of
 * an uncompiled family. Such fixtures iterate the compiled families here;
 * a case that exists for one family only is compiled under that family's
 * macro and registered with REQUIRES_BACKEND, so CMake reports the omission
 * instead of a test failing or silently passing.
 */
#pragma once
#include "backends/DeviceId.h"
#include <vector>

namespace llaminar2::test {

/**
 * @param ordinal Fake or real device ordinal to use for every family.
 * @return One identity per compiled GPU family, CUDA first.
 */
inline std::vector<DeviceId> compiledGPUDevices(int ordinal)
{
    std::vector<DeviceId> devices;
#if defined(HAVE_CUDA)
    devices.push_back(DeviceId::cuda(ordinal));
#endif
#if defined(HAVE_ROCM)
    devices.push_back(DeviceId::rocm(ordinal));
#endif
    (void)ordinal;
    return devices;
}

/** @return Every compiled GPU backend family, CUDA first. */
inline std::vector<DeviceType> compiledGPUTypes()
{
    std::vector<DeviceType> types;
    for (const auto device : compiledGPUDevices(0))
        types.push_back(device.type);
    return types;
}

} // namespace llaminar2::test
