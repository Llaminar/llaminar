/**
 * @file Test__ROCmPrefillGraphCacheExecution.cpp
 * @brief ROCm integration test entry point for exact-bucket prefill graph cache execution.
 *
 * Binds the shared prefill graph-cache lifecycle test to the ROCm backend so
 * the test is named, linked, and scheduled independently from CUDA coverage.
 */

#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;

namespace
{
    /// @brief Registers the ROCm device-context factory for this test binary.
    void ensurePrefillGraphCacheBackendRegistered()
    {
        ensureAMDFactoryRegistered();
    }

    /// @brief Returns whether the linked ROCm backend has an available device.
    bool hasPrefillGraphCacheBackendSupport()
    {
        return GPUDeviceContextPool::instance().hasAMDSupport();
    }

    /// @brief Returns the ROCm device used by this integration test.
    DeviceId prefillGraphCacheBackendDeviceId()
    {
        return DeviceId::rocm(0);
    }

    /// @brief Returns the gtest skip message when ROCm is unavailable.
    const char *prefillGraphCacheBackendSkipMessage()
    {
        return "ROCm not available";
    }
}

#include "Test__PrefillGraphCacheExecutionCommon.h"
