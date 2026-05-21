/**
 * @file Test__CUDAPrefillGraphCacheExecution.cpp
 * @brief CUDA integration test entry point for exact-bucket prefill graph cache execution.
 *
 * Binds the shared prefill graph-cache lifecycle test to the CUDA backend so
 * CUDA graph capture/replay is exercised as a first-class integration test.
 */

#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;

namespace
{
    /// @brief Registers the CUDA device-context factory for this test binary.
    void ensurePrefillGraphCacheBackendRegistered()
    {
        ensureNvidiaFactoryRegistered();
    }

    /// @brief Returns whether the linked CUDA backend has an available device.
    bool hasPrefillGraphCacheBackendSupport()
    {
        return GPUDeviceContextPool::instance().hasNvidiaSupport();
    }

    /// @brief Returns the CUDA device used by this integration test.
    DeviceId prefillGraphCacheBackendDeviceId()
    {
        return DeviceId::cuda(0);
    }

    /// @brief Returns the gtest skip message when CUDA is unavailable.
    const char *prefillGraphCacheBackendSkipMessage()
    {
        return "CUDA not available";
    }
}

#include "Test__PrefillGraphCacheExecutionCommon.h"
