/**
 * @file Test__NCCLDeviceScope.cpp
 * @brief Native NCCL preparation must not acquire unselected CUDA devices.
 *
 * This executable starts with no CUDA contexts and observes primary-context
 * state through the canonical Driver API table, which does not create one.
 * Initializing a one-member collective must leave every excluded GPU untouched.
 * The regression guards against an unused all-visible-GPU copy communicator
 * that duplicated native buffers and bypassed the declared device membership.
 */
#include "backends/cuda/CUDADriverApi.h"
#include "collective/DeviceGroup.h"
#include "collective/backends/NCCLBackend.h"

#include <gtest/gtest.h>

using namespace llaminar2;

/**
 * @test Preparation and retirement preserve the exact declared device scope.
 *
 * Selecting the last visible device also catches accidental GPU-zero ownership.
 * Repeat the backend lifecycle without resetting CUDA: native context retention
 * on a selected participant is valid, activation of an excluded one is not.
 */
TEST(NCCLDeviceScope, LifecycleDoesNotInitializeExcludedDevices)
{
    const auto &driver = CUDADriverApi::instance();
    ASSERT_EQ(driver.init(0), CUDA_SUCCESS);
    int count = 0;
    ASSERT_EQ(driver.deviceGetCount(&count), CUDA_SUCCESS);
    ASSERT_GE(count, 2) << "Device-scope regression requires two visible CUDA GPUs";

    // Driver observation, unlike cudaSetDevice, cannot initialize the runtime
    // context whose absence this test is intended to prove.
    for (int ordinal = 0; ordinal < count; ++ordinal)
    {
        CUdevice device{};
        unsigned flags = 0;
        int active = -1;
        ASSERT_EQ(driver.deviceGet(&device, ordinal), CUDA_SUCCESS);
        ASSERT_EQ(driver.devicePrimaryCtxGetState(device, &flags, &active), CUDA_SUCCESS);
        ASSERT_EQ(active, 0) << "The isolated process already owns CUDA:" << ordinal;
    }

    const int selected = count - 1;
    const auto group = DeviceGroupBuilder()
        .setName("selected-cuda-only")
        .addDevice(DeviceId::cuda(selected))
        .build();
    NCCLBackend backend;
    for (int generation = 0; generation < 3; ++generation)
    {
        SCOPED_TRACE(generation);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();
        ASSERT_TRUE(backend.isInitialized());
        EXPECT_EQ(backend.numRanks(), 1);
        for (int ordinal = 0; ordinal < count; ++ordinal)
        {
            CUdevice device{};
            unsigned flags = 0;
            int active = -1;
            ASSERT_EQ(driver.deviceGet(&device, ordinal), CUDA_SUCCESS);
            ASSERT_EQ(driver.devicePrimaryCtxGetState(device, &flags, &active), CUDA_SUCCESS);
            EXPECT_EQ(active, ordinal == selected ? 1 : 0) << "CUDA:" << ordinal;
        }
        backend.shutdown();
        EXPECT_FALSE(backend.isInitialized());
    }
}
