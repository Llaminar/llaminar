/**
 * @file core_gtest_main.cpp
 * @brief Core-linked test entry with lexical ownership of native GPU workers.
 *
 * The process guardian owns filesystem/MPI ancestry. Inside its actual child,
 * this frame retires workers before returning to runtime exit callbacks, even
 * when a later test initializes another vendor, device or native library.
 * Construction performs no GPU discovery; device-free core tests remain so.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include <gtest/gtest.h>

/** @brief Run tests inside the same native lifetime boundary as the application. */
int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_lifetime;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
