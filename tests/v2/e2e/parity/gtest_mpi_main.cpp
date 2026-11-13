/**
 * @file gtest_mpi_main.cpp
 * @brief Custom GTest main with MPI initialization for E2E tests
 * @author David Sanftenberg
 */

#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include "backends/GPUDeviceContextPool.h"

int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    // Initialize MPI before GTest
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Shutdown GPU contexts before MPI_Finalize to avoid SIGSEGV in
    // HIP runtime teardown (static destruction order fiasco).
    llaminar2::GPUDeviceProcessLifetime::retireActive();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
