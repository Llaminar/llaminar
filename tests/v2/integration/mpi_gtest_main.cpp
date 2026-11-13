/**
 * @file mpi_gtest_main.cpp
 * @brief MPI integration entrypoint with original production process ownership.
 *
 * The guardian owns process ancestry, and MPIProcessSession owns the actual
 * MPI_THREAD_MULTIPLE initialization and CPU collective process-pool lifetime.
 * Native CPU domains borrow that live original session; raw MPI initialization
 * alone cannot authorize their retained process resources. Canonical shutdown
 * closes collective admission, retires routers/kernel caches and GPU workers,
 * proves CPU domain retirement, then finalizes MPI and releases its pool.
 */

#include "app/MPIProcessSession.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"

#include <gtest/gtest.h>

/**
 * @brief Run the existing MPI integration cases within the original runtime scope.
 * @param argc Original process arguments, forwarded through guardian and MPI.
 * @param argv Original process argument vector, then parsed by GTest.
 * @return Original owned child status or the complete test result.
 * @throws std::exception For rejected production MPI initialization/shutdown.
 */
int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;
    // Declared after the GPU lifetime so canonical MPI shutdown clears caches
    // and workers before finalization; later GPU scope teardown is idempotent.
    auto mpi_process = llaminar2::MPIProcessSession::initialize(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
