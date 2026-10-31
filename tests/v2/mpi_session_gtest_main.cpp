/**
 * @file mpi_session_gtest_main.cpp
 * @brief Process-entry ownership for fixtures that initialize MPI inside GTest.
 *
 * A test environment or test body may initialize MPI only after the executable
 * has entered the same ancestor lifetime used by production. This main performs
 * no MPI initialization itself, so each fixture retains its original thread
 * support and initialization/finalization policy.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include <gtest/gtest.h>

/**
 * @brief Own the original process before starting registered test environments.
 * @param argc Original argument count, before GTest parses its options.
 * @param argv Original arguments, forwarded unchanged to the owned child.
 * @return Original child status or the owned process's GTest result.
 */
int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
