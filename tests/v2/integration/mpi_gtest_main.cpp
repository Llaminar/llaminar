/**
 * @file mpi_gtest_main.cpp
 * @brief Custom GTest main with background-safe MPI initialization.
 *
 * Integration binaries use the same MPI_THREAD_MULTIPLE contract as the
 * production runtime. This permits real tests of maintenance/progress threads
 * while remaining compatible with tests that make MPI calls only on main.
 */

#include <gtest/gtest.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    const int init_result = MPI_Init_thread(
        &argc,
        &argv,
        MPI_THREAD_MULTIPLE,
        &provided);
    if (init_result != MPI_SUCCESS || provided < MPI_THREAD_MULTIPLE)
        return 2;

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    const int result = RUN_ALL_TESTS();

    // Finalize MPI
    if (MPI_Finalize() != MPI_SUCCESS)
        return 3;

    return result;
}
