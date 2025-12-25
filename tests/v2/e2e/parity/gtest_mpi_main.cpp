/**
 * @file gtest_mpi_main.cpp
 * @brief Custom GTest main with MPI initialization for E2E tests
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    // Initialize MPI before GTest
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
