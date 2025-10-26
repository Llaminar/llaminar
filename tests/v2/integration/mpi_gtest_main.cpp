/**
 * @file mpi_gtest_main.cpp
 * @brief Custom GTest main that initializes MPI for integration tests
 */

#include <gtest/gtest.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
