/**
 * @file gtest.h
 * @brief Inert GTest surface used only by the actual MPI entrypoint replay.
 *
 * Target-private include ordering substitutes these calls without linking or
 * running GTest, MPI, or GPU libraries. The replay compiles the real entry body.
 */
#pragma once
namespace mpi_entry_replay
{
    void initializeTests(int *argc, char **argv);
    int runTests();
}
namespace testing
{
    inline void InitGoogleTest(int *argc, char **argv)
    {
        mpi_entry_replay::initializeTests(argc, argv);
    }
}
#define RUN_ALL_TESTS() ::mpi_entry_replay::runTests()
