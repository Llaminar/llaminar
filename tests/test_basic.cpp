#include "../src/common.h"
#include "../src/argument_parser.h"
#include "test_mpi_utils.h" // unified MPI helpers
#include <gtest/gtest.h>
#include <iostream>

// Basic MPI test function
void runBasicTest(int rank, int size)
{
    std::cout << "Hello from Llaminar! Process " << rank
              << " of " << size << " processes." << std::endl;
}

TEST(BasicMPISmoke, RanksPrintHello)
{
    int rank = llaminar::test_util::MPIEnvironment::rank();
    int world = llaminar::test_util::MPIEnvironment::world();
    // Just invoke the legacy print function; not asserting output capture here.
    runBasicTest(rank, world);
    SUCCEED();
}

LLAMINAR_DEFINE_GTEST_MPI_MAIN();