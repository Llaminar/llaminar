#include "TestMpiUtils.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <thread>

TEST(MPIFinalizeProbe, SimpleBarrier)
{
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    // Simple collective to ensure MPI working.
    int value = rank + 1;
    int sum = 0;
    MPI_Allreduce(&value, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(sum, world * (world + 1) / 2) << "Allreduce mismatch";
    // Sleep briefly to allow async progress threads (if any) to settle.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

LLAMINAR_DEFINE_GTEST_MPI_MAIN();
