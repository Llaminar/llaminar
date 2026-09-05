/**
 * @file NodeExpertOverlayParityMain.cpp
 * @brief Shared MPI and backend lifetime entrypoint for node-overlay matrices.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParitySupport.h"

/** Run registered cells inside one explicit MPI/backend process lifetime. */
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /*
     * This dedicated integration main does not pass through RuntimeInitPhase,
     * which normally publishes rank identity to logging and PerfStats.  Set it
     * at the same MPI boundary so rank-qualified diagnostic exports cannot be
     * collapsed into (and race on) rank zero's file.
     */
    Logger::getInstance().setRank(rank);
    std::cout << "[Rank " << rank
              << "] Qwen3.5 MoE typed ExpertOverlay parity test\n";

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
