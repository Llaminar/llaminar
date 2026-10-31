/**
 * @file Test__Qwen36_LocalPP_Prefix_Parity.cpp
 * @brief Native MPI fixture with an explicit process-entry backing owner.
 *
 * The executable enters the shared ancestor session before parsing arguments,
 * starting test environments, or initializing MPI. Direct binary runs retain
 * their actual descendants and runtime backing through terminal retirement.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include "Qwen36DenseParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    DensePrefixRestoreParityCase localPPCase()
    {
        return qwen36DensePrefixParityCase(
            "Qwen3.6 dense LocalPP parity",
            DensePrefixParityTopology::LocalPP);
    }
}

TEST(Qwen36LocalPPPrefixParity, PrefixRestoreFullHit)
{
    runDensePrefixRestoreParity(localPPCase(), PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36LocalPPPrefixParity, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(localPPCase(), false);
}

TEST(Qwen36LocalPPPrefixParity, MTPGreedyDepth3MatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(localPPCase(), false, 3);
}

TEST(Qwen36LocalPPPrefixParity, MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseDynamicMTPParity(localPPCase(), false);
}

TEST(Qwen36LocalPPPrefixParity, StochasticMTPVerifierMatchesAfterClearCache)
{
    runDenseStochasticMTPVerifierParity(localPPCase());
}

TEST(Qwen36LocalPPPrefixParity, PrefixCacheMTPRestore)
{
    runDenseMTPParity(localPPCase(), true);
}

int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    llaminar2::GPUDeviceProcessLifetime::retireActive();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
