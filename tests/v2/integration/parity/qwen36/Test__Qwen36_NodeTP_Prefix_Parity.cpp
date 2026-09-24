/**
 * @file Test__Qwen36_NodeTP_Prefix_Parity.cpp
 * @brief Qwen3.6 dense NodeTP prefix-cache and stochastic MTP parity.
 */

#include "Qwen36DenseParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    DensePrefixRestoreParityCase nodeTPCase()
    {
        return qwen36DensePrefixParityCase(
            "Qwen3.6 dense NodeTP parity",
            DensePrefixParityTopology::NodeTP);
    }
}

TEST(Qwen36NodeTPPrefixParity, PrefixRestoreFullHit)
{
    runDensePrefixRestoreParity(nodeTPCase(), PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36NodeTPPrefixParity, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(nodeTPCase(), false);
}

TEST(Qwen36NodeTPPrefixParity, MTPGreedyDepth3MatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(nodeTPCase(), false, 3);
}

TEST(Qwen36NodeTPPrefixParity, MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseDynamicMTPParity(nodeTPCase(), false);
}

TEST(Qwen36NodeTPPrefixParity, PrefixCacheMTPRestore)
{
    runDenseMTPParity(nodeTPCase(), true);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
