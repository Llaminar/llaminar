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

TEST(Qwen36LocalPPPrefixParity, PrefixCacheMTPRestore)
{
    runDenseMTPParity(localPPCase(), true);
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
