#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    MoEPrefixRestoreParityCase singleDeviceCase()
    {
        return qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE SingleDevice parity",
            MoEPrefixParityTopology::SingleDevice);
    }
}

TEST(Qwen36MoESingleDevicePrefixMTPParity, PrefixRestoreFullHit)
{
    runMoEPrefixRestoreParity(singleDeviceCase(), PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36MoESingleDevicePrefixMTPParity, PrefixRestorePartialHit)
{
    runMoEPrefixRestoreParity(singleDeviceCase(), PrefixRestoreParityMode::PartialHit);
}

TEST(Qwen36MoESingleDevicePrefixMTPParity, IncrementalDecodeMatchesFullContext)
{
    runMoEIncrementalDecodeMatchesFullContext(singleDeviceCase());
}

TEST(Qwen36MoESingleDevicePrefixMTPParity, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runMoEMTPParity(singleDeviceCase(), false);
}

TEST(Qwen36MoESingleDevicePrefixMTPParity, PrefixCacheMTPRestore)
{
    runMoEMTPParity(singleDeviceCase(), true);
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
