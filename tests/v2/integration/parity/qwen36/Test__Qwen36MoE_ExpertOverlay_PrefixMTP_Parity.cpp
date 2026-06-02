#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "utils/Logger.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    MoEPrefixRestoreParityCase expertOverlayCase()
    {
        return qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE ExpertOverlay ROCm2TP hot + CPU2LocalTP cold parity",
            MoEPrefixParityTopology::ExpertOverlayRocm2TPHotCpu2LocalTPCold);
    }

    MoEPrefixRestoreParityCase rocmOnlyExpertOverlayCase()
    {
        return qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE ExpertOverlay ROCm2TP hot-only parity",
            MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly);
    }
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_ROCm2TPHotOnly)
{
    runMoEMTPParity(rocmOnlyExpertOverlayCase(), false);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_ROCm2TPHotOnly)
{
    runMoEMTPParity(rocmOnlyExpertOverlayCase(), true);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_ROCm2TPHot_CPU2LocalTPCold)
{
    runMoEMTPParity(expertOverlayCase(), false);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_ROCm2TPHot_CPU2LocalTPCold)
{
    runMoEMTPParity(expertOverlayCase(), true);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    initializeLogging();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
