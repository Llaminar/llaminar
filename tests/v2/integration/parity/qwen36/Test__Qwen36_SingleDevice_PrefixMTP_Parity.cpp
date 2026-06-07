#include "Qwen36DenseParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    DensePrefixRestoreParityCase singleDeviceCase()
    {
        return qwen36DensePrefixParityCase(
            "Qwen3.6 dense SingleDevice parity",
            DensePrefixParityTopology::SingleDevice);
    }
}

TEST(Qwen36SingleDevicePrefixMTPParity, PrefixRestoreFullHit)
{
    runDensePrefixRestoreParity(singleDeviceCase(), PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36SingleDevicePrefixMTPParity, PrefixRestorePartialHit)
{
    runDensePrefixRestoreParity(singleDeviceCase(), PrefixRestoreParityMode::PartialHit);
}

TEST(Qwen36SingleDevicePrefixMTPParity, SplitPrefillMatchesPyTorchDecodeTokens)
{
    runDenseSplitPrefillParity(singleDeviceCase(), 4);
}

TEST(Qwen36SingleDevicePrefixMTPParity, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(singleDeviceCase(), false);
}

TEST(Qwen36SingleDevicePrefixMTPParity, MTPGreedyDepth3MatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(singleDeviceCase(), false, 3);
}

TEST(Qwen36SingleDevicePrefixMTPParity, Phase138DirectDepth3MatchesPyTorchDecodeTokens)
{
    ScopedEnvironmentValues phase138_env({
        {"LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE", "vllm_style_spec_decode"},
        {"LLAMINAR_MTP_PHASE138_DIRECT_CANDIDATE", "1"},
    });
    runDenseMTPParity(singleDeviceCase(), false, 3);
}

TEST(Qwen36SingleDevicePrefixMTPParity, Phase138DirectDepth1PrefixCacheMTPRestore)
{
    ScopedEnvironmentValues phase138_env({
        {"LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE", "vllm_style_spec_decode"},
        {"LLAMINAR_MTP_PHASE138_DIRECT_CANDIDATE", "1"},
    });
    runDenseMTPParity(singleDeviceCase(), true, 1);
}

TEST(Qwen36SingleDevicePrefixMTPParity, MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseDynamicMTPParity(singleDeviceCase(), false);
}

TEST(Qwen36SingleDevicePrefixMTPParity, PrefixCacheMTPRestore)
{
    runDenseMTPParity(singleDeviceCase(), true);
}

TEST(Qwen36SingleDevicePrefixMTPParity, PrefixCacheMTPDynamicDepthRestore)
{
    runDenseDynamicMTPParity(singleDeviceCase(), true);
}

TEST(Qwen36SingleDevicePrefixMTPParity, MTPStochasticSamplingVerifierRuns)
{
    runDenseStochasticMTPVerifierParity(singleDeviceCase());
}

TEST(Qwen36SingleDevicePrefixMTPParity, MTPVerifierRowsPostSidecarMatchRestoredReplay)
{
    runDenseMTPVerifierRowsPostSidecarEquivalence(singleDeviceCase());
}

TEST(Qwen36SingleDevicePrefixMTPParity, M2VerifierLongPrefixMatchesSequential)
{
    runDenseM2VerifierLongPrefixMatchesSequential(singleDeviceCase());
}

TEST(Qwen36SingleDevicePrefixMTPParity, OneRowRestoreLongPrefixMatchesSequential)
{
    runDenseOneRowRestoreLongPrefixMatchesSequential(singleDeviceCase());
}

TEST(Qwen36SingleDevicePrefixMTPParity, M4VerifierLongPrefixIsNotDecodeEquivalent)
{
    runDenseM4VerifierLongPrefixIsNotDecodeEquivalent(singleDeviceCase());
}

TEST(Qwen36SingleDevicePrefixMTPParity, SidecarChainVerifierStateShortcutCandidateIsNotDecodeEquivalent)
{
    runDenseM4SidecarChainVerifierStateShortcutCandidate(
        singleDeviceCase(),
        /*expect_decode_equivalent=*/false);
}

TEST(Qwen36SingleDevicePrefixMTPParity, AllPositionCatchupCandidateFailsCommitReplay)
{
    runDenseAllPositionCatchupCandidateFailsCommitReplay(
        singleDeviceCase(),
        /*use_benchmark_prompt=*/true);
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
