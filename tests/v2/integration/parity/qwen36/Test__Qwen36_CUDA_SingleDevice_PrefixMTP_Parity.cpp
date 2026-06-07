#include "Qwen36DenseParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    DensePrefixRestoreParityCase cudaSingleDeviceCase()
    {
        auto test_case = qwen36DensePrefixParityCase(
            "Qwen3.6 dense CUDA SingleDevice parity",
            DensePrefixParityTopology::SingleDevice);
        test_case.devices = {GlobalDeviceAddress::cuda(0)};
        test_case.required_cuda_devices = 1;
        test_case.required_rocm_devices = 0;
        return test_case;
    }
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, PrefixRestoreFullHit)
{
    runDensePrefixRestoreParity(cudaSingleDeviceCase(), PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, PrefixRestorePartialHit)
{
    runDensePrefixRestoreParity(cudaSingleDeviceCase(), PrefixRestoreParityMode::PartialHit);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, SplitPrefillMatchesPyTorchDecodeTokens)
{
    runDenseSplitPrefillParity(cudaSingleDeviceCase(), 4);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(cudaSingleDeviceCase(), false);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseDynamicMTPParity(cudaSingleDeviceCase(), false);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, PrefixCacheMTPRestore)
{
    runDenseMTPParity(cudaSingleDeviceCase(), true);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, PrefixCacheMTPDynamicDepthRestore)
{
    runDenseDynamicMTPParity(cudaSingleDeviceCase(), true);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, MTPBenchmarkPromptDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseBenchmarkStyleDynamicMTPParitySinglePass(cudaSingleDeviceCase());
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, MTPBenchmarkPromptFixedDepth1MatchesPyTorchDecodeTokens)
{
    runDenseBenchmarkStyleFixedMTPParity(cudaSingleDeviceCase(), 1);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, MTPBenchmarkPromptFixedDepth3MatchesPyTorchDecodeTokens)
{
    runDenseBenchmarkStyleFixedMTPParity(cudaSingleDeviceCase(), 3);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, Phase138VllmStyleCandidateEquivalence)
{
    runDensePhase138VllmStyleCandidateEquivalence(cudaSingleDeviceCase(), 2);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, Phase138VllmStyleCandidatePrefixRestoreEquivalence)
{
    runDensePhase138VllmStyleCandidatePrefixRestoreEquivalence(cudaSingleDeviceCase(), 1);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, NoMTPBenchmarkStyleFreshRunnerDeterminism)
{
    runDenseNoMTPBenchmarkStyleFreshRunnerDeterminism(
        cudaSingleDeviceCase(),
        /*decode_token_budget=*/128,
        /*reused_cycle_count=*/4);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, MTPVerifierRowsPostSidecarMatchRestoredReplay)
{
    runDenseMTPVerifierRowsPostSidecarEquivalence(cudaSingleDeviceCase());
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, M2VerifierLongPrefixMatchesSequential)
{
    runDenseM2VerifierLongPrefixMatchesSequential(cudaSingleDeviceCase());
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, OneRowRestoreLongPrefixMatchesSequential)
{
    runDenseOneRowRestoreLongPrefixMatchesSequential(cudaSingleDeviceCase());
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, M2VerifierFinalStateLongPrefixIsNotDecodeEquivalent)
{
    runDenseM2VerifierFinalStateLongPrefixIsNotDecodeEquivalent(cudaSingleDeviceCase());
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, M4VerifierLongPrefixIsNotDecodeEquivalent)
{
    runDenseM4VerifierLongPrefixIsNotDecodeEquivalent(cudaSingleDeviceCase());
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, M4VerifierRow3RestoreIsNotDecodeEquivalent)
{
    runDenseM4VerifierRow3RestoreFeedsNextVerifierRows(cudaSingleDeviceCase());
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, SidecarChainVerifierStateShortcutCandidateIsNotDecodeEquivalent)
{
    runDenseM4SidecarChainVerifierStateShortcutCandidate(
        cudaSingleDeviceCase(),
        /*expect_decode_equivalent=*/false);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, AllPositionCatchupCandidateFailsCommitReplay)
{
    runDenseAllPositionCatchupCandidateFailsCommitReplay(
        cudaSingleDeviceCase(),
        /*use_benchmark_prompt=*/true);
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, MTPEnabledForwardOnlyMatchesNoMTPDecode)
{
    runDenseMTPEnabledForwardOnlyMatchesNoMTP(cudaSingleDeviceCase());
}

TEST(Qwen36CUDASingleDevicePrefixMTPParity, MTPStochasticSamplingVerifierRuns)
{
    runDenseStochasticMTPVerifierParity(cudaSingleDeviceCase());
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
