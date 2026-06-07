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

TEST(Qwen36SingleDevicePrefixMTPParity, Phase138VllmStyleCandidateEquivalence)
{
    runDensePhase138VllmStyleCandidateEquivalence(singleDeviceCase(), 2);
}

TEST(Qwen36SingleDevicePrefixMTPParity, Phase138VllmStyleCandidatePrefixRestoreEquivalence)
{
    runDensePhase138VllmStyleCandidatePrefixRestoreEquivalence(singleDeviceCase(), 3);
}

TEST(Qwen36SingleDevicePrefixMTPParity, Phase138VllmStyleCandidateStopTokenEquivalence)
{
    runDensePhase138VllmStyleCandidateStopTokenEquivalence(singleDeviceCase(), 2);
}

TEST(Qwen36SingleDevicePrefixMTPParity, NoMTPPhase138ContinuationMatchesPyTorch)
{
    runDenseNoMTPPhase138ContinuationMatchesPyTorch(singleDeviceCase(), 8);
}

TEST(Qwen36SingleDevicePrefixMTPParity, NoMTPPhase138ThinkContinuationStageParity)
{
    runDenseNoMTPPhase138ThinkContinuationStageParity(singleDeviceCase());
}

TEST(Qwen36SingleDevicePrefixMTPParity, Phase138VllmStyleCandidateContinuationEquivalence)
{
    runDensePhase138VllmStyleCandidateContinuationEquivalence(singleDeviceCase(), 3, 8);
}

TEST(Qwen36SingleDevicePrefixMTPParity, MTPShiftedRowCommitPreservesVerifierForward)
{
    runDenseMTPShiftedRowCommitPreservesVerifierForward(singleDeviceCase());
}

TEST(Qwen36SingleDevicePrefixMTPParity, MTPFirstTransactionLeavesSequentialState)
{
    runDenseMTPFirstTransactionLeavesSequentialState(singleDeviceCase());
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

TEST(Qwen36SingleDevicePrefixMTPParity, OneRowRestoreLongPrefixMatchesSequential)
{
    runDenseOneRowRestoreLongPrefixMatchesSequential(singleDeviceCase());
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
