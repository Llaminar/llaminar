#include "Qwen36DenseParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    /**
     * @brief Build a CUDA-only two-device LocalTP dense MTP fixture.
     *
     * Keeping the CUDA device list in the test definition, rather than relying
     * on the generic LocalTP helper default, makes the test name itself prove
     * which backend lane is being exercised.  The runtime availability guard in
     * the shared parity helper will skip cleanly if the host has fewer than two
     * CUDA devices.
     */
    DensePrefixRestoreParityCase cudaLocalTPCase()
    {
        auto test_case = qwen36DensePrefixParityCase(
            "Qwen3.6 dense CUDA LocalTP parity",
            DensePrefixParityTopology::LocalTP);
        test_case.devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
        };
        test_case.required_cuda_devices = 2;
        test_case.required_rocm_devices = 0;
        return test_case;
    }

    /**
     * @brief Build a ROCm-only two-device LocalTP dense MTP fixture.
     *
     * This is the explicit counterpart to @ref cudaLocalTPCase.  It preserves
     * the current ROCm coverage while making it impossible to mistake a green
     * generic LocalTP run for CUDA coverage.
     */
    DensePrefixRestoreParityCase rocmLocalTPCase()
    {
        auto test_case = qwen36DensePrefixParityCase(
            "Qwen3.6 dense ROCm LocalTP parity",
            DensePrefixParityTopology::LocalTP);
        test_case.devices = {
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::rocm(1),
        };
        test_case.required_cuda_devices = 0;
        test_case.required_rocm_devices = 2;
        return test_case;
    }
}

TEST(Qwen36CUDALocalTPPrefixMTPParity, PrefixRestoreFullHit)
{
    runDensePrefixRestoreParity(cudaLocalTPCase(), PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36CUDALocalTPPrefixMTPParity, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(cudaLocalTPCase(), false);
}

TEST(Qwen36CUDALocalTPPrefixMTPParity, MTPGreedyDepth3MatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(cudaLocalTPCase(), false, 3);
}

TEST(Qwen36CUDALocalTPPrefixMTPParity, MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseDynamicMTPParity(cudaLocalTPCase(), false);
}

TEST(Qwen36CUDALocalTPPrefixMTPParity, PrefixCacheMTPRestore)
{
    runDenseMTPParity(cudaLocalTPCase(), true);
}

TEST(Qwen36ROCmLocalTPPrefixMTPParity, PrefixRestoreFullHit)
{
    runDensePrefixRestoreParity(rocmLocalTPCase(), PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36ROCmLocalTPPrefixMTPParity, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(rocmLocalTPCase(), false);
}

TEST(Qwen36ROCmLocalTPPrefixMTPParity, MTPGreedyDepth3MatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(rocmLocalTPCase(), false, 3);
}

TEST(Qwen36ROCmLocalTPPrefixMTPParity, MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseDynamicMTPParity(rocmLocalTPCase(), false);
}

TEST(Qwen36ROCmLocalTPPrefixMTPParity, PrefixCacheMTPRestore)
{
    runDenseMTPParity(rocmLocalTPCase(), true);
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
