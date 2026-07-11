/**
 * @file Test__Qwen36MoE_NodeLocalTP_PrefixMTP_Parity.cpp
 * @brief Qwen3.6 MoE prefix-cache and stochastic MTP coverage for CPU NodeLocalTP.
 *
 * CPU tensor parallelism in Llaminar is represented as NodeLocalTP: each CPU
 * socket is a separate MPI rank with its own weight manager, and collectives
 * coordinate the tensor-parallel MoE verifier path.  These tests are the MoE
 * counterpart to the dense Qwen3.6 NodeLocalTP prefix-MTP suite and close the
 * CPUx2 coverage gap in the stochastic depth matrix.
 */

#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <mpi.h>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    /**
     * @brief Build the baseline CPUx2 NodeLocalTP MoE MTP fixture.
     *
     * @return Qwen3.6 MoE fixture using two CPU MPI ranks as a NodeLocalTP domain.
     */
    MoEPrefixRestoreParityCase cpuNodeLocalTPCase()
    {
        auto test_case = qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE CPUx2 NodeLocalTP parity",
            MoEPrefixParityTopology::NodeLocalTP);
        test_case.decode_steps = 2;
        return test_case;
    }

    /**
     * @brief Build the CPUx2 NodeLocalTP stochastic benchmark fixture.
     *
     * @return NodeLocalTP fixture with a longer prompt and four decode steps so
     * fixed depth-3 and dynamic-depth stochastic MTP both execute real verifier
     * windows.
     */
    MoEPrefixRestoreParityCase cpuNodeLocalTPBenchmarkPromptCase()
    {
        auto test_case = cpuNodeLocalTPCase();
        test_case.name =
            "Qwen3.6 MoE CPUx2 NodeLocalTP benchmark-prompt MTP diagnostic";
        test_case.prompt = qwen36MoEBenchmarkPrompt();
        test_case.metadata_envs = {
            "LLAMINAR_QWEN36_MOE_NODELOCALTP_MTP_DIAGNOSTIC_METADATA"};
        test_case.default_metadata_path =
            "pytorch_qwen36_moe_cpu2_nodelocaltp_mtp_diagnostic_snapshots/metadata.txt";
        test_case.decode_steps = 4;
        test_case.max_seq_len = 768;
        return test_case;
    }
} // namespace

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, PrefixRestoreFullHit)
{
    runMoEPrefixRestoreParity(
        cpuNodeLocalTPCase(),
        PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, MTPGreedyMatchesBaselineTokens)
{
    runMoEMTPParity(cpuNodeLocalTPCase(), false);
}

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, MTPGreedyDepth3MatchesBaselineTokens)
{
    runMoEMTPParity(cpuNodeLocalTPBenchmarkPromptCase(), false, 3);
}

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, PrefixCacheMTPRestore)
{
    runMoEMTPParity(cpuNodeLocalTPCase(), true);
}

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, StochasticMTPDepth1VerifierMatchesAfterClearCache)
{
    runMoEStochasticMTPVerifierParity(cpuNodeLocalTPBenchmarkPromptCase(), 1);
}

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, StochasticMTPDepth2VerifierMatchesAfterClearCache)
{
    runMoEStochasticMTPVerifierParity(cpuNodeLocalTPBenchmarkPromptCase(), 2);
}

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, StochasticMTPDepth3VerifierMatchesAfterClearCache)
{
    runMoEStochasticMTPVerifierParity(
        cpuNodeLocalTPBenchmarkPromptCase(),
        3,
        true);
}

/**
 * @brief Reproduce maximum-depth stochastic verification from a restored prefix.
 *
 * Dynamic-depth coverage can demote after the first request and accidentally
 * exercise only a shallower restored-prefix verifier.  Keep this fixed-depth
 * cell so the maximum four-row group is independently proven after both the
 * first prefix restore and the subsequent explicit request reset.
 */
TEST(Qwen36MoENodeLocalTPPrefixMTPParity, StochasticMTPDepth3VerifierMatchesAfterPrefixRestore)
{
    runMoEStochasticMTPVerifierParity(
        cpuNodeLocalTPBenchmarkPromptCase(),
        3,
        true,
        MTPDepthPolicyConfig{},
        true);
}

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterClearCache)
{
    runMoEStochasticMTPVerifierParity(
        cpuNodeLocalTPBenchmarkPromptCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3));
}

TEST(Qwen36MoENodeLocalTPPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore)
{
    runMoEStochasticMTPVerifierParity(
        cpuNodeLocalTPBenchmarkPromptCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int local_result = RUN_ALL_TESTS();

    int global_result = local_result;
    MPI_Allreduce(
        &local_result,
        &global_result,
        1,
        MPI_INT,
        MPI_MAX,
        MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(global_result);
}
