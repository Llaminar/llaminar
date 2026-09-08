/**
 * @file Test__Qwen36MoE_NodeTP_PrefixMTP_Parity.cpp
 * @brief Qwen3.6 MoE NodeTP prefix-cache and stochastic MTP parity.
 *
 * CPU tensor parallelism in Llaminar is represented as NodeTP: each CPU
 * socket is a separate MPI rank with its own weight manager, and collectives
 * coordinate the tensor-parallel MoE verifier path.  These tests are the MoE
 * counterpart to the dense Qwen3.6 NodeTP prefix-MTP suite and close the
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
     * @brief Build the baseline CPUx2 NodeTP MoE MTP fixture.
     *
     * @return Qwen3.6 MoE fixture using two CPU MPI ranks as a NodeTP domain.
     */
    MoEPrefixRestoreParityCase cpuNodeTPCase()
    {
        auto test_case = qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE CPUx2 NodeTP parity",
            MoEPrefixParityTopology::NodeTP);
        test_case.decode_steps = 2;
        return test_case;
    }

    /**
     * @brief Build the CPUx2 NodeTP stochastic benchmark fixture.
     *
     * @return NodeTP fixture with a longer prompt and four decode steps so
     * fixed depth-3 and dynamic-depth stochastic MTP both execute real verifier
     * windows.
     */
    MoEPrefixRestoreParityCase cpuNodeTPBenchmarkPromptCase()
    {
        auto test_case = cpuNodeTPCase();
        test_case.name =
            "Qwen3.6 MoE CPUx2 NodeTP benchmark-prompt MTP diagnostic";
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

TEST(Qwen36MoENodeTPPrefixMTPParity, PrefixRestoreFullHit)
{
    runMoEPrefixRestoreParity(
        cpuNodeTPCase(),
        PrefixRestoreParityMode::FullHit);
}

TEST(Qwen36MoENodeTPPrefixMTPParity, MTPGreedyMatchesBaselineTokens)
{
    runMoEMTPParity(cpuNodeTPCase(), false);
}

TEST(Qwen36MoENodeTPPrefixMTPParity, MTPGreedyDepth3MatchesBaselineTokens)
{
    runMoEMTPParity(cpuNodeTPBenchmarkPromptCase(), false, 3);
}

TEST(Qwen36MoENodeTPPrefixMTPParity, PrefixCacheMTPRestore)
{
    runMoEMTPParity(cpuNodeTPCase(), true);
}

TEST(Qwen36MoENodeTPPrefixMTPParity, StochasticMTPDepth1VerifierMatchesAfterClearCache)
{
    runMoEStochasticMTPVerifierParity(cpuNodeTPBenchmarkPromptCase(), 1);
}

TEST(Qwen36MoENodeTPPrefixMTPParity, StochasticMTPDepth2VerifierMatchesAfterClearCache)
{
    runMoEStochasticMTPVerifierParity(cpuNodeTPBenchmarkPromptCase(), 2);
}

TEST(Qwen36MoENodeTPPrefixMTPParity, StochasticMTPDepth3VerifierMatchesAfterClearCache)
{
    runMoEStochasticMTPVerifierParity(
        cpuNodeTPBenchmarkPromptCase(),
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
TEST(Qwen36MoENodeTPPrefixMTPParity, StochasticMTPDepth3VerifierMatchesAfterPrefixRestore)
{
    runMoEStochasticMTPVerifierParity(
        cpuNodeTPBenchmarkPromptCase(),
        3,
        true,
        MTPDepthPolicyConfig{},
        true);
}

TEST(Qwen36MoENodeTPPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterClearCache)
{
    runMoEStochasticMTPVerifierParity(
        cpuNodeTPBenchmarkPromptCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3));
}

TEST(Qwen36MoENodeTPPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore)
{
    runMoEStochasticMTPVerifierParity(
        cpuNodeTPBenchmarkPromptCase(),
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
