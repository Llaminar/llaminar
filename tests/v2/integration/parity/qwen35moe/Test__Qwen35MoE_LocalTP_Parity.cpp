/**
 * @file Test__Qwen35MoE_LocalTP_Parity.cpp
 * @brief Typed two-ROCm ExpertOverlay parity campaign for Qwen3.5 MoE.
 *
 * This binary declares only the authenticated model and physical topology.
 * The shared model-parity generator expands the required ordinal/random and
 * static/dynamic residency policies. Every emitted cell executes the same
 * live production runner and proves fresh, complete, and partial prefix-cache
 * behavior in addition to its numerical checkpoints and PerfStats contract.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35MoEModelParityDefinitions.h"
#include "Qwen35MoEParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

/** @return Standard matrix expanded over the two-ROCm overlay topology. */
static const std::vector<ModelParityCase> &qwen35MoELocalTPCases()
{
    static const auto cases = expandModelParityDefinition(
        qwen35MoEParityDefinition(
            qwen35MoE35BQ4KXLParityModel(),
            qwen35MoERocm2LocalTPTopology(),
            qwen35MoEMultiDeviceThresholds()));
    return cases;
}

class Qwen35MoELocalTPParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoELocalTPParityTest>,
      public ModelParityCaseParameter
{};

TEST_P(Qwen35MoELocalTPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35MoELocalTP,
    Qwen35MoELocalTPParityTest,
    ::testing::ValuesIn(qwen35MoELocalTPCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    {
        return info.param.testName();
    });

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
