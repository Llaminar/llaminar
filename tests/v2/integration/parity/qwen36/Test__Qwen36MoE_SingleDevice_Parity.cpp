/**
 * @file Test__Qwen36MoE_SingleDevice_Parity.cpp
 * @brief Single-device Qwen3.6 MoE and Ornith fine-tune production parity.
 *
 * The shared typed matrix expands MTP off, fixed depths 1/2/3/15, and dynamic
 * depth while every case proves fresh, full, and partial prefix restore. Qwen3.6
 * MoE divergences retain the standard snapshot CSV diagnostics from one live
 * production runner lifetime.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../qwen35moe/Qwen35MoEParityTestBase.h"
#include "Qwen36ModelParityDefinitions.h"
#include "Ornith15ModelParityDefinitions.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <array>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    /** @return Canonically expanded CPU, CUDA, and ROCm cases. */
    const std::vector<ModelParityCase> &qwen36MoESingleDeviceCases()
    {
        static const auto cases = []
        {
            const std::array definitions = {
                qwen36MoEParityDefinition(
                    qwen36SingleDeviceTopology(
                        "CPU0",
                        GlobalDeviceAddress::cpu()),
                    "pytorch_qwen36_moe_singledevice_cpu_snapshots",
                    qwen36MoESingleDeviceThresholds()),
                qwen36MoEParityDefinition(
                    qwen36SingleDeviceTopology(
                        "CUDA0",
                        GlobalDeviceAddress::cuda(0)),
                    "pytorch_qwen36_moe_singledevice_cuda_snapshots",
                    qwen36MoESingleDeviceThresholds()),
                qwen36MoEParityDefinition(
                    qwen36SingleDeviceTopology(
                        "ROCm0",
                        GlobalDeviceAddress::rocm(0)),
                    "pytorch_qwen36_moe_singledevice_rocm_snapshots",
                    qwen36MoESingleDeviceThresholds()),
            };
            std::vector<ModelParityCase> expanded;
            for (const auto &definition : withOrnith15CertificationModels(definitions))
            {
                auto definition_cases =
                    expandModelParityDefinition(definition);
                expanded.insert(
                    expanded.end(),
                    std::make_move_iterator(definition_cases.begin()),
                    std::make_move_iterator(definition_cases.end()));
            }
            return expanded;
        }();
        return cases;
    }
}

class Qwen36MoESingleDeviceParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen36MoESingleDeviceParityTest>,
      public ModelParityCaseParameter
{};

TEST_P(Qwen36MoESingleDeviceParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen36MoE,
    Qwen36MoESingleDeviceParityTest,
    ::testing::ValuesIn(qwen36MoESingleDeviceCases()),
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
