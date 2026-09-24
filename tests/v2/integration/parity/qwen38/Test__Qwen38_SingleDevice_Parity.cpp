/**
 * @file Test__Qwen38_SingleDevice_Parity.cpp
 * @brief Classic single-device Qwen3.8 dense math parity tests.
 *
 * The standard typed matrix proves main-graph and recursive-MTP checkpoints,
 * exact grouped-versus-serial tokens, and the mandatory fresh/full/partial
 * prefix lifecycle against one production runner on CPU, CUDA, and ROCm.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../qwen35/Qwen35ParityTestBase.h"
#include "Qwen38ModelParityDefinitions.h"
#include "../qwen36/Qwen36ModelParityDefinitions.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;
using namespace llaminar2::test::parity::qwen38;

namespace
{
    /** @return Canonically expanded dense CPU, CUDA, and ROCm cases. */
    const std::vector<ModelParityCase> &qwen38DenseSingleDeviceCases()
    {
        static const auto cases = []
        {
            const std::array definitions = {
                qwen38DenseParityDefinition(
                    qwen36::qwen36SingleDeviceTopology(
                        "CPU0", GlobalDeviceAddress::cpu()),
                    "pytorch_qwen38_dense_singledevice_cpu_snapshots"),
                qwen38DenseParityDefinition(
                    qwen36::qwen36SingleDeviceTopology(
                        "CUDA0", GlobalDeviceAddress::cuda(0)),
                    "pytorch_qwen38_dense_singledevice_cuda_snapshots"),
                qwen38DenseParityDefinition(
                    qwen36::qwen36SingleDeviceTopology(
                        "ROCm0", GlobalDeviceAddress::rocm(0)),
                    "pytorch_qwen38_dense_singledevice_rocm_snapshots"),
            };
            std::vector<ModelParityCase> expanded;
            for (const auto &definition : definitions)
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
} // namespace

/**
 * @brief Parameterized wrapper around the Qwen3.5 hybrid-GDN parity base.
 *
 * Qwen3.8 dense GGUFs use the same hybrid GDN/full-attention graph family as
 * Qwen3.5, with trailing MTP sidecar tensors. The shared production campaign
 * expands MTP off, fixed depths 1/2/3/15, and device-owned dynamic depth from
 * the typed Qwen3.8 declaration.
 */
class Qwen38DenseSingleDeviceParityTest
    : public Qwen35ConfigDrivenParityTest<Qwen38DenseSingleDeviceParityTest>,
      public ModelParityCaseParameter
{};

TEST_P(Qwen38DenseSingleDeviceParityTest, ProductionParity)
{
    runProductionParityCampaign();
}
INSTANTIATE_TEST_SUITE_P(
    Qwen38Dense,
    Qwen38DenseSingleDeviceParityTest,
    ::testing::ValuesIn(qwen38DenseSingleDeviceCases()),
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
