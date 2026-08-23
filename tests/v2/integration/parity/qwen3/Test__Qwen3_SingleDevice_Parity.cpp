/**
 * @file Test__Qwen3_SingleDevice_Parity.cpp
 * @brief Generated single-device Qwen3 production-parity matrix.
 *
 * The typed declarations below preserve the established CPU, CUDA, and ROCm
 * numerical contracts while central expansion owns activation/KV products and
 * the mandatory fresh/full/partial prefix-restore lifecycle.
 */

#include "Qwen3ModelParityDefinitions.h"
#include "Qwen3ParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen3;

namespace
{
    /** @return Exact one-participant topology for a production backend. */
    ModelParityTopologyDefinition singleDeviceTopology(
        std::string test_id,
        GlobalDeviceAddress address)
    {
        return ModelParityTopologyDefinition{
            .test_id = std::move(test_id),
            .kind = ModelParityTopologyKind::SingleDevice,
            .participants = {{std::move(address), 0}},
            .collective = Collective::None,
            .mpi_ranks = 1,
        };
    }

    /** @return Typed Qwen3 definitions replacing the legacy record table. */
    std::vector<ModelParityDefinition> qwen3SingleDeviceDefinitions()
    {
        const auto cpu = singleDeviceTopology(
            "CPU0", GlobalDeviceAddress::cpu());
        const auto cuda = singleDeviceTopology(
            "CUDA0", GlobalDeviceAddress::cuda(0));
        const auto rocm = singleDeviceTopology(
            "ROCm0", GlobalDeviceAddress::rocm(0));

        const auto common = qwen3ParityThresholds(0.01f);
        const auto compressed =
            qwen3ParityThresholds(kQwen3CompressedKVKLThreshold);
        const auto rocm_default = qwen3ParityThresholds(0.04f, 60.0f, 60.0f);
        const auto rocm_compressed = qwen3ParityThresholds(
            kQwen3CompressedKVKLThreshold, 60.0f, 60.0f);

        return {
            qwen3Q80ParityDefinition(
                cpu,
                common,
                {
                    KVCachePrecision::FP16,
                    KVCachePrecision::Q8_1,
                    KVCachePrecision::Q16_1,
                    KVCachePrecision::TQ,
                },
                {
                    {ActivationPrecision::FP32,
                     KVCachePrecision::Q8_1,
                     compressed},
                    {ActivationPrecision::FP32,
                     KVCachePrecision::TQ,
                     compressed},
                }),
            qwen3Q80ParityDefinition(
                cuda,
                common,
                {
                    KVCachePrecision::FP16,
                    KVCachePrecision::Q8_1,
                    KVCachePrecision::TQ,
                },
                {
                    {ActivationPrecision::FP32,
                     KVCachePrecision::Q8_1,
                     compressed},
                    {ActivationPrecision::FP32,
                     KVCachePrecision::TQ,
                     compressed},
                }),
            qwen3Q80ParityDefinition(
                rocm,
                rocm_default,
                {
                    KVCachePrecision::FP16,
                    KVCachePrecision::FP32,
                    KVCachePrecision::Q8_1,
                    KVCachePrecision::TQ,
                },
                {
                    {ActivationPrecision::FP32,
                     KVCachePrecision::Q8_1,
                     rocm_compressed},
                    {ActivationPrecision::FP32,
                     KVCachePrecision::TQ,
                     rocm_compressed},
                }),
        };
    }

    /** @return Stable generated case storage used by GoogleTest discovery. */
    const std::vector<ModelParityCase> &qwen3SingleDeviceCases()
    {
        static const auto cases = []
        {
            std::vector<ModelParityCase> expanded;
            for (const auto &definition : qwen3SingleDeviceDefinitions())
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

class Qwen3SingleDeviceParityTest
    : public Qwen3ConfigDrivenParityTest<Qwen3SingleDeviceParityTest>,
      public ModelParityCaseParameter
{
};

TEST_P(Qwen3SingleDeviceParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen3,
    Qwen3SingleDeviceParityTest,
    ::testing::ValuesIn(qwen3SingleDeviceCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    { return info.param.testName(); });

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();
    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
