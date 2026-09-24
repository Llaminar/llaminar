/**
 * @file Test__Qwen35_SingleDevice_Parity.cpp
 * @brief Generated single-device dense Qwen3.5 production-parity matrix.
 *
 * The typed definitions preserve every established CPU, CUDA, and ROCm
 * numerical contract for the 0.8B, 4B, extended-decode, and 27B reference
 * packs. Canonical expansion owns precision products and every emitted case
 * proves fresh inference plus full and partial production prefix restore.
 */

#include "Qwen35ModelParityDefinitions.h"
#include "Qwen35ParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;

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

    /** @return Numerical contract for Q4_0 0.8B CPU/CUDA execution. */
    BackendThresholds qwen35_08BThresholds()
    {
        return BackendThresholds{
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 6,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.02f,
        };
    }

    /** @return ROCm numerical contract for the Q4_0 0.8B model. */
    BackendThresholds qwen35_08BRocmThresholds()
    {
        auto thresholds = qwen35_08BThresholds();
        thresholds.kl_threshold = 0.04f;
        thresholds.min_top1_accuracy = 60.0f;
        thresholds.min_top5_accuracy = 60.0f;
        return thresholds;
    }

    /** @return CPU numerical contract for the Q8_0 4B model. */
    BackendThresholds qwen35_4BCpuThresholds()
    {
        return BackendThresholds{
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.93f,
            .early_layers_count = 8,
            .min_early_layers_passed = 8,
            .kl_threshold = 0.03f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f,
            .pytorch_top1_in_topk = 3,
        };
    }

    /** @return CUDA numerical contract for the Q8_0 4B model. */
    BackendThresholds qwen35_4BCudaThresholds()
    {
        auto thresholds = qwen35_4BCpuThresholds();
        thresholds.kl_threshold = 0.06f;
        return thresholds;
    }

    /** @return ROCm numerical contract for the Q8_0 4B model. */
    BackendThresholds qwen35_4BRocmThresholds()
    {
        auto thresholds = qwen35_4BCpuThresholds();
        thresholds.kl_threshold = 0.07f;
        return thresholds;
    }

    /** @return Twenty-step drift-characterization contract for 4B CPU. */
    BackendThresholds qwen35_4BDecode20Thresholds()
    {
        return BackendThresholds{
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 8,
            .min_early_layers_passed = 8,
            .kl_threshold = 0.50f,
            .min_top1_accuracy = 50.0f,
            .min_top5_accuracy = 60.0f,
            .min_decode_pass_rate = 0.60f,
            .pytorch_top1_in_topk = 0,
        };
    }

    /** @return Diagnostic contract for the dense Q8_0 27B model. */
    BackendThresholds qwen35_27BThresholds()
    {
        return BackendThresholds{
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 8,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.20f,
            .min_top1_accuracy = 40.0f,
            .min_top5_accuracy = 60.0f,
            .pytorch_top1_in_topk = 0,
        };
    }

    /** @return Typed definitions replacing the legacy per-cell record table. */
    std::vector<ModelParityDefinition> qwen35SingleDeviceDefinitions()
    {
        const auto cpu = singleDeviceTopology(
            "CPU0", GlobalDeviceAddress::cpu());
        const auto cuda = singleDeviceTopology(
            "CUDA0", GlobalDeviceAddress::cuda(0));
        const auto rocm = singleDeviceTopology(
            "ROCm0", GlobalDeviceAddress::rocm(0));

        return {
            qwen35ParityDefinition(
                qwen35_08B_Q40ParityModel(), cpu,
                qwen35_08BThresholds(),
                {KVCachePrecision::FP16,
                 KVCachePrecision::Q8_1,
                 KVCachePrecision::Q16_1}),
            qwen35ParityDefinition(
                qwen35_08B_Q40ParityModel(), cuda,
                qwen35_08BThresholds(),
                {KVCachePrecision::FP16,
                 KVCachePrecision::Q8_1}),
            qwen35ParityDefinition(
                qwen35_08B_Q40ParityModel(), rocm,
                qwen35_08BRocmThresholds(),
                {KVCachePrecision::FP16,
                 KVCachePrecision::Q8_1}),
            qwen35ParityDefinition(
                qwen35_4B_Q80ParityModel(), cpu,
                qwen35_4BCpuThresholds()),
            qwen35ParityDefinition(
                qwen35_4B_Q80ParityModel(), cuda,
                qwen35_4BCudaThresholds()),
            qwen35ParityDefinition(
                qwen35_4B_Q80ParityModel(), rocm,
                qwen35_4BRocmThresholds()),
            qwen35ParityDefinition(
                qwen35_4B_Q80Decode20ParityModel(), cpu,
                qwen35_4BDecode20Thresholds()),
            qwen35ParityDefinition(
                qwen35_27B_Q80ParityModel(), cpu,
                qwen35_27BThresholds(),
                {KVCachePrecision::FP16,
                 KVCachePrecision::Q8_1}),
        };
    }

    /** @return Stable generated case storage used by GoogleTest discovery. */
    const std::vector<ModelParityCase> &qwen35SingleDeviceCases()
    {
        static const auto cases = []
        {
            std::vector<ModelParityCase> expanded;
            for (const auto &definition : qwen35SingleDeviceDefinitions())
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

class Qwen35SingleDeviceParityTest
    : public Qwen35ConfigDrivenParityTest<Qwen35SingleDeviceParityTest>,
      public ModelParityCaseParameter
{};

TEST_P(Qwen35SingleDeviceParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35,
    Qwen35SingleDeviceParityTest,
    ::testing::ValuesIn(qwen35SingleDeviceCases()),
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
