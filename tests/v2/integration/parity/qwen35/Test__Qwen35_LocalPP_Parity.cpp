/**
 * @file Test__Qwen35_LocalPP_Parity.cpp
 * @brief Generated rank-local pipeline-parity matrix for dense Qwen3.5.
 *
 * Each definition names the real participant addresses and pipeline layer
 * policy consumed by the production runner. The canonical campaign expands
 * precision axes and proves fresh, full-prefix, and partial-prefix execution
 * without a topology-specific test lifecycle.
 */

#include "Qwen35ModelParityDefinitions.h"
#include "Qwen35ParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include <array>
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
    /** @return Established Qwen3.5-0.8B pipeline numerical contract. */
    BackendThresholds qwen35_08BPPThresholds()
    {
        return BackendThresholds{
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.06f,
        };
    }

    /** @return Established Qwen3.5-4B pipeline numerical contract. */
    BackendThresholds qwen35_4BPPThresholds()
    {
        return BackendThresholds{
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.05f,
        };
    }

    /** @return Established Qwen3.5-27B Q4_K_M pipeline contract. */
    BackendThresholds qwen35_27BPPThresholds()
    {
        return BackendThresholds{
            .cosine_threshold = 0.998f,
            .decode_cosine_threshold = 0.996f,
            .early_layers_count = 8,
            .min_early_layers_passed = 7,
            .kl_threshold = 0.01f,
        };
    }

    /** @return One typed two-stage rank-local pipeline definition. */
    ModelParityDefinition makeQwen35LocalPPDefinition(
        ModelParityModelDefinition model,
        std::string topology_id,
        GlobalDeviceAddress first,
        GlobalDeviceAddress second,
        BackendThresholds thresholds,
        std::vector<float> pipeline_weights = {})
    {
        return qwen35ParityDefinition(
            std::move(model),
            ModelParityTopologyDefinition{
                .test_id = std::move(topology_id),
                .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
                .participants = {
                    {std::move(first), 0},
                    {std::move(second), 0},
                },
                .collective = Collective::None,
                .mpi_ranks = 1,
                .pipeline_stage_sizes = {1, 1},
                .pipeline_weights = std::move(pipeline_weights),
            },
            std::move(thresholds));
    }

    /** @return Canonically expanded dense Qwen3.5 local-pipeline cases. */
    const std::vector<ModelParityCase> &qwen35LocalPPCases()
    {
        static const auto cases = []
        {
            const std::array definitions = {
                makeQwen35LocalPPDefinition(
                    qwen35_08B_Q40ParityModel(), "LocalPP_HOST_2xCPU",
                    GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1),
                    qwen35_08BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_08B_Q40ParityModel(), "LocalPP_NCCL_2xCUDA",
                    GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1),
                    qwen35_08BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_08B_Q40ParityModel(), "LocalPP_RCCL_2xROCm",
                    GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1),
                    qwen35_08BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_08B_Q40ParityModel(),
                    "LocalPP_Heterogeneous_CUDA_ROCm",
                    GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0),
                    qwen35_08BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_4B_Q80ParityModel(), "LocalPP_HOST_2xCPU",
                    GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1),
                    qwen35_4BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_4B_Q80ParityModel(), "LocalPP_NCCL_2xCUDA",
                    GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1),
                    qwen35_4BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_4B_Q80ParityModel(), "LocalPP_RCCL_2xROCm",
                    GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1),
                    qwen35_4BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_4B_Q80ParityModel(),
                    "LocalPP_Heterogeneous_CUDA_ROCm",
                    GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0),
                    qwen35_4BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_27B_Q4KMParityModel(), "LocalPP_NCCL_2xCUDA",
                    GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1),
                    qwen35_27BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_27B_Q4KMParityModel(), "LocalPP_RCCL_2xROCm",
                    GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1),
                    qwen35_27BPPThresholds()),
                makeQwen35LocalPPDefinition(
                    qwen35_27B_Q4KMParityModel(),
                    "LocalPP_Heterogeneous_CUDA_ROCm",
                    GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0),
                    qwen35_27BPPThresholds(), {0.31f, 0.69f}),
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

class Qwen35LocalPPParityTest
    : public Qwen35ConfigDrivenParityTest<Qwen35LocalPPParityTest>,
      public ModelParityCaseParameter
{};

TEST_P(Qwen35LocalPPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35,
    Qwen35LocalPPParityTest,
    ::testing::ValuesIn(qwen35LocalPPCases()),
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
