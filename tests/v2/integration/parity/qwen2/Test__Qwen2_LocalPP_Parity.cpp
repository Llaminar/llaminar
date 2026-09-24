/**
 * @file Test__Qwen2_LocalPP_Parity.cpp
 * @brief Local Pipeline Parallelism Qwen2 parity tests
 *
 * Tests that Local PP inference (layers split across devices within a single
 * node with activation transfer between stages) produces results matching
 * PyTorch reference outputs.
 *
 * Configurations:
 *   - LocalPP_RCCL_2xROCm:          2x AMD GPU via RCCL
 *   - LocalPP_NCCL_2xCUDA:          2x NVIDIA GPU via NCCL
 *   - LocalPP_HOST_CUDA_ROCm:        Heterogeneous CUDA+ROCm via HOST backend
 *   - LocalPP_HOST_CUDA_CPU:         CUDA + CPU via HOST backend
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen2ModelParityDefinitions.h"
#include "Qwen2ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

#include <array>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

/** @return Numerical contract for one pure pipeline topology. */
static BackendThresholds localPPThresholds(float kl_threshold)
{
    return BackendThresholds{
        .cosine_threshold = 0.97f,
        .decode_cosine_threshold = 0.97f,
        .early_layers_count = 6,
        .min_early_layers_passed = 4,
        .kl_threshold = kl_threshold,
    };
}

/** @return One typed two-stage rank-local pipeline definition. */
static ModelParityDefinition makeLocalPPDefinition(
    std::string topology_id,
    GlobalDeviceAddress first,
    GlobalDeviceAddress second,
    float kl_threshold)
{
    return qwen2Q40ParityDefinition(
        ModelParityTopologyDefinition{
            .test_id = std::move(topology_id),
            .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
            .participants = {{std::move(first), 0}, {std::move(second), 0}},
            .mpi_ranks = 1,
            .pipeline_stage_sizes = {1, 1},
        },
        localPPThresholds(kl_threshold));
}

/** @return Canonically expanded Qwen2 pure-pipeline cases. */
static const std::vector<ModelParityCase> &qwen2LocalPPCases()
{
    static const auto cases = []
    {
        const std::array definitions = {
            makeLocalPPDefinition(
                "LocalPP_RCCL_2xROCm",
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1), 0.015f),
            makeLocalPPDefinition(
                "LocalPP_NCCL_2xCUDA",
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1), 0.015f),
            makeLocalPPDefinition(
                "LocalPP_Heterogeneous_CUDA_ROCm",
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::rocm(0), 0.025f),
            makeLocalPPDefinition(
                "LocalPP_Heterogeneous_CUDA_CPU",
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cpu(), 0.015f),
        };
        std::vector<ModelParityCase> expanded;
        for (const auto &definition : definitions)
        {
            auto definition_cases = expandModelParityDefinition(definition);
            expanded.insert(
                expanded.end(),
                std::make_move_iterator(definition_cases.begin()),
                std::make_move_iterator(definition_cases.end()));
        }
        return expanded;
    }();
    return cases;
}

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen2LocalPPParityTest : public ConfigDrivenParityTest<Qwen2LocalPPParityTest>,
                               public ModelParityCaseParameter
{};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen2LocalPPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen2,
    Qwen2LocalPPParityTest,
    ::testing::ValuesIn(qwen2LocalPPCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    {
        return info.param.testName();
    });

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    // CRITICAL: Shutdown GlobalBackendRouter before MPI_Finalize to ensure
    // NCCLCoordinator cleanup happens while CUDA runtime is still active.
    // Without this, the static GlobalBackendRouter::instance_ is destroyed
    // during atexit handlers after CUDA has shut down, causing crashes.
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    // Skip static destructors — see Test__Qwen2_SingleDevice_Parity.cpp for rationale.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
