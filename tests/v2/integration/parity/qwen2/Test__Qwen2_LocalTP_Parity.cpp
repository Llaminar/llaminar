/**
 * @file Test__Qwen2_LocalTP_Parity.cpp
 * @brief Local Tensor Parallelism Qwen2 parity tests
 *
 * Tests that Local TP inference (weight-sharded across multiple devices
 * within a single node) produces results matching PyTorch reference outputs.
 *
 * Configurations:
 *   - LocalTP_NCCL_2xCUDA:          2x NVIDIA GPU via NCCL
 *   - LocalTP_RCCL_2xROCm:          2x AMD GPU via RCCL
 *   - LocalTP_RCCL_4xROCm:          4x AMD GPU via RCCL
 *   - LocalTP_HOST_CUDA_ROCm:        Heterogeneous CUDA+ROCm via HOST backend
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
// Common Excluded Stages for TP
// =============================================================================

// Sharded outputs can't be compared directly against single-device PyTorch snapshots
static const std::vector<std::string> kTPExcludedStages = {
    "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "Q_ROPE", "K_ROPE",
    "ATTENTION_CONTEXT",
    "FFN_GATE", "FFN_UP", "FFN_SWIGLU"};

// =============================================================================
// Test Configuration Definitions
// =============================================================================

/** @return One typed rank-local TP definition. */
static ModelParityDefinition makeLocalTPDefinition(
    std::string topology_id,
    std::vector<ModelParityParticipant> participants,
    Collective collective,
    float cosine_threshold,
    float kl_threshold)
{
    return qwen2Q40ParityDefinition(
        ModelParityTopologyDefinition{
            .test_id = std::move(topology_id),
            .kind = ModelParityTopologyKind::RankLocalTensorParallel,
            .participants = std::move(participants),
            .collective = collective,
            .mpi_ranks = 1,
        },
        BackendThresholds{
            .cosine_threshold = cosine_threshold,
            .decode_cosine_threshold = cosine_threshold,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = kl_threshold,
            .excluded_stages = kTPExcludedStages,
        });
}

/** @return Canonically expanded local-TP topology cases. */
static const std::vector<ModelParityCase> &qwen2LocalTPCases()
{
    static const auto cases = []
    {
        const std::array definitions = {
            makeLocalTPDefinition(
                "LocalTP_NCCL_2xCUDA",
                {{GlobalDeviceAddress::cuda(0), 0},
                 {GlobalDeviceAddress::cuda(1), 0}},
                Collective::NCCL,
                0.90f,
                0.35f),
            makeLocalTPDefinition(
                "LocalTP_RCCL_2xROCm",
                {{GlobalDeviceAddress::rocm(0), 0},
                 {GlobalDeviceAddress::rocm(1), 0}},
                Collective::RCCL,
                0.90f,
                0.40f),
            makeLocalTPDefinition(
                "LocalTP_RCCL_4xROCm",
                {{GlobalDeviceAddress::rocm(0), 0},
                 {GlobalDeviceAddress::rocm(1), 0},
                 {GlobalDeviceAddress::rocm(2), 0},
                 {GlobalDeviceAddress::rocm(3), 0}},
                Collective::RCCL,
                0.96f,
                0.02f),
            makeLocalTPDefinition(
                "LocalTP_Heterogeneous_CUDA_ROCm",
                {{GlobalDeviceAddress::cuda(0), 0},
                 {GlobalDeviceAddress::rocm(0), 0}},
                Collective::HETEROGENEOUS,
                0.90f,
                0.50f),
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

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen2LocalTPParityTest : public ConfigDrivenParityTest<Qwen2LocalTPParityTest>,
                               public ModelParityCaseParameter
{};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen2LocalTPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen2,
    Qwen2LocalTPParityTest,
    ::testing::ValuesIn(qwen2LocalTPCases()),
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
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    // Skip static destructors — see Test__Qwen2_SingleDevice_Parity.cpp for rationale.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
