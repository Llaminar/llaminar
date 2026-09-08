/**
 * @file Test__Qwen2_HybridPPTP_Parity.cpp
 * @brief Hybrid Pipeline Parallelism + Tensor Parallelism Qwen2 parity tests
 *
 * Tests that hybrid PP+TP inference (PP between stages, where at least one
 * stage uses TP internally) produces results matching PyTorch reference outputs.
 *
 * These are the most complex parallelism configurations, combining layer
 * splitting (PP) with weight sharding (TP) within stages.
 *
 * Configurations:
 *   - LocalPP_TP2xCUDA_ROCm:    Stage 0 = TP(2xCUDA/NCCL), Stage 1 = ROCm
 *   - LocalPP_TP2xROCm_CUDA:    Stage 0 = TP(2xROCm/RCCL), Stage 1 = CUDA
 *   - LocalPP_TP2xROCm_CPU:     Stage 0 = TP(2xROCm/RCCL), Stage 1 = CPU
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

/** @return One typed two-stage PP+TP definition. */
static ModelParityDefinition makeHybridPPTPDefinition(
    std::string topology_id,
    std::vector<ModelParityParticipant> participants,
    Collective tensor_parallel_collective)
{
    return qwen2Q40ParityDefinition(
        ModelParityTopologyDefinition{
            .test_id = std::move(topology_id),
            .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
            .participants = std::move(participants),
            .mpi_ranks = 1,
            .pipeline_stage_sizes = {2, 1},
            .tensor_parallel_collective = tensor_parallel_collective,
        },
        BackendThresholds{
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.015f,
            .excluded_stages = kTPExcludedStages,
        });
}

/** @return Canonically expanded Qwen2 PP+TP cases. */
static const std::vector<ModelParityCase> &qwen2HybridPPTPCases()
{
    static const auto cases = []
    {
        const std::array definitions = {
            makeHybridPPTPDefinition(
                "LocalPP_TP2xCUDA_ROCm",
                {{GlobalDeviceAddress::cuda(0), 0},
                 {GlobalDeviceAddress::cuda(1), 0},
                 {GlobalDeviceAddress::rocm(0), 0}},
                Collective::NCCL),
            makeHybridPPTPDefinition(
                "LocalPP_TP2xROCm_CUDA",
                {{GlobalDeviceAddress::rocm(0), 0},
                 {GlobalDeviceAddress::rocm(1), 0},
                 {GlobalDeviceAddress::cuda(0), 0}},
                Collective::RCCL),
            makeHybridPPTPDefinition(
                "LocalPP_TP2xROCm_CPU",
                {{GlobalDeviceAddress::rocm(0), 0},
                 {GlobalDeviceAddress::rocm(1), 0},
                 {GlobalDeviceAddress::cpu(), 0}},
                Collective::RCCL),
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

class Qwen2HybridPPTPParityTest : public ConfigDrivenParityTest<Qwen2HybridPPTPParityTest>,
                                  public ModelParityCaseParameter
{};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen2HybridPPTPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen2,
    Qwen2HybridPPTPParityTest,
    ::testing::ValuesIn(qwen2HybridPPTPCases()),
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
