/**
 * @file Test__Qwen35_LocalTP_Parity.cpp
 * @brief Local Tensor Parallelism Qwen3.5 parity tests
 *
 * Tests that Local TP inference (weight-sharded across multiple devices
 * within a single node) produces results matching PyTorch reference outputs
 * for Qwen3.5 models with GDN (Gated Delta Network) + FA (Full Attention)
 * hybrid architecture.
 *
 * NOTE: CPU TP tests use GlobalTP (MPI) domain, not LocalTP, because
 * DeviceId::cpu() is a singleton — LocalTP cannot distinguish multiple
 * CPU sockets. See Test__Qwen35_GlobalTP_Parity.cpp for multi-socket CPU TP.
 *
 * Configurations (Qwen3.5-0.8B Q4_0):
 *   - LocalTP_NCCL_2xCUDA_08B:       2x NVIDIA GPU via NCCL (skipped: no GPU kernels)
 *   - LocalTP_RCCL_2xROCm_08B:       2x AMD GPU via RCCL (skipped: no GPU kernels)
 *   - LocalTP_HOST_CUDA_ROCm_08B:    Heterogeneous CUDA+ROCm (skipped: no GPU kernels)
 *
 * Configurations (Qwen3.5-4B Q8_0):
 *   - LocalTP_NCCL_2xCUDA_4B:        2x NVIDIA GPU via NCCL (skipped: no GPU kernels)
 *   - LocalTP_RCCL_2xROCm_4B:        2x AMD GPU via RCCL (skipped: no GPU kernels)
 *   - LocalTP_HOST_CUDA_ROCm_4B:     Heterogeneous CUDA+ROCm (skipped: no GPU kernels)
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35ModelParityDefinitions.h"
#include "Qwen35ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

#include <array>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;

// =============================================================================
// Common Excluded Stages for TP
// =============================================================================

// Sharded outputs can't be compared directly against single-device PyTorch snapshots.
// Includes Qwen2-standard exclusions plus GDN-specific sharded stages.
static const std::vector<std::string> kTPExcludedStages = {
    // Standard attention projections (FA layers)
    "Q_PROJECTION",
    "K_PROJECTION",
    "V_PROJECTION",
    "Q_ROPE",
    "K_ROPE",
    "ATTENTION_CONTEXT",
    // FFN sharded intermediates
    "FFN_GATE",
    "FFN_UP",
    "FFN_SWIGLU",
    // GDN-specific: QKV projection covers gdn_proj (same snapshot key)
    "QKV_PROJECTION",
    // GDN-specific: recurrence output is per-local-heads under TP
    "GDN_DELTA_RULE_OUTPUT",
    // GDN-specific: gated norm output is per-local-heads under TP
    "GDN_NORM_GATE_OUTPUT",
};

// =============================================================================
// Test Configuration Definitions — Qwen3.5-0.8B (Q4_0)
// =============================================================================

/** @return Numerical contract for one dense Qwen3.5 local-TP topology. */
static BackendThresholds qwen35LocalTPThresholds()
{
    return BackendThresholds{
        .cosine_threshold = 0.90f,
        .decode_cosine_threshold = 0.90f,
        .early_layers_count = 6,
        .min_early_layers_passed = 4,
        .kl_threshold = 0.06f,
        .excluded_stages = kTPExcludedStages,
    };
}

/** @return One typed rank-local TP definition. */
static ModelParityDefinition makeQwen35LocalTPDefinition(
    ModelParityModelDefinition model,
    std::string topology_id,
    std::vector<ModelParityParticipant> participants,
    Collective collective)
{
    return qwen35ParityDefinition(
        std::move(model),
        ModelParityTopologyDefinition{
            .test_id = std::move(topology_id),
            .kind = ModelParityTopologyKind::RankLocalTensorParallel,
            .participants = std::move(participants),
            .collective = collective,
            .mpi_ranks = 1,
        },
        qwen35LocalTPThresholds());
}

/** @return Canonically expanded dense Qwen3.5 local-TP cases. */
static const std::vector<ModelParityCase> &qwen35LocalTPCases()
{
    static const auto cases = []
    {
        const std::array definitions = {
            makeQwen35LocalTPDefinition(
                qwen35_08B_Q40ParityModel(), "LocalTP_NCCL_2xCUDA",
                {{GlobalDeviceAddress::cuda(0), 0},
                 {GlobalDeviceAddress::cuda(1), 0}},
                Collective::NCCL),
            makeQwen35LocalTPDefinition(
                qwen35_08B_Q40ParityModel(), "LocalTP_RCCL_2xROCm",
                {{GlobalDeviceAddress::rocm(0), 0},
                 {GlobalDeviceAddress::rocm(1), 0}},
                Collective::RCCL),
            makeQwen35LocalTPDefinition(
                qwen35_08B_Q40ParityModel(), "LocalTP_Heterogeneous_CUDA_ROCm",
                {{GlobalDeviceAddress::cuda(0), 0},
                 {GlobalDeviceAddress::rocm(0), 0}},
                Collective::HETEROGENEOUS),
            makeQwen35LocalTPDefinition(
                qwen35_4B_Q80ParityModel(), "LocalTP_NCCL_2xCUDA",
                {{GlobalDeviceAddress::cuda(0), 0},
                 {GlobalDeviceAddress::cuda(1), 0}},
                Collective::NCCL),
            makeQwen35LocalTPDefinition(
                qwen35_4B_Q80ParityModel(), "LocalTP_RCCL_2xROCm",
                {{GlobalDeviceAddress::rocm(0), 0},
                 {GlobalDeviceAddress::rocm(1), 0}},
                Collective::RCCL),
            makeQwen35LocalTPDefinition(
                qwen35_4B_Q80ParityModel(), "LocalTP_Heterogeneous_CUDA_ROCm",
                {{GlobalDeviceAddress::cuda(0), 0},
                 {GlobalDeviceAddress::rocm(0), 0}},
                Collective::HETEROGENEOUS),
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

class Qwen35LocalTPParityTest : public Qwen35ConfigDrivenParityTest<Qwen35LocalTPParityTest>,
                                public ModelParityCaseParameter
{};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen35LocalTPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen35,
    Qwen35LocalTPParityTest,
    ::testing::ValuesIn(qwen35LocalTPCases()),
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
