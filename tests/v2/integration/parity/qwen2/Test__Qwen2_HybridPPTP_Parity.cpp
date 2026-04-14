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
#include "Qwen2ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

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

static const std::vector<TestConfig> kHybridPPTPConfigs = {
    // Stage 0 = TP(2xCUDA), Stage 1 = ROCm
    {
        .name = "LocalPP_TP2xCUDA_ROCm",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,     // PP between TP domain and ROCm
        .collective = Collective::HETEROGENEOUS, // Cross-vendor transfer between stages
        .thresholds = {
            .cosine_threshold = 0.96f,        // Observed: 0.998 prefill cosine
            .decode_cosine_threshold = 0.95f, // Observed: 0.984 avg decode cosine
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.015f,               // Observed: 0.003 prefill KL (was 0.50 = 167x over-relaxed)
            .excluded_stages = kTPExcludedStages, // TP excluded stages apply to stage 0
        },
        .pp_stage_sizes = {2, 1},          // Stage 0: 2 devices (TP), Stage 1: 1 device
        .tp_collective = Collective::NCCL, // TP collective within stage 0 (CUDA-CUDA)
    },
    // Stage 0 = TP(2xROCm), Stage 1 = single CUDA
    {
        .name = "LocalPP_TP2xROCm_CUDA",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalPP,     // PP between TP domain and CUDA
        .collective = Collective::HETEROGENEOUS, // Cross-vendor transfer between stages
        .thresholds = {
            .cosine_threshold = 0.96f,        // Observed: 0.998 prefill cosine
            .decode_cosine_threshold = 0.95f, // Observed: 0.984 avg decode cosine
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.015f,               // Observed: 0.002 prefill KL (was 0.50 = 239x over-relaxed)
            .excluded_stages = kTPExcludedStages, // TP excluded stages apply to stage 0
        },
        .pp_stage_sizes = {2, 1},          // Stage 0: 2 devices (TP), Stage 1: 1 device
        .tp_collective = Collective::RCCL, // TP collective within stage 0 (ROCm-ROCm)
    },
    // Stage 0 = TP(2xROCm), Stage 1 = CPU
    {
        .name = "LocalPP_TP2xROCm_CPU",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm, ParityDeviceType::CPU},
        .parallelism = Parallelism::LocalPP, // PP between TP domain and CPU
        .collective = Collective::None,      // HOST backend for GPU→CPU transfer
        .thresholds = {
            .cosine_threshold = 0.96f,        // Observed: 0.998 prefill cosine
            .decode_cosine_threshold = 0.95f, // Observed: 0.984 avg decode cosine
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.015f,               // Observed: 0.002 prefill KL (was 0.50 = 208x over-relaxed)
            .excluded_stages = kTPExcludedStages, // TP excluded stages apply to stage 0
        },
        .pp_stage_sizes = {2, 1},          // Stage 0: 2 devices (TP), Stage 1: 1 device
        .tp_collective = Collective::RCCL, // TP collective within stage 0 (ROCm-ROCm)
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen2HybridPPTPParityTest : public ConfigDrivenParityTest<Qwen2HybridPPTPParityTest>,
                                  public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen2HybridPPTPParityTest, PrefillParity)
{
    // Hybrid PP+TP: pipeline uses LocalPPTestRunner wrapping pre-compiled
    // TP MDO + single-device DGO. Use non-TP parity comparison since the
    // outer runner is not a MultiDeviceOrchestrator (TP-specific snapshot
    // access requires MDO cast). The combined output is still compared
    // against PyTorch reference for correctness.
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runPrefillParity();
    assertParity(summary);
}

TEST_P(Qwen2HybridPPTPParityTest, DecodeParity)
{
    // Hybrid PP+TP: use non-TP decode parity comparison for the same reason
    // as PrefillParity — the outer runner is a LocalPPTestRunner, not MDO.
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen2HybridPPTPParityTest, SnapshotInfrastructure)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    auto embedding = loadPyTorchSnapshot("EMBEDDING");
    ASSERT_FALSE(embedding.empty()) << "Failed to load EMBEDDING snapshot";

    ASSERT_TRUE(runner_ != nullptr);
    runner_->forward(config_.token_ids.data(), config_.token_ids.size());

    auto keys = runner_->getSnapshotKeys();
    EXPECT_GT(keys.size(), 0) << "No snapshots captured";

    bool has_embedding = std::find(keys.begin(), keys.end(), "EMBEDDING") != keys.end();
    bool has_lm_head = std::find(keys.begin(), keys.end(), "LM_HEAD") != keys.end();
    EXPECT_TRUE(has_embedding) << "Missing EMBEDDING snapshot";
    EXPECT_TRUE(has_lm_head) << "Missing LM_HEAD snapshot";
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen2,
    Qwen2HybridPPTPParityTest,
    ::testing::ValuesIn(kHybridPPTPConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
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
    return result;
}
