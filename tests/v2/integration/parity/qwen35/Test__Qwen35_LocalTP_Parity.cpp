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
#include "Qwen35ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

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

static const auto kQwen35_08B_TP_Thresholds = BackendThresholds{
    .cosine_threshold = 0.90f,
    .decode_cosine_threshold = 0.90f,
    .early_layers_count = 6,
    .min_early_layers_passed = 4,
    .kl_threshold = 0.06f, // Was 0.35 = very over-relaxed; no GPU results yet, conservative estimate
    .excluded_stages = kTPExcludedStages,
};

static const auto kQwen35_4B_TP_Thresholds = BackendThresholds{
    .cosine_threshold = 0.90f,
    .decode_cosine_threshold = 0.90f,
    .early_layers_count = 6,
    .min_early_layers_passed = 4,
    .kl_threshold = 0.06f, // Was 0.35 = very over-relaxed; no GPU results yet, conservative estimate
    .excluded_stages = kTPExcludedStages,
};

static const std::vector<TestConfig> kLocalTPConfigs = {
    // =========================================================================
    // Qwen3.5-0.8B (Q4_0) — GPU configs (skipped: no GPU kernels yet)
    // =========================================================================
    {
        .name = "LocalTP_NCCL_2xCUDA_08B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::NCCL,
        .thresholds = kQwen35_08B_TP_Thresholds,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalTP_RCCL_2xROCm_08B",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::RCCL,
        .thresholds = kQwen35_08B_TP_Thresholds,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalTP_HETEROGENEOUS_CUDA_ROCm_08B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::HETEROGENEOUS,
        .thresholds = kQwen35_08B_TP_Thresholds,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    // =========================================================================
    // Qwen3.5-4B (Q8_0) — GPU configs (skipped: no GPU kernels yet)
    // =========================================================================
    {
        .name = "LocalTP_NCCL_2xCUDA_4B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::NCCL,
        .thresholds = kQwen35_4B_TP_Thresholds,
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalTP_RCCL_2xROCm_4B",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::RCCL,
        .thresholds = kQwen35_4B_TP_Thresholds,
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalTP_HETEROGENEOUS_CUDA_ROCm_4B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::HETEROGENEOUS,
        .thresholds = kQwen35_4B_TP_Thresholds,
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35LocalTPParityTest : public Qwen35ConfigDrivenParityTest<Qwen35LocalTPParityTest>,
                                public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen35LocalTPParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runTPPrefillParity();
    assertTPParity(summary);
}

TEST_P(Qwen35LocalTPParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runTPDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen35LocalTPParityTest, SnapshotInfrastructure)
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
    Qwen35,
    Qwen35LocalTPParityTest,
    ::testing::ValuesIn(kLocalTPConfigs),
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
