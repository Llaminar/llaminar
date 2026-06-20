/**
 * @file Test__Qwen35_LocalPP_Parity.cpp
 * @brief Local Pipeline Parallelism Qwen3.5 parity tests
 *
 * Tests that Local PP inference (layers split across devices within a single
 * node with activation transfer between stages) produces results matching
 * PyTorch reference outputs for Qwen3.5 models with GDN + FA hybrid
 * architecture.
 *
 * Configurations (Qwen3.5-0.8B Q4_0):
 *   - LocalPP_HOST_2xCPU_08B:        2x CPU via HOST backend
 *   - LocalPP_NCCL_2xCUDA_08B:       2x NVIDIA GPU via NCCL (skipped: no GPU kernels)
 *   - LocalPP_RCCL_2xROCm_08B:       2x AMD GPU via RCCL (skipped: no GPU kernels)
 *   - LocalPP_HOST_CUDA_ROCm_08B:    Heterogeneous CUDA+ROCm (skipped: no GPU kernels)
 *
 * Configurations (Qwen3.5-4B Q8_0):
 *   - LocalPP_HOST_2xCPU_4B:         2x CPU via HOST backend
 *   - LocalPP_NCCL_2xCUDA_4B:        2x NVIDIA GPU via NCCL (skipped: no GPU kernels)
 *   - LocalPP_RCCL_2xROCm_4B:        2x AMD GPU via RCCL (skipped: no GPU kernels)
 *   - LocalPP_HOST_CUDA_ROCm_4B:     Heterogeneous CUDA+ROCm (skipped: no GPU kernels)
 *
 * Configurations (Qwen3.5-27B Q4_K_M — dense 64-layer hybrid GDN+FA):
 *   - LocalPP_NCCL_2xCUDA_27B:       2x NVIDIA GPU via NCCL
 *   - LocalPP_RCCL_2xROCm_27B:       2x AMD GPU via RCCL
 *   - LocalPP_HETEROGENEOUS_CUDA_ROCm_27B: Heterogeneous CUDA+ROCm via HOST
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;

// =============================================================================
// Test Configuration Definitions — Qwen3.5-0.8B (Q4_0)
// =============================================================================

static const auto kQwen35_08B_PP_Thresholds = BackendThresholds{
    .cosine_threshold = 0.96f,        // Observed min: 0.989 (RCCL), was 0.95
    .decode_cosine_threshold = 0.95f, // Observed min: 0.983 avg decode, was 0.90
    .early_layers_count = 6,
    .min_early_layers_passed = 4,
    .kl_threshold = 0.06f, // Observed max: 0.017 (RCCL), was 0.20 = 12x over-relaxed
};

static const auto kQwen35_4B_PP_Thresholds = BackendThresholds{
    .cosine_threshold = 0.96f,        // Observed min: 0.990 (RCCL), was 0.95
    .decode_cosine_threshold = 0.95f, // Observed min: 0.991 avg decode, was 0.90
    .early_layers_count = 6,
    .min_early_layers_passed = 4,
    .kl_threshold = 0.05f, // Observed max: 0.014 (HETERO), was 0.20 = 14x over-relaxed
};

// Q4_K_M quantization diverges more than Q8_0 due to lower precision; 64 layers
// accumulate more drift so thresholds are relaxed relative to 4B/0.8B configs.
// Empirical results: min cosine ~0.9996, KL ~0.0001 across CUDA/ROCm/heterogeneous.
static const auto kQwen35_27B_PP_Thresholds = BackendThresholds{
    .cosine_threshold = 0.998f,        // Observed: 0.9996+ across all configs
    .decode_cosine_threshold = 0.996f, // Observed: 0.9996+ in decode steps
    .early_layers_count = 8,
    .min_early_layers_passed = 7,
    .kl_threshold = 0.01f, // Observed: 0.0001-0.0002
};

static const std::vector<TestConfig> kLocalPPConfigs = {
    // =========================================================================
    // Qwen3.5-0.8B (Q4_0) — CPU-only PP (HOST backend)
    // =========================================================================
    {
        .name = "LocalPP_HOST_2xCPU_08B",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_08B_PP_Thresholds,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    // =========================================================================
    // Qwen3.5-0.8B (Q4_0) — GPU configs (skipped: no GPU kernels yet)
    // =========================================================================
    {
        .name = "LocalPP_NCCL_2xCUDA_08B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_08B_PP_Thresholds,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalPP_RCCL_2xROCm_08B",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_08B_PP_Thresholds,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalPP_HETEROGENEOUS_CUDA_ROCm_08B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_08B_PP_Thresholds,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    // =========================================================================
    // Qwen3.5-4B (Q8_0) — CPU-only PP (HOST backend)
    // =========================================================================
    {
        .name = "LocalPP_HOST_2xCPU_4B",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_4B_PP_Thresholds,
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    // =========================================================================
    // Qwen3.5-4B (Q8_0) — GPU configs (skipped: no GPU kernels yet)
    // =========================================================================
    {
        .name = "LocalPP_NCCL_2xCUDA_4B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_4B_PP_Thresholds,
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalPP_RCCL_2xROCm_4B",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_4B_PP_Thresholds,
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalPP_HETEROGENEOUS_CUDA_ROCm_4B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_4B_PP_Thresholds,
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    // =========================================================================
    // Qwen3.5-27B (Q4_K_M) — dense 64-layer hybrid GDN+FA, pipeline split 0-31 / 32-63
    // Critical for validating large-model PP correctness with Q4 quantization
    // on production GPU configurations (CUDA and ROCm).
    // =========================================================================
    {
        .name = "LocalPP_NCCL_2xCUDA_27B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_27B_PP_Thresholds,
        .model_path = "models/Qwen3.5-27B-Q4_K_M.gguf",
        .snapshot_dir = "pytorch_qwen35_27b_q4km_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalPP_RCCL_2xROCm_27B",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_27B_PP_Thresholds,
        .model_path = "models/Qwen3.5-27B-Q4_K_M.gguf",
        .snapshot_dir = "pytorch_qwen35_27b_q4km_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "LocalPP_HETEROGENEOUS_CUDA_ROCm_27B",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .thresholds = kQwen35_27B_PP_Thresholds,
        .model_path = "models/Qwen3.5-27B-Q4_K_M.gguf",
        .snapshot_dir = "pytorch_qwen35_27b_q4km_snapshots",
        .pp_weights = {0.31f, 0.69f}, // CUDA gets ~20 layers, ROCm gets ~44 (CUDA has less VRAM)
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35LocalPPParityTest : public Qwen35ConfigDrivenParityTest<Qwen35LocalPPParityTest>,
                                public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen35LocalPPParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runPrefillParity();
    assertParity(summary);
}

TEST_P(Qwen35LocalPPParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen35LocalPPParityTest, SnapshotInfrastructure)
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
    Qwen35LocalPPParityTest,
    ::testing::ValuesIn(kLocalPPConfigs),
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

    // Skip static destructors — see Test__Qwen2_SingleDevice_Parity.cpp for rationale.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
