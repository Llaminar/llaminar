/**
 * @file Test__Qwen35_SingleDevice_Parity.cpp
 * @brief Single-device Qwen3.5 parity tests (CPU, CUDA, ROCm)
 *
 * Tests that single-device Qwen3.5 inference produces results matching
 * PyTorch reference outputs. Validates:
 *   - GDN (Gated Delta Network) layer integration (conv1d, delta-rule, gated norm)
 *   - Full Attention layer integration (GQA with QK norms, partial RoPE)
 *   - Heterogeneous layer dispatch (GDN vs FA selected per layer index)
 *   - Attention output gating (shared by both layer types)
 *   - SwiGLU FFN (shared by both layer types)
 *
 * Configurations:
 *   - CPU: Full-precision baseline with FP16 and Q8_1 KV cache
 *   - CUDA: Single NVIDIA GPU
 *   - ROCm: Single AMD GPU
 *
 * Model: Qwen3.5-0.8B-Q4_0.gguf (Q4_0 quantization, expect wider tolerances)
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "Qwen35ParityTestBase.h"
#include "collective/BackendRouter.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

// NOTE: Qwen3.5-0.8B uses Q4_0 quantization which diverges more from FP32 reference
// than Q8_0. Additionally, GDN layers use recurrent delta-rule which may accumulate
// small numerical differences across sequence positions. Thresholds are set
// conservatively and should be tightened once baseline numbers are established.

static const std::vector<TestConfig> kQwen35SingleDeviceConfigs = {
    {
        .name = "Qwen35_CPU_KV_FP16",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.90f,        // Q4_0 + GDN diverges more than Q8_0
            .decode_cosine_threshold = 0.85f, // GDN recurrence accumulates drift
            .early_layers_count = 6,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.02f,
        },
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "Qwen35_CPU_KV_Q8_1",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 6,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.02f,
        },
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
    {
        .name = "Qwen35_CPU_KV_Q16_1",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 6,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.02f,
        },
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q16_1,
    },
    {
        .name = "Qwen35_CUDA_KV_FP16",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 6,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.02f,
        },
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "Qwen35_CUDA_KV_Q8_1",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 6,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.02f,
        },
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
    {
        .name = "Qwen35_ROCm_KV_FP16",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 6,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.02f,
            .min_top1_accuracy = 60.0f,
            .min_top5_accuracy = 60.0f,
        },
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "Qwen35_ROCm_KV_Q8_1",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 6,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.02f,
            .min_top1_accuracy = 60.0f,
            .min_top5_accuracy = 60.0f,
        },
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35SingleDeviceParityTest : public Qwen35ConfigDrivenParityTest<Qwen35SingleDeviceParityTest>,
                                     public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen35SingleDeviceParityTest, PrefillParity)
{
    auto summary = runSingleDevicePrefillParity();
    assertParity(summary);
}

TEST_P(Qwen35SingleDeviceParityTest, DecodeParity)
{
    auto summary = runSingleDeviceDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen35SingleDeviceParityTest, SnapshotInfrastructure)
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
    Qwen35SingleDeviceParityTest,
    ::testing::ValuesIn(kQwen35SingleDeviceConfigs),
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

    MPI_Finalize();
    return result;
}
