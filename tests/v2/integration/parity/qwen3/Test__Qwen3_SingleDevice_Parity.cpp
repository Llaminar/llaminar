/**
 * @file Test__Qwen3_SingleDevice_Parity.cpp
 * @brief Single-device Qwen3 parity tests (CPU, CUDA, ROCm)
 *
 * Tests that single-device Qwen3 inference produces results matching
 * PyTorch reference outputs. Validates the QKNormStage integration
 * and the absence of QKV biases.
 *
 * Configurations:
 *   - CPU: Full-precision baseline with FP16, Q8_1, and Q16_1 KV cache
 *   - CUDA: Single NVIDIA GPU
 *   - ROCm: Single AMD GPU
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "Qwen3ParityTestBase.h"
#include "collective/BackendRouter.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen3;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

static const std::vector<TestConfig> kQwen3SingleDeviceConfigs = {
    {
        .name = "Qwen3_CPU_KV_FP16",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 quantized GEMM diverges from FP32 reference
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "Qwen3_CPU_KV_Q8_1",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 quantized GEMM diverges from FP32 reference
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
    {
        .name = "Qwen3_CPU_KV_Q16_1",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 quantized GEMM diverges from FP32 reference
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q16_1,
    },
    {
        .name = "Qwen3_CPU_KV_TQ4",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
            .min_top1_accuracy = 60.0f, // TQ4 on 28-layer Qwen3 diverges at later decode steps
            .min_top5_accuracy = 80.0f,
            .pytorch_top1_in_topk = 0, // Disable strict topK gate for TQ4 (cumulative quantization error)
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::TQ4,
    },
    {
        .name = "Qwen3_CPU_KV_TQ",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 quantized GEMM diverges from FP32 reference
            .decode_cosine_threshold = 0.98f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::TQ,
    },
    {
        .name = "Qwen3_CUDA_KV_FP16",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 Layer 1 cosine ~0.945 is borderline at 0.95
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "Qwen3_CUDA_KV_Q8_1",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 Layer 1 cosine ~0.945 is borderline at 0.95
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
    {
        .name = "Qwen3_ROCm_KV_FP16",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 quantized GEMM diverges equally across backends
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
            .min_top1_accuracy = 60.0f,
            .min_top5_accuracy = 60.0f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "Qwen3_ROCm_KV_FP32",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 quantized GEMM diverges equally across backends
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
            .min_top1_accuracy = 60.0f,
            .min_top5_accuracy = 60.0f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP32,
    },
    {
        .name = "Qwen3_ROCm_KV_Q8_1",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.94f, // Q8_0 quantized GEMM diverges equally across backends
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
            .min_top1_accuracy = 60.0f,
            .min_top5_accuracy = 60.0f,
        },
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen3SingleDeviceParityTest : public Qwen3ConfigDrivenParityTest<Qwen3SingleDeviceParityTest>,
                                    public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen3SingleDeviceParityTest, PrefillParity)
{
    auto summary = runSingleDevicePrefillParity();
    assertParity(summary);
}

TEST_P(Qwen3SingleDeviceParityTest, DecodeParity)
{
    auto summary = runSingleDeviceDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen3SingleDeviceParityTest, SnapshotInfrastructure)
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
    Qwen3,
    Qwen3SingleDeviceParityTest,
    ::testing::ValuesIn(kQwen3SingleDeviceConfigs),
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
