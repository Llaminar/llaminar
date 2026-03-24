/**
 * @file Test__Qwen2_SingleDevice_Parity.cpp
 * @brief Single-device Qwen2 parity tests (CPU, CUDA, ROCm)
 *
 * Tests that single-device inference produces results matching
 * PyTorch reference outputs. No parallelism involved.
 *
 * Configurations:
 *   - CPU: Full-precision baseline
 *   - CUDA: Single NVIDIA GPU
 *   - ROCm: Single AMD GPU
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "Qwen2ParityTestBase.h"
#include "collective/BackendRouter.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

static const std::vector<TestConfig> kSingleDeviceConfigs = {
    {
        .name = "CPU_KV_FP16",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f, // Q4_0 quantized GEMM diverges from FP32 reference equally on CPU and GPU
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.005f,
            .min_top1_accuracy = 90.0f,
            .min_top5_accuracy = 95.0f,
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "CPU_KV_Q8_1",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f, // Q4_0 quantized GEMM diverges from FP32 reference equally on CPU and GPU
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.005f,
            .min_top1_accuracy = 90.0f,
            .min_top5_accuracy = 95.0f,
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
    {
        .name = "CPU_KV_Q16_1",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f, // Q4_0 quantized GEMM diverges from FP32 reference equally on CPU and GPU
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.005f,
            .min_top1_accuracy = 90.0f,
            .min_top5_accuracy = 95.0f,
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q16_1,
    },
    {
        .name = "CUDA_KV_FP16",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.008f, // CUDA non-determinism causes KL to fluctuate 0.002-0.006 between runs
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "CUDA_KV_Q8_1",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.008f, // CUDA non-determinism causes KL to fluctuate 0.002-0.006 between runs
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
            .pytorch_top1_in_topk = 5, // Q8_1 KV quantization pushes ref token past top-3 (both CUDA+ROCm)
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
    {
        .name = "ROCm_KV_FP16",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.005f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "ROCm_KV_FP32",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.005f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP32,
    },
    {
        .name = "ROCm_KV_Q8_1",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.005f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
            .pytorch_top1_in_topk = 5, // Q8_1 KV quantization pushes ref token past top-3 (both CUDA+ROCm)
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
    // =========================================================================
    // Q8_0 model configs — exercises the native-VNNI code path (codebook 18)
    // with per-block-of-32 FP16 weight scales preserved.
    // (Q4_0 above also exercises the native-VNNI path, codebook 0)
    // =========================================================================
    {
        .name = "CUDA_Q8_0_KV_FP16",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f, // Q8_0 native-VNNI integer GEMM diverges more from FP32 reference
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
        },
        .model_path = "models/qwen2.5-0.5b-instruct-q8_0.gguf",
        .snapshot_dir = "pytorch_qwen2_snapshots_q8_0",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "CUDA_Q8_0_KV_Q8_1",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f, // Q8_0 native-VNNI integer GEMM diverges more from FP32 reference
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
            .pytorch_top1_in_topk = 5,
        },
        .model_path = "models/qwen2.5-0.5b-instruct-q8_0.gguf",
        .snapshot_dir = "pytorch_qwen2_snapshots_q8_0",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
    {
        .name = "ROCm_Q8_0_KV_FP16",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f, // Q8_0 native-VNNI integer GEMM diverges more from FP32 reference
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
        },
        .model_path = "models/qwen2.5-0.5b-instruct-q8_0.gguf",
        .snapshot_dir = "pytorch_qwen2_snapshots_q8_0",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "ROCm_Q8_0_KV_Q8_1",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f, // Q8_0 native-VNNI integer GEMM diverges more from FP32 reference
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 95.0f,
            .pytorch_top1_in_topk = 5, // Q8_1 KV quantization pushes ref token past top-3
        },
        .model_path = "models/qwen2.5-0.5b-instruct-q8_0.gguf",
        .snapshot_dir = "pytorch_qwen2_snapshots_q8_0",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::Q8_1,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen2SingleDeviceParityTest : public ConfigDrivenParityTest<Qwen2SingleDeviceParityTest>,
                                    public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen2SingleDeviceParityTest, PrefillParity)
{
    auto summary = runSingleDevicePrefillParity();
    assertParity(summary);
}

TEST_P(Qwen2SingleDeviceParityTest, DecodeParity)
{
    auto summary = runSingleDeviceDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen2SingleDeviceParityTest, SnapshotInfrastructure)
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
    Qwen2SingleDeviceParityTest,
    ::testing::ValuesIn(kSingleDeviceConfigs),
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
