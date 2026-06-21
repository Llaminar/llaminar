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
#include <unistd.h>
#include "Qwen2ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

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
            .kl_threshold = 0.006f, // Relaxed: AVX2 fallback path introduces minor numeric drift in Q16_1 attention
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
            .kl_threshold = 0.009f, // CUDA non-determinism causes KL to fluctuate 0.002-0.008 between runs
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f, // CUDA non-determinism can shift one token out of top-5 (4/5 = 80%)
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
    // TurboQuant KV cache configs — exercises the TQ (split TQ8-K/TQ4-V) path.
    // These stay under the same end-to-end parity harness as the other KV
    // precisions because the acceptance criterion is real inference quality.
    // Thresholds match the corresponding Q8_1 configs.
    // =========================================================================
    {
        .name = "CPU_KV_TQ",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.005f,
            .min_top1_accuracy = 90.0f,
            .min_top5_accuracy = 95.0f,
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::TQ,
    },
    // =========================================================================
    // Q8_0 model configs — exercises the native-VNNI code path (codebook 18)
    // with per-block-of-32 FP16 weight scales preserved.
    // (Q4_0 above also exercises the native-VNNI path, codebook 0)
    // =========================================================================
    {
        .name = "CPU_Q8_0_KV_FP16",
        .devices = {ParityDeviceType::CPU},
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
        .name = "CPU_Q8_0_KV_Q8_1",
        .devices = {ParityDeviceType::CPU},
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
        .name = "CUDA_Q8_0_ChatCalc_KV_FP16",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.01f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f, // Short chat prompt: one shifted token out of 5 still preserves top-1/KL.
        },
        .model_path = "models/qwen2.5-0.5b-instruct-q8_0.gguf",
        .snapshot_dir = "pytorch_qwen2_snapshots_q8_0_chat_calc",
        .prompt = R"(<|im_start|>system
You are a calculator. Reply with only the numeric answer, no explanation.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
)",
        .token_ids = {151644, 8948, 198, 2610, 525, 264, 29952, 13, 17841, 448, 1172, 279, 24064, 4226, 11, 902, 16148, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198},
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
        .decode_steps = 2,
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
    // =========================================================================
    // CUDA TurboQuant KV cache configs — GPU TQ8-K/TQ4-V split parity
    // TQ adds more quantization noise than Q8_1 or FP16, so thresholds
    // must be at least as loose as CUDA_KV_FP16 (which already uses 80%).
    // =========================================================================
    {
        .name = "CUDA_KV_TQ",
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
            .min_top5_accuracy = 80.0f, // TQ4 V quantization + CUDA non-determinism can shift one token out of top-5 (4/5 = 80%)
            .pytorch_top1_in_topk = 5,  // Q8_1 KV quantization pushes ref token past top-3 (both CUDA+ROCm)
        },
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::TQ,
    },
    // =========================================================================
    // ROCm TurboQuant KV cache configs — Thresholds match ROCm_KV_Q8_1
    // =========================================================================
    {
        .name = "ROCm_KV_TQ",
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
        .kv_cache_precision = KVCachePrecision::TQ,
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
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    // Skip C++ static destructors via _exit() to avoid CUDA/ROCm atexit
    // handler races. Meyers singletons (GPUDeviceWorkerPool, CUDABackend,
    // CUDAConcurrentPrefillPool, etc.) may call CUDA/HIP APIs after the
    // runtime's own atexit handler has torn down the driver context,
    // causing intermittent SIGSEGV that mpirun reports as non-zero exit.
    // Same pattern as Main.cpp.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
