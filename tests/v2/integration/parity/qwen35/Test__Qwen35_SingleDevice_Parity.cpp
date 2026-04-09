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
    // =========================================================================
    // Qwen3.5-0.8B (Q4_0) — n_k_heads == n_v_heads == 16
    // =========================================================================
    {
        .name = "Qwen35_08B_CPU_KV_FP16",
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
        .name = "Qwen35_08B_CPU_KV_Q8_1",
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
        .name = "Qwen35_08B_CPU_KV_Q16_1",
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
        .name = "Qwen35_08B_CUDA_KV_FP16",
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
        .name = "Qwen35_08B_CUDA_KV_Q8_1",
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
        .name = "Qwen35_08B_ROCm_KV_FP16",
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
        .name = "Qwen35_08B_ROCm_KV_Q8_1",
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

    // =========================================================================
    // Qwen3.5-4B (Q8_0) — n_k_heads=16, n_v_heads=32 (tests repeat_interleave)
    //
    // ACTIVATION ROTATION mitigates the 4B model's massive activation outliers
    // (kurtosis up to 1191 at layer 11). Block-diagonal orthogonal rotation
    // (block_dim=128) spreads outlier energy across dimensions before Q8_1
    // quantization, improving worst-layer cosine from ~0.85 to ~0.93+.
    // Remaining gap vs FP32 is due to Q8_0 weight quantization, not outliers.
    //
    // Post exp() polynomial fix: LM_HEAD cosine 0.963→0.989, KL 0.68→0.04,
    // Top-1 0%→100%. Thresholds tightened accordingly.
    // =========================================================================
    {
        .name = "Qwen35_4B_CPU_KV_FP16",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,        // Observed: ~0.98+ post exp() fix
            .decode_cosine_threshold = 0.93f, // Observed: ~0.97+ post exp() fix
            .early_layers_count = 8,
            .min_early_layers_passed = 8, // All 8 early layers pass with rotation
            .kl_threshold = 0.10f,        // Observed: 0.04 prefill post exp() fix
            .min_top1_accuracy = 80.0f,   // Observed: 100% post exp() fix
            .min_top5_accuracy = 80.0f,   // Observed: 100% post exp() fix
            .pytorch_top1_in_topk = 3,    // Re-enabled: exp() fix restored accuracy
        },
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },

    // =========================================================================
    // Qwen3.5-4B Extended Decode (20 steps)
    //
    // Tests decode parity over 20 tokens to detect GDN state drift that may
    // not manifest in the standard 5-step test. Uses separate snapshot dir with
    // 20 decode steps from PyTorch reference. Thresholds are intentionally loose
    // to characterize divergence rather than gate— the primary goal is to see
    // the cosine-over-steps trend and token match rate over a longer horizon.
    // =========================================================================
    {
        .name = "Qwen35_4B_CPU_ExtendedDecode20",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.90f, // Looser: expect drift over 20 steps
            .early_layers_count = 8,
            .min_early_layers_passed = 8,
            .kl_threshold = 0.50f,      // Looser: characterize, not gate
            .min_top1_accuracy = 50.0f, // Looser: expect some divergence
            .min_top5_accuracy = 60.0f,
            .min_decode_pass_rate = 0.60f, // At least 60% of steps pass
            .pytorch_top1_in_topk = 0,     // Disabled: focus on trend, not gating
        },
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_decode20_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
        .decode_steps = 20,
    },
    {
        .name = "Qwen35_4B_CUDA_KV_FP16",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.93f,
            .early_layers_count = 8,
            .min_early_layers_passed = 8,
            .kl_threshold = 0.10f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f,
            .pytorch_top1_in_topk = 3,
        },
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    {
        .name = "Qwen35_4B_ROCm_KV_FP16",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.93f,
            .early_layers_count = 8,
            .min_early_layers_passed = 8,
            .kl_threshold = 0.10f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f,
            .pytorch_top1_in_topk = 3,
        },
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
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
