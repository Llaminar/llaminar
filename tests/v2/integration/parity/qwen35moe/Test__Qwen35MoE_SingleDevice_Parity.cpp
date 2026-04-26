/**
 * @file Test__Qwen35MoE_SingleDevice_Parity.cpp
 * @brief Single-device Qwen3.5 MoE parity tests (CPU, CUDA, ROCm)
 *
 * Tests that single-device Qwen3.5 MoE inference produces results matching
 * PyTorch reference outputs. Validates:
 *   - GDN (Gated Delta Network) layer integration (same as dense Qwen3.5)
 *   - Full Attention layer integration (same as dense Qwen3.5)
 *   - MoE Router: softmax top-k expert selection (256 experts, top-8)
 *   - MoE Expert FFN: per-expert SwiGLU + weighted combine
 *   - Shared Expert: always-active dense SwiGLU + sigmoid gate
 *   - MoE Combined Output: routed + gated shared expert
 *   - Residual connections across the MoE FFN block
 *
 * Configurations:
 *   - CPU: Full-precision baseline with FP16 KV cache
 *   - CUDA: Single NVIDIA GPU (SKIP for now — model too large for single 3090)
 *   - ROCm: Single AMD GPU (SKIP for now — requires testing)
 *
 * Model: Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf at /opt/llaminar-models/
 *   - Q4_K/Q5_K/Q6_K quantization (expect wider tolerances than Q8_0)
 *   - 40 layers, 256 experts (top-8), 2048 hidden dim, 512 expert intermediate
 *
 * NOTE: Decode attention drops (0.04-0.12 cosine/layer) are expected due to
 * Llaminar's FP16 KV cache vs PyTorch's FP32 DynamicCache. MoE expert drops
 * (0.01-0.035/layer) are from block-wise GEMM accumulation order differences.
 * End-to-end token predictions match (100% top-1 across 5 decode steps).
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35MoEParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

// NOTE: Qwen3.5-35B MoE uses mixed Q4_K/Q5_K/Q6_K quantization which diverges
// more from FP32 reference than Q8_0. Additionally:
// - MoE routing introduces discrete expert selection (potential divergence source)
// - GDN layers use recurrent delta-rule (accumulates numerical differences)
// - 35B model with 256 experts — large model increases error accumulation
// - FP16 KV cache truncation causes 0.04-0.12 attention cosine drops in decode
// Thresholds are calibrated from observed single-run baselines with ~4-5% margin.

static const std::vector<TestConfig> kQwen35MoESingleDeviceConfigs = {
    // =========================================================================
    // Qwen3.5-35B MoE (Q4_K_XL) — CPU baseline
    //
    // This is the primary single-device configuration. CPU execution is fully
    // deterministic and exercises the scalar reference MoEExpertComputeStage code path.
    // The model is at /opt/llaminar-models/ (not in the models/ workspace dir).
    //
    // Expert routing: 256 experts, top-8 selection, norm_topk_prob=true
    // Shared expert: always-active with sigmoid gate
    // =========================================================================
    {
        .name = "Qwen35MoE_35B_CPU_KV_FP16",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.96f,        // Worst observed: 0.9725 layer avg
            .decode_cosine_threshold = 0.98f, // Worst observed: 0.9952 LM_HEAD avg
            .early_layers_count = 6,
            .min_early_layers_passed = 5, // 6/6 observed; require 5
            .kl_threshold = 0.03f,         // LM_HEAD observed: 0.005-0.014 (run-to-run variance with dynamic rebalance)
            .min_top1_accuracy = 0.80f,   // Observed: 100%
            .min_top5_accuracy = 0.60f,   // Observed: 80% (0.8 on 2 steps)
            .pytorch_top1_in_topk = 3,    // Observed: 5/5 RefInTop3
        },
        .model_path = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        .snapshot_dir = "pytorch_qwen35_moe_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    // =========================================================================
    // GPU configs (CUDA, ROCm) — DISABLED
    //
    // The 35B MoE model needs ~21GB for weights alone + expert GEMM slabs.
    // Single-GPU parity tests OOM on both 24GB (RTX 3090) and 32GB (MI100).
    // Re-enable once weight streaming or a smaller MoE checkpoint is available.
    // =========================================================================
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35MoESingleDeviceParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoESingleDeviceParityTest>,
      public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen35MoESingleDeviceParityTest, PrefillParity)
{
    auto summary = runSingleDevicePrefillParity();
    assertParity(summary);
}

TEST_P(Qwen35MoESingleDeviceParityTest, DecodeParity)
{
    auto summary = runSingleDeviceDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen35MoESingleDeviceParityTest, SnapshotInfrastructure)
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

    // MoE-specific: verify that FFN_RESIDUAL snapshots exist (MoE combined output + residual)
    bool has_ffn_residual = false;
    for (const auto &key : keys)
    {
        if (key.find("FFN_RESIDUAL") != std::string::npos)
        {
            has_ffn_residual = true;
            break;
        }
    }
    EXPECT_TRUE(has_ffn_residual) << "Missing FFN_RESIDUAL snapshot (MoE combined output path)";
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen35MoE,
    Qwen35MoESingleDeviceParityTest,
    ::testing::ValuesIn(kQwen35MoESingleDeviceConfigs),
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

    // Skip static destructors — avoid CUDA/ROCm atexit races.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
