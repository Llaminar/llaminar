/**
 * @file Test__Qwen2_LocalTP_Parity.cpp
 * @brief Local Tensor Parallelism Qwen2 parity tests
 *
 * Tests that Local TP inference (weight-sharded across multiple devices
 * within a single node) produces results matching PyTorch reference outputs.
 *
 * Configurations:
 *   - LocalTP_NCCL_2xCUDA:          2x NVIDIA GPU via NCCL
 *   - LocalTP_RCCL_2xROCm:          2x AMD GPU via RCCL
 *   - LocalTP_RCCL_4xROCm:          4x AMD GPU via RCCL
 *   - LocalTP_HOST_CUDA_ROCm:        Heterogeneous CUDA+ROCm via HOST backend
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

static const std::vector<TestConfig> kLocalTPConfigs = {
    {
        .name = "LocalTP_NCCL_2xCUDA",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::NCCL,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.35f, // INT8 CUTLASS GEMM + column-parallel TP adds quantization variance
            .excluded_stages = kTPExcludedStages,
        },
    },
    {
        .name = "LocalTP_RCCL_2xROCm",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::RCCL,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.40f, // Relaxed - RCCL with host staging adds variance
            .excluded_stages = kTPExcludedStages,
        },
    },
    {
        .name = "LocalTP_RCCL_4xROCm",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm, ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::RCCL,
        .thresholds = {
            .cosine_threshold = 0.96f,        // Observed: 0.999 prefill cosine
            .decode_cosine_threshold = 0.96f, // Observed: 0.999 avg decode cosine
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.02f, // Observed: 0.001 prefill KL (was 0.50 = 500x over-relaxed)
            .excluded_stages = kTPExcludedStages,
        },
    },
    {
        .name = "LocalTP_HETEROGENEOUS_CUDA_ROCm",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm}, // Heterogeneous!
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::HETEROGENEOUS,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.50f, // Relaxed - heterogeneous TP with HOST backend adds variance
            .excluded_stages = kTPExcludedStages,
        },
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen2LocalTPParityTest : public ConfigDrivenParityTest<Qwen2LocalTPParityTest>,
                               public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen2LocalTPParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runTPPrefillParity();
    assertTPParity(summary);
}

TEST_P(Qwen2LocalTPParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runTPDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen2LocalTPParityTest, SnapshotInfrastructure)
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
    Qwen2LocalTPParityTest,
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

    // Skip static destructors — see Test__Qwen2_SingleDevice_Parity.cpp for rationale.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
