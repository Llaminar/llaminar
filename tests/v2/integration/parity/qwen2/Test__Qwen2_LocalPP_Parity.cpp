/**
 * @file Test__Qwen2_LocalPP_Parity.cpp
 * @brief Local Pipeline Parallelism Qwen2 parity tests
 *
 * Tests that Local PP inference (layers split across devices within a single
 * node with activation transfer between stages) produces results matching
 * PyTorch reference outputs.
 *
 * Configurations:
 *   - LocalPP_RCCL_2xROCm:          2x AMD GPU via RCCL
 *   - LocalPP_NCCL_2xCUDA:          2x NVIDIA GPU via NCCL
 *   - LocalPP_HOST_CUDA_ROCm:        Heterogeneous CUDA+ROCm via HOST backend
 *   - LocalPP_HOST_CUDA_CPU:         CUDA + CPU via HOST backend
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

static const std::vector<TestConfig> kLocalPPConfigs = {
    // PP splits layers across devices (unlike TP which shards weights).
    // Each stage processes a subset of layers and transfers activations.
    {
        .name = "LocalPP_RCCL_2xROCm",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .thresholds = {
            .cosine_threshold = 0.97f,        // Observed: 0.998 prefill cosine
            .decode_cosine_threshold = 0.97f, // Observed: 0.999 avg decode cosine
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.015f, // Observed: 0.003 prefill KL (was 0.20 = 67x over-relaxed)
        },
    },
    {
        .name = "LocalPP_NCCL_2xCUDA",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalPP,
        .thresholds = {
            .cosine_threshold = 0.97f, // Observed: 0.999 prefill cosine (NCCL, no results yet — matching RCCL)
            .decode_cosine_threshold = 0.97f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.015f, // Was 0.20 = over-relaxed, tightened to match RCCL
        },
    },
    {
        .name = "LocalPP_HETEROGENEOUS_CUDA_ROCm",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .thresholds = {
            .cosine_threshold = 0.97f,        // Observed: 0.999 prefill cosine
            .decode_cosine_threshold = 0.97f, // Observed: 0.999 avg decode cosine
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.025f, // Observed: 0.006 prefill KL (was 0.30 = 54x over-relaxed)
        },
    },
    {
        .name = "LocalPP_HOST_CUDA_CPU",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CPU},
        .parallelism = Parallelism::LocalPP,
        .thresholds = {
            .cosine_threshold = 0.97f,        // Observed: 0.999 prefill cosine
            .decode_cosine_threshold = 0.97f, // Observed: 0.999 avg decode cosine
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.015f, // Observed: 0.004 prefill KL (was 0.20 = 54x over-relaxed)
        },
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen2LocalPPParityTest : public ConfigDrivenParityTest<Qwen2LocalPPParityTest>,
                               public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen2LocalPPParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runPrefillParity();
    assertParity(summary);
}

TEST_P(Qwen2LocalPPParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

TEST_P(Qwen2LocalPPParityTest, SnapshotInfrastructure)
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
    Qwen2LocalPPParityTest,
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
    // Without this, the static GlobalBackendRouter::instance_ is destroyed
    // during atexit handlers after CUDA has shut down, causing crashes.
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    // Skip static destructors — see Test__Qwen2_SingleDevice_Parity.cpp for rationale.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
