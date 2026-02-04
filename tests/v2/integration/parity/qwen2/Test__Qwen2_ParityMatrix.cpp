/**
 * @file Test__Qwen2_ParityMatrix.cpp
 * @brief Declarative test matrix for Qwen2 PyTorch parity tests
 *
 * This file defines the configuration space for Qwen2 parity tests:
 *   - devices: List of device types (CPU, CUDA, ROCm) - heterogeneous supported
 *   - parallelism: None, LocalTP
 *   - collectiveBackend: None, NCCL, RCCL, PCIeBAR
 *
 * All imperative setup is handled by ConfigDrivenParityTest in Qwen2ParityTestBase.h.
 * This file only defines configurations and instantiates tests.
 *
 * @author David Sanftenberg
 * @date 2026-01-30
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "Qwen2ParityTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

// Common excluded stages for LocalTP (sharded outputs can't be compared directly)
static const std::vector<std::string> kTPExcludedStages = {
    "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "Q_ROPE", "K_ROPE",
    "ATTENTION_CONTEXT",
    "FFN_GATE", "FFN_UP", "FFN_SWIGLU"};

/**
 * @brief All test configurations in the parity test matrix
 */
static const std::vector<TestConfig> kTestConfigs = {
    // =========================================================================
    // Single-Device Tests
    // =========================================================================
    {
        .name = "CPU",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.999f,
            .decode_cosine_threshold = 0.99f,
            .early_layers_count = 4,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.15f,
        },
    },
    {
        .name = "CUDA",
        .devices = {ParityDeviceType::CUDA},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.95f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.10f,
        },
    },
    {
        .name = "ROCm",
        .devices = {ParityDeviceType::ROCm},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.95f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.10f,
        },
    },

    // =========================================================================
    // LOCAL Tensor Parallelism
    // =========================================================================
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
            .kl_threshold = 0.25f, // Relaxed from 0.20 - TP introduces quantization variance
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
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.50f, // Relaxed - 4-way TP + RCCL host staging adds more variance
            .excluded_stages = kTPExcludedStages,
        },
    },
    {
        .name = "LocalTP_PCIeBAR_CUDA_ROCm",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm}, // Heterogeneous!
        .parallelism = Parallelism::LocalTP,
        .collective = Collective::PCIeBAR,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.50f, // Relaxed - heterogeneous TP with PCIe BAR adds variance
            .excluded_stages = kTPExcludedStages,
        },
    },

    // =========================================================================
    // LOCAL Pipeline Parallelism
    // =========================================================================
    // PP splits layers across devices (unlike TP which shards weights).
    // Each stage processes a subset of layers and transfers activations.
    {
        .name = "LocalPP_RCCL_2xROCm",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .collective = Collective::RCCL,
        .thresholds = {
            .cosine_threshold = 0.95f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.20f,
        },
    },
    {
        .name = "LocalPP_NCCL_2xCUDA",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalPP,
        .collective = Collective::NCCL,
        .thresholds = {
            .cosine_threshold = 0.95f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.20f,
        },
    },
    {
        .name = "LocalPP_PCIeBAR_CUDA_ROCm",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::ROCm},
        .parallelism = Parallelism::LocalPP,
        .collective = Collective::PCIeBAR,
        .thresholds = {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.85f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.30f, // Relaxed - cross-vendor PP adds variance
        },
    },
    {
        .name = "LocalPP_HOST_CUDA_CPU",
        .devices = {ParityDeviceType::CUDA, ParityDeviceType::CPU},
        .parallelism = Parallelism::LocalPP,
        .collective = Collective::None, // HOST backend auto-selected for CPU
        .thresholds = {
            .cosine_threshold = 0.95f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.20f,
        },
    },

    // =========================================================================
    // Hybrid: LOCAL PP of LOCAL TP domains
    // =========================================================================
    // These tests combine PP (layer split) with TP (weight shard) within stages.
    // E.g., Stage 0 = TP(2xROCm), Stage 1 = single CUDA
    {
        .name = "LocalPP_TP2xROCm_CUDA",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm, ParityDeviceType::CUDA},
        .parallelism = Parallelism::LocalPP, // PP between TP domain and CUDA
        .collective = Collective::PCIeBAR,   // Cross-vendor transfer between stages
        .thresholds = {
            .cosine_threshold = 0.85f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.50f,                // Relaxed - combined TP+PP adds variance
            .excluded_stages = kTPExcludedStages, // TP excluded stages apply to stage 0
        },
        .pp_stage_sizes = {2, 1},          // Stage 0: 2 devices (TP), Stage 1: 1 device
        .tp_collective = Collective::RCCL, // TP collective within stage 0 (ROCm-ROCm)
    },
    {
        .name = "LocalPP_TP2xROCm_CPU",
        .devices = {ParityDeviceType::ROCm, ParityDeviceType::ROCm, ParityDeviceType::CPU},
        .parallelism = Parallelism::LocalPP, // PP between TP domain and CPU
        .collective = Collective::None,      // HOST backend for GPU→CPU transfer
        .thresholds = {
            .cosine_threshold = 0.85f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.50f,                // Relaxed - combined TP+PP adds variance
            .excluded_stages = kTPExcludedStages, // TP excluded stages apply to stage 0
        },
        .pp_stage_sizes = {2, 1},          // Stage 0: 2 devices (TP), Stage 1: 1 device
        .tp_collective = Collective::RCCL, // TP collective within stage 0 (ROCm-ROCm)
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

/**
 * @brief Parameterized test fixture - just provides config to ConfigDrivenParityTest
 */
class Qwen2ParityMatrixTest : public ConfigDrivenParityTest<Qwen2ParityMatrixTest>,
                              public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen2ParityMatrixTest, PrefillParity)
{
    if (cfg().is_local_pp())
    {
        // LocalPP parity test - setup PP pipeline, then run parity (don't overwrite runner_)
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
        auto summary = runPrefillParity();
        assertParity(summary);
    }
    else if (cfg().is_local_tp())
    {
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
        auto summary = runTPPrefillParity();
        assertTPParity(summary);
    }
    else
    {
        auto summary = runSingleDevicePrefillParity();
        assertParity(summary);
    }
}

TEST_P(Qwen2ParityMatrixTest, DecodeParity)
{
    if (cfg().is_local_pp())
    {
        // LocalPP decode parity test - setup PP pipeline, then run parity
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
        auto summary = runDecodeParity();
        assertDecodeParity(summary);
    }
    else if (cfg().is_local_tp())
    {
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
        auto summary = runTPDecodeParity();
        assertDecodeParity(summary);
    }
    else
    {
        auto summary = runSingleDeviceDecodeParity();
        assertDecodeParity(summary);
    }
}

TEST_P(Qwen2ParityMatrixTest, SnapshotInfrastructure)
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
    Qwen2ParityMatrixTest,
    ::testing::ValuesIn(kTestConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
    });

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

/**
 * @brief Custom main() to initialize MPI before running tests
 *
 * LocalTP tests require MPI to be initialized. When run via ctest with mpirun,
 * MPI_Init must still be called by the process itself.
 */
int main(int argc, char **argv)
{
    // Initialize MPI with thread support for LocalTP tests
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize GoogleTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
