/**
 * @file Qwen2ParityTestBase.h
 * @brief Base class and macros for Qwen2 PyTorch parity tests
 *
 * Provides model-specific infrastructure for Qwen2 parity testing.
 * Backend-specific tests (CPU, CUDA, ROCm) inherit from this and
 * only need to provide configuration - the test cases are generated
 * automatically via INSTANTIATE_QWEN2_PARITY_TESTS macro.
 *
 * Usage:
 *   class Test__Qwen2_CPU_vs_PyTorch : public Qwen2ParityTestBase {
 *   protected:
 *       BackendThresholds getBackendThresholds() override {
 *           return {.cosine_threshold=0.999f, .early_layers_count=4, ...};
 *       }
 *       DeviceId getDevice() override { return DeviceId::cpu(); }
 *       std::string getBackendName() override { return "CPU"; }
 *   };
 *   INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_CPU_vs_PyTorch);
 *
 * @author David Sanftenberg
 * @date 2026-01-11
 */

#pragma once

#include "../ParityTestBase.h"

namespace llaminar2::test::parity::qwen2
{

    /**
     * @brief Backend-specific threshold configuration
     *
     * Different backends have different quantization characteristics:
     * - CPU (Q8_1): Per-block quantization, tighter thresholds
     * - CUDA (INT8): Per-row symmetric, relaxed thresholds
     * - ROCm: Similar to CUDA
     */
    struct BackendThresholds
    {
        float cosine_threshold = 0.99f;  ///< Minimum cosine similarity for layer pass
        int early_layers_count = 6;      ///< Number of early layers to check strictly
        int min_early_layers_passed = 6; ///< Minimum early layers that must pass
        float kl_threshold = 0.15f;      ///< Maximum KL divergence for LM_HEAD
    };

    /**
     * @brief Base class for Qwen2-specific parity tests
     *
     * Inherits from ParityTestBase and adds Qwen2-specific configuration.
     * Subclasses only need to implement:
     * - getBackendThresholds() - Return backend-specific thresholds
     * - getDevice() - Return DeviceId for inference
     * - getBackendName() - Return display name
     * - setupDeviceSpecific() (optional) - Device initialization
     */
    class Qwen2ParityTestBase : public ParityTestBase
    {
    protected:
        /**
         * @brief Get backend-specific threshold configuration
         * @return BackendThresholds struct with cosine/KL thresholds
         */
        virtual BackendThresholds getBackendThresholds() = 0;

        void SetUp() override
        {
            // Apply backend-specific thresholds
            auto thresholds = getBackendThresholds();
            config_.cosine_threshold = thresholds.cosine_threshold;
            config_.use_avg_cosine = true; // Always use average for consistency
            config_.early_layers_count = thresholds.early_layers_count;
            config_.min_early_layers_passed = thresholds.min_early_layers_passed;
            config_.kl_threshold = thresholds.kl_threshold;

            // Qwen2-specific model configuration (already defaults in ParityConfig)
            // config_.model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
            // config_.snapshot_dir = "pytorch_qwen2_snapshots";

            // Call parent setup (regenerates snapshots, etc.)
            ParityTestBase::SetUp();
        }
    };

} // namespace llaminar2::test::parity::qwen2

/**
 * @brief Macro to instantiate standard Qwen2 parity test cases
 *
 * Generates the common test cases for any Qwen2 parity test fixture:
 * - PrefillParity_LayerByLayer: Main parity test
 * - SnapshotInfrastructure: Verifies snapshot loading works
 *
 * Usage:
 *   class Test__Qwen2_CPU_vs_PyTorch : public Qwen2ParityTestBase { ... };
 *   INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_CPU_vs_PyTorch);
 */
#define INSTANTIATE_QWEN2_PARITY_TESTS(TestFixture)                                          \
    TEST_F(TestFixture, PrefillParity_LayerByLayer)                                          \
    {                                                                                        \
        auto summary = runPrefillParity();                                                   \
        assertParity(summary);                                                               \
    }                                                                                        \
                                                                                             \
    TEST_F(TestFixture, SnapshotInfrastructure)                                              \
    {                                                                                        \
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";                             \
                                                                                             \
        auto embedding = loadPyTorchSnapshot("EMBEDDING");                                   \
        ASSERT_FALSE(embedding.empty()) << "Failed to load EMBEDDING snapshot";              \
                                                                                             \
        ASSERT_TRUE(runner_ != nullptr);                                                     \
        runner_->forward(config_.token_ids.data(), config_.token_ids.size());                \
                                                                                             \
        auto keys = runner_->getSnapshotKeys();                                              \
        EXPECT_GT(keys.size(), 0) << "No snapshots captured";                                \
                                                                                             \
        bool has_embedding = std::find(keys.begin(), keys.end(), "EMBEDDING") != keys.end(); \
        bool has_lm_head = std::find(keys.begin(), keys.end(), "LM_HEAD") != keys.end();     \
        EXPECT_TRUE(has_embedding) << "Missing EMBEDDING snapshot";                          \
        EXPECT_TRUE(has_lm_head) << "Missing LM_HEAD snapshot";                              \
    }
