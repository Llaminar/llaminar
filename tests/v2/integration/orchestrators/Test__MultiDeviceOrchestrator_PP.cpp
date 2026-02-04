/**
 * @file Test__MultiDeviceOrchestrator_PP.cpp
 * @brief Integration tests for MultiDeviceOrchestrator Pipeline Parallelism mode
 *
 * Tests the PP mode of MultiDeviceOrchestrator:
 * - PP stage runner creation with layer-partitioned model contexts
 * - Sequential forward execution through stages
 * - Activation transfer between stages via LocalPPContext
 * - Logits output from final stage
 *
 * Note: These tests run on CPU (no GPU required) to verify the orchestration
 * logic. GPU-specific PP tests are in parity test suite.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <fstream>

#include "execution/local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/config/RuntimeConfig.h"
#include "loaders/ModelContext.h"
#include "backends/DeviceId.h"
#include "backends/GlobalDeviceAddress.h"
#include "collective/ILocalPPContext.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MultiDeviceOrchestrator_PP : public ::testing::Test
{
protected:
    static constexpr const char *TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    static constexpr int MAX_SEQ_LEN = 64;
    static constexpr int BATCH_SIZE = 1;
    static constexpr int NUM_LAYERS = 24;

    void SetUp() override
    {
        // Check model exists
        std::ifstream f(TEST_MODEL_PATH);
        if (!f.good())
        {
            GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
        }

        // Load full model context for reference
        model_ctx_ = ModelContext::create(TEST_MODEL_PATH);
        ASSERT_NE(model_ctx_, nullptr);
        ASSERT_EQ(model_ctx_->blockCount(), NUM_LAYERS) << "Expected 24 layer model";
    }

    /**
     * @brief Create PPStageConfig for a given layer range
     */
    MultiDeviceOrchestrator::PPStageConfig createPPStageConfig(
        int first_layer, int last_layer, DeviceId device,
        bool has_embedding = false, bool has_lm_head = false)
    {
        MultiDeviceOrchestrator::PPStageConfig config;
        config.first_layer = first_layer;
        config.last_layer = last_layer;
        config.has_embedding = has_embedding;
        config.has_lm_head = has_lm_head;
        config.stage_devices.push_back(GlobalDeviceAddress::cpu());
        return config;
    }

    /**
     * @brief Create 2-stage PP config on CPU
     */
    MultiDeviceOrchestrator::Config create2StageCPUConfig()
    {
        MultiDeviceOrchestrator::Config config;
        config.max_seq_len = MAX_SEQ_LEN;
        config.batch_size = BATCH_SIZE;
        config.activation_precision = ActivationPrecision::FP32;
        config.mode = MultiDeviceOrchestrator::ParallelismMode::PP;

        // Stage 0: layers [0, 12) with embedding
        config.pp_stages.push_back(createPPStageConfig(
            0, 12, DeviceId::cpu(), true, false));

        // Stage 1: layers [12, 24) with LM head
        config.pp_stages.push_back(createPPStageConfig(
            12, 24, DeviceId::cpu(), false, true));

        return config;
    }

    std::shared_ptr<ModelContext> model_ctx_;
};

// =============================================================================
// PP Configuration Tests
// =============================================================================

/**
 * @test PP mode detection from config
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, ConfigDetectsMode_PP)
{
    MultiDeviceOrchestrator::Config config;
    config.max_seq_len = MAX_SEQ_LEN;
    config.batch_size = BATCH_SIZE;
    config.mode = MultiDeviceOrchestrator::ParallelismMode::AUTO;

    // No TP devices, only PP stages
    config.pp_stages.push_back(createPPStageConfig(0, 12, DeviceId::cpu(), true, false));
    config.pp_stages.push_back(createPPStageConfig(12, 24, DeviceId::cpu(), false, true));

    EXPECT_EQ(config.detectMode(), MultiDeviceOrchestrator::ParallelismMode::PP);
}

/**
 * @test 2-stage PP config validates successfully
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Config_2Stage_IsValid)
{
    auto config = create2StageCPUConfig();
    EXPECT_TRUE(config.validate()) << "2-stage CPU PP config should be valid";
}

/**
 * @test PP config with layer gap fails validation
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Config_LayerGap_IsInvalid)
{
    MultiDeviceOrchestrator::Config config;
    config.max_seq_len = MAX_SEQ_LEN;
    config.batch_size = BATCH_SIZE;
    config.mode = MultiDeviceOrchestrator::ParallelismMode::PP;

    // Gap: [0, 10), [12, 24) - missing layers 10-11
    config.pp_stages.push_back(createPPStageConfig(0, 10, DeviceId::cpu(), true, false));
    config.pp_stages.push_back(createPPStageConfig(12, 24, DeviceId::cpu(), false, true));

    EXPECT_FALSE(config.validate()) << "PP config with layer gap should be invalid";
}

/**
 * @test PP config layer boundaries extraction
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Config_LayerBoundaries)
{
    auto config = create2StageCPUConfig();
    auto boundaries = config.buildLayerBoundaries();

    ASSERT_EQ(boundaries.size(), 3);
    EXPECT_EQ(boundaries[0], 0);  // Start of stage 0
    EXPECT_EQ(boundaries[1], 12); // Start of stage 1
    EXPECT_EQ(boundaries[2], 24); // End of stage 1
}

// =============================================================================
// PP Orchestrator Creation Tests (require model loading)
// =============================================================================

/**
 * @test MultiDeviceOrchestrator construction in PP mode (CPU)
 *
 * This test verifies that the orchestrator can be constructed in PP mode
 * and creates the expected number of stage runners.
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, ConstructionCreatesStageRunners)
{
    auto config = create2StageCPUConfig();

    // Create orchestrator - this should create 2 PP stage runners
    auto orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, config);
    ASSERT_NE(orchestrator, nullptr);

    // Verify PP mode was selected
    EXPECT_EQ(orchestrator->effectiveMode(), MultiDeviceOrchestrator::ParallelismMode::PP);

    // Verify vocabulary size is available (implies successful initialization)
    EXPECT_GT(orchestrator->vocab_size(), 0) << "Should have valid vocab_size from model";
}

// =============================================================================
// PP Forward Execution Tests (require model loading + inference)
// NOTE: These tests are DISABLED because DeviceGraphOrchestrator::forward()
// currently validates that token_ids is non-null even for PP middle stages
// that receive hidden state input via setHiddenState(). This is a limitation
// in the graph build session validation that needs to be addressed in
// DeviceGraphOrchestrator before these tests can pass.
// =============================================================================

/**
 * @test PP forward produces logits
 *
 * Verifies that forward() in PP mode produces non-null logits output.
 * This is an integration test that exercises the full PP execution path.
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Forward_ProducesLogits)
{
    auto config = create2StageCPUConfig();
    auto orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, config);
    ASSERT_NE(orchestrator, nullptr);

    // Run forward with a simple prompt
    std::vector<int> tokens = {151644, 8948, 198}; // Simple system token sequence
    bool success = orchestrator->forward(tokens.data(), static_cast<int>(tokens.size()));
    EXPECT_TRUE(success) << "PP forward should succeed";

    // Check logits are available
    const float *logits = orchestrator->logits();
    EXPECT_NE(logits, nullptr) << "PP forward should produce logits";
}

/**
 * @test PP forward vs single-device forward produces similar results
 *
 * This is a smoke test to verify PP doesn't produce wildly different results.
 * More rigorous parity testing is done in the parity test suite.
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, Forward_SimilarToSingleDevice)
{
    // Create PP orchestrator
    auto pp_config = create2StageCPUConfig();
    auto pp_orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, pp_config);
    ASSERT_NE(pp_orchestrator, nullptr);

    // Create single-device orchestrator (TP mode with 1 device)
    MultiDeviceOrchestrator::Config tp_config;
    tp_config.max_seq_len = MAX_SEQ_LEN;
    tp_config.batch_size = BATCH_SIZE;
    tp_config.activation_precision = ActivationPrecision::FP32;
    tp_config.mode = MultiDeviceOrchestrator::ParallelismMode::TP;
    tp_config.devices.push_back(GlobalDeviceAddress::cpu());

    auto tp_orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, tp_config);
    ASSERT_NE(tp_orchestrator, nullptr);

    // Run same tokens through both
    std::vector<int> tokens = {151644, 8948, 198};

    bool pp_success = pp_orchestrator->forward(tokens.data(), static_cast<int>(tokens.size()));
    EXPECT_TRUE(pp_success);

    bool tp_success = tp_orchestrator->forward(tokens.data(), static_cast<int>(tokens.size()));
    EXPECT_TRUE(tp_success);

    // Compare logits - they should be close (not necessarily identical due to
    // potential numerical differences in activation transfer)
    const float *pp_logits = pp_orchestrator->logits();
    const float *tp_logits = tp_orchestrator->logits();
    ASSERT_NE(pp_logits, nullptr);
    ASSERT_NE(tp_logits, nullptr);

    // Find argmax for both - they should pick the same top token
    int vocab = pp_orchestrator->vocab_size();
    ASSERT_GT(vocab, 0);

    int pp_argmax = 0, tp_argmax = 0;
    float pp_max = pp_logits[0], tp_max = tp_logits[0];
    for (int i = 1; i < vocab; ++i)
    {
        if (pp_logits[i] > pp_max)
        {
            pp_max = pp_logits[i];
            pp_argmax = i;
        }
        if (tp_logits[i] > tp_max)
        {
            tp_max = tp_logits[i];
            tp_argmax = i;
        }
    }

    // They should predict the same top token (strong requirement for correct PP)
    EXPECT_EQ(pp_argmax, tp_argmax)
        << "PP and single-device should predict same top token. "
        << "PP=" << pp_argmax << " TP=" << tp_argmax;
}

// =============================================================================
// PP Cache Clear Tests
// =============================================================================

/**
 * @test PP clear_cache propagates to all stage runners
 *
 * DISABLED: Requires DeviceGraphOrchestrator fix for PP middle stage forward()
 */
TEST_F(Test__MultiDeviceOrchestrator_PP, ClearCache_PropagatestoStages)
{
    auto config = create2StageCPUConfig();
    auto orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, config);
    ASSERT_NE(orchestrator, nullptr);

    // Run forward to populate caches
    std::vector<int> tokens = {151644};
    EXPECT_TRUE(orchestrator->forward(tokens.data(), 1));

    // Clear should not throw
    EXPECT_NO_THROW(orchestrator->clear_cache());

    // Should be able to run forward again after clear
    EXPECT_TRUE(orchestrator->forward(tokens.data(), 1));
}
