/**
 * @file Test__PPStageRunner.cpp
 * @brief Integration tests for createPPStageRunner factory function
 *
 * Tests the Pipeline Parallelism stage runner creation and configuration:
 * - First stage (embedding + initial layers)
 * - Middle stage (layers only, no embedding/LM head)
 * - Last stage (final layers + LM head)
 * - Invalid configuration rejection
 *
 * These tests require the test model at models/qwen2.5-0.5b-instruct-q4_0.gguf.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <fstream>
#include <memory>

#include "execution/factory/InferenceRunnerFactory.h"
#include "loaders/ModelContext.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

namespace
{

    // Test model path (relative to workspace root, set as WORKING_DIRECTORY in CMake)
    const std::string TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__PPStageRunner : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Skip if model doesn't exist
            std::ifstream f(TEST_MODEL_PATH);
            if (!f.good())
            {
                GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
            }
        }

        /**
         * @brief Load the test model into a ModelContext
         * @return Shared pointer to ModelContext, or nullptr on failure
         */
        std::shared_ptr<ModelContext> loadModel()
        {
            return ModelContext::create(TEST_MODEL_PATH);
        }

        /**
         * @brief Load a PP stage context with layer-partitioned weights
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding Whether this stage owns embedding
         * @param has_lm_head Whether this stage owns LM head
         * @return Shared pointer to ModelContext, or nullptr on failure
         */
        std::shared_ptr<ModelContext> loadPPStageContext(
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head)
        {
            return ModelContext::createForPPStage(
                TEST_MODEL_PATH,
                first_layer,
                last_layer,
                has_embedding,
                has_lm_head);
        }
    };

    // =============================================================================
    // First Stage Tests (Embedding + Initial Layers)
    // =============================================================================

    /**
     * @test Create first PP stage runner (embedding + layers 0-11)
     *
     * Validates:
     * - Factory returns non-null runner
     * - Stage can be created with has_embedding=true
     */
    TEST_F(Test__PPStageRunner, CreateFirstStageRunner)
    {
        // Qwen2.5-0.5B has 24 layers
        auto stage_ctx = loadPPStageContext(0, 12, true, false);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = 12; // First 12 layers (out of 24)
        config.has_embedding = true;
        config.has_lm_head = false;

        ASSERT_TRUE(config.isValid()) << "PP config should be valid";
        EXPECT_EQ(config.layerCount(), 12);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr) << "createPPStageRunner returned nullptr for first stage";
    }

    // =============================================================================
    // Middle Stage Tests (Layers Only)
    // =============================================================================

    /**
     * @test Create middle PP stage runner (layers 8-16, no embedding, no LM head)
     *
     * Validates:
     * - Factory returns non-null runner for middle stage
     * - Stage can be created without embedding or LM head
     */
    TEST_F(Test__PPStageRunner, CreateMiddleStageRunner)
    {
        // Create a middle stage (no embedding, no LM head)
        auto stage_ctx = loadPPStageContext(8, 16, false, false);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 8;
        config.last_layer = 16;
        config.has_embedding = false;
        config.has_lm_head = false;

        ASSERT_TRUE(config.isValid()) << "PP config should be valid";
        EXPECT_EQ(config.layerCount(), 8);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr) << "createPPStageRunner returned nullptr for middle stage";
    }

    // =============================================================================
    // Last Stage Tests (Final Layers + LM Head)
    // =============================================================================

    /**
     * @test Create last PP stage runner (layers 12-24 + LM head)
     *
     * Validates:
     * - Factory returns non-null runner for last stage
     * - Stage can be created with has_lm_head=true
     */
    TEST_F(Test__PPStageRunner, CreateLastStageRunner)
    {
        // Get the full model to query layer count
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        int num_layers = full_ctx->blockCount();
        ASSERT_EQ(num_layers, 24) << "Qwen2.5-0.5B should have 24 layers";

        // Create last stage context (layers 12-24, has LM head)
        auto stage_ctx = loadPPStageContext(12, num_layers, false, true);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 12;
        config.last_layer = num_layers;
        config.has_embedding = false;
        config.has_lm_head = true;

        ASSERT_TRUE(config.isValid()) << "PP config should be valid";
        EXPECT_EQ(config.layerCount(), 12);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr) << "createPPStageRunner returned nullptr for last stage";
    }

    // =============================================================================
    // Full Model Single Stage Tests
    // =============================================================================

    /**
     * @test Create single-stage PP runner that covers all layers
     *
     * This is equivalent to non-PP mode: one stage has everything.
     */
    TEST_F(Test__PPStageRunner, CreateSingleStageRunnerAllLayers)
    {
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        int num_layers = full_ctx->blockCount();

        // Create a stage that covers all layers (degenerate PP with 1 stage)
        auto stage_ctx = loadPPStageContext(0, num_layers, true, true);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = num_layers;
        config.has_embedding = true;
        config.has_lm_head = true;

        ASSERT_TRUE(config.isValid()) << "PP config should be valid";
        EXPECT_EQ(config.layerCount(), num_layers);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr) << "createPPStageRunner returned nullptr for full model stage";
    }

    // =============================================================================
    // Invalid Configuration Tests
    // =============================================================================

    /**
     * @test Reject invalid PP config where first_layer > last_layer
     *
     * Either the factory should return nullptr, or FactoryPPStageConfig::isValid() should fail.
     */
    TEST_F(Test__PPStageRunner, RejectsInvalidConfigFirstGreaterThanLast)
    {
        auto model_ctx = loadModel();
        ASSERT_NE(model_ctx, nullptr);

        FactoryPPStageConfig invalid_config;
        invalid_config.first_layer = 10;
        invalid_config.last_layer = 5; // Invalid: first > last
        invalid_config.has_embedding = false;
        invalid_config.has_lm_head = false;

        // Config validation should fail
        EXPECT_FALSE(invalid_config.isValid())
            << "Config with first_layer > last_layer should be invalid";

        // Factory should either return nullptr or throw for invalid config
        // We use a model context that wasn't layer-partitioned; the factory
        // should still validate the FactoryPPStageConfig before proceeding.
        auto runner = createPPStageRunner(model_ctx, DeviceId::cpu(), invalid_config);
        EXPECT_EQ(runner, nullptr)
            << "createPPStageRunner should return nullptr for invalid config";
    }

    /**
     * @test Reject invalid PP config where first_layer == last_layer (zero layers)
     */
    TEST_F(Test__PPStageRunner, RejectsInvalidConfigZeroLayers)
    {
        auto model_ctx = loadModel();
        ASSERT_NE(model_ctx, nullptr);

        FactoryPPStageConfig invalid_config;
        invalid_config.first_layer = 5;
        invalid_config.last_layer = 5; // Invalid: zero layers
        invalid_config.has_embedding = false;
        invalid_config.has_lm_head = false;

        EXPECT_FALSE(invalid_config.isValid())
            << "Config with first_layer == last_layer should be invalid";

        auto runner = createPPStageRunner(model_ctx, DeviceId::cpu(), invalid_config);
        EXPECT_EQ(runner, nullptr)
            << "createPPStageRunner should return nullptr for zero-layer config";
    }

    /**
     * @test Reject invalid PP config with negative first_layer
     */
    TEST_F(Test__PPStageRunner, RejectsInvalidConfigNegativeFirstLayer)
    {
        auto model_ctx = loadModel();
        ASSERT_NE(model_ctx, nullptr);

        FactoryPPStageConfig invalid_config;
        invalid_config.first_layer = -1;
        invalid_config.last_layer = 10;
        invalid_config.has_embedding = true;
        invalid_config.has_lm_head = false;

        EXPECT_FALSE(invalid_config.isValid())
            << "Config with negative first_layer should be invalid";

        auto runner = createPPStageRunner(model_ctx, DeviceId::cpu(), invalid_config);
        EXPECT_EQ(runner, nullptr)
            << "createPPStageRunner should return nullptr for negative first_layer";
    }

    // =============================================================================
    // Null Model Context Tests
    // =============================================================================

    /**
     * @test Reject null model context
     */
    TEST_F(Test__PPStageRunner, RejectsNullModelContext)
    {
        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = 12;
        config.has_embedding = true;
        config.has_lm_head = false;

        ASSERT_TRUE(config.isValid());

        auto runner = createPPStageRunner(nullptr, DeviceId::cpu(), config);
        EXPECT_EQ(runner, nullptr)
            << "createPPStageRunner should return nullptr for null model context";
    }

    // =============================================================================
    // Layer Range Boundary Tests
    // =============================================================================

    /**
     * @test Create PP stage with single layer
     */
    TEST_F(Test__PPStageRunner, CreateSingleLayerStage)
    {
        // Create a minimal stage with just one layer (layer 5)
        auto stage_ctx = loadPPStageContext(5, 6, false, false);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 5;
        config.last_layer = 6; // Single layer
        config.has_embedding = false;
        config.has_lm_head = false;

        ASSERT_TRUE(config.isValid());
        EXPECT_EQ(config.layerCount(), 1);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr)
            << "createPPStageRunner returned nullptr for single-layer stage";
    }

    /**
     * @test Verify PP config layer count calculation
     */
    TEST_F(Test__PPStageRunner, LayerCountCalculation)
    {
        FactoryPPStageConfig config;

        // Two-stage PP: 24 layers total
        config.first_layer = 0;
        config.last_layer = 12;
        EXPECT_EQ(config.layerCount(), 12);

        config.first_layer = 12;
        config.last_layer = 24;
        EXPECT_EQ(config.layerCount(), 12);

        // Three-stage PP: 24 layers = 8 + 8 + 8
        config.first_layer = 0;
        config.last_layer = 8;
        EXPECT_EQ(config.layerCount(), 8);

        config.first_layer = 8;
        config.last_layer = 16;
        EXPECT_EQ(config.layerCount(), 8);

        config.first_layer = 16;
        config.last_layer = 24;
        EXPECT_EQ(config.layerCount(), 8);
    }

} // anonymous namespace
