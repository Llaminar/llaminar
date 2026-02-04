/**
 * @file Test__FactoryPPStageConfig.cpp
 * @brief Unit tests for FactoryPPStageConfig struct in InferenceRunnerFactory
 *
 * Tests validation and layer counting for Pipeline Parallelism stage configuration.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>

#include "execution/factory/InferenceRunnerFactory.h"

using namespace llaminar2;

// =============================================================================
// FactoryPPStageConfig Validation Tests
// =============================================================================

/**
 * @test Valid config with all layers (first and last stage combined)
 */
TEST(Test__FactoryPPStageConfig, ValidConfigWithAllLayers)
{
    FactoryPPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 24;
    config.has_embedding = true;
    config.has_lm_head = true;

    EXPECT_TRUE(config.isValid());
    EXPECT_EQ(config.layerCount(), 24);
}

/**
 * @test Valid config for a middle stage (no embedding, no lm_head)
 */
TEST(Test__FactoryPPStageConfig, ValidConfigMiddleStage)
{
    FactoryPPStageConfig config;
    config.first_layer = 6;
    config.last_layer = 12;
    config.has_embedding = false;
    config.has_lm_head = false;

    EXPECT_TRUE(config.isValid());
    EXPECT_EQ(config.layerCount(), 6);
}

/**
 * @test Valid config for first stage only (has embedding, no lm_head)
 */
TEST(Test__FactoryPPStageConfig, ValidConfigFirstStageOnly)
{
    FactoryPPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 6;
    config.has_embedding = true;
    config.has_lm_head = false;

    EXPECT_TRUE(config.isValid());
    EXPECT_EQ(config.layerCount(), 6);
}

/**
 * @test Valid config for last stage only (no embedding, has lm_head)
 */
TEST(Test__FactoryPPStageConfig, ValidConfigLastStageOnly)
{
    FactoryPPStageConfig config;
    config.first_layer = 18;
    config.last_layer = 24;
    config.has_embedding = false;
    config.has_lm_head = true;

    EXPECT_TRUE(config.isValid());
    EXPECT_EQ(config.layerCount(), 6);
}

/**
 * @test Invalid config with negative first_layer
 */
TEST(Test__FactoryPPStageConfig, InvalidNegativeFirstLayer)
{
    FactoryPPStageConfig config;
    config.first_layer = -1;
    config.last_layer = 10;

    EXPECT_FALSE(config.isValid());
}

/**
 * @test Invalid config where first_layer equals last_layer (zero layers)
 */
TEST(Test__FactoryPPStageConfig, InvalidFirstLayerEqualLastLayer)
{
    FactoryPPStageConfig config;
    config.first_layer = 5;
    config.last_layer = 5;

    EXPECT_FALSE(config.isValid());
}

/**
 * @test Invalid config where first_layer is greater than last_layer
 */
TEST(Test__FactoryPPStageConfig, InvalidFirstLayerGreaterThanLastLayer)
{
    FactoryPPStageConfig config;
    config.first_layer = 10;
    config.last_layer = 5;

    EXPECT_FALSE(config.isValid());
}

/**
 * @test Default-constructed config has expected initial values
 */
TEST(Test__FactoryPPStageConfig, DefaultConstructedValues)
{
    FactoryPPStageConfig config;

    // Default values as per struct definition
    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 0);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_FALSE(config.has_lm_head);

    // Default config is NOT valid (first_layer == last_layer means 0 layers)
    EXPECT_FALSE(config.isValid());
}

// =============================================================================
// FactoryPPStageConfig Layer Count Tests
// =============================================================================

/**
 * @test Layer count calculation for single layer stage
 */
TEST(Test__FactoryPPStageConfig, LayerCountSingleLayer)
{
    FactoryPPStageConfig config;
    config.first_layer = 5;
    config.last_layer = 6;

    EXPECT_EQ(config.layerCount(), 1);
}

/**
 * @test Layer count calculation for multi-layer stage
 */
TEST(Test__FactoryPPStageConfig, LayerCountMultiLayer)
{
    FactoryPPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 8;

    EXPECT_EQ(config.layerCount(), 8);
}

/**
 * @test Layer count is consistent with [first, last) range semantics
 */
TEST(Test__FactoryPPStageConfig, LayerCountRangeSemantics)
{
    // For a 24-layer model split into 3 stages:
    // Stage 0: layers [0, 8)  -> 8 layers
    // Stage 1: layers [8, 16) -> 8 layers
    // Stage 2: layers [16, 24) -> 8 layers

    FactoryPPStageConfig stage0{.first_layer = 0, .last_layer = 8, .has_embedding = true, .has_lm_head = false};
    FactoryPPStageConfig stage1{.first_layer = 8, .last_layer = 16, .has_embedding = false, .has_lm_head = false};
    FactoryPPStageConfig stage2{.first_layer = 16, .last_layer = 24, .has_embedding = false, .has_lm_head = true};

    EXPECT_EQ(stage0.layerCount(), 8);
    EXPECT_EQ(stage1.layerCount(), 8);
    EXPECT_EQ(stage2.layerCount(), 8);

    // Total layers across all stages should equal 24
    EXPECT_EQ(stage0.layerCount() + stage1.layerCount() + stage2.layerCount(), 24);

    // All stages should be valid
    EXPECT_TRUE(stage0.isValid());
    EXPECT_TRUE(stage1.isValid());
    EXPECT_TRUE(stage2.isValid());
}

// =============================================================================
// FactoryPPStageConfig Edge Cases
// =============================================================================

/**
 * @test Very large layer indices are valid
 */
TEST(Test__FactoryPPStageConfig, LargeLayerIndices)
{
    FactoryPPStageConfig config;
    config.first_layer = 1000;
    config.last_layer = 2000;
    config.has_embedding = false;
    config.has_lm_head = false;

    EXPECT_TRUE(config.isValid());
    EXPECT_EQ(config.layerCount(), 1000);
}

/**
 * @test Designated initializer syntax works correctly
 */
TEST(Test__FactoryPPStageConfig, DesignatedInitializerSyntax)
{
    FactoryPPStageConfig config{
        .first_layer = 12,
        .last_layer = 24,
        .has_embedding = false,
        .has_lm_head = true};

    EXPECT_EQ(config.first_layer, 12);
    EXPECT_EQ(config.last_layer, 24);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
    EXPECT_TRUE(config.isValid());
    EXPECT_EQ(config.layerCount(), 12);
}
