/**
 * @file Test__PPStageConfig.cpp
 * @brief Unit tests for PPStageConfig (Unified PP Graph Architecture Phase 1.2)
 *
 * Tests the PPStageConfig structure which defines pipeline parallel stages
 * for the unified PP graph architecture.
 *
 * @see docs/v2/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "config/PPStageConfig.h"

using namespace llaminar2;

// =============================================================================
// Default Construction Tests
// =============================================================================

TEST(Test__PPStageConfig, DefaultConstruction)
{
    PPStageConfig config;

    // Default values
    EXPECT_EQ(config.stage_id, 0);
    EXPECT_TRUE(config.domain_name.empty());
    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 0);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_FALSE(config.has_lm_head);
}

// =============================================================================
// numLayers Tests
// =============================================================================

TEST(Test__PPStageConfig, NumLayers_ReturnsCorrectCount)
{
    PPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 8;

    EXPECT_EQ(config.numLayers(), 8);
}

TEST(Test__PPStageConfig, NumLayers_ZeroForSameFirstAndLast)
{
    PPStageConfig config;
    config.first_layer = 5;
    config.last_layer = 5;

    EXPECT_EQ(config.numLayers(), 0);
}

TEST(Test__PPStageConfig, NumLayers_MiddleStageRange)
{
    PPStageConfig config;
    config.first_layer = 8;
    config.last_layer = 16;

    EXPECT_EQ(config.numLayers(), 8);
}

// =============================================================================
// containsLayer Tests
// =============================================================================

TEST(Test__PPStageConfig, ContainsLayer_TrueForLayerInRange)
{
    PPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 8;

    EXPECT_TRUE(config.containsLayer(0));
    EXPECT_TRUE(config.containsLayer(4));
    EXPECT_TRUE(config.containsLayer(7));
}

TEST(Test__PPStageConfig, ContainsLayer_FalseForLayerOutOfRange)
{
    PPStageConfig config;
    config.first_layer = 8;
    config.last_layer = 16;

    // Below range
    EXPECT_FALSE(config.containsLayer(0));
    EXPECT_FALSE(config.containsLayer(7));

    // Above range
    EXPECT_FALSE(config.containsLayer(16));
    EXPECT_FALSE(config.containsLayer(20));
}

TEST(Test__PPStageConfig, ContainsLayer_ExclusiveUpperBound)
{
    PPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 8;

    // Layer 7 is in range (last valid layer)
    EXPECT_TRUE(config.containsLayer(7));

    // Layer 8 is NOT in range (upper bound is exclusive)
    EXPECT_FALSE(config.containsLayer(8));
}

TEST(Test__PPStageConfig, ContainsLayer_NegativeLayerFalse)
{
    PPStageConfig config;
    config.first_layer = 0;
    config.last_layer = 8;

    EXPECT_FALSE(config.containsLayer(-1));
}

// =============================================================================
// isFirstStage / isLastStage Tests
// =============================================================================

TEST(Test__PPStageConfig, IsFirstStage_TrueWhenHasEmbedding)
{
    PPStageConfig config;
    config.has_embedding = true;
    config.has_lm_head = false;

    EXPECT_TRUE(config.isFirstStage());
}

TEST(Test__PPStageConfig, IsFirstStage_FalseWhenNoEmbedding)
{
    PPStageConfig config;
    config.has_embedding = false;
    config.has_lm_head = true;

    EXPECT_FALSE(config.isFirstStage());
}

TEST(Test__PPStageConfig, IsLastStage_TrueWhenHasLMHead)
{
    PPStageConfig config;
    config.has_embedding = false;
    config.has_lm_head = true;

    EXPECT_TRUE(config.isLastStage());
}

TEST(Test__PPStageConfig, IsLastStage_FalseWhenNoLMHead)
{
    PPStageConfig config;
    config.has_embedding = true;
    config.has_lm_head = false;

    EXPECT_FALSE(config.isLastStage());
}

// =============================================================================
// Validation Tests - Failures
// =============================================================================

TEST(Test__PPStageConfig, Validate_FailsOnNegativeStageId)
{
    PPStageConfig config;
    config.stage_id = -1;
    config.domain_name = "test_domain";
    config.first_layer = 0;
    config.last_layer = 8;

    std::string error;
    EXPECT_FALSE(config.validate(24, &error));
    EXPECT_TRUE(error.find("stage_id") != std::string::npos);
}

TEST(Test__PPStageConfig, Validate_FailsOnEmptyDomainName)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = ""; // empty
    config.first_layer = 0;
    config.last_layer = 8;

    std::string error;
    EXPECT_FALSE(config.validate(24, &error));
    EXPECT_TRUE(error.find("domain_name") != std::string::npos);
}

TEST(Test__PPStageConfig, Validate_FailsOnNegativeFirstLayer)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = "test_domain";
    config.first_layer = -1;
    config.last_layer = 8;

    std::string error;
    EXPECT_FALSE(config.validate(24, &error));
    EXPECT_TRUE(error.find("first_layer") != std::string::npos);
}

TEST(Test__PPStageConfig, Validate_FailsOnInvalidLayerRange)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = "test_domain";
    config.first_layer = 8;
    config.last_layer = 8; // Same as first (zero layers)

    std::string error;
    EXPECT_FALSE(config.validate(24, &error));
    EXPECT_TRUE(error.find("last_layer") != std::string::npos);
}

TEST(Test__PPStageConfig, Validate_FailsOnLastLayerLessThanFirst)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = "test_domain";
    config.first_layer = 16;
    config.last_layer = 8; // Less than first

    std::string error;
    EXPECT_FALSE(config.validate(24, &error));
    EXPECT_TRUE(error.find("last_layer") != std::string::npos);
}

TEST(Test__PPStageConfig, Validate_FailsOnLayerExceedsTotalLayers)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = "test_domain";
    config.first_layer = 0;
    config.last_layer = 32; // Exceeds total_layers=24

    std::string error;
    EXPECT_FALSE(config.validate(24, &error));
    EXPECT_TRUE(error.find("total_layers") != std::string::npos);
}

TEST(Test__PPStageConfig, Validate_FailsOnEmbeddingNotAtLayerZero)
{
    PPStageConfig config;
    config.stage_id = 1;
    config.domain_name = "test_domain";
    config.first_layer = 8; // Not zero!
    config.last_layer = 16;
    config.has_embedding = true; // But has embedding

    std::string error;
    EXPECT_FALSE(config.validate(24, &error));
    EXPECT_TRUE(error.find("has_embedding") != std::string::npos);
}

TEST(Test__PPStageConfig, Validate_FailsOnLMHeadNotAtLastLayer)
{
    PPStageConfig config;
    config.stage_id = 1;
    config.domain_name = "test_domain";
    config.first_layer = 8;
    config.last_layer = 16; // Not total_layers!
    config.has_lm_head = true; // But has LM head

    std::string error;
    EXPECT_FALSE(config.validate(24, &error));
    EXPECT_TRUE(error.find("has_lm_head") != std::string::npos);
}

// =============================================================================
// Validation Tests - Success
// =============================================================================

TEST(Test__PPStageConfig, Validate_SucceedsForValidConfig)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = "gpu_tp";
    config.first_layer = 0;
    config.last_layer = 24;
    config.has_embedding = true;
    config.has_lm_head = true;

    std::string error;
    EXPECT_TRUE(config.validate(24, &error));
    EXPECT_TRUE(error.empty());
}

TEST(Test__PPStageConfig, Validate_SucceedsForFirstStage)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = "gpu_nvidia";
    config.first_layer = 0;
    config.last_layer = 8;
    config.has_embedding = true;
    config.has_lm_head = false;

    std::string error;
    EXPECT_TRUE(config.validate(24, &error));
}

TEST(Test__PPStageConfig, Validate_SucceedsForMiddleStage)
{
    PPStageConfig config;
    config.stage_id = 1;
    config.domain_name = "gpu_amd";
    config.first_layer = 8;
    config.last_layer = 16;
    config.has_embedding = false;
    config.has_lm_head = false;

    std::string error;
    EXPECT_TRUE(config.validate(24, &error));
}

TEST(Test__PPStageConfig, Validate_SucceedsForLastStage)
{
    PPStageConfig config;
    config.stage_id = 2;
    config.domain_name = "cpu_tp";
    config.first_layer = 16;
    config.last_layer = 24;
    config.has_embedding = false;
    config.has_lm_head = true;

    std::string error;
    EXPECT_TRUE(config.validate(24, &error));
}

TEST(Test__PPStageConfig, Validate_NullErrorMsgWorks)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = "test";
    config.first_layer = 0;
    config.last_layer = 8;

    // Should not crash with nullptr
    EXPECT_TRUE(config.validate(24, nullptr));
}

// =============================================================================
// Factory Method Tests
// =============================================================================

TEST(Test__PPStageConfig, FullModel_CreatesCorrectConfig)
{
    PPStageConfig config = PPStageConfig::fullModel(24, "gpu_tp");

    EXPECT_EQ(config.stage_id, 0);
    EXPECT_EQ(config.domain_name, "gpu_tp");
    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 24);
    EXPECT_TRUE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
    EXPECT_EQ(config.numLayers(), 24);

    // Should validate
    std::string error;
    EXPECT_TRUE(config.validate(24, &error));
}

TEST(Test__PPStageConfig, FirstStage_SetsHasEmbedding)
{
    PPStageConfig config = PPStageConfig::firstStage(0, "nvidia_domain", 0, 8);

    EXPECT_EQ(config.stage_id, 0);
    EXPECT_EQ(config.domain_name, "nvidia_domain");
    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 8);
    EXPECT_TRUE(config.has_embedding);
    EXPECT_FALSE(config.has_lm_head);
    EXPECT_TRUE(config.isFirstStage());
    EXPECT_FALSE(config.isLastStage());

    // Should validate
    std::string error;
    EXPECT_TRUE(config.validate(24, &error));
}

TEST(Test__PPStageConfig, MiddleStage_NoEmbeddingNoLMHead)
{
    PPStageConfig config = PPStageConfig::middleStage(1, "amd_domain", 8, 16);

    EXPECT_EQ(config.stage_id, 1);
    EXPECT_EQ(config.domain_name, "amd_domain");
    EXPECT_EQ(config.first_layer, 8);
    EXPECT_EQ(config.last_layer, 16);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_FALSE(config.has_lm_head);
    EXPECT_FALSE(config.isFirstStage());
    EXPECT_FALSE(config.isLastStage());

    // Should validate
    std::string error;
    EXPECT_TRUE(config.validate(24, &error));
}

TEST(Test__PPStageConfig, LastStage_SetsHasLMHead)
{
    PPStageConfig config = PPStageConfig::lastStage(2, "cpu_domain", 16, 24);

    EXPECT_EQ(config.stage_id, 2);
    EXPECT_EQ(config.domain_name, "cpu_domain");
    EXPECT_EQ(config.first_layer, 16);
    EXPECT_EQ(config.last_layer, 24);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
    EXPECT_FALSE(config.isFirstStage());
    EXPECT_TRUE(config.isLastStage());

    // Should validate
    std::string error;
    EXPECT_TRUE(config.validate(24, &error));
}

// =============================================================================
// Integration: Factory Methods with Validation
// =============================================================================

TEST(Test__PPStageConfig, ThreeStagesPipeline_AllValidate)
{
    const int total_layers = 24;

    PPStageConfig stage0 = PPStageConfig::firstStage(0, "gpu_nvidia", 0, 8);
    PPStageConfig stage1 = PPStageConfig::middleStage(1, "gpu_amd", 8, 16);
    PPStageConfig stage2 = PPStageConfig::lastStage(2, "cpu_tp", 16, 24);

    std::string error;

    // All stages should validate
    EXPECT_TRUE(stage0.validate(total_layers, &error)) << "Stage 0 error: " << error;
    EXPECT_TRUE(stage1.validate(total_layers, &error)) << "Stage 1 error: " << error;
    EXPECT_TRUE(stage2.validate(total_layers, &error)) << "Stage 2 error: " << error;

    // Verify layer coverage is complete and non-overlapping
    EXPECT_EQ(stage0.first_layer, 0);
    EXPECT_EQ(stage0.last_layer, stage1.first_layer);
    EXPECT_EQ(stage1.last_layer, stage2.first_layer);
    EXPECT_EQ(stage2.last_layer, total_layers);

    // Verify total layer count
    EXPECT_EQ(stage0.numLayers() + stage1.numLayers() + stage2.numLayers(), total_layers);

    // Verify exactly one embedding and one LM head
    EXPECT_TRUE(stage0.has_embedding && !stage1.has_embedding && !stage2.has_embedding);
    EXPECT_TRUE(!stage0.has_lm_head && !stage1.has_lm_head && stage2.has_lm_head);
}
