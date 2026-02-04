/**
 * @file Test__ModelContextConfig.cpp
 * @brief Unit tests for ModelContextConfig struct
 *
 * Tests configuration creation helpers, validation, and field accessors
 * for various parallelism scenarios (PP, TP, combined).
 */

#include <gtest/gtest.h>
#include "loaders/ModelContextConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"

using namespace llaminar2;

class Test__ModelContextConfig : public ::testing::Test {};

// =============================================================================
// defaults() Tests
// =============================================================================

TEST(Test__ModelContextConfig, Defaults_ReturnsFullModelConfig)
{
    auto config = ModelContextConfig::defaults();

    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, -1);
    EXPECT_TRUE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
    EXPECT_EQ(config.shard_index, 0);
    EXPECT_EQ(config.total_shards, 1);
    EXPECT_FALSE(config.isLayerPartitioned());
    EXPECT_FALSE(config.isSharded());
}

// =============================================================================
// forPPStage() Tests
// =============================================================================

TEST(Test__ModelContextConfig, ForPPStage_FirstOfTwo_HasEmbedding)
{
    auto config = ModelContextConfig::forPPStage(0, 2, 24);

    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 11); // 0-11 = 12 layers
    EXPECT_TRUE(config.has_embedding);
    EXPECT_FALSE(config.has_lm_head);
    EXPECT_TRUE(config.isLayerPartitioned());
    // Note: forPPStage uses defaults() which sets REPLICATED strategy
    // The strategy indicates weight distribution, not layer partitioning
    EXPECT_EQ(config.strategy, WeightDistributionStrategy::REPLICATED);
}

TEST(Test__ModelContextConfig, ForPPStage_LastOfTwo_HasLmHead)
{
    auto config = ModelContextConfig::forPPStage(1, 2, 24);

    EXPECT_EQ(config.first_layer, 12);
    EXPECT_EQ(config.last_layer, 23);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
}

TEST(Test__ModelContextConfig, ForPPStage_ThreeStages_CorrectLayerRanges)
{
    // 24 layers / 3 stages = 8 layers each
    auto stage0 = ModelContextConfig::forPPStage(0, 3, 24);
    auto stage1 = ModelContextConfig::forPPStage(1, 3, 24);
    auto stage2 = ModelContextConfig::forPPStage(2, 3, 24);

    // Stage 0: layers 0-7
    EXPECT_EQ(stage0.first_layer, 0);
    EXPECT_EQ(stage0.last_layer, 7);
    EXPECT_TRUE(stage0.has_embedding);
    EXPECT_FALSE(stage0.has_lm_head);

    // Stage 1: layers 8-15
    EXPECT_EQ(stage1.first_layer, 8);
    EXPECT_EQ(stage1.last_layer, 15);
    EXPECT_FALSE(stage1.has_embedding);
    EXPECT_FALSE(stage1.has_lm_head);

    // Stage 2: layers 16-23
    EXPECT_EQ(stage2.first_layer, 16);
    EXPECT_EQ(stage2.last_layer, 23);
    EXPECT_FALSE(stage2.has_embedding);
    EXPECT_TRUE(stage2.has_lm_head);
}

TEST(Test__ModelContextConfig, ForPPStage_UnevenDivision_HandlesRemainder)
{
    // 25 layers / 3 stages: first 2 get 9, last gets 7 (or similar distribution)
    auto stage0 = ModelContextConfig::forPPStage(0, 3, 25);
    auto stage1 = ModelContextConfig::forPPStage(1, 3, 25);
    auto stage2 = ModelContextConfig::forPPStage(2, 3, 25);

    // Verify all layers covered without overlap
    int total_layers = (stage0.last_layer - stage0.first_layer + 1) +
                       (stage1.last_layer - stage1.first_layer + 1) +
                       (stage2.last_layer - stage2.first_layer + 1);
    EXPECT_EQ(total_layers, 25);

    // Verify no gaps
    EXPECT_EQ(stage1.first_layer, stage0.last_layer + 1);
    EXPECT_EQ(stage2.first_layer, stage1.last_layer + 1);
    EXPECT_EQ(stage2.last_layer, 24); // Last layer is 24 (0-indexed)
}

// =============================================================================
// forTPShard() Tests
// =============================================================================

TEST(Test__ModelContextConfig, ForTPShard_SetsShardInfo)
{
    auto config = ModelContextConfig::forTPShard(1, 4);

    EXPECT_EQ(config.shard_index, 1);
    EXPECT_EQ(config.total_shards, 4);
    EXPECT_FLOAT_EQ(config.work_fraction, 0.25f);
    EXPECT_TRUE(config.isSharded());
    EXPECT_FALSE(config.isLayerPartitioned());
    EXPECT_EQ(config.strategy, WeightDistributionStrategy::SHARDED);
}

// =============================================================================
// validate() Tests
// =============================================================================

TEST(Test__ModelContextConfig, Validate_ValidConfig_ReturnsEmpty)
{
    auto config = ModelContextConfig::defaults();
    auto errors = config.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(Test__ModelContextConfig, Validate_InvalidShardIndex_ReturnsError)
{
    auto config = ModelContextConfig::defaults();
    config.shard_index = 5;
    config.total_shards = 4;

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("shard_index") != std::string::npos);
}

TEST(Test__ModelContextConfig, Validate_InvalidLayerRange_ReturnsError)
{
    auto config = ModelContextConfig::defaults();
    config.first_layer = 10;
    config.last_layer = 5;

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
}

// =============================================================================
// fromExecutionPlan() Tests
// =============================================================================

TEST(Test__ModelContextConfig, FromExecutionPlan_CopiesAllFields)
{
    RankExecutionPlan plan;
    plan.first_layer = 12;
    plan.last_layer = 23;
    plan.has_embedding = false;
    plan.has_lm_head = true;
    plan.weight_shard.shard_index = 1;
    plan.weight_shard.total_shards = 2;
    plan.weight_shard.work_fraction = 0.6f;

    auto config = ModelContextConfig::fromExecutionPlan(plan);

    EXPECT_EQ(config.first_layer, 12);
    EXPECT_EQ(config.last_layer, 23);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
    EXPECT_EQ(config.shard_index, 1);
    EXPECT_EQ(config.total_shards, 2);
    EXPECT_FLOAT_EQ(config.work_fraction, 0.6f);
    EXPECT_TRUE(config.isLayerPartitioned());
    EXPECT_TRUE(config.isSharded());
}

// =============================================================================
// toString() Tests
// =============================================================================

TEST(Test__ModelContextConfig, ToString_ContainsKeyInfo)
{
    auto config = ModelContextConfig::forPPStage(0, 2, 24);
    auto str = config.toString();

    EXPECT_TRUE(str.find("layers") != std::string::npos ||
                str.find("layer") != std::string::npos ||
                str.find("Layer") != std::string::npos);
    EXPECT_TRUE(str.find("0") != std::string::npos);  // first_layer
    EXPECT_TRUE(str.find("11") != std::string::npos); // last_layer
}

// =============================================================================
// isLayerPartitioned() Tests
// =============================================================================

TEST(Test__ModelContextConfig, IsLayerPartitioned_DetectsPartitioning)
{
    // Default is not partitioned
    auto config1 = ModelContextConfig::defaults();
    EXPECT_FALSE(config1.isLayerPartitioned());

    // Non-zero first_layer means partitioned
    auto config2 = ModelContextConfig::defaults();
    config2.first_layer = 5;
    EXPECT_TRUE(config2.isLayerPartitioned());

    // Non -1 last_layer means partitioned
    auto config3 = ModelContextConfig::defaults();
    config3.last_layer = 10;
    EXPECT_TRUE(config3.isLayerPartitioned());

    // Missing embedding means partitioned
    auto config4 = ModelContextConfig::defaults();
    config4.has_embedding = false;
    EXPECT_TRUE(config4.isLayerPartitioned());

    // Missing lm_head means partitioned
    auto config5 = ModelContextConfig::defaults();
    config5.has_lm_head = false;
    EXPECT_TRUE(config5.isLayerPartitioned());
}

// =============================================================================
// isSharded() Tests
// =============================================================================

TEST(Test__ModelContextConfig, IsSharded_DetectsSharding)
{
    // Default is not sharded
    auto config1 = ModelContextConfig::defaults();
    EXPECT_FALSE(config1.isSharded());

    // total_shards > 1 means sharded
    auto config2 = ModelContextConfig::defaults();
    config2.total_shards = 2;
    EXPECT_TRUE(config2.isSharded());

    // Single shard is not sharded
    auto config3 = ModelContextConfig::defaults();
    config3.total_shards = 1;
    config3.shard_index = 0;
    EXPECT_FALSE(config3.isSharded());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(Test__ModelContextConfig, ForPPStage_SingleStage_ReturnsFullModel)
{
    auto config = ModelContextConfig::forPPStage(0, 1, 24);

    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 23);
    EXPECT_TRUE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
}

TEST(Test__ModelContextConfig, ForTPShard_SingleShard_ReturnsUnsharded)
{
    auto config = ModelContextConfig::forTPShard(0, 1);

    EXPECT_EQ(config.shard_index, 0);
    EXPECT_EQ(config.total_shards, 1);
    EXPECT_FLOAT_EQ(config.work_fraction, 1.0f);
    EXPECT_FALSE(config.isSharded());
}

TEST(Test__ModelContextConfig, ForPPStage_FourStagesSmallModel_CorrectDistribution)
{
    // 8 layers / 4 stages = 2 layers each
    auto stage0 = ModelContextConfig::forPPStage(0, 4, 8);
    auto stage1 = ModelContextConfig::forPPStage(1, 4, 8);
    auto stage2 = ModelContextConfig::forPPStage(2, 4, 8);
    auto stage3 = ModelContextConfig::forPPStage(3, 4, 8);

    // Verify layer coverage
    EXPECT_EQ(stage0.first_layer, 0);
    EXPECT_EQ(stage1.first_layer, stage0.last_layer + 1);
    EXPECT_EQ(stage2.first_layer, stage1.last_layer + 1);
    EXPECT_EQ(stage3.first_layer, stage2.last_layer + 1);
    EXPECT_EQ(stage3.last_layer, 7);

    // Verify embedding/lm_head
    EXPECT_TRUE(stage0.has_embedding);
    EXPECT_FALSE(stage0.has_lm_head);
    EXPECT_FALSE(stage1.has_embedding);
    EXPECT_FALSE(stage1.has_lm_head);
    EXPECT_FALSE(stage2.has_embedding);
    EXPECT_FALSE(stage2.has_lm_head);
    EXPECT_FALSE(stage3.has_embedding);
    EXPECT_TRUE(stage3.has_lm_head);
}
