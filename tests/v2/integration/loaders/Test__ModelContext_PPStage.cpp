/**
 * @file Test__ModelContext_PPStage.cpp
 * @brief Integration tests for ModelContext PP stage loading
 *
 * Validates that ModelContext with ModelContextConfig correctly loads
 * partial weights for Pipeline Parallelism stages.
 *
 * Test Coverage:
 * - PP stage 0 loads only first half of layers + embedding
 * - PP stage 1 loads only second half of layers + lm_head
 * - Global weights (token_embd, output_norm, output) are filtered correctly
 * - Memory is reduced compared to full model loading
 * - Multi-stage PP (3+ stages) divides layers correctly
 * - Explicit config (custom layer ranges) works correctly
 *
 * NOTE: ModelContextConfig uses INCLUSIVE last_layer. ModelContext::create()
 * converts to exclusive via `last_layer + 1` before passing to WeightManager.
 * The conversion is correct: config.last_layer=11 → setLayerRange(0, 12) → layers 0-11 loaded.
 */

#include <gtest/gtest.h>
#include "loaders/ModelContext.h"
#include "loaders/ModelContextConfig.h"
#include "utils/Logger.h"
#include <filesystem>

using namespace llaminar2;

// Test model path - uses the standard test model
static const std::string TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

class Test__ModelContext_PPStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Skip if model not available
        if (!std::filesystem::exists(TEST_MODEL_PATH))
        {
            GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
        }
    }
};

// Test 1: First PP stage (stage 0) loads embedding but not lm_head
TEST_F(Test__ModelContext_PPStage, FirstStage_LoadsEmbedding_NotLmHead)
{
    // Create config for first PP stage (2 stages total)
    // Qwen2.5-0.5b has 24 layers, so stage 0 gets layers 0-11 (inclusive)
    // forPPStage calculates: first=0, last=11 (inclusive)
    // ModelContext::create converts to exclusive: setLayerRange(0, 12) → layers 0-11 loaded
    auto config = ModelContextConfig::forPPStage(0, 2, 24);

    // Verify the config values
    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 11); // Inclusive in config
    EXPECT_TRUE(config.has_embedding);
    EXPECT_FALSE(config.has_lm_head);

    // Create model context
    auto ctx = ModelContext::create(TEST_MODEL_PATH, config);
    ASSERT_NE(ctx, nullptr) << "Failed to create ModelContext";

    auto weight_mgr = ctx->concreteWeightManager();
    ASSERT_NE(weight_mgr, nullptr);

    // Embedding should be loadable
    auto embedding = weight_mgr->getWeightForDevice("token_embd.weight");
    EXPECT_NE(embedding, nullptr) << "Stage 0 should have embedding";

    // LM head should NOT be loadable (filtered out)
    auto lm_head = weight_mgr->getWeightForDevice("output.weight");
    EXPECT_EQ(lm_head, nullptr) << "Stage 0 should NOT have lm_head";

    auto output_norm = weight_mgr->getWeightForDevice("output_norm.weight");
    EXPECT_EQ(output_norm, nullptr) << "Stage 0 should NOT have output_norm";

    // Layer 0 weights should be available
    auto layer0_q = weight_mgr->getWeightForDevice("blk.0.attn_q.weight");
    EXPECT_NE(layer0_q, nullptr) << "Stage 0 should have layer 0";

    // Layer 10 should be available
    auto layer10_q = weight_mgr->getWeightForDevice("blk.10.attn_q.weight");
    EXPECT_NE(layer10_q, nullptr) << "Stage 0 should have layer 10";

    // Layer 11 should be available (last_layer=11 is inclusive)
    auto layer11_q = weight_mgr->getWeightForDevice("blk.11.attn_q.weight");
    EXPECT_NE(layer11_q, nullptr) << "Stage 0 should have layer 11 (inclusive boundary)";

    // Layer 12 should NOT be available (outside range)
    auto layer12_q = weight_mgr->getWeightForDevice("blk.12.attn_q.weight");
    EXPECT_EQ(layer12_q, nullptr) << "Stage 0 should NOT have layer 12";
}

// Test 2: Last PP stage (stage 1) loads lm_head but not embedding
TEST_F(Test__ModelContext_PPStage, LastStage_LoadsLmHead_NotEmbedding)
{
    // Create config for last PP stage (2 stages total)
    // Stage 1 gets layers 12-23 (inclusive in config)
    // ModelContext::create converts to exclusive: setLayerRange(12, 24) → layers 12-23 loaded
    auto config = ModelContextConfig::forPPStage(1, 2, 24);

    // Verify the config values
    EXPECT_EQ(config.first_layer, 12);
    EXPECT_EQ(config.last_layer, 23); // Inclusive in config
    EXPECT_FALSE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);

    auto ctx = ModelContext::create(TEST_MODEL_PATH, config);
    ASSERT_NE(ctx, nullptr);

    auto weight_mgr = ctx->concreteWeightManager();

    // LM head should be loadable
    auto lm_head = weight_mgr->getWeightForDevice("output.weight");
    EXPECT_NE(lm_head, nullptr) << "Stage 1 should have lm_head";

    auto output_norm = weight_mgr->getWeightForDevice("output_norm.weight");
    EXPECT_NE(output_norm, nullptr) << "Stage 1 should have output_norm";

    // Embedding should NOT be loadable
    auto embedding = weight_mgr->getWeightForDevice("token_embd.weight");
    EXPECT_EQ(embedding, nullptr) << "Stage 1 should NOT have embedding";

    // Layer 12 should be available
    auto layer12_q = weight_mgr->getWeightForDevice("blk.12.attn_q.weight");
    EXPECT_NE(layer12_q, nullptr) << "Stage 1 should have layer 12";

    // Layer 22 should be available
    auto layer22_q = weight_mgr->getWeightForDevice("blk.22.attn_q.weight");
    EXPECT_NE(layer22_q, nullptr) << "Stage 1 should have layer 22";

    // Layer 23 should be available (last_layer=23 is inclusive)
    auto layer23_q = weight_mgr->getWeightForDevice("blk.23.attn_q.weight");
    EXPECT_NE(layer23_q, nullptr) << "Stage 1 should have layer 23 (inclusive boundary)";

    // Layer 11 should NOT be available
    auto layer11_q = weight_mgr->getWeightForDevice("blk.11.attn_q.weight");
    EXPECT_EQ(layer11_q, nullptr) << "Stage 1 should NOT have layer 11";
}

// Test 3: Full model loads everything
TEST_F(Test__ModelContext_PPStage, FullModel_LoadsEverything)
{
    auto config = ModelContextConfig::defaults();

    auto ctx = ModelContext::create(TEST_MODEL_PATH, config);
    ASSERT_NE(ctx, nullptr);

    auto weight_mgr = ctx->concreteWeightManager();

    // All global weights should be available
    EXPECT_NE(weight_mgr->getWeightForDevice("token_embd.weight"), nullptr);
    EXPECT_NE(weight_mgr->getWeightForDevice("output.weight"), nullptr);
    EXPECT_NE(weight_mgr->getWeightForDevice("output_norm.weight"), nullptr);

    // All layers should be available
    for (int i = 0; i < 24; ++i)
    {
        std::string name = "blk." + std::to_string(i) + ".attn_q.weight";
        EXPECT_NE(weight_mgr->getWeightForDevice(name), nullptr)
            << "Full model should have layer " << i;
    }
}

// Test 4: Three-stage PP divides correctly
TEST_F(Test__ModelContext_PPStage, ThreeStages_DividesCorrectly)
{
    // 24 layers / 3 stages = 8 layers each
    // Each stage loads all assigned layers (inclusive boundary)

    // Stage 0: layers 0-7 (inclusive)
    auto config0 = ModelContextConfig::forPPStage(0, 3, 24);
    EXPECT_EQ(config0.first_layer, 0);
    EXPECT_EQ(config0.last_layer, 7);

    auto ctx0 = ModelContext::create(TEST_MODEL_PATH, config0);
    ASSERT_NE(ctx0, nullptr);
    auto wm0 = ctx0->concreteWeightManager();

    EXPECT_NE(wm0->getWeightForDevice("token_embd.weight"), nullptr) << "Stage 0 has embedding";
    EXPECT_EQ(wm0->getWeightForDevice("output.weight"), nullptr) << "Stage 0 no lm_head";
    EXPECT_NE(wm0->getWeightForDevice("blk.0.attn_q.weight"), nullptr);
    EXPECT_NE(wm0->getWeightForDevice("blk.6.attn_q.weight"), nullptr);
    EXPECT_NE(wm0->getWeightForDevice("blk.7.attn_q.weight"), nullptr) << "Inclusive boundary";
    EXPECT_EQ(wm0->getWeightForDevice("blk.8.attn_q.weight"), nullptr);

    // Stage 1: layers 8-15 (inclusive)
    auto config1 = ModelContextConfig::forPPStage(1, 3, 24);
    EXPECT_EQ(config1.first_layer, 8);
    EXPECT_EQ(config1.last_layer, 15);

    auto ctx1 = ModelContext::create(TEST_MODEL_PATH, config1);
    ASSERT_NE(ctx1, nullptr);
    auto wm1 = ctx1->concreteWeightManager();

    EXPECT_EQ(wm1->getWeightForDevice("token_embd.weight"), nullptr) << "Stage 1 no embedding";
    EXPECT_EQ(wm1->getWeightForDevice("output.weight"), nullptr) << "Stage 1 no lm_head";
    EXPECT_EQ(wm1->getWeightForDevice("blk.7.attn_q.weight"), nullptr);
    EXPECT_NE(wm1->getWeightForDevice("blk.8.attn_q.weight"), nullptr);
    EXPECT_NE(wm1->getWeightForDevice("blk.14.attn_q.weight"), nullptr);
    EXPECT_NE(wm1->getWeightForDevice("blk.15.attn_q.weight"), nullptr) << "Inclusive boundary";
    EXPECT_EQ(wm1->getWeightForDevice("blk.16.attn_q.weight"), nullptr);

    // Stage 2: layers 16-23 (inclusive)
    auto config2 = ModelContextConfig::forPPStage(2, 3, 24);
    EXPECT_EQ(config2.first_layer, 16);
    EXPECT_EQ(config2.last_layer, 23);

    auto ctx2 = ModelContext::create(TEST_MODEL_PATH, config2);
    ASSERT_NE(ctx2, nullptr);
    auto wm2 = ctx2->concreteWeightManager();

    EXPECT_EQ(wm2->getWeightForDevice("token_embd.weight"), nullptr) << "Stage 2 no embedding";
    EXPECT_NE(wm2->getWeightForDevice("output.weight"), nullptr) << "Stage 2 has lm_head";
    EXPECT_EQ(wm2->getWeightForDevice("blk.15.attn_q.weight"), nullptr);
    EXPECT_NE(wm2->getWeightForDevice("blk.16.attn_q.weight"), nullptr);
    EXPECT_NE(wm2->getWeightForDevice("blk.22.attn_q.weight"), nullptr);
    EXPECT_NE(wm2->getWeightForDevice("blk.23.attn_q.weight"), nullptr) << "Inclusive boundary";
}

// Test 5: Explicit config (not using forPPStage helper)
TEST_F(Test__ModelContext_PPStage, ExplicitConfig_WorksCorrectly)
{
    ModelContextConfig config;
    config.first_layer = 5;
    config.last_layer = 10; // Inclusive in config, but WeightManager uses exclusive
    config.has_embedding = false;
    config.has_lm_head = false;
    config.strategy = WeightDistributionStrategy::LAYER_PARTITIONED;

    auto ctx = ModelContext::create(TEST_MODEL_PATH, config);
    ASSERT_NE(ctx, nullptr);

    auto wm = ctx->concreteWeightManager();

    // No global weights
    EXPECT_EQ(wm->getWeightForDevice("token_embd.weight"), nullptr);
    EXPECT_EQ(wm->getWeightForDevice("output.weight"), nullptr);

    // Layers 5-10 (inclusive boundary)
    EXPECT_EQ(wm->getWeightForDevice("blk.4.attn_q.weight"), nullptr);
    EXPECT_NE(wm->getWeightForDevice("blk.5.attn_q.weight"), nullptr);
    EXPECT_NE(wm->getWeightForDevice("blk.9.attn_q.weight"), nullptr);
    EXPECT_NE(wm->getWeightForDevice("blk.10.attn_q.weight"), nullptr) << "Inclusive boundary";
    EXPECT_EQ(wm->getWeightForDevice("blk.11.attn_q.weight"), nullptr);
}

// Test 6: Config validation for isLayerPartitioned
TEST_F(Test__ModelContext_PPStage, ConfigValidation_IsLayerPartitioned)
{
    // Default config is not layer partitioned
    auto defaults = ModelContextConfig::defaults();
    EXPECT_FALSE(defaults.isLayerPartitioned());

    // forPPStage creates layer partitioned config
    auto pp_config = ModelContextConfig::forPPStage(0, 2, 24);
    EXPECT_TRUE(pp_config.isLayerPartitioned());

    // Manual config with explicit range
    ModelContextConfig manual;
    manual.first_layer = 5;
    manual.last_layer = 10;
    manual.strategy = WeightDistributionStrategy::LAYER_PARTITIONED;
    EXPECT_TRUE(manual.isLayerPartitioned());
}

// Test 7: Verify config validation detects invalid configurations
TEST_F(Test__ModelContext_PPStage, ConfigValidation_DetectsInvalidRange)
{
    ModelContextConfig config;
    config.first_layer = 15;
    config.last_layer = 10; // Invalid: first > last
    config.strategy = WeightDistributionStrategy::LAYER_PARTITIONED;

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty()) << "Should detect invalid layer range";
}

// Test 8: Verify toString produces readable output
TEST_F(Test__ModelContext_PPStage, ConfigToString_ProducesReadableOutput)
{
    auto config = ModelContextConfig::forPPStage(1, 3, 24);
    std::string str = config.toString();

    // Should contain key information - check actual output format
    // Output format: "ModelContextConfig{layers=[8,15], emb=false, lm=false, shard=0/1}"
    EXPECT_NE(str.find("ModelContextConfig"), std::string::npos);
    EXPECT_NE(str.find("layers="), std::string::npos);
    EXPECT_NE(str.find("emb="), std::string::npos);
    EXPECT_NE(str.find("lm="), std::string::npos);
}

// Test 9: Memory reduction with PP (qualitative check)
TEST_F(Test__ModelContext_PPStage, PPStage_ReducesLoadedWeightCount)
{
    // Load full model and count weights
    auto full_config = ModelContextConfig::defaults();
    auto full_ctx = ModelContext::create(TEST_MODEL_PATH, full_config);
    ASSERT_NE(full_ctx, nullptr);
    auto full_wm = full_ctx->concreteWeightManager();

    // Load only stage 0 (half the layers)
    auto stage0_config = ModelContextConfig::forPPStage(0, 2, 24);
    auto stage0_ctx = ModelContext::create(TEST_MODEL_PATH, stage0_config);
    ASSERT_NE(stage0_ctx, nullptr);
    auto stage0_wm = stage0_ctx->concreteWeightManager();

    // Count layer weights that are loadable
    int full_layer_count = 0;
    int stage0_layer_count = 0;
    for (int i = 0; i < 24; ++i)
    {
        std::string name = "blk." + std::to_string(i) + ".attn_q.weight";
        if (full_wm->getWeightForDevice(name))
            full_layer_count++;
        if (stage0_wm->getWeightForDevice(name))
            stage0_layer_count++;
    }

    EXPECT_EQ(full_layer_count, 24) << "Full model should have all 24 layers";
    // Stage 0 gets 12 layers (0-11 inclusive)
    EXPECT_EQ(stage0_layer_count, 12) << "Stage 0 should have 12 layers (inclusive boundary)";
    EXPECT_LT(stage0_layer_count, full_layer_count) << "PP stage should load fewer layers";
}

// Test 10: Verifies inclusive last_layer semantics work correctly
// ModelContextConfig.last_layer is inclusive; ModelContext::create() converts to
// exclusive via +1 before passing to WeightManager::setLayerRange().
TEST_F(Test__ModelContext_PPStage, InclusiveLastLayerSemanticsAreCorrect)
{
    auto config = ModelContextConfig::forPPStage(0, 2, 24);

    // forPPStage sets last_layer = 11 (meaning layers 0-11 inclusive, 12 layers)
    EXPECT_EQ(config.last_layer, 11) << "Config says last_layer=11 (inclusive)";

    auto ctx = ModelContext::create(TEST_MODEL_PATH, config);
    ASSERT_NE(ctx, nullptr);
    auto wm = ctx->concreteWeightManager();

    // ModelContext::create converts inclusive→exclusive: setLayerRange(0, 12)
    // So layers 0-11 are all loaded correctly
    auto layer10 = wm->getWeightForDevice("blk.10.attn_q.weight");
    auto layer11 = wm->getWeightForDevice("blk.11.attn_q.weight");
    auto layer12 = wm->getWeightForDevice("blk.12.attn_q.weight");

    EXPECT_NE(layer10, nullptr) << "Layer 10 should be loaded";
    EXPECT_NE(layer11, nullptr) << "Layer 11 should be loaded (inclusive boundary)";
    EXPECT_EQ(layer12, nullptr) << "Layer 12 should NOT be loaded (outside range)";
}
