/**
 * @file Test__WeightManager_LayerPartitioned.cpp
 * @brief Unit tests for WeightManager LAYER_PARTITIONED strategy
 *
 * Tests verify:
 * - Layer range filtering works correctly
 * - Embedding weights are filtered based on has_embedding flag
 * - LM head weights are filtered based on has_lm_head flag
 * - Layer weights outside range return nullptr
 * - Layer weights inside range are loaded correctly
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "loaders/WeightManager.h"
#include "loaders/ModelContext.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__WeightManager_LayerPartitioned : public ::testing::Test
{
protected:
    static constexpr const char* TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    void SetUp() override
    {
        // Check model exists
        std::ifstream f(TEST_MODEL_PATH);
        if (!f.good())
        {
            GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
        }
    }

    /**
     * @brief Create a ModelContext for a PP stage with layer filtering
     */
    std::shared_ptr<ModelContext> createPPStageContext(
        int first_layer, int last_layer,
        bool has_embedding, bool has_lm_head)
    {
        return ModelContext::createForPPStage(
            TEST_MODEL_PATH,
            first_layer, last_layer,
            has_embedding, has_lm_head);
    }
};

// =============================================================================
// isWeightInLayerRange Tests
// =============================================================================

/**
 * @test Token embedding weight filtered based on has_embedding flag
 */
TEST_F(Test__WeightManager_LayerPartitioned, EmbeddingWeight_FilteredByFlag)
{
    // Stage 0: has embedding
    auto ctx0 = createPPStageContext(0, 12, true, false);
    ASSERT_NE(ctx0, nullptr);
    auto emb0 = ctx0->getWeight("token_embd.weight");
    EXPECT_NE(emb0, nullptr) << "Stage 0 should have embedding";

    // Stage 1: no embedding
    auto ctx1 = createPPStageContext(12, 24, false, true);
    ASSERT_NE(ctx1, nullptr);
    auto emb1 = ctx1->getWeight("token_embd.weight");
    EXPECT_EQ(emb1, nullptr) << "Stage 1 should NOT have embedding";
}

/**
 * @test Output norm weight filtered based on has_lm_head flag
 */
TEST_F(Test__WeightManager_LayerPartitioned, OutputNormWeight_FilteredByFlag)
{
    // Stage 0: no LM head
    auto ctx0 = createPPStageContext(0, 12, true, false);
    ASSERT_NE(ctx0, nullptr);
    auto norm0 = ctx0->getWeight("output_norm.weight");
    EXPECT_EQ(norm0, nullptr) << "Stage 0 should NOT have output_norm";

    // Stage 1: has LM head
    auto ctx1 = createPPStageContext(12, 24, false, true);
    ASSERT_NE(ctx1, nullptr);
    auto norm1 = ctx1->getWeight("output_norm.weight");
    EXPECT_NE(norm1, nullptr) << "Stage 1 should have output_norm";
}

/**
 * @test LM head weight filtered based on has_lm_head flag
 */
TEST_F(Test__WeightManager_LayerPartitioned, LMHeadWeight_FilteredByFlag)
{
    // Stage 0: no LM head
    auto ctx0 = createPPStageContext(0, 12, true, false);
    ASSERT_NE(ctx0, nullptr);
    auto head0 = ctx0->getWeight("output.weight");
    EXPECT_EQ(head0, nullptr) << "Stage 0 should NOT have output.weight";

    // Stage 1: has LM head
    auto ctx1 = createPPStageContext(12, 24, false, true);
    ASSERT_NE(ctx1, nullptr);
    auto head1 = ctx1->getWeight("output.weight");
    EXPECT_NE(head1, nullptr) << "Stage 1 should have output.weight";
}

/**
 * @test Layer weights outside range return nullptr
 */
TEST_F(Test__WeightManager_LayerPartitioned, LayerWeight_OutsideRange_ReturnsNullptr)
{
    // Stage 0: layers [0, 12)
    auto ctx0 = createPPStageContext(0, 12, true, false);
    ASSERT_NE(ctx0, nullptr);

    // Layer 0 attention Q should be loaded
    auto layer0_q = ctx0->getWeight("blk.0.attn_q.weight");
    EXPECT_NE(layer0_q, nullptr) << "Layer 0 should be in range";

    // Layer 11 should also be loaded (last layer in range)
    auto layer11_q = ctx0->getWeight("blk.11.attn_q.weight");
    EXPECT_NE(layer11_q, nullptr) << "Layer 11 should be in range";

    // Layer 12 should NOT be loaded (outside range)
    auto layer12_q = ctx0->getWeight("blk.12.attn_q.weight");
    EXPECT_EQ(layer12_q, nullptr) << "Layer 12 should NOT be in range for stage 0";

    // Layer 23 should NOT be loaded
    auto layer23_q = ctx0->getWeight("blk.23.attn_q.weight");
    EXPECT_EQ(layer23_q, nullptr) << "Layer 23 should NOT be in range for stage 0";
}

/**
 * @test Layer weights inside range are loaded correctly
 */
TEST_F(Test__WeightManager_LayerPartitioned, LayerWeight_InsideRange_Loaded)
{
    // Stage 1: layers [12, 24)
    auto ctx1 = createPPStageContext(12, 24, false, true);
    ASSERT_NE(ctx1, nullptr);

    // Layer 0 should NOT be loaded (before range)
    auto layer0_q = ctx1->getWeight("blk.0.attn_q.weight");
    EXPECT_EQ(layer0_q, nullptr) << "Layer 0 should NOT be in range for stage 1";

    // Layer 11 should NOT be loaded (before range)
    auto layer11_q = ctx1->getWeight("blk.11.attn_q.weight");
    EXPECT_EQ(layer11_q, nullptr) << "Layer 11 should NOT be in range for stage 1";

    // Layer 12 should be loaded (first in range)
    auto layer12_q = ctx1->getWeight("blk.12.attn_q.weight");
    EXPECT_NE(layer12_q, nullptr) << "Layer 12 should be in range for stage 1";

    // Layer 23 should be loaded (last in range)
    auto layer23_q = ctx1->getWeight("blk.23.attn_q.weight");
    EXPECT_NE(layer23_q, nullptr) << "Layer 23 should be in range for stage 1";
}

/**
 * @test All attention weights for a layer are loaded when in range
 */
TEST_F(Test__WeightManager_LayerPartitioned, AllAttentionWeights_InsideRange_Loaded)
{
    auto ctx = createPPStageContext(0, 12, true, false);
    ASSERT_NE(ctx, nullptr);

    // Check all attention weights for layer 5
    EXPECT_NE(ctx->getWeight("blk.5.attn_q.weight"), nullptr);
    EXPECT_NE(ctx->getWeight("blk.5.attn_k.weight"), nullptr);
    EXPECT_NE(ctx->getWeight("blk.5.attn_v.weight"), nullptr);
    EXPECT_NE(ctx->getWeight("blk.5.attn_output.weight"), nullptr);
    EXPECT_NE(ctx->getWeight("blk.5.attn_norm.weight"), nullptr);
}

/**
 * @test All FFN weights for a layer are loaded when in range
 */
TEST_F(Test__WeightManager_LayerPartitioned, AllFFNWeights_InsideRange_Loaded)
{
    auto ctx = createPPStageContext(0, 12, true, false);
    ASSERT_NE(ctx, nullptr);

    // Check all FFN weights for layer 5
    EXPECT_NE(ctx->getWeight("blk.5.ffn_gate.weight"), nullptr);
    EXPECT_NE(ctx->getWeight("blk.5.ffn_up.weight"), nullptr);
    EXPECT_NE(ctx->getWeight("blk.5.ffn_down.weight"), nullptr);
    EXPECT_NE(ctx->getWeight("blk.5.ffn_norm.weight"), nullptr);
}

/**
 * @test WeightManager reports layer range correctly
 */
TEST_F(Test__WeightManager_LayerPartitioned, WeightManager_ReportsLayerRange)
{
    auto ctx = createPPStageContext(5, 15, false, false);
    ASSERT_NE(ctx, nullptr);

    auto wm = ctx->concreteWeightManager();
    ASSERT_NE(wm, nullptr);

    EXPECT_TRUE(wm->hasLayerRange());
    auto [first, last] = wm->layerRange();
    EXPECT_EQ(first, 5);
    EXPECT_EQ(last, 15);
    EXPECT_FALSE(wm->hasEmbedding());
    EXPECT_FALSE(wm->hasLMHead());
}

/**
 * @test Strategy is LAYER_PARTITIONED when created with createForPPStage
 */
TEST_F(Test__WeightManager_LayerPartitioned, Strategy_IsLayerPartitioned)
{
    auto ctx = createPPStageContext(0, 12, true, false);
    ASSERT_NE(ctx, nullptr);

    auto wm = ctx->concreteWeightManager();
    ASSERT_NE(wm, nullptr);

    EXPECT_EQ(wm->strategy(), WeightDistributionStrategy::LAYER_PARTITIONED);
}

// =============================================================================
// Memory Savings Tests
// =============================================================================

/**
 * @test Stage 0 context loads fewer weights than full model
 *
 * This verifies the memory savings from layer partitioning.
 */
TEST_F(Test__WeightManager_LayerPartitioned, Stage0_LoadsFewerWeights)
{
    // Full model context
    auto full_ctx = ModelContext::create(TEST_MODEL_PATH);
    ASSERT_NE(full_ctx, nullptr);

    // Stage 0: first half
    auto stage0_ctx = createPPStageContext(0, 12, true, false);
    ASSERT_NE(stage0_ctx, nullptr);

    // Count accessible weights
    int full_count = 0;
    int stage0_count = 0;

    // Test a sampling of weights
    std::vector<std::string> test_weights = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.5.attn_q.weight",
        "blk.11.attn_q.weight",
        "blk.12.attn_q.weight",
        "blk.20.attn_q.weight",
        "blk.23.attn_q.weight"
    };

    for (const auto& name : test_weights)
    {
        if (full_ctx->getWeight(name) != nullptr) full_count++;
        if (stage0_ctx->getWeight(name) != nullptr) stage0_count++;
    }

    // Stage 0 should load embedding + layers 0-11
    // But NOT output_norm, output, or layers 12-23
    EXPECT_LT(stage0_count, full_count)
        << "Stage 0 should load fewer weights than full model";

    // Specifically:
    // - token_embd.weight: yes
    // - output_norm.weight: no
    // - output.weight: no
    // - blk.0-11: yes (3 weights in test)
    // - blk.12-23: no (3 weights in test)
    // Expected: 4 weights for stage 0, 9 for full
    EXPECT_EQ(stage0_count, 4) << "Stage 0: embedding + 3 layer weights in range";
    EXPECT_EQ(full_count, 9) << "Full model: all 9 test weights";
}

/**
 * @test Stage 1 context loads only second half weights
 */
TEST_F(Test__WeightManager_LayerPartitioned, Stage1_LoadsSecondHalf)
{
    // Stage 1: second half
    auto stage1_ctx = createPPStageContext(12, 24, false, true);
    ASSERT_NE(stage1_ctx, nullptr);

    // Check what's loaded
    EXPECT_EQ(stage1_ctx->getWeight("token_embd.weight"), nullptr)
        << "Stage 1 should NOT have embedding";
    EXPECT_NE(stage1_ctx->getWeight("output_norm.weight"), nullptr)
        << "Stage 1 should have output_norm";
    EXPECT_NE(stage1_ctx->getWeight("output.weight"), nullptr)
        << "Stage 1 should have LM head";

    // Layer 11 (before range)
    EXPECT_EQ(stage1_ctx->getWeight("blk.11.attn_q.weight"), nullptr)
        << "Layer 11 should NOT be in stage 1";

    // Layer 12 (first in range)
    EXPECT_NE(stage1_ctx->getWeight("blk.12.attn_q.weight"), nullptr)
        << "Layer 12 should be in stage 1";
}

