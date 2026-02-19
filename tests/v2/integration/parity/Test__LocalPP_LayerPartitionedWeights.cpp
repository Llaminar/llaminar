/**
 * @file Test__LocalPP_LayerPartitionedWeights.cpp
 * @brief Integration test for LocalPP with LAYER_PARTITIONED weight loading
 *
 * Verifies that Pipeline Parallelism stages only load their assigned layer weights,
 * achieving ~50% memory reduction for 2-stage PP.
 *
 * Tests:
 * - Stage 0 loads embedding + first half of layers
 * - Stage 1 loads second half of layers + LM head
 * - Each stage's WeightManager filters out-of-range weights
 * - Total weights across stages equals full model weights
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <fstream>

#include "loaders/ModelContext.h"
#include "loaders/WeightManager.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalPP_LayerPartitionedWeights : public ::testing::Test
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
     * @brief Count how many layer weights are loadable from a context
     */
    int countLoadableLayerWeights(std::shared_ptr<ModelContext> ctx, int num_layers)
    {
        int count = 0;
        for (int layer = 0; layer < num_layers; ++layer)
        {
            std::string name = "blk." + std::to_string(layer) + ".attn_q.weight";
            if (ctx->getWeightForDevice(name) != nullptr)
            {
                count++;
            }
        }
        return count;
    }

    /**
     * @brief Get set of loadable weight names from a context
     */
    std::set<std::string> getLoadableWeights(
        std::shared_ptr<ModelContext> ctx,
        const std::vector<std::string>& weight_names)
    {
        std::set<std::string> loadable;
        for (const auto& name : weight_names)
        {
            if (ctx->getWeightForDevice(name) != nullptr)
            {
                loadable.insert(name);
            }
        }
        return loadable;
    }
};

// =============================================================================
// 2-Stage PP Tests
// =============================================================================

/**
 * @test 2-stage PP: Stage 0 loads first half, Stage 1 loads second half
 */
TEST_F(Test__LocalPP_LayerPartitionedWeights, TwoStage_LayerPartitioning)
{
    // Get model metadata
    auto full_ctx = ModelContext::create(TEST_MODEL_PATH);
    ASSERT_NE(full_ctx, nullptr);
    int num_layers = full_ctx->blockCount();
    ASSERT_EQ(num_layers, 24) << "Qwen2.5-0.5B should have 24 layers";

    // Stage 0: layers [0, 12), embedding, no LM head
    auto stage0_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 0, 12, true, false);
    ASSERT_NE(stage0_ctx, nullptr);

    // Stage 1: layers [12, 24), no embedding, LM head
    auto stage1_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 12, 24, false, true);
    ASSERT_NE(stage1_ctx, nullptr);

    // Verify stage 0 layer counts
    int stage0_layers = countLoadableLayerWeights(stage0_ctx, num_layers);
    EXPECT_EQ(stage0_layers, 12) << "Stage 0 should have 12 layers";

    // Verify stage 1 layer counts
    int stage1_layers = countLoadableLayerWeights(stage1_ctx, num_layers);
    EXPECT_EQ(stage1_layers, 12) << "Stage 1 should have 12 layers";

    // Verify disjoint layer sets
    for (int layer = 0; layer < 12; ++layer)
    {
        std::string name = "blk." + std::to_string(layer) + ".attn_q.weight";
        EXPECT_NE(stage0_ctx->getWeightForDevice(name), nullptr)
            << "Stage 0 should have layer " << layer;
        EXPECT_EQ(stage1_ctx->getWeightForDevice(name), nullptr)
            << "Stage 1 should NOT have layer " << layer;
    }
    for (int layer = 12; layer < 24; ++layer)
    {
        std::string name = "blk." + std::to_string(layer) + ".attn_q.weight";
        EXPECT_EQ(stage0_ctx->getWeightForDevice(name), nullptr)
            << "Stage 0 should NOT have layer " << layer;
        EXPECT_NE(stage1_ctx->getWeightForDevice(name), nullptr)
            << "Stage 1 should have layer " << layer;
    }
}

/**
 * @test 2-stage PP: Special weights (embedding, norm, LM head) partitioning
 */
TEST_F(Test__LocalPP_LayerPartitionedWeights, TwoStage_SpecialWeights)
{
    // Stage 0: has embedding, no LM head
    auto stage0_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 0, 12, true, false);
    ASSERT_NE(stage0_ctx, nullptr);

    // Stage 1: no embedding, has LM head
    auto stage1_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 12, 24, false, true);
    ASSERT_NE(stage1_ctx, nullptr);

    // Embedding
    EXPECT_NE(stage0_ctx->getWeightForDevice("token_embd.weight"), nullptr)
        << "Stage 0 should have embedding";
    EXPECT_EQ(stage1_ctx->getWeightForDevice("token_embd.weight"), nullptr)
        << "Stage 1 should NOT have embedding";

    // Output norm
    EXPECT_EQ(stage0_ctx->getWeightForDevice("output_norm.weight"), nullptr)
        << "Stage 0 should NOT have output_norm";
    EXPECT_NE(stage1_ctx->getWeightForDevice("output_norm.weight"), nullptr)
        << "Stage 1 should have output_norm";

    // LM head
    EXPECT_EQ(stage0_ctx->getWeightForDevice("output.weight"), nullptr)
        << "Stage 0 should NOT have LM head";
    EXPECT_NE(stage1_ctx->getWeightForDevice("output.weight"), nullptr)
        << "Stage 1 should have LM head";
}

/**
 * @test Union of stage weights equals full model weights
 */
TEST_F(Test__LocalPP_LayerPartitionedWeights, TwoStage_UnionEqualsFullModel)
{
    // Get full model weights
    auto full_ctx = ModelContext::create(TEST_MODEL_PATH);
    ASSERT_NE(full_ctx, nullptr);

    // Build list of test weights
    std::vector<std::string> test_weights = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight"
    };
    
    int num_layers = full_ctx->blockCount();
    for (int layer = 0; layer < num_layers; ++layer)
    {
        test_weights.push_back("blk." + std::to_string(layer) + ".attn_q.weight");
        test_weights.push_back("blk." + std::to_string(layer) + ".ffn_gate.weight");
    }

    // Stage contexts
    auto stage0_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 0, 12, true, false);
    auto stage1_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 12, 24, false, true);

    // Get loadable weights from each
    auto full_weights = getLoadableWeights(full_ctx, test_weights);
    auto stage0_weights = getLoadableWeights(stage0_ctx, test_weights);
    auto stage1_weights = getLoadableWeights(stage1_ctx, test_weights);

    // Union of stage weights
    std::set<std::string> union_weights;
    union_weights.insert(stage0_weights.begin(), stage0_weights.end());
    union_weights.insert(stage1_weights.begin(), stage1_weights.end());

    // Verify union equals full
    EXPECT_EQ(union_weights, full_weights)
        << "Union of stage weights should equal full model weights";

    // Verify intersection is empty (no overlap)
    std::set<std::string> intersection;
    for (const auto& w : stage0_weights)
    {
        if (stage1_weights.count(w))
        {
            intersection.insert(w);
        }
    }
    EXPECT_TRUE(intersection.empty())
        << "Stage weights should not overlap";
}

// =============================================================================
// 3-Stage PP Tests
// =============================================================================

/**
 * @test 3-stage PP: Equal layer distribution
 */
TEST_F(Test__LocalPP_LayerPartitionedWeights, ThreeStage_LayerPartitioning)
{
    // 24 layers / 3 stages = 8 layers per stage
    auto stage0_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 0, 8, true, false);
    auto stage1_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 8, 16, false, false);
    auto stage2_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 16, 24, false, true);

    ASSERT_NE(stage0_ctx, nullptr);
    ASSERT_NE(stage1_ctx, nullptr);
    ASSERT_NE(stage2_ctx, nullptr);

    // Verify layer counts
    int stage0_layers = countLoadableLayerWeights(stage0_ctx, 24);
    int stage1_layers = countLoadableLayerWeights(stage1_ctx, 24);
    int stage2_layers = countLoadableLayerWeights(stage2_ctx, 24);

    EXPECT_EQ(stage0_layers, 8) << "Stage 0: layers 0-7";
    EXPECT_EQ(stage1_layers, 8) << "Stage 1: layers 8-15";
    EXPECT_EQ(stage2_layers, 8) << "Stage 2: layers 16-23";

    // Verify special weights
    EXPECT_NE(stage0_ctx->getWeightForDevice("token_embd.weight"), nullptr);
    EXPECT_EQ(stage1_ctx->getWeightForDevice("token_embd.weight"), nullptr);
    EXPECT_EQ(stage2_ctx->getWeightForDevice("token_embd.weight"), nullptr);

    EXPECT_EQ(stage0_ctx->getWeightForDevice("output.weight"), nullptr);
    EXPECT_EQ(stage1_ctx->getWeightForDevice("output.weight"), nullptr);
    EXPECT_NE(stage2_ctx->getWeightForDevice("output.weight"), nullptr);
}

// =============================================================================
// Strategy Verification Tests
// =============================================================================

/**
 * @test Strategy is LAYER_PARTITIONED for PP stage contexts
 */
TEST_F(Test__LocalPP_LayerPartitionedWeights, Strategy_IsLayerPartitioned)
{
    auto stage_ctx = ModelContext::createForPPStage(
        TEST_MODEL_PATH, 0, 12, true, false);
    ASSERT_NE(stage_ctx, nullptr);

    auto wm = stage_ctx->concreteWeightManager();
    EXPECT_EQ(wm->strategy(), WeightDistributionStrategy::LAYER_PARTITIONED);
}

/**
 * @test Regular ModelContext::create uses REPLICATED strategy
 */
TEST_F(Test__LocalPP_LayerPartitionedWeights, RegularContext_UsesReplicated)
{
    auto ctx = ModelContext::create(TEST_MODEL_PATH);
    ASSERT_NE(ctx, nullptr);

    auto wm = ctx->concreteWeightManager();
    EXPECT_EQ(wm->strategy(), WeightDistributionStrategy::REPLICATED);
}

