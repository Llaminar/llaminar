/**
 * @file Test__PartialWeightLoader.cpp
 * @brief Unit tests for IPartialWeightLoader
 *
 * Part of Phase 3: Pipeline Parallelism Integration
 *
 * Tests verify:
 * - weightsForLayerRange for full model
 * - weightsForLayerRange for first half (embedding, no lm_head)
 * - weightsForLayerRange for second half (lm_head, no embedding)
 * - Memory estimation
 * - Weight name generation patterns
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <algorithm>

#include "loaders/IPartialWeightLoader.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__PartialWeightLoader : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create loader with Qwen2-7B config
        model_config_ = ModelConfig::qwen2_7b();
        loader_ = createPartialWeightLoader(&model_config_);
    }

    /**
     * @brief Create execution plan for a PP stage
     */
    RankExecutionPlan createPlan(int rank, int first_layer, int last_layer,
                                 bool has_embedding, bool has_lm_head)
    {
        RankExecutionPlan plan;
        plan.rank = rank;
        plan.hostname = "localhost";
        plan.numa_node = 0;
        plan.pp_stage_id = rank;
        plan.first_layer = first_layer;
        plan.last_layer = last_layer;
        plan.has_embedding = has_embedding;
        plan.has_lm_head = has_lm_head;
        plan.prev_rank = (rank > 0) ? std::optional<int>(rank - 1) : std::nullopt;
        plan.next_rank = has_lm_head ? std::nullopt : std::optional<int>(rank + 1);
        plan.primary_device = GlobalDeviceAddress::cpu();
        return plan;
    }

    /**
     * @brief Check if a weight name is in the list
     */
    bool hasWeight(const std::vector<std::string> &weights, const std::string &name)
    {
        return std::find(weights.begin(), weights.end(), name) != weights.end();
    }

    ModelConfig model_config_;
    std::unique_ptr<IPartialWeightLoader> loader_;
};

// =============================================================================
// weightsForLayerRange Tests
// =============================================================================

/**
 * @test weightsForLayerRange includes embedding when requested
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_IncludesEmbedding)
{
    auto weights = loader_->weightsForLayerRange(0, 4, true, false);

    EXPECT_TRUE(hasWeight(weights, "token_embd.weight"));
    EXPECT_FALSE(hasWeight(weights, "output_norm.weight"));
    EXPECT_FALSE(hasWeight(weights, "output.weight"));
}

/**
 * @test weightsForLayerRange includes lm_head when requested
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_IncludesLMHead)
{
    auto weights = loader_->weightsForLayerRange(24, 27, false, true);

    EXPECT_FALSE(hasWeight(weights, "token_embd.weight"));
    EXPECT_TRUE(hasWeight(weights, "output_norm.weight"));
    EXPECT_TRUE(hasWeight(weights, "output.weight"));
}

/**
 * @test weightsForLayerRange returns correct layer weights
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_ContainsLayerWeights)
{
    auto weights = loader_->weightsForLayerRange(5, 7, false, false);

    // Should have weights for layers 5, 6, 7 (3 layers)
    // Each layer has 9 weights: 5 attention + 4 FFN

    // Layer 5 attention
    EXPECT_TRUE(hasWeight(weights, "blk.5.attn_q.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.5.attn_k.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.5.attn_v.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.5.attn_output.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.5.attn_norm.weight"));

    // Layer 5 FFN
    EXPECT_TRUE(hasWeight(weights, "blk.5.ffn_gate.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.5.ffn_up.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.5.ffn_down.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.5.ffn_norm.weight"));

    // Layer 7 (should be included)
    EXPECT_TRUE(hasWeight(weights, "blk.7.attn_q.weight"));

    // Layer 4 (should NOT be included)
    EXPECT_FALSE(hasWeight(weights, "blk.4.attn_q.weight"));

    // Layer 8 (should NOT be included)
    EXPECT_FALSE(hasWeight(weights, "blk.8.attn_q.weight"));
}

/**
 * @test weightsForLayerRange returns correct count for 3 layers
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_CorrectCount)
{
    // 3 layers, no embedding, no lm_head
    auto weights = loader_->weightsForLayerRange(5, 7, false, false);

    // 3 layers * 9 weights = 27
    EXPECT_EQ(weights.size(), 27);
}

/**
 * @test weightsForLayerRange for full model
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_FullModel)
{
    // Qwen2-7B has 28 layers
    auto weights = loader_->weightsForLayerRange(0, 27, true, true);

    // 28 layers * 9 weights + embedding + output_norm + output = 254
    EXPECT_EQ(weights.size(), 28 * 9 + 3);

    // First and last items
    EXPECT_TRUE(hasWeight(weights, "token_embd.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.0.attn_q.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.27.ffn_norm.weight"));
    EXPECT_TRUE(hasWeight(weights, "output.weight"));
}

/**
 * @test weightsForLayerRange for first half of model (embedding, no lm_head)
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_FirstHalf)
{
    // First 14 layers (0-13)
    auto weights = loader_->weightsForLayerRange(0, 13, true, false);

    // 14 layers * 9 + 1 embedding = 127
    EXPECT_EQ(weights.size(), 14 * 9 + 1);

    EXPECT_TRUE(hasWeight(weights, "token_embd.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.13.attn_q.weight"));
    EXPECT_FALSE(hasWeight(weights, "blk.14.attn_q.weight"));
    EXPECT_FALSE(hasWeight(weights, "output.weight"));
}

/**
 * @test weightsForLayerRange for second half of model (lm_head, no embedding)
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_SecondHalf)
{
    // Last 14 layers (14-27)
    auto weights = loader_->weightsForLayerRange(14, 27, false, true);

    // 14 layers * 9 + output_norm + output = 128
    EXPECT_EQ(weights.size(), 14 * 9 + 2);

    EXPECT_FALSE(hasWeight(weights, "token_embd.weight"));
    EXPECT_FALSE(hasWeight(weights, "blk.13.attn_q.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.14.attn_q.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.27.ffn_norm.weight"));
    EXPECT_TRUE(hasWeight(weights, "output_norm.weight"));
    EXPECT_TRUE(hasWeight(weights, "output.weight"));
}

// =============================================================================
// Per-Layer Weight Methods Tests
// =============================================================================

/**
 * @test attentionWeightsForLayer returns 5 weights
 */
TEST_F(Test__PartialWeightLoader, AttentionWeightsForLayer_ReturnsFiveWeights)
{
    auto weights = loader_->attentionWeightsForLayer(3);

    EXPECT_EQ(weights.size(), 5);
    EXPECT_TRUE(hasWeight(weights, "blk.3.attn_q.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.3.attn_k.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.3.attn_v.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.3.attn_output.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.3.attn_norm.weight"));
}

/**
 * @test ffnWeightsForLayer returns 4 weights
 */
TEST_F(Test__PartialWeightLoader, FFNWeightsForLayer_ReturnsFourWeights)
{
    auto weights = loader_->ffnWeightsForLayer(10);

    EXPECT_EQ(weights.size(), 4);
    EXPECT_TRUE(hasWeight(weights, "blk.10.ffn_gate.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.10.ffn_up.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.10.ffn_down.weight"));
    EXPECT_TRUE(hasWeight(weights, "blk.10.ffn_norm.weight"));
}

/**
 * @test allWeightsForLayer returns 9 weights
 */
TEST_F(Test__PartialWeightLoader, AllWeightsForLayer_ReturnsNineWeights)
{
    auto weights = loader_->allWeightsForLayer(15);

    EXPECT_EQ(weights.size(), 9);
}

// =============================================================================
// getWeightInfoForPlan Tests
// =============================================================================

/**
 * @test getWeightInfoForPlan returns correct info for first stage
 */
TEST_F(Test__PartialWeightLoader, GetWeightInfoForPlan_FirstStage)
{
    auto plan = createPlan(0, 0, 13, true, false);
    auto info = loader_->getWeightInfoForPlan(plan);

    EXPECT_EQ(info.layer_count, 14);
    EXPECT_TRUE(info.has_embedding);
    EXPECT_FALSE(info.has_lm_head);
    EXPECT_EQ(info.weight_names.size(), 14 * 9 + 1);
}

/**
 * @test getWeightInfoForPlan returns correct info for last stage
 */
TEST_F(Test__PartialWeightLoader, GetWeightInfoForPlan_LastStage)
{
    auto plan = createPlan(1, 14, 27, false, true);
    auto info = loader_->getWeightInfoForPlan(plan);

    EXPECT_EQ(info.layer_count, 14);
    EXPECT_FALSE(info.has_embedding);
    EXPECT_TRUE(info.has_lm_head);
    EXPECT_EQ(info.weight_names.size(), 14 * 9 + 2);
}

// =============================================================================
// Memory Estimation Tests
// =============================================================================

/**
 * @test estimateMemoryForPlan returns non-zero for valid plan
 */
TEST_F(Test__PartialWeightLoader, EstimateMemory_ReturnsNonZero)
{
    auto plan = createPlan(0, 0, 13, true, false);
    size_t estimate = loader_->estimateMemoryForPlan(plan);

    EXPECT_GT(estimate, 0);
}

/**
 * @test Memory estimate for half model is roughly half of full model
 */
TEST_F(Test__PartialWeightLoader, EstimateMemory_HalfModelIsRoughlyHalf)
{
    auto full_plan = createPlan(0, 0, 27, true, true);
    auto half_plan = createPlan(0, 0, 13, true, false);

    size_t full_estimate = loader_->estimateMemoryForPlan(full_plan);
    size_t half_estimate = loader_->estimateMemoryForPlan(half_plan);

    // Half should be roughly 40-60% of full (embedding is shared overhead)
    double ratio = static_cast<double>(half_estimate) / full_estimate;
    EXPECT_GT(ratio, 0.3);
    EXPECT_LT(ratio, 0.7);
}

/**
 * @test Memory estimate scales with layer count
 */
TEST_F(Test__PartialWeightLoader, EstimateMemory_ScalesWithLayers)
{
    auto plan_10_layers = createPlan(0, 0, 9, false, false);
    auto plan_20_layers = createPlan(0, 0, 19, false, false);

    size_t estimate_10 = loader_->estimateMemoryForPlan(plan_10_layers);
    size_t estimate_20 = loader_->estimateMemoryForPlan(plan_20_layers);

    // 20 layers should be roughly 2x the memory of 10 layers
    double ratio = static_cast<double>(estimate_20) / estimate_10;
    EXPECT_GT(ratio, 1.5);
    EXPECT_LT(ratio, 2.5);
}

// =============================================================================
// Factory Tests
// =============================================================================

/**
 * @test Factory creates loader without model config
 */
TEST_F(Test__PartialWeightLoader, Factory_CreateWithoutConfig)
{
    auto loader = createPartialWeightLoader(nullptr);
    ASSERT_NE(loader, nullptr);

    // Should still work, just with fallback estimates
    auto weights = loader->weightsForLayerRange(0, 4, true, false);
    EXPECT_EQ(weights.size(), 5 * 9 + 1);
}

/**
 * @test Factory creates loader with model config
 */
TEST_F(Test__PartialWeightLoader, Factory_CreateWithConfig)
{
    ModelConfig config = ModelConfig::qwen2_0_5b();
    auto loader = createPartialWeightLoader(&config);
    ASSERT_NE(loader, nullptr);

    // Verify it uses the config for better estimates
    auto plan = createPlan(0, 0, 11, true, false);
    size_t estimate = loader->estimateMemoryForPlan(plan);
    EXPECT_GT(estimate, 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

/**
 * @test weightsForLayerRange handles single layer
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_SingleLayer)
{
    auto weights = loader_->weightsForLayerRange(5, 5, false, false);

    // Single layer: 9 weights
    EXPECT_EQ(weights.size(), 9);
    EXPECT_TRUE(hasWeight(weights, "blk.5.attn_q.weight"));
    EXPECT_FALSE(hasWeight(weights, "blk.4.attn_q.weight"));
    EXPECT_FALSE(hasWeight(weights, "blk.6.attn_q.weight"));
}

/**
 * @test weightsForLayerRange handles invalid range (first > last)
 */
TEST_F(Test__PartialWeightLoader, WeightsForLayerRange_InvalidRange)
{
    auto weights = loader_->weightsForLayerRange(10, 5, false, false);

    // Invalid range should return empty layer weights (but may include non-layer weights)
    // For the case of no embedding and no lm_head, should be empty
    EXPECT_EQ(weights.size(), 0);
}

/**
 * @test getWeightInfoForPlan handles zero layers
 */
TEST_F(Test__PartialWeightLoader, GetWeightInfoForPlan_ZeroLayers)
{
    RankExecutionPlan plan;
    plan.rank = 0;
    plan.first_layer = 0;
    plan.last_layer = -1; // Invalid/empty
    plan.has_embedding = true;
    plan.has_lm_head = false;

    auto info = loader_->getWeightInfoForPlan(plan);

    // Should still have embedding
    EXPECT_EQ(info.layer_count, 0);
    EXPECT_TRUE(info.has_embedding);
    EXPECT_EQ(info.weight_names.size(), 1); // Just embedding
}
