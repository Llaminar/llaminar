/**
 * @file Test__PlacementStrategy.cpp
 * @brief Unit tests for PlacementStrategy and PlacementPlan
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>

#include "../../src/v2/execution/PlacementPlan.h"
#include "../../src/v2/execution/PlacementStrategy.h"

using namespace llaminar2;

// =============================================================================
// PlacementPlan Tests
// =============================================================================

class Test__PlacementPlan : public ::testing::Test
{
protected:
    PlacementPlan createTestPlan(int n_layers, int world_size)
    {
        PlacementPlan plan;
        plan.n_layers = n_layers;
        plan.world_size = world_size;
        plan.ranks_per_node = 1;
        plan.node_count = world_size;
        plan.architecture = "test";
        plan.strategy_name = "TestStrategy";

        plan.layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            plan.layers[i].layer_idx = i;
            plan.layers[i].owner_rank = 0;
            plan.layers[i].device = PlacementDevice::CPU;
        }

        return plan;
    }
};

TEST_F(Test__PlacementPlan, IsValidRequiresLayers)
{
    PlacementPlan plan;
    plan.n_layers = 0;
    plan.world_size = 1;
    EXPECT_FALSE(plan.isValid());
}

TEST_F(Test__PlacementPlan, IsValidRequiresWorldSize)
{
    PlacementPlan plan;
    plan.n_layers = 24;
    plan.world_size = 0;
    EXPECT_FALSE(plan.isValid());
}

TEST_F(Test__PlacementPlan, IsValidRequiresMatchingLayerCount)
{
    PlacementPlan plan;
    plan.n_layers = 24;
    plan.world_size = 1;
    plan.layers.resize(10); // Wrong size
    for (auto &l : plan.layers)
    {
        l.owner_rank = 0;
    }
    EXPECT_FALSE(plan.isValid());
}

TEST_F(Test__PlacementPlan, IsValidPassesWithCorrectSetup)
{
    auto plan = createTestPlan(24, 2);
    EXPECT_TRUE(plan.isValid());
}

TEST_F(Test__PlacementPlan, UsesGPUReturnsFalseForCPUOnly)
{
    auto plan = createTestPlan(24, 1);
    EXPECT_FALSE(plan.usesGPU());
}

TEST_F(Test__PlacementPlan, UsesGPUReturnsTrueForGPULayers)
{
    auto plan = createTestPlan(24, 1);
    plan.layers[0].device = PlacementDevice::GPU_0;
    EXPECT_TRUE(plan.usesGPU());
}

TEST_F(Test__PlacementPlan, UsesTensorParallelismForMultiRank)
{
    auto plan = createTestPlan(24, 2);
    EXPECT_TRUE(plan.usesTensorParallelism());
}

TEST_F(Test__PlacementPlan, NoTensorParallelismForSingleRank)
{
    auto plan = createTestPlan(24, 1);
    EXPECT_FALSE(plan.usesTensorParallelism());
}

TEST_F(Test__PlacementPlan, GetLayerPlacementReturnsCorrectLayer)
{
    auto plan = createTestPlan(24, 1);
    plan.layers[5].device = PlacementDevice::GPU_0;

    EXPECT_EQ(plan.getLayerPlacement(5).device, PlacementDevice::GPU_0);
    EXPECT_EQ(plan.getLayerPlacement(0).device, PlacementDevice::CPU);
}

TEST_F(Test__PlacementPlan, GetLayerPlacementHandlesOutOfBounds)
{
    auto plan = createTestPlan(24, 1);

    // Out of bounds should return default
    const auto &oob = plan.getLayerPlacement(100);
    EXPECT_EQ(oob.layer_idx, -1); // Default value
}

TEST_F(Test__PlacementPlan, GetActiveRanksReturnsCorrectRanks)
{
    auto plan = createTestPlan(24, 2);
    // All layers owned by rank 0
    auto active = plan.getActiveRanks();
    EXPECT_EQ(active.size(), 1u);
    EXPECT_EQ(active[0], 0);
}

TEST_F(Test__PlacementPlan, ToStringProducesOutput)
{
    auto plan = createTestPlan(24, 1);
    std::string str = plan.toString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("PlacementPlan"), std::string::npos);
    EXPECT_NE(str.find("TestStrategy"), std::string::npos);
}

// =============================================================================
// PlacementDevice Tests
// =============================================================================

TEST(Test__PlacementDevice, ToDeviceIdMapsCorrectly)
{
    EXPECT_EQ(toDeviceId(PlacementDevice::CPU), DeviceId::cpu());
    EXPECT_EQ(toDeviceId(PlacementDevice::GPU_0), DeviceId::cuda(0));
    EXPECT_EQ(toDeviceId(PlacementDevice::GPU_1), DeviceId::cuda(1));
    EXPECT_EQ(toDeviceId(PlacementDevice::GPU_2), DeviceId::cuda(2));
    EXPECT_EQ(toDeviceId(PlacementDevice::GPU_3), DeviceId::cuda(3));
    EXPECT_EQ(toDeviceId(PlacementDevice::GPU_ANY), DeviceId::cuda(0));
    EXPECT_EQ(toDeviceId(PlacementDevice::REPLICATED), DeviceId::cpu());
}

// =============================================================================
// LayerPlacement Tests
// =============================================================================

TEST(Test__LayerPlacement, GetDeviceWithoutSplit)
{
    LayerPlacement lp;
    lp.device = PlacementDevice::GPU_0;
    lp.split_attention_ffn = false;

    EXPECT_EQ(lp.getAttentionDevice(), DeviceId::cuda(0)); // GPU_0
    EXPECT_EQ(lp.getFFNDevice(), DeviceId::cuda(0));
}

TEST(Test__LayerPlacement, GetDeviceWithSplit)
{
    LayerPlacement lp;
    lp.device = PlacementDevice::CPU;
    lp.attention_device = PlacementDevice::GPU_0;
    lp.ffn_device = PlacementDevice::GPU_1;
    lp.split_attention_ffn = true;

    EXPECT_EQ(lp.getAttentionDevice(), DeviceId::cuda(0)); // GPU_0
    EXPECT_EQ(lp.getFFNDevice(), DeviceId::cuda(1));       // GPU_1
}

// =============================================================================
// CPUOnlyStrategy Tests
// =============================================================================

class Test__CPUOnlyStrategy : public ::testing::Test
{
protected:
    PlacementInput createBasicInput(int n_layers, int world_size)
    {
        PlacementInput input;
        input.architecture = "qwen2";
        input.n_layers = n_layers;
        input.d_model = 896;
        input.d_ff = 4864;
        input.vocab_size = 151936;
        input.n_heads = 14;
        input.n_kv_heads = 2;
        input.quant_type = "Q4_0";
        input.estimated_memory_bytes = 500000000; // 500MB

        input.world_size = world_size;
        input.ranks_per_node = world_size;
        input.node_count = 1;
        input.any_rank_has_gpu = false;
        input.total_gpu_memory = 0;
        input.total_cpu_memory = 64ULL * 1024 * 1024 * 1024; // 64GB

        input.rank_compute_weights.resize(world_size, 1.0f);

        return input;
    }
};

TEST_F(Test__CPUOnlyStrategy, IsApplicableWhenForcedCPU)
{
    CPUOnlyStrategy strategy;
    auto input = createBasicInput(24, 1);
    input.force_cpu_only = true;
    input.any_rank_has_gpu = true; // Even with GPU

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__CPUOnlyStrategy, IsApplicableWhenNoGPU)
{
    CPUOnlyStrategy strategy;
    auto input = createBasicInput(24, 1);
    input.any_rank_has_gpu = false;

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__CPUOnlyStrategy, AlwaysApplicableEvenWithGPU)
{
    CPUOnlyStrategy strategy;
    auto input = createBasicInput(24, 1);
    input.any_rank_has_gpu = true;
    input.force_cpu_only = false;

    // CPUOnlyStrategy is ALWAYS applicable - user can explicitly choose CPU
    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__CPUOnlyStrategy, ComputeReturnsCPUPlan)
{
    CPUOnlyStrategy strategy;
    auto input = createBasicInput(24, 1);

    PlacementPlan plan = strategy.compute(input);

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.n_layers, 24);
    EXPECT_EQ(plan.strategy_name, "CPUOnly");
    EXPECT_FALSE(plan.has_gpu);
    EXPECT_FALSE(plan.usesGPU());
}

TEST_F(Test__CPUOnlyStrategy, ComputeSetsAllLayersToCPU)
{
    CPUOnlyStrategy strategy;
    auto input = createBasicInput(24, 1);

    PlacementPlan plan = strategy.compute(input);

    for (int i = 0; i < 24; ++i)
    {
        EXPECT_EQ(plan.layers[i].device, PlacementDevice::CPU)
            << "Layer " << i << " should be CPU";
        EXPECT_EQ(plan.layers[i].layer_idx, i);
    }
}

TEST_F(Test__CPUOnlyStrategy, ComputeSetsGlobalTensorsToCPU)
{
    CPUOnlyStrategy strategy;
    auto input = createBasicInput(24, 1);

    PlacementPlan plan = strategy.compute(input);

    EXPECT_EQ(plan.global.embedding_device, PlacementDevice::CPU);
    EXPECT_EQ(plan.global.lm_head_device, PlacementDevice::CPU);
    EXPECT_EQ(plan.global.final_norm_device, PlacementDevice::CPU);
}

TEST_F(Test__CPUOnlyStrategy, ComputeShardsLargeVocab)
{
    CPUOnlyStrategy strategy;
    auto input = createBasicInput(24, 2);
    input.vocab_size = 200000; // Large vocab

    PlacementPlan plan = strategy.compute(input);

    // Large vocab with multi-rank should shard
    EXPECT_TRUE(plan.global.shard_embedding);
    EXPECT_TRUE(plan.global.shard_lm_head);
}

TEST_F(Test__CPUOnlyStrategy, ComputeNoShardSmallVocab)
{
    CPUOnlyStrategy strategy;
    auto input = createBasicInput(24, 2);
    input.vocab_size = 10000; // Small vocab

    PlacementPlan plan = strategy.compute(input);

    // Small vocab shouldn't shard
    EXPECT_FALSE(plan.global.shard_embedding);
    EXPECT_FALSE(plan.global.shard_lm_head);
}

// =============================================================================
// GPUFirstStrategy Tests (Placeholder - Falls Back to CPU)
// =============================================================================

TEST(Test__GPUFirstStrategy, IsApplicableWhenGPUAvailable)
{
    GPUFirstStrategy strategy;
    PlacementInput input;
    input.any_rank_has_gpu = true;
    input.force_cpu_only = false;

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST(Test__GPUFirstStrategy, NotApplicableWhenForcedCPU)
{
    GPUFirstStrategy strategy;
    PlacementInput input;
    input.any_rank_has_gpu = true;
    input.force_cpu_only = true;

    EXPECT_FALSE(strategy.isApplicable(input));
}

TEST(Test__GPUFirstStrategy, ComputeThrowsNotImplemented)
{
    GPUFirstStrategy strategy;
    PlacementInput input;
    input.n_layers = 24;
    input.world_size = 1;
    input.ranks_per_node = 1;
    input.node_count = 1;
    input.any_rank_has_gpu = true;

    // GPUFirst strategy is not yet implemented - should throw, not silently fallback
    EXPECT_THROW(strategy.compute(input), std::runtime_error);
}

// =============================================================================
// PlacementStrategyFactory Tests
// =============================================================================

TEST(Test__PlacementStrategyFactory, CreateCPUOnlyByName)
{
    auto strategy = PlacementStrategyFactory::create("CPUOnly");
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "CPUOnly");
}

TEST(Test__PlacementStrategyFactory, CreateCPUOnlyByAlternateNames)
{
    auto s1 = PlacementStrategyFactory::create("cpu");
    auto s2 = PlacementStrategyFactory::create("cpu_only");

    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);
    EXPECT_EQ(s1->name(), "CPUOnly");
    EXPECT_EQ(s2->name(), "CPUOnly");
}

TEST(Test__PlacementStrategyFactory, CreateGPUFirstByName)
{
    auto strategy = PlacementStrategyFactory::create("GPUFirst");
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "GPUFirst");
}

TEST(Test__PlacementStrategyFactory, CreateReturnsNullForUnknown)
{
    auto strategy = PlacementStrategyFactory::create("NonexistentStrategy");
    EXPECT_EQ(strategy, nullptr);
}

TEST(Test__PlacementStrategyFactory, AutoSelectRespectsUserChoice)
{
    PlacementInput input;
    input.preferred_strategy = "CPUOnly";
    input.any_rank_has_gpu = true; // GPU available but user explicitly wants CPU

    // User explicitly chose CPUOnly, so this should work
    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "CPUOnly");
}

TEST(Test__PlacementStrategyFactory, AutoSelectThrowsForGPUWithGPUAvailable)
{
    PlacementInput input;
    input.any_rank_has_gpu = true;
    // No preferred_strategy and no force_cpu_only - would auto-select for GPU
    // But GPU strategies aren't implemented, so should throw

    EXPECT_THROW(PlacementStrategyFactory::autoSelect(input), std::runtime_error);
}

TEST(Test__PlacementStrategyFactory, AutoSelectThrowsForInvalidStrategy)
{
    PlacementInput input;
    input.preferred_strategy = "NonexistentStrategy";

    EXPECT_THROW(PlacementStrategyFactory::autoSelect(input), std::runtime_error);
}

TEST(Test__PlacementStrategyFactory, AutoSelectRespectsForceCPU)
{
    PlacementInput input;
    input.force_cpu_only = true;
    input.any_rank_has_gpu = true;

    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "CPUOnly");
}

TEST(Test__PlacementStrategyFactory, AutoSelectDefaultsToCPUOnly)
{
    PlacementInput input;
    input.any_rank_has_gpu = false;

    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);
    // Currently defaults to CPUOnly until GPU support is implemented
    EXPECT_EQ(strategy->name(), "CPUOnly");
}

TEST(Test__PlacementStrategyFactory, AvailableStrategiesNotEmpty)
{
    auto strategies = PlacementStrategyFactory::availableStrategies();
    EXPECT_FALSE(strategies.empty());
    EXPECT_NE(std::find(strategies.begin(), strategies.end(), "CPUOnly"), strategies.end());
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
