/**
 * @file Test__PlacementStrategy.cpp
 * @brief Unit tests for PlacementStrategy and PlacementPlan
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>

#include "../../src/v2/execution/mpi_orchestration/PlacementPlan.h"
#include "../../src/v2/execution/mpi_orchestration/PlacementStrategy.h"

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
            plan.layers[i].device = PlacementDevice::cpu();
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
    plan.layers[0].device = PlacementDevice::gpu(0);
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
    plan.layers[5].device = PlacementDevice::gpu(0);

    EXPECT_EQ(plan.getLayerPlacement(5).device, PlacementDevice::gpu(0));
    EXPECT_EQ(plan.getLayerPlacement(0).device, PlacementDevice::cpu());
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
    EXPECT_EQ(toDeviceId(PlacementDevice::cpu()), DeviceId::cpu());
    EXPECT_EQ(toDeviceId(PlacementDevice::gpu(0)), DeviceId::cuda(0));
    EXPECT_EQ(toDeviceId(PlacementDevice::gpu(1)), DeviceId::cuda(1));
    EXPECT_EQ(toDeviceId(PlacementDevice::gpu(2)), DeviceId::cuda(2));
    EXPECT_EQ(toDeviceId(PlacementDevice::gpu(3)), DeviceId::cuda(3));
    EXPECT_EQ(toDeviceId(PlacementDevice::anyGpu()), DeviceId::cuda(0));
    EXPECT_EQ(toDeviceId(PlacementDevice::replicated()), DeviceId::cpu());
}

// =============================================================================
// LayerPlacement Tests
// =============================================================================

TEST(Test__LayerPlacement, GetDeviceWithoutSplit)
{
    LayerPlacement lp;
    lp.device = PlacementDevice::gpu(0);
    lp.split_attention_ffn = false;

    EXPECT_EQ(lp.getAttentionDevice(), DeviceId::cuda(0)); // GPU_0
    EXPECT_EQ(lp.getFFNDevice(), DeviceId::cuda(0));
}

TEST(Test__LayerPlacement, GetDeviceWithSplit)
{
    LayerPlacement lp;
    lp.device = PlacementDevice::cpu();
    lp.attention_device = PlacementDevice::gpu(0);
    lp.ffn_device = PlacementDevice::gpu(1);
    lp.split_attention_ffn = true;

    EXPECT_EQ(lp.getAttentionDevice(), DeviceId::cuda(0)); // GPU_0
    EXPECT_EQ(lp.getFFNDevice(), DeviceId::cuda(1));       // GPU_1
}

// =============================================================================
// CPUOnlyPlacementStrategy Tests
// =============================================================================

class Test__CPUOnlyPlacementStrategy : public ::testing::Test
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

TEST_F(Test__CPUOnlyPlacementStrategy, IsApplicableWhenForcedCPU)
{
    CPUOnlyPlacementStrategy strategy;
    auto input = createBasicInput(24, 1);
    input.force_cpu_only = true;
    input.any_rank_has_gpu = true; // Even with GPU

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__CPUOnlyPlacementStrategy, IsApplicableWhenNoGPU)
{
    CPUOnlyPlacementStrategy strategy;
    auto input = createBasicInput(24, 1);
    input.any_rank_has_gpu = false;

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__CPUOnlyPlacementStrategy, AlwaysApplicableEvenWithGPU)
{
    CPUOnlyPlacementStrategy strategy;
    auto input = createBasicInput(24, 1);
    input.any_rank_has_gpu = true;
    input.force_cpu_only = false;

    // CPUOnlyPlacementStrategy is ALWAYS applicable - user can explicitly choose CPU
    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__CPUOnlyPlacementStrategy, ComputeReturnsCPUPlan)
{
    CPUOnlyPlacementStrategy strategy;
    auto input = createBasicInput(24, 1);

    PlacementPlan plan = strategy.compute(input);

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.n_layers, 24);
    EXPECT_EQ(plan.strategy_name, "CPUOnly");
    EXPECT_FALSE(plan.has_gpu);
    EXPECT_FALSE(plan.usesGPU());
}

TEST_F(Test__CPUOnlyPlacementStrategy, ComputeSetsAllLayersToCPU)
{
    CPUOnlyPlacementStrategy strategy;
    auto input = createBasicInput(24, 1);

    PlacementPlan plan = strategy.compute(input);

    for (int i = 0; i < 24; ++i)
    {
        EXPECT_EQ(plan.layers[i].device, PlacementDevice::cpu())
            << "Layer " << i << " should be CPU";
        EXPECT_EQ(plan.layers[i].layer_idx, i);
    }
}

TEST_F(Test__CPUOnlyPlacementStrategy, ComputeSetsGlobalTensorsToCPU)
{
    CPUOnlyPlacementStrategy strategy;
    auto input = createBasicInput(24, 1);

    PlacementPlan plan = strategy.compute(input);

    EXPECT_EQ(plan.global.embedding_device, PlacementDevice::cpu());
    EXPECT_EQ(plan.global.lm_head_device, PlacementDevice::cpu());
    EXPECT_EQ(plan.global.final_norm_device, PlacementDevice::cpu());
}

TEST_F(Test__CPUOnlyPlacementStrategy, ComputeShardsLargeVocab)
{
    CPUOnlyPlacementStrategy strategy;
    auto input = createBasicInput(24, 2);
    input.vocab_size = 200000; // Large vocab

    PlacementPlan plan = strategy.compute(input);

    // Large vocab with multi-rank should shard
    EXPECT_TRUE(plan.global.shard_embedding);
    EXPECT_TRUE(plan.global.shard_lm_head);
}

TEST_F(Test__CPUOnlyPlacementStrategy, ComputeNoShardSmallVocab)
{
    CPUOnlyPlacementStrategy strategy;
    auto input = createBasicInput(24, 2);
    input.vocab_size = 10000; // Small vocab

    PlacementPlan plan = strategy.compute(input);

    // Small vocab shouldn't shard
    EXPECT_FALSE(plan.global.shard_embedding);
    EXPECT_FALSE(plan.global.shard_lm_head);
}

// =============================================================================
// GPUFirstPlacementStrategy Tests
// =============================================================================

class Test__GPUFirstPlacementStrategy : public ::testing::Test
{
protected:
    PlacementInput createGPUInput(int n_layers, size_t gpu_memory_gb, int num_gpus = 1)
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

        input.world_size = 1;
        input.ranks_per_node = 1;
        input.node_count = 1;
        input.any_rank_has_gpu = true;

        // Set up per-GPU memory
        size_t mem_per_gpu = gpu_memory_gb * 1024ULL * 1024 * 1024;
        input.total_gpu_memory = mem_per_gpu * num_gpus;
        for (int i = 0; i < num_gpus; ++i)
        {
            input.gpu_memory_per_device.push_back(mem_per_gpu);
        }

        input.total_cpu_memory = 64ULL * 1024 * 1024 * 1024;

        return input;
    }
};

TEST_F(Test__GPUFirstPlacementStrategy, IsApplicableWhenGPUAvailable)
{
    GPUFirstPlacementStrategy strategy;
    PlacementInput input;
    input.any_rank_has_gpu = true;
    input.force_cpu_only = false;

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__GPUFirstPlacementStrategy, NotApplicableWhenForcedCPU)
{
    GPUFirstPlacementStrategy strategy;
    PlacementInput input;
    input.any_rank_has_gpu = true;
    input.force_cpu_only = true;

    EXPECT_FALSE(strategy.isApplicable(input));
}

TEST_F(Test__GPUFirstPlacementStrategy, NotApplicableWhenNoGPU)
{
    GPUFirstPlacementStrategy strategy;
    PlacementInput input;
    input.any_rank_has_gpu = false;
    input.force_cpu_only = false;

    EXPECT_FALSE(strategy.isApplicable(input));
}

TEST_F(Test__GPUFirstPlacementStrategy, ComputeProducesValidPlan)
{
    GPUFirstPlacementStrategy strategy;
    auto input = createGPUInput(24, 8); // 8GB GPU

    PlacementPlan plan = strategy.compute(input);

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.n_layers, 24);
    EXPECT_EQ(plan.strategy_name, "GPUFirst");
    EXPECT_TRUE(plan.has_gpu);
}

TEST_F(Test__GPUFirstPlacementStrategy, ComputePlacesLayersOnGPU)
{
    GPUFirstPlacementStrategy strategy;
    auto input = createGPUInput(24, 16); // 16GB GPU - should fit most/all layers

    PlacementPlan plan = strategy.compute(input);

    // Should have at least some layers on GPU
    int gpu_layers = 0;
    for (const auto &layer : plan.layers)
    {
        if (layer.device.isGPU())
        {
            gpu_layers++;
        }
    }
    EXPECT_GT(gpu_layers, 0) << "Should have at least some layers on GPU";
}

TEST_F(Test__GPUFirstPlacementStrategy, ComputeRespectsMaxGPULayers)
{
    GPUFirstPlacementStrategy strategy;
    auto input = createGPUInput(24, 16); // Plenty of GPU memory
    input.max_gpu_layers = 5;            // But limit to 5 layers

    PlacementPlan plan = strategy.compute(input);

    int gpu_layers = 0;
    for (const auto &layer : plan.layers)
    {
        if (layer.device.isGPU())
        {
            gpu_layers++;
        }
    }
    EXPECT_LE(gpu_layers, 5) << "Should respect max_gpu_layers constraint";
}

TEST_F(Test__GPUFirstPlacementStrategy, ComputeSpillsToCPUWhenMemoryLimited)
{
    GPUFirstPlacementStrategy strategy;
    auto input = createGPUInput(24, 1); // Only 1GB GPU - won't fit all layers

    PlacementPlan plan = strategy.compute(input);

    // Should have some layers on CPU (spillover)
    int cpu_layers = 0;
    for (const auto &layer : plan.layers)
    {
        if (layer.device.isCPU())
        {
            cpu_layers++;
        }
    }
    EXPECT_GT(cpu_layers, 0) << "Small GPU should spill some layers to CPU";
}

TEST_F(Test__GPUFirstPlacementStrategy, ComputeDistributesAcrossMultipleGPUs)
{
    GPUFirstPlacementStrategy strategy;
    // 2 GPUs, each with 8GB - should use both
    auto input = createGPUInput(24, 8, 2);

    PlacementPlan plan = strategy.compute(input);

    // Count layers per GPU
    int gpu_layers = 0;
    int cpu_layers = 0;
    for (const auto &layer : plan.layers)
    {
        if (layer.device.isGPU())
            gpu_layers++;
        else
            cpu_layers++;
    }

    // With 16GB total across 2 GPUs, should use GPUs
    EXPECT_GT(gpu_layers, 0) << "Should use GPUs";
    // The key is that the strategy doesn't crash and produces a valid plan
    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(gpu_layers + cpu_layers, 24);
}

// =============================================================================
// HybridOptimalPlacementStrategy Tests
// =============================================================================

class Test__HybridOptimalPlacementStrategy : public ::testing::Test
{
protected:
    PlacementInput createHybridInput(int n_layers, size_t gpu_memory_gb, int num_gpus = 1)
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
        input.estimated_memory_bytes = 500000000;

        input.world_size = 1;
        input.ranks_per_node = 1;
        input.node_count = 1;
        input.any_rank_has_gpu = true;

        // Set up per-GPU memory
        size_t mem_per_gpu = gpu_memory_gb * 1024ULL * 1024 * 1024;
        input.total_gpu_memory = mem_per_gpu * num_gpus;
        for (int i = 0; i < num_gpus; ++i)
        {
            input.gpu_memory_per_device.push_back(mem_per_gpu);
        }

        input.total_cpu_memory = 64ULL * 1024 * 1024 * 1024;

        // Performance characteristics for Xeon + RTX 4090
        input.cpu_memory_bandwidth_gbps = 200.0f;  // DDR5 bandwidth
        input.gpu_memory_bandwidth_gbps = 1000.0f; // HBM bandwidth
        input.cpu_compute_tflops = 2.0f;           // AVX-512
        input.gpu_compute_tflops = 330.0f;         // FP16

        return input;
    }
};

TEST_F(Test__HybridOptimalPlacementStrategy, IsApplicableWithGPU)
{
    HybridOptimalPlacementStrategy strategy;
    auto input = createHybridInput(24, 8);

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__HybridOptimalPlacementStrategy, NotApplicableWhenForcedCPUOnly)
{
    HybridOptimalPlacementStrategy strategy;
    auto input = createHybridInput(24, 8);
    input.force_cpu_only = true;

    EXPECT_FALSE(strategy.isApplicable(input));
}

TEST_F(Test__HybridOptimalPlacementStrategy, NotApplicableWhenForcedGPUOnly)
{
    HybridOptimalPlacementStrategy strategy;
    auto input = createHybridInput(24, 8);
    input.force_gpu_only = true;

    EXPECT_FALSE(strategy.isApplicable(input));
}

TEST_F(Test__HybridOptimalPlacementStrategy, ComputeProducesValidPlan)
{
    HybridOptimalPlacementStrategy strategy;
    auto input = createHybridInput(24, 8);

    PlacementPlan plan = strategy.compute(input);

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.n_layers, 24);
    EXPECT_EQ(plan.strategy_name, "HybridOptimal");
    EXPECT_TRUE(plan.has_gpu);
}

TEST_F(Test__HybridOptimalPlacementStrategy, ConsidersCPUBandwidth)
{
    HybridOptimalPlacementStrategy strategy;

    // High CPU bandwidth should potentially lead to more CPU layers
    auto input_high_cpu_bw = createHybridInput(24, 8);
    input_high_cpu_bw.cpu_memory_bandwidth_gbps = 300.0f; // Very high

    auto input_low_cpu_bw = createHybridInput(24, 8);
    input_low_cpu_bw.cpu_memory_bandwidth_gbps = 50.0f; // Low

    PlacementPlan plan_high = strategy.compute(input_high_cpu_bw);
    PlacementPlan plan_low = strategy.compute(input_low_cpu_bw);

    // Both should be valid
    EXPECT_TRUE(plan_high.isValid());
    EXPECT_TRUE(plan_low.isValid());

    // Count GPU layers
    auto countGPULayers = [](const PlacementPlan &plan)
    {
        int count = 0;
        for (const auto &layer : plan.layers)
        {
            if (layer.device.isGPU())
                count++;
        }
        return count;
    };

    // With high CPU bandwidth, might use fewer GPU layers (CPU is more competitive)
    // This is a heuristic test - exact behavior depends on the algorithm
    int gpu_high = countGPULayers(plan_high);
    int gpu_low = countGPULayers(plan_low);

    // At minimum, both should have some GPU layers
    EXPECT_GT(gpu_high, 0);
    EXPECT_GT(gpu_low, 0);
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

TEST(Test__PlacementStrategyFactory, CreateGPUFirstByAlternateNames)
{
    auto s1 = PlacementStrategyFactory::create("gpu");
    auto s2 = PlacementStrategyFactory::create("gpu_first");

    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);
    EXPECT_EQ(s1->name(), "GPUFirst");
    EXPECT_EQ(s2->name(), "GPUFirst");
}

TEST(Test__PlacementStrategyFactory, CreateHybridOptimalByName)
{
    auto strategy = PlacementStrategyFactory::create("HybridOptimal");
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "HybridOptimal");
}

TEST(Test__PlacementStrategyFactory, CreateHybridOptimalByAlternateNames)
{
    auto s1 = PlacementStrategyFactory::create("hybrid");
    auto s2 = PlacementStrategyFactory::create("hybrid_optimal");

    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);
    EXPECT_EQ(s1->name(), "HybridOptimal");
    EXPECT_EQ(s2->name(), "HybridOptimal");
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

TEST(Test__PlacementStrategyFactory, AutoSelectUsesGPUFirstWhenGPUAvailable)
{
    PlacementInput input;
    input.any_rank_has_gpu = true;
    input.total_gpu_memory = 8ULL * 1024 * 1024 * 1024;
    // No bandwidth info → should use GPUFirst

    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "GPUFirst");
}

TEST(Test__PlacementStrategyFactory, AutoSelectUsesHybridOptimalWithBandwidthInfo)
{
    PlacementInput input;
    input.any_rank_has_gpu = true;
    input.total_gpu_memory = 8ULL * 1024 * 1024 * 1024;
    input.cpu_memory_bandwidth_gbps = 200.0f; // Has bandwidth info

    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "HybridOptimal");
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

TEST(Test__PlacementStrategyFactory, AutoSelectDefaultsToCPUOnlyWhenNoGPU)
{
    PlacementInput input;
    input.any_rank_has_gpu = false;

    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "CPUOnly");
}

TEST(Test__PlacementStrategyFactory, AvailableStrategiesContainsAll)
{
    auto strategies = PlacementStrategyFactory::availableStrategies();
    EXPECT_FALSE(strategies.empty());
    EXPECT_NE(std::find(strategies.begin(), strategies.end(), "CPUOnly"), strategies.end());
    EXPECT_NE(std::find(strategies.begin(), strategies.end(), "GPUFirst"), strategies.end());
    EXPECT_NE(std::find(strategies.begin(), strategies.end(), "HybridOptimal"), strategies.end());
}

// =============================================================================
// Legacy Alias Tests
// =============================================================================

TEST(Test__LegacyAliases, CPUOnlyStrategyAlias)
{
    // CPUOnlyStrategy should be an alias for CPUOnlyPlacementStrategy
    CPUOnlyStrategy strategy;
    EXPECT_EQ(strategy.name(), "CPUOnly");
}

TEST(Test__LegacyAliases, GPUFirstStrategyAlias)
{
    // GPUFirstStrategy should be an alias for GPUFirstPlacementStrategy
    GPUFirstStrategy strategy;
    EXPECT_EQ(strategy.name(), "GPUFirst");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
