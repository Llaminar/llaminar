/**
 * @file Test__ExecutionPlanBuilder.cpp
 * @brief Unit tests for ExecutionPlanBuilder
 *
 * Tests:
 * - Building plans for single device (no TP, no PP)
 * - Building plans for LOCAL TP (multiple devices on one rank)
 * - Building plans for GLOBAL TP (multiple ranks)
 * - Building plans for PP (multiple stages)
 * - Building plans with named domains (Scenario 7 style)
 * - Weight shard calculation
 * - Backend selection
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"

using namespace llaminar2;

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * @brief Helper to create a minimal ClusterInventory for testing
 */
class ClusterInventoryBuilder
{
public:
    ClusterInventoryBuilder &addRank(
        int rank,
        const std::string &hostname,
        int numa_node,
        std::vector<std::pair<DeviceType, int>> gpus = {})
    {
        RankInventory rank_inv;
        rank_inv.rank = rank;
        rank_inv.hostname = hostname;
        rank_inv.node_id = rank; // Simple: one rank per node
        rank_inv.local_rank = 0;
        rank_inv.numa_nodes = 2;

        for (const auto &[type, ordinal] : gpus)
        {
            DeviceInfo gpu;
            gpu.type = type;
            gpu.local_device_id = ordinal;
            gpu.numa_node = numa_node;
            gpu.memory_bytes = 16ULL * 1024 * 1024 * 1024; // 16GB
            gpu.compute_units = 108;
            gpu.tflops_fp16 = 100.0f;
            rank_inv.gpus.push_back(gpu);
        }

        ranks_.push_back(std::move(rank_inv));
        return *this;
    }

    ClusterInventory build()
    {
        ClusterInventory cluster;
        cluster.world_size = static_cast<int>(ranks_.size());
        cluster.ranks = std::move(ranks_);
        cluster.buildNodeAggregations();
        return cluster;
    }

private:
    std::vector<RankInventory> ranks_;
};

class Test__ExecutionPlanBuilder : public ::testing::Test
{
protected:
    std::unique_ptr<IExecutionPlanBuilder> builder;
    ModelConfig model;
    OrchestrationConfig config;

    void SetUp() override
    {
        builder = createExecutionPlanBuilder();
        model = ModelConfig::qwen2_7b();
        config = OrchestrationConfig::defaults();
    }
};

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, CreateBuilder_ReturnsNonNull)
{
    EXPECT_NE(builder, nullptr);
}

// ============================================================================
// Single Device Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_SingleRank_SingleGPU)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_EQ(plan.rank, 0);
    EXPECT_EQ(plan.hostname, "localhost");
    EXPECT_EQ(plan.first_layer, 0);
    EXPECT_EQ(plan.last_layer, model.n_layers - 1);
    EXPECT_TRUE(plan.has_embedding);
    EXPECT_TRUE(plan.has_lm_head);
    EXPECT_FALSE(plan.usesPipelineParallel());
    EXPECT_FALSE(plan.usesGlobalTP());

    // Validate the plan
    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_SingleRank_NoGPU_UsesCPU)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {}) // No GPUs
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_TRUE(plan.primary_device.isCPU());

    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

// ============================================================================
// Local TP Tests (Multiple Devices on One Rank)
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_LocalTP_TwoGPUs)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}})
                       .build();

    config.tp_degree = 2;
    config.tp_scope = TPScope::LOCAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_EQ(plan.local_tp_devices.size(), 2);
    EXPECT_TRUE(plan.usesLocalTP());
    EXPECT_FALSE(plan.usesGlobalTP());
    EXPECT_EQ(plan.local_tp_backend, CollectiveBackendType::NCCL);

    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_LocalTP_MixedGPUs_UsesPCIeBAR)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}, {DeviceType::ROCm, 0}})
                       .build();

    config.tp_degree = 2;
    config.tp_scope = TPScope::LOCAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_EQ(plan.local_tp_devices.size(), 2);
    EXPECT_EQ(plan.local_tp_backend, CollectiveBackendType::PCIE_BAR);

    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

// ============================================================================
// Global TP Tests (Multiple Ranks)
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_GlobalTP_TwoRanks)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    config.tp_degree = 2;
    config.tp_scope = TPScope::GLOBAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 2);

    // Both ranks should have all layers (TP shards work, not layers)
    for (int r = 0; r < 2; ++r)
    {
        const auto &plan = plans[r];
        EXPECT_EQ(plan.rank, r);
        EXPECT_EQ(plan.first_layer, 0);
        EXPECT_EQ(plan.last_layer, model.n_layers - 1);
        EXPECT_TRUE(plan.usesGlobalTP());
        EXPECT_EQ(plan.global_tp_rank_in_domain, r % 2);

        auto errors = plan.validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << r << " errors: " << (errors.empty() ? "" : errors[0]);
    }
}

// ============================================================================
// Pipeline Parallelism Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_PP_TwoStages)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    config.pp_degree = 2;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 2);

    // Stage 0: first half of layers
    const auto &plan0 = plans[0];
    EXPECT_EQ(plan0.pp_stage_id, 0);
    EXPECT_EQ(plan0.first_layer, 0);
    EXPECT_TRUE(plan0.has_embedding);
    EXPECT_FALSE(plan0.has_lm_head);
    EXPECT_FALSE(plan0.prev_rank.has_value());
    EXPECT_TRUE(plan0.next_rank.has_value());
    EXPECT_EQ(*plan0.next_rank, 1);

    // Stage 1: second half of layers
    const auto &plan1 = plans[1];
    EXPECT_EQ(plan1.pp_stage_id, 1);
    EXPECT_EQ(plan1.last_layer, model.n_layers - 1);
    EXPECT_FALSE(plan1.has_embedding);
    EXPECT_TRUE(plan1.has_lm_head);
    EXPECT_TRUE(plan1.prev_rank.has_value());
    EXPECT_EQ(*plan1.prev_rank, 0);
    EXPECT_FALSE(plan1.next_rank.has_value());

    // Verify layers are split evenly (28 layers / 2 = 14 each)
    EXPECT_EQ(plan0.layerCount() + plan1.layerCount(), model.n_layers);

    for (int r = 0; r < 2; ++r)
    {
        auto errors = plans[r].validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << r << " errors: " << (errors.empty() ? "" : errors[0]);
    }
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_PP_FourStages)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "node0", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "node0", 1, {{DeviceType::CUDA, 0}})
                       .addRank(2, "node1", 0, {{DeviceType::CUDA, 0}})
                       .addRank(3, "node1", 1, {{DeviceType::CUDA, 0}})
                       .build();

    config.pp_degree = 4;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 4);

    // Verify layer coverage is complete
    int total_layers = 0;
    for (const auto &plan : plans)
    {
        total_layers += plan.layerCount();

        auto errors = plan.validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << plan.rank << " errors: " << (errors.empty() ? "" : errors[0]);
    }
    EXPECT_EQ(total_layers, model.n_layers);

    // Verify PP chain
    EXPECT_FALSE(plans[0].prev_rank.has_value());
    EXPECT_TRUE(plans[0].next_rank.has_value());

    EXPECT_TRUE(plans[1].prev_rank.has_value());
    EXPECT_TRUE(plans[1].next_rank.has_value());

    EXPECT_TRUE(plans[2].prev_rank.has_value());
    EXPECT_TRUE(plans[2].next_rank.has_value());

    EXPECT_TRUE(plans[3].prev_rank.has_value());
    EXPECT_FALSE(plans[3].next_rank.has_value());
}

// ============================================================================
// Named Domain Tests (Scenario 7 Style)
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_TwoDomainsPP)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    // Define two domains
    DomainDefinition gpu0;
    gpu0.name = "gpu0";
    gpu0.devices = {GlobalDeviceAddress::cuda(0, 0)};
    gpu0.backend = CollectiveBackendType::NCCL;

    DomainDefinition gpu1;
    gpu1.name = "gpu1";
    gpu1.devices = {GlobalDeviceAddress::cuda(0, 1)};
    gpu1.backend = CollectiveBackendType::NCCL;

    config.domain_definitions = {gpu0, gpu1};

    // Define PP stages
    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "gpu0";
    stage0.first_layer = 0;
    stage0.last_layer = 13;

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "gpu1";
    stage1.first_layer = 14;
    stage1.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 2);

    // Verify domain-based assignment
    EXPECT_EQ(plans[0].first_layer, 0);
    EXPECT_EQ(plans[0].last_layer, 13);
    EXPECT_TRUE(plans[0].has_embedding);

    EXPECT_EQ(plans[1].first_layer, 14);
    EXPECT_EQ(plans[1].last_layer, 27);
    EXPECT_TRUE(plans[1].has_lm_head);

    for (int r = 0; r < 2; ++r)
    {
        auto errors = plans[r].validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << r << " errors: " << (errors.empty() ? "" : errors[0]);
    }
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_MixedVendorTP)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}, {DeviceType::ROCm, 0}})
                       .build();

    // Define mixed-vendor domain
    DomainDefinition mixed;
    mixed.name = "mixed_tp";
    mixed.devices = {
        GlobalDeviceAddress::cuda(0, 0),
        GlobalDeviceAddress::rocm(0, 0)};
    mixed.weights = {0.6f, 0.4f}; // CUDA is faster
    mixed.backend = CollectiveBackendType::PCIE_BAR;

    config.domain_definitions = {mixed};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // Verify domain participation
    ASSERT_FALSE(plan.my_domains.empty());
    const auto &domain = plan.my_domains[0];
    EXPECT_EQ(domain.domain_name, "mixed_tp");
    EXPECT_EQ(domain.devices.size(), 2);
    EXPECT_EQ(domain.weights.size(), 2);
    EXPECT_FLOAT_EQ(domain.weights[0], 0.6f);
    EXPECT_EQ(domain.backend, CollectiveBackendType::PCIE_BAR);

    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

// ============================================================================
// Weight Shard Calculation Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, WeightShard_SingleRank_NoSharding)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &shard = plans[0].weight_shard;

    EXPECT_EQ(shard.shard_index, 0);
    EXPECT_EQ(shard.total_shards, 1);
    EXPECT_FLOAT_EQ(shard.work_fraction, 1.0f);
    EXPECT_FALSE(shard.isSharded());
}

TEST_F(Test__ExecutionPlanBuilder, WeightShard_GlobalTP_EqualSharding)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .addRank(2, "localhost", 2, {{DeviceType::CUDA, 0}})
                       .addRank(3, "localhost", 3, {{DeviceType::CUDA, 0}})
                       .build();

    config.tp_degree = 4;
    config.tp_scope = TPScope::GLOBAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 4);

    for (int r = 0; r < 4; ++r)
    {
        const auto &shard = plans[r].weight_shard;
        EXPECT_TRUE(shard.isSharded());
        EXPECT_EQ(shard.total_shards, 4);
        EXPECT_FLOAT_EQ(shard.work_fraction, 0.25f);
    }
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_ValidSimpleConfig)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_TRUE(errors.empty());
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_PPDegreeExceedsRanks)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    config.pp_degree = 4; // Only 1 rank

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_PPDegreeExceedsLayers)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    ModelConfig tiny_model;
    tiny_model.n_layers = 1;
    tiny_model.n_heads = 1;
    tiny_model.n_kv_heads = 1;
    tiny_model.hidden_size = 64;

    config.pp_degree = 2; // Only 1 layer

    auto errors = builder->validateConfig(config, tiny_model, cluster);
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_UndefinedDomain)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    // Reference domain that doesn't exist
    PPStageDefinition stage;
    stage.stage_id = 0;
    stage.domain_name = "nonexistent";
    stage.first_layer = 0;
    stage.last_layer = 27;

    config.pp_stage_definitions = {stage};

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_OverlappingLayers)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    DomainDefinition gpu0;
    gpu0.name = "gpu0";
    gpu0.devices = {GlobalDeviceAddress::cuda(0, 0)};

    DomainDefinition gpu1;
    gpu1.name = "gpu1";
    gpu1.devices = {GlobalDeviceAddress::cuda(0, 1)};

    config.domain_definitions = {gpu0, gpu1};

    // Overlapping layer ranges
    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "gpu0";
    stage0.first_layer = 0;
    stage0.last_layer = 15; // Overlaps with stage1

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "gpu1";
    stage1.first_layer = 10; // Overlap!
    stage1.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1};

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_FALSE(errors.empty());
}

// ============================================================================
// buildPlanForRank Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlanForRank_ValidRank)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    auto plan = builder->buildPlanForRank(config, model, cluster, 1);

    EXPECT_EQ(plan.rank, 1);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlanForRank_InvalidRank_ReturnsEmpty)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto plan = builder->buildPlanForRank(config, model, cluster, 5);

    // Invalid rank should return empty/default plan
    EXPECT_EQ(plan.rank, 0); // Default
}

// ============================================================================
// ModelConfig Tests
// ============================================================================

TEST(Test__ModelConfig, QWen2_7B_HasCorrectValues)
{
    auto config = ModelConfig::qwen2_7b();

    EXPECT_EQ(config.n_layers, 28);
    EXPECT_EQ(config.n_heads, 28);
    EXPECT_EQ(config.n_kv_heads, 4);
    EXPECT_EQ(config.hidden_size, 3584);
}

TEST(Test__ModelConfig, Validate_ValidConfig)
{
    auto config = ModelConfig::qwen2_7b();
    auto errors = config.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(Test__ModelConfig, Validate_ZeroLayers)
{
    ModelConfig config;
    config.n_layers = 0;
    config.n_heads = 28;
    config.n_kv_heads = 4;
    config.hidden_size = 3584;

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(Test__ModelConfig, ToString_ContainsName)
{
    auto config = ModelConfig::qwen2_7b();
    auto str = config.toString();
    EXPECT_NE(str.find("Qwen2-7B"), std::string::npos);
}

// ============================================================================
// GPU Selection Priority Tests
// ============================================================================
// These tests verify the critical fix where GPU is selected as primary_device
// when GPUs are available in ClusterInventory, rather than defaulting to CPU.

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_WithGPUs_SelectsGPUAsPrimary)
{
    // Regression test: Ensure GPU is selected when available in inventory
    // This was broken when gatherClusterInventory() didn't enumerate GPUs
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // GPU must be selected as primary device, not CPU
    EXPECT_FALSE(plan.primary_device.isCPU())
        << "GPU should be selected as primary_device when available in ClusterInventory";
    EXPECT_EQ(plan.primary_device.device_type, DeviceType::CUDA);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_WithMultipleGPUs_NoTPConfig_SelectsFirstGPU)
{
    // When multiple GPUs exist but no TP is configured, first GPU should be primary
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}, {DeviceType::ROCm, 0}})
                       .build();

    // No TP configuration - single device mode
    config.tp_degree = 1;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_FALSE(plan.primary_device.isCPU());
    EXPECT_EQ(plan.primary_device.device_type, DeviceType::CUDA);
    EXPECT_EQ(plan.primary_device.device_ordinal, 0);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_EmptyInventory_FallsBackToCPU)
{
    // When ClusterInventory has no GPUs, CPU must be selected
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {}) // No GPUs
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_TRUE(plan.primary_device.isCPU())
        << "CPU fallback should be used when no GPUs in ClusterInventory";
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_MixedVendorGPUs_SelectsFirstAvailable)
{
    // Mixed vendor scenario (CUDA + ROCm) - should select first GPU
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::ROCm, 0}})
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // Primary device should be the first GPU (CUDA:0)
    EXPECT_FALSE(plan.primary_device.isCPU());
    // Either vendor is acceptable as long as it's a GPU
    EXPECT_TRUE(plan.primary_device.device_type == DeviceType::CUDA ||
                plan.primary_device.device_type == DeviceType::ROCm);
}

TEST_F(Test__ExecutionPlanBuilder, ClusterInventory_TotalGPUs_MatchesAddedDevices)
{
    // Verify ClusterInventory correctly counts total GPUs
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}})
                       .addRank(1, "remotehost", 1,
                                {{DeviceType::ROCm, 0}})
                       .build();

    EXPECT_EQ(cluster.world_size, 2);
    EXPECT_EQ(cluster.ranks[0].gpus.size(), 2);
    EXPECT_EQ(cluster.ranks[1].gpus.size(), 1);

    // Total GPUs across cluster
    int total_gpus = 0;
    for (const auto &rank : cluster.ranks)
    {
        total_gpus += static_cast<int>(rank.gpus.size());
    }
    EXPECT_EQ(total_gpus, 3);
}
