/**
 * @file Test__HeterogeneousMultiDomainStrategy.cpp
 * @brief Unit tests for HeterogeneousMultiDomainStrategy
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "execution/mpi_orchestration/placement/HeterogeneousMultiDomainStrategy.h"
#include "execution/mpi_orchestration/PlacementStrategy.h"
#include "execution/mpi_orchestration/DeviceInventory.h"

using namespace llaminar2;

// =============================================================================
// Test Fixtures and Helpers
// =============================================================================

class Test__HeterogeneousMultiDomainStrategy : public ::testing::Test
{
protected:
    /**
     * @brief Create a cluster inventory with specified configuration
     */
    ClusterInventory createCluster(
        int num_nodes,
        int ranks_per_node,
        bool with_nvidia = true,
        bool with_amd = false,
        size_t gpu_memory_gb = 16)
    {
        ClusterInventory inventory;
        inventory.world_size = num_nodes * ranks_per_node;
        inventory.node_count = num_nodes;

        int rank = 0;
        for (int node = 0; node < num_nodes; ++node)
        {
            NodeInventory node_inv;
            node_inv.node_id = node;
            node_inv.hostname = "node" + std::to_string(node);

            for (int local = 0; local < ranks_per_node; ++local)
            {
                RankInventory rank_inv;
                rank_inv.rank = rank;
                rank_inv.node_id = node;
                rank_inv.local_rank = local;
                rank_inv.hostname = node_inv.hostname;
                rank_inv.cpu_cores = 32;
                rank_inv.cpu_memory_bytes = 64ULL * 1024 * 1024 * 1024;

                // CPU info
                rank_inv.cpu.type = DeviceType::CPU;
                rank_inv.cpu.compute_units = 32;
                rank_inv.cpu.memory_bytes = rank_inv.cpu_memory_bytes;
                rank_inv.cpu.tflops_fp16 = 0.5f; // ~0.5 TFLOPS for AVX-512

                // GPU info
                int gpu_idx = 0;
                if (with_nvidia)
                {
                    DeviceInfo nvidia_gpu;
                    nvidia_gpu.type = DeviceType::CUDA;
                    nvidia_gpu.local_device_id = gpu_idx++;
                    nvidia_gpu.memory_bytes = gpu_memory_gb * 1024ULL * 1024 * 1024;
                    nvidia_gpu.compute_units = 108; // ~A100
                    nvidia_gpu.tflops_fp16 = 312.0f;
                    nvidia_gpu.tflops_int8 = 624.0f;
                    nvidia_gpu.name = "NVIDIA A100";
                    rank_inv.gpus.push_back(nvidia_gpu);
                }
                if (with_amd)
                {
                    DeviceInfo amd_gpu;
                    amd_gpu.type = DeviceType::ROCm;
                    amd_gpu.local_device_id = gpu_idx++;
                    amd_gpu.memory_bytes = gpu_memory_gb * 1024ULL * 1024 * 1024;
                    amd_gpu.compute_units = 120; // ~MI250X
                    amd_gpu.tflops_fp16 = 383.0f;
                    amd_gpu.tflops_int8 = 383.0f;
                    amd_gpu.name = "AMD MI250X";
                    rank_inv.gpus.push_back(amd_gpu);
                }

                inventory.ranks.push_back(rank_inv);
                node_inv.ranks.push_back(rank);
                node_inv.total_gpus += rank_inv.gpuCount();
                node_inv.total_gpu_memory += rank_inv.totalGPUMemory();
                node_inv.total_cpu_memory += rank_inv.cpu_memory_bytes;
                node_inv.total_cpu_cores += rank_inv.cpu_cores;

                ++rank;
            }

            inventory.nodes.push_back(node_inv);
        }

        // Update totals
        for (const auto &node : inventory.nodes)
        {
            inventory.total_gpus += node.total_gpus;
            inventory.total_gpu_memory += node.total_gpu_memory;
            inventory.total_cpu_memory += node.total_cpu_memory;
        }

        return inventory;
    }

    /**
     * @brief Create PlacementInput from cluster inventory
     */
    PlacementInput createInput(const ClusterInventory &inventory, int n_layers)
    {
        PlacementInput input;
        input.architecture = "qwen2";
        input.n_layers = n_layers;
        input.d_model = 4096;
        input.d_ff = 11008;
        input.vocab_size = 151936;
        input.n_heads = 32;
        input.n_kv_heads = 8;
        input.quant_type = "Q4_0";
        input.estimated_memory_bytes = 7000000000; // 7GB

        input.cluster_inventory = inventory;
        input.world_size = inventory.world_size;
        input.node_count = inventory.node_count;
        input.ranks_per_node = inventory.world_size / std::max(1, inventory.node_count);
        input.any_rank_has_gpu = inventory.hasAnyGPU();
        input.total_gpu_memory = inventory.total_gpu_memory;
        input.total_cpu_memory = inventory.total_cpu_memory;

        return input;
    }
};

// =============================================================================
// Basic Strategy Tests
// =============================================================================

TEST_F(Test__HeterogeneousMultiDomainStrategy, ConstructsWithDefaultConfig)
{
    HeterogeneousMultiDomainStrategy strategy;
    EXPECT_EQ(strategy.name(), "HeterogeneousMultiDomain");
    EXPECT_TRUE(strategy.config().enable_gpu_tp);
    EXPECT_TRUE(strategy.config().enable_cpu_tp);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, ConstructsWithCustomConfig)
{
    HeterogeneousConfig config;
    config.enable_gpu_tp = false;
    config.cpu_compute_fraction = 0.5f;

    HeterogeneousMultiDomainStrategy strategy(config);
    EXPECT_FALSE(strategy.config().enable_gpu_tp);
    EXPECT_FLOAT_EQ(strategy.config().cpu_compute_fraction, 0.5f);
}

// =============================================================================
// Applicability Tests
// =============================================================================

TEST_F(Test__HeterogeneousMultiDomainStrategy, IsApplicableWithMultipleNodes)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);
    auto input = createInput(inventory, 24);

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, IsApplicableWithHeterogeneousGPUs)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 1, true, true); // NVIDIA + AMD
    auto input = createInput(inventory, 24);

    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, IsApplicableWithGPUAndCPU)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 2, true, false);
    auto input = createInput(inventory, 24);

    // GPU + CPU hybrid is applicable
    EXPECT_TRUE(strategy.isApplicable(input));
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, NotApplicableWhenForcedCPUOnly)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, true);
    auto input = createInput(inventory, 24);
    input.force_cpu_only = true;

    EXPECT_FALSE(strategy.isApplicable(input));
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, NotApplicableForSingleNodeHomogeneous)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 1, true, false); // Single node, single GPU type
    auto input = createInput(inventory, 24);

    // Disable CPU TP to make it truly homogeneous
    HeterogeneousConfig config;
    config.enable_cpu_tp = false;
    HeterogeneousMultiDomainStrategy strict_strategy(config);

    EXPECT_FALSE(strict_strategy.isApplicable(input));
}

// =============================================================================
// Domain Assignment Tests
// =============================================================================

TEST_F(Test__HeterogeneousMultiDomainStrategy, ComputeDomainAssignmentsForSingleNode)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 2, true, false);

    auto domains = strategy.computeDomainAssignments(inventory, 24);

    // Should have GPU domain per rank + one CPU domain
    EXPECT_GE(domains.size(), 2u);

    // Check layer coverage
    int total_layers = 0;
    for (const auto &domain : domains)
    {
        total_layers += domain.layerCount();
        EXPECT_TRUE(domain.isValid());
    }
    EXPECT_EQ(total_layers, 24);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, ComputeDomainAssignmentsForMultiNode)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);

    auto domains = strategy.computeDomainAssignments(inventory, 28);

    // Should have domains for both nodes
    int node0_domains = 0, node1_domains = 0;
    for (const auto &domain : domains)
    {
        if (domain.node_id == 0)
            node0_domains++;
        else if (domain.node_id == 1)
            node1_domains++;
    }
    EXPECT_GT(node0_domains, 0);
    EXPECT_GT(node1_domains, 0);

    // Check layer coverage
    int total_layers = 0;
    for (const auto &domain : domains)
    {
        total_layers += domain.layerCount();
    }
    EXPECT_EQ(total_layers, 28);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, GPUDomainsGetMoreLayersThanCPU)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 2, true, false);

    auto domains = strategy.computeDomainAssignments(inventory, 24);

    int gpu_layers = 0, cpu_layers = 0;
    for (const auto &domain : domains)
    {
        if (domain.type == TPDomainType::GPU_INTRA_RANK)
        {
            gpu_layers += domain.layerCount();
        }
        else
        {
            cpu_layers += domain.layerCount();
        }
    }

    // GPU should get more layers (higher compute weight)
    EXPECT_GT(gpu_layers, cpu_layers);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, HeterogeneousGPUsCreateSeparateDomains)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 2, true, true); // NVIDIA + AMD per rank

    auto domains = strategy.computeDomainAssignments(inventory, 24);

    // Each rank should have its own GPU domain
    int gpu_domains = 0;
    for (const auto &domain : domains)
    {
        if (domain.type == TPDomainType::GPU_INTRA_RANK)
        {
            gpu_domains++;
            // Each GPU domain should have both NVIDIA and AMD
            EXPECT_EQ(domain.devices.size(), 2u);
        }
    }
    EXPECT_EQ(gpu_domains, 2); // One per rank
}

// =============================================================================
// Plan Generation Tests
// =============================================================================

TEST_F(Test__HeterogeneousMultiDomainStrategy, GeneratePlanCreatesValidPlan)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);

    auto plan = strategy.generatePlan(inventory, 24);

    EXPECT_EQ(plan.total_layers, 24);
    EXPECT_EQ(plan.world_size, 4);
    EXPECT_EQ(plan.node_count, 2);
    EXPECT_FALSE(plan.domains.empty());
    EXPECT_FALSE(plan.stages.empty());
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, GeneratePlanCreatesPipelineStages)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);

    auto plan = strategy.generatePlan(inventory, 24);

    // Should have one stage per node
    EXPECT_EQ(plan.stages.size(), 2u);

    // Stages should cover all layers
    int covered = 0;
    for (const auto &stage : plan.stages)
    {
        covered += (stage.layer_end - stage.layer_start);
        EXPECT_FALSE(stage.ranks.empty());
    }
    EXPECT_EQ(covered, 24);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, GetDomainForLayerReturnsCorrect)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 2, true, false);

    auto plan = strategy.generatePlan(inventory, 24);

    // Every layer should have a domain
    for (int layer = 0; layer < 24; ++layer)
    {
        const auto *domain = plan.getDomainForLayer(layer);
        ASSERT_NE(domain, nullptr) << "Layer " << layer << " has no domain";
        EXPECT_GE(layer, domain->layer_start);
        EXPECT_LT(layer, domain->layer_end);
    }
}

// =============================================================================
// PlacementPlan Computation Tests
// =============================================================================

TEST_F(Test__HeterogeneousMultiDomainStrategy, ComputeReturnsValidPlan)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);
    auto input = createInput(inventory, 24);

    PlacementPlan plan = strategy.compute(input);

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.n_layers, 24);
    EXPECT_EQ(plan.strategy_name, "HeterogeneousMultiDomain");
    EXPECT_TRUE(plan.has_gpu);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, ComputeAssignsLayersToGPU)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 2, true, false);
    auto input = createInput(inventory, 24);

    PlacementPlan plan = strategy.compute(input);

    // At least some layers should be on GPU
    int gpu_layers = 0;
    for (const auto &layer : plan.layers)
    {
        if (layer.device.isGPU())
        {
            gpu_layers++;
        }
    }
    EXPECT_GT(gpu_layers, 0);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, ComputeHandlesCPUOnlyCluster)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, false, false); // No GPUs
    auto input = createInput(inventory, 24);

    PlacementPlan plan = strategy.compute(input);

    EXPECT_TRUE(plan.isValid());
    EXPECT_FALSE(plan.usesGPU());

    // All layers should be CPU
    for (const auto &layer : plan.layers)
    {
        EXPECT_TRUE(layer.device.isCPU());
    }
}

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(Test__HeterogeneousMultiDomainStrategy, PlanSerializationRoundtrip)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, true);

    auto original = strategy.generatePlan(inventory, 24);

    // Serialize
    auto data = original.serialize();
    EXPECT_FALSE(data.empty());

    // Deserialize
    auto restored = HeterogeneousPlan::deserialize(data);

    // Verify
    EXPECT_EQ(restored.total_layers, original.total_layers);
    EXPECT_EQ(restored.world_size, original.world_size);
    EXPECT_EQ(restored.node_count, original.node_count);
    EXPECT_EQ(restored.stages.size(), original.stages.size());
    EXPECT_EQ(restored.domains.size(), original.domains.size());

    // Check domain details
    for (size_t i = 0; i < original.domains.size(); ++i)
    {
        const auto &orig = original.domains[i];
        const auto &rest = restored.domains[i];

        EXPECT_EQ(rest.domain_id, orig.domain_id);
        EXPECT_EQ(rest.type, orig.type);
        EXPECT_EQ(rest.node_id, orig.node_id);
        EXPECT_EQ(rest.layer_start, orig.layer_start);
        EXPECT_EQ(rest.layer_end, orig.layer_end);
        EXPECT_EQ(rest.ranks.size(), orig.ranks.size());
        EXPECT_EQ(rest.devices.size(), orig.devices.size());
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(Test__HeterogeneousMultiDomainStrategy, HandlesSingleRankCluster)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 1, true, false);
    auto input = createInput(inventory, 24);

    PlacementPlan plan = strategy.compute(input);

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.world_size, 1);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, HandlesVeryFewLayers)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);
    auto input = createInput(inventory, 4); // Very few layers

    PlacementPlan plan = strategy.compute(input);

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.n_layers, 4);
}

TEST_F(Test__HeterogeneousMultiDomainStrategy, HandlesMoreDomainsThanLayers)
{
    HeterogeneousConfig config;
    config.min_layers_per_domain = 1;
    HeterogeneousMultiDomainStrategy strategy(config);

    auto inventory = createCluster(2, 4, true, false); // 8 ranks = potentially 8+ domains
    auto input = createInput(inventory, 4);            // Only 4 layers

    auto domains = strategy.computeDomainAssignments(inventory, 4);

    // Check that all layers are covered (some domains may have 0 layers)
    int total_layers = 0;
    for (const auto &domain : domains)
    {
        total_layers += domain.layerCount();
    }
    EXPECT_EQ(total_layers, 4);
}

// =============================================================================
// Determinism Tests
// =============================================================================

TEST_F(Test__HeterogeneousMultiDomainStrategy, IsDeterministic)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, true);
    auto input = createInput(inventory, 24);

    // Compute plan multiple times
    PlacementPlan plan1 = strategy.compute(input);
    PlacementPlan plan2 = strategy.compute(input);
    PlacementPlan plan3 = strategy.compute(input);

    // All should be identical
    EXPECT_EQ(plan1.layers.size(), plan2.layers.size());
    EXPECT_EQ(plan2.layers.size(), plan3.layers.size());

    for (size_t i = 0; i < plan1.layers.size(); ++i)
    {
        EXPECT_EQ(plan1.layers[i].device, plan2.layers[i].device);
        EXPECT_EQ(plan2.layers[i].device, plan3.layers[i].device);
        EXPECT_EQ(plan1.layers[i].owner_rank, plan2.layers[i].owner_rank);
        EXPECT_EQ(plan2.layers[i].owner_rank, plan3.layers[i].owner_rank);
    }
}

// =============================================================================
// DomainAssignment Structure Tests
// =============================================================================

TEST(Test__DomainAssignment, IsValidRequiresDevicesAndRanks)
{
    DomainAssignment domain;
    domain.domain_id = 0;
    domain.layer_start = 0;
    domain.layer_end = 10;

    // Missing devices and ranks
    EXPECT_FALSE(domain.isValid());

    // Add devices but no ranks
    domain.devices.push_back(DeviceId::cuda(0));
    EXPECT_FALSE(domain.isValid());

    // Add ranks - now valid
    domain.ranks.push_back(0);
    EXPECT_TRUE(domain.isValid());
}

TEST(Test__DomainAssignment, LayerCountCalculatesCorrectly)
{
    DomainAssignment domain;
    domain.layer_start = 5;
    domain.layer_end = 15;

    EXPECT_EQ(domain.layerCount(), 10);
}

TEST(Test__DomainAssignment, ToStringProducesOutput)
{
    DomainAssignment domain;
    domain.domain_id = 0;
    domain.type = TPDomainType::GPU_INTRA_RANK;
    domain.ranks = {0};
    domain.devices = {DeviceId::cuda(0), DeviceId::rocm(1)};
    domain.node_id = 0;
    domain.layer_start = 0;
    domain.layer_end = 12;
    domain.compute_weight = 100.0f;

    std::string str = domain.toString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("GPU_TP"), std::string::npos);
    EXPECT_NE(str.find("layers="), std::string::npos);
}

// =============================================================================
// HeterogeneousPlan Structure Tests
// =============================================================================

TEST(Test__HeterogeneousPlan, ToStringProducesOutput)
{
    HeterogeneousPlan plan;
    plan.total_layers = 24;
    plan.world_size = 4;
    plan.node_count = 2;

    std::string str = plan.toString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("HeterogeneousPlan"), std::string::npos);
    EXPECT_NE(str.find("total_layers"), std::string::npos);
}
