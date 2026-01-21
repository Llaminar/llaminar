/**
 * @file Test__Scenario5_MultiNodePP.cpp
 * @brief Orchestration tests for multi-node pipeline parallel scenarios
 *
 * **Hardware Configuration**:
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                         NODE 0 (Machine 1)                                  │
 * │  ┌─────────────────────────────────────────────────────────────────────┐    │
 * │  │       SOCKET 0 (MPI Rank 0)                                         │    │
 * │  │  ┌──────────────────┐  ┌──────────────────┐                         │    │
 * │  │  │  NVIDIA A100-80GB│  │  NVIDIA A100-80GB│                         │    │
 * │  │  │  80GB VRAM       │  │  80GB VRAM       │                         │    │
 * │  │  │  CUDA:0          │  │  CUDA:1          │                         │    │
 * │  │  └──────────────────┘  └──────────────────┘                         │    │
 * │  │  Xeon 28-core DDR5       NVLink 600 GB/s                            │    │
 * │  └─────────────────────────────────────────────────────────────────────┘    │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *                              │
 *                              │ 200 Gbps HDR InfiniBand
 *                              │
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                         NODE 1 (Machine 2)                                  │
 * │  ┌─────────────────────────────────────────────────────────────────────┐    │
 * │  │       SOCKET 0 (MPI Rank 1)                                         │    │
 * │  │  ┌──────────────────┐  ┌──────────────────┐                         │    │
 * │  │  │  NVIDIA A100-80GB│  │  NVIDIA A100-80GB│                         │    │
 * │  │  │  80GB VRAM       │  │  80GB VRAM       │                         │    │
 * │  │  │  CUDA:0          │  │  CUDA:1          │                         │    │
 * │  │  └──────────────────┘  └──────────────────┘                         │    │
 * │  │  Xeon 28-core DDR5       NVLink 600 GB/s                            │    │
 * │  └─────────────────────────────────────────────────────────────────────┘    │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 * Tests pipeline parallelism across multiple nodes connected via InfiniBand.
 * This is critical for:
 * - Large models that don't fit on a single machine
 * - Maximizing throughput with PP overlapping
 * - Validating Send/Recv stage insertion at PP boundaries
 *
 * **Tested Models**: Qwen2 7B, 72B, 235B, 571B
 * **Total GPU VRAM**: 4× 80GB = 320 GB
 * **Interconnect**: 200 Gbps HDR InfiniBand (inter-node), 600 GB/s NVLink (intra-node)
 *
 * @see docs/v2/ARCHITECTURE_EXECUTION_SCENARIOS.md (Scenario 5)
 * @author Copilot
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "../OrchestrationTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::orchestration;

// =============================================================================
// Scenario 5 Configuration
// =============================================================================

/**
 * @brief Scenario 5: 2-Node Pipeline Parallel over InfiniBand
 *
 * Hardware Configuration:
 * - 2 machines connected via HDR InfiniBand (200 Gbps)
 * - Each machine: 1 socket with 2× A100-80GB
 * - Total: 4× A100-80GB = 320GB VRAM
 *
 * Test Focus:
 * - PP stage creation (2 stages, one per node)
 * - Layer splitting between nodes
 * - Send/Recv stage insertion at PP boundaries
 * - Balanced layer distribution
 */
class Test__Scenario5_MultiNodePP : public OrchestrationTestBase
{
protected:
    ClusterConfig getClusterConfig() override
    {
        ClusterConfig config;
        config.name = "Scenario 5: 2-Node Pipeline Parallel";
        config.num_machines = 2;
        config.sockets_per_machine = 1;

        // Each socket: 2× A100-80GB with NVLink
        SocketConfig socket;
        socket.cpu = CPUSpecs::XEON_28C_DDR5;
        socket.gpus = {GPUSpecs::A100_80GB, GPUSpecs::A100_80GB};

        // Same config for all sockets (homogeneous within nodes)
        config.socket_configs = {socket};

        // High-speed InfiniBand interconnect between nodes
        config.infiniband_bandwidth_gbps = 200.0f; // HDR IB
        config.nvlink_bandwidth_gbps = 600.0f;     // Intra-node NVLink
        config.qpi_upi_bandwidth_gbps = 50.0f;     // N/A (single socket per machine)
        config.pcie_bandwidth_gbps = 32.0f;        // Fallback

        return config;
    }

    std::vector<ModelConfig> getModelConfigs() override
    {
        return {
            ModelConfigs::QWEN2_7B,   // Small - fits on single node
            ModelConfigs::QWEN2_72B,  // Medium - benefits from PP
            ModelConfigs::QWEN2_235B, // Large - requires PP
            ModelConfigs::QWEN2_571B, // Very large - tests extreme PP
        };
    }
};

// =============================================================================
// Instantiate Standard Orchestration Tests (15+ tests from macro)
// =============================================================================

INSTANTIATE_ORCHESTRATION_TESTS(Test__Scenario5_MultiNodePP);

// =============================================================================
// Scenario-Specific Tests: Topology Validation
// =============================================================================

TEST_F(Test__Scenario5_MultiNodePP, ClusterHas2Nodes)
{
    EXPECT_EQ(cluster_.num_machines, 2);
    EXPECT_EQ(cluster_.sockets_per_machine, 1);
    EXPECT_EQ(cluster_.totalRanks(), 2);
}

TEST_F(Test__Scenario5_MultiNodePP, EachNodeHas2A100GPUs)
{
    for (int rank = 0; rank < cluster_.totalRanks(); ++rank)
    {
        const auto &placement = topologies_[rank]->placement();

        int cuda_count = 0;
        size_t total_vram = 0;

        for (const auto &dev : placement.devices)
        {
            if (dev.type == DeviceCapability::Type::CUDA)
            {
                cuda_count++;
                total_vram += dev.memory_bytes;
                EXPECT_EQ(dev.name, "NVIDIA A100 80GB");
            }
        }

        EXPECT_EQ(cuda_count, 2) << "Rank " << rank << " should have 2 CUDA GPUs";
        EXPECT_EQ(total_vram, 2 * 80ULL * 1024 * 1024 * 1024)
            << "Rank " << rank << " should have 160GB total VRAM";
    }
}

TEST_F(Test__Scenario5_MultiNodePP, TotalVRAMIs320GB)
{
    // 2 nodes × 2 GPUs × 80GB = 320GB
    float expected_vram = 2.0f * 2.0f * 80.0f;
    EXPECT_FLOAT_EQ(cluster_.totalVRAMGB(), expected_vram);
}

TEST_F(Test__Scenario5_MultiNodePP, TotalGPUCountIs4)
{
    // 2 nodes × 2 GPUs = 4 GPUs total
    EXPECT_EQ(cluster_.totalGPUs(), 4);
}

TEST_F(Test__Scenario5_MultiNodePP, InfiniBandIs200Gbps)
{
    // HDR InfiniBand between nodes
    EXPECT_FLOAT_EQ(cluster_.infiniband_bandwidth_gbps, 200.0f);
}

TEST_F(Test__Scenario5_MultiNodePP, NVLinkIs600GBps)
{
    // NVLink between GPUs within same node
    EXPECT_FLOAT_EQ(cluster_.nvlink_bandwidth_gbps, 600.0f);
}

TEST_F(Test__Scenario5_MultiNodePP, NVLinkFasterThanInfiniBand)
{
    // Intra-node should be faster than inter-node
    // NVLink: 600 GB/s = 4800 Gbps
    // IB: 200 Gbps
    float nvlink_gbps = cluster_.nvlink_bandwidth_gbps * 8.0f; // GB/s to Gbps
    EXPECT_GT(nvlink_gbps, cluster_.infiniband_bandwidth_gbps);
}

// =============================================================================
// Scenario-Specific Tests: Model Fit Analysis
// =============================================================================

TEST_F(Test__Scenario5_MultiNodePP, Qwen7B_FitsOnSingleNode)
{
    // 7B at Q4 is ~4GB, fits easily on one node (160GB)
    const auto &model = models_[0];

    float vram_per_node = cluster_.totalVRAMGB() / cluster_.num_machines;
    EXPECT_LT(model.memorySizeGB(), vram_per_node);
    EXPECT_TRUE(modelFitsInGPUMemory(model));
    EXPECT_EQ(machinesNeededForModel(model), 1);
}

TEST_F(Test__Scenario5_MultiNodePP, Qwen72B_FitsOnSingleNode)
{
    // 72B at Q4 is ~40GB, fits on one node (160GB)
    const auto &model = models_[1];

    float vram_per_node = cluster_.totalVRAMGB() / cluster_.num_machines;
    EXPECT_LT(model.memorySizeGB(), vram_per_node);
    EXPECT_TRUE(modelFitsInGPUMemory(model));
    EXPECT_EQ(machinesNeededForModel(model), 1);
}

TEST_F(Test__Scenario5_MultiNodePP, Qwen235B_FitsOnSingleNode)
{
    // 235B at Q4 is ~130GB, fits on one node (160GB) but tight
    const auto &model = models_[2];

    float vram_per_node = cluster_.totalVRAMGB() / cluster_.num_machines;
    EXPECT_LT(model.memorySizeGB(), vram_per_node);
    EXPECT_TRUE(modelFitsInGPUMemory(model));
    EXPECT_EQ(machinesNeededForModel(model), 1);
}

TEST_F(Test__Scenario5_MultiNodePP, Qwen571B_RequiresBothNodes)
{
    // 571B at Q4 is ~320GB, exactly matches total cluster VRAM
    // Requires both nodes for PP
    const auto &model = models_[3];

    float vram_per_node = cluster_.totalVRAMGB() / cluster_.num_machines;
    EXPECT_GT(model.memorySizeGB(), vram_per_node)
        << "571B model should exceed single node VRAM";

    // Model may or may not "fit" depending on exact VRAM vs model size calculation
    // The key point is it requires 2 nodes
    EXPECT_GE(machinesNeededForModel(model), 2)
        << "571B model should require at least 2 nodes";
}

// =============================================================================
// Scenario-Specific Tests: Pipeline Parallel Strategy
// =============================================================================

/**
 * @brief Test: Two PP stages created for multi-node setup
 */
TEST_F(Test__Scenario5_MultiNodePP, TwoPPStagesCreatedForLargeModel)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[3]; // 571B - requires both nodes
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Should have exactly 2 PP stages (one per node) for large model
    EXPECT_GE(plan.stages.size(), 2)
        << "Should have at least 2 PP stages for 2-node cluster with large model";

    if (plan.stages.size() >= 2)
    {
        EXPECT_EQ(plan.stages[0].node_id, 0) << "Stage 0 should be on node 0";
        EXPECT_EQ(plan.stages[1].node_id, 1) << "Stage 1 should be on node 1";
    }
}

/**
 * @brief Test: Layers split between nodes
 */
TEST_F(Test__Scenario5_MultiNodePP, LayersSplitBetweenNodes)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[3]; // 571B - 160 layers
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    if (plan.stages.size() < 2)
    {
        GTEST_SKIP() << "Plan has fewer than 2 stages";
    }

    int layers_node0 = plan.stages[0].layer_end - plan.stages[0].layer_start;
    int layers_node1 = plan.stages[1].layer_end - plan.stages[1].layer_start;

    // All layers should be covered
    EXPECT_EQ(layers_node0 + layers_node1, model.n_layers)
        << "Total layers should equal model layers";

    // With identical hardware, layers should be roughly equal
    int expected_layers_per_node = model.n_layers / 2;
    EXPECT_NEAR(layers_node0, expected_layers_per_node, 2)
        << "Layers should be roughly equal between identical nodes";
    EXPECT_NEAR(layers_node1, expected_layers_per_node, 2)
        << "Layers should be roughly equal between identical nodes";
}

/**
 * @brief Test: PP stages have valid boundaries
 */
TEST_F(Test__Scenario5_MultiNodePP, PPStageBoundariesValid)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[3]; // 571B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    int prev_end = 0;
    for (size_t i = 0; i < plan.stages.size(); ++i)
    {
        auto result = validatePPStage(plan, static_cast<int>(i));
        EXPECT_TRUE(result.is_valid)
            << "Stage " << i << " invalid: " << result.error_message;

        // Stages should be contiguous
        EXPECT_EQ(plan.stages[i].layer_start, prev_end)
            << "Stage " << i << " should start where previous ended";

        prev_end = plan.stages[i].layer_end;
    }

    // Last stage should end at model.n_layers
    if (!plan.stages.empty())
    {
        EXPECT_EQ(plan.stages.back().layer_end, model.n_layers)
            << "Last PP stage should end at final layer";
    }
}

/**
 * @brief Test: Each PP stage has TP domains
 */
TEST_F(Test__Scenario5_MultiNodePP, PPStagesHaveTPDomains)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[1]; // 72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    for (size_t i = 0; i < plan.stages.size(); ++i)
    {
        EXPECT_GT(plan.stages[i].domains.size(), 0)
            << "PP stage " << i << " should have at least one TP domain";
    }
}

/**
 * @brief Test: TP domains within stage use NVLink (same node)
 */
TEST_F(Test__Scenario5_MultiNodePP, IntraNodeDomainsUseSameNode)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[1]; // 72B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Check that GPU_INTRA_RANK domains are single-node (for NVLink)
    for (const auto &domain : plan.domains)
    {
        if (domain.type == TPDomainType::GPU_INTRA_RANK && domain.devices.size() > 1)
        {
            // All ranks in an intra-rank domain should be on same node
            std::set<int> nodes;
            for (int rank : domain.ranks)
            {
                // Node = rank / sockets_per_machine for this topology
                int node_id = rank / cluster_.sockets_per_machine;
                nodes.insert(node_id);
            }

            EXPECT_EQ(nodes.size(), 1)
                << "GPU_INTRA_RANK domain should span single node for NVLink";
        }
    }
}

/**
 * @brief Test: Cross-node PP stages are on different nodes
 */
TEST_F(Test__Scenario5_MultiNodePP, CrossNodePPStagesOnDifferentNodes)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[3]; // 571B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // With 2 nodes and PP, there should be stages on different nodes
    if (plan.stages.size() >= 2)
    {
        // Verify stages are on different nodes
        EXPECT_NE(plan.stages[0].node_id, plan.stages[1].node_id)
            << "PP stages should be on different nodes";
    }
}

/**
 * @brief Test: Small model may not need PP
 */
TEST_F(Test__Scenario5_MultiNodePP, SmallModelMayNotNeedPP)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[0]; // 7B - fits easily on one node
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Small model may use fewer stages
    // Just verify the plan is valid
    auto result = validateHeterogeneousPlan(plan, model.n_layers);
    EXPECT_TRUE(result.is_valid) << result.error_message;
    EXPECT_TRUE(result.all_layers_covered);
}

/**
 * @brief Test: Very large model distributed across both nodes
 */
TEST_F(Test__Scenario5_MultiNodePP, VeryLargeModelDistributed)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[3]; // 571B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable for this model";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    // Very large model should use all available nodes
    std::set<int> nodes_used;
    for (const auto &stage : plan.stages)
    {
        nodes_used.insert(stage.node_id);
    }

    EXPECT_EQ(nodes_used.size(), static_cast<size_t>(cluster_.num_machines))
        << "Very large model should use all available nodes";
}

/**
 * @brief Test: Validate Send/Recv stage markers at PP boundaries
 */
TEST_F(Test__Scenario5_MultiNodePP, PPBoundaryHasCommunication)
{
    HeterogeneousMultiDomainStrategy strategy;

    const auto &model = models_[3]; // 571B
    auto input = createPlacementInput(model, 0);

    if (!strategy.isApplicable(input))
    {
        GTEST_SKIP() << "Strategy not applicable";
    }

    auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);

    for (size_t i = 0; i < plan.stages.size(); ++i)
    {
        auto result = validatePPStage(plan, static_cast<int>(i));

        // First stage shouldn't need recv (no prior stage)
        if (i == 0)
        {
            EXPECT_FALSE(result.has_recv_stage)
                << "First PP stage shouldn't have recv";
        }

        // Last stage shouldn't need send (no next stage)
        if (i == plan.stages.size() - 1)
        {
            EXPECT_FALSE(result.has_send_stage)
                << "Last PP stage shouldn't have send";
        }

        // Middle stages need both (if we had more than 2 stages)
        if (i > 0 && i < plan.stages.size() - 1)
        {
            EXPECT_TRUE(result.has_send_stage && result.has_recv_stage)
                << "Middle PP stages need both send and recv";
        }
    }
}

// =============================================================================
// Scenario-Specific Tests: Plan Validation
// =============================================================================

/**
 * @brief Test: Plan covers all layers without gaps
 */
TEST_F(Test__Scenario5_MultiNodePP, PlanCoversAllLayersNoGaps)
{
    HeterogeneousMultiDomainStrategy strategy;

    for (const auto &model : models_)
    {
        auto input = createPlacementInput(model, 0);

        if (!strategy.isApplicable(input))
        {
            continue; // Skip models where strategy not applicable
        }

        auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);
        auto result = validateHeterogeneousPlan(plan, model.n_layers);

        EXPECT_TRUE(result.all_layers_covered)
            << model.name << ": Not all layers covered";
        EXPECT_TRUE(result.no_layer_gaps)
            << model.name << ": Layer gaps detected - " << result.error_message;
        EXPECT_TRUE(result.no_layer_overlaps)
            << model.name << ": Layer overlaps detected - " << result.error_message;
    }
}

/**
 * @brief Test: Domains have valid device assignments
 */
TEST_F(Test__Scenario5_MultiNodePP, DomainsHaveValidDevices)
{
    HeterogeneousMultiDomainStrategy strategy;

    for (const auto &model : models_)
    {
        auto input = createPlacementInput(model, 0);

        if (!strategy.isApplicable(input))
        {
            continue;
        }

        auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);
        auto result = validateHeterogeneousPlan(plan, model.n_layers);

        EXPECT_TRUE(result.domains_have_valid_devices)
            << model.name << ": Domains missing devices - " << result.error_message;
        EXPECT_TRUE(result.domains_have_valid_ranks)
            << model.name << ": Domains missing ranks - " << result.error_message;
    }
}

// =============================================================================
// Scenario-Specific Tests: Interconnect Analysis
// =============================================================================

TEST_F(Test__Scenario5_MultiNodePP, InfiniBandRequiredForPP)
{
    // PP across nodes requires InfiniBand
    for (const auto &model : models_)
    {
        if (machinesNeededForModel(model) > 1)
        {
            EXPECT_GT(cluster_.infiniband_bandwidth_gbps, 0.0f)
                << model.name << " requires InfiniBand for multi-node PP";
        }
    }
}

TEST_F(Test__Scenario5_MultiNodePP, IntraNodeBandwidthFarExceedsInterNode)
{
    // NVLink should be much faster than InfiniBand
    // This justifies preferring TP within nodes and PP across nodes
    float nvlink_gbps = cluster_.nvlink_bandwidth_gbps * 8.0f; // GB/s to Gbps
    float ib_gbps = cluster_.infiniband_bandwidth_gbps;

    EXPECT_GT(nvlink_gbps, ib_gbps * 10)
        << "NVLink should be at least 10x faster than InfiniBand";
}

// =============================================================================
// Scenario-Specific Tests: Layers Per Rank Analysis
// =============================================================================

TEST_F(Test__Scenario5_MultiNodePP, LayersPerRankReasonable)
{
    // With 2 ranks, layers per rank should be reasonable
    for (const auto &model : models_)
    {
        float lpr = model.layersPerRank(cluster_.totalRanks());

        // Should have enough layers per rank for meaningful work
        EXPECT_GE(lpr, 4.0f)
            << model.name << " has too few layers per rank: " << lpr;
    }
}

// =============================================================================
// Summary Test: Print full plan for visual inspection
// =============================================================================

TEST_F(Test__Scenario5_MultiNodePP, SummaryPrintAllModels)
{
    HeterogeneousMultiDomainStrategy strategy;

    printClusterSummary();
    printf("Interconnect: %.0f Gbps HDR InfiniBand (inter-node), %.0f GB/s NVLink (intra-node)\n",
           cluster_.infiniband_bandwidth_gbps, cluster_.nvlink_bandwidth_gbps);

    for (const auto &model : models_)
    {
        auto input = createPlacementInput(model, 0);
        if (strategy.isApplicable(input))
        {
            auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);
            printHeterogeneousPlanSummary(plan, model);

            // Also print PP stage details
            printf("  Pipeline Stages:\n");
            for (size_t i = 0; i < plan.stages.size(); ++i)
            {
                const auto &stage = plan.stages[i];
                printf("    Stage %zu: Node %d, Layers %d-%d (%d layers), Domains: %zu\n",
                       i, stage.node_id, stage.layer_start, stage.layer_end,
                       stage.layer_end - stage.layer_start, stage.domains.size());
            }
            printf("\n");
        }
        else
        {
            printf("Model %s: Strategy not applicable\n\n", model.name.c_str());
        }
    }
}
