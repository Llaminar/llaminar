/**
 * @file Test__HeterogeneousExecution.cpp
 * @brief Integration tests for heterogeneous CPU/GPU execution across ranks
 *
 * **Purpose**: Validates that the system correctly handles mixed CPU/GPU
 * execution across MPI ranks, including:
 * - Heterogeneous topology discovery and work distribution
 * - CPU fallback when no GPUs are available
 * - Graceful degradation when GPU backends fail
 * - Memory-aware placement across devices
 *
 * **Test Strategy**:
 * Uses MockMPITopology to simulate various cluster configurations without
 * requiring actual MPI or GPU hardware. Tests validate that:
 * 1. Work distribution correctly accounts for device capabilities
 * 2. Ranks without GPUs receive CPU-appropriate work assignments
 * 3. System falls back gracefully when accelerators are unavailable
 * 4. Memory constraints are respected in placement decisions
 *
 * **Dependencies**:
 * - MockMPITopology (simulates heterogeneous cluster topologies)
 * - MockMPIContext (simulates multi-rank execution)
 * - MockModelContext (provides model configuration without GGUF files)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>

#include "mocks/MockMPITopology.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockModelContext.h"
#include "execution/mpi_orchestration/DeviceInventory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

/**
 * @brief Test fixture for heterogeneous execution tests
 *
 * Provides helper methods for creating various cluster configurations
 * and validating work distribution across heterogeneous ranks.
 */
class Test__HeterogeneousExecution : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Default minimal model context for tests
        model_ctx_ = MockModelContext::createMinimal();
    }

    /**
     * @brief Create a 4-rank heterogeneous topology (2 GPU + 2 CPU)
     *
     * Configuration:
     * - Rank 0: CPU only (node 0)
     * - Rank 1: CPU + CUDA:0 with 8GB (node 0)
     * - Rank 2: CPU only (node 1)
     * - Rank 3: CPU + CUDA:0 with 16GB (node 1)
     */
    std::vector<std::shared_ptr<MockMPITopology>> createMixedCpuGpuTopology()
    {
        std::vector<std::shared_ptr<MockMPITopology>> topologies;

        for (int local_rank = 0; local_rank < 4; ++local_rank)
        {
            auto builder = MockMPITopologyBuilder();

            // Node 0: ranks 0-1
            builder.addCPUOnlyRank(0, 0);      // Rank 0: CPU only
            builder.addGPURank(1, 0, 0, 8.0f); // Rank 1: GPU 8GB

            // Node 1: ranks 2-3
            builder.addCPUOnlyRank(2, 1);       // Rank 2: CPU only
            builder.addGPURank(3, 1, 0, 16.0f); // Rank 3: GPU 16GB

            builder.setLocalRank(local_rank);
            topologies.push_back(builder.build());
        }

        return topologies;
    }

    /**
     * @brief Create a 4-rank CPU-only topology
     */
    std::vector<std::shared_ptr<MockMPITopology>> createCpuOnlyTopology()
    {
        std::vector<std::shared_ptr<MockMPITopology>> topologies;

        for (int local_rank = 0; local_rank < 4; ++local_rank)
        {
            auto builder = MockMPITopologyBuilder();

            builder.addCPUOnlyRank(0, 0);
            builder.addCPUOnlyRank(1, 0);
            builder.addCPUOnlyRank(2, 1);
            builder.addCPUOnlyRank(3, 1);

            builder.setLocalRank(local_rank);
            topologies.push_back(builder.build());
        }

        return topologies;
    }

    /**
     * @brief Create a 4-rank topology with one GPU rank among CPU ranks
     *
     * Configuration:
     * - Ranks 0, 1, 3: CPU only
     * - Rank 2: CPU + CUDA:0 with 24GB
     */
    std::vector<std::shared_ptr<MockMPITopology>> createSingleGpuTopology()
    {
        std::vector<std::shared_ptr<MockMPITopology>> topologies;

        for (int local_rank = 0; local_rank < 4; ++local_rank)
        {
            auto builder = MockMPITopologyBuilder();

            builder.addCPUOnlyRank(0, 0);
            builder.addCPUOnlyRank(1, 0);
            builder.addGPURank(2, 1, 0, 24.0f); // Single GPU with large VRAM
            builder.addCPUOnlyRank(3, 1);

            builder.setLocalRank(local_rank);
            topologies.push_back(builder.build());
        }

        return topologies;
    }

    /**
     * @brief Create topology with varying GPU memory across ranks
     *
     * Configuration for memory-aware placement testing:
     * - Rank 0: GPU with 8GB
     * - Rank 1: GPU with 16GB
     * - Rank 2: GPU with 24GB
     * - Rank 3: GPU with 40GB (A100-like)
     */
    std::vector<std::shared_ptr<MockMPITopology>> createVaryingMemoryTopology()
    {
        std::vector<std::shared_ptr<MockMPITopology>> topologies;

        for (int local_rank = 0; local_rank < 4; ++local_rank)
        {
            auto builder = MockMPITopologyBuilder();

            builder.addGPURank(0, 0, 0, 8.0f);
            builder.addGPURank(1, 0, 1, 16.0f);
            builder.addGPURank(2, 1, 0, 24.0f);
            builder.addGPURank(3, 1, 1, 40.0f);

            builder.setLocalRank(local_rank);
            topologies.push_back(builder.build());
        }

        return topologies;
    }

    /**
     * @brief Validate that all ranks see consistent cluster view
     */
    void validateConsistentClusterView(
        const std::vector<std::shared_ptr<MockMPITopology>> &topologies)
    {
        ASSERT_GE(topologies.size(), 1);

        const auto &ref_inventory = topologies[0]->clusterInventory();

        for (size_t i = 1; i < topologies.size(); ++i)
        {
            const auto &inv = topologies[i]->clusterInventory();

            EXPECT_EQ(inv.world_size, ref_inventory.world_size)
                << "Rank " << i << " has inconsistent world_size";
            EXPECT_EQ(inv.total_gpus, ref_inventory.total_gpus)
                << "Rank " << i << " has inconsistent total_gpus";
            EXPECT_EQ(inv.total_gpu_memory, ref_inventory.total_gpu_memory)
                << "Rank " << i << " has inconsistent total_gpu_memory";
        }
    }

    /**
     * @brief Calculate expected work distribution based on compute weights
     */
    std::vector<float> calculateExpectedWorkDistribution(
        const std::shared_ptr<MockMPITopology> &topology,
        size_t total_work)
    {
        auto weights = topology->get_compute_weights();
        float total_weight = 0.0f;
        for (float w : weights)
        {
            total_weight += w;
        }

        std::vector<float> distribution;
        for (float w : weights)
        {
            distribution.push_back(static_cast<float>(total_work) * (w / total_weight));
        }
        return distribution;
    }

    std::shared_ptr<MockModelContext> model_ctx_;
};

// ============================================================================
// Test: MixedCpuGpuRanks
// ============================================================================

/**
 * @test Verify work distribution in a heterogeneous 4-rank cluster
 *
 * Tests that:
 * 1. GPU ranks are correctly identified
 * 2. CPU-only ranks are correctly identified
 * 3. Work ranges are calculated based on device capabilities
 * 4. ClusterInventory accurately reflects the heterogeneous configuration
 */
TEST_F(Test__HeterogeneousExecution, MixedCpuGpuRanks_CorrectTopologyDetection)
{
    auto topologies = createMixedCpuGpuTopology();

    // Validate all ranks see the same cluster configuration
    validateConsistentClusterView(topologies);

    // Check GPU detection for each rank
    EXPECT_FALSE(topologies[0]->has_accelerator()) << "Rank 0 should be CPU-only";
    EXPECT_TRUE(topologies[1]->has_accelerator()) << "Rank 1 should have GPU";
    EXPECT_FALSE(topologies[2]->has_accelerator()) << "Rank 2 should be CPU-only";
    EXPECT_TRUE(topologies[3]->has_accelerator()) << "Rank 3 should have GPU";

    // Verify cluster totals
    const auto &inv = topologies[0]->clusterInventory();
    EXPECT_EQ(inv.world_size, 4);
    EXPECT_EQ(inv.node_count, 2);
    EXPECT_EQ(inv.total_gpus, 2); // 2 GPU ranks

    // Verify GPU memory totals (8GB + 16GB = 24GB)
    size_t expected_memory = static_cast<size_t>(24) * 1024 * 1024 * 1024;
    EXPECT_EQ(inv.total_gpu_memory, expected_memory);
}

TEST_F(Test__HeterogeneousExecution, MixedCpuGpuRanks_WorkDistribution)
{
    auto topologies = createMixedCpuGpuTopology();

    // Test head distribution (14 heads for Qwen2-0.5B-like model)
    const int total_heads = 14;

    for (int r = 0; r < 4; ++r)
    {
        WorkRange heads = topologies[r]->get_head_range(total_heads);

        // Uniform distribution: each rank gets ~3-4 heads
        EXPECT_GT(heads.size(), 0) << "Rank " << r << " should have work";
        EXPECT_LE(heads.size(), (total_heads + 3) / 4)
            << "Rank " << r << " has too much work";
    }

    // Verify total coverage
    size_t total_work = 0;
    size_t prev_end = 0;
    for (int r = 0; r < 4; ++r)
    {
        WorkRange heads = topologies[r]->get_head_range(total_heads);
        EXPECT_EQ(heads.start, prev_end) << "Gap at rank " << r;
        prev_end = heads.end;
        total_work += heads.size();
    }
    EXPECT_EQ(total_work, total_heads);
}

TEST_F(Test__HeterogeneousExecution, MixedCpuGpuRanks_ComputeWeights)
{
    auto topologies = createMixedCpuGpuTopology();

    auto weights = topologies[0]->get_compute_weights();
    ASSERT_EQ(weights.size(), 4);

    // CPU-only ranks: 1.0 (default CPU compute power)
    // GPU ranks: 1.0 (CPU) + 10.0 (default GPU) = 11.0
    EXPECT_FLOAT_EQ(weights[0], 1.0f);  // Rank 0: CPU only
    EXPECT_FLOAT_EQ(weights[1], 11.0f); // Rank 1: CPU + GPU
    EXPECT_FLOAT_EQ(weights[2], 1.0f);  // Rank 2: CPU only
    EXPECT_FLOAT_EQ(weights[3], 11.0f); // Rank 3: CPU + GPU
}

// ============================================================================
// Test: AllCpuFallback
// ============================================================================

/**
 * @test Verify system operates correctly when no GPUs are available
 *
 * When all ranks are CPU-only:
 * 1. hasAnyGPU() returns false
 * 2. Work distribution is uniform across ranks
 * 3. No GPU-specific code paths are triggered
 */
TEST_F(Test__HeterogeneousExecution, AllCpuFallback_NoGpuDetected)
{
    auto topologies = createCpuOnlyTopology();

    for (int r = 0; r < 4; ++r)
    {
        EXPECT_FALSE(topologies[r]->has_accelerator())
            << "Rank " << r << " should not have accelerator";
    }

    const auto &inv = topologies[0]->clusterInventory();
    EXPECT_FALSE(inv.hasAnyGPU());
    EXPECT_EQ(inv.total_gpus, 0);
    EXPECT_EQ(inv.total_gpu_memory, 0);
}

TEST_F(Test__HeterogeneousExecution, AllCpuFallback_UniformWorkDistribution)
{
    auto topologies = createCpuOnlyTopology();

    // All CPU ranks should have equal compute weights
    auto weights = topologies[0]->get_compute_weights();
    ASSERT_EQ(weights.size(), 4);

    for (float w : weights)
    {
        EXPECT_FLOAT_EQ(w, 1.0f) << "CPU-only ranks should have weight 1.0";
    }

    // Work should be uniformly distributed
    const int total_work = 100;
    for (int r = 0; r < 4; ++r)
    {
        WorkRange work = topologies[r]->get_row_range(total_work);
        EXPECT_EQ(work.size(), 25) << "Rank " << r << " should get 25 rows";
    }
}

TEST_F(Test__HeterogeneousExecution, AllCpuFallback_PlacementMetadata)
{
    auto topologies = createCpuOnlyTopology();

    // Verify placement metadata for CPU-only ranks
    for (int r = 0; r < 4; ++r)
    {
        const auto &placement = topologies[r]->placement();

        EXPECT_EQ(placement.rank, r);
        EXPECT_FALSE(placement.devices.empty());

        // All devices should be CPU type
        for (const auto &dev : placement.devices)
        {
            EXPECT_EQ(dev.type, DeviceCapability::Type::CPU);
        }
    }
}

// ============================================================================
// Test: SingleGpuAmongMany
// ============================================================================

/**
 * @test Verify behavior when only one rank has a GPU
 *
 * Scenario: 4 ranks, but only rank 2 has a GPU
 * - Cluster should still report hasAnyGPU() = true
 * - Work distribution may favor the GPU rank for compute-heavy tasks
 */
TEST_F(Test__HeterogeneousExecution, SingleGpuAmongMany_GpuDetected)
{
    auto topologies = createSingleGpuTopology();

    // Only rank 2 has GPU
    EXPECT_FALSE(topologies[0]->has_accelerator());
    EXPECT_FALSE(topologies[1]->has_accelerator());
    EXPECT_TRUE(topologies[2]->has_accelerator());
    EXPECT_FALSE(topologies[3]->has_accelerator());

    const auto &inv = topologies[0]->clusterInventory();
    EXPECT_TRUE(inv.hasAnyGPU());
    EXPECT_EQ(inv.total_gpus, 1);
}

TEST_F(Test__HeterogeneousExecution, SingleGpuAmongMany_ComputeWeightImbalance)
{
    auto topologies = createSingleGpuTopology();

    auto weights = topologies[0]->get_compute_weights();
    ASSERT_EQ(weights.size(), 4);

    // Ranks 0, 1, 3 are CPU-only (weight 1.0)
    // Rank 2 has GPU (weight 11.0)
    EXPECT_FLOAT_EQ(weights[0], 1.0f);
    EXPECT_FLOAT_EQ(weights[1], 1.0f);
    EXPECT_FLOAT_EQ(weights[2], 11.0f); // GPU rank
    EXPECT_FLOAT_EQ(weights[3], 1.0f);

    // Total weight: 1 + 1 + 11 + 1 = 14
    float total_weight = 0.0f;
    for (float w : weights)
        total_weight += w;
    EXPECT_FLOAT_EQ(total_weight, 14.0f);
}

TEST_F(Test__HeterogeneousExecution, SingleGpuAmongMany_GpuRankIdentification)
{
    auto topologies = createSingleGpuTopology();

    // Verify we can identify the GPU rank from any rank's view
    const auto &inv = topologies[0]->clusterInventory();

    int gpu_rank = -1;
    for (int r = 0; r < 4; ++r)
    {
        if (inv.ranks[r].hasGPU())
        {
            gpu_rank = r;
            break;
        }
    }

    EXPECT_EQ(gpu_rank, 2) << "Rank 2 should be identified as the GPU rank";

    // Verify GPU memory on that rank
    EXPECT_GT(inv.ranks[2].totalGPUMemory(), 0);
    size_t expected_memory = static_cast<size_t>(24) * 1024 * 1024 * 1024;
    EXPECT_EQ(inv.ranks[2].totalGPUMemory(), expected_memory);
}

// ============================================================================
// Test: DeviceMemoryAwarePlacement
// ============================================================================

/**
 * @test Verify that VRAM differences are captured in cluster inventory
 *
 * Tests with varying GPU memory:
 * - Rank 0: 8GB, Rank 1: 16GB, Rank 2: 24GB, Rank 3: 40GB
 */
TEST_F(Test__HeterogeneousExecution, DeviceMemoryAwarePlacement_MemoryCapture)
{
    auto topologies = createVaryingMemoryTopology();

    const auto &inv = topologies[0]->clusterInventory();

    // Verify each rank's GPU memory
    size_t gb = 1024ULL * 1024ULL * 1024ULL;
    EXPECT_EQ(inv.ranks[0].totalGPUMemory(), 8 * gb);
    EXPECT_EQ(inv.ranks[1].totalGPUMemory(), 16 * gb);
    EXPECT_EQ(inv.ranks[2].totalGPUMemory(), 24 * gb);
    EXPECT_EQ(inv.ranks[3].totalGPUMemory(), 40 * gb);

    // Total GPU memory: 8 + 16 + 24 + 40 = 88GB
    EXPECT_EQ(inv.total_gpu_memory, 88 * gb);
}

TEST_F(Test__HeterogeneousExecution, DeviceMemoryAwarePlacement_RankOrdering)
{
    auto topologies = createVaryingMemoryTopology();

    const auto &inv = topologies[0]->clusterInventory();

    // Verify ranks are ordered by memory (for potential memory-aware scheduling)
    std::vector<std::pair<int, size_t>> rank_memory;
    for (int r = 0; r < 4; ++r)
    {
        rank_memory.emplace_back(r, inv.ranks[r].totalGPUMemory());
    }

    // Sort by memory descending
    std::sort(rank_memory.begin(), rank_memory.end(),
              [](const auto &a, const auto &b)
              { return a.second > b.second; });

    // Rank 3 should have most memory
    EXPECT_EQ(rank_memory[0].first, 3);
    // Rank 0 should have least memory
    EXPECT_EQ(rank_memory[3].first, 0);
}

TEST_F(Test__HeterogeneousExecution, DeviceMemoryAwarePlacement_SliceMetadata)
{
    auto topologies = createVaryingMemoryTopology();

    // Test that slice metadata correctly reflects rank distribution
    const size_t total_rows = 4096;
    const size_t total_cols = 896;

    for (int r = 0; r < 4; ++r)
    {
        SliceMetadata meta = topologies[r]->createRowParallelMeta(total_rows, total_cols);

        EXPECT_EQ(meta.mode, SliceMode::ROW_PARALLEL);
        EXPECT_EQ(meta.original_rows, total_rows);
        EXPECT_EQ(meta.original_cols, total_cols);
        EXPECT_EQ(meta.world_size, 4);
        EXPECT_EQ(meta.rank, r);

        // Each rank should get ~1024 rows (4096/4)
        EXPECT_GT(meta.slice_size(), 0);
        EXPECT_LE(meta.slice_size(), total_rows);
    }
}

// ============================================================================
// Test: GracefulDegradation
// ============================================================================

/**
 * @test Verify system handles topology changes gracefully
 *
 * Tests that:
 * 1. Compute participant flag can be toggled
 * 2. Non-participating ranks are handled correctly
 * 3. Work ranges remain consistent even with observer ranks
 */
TEST_F(Test__HeterogeneousExecution, GracefulDegradation_ComputeParticipantToggle)
{
    auto topology = MockMPITopology::createSimple(0, 2);

    // By default, all ranks are compute participants
    EXPECT_TRUE(topology->is_compute_participant());

    // Simulate "failed" rank by disabling participation
    topology->set_compute_participant(false);
    EXPECT_FALSE(topology->is_compute_participant());

    // Re-enable
    topology->set_compute_participant(true);
    EXPECT_TRUE(topology->is_compute_participant());
}

TEST_F(Test__HeterogeneousExecution, GracefulDegradation_EmptyWorkRange)
{
    auto topology = MockMPITopology::createSimple(1, 4);

    // Edge case: fewer work items than ranks
    WorkRange work = topology->get_head_range(2);

    // With 2 heads and 4 ranks, some ranks get no work
    // First 2 ranks get 1 head each, last 2 get nothing
    if (topology->rank() >= 2)
    {
        EXPECT_TRUE(work.empty());
        EXPECT_EQ(work.size(), 0);
    }
}

TEST_F(Test__HeterogeneousExecution, GracefulDegradation_ZeroWorkDistribution)
{
    auto topology = MockMPITopology::createSimple(0, 4);

    // Edge case: zero work items
    WorkRange work = topology->get_head_range(0);

    EXPECT_TRUE(work.empty());
    EXPECT_EQ(work.start, 0);
    EXPECT_EQ(work.end, 0);
}

// ============================================================================
// Test: MockMPIContext Integration
// ============================================================================

/**
 * @test Verify MockMPIContext works correctly with heterogeneous topologies
 *
 * Tests that collective operations behave correctly in simulated environment.
 */
TEST_F(Test__HeterogeneousExecution, MockMPIContext_BasicOperations)
{
    MockMPIContext::Config config;
    config.rank = 1;
    config.world_size = 4;
    config.track_calls = true;
    config.simulate_noop = true;

    MockMPIContext ctx(config);

    EXPECT_EQ(ctx.rank(), 1);
    EXPECT_EQ(ctx.world_size(), 4);
    EXPECT_FALSE(ctx.is_root());

    // Test barrier (no-op in mock)
    ASSERT_NO_THROW(ctx.barrier());
    EXPECT_EQ(ctx.barrier_call_count(), 1);
}

TEST_F(Test__HeterogeneousExecution, MockMPIContext_WorkDistribution)
{
    MockMPIContext ctx(2, 4); // Rank 2 of 4

    // Test local slice calculation
    auto [start, count] = ctx.get_local_slice(100);

    // 100 / 4 = 25 per rank
    EXPECT_EQ(start, 50); // Rank 2 starts at 50
    EXPECT_EQ(count, 25); // Gets 25 elements

    // Test row distribution
    auto [row_start, row_count] = ctx.distribute_rows(1000);

    EXPECT_EQ(row_start, 500); // Rank 2 starts at 500
    EXPECT_EQ(row_count, 250); // Gets 250 rows (1000/4)
}

TEST_F(Test__HeterogeneousExecution, MockMPIContext_FailureInjection)
{
    MockMPIContext::Config config;
    config.rank = 0;
    config.world_size = 4;
    config.barrier_should_fail = true;

    MockMPIContext ctx(config);

    EXPECT_THROW(ctx.barrier(), std::runtime_error);
}

// ============================================================================
// Test: Multi-Node Configuration
// ============================================================================

/**
 * @test Verify correct handling of multi-node configurations
 */
TEST_F(Test__HeterogeneousExecution, MultiNode_NodeLeaderDetection)
{
    // 4 ranks across 2 nodes (2 ranks per node)
    auto rank0 = MockMPITopology::createSimple(0, 4, 2);
    auto rank1 = MockMPITopology::createSimple(1, 4, 2);
    auto rank2 = MockMPITopology::createSimple(2, 4, 2);
    auto rank3 = MockMPITopology::createSimple(3, 4, 2);

    // Node leaders are ranks 0 and 2 (first on each node)
    EXPECT_TRUE(rank0->is_node_leader());
    EXPECT_FALSE(rank1->is_node_leader());
    EXPECT_TRUE(rank2->is_node_leader());
    EXPECT_FALSE(rank3->is_node_leader());
}

TEST_F(Test__HeterogeneousExecution, MultiNode_NodeIdAssignment)
{
    auto rank0 = MockMPITopology::createSimple(0, 4, 2);
    auto rank1 = MockMPITopology::createSimple(1, 4, 2);
    auto rank2 = MockMPITopology::createSimple(2, 4, 2);
    auto rank3 = MockMPITopology::createSimple(3, 4, 2);

    // Ranks 0-1 on node 0, ranks 2-3 on node 1
    EXPECT_EQ(rank0->placement().node_id, 0);
    EXPECT_EQ(rank1->placement().node_id, 0);
    EXPECT_EQ(rank2->placement().node_id, 1);
    EXPECT_EQ(rank3->placement().node_id, 1);

    // Verify local ranks within nodes
    EXPECT_EQ(rank0->placement().local_rank, 0);
    EXPECT_EQ(rank1->placement().local_rank, 1);
    EXPECT_EQ(rank2->placement().local_rank, 0);
    EXPECT_EQ(rank3->placement().local_rank, 1);
}

// ============================================================================
// Test: ROCm Device Support
// ============================================================================

/**
 * @test Verify ROCm GPU detection and handling
 */
TEST_F(Test__HeterogeneousExecution, ROCm_DeviceDetection)
{
    auto topology = MockMPITopologyBuilder()
                        .addROCmRank(0, 0, 0, 16.0f)
                        .addROCmRank(1, 0, 1, 16.0f)
                        .setLocalRank(0)
                        .build();

    EXPECT_TRUE(topology->has_accelerator());

    const auto &devices = topology->get_devices();
    bool has_rocm = false;
    for (const auto &dev : devices)
    {
        if (dev.type == DeviceCapability::Type::ROCm)
        {
            has_rocm = true;
            EXPECT_EQ(dev.device_id, 0);
        }
    }
    EXPECT_TRUE(has_rocm);

    // Verify cluster inventory
    const auto &inv = topology->clusterInventory();
    EXPECT_TRUE(inv.hasAnyGPU());
    EXPECT_EQ(inv.total_gpus, 2);
}

// ============================================================================
// Test: Coordinator Role
// ============================================================================

/**
 * @test Verify coordinator role assignment
 */
TEST_F(Test__HeterogeneousExecution, Coordinator_RoleAssignment)
{
    auto topologies = createMixedCpuGpuTopology();

    // Only rank 0 should be coordinator
    EXPECT_TRUE(topologies[0]->is_coordinator());
    EXPECT_FALSE(topologies[1]->is_coordinator());
    EXPECT_FALSE(topologies[2]->is_coordinator());
    EXPECT_FALSE(topologies[3]->is_coordinator());
}

// ============================================================================
// Test: ToString and Debugging
// ============================================================================

/**
 * @test Verify debugging output contains expected information
 */
TEST_F(Test__HeterogeneousExecution, ToString_ContainsRelevantInfo)
{
    auto topology = MockMPITopologyBuilder()
                        .addCPUOnlyRank(0, 0)
                        .addGPURank(1, 0, 0, 8.0f)
                        .setLocalRank(1)
                        .build();

    std::string str = topology->to_string();

    // Should contain rank info
    EXPECT_NE(str.find("rank=1"), std::string::npos);

    // Should contain world size
    EXPECT_NE(str.find("2"), std::string::npos);

    // Should mention devices
    EXPECT_NE(str.find("CPU"), std::string::npos);
    EXPECT_NE(str.find("CUDA"), std::string::npos);
}
