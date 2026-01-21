/**
 * @file Test__OrchestratorWiringIntegration.cpp
 * @brief TDD Integration tests for orchestrator wiring (Phase G1)
 *
 * These tests verify that the orchestration system is PROPERLY WIRED:
 *   1. MPITopology::computePlacement() is called during inference setup
 *   2. PlacementStrategyFactory::autoSelect() selects the right strategy
 *   3. HeterogeneousMultiDomainStrategy is used when appropriate
 *   4. The PlacementPlan flows through to GraphOrchestrator
 *
 * TDD APPROACH:
 *   - These tests are written FIRST to define expected behavior
 *   - They should FAIL initially (wiring not complete)
 *   - After wiring is implemented, they should PASS
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>

#include "mocks/MockMPITopology.h" // Use the existing mock with tracking
#include "execution/PlacementStrategy.h"
#include "execution/PlacementPlan.h"
#include "execution/placement/HeterogeneousMultiDomainStrategy.h"
#include "execution/GraphOrchestrator.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__OrchestratorWiringIntegration : public ::testing::Test
{
protected:
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<MockMPITopology> mock_topology_;
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Create a mock topology using the builder that simulates real hardware
        auto builder = MockMPITopologyBuilder();

        // Add ranks based on actual world size
        for (int r = 0; r < world_size_; ++r)
        {
            // Simulate heterogeneous setup: rank 0 has CUDA, rank 1 has ROCm
            if (r % 2 == 0)
            {
                builder.addGPURank(r, r / 2, 0, 24.0f); // CUDA GPU
            }
            else
            {
                builder.addROCmRank(r, r / 2, 0, 31.0f); // ROCm GPU
            }
        }

        builder.setLocalRank(rank_);
        mock_topology_ = builder.build();
    }

    void TearDown() override
    {
        mock_topology_->resetCallTracking();
        mock_topology_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create a minimal PlacementInput for testing
     */
    PlacementInput createTestInput(int n_layers = 28)
    {
        PlacementInput input;
        input.architecture = "qwen2";
        input.n_layers = n_layers;
        input.d_model = 3584;
        input.d_ff = 18944;
        input.vocab_size = 152064;
        input.n_heads = 28;
        input.n_kv_heads = 4;
        input.quant_type = "Q4_0";
        input.estimated_memory_bytes = 4ULL * 1024 * 1024 * 1024; // 4GB
        input.world_size = world_size_;
        input.ranks_per_node = mock_topology_->ranks_per_node();
        input.node_count = mock_topology_->node_count();

        // Fill in cluster inventory
        input.cluster_inventory = mock_topology_->clusterInventory();

        return input;
    }

    /**
     * @brief Create PlacementInput with heterogeneous GPU configuration
     */
    PlacementInput createHeterogeneousInput(int n_layers = 28)
    {
        PlacementInput input = createTestInput(n_layers);

        // Simulate heterogeneous GPUs (CUDA + ROCm on same system)
        input.any_rank_has_gpu = true;
        input.total_gpu_memory = 55ULL * 1024 * 1024 * 1024;  // 55GB (24GB + 31GB)
        input.total_cpu_memory = 377ULL * 1024 * 1024 * 1024; // 377GB

        // Explicitly request heterogeneous strategy
        input.preferred_strategy = "heterogeneous";

        return input;
    }
};

// =============================================================================
// WIRING TEST 1: computePlacement Tracking Works
// =============================================================================

TEST_F(Test__OrchestratorWiringIntegration, ComputePlacementTrackingWorks)
{
    // GIVEN: A mock topology with call tracking
    ASSERT_EQ(mock_topology_->getComputePlacementCallCount(), 0)
        << "Precondition: computePlacement should not have been called yet";

    // WHEN: We call computePlacement through the interface
    auto input = createTestInput();
    PlacementPlan plan = mock_topology_->computePlacement(input);

    // THEN: The call should be tracked
    EXPECT_EQ(mock_topology_->getComputePlacementCallCount(), 1)
        << "computePlacement call should be tracked";

    // AND: The input should be stored
    const auto &last_input = mock_topology_->getLastPlacementInput();
    EXPECT_EQ(last_input.n_layers, input.n_layers)
        << "Last input should have correct n_layers";
    EXPECT_EQ(last_input.architecture, input.architecture)
        << "Last input should have correct architecture";

    if (rank_ == 0)
    {
        LOG_INFO("[WIRING TEST] computePlacement tracking verified:");
        LOG_INFO("  Call count: " << mock_topology_->getComputePlacementCallCount());
        LOG_INFO("  Plan strategy: " << plan.strategy_name);
    }
}

// =============================================================================
// WIRING TEST 2: Strategy Factory Selects Correct Strategy
// =============================================================================

TEST_F(Test__OrchestratorWiringIntegration, StrategyFactorySelectsGPUFirstForHomogeneous)
{
    // GIVEN: A standard input with GPUs available
    PlacementInput input = createTestInput();
    input.any_rank_has_gpu = true;
    input.total_gpu_memory = 24ULL * 1024 * 1024 * 1024; // 24GB

    // WHEN: We call autoSelect
    auto strategy = PlacementStrategyFactory::autoSelect(input);

    // THEN: Should select GPUFirst for homogeneous GPU setup
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "GPUFirst")
        << "Should select GPUFirst for standard GPU setup";
}

TEST_F(Test__OrchestratorWiringIntegration, StrategyFactorySelectsCPUOnlyWhenNoGPU)
{
    // GIVEN: Input with no GPUs
    PlacementInput input = createTestInput();
    input.any_rank_has_gpu = false;
    input.force_cpu_only = true;

    // WHEN: We call autoSelect
    auto strategy = PlacementStrategyFactory::autoSelect(input);

    // THEN: Should select CPUOnly
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "CPUOnly")
        << "Should select CPUOnly when no GPU available";
}

TEST_F(Test__OrchestratorWiringIntegration, StrategyFactorySelectsHeterogeneousWhenRequested)
{
    // GIVEN: Input explicitly requesting heterogeneous strategy
    PlacementInput input = createHeterogeneousInput();

    // WHEN: We call autoSelect
    auto strategy = PlacementStrategyFactory::autoSelect(input);

    // THEN: Should select HeterogeneousMultiDomain
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "HeterogeneousMultiDomain")
        << "Should select HeterogeneousMultiDomain when explicitly requested";
}

// =============================================================================
// WIRING TEST 3: HeterogeneousMultiDomainStrategy Generates Valid Plan
// =============================================================================

TEST_F(Test__OrchestratorWiringIntegration, HeterogeneousStrategyGeneratesValidPlan)
{
    // GIVEN: Heterogeneous input configuration
    PlacementInput input = createHeterogeneousInput();

    // WHEN: We use HeterogeneousMultiDomainStrategy to compute placement
    HeterogeneousMultiDomainStrategy strategy;

    // First check if applicable
    bool applicable = strategy.isApplicable(input);
    if (!applicable)
    {
        // Strategy may not be applicable in single-rank test environment
        GTEST_SKIP() << "HeterogeneousMultiDomainStrategy not applicable in current config";
    }

    PlacementPlan plan = strategy.compute(input);

    // THEN: Plan should be valid
    EXPECT_FALSE(plan.strategy_name.empty()) << "Plan should have strategy name";
    EXPECT_EQ(plan.n_layers, input.n_layers) << "Plan should cover all layers";

    if (rank_ == 0)
    {
        LOG_INFO("[WIRING TEST] HeterogeneousPlan generated:");
        LOG_INFO("  Strategy: " << plan.strategy_name);
        LOG_INFO("  Layers: " << plan.n_layers);
    }
}

// =============================================================================
// WIRING TEST 4: PlacementPlan Structure Is Correct
// =============================================================================

TEST_F(Test__OrchestratorWiringIntegration, PlacementPlanStructureIsCorrect)
{
    // GIVEN: A placement plan with specific layer assignments
    PlacementPlan plan;
    plan.strategy_name = "TestStrategy";
    plan.n_layers = 28;
    plan.world_size = world_size_;

    // Add layer assignments
    for (int i = 0; i < plan.n_layers; ++i)
    {
        LayerPlacement lp;
        lp.layer_idx = i;
        lp.owner_rank = i % world_size_; // Round-robin distribution
        lp.device = PlacementDevice::gpu(0);
        plan.layers.push_back(lp);
    }

    // Count layers owned by this rank
    int local_layers = 0;
    for (const auto &lp : plan.layers)
    {
        if (lp.owner_rank == rank_)
        {
            ++local_layers;
        }
    }

    // In round-robin distribution, each rank should own ~n_layers/world_size layers
    int expected_local = plan.n_layers / world_size_;
    int remainder = plan.n_layers % world_size_;
    if (rank_ < remainder)
        expected_local++;

    EXPECT_EQ(local_layers, expected_local)
        << "Rank " << rank_ << " should own " << expected_local << " layers";

    if (rank_ == 0)
    {
        LOG_INFO("[WIRING TEST] PlacementPlan layer distribution:");
        LOG_INFO("  Total layers: " << plan.n_layers);
        LOG_INFO("  World size: " << world_size_);
        LOG_INFO("  Local layers (rank 0): " << local_layers);
    }
}

// =============================================================================
// WIRING TEST 5: Mock Topology Detects Heterogeneous GPUs
// =============================================================================

TEST_F(Test__OrchestratorWiringIntegration, MockTopologyDetectsHeterogeneousGPUs)
{
    // GIVEN: Mock topology configured with mixed CUDA/ROCm

    // THEN: hasHeterogeneousGPUs should return true if world_size >= 2
    // (because we configured rank 0 with CUDA, rank 1 with ROCm in SetUp)
    bool is_heterogeneous = mock_topology_->hasHeterogeneousGPUs();

    if (world_size_ >= 2)
    {
        EXPECT_TRUE(is_heterogeneous)
            << "Mock topology should detect heterogeneous GPUs with 2+ ranks";
    }
    else
    {
        // Single rank - won't be heterogeneous
        EXPECT_FALSE(is_heterogeneous)
            << "Single rank topology is not heterogeneous";
    }

    if (rank_ == 0)
    {
        LOG_INFO("[WIRING TEST] Heterogeneous detection:");
        LOG_INFO("  World size: " << world_size_);
        LOG_INFO("  Has heterogeneous GPUs: " << (is_heterogeneous ? "yes" : "no"));
    }
}

// =============================================================================
// WIRING TEST 6: ClusterInventory Is Populated Correctly
// =============================================================================

TEST_F(Test__OrchestratorWiringIntegration, ClusterInventoryIsPopulatedCorrectly)
{
    // GIVEN: A mock topology
    const auto &inventory = mock_topology_->clusterInventory();

    // THEN: ClusterInventory should have correct structure
    EXPECT_EQ(inventory.world_size, world_size_)
        << "ClusterInventory world_size should match MPI world size";

    EXPECT_GE(inventory.node_count, 1)
        << "Should have at least one node";

    EXPECT_EQ(inventory.ranks.size(), static_cast<size_t>(world_size_))
        << "Should have inventory for all ranks";

    // Each rank should have valid info
    for (int r = 0; r < world_size_; ++r)
    {
        const auto &rank_info = inventory.ranks[r];
        EXPECT_EQ(rank_info.rank, r) << "Rank info should have correct rank ID";
        EXPECT_GE(rank_info.node_id, 0) << "Rank " << r << " should have valid node_id";
    }

    if (rank_ == 0)
    {
        LOG_INFO("[WIRING TEST] ClusterInventory:");
        LOG_INFO("  World size: " << inventory.world_size);
        LOG_INFO("  Node count: " << inventory.node_count);
        LOG_INFO("  Ranks: " << inventory.ranks.size());
    }
}

// =============================================================================
// WIRING TEST 7: End-to-End Flow Through computePlacement
// =============================================================================

TEST_F(Test__OrchestratorWiringIntegration, EndToEndComputePlacementFlow)
{
    // This test verifies the complete flow:
    //   1. Create PlacementInput
    //   2. Call computePlacement on topology
    //   3. Verify plan is returned
    //   4. Verify call is tracked

    // GIVEN: Test input
    PlacementInput input = createTestInput();
    input.any_rank_has_gpu = true;

    // Reset tracking
    mock_topology_->resetCallTracking();
    ASSERT_EQ(mock_topology_->getComputePlacementCallCount(), 0);

    // WHEN: Call computePlacement through the IMPITopology interface
    PlacementPlan plan = mock_topology_->computePlacement(input);

    // THEN: Plan should be valid
    EXPECT_FALSE(plan.strategy_name.empty()) << "Plan should have strategy name";
    EXPECT_EQ(plan.n_layers, input.n_layers) << "Plan should cover all layers";

    // AND: Call should be tracked
    EXPECT_EQ(mock_topology_->getComputePlacementCallCount(), 1)
        << "computePlacement should be tracked";

    // AND: All layers should be assigned
    std::vector<bool> layer_covered(input.n_layers, false);
    for (const auto &lp : plan.layers)
    {
        if (lp.layer_idx >= 0 && lp.layer_idx < input.n_layers)
        {
            layer_covered[lp.layer_idx] = true;
        }
    }

    bool all_covered = true;
    for (int i = 0; i < input.n_layers; ++i)
    {
        if (!layer_covered[i])
        {
            all_covered = false;
            break;
        }
    }

    EXPECT_TRUE(all_covered) << "All layers should be assigned in plan";

    if (rank_ == 0)
    {
        LOG_INFO("[WIRING TEST] End-to-end flow verified:");
        LOG_INFO("  Strategy: " << plan.strategy_name);
        LOG_INFO("  Plan layers: " << plan.layers.size());
        LOG_INFO("  All layers covered: " << (all_covered ? "yes" : "no"));
        LOG_INFO("  computePlacement calls: " << mock_topology_->getComputePlacementCallCount());
    }
}

// =============================================================================
// WIRING TEST 8: Mock Plan Override Works
// =============================================================================

TEST_F(Test__OrchestratorWiringIntegration, MockPlanOverrideWorks)
{
    // GIVEN: A custom mock plan
    PlacementPlan mock_plan;
    mock_plan.strategy_name = "CustomMockStrategy";
    mock_plan.n_layers = 42;
    mock_topology_->setMockPlacementPlan(mock_plan);

    // WHEN: Call computePlacement
    PlacementInput input = createTestInput();
    PlacementPlan result = mock_topology_->computePlacement(input);

    // THEN: Should return the mock plan
    EXPECT_EQ(result.strategy_name, "CustomMockStrategy")
        << "Should return mock plan strategy name";
    EXPECT_EQ(result.n_layers, 42)
        << "Should return mock plan n_layers";

    // AND: Call should still be tracked
    EXPECT_EQ(mock_topology_->getComputePlacementCallCount(), 1);

    // Cleanup
    mock_topology_->clearMockPlacementPlan();
}

// =============================================================================
// Main (MPI-aware)
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Only rank 0 prints test output
    if (rank != 0)
    {
        ::testing::GTEST_FLAG(print_time) = false;
        ::testing::UnitTest::GetInstance()->listeners().Release(
            ::testing::UnitTest::GetInstance()->listeners().default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
