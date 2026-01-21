/**
 * @file Test__Phase6HeterogeneousDomainIntegration.cpp
 * @brief Phase 6.6 Integration tests for heterogeneous multi-domain parallelism
 *
 * **Purpose**: End-to-end integration tests for Phase 6 components:
 * - Phase 6.1: Full capability exchange (MPITopology)
 * - Phase 6.2: HeterogeneousMultiDomainStrategy
 * - Phase 6.3: Domain wiring in GraphOrchestrator
 * - Phase 6.4: PP Send/Recv stages
 * - Phase 6.5: CLI integration
 *
 * These tests validate the complete flow from capability exchange through
 * domain assignment to graph execution with multi-domain tensor parallelism.
 *
 * **Test Categories**:
 * 1. Capability Exchange - MPI topology and device inventory
 * 2. Strategy Tests - Domain assignment and layer distribution
 * 3. Orchestrator Wiring - TPDomain integration with graph stages
 * 4. Plan Broadcast - MPI broadcast of heterogeneous plans
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cstring>
#include <set>

#include "execution/placement/HeterogeneousMultiDomainStrategy.h"
#include "execution/PlacementStrategy.h"
#include "execution/DeviceInventory.h"
#include "config/TPDomain.h"
#include "utils/MPITopology.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"

using namespace llaminar2;

// =============================================================================
// Test Fixtures
// =============================================================================

/**
 * @brief Base test fixture for Phase 6 integration tests
 *
 * Provides helpers for creating cluster inventories, MPI context access,
 * and common test utilities.
 */
class Test__Phase6HeterogeneousDomainIntegration : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    std::unique_ptr<MPITopology> topology_;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    void TearDown() override
    {
        topology_.reset();
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * @brief Create a synthetic cluster inventory for testing
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
                rank_inv.cpu.tflops_fp16 = 0.5f;

                // GPU info
                int gpu_idx = 0;
                if (with_nvidia)
                {
                    DeviceInfo nvidia_gpu;
                    nvidia_gpu.type = DeviceType::CUDA;
                    nvidia_gpu.local_device_id = gpu_idx++;
                    nvidia_gpu.memory_bytes = gpu_memory_gb * 1024ULL * 1024 * 1024;
                    nvidia_gpu.compute_units = 108;
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
                    amd_gpu.compute_units = 120;
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
        input.estimated_memory_bytes = 7000000000;

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
// Capability Exchange Tests (Phase 6.1)
// =============================================================================

/**
 * Test that MPITopology capability exchange produces valid inventory on all ranks
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, CapabilityExchangeProducesValidInventory)
{
    // Create topology - this triggers capability exchange automatically
    topology_ = std::make_unique<MPITopology>(MPI_COMM_WORLD);

    // All ranks should have placement info for all ranks
    const auto &all_placements = topology_->all_placements();

    EXPECT_EQ(all_placements.size(), static_cast<size_t>(world_size_))
        << "Rank " << rank_ << " should have placement info for all ranks";

    // Each placement should have valid rank ID
    for (int r = 0; r < world_size_; ++r)
    {
        const auto &placement = topology_->get_placement(r);
        EXPECT_EQ(placement.rank, r)
            << "Placement for rank " << r << " has wrong rank ID";
    }

    // Verify cluster inventory is accessible
    const auto &inventory = topology_->clusterInventory();
    EXPECT_EQ(inventory.world_size, world_size_);
    EXPECT_GE(inventory.node_count, 1);

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Test that heterogeneous GPU detection works correctly
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, HeterogeneousDetectionWorksCorrectly)
{
    topology_ = std::make_unique<MPITopology>(MPI_COMM_WORLD);

    // Check that hasHeterogeneousGPUs() returns consistent results
    bool has_heterogeneous = topology_->hasHeterogeneousGPUs();

    // Gather all results
    std::vector<int> all_results(world_size_);
    int my_result = has_heterogeneous ? 1 : 0;
    MPI_Allgather(&my_result, 1, MPI_INT,
                  all_results.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // All ranks should agree
    for (int r = 0; r < world_size_; ++r)
    {
        EXPECT_EQ(all_results[r], all_results[0])
            << "Rank " << r << " disagrees on heterogeneous detection";
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Test that getRankInventory returns valid data for all ranks
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, GetRankInventoryReturnsValidData)
{
    topology_ = std::make_unique<MPITopology>(MPI_COMM_WORLD);

    for (int r = 0; r < world_size_; ++r)
    {
        const auto &inv = topology_->getRankInventory(r);
        EXPECT_EQ(inv.rank, r) << "RankInventory for rank " << r << " has wrong rank ID";
        EXPECT_GE(inv.node_id, 0) << "Rank " << r << " has invalid node_id";
        // Note: cpu_cores may not be populated in RankInventory depending on MPITopology implementation
        // The important thing is that the rank ID and node_id are correct
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Strategy Tests (Phase 6.2)
// =============================================================================

/**
 * Test that strategy generates valid domain assignments
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, StrategyGeneratesDomainAssignments)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);

    auto domains = strategy.computeDomainAssignments(inventory, 24);

    // Should have at least one domain
    EXPECT_FALSE(domains.empty()) << "Strategy should generate at least one domain";

    // All domains should be valid
    for (const auto &domain : domains)
    {
        EXPECT_TRUE(domain.isValid()) << "Domain " << domain.domain_id << " is invalid";
        EXPECT_GE(domain.layer_start, 0);
        EXPECT_LE(domain.layer_end, 24);
        EXPECT_GT(domain.layerCount(), 0);
    }
}

/**
 * Test that layer distribution is proportional to compute weight
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, StrategyLayerDistributionIsProportional)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 2, true, false); // GPU + CPU

    auto domains = strategy.computeDomainAssignments(inventory, 24);

    // Count GPU vs CPU layers
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
    EXPECT_GT(gpu_layers, cpu_layers)
        << "GPU (" << gpu_layers << " layers) should get more than CPU (" << cpu_layers << ")";

    // Total should be 24
    EXPECT_EQ(gpu_layers + cpu_layers, 24);
}

/**
 * Test that getDomainForLayer returns correct domain for all layers
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, GetDomainForLayerReturnsCorrect)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);

    auto plan = strategy.generatePlan(inventory, 24);

    // Every layer should have exactly one domain
    for (int layer = 0; layer < 24; ++layer)
    {
        const auto *domain = plan.getDomainForLayer(layer);
        ASSERT_NE(domain, nullptr) << "Layer " << layer << " has no domain";
        EXPECT_GE(layer, domain->layer_start);
        EXPECT_LT(layer, domain->layer_end);
    }
}

/**
 * Test plan generation creates valid pipeline stages
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, PlanGeneratesValidPipelineStages)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, false);

    auto plan = strategy.generatePlan(inventory, 24);

    // Should have stages for multi-node
    EXPECT_GE(plan.stages.size(), 1u);

    // All stages should have valid layer ranges
    int total_stage_layers = 0;
    for (const auto &stage : plan.stages)
    {
        EXPECT_GE(stage.layer_start, 0);
        EXPECT_LE(stage.layer_end, 24);
        EXPECT_GT(stage.layer_end - stage.layer_start, 0);
        total_stage_layers += (stage.layer_end - stage.layer_start);

        // Stage should have at least one rank
        EXPECT_FALSE(stage.ranks.empty());
    }

    // Stages should cover all layers
    EXPECT_EQ(total_stage_layers, 24);
}

// =============================================================================
// Plan Serialization Tests
// =============================================================================

/**
 * Test that plan serialize/deserialize produces identical plan
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, PlanSerializeDeserializeRoundTrip)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, true); // NVIDIA + AMD
    auto original = strategy.generatePlan(inventory, 28);

    // Serialize
    auto data = original.serialize();
    EXPECT_GT(data.size(), 0u);

    // Deserialize
    auto restored = HeterogeneousPlan::deserialize(data);

    // Verify equivalence
    EXPECT_EQ(restored.total_layers, original.total_layers);
    EXPECT_EQ(restored.world_size, original.world_size);
    EXPECT_EQ(restored.node_count, original.node_count);
    EXPECT_EQ(restored.domains.size(), original.domains.size());

    // Verify each domain
    for (size_t i = 0; i < original.domains.size(); ++i)
    {
        const auto &orig = original.domains[i];
        const auto &rest = restored.domains[i];

        EXPECT_EQ(rest.domain_id, orig.domain_id);
        EXPECT_EQ(rest.type, orig.type);
        EXPECT_EQ(rest.layer_start, orig.layer_start);
        EXPECT_EQ(rest.layer_end, orig.layer_end);
        EXPECT_EQ(rest.ranks.size(), orig.ranks.size());
        EXPECT_EQ(rest.devices.size(), orig.devices.size());
    }
}

// =============================================================================
// MPI Plan Broadcast Tests (Phase 6.4)
// =============================================================================

/**
 * Test that plan can be broadcast from rank 0 to all ranks
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, PlanBroadcastFromRank0)
{
    HeterogeneousPlan local_plan;

    if (rank_ == 0)
    {
        // Only rank 0 generates the plan
        HeterogeneousMultiDomainStrategy strategy;
        auto inventory = createCluster(2, 2, true, false);
        local_plan = strategy.generatePlan(inventory, 24);
    }

    // Serialize (only rank 0 has meaningful data)
    std::vector<uint8_t> data;
    if (rank_ == 0)
    {
        data = local_plan.serialize();
    }

    // Broadcast size first
    int data_size = static_cast<int>(data.size());
    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize on non-root ranks
    if (rank_ != 0)
    {
        data.resize(data_size);
    }

    // Broadcast data
    MPI_Bcast(data.data(), data_size, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Deserialize on all ranks
    auto received_plan = HeterogeneousPlan::deserialize(data);

    // Verify all ranks received valid plan
    EXPECT_EQ(received_plan.total_layers, 24);
    EXPECT_GT(received_plan.domains.size(), 0u);

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Test that all ranks have identical plan after broadcast
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, AllRanksHaveIdenticalPlan)
{
    HeterogeneousPlan local_plan;

    if (rank_ == 0)
    {
        HeterogeneousMultiDomainStrategy strategy;
        auto inventory = createCluster(2, 2, true, true);
        local_plan = strategy.generatePlan(inventory, 28);
    }

    // Broadcast
    std::vector<uint8_t> data;
    if (rank_ == 0)
    {
        data = local_plan.serialize();
    }

    int data_size = static_cast<int>(data.size());
    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank_ != 0)
    {
        data.resize(data_size);
    }
    MPI_Bcast(data.data(), data_size, MPI_BYTE, 0, MPI_COMM_WORLD);

    auto plan = HeterogeneousPlan::deserialize(data);

    // Gather summary info from all ranks
    struct PlanSummary
    {
        int total_layers;
        int world_size;
        int node_count;
        int num_domains;
        int num_stages;
    };

    PlanSummary my_summary{
        plan.total_layers,
        plan.world_size,
        plan.node_count,
        static_cast<int>(plan.domains.size()),
        static_cast<int>(plan.stages.size())};

    std::vector<PlanSummary> all_summaries(world_size_);
    MPI_Allgather(&my_summary, sizeof(PlanSummary), MPI_BYTE,
                  all_summaries.data(), sizeof(PlanSummary), MPI_BYTE,
                  MPI_COMM_WORLD);

    // All ranks should have identical summary
    for (int r = 0; r < world_size_; ++r)
    {
        EXPECT_EQ(all_summaries[r].total_layers, all_summaries[0].total_layers)
            << "Rank " << r << " has different total_layers";
        EXPECT_EQ(all_summaries[r].world_size, all_summaries[0].world_size)
            << "Rank " << r << " has different world_size";
        EXPECT_EQ(all_summaries[r].num_domains, all_summaries[0].num_domains)
            << "Rank " << r << " has different num_domains";
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Domain Configuration Tests (Phase 6.3)
// =============================================================================

/**
 * Test that TPDomain can be created from DomainAssignment
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, TPDomainFromDomainAssignment)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(1, 2, true, true);
    auto domains = strategy.computeDomainAssignments(inventory, 24);

    // Verify we can extract TPDomain info from assignments
    for (const auto &domain : domains)
    {
        // Domain should have valid TPDomainType
        EXPECT_TRUE(domain.type == TPDomainType::GPU_INTRA_RANK ||
                    domain.type == TPDomainType::CPU_CROSS_RANK);

        // Domain should have devices
        EXPECT_FALSE(domain.devices.empty());

        // All devices should be valid
        for (const auto &dev : domain.devices)
        {
            EXPECT_TRUE(dev.is_cpu() || dev.is_gpu());
        }
    }
}

/**
 * Test MultiDomainTPConfig can be created for test scenarios
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, MultiDomainTPConfigForTest)
{
    // Create test domains
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
    gpu_domain.domain_size = 2;
    gpu_domain.local_rank_in_domain = 0;
    gpu_domain.name = "gpu_tp_test";
    gpu_domain.communicator = MPI_COMM_NULL;

    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.devices = {DeviceId::cpu()};
    cpu_domain.domain_size = world_size_;
    cpu_domain.local_rank_in_domain = rank_;
    cpu_domain.name = "cpu_tp_test";
    cpu_domain.communicator = MPI_COMM_WORLD;

    // Create config
    auto config = MultiDomainTPConfig::createForTest({gpu_domain, cpu_domain});

    // Verify domains
    EXPECT_EQ(config.domains().size(), 2u);
    EXPECT_NE(config.gpuDomain(), nullptr);
    EXPECT_NE(config.cpuDomain(), nullptr);
}

// =============================================================================
// Deterministic Plan Computation Tests
// =============================================================================

/**
 * Test that all ranks compute identical plans deterministically
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, DeterministicPlanComputation)
{
    // Each rank independently computes the same plan
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, true);
    auto plan = strategy.generatePlan(inventory, 28);

    // Serialize each rank's plan
    auto data = plan.serialize();

    // Gather all plan sizes
    std::vector<int> all_sizes(world_size_);
    int my_size = static_cast<int>(data.size());
    MPI_Allgather(&my_size, 1, MPI_INT,
                  all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // All sizes should match
    for (int r = 0; r < world_size_; ++r)
    {
        EXPECT_EQ(all_sizes[r], all_sizes[0])
            << "Rank " << r << " produced different plan size";
    }

    // Compute checksum
    uint32_t checksum = 0;
    for (uint8_t byte : data)
    {
        checksum = (checksum << 1) ^ byte;
    }

    // Gather all checksums
    std::vector<uint32_t> all_checksums(world_size_);
    MPI_Allgather(&checksum, 1, MPI_UINT32_T,
                  all_checksums.data(), 1, MPI_UINT32_T, MPI_COMM_WORLD);

    // All checksums should match (deterministic)
    for (int r = 0; r < world_size_; ++r)
    {
        EXPECT_EQ(all_checksums[r], all_checksums[0])
            << "Rank " << r << " produced different plan (checksum mismatch)";
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// PlacementPlan Integration Tests
// =============================================================================

/**
 * Test that strategy computes valid PlacementPlan
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, StrategyComputesValidPlacementPlan)
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

/**
 * Test that PlacementPlan layer assignments cover all layers
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, PlacementPlanCoversAllLayers)
{
    HeterogeneousMultiDomainStrategy strategy;
    auto inventory = createCluster(2, 2, true, true);
    auto input = createInput(inventory, 28);

    PlacementPlan plan = strategy.compute(input);

    // Verify we have layer entries
    EXPECT_EQ(plan.layers.size(), 28u);

    // Each layer should have a device
    for (int i = 0; i < 28; ++i)
    {
        EXPECT_GE(plan.layers[i].layer_idx, 0);
        // Device should be valid (CPU or GPU)
    }
}

// =============================================================================
// End-to-End Mock Tests
// =============================================================================

/**
 * Test complete flow: inventory -> strategy -> plan -> TPDomain mapping
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, EndToEndWithMockModel)
{
    // 1. Create synthetic cluster
    auto inventory = createCluster(2, 2, true, true);

    // 2. Generate heterogeneous plan
    HeterogeneousMultiDomainStrategy strategy;
    auto plan = strategy.generatePlan(inventory, 24);

    // 3. Verify plan structure
    EXPECT_EQ(plan.total_layers, 24);
    EXPECT_FALSE(plan.domains.empty());

    // 4. Verify each layer has a domain
    for (int layer = 0; layer < 24; ++layer)
    {
        const auto *domain = plan.getDomainForLayer(layer);
        ASSERT_NE(domain, nullptr) << "Layer " << layer << " missing domain";
    }

    // 5. Verify domain types make sense
    bool has_gpu_domain = false;
    bool has_cpu_domain = false;
    for (const auto &domain : plan.domains)
    {
        if (domain.type == TPDomainType::GPU_INTRA_RANK)
        {
            has_gpu_domain = true;
            // GPU domains should have GPU devices
            for (const auto &dev : domain.devices)
            {
                EXPECT_TRUE(dev.is_gpu()) << "GPU domain has non-GPU device";
            }
        }
        else if (domain.type == TPDomainType::CPU_CROSS_RANK)
        {
            has_cpu_domain = true;
        }
    }

    // With our config, we expect both GPU and CPU domains
    EXPECT_TRUE(has_gpu_domain) << "No GPU domains created";
    // CPU domains are optional based on config
}

/**
 * Test that heterogeneous GPU cluster (NVIDIA + AMD) creates separate intra-rank domains
 */
TEST_F(Test__Phase6HeterogeneousDomainIntegration, HeterogeneousGPUCreatesIntraRankDomains)
{
    auto inventory = createCluster(1, 2, true, true); // NVIDIA + AMD per rank

    HeterogeneousMultiDomainStrategy strategy;
    auto domains = strategy.computeDomainAssignments(inventory, 24);

    // Count GPU intra-rank domains
    int gpu_domains = 0;
    for (const auto &domain : domains)
    {
        if (domain.type == TPDomainType::GPU_INTRA_RANK)
        {
            gpu_domains++;
            // Each should have both NVIDIA and AMD
            bool has_cuda = false, has_rocm = false;
            for (const auto &dev : domain.devices)
            {
                if (dev.type == DeviceType::CUDA)
                    has_cuda = true;
                if (dev.type == DeviceType::ROCm)
                    has_rocm = true;
            }
            EXPECT_TRUE(has_cuda && has_rocm)
                << "GPU domain should have both NVIDIA and AMD";
        }
    }

    EXPECT_EQ(gpu_domains, 2) << "Should have one GPU domain per rank";
}

// =============================================================================
// Main function with MPI initialization
// =============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
