/**
 * @file Test__PlacementStrategy_MPI_Integration.cpp
 * @brief Multi-rank integration tests for PlacementStrategy (Phase G0)
 *
 * Tests that PlacementStrategy and PlacementPlan work correctly with
 * real MPI communication:
 *   - All ranks compute identical placement plans (determinism)
 *   - Plans integrate correctly with MPITopology
 *   - WeightPlacementMap correctly populated from plan
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <functional>

#include "execution/mpi_orchestration/PlacementStrategy.h"
#include "execution/mpi_orchestration/PlacementPlan.h"
#include "utils/MPITopology.h"
#include "utils/MPIContext.h"
#include "loaders/WeightManager.h"
#include "utils/Logger.h"

using namespace llaminar2;

/**
 * @brief Test fixture for PlacementStrategy MPI integration tests
 */
class Test__PlacementStrategy_MPI_Integration : public ::testing::Test
{
protected:
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<MPITopology> topology_;
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
        topology_ = std::make_unique<MPITopology>(MPI_COMM_WORLD);
    }

    void TearDown() override
    {
        topology_.reset();
        mpi_ctx_->barrier();
    }

    // Create a PlacementInput for testing
    PlacementInput createTestInput(int n_layers = 24)
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
        input.estimated_memory_bytes = 500 * 1024 * 1024; // 500MB
        input.world_size = world_size_;
        input.force_cpu_only = false;
        input.force_gpu_only = false;
        input.max_gpu_layers = -1;
        input.preferred_strategy = "";

        // Use topology info for compute weights
        input.rank_compute_weights = topology_->get_compute_weights();

        return input;
    }

    // Hash a PlacementPlan for comparison
    // Hash a PlacementDevice into an integer for comparison
    static uint64_t hashDevice(const PlacementDevice &device)
    {
        return (static_cast<uint64_t>(device.type) << 8) | static_cast<uint64_t>(device.gpu_index);
    }

    uint64_t hashPlan(const PlacementPlan &plan)
    {
        uint64_t hash = 0;
        hash ^= std::hash<int>()(plan.n_layers);
        hash ^= std::hash<int>()(plan.world_size) << 1;
        hash ^= std::hash<std::string>()(plan.strategy_name) << 2;

        for (const auto &layer : plan.layers)
        {
            hash ^= std::hash<int>()(layer.layer_idx) << 3;
            hash ^= std::hash<int>()(layer.owner_rank) << 4;
            hash ^= std::hash<uint64_t>()(hashDevice(layer.device)) << 5;
        }

        return hash;
    }

    static int countGPULayers(const PlacementPlan &plan)
    {
        return static_cast<int>(std::count_if(
            plan.layers.begin(), plan.layers.end(),
            [](const LayerPlacement &layer)
            {
                return layer.device.isGPU();
            }));
    }
};

// =============================================================================
// Determinism Tests
// =============================================================================

TEST_F(Test__PlacementStrategy_MPI_Integration, CPUOnlyStrategyDeterministicAcrossRanks)
{
    // All ranks should produce identical plans when running CPUOnlyStrategy
    auto input = createTestInput();
    input.force_cpu_only = true;

    CPUOnlyStrategy strategy;
    PlacementPlan plan = strategy.compute(input);

    // Hash the plan
    uint64_t my_hash = hashPlan(plan);

    // Gather all hashes
    std::vector<uint64_t> all_hashes(world_size_);
    MPI_Allgather(&my_hash, 1, MPI_UINT64_T,
                  all_hashes.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

    // All hashes must be identical
    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_hashes[r], all_hashes[0])
            << "Rank " << r << " produced different plan hash than rank 0";
    }
}

TEST_F(Test__PlacementStrategy_MPI_Integration, AutoSelectProducesSameStrategyOnAllRanks)
{
    // PlacementStrategyFactory::autoSelect should pick same strategy on all ranks
    auto input = createTestInput();

    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);

    std::string my_strategy_name = strategy->name();

    // Exchange strategy names (as hashes for simplicity)
    size_t my_hash = std::hash<std::string>()(my_strategy_name);
    std::vector<size_t> all_hashes(world_size_);
    MPI_Allgather(&my_hash, sizeof(size_t), MPI_BYTE,
                  all_hashes.data(), sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_hashes[r], all_hashes[0])
            << "Rank " << r << " selected different strategy";
    }

    if (rank_ == 0)
    {
        LOG_INFO("[AutoSelect] All ranks selected strategy: " << my_strategy_name);
    }
}

TEST_F(Test__PlacementStrategy_MPI_Integration, ComputePlacementViaTopologyIsDeterministic)
{
    // MPITopology::computePlacement should produce identical plans on all ranks
    PlacementPlan plan = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q4_0",
        500 * 1024 * 1024, "");

    // Gather detailed plan info for comparison
    struct PlanDetails
    {
        int n_layers;
        int world_size;
        uint64_t strategy_hash;
        uint64_t layers_hash;
    };

    uint64_t layers_hash = 0;
    for (size_t i = 0; i < plan.layers.size(); ++i)
    {
        layers_hash ^= (static_cast<uint64_t>(plan.layers[i].layer_idx) << (i % 64));
        layers_hash ^= (static_cast<uint64_t>(plan.layers[i].owner_rank) << ((i + 1) % 64));
        layers_hash ^= (hashDevice(plan.layers[i].device) << ((i + 2) % 64));
    }

    PlanDetails my_details{
        plan.n_layers,
        plan.world_size,
        std::hash<std::string>()(plan.strategy_name),
        layers_hash};

    std::vector<PlanDetails> all_details(world_size_);
    MPI_Allgather(&my_details, sizeof(PlanDetails), MPI_BYTE,
                  all_details.data(), sizeof(PlanDetails), MPI_BYTE, MPI_COMM_WORLD);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_details[r].n_layers, all_details[0].n_layers)
            << "Rank " << r << " has different n_layers";
        EXPECT_EQ(all_details[r].world_size, all_details[0].world_size)
            << "Rank " << r << " has different world_size";
        EXPECT_EQ(all_details[r].strategy_hash, all_details[0].strategy_hash)
            << "Rank " << r << " has different strategy";
        EXPECT_EQ(all_details[r].layers_hash, all_details[0].layers_hash)
            << "Rank " << r << " has different layers configuration";
    }
}

// =============================================================================
// Plan Validity Tests
// =============================================================================

TEST_F(Test__PlacementStrategy_MPI_Integration, ComputedPlanIsValid)
{
    PlacementPlan plan = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q4_0",
        500 * 1024 * 1024, "");

    EXPECT_TRUE(plan.isValid()) << "Computed plan should be valid";
    EXPECT_EQ(plan.n_layers, 24) << "Plan should have correct n_layers";
    EXPECT_EQ(plan.world_size, world_size_) << "Plan should have correct world_size";
    EXPECT_EQ(plan.layers.size(), 24u) << "Plan should have layer placement for all layers";
}

TEST_F(Test__PlacementStrategy_MPI_Integration, AllLayersHaveValidPlacement)
{
    PlacementPlan plan = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q4_0",
        500 * 1024 * 1024, "");

    for (int i = 0; i < 24; ++i)
    {
        const auto &layer = plan.getLayerPlacement(i);
        EXPECT_EQ(layer.layer_idx, i) << "Layer " << i << " has wrong index";
        EXPECT_GE(layer.owner_rank, 0) << "Layer " << i << " has invalid owner_rank";
        EXPECT_LT(layer.owner_rank, world_size_) << "Layer " << i << " owner_rank out of range";
        // Device should be either CPU or GPU with valid index
        EXPECT_TRUE(layer.device.isCPU() || layer.device.isGPU())
            << "Layer " << i << " has invalid device type";
        if (layer.device.isGPU())
        {
            EXPECT_GE(layer.device.gpu_index, 0) << "Layer " << i << " has negative GPU index";
            EXPECT_LT(layer.device.gpu_index, 32) << "Layer " << i << " has GPU index >= 32";
        }
    }
}

// =============================================================================
// Strategy Selection Tests
// =============================================================================

TEST_F(Test__PlacementStrategy_MPI_Integration, ForceCPUOnlySelectsCPUOnlyStrategy)
{
    auto input = createTestInput();
    input.force_cpu_only = true;

    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "CPUOnly");
}

TEST_F(Test__PlacementStrategy_MPI_Integration, PreferredStrategyIsRespected)
{
    auto input = createTestInput();
    input.preferred_strategy = "CPUOnly";

    auto strategy = PlacementStrategyFactory::autoSelect(input);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->name(), "CPUOnly");
}

TEST_F(Test__PlacementStrategy_MPI_Integration, AvailableStrategiesListIsConsistent)
{
    auto strategies = PlacementStrategyFactory::availableStrategies();

    // Gather strategy count from all ranks
    int my_count = static_cast<int>(strategies.size());
    std::vector<int> all_counts(world_size_);
    MPI_Allgather(&my_count, 1, MPI_INT,
                  all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_counts[r], all_counts[0])
            << "Rank " << r << " has different number of available strategies";
    }

    // Should have at least CPUOnly and GPUFirst
    EXPECT_GE(strategies.size(), 2u);

    bool has_cpu_only = std::find(strategies.begin(), strategies.end(), "CPUOnly") != strategies.end();
    bool has_gpu_first = std::find(strategies.begin(), strategies.end(), "GPUFirst") != strategies.end();

    EXPECT_TRUE(has_cpu_only) << "CPUOnly strategy should be available";
    EXPECT_TRUE(has_gpu_first) << "GPUFirst strategy should be available";
}

// =============================================================================
// Tensor Parallelism Flag Tests
// =============================================================================

TEST_F(Test__PlacementStrategy_MPI_Integration, TensorParallelismEnabledForMultiRank)
{
    PlacementPlan plan = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q4_0",
        500 * 1024 * 1024, "");

    if (world_size_ > 1)
    {
        EXPECT_TRUE(plan.usesTensorParallelism())
            << "Multi-rank should use tensor parallelism";
    }
    else
    {
        EXPECT_FALSE(plan.usesTensorParallelism())
            << "Single-rank should not use tensor parallelism";
    }
}

TEST_F(Test__PlacementStrategy_MPI_Integration, GPUUsageCorrectlyReported)
{
    PlacementPlan plan = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q4_0",
        500 * 1024 * 1024, "");

    // Without actual GPUs, should be CPU-only
    // This test verifies usesGPU() is consistent across ranks
    bool my_uses_gpu = plan.usesGPU();

    int uses_gpu_int = my_uses_gpu ? 1 : 0;
    std::vector<int> all_uses_gpu(world_size_);
    MPI_Allgather(&uses_gpu_int, 1, MPI_INT,
                  all_uses_gpu.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_uses_gpu[r], all_uses_gpu[0])
            << "Rank " << r << " has different usesGPU() result";
    }
}

// =============================================================================
// Plan String Representation Tests
// =============================================================================

TEST_F(Test__PlacementStrategy_MPI_Integration, ToStringProducesIdenticalOutput)
{
    PlacementPlan plan = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q4_0",
        500 * 1024 * 1024, "");

    std::string my_str = plan.toString();

    // Hash the string for comparison
    size_t my_hash = std::hash<std::string>()(my_str);
    std::vector<size_t> all_hashes(world_size_);
    MPI_Allgather(&my_hash, sizeof(size_t), MPI_BYTE,
                  all_hashes.data(), sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_hashes[r], all_hashes[0])
            << "Rank " << r << " has different toString() output";
    }

    // Verify toString contains expected content
    EXPECT_FALSE(my_str.empty());
    EXPECT_NE(my_str.find("PlacementPlan"), std::string::npos);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__PlacementStrategy_MPI_Integration, SmallModelPlacement)
{
    // Test with a small model (fewer layers than ranks)
    int n_layers = std::min(4, world_size_);

    PlacementPlan plan = topology_->computePlacement(
        "test_small", n_layers, 256, 512, 1000, 4, 1, "Q4_0",
        10 * 1024 * 1024, "");

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.n_layers, n_layers);
    EXPECT_EQ(static_cast<int>(plan.layers.size()), n_layers);
}

TEST_F(Test__PlacementStrategy_MPI_Integration, LargeModelPlacement)
{
    // Test with a large model configuration
    PlacementPlan plan = topology_->computePlacement(
        "qwen2_72b", 80, 8192, 29568, 152064, 64, 8, "Q4_K",
        40ULL * 1024 * 1024 * 1024, "" // 40GB
    );

    EXPECT_TRUE(plan.isValid());
    EXPECT_EQ(plan.n_layers, 80);
    EXPECT_EQ(plan.layers.size(), 80u);
}

TEST_F(Test__PlacementStrategy_MPI_Integration, DifferentQuantTypesProduceSamePlan)
{
    // Plan should be the same regardless of quant type (for CPU-only)
    PlacementPlan plan_q4_0 = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q4_0",
        500 * 1024 * 1024, "CPUOnly");

    PlacementPlan plan_q8_0 = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q8_0",
        800 * 1024 * 1024, "CPUOnly");

    // Layer placements should be identical for CPU-only strategy
    ASSERT_EQ(plan_q4_0.layers.size(), plan_q8_0.layers.size());

    for (size_t i = 0; i < plan_q4_0.layers.size(); ++i)
    {
        EXPECT_EQ(plan_q4_0.layers[i].device, plan_q8_0.layers[i].device)
            << "Layer " << i << " has different device for different quant types";
        EXPECT_EQ(plan_q4_0.layers[i].owner_rank, plan_q8_0.layers[i].owner_rank)
            << "Layer " << i << " has different owner_rank for different quant types";
    }
}

TEST_F(Test__PlacementStrategy_MPI_Integration, KVPrecisionMonotonicLayerFitViaTopologyConvenience)
{
    if (!topology_->has_accelerator())
    {
        GTEST_SKIP() << "No accelerator detected; KV precision does not affect CPU-only placement.";
    }

    constexpr int n_layers = 4096;
    constexpr size_t estimated_memory = 0;

    PlacementPlan plan_auto = topology_->computePlacement(
        "kv_precision_probe", n_layers, 1024, 4096, 32000, 16, 16, "Q4_0",
        estimated_memory, "auto", "GPUFirst");

    PlacementPlan plan_fp16 = topology_->computePlacement(
        "kv_precision_probe", n_layers, 1024, 4096, 32000, 16, 16, "Q4_0",
        estimated_memory, "fp16", "GPUFirst");

    PlacementPlan plan_q8_1 = topology_->computePlacement(
        "kv_precision_probe", n_layers, 1024, 4096, 32000, 16, 16, "Q4_0",
        estimated_memory, "q8_1", "GPUFirst");

    const int gpu_layers_auto = countGPULayers(plan_auto);
    const int gpu_layers_fp16 = countGPULayers(plan_fp16);
    const int gpu_layers_q8_1 = countGPULayers(plan_q8_1);

    EXPECT_GE(gpu_layers_fp16, gpu_layers_auto)
        << "FP16 KV cache should fit at least as many GPU layers as AUTO";
    EXPECT_GE(gpu_layers_q8_1, gpu_layers_fp16)
        << "Q8_1 KV cache should fit at least as many GPU layers as FP16";

    struct LayerFitCounts
    {
        int auto_layers;
        int fp16_layers;
        int q8_1_layers;
    };

    LayerFitCounts my_counts{gpu_layers_auto, gpu_layers_fp16, gpu_layers_q8_1};
    std::vector<LayerFitCounts> all_counts(world_size_);
    MPI_Allgather(&my_counts, sizeof(LayerFitCounts), MPI_BYTE,
                  all_counts.data(), sizeof(LayerFitCounts), MPI_BYTE, MPI_COMM_WORLD);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_counts[r].auto_layers, all_counts[0].auto_layers)
            << "Rank " << r << " has different AUTO layer-fit count";
        EXPECT_EQ(all_counts[r].fp16_layers, all_counts[0].fp16_layers)
            << "Rank " << r << " has different FP16 layer-fit count";
        EXPECT_EQ(all_counts[r].q8_1_layers, all_counts[0].q8_1_layers)
            << "Rank " << r << " has different Q8_1 layer-fit count";
    }
}

// MPI-aware main function
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
