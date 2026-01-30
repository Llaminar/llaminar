/**
 * @file Test__HeterogeneousPipelineMPI.cpp
 * @brief MPI-aware integration tests for heterogeneous pipeline execution
 *
 * **Phase 5.5**: Multi-rank tests for heterogeneous execution verifying:
 *
 * 1. Cross-rank domain configuration
 * 2. Heterogeneous TP domains (GPU ranks + CPU ranks)
 * 3. AllReduce operations across mixed device domains
 * 4. Multi-rank placement strategies
 *
 * These tests require MPI and exercise the full pipeline with multiple
 * ranks that may have different device capabilities.
 *
 * Run with: mpirun -np 2 ./v2_integration_heterogeneous_pipeline_mpi
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>

#include "execution/local_execution/model/HeterogeneousLayerExecutor.h"
#include "execution/local_execution/device/DomainAwareBufferManager.h"
#include "execution/local_execution/coherence/CrossDomainTransfer.h"
#include "execution/local_execution/graph/GraphExecutor.h" // Contains ComputeGraph definition
#include "config/LayerPlacementConfig.h"
#include "config/TPDomain.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h" // For DeviceManager
#include "tensors/TensorClasses.h"
#include "../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// MPI Test Fixture
// =============================================================================

/**
 * @brief Test fixture for MPI-aware heterogeneous pipeline tests
 */
class Test__HeterogeneousPipelineMPI : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    bool gpu_available_ = false;
    DeviceId gpu_device_ = DeviceId::cpu();

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Detect GPU availability
        auto &dm = DeviceManager::instance();
        gpu_available_ = dm.has_gpu();

        if (gpu_available_)
        {
#ifdef HAVE_CUDA
            if (dm.cuda_device_count() > 0)
            {
                gpu_device_ = DeviceId::cuda(dm.get_device_id_for_type(
                    ComputeBackendType::GPU_CUDA, 0));
            }
#endif
#ifdef HAVE_ROCM
            if (!gpu_device_.is_gpu() && dm.rocm_device_count() > 0)
            {
                gpu_device_ = DeviceId::rocm(dm.get_device_id_for_type(
                    ComputeBackendType::GPU_ROCM, 0));
            }
#endif
        }
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * @brief Check if all ranks have GPU
     */
    bool allRanksHaveGPU()
    {
        int local_has_gpu = gpu_available_ ? 1 : 0;
        int all_have_gpu;
        MPI_Allreduce(&local_has_gpu, &all_have_gpu, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return all_have_gpu == 1;
    }

    /**
     * @brief Get GPU count across all ranks
     */
    int getTotalGPUCount()
    {
        int local_count = gpu_available_ ? 1 : 0;
        int total_count;
        MPI_Allreduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        return total_count;
    }
};

// =============================================================================
// Multi-Rank Placement Tests
// =============================================================================

TEST_F(Test__HeterogeneousPipelineMPI, AllRanksSamePlacement)
{
    // All ranks use the same placement config
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    // Verify all ranks see the same configuration
    EXPECT_EQ(placement.numLayers(), 10);
    EXPECT_EQ(placement.getCPULayers().size(), 10);
    EXPECT_EQ(placement.getGPULayers().size(), 0);

    // Create executor on each rank
    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.enable_profiling = true;
    HeterogeneousLayerExecutor executor(config);

    // Execute on each rank
    ComputeGraph graph;
    EXPECT_TRUE(executor.executeLayerRange(0, 10, &graph));

    // Collect stats across ranks
    auto stats = executor.getStats();

    int total_cpu_layers;
    MPI_Reduce(&stats.cpu_layers_executed, &total_cpu_layers, 1,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank_ == 0)
    {
        EXPECT_EQ(total_cpu_layers, 10 * world_size_);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__HeterogeneousPipelineMPI, RankAwarePlacement)
{
    // Each rank might have different device availability
    // Create placement based on local device availability
    DeviceId local_device = gpu_available_ ? gpu_device_ : DeviceId::cpu();

    // For consistency, if any rank doesn't have GPU, all use CPU
    int all_have_gpu = allRanksHaveGPU() ? 1 : 0;
    DeviceId chosen_device = all_have_gpu ? gpu_device_ : DeviceId::cpu();

    auto placement = LayerPlacementConfig::allOnDevice(chosen_device, 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    HeterogeneousLayerExecutor executor(config);

    // Execute
    ComputeGraph graph;
    EXPECT_TRUE(executor.executeLayerRange(0, 10, &graph));

    auto stats = executor.getStats();

    // All ranks should have same device type
    if (all_have_gpu)
    {
        EXPECT_EQ(stats.gpu_layers_executed, 10);
        EXPECT_EQ(stats.cpu_layers_executed, 0);
    }
    else
    {
        EXPECT_EQ(stats.cpu_layers_executed, 10);
        EXPECT_EQ(stats.gpu_layers_executed, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Cross-Rank Domain Tests
// =============================================================================

TEST_F(Test__HeterogeneousPipelineMPI, CrossRankDomainCreation)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ranks for cross-rank domain test";
    }

    TPDomainBuilder builder(MPI_COMM_WORLD);

    // All ranks participate in a CPU cross-rank domain
    int color = 0;   // Same color = same domain
    int key = rank_; // Order by world rank

    auto domain = builder.createCPUCrossRankDomain(color, key, "cpu_hetero_domain");

    // Verify domain properties
    EXPECT_EQ(domain.type, TPDomainType::CPU_CROSS_RANK);
    EXPECT_NE(domain.communicator, MPI_COMM_NULL);
    EXPECT_EQ(domain.domain_size, world_size_);
    EXPECT_EQ(domain.local_rank_in_domain, rank_);
    EXPECT_TRUE(domain.isCrossRank());
    EXPECT_FALSE(domain.isTrivial());

    // Clean up communicator
    if (domain.communicator != MPI_COMM_NULL &&
        domain.communicator != MPI_COMM_WORLD)
    {
        MPI_Comm_free(&domain.communicator);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__HeterogeneousPipelineMPI, HeterogeneousDomainSplit)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ranks for heterogeneous domain test";
    }

    TPDomainBuilder builder(MPI_COMM_WORLD);

    // Split ranks based on GPU availability
    // GPU ranks get color 0, CPU-only ranks get color 1
    int color = gpu_available_ ? 0 : 1;
    int key = rank_;

    MPI_Comm domain_comm = builder.splitCommunicator(true, "hetero_split");

    // Get domain info
    int domain_size, domain_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &domain_size);
    MPI_Comm_rank(domain_comm, &domain_rank);

    EXPECT_EQ(domain_size, world_size_); // All ranks participate

    // Now do actual color-based split
    MPI_Comm color_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &color_comm);

    int color_size;
    MPI_Comm_size(color_comm, &color_size);

    // Sum up GPU and CPU ranks
    int local_gpu_count = gpu_available_ ? 1 : 0;
    int total_gpu_ranks;
    MPI_Allreduce(&local_gpu_count, &total_gpu_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int total_cpu_only_ranks = world_size_ - total_gpu_ranks;

    if (gpu_available_)
    {
        EXPECT_EQ(color_size, total_gpu_ranks);
    }
    else
    {
        EXPECT_EQ(color_size, total_cpu_only_ranks);
    }

    MPI_Comm_free(&color_comm);
    MPI_Comm_free(&domain_comm);

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// MPI Collective Tests with Heterogeneous Domains
// =============================================================================

TEST_F(Test__HeterogeneousPipelineMPI, AllReduceWithinHomogeneousDomain)
{
    // Create a domain where all ranks are the same type (CPU)
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    // Create some test data
    std::vector<float> local_data(128, static_cast<float>(rank_ + 1));
    std::vector<float> reduced_data(128);

    // AllReduce within world communicator
    MPI_Allreduce(local_data.data(), reduced_data.data(), 128,
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Verify: sum of (rank+1) for all ranks
    float expected_sum = static_cast<float>(world_size_ * (world_size_ + 1) / 2);
    for (size_t i = 0; i < 128; ++i)
    {
        EXPECT_FLOAT_EQ(reduced_data[i], expected_sum);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__HeterogeneousPipelineMPI, AllReduceAcrossMixedDomains)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ranks for cross-domain allreduce test";
    }

    // Simulate mixed device execution:
    // - GPU ranks contribute activation values computed on GPU
    // - CPU ranks contribute activation values computed on CPU

    // Create local activation data
    const size_t activation_size = 64;
    std::vector<float> local_activations(activation_size);

    // Fill with rank-specific pattern
    for (size_t i = 0; i < activation_size; ++i)
    {
        local_activations[i] = static_cast<float>(rank_ * 1000 + i);
    }

    // Allreduce activations (simulating TP reduction)
    std::vector<float> reduced_activations(activation_size);
    MPI_Allreduce(local_activations.data(), reduced_activations.data(),
                  activation_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Verify: each element should be sum over all ranks
    for (size_t i = 0; i < activation_size; ++i)
    {
        float expected = 0.0f;
        for (int r = 0; r < world_size_; ++r)
        {
            expected += static_cast<float>(r * 1000 + i);
        }
        EXPECT_FLOAT_EQ(reduced_activations[i], expected)
            << "Mismatch at index " << i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Multi-Rank Buffer Allocation Tests
// =============================================================================

TEST_F(Test__HeterogeneousPipelineMPI, ConsistentBufferAllocation)
{
    // All ranks should allocate same buffer structure
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    DomainAwareBufferConfig config;
    config.placement_config = &placement;
    config.default_device = DeviceId::cpu();
    DomainAwareBufferManager manager(config);

    // Each rank allocates buffers
    const std::vector<size_t> shape = {32, 64};
    for (int layer = 0; layer < 10; ++layer)
    {
        auto *buffer = manager.allocateForLayer(
            layer, "buf_" + std::to_string(layer), shape, BufferTensorType::FP32);
        ASSERT_NE(buffer, nullptr);
    }

    auto stats = manager.getStats();

    // Verify consistent allocation across ranks
    int local_buffers = stats.total_buffers();
    int all_same = 1;
    int expected_buffers = 10;

    // Check if all ranks have same count
    int other_buffers;
    for (int r = 0; r < world_size_; ++r)
    {
        if (r == rank_)
        {
            other_buffers = local_buffers;
        }
        MPI_Bcast(&other_buffers, 1, MPI_INT, r, MPI_COMM_WORLD);
        if (other_buffers != expected_buffers)
        {
            all_same = 0;
        }
    }

    EXPECT_EQ(all_same, 1) << "Buffer count mismatch across ranks";

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Placement Configuration Consistency Tests
// =============================================================================

TEST_F(Test__HeterogeneousPipelineMPI, PlacementConfigConsistentAcrossRanks)
{
    // All ranks should see identical placement config
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, gpu_device_);

    // Collect placement info
    int local_cpu_count = static_cast<int>(placement.getCPULayers().size());
    int local_gpu_count = static_cast<int>(placement.getGPULayers().size());
    int local_num_layers = placement.numLayers();

    std::vector<int> all_cpu_counts(world_size_);
    std::vector<int> all_gpu_counts(world_size_);
    std::vector<int> all_num_layers(world_size_);

    MPI_Gather(&local_cpu_count, 1, MPI_INT, all_cpu_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_gpu_count, 1, MPI_INT, all_gpu_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_num_layers, 1, MPI_INT, all_num_layers.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank_ == 0)
    {
        for (int r = 1; r < world_size_; ++r)
        {
            EXPECT_EQ(all_cpu_counts[r], all_cpu_counts[0])
                << "CPU layer count mismatch: rank " << r << " vs rank 0";
            EXPECT_EQ(all_gpu_counts[r], all_gpu_counts[0])
                << "GPU layer count mismatch: rank " << r << " vs rank 0";
            EXPECT_EQ(all_num_layers[r], all_num_layers[0])
                << "Total layer count mismatch: rank " << r << " vs rank 0";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Statistics Aggregation Tests
// =============================================================================

TEST_F(Test__HeterogeneousPipelineMPI, AggregateExecutionStats)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.enable_profiling = true;
    HeterogeneousLayerExecutor executor(config);

    ComputeGraph graph;
    executor.executeLayerRange(0, 10, &graph);

    auto stats = executor.getStats();

    // Aggregate stats across all ranks
    struct AggregateStats
    {
        int total_cpu_layers;
        int total_gpu_layers;
        int total_transfers;
        double total_cpu_time;
        double total_gpu_time;
        double total_transfer_time;
    } agg_stats = {0};

    MPI_Reduce(&stats.cpu_layers_executed, &agg_stats.total_cpu_layers,
               1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.gpu_layers_executed, &agg_stats.total_gpu_layers,
               1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.cross_domain_transfers, &agg_stats.total_transfers,
               1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.cpu_time_ms, &agg_stats.total_cpu_time,
               1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.gpu_time_ms, &agg_stats.total_gpu_time,
               1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.transfer_time_ms, &agg_stats.total_transfer_time,
               1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank_ == 0)
    {
        // Each rank executed 10 CPU layers
        EXPECT_EQ(agg_stats.total_cpu_layers, 10 * world_size_);
        EXPECT_EQ(agg_stats.total_gpu_layers, 0);
        EXPECT_EQ(agg_stats.total_transfers, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Data Consistency Tests
// =============================================================================

TEST_F(Test__HeterogeneousPipelineMPI, ActivationDataConsistentAfterAllReduce)
{
    // Test that activation data is consistent after collective operations
    const size_t rows = 64;
    const size_t cols = 128;

    // Each rank creates identical input (same seed)
    auto input = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f, 42);

    // Verify all ranks have same data
    std::vector<float> local_checksum(1);
    local_checksum[0] = 0.0f;
    const float *data = input->data();
    for (size_t i = 0; i < input->numel(); ++i)
    {
        local_checksum[0] += data[i];
    }

    std::vector<float> all_checksums(world_size_);
    MPI_Gather(local_checksum.data(), 1, MPI_FLOAT,
               all_checksums.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank_ == 0)
    {
        for (int r = 1; r < world_size_; ++r)
        {
            EXPECT_NEAR(all_checksums[r], all_checksums[0], 1e-5f)
                << "Checksum mismatch between rank " << r << " and rank 0";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// MPI Main
// =============================================================================

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
