/**
 * @file Test__ClusterInventoryGatherer.cpp
 * @brief Unit tests for ClusterInventoryGatherer free function
 *
 * Tests the extracted gatherClusterInventory() function that was
 * previously part of OrchestrationRunner.
 *
 * @date April 2026
 */

#include <gtest/gtest.h>

#include "planning/ClusterInventoryGatherer.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =========================================================================
// Single-Rank Tests (no real MPI needed)
// =========================================================================

TEST(Test__ClusterInventoryGatherer, SingleRank_PopulatesLocalDevices)
{
    // nullptr mpi_ctx → single-rank path
    auto inventory = gatherClusterInventory(nullptr);

    EXPECT_EQ(inventory.world_size, 1);
    EXPECT_EQ(inventory.node_count, 1);
    EXPECT_EQ(inventory.ranks.size(), 1u);

    const auto& rank0 = inventory.ranks[0];
    EXPECT_EQ(rank0.rank, 0);
    EXPECT_EQ(rank0.hostname, "localhost");
    // Should have discovered at least the CPU
    EXPECT_EQ(rank0.cpu.type, DeviceType::CPU);
}

TEST(Test__ClusterInventoryGatherer, SingleRank_WorldSizeOne)
{
    // MPI context with world_size=1 should use the local-only path
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
    auto inventory = gatherClusterInventory(mpi_ctx);

    EXPECT_EQ(inventory.world_size, 1);
    EXPECT_EQ(inventory.ranks.size(), 1u);
}

TEST(Test__ClusterInventoryGatherer, ExplicitTPDevices_OverridesDetected)
{
    // Provide explicit TP devices — should override detected GPUs
    std::vector<GlobalDeviceAddress> tp_devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1),
    };

    auto inventory = gatherClusterInventory(nullptr, tp_devices);

    EXPECT_EQ(inventory.world_size, 1);
    EXPECT_EQ(inventory.total_gpus, 2);
    ASSERT_EQ(inventory.ranks[0].gpus.size(), 2u);
    EXPECT_EQ(inventory.ranks[0].gpus[0].type, DeviceType::CUDA);
    EXPECT_EQ(inventory.ranks[0].gpus[0].local_device_id, 0);
    EXPECT_EQ(inventory.ranks[0].gpus[1].type, DeviceType::CUDA);
    EXPECT_EQ(inventory.ranks[0].gpus[1].local_device_id, 1);
}

TEST(Test__ClusterInventoryGatherer, ExplicitROCmDevices_SetsCorrectType)
{
    std::vector<GlobalDeviceAddress> tp_devices = {
        GlobalDeviceAddress::rocm(0),
    };

    auto inventory = gatherClusterInventory(nullptr, tp_devices);

    EXPECT_EQ(inventory.total_gpus, 1);
    EXPECT_EQ(inventory.ranks[0].gpus[0].type, DeviceType::ROCm);
}

TEST(Test__ClusterInventoryGatherer, EmptyTPDevices_UsesDetected)
{
    // Empty explicit list → uses whatever DeviceManager finds
    std::vector<GlobalDeviceAddress> empty;
    auto inventory = gatherClusterInventory(nullptr, empty);

    EXPECT_EQ(inventory.world_size, 1);
    // GPU count depends on hardware, but should not crash
    EXPECT_GE(inventory.total_gpus, 0);
}
