/**
 * @file Test__PhysicalInventoryAggregation.cpp
 * @brief Pure adversarial coverage of physical discovery projection.
 *
 * Repeated rank views, ordinal remapping and hostname aliases must not multiply
 * physical capacity. These tests open no devices and maintain no live ledger;
 * PhysicalMemoryAuthority remains the only admission/allocation authority.
 */
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "backends/DeviceUUID.h"
#include "planning/RankHardwareOwnership.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>

using namespace llaminar2;

namespace
{
    /** @brief Two MPI observers of one node, with intentionally remapped ordinals. */
    ClusterInventory overlappingInventory()
    {
        ClusterInventory inventory;
        inventory.world_size = 2;
        for (int peer = 0; peer < 2; ++peer)
        {
            RankInventory rank;
            rank.rank = peer;
            rank.local_rank = peer;
            rank.node_id = 0;
            rank.hostname = peer == 0 ? "hostname" : "hostname-alias";
            rank.cpu.numa_node = peer;
            rank.cpu_memory_bytes = 100;
            rank.cpu_cores = 2;
            for (int numa = 0; numa < 2; ++numa)
            {
                CPUSocketInfo socket;
                socket.socket_id = numa;
                socket.numa_node = numa;
                socket.memory_bytes = 100;
                socket.physical_cores = {numa * 2, numa * 2 + 1};
                rank.cpu_socket_info.push_back(socket);
            }
            for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            {
                DeviceInfo gpu;
                gpu.type = backend;
                gpu.uuid = "same-byte-uuid-across-vendors";
                gpu.local_device_id = peer == 0 ? 5 : 0;
                gpu.memory_bytes = 50;
                gpu.free_memory_bytes = 40 - peer; // Time-varying observations are legal.
                gpu.compute_units = 16;
                rank.gpus.push_back(gpu);
            }
            inventory.ranks.push_back(rank);
        }
        return inventory;
    }
}

TEST(PhysicalInventoryAggregation, SharedGPUViewsAreNotAdditionalCapacity)
{
    auto inventory = overlappingInventory();
    inventory.buildNodeAggregations();
    EXPECT_EQ(inventory.total_gpus, 2);
    EXPECT_EQ(inventory.total_gpu_memory, 100u);
    EXPECT_EQ(inventory.total_cpu_memory, 200u);
    ASSERT_EQ(inventory.nodes.size(), 1u);
    EXPECT_EQ(inventory.nodes[0].total_cpu_cores, 4);
    EXPECT_EQ(inventory.nodes[0].ranks, (std::vector<int>{0, 1}));
}

TEST(PhysicalInventoryAggregation, RankConnectionsUsePhysicalMembershipNotNamesOrRates)
{
    auto inventory = overlappingInventory();
    const auto local = inventory.connectionBetweenRanks(0, 1);
    EXPECT_EQ(local.locality(), RankConnectionLocality::SameNode); // Different hostname aliases.
    EXPECT_EQ(local.sourceRank(), 0);
    EXPECT_EQ(local.destinationRank(), 1);
    EXPECT_EQ(local.sourceNode(), 0);
    EXPECT_EQ(local.destinationNode(), 0);
    EXPECT_EQ(inventory.connectionBetweenRanks(1, 1).locality(), RankConnectionLocality::SameRank);
    for (float bandwidth : {0.001f, 100000.0f})
    {
        inventory.ranks[1].cpu.memory_bandwidth_gbps = bandwidth;
        inventory.ranks[1].cpu.tflops_fp16 = bandwidth;
        EXPECT_EQ(inventory.connectionBetweenRanks(0, 1), local);
    }
    // Equal hostnames and NUMA IDs cannot turn different machines into peers.
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    inventory.ranks[1].cpu.numa_node = inventory.ranks[0].cpu.numa_node;
    inventory.ranks[1].node_id = 41;
    for (float bandwidth : {0.001f, 100000.0f})
    {
        inventory.ranks[1].cpu.memory_bandwidth_gbps = bandwidth;
        const auto remote = inventory.connectionBetweenRanks(0, 1);
        EXPECT_EQ(remote.locality(), RankConnectionLocality::CrossNode);
        EXPECT_EQ(remote.destinationNode(), 41);
        EXPECT_EQ(inventory.connectionBetweenRanks(1, 0).locality(), remote.locality());
    }
}

TEST(PhysicalInventoryAggregation, MissingRankIdentityCannotDefaultToLocality)
{
    auto inventory = overlappingInventory();
    for (int bad : {-1, 2, 99})
    {
        EXPECT_THROW(inventory.connectionBetweenRanks(bad, bad), std::invalid_argument);
        EXPECT_THROW(inventory.connectionBetweenRanks(0, bad), std::invalid_argument);
        EXPECT_THROW(inventory.connectionBetweenRanks(bad, 0), std::invalid_argument);
    }
    for (int defect = 0; defect != 4; ++defect)
    {
        auto malformed = inventory;
        if (defect == 0) malformed.world_size = 3;
        if (defect == 1) malformed.ranks.pop_back();
        if (defect == 2) malformed.ranks[1].rank = 0;
        if (defect == 3) malformed.ranks[1].node_id = -1;
        EXPECT_THROW(malformed.connectionBetweenRanks(0, 1), std::invalid_argument);
    }
}

TEST(PhysicalInventoryAggregation, CPUOwnershipRequiresObservedNUMAIdentity)
{
    auto rank = overlappingInventory().ranks.front();
    rank.local_rank = 9;
    rank.cpu.numa_node = 7;
    EXPECT_TRUE(rankOwnsCPUNode(rank, 7));
    EXPECT_TRUE(rankOwnsCPUNode(rank, -1));
    EXPECT_FALSE(rankOwnsCPUNode(rank, 9));
    EXPECT_FALSE(rankOwnsCPUNode(rank, 0)); // Visible remote memory is not local ownership.
    EXPECT_FALSE(rankOwnsCPUNode(rank, -2));

    rank.cpu.numa_node = -1; // Explicit whole-host observation.
    rank.cpu_socket_info[0].numa_node = 3;
    rank.cpu_socket_info[1].numa_node = 7;
    EXPECT_TRUE(rankOwnsCPUNode(rank, 3));
    EXPECT_TRUE(rankOwnsCPUNode(rank, 7));
    EXPECT_TRUE(rankOwnsCPUNode(rank, -1));
    EXPECT_FALSE(rankOwnsCPUNode(rank, 0));
    rank.cpu_socket_info.clear();
    EXPECT_FALSE(rankOwnsCPUNode(rank, 7));
    EXPECT_FALSE(rankOwnsCPUNode(rank, -1));
}

TEST(PhysicalInventoryAggregation, RebuildIsIdempotentAndEmptyMeansNoNodes)
{
    auto inventory = overlappingInventory();
    for (int repeat = 0; repeat < 20; ++repeat)
    {
        inventory.buildNodeAggregations();
        EXPECT_EQ(inventory.total_gpu_memory, 100u);
        EXPECT_EQ(inventory.total_cpu_memory, 200u);
        EXPECT_EQ(inventory.nodes[0].ranks.size(), 2u);
    }
    inventory.ranks.clear();
    inventory.buildNodeAggregations();
    EXPECT_TRUE(inventory.nodes.empty());
    EXPECT_EQ(inventory.node_count, 0);
    EXPECT_EQ(inventory.total_gpus, 0);
    EXPECT_EQ(inventory.total_cpu_memory, 0u);
}

TEST(PhysicalInventoryAggregation, SameUUIDOnDifferentNodesRemainsDistinct)
{
    auto inventory = overlappingInventory();
    inventory.ranks[1].node_id = 1;
    inventory.buildNodeAggregations();
    EXPECT_EQ(inventory.total_gpus, 4);
    EXPECT_EQ(inventory.total_gpu_memory, 200u);
    EXPECT_EQ(inventory.total_cpu_memory, 400u);
}

TEST(PhysicalInventoryAggregation, DistinctPartitionsMayShareTheSameOrdinalAndBus)
{
    auto inventory = overlappingInventory();
    inventory.ranks[1].gpus[0].uuid = "different-partition";
    inventory.ranks[1].gpus[0].local_device_id = 5;
    inventory.buildNodeAggregations();
    EXPECT_EQ(inventory.total_gpus, 3);
    EXPECT_EQ(inventory.total_gpu_memory, 150u);
}

TEST(PhysicalInventoryAggregation, RejectsMissingAndConflictingObservedIdentities)
{
    for (int defect = 0; defect < 6; ++defect)
    {
        auto inventory = overlappingInventory();
        inventory.buildNodeAggregations();
        auto &rank = inventory.ranks[1];
        if (defect == 0) rank.gpus[0].uuid.clear();
        if (defect == 1) ++rank.gpus[0].memory_bytes;
        if (defect == 2) ++rank.gpus[0].compute_units;
        if (defect == 3) ++rank.cpu_socket_info[0].memory_bytes;
        if (defect == 4) rank.node_id = -1;
        if (defect == 5) ++rank.gpus[0].last_level_cache_bytes;
        EXPECT_THROW(inventory.buildNodeAggregations(), std::invalid_argument);
        EXPECT_EQ(inventory.total_gpu_memory, 100u); // No half-published summary.
        EXPECT_EQ(inventory.total_cpu_memory, 200u);
    }
}

TEST(PhysicalInventoryAggregation, RepeatedNUMAViewsWithoutSocketDetailAreUnioned)
{
    auto inventory = overlappingInventory();
    for (auto &rank : inventory.ranks)
    {
        rank.cpu_socket_info.clear();
        rank.cpu.numa_node = 7;
    }
    inventory.buildNodeAggregations();
    EXPECT_EQ(inventory.total_cpu_memory, 100u);
    EXPECT_EQ(inventory.nodes[0].total_cpu_cores, 2);
    inventory.ranks[0].cpu.numa_node = -1;
    EXPECT_THROW(inventory.buildNodeAggregations(), std::invalid_argument);
}

TEST(PhysicalInventoryAggregation, UUIDEncodingPreservesUnsignedDriverBytes)
{
    std::array<char, 16> bytes{};
    EXPECT_THROW(formatDeviceUUID(bytes), std::invalid_argument);
    bytes[0] = static_cast<char>(0x80);
    bytes[15] = static_cast<char>(0xff);
    EXPECT_EQ(formatDeviceUUID(bytes), "800000000000000000000000000000ff");
}

TEST(PhysicalInventoryAggregation, CapacityAndCoreOverflowCannotPublishASummary)
{
    auto inventory = overlappingInventory();
    inventory.ranks[1].node_id = 1;
    for (auto &rank : inventory.ranks)
    {
        rank.cpu_socket_info.clear();
        rank.cpu_memory_bytes = std::numeric_limits<size_t>::max();
    }
    EXPECT_THROW(inventory.buildNodeAggregations(), std::overflow_error);
    EXPECT_TRUE(inventory.nodes.empty());
    for (auto &rank : inventory.ranks)
    {
        rank.node_id = 0;
        rank.cpu_memory_bytes = 10;
        rank.cpu_cores = std::numeric_limits<int>::max();
    }
    EXPECT_THROW(inventory.buildNodeAggregations(), std::overflow_error);
    inventory.ranks[0].cpu_cores = -1;
    EXPECT_THROW(inventory.buildNodeAggregations(), std::invalid_argument);
}
