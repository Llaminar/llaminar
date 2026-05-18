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

#include <unistd.h>

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

    const auto &rank0 = inventory.ranks[0];
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

// =========================================================================
// CPU Hardware Enrichment Tests (data correctness — fixes from May 2026)
// =========================================================================

TEST(Test__ClusterInventoryGatherer, SingleRank_CPUCoresAreReasonable)
{
    // Regression: cpu_cores was hardcoded to 1, now detected from HardwareInventory
    auto inventory = gatherClusterInventory(nullptr);
    const auto &rank0 = inventory.ranks[0];

    // Any real machine has at least 1 physical core; HardwareInventory should detect > 0
    EXPECT_GT(rank0.cpu_cores, 0)
        << "cpu_cores should be detected from HardwareInventory, not left at default 0/1";

    // cpu_sockets should be at least 1
    EXPECT_GE(rank0.cpu_sockets, 1)
        << "cpu_sockets should be detected (was previously left at 0)";
}

TEST(Test__ClusterInventoryGatherer, SingleRank_CPUComputeUnitsAreThreads)
{
    // Regression: cpu.compute_units was set to memory_channels, not logical threads
    auto inventory = gatherClusterInventory(nullptr);
    const auto &rank0 = inventory.ranks[0];

    // compute_units should be >= cpu_cores (threads >= physical cores)
    EXPECT_GE(rank0.cpu.compute_units, rank0.cpu_cores)
        << "cpu.compute_units (logical threads) must be >= cpu_cores (physical cores)";

    // Sanity: if HT is present, threads = 2x cores
    // If no HT, threads == cores. Either way, should be > 0
    EXPECT_GT(rank0.cpu.compute_units, 0)
        << "cpu.compute_units should represent logical thread count, not memory channels";
}

TEST(Test__ClusterInventoryGatherer, SingleRank_CPUSocketInfoPopulated)
{
    // Regression: cpu_socket_info was empty (never populated in gatherer)
    auto inventory = gatherClusterInventory(nullptr);
    const auto &rank0 = inventory.ranks[0];

    EXPECT_FALSE(rank0.cpu_socket_info.empty())
        << "cpu_socket_info should be populated from HardwareInventory::detect()";

    // First socket should have a model name
    EXPECT_FALSE(rank0.cpu_socket_info[0].model_name.empty())
        << "Socket model name should be detected from /proc/cpuinfo";

    // Socket should have some physical cores
    EXPECT_GT(rank0.cpu_socket_info[0].num_physical_cores(), 0);
}

TEST(Test__ClusterInventoryGatherer, SingleRank_MemoryIsPerSocket)
{
    // Regression: memory was full-machine sysconf value, causing double-counting
    // when multiple ranks on same node aggregated their values.
    auto inventory = gatherClusterInventory(nullptr);
    const auto &rank0 = inventory.ranks[0];

    // Memory should be > 0
    EXPECT_GT(rank0.cpu_memory_bytes, 0u)
        << "cpu_memory_bytes should be populated";

    // For single-rank (local_rank=0 with only 1 socket in fallback path,
    // or the full machine total), memory should not exceed what sysconf reports.
    // Key invariant: if multiple sockets exist, per-rank memory < total machine memory.
    if (rank0.cpu_sockets > 1)
    {
        // In single-rank mode with local_rank=0, enrichCPUInfo hits the socket[0] path
        // Memory should be roughly total/sockets (the NUMA-local portion)
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        size_t total_system_ram = static_cast<size_t>(pages) * static_cast<size_t>(page_size);

        // Per-socket memory should be significantly less than total
        // (with some tolerance for NUMA asymmetry)
        EXPECT_LT(rank0.cpu_memory_bytes, total_system_ram)
            << "Per-socket memory should be less than total system RAM "
            << "(was previously the full sysconf value causing 2x over-report)";
    }
}

TEST(Test__ClusterInventoryGatherer, SingleRank_CPUDeviceInfoNameSet)
{
    // The CPU DeviceInfo.name should have a meaningful processor name, not be empty
    auto inventory = gatherClusterInventory(nullptr);
    const auto &rank0 = inventory.ranks[0];

    EXPECT_FALSE(rank0.cpu.name.empty())
        << "cpu.name should be populated from HardwareInventory (model_name)";
}

// =========================================================================
// Node Aggregation Tests (buildNodeAggregations correctness)
// =========================================================================

TEST(Test__ClusterInventoryGatherer, NodeAggregation_PerSocketMemorySumsCorrectly)
{
    // Simulate 2 ranks on 1 node, each reporting per-socket memory
    // Regression: both ranks reported full-machine RAM → 2x over-count
    ClusterInventory inventory;
    inventory.world_size = 2;

    RankInventory rank0;
    rank0.rank = 0;
    rank0.node_id = 0;
    rank0.hostname = "node0";
    rank0.cpu_cores = 28;                                 // 28 cores on socket 0
    rank0.cpu_memory_bytes = 384ULL * 1024 * 1024 * 1024; // 384 GB (socket 0)

    RankInventory rank1;
    rank1.rank = 1;
    rank1.node_id = 0;
    rank1.hostname = "node0";
    rank1.cpu_cores = 28;                                 // 28 cores on socket 1
    rank1.cpu_memory_bytes = 384ULL * 1024 * 1024 * 1024; // 384 GB (socket 1)

    inventory.ranks = {rank0, rank1};
    inventory.buildNodeAggregations();

    ASSERT_EQ(inventory.nodes.size(), 1u);
    const auto &node = inventory.nodes[0];

    // Total memory should be 768 GB (sum of both sockets), NOT 1.5 TB
    size_t expected_total = 768ULL * 1024 * 1024 * 1024;
    EXPECT_EQ(node.total_cpu_memory, expected_total)
        << "Node total memory should be sum of per-socket values (768 GB), "
        << "not 2x full-machine sysconf values";

    // Total cores should be 56 (28 per socket × 2 ranks)
    EXPECT_EQ(node.total_cpu_cores, 56)
        << "Node total cores should sum per-socket cores from each rank";
}

TEST(Test__ClusterInventoryGatherer, NodeAggregation_MultiNode)
{
    // 4 ranks across 2 nodes (2 ranks per node, 1 socket per rank)
    ClusterInventory inventory;
    inventory.world_size = 4;

    RankInventory ranks[4];
    for (int i = 0; i < 4; ++i)
    {
        ranks[i].rank = i;
        ranks[i].node_id = i / 2; // ranks 0,1 on node 0; ranks 2,3 on node 1
        ranks[i].hostname = "node" + std::to_string(i / 2);
        ranks[i].cpu_cores = 32;
        ranks[i].cpu_memory_bytes = 256ULL * 1024 * 1024 * 1024; // 256 GB per socket
    }

    inventory.ranks = {ranks[0], ranks[1], ranks[2], ranks[3]};
    inventory.buildNodeAggregations();

    ASSERT_EQ(inventory.nodes.size(), 2u);

    // Each node: 2 ranks × 32 cores = 64 total cores
    EXPECT_EQ(inventory.nodes[0].total_cpu_cores, 64);
    EXPECT_EQ(inventory.nodes[1].total_cpu_cores, 64);

    // Each node: 2 ranks × 256 GB = 512 GB total
    size_t expected_per_node = 512ULL * 1024 * 1024 * 1024;
    EXPECT_EQ(inventory.nodes[0].total_cpu_memory, expected_per_node);
    EXPECT_EQ(inventory.nodes[1].total_cpu_memory, expected_per_node);

    // Cluster total: 1024 GB
    EXPECT_EQ(inventory.total_cpu_memory, 2 * expected_per_node);
}
