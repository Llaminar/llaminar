/**
 * @file Test__ClusterInventoryGatherer.cpp
 * @brief Device-free coverage of canonical hardware-to-rank projection.
 *
 * Synthetic observations exercise asymmetric NUMA ownership, sparse ordinals,
 * mixed backends, full capacity/link preservation and malformed observations.
 * Real DeviceManager/MPI discovery belongs in ClusterInventoryBootstrap, not
 * the Unit gate. A selector cannot fabricate a GPU because it is not an input.
 */
#include <gtest/gtest.h>
#include "planning/ClusterInventoryGatherer.h"
#include "utils/MPITopology.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <array>
#include <stdexcept>

using namespace llaminar2;

namespace
{
    /** @brief Two unequal, non-contiguously numbered synthetic NUMA endpoints. */
    HardwareInventory hardwareFixture()
    {
        HardwareInventory hardware;
        hardware.cpu_execution = test::kSyntheticCPUExecutionGeometry;
        CPUSocketInfo first;
        first.socket_id = 0;
        first.numa_node = 3;
        first.model_name = "fixture CPU";
        first.physical_cores = {2, 4};
        first.ht_threads = {10, 12};
        first.memory_bytes = 1000;
        first.available_memory_bytes = 600;
        CPUSocketInfo second = first;
        second.socket_id = 1;
        second.numa_node = 7;
        second.physical_cores = {1, 3, 5};
        second.ht_threads = {9, 11, 13};
        second.memory_bytes = 2000;
        second.available_memory_bytes = 1700;
        hardware.cpu_sockets = {first, second};
        hardware.cpu_last_level_cache_bytes = {{0, 32u << 20}, {1, 96u << 20}};
        return hardware;
    }

    /** @brief Fully initialized driver observation; never opens an accelerator. */
    ComputeDevice gpuFixture(ComputeBackendType backend, int ordinal)
    {
        ComputeDevice device{};
        device.type = backend;
        device.device_id = ordinal;
        device.numa_node = 7;
        device.name = "fixture GPU";
        device.uuid = "fixture-" + std::to_string(static_cast<int>(backend)) + "-" + std::to_string(ordinal);
        device.total_memory_bytes = 10000;
        device.free_memory_bytes = 8123;
        device.compute_units = 42;
        device.last_level_cache_bytes = 4u << 20;
        device.compute_capability = 86;
        device.pcie.pcie_gen = 3;
        device.pcie.link_width = 8;
        device.pcie.link_speed_gts = 8.0;
        device.pcie.max_width = 16;
        device.pcie.max_speed_gts = 16.0;
        device.pcie.degraded = true;
        device.pcie.bottleneck_bdf = "0000:81:00.0";
        return device;
    }

    /** @brief Same projection contract for CUDA, ROCm, Vulkan and Metal. */
    class InventoryGPUProjection : public ::testing::TestWithParam<ComputeBackendType> {};
}

TEST(Test__ClusterInventoryGatherer, SingleRankWholeHostIsNotSocketZero)
{
    const auto rank = makeRankInventory(hardwareFixture(), {}, {});
    EXPECT_EQ(rank.cpu_cores, 5);
    EXPECT_EQ(rank.cpu.compute_units, 10);
    EXPECT_EQ(rank.cpu.memory_bytes, 3000u);
    EXPECT_EQ(rank.cpu.free_memory_bytes, 2300u);
    EXPECT_EQ(rank.cpu_memory_bytes, rank.cpu.memory_bytes);
    EXPECT_EQ(rank.cpu.numa_node, -1);
    EXPECT_EQ(rank.cpu_sockets, 2);
    EXPECT_EQ(rank.numa_nodes, 2);
    EXPECT_EQ(rank.cpu_socket_info.size(), 2u);
    EXPECT_EQ(rank.cpu.last_level_cache_bytes, 128u << 20);
}

TEST(Test__ClusterInventoryGatherer, ActualNUMAMembershipNotLocalRankChoosesCPU)
{
    const auto rank = makeRankInventory(hardwareFixture(), {},
        {.rank=4, .node=2, .local_rank=0, .hostname="alias", .cpu_numa_node=7});
    EXPECT_EQ(rank.rank, 4);
    EXPECT_EQ(rank.node_id, 2);
    EXPECT_EQ(rank.local_rank, 0);
    EXPECT_EQ(rank.hostname, "alias");
    EXPECT_EQ(rank.cpu.numa_node, 7);
    EXPECT_EQ(rank.cpu_cores, 3);
    EXPECT_EQ(rank.cpu.compute_units, 6);
    EXPECT_EQ(rank.cpu.memory_bytes, 2000u);
    EXPECT_EQ(rank.cpu.free_memory_bytes, 1700u);
    EXPECT_EQ(rank.cpu.last_level_cache_bytes, 96u << 20);
}

TEST(Test__ClusterInventoryGatherer, ExecutionWorkersRemainDistinctFromPhysicalCores)
{
    for (const int workers : {1, 2, 7})
    {
        const auto rank = makeRankInventory(hardwareFixture(), {},
            {.cpu_numa_node=7, .cpu_worker_threads=workers});
        EXPECT_EQ(rank.cpu_cores, 3);
        EXPECT_EQ(rank.cpu.compute_units, 6);
        EXPECT_EQ(rank.cpuWorkerThreads(), workers);
        const auto wire = MPITopology::serializeRankInventory(rank);
        const auto restored = MPITopology::deserializeRankInventory(wire.data(), wire.size());
        EXPECT_EQ(restored.cpuWorkerThreads(), workers);
        EXPECT_EQ(restored.cpu_cores, 3);
    }
    const auto hardware_only = makeRankInventory(hardwareFixture(), {}, {});
    EXPECT_THROW(hardware_only.cpuWorkerThreads(), std::invalid_argument);
    EXPECT_THROW(makeRankInventory(hardwareFixture(), {}, {.cpu_worker_threads=-1}), std::invalid_argument);
}

TEST(Test__ClusterInventoryGatherer, MissingOrMalformedCPUObservationIsFatal)
{
    EXPECT_THROW(makeRankInventory({}, {}, {}), std::invalid_argument);
    EXPECT_THROW(makeRankInventory(hardwareFixture(), {}, {.rank=-1}), std::invalid_argument);
    EXPECT_THROW(makeRankInventory(hardwareFixture(), {}, {.cpu_numa_node=0}), std::invalid_argument);
    auto hardware = hardwareFixture();
    hardware.cpu_sockets[0].available_memory_bytes = 1001;
    EXPECT_THROW(makeRankInventory(hardware, {}, {}), std::invalid_argument);
}

TEST(Test__ClusterInventoryGatherer, BoundCPUDoesNotHideOtherSocketAccelerators)
{
    auto hardware = hardwareFixture();
    hardware.cuda_devices = {gpuFixture(ComputeBackendType::GPU_CUDA, 9)};
    hardware.rocm_devices = {gpuFixture(ComputeBackendType::GPU_ROCM, 2)};
    for (const int cpu_node : {3, 7})
    {
        const auto rank = makeRankInventory(hardware, {.cpu_numa_node=cpu_node});
        ASSERT_EQ(rank.gpus.size(), 2u);
        EXPECT_EQ(rank.cpu.numa_node, cpu_node);
        EXPECT_EQ(rank.cpu.memory_bytes, cpu_node == 3 ? 1000u : 2000u);
        EXPECT_EQ(rank.gpus[0].uuid, hardware.cuda_devices[0].uuid);
        EXPECT_EQ(rank.gpus[1].uuid, hardware.rocm_devices[0].uuid);
        EXPECT_EQ(rank.gpus[0].numa_node, 7);
        EXPECT_EQ(rank.gpus[1].numa_node, 7);
    }
    // A genuinely excluded backend remains absent. Host-wide does not mean
    // entering a vendor the startup policy deliberately excluded.
    hardware.cuda_devices.clear();
    const auto rank = makeRankInventory(hardware, {.cpu_numa_node=3});
    ASSERT_EQ(rank.gpus.size(), 1u);
    EXPECT_EQ(rank.gpus.front().type, DeviceType::ROCm);
}

TEST_P(InventoryGPUProjection, PreservesSparseOrdinalCapacityAndPCIe)
{
    const std::array visible{gpuFixture(GetParam(), 9), gpuFixture(GetParam(), 2)};
    const auto rank = makeRankInventory(hardwareFixture(), visible, {});
    ASSERT_EQ(rank.gpus.size(), 2u);
    EXPECT_EQ(rank.gpus[0].local_device_id, 9);
    EXPECT_EQ(rank.gpus[1].local_device_id, 2);
    EXPECT_EQ(rank.gpus[0].uuid, visible[0].uuid);
    EXPECT_EQ(rank.gpus[1].uuid, visible[1].uuid);
    for (const auto &gpu : rank.gpus)
    {
        EXPECT_EQ(gpu.memory_bytes, 10000u);
        EXPECT_EQ(gpu.free_memory_bytes, 8123u);
        EXPECT_EQ(gpu.compute_units, 42);
        EXPECT_EQ(gpu.last_level_cache_bytes, 4u << 20);
        EXPECT_EQ(gpu.name, "fixture GPU");
        EXPECT_EQ(gpu.numa_node, 7);
        EXPECT_EQ(gpu.pcie_gen, 3);
        EXPECT_EQ(gpu.pcie_width, 8);
        EXPECT_EQ(gpu.pcie_max_width, 16);
        EXPECT_EQ(gpu.pcie_speed_gts, 8.0);
        EXPECT_EQ(gpu.pcie_max_speed_gts, 16.0);
        EXPECT_TRUE(gpu.pcie_degraded);
        EXPECT_EQ(gpu.pcie_bottleneck_bdf, "0000:81:00.0");
        EXPECT_EQ(gpu.tflops_fp16, 0.0) << "Missing throughput is not a fabricated estimate";
    }
}

TEST_P(InventoryGPUProjection, RejectsInventedCapacityAndDuplicateOrdinals)
{
    auto gpu = gpuFixture(GetParam(), 4);
    std::array duplicate{gpu, gpu};
    EXPECT_THROW(makeRankInventory(hardwareFixture(), duplicate, {}), std::invalid_argument);
    gpu.device_id = -1;
    EXPECT_THROW(makeRankInventory(hardwareFixture(), std::span(&gpu, 1), {}), std::invalid_argument);
    gpu.device_id = 4;
    gpu.total_memory_bytes = 0;
    EXPECT_THROW(makeRankInventory(hardwareFixture(), std::span(&gpu, 1), {}), std::invalid_argument);
    gpu.total_memory_bytes = 8000;
    EXPECT_THROW(makeRankInventory(hardwareFixture(), std::span(&gpu, 1), {}), std::invalid_argument);
}
INSTANTIATE_TEST_SUITE_P(AllBackends, InventoryGPUProjection,
    ::testing::Values(ComputeBackendType::GPU_CUDA, ComputeBackendType::GPU_ROCM,
                      ComputeBackendType::GPU_VULKAN, ComputeBackendType::GPU_METAL));

TEST(Test__ClusterInventoryGatherer, SameOrdinalOnDifferentBackendsRemainsDistinct)
{
    const std::array visible{gpuFixture(ComputeBackendType::GPU_CUDA, 2),
                             gpuFixture(ComputeBackendType::GPU_ROCM, 2)};
    const auto rank = makeRankInventory(hardwareFixture(), visible, {});
    ASSERT_EQ(rank.gpus.size(), 2u);
    EXPECT_EQ(rank.gpus[0].type, DeviceType::CUDA);
    EXPECT_EQ(rank.gpus[1].type, DeviceType::ROCm);
}

TEST(Test__ClusterInventoryGatherer, P2PReordersAndFiltersBothBackendsWithoutLosingDirectedEdges)
{
    for (const auto backend : {ComputeBackendType::GPU_CUDA, ComputeBackendType::GPU_ROCM})
    {
        auto hardware = hardwareFixture();
        P2PMatrix peers;
        peers.backend = backend;
        peers.device_ids = {2, 7, 9};
        peers.can_access = {{true, false, true}, {false, true, false}, {false, false, true}};
        (backend == ComputeBackendType::GPU_CUDA ? hardware.cuda_p2p : hardware.rocm_p2p) = peers;
        const std::array visible{gpuFixture(backend, 9), gpuFixture(backend, 2)};
        const auto rank = makeRankInventory(hardware, visible, {});
        const auto &matrix = backend == ComputeBackendType::GPU_CUDA ? rank.p2p_cuda : rank.p2p_rocm;
        const int count = backend == ComputeBackendType::GPU_CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
        EXPECT_EQ(count, 2);
        EXPECT_EQ(matrix, (std::vector<bool>{true, false, true, true}));
        const auto bytes = MPITopology::serializeRankInventory(rank);
        const auto decoded = MPITopology::deserializeRankInventory(bytes.data(), bytes.size());
        EXPECT_EQ(rank.cpu_execution, hardware.cpu_execution);
        EXPECT_EQ(decoded.cpu_execution, hardware.cpu_execution);
        EXPECT_EQ(decoded.cpu.last_level_cache_bytes, rank.cpu.last_level_cache_bytes);
        EXPECT_EQ(decoded.p2p_cuda, rank.p2p_cuda);
        EXPECT_EQ(decoded.p2p_rocm, rank.p2p_rocm);
        EXPECT_EQ(decoded.gpus[0].free_memory_bytes, rank.gpus[0].free_memory_bytes);
        EXPECT_EQ(decoded.gpus[0].pcie_bottleneck_bdf, rank.gpus[0].pcie_bottleneck_bdf);
        EXPECT_EQ(decoded.gpus[0].last_level_cache_bytes, rank.gpus[0].last_level_cache_bytes);
    }
}

TEST(Test__ClusterInventoryGatherer, IncompleteP2PObservationIsNotFalseEvidence)
{
    auto hardware = hardwareFixture();
    P2PMatrix peers;
    peers.device_ids = {2};
    peers.can_access = {{true}};
    hardware.cuda_p2p = peers;
    const std::array visible{gpuFixture(ComputeBackendType::GPU_CUDA, 9)};
    EXPECT_THROW(makeRankInventory(hardware, visible, {}), std::invalid_argument);
    hardware.cuda_p2p->can_access.clear();
    EXPECT_THROW(makeRankInventory(hardware, {}, {}), std::invalid_argument);
}

TEST(Test__ClusterInventoryGatherer, TopologyMovePreservesInstalledSnapshot)
{
    MPITopology original(1, 2, 2, MPI_COMM_NULL);
    ASSERT_EQ(original.clusterInventory().ranks.size(), 2u);
    EXPECT_EQ(original.clusterInventory().ranks[0].rank, 0);
    EXPECT_EQ(original.clusterInventory().ranks[1].rank, 1);
    MPITopology moved(std::move(original));
    EXPECT_EQ(moved.clusterInventory().world_size, 2);
    EXPECT_EQ(moved.getRankInventory(1).hostname, "explicit");
    MPITopology assigned(0, 1, 1, MPI_COMM_NULL);
    assigned = std::move(moved);
    EXPECT_EQ(assigned.clusterInventory().ranks.size(), 2u);
    EXPECT_EQ(assigned.getRankInventory(1).rank, 1);
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
    rank0.cpu.numa_node = 0;
    rank0.cpu_memory_bytes = 384ULL * 1024 * 1024 * 1024; // 384 GB (socket 0)

    RankInventory rank1;
    rank1.rank = 1;
    rank1.node_id = 0;
    rank1.hostname = "node0";
    rank1.cpu_cores = 28;                                 // 28 cores on socket 1
    rank1.cpu.numa_node = 1;
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
        ranks[i].cpu.numa_node = i % 2;
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
