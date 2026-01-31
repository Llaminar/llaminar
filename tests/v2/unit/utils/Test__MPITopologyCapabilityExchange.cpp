/**
 * @file Test__MPITopologyCapabilityExchange.cpp
 * @brief Unit tests for MPITopology capability exchange serialization
 *
 * Tests the serialization/deserialization of RankInventory structures
 * and the heterogeneous GPU detection logic.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "utils/MPITopology.h"
#include "execution/mpi_orchestration/DeviceInventory.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__MPITopologyCapabilityExchange : public ::testing::Test
{
protected:
    /**
     * @brief Create a test RankInventory with specified configuration
     */
    RankInventory createTestInventory(
        int rank,
        const std::string &hostname,
        int num_cuda_gpus,
        int num_rocm_gpus)
    {
        RankInventory inv;
        inv.rank = rank;
        inv.node_id = rank / 2; // 2 ranks per node
        inv.local_rank = rank % 2;
        inv.hostname = hostname;
        inv.cpu_cores = 32;
        inv.cpu_sockets = 2;
        inv.numa_nodes = 2;
        inv.cpu_memory_bytes = 128ULL * 1024 * 1024 * 1024; // 128GB

        // CPU device info
        inv.cpu.type = DeviceType::CPU;
        inv.cpu.local_device_id = 0;
        inv.cpu.memory_bytes = inv.cpu_memory_bytes;
        inv.cpu.compute_units = inv.cpu_cores;
        inv.cpu.memory_bandwidth_gbps = 200.0f;
        inv.cpu.name = "AMD EPYC 7763";
        inv.cpu.tflops_fp16 = 1.0f; // Baseline compute weight

        // Add CUDA GPUs
        for (int i = 0; i < num_cuda_gpus; ++i)
        {
            DeviceInfo gpu;
            gpu.type = DeviceType::CUDA;
            gpu.local_device_id = i;
            gpu.memory_bytes = 80ULL * 1024 * 1024 * 1024; // 80GB
            gpu.free_memory_bytes = 75ULL * 1024 * 1024 * 1024;
            gpu.compute_units = 108; // A100 SM count
            gpu.compute_capability_major = 8;
            gpu.compute_capability_minor = 0;
            gpu.tflops_fp16 = 312.0f;
            gpu.tflops_int8 = 624.0f;
            gpu.memory_bandwidth_gbps = 2039.0f;
            gpu.name = "NVIDIA A100-SXM4-80GB";
            gpu.uuid = "GPU-" + std::to_string(rank) + "-cuda-" + std::to_string(i);
            gpu.supports_p2p = true;
            gpu.pcie_bus_id = 0x41 + i;
            gpu.numa_node = i % 2;
            inv.gpus.push_back(gpu);
        }

        // Add ROCm GPUs
        for (int i = 0; i < num_rocm_gpus; ++i)
        {
            DeviceInfo gpu;
            gpu.type = DeviceType::ROCm;
            gpu.local_device_id = num_cuda_gpus + i;
            gpu.memory_bytes = 64ULL * 1024 * 1024 * 1024; // 64GB
            gpu.free_memory_bytes = 60ULL * 1024 * 1024 * 1024;
            gpu.compute_units = 120; // MI250X CU count per die
            gpu.compute_capability_major = 9;
            gpu.compute_capability_minor = 0;
            gpu.tflops_fp16 = 383.0f;
            gpu.tflops_int8 = 383.0f;
            gpu.memory_bandwidth_gbps = 3276.0f;
            gpu.name = "AMD Instinct MI250X";
            gpu.uuid = "GPU-" + std::to_string(rank) + "-rocm-" + std::to_string(i);
            gpu.supports_p2p = true;
            gpu.pcie_bus_id = 0x81 + i;
            gpu.numa_node = i % 2;
            inv.gpus.push_back(gpu);
        }

        return inv;
    }

    /**
     * @brief Compare two DeviceInfo structs for equality
     */
    void expectDeviceInfoEqual(const DeviceInfo &a, const DeviceInfo &b, const std::string &context)
    {
        EXPECT_EQ(static_cast<int>(a.type), static_cast<int>(b.type)) << context << " type mismatch";
        EXPECT_EQ(a.local_device_id, b.local_device_id) << context << " device_id mismatch";
        EXPECT_EQ(a.memory_bytes, b.memory_bytes) << context << " memory_bytes mismatch";
        EXPECT_EQ(a.free_memory_bytes, b.free_memory_bytes) << context << " free_memory_bytes mismatch";
        EXPECT_EQ(a.compute_units, b.compute_units) << context << " compute_units mismatch";
        EXPECT_EQ(a.compute_capability_major, b.compute_capability_major) << context << " cc_major mismatch";
        EXPECT_EQ(a.compute_capability_minor, b.compute_capability_minor) << context << " cc_minor mismatch";
        EXPECT_FLOAT_EQ(a.tflops_fp16, b.tflops_fp16) << context << " tflops_fp16 mismatch";
        EXPECT_FLOAT_EQ(a.tflops_int8, b.tflops_int8) << context << " tflops_int8 mismatch";
        EXPECT_FLOAT_EQ(a.memory_bandwidth_gbps, b.memory_bandwidth_gbps) << context << " bandwidth mismatch";
        EXPECT_EQ(a.name, b.name) << context << " name mismatch";
        EXPECT_EQ(a.uuid, b.uuid) << context << " uuid mismatch";
        EXPECT_EQ(a.supports_p2p, b.supports_p2p) << context << " supports_p2p mismatch";
        EXPECT_EQ(a.pcie_bus_id, b.pcie_bus_id) << context << " pcie_bus_id mismatch";
        EXPECT_EQ(a.numa_node, b.numa_node) << context << " numa_node mismatch";
    }

    /**
     * @brief Compare two RankInventory structs for equality
     */
    void expectInventoryEqual(const RankInventory &a, const RankInventory &b)
    {
        EXPECT_EQ(a.rank, b.rank) << "rank mismatch";
        EXPECT_EQ(a.node_id, b.node_id) << "node_id mismatch";
        EXPECT_EQ(a.local_rank, b.local_rank) << "local_rank mismatch";
        EXPECT_EQ(a.hostname, b.hostname) << "hostname mismatch";
        EXPECT_EQ(a.cpu_cores, b.cpu_cores) << "cpu_cores mismatch";
        EXPECT_EQ(a.cpu_sockets, b.cpu_sockets) << "cpu_sockets mismatch";
        EXPECT_EQ(a.numa_nodes, b.numa_nodes) << "numa_nodes mismatch";
        EXPECT_EQ(a.cpu_memory_bytes, b.cpu_memory_bytes) << "cpu_memory_bytes mismatch";

        expectDeviceInfoEqual(a.cpu, b.cpu, "CPU");

        ASSERT_EQ(a.gpus.size(), b.gpus.size()) << "GPU count mismatch";
        for (size_t i = 0; i < a.gpus.size(); ++i)
        {
            expectDeviceInfoEqual(a.gpus[i], b.gpus[i], "GPU[" + std::to_string(i) + "]");
        }
    }
};

// =============================================================================
// Serialization/Deserialization Tests
// =============================================================================

TEST_F(Test__MPITopologyCapabilityExchange, SerializeDeserialize_CPUOnly)
{
    // Test with CPU-only inventory (no GPUs)
    RankInventory original = createTestInventory(0, "cpu-only-node", 0, 0);

    // Serialize
    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);
    EXPECT_GT(data.size(), 0) << "Serialized data should not be empty";

    // Deserialize
    RankInventory restored = MPITopology::deserializeRankInventory(data.data(), data.size());

    // Verify
    expectInventoryEqual(original, restored);
}

TEST_F(Test__MPITopologyCapabilityExchange, SerializeDeserialize_SingleCUDAGPU)
{
    // Test with single CUDA GPU
    RankInventory original = createTestInventory(0, "cuda-node-0", 1, 0);

    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);
    RankInventory restored = MPITopology::deserializeRankInventory(data.data(), data.size());

    expectInventoryEqual(original, restored);
    EXPECT_EQ(restored.gpus.size(), 1);
    EXPECT_EQ(restored.gpus[0].type, DeviceType::CUDA);
}

TEST_F(Test__MPITopologyCapabilityExchange, SerializeDeserialize_SingleROCmGPU)
{
    // Test with single ROCm GPU
    RankInventory original = createTestInventory(1, "rocm-node-0", 0, 1);

    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);
    RankInventory restored = MPITopology::deserializeRankInventory(data.data(), data.size());

    expectInventoryEqual(original, restored);
    EXPECT_EQ(restored.gpus.size(), 1);
    EXPECT_EQ(restored.gpus[0].type, DeviceType::ROCm);
}

TEST_F(Test__MPITopologyCapabilityExchange, SerializeDeserialize_MultipleCUDAGPUs)
{
    // Test with multiple CUDA GPUs (typical multi-GPU node)
    RankInventory original = createTestInventory(0, "multi-cuda-node", 4, 0);

    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);
    RankInventory restored = MPITopology::deserializeRankInventory(data.data(), data.size());

    expectInventoryEqual(original, restored);
    EXPECT_EQ(restored.gpus.size(), 4);
    for (const auto &gpu : restored.gpus)
    {
        EXPECT_EQ(gpu.type, DeviceType::CUDA);
    }
}

TEST_F(Test__MPITopologyCapabilityExchange, SerializeDeserialize_MixedGPUs)
{
    // Test with mixed CUDA and ROCm GPUs (heterogeneous node)
    RankInventory original = createTestInventory(0, "heterogeneous-node", 2, 2);

    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);
    RankInventory restored = MPITopology::deserializeRankInventory(data.data(), data.size());

    expectInventoryEqual(original, restored);
    EXPECT_EQ(restored.gpus.size(), 4);

    // Verify GPU types are preserved
    int cuda_count = 0;
    int rocm_count = 0;
    for (const auto &gpu : restored.gpus)
    {
        if (gpu.type == DeviceType::CUDA)
            cuda_count++;
        else if (gpu.type == DeviceType::ROCm)
            rocm_count++;
    }
    EXPECT_EQ(cuda_count, 2);
    EXPECT_EQ(rocm_count, 2);
}

TEST_F(Test__MPITopologyCapabilityExchange, SerializeDeserialize_LongHostname)
{
    // Test with a very long hostname (edge case)
    std::string long_hostname = "very-long-hostname-that-might-cause-issues-in-serialization-" +
                                std::string(100, 'x') + ".example.com";
    RankInventory original = createTestInventory(0, long_hostname, 1, 0);

    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);
    RankInventory restored = MPITopology::deserializeRankInventory(data.data(), data.size());

    EXPECT_EQ(restored.hostname, long_hostname);
}

TEST_F(Test__MPITopologyCapabilityExchange, SerializeDeserialize_EmptyStrings)
{
    // Test with empty name/uuid strings
    RankInventory original;
    original.rank = 0;
    original.node_id = 0;
    original.local_rank = 0;
    original.hostname = ""; // Empty hostname
    original.cpu_cores = 1;
    original.cpu_sockets = 1;
    original.numa_nodes = 1;
    original.cpu_memory_bytes = 1024;

    original.cpu.type = DeviceType::CPU;
    original.cpu.name = ""; // Empty name
    original.cpu.uuid = ""; // Empty UUID

    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);
    RankInventory restored = MPITopology::deserializeRankInventory(data.data(), data.size());

    EXPECT_EQ(restored.hostname, "");
    EXPECT_EQ(restored.cpu.name, "");
    EXPECT_EQ(restored.cpu.uuid, "");
}

TEST_F(Test__MPITopologyCapabilityExchange, SerializeDeserialize_LargeMemoryValues)
{
    // Test with very large memory values (multi-terabyte systems)
    RankInventory original = createTestInventory(0, "large-memory-node", 8, 0);
    original.cpu_memory_bytes = 2ULL * 1024 * 1024 * 1024 * 1024; // 2TB

    for (auto &gpu : original.gpus)
    {
        gpu.memory_bytes = 192ULL * 1024 * 1024 * 1024; // 192GB HBM3
    }

    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);
    RankInventory restored = MPITopology::deserializeRankInventory(data.data(), data.size());

    EXPECT_EQ(restored.cpu_memory_bytes, 2ULL * 1024 * 1024 * 1024 * 1024);
    for (const auto &gpu : restored.gpus)
    {
        EXPECT_EQ(gpu.memory_bytes, 192ULL * 1024 * 1024 * 1024);
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(Test__MPITopologyCapabilityExchange, Deserialize_TruncatedData)
{
    RankInventory original = createTestInventory(0, "test-node", 1, 0);
    std::vector<uint8_t> data = MPITopology::serializeRankInventory(original);

    // Truncate the data
    data.resize(data.size() / 2);

    // Should throw on truncated data
    EXPECT_THROW(
        MPITopology::deserializeRankInventory(data.data(), data.size()),
        std::runtime_error);
}

TEST_F(Test__MPITopologyCapabilityExchange, Deserialize_EmptyData)
{
    std::vector<uint8_t> empty_data;

    // Should throw on empty data
    EXPECT_THROW(
        MPITopology::deserializeRankInventory(empty_data.data(), 0),
        std::runtime_error);
}

// =============================================================================
// Heterogeneous Detection Tests
// =============================================================================

TEST_F(Test__MPITopologyCapabilityExchange, ClusterInventory_HomogeneousCUDA)
{
    // Build a cluster inventory with only CUDA GPUs
    ClusterInventory inventory;
    inventory.world_size = 2;
    inventory.node_count = 1;
    inventory.ranks.resize(2);

    inventory.ranks[0] = createTestInventory(0, "node-0", 2, 0);
    inventory.ranks[1] = createTestInventory(1, "node-0", 2, 0);
    inventory.buildNodeAggregations();

    // Check that we can detect it's not heterogeneous
    bool has_cuda = false;
    bool has_rocm = false;
    for (const auto &rank : inventory.ranks)
    {
        for (const auto &gpu : rank.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
                has_cuda = true;
            if (gpu.type == DeviceType::ROCm)
                has_rocm = true;
        }
    }

    EXPECT_TRUE(has_cuda);
    EXPECT_FALSE(has_rocm);
    EXPECT_FALSE(has_cuda && has_rocm); // Not heterogeneous
}

TEST_F(Test__MPITopologyCapabilityExchange, ClusterInventory_HomogeneousROCm)
{
    // Build a cluster inventory with only ROCm GPUs
    ClusterInventory inventory;
    inventory.world_size = 2;
    inventory.node_count = 1;
    inventory.ranks.resize(2);

    inventory.ranks[0] = createTestInventory(0, "node-0", 0, 2);
    inventory.ranks[1] = createTestInventory(1, "node-0", 0, 2);
    inventory.buildNodeAggregations();

    bool has_cuda = false;
    bool has_rocm = false;
    for (const auto &rank : inventory.ranks)
    {
        for (const auto &gpu : rank.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
                has_cuda = true;
            if (gpu.type == DeviceType::ROCm)
                has_rocm = true;
        }
    }

    EXPECT_FALSE(has_cuda);
    EXPECT_TRUE(has_rocm);
    EXPECT_FALSE(has_cuda && has_rocm); // Not heterogeneous
}

TEST_F(Test__MPITopologyCapabilityExchange, ClusterInventory_Heterogeneous)
{
    // Build a cluster inventory with both CUDA and ROCm GPUs
    ClusterInventory inventory;
    inventory.world_size = 2;
    inventory.node_count = 2;
    inventory.ranks.resize(2);

    inventory.ranks[0] = createTestInventory(0, "cuda-node", 2, 0); // CUDA node
    inventory.ranks[1] = createTestInventory(1, "rocm-node", 0, 2); // ROCm node
    inventory.buildNodeAggregations();

    bool has_cuda = false;
    bool has_rocm = false;
    for (const auto &rank : inventory.ranks)
    {
        for (const auto &gpu : rank.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
                has_cuda = true;
            if (gpu.type == DeviceType::ROCm)
                has_rocm = true;
        }
    }

    EXPECT_TRUE(has_cuda);
    EXPECT_TRUE(has_rocm);
    EXPECT_TRUE(has_cuda && has_rocm); // IS heterogeneous
}

TEST_F(Test__MPITopologyCapabilityExchange, ClusterInventory_CPUOnly)
{
    // Build a cluster inventory with no GPUs
    ClusterInventory inventory;
    inventory.world_size = 2;
    inventory.node_count = 1;
    inventory.ranks.resize(2);

    inventory.ranks[0] = createTestInventory(0, "cpu-node", 0, 0);
    inventory.ranks[1] = createTestInventory(1, "cpu-node", 0, 0);
    inventory.buildNodeAggregations();

    bool has_cuda = false;
    bool has_rocm = false;
    for (const auto &rank : inventory.ranks)
    {
        for (const auto &gpu : rank.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
                has_cuda = true;
            if (gpu.type == DeviceType::ROCm)
                has_rocm = true;
        }
    }

    EXPECT_FALSE(has_cuda);
    EXPECT_FALSE(has_rocm);
    EXPECT_FALSE(has_cuda && has_rocm); // Not heterogeneous (no GPUs at all)
}

// =============================================================================
// Node Aggregation Tests
// =============================================================================

TEST_F(Test__MPITopologyCapabilityExchange, NodeAggregation_TotalGPUs)
{
    ClusterInventory inventory;
    inventory.world_size = 4;
    inventory.node_count = 2;
    inventory.ranks.resize(4);

    // Node 0: 2 ranks, 4 GPUs each = 8 GPUs
    inventory.ranks[0] = createTestInventory(0, "node-0", 4, 0);
    inventory.ranks[0].node_id = 0;
    inventory.ranks[1] = createTestInventory(1, "node-0", 4, 0);
    inventory.ranks[1].node_id = 0;

    // Node 1: 2 ranks, 2 GPUs each = 4 GPUs
    inventory.ranks[2] = createTestInventory(2, "node-1", 2, 0);
    inventory.ranks[2].node_id = 1;
    inventory.ranks[3] = createTestInventory(3, "node-1", 2, 0);
    inventory.ranks[3].node_id = 1;

    inventory.buildNodeAggregations();

    EXPECT_EQ(inventory.nodes.size(), 2);
    EXPECT_EQ(inventory.nodes[0].total_gpus, 8);
    EXPECT_EQ(inventory.nodes[1].total_gpus, 4);
    EXPECT_EQ(inventory.total_gpus, 12);
}

TEST_F(Test__MPITopologyCapabilityExchange, NodeAggregation_TotalMemory)
{
    ClusterInventory inventory;
    inventory.world_size = 2;
    inventory.node_count = 1;
    inventory.ranks.resize(2);

    inventory.ranks[0] = createTestInventory(0, "node-0", 2, 0);
    inventory.ranks[0].node_id = 0;
    inventory.ranks[1] = createTestInventory(1, "node-0", 2, 0);
    inventory.ranks[1].node_id = 0;

    inventory.buildNodeAggregations();

    // Each rank has 2 GPUs x 80GB = 160GB
    // Total for node = 320GB
    size_t expected_gpu_memory = 4 * 80ULL * 1024 * 1024 * 1024;
    EXPECT_EQ(inventory.nodes[0].total_gpu_memory, expected_gpu_memory);
    EXPECT_EQ(inventory.total_gpu_memory, expected_gpu_memory);
}

// =============================================================================
// Compute Weight Tests
// =============================================================================

TEST_F(Test__MPITopologyCapabilityExchange, ComputeWeight_CPUOnly)
{
    RankInventory inv = createTestInventory(0, "cpu-node", 0, 0);

    // CPU baseline weight should be tflops_int8
    float weight = inv.totalComputeWeight();
    EXPECT_GT(weight, 0.0f);
}

TEST_F(Test__MPITopologyCapabilityExchange, ComputeWeight_WithGPUs)
{
    RankInventory inv = createTestInventory(0, "gpu-node", 2, 0);

    float weight = inv.totalComputeWeight();

    // Weight should include CPU + 2 GPUs
    // Each A100 has tflops_int8 = 624
    float cpu_weight = inv.cpu.computeWeight();
    float expected_min = cpu_weight + 2 * 600.0f; // Rough lower bound

    EXPECT_GT(weight, expected_min);
}

// =============================================================================
// Serialization Size Tests
// =============================================================================

TEST_F(Test__MPITopologyCapabilityExchange, SerializationSize_Reasonable)
{
    // Ensure serialized size is reasonable (not exploding)
    RankInventory inv = createTestInventory(0, "node-0", 8, 0);

    std::vector<uint8_t> data = MPITopology::serializeRankInventory(inv);

    // Should be less than 4KB for 8 GPUs
    EXPECT_LT(data.size(), 4096);

    // But should be at least a few hundred bytes
    EXPECT_GT(data.size(), 200);
}

TEST_F(Test__MPITopologyCapabilityExchange, SerializationSize_ScalesWithGPUs)
{
    RankInventory inv1 = createTestInventory(0, "node-0", 1, 0);
    RankInventory inv8 = createTestInventory(0, "node-0", 8, 0);

    std::vector<uint8_t> data1 = MPITopology::serializeRankInventory(inv1);
    std::vector<uint8_t> data8 = MPITopology::serializeRankInventory(inv8);

    // 8 GPUs should use more space than 1 GPU
    EXPECT_GT(data8.size(), data1.size());

    // But not 8x more (due to fixed overhead)
    EXPECT_LT(data8.size(), data1.size() * 8);
}
