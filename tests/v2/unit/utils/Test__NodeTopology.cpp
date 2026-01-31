/**
 * @file Test__NodeTopology.cpp
 * @brief Unit tests for NodeTopology class
 *
 * Tests system topology detection including NUMA nodes, CPU sockets,
 * and inter-socket link detection for CPU tensor parallelism.
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include <gtest/gtest.h>
#include "utils/NodeTopology.h"

using namespace llaminar2;

// ============================================================================
// Basic Detection Tests
// ============================================================================

TEST(Test__NodeTopology, DetectsAtLeastOneSocket)
{
    auto topology = NodeTopology::detect();
    EXPECT_GE(topology.numSockets(), 1) << "System should have at least one socket";
}

TEST(Test__NodeTopology, DetectsAtLeastOneNUMANode)
{
    auto topology = NodeTopology::detect();
    EXPECT_GE(topology.numNUMANodes(), 1) << "System should have at least one NUMA node";
}

TEST(Test__NodeTopology, DetectsAtLeastOneCPU)
{
    auto topology = NodeTopology::detect();
    EXPECT_GE(topology.numCPUs(), 1) << "System should have at least one CPU";
}

TEST(Test__NodeTopology, NUMANodesNotEmpty)
{
    auto topology = NodeTopology::detect();
    EXPECT_FALSE(topology.numaNodes().empty()) << "NUMA nodes vector should not be empty";
}

TEST(Test__NodeTopology, EachNUMANodeHasCPUs)
{
    auto topology = NodeTopology::detect();
    for (const auto &node : topology.numaNodes())
    {
        EXPECT_FALSE(node.cpu_ids.empty())
            << "NUMA node " << node.node_id << " should have CPUs";
    }
}

// ============================================================================
// CPU Mapping Tests
// ============================================================================

TEST(Test__NodeTopology, CPUToSocketMapping)
{
    auto topology = NodeTopology::detect();

    // Every CPU should map to a valid socket
    for (const auto &node : topology.numaNodes())
    {
        for (int cpu_id : node.cpu_ids)
        {
            int socket = topology.socketForCPU(cpu_id);
            EXPECT_GE(socket, 0) << "CPU " << cpu_id << " should map to a valid socket";
            EXPECT_LT(socket, topology.numSockets())
                << "CPU " << cpu_id << " socket should be < numSockets";
        }
    }
}

TEST(Test__NodeTopology, CPUToNUMAMapping)
{
    auto topology = NodeTopology::detect();

    // Every CPU should map to a valid NUMA node
    for (const auto &node : topology.numaNodes())
    {
        for (int cpu_id : node.cpu_ids)
        {
            int numa_node = topology.numaNodeForCPU(cpu_id);
            EXPECT_GE(numa_node, 0) << "CPU " << cpu_id << " should map to a valid NUMA node";
            EXPECT_LT(numa_node, topology.numNUMANodes())
                << "CPU " << cpu_id << " NUMA node should be < numNUMANodes";
        }
    }
}

TEST(Test__NodeTopology, InvalidCPUReturnsNegative)
{
    auto topology = NodeTopology::detect();

    // CPU ID that doesn't exist should return -1
    int invalid_cpu = 99999;
    EXPECT_EQ(topology.socketForCPU(invalid_cpu), -1);
    EXPECT_EQ(topology.numaNodeForCPU(invalid_cpu), -1);
}

// ============================================================================
// CreateForTest Tests
// ============================================================================

TEST(Test__NodeTopology, CreateForTestSingleSocket)
{
    auto topology = NodeTopology::createForTest(1, 1, 8); // 1 socket, 1 NUMA node, 8 cores

    EXPECT_EQ(topology.numSockets(), 1);
    EXPECT_EQ(topology.numNUMANodes(), 1);
    EXPECT_EQ(topology.numCPUs(), 8);
    EXPECT_FALSE(topology.hasInterSocketLinks());
    EXPECT_TRUE(topology.interSocketLinks().empty());
}

TEST(Test__NodeTopology, CreateForTestDualSocket)
{
    auto topology = NodeTopology::createForTest(2, 1, 8); // 2 sockets, 1 NUMA node each, 8 cores each

    EXPECT_EQ(topology.numSockets(), 2);
    EXPECT_EQ(topology.numNUMANodes(), 2);
    EXPECT_EQ(topology.numCPUs(), 16);
    EXPECT_TRUE(topology.hasInterSocketLinks());
    EXPECT_EQ(topology.interSocketLinks().size(), 1); // socket0 <-> socket1
}

TEST(Test__NodeTopology, CreateForTestQuadSocketMesh)
{
    auto topology = NodeTopology::createForTest(4, 2, 8); // 4 sockets, 2 NUMA nodes each, 8 cores each

    EXPECT_EQ(topology.numSockets(), 4);
    EXPECT_EQ(topology.numNUMANodes(), 8);
    EXPECT_EQ(topology.numCPUs(), 64);
    EXPECT_TRUE(topology.hasInterSocketLinks());
    // Mesh: 4 sockets = C(4,2) = 6 links
    EXPECT_EQ(topology.interSocketLinks().size(), 6);
}

TEST(Test__NodeTopology, CreateForTestMapsAllCPUs)
{
    auto topology = NodeTopology::createForTest(2, 2, 4); // 2 sockets, 2 NUMA nodes each, 4 cores each

    int total_cpus = topology.numCPUs();
    EXPECT_EQ(total_cpus, 16);

    for (int cpu = 0; cpu < total_cpus; ++cpu)
    {
        EXPECT_GE(topology.socketForCPU(cpu), 0) << "CPU " << cpu << " should have valid socket";
        EXPECT_GE(topology.numaNodeForCPU(cpu), 0) << "CPU " << cpu << " should have valid NUMA node";
    }
}

// ============================================================================
// MPI Rank Grouping Tests
// ============================================================================

TEST(Test__NodeTopology, GroupRanksBySocketSingleSocket)
{
    auto topology = NodeTopology::createForTest(1, 1, 8);

    auto groups = topology.groupRanksBySocket(4);

    EXPECT_EQ(groups.size(), 1);
    EXPECT_EQ(groups[0].size(), 4); // All 4 ranks on socket 0

    // Verify ranks 0-3 are in socket 0
    for (int rank = 0; rank < 4; ++rank)
    {
        EXPECT_NE(std::find(groups[0].begin(), groups[0].end(), rank), groups[0].end());
    }
}

TEST(Test__NodeTopology, GroupRanksBySocketDualSocket)
{
    auto topology = NodeTopology::createForTest(2, 1, 8);

    auto groups = topology.groupRanksBySocket(4);

    EXPECT_EQ(groups.size(), 2);
    EXPECT_EQ(groups[0].size(), 2); // Ranks 0, 2 on socket 0
    EXPECT_EQ(groups[1].size(), 2); // Ranks 1, 3 on socket 1

    // Verify rank distribution (round-robin)
    EXPECT_NE(std::find(groups[0].begin(), groups[0].end(), 0), groups[0].end());
    EXPECT_NE(std::find(groups[0].begin(), groups[0].end(), 2), groups[0].end());
    EXPECT_NE(std::find(groups[1].begin(), groups[1].end(), 1), groups[1].end());
    EXPECT_NE(std::find(groups[1].begin(), groups[1].end(), 3), groups[1].end());
}

TEST(Test__NodeTopology, GroupRanksBySocketUnevenDistribution)
{
    auto topology = NodeTopology::createForTest(2, 1, 8);

    // 5 ranks on 2 sockets = uneven (3 + 2)
    auto groups = topology.groupRanksBySocket(5);

    EXPECT_EQ(groups.size(), 2);
    EXPECT_EQ(groups[0].size(), 3); // Ranks 0, 2, 4 on socket 0
    EXPECT_EQ(groups[1].size(), 2); // Ranks 1, 3 on socket 1
}

// ============================================================================
// SuggestSocketForRank Tests
// ============================================================================

TEST(Test__NodeTopology, SuggestSocketForRank)
{
    auto topology = NodeTopology::createForTest(2, 1, 8);

    // Test round-robin distribution
    EXPECT_EQ(topology.suggestSocketForRank(4, 0), 0); // rank 0 -> socket 0
    EXPECT_EQ(topology.suggestSocketForRank(4, 1), 1); // rank 1 -> socket 1
    EXPECT_EQ(topology.suggestSocketForRank(4, 2), 0); // rank 2 -> socket 0
    EXPECT_EQ(topology.suggestSocketForRank(4, 3), 1); // rank 3 -> socket 1
}

TEST(Test__NodeTopology, SuggestSocketForRankSingleSocket)
{
    auto topology = NodeTopology::createForTest(1, 1, 8);

    // All ranks should go to socket 0
    for (int rank = 0; rank < 8; ++rank)
    {
        EXPECT_EQ(topology.suggestSocketForRank(8, rank), 0);
    }
}

TEST(Test__NodeTopology, SuggestSocketForRankSingleRank)
{
    auto topology = NodeTopology::createForTest(2, 1, 8);

    // Single rank should go to socket 0
    EXPECT_EQ(topology.suggestSocketForRank(1, 0), 0);
}

// ============================================================================
// Inter-Socket Link Tests
// ============================================================================

TEST(Test__NodeTopology, InterSocketBandwidth)
{
    auto topology = NodeTopology::createForTest(2, 1, 8);

    float bandwidth = topology.interSocketBandwidthGBps();
    EXPECT_GT(bandwidth, 0.0f) << "Dual-socket should have positive bandwidth";
    // UPI bandwidth is typically ~50 GB/s, Infinity Fabric ~100 GB/s
    EXPECT_GE(bandwidth, 40.0f);
    EXPECT_LE(bandwidth, 150.0f);
}

TEST(Test__NodeTopology, InterSocketBandwidthSingleSocket)
{
    auto topology = NodeTopology::createForTest(1, 1, 8);

    float bandwidth = topology.interSocketBandwidthGBps();
    EXPECT_EQ(bandwidth, 0.0f) << "Single socket should have zero inter-socket bandwidth";
}

TEST(Test__NodeTopology, InterSocketLinksHaveValidSockets)
{
    auto topology = NodeTopology::createForTest(4, 1, 8);

    for (const auto &link : topology.interSocketLinks())
    {
        EXPECT_GE(link.socket_a, 0);
        EXPECT_LT(link.socket_a, topology.numSockets());
        EXPECT_GE(link.socket_b, 0);
        EXPECT_LT(link.socket_b, topology.numSockets());
        EXPECT_NE(link.socket_a, link.socket_b) << "Link should connect different sockets";
        EXPECT_GT(link.bandwidth_gbps, 0.0f);
        EXPECT_GT(link.latency_ns, 0.0f);
    }
}

// ============================================================================
// NUMA Nodes For Socket Tests
// ============================================================================

TEST(Test__NodeTopology, NUMANodesForSocket)
{
    auto topology = NodeTopology::createForTest(2, 2, 4); // 2 sockets, 2 NUMA nodes each

    auto nodes_socket0 = topology.numaNodesForSocket(0);
    auto nodes_socket1 = topology.numaNodesForSocket(1);

    EXPECT_EQ(nodes_socket0.size(), 2);
    EXPECT_EQ(nodes_socket1.size(), 2);

    // Verify all returned nodes belong to the correct socket
    for (const auto *node : nodes_socket0)
    {
        EXPECT_EQ(node->socket_id, 0);
    }
    for (const auto *node : nodes_socket1)
    {
        EXPECT_EQ(node->socket_id, 1);
    }
}

TEST(Test__NodeTopology, NUMANodesForInvalidSocket)
{
    auto topology = NodeTopology::createForTest(2, 1, 8);

    auto nodes = topology.numaNodesForSocket(99); // Invalid socket
    EXPECT_TRUE(nodes.empty());
}

// ============================================================================
// ToString Tests
// ============================================================================

TEST(Test__NodeTopology, ToStringNotEmpty)
{
    auto topology = NodeTopology::detect();
    std::string str = topology.toString();

    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("NodeTopology"), std::string::npos);
    EXPECT_NE(str.find("sockets"), std::string::npos);
    EXPECT_NE(str.find("numa_nodes"), std::string::npos);
}

TEST(Test__NodeTopology, ToStringContainsCorrectCounts)
{
    auto topology = NodeTopology::createForTest(2, 2, 4);
    std::string str = topology.toString();

    // Should contain "sockets: 2"
    EXPECT_NE(str.find("2"), std::string::npos);
    // Should contain socket/NUMA info
    EXPECT_NE(str.find("NUMA"), std::string::npos);
}

TEST(Test__NodeTopology, ToStringShowsInterSocketLinks)
{
    auto topology = NodeTopology::createForTest(2, 1, 8);
    std::string str = topology.toString();

    // Should mention inter-socket links
    EXPECT_NE(str.find("inter_socket_links"), std::string::npos);
    EXPECT_NE(str.find("socket0"), std::string::npos);
    EXPECT_NE(str.find("socket1"), std::string::npos);
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST(Test__NodeTopology, TotalCPUsMatchNUMANodesCPUs)
{
    auto topology = NodeTopology::detect();

    int sum_cpus = 0;
    for (const auto &node : topology.numaNodes())
    {
        sum_cpus += static_cast<int>(node.cpu_ids.size());
    }

    EXPECT_EQ(sum_cpus, topology.numCPUs())
        << "Total CPUs should equal sum of CPUs across all NUMA nodes";
}

TEST(Test__NodeTopology, AllNUMANodesHaveValidSocketId)
{
    auto topology = NodeTopology::detect();

    for (const auto &node : topology.numaNodes())
    {
        EXPECT_GE(node.socket_id, 0);
        EXPECT_LT(node.socket_id, topology.numSockets())
            << "NUMA node " << node.node_id << " has invalid socket_id " << node.socket_id;
    }
}

TEST(Test__NodeTopology, DetectionIsIdempotent)
{
    // Calling detect() multiple times should give same results
    auto topology1 = NodeTopology::detect();
    auto topology2 = NodeTopology::detect();

    EXPECT_EQ(topology1.numSockets(), topology2.numSockets());
    EXPECT_EQ(topology1.numNUMANodes(), topology2.numNUMANodes());
    EXPECT_EQ(topology1.numCPUs(), topology2.numCPUs());
}
