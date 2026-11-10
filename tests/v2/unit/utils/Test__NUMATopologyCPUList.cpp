/**
 * @file Test__NUMATopologyCPUList.cpp
 * @brief Exact cpulist membership for procfs NUMA node detection.
 *
 * The former procfs path matched node cpulists by substring and only probed
 * nodes 0-7: CPU 1 was falsely found in "10-19", a range such as "8-9" was
 * not recognised as containing CPU 9 by the number scan, and nodes 8+ were
 * never consulted. Membership is now parsed exactly over every node.
 */
#include "utils/NUMATopology.h"

#include <gtest/gtest.h>
#include <type_traits>

using llaminar2::NUMATopology;

/** @test Lists parse exactly; malformed and reversed tokens are rejected. */
TEST(NUMATopologyCPUList, ParsesRangesAndSinglesExactly)
{
    EXPECT_EQ(NUMATopology::parseCPUListExact(" 0-3,8,10-11\n"),
              (std::vector<int>{0, 1, 2, 3, 8, 10, 11}));
    EXPECT_TRUE(NUMATopology::parseCPUListExact("").empty());
    for (const char *bad : {"1-", "-3", "a", "3-1", "1,,2", "1 2"})
        EXPECT_THROW((void)NUMATopology::parseCPUListExact(bad), std::invalid_argument) << bad;
}

/** @test Substring-sharing ranges do not claim a CPU in either direction. */
TEST(NUMATopologyCPUList, MembershipIsNotASubstringMatch)
{
    const std::vector<std::pair<int, std::string>> nodes{{0, "10-19"}, {1, "0-9"}};
    EXPECT_EQ(NUMATopology::nodeOwningCPU(1, nodes), 1);  // not node 0 via "10"
    EXPECT_EQ(NUMATopology::nodeOwningCPU(9, nodes), 1);  // "0-9" contains 9
    EXPECT_EQ(NUMATopology::nodeOwningCPU(15, nodes), 0);
    EXPECT_EQ(NUMATopology::nodeOwningCPU(20, nodes), -1);
}

/** @test Sparse and high node ids are consulted; duplicate owners are rejected. */
TEST(NUMATopologyCPUList, ConsultsEveryNodeAndRejectsDuplicates)
{
    const std::vector<std::pair<int, std::string>> grace_like{
        {0, "0-71"}, {1, "72-143"}, {2, ""}, {9, ""}, {12, "144-215"}};
    EXPECT_EQ(NUMATopology::nodeOwningCPU(150, grace_like), 12);
    EXPECT_EQ(NUMATopology::nodeOwningCPU(80, grace_like), 1);
    EXPECT_THROW((void)NUMATopology::nodeOwningCPU(3, {{0, "0-7"}, {1, "3"}}), std::invalid_argument);
    EXPECT_THROW((void)NUMATopology::nodeOwningCPU(3, {{0, "0-x"}}), std::invalid_argument);
}

#include "utils/NodeTopology.h"
#include "backends/DeviceRegistry.h"

#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace
{
    /** @brief Write one sysfs-shaped node directory under a temporary root. */
    void writeNode(const std::filesystem::path &root, const std::string &name,
                   const char *cpulist, const char *meminfo_kb)
    {
        std::filesystem::create_directories(root / name);
        if (cpulist)
            std::ofstream(root / name / "cpulist") << cpulist << "\n";
        if (meminfo_kb)
            std::ofstream(root / name / "meminfo")
                << "Node " << name.substr(4) << " MemTotal:       " << meminfo_kb << " kB\n";
    }
}

/** @test Grace-shaped sysfs: CPU-less memory nodes are not CPU endpoints. */
TEST(NUMATopologyCPUList, CPULessMemoryNodesAreNotCPUEndpoints)
{
    const auto root = std::filesystem::temp_directory_path() /
        ("llaminar_grace_nodes_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    // Directory creation order deliberately differs from node order.
    writeNode(root, "node12", "", "4194304");     // GPU/CXL memory, no CPUs
    writeNode(root, "node1", "72-143", "16777216");
    writeNode(root, "node4", nullptr, "8388608"); // memory-only, no cpulist
    writeNode(root, "node0", "0-71", "16777216");
    std::filesystem::create_directories(root / "nodeX");
    std::filesystem::create_directories(root / "possible");

    const auto nodes = llaminar2::NodeTopology::detectNUMANodesAt(root.string());
    std::filesystem::remove_all(root);
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_EQ(nodes[0].node_id, 0);
    EXPECT_EQ(nodes[1].node_id, 1);
    EXPECT_EQ(nodes[0].cpu_ids.size(), 72u);
    EXPECT_EQ(nodes[0].memory_bytes, 16777216ull * 1024ull);
}

/** @test The device registry's CPU endpoints are exactly NodeTopology's nodes. */
TEST(NUMATopologyCPUList, RegistryCPUEndpointsFollowNodeTopology)
{
    if (!std::filesystem::exists("/sys/devices/system/node"))
        GTEST_SKIP() << "Host exposes no NUMA sysfs";
    auto &registry = llaminar2::DeviceRegistry::instance();
    registry.discover();
    std::vector<int> registered;
    for (const auto &device : registry.devicesByType(llaminar2::DeviceType::CPU))
        registered.push_back(device.numa_node);
    std::vector<int> expected;
    const auto topology = llaminar2::NodeTopology::detect();
    for (const auto &node : topology.numaNodes())
        expected.push_back(node.node_id);
    EXPECT_EQ(registered, expected);
}

/** @test Failed detection never publishes a usable node; success always does. */
TEST(NUMATopologyCPUList, UndetectedAffinityIsNeverNodeZero)
{
    EXPECT_EQ(llaminar2::NUMAInfo{}.local_numa_node, -1);
    const auto info = NUMATopology::detectLocalNUMANode();
    if (info.detection_succeeded)
        EXPECT_GE(info.local_numa_node, 0);
    else
    {
        EXPECT_EQ(info.local_numa_node, -1);
        EXPECT_EQ(info.detection_method, "unknown");
    }
}

/** @test A temporary topology yields an owned node list, never a dangling one. */
TEST(NUMATopologyCPUList, TemporaryTopologyNodeListIsOwned)
{
    static_assert(std::is_same_v<decltype(llaminar2::NodeTopology::detect().numaNodes()),
                                 std::vector<llaminar2::NUMANode>>,
                  "numaNodes() on a temporary must return by value");
    const auto topology = llaminar2::NodeTopology::createForTest(2, 2, 4);
    static_assert(std::is_same_v<decltype(topology.numaNodes()),
                                 const std::vector<llaminar2::NUMANode> &>);
    std::vector<int> ids;
    for (const auto &node : llaminar2::NodeTopology::createForTest(2, 2, 4).numaNodes())
        ids.push_back(node.node_id);
    std::vector<int> expected;
    for (const auto &node : topology.numaNodes())
        expected.push_back(node.node_id);
    EXPECT_EQ(ids, expected);
}
