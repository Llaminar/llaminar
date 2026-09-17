/**
 * @file Test__NodeDetection.cpp
 * @brief Device-free physical membership validation and synthetic label fixtures.
 *
 * Real MPI membership is exercised by ClusterInventoryBootstrap. These tests
 * reject malformed shared-node leaders without opening MPI or device runtimes;
 * hostname grouping remains only a convenience for explicit synthetic fixtures.
 */

#include <gtest/gtest.h>
#include "utils/NodeDetection.h"
#include <stdexcept>

using namespace llaminar2;

// =============================================================================
// fromHostnames() tests — pure logic, no MPI needed
// =============================================================================

TEST(Test__NodeDetection, FromHostnames_Empty)
{
    auto result = NodeDetection::fromHostnames({});
    EXPECT_EQ(result.node_count, 0);
    EXPECT_TRUE(result.node_ids.empty());
    EXPECT_TRUE(result.hostnames.empty());
}

TEST(Test__NodeDetection, FromHostnames_SingleHost)
{
    auto result = NodeDetection::fromHostnames({"node-a"});
    EXPECT_EQ(result.node_count, 1);
    ASSERT_EQ(result.node_ids.size(), 1u);
    EXPECT_EQ(result.node_ids[0], 0);
    EXPECT_EQ(result.hostnames[0], "node-a");
}

TEST(Test__NodeDetection, FromHostnames_AllSameHost)
{
    auto result = NodeDetection::fromHostnames({"host1", "host1", "host1", "host1"});
    EXPECT_EQ(result.node_count, 1);
    ASSERT_EQ(result.node_ids.size(), 4u);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(result.node_ids[i], 0) << "rank " << i;
    }
}

TEST(Test__NodeDetection, FromHostnames_TwoHosts_Contiguous)
{
    // Ranks 0,1 on node-a, ranks 2,3 on node-b
    auto result = NodeDetection::fromHostnames({"node-a", "node-a", "node-b", "node-b"});
    EXPECT_EQ(result.node_count, 2);
    ASSERT_EQ(result.node_ids.size(), 4u);
    EXPECT_EQ(result.node_ids[0], 0);
    EXPECT_EQ(result.node_ids[1], 0);
    EXPECT_EQ(result.node_ids[2], 1);
    EXPECT_EQ(result.node_ids[3], 1);
}

TEST(Test__NodeDetection, FromHostnames_TwoHosts_Interleaved)
{
    // Non-contiguous rank assignment: ranks alternate between nodes
    auto result = NodeDetection::fromHostnames({"node-a", "node-b", "node-a", "node-b"});
    EXPECT_EQ(result.node_count, 2);
    ASSERT_EQ(result.node_ids.size(), 4u);
    // node-a first seen → id 0, node-b second seen → id 1
    EXPECT_EQ(result.node_ids[0], 0);
    EXPECT_EQ(result.node_ids[1], 1);
    EXPECT_EQ(result.node_ids[2], 0);
    EXPECT_EQ(result.node_ids[3], 1);
}

TEST(Test__NodeDetection, FromHostnames_ThreeHosts)
{
    auto result = NodeDetection::fromHostnames(
        {"alpha", "beta", "gamma", "alpha", "beta", "gamma"});
    EXPECT_EQ(result.node_count, 3);
    ASSERT_EQ(result.node_ids.size(), 6u);
    // Sequential IDs by first appearance
    EXPECT_EQ(result.node_ids[0], 0); // alpha
    EXPECT_EQ(result.node_ids[1], 1); // beta
    EXPECT_EQ(result.node_ids[2], 2); // gamma
    EXPECT_EQ(result.node_ids[3], 0); // alpha again
    EXPECT_EQ(result.node_ids[4], 1); // beta again
    EXPECT_EQ(result.node_ids[5], 2); // gamma again
}

TEST(Test__NodeDetection, FromHostnames_FirstAppearanceOrder)
{
    // Verify IDs are assigned in order of first appearance, not alphabetically
    auto result = NodeDetection::fromHostnames({"zebra", "alpha", "mango"});
    EXPECT_EQ(result.node_count, 3);
    EXPECT_EQ(result.node_ids[0], 0); // zebra first
    EXPECT_EQ(result.node_ids[1], 1); // alpha second
    EXPECT_EQ(result.node_ids[2], 2); // mango third
}

TEST(Test__NodeDetection, FromHostnames_HostnamesPreserved)
{
    std::vector<std::string> input = {"host-x", "host-y", "host-x"};
    auto result = NodeDetection::fromHostnames(input);
    ASSERT_EQ(result.hostnames.size(), 3u);
    EXPECT_EQ(result.hostnames[0], "host-x");
    EXPECT_EQ(result.hostnames[1], "host-y");
    EXPECT_EQ(result.hostnames[2], "host-x");
}

/** Shared membership, not labels, is the physical topology authority. */
TEST(Test__NodeDetection, SharedLeadersHandleInterleavedNodes)
{
    const auto result = NodeDetection::fromSharedNodeLeaders({0, 1, 0, 1, 4, 4});
    EXPECT_EQ(result.node_ids, (std::vector<int>{0, 1, 0, 1, 2, 2}));
    EXPECT_EQ(result.node_count, 3);
}

/** Unequal slots and noncontiguous placement need no rank/socket formula. */
TEST(Test__NodeDetection, SharedLeadersHandleUnequalNodeMembership)
{
    const auto result = NodeDetection::fromSharedNodeLeaders({0, 0, 2, 0, 2, 5});
    EXPECT_EQ(result.node_ids, (std::vector<int>{0, 0, 1, 0, 1, 2}));
    EXPECT_EQ(result.node_count, 3);
}

TEST(Test__NodeDetection, SharedLeadersRejectUnauthenticatedMembership)
{
    for (const auto &leaders : std::vector<std::vector<int>>{
             {-1}, {1}, {0, 2}, {0, 0, 1}, {0, 7, 0}, {0, -1}})
        EXPECT_THROW(NodeDetection::fromSharedNodeLeaders(leaders), std::invalid_argument);
}

TEST(Test__NodeDetection, SharedLeadersEmptyAndSingleRank)
{
    EXPECT_EQ(NodeDetection::fromSharedNodeLeaders({}).node_count, 0);
    EXPECT_EQ(NodeDetection::fromSharedNodeLeaders({0}).node_ids, (std::vector<int>{0}));
}
