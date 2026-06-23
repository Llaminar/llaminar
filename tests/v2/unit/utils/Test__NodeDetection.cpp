/**
 * @file Test__NodeDetection.cpp
 * @brief Unit tests for NodeDetection — canonical hostname-based node detection
 */

#include <gtest/gtest.h>
#include "utils/NodeDetection.h"
#include <array>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <unistd.h>

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

// =============================================================================
// detect() test — requires MPI, uses MPI_COMM_SELF (single rank)
// =============================================================================

TEST(Test__NodeDetection, Detect_SingleRank)
{
    // MPI_COMM_SELF is a valid communicator with exactly 1 rank
    auto result = NodeDetection::detect(MPI_COMM_SELF);
    EXPECT_EQ(result.node_count, 1);
    ASSERT_EQ(result.node_ids.size(), 1u);
    EXPECT_EQ(result.node_ids[0], 0);
    ASSERT_EQ(result.hostnames.size(), 1u);
    EXPECT_FALSE(result.hostnames[0].empty());
}

TEST(Test__NodeDetection, Detect_NullComm)
{
    auto result = NodeDetection::detect(MPI_COMM_NULL);
    EXPECT_EQ(result.node_count, 0);
    EXPECT_TRUE(result.node_ids.empty());
}

// =============================================================================
// Helper: write a temporary hostfile for testing
// =============================================================================

namespace
{
    class TempHostfile
    {
    public:
        explicit TempHostfile(const std::string &content)
        {
            std::array<char, 64> path_template{};
            std::snprintf(path_template.data(), path_template.size(), "/tmp/llaminar-hostfile-XXXXXX");

            int fd = ::mkstemp(path_template.data());
            if (fd == -1)
                throw std::runtime_error("mkstemp failed for TempHostfile");
            ::close(fd);

            path_ = path_template.data();
            std::ofstream ofs(path_);
            ofs << content;
        }
        ~TempHostfile() { std::remove(path_.c_str()); }
        const std::string &path() const { return path_; }

    private:
        std::string path_;
    };
} // namespace

// =============================================================================
// parseHostfile() tests
// =============================================================================

TEST(Test__NodeDetection, ParseHostfile_Basic)
{
    TempHostfile hf("node-a slots=4\nnode-b slots=2\n");
    auto nodes = NodeDetection::parseHostfile(hf.path());
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_EQ(nodes[0].first, "node-a");
    EXPECT_EQ(nodes[0].second, 0);
    EXPECT_EQ(nodes[1].first, "node-b");
    EXPECT_EQ(nodes[1].second, 1);
}

TEST(Test__NodeDetection, ParseHostfile_DuplicateHostname)
{
    // Same hostname appears twice — should be deduplicated
    TempHostfile hf("node-a slots=2\nnode-b slots=2\nnode-a slots=1\n");
    auto nodes = NodeDetection::parseHostfile(hf.path());
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_EQ(nodes[0].first, "node-a");
    EXPECT_EQ(nodes[1].first, "node-b");
}

TEST(Test__NodeDetection, ParseHostfile_CommentsAndBlankLines)
{
    TempHostfile hf("# This is a comment\n\nnode-a\n  # another comment\nnode-b\n\n");
    auto nodes = NodeDetection::parseHostfile(hf.path());
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_EQ(nodes[0].first, "node-a");
    EXPECT_EQ(nodes[1].first, "node-b");
}

TEST(Test__NodeDetection, ParseHostfile_NoSlotsSpecifier)
{
    TempHostfile hf("node-a\nnode-b\n");
    auto nodes = NodeDetection::parseHostfile(hf.path());
    ASSERT_EQ(nodes.size(), 2u);
    EXPECT_EQ(nodes[0].first, "node-a");
    EXPECT_EQ(nodes[1].first, "node-b");
}

TEST(Test__NodeDetection, ParseHostfile_NonexistentFile)
{
    auto nodes = NodeDetection::parseHostfile("/nonexistent/path/hostfile.txt");
    EXPECT_TRUE(nodes.empty());
}

// =============================================================================
// fromHostnames(hostnames, hostfile_path) tests — hostfile-aware ordering
// =============================================================================

TEST(Test__NodeDetection, FromHostnames_WithHostfile_OverridesOrder)
{
    // Hostfile says: node-b is node 0, node-a is node 1
    TempHostfile hf("node-b slots=2\nnode-a slots=2\n");

    // MPI ranks: rank 0 on node-a, rank 1 on node-a, rank 2 on node-b, rank 3 on node-b
    auto result = NodeDetection::fromHostnames(
        {"node-a", "node-a", "node-b", "node-b"}, hf.path());

    EXPECT_EQ(result.node_count, 2);
    ASSERT_EQ(result.node_ids.size(), 4u);
    // Hostfile ordering: node-b=0, node-a=1
    EXPECT_EQ(result.node_ids[0], 1); // node-a → hostfile id 1
    EXPECT_EQ(result.node_ids[1], 1);
    EXPECT_EQ(result.node_ids[2], 0); // node-b → hostfile id 0
    EXPECT_EQ(result.node_ids[3], 0);
}

TEST(Test__NodeDetection, FromHostnames_WithHostfile_UnknownHostFallback)
{
    // Hostfile only knows about node-a
    TempHostfile hf("node-a slots=2\n");

    // MPI has ranks on node-a and node-unknown
    auto result = NodeDetection::fromHostnames(
        {"node-a", "node-unknown", "node-a"}, hf.path());

    EXPECT_EQ(result.node_count, 2);
    ASSERT_EQ(result.node_ids.size(), 3u);
    EXPECT_EQ(result.node_ids[0], 0); // node-a → from hostfile
    EXPECT_EQ(result.node_ids[1], 1); // node-unknown → new sequential id
    EXPECT_EQ(result.node_ids[2], 0); // node-a → from hostfile
}

TEST(Test__NodeDetection, FromHostnames_WithEmptyHostfile_FallsBackToDefault)
{
    TempHostfile hf("# only comments\n\n");

    auto result = NodeDetection::fromHostnames(
        {"node-a", "node-b"}, hf.path());

    // Should fall back to first-appearance ordering
    EXPECT_EQ(result.node_count, 2);
    EXPECT_EQ(result.node_ids[0], 0); // node-a first
    EXPECT_EQ(result.node_ids[1], 1); // node-b second
}

TEST(Test__NodeDetection, FromHostnames_WithNonexistentHostfile_FallsBack)
{
    auto result = NodeDetection::fromHostnames(
        {"node-a", "node-b"}, "/nonexistent/hostfile");

    // Should fall back to first-appearance ordering
    EXPECT_EQ(result.node_count, 2);
    EXPECT_EQ(result.node_ids[0], 0);
    EXPECT_EQ(result.node_ids[1], 1);
}

// =============================================================================
// detect() with hostfile — uses MPI_COMM_SELF (single rank)
// =============================================================================

TEST(Test__NodeDetection, Detect_WithHostfile_SingleRank)
{
    // Even with a hostfile, detect should work for a single rank
    TempHostfile hf("some-host slots=1\n");
    auto result = NodeDetection::detect(MPI_COMM_SELF, hf.path());
    EXPECT_GE(result.node_count, 1);
    ASSERT_EQ(result.node_ids.size(), 1u);
    ASSERT_EQ(result.hostnames.size(), 1u);
    EXPECT_FALSE(result.hostnames[0].empty());
}

TEST(Test__NodeDetection, Detect_WithEmptyHostfilePath_DefaultBehavior)
{
    // Empty hostfile path = default behavior
    auto result = NodeDetection::detect(MPI_COMM_SELF, "");
    EXPECT_EQ(result.node_count, 1);
    ASSERT_EQ(result.node_ids.size(), 1u);
    EXPECT_EQ(result.node_ids[0], 0);
}
