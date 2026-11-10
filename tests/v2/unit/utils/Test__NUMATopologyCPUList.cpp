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
