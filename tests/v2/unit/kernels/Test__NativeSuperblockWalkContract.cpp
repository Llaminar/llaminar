/**
 * @file Test__NativeSuperblockWalkContract.cpp
 * @brief Device-free admission and progress checks for physical superblock walks
 *        and the production ordered-partition interval authority.
 *
 * Every NEON and x86 ordered-partition kernel forms its partition ranges with
 * orderedPartitionInterval() after its plan was admitted once per call (NEON
 * through admitsOrderedPartitionPlan() in requireExecutable, the fatal
 * wrappers elsewhere). These tests call those production helpers directly (no
 * re-implemented formula) and prove that every interval of an admitted plan is
 * an admitted superblock walk that strictly progresses, that the intervals tile
 * [begin, end) in order, and that invalid plans and ranges are rejected.
 */
#include <gtest/gtest.h>
#include <climits>
#include <cstdint>
#include <vector>
#include "kernels/cpu/gemm/CPUNativeSuperblockWalkContract.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedPreparedFootprint.h"
using namespace llaminar2::cpu::native_quantised;

namespace {
/** @brief One ordered plan over [begin, end) of a row with `blocks` blocks. */
struct Plan {
    int begin;
    int end;
    int blocks;
    int partitions;
    int width;
};

/** @return The canonical covering width ceil((end - begin) / partitions), formed in int64. */
int coveringWidth(int begin, int end, int partitions) {
    return static_cast<int>((std::int64_t(end) - begin + partitions - 1) / partitions);
}

/** @brief Singleton, odd, 37 and overpartitioned plans, at zero and nonzero begin, up to INT_MAX. */
std::vector<Plan> plans() {
    std::vector<Plan> result;
    for (int blocks : {0, 1, 7, 8, 9, 31, INT_MAX})
        for (int begin : {0, 1, 5, blocks / 2, blocks})
            for (int partitions : {1, 3, 17, 37}) {
                if (begin > blocks) continue;
                const int end = blocks;
                result.push_back({begin, end, blocks, partitions, coveringWidth(begin, end, partitions)});
                // Nonzero begin with an end inside the row (x86 tiles).
                if (end - begin > 2) result.push_back({begin, end - 1, blocks, partitions, coveringWidth(begin, end - 1, partitions)});
            }
    // Explicitly overpartitioned plans: more partitions than blocks, and widths
    // whose unsaturated starts lie beyond end (the reversed-range case).
    result.push_back({0, 8, 8, 37, 1});
    result.push_back({5, 13, 16, 37, 1});
    result.push_back({0, 9, 9, 3, 8});
    result.push_back({3, 12, 12, 17, 9});
    // INT_MAX extents whose unsaturated int start would overflow.
    result.push_back({0, INT_MAX, INT_MAX, 3, INT_MAX});
    result.push_back({INT_MAX - 10, INT_MAX, INT_MAX, 37, INT_MAX});
    result.push_back({1, INT_MAX, INT_MAX, 37, coveringWidth(1, INT_MAX, 37)});
    return result;
}
} // namespace

TEST(NativeSuperblockWalkContract, OrderedPartitionIntervalsTileAdmittedPlans) {
    for (const Plan &plan : plans())
        for (int phase = 0; phase < 8; ++phase) {
            SCOPED_TRACE(::testing::Message() << "begin=" << plan.begin << " end=" << plan.end << " blocks=" << plan.blocks
                                              << " partitions=" << plan.partitions << " width=" << plan.width << " phase=" << phase);
            ASSERT_TRUE(admitsOrderedPartitionPlan(plan.begin, plan.end, std::size_t(plan.blocks), plan.partitions, plan.width, phase));
            requireOrderedPartitionPlan(plan.begin, plan.end, std::size_t(plan.blocks), plan.partitions, plan.width, phase);
            const CPUNativeQuantisedBlockGeometry geometry{.blocks_per_row = std::size_t(plan.blocks), .phase = std::size_t(phase)};
            int expected_first = plan.begin;
            for (int partition = 0; partition < plan.partitions; ++partition) {
                const OrderedPartitionInterval interval = orderedPartitionInterval(plan.begin, plan.end, partition, plan.width);
                // Ordered, contiguous and inside [begin, end): never reversed.
                ASSERT_EQ(interval.first, expected_first) << "partition " << partition;
                ASSERT_LE(interval.first, interval.last) << "partition " << partition;
                ASSERT_LE(interval.last, plan.end) << "partition " << partition;
                // A nonempty interval equals the unsaturated form.
                if (interval.first < interval.last) {
                    const std::int64_t unsaturated = std::int64_t(plan.begin) + std::int64_t(partition) * plan.width;
                    EXPECT_EQ(interval.first, unsaturated);
                    EXPECT_EQ(interval.last, std::min<std::int64_t>(unsaturated + plan.width, plan.end));
                }
                // Each interval is an admitted walk by construction.
                ASSERT_TRUE(admitsSuperblockWalk(interval.first, interval.last, std::size_t(plan.blocks), phase));
                requireSuperblockWalk(interval.first, interval.last, std::size_t(plan.blocks), phase);
                // Sample both ends, including the ragged final superblock, without walking INT_MAX blocks.
                for (int block : {interval.first, interval.last - 1}) {
                    if (block >= interval.first && block < interval.last) {
                        EXPECT_GT(std::min<std::size_t>(interval.last, geometry.superblockEnd(block)), std::size_t(block));
                    }
                }
                expected_first = interval.last;
            }
            EXPECT_EQ(expected_first, plan.end) << "the plan's partitions cover [begin, end)";
        }
}

TEST(NativeSuperblockWalkContract, RejectsOriginalStuckRange) {
    EXPECT_DEATH(requireSuperblockWalk(8, 9, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(-1, 1, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(2, 1, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(0, 1, std::size_t(INT_MAX) + 1), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(0, 1, 8, 8), "Invalid CPU superblock walk range");
}

TEST(NativeSuperblockWalkContract, RejectsInvalidOrderedPartitionPlans) {
    // The noexcept predicates NEON's once-per-call admission uses reject the same plans.
    EXPECT_FALSE(admitsSuperblockWalk(8, 9, 8));
    EXPECT_FALSE(admitsOrderedPartitionPlan(8, 9, 8, 1, 1));
    EXPECT_FALSE(admitsOrderedPartitionPlan(0, 8, 8, 1, 8, 8));
    EXPECT_FALSE(admitsOrderedPartitionPlan(0, 8, 8, 0, 8));
    EXPECT_FALSE(admitsOrderedPartitionPlan(0, 8, 8, 2, -1));
    EXPECT_FALSE(admitsOrderedPartitionPlan(0, 9, 9, 2, 4));
    EXPECT_FALSE(admitsOrderedPartitionPlan(0, 8, std::size_t(INT_MAX) + 1, 1, 8));
    EXPECT_FALSE(admitsOrderedPartitionPlan(1, INT_MAX, INT_MAX, 37, coveringWidth(1, INT_MAX, 37) - 1));
    EXPECT_DEATH(requireOrderedPartitionPlan(8, 9, 8, 1, 1), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireOrderedPartitionPlan(0, 8, 8, 1, 8, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireOrderedPartitionPlan(0, 8, 8, 0, 8), "Invalid CPU ordered partition plan");
    EXPECT_DEATH(requireOrderedPartitionPlan(0, 8, 8, 2, -1), "Invalid CPU ordered partition plan");
    EXPECT_DEATH(requireOrderedPartitionPlan(0, 9, 9, 2, 4), "Invalid CPU ordered partition plan");
    EXPECT_DEATH(requireOrderedPartitionPlan(1, INT_MAX, INT_MAX, 37, coveringWidth(1, INT_MAX, 37) - 1),
                 "Invalid CPU ordered partition plan");
}
