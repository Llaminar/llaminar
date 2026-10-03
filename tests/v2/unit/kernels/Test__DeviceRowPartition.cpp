/**
 * @file Test__DeviceRowPartition.cpp
 * @brief Device-free proof of exact live-row ownership for captured distribution.
 *
 * Exhaustive small counts and large arithmetic sentinels prove disjoint complete
 * coverage, bounded imbalance, monotone per-owner storage, and rejection of
 * corrupt counts. No allocation capacity may be mistaken for a wire extent.
 */
#include "kernels/common/DeviceRowPartition.h"
#include "kernels/common/MoERouterRowPacket.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <climits>

using namespace llaminar2;

/** @test Every live row has one owner for degrees through eight, including empties. */
TEST(DeviceRowPartition, ExhaustiveLivePrefixes)
{
    for (int capacity : {1, 2, 15, 16, 17, 63, 65, 129, 512})
        for (int degree = 1; degree <= 8; ++degree)
            for (int live = 0; live <= capacity; ++live)
            {
                int end = 0, minimum = INT_MAX, maximum = 0;
                for (int rank = 0; rank < degree; ++rank)
                {
                    const auto partition = DeviceRowPartition::balanced(rank, degree);
                    const auto span = partition.resolveFor(capacity, live);
                    ASSERT_TRUE(span.valid());
                    EXPECT_EQ(span.first, end);
                    EXPECT_LE(span.count, partition.capacityFor(capacity));
                    EXPECT_LE(span.count, partition.resolveFor(capacity + 1, live + 1).count);
                    for (int row = span.first; row < span.first + span.count; ++row)
                        EXPECT_EQ(partition.ownerFor(capacity, live, row), rank);
                    EXPECT_EQ(partition.ownerFor(capacity, live, live), -1);
                    EXPECT_EQ(partition.ownerFor(capacity, live, -1), -1);
                    minimum = std::min(minimum, span.count);
                    maximum = std::max(maximum, span.count);
                    end += span.count;
                }
                EXPECT_EQ(end, live);
                EXPECT_LE(maximum - minimum, 1);
            }
}

/** @test Quotient/remainder arithmetic never multiplies two unbounded coordinates. */
TEST(DeviceRowPartition, LargeCountsDoNotOverflow)
{
    for (int degree : {1, 2, 3, 7, 8, 127, INT_MAX})
        for (int rank : {0, degree / 2, degree - 1})
            for (int live : {0, 1, 513, INT_MAX - 1, INT_MAX})
            {
                const auto partition = DeviceRowPartition::balanced(rank, degree);
                const auto span = partition.resolveFor(INT_MAX, live);
                ASSERT_TRUE(span.valid());
                const int64_t quotient = int64_t(live) / degree;
                const int64_t remainder = live % degree;
                EXPECT_EQ(span.first, quotient * rank + std::min<int64_t>(rank, remainder));
                EXPECT_EQ(span.count, quotient + (rank < remainder));
                EXPECT_LE(int64_t(span.first) + span.count, live);
                if (span.count)
                {
                    EXPECT_EQ(partition.ownerFor(INT_MAX, live, span.first), rank);
                    EXPECT_EQ(partition.ownerFor(INT_MAX, live, span.first + span.count - 1), rank);
                }
            }
}

/** @test Malformed membership/counts fail, rather than inventing a useful row. */
TEST(DeviceRowPartition, RejectsInvalidGeometry)
{
    EXPECT_THROW((void)DeviceRowPartition::balanced(0, 0), std::invalid_argument);
    EXPECT_THROW((void)DeviceRowPartition::balanced(-1, 2), std::invalid_argument);
    EXPECT_THROW((void)DeviceRowPartition::balanced(2, 2), std::invalid_argument);
    const auto partition = DeviceRowPartition::balanced(1, 2);
    EXPECT_FALSE(partition.resolveFor(0, 0).valid());
    EXPECT_FALSE(partition.resolveFor(16, -1).valid());
    EXPECT_FALSE(partition.resolveFor(16, 17).valid());
    EXPECT_THROW((void)partition.capacityFor(-1), std::invalid_argument);
    EXPECT_FALSE(DeviceRowPartition::resolveMember(16, 3, -1, 2).valid());
    EXPECT_FALSE(DeviceRowPartition::resolveMember(16, 3, 2, 2).valid());
    EXPECT_FALSE(DeviceRowPartition::resolveMember(16, 3, 0, 0).valid());
}

/** @test Equal peer strides admit unequal live packets without communicating the tail. */
TEST(DeviceRowPartition, PacketCapacityIsNotTheLiveExtent)
{
    EXPECT_FALSE(MoERouterRowPacketLayout{}.valid());
    for (int degree : {2, 3, 4, 8})
        for (int capacity : {2, 65, 512})
            for (int rank = 0; rank < degree; ++rank)
            {
                const MoERouterRowPacketLayout layout{DeviceRowPartition::balanced(rank, degree), capacity, 8};
                ASSERT_TRUE(layout.valid());
                EXPECT_EQ(layout.packetBytes(), size_t((capacity + degree - 1) / degree) * 8 * 8);
                for (int live : {0, 1, capacity - 1, capacity})
                {
                    const auto span = layout.partition.resolveFor(capacity, live);
                    EXPECT_LE(size_t(span.count) * 8 * 8, layout.packetBytes());
                }
            }
}
