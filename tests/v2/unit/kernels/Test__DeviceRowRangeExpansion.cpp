/**
 * @file Test__DeviceRowRangeExpansion.cpp
 * @brief Device-free count-authority, expansion, capture identity and overflow controls.
 */
#include "kernels/common/DeviceRowRange.h"
#include <gtest/gtest.h>
#include <limits>
using namespace llaminar2;
TEST(DeviceRowRangeExpansion, FactorOnePreservesEveryOriginalRowAndIdentity)
{
    std::int32_t count = 3;
    for (auto original : {DeviceRowRange::fullyActive(9), DeviceRowRange::deviceCounted(9, &count)})
        for (int first = 0; first < 9; ++first)
            for (int width = 1; width <= 9 - first; ++width)
            {
                const auto tile = original.slice(first, width);
                const auto same = tile.repeatEachRow(1);
                EXPECT_EQ(tile.captureIdentity(), same.captureIdentity());
                for (int published = -1; published <= 10; ++published)
                    EXPECT_EQ(tile.activeRowsFor(published), same.activeRowsFor(published));
            }
}
TEST(DeviceRowRangeExpansion, ExpandsOriginalAuthorityAndClipsPhysicalSlices)
{
    std::int32_t count = 0;
    const auto rows = DeviceRowRange::deviceCounted(7, &count).repeatEachRow(4);
    EXPECT_EQ(rows.countOwner(), &count);
    EXPECT_EQ(rows.capacity(), 28);
    EXPECT_EQ(rows.countCapacity(), 7);
    for (int first = 0; first < 28; ++first)
        for (int width = 1; width <= 28 - first; ++width)
            for (int published = -1; published <= 8; ++published)
            {
                const int expected = published < 0 || published > 7 ? -1 :
                    std::min(width, std::max(0, 4 * published - first));
                EXPECT_EQ(rows.slice(first, width).activeRowsFor(published), expected);
            }
}
TEST(DeviceRowRangeExpansion, RepetitionAndSlicingComposeWithOneBorrowedCount)
{
    std::int32_t count = 2;
    const auto a = DeviceRowRange::deviceCounted(7, &count).slice(1, 3).repeatEachRow(4);
    const auto b = DeviceRowRange::deviceCounted(7, &count).repeatEachRow(2).slice(2, 6).repeatEachRow(2);
    EXPECT_EQ(a.captureIdentity(), b.captureIdentity());
    EXPECT_EQ(a.activeRowsFor(2), 4);
    EXPECT_EQ(a.activeRowsFor(0), 0);
    EXPECT_EQ(a.activeRowsFor(7), 12);
}
TEST(DeviceRowRangeExpansion, CaptureIdentityDistinguishesEveryExpandedField)
{
    std::int32_t first = 1, other = 1;
    auto rows = DeviceRowRange::deviceCounted(8, &first).repeatEachRow(4);
    EXPECT_NE(rows.captureIdentity(), DeviceRowRange::deviceCounted(8, &other).repeatEachRow(4).captureIdentity());
    EXPECT_NE(rows.captureIdentity(), DeviceRowRange::deviceCounted(8, &first).captureIdentity());
    EXPECT_NE(rows.captureIdentity(), rows.slice(1, 31).captureIdentity());
    EXPECT_NE(rows.captureIdentity(), rows.slice(0, 31).captureIdentity());
    EXPECT_NE(rows.slice(0, 16).captureIdentity(),
        DeviceRowRange::deviceCounted(4, &first).repeatEachRow(8).slice(0, 16).captureIdentity());
    EXPECT_EQ(rows.captureIdentity().schema_version, 2);
    const auto identity = rows.captureIdentity();
    first = 7;
    EXPECT_EQ(rows.captureIdentity(), identity); // Contents are execution state, never capture identity.
}
TEST(DeviceRowRangeExpansion, RejectsInvalidFactorOverflowAndPhysicalSlice)
{
    const auto rows = DeviceRowRange::fullyActive(9);
    EXPECT_THROW((void)rows.repeatEachRow(0), std::invalid_argument);
    EXPECT_THROW((void)rows.repeatEachRow(-1), std::invalid_argument);
    EXPECT_THROW((void)rows.repeatEachRow(std::numeric_limits<int>::max()), std::overflow_error);
    EXPECT_THROW((void)DeviceRowRange::fullyActive(std::numeric_limits<int>::max()).repeatEachRow(2), std::overflow_error);
    const auto expanded = rows.repeatEachRow(4);
    EXPECT_THROW((void)expanded.slice(-1, 1), std::out_of_range);
    EXPECT_THROW((void)expanded.slice(0, 0), std::out_of_range);
    EXPECT_THROW((void)expanded.slice(35, 2), std::out_of_range);
    const auto boundary = DeviceRowRange::fullyActive(std::numeric_limits<int>::max()).repeatEachRow(1);
    EXPECT_EQ(boundary.activeRowsFor(std::numeric_limits<int>::max()), std::numeric_limits<int>::max());
}
