/**
 * @file Test__DeviceRequestRowRanges.cpp
 * @brief Device-free admission and indexing proof for captured request rows.
 *
 * A length pointer is an identity only: deliberately invalid addresses prove
 * factories and pure validation never dereference device metadata on the host.
 * Empty requests preserve following requests' physical coordinates.
 */
#include <gtest/gtest.h>
#include "kernels/common/DeviceRequestRowRanges.h"
#include "kernels/gdn/GDNDeinterleaveRows.h"
#include <limits>

using namespace llaminar2;

TEST(DeviceRequestRowRanges, FixedRowsHaveNoMutableAuthority)
{
    const auto rows = DeviceRequestRowRanges::fullyActive(16);
    EXPECT_EQ(rows.requests(), 1);
    EXPECT_EQ(rows.rowsPerRequest(), 16);
    EXPECT_EQ(rows.physicalRows(), 16);
    EXPECT_EQ(rows.lengthOwner(), nullptr);
    EXPECT_THROW((void)DeviceRequestRowRanges::fullyActive(0), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::fullyActive(-1), std::invalid_argument);
}

TEST(DeviceRequestRowRanges, RequestPrefixesRemainDisjointAndChecked)
{
    const auto *identity = reinterpret_cast<const std::int32_t *>(std::uintptr_t{0x12340});
    const auto rows = DeviceRequestRowRanges::deviceCounted(3, 16, identity);
    EXPECT_EQ(rows.lengthOwner(), identity);
    EXPECT_EQ(rows.physicalRows(), 48);
    for (int request = 0; request < 3; ++request)
        for (int length = 0; length <= 16; ++length)
            EXPECT_EQ(rows.activeRowsFor(request, length), length);
    EXPECT_EQ(rows.activeRowsFor(0, 0), 0);
    EXPECT_EQ(rows.activeRowsFor(2, 16), 16);
    EXPECT_EQ(rows.activeRowsFor(-1, 1), -1);
    EXPECT_EQ(rows.activeRowsFor(3, 1), -1);
    EXPECT_EQ(rows.activeRowsFor(0, -1), -1);
    EXPECT_EQ(rows.activeRowsFor(0, 17), -1);
}

TEST(DeviceRequestRowRanges, InvalidAdmissionCannotOverflowOrOmitAuthority)
{
    const std::int32_t owner = 0;
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(1, 16, nullptr), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(0, 16, &owner), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(2, 0, &owner), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(2, std::numeric_limits<int>::max(), &owner),
        std::invalid_argument);
    const auto largest = DeviceRequestRowRanges::deviceCounted(1, std::numeric_limits<int>::max(), &owner);
    EXPECT_EQ(largest.physicalRows(), std::numeric_limits<int>::max());
}

TEST(DeviceRequestRowRanges, DeinterleaveGeometryRejectsUnrepresentableMatrices)
{
    const std::int32_t owner = 0;
    const auto rows = DeviceRequestRowRanges::deviceCounted(2, 16, &owner);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 128, 128), 16 * 40 * 128 * 3);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 0, 40, 128, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, -1, 128, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 0, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 128, -1), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max()), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(DeviceRequestRowRanges::fullyActive(
        std::numeric_limits<int>::max()), 1, 1, 1, 1), 0);
}
