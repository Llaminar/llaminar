/**
 * @file Test__HostRegistrationPageBoundary.cpp
 * @brief Device-free geometry and actual VMA invariants for host registration.
 *
 * Cover arbitrary byte offsets, one-page deduplication, range overflow, data
 * preservation and interior huge-page eligibility. No GPU or model is loaded.
 */
#include "memory/HostRegistrationPageBoundary.h"
#include "../../utils/HostPageMappingTestUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>

using namespace llaminar2;

TEST(HostRegistrationPageBoundary, ExactEdgePagesForEveryOffsetAndTail)
{
    constexpr std::size_t page = 4096u;
    for (std::size_t offset = 0; offset < page; ++offset)
        for (const auto bytes : {1u, 13u, 4096u, 8193u})
        {
            const auto boundary = HostRegistrationPageBoundary::forRange(0x10000u + offset, bytes, page);
            const auto last = (0x10000u + offset + bytes - 1u) & ~(page - 1u);
            ASSERT_EQ(boundary.pageBytes(), page);
            ASSERT_EQ(boundary.pages().front(), 0x10000u);
            ASSERT_EQ(boundary.pages().back(), last);
            ASSERT_EQ(boundary.pages().size(), last == 0x10000u ? 1u : 2u);
        }
}

TEST(HostRegistrationPageBoundary, RejectsInvalidGeometryBeforeAnySystemCall)
{
    EXPECT_THROW((void)HostRegistrationPageBoundary::forRange(0u, 1u, 4096u), std::invalid_argument);
    EXPECT_THROW((void)HostRegistrationPageBoundary::forRange(1u, 0u, 4096u), std::invalid_argument);
    EXPECT_THROW((void)HostRegistrationPageBoundary::forRange(1u, 1u, 0u), std::invalid_argument);
    EXPECT_THROW((void)HostRegistrationPageBoundary::forRange(1u, 1u, 4095u), std::invalid_argument);
    EXPECT_THROW((void)HostRegistrationPageBoundary::forRange(
        std::numeric_limits<std::uintptr_t>::max() - 7u, 9u, 4096u), std::overflow_error);
}

TEST(HostRegistrationPageBoundary, IsolatesOnlyEdgesRetainsPayloadAndInteriorHugePages)
{
    test::HostPageMapping mapping;
    auto *const first = mapping.data();
    auto *const last = first + test::HostPageMapping::live_bytes - 1u;
    auto *const middle = first + 2u * test::HostPageMapping::huge_bytes;
    ASSERT_TRUE(test::HostPageMapping::hasFlag(first, "hg"));
    HostRegistrationPageBoundary::prepare(first + 13u, test::HostPageMapping::live_bytes - 26u);
    EXPECT_TRUE(test::HostPageMapping::hasFlag(first, "nh"));
    EXPECT_TRUE(test::HostPageMapping::hasFlag(last, "nh"));
    EXPECT_TRUE(test::HostPageMapping::hasFlag(middle, "hg"));
    EXPECT_TRUE(test::HostPageMapping::hasFlag(first + test::HostPageMapping::live_bytes, "hg"));
    EXPECT_TRUE(std::all_of(first, first + test::HostPageMapping::payload_bytes,
        [](unsigned char byte) { return byte == 0x53; }));
    // Idempotent preparation must not rejoin the boundary with its neighbour.
    HostRegistrationPageBoundary::prepare(first + 13u, test::HostPageMapping::live_bytes - 26u);
    mapping.reclaimNeighbour();
    EXPECT_TRUE(std::all_of(first, first + test::HostPageMapping::live_bytes,
        [](unsigned char byte) { return byte == 0x53; }));
}
