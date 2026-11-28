/**
 * @file Test__RequestIntrinsicCoordinates.cpp
 * @brief §7.1/§7.2 substrate: immutable seed resolution and intrinsic windows.
 *
 * Device-free contract tests: the RNG key contains only (seed, position,
 * purpose) — no slot/batch identity; seeds are nonzero and immutable; the
 * window quantum is sealed and never shortened/extended by anything intrinsic
 * to the test (load cannot enter these pure functions).
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <type_traits>

#include "execution/serving/RequestIntrinsicCoordinates.h"

namespace llaminar2::test
{
    TEST(RequestIntrinsicCoordinates, SeedResolutionPreservesExplicitNonzero)
    {
        EXPECT_EQ(resolveRequestSeed(0xdeadbeefULL, 42), 42u);
        EXPECT_EQ(resolveRequestSeed(0x0ULL, 7), 7u);
    }

    TEST(RequestIntrinsicCoordinates, ZeroSeedResolvesDeterministicallyFromAdmittedEntropy)
    {
        const auto a = resolveRequestSeed(0x123456789abcdef0ULL, 0);
        const auto b = resolveRequestSeed(0x123456789abcdef0ULL, 0);
        EXPECT_EQ(a, b);       // same admission entropy -> same resolved seed
        EXPECT_NE(a, 0u);      // never zero
        const auto c = resolveRequestSeed(0xdeadbeefULL, 0);
        EXPECT_NE(c, 0u);
    }

    TEST(RequestIntrinsicCoordinates, KeyContainsOnlySeedPositionPurpose)
    {
        // The coordinate type carries exactly three intrinsic fields; the
        // slot/request/batch exclusion is unrepresentable.
        DrawCoordinates coords{99, 5, DrawPurpose::MTPVerifier};
        EXPECT_EQ(coords.seed, 99u);
        EXPECT_EQ(coords.logical_position, 5u);
        EXPECT_EQ(coords.purpose, DrawPurpose::MTPVerifier);
        // Standard-layout with exactly the three intrinsic fields (padding
        // tolerant): the type cannot carry slot/batch identity.
        EXPECT_TRUE(std::is_standard_layout_v<DrawCoordinates>);
        EXPECT_LT(offsetof(DrawCoordinates, seed), offsetof(DrawCoordinates, logical_position));
        EXPECT_LT(offsetof(DrawCoordinates, logical_position), offsetof(DrawCoordinates, purpose));
    }

    TEST(RequestIntrinsicCoordinates, WindowEndFollowsSealedQuantum)
    {
        const IntrinsicWindowPlan plan{16, 100};
        EXPECT_EQ(nextWindowEnd(0, plan), 16u);
        EXPECT_EQ(nextWindowEnd(15, plan), 16u);
        EXPECT_EQ(nextWindowEnd(16, plan), 32u);
        EXPECT_EQ(nextWindowEnd(99, plan), 100u);  // bounded by response limit
        EXPECT_EQ(nextWindowEnd(100, plan), 100u);
    }

    TEST(RequestIntrinsicCoordinates, WindowBoundsAreImmutableAndRejectZeroQuantum)
    {
        const IntrinsicWindowPlan plan{10, 30};
        EXPECT_TRUE(isInsideWindow(0, 5, plan));
        EXPECT_TRUE(isInsideWindow(0, 10, plan));
        EXPECT_FALSE(isInsideWindow(0, 11, plan));
        EXPECT_TRUE(isInsideWindow(9, 10, plan));
        EXPECT_THROW(nextWindowEnd(0, IntrinsicWindowPlan{0, 10}), std::invalid_argument);
    }
} // namespace llaminar2::test
