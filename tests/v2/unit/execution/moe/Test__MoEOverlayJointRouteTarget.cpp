/**
 * @file Test__MoEOverlayJointRouteTarget.cpp
 * @brief Replay fixed captured geometry against changing original-bank targets.
 *
 * Native packet capture is tested separately. These metadata tests exercise the
 * same typed production resolver without claiming CPU worker readiness.
 */
#include "execution/moe/MoEOverlayJointRouteTarget.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
using namespace llaminar2;
/** @test Whole ownership remains unchanged and cohorts never become scalar owners. */
TEST(MoEOverlayJointRouteTarget, WholeAndCohortEncodingsAreDisjoint)
{
    EXPECT_EQ(MoEOverlayRouteTarget::cpuCohort(0).word(), -2);
    EXPECT_EQ(MoEOverlayRouteTarget::cpuCohort(4).word(), -6);
    EXPECT_EQ(MoEOverlayRouteTarget::cpuCohort(std::numeric_limits<int>::max() - 1).word(),
              std::numeric_limits<int>::min());
    const MoEOverlayJointRouteTarget ordinary;
    for (int owner : {-1, 0, 1, 8, 71})
    {
        int selected = -99;
        ASSERT_TRUE(ordinary.resolve(3, owner, &selected));
        EXPECT_EQ(selected, owner);
    }
    int selected = 99;
    EXPECT_TRUE(ordinary.resolve(3, MoEOverlayRouteTarget::cpuCohort(4).word(), &selected));
    EXPECT_EQ(selected, -1);
    EXPECT_FALSE(ordinary.resolve(3, 0, nullptr));
    EXPECT_THROW(MoEOverlayRouteTarget::whole(-1), std::invalid_argument);
    EXPECT_THROW(MoEOverlayRouteTarget::cpuCohort(-1), std::invalid_argument);
    EXPECT_THROW(MoEOverlayRouteTarget::cpuCohort(std::numeric_limits<int>::max()),
                 std::invalid_argument);
}
/** @test One captured fixed family follows bank1/2/3 membership without recapture or source-ID
 * bits. */
TEST(MoEOverlayJointRouteTarget, LiveBankMembershipReplayRetainsFixedCohortThroughEightMembers)
{
    for (int degree = 1; degree <= 8; ++degree)
    {
        std::array<MoEOverlayJointRouteTarget, 8> lanes;
        for (int member = 0; member < degree; ++member)
            lanes[member] =
                MoEOverlayJointRouteTarget::fromOriginalCohort(512, 7 + 13 * member, 9, 4);
        for (int epoch : {1, 2, 3, 2, 1})
        {
            // Genuine bank words, not a second retained bitmap. Membership swaps original IDs.
            const int incoming = epoch == 2 ? 1 : 64;
            for (int expert : {0, 1, 64, 511})
                for (int member = 0; member < degree; ++member)
                {
                    int selected = -99;
                    const int word =
                        expert == incoming ? MoEOverlayRouteTarget::cpuCohort(4).word() : 0;
                    ASSERT_TRUE(lanes[member].resolve(expert, word, &selected));
                    EXPECT_EQ(selected, expert == incoming ? 7 + 13 * member : 0);
                    EXPECT_TRUE(lanes[member].validFor(512, 7 + 13 * member, 9));
                    EXPECT_FALSE(lanes[member].validFor(512, 7 + 13 * member, 10));
                }
        }
        int selected = 99;
        EXPECT_TRUE(lanes[0].resolve(1, MoEOverlayRouteTarget::cpuCohort(5).word(), &selected));
        EXPECT_EQ(selected, -1);
        EXPECT_FALSE(
            lanes[0].resolve(1, 7, &selected)); // Conflicting whole owner for a fixed shard lane.
        EXPECT_FALSE(lanes[0].resolve(512, -1, &selected));
    }
}

/** @test Retired expert bitmaps cannot weaken the original fixed-geometry admission checks. */
TEST(MoEOverlayJointRouteTarget, RejectsMissingOrForeignFixedGeometry)
{
    EXPECT_THROW(MoEOverlayJointRouteTarget::fromOriginalCohort(0, 7, 9, 4), std::invalid_argument);
    EXPECT_THROW(MoEOverlayJointRouteTarget::fromOriginalCohort(513, 7, 9, 4),
                 std::invalid_argument);
    EXPECT_THROW(MoEOverlayJointRouteTarget::fromOriginalCohort(512, -1, 9, 4),
                 std::invalid_argument);
    EXPECT_THROW(MoEOverlayJointRouteTarget::fromOriginalCohort(512, 7, -1, 4),
                 std::invalid_argument);
    EXPECT_THROW(MoEOverlayJointRouteTarget::fromOriginalCohort(512, 7, 9, -1),
                 std::invalid_argument);
    const auto lane = MoEOverlayJointRouteTarget::fromOriginalCohort(512, 7, 9, 4);
    EXPECT_FALSE(lane.validFor(511, 7, 9));
    EXPECT_FALSE(lane.validFor(512, 8, 9));
    EXPECT_FALSE(lane.validFor(512, 7, 10));
    int untouched = 91;
    EXPECT_FALSE(lane.resolve(-1, 0, &untouched));
    EXPECT_EQ(untouched, 91);
    EXPECT_FALSE(lane.resolve(512, 0, &untouched));
    EXPECT_EQ(untouched, 91);
}
