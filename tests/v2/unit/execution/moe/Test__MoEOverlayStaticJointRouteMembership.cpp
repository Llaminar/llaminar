/**
 * @file Test__MoEOverlayStaticJointRouteMembership.cpp
 * @brief Replay capture identity and source-cohort selection without a device.
 *
 * These tests call the production lane resolver, including empty routing work
 * and noncontiguous participant IDs. Native packet compaction/capture and real
 * CPU NUMA execution are separate integration obligations.
 */
#include "execution/moe/MoEOverlayStaticJointRouteMembership.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>

using llaminar2::MoEOverlayStaticJointRouteMembership;

/** @test Ordinary ownership remains independent of immutable CPU joint projections. */
TEST(MoEOverlayStaticJointRouteMembership, OrdinaryRoutingIsUnchanged)
{
    const MoEOverlayStaticJointRouteMembership ordinary;
    EXPECT_FALSE(ordinary.enabled());
    EXPECT_TRUE(ordinary.validFor(512, 71, 19));
    EXPECT_TRUE(ordinary.acceptsEpoch(902));
    for (const auto target : {-1, 0, 1, 8, 71})
    {
        std::int32_t resolved = -99;
        ASSERT_TRUE(ordinary.resolve(3, target, &resolved));
        EXPECT_EQ(resolved, target);
    }
    std::int32_t untouched = 71;
    EXPECT_FALSE(ordinary.resolve(3, -2, &untouched));
    EXPECT_EQ(untouched, 71);
    EXPECT_FALSE(ordinary.resolve(3, 1, nullptr));
}

/** @test Every 2..8-member projection receives joint routes, without an ordinal-ID assumption. */
TEST(MoEOverlayStaticJointRouteMembership, FanoutAndPartialReplayThroughEightMembers)
{
    constexpr std::array<std::int32_t, 5> joint_experts{1, 7, 31, 32, 511};
    constexpr std::array<std::int32_t, 12> routed_experts{1, 0, 7, 31, 32, 511, 6, 7, 8, 1, 511, 32};
    for (int members = 2; members <= 8; ++members)
    {
        std::vector<MoEOverlayStaticJointRouteMembership> lanes;
        for (int member = 0; member < members; ++member)
            lanes.push_back(MoEOverlayStaticJointRouteMembership::fromProjection(
                512, 3 + member * 5, 19, 37, joint_experts));
        // Repeated large/small/empty work must not carry mutable routing state.
        for (int replay = 0; replay < 20; ++replay)
            for (const std::size_t live : {12u, 1u, 0u, 5u, 12u})
                for (int member = members - 1; member >= 0; --member)
                    for (std::size_t slot = 0; slot < live; ++slot)
                    {
                        const auto expert = routed_experts[slot];
                        const bool joint = expert == 1 || expert == 7 || expert == 31 || expert == 32 || expert == 511;
                        const int durable = joint ? -1 : 99;
                        std::int32_t selected = -99;
                        ASSERT_TRUE(lanes[member].resolve(expert, durable, &selected));
                        EXPECT_EQ(selected, joint ? 3 + member * 5 : 99);
                    }
    }
}

/** @test Mismatched capture identity and conflicting owner data are not dispatchable. */
TEST(MoEOverlayStaticJointRouteMembership, RejectsStaleForeignAndMalformedProjection)
{
    constexpr std::array<std::int32_t, 3> experts{0, 32, 511};
    const auto lane = MoEOverlayStaticJointRouteMembership::fromProjection(512, 19, 7, 43, experts);
    ASSERT_TRUE(lane.enabled());
    EXPECT_TRUE(lane.validFor(512, 19, 7));
    EXPECT_FALSE(lane.validFor(511, 19, 7));
    EXPECT_FALSE(lane.validFor(512, 20, 7));
    EXPECT_FALSE(lane.validFor(512, 19, 8));
    EXPECT_TRUE(lane.acceptsEpoch(43));
    EXPECT_FALSE(lane.acceptsEpoch(0));
    EXPECT_FALSE(lane.acceptsEpoch(42));
    EXPECT_FALSE(lane.acceptsEpoch(44));
    std::int32_t untouched = 77;
    EXPECT_FALSE(lane.resolve(32, 19, &untouched));
    EXPECT_EQ(untouched, 77);
    EXPECT_FALSE(lane.contains(-1));
    EXPECT_FALSE(lane.contains(512));
    EXPECT_FALSE(lane.contains(31));
    EXPECT_THROW(MoEOverlayStaticJointRouteMembership::fromProjection(511, 19, 7, 43, experts), std::invalid_argument);
    EXPECT_THROW(MoEOverlayStaticJointRouteMembership::fromProjection(512, -1, 7, 43, experts), std::invalid_argument);
    EXPECT_THROW(MoEOverlayStaticJointRouteMembership::fromProjection(512, 19, -1, 43, experts), std::invalid_argument);
    EXPECT_THROW(MoEOverlayStaticJointRouteMembership::fromProjection(512, 19, 7, 0, experts), std::invalid_argument);
    EXPECT_THROW(MoEOverlayStaticJointRouteMembership::fromProjection(513, 19, 7, 43, experts), std::invalid_argument);
    const std::array<std::int32_t, 2> duplicate{7, 7};
    EXPECT_THROW(MoEOverlayStaticJointRouteMembership::fromProjection(512, 19, 7, 43, duplicate), std::invalid_argument);
}
