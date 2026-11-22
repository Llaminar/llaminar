/**
 * @file Test__MTPRowPolicy.cpp
 * @brief §6/§8 MTP row policy and exactly-once verifier publication contract.
 *
 * Depth policy + learner/sidecar identity are immutable; a logical position
 * may be accepted exactly once and never after rollback; rollback targets an
 * earlier unaccepted frontier. Device-free.
 */

#include <gtest/gtest.h>

#include "execution/serving/MTPRowPolicy.h"

namespace llaminar2::test
{
    TEST(MTPRowPolicy, AcceptanceIsExactlyOncePerPosition)
    {
        MTPVerifierPosition pos{5, false, false};
        acceptPosition(pos);
        EXPECT_TRUE(pos.accepted);
        EXPECT_THROW(acceptPosition(pos), std::logic_error);
    }

    TEST(MTPRowPolicy, PublishAfterRollbackIsRefused)
    {
        MTPVerifierPosition pos{5, false, false};
        rollbackTo(pos, 3);
        EXPECT_TRUE(pos.rolled_back);
        EXPECT_THROW(acceptPosition(pos), std::logic_error);
    }

    TEST(MTPRowPolicy, RollbackRefusesAcceptedOrNonEarlierTargets)
    {
        MTPVerifierPosition pos{4, true, false};
        EXPECT_THROW(rollbackTo(pos, 2), std::logic_error);
        MTPVerifierPosition frontier{4, false, false};
        EXPECT_THROW(rollbackTo(frontier, 4), std::logic_error);  // not strictly earlier
        EXPECT_THROW(rollbackTo(frontier, 9), std::logic_error);  // beyond current
    }

    TEST(MTPRowPolicy, PolicyRowsAreImmutableDescriptors)
    {
        MTPRowPolicy policy{4, MTPDepthPolicy::Fixed, 0xabc, 0xdef};
        EXPECT_EQ(policy.max_depth, 4u);
        EXPECT_EQ(policy.depth_policy, MTPDepthPolicy::Fixed);
        EXPECT_EQ(policy.learner_state_identity, 0xabcu);
        EXPECT_EQ(policy.sidecar_identity, 0xdefu);
    }
} // namespace llaminar2::test
