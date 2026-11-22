/**
 * @file Test__MultiDeviceIdentity.cpp
 * @brief §12 multi-device identity/agreement substrate.
 *
 * Slot identity maps to every participant; only the declared continuation
 * may sample; epoch ordering is typed; participation digests are device
 * authored and only compared (a host-authored value never agrees). Device-free.
 */

#include <gtest/gtest.h>

#include "execution/serving/MultiDeviceIdentity.h"

namespace llaminar2::test
{
    TEST(MultiDeviceIdentity, SlotIdentitySharedAcrossParticipants)
    {
        const MultiDeviceSlotIdentity rank0{0, 7};
        const MultiDeviceSlotIdentity rank1{0, 7}; // same slot+generation on every shard
        EXPECT_TRUE(rank0.sameIdentity(rank1));
        const MultiDeviceSlotIdentity replaced{0, 8};
        EXPECT_FALSE(rank0.sameIdentity(replaced));
    }

    TEST(MultiDeviceIdentity, OnlyDeclaredContinuationSamples)
    {
        ContinuationGuard follower{false};
        EXPECT_THROW(follower.assertMaySample(), std::logic_error);
        ContinuationGuard continuation{true};
        EXPECT_NO_THROW(continuation.assertMaySample());
    }

    TEST(MultiDeviceIdentity, EpochOrderingIsOrdered)
    {
        const SlotQualifiedEpochs a{3, 1};
        const SlotQualifiedEpochs b{4, 0};
        EXPECT_TRUE(a.ordersBefore(b));
        const SlotQualifiedEpochs c{4, 1};
        EXPECT_TRUE(b.ordersBefore(c));
        EXPECT_FALSE(c.ordersBefore(c));
    }

    TEST(MultiDeviceIdentity, ParticipationDigestIsDeviceAuthoredAndComparedOnly)
    {
        const ParticipationDigest d1{0xabc, true};
        const ParticipationDigest d2{0xabc, true};
        EXPECT_TRUE(d1.agreesWith(d2));
        const ParticipationDigest host{0xabc, false}; // host-authored: no oracle
        EXPECT_FALSE(d1.agreesWith(host));
        EXPECT_FALSE(d1.agreesWith(ParticipationDigest{0xdef, true}));
    }
} // namespace llaminar2::test
