/**
 * @file Test__EffectiveUseOrdering.cpp
 * @brief Effective-use receipt, idempotence, watermark and pending-pin contracts.
 */
#include "kernels/cpu/attention/EffectiveUseOrdering.h"
#include <gtest/gtest.h>

using namespace llaminar2;

/** @test Accepted touches advance a watermark; duplicates are idempotent. */
TEST(EffectiveUseOrdering, AcceptanceAndWatermarkContracts)
{
    EffectiveUseTracker tracker;
    EffectiveUseReceipt first{1, 1, 1, 0};
    EffectiveUseReceipt second{1, 2, 1, 0};
    EXPECT_TRUE(tracker.accept(first));
    EXPECT_EQ(tracker.watermark(), 1u);
    EXPECT_EQ(tracker.acknowledged(first), 1u);
    EXPECT_FALSE(tracker.accept(first)); // Idempotent duplicate.
    EXPECT_EQ(tracker.watermark(), 1u);
    EXPECT_TRUE(tracker.accept(second));
    EXPECT_EQ(tracker.watermark(), 2u);
    EXPECT_EQ(tracker.acknowledged(second), 2u);
}

/** @test Obsolete incarnation cannot refresh under the same content version. */
TEST(EffectiveUseOrdering, ObsoleteIncarnationDoesNotRefreshReplacement)
{
    EffectiveUseTracker tracker;
    EffectiveUseReceipt old_incarnation{1, 1, 7, 0};
    EffectiveUseReceipt replacement{1, 1, 9, 0};
    EXPECT_TRUE(tracker.accept(old_incarnation));
    EXPECT_TRUE(tracker.accept(replacement));
    EXPECT_EQ(tracker.acknowledged(old_incarnation), 1u);
    EXPECT_EQ(tracker.acknowledged(replacement), 2u);
    // A touch for an even older incarnation is a distinct entry, never a refresh.
    EffectiveUseReceipt ancient{1, 1, 3, 0};
    EXPECT_TRUE(tracker.accept(ancient));
    EXPECT_EQ(tracker.acknowledged(ancient), 3u);
    EXPECT_EQ(tracker.watermark(), 3u);
}

/** @test Eviction gate: pending touch pins; acknowledgement releases; reordered receipts never defeat it. */
TEST(EffectiveUseOrdering, EvictionGatePinsThenAcknowledgesOrderly)
{
    EffectiveUseGate gate;
    EffectiveUseReceipt first{1, 1, 1, 0};
    EffectiveUseReceipt second{1, 2, 1, 0};
    EffectiveUseReceipt obsolete{0, 1, 7, 0};
    // A new accepted touch pins its backing: eviction is blocked.
    gate.pin(first);
    EXPECT_TRUE(gate.hasPending());
    EXPECT_FALSE(gate.mayEvict(first));
    // Acknowledgment is the ordering point: the pin releases and the exact
    // incarnation becomes evictable while ordinary recency governs.
    gate.acknowledge(first);
    EXPECT_FALSE(gate.hasPending());
    EXPECT_TRUE(gate.mayEvict(first));
    // Reordered receipts: a second accepted touch still blocks the gate;
    // acknowledging the first again is idempotent and cannot release it.
    gate.pin(second);
    gate.acknowledge(first); // Obsolete acknowledgement: idempotent.
    EXPECT_TRUE(gate.hasPending());
    EXPECT_FALSE(gate.mayEvict(second));
    gate.acknowledge(second);
    EXPECT_TRUE(gate.mayEvict(second));
    // Foreign/obsolete receipts are never evictable through this gate.
    EXPECT_FALSE(gate.mayEvict(obsolete));
    // Duplicate pin of an already-pinned touch keeps exactly one pin.
    gate.pin(first);
    gate.pin(first);
    EXPECT_TRUE(gate.hasPending());
    gate.acknowledge(first);
    EXPECT_FALSE(gate.hasPending());
    EXPECT_TRUE(gate.mayEvict(first));
}

/** @test Bounded pending pin guards unacknowledged accepted use; capacity refuses overflow. */
TEST(EffectiveUseOrdering, PendingPinAndBoundedCapacity)
{
    EffectiveUseTracker tracker;
    tracker.pinForPendingUse();
    EXPECT_TRUE(tracker.hasPendingPin());
    tracker.releasePendingUse();
    EXPECT_FALSE(tracker.hasPendingPin());
    tracker.releasePendingUse(); // Idempotent release.
    EXPECT_FALSE(tracker.hasPendingPin());
    // Capacity: accept up to the fixed limit, then refuse (typed).
    for (std::size_t i = 0; i < EffectiveUseTracker::capacity(); ++i)
        EXPECT_TRUE(tracker.accept({1, 1, static_cast<std::uint64_t>(i + 1), 0}));
    EXPECT_THROW(tracker.accept({1, 1, 9999, 0}), std::logic_error);
    EXPECT_EQ(tracker.watermark(), EffectiveUseTracker::capacity());
}
