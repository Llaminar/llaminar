/**
 * @file Test__EffectiveUseOrdering.cpp
 * @brief Effective-use receipt, membership-sized journal and eviction gate contracts.
 */
#include "kernels/cpu/attention/EffectiveUseOrdering.h"
#include <gtest/gtest.h>

using namespace llaminar2;

/** @test Touches accept, refresh, duplicate and retire with typed outcomes. */
TEST(EffectiveUseOrdering, MembershipJournalAcceptsRefreshesAndReclaims)
{
    EffectiveUseJournal journal(2); // Two live incarnations.
    EffectiveUseReceipt first{1, 1, 0, 1};
    EffectiveUseReceipt refresh{1, 1, 0, 2};      // Newer touch, same incarnation.
    EffectiveUseReceipt obsolete{1, 1, 0, 0};     // Zero ordinal: obsolete.
    EffectiveUseReceipt second{1, 1, 1, 3};       // Another incarnation.
    EXPECT_EQ(journal.touch(first), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.watermark(), 1u);
    EXPECT_EQ(journal.touch(first), EffectiveUseStatus::Duplicate); // Idempotent.
    EXPECT_EQ(journal.watermark(), 1u);
    EXPECT_EQ(journal.touch(refresh), EffectiveUseStatus::Refreshed); // No-copy refresh.
    EXPECT_EQ(journal.watermark(), 2u);
    EXPECT_EQ(journal.touch(obsolete), EffectiveUseStatus::Obsolete); // Zero ordinal.
    EXPECT_EQ(journal.touch(second), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.watermark(), 3u);
    // Retirement reclaims capacity: a full journal can accept again after retire.
    EXPECT_TRUE(journal.retire(0));
    EffectiveUseReceipt third{1, 1, 2, 4};
    EXPECT_EQ(journal.touch(third), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch(third), EffectiveUseStatus::Duplicate);
    // Retiring an unknown incarnation is idempotent.
    EXPECT_FALSE(journal.retire(9));
}

/** @test Typed journal-full refusal; bounded by live membership, never silent. */
TEST(EffectiveUseOrdering, JournalFullIsTypedNotSilent)
{
    EffectiveUseJournal journal(2);
    EXPECT_EQ(journal.touch({1, 1, 0, 1}), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch({1, 1, 1, 2}), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch({1, 1, 2, 3}), EffectiveUseStatus::JournalFull);
    // Refresh still works on a live incarnation (reclamation by refresh).
    EXPECT_EQ(journal.touch({1, 1, 0, 4}), EffectiveUseStatus::Refreshed);
}

/** @test Acknowledgment is ordinal-exact; delayed acks never release newer pins. */
TEST(EffectiveUseOrdering, AcknowledgementIsOrdinalExact)
{
    EffectiveUseJournal journal(3);
    EffectiveUseReceipt old_touch{1, 1, 0, 1};
    EffectiveUseReceipt fresh_touch{1, 1, 0, 2};
    EXPECT_EQ(journal.touch(old_touch), EffectiveUseStatus::Accepted);
    EXPECT_FALSE(journal.mayEvict(0)); // Pending pin.
    // A DELAYED acknowledgement of an older ordinal cannot release a newer
    // re-pinned touch (it releases nothing).
    EXPECT_EQ(journal.touch(fresh_touch), EffectiveUseStatus::Refreshed);
    // A DELAYED acknowledgement of an OLD ordinal cannot release the NEW pin.
    EXPECT_FALSE(journal.acknowledge(old_touch));
    EXPECT_FALSE(journal.mayEvict(0));            // new pin still protected.
    EXPECT_TRUE(journal.acknowledge(fresh_touch));
    EXPECT_TRUE(journal.mayEvict(0));
    // Duplicate/foreign acks are idempotent.
    EXPECT_FALSE(journal.acknowledge({9, 9, 7, 99}));
    EXPECT_TRUE(journal.acknowledge(fresh_touch));
}

/** @test Long-lifetime behavior: many successful touches reuse reclaimed slots. */
TEST(EffectiveUseOrdering, ManyTouchesOverTimeReclaimAndStayBounded)
{
    EffectiveUseJournal journal(2);
    std::uint64_t ordinal = 0;
    for (std::size_t generation = 0; generation < 40; ++generation)
    {
        const auto incarnation = generation % 2;
        const auto status = journal.touch(
            EffectiveUseReceipt{1, 1, incarnation, ++ordinal});
        EXPECT_EQ(status, EffectiveUseStatus::Accepted);
        EXPECT_TRUE(journal.acknowledge(
            EffectiveUseReceipt{1, 1, incarnation, ordinal}));
        EXPECT_TRUE(journal.mayEvict(incarnation));
        // The other incarnation still occupies its slot.
        journal.retire(static_cast<std::uint64_t>(1 - incarnation));
    }
}
