/**
 * @file Test__EffectiveUseOrdering.cpp
 * @brief Effective-use receipt, membership-sized journal and eviction gate contracts.
 */
#include "kernels/cpu/attention/EffectiveUseOrdering.h"
#include <gtest/gtest.h>
#include <vector>

using namespace llaminar2;

/** @test Touches accept, refresh, duplicate and retire with typed outcomes. */
TEST(EffectiveUseOrdering, MembershipJournalAcceptsRefreshesAndReclaims)
{
    std::vector<EffectiveUseJournal::Entry> journal_storage(2);
    EffectiveUseJournal journal(journal_storage); // Two live incarnations.
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
    // TRUE retirement: a pending obligation refuses retirement; after
    // acknowledging the whole frontier the slot reclaims.
    EXPECT_FALSE(journal.retire(0)); // first+refresh still pending.
    EXPECT_TRUE(journal.acknowledge(refresh)); // Frontier: both released.
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
    std::vector<EffectiveUseJournal::Entry> journal_storage(2);
    EffectiveUseJournal journal(journal_storage);
    EXPECT_EQ(journal.touch({1, 1, 0, 1}), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch({1, 1, 1, 2}), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch({1, 1, 2, 3}), EffectiveUseStatus::JournalFull);
    // Refresh still works on a live incarnation (reclamation by refresh).
    EXPECT_EQ(journal.touch({1, 1, 0, 4}), EffectiveUseStatus::Refreshed);
}

/** @test Cumulative frontier: all live touches pending until the full identity acks. */
TEST(EffectiveUseOrdering, FullIdentityAndCumulativeFrontier)
{
    std::vector<EffectiveUseJournal::Entry> journal_storage(3);
    EffectiveUseJournal journal(journal_storage);
    EffectiveUseReceipt touch1{1, 1, 0, 1};
    EffectiveUseReceipt touch2{1, 1, 0, 2};
    EXPECT_EQ(journal.touch(touch1), EffectiveUseStatus::Accepted);
    EXPECT_FALSE(journal.mayEvict(0)); // touch1 pending.
    // R9 repro: touch1, touch2 on the same replica; ack2 with the FULL
    // identity (same content/table/incarnation) releases BOTH (cumulative).
    EXPECT_EQ(journal.touch(touch2), EffectiveUseStatus::Refreshed);
    EXPECT_FALSE(journal.acknowledge(touch1)); // Delayed old ack: releases nothing.
    EXPECT_FALSE(journal.mayEvict(0));          // touch1 AND touch2 pending.
    EXPECT_TRUE(journal.acknowledge(touch2));
    EXPECT_TRUE(journal.mayEvict(0));
    // R9 repro: wrong content/table at matching incarnation+ordinal refuses.
    EXPECT_FALSE(journal.acknowledge({1, 2, 0, 2})); // Wrong table_version.
    EXPECT_FALSE(journal.acknowledge({2, 1, 0, 2})); // Wrong content_version.
    EXPECT_FALSE(journal.acknowledge({9, 9, 7, 99})); // Foreign incarnation.
    // A lower sequence than the live frontier is obsolete and never re-refreshes.
    EXPECT_EQ(journal.touch(touch1), EffectiveUseStatus::Obsolete);
}

/** @test Long-lifetime behavior: many successful touches reuse reclaimed slots. */
TEST(EffectiveUseOrdering, ManyTouchesOverTimeReclaimAndStayBounded)
{
    std::vector<EffectiveUseJournal::Entry> journal_storage(2);
    EffectiveUseJournal journal(journal_storage);
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
        // TRUE retirement after full acknowledgement releases the slot; the
        // other incarnation still occupies its own slot meanwhile.
        EXPECT_TRUE(journal.retire(incarnation));
    }
}
