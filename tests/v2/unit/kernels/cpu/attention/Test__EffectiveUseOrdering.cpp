/**
 * @file Test__EffectiveUseOrdering.cpp
 * @brief Per-operation effective-use lease contracts (R10 model).
 */
#include "kernels/cpu/attention/EffectiveUseOrdering.h"
#include <gtest/gtest.h>
#include <vector>

using namespace llaminar2;

namespace
{
struct JournalHarness
{
    std::vector<EffectiveUseJournal::Entry> store;
    EffectiveUseJournal journal;
    explicit JournalHarness(std::size_t n) : store(n), journal(store) {}
};
}

/** @test Every touch is an independent lease; ack releases exactly its own. */
TEST(EffectiveUseOrdering, PerOperationLeasesReleaseIndependently)
{
    JournalHarness harness(4);
    auto &journal = harness.journal;
    EXPECT_TRUE(journal.bound());
    EffectiveUseReceipt touch1{1, 1, 1, 1};
    EffectiveUseReceipt touch2{1, 1, 1, 2};
    EXPECT_EQ(journal.touch(touch1), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch(touch2), EffectiveUseStatus::Accepted);
    EXPECT_TRUE(journal.hasPending());
    // Ack of lease2 never releases lease1 (independent consumers).
    EXPECT_TRUE(journal.acknowledge(touch2));
    EXPECT_FALSE(journal.mayEvict(1));
    EXPECT_TRUE(journal.acknowledge(touch1));
    EXPECT_TRUE(journal.mayEvict(1));
    EXPECT_FALSE(journal.hasPending());
}

/** @test A duplicate ordinal issues no second lease; older ordinals are obsolete. */
TEST(EffectiveUseOrdering, DuplicateAndObsoleteOrdinalsRefuse)
{
    JournalHarness harness(4);
    auto &journal = harness.journal;
    EffectiveUseReceipt touch1{1, 1, 1, 1};
    EXPECT_EQ(journal.touch(touch1), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch(touch1), EffectiveUseStatus::Duplicate);
    EXPECT_EQ(journal.touch({1, 1, 1, 0}), EffectiveUseStatus::Obsolete); // Zero.
    EXPECT_TRUE(journal.acknowledge(touch1));
    // A fresh ordinal after release is a NEW lease; a zero ordinal stays obsolete.
    EXPECT_EQ(journal.touch({1, 1, 1, 3}), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch({1, 1, 1, 0}), EffectiveUseStatus::Obsolete);
}

/** @test Forged full identity refuses; matching identity releases one lease. */
TEST(EffectiveUseOrdering, FullIdentityAuth)
{
    JournalHarness harness(4);
    auto &journal = harness.journal;
    EXPECT_EQ(journal.touch({1, 1, 5, 3}), EffectiveUseStatus::Accepted);
    // Wrong content/table at the same incarnation+ordinal: no release.
    EXPECT_FALSE(journal.acknowledge({2, 1, 5, 3}));
    EXPECT_FALSE(journal.acknowledge({1, 2, 5, 3}));
    EXPECT_FALSE(journal.acknowledge({1, 1, 6, 3}));
    EXPECT_FALSE(journal.mayEvict(5));
    EXPECT_TRUE(journal.acknowledge({1, 1, 5, 3}));
    EXPECT_TRUE(journal.mayEvict(5));
}

/** @test Journal full is typed; live leases block eviction until acknowledged. */
TEST(EffectiveUseOrdering, JournalFullIsTyped)
{
    JournalHarness harness(2);
    auto &journal = harness.journal;
    EXPECT_EQ(journal.touch({1, 1, 0, 1}), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch({1, 1, 1, 2}), EffectiveUseStatus::Accepted);
    EXPECT_EQ(journal.touch({1, 1, 2, 3}), EffectiveUseStatus::JournalFull);
    EXPECT_TRUE(journal.hasPending());
    EXPECT_FALSE(journal.mayEvict(0));
    EXPECT_TRUE(journal.acknowledge({1, 1, 0, 1}));
    EXPECT_TRUE(journal.mayEvict(0));
    EXPECT_FALSE(journal.mayEvict(1)); // Still live.
}

/** @test Unbound/empty journal never reaches storage; rebind is guarded. */
TEST(EffectiveUseOrdering, UnboundAndEmptyJournalRefuseTyped)
{
    EffectiveUseJournal unbound;
    EXPECT_FALSE(unbound.bound());
    EXPECT_EQ(unbound.touch({1, 1, 1, 1}), EffectiveUseStatus::JournalFull);
    EXPECT_FALSE(unbound.acknowledge({1, 1, 1, 1}));
    // An empty extent is unbounded storage: construction binds to nothing and
    // every operation refuses typed (no null-entry access, no capacity-1).
    std::vector<EffectiveUseJournal::Entry> empty_store;
    EffectiveUseJournal empty(empty_store);
    EXPECT_FALSE(empty.bound());
    EXPECT_EQ(empty.capacity(), 0u);
    EXPECT_EQ(empty.touch({1, 1, 1, 1}), EffectiveUseStatus::JournalFull);
    // Rebind while active refuses; rebind while inert binds.
    std::vector<EffectiveUseJournal::Entry> small(2);
    EffectiveUseJournal fresh(small);
    EXPECT_EQ(fresh.touch({1, 1, 1, 1}), EffectiveUseStatus::Accepted);
    std::vector<EffectiveUseJournal::Entry> larger(4);
    EXPECT_FALSE(fresh.rebind(larger)); // Active: forbidden.
    EXPECT_TRUE(fresh.acknowledge({1, 1, 1, 1}));
    EXPECT_TRUE(fresh.rebind(larger));
}

/** @test Many touches over time: leases release and slots reclaim. */
TEST(EffectiveUseOrdering, ManyTouchesOverTimeReclaimAndStayBounded)
{
    JournalHarness harness(2);
    auto &journal = harness.journal;
    std::uint64_t ordinal = 0;
    for (std::size_t generation = 0; generation < 40; ++generation)
    {
        const auto incarnation = static_cast<std::uint64_t>(generation % 2);
        EXPECT_EQ(journal.touch(EffectiveUseReceipt{1, 1, incarnation, ++ordinal}),
            EffectiveUseStatus::Accepted);
        EXPECT_TRUE(journal.acknowledge(EffectiveUseReceipt{1, 1, incarnation, ordinal}));
        EXPECT_TRUE(journal.mayEvict(incarnation));
    }
}
