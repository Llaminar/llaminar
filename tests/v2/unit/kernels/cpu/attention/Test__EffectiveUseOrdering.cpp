/**
 * @file Test__EffectiveUseOrdering.cpp
 * @brief Per-operation lease + retained replica-record contracts (R11 model).
 */
#include "kernels/cpu/attention/EffectiveUseOrdering.h"
#include <gtest/gtest.h>
#include <vector>

using namespace llaminar2;
namespace
{
struct Harness
{
    std::vector<EffectiveUseJournal::LeaseCell> cells;
    std::vector<EffectiveUseJournal::ReplicaRecord> records;
    EffectiveUseJournal journal;
    Harness(std::size_t lease_cells, std::size_t replica_records)
        : cells(lease_cells), records(replica_records), journal(cells, records) {}
};
EffectiveUseReceipt receipt(std::uint64_t incarnation, std::uint64_t sequence)
{
    return EffectiveUseReceipt{1, 1, incarnation, sequence};
}
}

/** @test Every touch mints its own lease; acks release exactly that lease. */
TEST(EffectiveUseOrdering, PerOperationLeasesReleaseIndependently)
{
    Harness h(4, 2);
    auto lease1 = h.journal.touch(receipt(1, 1));
    ASSERT_TRUE(lease1);
    auto lease2 = h.journal.touch(receipt(1, 2));
    ASSERT_TRUE(lease2);
    EXPECT_TRUE(h.journal.hasPending());
    // Ack of lease2 never releases lease1 (independent consumers).
    EXPECT_TRUE(lease2->acknowledge());
    EXPECT_FALSE(lease2->acknowledge()); // Idempotent refusal.
    EXPECT_FALSE(h.journal.mayEvict(1));
    EXPECT_TRUE(lease1->acknowledge());
    EXPECT_TRUE(h.journal.mayEvict(1));
    EXPECT_FALSE(h.journal.hasPending());
}

/** @test Rollback-free refusal: duplicate/obsolete/full preserve prior state. */
TEST(EffectiveUseOrdering, RefusalsAreRollbackFree)
{
    Harness h(1, 2); // One lease cell.
    auto lease1 = h.journal.touch(receipt(1, 1));
    ASSERT_TRUE(lease1);
    // Duplicate ordinal refuses without mutation.
    EXPECT_FALSE(h.journal.touch(receipt(1, 1)));
    EXPECT_FALSE(h.journal.touch(receipt(1, 0))); // Zero ordinal (sealed).
    // Full capacity: the SECOND live lease refuses; first lease still live.
    auto lease2 = h.journal.touch(receipt(2, 2));
    EXPECT_FALSE(lease2.has_value());
    EXPECT_TRUE(lease1.has_value());
    EXPECT_TRUE(lease1->acknowledge());
    EXPECT_TRUE(h.journal.mayEvict(1));
}

/** @test Full-identity auth: foreign content/table never acknowledges. */
TEST(EffectiveUseOrdering, FullIdentityAuth)
{
    Harness h(4, 4);
    auto lease = h.journal.touch(EffectiveUseReceipt{1, 1, 5, 3});
    ASSERT_TRUE(lease);
    EffectiveUseLease forged;
    EXPECT_FALSE(forged.acknowledge()); // Empty lease.
    EXPECT_FALSE(h.journal.acknowledge(EffectiveUseLease())); // Not via journal.
    EXPECT_TRUE(lease->acknowledge());
    EXPECT_TRUE(h.journal.mayEvict(5));
}

/** @test Retained replica record: cell reuse never erases identity. */
TEST(EffectiveUseOrdering, CellReuseKeepsReplicaIdentity)
{
    Harness h(1, 2); // One lease cell, two replica records.
    auto a = h.journal.touch(receipt(1, 1));
    ASSERT_TRUE(a);
    EXPECT_TRUE(a->acknowledge());
    EXPECT_TRUE(h.journal.mayEvict(1));
    // B reuses the cell; A's retained record stays evictable (independent).
    auto b = h.journal.touch(receipt(2, 2));
    ASSERT_TRUE(b);
    EXPECT_TRUE(h.journal.mayEvict(1));   // A still evictable.
    EXPECT_FALSE(h.journal.mayEvict(2));  // B pending.
    EXPECT_TRUE(b->acknowledge());
    EXPECT_TRUE(h.journal.mayEvict(2));
    // REPLAYING A's old receipt after B released the cell: refused (obsolete
    // ordinal against A's retained auth, no new lease, no watermark bump).
    auto replay = h.journal.touch(receipt(1, 1));
    EXPECT_FALSE(replay.has_value());
    EXPECT_EQ(h.journal.watermark(), 2u);
}

/** @test Explicit retirement refuses while any touch is unacknowledged. */
TEST(EffectiveUseOrdering, RetirementRefusesWhileLive)
{
    Harness h(4, 2);
    auto lease = h.journal.touch(receipt(1, 1));
    ASSERT_TRUE(lease);
    EXPECT_FALSE(h.journal.retireReplica(1)); // Live obligation: refuse.
    EXPECT_TRUE(lease->acknowledge());
    EXPECT_TRUE(h.journal.retireReplica(1));  // Fully acked: retire.
    EXPECT_FALSE(h.journal.retireReplica(9)); // Untracked: idempotent refuse.
}

/** @test Many touches over time: leases release, records retain, bounded. */
TEST(EffectiveUseOrdering, ManyTouchesOverTimeReclaimAndStayBounded)
{
    Harness h(2, 2);
    std::uint64_t ordinal = 0;
    for (std::size_t generation = 0; generation < 40; ++generation)
    {
        const auto incarnation = static_cast<std::uint64_t>(generation % 2);
        auto lease = h.journal.touch(receipt(incarnation, ++ordinal));
        ASSERT_TRUE(lease) << "generation " << generation;
        EXPECT_TRUE(lease->acknowledge());
        EXPECT_TRUE(h.journal.mayEvict(incarnation));
    }
}
