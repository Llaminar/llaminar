/**
 * @file Test__PageTableAdmission.cpp
 * @brief All-class table+pool atomic admission contracts (plan P1).
 *
 * Constructed lifecycle tests. They verify that a combined pool growth +
 * table bank admission either returns both cells or refuses without leaving a
 * usable partial reservation, across exact fit, exhaustion, aborted update and
 * foreign-authority rejection.
 */
#include "planning/PageTableAdmission.h"
#include <gtest/gtest.h>
#include <vector>

using namespace llaminar2;
namespace
{
/** @brief One exact PMA authority with a small pool + table. */
struct CombinedFixture
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::shared_ptr<PhysicalPagePool> pool;
    std::shared_ptr<CPUFP32PagedKVTable> table;
    CombinedFixture(size_t frames = 3, size_t banks = 2)
    {
        const auto device = DeviceId::cpu();
        const PhysicalPagePoolGeometry pool_geometry{frames, 64, 4, frames};
        const CPUFP32PagedKVTableGeometry table_geometry{1, banks, 2, 1, 64, 0};
        const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
            CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
        table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    }
};
}

/** @test Combined admission returns both cells; refusal before pool exhaustion rolls back. */
TEST(PageTableAdmission, CombinedAdmissionReturnsBothOrRollsBack)
{
    CombinedFixture fx;
    auto admitted = admitPoolAndTable(*fx.pool, *fx.table, 2);
    ASSERT_TRUE(admitted);
    EXPECT_EQ(admitted->grant.remainingCommitment(), 2u);
    // Attach one page and publish through the reserved bank.
    std::vector<float> rows(2 * 4, 1.0f);
    auto frame = admitted->grant.assignFrame();
    auto plan = admitted->grant.prepareFreshWrite(frame, rows.size() * sizeof(float));
    ASSERT_TRUE(plan);
    plan->copyFrom(std::as_bytes(std::span(rows)));
    admitted->table_update.append(CPUFP32KVPage(plan->commit(), 1, 4));
    auto snapshot = admitted->table_update.commit();
    EXPECT_NO_THROW(snapshot.view());
}

/** @test Pool exhaustion refuses before the table reservation is usable. */
TEST(PageTableAdmission, PoolExhaustionRollsBackTableReservation)
{
    CombinedFixture fx(/*frames=*/1, /*banks=*/2);
    // Consume the only frame.
    auto holder = fx.pool->tryAdmit(1);
    ASSERT_TRUE(holder);
    (void)holder->assignFrame();
    // Combined admission must refuse; no table bank may remain reserved.
    auto refused = admitPoolAndTable(*fx.pool, *fx.table, 1);
    EXPECT_FALSE(refused);
    // The table must still offer a fresh bank afterwards.
    auto update = fx.table->tryBeginUpdate();
    ASSERT_TRUE(update);
}

/** @test Table-bank exhaustion refuses before the pool grant is usable. */
TEST(PageTableAdmission, TableExhaustionRollsBackPoolGrant)
{
    CombinedFixture fx(3, /*banks=*/1);
    auto first = admitPoolAndTable(*fx.pool, *fx.table, 1);
    ASSERT_TRUE(first);
    // Holding the only bank; a second combined admission must refuse and roll
    // back the pool grant (no leaked protection).
    auto second = admitPoolAndTable(*fx.pool, *fx.table, 1);
    EXPECT_FALSE(second);
    // No protected growth leaked by the refusal: one frame is already held by
    // the first admission, so exactly two remain available.
    auto check = fx.pool->tryAdmit(2);
    ASSERT_TRUE(check);
    EXPECT_EQ(check->remainingCommitment(), 2u);
}

/** @test Foreign-authority combination is rejected before acquiring anything. */
TEST(PageTableAdmission, ForeignAuthorityIsRejectedBeforeAcquisition)
{
    CombinedFixture fx;
    CombinedFixture foreign;
    EXPECT_THROW(admitPoolAndTable(*fx.pool, *foreign.table, 1), std::invalid_argument);
    auto check = fx.pool->tryAdmit(3);
    ASSERT_TRUE(check);
    EXPECT_EQ(check->remainingCommitment(), 3u);
}

/** @test Aborted combined admission keeps both classes available for the next attempt. */
TEST(PageTableAdmission, AbortedCombinedAdmissionLeavesBothClassesAvailable)
{
    CombinedFixture fx;
    {
        auto admitted = admitPoolAndTable(*fx.pool, *fx.table, 1);
        ASSERT_TRUE(admitted);
        // Abort both cells: grant + reserved table bank roll back by RAII.
    }
    auto again = admitPoolAndTable(*fx.pool, *fx.table, 1);
    ASSERT_TRUE(again);
    EXPECT_EQ(again->grant.remainingCommitment(), 1u);
}

/** @test Witnessed admission proves completion order before acquiring anything. */
TEST(PageTableAdmission, WitnessedAdmissionProvesBeforeAcquisition)
{
    // 2 frames, 2 banks, budget 8/headroom 4: one honored request fits.
    CombinedFixture fx(/*frames=*/2, /*banks=*/2);
    fx.table.reset();
    fx.table = CPUFP32PagedKVTable::create(fx.authority, DeviceId::cpu(),
        CPUFP32PagedKVTableGeometry{1, 2, 2, 1, 8, 4});
    PageTableAdmissionWorkspace workspace(fx.pool->geometry().request_records + 1);
    auto admission = fx.table->reserveAcceptedWork(1);
    ASSERT_TRUE(admission);
    // Peak 2 / initial 2 against 2 frames: the proof holds and both cells
    // are acquired, consuming exactly one reserved issuance.
    auto admitted = admitPoolTableWitnessed(*fx.pool, *fx.table, workspace, 2, 2, *admission);
    ASSERT_TRUE(admitted);
    EXPECT_EQ(admitted->grant.remainingCommitment(), 2u);
    EXPECT_EQ(admission->remaining(), 0u);
    // Now the pool is full: a second witnessed request with a fresh
    // reservation must be REFUSED by the proof (no completion order for a
    // second 2-frame peak on 2 frames), leaving the second reservation
    // unconsumed and the table bank available.
    auto second = fx.table->reserveAcceptedWork(1);
    ASSERT_TRUE(second);
    EXPECT_FALSE(admitPoolTableWitnessed(*fx.pool, *fx.table, workspace, 2, 2, *second));
    EXPECT_EQ(second->remaining(), 1u);                 // No issuance consumed.
    EXPECT_EQ(fx.table->outstandingReserved(), 1u);     // Still outstanding.
    EXPECT_EQ(fx.table->issuedVersions(), 1u);          // Only the first issued.
    // A witnessed request whose initial grant fits the remaining frame is
    // also refused by the witness (peak 2 on full pool) without mutation.
    EXPECT_FALSE(admitPoolTableWitnessed(*fx.pool, *fx.table, workspace, 1, 1, *second));
    EXPECT_EQ(second->remaining(), 1u);
}

/** @test B4: observed live class demand is part of the witness (no hard-coded per-request 1). */
TEST(PageTableAdmission, WitnessedAdmissionCountsObservedLiveClassDemand)
{
    CombinedFixture fx(/*frames=*/3, /*banks=*/2);
    fx.table.reset(); // Rebuild the table with a reservation-capable geometry.
    fx.table = CPUFP32PagedKVTable::create(fx.authority, DeviceId::cpu(),
        CPUFP32PagedKVTableGeometry{1, 2, 2, 1, 8, 4});
    PageTableAdmissionWorkspace workspace(fx.pool->geometry().request_records + 1);
    auto reservation = fx.table->reserveAcceptedWork(1);
    ASSERT_TRUE(reservation);
    // Publish one bank (live demand: 1 published bank, 1 live lease after a touch).
    auto update = fx.table->tryBeginReserved(*reservation);
    ASSERT_TRUE(update);
    auto grant = fx.pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    const auto frame = grant->assignFrame();
    std::vector<float> rows(2 * 4, 1.0f);
    auto write = grant->prepareAppend(frame, rows.size() * sizeof(float));
    ASSERT_TRUE(write);
    write->copyFrom(std::as_bytes(std::span(rows)));
    update->append(CPUFP32KVPage(write->commit(), 1, 4));
    auto snapshot = update->commit();
    auto lease = fx.table->touchEffectiveUse(snapshot);
    ASSERT_TRUE(lease.has_value());
    // Observe the real demand: 1 published bank + 1 live lease must be counted
    // before the proposed request's own peak is added.
    const auto observed = fx.table->liveClassDemand();
    EXPECT_EQ(observed.published_banks, 1u);
    EXPECT_EQ(observed.live_leases, 1u);
    // The composed witness uses the observation: with 2 banks and 1 observed,
    // one more proposed request fits; a third would exceed the bank class.
    auto second = fx.table->reserveAcceptedWork(1);
    ASSERT_TRUE(second);
    auto admitted = admitPoolTableWitnessed(*fx.pool, *fx.table, workspace, 1, 1, *second);
    EXPECT_TRUE(admitted) << "observed live bank demand should allow one more";
    EXPECT_TRUE(lease->acknowledge());
}

/** @test Reserved all-resource admission consumes exactly one accepted-work issuance. */
TEST(PageTableAdmission, ReservedAdmissionConsumesOneIssuanceAndRollsBackTogether)
{
    // Budget 8 with headroom 4: the reserved path is usable alongside normal.
    CombinedFixture fx(/*frames=*/2, /*banks=*/2);
    fx.table.reset(); // Rebuild the table with a reservation-capable geometry.
    fx.table = CPUFP32PagedKVTable::create(fx.authority, DeviceId::cpu(),
        CPUFP32PagedKVTableGeometry{1, 2, 2, 1, 8, 4});
    auto reservation = fx.table->reserveAcceptedWork(1);
    ASSERT_TRUE(reservation);
    EXPECT_EQ(fx.table->outstandingReserved(), 1u);
    auto admitted = admitPoolTableReserved(*fx.pool, *fx.table, 1, *reservation);
    ASSERT_TRUE(admitted);
    EXPECT_EQ(admitted->grant.remainingCommitment(), 1u);
    EXPECT_EQ(reservation->remaining(), 0u);
    EXPECT_EQ(fx.table->outstandingReserved(), 0u);
    EXPECT_EQ(fx.table->issuedVersions(), 1u);
    // The accepted update is independently usable: attach + commit.
    std::vector<float> rows(2 * 4, 2.0f);
    auto frame = admitted->grant.assignFrame();
    auto plan = admitted->grant.prepareFreshWrite(frame, rows.size() * sizeof(float));
    ASSERT_TRUE(plan);
    plan->copyFrom(std::as_bytes(std::span(rows)));
    admitted->table_update.append(CPUFP32KVPage(plan->commit(), 1, 4));
    EXPECT_NO_THROW(admitted->table_update.commit());
}

/** @test Foreign/exhausted reservation refuses before any class is acquired. */
TEST(PageTableAdmission, ReservedAdmissionRefusesForeignOrExhaustedHandles)
{
    CombinedFixture fx;
    fx.table.reset();
    fx.table = CPUFP32PagedKVTable::create(fx.authority, DeviceId::cpu(),
        CPUFP32PagedKVTableGeometry{1, 2, 2, 1, 8, 4});
    CombinedFixture foreign;
    foreign.table.reset();
    foreign.table = CPUFP32PagedKVTable::create(foreign.authority, DeviceId::cpu(),
        CPUFP32PagedKVTableGeometry{1, 2, 2, 1, 8, 4});
    auto foreign_reservation = foreign.table->reserveAcceptedWork(1);
    ASSERT_TRUE(foreign_reservation);
    // Foreign handle: same pool, but the handle was minted by another table.
    EXPECT_FALSE(admitPoolTableReserved(*fx.pool, *fx.table, 1, *foreign_reservation));
    EXPECT_EQ(fx.pool->tryAdmit(3)->remainingCommitment(), 3u); // No growth leaked.
    EXPECT_EQ(foreign.table->outstandingReserved(), 1u);        // No capacity consumed.
    // Exhausted handle: consume it first, then admission must refuse upfront.
    auto own = fx.table->reserveAcceptedWork(1);
    ASSERT_TRUE(own);
    auto first = admitPoolTableReserved(*fx.pool, *fx.table, 1, *own);
    ASSERT_TRUE(first);
    EXPECT_EQ(first->grant.remainingCommitment(), 1u);
    auto after_pool = fx.pool->tryAdmit(2);
    ASSERT_TRUE(after_pool); // Two frames remain (3-frames fixture).
    EXPECT_FALSE(admitPoolTableReserved(*fx.pool, *fx.table, 1, *own));
    EXPECT_EQ(after_pool->remainingCommitment(), 2u); // Untouched by the refusal.
}
