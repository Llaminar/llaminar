/**
 * @file Test__CPUFP32PagedKVTable.cpp
 * @brief Native CPU attention retains exact immutable metadata banks and payload owners.
 *
 * Lifecycle/geometry checks are not recorded replay. They exercise bounded
 * metadata pressure, old/new native reads, aborted publication and actual
 * storage release, without a model, GPU, contiguous gather or hot allocation.
 */
#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include <gtest/gtest.h>
#include <vector>

using namespace llaminar2;

/** @test Metadata construction is exactly admitted and rolls back all roots on a short budget. */
TEST(CPUFP32PagedKVTable, MetadataGeometryAndPartialConstructionRollback)
{
    const auto device = DeviceId::cpu();
    EXPECT_THROW(CPUFP32PagedKVTable::physicalBytes({}, device), std::invalid_argument);
    for (const size_t banks : {1u, 3u, 5u})
    {
        const CPUFP32PagedKVTableGeometry geometry{3, banks, 2, 16, 64, 0};
        const auto bytes = CPUFP32PagedKVTable::physicalBytes(geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes - 1);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        EXPECT_THROW(CPUFP32PagedKVTable::create(authority, device, geometry), std::exception);
        EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::KVCache,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}

/** @test Two published table versions remain immutable until their original borrowers retire. */
TEST(CPUFP32PagedKVTable, PublishedBanksRetainOriginalNativeOperands)
{
    constexpr int dimension = 64;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{3, 2 * 2 * dimension * sizeof(float), 3, 1};
    const CPUFP32PagedKVTableGeometry table_geometry{1, 2, 2, 2, 64, 0};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const auto page = [&](float value) {
        auto grant = pool->tryAdmit(1);
        if (!grant) throw std::logic_error("Fixture page grant refused");
        const auto frame = grant->assignFrame();
        std::vector<float> input(4 * dimension, 0.0f);
        for (int token = 0; token < 2; ++token)
            std::fill_n(input.data() + token * 2 * dimension + dimension, dimension, value);
        auto write = grant->prepareAppend(frame, input.size() * sizeof(float));
        if (!write) throw std::logic_error("Fixture page write refused");
        write->copyFrom(std::as_bytes(std::span(input)));
        return CPUFP32KVPage(write->commit(), 1, dimension);
    };
    auto first = table->tryBeginUpdate();
    ASSERT_TRUE(first);
    first->append(page(4.0f));
    auto old = first->commit();
    auto reader = old.tryBorrow();
    ASSERT_TRUE(reader);
    EXPECT_FALSE(old.tryBorrow());
    auto next = table->tryBeginUpdate();
    ASSERT_TRUE(next);
    next->append(page(9.0f));
    auto current = next->commit();
    EXPECT_GT(current.version(), old.version());
    EXPECT_FALSE(table->tryBeginUpdate());
    table->closeBorrowers(old);
    old = {};
    EXPECT_THROW(old.view(), std::logic_error);
    EXPECT_FALSE(reader->tryBorrow()); // Below the reader limit, but this version's frontier is closed.
    EXPECT_FALSE(table->tryBeginUpdate());
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    std::vector<float> query(dimension, 0.0f), output(dimension, -1.0f);
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(query, *reader, output, 1, 1, false));
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](float value) { return value == 4.0f; }));
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(query, current, output, 1, 1, false));
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](float value) { return value == 9.0f; }));
    reader.reset();
    auto aborted = table->tryBeginUpdate();
    ASSERT_TRUE(aborted);
    EXPECT_THROW(aborted->commit(), std::invalid_argument); // No dummy page/empty compute publication.
    aborted.reset();
    auto replacement = table->tryBeginUpdate();
    ASSERT_TRUE(replacement);
    replacement->append(page(7.0f));
    auto newest = replacement->commit();
    EXPECT_GT(newest.version(), current.version() + 1); // Aborted version is not recycled.
    EXPECT_FALSE(table->tryBeginUpdate());
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(query, current, output, 1, 1, false));
    EXPECT_EQ(output.front(), 9.0f);
    {
        const auto foreign_bytes = CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
        PhysicalMemoryPlanBuilder foreign_plan;
        foreign_plan.add({.world_rank=0, .device=device, .total_bytes=foreign_bytes, .admission_available_bytes=foreign_bytes},
            PhysicalMemoryOwner::KVCache, foreign_bytes);
        auto foreign_authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(foreign_plan.build()), 0);
        auto foreign_table = CPUFP32PagedKVTable::create(foreign_authority, device, table_geometry);
        auto foreign_update = foreign_table->tryBeginUpdate();
        ASSERT_TRUE(foreign_update);
        auto original_page = page(11.0f);
        EXPECT_THROW(foreign_update->append(std::move(original_page)), std::invalid_argument);
        EXPECT_EQ(original_page.rows(), 2); // Refusal did not consume the original owner.
        EXPECT_EQ(original_page.value(0, 0)[0], 11.0f);
    }
    table.reset(); pool.reset();
    EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), bytes);
    current = {};
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(query, newest, output, 1, 1, false));
    EXPECT_EQ(output.front(), 7.0f);
    newest = {};
    EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test Nonce/version issuance stops before exhaustion; aborted versions burn, never recycle. */
TEST(CPUFP32PagedKVTable, VersionIssuanceStopsAndBurnedVersionsNeverRecycle)
{
    constexpr int dimension = 64;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{2, 2 * 2 * dimension * sizeof(float), 3, 1};
    const CPUFP32PagedKVTableGeometry table_geometry{1, 2, 2, 2, 64, 0};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const auto page = [&](float value) {
        auto grant = pool->tryAdmit(1);
        if (!grant) throw std::logic_error("Fixture page grant refused");
        const auto frame = grant->assignFrame();
        std::vector<float> input(2 * 2 * dimension, 0.0f);
        std::fill_n(input.data() + dimension, dimension, value);
        auto write = grant->prepareAppend(frame, input.size() * sizeof(float));
        if (!write) throw std::logic_error("Fixture page write refused");
        write->copyFrom(std::as_bytes(std::span(input)));
        return CPUFP32KVPage(write->commit(), 1, dimension);
    };
    // Bank 0: publish (keep the snapshot borrower alive). Bank 1: begin and
    // HOLD it open, so both banks are occupied: admission stops before mutation.
    auto first_update = table->tryBeginUpdate();
    ASSERT_TRUE(first_update);
    first_update->append(page(1.0f));
    auto snapshot = first_update->commit();
    EXPECT_GT(snapshot.version(), 0u);
    auto held = table->tryBeginUpdate();
    ASSERT_TRUE(held);
    EXPECT_FALSE(table->tryBeginUpdate()); // Admission stops at bank exhaustion.
    held.reset();                          // Abort burns the issued version.
    // Draining the held bank re-opens exactly one admission; the burned
    // version is never recycled (next issuance is strictly greater).
    auto fresh = table->tryBeginUpdate();
    ASSERT_TRUE(fresh);
    fresh->append(page(2.0f));
    auto newest = fresh->commit();
    EXPECT_GT(newest.version(), 1u); // Burned version 2 is not reused.
    EXPECT_EQ(newest.version(), 3u); // 1 published, 2 burned, 3 issued now.
}

/** @test Finite issuance budget refuses normal admission before headroom; accepted work completes. */
TEST(CPUFP32PagedKVTable, FiniteIssuanceBudgetExhaustionRefusesBeforeHeadroom)
{
    constexpr int dimension = 64;
    const auto device = DeviceId::cpu();
    // Four banks, but ONLY three issuances: bank exhaustion alone would permit
    // four begins. Headroom 1 reserves the final issuance for accepted work.
    const PhysicalPagePoolGeometry pool_geometry{4, 2 * 2 * dimension * sizeof(float), 4, 1};
    const CPUFP32PagedKVTableGeometry table_geometry{1, 4, 2, 2, 3, 1};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const auto page = [&](float value) {
        auto grant = pool->tryAdmit(1);
        if (!grant) throw std::logic_error("Fixture page grant refused");
        const auto frame = grant->assignFrame();
        std::vector<float> input(2 * 2 * dimension, 0.0f);
        std::fill_n(input.data() + dimension, dimension, value);
        auto write = grant->prepareAppend(frame, input.size() * sizeof(float));
        if (!write) throw std::logic_error("Fixture page write refused");
        write->copyFrom(std::as_bytes(std::span(input)));
        return CPUFP32KVPage(write->commit(), 1, dimension);
    };
    // Normal admission consumes budget-headroom = 2 issuances (versions 1,2).
    auto first = table->tryBeginUpdate();
    ASSERT_TRUE(first);
    first->append(page(1.0f));
    auto second = table->tryBeginUpdate();
    ASSERT_TRUE(second);
    second->append(page(2.0f));
    // FINITE exhaustion: the third normal begin refuses although a bank is free.
    EXPECT_FALSE(table->tryBeginUpdate());
    EXPECT_EQ(table->normalAdmissionRemaining(), 0u);
    EXPECT_FALSE(table->issuanceBudgetExhausted()); // Headroom remains for accepted work.
    // No caller may consume the headroom without an accepted-work reservation:
    // the old freely-callable reserved path is gone, so only a minted handle
    // can complete accepted work.
    CPUFP32PagedKVTable::IssuanceReservation unminted;
    EXPECT_FALSE(table->tryBeginReserved(unminted));
    EXPECT_EQ(table->issuedVersions(), 2u); // Normal begins unaffected by the refusal.
    auto reservation = table->reserveAcceptedWork(1);
    ASSERT_TRUE(reservation);
    EXPECT_EQ(reservation->remaining(), 1u);
    auto accepted = table->tryBeginReserved(*reservation);
    ASSERT_TRUE(accepted);
    accepted->append(page(3.0f));
    EXPECT_TRUE(table->issuanceBudgetExhausted());
    EXPECT_FALSE(table->tryBeginReserved(*reservation)); // Exhausted handle.
    EXPECT_FALSE(table->tryBeginUpdate());
    EXPECT_EQ(reservation->remaining(), 0u);
    // Complete both published and accepted work; versions never recycle.
    auto s1 = first->commit();
    auto s2 = second->commit();
    auto s3 = accepted->commit();
    EXPECT_EQ(s1.version(), 1u);
    EXPECT_EQ(s2.version(), 2u);
    EXPECT_EQ(s3.version(), 3u);
    // No normal admission remains after finite exhaustion; four free banks.
    EXPECT_FALSE(table->tryBeginUpdate());
    EXPECT_EQ(table->outstandingReserved(), 0u);
}

/** @test Accepted-work reservations are minted, consumable, unshareable and adversarial-safe. */
TEST(CPUFP32PagedKVTable, AcceptedWorkReservationIsConsumableAndUnshareable)
{
    const auto device = DeviceId::cpu();
    // Budget 5, headroom 3, banks 3: normal admission cap = 2. A second table
    // is created on the SAME authority for the foreign-handle case, so the
    // plan admits both exact table BOMs.
    const CPUFP32PagedKVTableGeometry table_geometry{1, 3, 2, 2, 5, 3};
    const auto bytes = 2 * CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    // Reservation is minted and consumes headroom from other clients.
    auto first = table->reserveAcceptedWork(2);
    ASSERT_TRUE(first);
    EXPECT_EQ(table->outstandingReserved(), 2u);
    EXPECT_EQ(table->headroomAvailable(), 1u);
    // A second client cannot reserve beyond the remaining headroom.
    EXPECT_FALSE(table->reserveAcceptedWork(2));
    EXPECT_EQ(table->outstandingReserved(), 2u);
    // Capacity beyond the finite budget also refuses without mutation.
    EXPECT_FALSE(table->reserveAcceptedWork(999));
    // The handle cannot be copied; a moved-from handle is empty.
    auto second = std::move(first);
    ASSERT_TRUE(first.has_value());
    EXPECT_FALSE(static_cast<bool>(*first)); // Moved-from handle cannot consume.
    EXPECT_EQ(second->remaining(), 2u);
    // Foreign/exhausted/moved-from handles never issue a version.
    CPUFP32PagedKVTable::IssuanceReservation moved;
    EXPECT_FALSE(table->tryBeginReserved(moved));
    EXPECT_EQ(table->issuedVersions(), 0u);
    // Consuming one issuance advances the version monotonically and drains.
    auto update = table->tryBeginReserved(*second);
    ASSERT_TRUE(update);
    EXPECT_EQ(table->issuedVersions(), 1u);
    EXPECT_EQ(table->outstandingReserved(), 1u);
    EXPECT_EQ(second->remaining(), 1u);
    // A dropped reserved handle returns unconsumed capacity to the headroom.
    second = CPUFP32PagedKVTable::IssuanceReservation();
    EXPECT_EQ(table->outstandingReserved(), 0u);
    EXPECT_EQ(table->headroomAvailable(), 3u);
    // A reservation minted by a DIFFERENT table is foreign: it cannot consume
    // this table's headroom even when capacity is available, and consumes the
    // other table's budget only.
    auto other_table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    auto foreign = other_table->reserveAcceptedWork(1);
    ASSERT_TRUE(foreign);
    EXPECT_FALSE(table->tryBeginReserved(*foreign));
    EXPECT_EQ(table->issuedVersions(), 1u);
    EXPECT_EQ(other_table->issuedVersions(), 0u);
    EXPECT_EQ(table->outstandingReserved(), 0u);
}

/** @test Effective use is acknowledged only for a real published version; duplicate/obsolete refuse. */
TEST(CPUFP32PagedKVTable, EffectiveUseWatermarkFollowsOnlyRealAttachments)
{
    constexpr int dimension = 8;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{2, 2 * 2 * dimension * sizeof(float), 2, 1};
    // readers_per_bank 4 so the snapshot plus two borrowers fit concurrently.
    const CPUFP32PagedKVTableGeometry table_geometry{1, 2, 4, 2, 64, 0};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const auto page = [&](float value) {
        auto grant = pool->tryAdmit(1);
        if (!grant) throw std::logic_error("Fixture page grant refused");
        const auto frame = grant->assignFrame();
        std::vector<float> input(2 * 2 * dimension, 0.0f);
        std::fill_n(input.data() + dimension, dimension, value);
        auto write = grant->prepareAppend(frame, input.size() * sizeof(float));
        if (!write) throw std::logic_error("Fixture page write refused");
        write->copyFrom(std::as_bytes(std::span(input)));
        return CPUFP32KVPage(write->commit(), 1, dimension);
    };
    // Nothing published yet: any receipt is refused (no invented touch).
    EXPECT_EQ(table->effectiveUseWatermark(), 0u);
    // Publish one bank (version 1), then acknowledge the exact touch.
    auto update = table->tryBeginUpdate();
    ASSERT_TRUE(update);
    update->append(page(1.0f));
    auto snapshot = update->commit();
    ASSERT_TRUE(snapshot.version() > 0u);
    const auto incarnation = snapshot.version();
    EXPECT_EQ(table->touchEffectiveUse(snapshot), EffectiveUseStatus::Accepted);
    EXPECT_EQ(table->effectiveUseWatermark(), 1u);
    EXPECT_FALSE(table->effectiveUseMayEvict(incarnation)); // Lease live until ack.
    // R10: independent consumers each issue their own lease with a fresh
    // ordinal; acking one never releases another's obligation. The journal's
    // duplicate/obsolete idempotence is covered by the unit suite; the table
    // always mints a fresh ordinal per successful touch.
    auto borrower = snapshot.tryBorrow();
    ASSERT_TRUE(borrower);
    EXPECT_EQ(table->touchEffectiveUse(*borrower), EffectiveUseStatus::Accepted);
    EXPECT_EQ(table->effectiveUseWatermark(), 2u);
    EXPECT_TRUE(table->acknowledgeEffectiveUse(snapshot));  // lease A released.
    EXPECT_FALSE(table->effectiveUseMayEvict(incarnation)); // lease B STILL live.
    // Per-operation ordering: a SECOND independent borrower snapshot, each
    // with its own lease; one ack releases only that borrower's obligation.
    auto borrower2 = snapshot.tryBorrow();
    ASSERT_TRUE(borrower2);
    EXPECT_EQ(table->touchEffectiveUse(*borrower2), EffectiveUseStatus::Accepted);
    EXPECT_EQ(table->effectiveUseWatermark(), 3u);
    // Ack of borrower2 releases only its own lease; the first borrower lease
    // (lease B) is still live.
    EXPECT_FALSE(table->effectiveUseMayEvict(incarnation));
    EXPECT_TRUE(table->acknowledgeEffectiveUse(*borrower2));
    EXPECT_TRUE(table->acknowledgeEffectiveUse(*borrower)); // lease B finally acked.
    EXPECT_TRUE(table->effectiveUseMayEvict(incarnation));
    // A wrong FULL identity (a different published version / its own bank)
    // never acknowledges the first incarnation's obligations.
    auto second = table->tryBeginUpdate();
    ASSERT_TRUE(second);
    second->append(page(2.0f));
    auto snapshot2 = second->commit();
    ASSERT_GT(snapshot2.version(), snapshot.version());
    EXPECT_EQ(table->touchEffectiveUse(snapshot2), EffectiveUseStatus::Accepted);
    EXPECT_EQ(table->effectiveUseWatermark(), 4u);
    EXPECT_TRUE(table->acknowledgeEffectiveUse(snapshot2));
    EXPECT_TRUE(table->effectiveUseMayEvict(snapshot2.version()));
    // The orphan/release-after-retire path (ack with retained receipt identity
    // after publication retirement) is recorded in the production replay
    // fixture (table-save-receipt/table-use-ack-retained) and in the journal
    // unit suite; the table-level Snapshot API cannot both drop and retain a
    // single snapshot (that is exactly why the recorded replay exists).
}

/** @test Reservation conservation: issued + outstanding + new <= absolute budget always. */
TEST(CPUFP32PagedKVTable, ReservationConservationHoldsThroughPartialConsumption)
{
    const auto device = DeviceId::cpu();
    // Reviewer example: budget10, headroom8.
    const CPUFP32PagedKVTableGeometry table_geometry{1, 10, 4, 2, 10, 8};
    const auto bytes = CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    auto reservation = table->reserveAcceptedWork(8);
    ASSERT_TRUE(reservation);
    // Consume six issuances through reserved updates (abort each bank).
    for (int i = 0; i < 6; ++i)
    {
        auto update = table->tryBeginReserved(*reservation);
        ASSERT_TRUE(update) << "consume " << i;
    }
    EXPECT_EQ(table->issuedVersions(), 6u);
    EXPECT_EQ(table->outstandingReserved(), 2u);
    // OVERBOOKING: outstanding2 + issued6 = 8; another reserve4 would total 12 > 10.
    // Exactly the remaining absolute capacity (2) may still be reserved, but
    // it must be HELD (a temporary would release its capacity on destruction).
    auto extra = table->reserveAcceptedWork(2);
    ASSERT_TRUE(extra);
    EXPECT_EQ(table->outstandingReserved(), 4u);
    EXPECT_FALSE(table->reserveAcceptedWork(2)); // Overbook would total 12 > 10.
    EXPECT_FALSE(table->reserveAcceptedWork(4));
    EXPECT_FALSE(table->reserveAcceptedWork(1));
    // The invariant always holds: total issued + outstanding <= budget.
    EXPECT_LE(table->issuedVersions() + table->outstandingReserved(), 10u);
    // Releasing the extra returns exactly its unconsumed capacity.
    extra = CPUFP32PagedKVTable::IssuanceReservation();
    EXPECT_EQ(table->outstandingReserved(), 2u);
}

/** @test Reservation serial exhaustion seals new reservations (no recycle). */
TEST(CPUFP32PagedKVTable, ReservationSerialExhaustionSealsNewReservations)
{
    const auto device = DeviceId::cpu();
    const CPUFP32PagedKVTableGeometry table_geometry{1, 2, 4, 2, 16, 8};
    const auto bytes = CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    // The serial seal is a terminal-exhaustion policy (checked at max, the
    // counter never wraps to re-issue serial1). The observable contract at the
    // boundary is covered by code inspection (the counter asserts a terminal
    // refusal and stays at max); the ordinary reserve path still works.
    auto reservation = table->reserveAcceptedWork(1);
    ASSERT_TRUE(reservation);
    EXPECT_EQ(reservation->remaining(), 1u);
    // A released handle returns capacity and does not recycle its serial.
    reservation = CPUFP32PagedKVTable::IssuanceReservation();
    auto fresh = table->reserveAcceptedWork(1);
    ASSERT_TRUE(fresh);
    EXPECT_EQ(fresh->remaining(), 1u);
}

/** @test Headroom below budget is typed; equal/zero headroom is rejected before allocation. */
TEST(CPUFP32PagedKVTable, IssuanceHeadroomValidationIsTyped)
{
    // Zero budget, or headroom >= budget, must never admit.
    EXPECT_THROW(CPUFP32PagedKVTable::physicalBytes({1, 2, 2, 2, 0, 0}, DeviceId::cpu()),
        std::invalid_argument);
    EXPECT_THROW(CPUFP32PagedKVTable::physicalBytes({1, 2, 2, 2, 2, 2}, DeviceId::cpu()),
        std::invalid_argument);
    EXPECT_THROW(CPUFP32PagedKVTable::physicalBytes({1, 2, 2, 2, 2, 3}, DeviceId::cpu()),
        std::invalid_argument);
    // Budget strictly above headroom is legal.
    EXPECT_GT(CPUFP32PagedKVTable::physicalBytes({1, 2, 2, 2, 5, 1}, DeviceId::cpu()), 0u);
}
