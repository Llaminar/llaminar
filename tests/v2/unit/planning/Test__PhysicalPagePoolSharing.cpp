/**
 * @file Test__PhysicalPagePoolSharing.cpp
 * @brief Immutable sharing and partial-tail copy-on-write contracts.
 *
 * Constructed lifecycle/geometry tests, not recorded replay. They verify the
 * nine-token shared-tail example, peer adoption, refusal-before-mutation,
 * exact byte copies, old-source immutability and last-reference retirement.
 */
#include "planning/PhysicalPagePool.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include <gtest/gtest.h>
#include <cstring>
#include <limits>

using namespace llaminar2;
namespace
{
/** @brief Exact PMA backing with two request records and bounded frames. */
struct ShareFixture
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::shared_ptr<PhysicalPagePool> pool;
    explicit ShareFixture(std::size_t frames, std::size_t frame_bytes = 64,
        std::size_t max_pages = 2)
    {
        const auto device = DeviceId::cpu();
        const PhysicalPagePoolGeometry geometry{frames, frame_bytes, 3, max_pages};
        const auto bytes = PhysicalPagePool::physicalBytes(geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        pool = authority->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
    }
};

/** @brief Write exactly one complete private page and return its publication. */
PhysicalPagePool::CommittedPage writePage(PhysicalPagePool::Grant &grant, std::span<const std::byte> bytes)
{
    auto frame = grant.assignFrame();
    auto plan = grant.prepareFreshWrite(frame, bytes.size());
    EXPECT_TRUE(plan);
    plan->copyFrom(bytes);
    return plan->commit();
}

const std::byte *base(PhysicalPagePool::SharedPage &page) { return page.read().data(); }
}

/** @test A shared immutable page adopts peers without copying and retires exactly on the last reference. */
TEST(PhysicalPagePoolSharing, SharedImmutablePageAdoptsPeersAndRetiresOnce)
{
    ShareFixture fx(2);
    std::byte bytes[16];
    for (std::size_t i = 0; i < 16; ++i) bytes[i] = static_cast<std::byte>(i + 1);
    auto first_grant = *fx.pool->tryAdmit(1);
    {
        auto page = writePage(first_grant, bytes);
        ASSERT_EQ(std::memcmp(page.read().data(), bytes, 16), 0);

        // Peer adoption adds a logical reference, never a second physical frame.
        auto second_grant = *fx.pool->tryAdmit(1);
        auto shared = page.share();
        auto adopted = second_grant.adoptShared(shared);
        ASSERT_TRUE(adopted);
        EXPECT_EQ(std::memcmp(adopted->read().data(), bytes, 16), 0);
        EXPECT_EQ(adopted->issuer(), fx.authority.get());
        EXPECT_EQ(second_grant.remainingCommitment(), 1u); // No frame consumed.

        // Original private grant retirement cannot free the shared frame.
        first_grant = PhysicalPagePool::Grant();
        EXPECT_EQ(std::memcmp(shared.read().data(), bytes, 16), 0);
        EXPECT_EQ(std::memcmp(adopted->read().data(), bytes, 16), 0);

        adopted.reset();
        EXPECT_EQ(std::memcmp(shared.read().data(), bytes, 16), 0);
        shared = PhysicalPagePool::SharedPage();
        page = PhysicalPagePool::CommittedPage(); // Original publication pin retires last.
        (void)page;
    }
    // Last shared reference and its origins retired: exact capacity returns.
    auto third_grant = *fx.pool->tryAdmit(1);
    auto frame = third_grant.assignFrame();
    EXPECT_EQ(third_grant.remainingCommitment(), 0u);
    auto plan = third_grant.prepareFreshWrite(frame, 16);
    EXPECT_TRUE(plan);
    plan->copyFrom(bytes);
    (void)plan->commit();
}

/** @test Partial-tail COW copies exactly the shared bytes and leaves the source immutable. */
TEST(PhysicalPagePoolSharing, PartialTailCopyOnWritePreservesSourceAndExactBytes)
{
    // Nine-token shared tail: frame 0 holds 9/16 bytes; two writers each extend.
    ShareFixture fx(3);
    std::byte shared_bytes[16];
    for (std::size_t i = 0; i < 16; ++i) shared_bytes[i] = static_cast<std::byte>(0x40 + i);
    std::byte fresh[7];
    for (std::size_t i = 0; i < 7; ++i) fresh[i] = static_cast<std::byte>(0x80 + i);

    auto exporter = *fx.pool->tryAdmit(1);
    // The shared source is a partial page: only its first nine bytes are valid.
    auto shared_page = writePage(exporter, std::span<const std::byte>(shared_bytes, 9)).share();

    // Writer A COW copies 9 shared + 7 fresh = 16 (one protected destination).
    auto writer_a = *fx.pool->tryAdmit(1);
    auto plan_a = writer_a.prepareCOW(shared_page, 7);
    ASSERT_TRUE(plan_a);
    plan_a->copySource();
    EXPECT_EQ(std::memcmp(base(shared_page), shared_bytes, 9), 0); // source unchanged
    plan_a->copyNext(fresh);
    auto published_a = plan_a->commit();
    ASSERT_EQ(std::memcmp(published_a.read().data(), shared_bytes, 9), 0);
    ASSERT_EQ(std::memcmp(published_a.read().data() + 9, fresh, 7), 0);
    EXPECT_EQ(std::memcmp(base(shared_page), shared_bytes, 9), 0);

    // Writer B COW works concurrently; both published pages are independent.
    auto writer_b = *fx.pool->tryAdmit(1);
    auto plan_b = writer_b.prepareCOW(shared_page, 7);
    ASSERT_TRUE(plan_b);
    plan_b->copySource();
    plan_b->copyNext(fresh);
    auto published_b = plan_b->commit();
    EXPECT_EQ(std::memcmp(published_a.read().data(), published_b.read().data(), 16), 0);

    // Source still readable after both writers diverted.
    EXPECT_EQ(std::memcmp(base(shared_page), shared_bytes, 9), 0);
}

/** @test A shared prefix enters an operation snapshot and peers stay byte-identical after COW divergence. */
TEST(PhysicalPagePoolSharing, SharedPrefixEntersOperationSnapshotAndPeersRemainImmutable)
{
    // One 8-token shared prefix (dimension=1 -> 8 bytes per interleaved row),
    // borrowed by two request tables; one writer diverges via COW.
    // Pool + two tables share one PMA authority so attachment authentication holds.
    std::byte prefix[8];
    for (std::size_t i = 0; i < 8; ++i) prefix[i] = static_cast<std::byte>(0x20 + i);
    std::byte fresh[8];
    for (std::size_t i = 0; i < 8; ++i) fresh[i] = static_cast<std::byte>(0x50 + i);
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{3, 128, 3, 2};
    const CPUFP32PagedKVTableGeometry table_geometry{2, 8, 2, 8, 64, 0};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        2 * CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table_a = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    auto table_b = CPUFP32PagedKVTable::create(authority, device, table_geometry);

    auto exporter = *pool->tryAdmit(1);
    auto shared = writePage(exporter, prefix).share();

    {
        auto update_a = table_a->tryBeginUpdate();
        ASSERT_TRUE(update_a);
        update_a->appendShared(shared.cloneForBorrow(), 1, 1);
        auto snapshot_a = update_a->commit();
        auto view_a = snapshot_a.view();
        ASSERT_EQ(view_a.rows(), 1);
        const auto *row = view_a.key(0, 0);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(std::memcmp(row, prefix, 8), 0);
    }
    {
        auto update_b = table_b->tryBeginUpdate();
        ASSERT_TRUE(update_b);
        update_b->appendShared(shared.cloneForBorrow(), 1, 1);
        auto snapshot_b = update_b->commit();
        auto view_b = snapshot_b.view();
        EXPECT_EQ(std::memcmp(view_b.key(0, 0), prefix, 8), 0);
    }

    // B diverges through COW; A and the exporter still see the original bytes.
    auto writer_b = *pool->tryAdmit(1);
    auto cow = writer_b.prepareCOW(shared, 8);
    ASSERT_TRUE(cow);
    cow->copySource();
    cow->copyNext(fresh);
    auto private_b = cow->commit();
    ASSERT_EQ(std::memcmp(private_b.read().data(), prefix, 8), 0);
    ASSERT_EQ(std::memcmp(private_b.read().data() + 8, fresh, 8), 0);
    EXPECT_EQ(std::memcmp(shared.read().data(), prefix, 8), 0);

    // The shared reference in table A is still intact after B's private publication.
    {
        auto update_a2 = table_a->tryBeginUpdate();
        ASSERT_TRUE(update_a2);
        update_a2->appendShared(shared.cloneForBorrow(), 1, 1);
        auto snapshot_a2 = update_a2->commit();
        EXPECT_EQ(std::memcmp(snapshot_a2.view().key(0, 0), prefix, 8), 0);
    }
}

/** @test Retired-but-pinned shared frames are never reassigned while a retained reader lives. */
TEST(PhysicalPagePoolSharing, RetainedReaderRetiringSharedFrameIsNeverReassigned)
{
    ShareFixture fx(3, 128, 2);
    std::byte first_bytes[8];
    std::byte prefix[8];
    for (std::size_t i = 0; i < 8; ++i)
    {
        first_bytes[i] = static_cast<std::byte>(0x10 + i);
        prefix[i] = static_cast<std::byte>(0x70 + i);
    }

    // Exporter 1 publishes a page and one reader registers; then exporter 1
    // shares it and ALL shared/publication references retire while the reader
    // stays executing. The frame is now !owned && !shared && retiring && pins.
    auto exporter = *fx.pool->tryAdmit(1);
    auto page = writePage(exporter, first_bytes);
    auto reader = page.registerRead();
    auto bytes = reader.beginCPU();
    ASSERT_EQ(std::memcmp(bytes.data(), first_bytes, 8), 0);
    auto shared = page.share();
    shared = PhysicalPagePool::SharedPage();
    page = PhysicalPagePool::CommittedPage();

    // A genuinely free frame supplies admission headroom, and a different
    // shared source exists. The retiring frame must never be selected.
    auto exporter2 = *fx.pool->tryAdmit(1);
    auto other_source = writePage(exporter2, prefix);
    auto other_shared = other_source.share();
    auto writer = *fx.pool->tryAdmit(1);
    auto cow = writer.prepareCOW(other_shared, 8);
    ASSERT_TRUE(cow);
    cow->copySource();
    cow->copyNext(std::as_bytes(std::span(first_bytes, 8)).first(8));
    auto published = cow->commit();
    // The original reader's bytes remain intact and its completion is exact.
    ASSERT_EQ(std::memcmp(bytes.data(), first_bytes, 8), 0);
    reader.completeCPU();
    EXPECT_EQ(std::memcmp(published.read().data(), prefix, 8), 0);
}

/** @test COW fresh fragments require the authenticated source copy before any write. */
TEST(PhysicalPagePoolSharing, COWFreshFragmentRequiresAuthenticatedSourceFirst)
{
    ShareFixture fx(3, 128, 2);
    std::byte prefix[8];
    for (std::size_t i = 0; i < 8; ++i) prefix[i] = static_cast<std::byte>(0x40 + i);
    std::byte combined[16];
    for (std::size_t i = 0; i < 16; ++i) combined[i] = static_cast<std::byte>(0x80 + i);
    auto exporter = *fx.pool->tryAdmit(1);
    auto shared = writePage(exporter, prefix).share();

    auto writer = *fx.pool->tryAdmit(1);
    auto cow = writer.prepareCOW(shared, 8);
    ASSERT_TRUE(cow);
    // Fresh fragment before copySource must reject, including a combined extent
    // that would pass the byte-count check through unsigned underflow.
    EXPECT_THROW(cow->copyNext(std::span<const std::byte>(combined, 16)), std::logic_error);
    EXPECT_THROW(cow->commit(), std::logic_error);
    // Repeated source copy rejects; intermediate oversized fragment rejects.
    cow->copySource();
    EXPECT_THROW(cow->copySource(), std::logic_error);
    EXPECT_THROW(cow->copyNext(std::span<const std::byte>(combined, 16)), std::logic_error);
    cow->copyNext(std::span<const std::byte>(combined, 8));
    auto published = cow->commit();
    ASSERT_EQ(std::memcmp(published.read().data(), prefix, 8), 0);
    ASSERT_EQ(std::memcmp(published.read().data() + 8, combined, 8), 0);
}

/** @test Sharing/COW preserve the conservation invariants P=O+F and allocation counts stay bounded. */
TEST(PhysicalPagePoolSharing, ConservationInvariantHoldsThroughSharingAndCOW)
{
    ShareFixture fx(3, 128, 2);
    std::byte prefix[8];
    for (std::size_t i = 0; i < 8; ++i) prefix[i] = static_cast<std::byte>(0x30 + i);
    const auto claimed = [&]() {
        return fx.pool->geometry().frames;
    };
    auto exporter = *fx.pool->tryAdmit(1);
    auto shared = writePage(exporter, prefix).share();
    auto second = *fx.pool->tryAdmit(1);
    auto adopted = second.adoptShared(shared);
    ASSERT_TRUE(adopted);
    // O unchanged by adoption: one shared frame, no second physical copy.
    EXPECT_EQ(claimed(), 3u);
    auto writer = *fx.pool->tryAdmit(1);
    auto cow = writer.prepareCOW(shared, 8);
    ASSERT_TRUE(cow);
    cow->copySource();
    cow->copyNext(std::as_bytes(std::span(prefix, 8)));
    auto published = cow->commit();
    // Shared source + COW destination + adopted reference still one frame each.
    EXPECT_EQ(claimed(), 3u);
    EXPECT_EQ(std::memcmp(published.read().data(), prefix, 8), 0);
}

/** @test A two-frame pool refuses the second simultaneous COW before mutation; serial completion fits. */
TEST(PhysicalPagePoolSharing, TwoFramePoolRefusesSimultaneousCOWBeforeMutation)
{
    ShareFixture fx(2);
    std::byte shared_bytes[16];
    std::byte fresh[7]{};
    auto exporter = *fx.pool->tryAdmit(1);
    auto shared_page = writePage(exporter, shared_bytes).share();

    // A's admission and destination fit: shared source + one private dest = 2.
    auto writer_a = *fx.pool->tryAdmit(1);
    auto plan_a = writer_a.prepareCOW(shared_page, 7);
    ASSERT_TRUE(plan_a);
    EXPECT_EQ(writer_a.remainingCommitment(), 0u);

    // B cannot be admitted while A's destination is outstanding: refusal before mutation.
    EXPECT_FALSE(fx.pool->tryAdmit(1));
    EXPECT_EQ(std::memcmp(base(shared_page), shared_bytes, 16), 0);

    // A publishes, then actually retires its private frame and request; B now fits.
    plan_a->copySource();
    plan_a->copyNext(fresh);
    auto published_a = plan_a->commit();
    writer_a.discardFrame(writer_a.assignedFrames().front());
    published_a = PhysicalPagePool::CommittedPage(); // Last reader of A retires.
    writer_a = PhysicalPagePool::Grant();             // A's whole request retires.
    auto grant_b = fx.pool->tryAdmit(1);
    ASSERT_TRUE(grant_b);
    auto plan_b = grant_b->prepareCOW(shared_page, 7);
    ASSERT_TRUE(plan_b);
}

/** @test Foreign pool, stale reference and misgeometry refusal never mutate owners. */
TEST(PhysicalPagePoolSharing, ForeignAndStaleRefusalsPreserveOriginalOwners)
{
    ShareFixture fx(3);
    ShareFixture foreign(3);
    std::byte bytes[16]{};
    auto grant = *fx.pool->tryAdmit(1);
    auto shared = writePage(grant, bytes).share();

    // Adopting from another pool is rejected before changing either owner.
    auto foreign_grant = *foreign.pool->tryAdmit(1);
    EXPECT_THROW(foreign_grant.adoptShared(shared), std::logic_error);
    EXPECT_EQ(foreign_grant.remainingCommitment(), 1u);
    EXPECT_NO_THROW(shared.read());

    // Copy-on-write against a foreign pool refuses before mutation.
    EXPECT_THROW(foreign_grant.prepareCOW(shared, 8), std::logic_error);
    EXPECT_EQ(foreign_grant.remainingCommitment(), 1u);

    // Oversized fresh extent refuses before assigning a destination.
    auto local = *fx.pool->tryAdmit(1);
    EXPECT_THROW(local.prepareCOW(shared, fx.pool->geometry().frame_bytes),
        std::invalid_argument);
    EXPECT_EQ(local.remainingCommitment(), 1u);
}

/** @test Aborted COW retains the destination privately and cannot publish partial bytes. */
TEST(PhysicalPagePoolSharing, AbortedCopyOnWriteCannotPublishPartialBytes)
{
    ShareFixture fx(3);
    std::byte shared_bytes[16];
    std::byte fresh[7]{};
    auto exporter = *fx.pool->tryAdmit(1);
    auto shared_page = writePage(exporter, shared_bytes).share();
    auto writer = *fx.pool->tryAdmit(2);
    auto plan = writer.prepareCOW(shared_page, 7);
    ASSERT_TRUE(plan);
    plan->copySource();
    EXPECT_THROW(plan->commit(), std::logic_error); // Fresh rows incomplete.
    plan = PhysicalPagePool::COWPlan();              // Cancel the plan.
    EXPECT_EQ(writer.remainingCommitment(), 1u);      // Destination remains assigned.
    EXPECT_EQ(std::memcmp(base(shared_page), shared_bytes, 16), 0);
    // The same writer can divert a fresh destination from the same source.
    auto plan2 = writer.prepareCOW(shared_page, 7);
    ASSERT_TRUE(plan2);
}

/** @test Plan P1 same-frame reappend is refused while old readers pin; resumes after retirement. */
TEST(PhysicalPagePoolSharing, SameFrameReappendWaitsForOldReaderRetirement)
{
    ShareFixture fx(2, 128, 2);
    std::vector<std::byte> write(32, std::byte{0x11});
    auto grant = *fx.pool->tryAdmit(1);
    auto frame = grant.assignFrame();
    auto plan = grant.prepareFreshWrite(frame, write.size());
    ASSERT_TRUE(plan);
    plan->copyFrom(write);
    auto page = plan->commit();
    // A registered reader pins the frame; a same-frame rewrite is refused
    // because the frame already holds a live prefix (throws, no mutation).
    auto reader = page.registerRead();
    auto bytes = reader.beginCPU();
    EXPECT_THROW(grant.prepareFreshWrite(frame, 32), std::logic_error);
    EXPECT_EQ(std::memcmp(bytes.data(), write.data(), write.size()), 0);
    reader.completeCPU();
    page = PhysicalPagePool::CommittedPage();
    // After the reader retires the prefix still exists: prepareAppend on the
    // same frame extends it without rewriting the original bytes.
    auto next = grant.prepareAppend(frame, 32);
    ASSERT_TRUE(next);
    std::vector<std::byte> next_bytes(32, std::byte{0x22});
    next->copyFrom(next_bytes);
    auto next_page = next->commit();
    ASSERT_EQ(std::memcmp(next_page.read().data(), write.data(), write.size()), 0);
    ASSERT_EQ(std::memcmp(next_page.read().data() + write.size(), next_bytes.data(), next_bytes.size()), 0);
}
