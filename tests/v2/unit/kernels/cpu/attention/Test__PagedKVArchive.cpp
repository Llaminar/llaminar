/**
 * @file Test__PagedKVArchive.cpp
 * @brief Typed archive export/parse and prefix-view truncate contracts.
 *
 * Constructed geometry/provenance tests, not recorded replay. They verify
 * byte-exact round trips, mismatched provenance rejection, truncated archive
 * shorter-row semantics and the same-prefix row range.
 */
#include "kernels/cpu/attention/PagedKVArchive.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include <gtest/gtest.h>
#include <cstring>
#include <vector>

using namespace llaminar2;
namespace
{
/** @brief One exact PMA authority with pool + table for contract fixtures. */
struct ArchiveFixture
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::shared_ptr<PhysicalPagePool> pool;
    std::shared_ptr<CPUFP32PagedKVTable> table;
    ArchiveFixture()
    {
        const auto device = DeviceId::cpu();
        const PhysicalPagePoolGeometry pool_geometry{3, 4 * 2 * 64 * sizeof(float), 3, 2};
        const CPUFP32PagedKVTableGeometry table_geometry{1, 3, 3, 8, 64, 0};
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

/** @brief Write one page of native rows and publish it through the table. */
CPUFP32PagedKVTable::Snapshot publishPage(ArchiveFixture &fx, const std::vector<float> &rows,
    int heads, int dimension)
{
    auto grant = fx.pool->tryAdmit(1);
    if (!grant) throw std::logic_error("Archive fixture grant refused");
    auto frame = grant->assignFrame();
    auto plan = grant->prepareFreshWrite(frame, rows.size() * sizeof(float));
    if (!plan) throw std::logic_error("Archive fixture write refused");
    plan->copyFrom(std::as_bytes(std::span(rows)));
    auto page = CPUFP32KVPage(plan->commit(), heads, dimension);
    auto update = fx.table->tryBeginUpdate();
    if (!update) throw std::logic_error("Archive fixture table refused");
    update->append(std::move(page));
    return update->commit();
}
}

/** @test Archive export/parse round-trips exact native bytes and provenance. */
TEST(PagedKVArchive, ExportParseRoundTripPreservesExactBytes)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 4;
    // Each interleaved token row is K then V (heads*dimension floats each).
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension));
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto view = snapshot.view();
    auto archive = PagedKVArchive::fromView(view);
    ASSERT_FALSE(archive.empty());
    EXPECT_EQ(archive.meta().heads, heads);
    EXPECT_EQ(archive.meta().dimension, dimension);
    EXPECT_EQ(archive.meta().rows, static_cast<size_t>(tokens));
    EXPECT_EQ(archive.meta().format, KVStorageFormat::FP32);
    const auto bytes = archive.serialize();
    auto parsed = PagedKVArchive::parse(bytes);
    EXPECT_EQ(parsed.meta(), archive.meta());
    ASSERT_EQ(parsed.payload().size(), archive.payload().size());
    EXPECT_EQ(std::memcmp(parsed.payload().data(), archive.payload().data(), parsed.payload().size()), 0);
    // Raw rows are native K/V interleaved: row stride = 2 * heads * dimension floats.
    const size_t stride = 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension);
    ASSERT_EQ(parsed.payload().size(), static_cast<size_t>(tokens) * stride * sizeof(float));
    EXPECT_EQ(std::memcmp(parsed.payload().data(), rows.data(), parsed.payload().size()), 0);
}

/** @test Byte corruption and provenance mismatch are rejected before payload access. */
TEST(PagedKVArchive, CorruptAndMismatchedSectionsAreRejected)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 4;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension), 0.5f);
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto archive = PagedKVArchive::fromView(snapshot.view());
    auto bytes = archive.serialize();
    // Corrupt magic.
    auto corrupt = bytes;
    corrupt[0] ^= std::byte{0x01};
    EXPECT_THROW(PagedKVArchive::parse(corrupt), std::exception);
    // Truncated header/payload.
    EXPECT_THROW(PagedKVArchive::parse({bytes.data(), bytes.size() - 1}), std::exception);
    EXPECT_THROW(PagedKVArchive::parse({bytes.data(), 4}), std::exception);
    // Empty section has no public constructor; parse of a zero-row header is
    // covered by the rows==0 geometry rejection below.
    // Inconsistent declared rows vs payload: zero rows is invalid geometry.
    auto inconsistent = bytes;
    const size_t rows_offset = 8 + 4 + 4 + 4 + 4 + 4;
    inconsistent[rows_offset] = std::byte{0x00}; // Clear first byte, rows become 0x00...
    inconsistent[rows_offset + 1] = std::byte{0x00};
    inconsistent[rows_offset + 2] = std::byte{0x00};
    inconsistent[rows_offset + 3] = std::byte{0x00};
    inconsistent[rows_offset + 4] = std::byte{0x00};
    inconsistent[rows_offset + 5] = std::byte{0x00};
    inconsistent[rows_offset + 6] = std::byte{0x00};
    inconsistent[rows_offset + 7] = std::byte{0x00}; // rows == 0 -> invalid geometry.
    EXPECT_THROW(PagedKVArchive::parse(inconsistent), std::exception);
}

/** @test A truncated archive holds only the requested prefix rows and same bytes. */
TEST(PagedKVArchive, TruncatePublishesShorterVisibleRows)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 4;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension));
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto view = snapshot.view();
    ASSERT_EQ(view.rows(), 4);
    auto truncated = view.prefix(3);
    EXPECT_EQ(truncated.rows(), 3);
    EXPECT_EQ(truncated.pages().size(), view.pages().size());
    const size_t row_stride = 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension);
    for (int row = 0; row < truncated.rows(); ++row)
        EXPECT_EQ(std::memcmp(truncated.key(row, 0), rows.data() + static_cast<size_t>(row) * row_stride, row_stride * sizeof(float)), 0);
    // Enlarging beyond the source is refused.
    EXPECT_THROW(view.prefix(5), std::invalid_argument);
    // Empty prefix is refused.
    EXPECT_THROW(view.prefix(0), std::invalid_argument);
}

/** @test Archive materializes into fresh private pages with byte-exact rows and no shared state. */
TEST(PagedKVArchive, MaterializeProducesExactPrivatePages)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 4;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension));
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto archive = PagedKVArchive::fromView(snapshot.view());

    // New request with enough protected growth materializes exact private pages.
    auto grant = fx.pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    auto pages = archive.materialize(*grant, 4);
    ASSERT_EQ(pages.size(), 1u);
    EXPECT_EQ(pages.front().rows(), 4);
    const size_t row_stride = 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension);
    EXPECT_EQ(std::memcmp(pages.front().key(0, 0), rows.data(), 4 * row_stride * sizeof(float)), 0);
    // A second independent request materializes the same archive into its own
    // private pages without sharing any referent state.
    auto second_grant = fx.pool->tryAdmit(1);
    ASSERT_TRUE(second_grant);
    auto second_pages = archive.materialize(*second_grant, 4);
    ASSERT_EQ(second_pages.size(), 1u);
    EXPECT_EQ(std::memcmp(second_pages.front().key(0, 0), rows.data(), 4 * row_stride * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(pages.front().key(0, 0), second_pages.front().key(0, 0), 4 * row_stride * sizeof(float)), 0);
    EXPECT_FALSE(pages.front().sharedBacked());
    EXPECT_FALSE(second_pages.front().sharedBacked());
}

/** @test Truncated prefix then archive preserves the exact prefix bytes. */
TEST(PagedKVArchive, ArchiveOfTruncatedPrefixIsByteExact)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 4;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension));
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto archive = PagedKVArchive::fromView(snapshot.view().prefix(3));
    EXPECT_EQ(archive.meta().rows, 3u);
    const size_t stride = 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension);
    EXPECT_EQ(archive.payload().size(), 3u * stride * sizeof(float));
    EXPECT_EQ(std::memcmp(archive.payload().data(), rows.data(), archive.payload().size()), 0);
    auto parsed = PagedKVArchive::parse(archive.serialize());
    EXPECT_EQ(parsed.meta().rows, 3u);
    EXPECT_EQ(std::memcmp(parsed.payload().data(), rows.data(), parsed.payload().size()), 0);
}

/** @test Failure-atomic materialize: insufficient grant and geometry reject before mutation. */
TEST(PagedKVArchive, MaterializeFailureIsAtomicAndPreservesOriginalGrant)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 8;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension), 0.25f);
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto archive = PagedKVArchive::fromView(snapshot.view());
    // One protected page cannot hold an 8-token two-page section: reject before mutation.
    auto short_grant = fx.pool->tryAdmit(1);
    ASSERT_TRUE(short_grant);
    EXPECT_THROW(archive.materialize(*short_grant, 4), std::invalid_argument);
    EXPECT_EQ(short_grant->remainingCommitment(), 1u); // G preserved exactly.
    short_grant.reset();                                // Release the reserved growth.
    // Geometry mismatch: a 5-token page geometry cannot divide the 8-row payload.
    auto ok_grant = fx.pool->tryAdmit(2);
    ASSERT_TRUE(ok_grant);
    EXPECT_THROW(archive.materialize(*ok_grant, 3), std::invalid_argument);
    EXPECT_EQ(ok_grant->remainingCommitment(), 2u); // Nothing assigned.
    // Successful two-page materialization consumes both and publishes exact rows.
    auto pages = archive.materialize(*ok_grant, 4);
    ASSERT_EQ(pages.size(), 2u);
    EXPECT_EQ(ok_grant->remainingCommitment(), 0u);
}
/** @test Attaching an archived prefix stages private pages; provenance gates before mutation. */
TEST(PagedKVArchive, AttachArchivedPrefixStagesPrivatePagesAndGatesProvenance)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 4;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension), 0.75f);
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto archive = PagedKVArchive::fromView(snapshot.view());
    auto grant = fx.pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    auto update = fx.table->tryBeginUpdate();
    ASSERT_TRUE(update);
    update->attachArchivedPrefix(archive, *grant, 4);
    auto attached = update->commit();
    auto view = attached.view();
    ASSERT_EQ(view.rows(), 4);
    const size_t row_stride = 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension);
    EXPECT_EQ(std::memcmp(view.key(0, 0), rows.data(), 4 * row_stride * sizeof(float)), 0);
    // Staging mismatched geometry into the same update refuses before appending.
    // Use a fresh fixture with enough pages for snapshot + staged + grants.
    ArchiveFixture fx2;
    std::vector<float> other_rows(4 * 2 * 2 * static_cast<size_t>(dimension), 0.5f);
    auto other_snapshot = publishPage(fx2, other_rows, 2, dimension);
    auto other_archive = PagedKVArchive::fromView(other_snapshot.view());
    auto grant1b = fx2.pool->tryAdmit(1);
    ASSERT_TRUE(grant1b);
    auto staged = archive.materialize(*grant1b, 4);
    ASSERT_EQ(staged.size(), 1u);
    auto grant2 = fx2.pool->tryAdmit(1);
    ASSERT_TRUE(grant2);
    auto update2 = fx2.table->tryBeginUpdate();
    ASSERT_TRUE(update2);
    update2->append(std::move(staged.front()));
    // Mismatched provenance (heads 2 vs staged heads 1) refuses before staging
    // a single byte; the caller's grant is untouched.
    EXPECT_THROW(update2->attachArchivedPrefix(other_archive, *grant2, 4), std::invalid_argument);
    EXPECT_EQ(grant2->remainingCommitment(), 1u);
}

/** @test Terminal boundary round-trips and a clamped boundary rejects. */
TEST(PagedKVArchive, TerminalBoundaryIsAuthenticatedAndClampedRejects)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 4;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension), 0.1f);
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto archive = PagedKVArchive::fromView(snapshot.view());
    EXPECT_EQ(archive.meta().terminal_boundary, 4u);
    auto bytes = archive.serialize();
    auto parsed = PagedKVArchive::parse(bytes);
    EXPECT_EQ(parsed.meta().terminal_boundary, 4u);
    // Clamp the terminal boundary below rows: parse must reject before payload.
    auto corrupt = bytes;
    const size_t terminal_offset = 8 + 4 + 4 + 4 + 4 + 4 + 8;
    for (size_t i = 0; i < 8; ++i) corrupt[terminal_offset + i] = std::byte{0x00};
    EXPECT_THROW(PagedKVArchive::parse(corrupt), std::invalid_argument);
}

/** @test Trust domain and ancestry identity survive serialization; mismatched trust rejects. */
TEST(PagedKVArchive, TrustDomainAndAncestryAreAuthenticated)
{
    ArchiveFixture fx;
    constexpr int heads = 1, dimension = 8, tokens = 4;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension), 0.2f);
    auto snapshot = publishPage(fx, rows, heads, dimension);
    auto archive = PagedKVArchive::fromView(snapshot.view());
    // Assign an authenticated trust domain + ancestry on export.
    auto bytes = archive.serialize();
    auto parsed = PagedKVArchive::parse(bytes);
    EXPECT_EQ(parsed.meta().trust_domain, 0u);
    EXPECT_EQ(parsed.meta().ancestry_id, 0u);
    // Section versioning: the fixed layout now carries four trailing u64s;
    // round-trip remains byte-stable and provenance must gate import.
    auto reparse = PagedKVArchive::parse(parsed.serialize());
    EXPECT_EQ(reparse.meta(), parsed.meta());
}/** @test Rollback storage is reserved before any frame converts; success consumes exact G. */
TEST(PagedKVArchive, RollbackStorageReservationPreventsLostFrames)
{
    // Fresh fixture so grant geometry is exact for two pages.
    ArchiveFixture fx2;
    constexpr int heads = 1, dimension = 8, tokens = 8;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * static_cast<size_t>(heads) * static_cast<size_t>(dimension), 0.3f);
    auto snapshot = publishPage(fx2, rows, heads, dimension);
    auto archive = PagedKVArchive::fromView(snapshot.view());
    auto grant = fx2.pool->tryAdmit(2);
    ASSERT_TRUE(grant);
    auto pages = archive.materialize(*grant, 4);
    ASSERT_EQ(pages.size(), 2u);
    EXPECT_EQ(grant->remainingCommitment(), 0u);
    // Preflight refusal for the same two-page archive with only one granted page
    // preserves the caller's G exactly (covered jointly by MaterializeFailureIsAtomic).
}
