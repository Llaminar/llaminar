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
        const CPUFP32PagedKVTableGeometry table_geometry{1, 3, 3, 8};
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
