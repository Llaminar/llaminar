/**
 * @file Test__PagedKVArchiveStore.cpp
 * @brief Durable finite-quota disk archive store contracts (plan P3 §11.3).
 */
#include "execution/prefix_cache/PagedKVArchiveStore.h"
#include "execution/prefix_cache/BlockHash.h"

#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include "kernels/cpu/attention/PagedKVArchive.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

namespace llaminar2
{
namespace
{
struct TempDir
{
    std::string path;
    TempDir()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = (std::filesystem::temp_directory_path() /
            ("llaminar_kv_store_" + std::to_string(stamp) + "_" +
             std::to_string(::getpid()))).string();
        std::filesystem::create_directories(path);
    }
    ~TempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

struct ArchiveFixture
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::shared_ptr<PhysicalPagePool> pool;
    std::shared_ptr<CPUFP32PagedKVTable> table;
    ArchiveFixture()
    {
        const auto device = DeviceId::cpu();
        const PhysicalPagePoolGeometry pool_geometry{4,
            static_cast<std::size_t>(8) * 2 * 2 * 8 * sizeof(float), 2, 4};
        const CPUFP32PagedKVTableGeometry table_geometry{2, 4, 4, 8, 32, 8};
        const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
            CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes,
            .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
        table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    }
};

PagedKVArchive makeArchive(ArchiveFixture &fx, int tokens = 4, int heads = 1, int dimension = 8)
{
    auto grant = fx.pool->tryAdmit(1);
    if (!grant) throw std::logic_error("Archive fixture grant refused");
    auto frame = grant->assignFrame();
    const auto width = static_cast<size_t>(heads) * static_cast<size_t>(dimension);
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * width);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
    auto plan = grant->prepareFreshWrite(frame, rows.size() * sizeof(float));
    if (!plan) throw std::logic_error("Archive fixture write refused");
    plan->copyFrom(std::as_bytes(std::span(rows)));
    auto page = CPUFP32KVPage(plan->commit(), heads, dimension);
    auto update = fx.table->tryBeginUpdate();
    if (!update) throw std::logic_error("Archive fixture table refused");
    update->append(std::move(page));
    auto snapshot = update->commit();
    return PagedKVArchive::fromView(snapshot.view());
}

PagedKVTierConfig tierWithDisk(const std::string &dir, std::size_t cap, std::size_t membership)
{
    PagedKVTierConfig tiers;
    tiers.disk_enabled = true;
    tiers.disk_cap_bytes = cap;
    tiers.disk_journal_membership = membership;
    tiers.disk_dir = dir;
    return tiers;
}
} // namespace

/** @test Round-trip: put verifies the exact admitted extent, get re-parses
 * the section with provenance AND digest verification; bytes are identical. */
TEST(PagedKVArchiveStore, PutGetRoundTripPreservesExactBytes)
{
    TempDir dir;
    const auto route = validatePagedKVDiskRoute(tierWithDisk(dir.path, 1 << 20, 8));
    ASSERT_TRUE(route.valid);
    PagedKVArchiveStore store(route);
    ArchiveFixture fx;
    auto archive = makeArchive(fx, /*tokens=*/4, /*heads=*/1, /*dimension=*/8);

    std::string refusal;
    auto key = store.put(archive, /*generation=*/1, /*namespace_id=*/7, &refusal);
    ASSERT_TRUE(key.has_value()) << refusal;
    EXPECT_EQ(key->generation, 1u);
    EXPECT_EQ(key->namespace_id, 7u);
    EXPECT_GT(key->digest, 0u);
    EXPECT_GE(key->bytes, archive.serialize().size());
    EXPECT_TRUE(store.quota().contains({key->generation, key->namespace_id,
        key->digest, key->bytes}));

    auto loaded = store.get(*key, &refusal);
    ASSERT_TRUE(loaded.has_value()) << refusal;
    EXPECT_EQ(loaded->meta(), archive.meta());
    const auto a = archive.serialize();
    const auto b = loaded->serialize();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);
}

/** @test Retirement removes exactly the extent and releases its quota. */
TEST(PagedKVArchiveStore, RetireRemovesExtentAndReleasesQuota)
{
    TempDir dir;
    const auto route = validatePagedKVDiskRoute(tierWithDisk(dir.path, 1 << 20, 8));
    ASSERT_TRUE(route.valid);
    PagedKVArchiveStore store(route);
    ArchiveFixture fx;
    auto archive = makeArchive(fx);
    auto key = store.put(archive, 1, 7);
    ASSERT_TRUE(key.has_value());
    const auto before = store.quota().bytesAdmitted();
    EXPECT_GT(before, 0u);
    EXPECT_TRUE(store.retire(*key));
    EXPECT_FALSE(store.quota().contains({key->generation, key->namespace_id,
        key->digest, key->bytes}));
    EXPECT_EQ(store.quota().bytesAdmitted(), 0u);
    // A second retire is a no-op refusal.
    EXPECT_FALSE(store.retire(*key));
    // A get on a retired extent refuses.
    std::string refusal;
    EXPECT_FALSE(store.get(*key, &refusal).has_value());
    EXPECT_FALSE(refusal.empty());
}

/** @test A quota that cannot hold the rounded extent refuses BEFORE storage:
 * no file, no quota admission, precise diagnostic. */
TEST(PagedKVArchiveStore, QuotaFullRefusesBeforeAnyFile)
{
    TempDir dir;
    const auto route = validatePagedKVDiskRoute(tierWithDisk(dir.path, 512, 8));
    ASSERT_TRUE(route.valid);
    PagedKVArchiveStore store(route);
    ArchiveFixture fx;
    auto archive = makeArchive(fx);
    std::string refusal;
    auto key = store.put(archive, 1, 7, &refusal);
    EXPECT_FALSE(key.has_value());
    EXPECT_FALSE(refusal.empty());
    EXPECT_EQ(store.quota().bytesAdmitted(), 0u);
    // No file was created in the directory.
    std::error_code ec;
    EXPECT_EQ(std::filesystem::directory_iterator(dir.path, ec) ==
              std::filesystem::directory_iterator(), true);
    // A later larger quota admits the same section (no burned state).
    const auto route2 = validatePagedKVDiskRoute(tierWithDisk(dir.path, 1 << 20, 8));
    PagedKVArchiveStore store2(route2);
    EXPECT_TRUE(store2.put(archive, 1, 7).has_value());
}

/** @test A corrupt/tampered extent refuses on read (digest bind + parse). */
TEST(PagedKVArchiveStore, CorruptAndTamperedExtentRefuse)
{
    TempDir dir;
    const auto route = validatePagedKVDiskRoute(tierWithDisk(dir.path, 1 << 20, 8));
    ASSERT_TRUE(route.valid);
    PagedKVArchiveStore store(route);
    ArchiveFixture fx;
    auto archive = makeArchive(fx);
    auto key = store.put(archive, 1, 7);
    ASSERT_TRUE(key.has_value());
    // Flip one payload byte in place: the file still exists, the parse may
    // succeed geometrically but the digest bind must refuse.
    {
        const auto path = dir.path + "/" + prefixHashHex(key->digest) + ".7.1";
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(f.good());
        f.seekp(64); // payload region (after the fixed header).
        char b = 0;
        f.read(&b, 1);
        b = static_cast<char>(b ^ 0x5A);
        f.seekp(64);
        f.write(&b, 1);
        f.flush();
    }
    std::string refusal;
    EXPECT_FALSE(store.get(*key, &refusal).has_value());
    EXPECT_FALSE(refusal.empty());
    // Retirement still works for the admitted identity (the file exists).
    EXPECT_TRUE(store.retire(*key));
}

/** @test Engine-only cleanup: only the exact temp pattern is removed; a
 * named extent and a foreign file are never touched. */
TEST(PagedKVArchiveStore, CleanupIsNamespaceSafe)
{
    TempDir dir;
    const auto route = validatePagedKVDiskRoute(tierWithDisk(dir.path, 1 << 20, 8));
    ASSERT_TRUE(route.valid);
    PagedKVArchiveStore store(route);
    ArchiveFixture fx;
    auto archive = makeArchive(fx);
    auto key = store.put(archive, 1, 7);
    ASSERT_TRUE(key.has_value());
    // Simulate a crash-orphaned engine temp AND a foreign file.
    {
        std::ofstream t(dir.path + "/.llaminar-kv-12345.tmp");
        t << "orphan";
    }
    {
        std::ofstream f(dir.path + "/foreign-file.txt");
        f << "keep me";
    }
    EXPECT_EQ(store.cleanOrphanedTemps(), 1u);
    EXPECT_TRUE(std::filesystem::exists(dir.path + "/foreign-file.txt"));
    // The named extent still exists and reads.
    std::string refusal;
    EXPECT_TRUE(store.get(*key, &refusal).has_value()) << refusal;
    // A second cleanup finds nothing more.
    EXPECT_EQ(store.cleanOrphanedTemps(), 0u);
}

/** @test Construction refuses a wrong directory (engine-only namespace). */
TEST(PagedKVArchiveStore, WrongDirectoryRefusesConstruction)
{
    TempDir dir;
    // The route directory names a plain FILE, not a directory.
    const auto bad = dir.path + "/not-a-dir";
    {
        std::ofstream f(bad);
        f << "x";
    }
    const auto route = validatePagedKVDiskRoute(tierWithDisk(bad, 1 << 20, 8));
    ASSERT_TRUE(route.valid);
    EXPECT_THROW(PagedKVArchiveStore{route}, std::invalid_argument);
    // A route that never existed also refuses.
    const auto route2 = validatePagedKVDiskRoute(
        tierWithDisk(dir.path + "/does-not-exist", 1 << 20, 8));
    ASSERT_TRUE(route2.valid);
    EXPECT_THROW(PagedKVArchiveStore{route2}, std::invalid_argument);
}

/** @test Pending/old replacement extents: two generations coexist under the
 * quota until the old one is explicitly retired (never overwritten). */
TEST(PagedKVArchiveStore, ReplacementGenerationsCoexistUntilRetired)
{
    TempDir dir;
    const auto route = validatePagedKVDiskRoute(tierWithDisk(dir.path, 1 << 20, 8));
    ASSERT_TRUE(route.valid);
    PagedKVArchiveStore store(route);
    ArchiveFixture fx;
    auto first = makeArchive(fx, 4);
    auto second = makeArchive(fx, 3);
    auto k1 = store.put(first, 1, 7);
    ASSERT_TRUE(k1.has_value());
    auto k2 = store.put(second, 2, 7);
    ASSERT_TRUE(k2.has_value());
    EXPECT_NE(k1->digest, k2->digest);
    // Both are admitted (old + new), never an in-place overwrite.
    EXPECT_TRUE(store.quota().contains({k1->generation, k1->namespace_id,
        k1->digest, k1->bytes}));
    EXPECT_TRUE(store.quota().contains({k2->generation, k2->namespace_id,
        k2->digest, k2->bytes}));
    // Both readable; then retire the old generation: only it is gone.
    std::string refusal;
    EXPECT_TRUE(store.get(*k1, &refusal).has_value()) << refusal;
    EXPECT_TRUE(store.get(*k2, &refusal).has_value()) << refusal;
    EXPECT_TRUE(store.retire(*k1));
    EXPECT_FALSE(store.quota().contains({k1->generation, k1->namespace_id,
        k1->digest, k1->bytes}));
    EXPECT_TRUE(store.get(*k2, &refusal).has_value()) << refusal;
}

/** @test Disabled tier: an off route admits nothing and refuses clearly. */
TEST(PagedKVArchiveStore, DisabledRouteNeverAdmits)
{
    TempDir dir;
    PagedKVTierConfig tiers;
    tiers.disk_enabled = false;
    const auto route = validatePagedKVDiskRoute(tiers);
    ASSERT_TRUE(route.valid);
    PagedKVArchiveStore store(route);
    ArchiveFixture fx;
    auto archive = makeArchive(fx);
    EXPECT_FALSE(store.put(archive, 1, 7).has_value());
    EXPECT_EQ(store.quota().bytesAdmitted(), 0u);
    EXPECT_EQ(store.quota().capacity(), 1u); // Reserved off-record (representable).
}
} // namespace llaminar2
