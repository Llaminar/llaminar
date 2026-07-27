#include <gtest/gtest.h>

#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixStateCache.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>

using namespace llaminar2;

namespace
{
    constexpr const char *kModelSha256 =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    PrefixPayloadLayout layoutBytes(size_t bytes)
    {
        PrefixPayloadLayout layout;
        layout.block_size = 1;
        layout.fa_layers = 1;
        layout.total_layers = 1;
        layout.bytes_per_fa_layer_k = bytes / 2;
        layout.bytes_per_fa_layer_v = bytes - layout.bytes_per_fa_layer_k;
        return layout;
    }

    PrefixCacheKey keyFor(int block)
    {
        return makePrefixCacheKey(0xbeef, 0, block, block, {block});
    }

    std::filesystem::path tempDir()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / ("llaminar_prefix_state_cache_" + std::to_string(stamp));
    }

    std::shared_ptr<DiskPrefixStorageBackend> makeDiskBackend(
        const std::filesystem::path &directory,
        size_t budget_bytes)
    {
        return std::make_shared<DiskPrefixStorageBackend>(
            directory / (std::string(kModelSha256) + ".kvcache"),
            budget_bytes,
            kModelSha256);
    }
} // namespace

TEST(Test__PrefixStateCacheLRU, InsertFindAndTouchUpdatesRecency)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(128);
    PrefixStateCache cache(128, backend);
    auto a = backend->allocate(keyFor(0), layoutBytes(32));
    auto b = backend->allocate(keyFor(1), layoutBytes(32));

    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.find(a.key).has_value());

    const auto keys = cache.keysMostRecentFirst();
    ASSERT_EQ(keys.size(), 2u);
    EXPECT_EQ(keys[0], a.key);
    EXPECT_EQ(keys[1], b.key);
    EXPECT_EQ(cache.stats().lookups, 1u);
    EXPECT_EQ(cache.stats().hits, 1u);
    EXPECT_EQ(cache.stats().stores, 2u);
}

TEST(Test__PrefixStateCacheLRU, EvictsLeastRecentlyUsedBlocksToFitBudget)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(96);
    PrefixStateCache cache(64, backend);
    auto a = backend->allocate(keyFor(0), layoutBytes(32));
    auto b = backend->allocate(keyFor(1), layoutBytes(32));
    auto c = backend->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.find(a.key).has_value()); // b becomes LRU.
    ASSERT_TRUE(cache.insert(c));

    EXPECT_TRUE(cache.contains(a.key));
    EXPECT_FALSE(cache.contains(b.key));
    EXPECT_TRUE(cache.contains(c.key));
    EXPECT_EQ(cache.usedBytes(), 64u);
    EXPECT_EQ(cache.stats().evictions, 1u);
}

/**
 * @brief Prove copied handles retain payload storage after metadata eviction.
 *
 * GPU restore uses this ownership property to avoid a contended cache lease:
 * lookup returns a cheap shared-owner copy, the cache remains free to evict its
 * metadata entry, and an event-retirement queue keeps that copied handle alive
 * until the asynchronous restore has stopped reading the payload.
 */
TEST(Test__PrefixStateCacheLRU, CopiedHandleOwnsPayloadAfterMetadataEviction)
{
    constexpr size_t kBlockBytes = 32;
    auto backend =
        std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 3);
    PrefixStateCache cache(kBlockBytes * 2, backend);

    auto a = backend->allocate(keyFor(0), layoutBytes(kBlockBytes));
    auto b = backend->allocate(keyFor(1), layoutBytes(kBlockBytes));
    auto c = backend->allocate(keyFor(2), layoutBytes(kBlockBytes));
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    ASSERT_TRUE(c.valid());
    ASSERT_NE(a.kv_storage, nullptr);
    std::fill(a.kv_storage->begin(), a.kv_storage->end(), uint8_t{0xa5});

    const PrefixCacheKey a_key = a.key;
    const PrefixCacheKey b_key = b.key;
    ASSERT_TRUE(cache.insert(std::move(a)));
    ASSERT_TRUE(cache.insert(std::move(b)));

    auto held_restore_source = cache.find(a_key);
    ASSERT_TRUE(held_restore_source.has_value());
    ASSERT_NE(held_restore_source->kv_storage, nullptr);
    std::weak_ptr<std::vector<uint8_t>> payload_lifetime =
        held_restore_source->kv_storage;

    /*
     * Touch B after obtaining A's restore handle so A becomes the metadata
     * LRU. Inserting C evicts A from the cache while the copied handle remains
     * the sole owner used by the simulated asynchronous restore.
     */
    ASSERT_TRUE(cache.find(b_key).has_value());
    ASSERT_TRUE(cache.insert(std::move(c)));
    EXPECT_FALSE(cache.contains(a_key));
    EXPECT_FALSE(payload_lifetime.expired());
    ASSERT_NE(held_restore_source->kv_storage, nullptr);
    EXPECT_TRUE(std::all_of(
        held_restore_source->kv_storage->begin(),
        held_restore_source->kv_storage->end(),
        [](uint8_t byte)
        {
            return byte == uint8_t{0xa5};
        }));

    held_restore_source.reset();
    EXPECT_TRUE(payload_lifetime.expired())
        << "The copied restore handle, not cache metadata or a manual lease, "
           "must own the evicted payload";
}

TEST(Test__PrefixStateCacheLRU, RetainedBlocksAreNotEvicted)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(96);
    PrefixStateCache cache(64, backend);
    auto a = backend->allocate(keyFor(0), layoutBytes(32));
    auto b = backend->allocate(keyFor(1), layoutBytes(32));
    auto c = backend->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.retain(a.key));
    ASSERT_TRUE(cache.insert(c)); // b can be evicted, a cannot.

    EXPECT_TRUE(cache.contains(a.key));
    EXPECT_FALSE(cache.contains(b.key));
    EXPECT_TRUE(cache.contains(c.key));
    EXPECT_FALSE(cache.erase(a.key));
    EXPECT_TRUE(cache.release(a.key));
    EXPECT_TRUE(cache.erase(a.key));
}

TEST(Test__PrefixStateCacheLRU, TooLargeBlockIsRejected)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(32);
    PrefixStateCache cache(32, backend);
    PrefixBlockHandle handle;
    handle.key = keyFor(99);
    handle.layout = layoutBytes(64);
    handle.tier = PrefixStorageTier::Ram;
    handle.total_bytes = 64;
    EXPECT_FALSE(cache.insert(handle));
}

TEST(Test__PrefixStateCacheLRU, RecordsRequestLevelStatsAndResidentPayloadBytes)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(256);
    PrefixStateCache cache(256, backend);
    auto layout = layoutBytes(32);
    layout.includes_hybrid_state = true;
    layout.hybrid_state_bytes = 16;
    layout.includes_mtp_state = true;
    layout.mtp_kv_bytes = 8;

    auto handle = backend->allocate(keyFor(3), layout);
    ASSERT_TRUE(cache.insert(handle));

    cache.recordRequestLookup(/*requested_tokens=*/5,
                              /*matched_tokens=*/3,
                              /*matched_blocks=*/1);
    cache.recordTerminalStateHit();

    EXPECT_EQ(cache.stats().partial_hits, 1u);
    EXPECT_EQ(cache.stats().matched_tokens, 3u);
    EXPECT_EQ(cache.stats().matched_blocks, 1u);
    EXPECT_EQ(cache.stats().terminal_state_hits, 1u);
    EXPECT_EQ(cache.stats().hybrid_state_bytes, 16u);
    EXPECT_EQ(cache.stats().mtp_state_bytes, 8u);

    ASSERT_TRUE(cache.erase(handle.key));
    EXPECT_EQ(cache.stats().hybrid_state_bytes, 0u);
    EXPECT_EQ(cache.stats().mtp_state_bytes, 0u);
}

TEST(Test__PrefixStateCacheLRU, ClearReleasesResidentEntriesAndPayloadAccounting)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(256);
    PrefixStateCache cache(256, backend);
    auto layout = layoutBytes(32);
    layout.includes_hybrid_state = true;
    layout.hybrid_state_bytes = 16;
    layout.includes_mtp_state = true;
    layout.mtp_kv_bytes = 8;

    auto a = backend->allocate(keyFor(0), layout);
    auto b = backend->allocate(keyFor(1), layout);
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_EQ(cache.usedBytes(), 112u);
    ASSERT_EQ(cache.stats().ram_bytes, 112u);
    ASSERT_EQ(cache.stats().hybrid_state_bytes, 32u);
    ASSERT_EQ(cache.stats().mtp_state_bytes, 16u);

    ASSERT_TRUE(cache.clear());

    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(cache.usedBytes(), 0u);
    EXPECT_EQ(cache.stats().ram_bytes, 0u);
    EXPECT_EQ(cache.stats().hybrid_state_bytes, 0u);
    EXPECT_EQ(cache.stats().mtp_state_bytes, 0u);
    EXPECT_FALSE(cache.contains(a.key));
    EXPECT_TRUE(backend->canStore(256))
        << "clear should release resident allocations back to the RAM backend";
}

TEST(Test__PrefixStateCacheLRU, ClearRefusesRetainedResidentEntries)
{
    auto backend = std::make_shared<RamPrefixStorageBackend>(128);
    PrefixStateCache cache(128, backend);
    auto handle = backend->allocate(keyFor(4), layoutBytes(32));
    ASSERT_TRUE(cache.insert(handle));
    ASSERT_TRUE(cache.retain(handle.key));

    EXPECT_FALSE(cache.clear());
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(cache.usedBytes(), 32u);
    EXPECT_TRUE(cache.contains(handle.key));

    ASSERT_TRUE(cache.release(handle.key));
    EXPECT_TRUE(cache.clear());
    EXPECT_EQ(cache.size(), 0u);
}

TEST(Test__PrefixStateCacheLRU, EvictedBlockPersistsToDiskAndHydratesOnFind)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    auto disk = makeDiskBackend(dir, 128);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(64, ram, disk);
    auto a = ram->allocate(keyFor(0), layoutBytes(32));
    auto b = ram->allocate(keyFor(1), layoutBytes(32));
    auto c = ram->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    ASSERT_TRUE(c.valid());
    std::fill(a.kv_storage->begin(), a.kv_storage->end(), 0xa5);

    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.insert(c));

    EXPECT_TRUE(cache.contains(a.key))
        << "disk-resident blocks remain addressable after RAM eviction";
    EXPECT_EQ(cache.usedBytes(), 64u);
    EXPECT_EQ(cache.stats().evictions, 1u);
    EXPECT_EQ(cache.stats().disk_bytes, 32u);

    auto hydrated = cache.find(a.key);
    ASSERT_TRUE(hydrated.has_value());
    ASSERT_NE(hydrated->kv_storage, nullptr);
    EXPECT_EQ(hydrated->tier, PrefixStorageTier::Ram);
    EXPECT_EQ(*hydrated->kv_storage, *a.kv_storage);
    EXPECT_EQ(cache.stats().disk_hydrations, 1u);
    EXPECT_EQ(cache.stats().promotions, 1u);
    EXPECT_GE(cache.stats().disk_bytes, 32u);

    cleanup();
}

TEST(Test__PrefixStateCacheLRU, RepeatedPromotionDemotionAndBottomTierEvictionAreExact)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    constexpr size_t kBlockBytes = 32;
    auto ram = std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 4);
    auto disk = makeDiskBackend(dir, kBlockBytes * 2);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(kBlockBytes * 2, ram, disk);

    auto a = ram->allocate(keyFor(0), layoutBytes(kBlockBytes));
    auto b = ram->allocate(keyFor(1), layoutBytes(kBlockBytes));
    auto c = ram->allocate(keyFor(2), layoutBytes(kBlockBytes));
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    ASSERT_TRUE(c.valid());

    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.find(a.key).has_value()); // B is the oldest RAM block.
    ASSERT_TRUE(cache.insert(c));

    EXPECT_TRUE(cache.isRamResident(a.key));
    EXPECT_FALSE(cache.isRamResident(b.key));
    EXPECT_TRUE(cache.isDiskResident(b.key));
    EXPECT_TRUE(cache.isRamResident(c.key));
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 1u);

    /*
     * B rises from disk to RAM. Making room demotes A, while B's durable disk
     * copy remains present and receives a persistent LRU touch.
     */
    ASSERT_TRUE(cache.find(b.key).has_value());
    EXPECT_FALSE(cache.isRamResident(a.key));
    EXPECT_TRUE(cache.isDiskResident(a.key));
    EXPECT_TRUE(cache.isRamResident(b.key));
    EXPECT_TRUE(cache.isDiskResident(b.key));
    EXPECT_TRUE(cache.isRamResident(c.key));
    EXPECT_EQ(cache.stats().disk_hydrations, 1u);
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 2u);

    /*
     * A now rises. C must be demoted first. Disk is already full, so the newly
     * arriving C record overwrites B, the oldest bottom-tier record. B remains
     * valid in RAM and can be demoted again on a later pressure cycle.
     */
    ASSERT_TRUE(cache.find(a.key).has_value());
    EXPECT_TRUE(cache.isRamResident(a.key));
    EXPECT_TRUE(cache.isDiskResident(a.key));
    EXPECT_TRUE(cache.isRamResident(b.key));
    EXPECT_FALSE(cache.isDiskResident(b.key));
    EXPECT_FALSE(cache.isRamResident(c.key));
    EXPECT_TRUE(cache.isDiskResident(c.key));
    EXPECT_EQ(cache.stats().disk_hydrations, 2u);
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 3u);
    EXPECT_EQ(cache.stats().disk_evictions, 1u);

    /*
     * C's second promotion proves the cycle is repeatable. B is demoted back
     * into disk and overwrites now-oldest A; no key is lost from the hierarchy.
     */
    ASSERT_TRUE(cache.find(c.key).has_value());
    EXPECT_TRUE(cache.isRamResident(a.key));
    EXPECT_FALSE(cache.isDiskResident(a.key));
    EXPECT_FALSE(cache.isRamResident(b.key));
    EXPECT_TRUE(cache.isDiskResident(b.key));
    EXPECT_TRUE(cache.isRamResident(c.key));
    EXPECT_TRUE(cache.isDiskResident(c.key));
    EXPECT_EQ(cache.stats().disk_hydrations, 3u);
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 4u);
    EXPECT_EQ(cache.stats().disk_evictions, 2u);
    EXPECT_EQ(cache.stats().evictions, 4u);

    cleanup();
}

TEST(Test__PrefixStateCacheLRU, ClearReleasesDiskEntries)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    auto disk = makeDiskBackend(dir, 128);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(64, ram, disk);
    auto a = ram->allocate(keyFor(0), layoutBytes(32));
    auto b = ram->allocate(keyFor(1), layoutBytes(32));
    auto c = ram->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.insert(c));
    ASSERT_TRUE(cache.contains(a.key));
    ASSERT_EQ(cache.stats().disk_bytes, 32u);

    ASSERT_TRUE(cache.clear());

    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(cache.usedBytes(), 0u);
    EXPECT_EQ(cache.stats().ram_bytes, 0u);
    EXPECT_EQ(cache.stats().disk_bytes, 0u);
    EXPECT_FALSE(cache.contains(a.key));
    EXPECT_FALSE(cache.contains(b.key));
    EXPECT_FALSE(cache.contains(c.key));

    cleanup();
}

TEST(Test__PrefixStateCacheLRU, DiskHydrationFailureRecordsReadFailureAndMiss)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    auto disk = makeDiskBackend(dir, 128);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(64, ram, disk);
    auto a = ram->allocate(keyFor(0), layoutBytes(32));
    auto b = ram->allocate(keyFor(1), layoutBytes(32));
    auto c = ram->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.insert(c));

    const auto archive_bytes = std::filesystem::file_size(disk->archivePath());
    ASSERT_GT(archive_bytes, 17u);
    std::fstream corrupt(
        disk->archivePath(),
        std::ios::binary | std::ios::in | std::ios::out);
    corrupt.seekg(static_cast<std::streamoff>(archive_bytes - 17));
    char byte = 0;
    corrupt.read(&byte, 1);
    byte ^= 0x33;
    corrupt.seekp(static_cast<std::streamoff>(archive_bytes - 17));
    corrupt.write(&byte, 1);
    corrupt.close();

    EXPECT_FALSE(cache.find(a.key).has_value());
    EXPECT_EQ(cache.stats().disk_read_failures, 1u);
    EXPECT_EQ(cache.stats().misses, 1u);
    EXPECT_FALSE(cache.contains(a.key));

    cleanup();
}
