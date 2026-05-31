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

TEST(Test__PrefixStateCacheLRU, EvictedBlockPersistsToDiskAndHydratesOnFind)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    auto disk = std::make_shared<DiskPrefixStorageBackend>(dir, 128);
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

TEST(Test__PrefixStateCacheLRU, DiskHydrationFailureRecordsReadFailureAndMiss)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    auto ram = std::make_shared<RamPrefixStorageBackend>(128);
    auto disk = std::make_shared<DiskPrefixStorageBackend>(dir, 128);
    PrefixStateCache cache(64, ram, disk);
    auto a = ram->allocate(keyFor(0), layoutBytes(32));
    auto b = ram->allocate(keyFor(1), layoutBytes(32));
    auto c = ram->allocate(keyFor(2), layoutBytes(32));
    ASSERT_TRUE(cache.insert(a));
    ASSERT_TRUE(cache.insert(b));
    ASSERT_TRUE(cache.insert(c));

    const auto kv_path = disk->blockPayloadPath(a.key, "kv.bin");
    std::ofstream corrupt(kv_path, std::ios::binary | std::ios::trunc);
    corrupt << "bad";
    corrupt.close();

    EXPECT_FALSE(cache.find(a.key).has_value());
    EXPECT_EQ(cache.stats().disk_read_failures, 1u);
    EXPECT_EQ(cache.stats().misses, 1u);
    EXPECT_FALSE(cache.contains(a.key));

    cleanup();
}
