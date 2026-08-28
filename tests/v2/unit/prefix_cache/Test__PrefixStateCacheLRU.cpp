/**
 * @file Test__PrefixStateCacheLRU.cpp
 * @brief Unit regressions for prefix-cache lookup, ownership, and tier LRU.
 */

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

    PrefixCacheKey keyForFingerprint(uint64_t fingerprint, int block)
    {
        return makePrefixCacheKey(
            fingerprint,
            0,
            block,
            block,
            {block});
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

/**
 * A placement-invalidating fingerprint transition must immediately recover
 * volatile capacity. Persisting stale blocks during that request boundary
 * would charge disk checksums, writes, and fsync to the next inference.
 */
TEST(
    Test__PrefixStateCacheLRU,
    FingerprintRebaseRetiresVolatileEntriesWithoutWritingDisk)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    constexpr size_t kBlockBytes = 32;
    constexpr uint64_t kOldFingerprint = 0x1111;
    constexpr uint64_t kNewFingerprint = 0x2222;
    auto ram = std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 3);
    auto disk = makeDiskBackend(dir, kBlockBytes * 4);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(kBlockBytes * 2, ram, disk);

    auto old_a = ram->allocate(
        keyForFingerprint(kOldFingerprint, 0),
        layoutBytes(kBlockBytes));
    auto old_b = ram->allocate(
        keyForFingerprint(kOldFingerprint, 1),
        layoutBytes(kBlockBytes));
    auto old_c = ram->allocate(
        keyForFingerprint(kOldFingerprint, 2),
        layoutBytes(kBlockBytes));
    ASSERT_TRUE(old_a.valid());
    ASSERT_TRUE(old_b.valid());
    ASSERT_TRUE(old_c.valid());
    ASSERT_TRUE(cache.insert(old_a));
    ASSERT_TRUE(cache.insert(old_b));
    ASSERT_TRUE(cache.insert(old_c));
    ASSERT_EQ(cache.stats().ram_to_disk_demotions, 1u);
    ASSERT_GT(disk->usedBytes(), 0u);
    const size_t durable_bytes_before_rebase = disk->usedBytes();

    const auto transition = cache.rebaseFingerprint(
        kOldFingerprint,
        kNewFingerprint);
    ASSERT_TRUE(transition.has_value());
    EXPECT_TRUE(transition->changed());
    EXPECT_EQ(transition->invalidated_ram_entries, 2u);
    EXPECT_EQ(transition->unindexed_disk_entries, 1u);
    EXPECT_EQ(transition->released_ram_bytes, kBlockBytes * 2);
    EXPECT_EQ(cache.usedBytes(), 0u);
    EXPECT_EQ(ram->usedBytes(), 0u);
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.isDiskResident(old_a.key));
    EXPECT_EQ(disk->usedBytes(), durable_bytes_before_rebase)
        << "Rebase must not serialize a delete or compact the durable archive";
    EXPECT_EQ(cache.stats().fingerprint_rebases, 1u);
    EXPECT_EQ(cache.stats().fingerprint_invalidated_ram_entries, 2u);
    EXPECT_EQ(cache.stats().fingerprint_unindexed_disk_entries, 1u);

    auto current = ram->allocate(
        keyForFingerprint(kNewFingerprint, 0),
        layoutBytes(kBlockBytes));
    ASSERT_TRUE(current.valid());
    ASSERT_TRUE(cache.insert(current));
    EXPECT_EQ(cache.stats().ram_to_disk_demotions, 1u)
        << "A new-epoch insert must consume recovered RAM rather than spill a stale epoch";

    cleanup();
}

/** A retained legacy lease rejects the complete rebase before any tier moves. */
TEST(Test__PrefixStateCacheLRU, BusyFingerprintRebaseIsAtomic)
{
    constexpr size_t kBlockBytes = 32;
    constexpr uint64_t kOldFingerprint = 0x1111;
    constexpr uint64_t kNewFingerprint = 0x2222;
    auto ram = std::make_shared<RamPrefixStorageBackend>(kBlockBytes * 2);
    PrefixStateCache cache(kBlockBytes * 2, ram);

    auto old = ram->allocate(
        keyForFingerprint(kOldFingerprint, 0),
        layoutBytes(kBlockBytes));
    auto current = ram->allocate(
        keyForFingerprint(kNewFingerprint, 0),
        layoutBytes(kBlockBytes));
    ASSERT_TRUE(old.valid());
    ASSERT_TRUE(current.valid());
    ASSERT_TRUE(cache.insert(old));
    ASSERT_TRUE(cache.insert(current));
    ASSERT_TRUE(cache.retain(old.key));

    EXPECT_FALSE(cache.rebaseFingerprint(
        kOldFingerprint,
        kNewFingerprint));
    EXPECT_TRUE(cache.isRamResident(old.key));
    EXPECT_TRUE(cache.isRamResident(current.key));
    EXPECT_EQ(cache.stats().fingerprint_rebases, 0u);

    EXPECT_TRUE(cache.release(old.key));
    ASSERT_TRUE(cache.rebaseFingerprint(
        kOldFingerprint,
        kNewFingerprint));
    EXPECT_FALSE(cache.isRamResident(old.key));
    EXPECT_TRUE(cache.isRamResident(current.key));
}

/**
 * @brief A prior terminal partial block must extend a later multi-turn prompt.
 *
 * Recurrent models store their continuation state on the terminal block. The
 * next request can append tokens inside that same logical block, so exact
 * full-width lookup would miss the only byte-correct continuation checkpoint.
 */
TEST(Test__PrefixStateCacheLRU, LongestTokenPrefixSelectsTerminalPartialBlock)
{
    constexpr uint64_t kFingerprint = 0xbeef;
    constexpr uint64_t kParentHash = 0x1234;
    constexpr int kBlockIndex = 2;
    constexpr int kTokenStart = 128;
    auto backend = std::make_shared<RamPrefixStorageBackend>(256);
    PrefixStateCache cache(256, backend);

    const std::vector<int32_t> terminal_tokens = {41, 42, 43};
    const PrefixCacheKey terminal_key = makePrefixCacheKey(
        kFingerprint,
        kParentHash,
        kBlockIndex,
        kTokenStart,
        terminal_tokens);
    auto terminal = backend->allocate(terminal_key, layoutBytes(32));
    terminal.has_hybrid_state = true;
    ASSERT_TRUE(cache.insert(terminal));

    const std::vector<int32_t> longer_block = {41, 42, 43, 44, 45};
    auto match = cache.findLongestTokenPrefix(
        kFingerprint,
        kParentHash,
        kBlockIndex,
        kTokenStart,
        longer_block);
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->key, terminal_key);
    EXPECT_TRUE(match->has_hybrid_state);
    EXPECT_EQ(cache.stats().lookups, 1u);
    EXPECT_EQ(cache.stats().hits, 1u);
    EXPECT_EQ(cache.stats().misses, 0u);

    const PrefixCacheKey full_key = makePrefixCacheKey(
        kFingerprint,
        kParentHash,
        kBlockIndex,
        kTokenStart,
        longer_block);
    auto full = backend->allocate(full_key, layoutBytes(32));
    ASSERT_TRUE(cache.insert(full));
    match = cache.findLongestTokenPrefix(
        kFingerprint,
        kParentHash,
        kBlockIndex,
        kTokenStart,
        longer_block);
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->key, full_key)
        << "An exact block must take precedence over an older terminal prefix";
}

/**
 * @brief Rich terminal replacement must retire stale RAM, VRAM, and disk copies.
 *
 * A block first observed inside a longer prompt carries ordinary KV payloads.
 * If a later request ends on that same key, harvest republishes it with MTP and
 * terminal state. The old disk record must not survive and reappear after the
 * richer resident block is demoted and hydrated again.
 */
TEST(Test__PrefixStateCacheLRU, PreparedReplacementCannotResurrectStaleDiskPayload)
{
    const auto dir = tempDir();
    const auto cleanup = [&]() { std::filesystem::remove_all(dir); };

    constexpr size_t kCacheBudget = 64;
    auto ram = std::make_shared<RamPrefixStorageBackend>(256);
    auto disk = makeDiskBackend(dir, 256);
    ASSERT_TRUE(disk->ready()) << disk->initializationError();
    PrefixStateCache cache(kCacheBudget, ram, disk);

    const PrefixCacheKey replaced_key = keyFor(7);
    auto old = ram->allocate(replaced_key, layoutBytes(32));
    ASSERT_TRUE(old.valid());
    std::fill(old.kv_storage->begin(), old.kv_storage->end(), 0x11);
    ASSERT_TRUE(cache.insert(old));

    auto pressure = ram->allocate(keyFor(8), layoutBytes(48));
    ASSERT_TRUE(pressure.valid());
    ASSERT_TRUE(cache.insert(pressure));
    ASSERT_TRUE(cache.isDiskResident(replaced_key));

    auto rich_layout = layoutBytes(32);
    rich_layout.includes_mtp_state = true;
    rich_layout.mtp_kv_bytes = 16;
    ASSERT_TRUE(cache.prepareInsert(replaced_key, rich_layout.totalBytes()));
    EXPECT_FALSE(cache.contains(replaced_key));

    auto rich = ram->allocate(replaced_key, rich_layout);
    ASSERT_TRUE(rich.valid());
    std::fill(rich.kv_storage->begin(), rich.kv_storage->end(), 0x22);
    ASSERT_NE(rich.mtp_storage, nullptr);
    std::fill(rich.mtp_storage->begin(), rich.mtp_storage->end(), 0x33);
    ASSERT_TRUE(cache.insert(rich));

    auto second_pressure = ram->allocate(keyFor(9), layoutBytes(32));
    ASSERT_TRUE(second_pressure.valid());
    ASSERT_TRUE(cache.insert(second_pressure));
    ASSERT_TRUE(cache.isDiskResident(replaced_key));

    const auto hydrated = cache.find(replaced_key);
    ASSERT_TRUE(hydrated.has_value());
    EXPECT_TRUE(hydrated->layout.includes_mtp_state);
    ASSERT_NE(hydrated->kv_storage, nullptr);
    ASSERT_NE(hydrated->mtp_storage, nullptr);
    EXPECT_TRUE(std::all_of(
        hydrated->kv_storage->begin(),
        hydrated->kv_storage->end(),
        [](uint8_t value) { return value == 0x22; }));
    EXPECT_TRUE(std::all_of(
        hydrated->mtp_storage->begin(),
        hydrated->mtp_storage->end(),
        [](uint8_t value) { return value == 0x33; }));

    cleanup();
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
