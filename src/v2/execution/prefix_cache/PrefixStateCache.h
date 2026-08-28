/**
 * @file PrefixStateCache.h
 * @brief Tier-aware prefix-block index, lookup, and LRU ownership contracts.
 *
 * Prefix blocks are keyed by model/runtime fingerprint, parent block, token
 * bytes, and logical position. The cache owns metadata and tier residency;
 * returned handles retain shared payload ownership so asynchronous consumers
 * do not hold a contended cache lease. Lookup APIs also understand terminal
 * partial blocks, which are important for multi-turn prompts whose previous
 * request boundary falls inside the next request's full cache block.
 */
#pragma once

#include "execution/prefix_cache/PrefixCacheStats.h"
#include "execution/prefix_cache/PrefixStateBlock.h"

#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class DeviceHotPrefixStorageBackend;
    class DiskPrefixStorageBackend;

    /**
     * @brief Total result of attempting to lease one preallocated hot slot.
     *
     * Busy is an ordinary concurrency outcome: an evicted cache key may still
     * be retained by the request currently restoring it. The caller continues
     * from the configured RAM tier without turning that temporary lease into a
     * correctness failure. Error denotes a broken lifecycle or backend edge.
     */
    enum class PrefixDeviceHotLeaseResult
    {
        Acquired,
        Ineligible,
        Busy,
        Error,
    };

    /**
     * @brief Total state transition produced by a runtime fingerprint rebase.
     *
     * `InvalidateOnRebalance` creates a new request namespace when the sole
     * MoE placement authority publishes. Old RAM and device-hot records cannot
     * satisfy the new namespace and must stop consuming the live capacity
     * budget. Durable disk records remain in the append-only archive so this
     * request-boundary transition never performs synchronous storage I/O; they
     * are merely removed from this runtime's lookup index and are reclaimed by
     * the disk tier's ordinary bounded-LRU policy.
     */
    struct PrefixFingerprintRebase
    {
        uint64_t previous_fingerprint = 0;
        uint64_t active_fingerprint = 0;
        size_t invalidated_ram_entries = 0;
        size_t invalidated_device_entries = 0;
        size_t unindexed_disk_entries = 0;
        size_t released_ram_bytes = 0;
        size_t released_device_bytes = 0;

        /** @return True when at least one stale tier record was retired. */
        bool changed() const noexcept
        {
            return invalidated_ram_entries != 0 ||
                   invalidated_device_entries != 0 ||
                   unindexed_disk_entries != 0;
        }
    };

    class PrefixStateCache
    {
    public:
        PrefixStateCache(size_t ram_budget_bytes,
                         std::shared_ptr<IPrefixStorageBackend> ram_backend,
                         std::shared_ptr<DiskPrefixStorageBackend> disk_backend = nullptr,
                         std::shared_ptr<DeviceHotPrefixStorageBackend> device_hot_backend = nullptr);

        /**
         * @brief Atomically publish durable RAM and optional device-hot handles.
         *
         * The optional hot handle must already be filled and event-published by
         * the producer stream. Passing an empty handle preserves RAM-only and
         * disk-backed cache configurations.
         */
        bool insert(
            PrefixBlockHandle handle,
            PrefixBlockHandle device_hot_handle = {});

        /**
         * @brief Reserve hot-tier capacity and allocate an unfilled VRAM block.
         *
         * The caller fills the returned block by D2D copies and passes it back
         * to insert(). This split lets serialized per-layer staging be copied
         * before the staging slot is reused.
         *
         * @param ram_archive Durable lower-tier payload and exact layout.
         * @param producer_stream Explicit GPU stream that will populate the slot.
         * @param device_hot_handle Receives the unfilled arena-backed replica.
         * @return Typed acquisition, eligibility, contention, or error result.
         */
        PrefixDeviceHotLeaseResult prepareDeviceHotCopy(
            const PrefixBlockHandle &ram_archive,
            void *producer_stream,
            PrefixBlockHandle *device_hot_handle);

        /**
         * @brief Return whether one block is eligible for the configured hot tier.
         *
         * Eligibility depends only on the immutable tier configuration and the
         * block's charged size. It deliberately ignores current occupancy:
         * prepareDeviceHotCopy() applies device-hot LRU pressure before it asks
         * the arena for a lease. Callers use this distinction to separate a
         * block that can never enter the tier from a temporarily Busy slot and
         * a genuine lifecycle Error.
         */
        bool deviceHotCapacityEligible(size_t bytes) const;

        /**
         * @brief Inspect an installed VRAM replica without changing cache stats.
         *
         * Prefix population may promote a cold terminal block after the lookup
         * result was formed. Terminal-state restoration occurs later in the same
         * request and uses this accessor to consume that promoted device owner
         * instead of uploading the stale lower-tier handle a second time.
         */
        std::optional<PrefixBlockHandle> deviceHotCopy(
            const PrefixCacheKey &key) const;

        /**
         * @brief Install compatible records discovered in a durable archive.
         *
         * This is called once after the runtime fingerprint and native payload
         * layout are known. It restores lookup metadata only; payload bytes
         * remain on disk until a matching request hydrates them.
         */
        bool installDiscoveredDiskEntries(
            uint64_t fingerprint,
            const PrefixPayloadLayout &layout,
            std::string *error = nullptr);

        /**
         * @brief Publish a completed cold-to-hot GPU promotion.
         *
         * The handle must already contain every serialized section and have a
         * readiness event recorded after the final copy. `repromotion` is true
         * when a RAM/disk hit is being returned to the hot tier.
         */
        bool installDeviceHotCopy(
            PrefixBlockHandle device_hot_handle,
            bool repromotion);
        std::optional<PrefixBlockHandle> find(const PrefixCacheKey &key);

        /**
         * @brief Find the longest installed token prefix of one logical block.
         *
         * A previous request can end partway through a cache block. Its
         * terminal block then owns the exact recurrent/hybrid state required
         * to continue a later, longer request, but its key hashes fewer tokens
         * than the later request's full block. This method probes candidate
         * widths from longest to shortest without charging every metadata
         * probe as a cache miss, then performs one ordinary find() so tier
         * hydration, LRU touch, and request statistics remain canonical.
         *
         * @param fingerprint Runtime/model fingerprint shared by all candidates.
         * @param parent_hash Stable hash of the preceding matched block.
         * @param block_index Logical block index within the token sequence.
         * @param token_start Logical token offset of this block.
         * @param tokens Current request's tokens for this block.
         * @return The longest installed handle, or std::nullopt when no token
         *         prefix is installed or the selected tier record cannot load.
         */
        std::optional<PrefixBlockHandle> findLongestTokenPrefix(
            uint64_t fingerprint,
            uint64_t parent_hash,
            int block_index,
            int token_start,
            const std::vector<int32_t> &tokens);

        bool contains(const PrefixCacheKey &key) const;
        bool retain(const PrefixCacheKey &key);
        bool release(const PrefixCacheKey &key);
        bool erase(const PrefixCacheKey &key);
        bool clear();

        /**
         * @brief Retire volatile records outside one active fingerprint.
         *
         * This is the typed publication boundary for
         * `InvalidateOnRebalance`. It never writes or fsyncs the disk archive:
         * obsolete disk records are no longer addressable by this runtime and
         * remain subject to the archive's ordinary capacity eviction. Shared
         * payload owners held by an already-admitted request remain valid even
         * after the cache drops its copies.
         *
         * The method first verifies that no legacy explicit cache lease is
         * held, then applies the transition atomically with respect to this
         * single-thread-owned cache. A zero fingerprint is rejected.
         *
         * @param previous_fingerprint Fingerprint being superseded.
         * @param active_fingerprint Newly published request fingerprint.
         * @return Transition evidence, or `std::nullopt` for an invalid/busy
         *         lifecycle that must not be hidden as a cache miss.
         */
        std::optional<PrefixFingerprintRebase> rebaseFingerprint(
            uint64_t previous_fingerprint,
            uint64_t active_fingerprint);

        /**
         * @brief Reserve RAM for a newly archived block and retire an old key.
         *
         * Terminal harvest can enrich an existing nonterminal block with
         * recurrent, MTP, and terminal-state payloads under the same token key.
         * The RAM backend accounts allocations by key, so allocating the new
         * payload before retiring every old tier copy would alias accounting and
         * could later resurrect a stale disk record. This method establishes the
         * required replacement boundary before the caller allocates storage.
         *
         * The operation rejects retained resident entries. On success no cache
         * tier contains @p key and enough cache-managed RAM capacity is available
         * for @p incoming_bytes. A later allocation or copy failure simply leaves
         * the key absent; stale payloads are never restored as a fallback.
         *
         * @param key Key that the new archive will publish.
         * @param incoming_bytes Total RAM bytes required by the new archive.
         * @return true when replacement and capacity preparation both succeed.
         */
        bool prepareInsert(
            const PrefixCacheKey &key,
            size_t incoming_bytes);

        void recordRequestLookup(int requested_tokens,
                                 int matched_tokens,
                                 int matched_blocks);
        void recordTerminalStateHit();

        size_t size() const { return entries_.size(); }
        size_t ramBudgetBytes() const { return ram_budget_bytes_; }
        size_t usedBytes() const { return used_bytes_; }
        const PrefixCacheStats &stats() const { return stats_; }
        std::vector<PrefixCacheKey> keysMostRecentFirst() const;
        bool isRamResident(const PrefixCacheKey &key) const;
        bool isDeviceHotResident(const PrefixCacheKey &key) const;
        bool isDiskResident(const PrefixCacheKey &key) const;

    private:
        struct Entry
        {
            PrefixStateBlock block;
            std::list<PrefixCacheKey>::iterator lru_it;
        };

        bool insertResident(PrefixBlockHandle handle, bool count_store, bool preserve_disk_entry = false);
        bool evictResident(const PrefixCacheKey &key);
        bool evictUntilFits(size_t incoming_bytes);
        bool removeDeviceHotEntry(
            const PrefixCacheKey &key,
            bool capacity_eviction);
        void touchDeviceHot(const PrefixCacheKey &key);
        bool persistResidentToDisk(const Entry &entry);
        bool removeDiskEntry(const PrefixCacheKey &key);
        void forgetDiskEntry(const PrefixCacheKey &key);
        void touch(Entry &entry);
        void addResidentStats(const PrefixBlockHandle &handle);
        void subtractResidentStats(const PrefixBlockHandle &handle);

        size_t ram_budget_bytes_ = 0;
        size_t used_bytes_ = 0;
        uint64_t tick_ = 0;
        PrefixCacheStats stats_;
        std::shared_ptr<IPrefixStorageBackend> ram_backend_;
        std::shared_ptr<DiskPrefixStorageBackend> disk_backend_;
        std::shared_ptr<DeviceHotPrefixStorageBackend> device_hot_backend_;
        std::unordered_map<PrefixCacheKey, Entry, PrefixCacheKeyHasher> entries_;
        std::unordered_map<PrefixCacheKey, PrefixBlockHandle, PrefixCacheKeyHasher> device_hot_entries_;
        std::unordered_map<PrefixCacheKey, PrefixBlockHandle, PrefixCacheKeyHasher> disk_entries_;
        std::list<PrefixCacheKey> lru_;
        std::list<PrefixCacheKey> device_hot_lru_;
    };

} // namespace llaminar2
