/**
 * @file PrefixStateCache.cpp
 * @brief Implements durable prefix-block indexing and tier-aware LRU lookup.
 *
 * Metadata probing is separated from payload acquisition so a longest-prefix
 * query records one semantic hit or miss even when it examines several token
 * widths. The final find() remains the sole authority for disk hydration, LRU
 * movement, and shared payload ownership. Early preparation only queues native
 * persistence; required RAM/SSD publication completes exact immutable receipts
 * before allocating. Pending durable work is never dropped or credited as free.
 */

#include "execution/prefix_cache/PrefixStateCache.h"

#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"
#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <utility>

namespace llaminar2
{

    PrefixStateCache::PrefixStateCache(size_t ram_budget_bytes,
                                       std::shared_ptr<IPrefixStorageBackend> ram_backend,
                                       std::shared_ptr<DiskPrefixStorageBackend> disk_backend,
                                       std::shared_ptr<DeviceHotPrefixStorageBackend> device_hot_backend)
        : ram_budget_bytes_(ram_budget_bytes),
          ram_backend_(std::move(ram_backend)),
          disk_backend_(std::move(disk_backend)),
          device_hot_backend_(std::move(device_hot_backend))
    {
    }

    bool PrefixStateCache::insert(
        PrefixBlockHandle handle,
        PrefixBlockHandle device_hot_handle)
    {
        const bool install_device_hot = device_hot_handle.valid();
        if (install_device_hot &&
            (device_hot_handle.tier != PrefixStorageTier::DeviceHot ||
             device_hot_handle.key != handle.key ||
             !device_hot_handle.layout.compatiblePayloadShape(handle.layout) ||
             !device_hot_backend_))
        {
            if (device_hot_backend_)
                device_hot_backend_->release(device_hot_handle);
            return false;
        }

        if (!insertResident(std::move(handle), /*count_store=*/true))
        {
            if (install_device_hot && device_hot_backend_)
                device_hot_backend_->release(device_hot_handle);
            return false;
        }

        if (install_device_hot)
        {
            if (!installDeviceHotCopy(
                    std::move(device_hot_handle),
                    /*repromotion=*/false))
            {
                return false;
            }
        }
        return true;
    }

    bool PrefixStateCache::insertResident(PrefixBlockHandle handle,
                                          bool count_store,
                                          bool preserve_disk_entry)
    {
        publishCompletedPersistence();
        if (!handle.valid() || handle.total_bytes > ram_budget_bytes_ || !ram_backend_)
        {
            return false;
        }

        const PrefixCacheKey resident_key = handle.key;
        auto existing = entries_.find(resident_key);
        if (existing != entries_.end())
        {
            /*
             * Replacing an allocation after it has already been created is
             * unsafe because RamPrefixStorageBackend accounts owners by key.
             * Production replacement must pass through prepareInsert() before
             * allocating the new handle.
             */
            return false;
        }
        if (!preserve_disk_entry &&
            (disk_entries_.find(resident_key) != disk_entries_.end() ||
             device_hot_entries_.find(resident_key) !=
                 device_hot_entries_.end()))
        {
            return false;
        }

        if (evictUntilFits(handle.total_bytes) != PrefixRamInsertPreparation::Prepared)
        {
            return false;
        }

        lru_.push_front(handle.key);
        PrefixStateBlock block;
        block.handle = std::move(handle);
        block.cached_tokens = block.handle.key.token_start + block.handle.key.token_count;
        block.block_index = block.handle.key.block_index;
        block.last_access_tick = ++tick_;

        Entry entry;
        entry.block = std::move(block);
        entry.lru_it = lru_.begin();
        used_bytes_ += entry.block.handle.total_bytes;
        stats_.inserts++;
        if (count_store)
        {
            stats_.stores++;
        }
        stats_.ram_bytes = used_bytes_;
        addResidentStats(entry.block.handle);
        entries_.emplace(entry.block.handle.key, std::move(entry));
        return true;
    }

    bool PrefixStateCache::installDiscoveredDiskEntries(
        uint64_t fingerprint,
        const PrefixPayloadLayout &layout,
        std::string *error)
    {
        if (!disk_backend_)
            return true;

        const auto discovered =
            disk_backend_->compatibleEntries(fingerprint, layout, error);
        for (const PrefixBlockHandle &handle : discovered)
        {
            if (!handle.valid() || handle.tier != PrefixStorageTier::Disk)
                continue;
            auto [iterator, inserted] =
                disk_entries_.emplace(handle.key, handle);
            if (!inserted)
            {
                const uint64_t previous = iterator->second.total_bytes;
                stats_.disk_bytes =
                    stats_.disk_bytes > previous
                        ? stats_.disk_bytes - previous
                        : 0;
                iterator->second = handle;
            }
            stats_.disk_bytes += handle.total_bytes;
        }
        return true;
    }

    bool PrefixStateCache::installDeviceHotCopy(
        PrefixBlockHandle device_hot_handle,
        bool repromotion)
    {
        if (!device_hot_backend_ ||
            !device_hot_handle.valid() ||
            device_hot_handle.tier != PrefixStorageTier::DeviceHot)
        {
            return false;
        }

        /*
         * Replacing the same key is an installation detail, not capacity
         * pressure. Keep the eviction counter reserved for true LRU demotions.
         */
        removeDeviceHotEntry(
            device_hot_handle.key,
            /*capacity_eviction=*/false);
        device_hot_lru_.push_front(device_hot_handle.key);
        device_hot_entries_.emplace(
            device_hot_handle.key,
            std::move(device_hot_handle));
        ++stats_.promotions;
        ++stats_.device_hot_promotions;
        if (repromotion)
            ++stats_.device_hot_repromotions;
        stats_.device_hot_bytes =
            static_cast<uint64_t>(device_hot_backend_->usedBytes());
        stats_.device_bytes = stats_.device_hot_bytes;
        return true;
    }

    std::optional<PrefixBlockHandle> PrefixStateCache::find(const PrefixCacheKey &key)
    {
        publishCompletedPersistence();
        if (pending_hydration_ && pending_hydration_->diskHandle().key != key)
            pending_hydration_.reset();
        stats_.lookups++;
        auto hot_it = device_hot_entries_.find(key);
        if (hot_it != device_hot_entries_.end())
        {
            ++stats_.hits;
            ++stats_.device_hot_direct_hits;
            touchDeviceHot(key);
            auto resident_it = entries_.find(key);
            if (resident_it != entries_.end())
                touch(resident_it->second);
            return hot_it->second;
        }

        auto it = entries_.find(key);
        if (it == entries_.end())
        {
            auto disk_it = disk_entries_.find(key);
            if ((!pending_hydration_ && disk_it == disk_entries_.end()) ||
                !disk_backend_ || !ram_backend_)
            {
                stats_.misses++;
                return std::nullopt;
            }

            std::string error;
            if (!pending_hydration_)
                pending_hydration_ = disk_backend_->beginVerifiedHydration(
                    key,
                    disk_it->second.layout,
                    &error);
            if (!pending_hydration_)
            {
                ++stats_.disk_read_failures;
                stats_.misses++;
                removeDiskEntry(key);
                return std::nullopt;
            }

            /*
             * The first pass proved the durable bytes through one bounded,
             * authority-owned window.  Only now may a victim leave RAM.  The
             * second pass below fills the final admitted RAM allocation
             * directly and verifies it again, eliminating both a full-block
             * staging peak and a redundant whole-block memcpy.
             */
            auto &hydration = *pending_hydration_;
            if (completePendingCapacity(hydration.totalBytes()) !=
                PrefixRamInsertPreparation::Prepared)
            {
                // Known durable writes complete before a selected restore is
                // judged unavailable. External request aliases have no archive
                // completion edge; keep both their charge and the verified inode
                // rather than discarding work or claiming a corrupt record.
                stats_.misses++;
                return std::nullopt;
            }

            auto concrete_ram =
                std::dynamic_pointer_cast<RamPrefixStorageBackend>(
                    ram_backend_);
            if (!concrete_ram)
            {
                ++stats_.disk_read_failures;
                stats_.misses++;
                return std::nullopt;
            }
            PrefixBlockHandle hydrated;
            if (!disk_backend_->hydrateVerified(
                    hydration,
                    *concrete_ram,
                    &hydrated,
                    &error))
            {
                ++stats_.disk_read_failures;
                ++stats_.misses;
                /*
                 * Another archive writer may have replaced or evicted the
                 * verified record while RAM capacity was prepared.  Never
                 * publish different bytes under the stale local index.
                 */
                removeDiskEntry(key);
                pending_hydration_.reset();
                return std::nullopt;
            }
            pending_hydration_.reset();

            if (!insertResident(hydrated, /*count_store=*/false, /*preserve_disk_entry=*/true))
            {
                ram_backend_->release(hydrated);
                ++stats_.disk_read_failures;
                stats_.misses++;
                return std::nullopt;
            }

            ++stats_.hits;
            ++stats_.disk_hydrations;
            ++stats_.promotions;
            auto hydrated_it = entries_.find(key);
            return hydrated_it == entries_.end()
                       ? std::optional<PrefixBlockHandle>{}
                       : std::optional<PrefixBlockHandle>{hydrated_it->second.block.handle};
        }
        stats_.hits++;
        touch(it->second);
        touchDeviceHot(key);
        return it->second.block.handle;
    }

    std::optional<PrefixBlockHandle> PrefixStateCache::findLongestTokenPrefix(
        uint64_t fingerprint,
        uint64_t parent_hash,
        int block_index,
        int token_start,
        const std::vector<int32_t> &tokens)
    {
        if (fingerprint == 0 || block_index < 0 || token_start < 0 ||
            tokens.empty())
        {
            return std::nullopt;
        }

        for (size_t token_count = tokens.size(); token_count > 0; --token_count)
        {
            std::vector<int32_t> candidate_tokens(
                tokens.begin(),
                tokens.begin() + static_cast<std::ptrdiff_t>(token_count));
            const PrefixCacheKey candidate = makePrefixCacheKey(
                fingerprint,
                parent_hash,
                block_index,
                token_start,
                candidate_tokens);
            if (contains(candidate))
            {
                return find(candidate);
            }
        }

        /*
         * Keep one request-level miss in the existing statistics vocabulary.
         * find() also preserves future behavior if a backend learns to resolve
         * an exact key without advertising it through contains().
         */
        return find(makePrefixCacheKey(
            fingerprint,
            parent_hash,
            block_index,
            token_start,
            tokens));
    }

    bool PrefixStateCache::contains(const PrefixCacheKey &key) const
    {
        return entries_.find(key) != entries_.end() ||
               device_hot_entries_.find(key) != device_hot_entries_.end() ||
               disk_entries_.find(key) != disk_entries_.end() ||
               (pending_hydration_ && pending_hydration_->diskHandle().key == key);
    }

    bool PrefixStateCache::retain(const PrefixCacheKey &key)
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
        {
            return false;
        }
        ++it->second.block.ref_count;
        touch(it->second);
        return true;
    }

    bool PrefixStateCache::release(const PrefixCacheKey &key)
    {
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.block.ref_count == 0)
        {
            return false;
        }
        --it->second.block.ref_count;
        return true;
    }

    bool PrefixStateCache::erase(const PrefixCacheKey &key)
    {
        publishCompletedPersistence();
        const bool retired_promotion =
            pending_hydration_ && pending_hydration_->diskHandle().key == key;
        if (retired_promotion)
            pending_hydration_.reset();
        auto it = entries_.find(key);
        if (it != entries_.end() && it->second.block.ref_count > 0)
        {
            return false;
        }
        bool erased = false;
        // A queued older put must precede a tombstone, not resurrect this key
        // after prepareInsert() publishes a richer terminal snapshot.
        if (retired_promotion || it != entries_.end() || disk_entries_.contains(key))
            retireArchiveKey(key);
        if (it != entries_.end())
        {
            erased = evictResident(key);
        }
        const bool erased_device_hot =
            removeDeviceHotEntry(key, /*capacity_eviction=*/false);
        const bool erased_disk = disk_entries_.contains(key);
        forgetDiskEntry(key);
        return retired_promotion || erased_disk || erased_device_hot || erased;
    }

    bool PrefixStateCache::clear()
    {
        publishCompletedPersistence();
        for (const auto &[key, entry] : entries_)
        {
            (void)key;
            if (entry.block.ref_count > 0)
            {
                return false;
            }
        }

        pending_hydration_.reset();

        for (auto &[key, entry] : entries_)
        {
            retireArchiveKey(key);
            // One key needs one tombstone even when both tiers retain it.
            forgetDiskEntry(key);
            subtractResidentStats(entry.block.handle);
            if (ram_backend_)
            {
                ram_backend_->release(entry.block.handle);
            }
        }
        entries_.clear();
        lru_.clear();
        used_bytes_ = 0;
        stats_.ram_bytes = 0;
        stats_.hybrid_state_bytes = 0;
        stats_.mtp_state_bytes = 0;

        for (auto &[key, handle] : device_hot_entries_)
        {
            (void)key;
            if (device_hot_backend_)
            {
                device_hot_backend_->release(handle);
            }
        }
        device_hot_entries_.clear();
        device_hot_lru_.clear();
        stats_.device_hot_bytes = 0;
        stats_.device_bytes = 0;

        for (auto &[key, handle] : disk_entries_)
        {
            (void)handle;
            retireArchiveKey(key);
        }
        disk_entries_.clear();
        stats_.disk_bytes = 0;
        // clear() is the explicit administrative purge, not request reset.
        // Its durable tombstones must survive restart before it returns.
        if (disk_backend_)
        {
            std::string error;
            if (!disk_backend_->waitForPersistence(&error))
                throw std::runtime_error(error);
            publishCompletedPersistence();
        }
        return true;
    }

    std::optional<PrefixFingerprintRebase>
    PrefixStateCache::rebaseFingerprint(
        uint64_t previous_fingerprint,
        uint64_t active_fingerprint)
    {
        publishCompletedPersistence();
        if (previous_fingerprint == 0 || active_fingerprint == 0 ||
            previous_fingerprint == active_fingerprint)
        {
            return std::nullopt;
        }

        /*
         * Explicit retain()/release() leases predate shared handle ownership
         * but remain a supported unit-level contract. Preflight every stale
         * resident before mutating any tier so a busy transition is all-or-
         * nothing rather than a partially rebased cache.
         */
        for (const auto &[key, entry] : entries_)
        {
            if (key.fingerprint != active_fingerprint &&
                entry.block.ref_count > 0)
            {
                return std::nullopt;
            }
        }

        if (pending_hydration_ &&
            pending_hydration_->diskHandle().key.fingerprint != active_fingerprint)
            pending_hydration_.reset();

        PrefixFingerprintRebase transition{
            .previous_fingerprint = previous_fingerprint,
            .active_fingerprint = active_fingerprint,
        };

        std::vector<PrefixCacheKey> stale_device_keys;
        stale_device_keys.reserve(device_hot_entries_.size());
        for (const auto &[key, handle] : device_hot_entries_)
        {
            (void)handle;
            if (key.fingerprint != active_fingerprint)
                stale_device_keys.push_back(key);
        }
        for (const PrefixCacheKey &key : stale_device_keys)
        {
            const auto found = device_hot_entries_.find(key);
            if (found == device_hot_entries_.end())
                continue;
            transition.released_device_bytes += found->second.total_bytes;
            if (removeDeviceHotEntry(key, /*capacity_eviction=*/false))
                ++transition.invalidated_device_entries;
        }

        std::vector<PrefixCacheKey> stale_ram_keys;
        stale_ram_keys.reserve(entries_.size());
        for (const auto &[key, entry] : entries_)
        {
            (void)entry;
            if (key.fingerprint != active_fingerprint)
                stale_ram_keys.push_back(key);
        }
        for (const PrefixCacheKey &key : stale_ram_keys)
        {
            const auto found = entries_.find(key);
            if (found == entries_.end())
                continue;
            transition.released_ram_bytes +=
                found->second.block.handle.total_bytes;
            if (evictResident(key))
                ++transition.invalidated_ram_entries;
        }

        std::vector<PrefixCacheKey> stale_disk_keys;
        stale_disk_keys.reserve(disk_entries_.size());
        for (const auto &[key, handle] : disk_entries_)
        {
            (void)handle;
            if (key.fingerprint != active_fingerprint)
                stale_disk_keys.push_back(key);
        }
        for (const PrefixCacheKey &key : stale_disk_keys)
        {
            if (disk_entries_.find(key) == disk_entries_.end())
                continue;
            forgetDiskEntry(key);
            ++transition.unindexed_disk_entries;
        }

        ++stats_.fingerprint_rebases;
        stats_.fingerprint_invalidated_ram_entries +=
            transition.invalidated_ram_entries;
        stats_.fingerprint_invalidated_device_entries +=
            transition.invalidated_device_entries;
        stats_.fingerprint_unindexed_disk_entries +=
            transition.unindexed_disk_entries;
        return transition;
    }

    PrefixRamInsertPreparation PrefixStateCache::prepareInsert(
        const PrefixCacheKey &key,
        size_t incoming_bytes)
    {
        publishCompletedPersistence();
        if (!key.valid() || incoming_bytes == 0 ||
            incoming_bytes > ram_budget_bytes_ || !ram_backend_)
        {
            return PrefixRamInsertPreparation::Error;
        }

        const auto resident = entries_.find(key);
        if (resident != entries_.end() &&
            resident->second.block.ref_count > 0)
        {
            return PrefixRamInsertPreparation::Error;
        }

        if (contains(key) && !erase(key))
        {
            return PrefixRamInsertPreparation::Error;
        }
        return prepareCapacity(incoming_bytes);
    }

    void PrefixStateCache::completeInsertPreparation(
        const PrefixCacheKey &key,
        size_t incoming_bytes)
    {
        auto preparation = prepareInsert(key, incoming_bytes);
        if (preparation == PrefixRamInsertPreparation::Busy)
            preparation = completePendingCapacity(incoming_bytes);
        if (preparation != PrefixRamInsertPreparation::Prepared)
        {
            const char *reason = preparation == PrefixRamInsertPreparation::Busy
                ? "retained physical owners have no pending archive completion"
                : "invalid geometry or unsafe replacement";
            throw std::runtime_error(
                "required prefix RAM publication rejected key=" + key.toHex() +
                " bytes=" + std::to_string(incoming_bytes) + " reason=" + reason);
        }
    }

    PrefixRamInsertPreparation PrefixStateCache::completePendingCapacity(
        size_t incoming_bytes)
    {
        auto preparation = prepareCapacity(incoming_bytes);
        while (preparation == PrefixRamInsertPreparation::Busy &&
               !pending_persistence_.empty())
        {
            // Cache metadata is single-thread-owned; the writer only publishes
            // this immutable receipt. Copy the ticket before publication can
            // retire the deque's front. Never wait for later unrelated work.
            const auto dependency = pending_persistence_.front().ticket;
            PerfStatsCollector::addCounter(
                "prefix_cache", "ram_archive_capacity_dependency_waits", 1.0,
                "archive_publication", "CPU",
                {{"receipt", std::to_string(dependency.identity())},
                 {"incoming_bytes", std::to_string(incoming_bytes)}});
            (void)dependency.waitForPublication();
            publishCompletedPersistence();
            preparation = prepareCapacity(incoming_bytes);
        }
        return preparation;
    }

    PrefixRamInsertPreparation PrefixStateCache::prepareCapacity(size_t incoming_bytes)
    {
        publishCompletedPersistence();
        if (incoming_bytes == 0 || incoming_bytes > ram_budget_bytes_ || !ram_backend_)
            return PrefixRamInsertPreparation::Error;
        const auto preparation = evictUntilFits(incoming_bytes);
        if (preparation != PrefixRamInsertPreparation::Prepared)
            return preparation;

        return evictUntilPhysicallyFits(incoming_bytes);
    }

    PrefixRamInsertPreparation PrefixStateCache::prepareHarvest(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        const PrefixHarvestSchedule &schedule,
        const PrefixPayloadLayout &layout,
        size_t runtime_capacity)
    {
        publishCompletedPersistence();
        if (!admission.supported || !admission.cache_enabled ||
            admission.fingerprint_key == 0u || layout.block_size <= 0 ||
            tokens.size() != static_cast<size_t>(schedule.promptTokens()) ||
            layout.block_size != admission.block_size)
            return PrefixRamInsertPreparation::Error;

        std::unordered_map<PrefixCacheKey, size_t, PrefixCacheKeyHasher> publications;
        const auto plan_frontier = [&](int frontier)
        {
            uint64_t parent_hash = 0;
            for (int start = 0, block = 0; start < frontier;
                 start += layout.block_size, ++block)
            {
                const int end = std::min(frontier, start + layout.block_size);
                const auto key = makePrefixCacheKey(
                    admission.fingerprint_key, parent_hash, block, start,
                    {tokens.begin() + start, tokens.begin() + end});
                parent_hash = key.stableHash();
                const bool terminal = end == frontier;
                if ((!terminal && layout.organization() ==
                        PrefixPayloadOrganization::RecurrentCheckpoint) ||
                    (contains(key) && (!terminal ||
                        admission.terminalHarvestDisposition(key, frontier) ==
                            PrefixTerminalHarvestDisposition::ReuseAdmittedArchive)))
                    continue;
                auto record_layout = layout;
                if (!terminal)
                {
                    record_layout.includes_hybrid_state = false;
                    record_layout.includes_terminal_hidden = false;
                    record_layout.includes_terminal_logits = false;
                }
                const size_t extra = terminal ? runtime_capacity : 0u;
                const size_t payload = record_layout.totalBytes();
                if (payload > std::numeric_limits<size_t>::max() - extra)
                    throw std::overflow_error("prefix publication payload overflow");
                auto &bytes = publications[key];
                // A reusable terminal becomes a nonterminal in the final
                // chain. Keep its richer immutable image, not two allocations.
                bytes = std::max(bytes, payload + extra);
            }
        };
        if (const auto checkpoint = schedule.reusableCheckpoint())
            plan_frontier(*checkpoint);
        plan_frontier(schedule.promptTokens());
        size_t incoming = 0u;
        for (const auto &[key, bytes] : publications)
        {
            if (bytes > std::numeric_limits<size_t>::max() - incoming)
                throw std::overflow_error("prefix publication batch overflow");
            incoming += bytes;
        }
        if (incoming == 0u)
            return PrefixRamInsertPreparation::Prepared;
        // A prompt larger than the bounded RAM tier is streamed through that
        // tier by ordinary per-record admission. Prepare at most its admitted
        // physical envelope; do not invent another reservation or enlarge it.
        return prepareCapacity(std::min(incoming, ram_budget_bytes_));
    }

    PrefixRamInsertPreparation PrefixStateCache::evictUntilPhysicallyFits(
        size_t incoming_bytes)
    {
        /*
         * Logical LRU bytes can fall before physical bytes: a request-held
         * handle or an in-flight DMA retains its admitted child lease after
         * cache eviction. Continue pressure until the backend's physical
         * authority can actually admit the new archive. Never let the caller
         * discover that mismatch after it has started copying the payload.
         */
        auto ram = std::dynamic_pointer_cast<RamPrefixStorageBackend>(ram_backend_);
        if (!ram)
            return PrefixRamInsertPreparation::Error;
        // Retire an orphaned hot replica first. Its CPU runtime-state alias
        // can be the only reason a logically evicted RAM payload is still
        // charged; persisting unrelated RAM cannot release that owner.
        const auto hot_candidates = device_hot_lru_;
        for (auto candidate = hot_candidates.rbegin();
             candidate != hot_candidates.rend() && !ram->canStore(incoming_bytes);
             ++candidate)
        {
            if (entries_.contains(*candidate))
                continue;
            if (!removeDeviceHotEntry(*candidate, /*capacity_eviction=*/true))
                return PrefixRamInsertPreparation::Error;
        }

        size_t pending_victim_bytes = 0u;
        const auto candidates = keysMostRecentFirst();
        for (auto candidate = candidates.rbegin(); candidate != candidates.rend(); ++candidate)
        {
            const auto available = ram->availableAllocationBytes();
            if (available >= incoming_bytes ||
                pending_victim_bytes >= incoming_bytes - available)
                break;
            const auto entry = entries_.find(*candidate);
            if (entry == entries_.end() || entry->second.block.ref_count > 0)
                continue;
            const auto persistence = persistResidentToDisk(entry->second);
            if (persistence == PrefixRamInsertPreparation::Error)
                return persistence;
            if (persistence == PrefixRamInsertPreparation::Busy)
            {
                pending_victim_bytes += entry->second.block.handle.total_bytes;
                continue;
            }
            if (!evictResident(*candidate))
                return PrefixRamInsertPreparation::Error;
            ++stats_.evictions;
            if (device_hot_entries_.contains(*candidate) &&
                !removeDeviceHotEntry(*candidate, /*capacity_eviction=*/true))
                return PrefixRamInsertPreparation::Error;
        }
        return ram->canStore(incoming_bytes) ? PrefixRamInsertPreparation::Prepared
                                            : PrefixRamInsertPreparation::Busy;
    }

    PrefixDeviceHotLeaseResult PrefixStateCache::prepareDeviceHotCopy(
        const PrefixBlockHandle &ram_archive,
        void *producer_stream,
        PrefixBlockHandle *device_hot_handle)
    {
        if (!device_hot_handle || !producer_stream || !device_hot_backend_ ||
            !ram_archive.valid())
        {
            return PrefixDeviceHotLeaseResult::Error;
        }
        if (!device_hot_backend_->capacityEligible(
                ram_archive.total_bytes))
        {
            return PrefixDeviceHotLeaseResult::Ineligible;
        }

        /*
         * Evict cache-owned LRU handles until an arena slot is physically
         * reusable. A request may still retain aliases to every evicted slot;
         * exhausting the LRU in that state is Busy, not an allocator failure.
         */
        while (!device_hot_backend_->canStore(
            ram_archive.total_bytes))
        {
            if (device_hot_lru_.empty())
                return PrefixDeviceHotLeaseResult::Busy;
            const PrefixCacheKey victim = device_hot_lru_.back();
            if (!removeDeviceHotEntry(
                    victim,
                    /*capacity_eviction=*/true))
            {
                return PrefixDeviceHotLeaseResult::Error;
            }
        }

        std::string error;
        if (device_hot_backend_->allocateDeviceBlock(
                ram_archive,
                producer_stream,
                device_hot_handle,
                &error))
        {
            return PrefixDeviceHotLeaseResult::Acquired;
        }
        return device_hot_backend_->canStore(ram_archive.total_bytes)
                   ? PrefixDeviceHotLeaseResult::Error
                   : PrefixDeviceHotLeaseResult::Busy;
    }

    bool PrefixStateCache::deviceHotCapacityEligible(size_t bytes) const
    {
        return device_hot_backend_ &&
               bytes > 0 &&
               device_hot_backend_->capacityEligible(bytes);
    }

    std::optional<PrefixBlockHandle> PrefixStateCache::deviceHotCopy(
        const PrefixCacheKey &key) const
    {
        const auto it = device_hot_entries_.find(key);
        return it == device_hot_entries_.end()
                   ? std::optional<PrefixBlockHandle>{}
                   : std::optional<PrefixBlockHandle>{it->second};
    }

    void PrefixStateCache::recordRequestLookup(int requested_tokens,
                                               int matched_tokens,
                                               int matched_blocks)
    {
        const int clamped_requested = std::max(0, requested_tokens);
        const int clamped_matched = std::max(0, std::min(matched_tokens, clamped_requested));
        if (clamped_matched > 0 && clamped_matched < clamped_requested)
        {
            ++stats_.partial_hits;
        }
        stats_.matched_tokens += static_cast<uint64_t>(clamped_matched);
        stats_.matched_blocks += static_cast<uint64_t>(std::max(0, matched_blocks));
    }

    void PrefixStateCache::recordTerminalStateHit()
    {
        ++stats_.terminal_state_hits;
    }

    std::vector<PrefixCacheKey> PrefixStateCache::keysMostRecentFirst() const
    {
        return std::vector<PrefixCacheKey>(lru_.begin(), lru_.end());
    }

    bool PrefixStateCache::isRamResident(const PrefixCacheKey &key) const
    {
        return entries_.find(key) != entries_.end();
    }

    bool PrefixStateCache::isDeviceHotResident(const PrefixCacheKey &key) const
    {
        return device_hot_entries_.find(key) != device_hot_entries_.end();
    }

    bool PrefixStateCache::isDiskResident(const PrefixCacheKey &key) const
    {
        return disk_entries_.find(key) != disk_entries_.end();
    }

    PrefixRamInsertPreparation PrefixStateCache::evictUntilFits(size_t incoming_bytes)
    {
        if (incoming_bytes > ram_budget_bytes_)
        {
            return PrefixRamInsertPreparation::Error;
        }
        const size_t resident_limit = ram_budget_bytes_ - incoming_bytes;
        size_t pending_victim_bytes = 0u;
        // This is a one-shot victim selection, not free-memory accounting.
        // Pending leases still consume their full PMA capacity. Select the
        // complete necessary LRU set now so one inference interval can overlap
        // every write, instead of discovering one additional victim at harvest.
        const auto candidates = keysMostRecentFirst();
        for (auto candidate = candidates.rbegin();
             candidate != candidates.rend() &&
             used_bytes_ > resident_limit &&
             pending_victim_bytes < used_bytes_ - resident_limit;
             ++candidate)
        {
            auto entry = entries_.find(*candidate);
            if (entry == entries_.end() || entry->second.block.ref_count > 0)
                continue;
            const auto persistence = persistResidentToDisk(entry->second);
            if (persistence == PrefixRamInsertPreparation::Error)
                return persistence;
            if (persistence == PrefixRamInsertPreparation::Busy)
            {
                pending_victim_bytes += entry->second.block.handle.total_bytes;
                continue;
            }
            if (!evictResident(*candidate))
                return PrefixRamInsertPreparation::Error;
            ++stats_.evictions;
        }
        return used_bytes_ <= resident_limit
                   ? PrefixRamInsertPreparation::Prepared
                   : PrefixRamInsertPreparation::Busy;
    }

    bool PrefixStateCache::removeDeviceHotEntry(
        const PrefixCacheKey &key,
        bool capacity_eviction)
    {
        auto it = device_hot_entries_.find(key);
        if (it == device_hot_entries_.end())
        {
            return false;
        }

        if (device_hot_backend_)
        {
            device_hot_backend_->release(it->second);
        }
        device_hot_entries_.erase(it);
        for (auto lru_it = device_hot_lru_.begin(); lru_it != device_hot_lru_.end(); ++lru_it)
        {
            if (*lru_it == key)
            {
                device_hot_lru_.erase(lru_it);
                break;
            }
        }
        stats_.device_hot_bytes = device_hot_backend_
                                      ? static_cast<uint64_t>(device_hot_backend_->usedBytes())
                                      : 0;
        stats_.device_bytes = stats_.device_hot_bytes;
        if (capacity_eviction)
            ++stats_.device_hot_evictions;
        return true;
    }

    void PrefixStateCache::touchDeviceHot(const PrefixCacheKey &key)
    {
        for (auto it = device_hot_lru_.begin(); it != device_hot_lru_.end(); ++it)
        {
            if (*it == key)
            {
                device_hot_lru_.erase(it);
                device_hot_lru_.push_front(key);
                return;
            }
        }
    }

    bool PrefixStateCache::evictResident(const PrefixCacheKey &key)
    {
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.block.ref_count > 0 || !ram_backend_)
        {
            return false;
        }
        subtractResidentStats(it->second.block.handle);
        used_bytes_ -= std::min(used_bytes_, it->second.block.handle.total_bytes);
        ram_backend_->release(it->second.block.handle);
        lru_.erase(it->second.lru_it);
        entries_.erase(it);
        stats_.ram_bytes = used_bytes_;
        return true;
    }

    PrefixRamInsertPreparation PrefixStateCache::persistResidentToDisk(Entry &entry)
    {
        if (!disk_backend_)
        {
            return PrefixRamInsertPreparation::Prepared;
        }

        const PrefixBlockHandle &handle = entry.block.handle;
        if (disk_entries_.find(handle.key) != disk_entries_.end())
        {
            return PrefixRamInsertPreparation::Prepared;
        }

        if (!entry.persistence.valid())
        {
            entry.persistence = disk_backend_->scheduleWrite(handle);
            pending_persistence_.push_back({handle.key, entry.persistence});
        }
        return PrefixRamInsertPreparation::Busy;
    }

    void PrefixStateCache::publishCompletedPersistence()
    {
        while (!pending_persistence_.empty())
        {
            const auto &pending = pending_persistence_.front();
            const auto publication = pending.ticket.publication();
            if (!publication)
                return;
            if (const auto *failed = std::get_if<PrefixArchivePersistenceFailure>(publication.get()))
            {
                ++stats_.disk_write_failures;
                throw std::runtime_error("background prefix archive publication failed: " + failed->diagnostic);
            }
            if (const auto *written = std::get_if<PrefixArchiveWritePublication>(publication.get()))
            {
                for (const PrefixCacheKey &evicted : written->evicted_keys)
                {
                    if (disk_entries_.contains(evicted))
                    {
                        forgetDiskEntry(evicted);
                        ++stats_.disk_evictions;
                    }
                }
                const auto resident = entries_.find(pending.key);
                if (resident != entries_.end() &&
                    resident->second.persistence.identity() == pending.ticket.identity())
                {
                    forgetDiskEntry(pending.key);
                    stats_.disk_bytes += written->disk_handle.total_bytes;
                    disk_entries_.emplace(pending.key, written->disk_handle);
                    resident->second.persistence = {};
                    ++stats_.ram_to_disk_demotions;
                }
            }
            pending_persistence_.pop_front();
        }
    }

    void PrefixStateCache::retireArchiveKey(const PrefixCacheKey &key)
    {
        if (!disk_backend_)
            return;
        pending_persistence_.push_back({key, disk_backend_->scheduleRetirement(key)});
    }

    bool PrefixStateCache::removeDiskEntry(const PrefixCacheKey &key)
    {
        auto it = disk_entries_.find(key);
        if (it == disk_entries_.end())
        {
            return false;
        }

        retireArchiveKey(key);
        const uint64_t bytes = static_cast<uint64_t>(it->second.total_bytes);
        stats_.disk_bytes = stats_.disk_bytes > bytes ? stats_.disk_bytes - bytes : 0;
        disk_entries_.erase(it);
        return true;
    }

    void PrefixStateCache::forgetDiskEntry(const PrefixCacheKey &key)
    {
        auto it = disk_entries_.find(key);
        if (it == disk_entries_.end())
            return;
        const uint64_t bytes = static_cast<uint64_t>(it->second.total_bytes);
        stats_.disk_bytes =
            stats_.disk_bytes > bytes ? stats_.disk_bytes - bytes : 0;
        disk_entries_.erase(it);
    }

    void PrefixStateCache::touch(Entry &entry)
    {
        lru_.erase(entry.lru_it);
        lru_.push_front(entry.block.handle.key);
        entry.lru_it = lru_.begin();
        entry.block.last_access_tick = ++tick_;
    }

    void PrefixStateCache::addResidentStats(const PrefixBlockHandle &handle)
    {
        if (handle.layout.includes_hybrid_state)
        {
            stats_.hybrid_state_bytes += static_cast<uint64_t>(handle.layout.hybrid_state_bytes);
        }
        if (handle.layout.includes_mtp_state)
        {
            stats_.mtp_state_bytes += static_cast<uint64_t>(handle.layout.mtpKVBytes());
        }
    }

    void PrefixStateCache::subtractResidentStats(const PrefixBlockHandle &handle)
    {
        if (handle.layout.includes_hybrid_state)
        {
            const uint64_t bytes = static_cast<uint64_t>(handle.layout.hybrid_state_bytes);
            stats_.hybrid_state_bytes = stats_.hybrid_state_bytes > bytes
                                            ? stats_.hybrid_state_bytes - bytes
                                            : 0;
        }
        if (handle.layout.includes_mtp_state)
        {
            const uint64_t bytes = static_cast<uint64_t>(handle.layout.mtpKVBytes());
            stats_.mtp_state_bytes = stats_.mtp_state_bytes > bytes
                                         ? stats_.mtp_state_bytes - bytes
                                         : 0;
        }
    }

} // namespace llaminar2
