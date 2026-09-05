/**
 * @file PrefixStateCache.cpp
 * @brief Implements durable prefix-block indexing and tier-aware LRU lookup.
 *
 * Metadata probing is separated from payload acquisition so a longest-prefix
 * query records one semantic hit or miss even when it examines several token
 * widths. The final find() remains the sole authority for disk hydration, LRU
 * movement, and shared payload ownership.
 */

#include "execution/prefix_cache/PrefixStateCache.h"

#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"
#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include <algorithm>
#include <cstddef>
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

        if (!evictUntilFits(handle.total_bytes))
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
            if (disk_it == disk_entries_.end() || !disk_backend_ || !ram_backend_)
            {
                stats_.misses++;
                return std::nullopt;
            }

            const PrefixBlockHandle disk_handle = disk_it->second;
            std::string error;
            auto hydration = disk_backend_->beginVerifiedHydration(
                    key,
                    disk_handle.layout,
                    &error);
            if (!hydration)
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
            if (!evictUntilFits(hydration->totalBytes()))
            {
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
                    *hydration,
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
                return std::nullopt;
            }

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
               disk_entries_.find(key) != disk_entries_.end();
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
        auto it = entries_.find(key);
        if (it != entries_.end() && it->second.block.ref_count > 0)
        {
            return false;
        }
        bool erased = false;
        if (it != entries_.end())
        {
            erased = evictResident(key);
        }
        const bool erased_device_hot =
            removeDeviceHotEntry(key, /*capacity_eviction=*/false);
        const bool erased_disk = removeDiskEntry(key);
        return erased_disk || erased_device_hot || erased;
    }

    bool PrefixStateCache::clear()
    {
        for (const auto &[key, entry] : entries_)
        {
            (void)key;
            if (entry.block.ref_count > 0)
            {
                return false;
            }
        }

        for (auto &[key, entry] : entries_)
        {
            (void)key;
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
            (void)key;
            if (disk_backend_)
            {
                disk_backend_->release(handle);
            }
        }
        disk_entries_.clear();
        stats_.disk_bytes = 0;
        return true;
    }

    std::optional<PrefixFingerprintRebase>
    PrefixStateCache::rebaseFingerprint(
        uint64_t previous_fingerprint,
        uint64_t active_fingerprint)
    {
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

    bool PrefixStateCache::prepareInsert(
        const PrefixCacheKey &key,
        size_t incoming_bytes)
    {
        if (!key.valid() || incoming_bytes == 0 ||
            incoming_bytes > ram_budget_bytes_)
        {
            return false;
        }

        const auto resident = entries_.find(key);
        if (resident != entries_.end() &&
            resident->second.block.ref_count > 0)
        {
            return false;
        }

        if (contains(key) && !erase(key))
        {
            return false;
        }
        return evictUntilFits(incoming_bytes);
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

    bool PrefixStateCache::evictUntilFits(size_t incoming_bytes)
    {
        if (incoming_bytes > ram_budget_bytes_)
        {
            return false;
        }
        while (used_bytes_ + incoming_bytes > ram_budget_bytes_)
        {
            bool evicted = false;
            for (auto it = lru_.rbegin(); it != lru_.rend(); ++it)
            {
                auto entry_it = entries_.find(*it);
                if (entry_it == entries_.end() || entry_it->second.block.ref_count > 0)
                {
                    continue;
                }
                const PrefixCacheKey victim = *it;
                if (!persistResidentToDisk(entry_it->second))
                {
                    continue;
                }
                evictResident(victim);
                ++stats_.evictions;
                evicted = true;
                break;
            }
            if (!evicted)
            {
                return false;
            }
        }
        return true;
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

    bool PrefixStateCache::persistResidentToDisk(const Entry &entry)
    {
        if (!disk_backend_)
        {
            return true;
        }

        const PrefixBlockHandle &handle = entry.block.handle;
        if (disk_entries_.find(handle.key) != disk_entries_.end())
        {
            return true;
        }

        PrefixBlockHandle disk_handle;
        std::vector<PrefixCacheKey> evicted_keys;
        std::string error;
        if (!disk_backend_->writeBlock(
                handle,
                &disk_handle,
                &evicted_keys,
                &error))
        {
            ++stats_.disk_write_failures;
            return false;
        }

        for (const PrefixCacheKey &evicted : evicted_keys)
        {
            forgetDiskEntry(evicted);
            ++stats_.disk_evictions;
        }
        stats_.disk_bytes += disk_handle.total_bytes;
        disk_entries_.emplace(disk_handle.key, std::move(disk_handle));
        ++stats_.ram_to_disk_demotions;
        return true;
    }

    bool PrefixStateCache::removeDiskEntry(const PrefixCacheKey &key)
    {
        auto it = disk_entries_.find(key);
        if (it == disk_entries_.end())
        {
            return false;
        }

        if (disk_backend_)
        {
            disk_backend_->release(it->second);
        }
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
