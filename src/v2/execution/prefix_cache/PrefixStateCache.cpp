#include "execution/prefix_cache/PrefixStateCache.h"

#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"
#include "execution/prefix_cache/DiskPrefixStorageBackend.h"

#include <algorithm>
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

    bool PrefixStateCache::insert(PrefixBlockHandle handle)
    {
        return insertResident(std::move(handle), /*count_store=*/true);
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
            if (existing->second.block.ref_count > 0)
            {
                return false;
            }
            erase(resident_key);
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
        auto [entry_it, inserted] = entries_.emplace(entry.block.handle.key, std::move(entry));
        if (inserted)
        {
            promoteResidentToDeviceHot(entry_it->second);
        }
        if (!preserve_disk_entry)
        {
            removeDiskEntry(resident_key);
        }
        return true;
    }

    std::optional<PrefixBlockHandle> PrefixStateCache::find(const PrefixCacheKey &key)
    {
        stats_.lookups++;
        auto it = entries_.find(key);
        if (it == entries_.end())
        {
            auto hot_it = device_hot_entries_.find(key);
            if (hot_it != device_hot_entries_.end() && device_hot_backend_ && ram_backend_)
            {
                const PrefixBlockHandle hot_handle = hot_it->second;
                if (evictUntilFits(hot_handle.total_bytes))
                {
                    PrefixBlockHandle hydrated;
                    std::string error;
                    if (device_hot_backend_->hydrateToRamBackend(
                            hot_handle,
                            *ram_backend_,
                            &hydrated,
                            &error) &&
                        insertResident(hydrated, /*count_store=*/false, /*preserve_disk_entry=*/true))
                    {
                        ++stats_.hits;
                        ++stats_.promotions;
                        touchDeviceHot(key);
                        auto hydrated_it = entries_.find(key);
                        return hydrated_it == entries_.end()
                                   ? std::optional<PrefixBlockHandle>{}
                                   : std::optional<PrefixBlockHandle>{hydrated_it->second.block.handle};
                    }
                    ram_backend_->release(hydrated);
                }
                removeDeviceHotEntry(key);
            }

            auto disk_it = disk_entries_.find(key);
            if (disk_it == disk_entries_.end() || !disk_backend_ || !ram_backend_)
            {
                stats_.misses++;
                return std::nullopt;
            }

            const PrefixBlockHandle disk_handle = disk_it->second;
            if (!evictUntilFits(disk_handle.total_bytes))
            {
                stats_.misses++;
                return std::nullopt;
            }

            PrefixBlockHandle hydrated;
            std::string error;
            if (!disk_backend_->readBlockIntoRamBackend(
                    key,
                    disk_handle.layout,
                    *ram_backend_,
                    &hydrated,
                    &error))
            {
                ++stats_.disk_read_failures;
                stats_.misses++;
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

    bool PrefixStateCache::contains(const PrefixCacheKey &key) const
    {
        return entries_.find(key) != entries_.end() ||
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
        removeDeviceHotEntry(key);
        return removeDiskEntry(key) || erased;
    }

    bool PrefixStateCache::reserveRam(size_t incoming_bytes)
    {
        return evictUntilFits(incoming_bytes);
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

    bool PrefixStateCache::evictDeviceHotUntilFits(size_t incoming_bytes)
    {
        if (!device_hot_backend_)
        {
            return false;
        }
        if (incoming_bytes > device_hot_backend_->budgetBytes())
        {
            return false;
        }
        while (device_hot_backend_->usedBytes() + incoming_bytes > device_hot_backend_->budgetBytes())
        {
            if (device_hot_lru_.empty())
            {
                return false;
            }
            const PrefixCacheKey victim = device_hot_lru_.back();
            if (!removeDeviceHotEntry(victim))
            {
                return false;
            }
        }
        return true;
    }

    bool PrefixStateCache::promoteResidentToDeviceHot(const Entry &entry)
    {
        if (!device_hot_backend_)
        {
            return false;
        }

        const PrefixBlockHandle &handle = entry.block.handle;
        if (device_hot_entries_.find(handle.key) != device_hot_entries_.end())
        {
            return true;
        }
        if (!evictDeviceHotUntilFits(handle.total_bytes))
        {
            return false;
        }

        PrefixBlockHandle hot_handle;
        std::string error;
        if (!device_hot_backend_->promoteFromRam(handle, &hot_handle, &error))
        {
            return false;
        }

        device_hot_lru_.push_front(hot_handle.key);
        stats_.promotions++;
        stats_.device_hot_bytes = static_cast<uint64_t>(device_hot_backend_->usedBytes());
        stats_.device_bytes = stats_.device_hot_bytes;
        device_hot_entries_.emplace(hot_handle.key, std::move(hot_handle));
        return true;
    }

    bool PrefixStateCache::removeDeviceHotEntry(const PrefixCacheKey &key)
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

        const size_t disk_budget = disk_backend_->budgetBytes();
        if (disk_budget != 0 && stats_.disk_bytes + handle.total_bytes > disk_budget)
        {
            ++stats_.disk_write_failures;
            return false;
        }

        PrefixBlockHandle disk_handle = disk_backend_->allocate(handle.key, handle.layout);
        if (!disk_handle.valid())
        {
            ++stats_.disk_write_failures;
            return false;
        }

        std::string error;
        if (!disk_backend_->writeBlock(handle, &error))
        {
            ++stats_.disk_write_failures;
            return false;
        }

        stats_.disk_bytes += disk_handle.total_bytes;
        disk_entries_.emplace(disk_handle.key, std::move(disk_handle));
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
