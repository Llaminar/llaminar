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

    class PrefixStateCache
    {
    public:
        PrefixStateCache(size_t ram_budget_bytes,
                         std::shared_ptr<IPrefixStorageBackend> ram_backend,
                         std::shared_ptr<DiskPrefixStorageBackend> disk_backend = nullptr,
                         std::shared_ptr<DeviceHotPrefixStorageBackend> device_hot_backend = nullptr);

        bool insert(PrefixBlockHandle handle);
        std::optional<PrefixBlockHandle> find(const PrefixCacheKey &key);
        bool contains(const PrefixCacheKey &key) const;
        bool retain(const PrefixCacheKey &key);
        bool release(const PrefixCacheKey &key);
        bool erase(const PrefixCacheKey &key);
        bool reserveRam(size_t incoming_bytes);
        void recordRequestLookup(int requested_tokens,
                                 int matched_tokens,
                                 int matched_blocks);
        void recordTerminalStateHit();

        size_t size() const { return entries_.size(); }
        size_t ramBudgetBytes() const { return ram_budget_bytes_; }
        size_t usedBytes() const { return used_bytes_; }
        const PrefixCacheStats &stats() const { return stats_; }
        std::vector<PrefixCacheKey> keysMostRecentFirst() const;

    private:
        struct Entry
        {
            PrefixStateBlock block;
            std::list<PrefixCacheKey>::iterator lru_it;
        };

        bool insertResident(PrefixBlockHandle handle, bool count_store, bool preserve_disk_entry = false);
        bool evictResident(const PrefixCacheKey &key);
        bool evictUntilFits(size_t incoming_bytes);
        bool evictDeviceHotUntilFits(size_t incoming_bytes);
        bool promoteResidentToDeviceHot(const Entry &entry);
        bool removeDeviceHotEntry(const PrefixCacheKey &key);
        void touchDeviceHot(const PrefixCacheKey &key);
        bool persistResidentToDisk(const Entry &entry);
        bool removeDiskEntry(const PrefixCacheKey &key);
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
