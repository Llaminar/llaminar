/**
 * @file PrefixStateCache.cpp
 * @brief Implements durable prefix-block indexing and tier-aware LRU lookup.
 *
 * Metadata probing is separated from payload acquisition so a longest-prefix
 * query records one semantic hit or miss even when it examines several token
 * widths. A coordinated lookup retains immutable disk offsets, then hydrates
 * rows plus the selected endpoint, without pinning old recurrent checkpoints.
 * The cache owns disk hydration, LRU and payload lifetime. Early preparation only queues native
 * persistence; required RAM/SSD publication completes exact immutable receipts
 * before allocating. Pending durable work is never dropped or credited as free.
 * Every RAM admission also polls the restore producer's exact completion edges,
 * because source reads can finish during a harvest or its durable capacity join.
 * A local durable index is only a lookup hint. Every selected RAM victim joins
 * the archive writer's fresh publication before releasing its cache alias.
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
    /**
     * @brief Own candidate sources until coordination seals one exact restore frontier.
     *
     * Metadata never borrows payload addresses. RAM candidates retain their
     * original owners; disk candidates retain only immutable inode offsets.
     * Before any disk allocation, all selected RAM sources shed unconsumed
     * checkpoint sections together, allowing ordinary eviction to reclaim them.
     */
    class PrefixCacheLookupPlan final : public IPrefixLookupPayloadPlan
    {
    public:
        /** @brief Bind a weak cache lifetime so stale plans cannot invoke a retired owner. */
        explicit PrefixCacheLookupPlan(std::weak_ptr<PrefixStateCache> cache) : cache_(std::move(cache)) {}

        /** @copydoc IPrefixLookupPayloadPlan::selectLongest */
        std::optional<PrefixBlockHandle> selectLongest(
            uint64_t fingerprint, uint64_t parent_hash, int block_index,
            int token_start, const std::vector<int32_t> &tokens) override
        {
            if (phase_ != Phase::Selecting)
                throw std::logic_error("prefix lookup selection already sealed");
            if (fingerprint == 0 || block_index < 0 || token_start < 0 || tokens.empty())
                return std::nullopt;
            auto cache = owner();
            cache->publishCompletedPersistence();
            ++cache->stats_.lookups;
            for (size_t count = tokens.size(); count > 0; --count)
            {
                const auto key = makePrefixCacheKey(fingerprint, parent_hash, block_index,
                    token_start, {tokens.begin(), tokens.begin() + count});
                Source source;
                if (const auto hot = cache->device_hot_entries_.find(key);
                    hot != cache->device_hot_entries_.end())
                {
                    source.resident = hot->second;
                    cache->touchDeviceHot(key);
                    if (const auto ram = cache->entries_.find(key); ram != cache->entries_.end())
                        cache->touch(ram->second);
                    ++cache->stats_.device_hot_direct_hits;
                }
                else if (const auto ram = cache->entries_.find(key); ram != cache->entries_.end())
                {
                    source.resident = ram->second.block.handle;
                    cache->touch(ram->second);
                }
                else if (const auto disk = cache->disk_entries_.find(key); disk != cache->disk_entries_.end())
                {
                    std::string error;
                    source.disk = cache->disk_backend_->captureLookupSource(key, disk->second.layout, &error);
                    if (!source.disk)
                    {
                        if (!error.empty()) throw std::runtime_error("prefix lookup snapshot failed: " + error);
                        cache->forgetDiskEntry(key);
                        continue;
                    }
                }
                else continue;
                const auto &handle = source.disk ? source.disk->diskHandle() : source.resident;
                source.metadata.key = handle.key;
                source.metadata.tier = handle.tier;
                source.metadata.layout = handle.layout;
                source.metadata.payload_identity = handle.payload_identity;
                source.metadata.total_bytes = handle.total_bytes;
                source.metadata.has_hybrid_state = handle.has_hybrid_state;
                source.metadata.has_terminal_hidden = handle.has_terminal_hidden;
                source.metadata.has_terminal_logits = handle.has_terminal_logits;
                source.metadata.has_model_runtime_state = handle.has_model_runtime_state;
                const auto metadata = source.metadata;
                sources_.push_back(std::move(source));
                ++cache->stats_.hits;
                return metadata;
            }
            ++cache->stats_.misses;
            return std::nullopt;
        }

        /** @copydoc IPrefixLookupPayloadPlan::boundedTokenCount */
        int boundedTokenCount(const std::vector<PrefixBlockHandle> &selected) const override
        {
            if (phase_ != Phase::Selecting)
                throw std::logic_error("prefix restore window must be chosen before materialization");
            auto ram = std::dynamic_pointer_cast<RamPrefixStorageBackend>(owner()->ram_backend_);
            if (!ram) throw std::logic_error("prefix restore window requires the RAM placement authority");
            size_t earlier_rows = 0u;
            int boundary = 0;
            for (const size_t index : selectedSourceIndices(selected))
            {
                const auto &block = sources_[index].metadata;
                const size_t rows = block.tier == PrefixStorageTier::DeviceHot ? 0u :
                    PrefixPayloadAllocationPlan::archive(
                        prefixReadLayout(block.layout, PrefixPayloadReadSet::SequenceRows)).totalBytes();
                const size_t endpoint = block.tier == PrefixStorageTier::DeviceHot
                    ? block.total_bytes - block.layout.totalBytes() : block.total_bytes;
                if (endpoint > std::numeric_limits<size_t>::max() - earlier_rows)
                    throw std::overflow_error("prefix restore working-set BOM overflow");
                const bool complete_endpoint = block.layout.hybrid_state_bytes == 0u || block.has_hybrid_state;
                if (complete_endpoint && ram->canRetainWorkingSet(
                        PrefixPayloadAllocationPlan::contiguous(earlier_rows + endpoint)))
                    boundary = block.key.token_start + block.key.token_count;
                if (rows > std::numeric_limits<size_t>::max() - earlier_rows)
                    throw std::overflow_error("prefix sequence-row BOM overflow");
                earlier_rows += rows;
            }
            return boundary;
        }

        /** @copydoc IPrefixLookupPayloadPlan::materialize */
        std::vector<PrefixBlockHandle> materialize(const std::vector<PrefixBlockHandle> &selected) override
        {
            if (selected.empty()) throw std::logic_error("cannot materialize an empty prefix hit");
            if (phase_ == Phase::Materialized)
            {
                if (selected.size() != selected_keys_.size())
                    throw std::logic_error("prefix lookup cannot select another materialized frontier");
                for (size_t index = 0; index < selected.size(); ++index)
                    if (selected[index].key != selected_keys_[index])
                        throw std::logic_error("prefix lookup materialized key changed");
                return materialized_;
            }
            if (phase_ != Phase::Selecting || selected.size() > sources_.size())
                throw std::logic_error("prefix lookup has no unconsumed source selection");
            const auto selected_indices = selectedSourceIndices(selected);
            if (boundedTokenCount(selected) != selected.back().key.token_start + selected.back().key.token_count)
                throw std::invalid_argument("prefix restore working set exceeds its bounded RAM window");
            std::vector<Source> sealed_sources;
            sealed_sources.reserve(selected.size());
            for (const size_t index : selected_indices)
            {
                selected_keys_.push_back(sources_[index].metadata.key);
                sealed_sources.push_back(std::move(sources_[index]));
            }
            phase_ = Phase::Materializing;
            sources_ = std::move(sealed_sources);
            materialized_.resize(selected.size());
            const bool recurrent_only = sources_.back().metadata.layout.organization() == PrefixPayloadOrganization::RecurrentCheckpoint;
            const size_t first = recurrent_only ? selected.size() - 1u : 0u;
            // Retire all unused resident sections before the first cold read
            // asks the one capacity authority to make space. No later source
            // can keep an obsolete recurrent image pinned through this loop.
            for (size_t index = 0; index < sources_.size(); ++index)
            {
                auto &source = sources_[index];
                if (index < first) { source = {}; continue; }
                if (source.resident.valid())
                {
                    auto lease = index + 1u == selected.size()
                        ? PrefixPayloadReadLease::wholeArchive(std::move(source.resident))
                        : PrefixPayloadReadLease::sequenceRows(std::move(source.resident));
                    materialized_[index] = lease.source();
                    source.resident = {};
                }
            }
            auto cache = owner();
            for (size_t index = first; index < sources_.size(); ++index)
                if (auto &disk = sources_[index].disk; disk)
                    materialized_[index] = cache->materializeLookupSource(*disk,
                        index + 1u == selected.size() ? PrefixPayloadReadSet::WholeArchive
                                                    : PrefixPayloadReadSet::SequenceRows);
            // These receipts describe actual retained owners after all reads
            // succeed. Earlier checkpoint sections were excluded here, before
            // the device restore can observe them; capacity is not read volume.
            for (size_t index = first; index < materialized_.size(); ++index)
            {
                const auto retained = materialized_[index].total_bytes;
                const auto archived = sources_[index].metadata.total_bytes;
                if (retained > archived)
                    throw std::logic_error("prefix read set exceeds its captured archive payload");
                const bool endpoint = index + 1u == materialized_.size();
                PerfStatsCollector::addCounter("prefix_cache", endpoint
                    ? "selected_restore_checkpoint_bytes" : "selected_restore_sequence_bytes",
                    static_cast<double>(retained), "materialize", "CPU");
                PerfStatsCollector::addCounter("prefix_cache", "selected_restore_unread_checkpoint_bytes",
                    static_cast<double>(archived - retained), "materialize", "CPU");
            }
            sources_.clear();
            if (first > 0u) materialized_.erase(materialized_.begin(), materialized_.begin() + first);
            phase_ = Phase::Materialized;
            return materialized_;
        }

        /** @copydoc IPrefixLookupPayloadPlan::terminal */
        PrefixBlockHandle terminal(const PrefixCacheKey &key) const override
        {
            if ((phase_ != Phase::Materialized && phase_ != Phase::Retired) ||
                materialized_.empty() || materialized_.back().key != key)
                throw std::logic_error("prefix harvest requires its materialized terminal frontier");
            return materialized_.back();
        }

        /** @copydoc IPrefixLookupPayloadPlan::retireForHarvest */
        PrefixBlockHandle retireForHarvest(std::optional<PrefixCacheKey> endpoint) override
        {
            if (endpoint)
            {
                auto last = terminal(*endpoint);
                materialized_.clear();
                materialized_.push_back(std::move(last));
            }
            else
            {
                if (phase_ == Phase::Materializing || !materialized_.empty())
                    throw std::logic_error("a materialized prefix cannot be retired as a cache miss");
                sources_.clear();
            }
            phase_ = Phase::Retired;
            return materialized_.empty() ? PrefixBlockHandle{} : materialized_.back();
        }

    private:
        /** @brief One candidate owns either resident bytes or immutable archive offsets. */
        struct Source
        {
            PrefixBlockHandle metadata;
            PrefixBlockHandle resident;
            std::optional<DiskPrefixStorageBackend::HydrationTicket> disk;
        };
        enum class Phase { Selecting, Materializing, Materialized, Retired };
        /**
         * @brief Authenticate a selected chain against retained source metadata.
         * @param selected Caller-selected identities in original lookup order.
         * @return Exact source indices; recurrent-only candidates may omit ancestors.
         * @throws std::logic_error for foreign, reordered, mixed-shape or gapped attention sources.
         * Payload geometry comes from the captured source, never mutable copies
         * of lookup metadata. This also prevents a forged recurrent-only layout
         * from discarding rows that the real selected endpoint still requires.
         */
        std::vector<size_t> selectedSourceIndices(const std::vector<PrefixBlockHandle> &selected) const
        {
            std::vector<size_t> indices;
            indices.reserve(selected.size());
            size_t cursor = 0u;
            int expected_start = 0;
            for (const auto &candidate : selected)
            {
                while (cursor < sources_.size() && sources_[cursor].metadata.key != candidate.key) ++cursor;
                if (cursor == sources_.size())
                    throw std::logic_error("prefix selection differs from its captured source identities");
                const auto &block = sources_[cursor].metadata;
                if (!indices.empty())
                {
                    const auto &first = sources_[indices.front()].metadata;
                    if (block.key.fingerprint != first.key.fingerprint ||
                        !block.layout.compatiblePayloadShape(first.layout))
                        throw std::logic_error("prefix selection mixes source namespaces or payload shapes");
                }
                if (block.layout.organization() != PrefixPayloadOrganization::RecurrentCheckpoint &&
                    block.key.token_start != expected_start)
                    throw std::logic_error("prefix attention restore has a sequence-row gap");
                expected_start = block.key.token_start + block.key.token_count;
                indices.push_back(cursor++);
            }
            return indices;
        }
        /** @return Live cache authority, rejecting a plan used after owner retirement. */
        std::shared_ptr<PrefixStateCache> owner() const
        {
            auto owner = cache_.lock();
            if (!owner) throw std::logic_error("prefix lookup cache owner has retired");
            return owner;
        }
        std::weak_ptr<PrefixStateCache> cache_;
        Phase phase_ = Phase::Selecting;
        std::vector<Source> sources_;
        std::vector<PrefixCacheKey> selected_keys_;
        std::vector<PrefixBlockHandle> materialized_;
    };

    std::shared_ptr<IPrefixLookupPayloadPlan> PrefixStateCache::beginLookup()
    {
        if (weak_from_this().expired())
            throw std::logic_error("prefix lookup requires its cache's shared lifetime");
        return std::make_shared<PrefixCacheLookupPlan>(weak_from_this());
    }

    PrefixBlockHandle PrefixStateCache::materializeLookupSource(
        DiskPrefixStorageBackend::HydrationTicket &source, PrefixPayloadReadSet read_set)
    {
        std::string error;
        if (!disk_backend_ || !disk_backend_->selectHydrationSections(source, read_set, &error))
        {
            ++stats_.disk_read_failures;
            throw std::runtime_error("prefix selected read metadata rejected: " + error);
        }
        const auto &metadata = source.diskHandle();
        const auto layout = prefixReadLayout(metadata.layout, read_set);
        const size_t runtime = read_set == PrefixPayloadReadSet::WholeArchive
            ? metadata.total_bytes - metadata.layout.totalBytes() : 0u;
        const auto allocation = PrefixPayloadAllocationPlan::archive(layout, runtime);
        if (completePendingCapacity(allocation) != PrefixRamInsertPreparation::Prepared)
            throw std::runtime_error("prefix selected restore cannot acquire its admitted RAM sections");
        auto ram = std::dynamic_pointer_cast<RamPrefixStorageBackend>(ram_backend_);
        PrefixBlockHandle payload;
        if (!ram || !disk_backend_->hydrateSelected(source, *ram, &payload, &error))
        {
            ++stats_.disk_read_failures;
            throw std::runtime_error("prefix selected read failed hydration: " + error);
        }
        if (payload.total_bytes == metadata.total_bytes)
        {
            if (!insertResident(payload, false, true))
            {
                ram->release(payload);
                throw std::runtime_error("prefix complete selected payload could not enter RAM index");
            }
            ++stats_.promotions;
        }
        else
        {
            // A projected read is not a weaker replacement for a durable
            // checkpoint. Only its consumer owns these sections; releasing
            // backend metadata keeps their actual leases charged until read
            // completion, while the complete archive stays on disk.
            if (!ram->release(payload))
                throw std::logic_error("prefix projected read lost its RAM allocation identity");
        }
        ++stats_.disk_hydrations;
        PerfStatsCollector::addCounter("prefix_cache", "selected_disk_read_bytes",
            static_cast<double>(source.hydratedPayloadBytes()), "restore", "CPU");
        return payload;
    }

    PrefixStateCache::PrefixStateCache(size_t ram_budget_bytes,
                                       std::shared_ptr<IPrefixStorageBackend> ram_backend,
                                       std::shared_ptr<DiskPrefixStorageBackend> disk_backend,
                                       std::shared_ptr<DeviceHotPrefixStorageBackend> device_hot_backend,
                                       const IPrefixRestoreSourceRetirement *restore_retirement)
        : ram_budget_bytes_(ram_budget_bytes),
          ram_backend_(std::move(ram_backend)),
          disk_backend_(std::move(disk_backend)),
          device_hot_backend_(std::move(device_hot_backend)),
          restore_retirement_(restore_retirement)
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
                pending_hydration_ = disk_backend_->beginHydration(
                    key,
                    disk_it->second.layout,
                    &error);
            if (!pending_hydration_)
            {
                ++stats_.disk_read_failures;
                stats_.misses++;
                if (!error.empty())
                    throw std::runtime_error("prefix source metadata admission failed: " + error);
                forgetDiskEntry(key);
                return std::nullopt;
            }

            /*
             * Metadata has retained a committed inode and exact read extents.
             * Admit final RAM owners, then read each section into them once.
             * There is no preliminary payload scan or staging copy.
             */
            auto &hydration = *pending_hydration_;
            if (completePendingCapacity(PrefixPayloadAllocationPlan::archive(
                    hydration.diskHandle().layout,
                    hydration.totalBytes() - hydration.diskHandle().layout.totalBytes())) !=
                PrefixRamInsertPreparation::Prepared)
            {
                // Known durable writes complete before a selected restore is
                // judged unavailable. External request aliases have no archive
                // completion edge; keep both their charge and the retained inode
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
            if (!disk_backend_->hydrateSelected(
                    hydration,
                    *concrete_ram,
                    &hydrated,
                    &error))
            {
                ++stats_.disk_read_failures;
                ++stats_.misses;
                /*
                 * Another archive writer may have replaced or evicted the
                 * retained record while RAM capacity was prepared.  Never
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
        const PrefixPayloadAllocationPlan &allocation)
    {
        const size_t incoming_bytes = allocation.totalBytes();
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
        return prepareCapacity(allocation);
    }

    void PrefixStateCache::completeInsertPreparation(
        const PrefixCacheKey &key,
        const PrefixPayloadAllocationPlan &allocation)
    {
        const size_t incoming_bytes = allocation.totalBytes();
        auto preparation = prepareInsert(key, allocation);
        if (preparation == PrefixRamInsertPreparation::Busy)
            preparation = completePendingCapacity(allocation);
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
        const PrefixPayloadAllocationPlan &allocation)
    {
        const size_t incoming_bytes = allocation.totalBytes();
        auto preparation = prepareCapacity(allocation);
        while (preparation == PrefixRamInsertPreparation::Busy &&
               !pending_persistence_.empty())
        {
            // Cache metadata is single-thread-owned; the writer only publishes
            // this immutable receipt. Copy the ticket before publication can
            // retire the deque's front. Never wait for later unrelated work.
            const auto dependency = pending_persistence_.front().ticket;
            PerfStatsCollector::addCounterWithSequence(
                "prefix_cache",
                "ram_archive_capacity_dependency_waits",
                1.0,
                {static_cast<uint64_t>(dependency.identity())},
                "archive_publication",
                "CPU",
                {{"incoming_bytes", std::to_string(incoming_bytes)}});
            (void)dependency.waitForPublication();
            publishCompletedPersistence();
            preparation = prepareCapacity(allocation);
        }
        return preparation;
    }

    PrefixRamInsertPreparation PrefixStateCache::prepareCapacity(const PrefixPayloadAllocationPlan &allocation)
    {
        const size_t incoming_bytes = allocation.totalBytes();
        // Completion can advance during a harvest loop or while joining a
        // necessary archive receipt. Reap exact restore readers at every
        // physical admission boundary, rather than only at harvest entry.
        if (restore_retirement_)
            restore_retirement_->retireCompletedPrefixRestoreSources();
        publishCompletedPersistence();
        if (incoming_bytes == 0 || incoming_bytes > ram_budget_bytes_ || !ram_backend_)
            return PrefixRamInsertPreparation::Error;
        const auto preparation = evictUntilFits(incoming_bytes);
        if (preparation != PrefixRamInsertPreparation::Prepared)
            return preparation;

        return evictUntilPhysicallyFits(allocation);
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
        for (const int checkpoint : schedule.reusableCheckpoints())
            plan_frontier(checkpoint);
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
        const PrefixPayloadAllocationPlan &allocation)
    {
        const size_t incoming_bytes = allocation.totalBytes();
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
             candidate != hot_candidates.rend() && !ram->canStore(allocation);
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
            const auto available = ram->availableArchiveBytes();
            if (ram->canStore(allocation) ||
                (pending_victim_bytes != 0u &&
                 pending_victim_bytes >= incoming_bytes - std::min(incoming_bytes, available)))
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
            if (!evictResident(*candidate, ResidentRetirement::Eviction))
                return PrefixRamInsertPreparation::Error;
            ++stats_.evictions;
            if (device_hot_entries_.contains(*candidate) &&
                !removeDeviceHotEntry(*candidate, /*capacity_eviction=*/true))
                return PrefixRamInsertPreparation::Error;
        }
        return ram->canStore(allocation) ? PrefixRamInsertPreparation::Prepared
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
            if (!evictResident(*candidate, ResidentRetirement::Eviction))
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

    bool PrefixStateCache::evictResident(const PrefixCacheKey &key, ResidentRetirement reason)
    {
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.block.ref_count > 0 || !ram_backend_)
        {
            return false;
        }
        subtractResidentStats(it->second.block.handle);
        used_bytes_ -= std::min(used_bytes_, it->second.block.handle.total_bytes);
        const size_t retired_bytes = it->second.block.handle.total_bytes;
        ram_backend_->release(it->second.block.handle);
        if (reason != ResidentRetirement::Removal)
        {
            if (const auto ram = std::dynamic_pointer_cast<RamPrefixStorageBackend>(ram_backend_))
            {
                if (reason == ResidentRetirement::Demotion) ram->recordDemotion(retired_bytes);
                else ram->recordEviction(retired_bytes);
            }
        }
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
        if (!entry.persistence.valid())
        {
            // Participant metadata cannot observe every peer/process eviction.
            // The sole writer checks identical backing under the archive lock;
            // its receipt retires this selected RAM victim, without another
            // payload write when the durable bytes are still present.
            entry.persistence = disk_backend_->scheduleWrite(handle);
            pending_persistence_.push_back({handle.key, entry.persistence});
            if (disk_entries_.contains(handle.key))
                PerfStatsCollector::addCounter(
                    "prefix_cache", "archive_backing_revalidations", 1.0,
                    "archive_publication", "CPU");
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
                    ++stats_.ram_to_disk_demotions;
                    if (resident->second.block.ref_count == 0)
                    {
                        // Consume this exact publication at the selected
                        // source's retirement boundary. Keeping a successful
                        // ticket on a resident would let later eviction trust
                        // old backing after another writer removed it.
                        if (!evictResident(pending.key, ResidentRetirement::Demotion))
                            throw std::logic_error("published prefix victim could not retire its RAM alias");
                        ++stats_.evictions;
                    }
                    else
                    {
                        // An explicit legacy retain acquired after selection
                        // postpones retirement. A later demotion must receive
                        // its own fresh receipt, not reuse this old one.
                        resident->second.persistence = {};
                    }
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
