/**
 * @file RamPrefixStorageBackend.cpp
 * @brief Implements bounded RAM prefix archives with exact physical leases.
 *
 * Cache indexing can retire a key while a request or in-flight copy still
 * owns its payload. The production reservation therefore follows the backing
 * allocation through the final shared handle, independently of logical LRU
 * occupancy. GPU backing is materialized once during setup and range placement
 * is event-aware; harvest, hydration, reset and eviction never pin or free RAM.
 * Complete GPU recurrent sections deliberately avoid a redundant CPU clear;
 * attention padding and optional terminal sections retain initialization.
 */

#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include "execution/prefix_cache/PrefixHostArena.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>

namespace llaminar2
{

    RamPrefixStorageBackend::RamPrefixStorageBackend(size_t budget_bytes)
        : RamPrefixStorageBackend(DeviceId::cpu(), budget_bytes)
    {
    }

    RamPrefixStorageBackend::RamPrefixStorageBackend(
        DeviceId producer_device,
        size_t budget_bytes)
        : producer_device_(producer_device),
          budget_bytes_(budget_bytes)
    {
    }

    std::shared_ptr<RamPrefixStorageBackend>
    RamPrefixStorageBackend::create(
        DeviceId producer_device,
        size_t budget_bytes,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        std::string *error)
    {
        const auto fail = [&](const std::string &message)
            -> std::shared_ptr<RamPrefixStorageBackend>
        {
            if (error)
                *error = message;
            return nullptr;
        };
        if (!producer_device.is_valid() || budget_bytes == 0u)
            return fail("RAM prefix tier requires a device and positive capacity");
        if (!memory_authority ||
            !memory_authority->contains(DeviceId::cpu()))
        {
            return fail(
                "RAM prefix tier requires the admitted rank-local CPU memory authority");
        }

        auto result = std::shared_ptr<RamPrefixStorageBackend>(
            new RamPrefixStorageBackend(producer_device, budget_bytes));
        try
        {
            // Reserve the entire bounded tier before any pageable or pinned
            // allocation exists. Concurrent participant caches on this rank
            // therefore cannot each spend the same admitted host bytes.
            result->reservation_ =
                memory_authority->reserveNewAllocations(
                    DeviceId::cpu(),
                    PhysicalMemoryOwner::PrefixHostTier,
                    budget_bytes);
            if (producer_device.is_gpu())
            {
                auto claim = result->claimHostAllocation(budget_bytes);
                auto backing = TransferEngine::instance().allocatePinnedHostBuffer(
                    budget_bytes, producer_device);
                result->arena_ = PrefixHostArena::create(
                    std::shared_ptr<void>(backing, backing->mutableData()),
                    budget_bytes, std::move(claim));
                PerfStatsCollector::addCounter("prefix_cache", "ram_arena_materializations",
                    1.0, "setup", producer_device.toString());
                PerfStatsCollector::addCounter("prefix_cache", "ram_arena_materialized_bytes",
                    static_cast<double>(budget_bytes), "setup", producer_device.toString());
            }
        }
        catch (const std::exception &exception)
        {
            return fail(exception.what());
        }
        return result;
    }

    std::shared_ptr<void> RamPrefixStorageBackend::claimHostAllocation(
        size_t bytes) const
    {
        if (bytes == 0u || !reservation_.valid())
            return {};
        auto lease = reservation_.claimAllocation(bytes);
        return std::static_pointer_cast<void>(
            std::make_shared<PhysicalMemorySuballocationLease>(
                std::move(lease)));
    }

    std::shared_ptr<void> RamPrefixStorageBackend::payloadOwner(const PrefixBlockHandle &handle) noexcept
    {
        for (const auto &owner : {handle.pinned_kv_storage, handle.pinned_hybrid_storage,
                                 handle.pinned_mtp_storage, handle.pinned_terminal_hidden_storage,
                                 handle.pinned_terminal_logits_storage})
            if (owner) return owner;
        for (const auto &owner : {handle.kv_storage, handle.hybrid_storage, handle.mtp_storage,
                                 handle.terminal_hidden_storage, handle.terminal_logits_storage})
            if (owner) return std::static_pointer_cast<void>(owner);
        return {};
    }

    bool RamPrefixStorageBackend::allocateSection(
        size_t bytes,
        SectionWriteCoverage coverage,
        std::shared_ptr<std::vector<uint8_t>> *pageable_owner,
        std::shared_ptr<void> *pinned_owner,
        void **payload,
        const std::shared_ptr<void> &arena_payload,
        size_t offset) const
    {
        if (!pageable_owner || !pinned_owner || !payload)
            return false;
        pageable_owner->reset();
        pinned_owner->reset();
        *payload = nullptr;
        if (bytes == 0)
            return true;

        if (!producer_device_.is_gpu())
        {
            *pageable_owner =
                std::make_shared<std::vector<uint8_t>>(bytes, uint8_t{0});
            *payload = (*pageable_owner)->data();
            return true;
        }

        if (!arena_payload)
            return false;
        void *raw = static_cast<uint8_t *>(arena_payload.get()) + offset;
        // Recurrent checkpoints overwrite their entire serialized section.
        // A host memset of large pinned images can dominate prefill (especially
        // with uncached mappings), yet contributes no observable archive data.
        // Keep initialization where short KV blocks or optional terminal rows
        // can leave allocated bytes outside the published payload.
        if (coverage == SectionWriteCoverage::Partial)
            std::memset(raw, 0, bytes);
        *pinned_owner = std::shared_ptr<void>(arena_payload, raw);
        *payload = raw;
        PerfStatsCollector::addCounter(
            "prefix_cache",
            coverage == SectionWriteCoverage::Partial
                ? "ram_payload_cpu_zeroed_bytes"
                : "ram_payload_full_overwrite_bytes",
            static_cast<double>(bytes),
            "allocate",
            producer_device_.toString());
        return true;
    }

    bool RamPrefixStorageBackend::canStore(size_t bytes) const
    {
        return bytes <= availableAllocationBytes();
    }

    size_t RamPrefixStorageBackend::availableAllocationBytes() const
    {
        const auto logical = budget_bytes_ - std::min(budget_bytes_, used_bytes_);
        if (producer_device_.is_gpu())
            return arena_ ? std::min(logical, arena_->availableBytes()) : 0u;
        return reservation_.valid()
                   ? std::min(logical, reservation_.remainingBytes())
                   : logical;
    }

    PrefixBlockHandle RamPrefixStorageBackend::allocate(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout)
    {
        return allocateWithDiagnostics(key, layout, nullptr);
    }

    PrefixBlockHandle RamPrefixStorageBackend::allocateWithDiagnostics(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout,
        std::string *error)
    {
        const auto fail = [&](std::string reason) -> PrefixBlockHandle
        {
            if (error)
                *error = std::move(reason);
            return {};
        };
        PrefixBlockHandle handle;
        handle.key = key;
        handle.tier = PrefixStorageTier::Ram;
        handle.layout = layout;
        constexpr size_t limit = std::numeric_limits<size_t>::max();
        if (layout.fa_layers < 0 || layout.bytes_per_fa_layer_k > limit - layout.bytes_per_fa_layer_v)
            return fail("invalid or overflowing RAM prefix attention geometry");
        const size_t layer_bytes = layout.bytes_per_fa_layer_k + layout.bytes_per_fa_layer_v;
        if (layout.fa_layers > 0 && layer_bytes > limit / static_cast<size_t>(layout.fa_layers))
            return fail("overflowing RAM prefix attention layer extent");
        const size_t kv_bytes = static_cast<size_t>(layout.fa_layers) * layer_bytes;
        const size_t hybrid_bytes = layout.includes_hybrid_state ? layout.hybrid_state_bytes : 0u;
        const size_t mtp_bytes = layout.includes_mtp_state ? layout.mtpKVBytes() : 0u;
        const size_t hidden_bytes = layout.includes_terminal_hidden ? layout.terminal_hidden_bytes : 0u;
        const size_t logits_bytes = layout.includes_terminal_logits ? layout.terminal_logits_bytes : 0u;
        for (size_t bytes : {kv_bytes, hybrid_bytes, mtp_bytes, hidden_bytes, logits_bytes})
        {
            if (bytes > limit - handle.total_bytes)
                return fail("overflowing RAM prefix serialized section extent");
            handle.total_bytes += bytes;
        }

        if (!key.valid() || handle.total_bytes == 0)
            return fail("invalid prefix key or zero-byte payload layout");
        if (producer_device_.is_gpu() && !arena_)
            return fail("GPU RAM prefix archives require an admitted persistent arena from create()");
        if (used_bytes_ > budget_bytes_ ||
            handle.total_bytes > budget_bytes_ - used_bytes_)
            return fail("logical RAM prefix tier capacity exhausted: requested=" +
                        std::to_string(handle.total_bytes) +
                        " used=" + std::to_string(used_bytes_) +
                        " budget=" + std::to_string(budget_bytes_));
        if (!producer_device_.is_gpu() && reservation_.valid() &&
            handle.total_bytes > reservation_.remainingBytes())
            return fail("physical RAM prefix tier capacity busy: requested=" +
                        std::to_string(handle.total_bytes) +
                        " materialized=" +
                        std::to_string(reservation_.materializedBytes()) +
                        " reserved=" +
                        std::to_string(reservation_.capacityBytes()));
        if (allocations_.find(key) != allocations_.end())
            return fail("prefix key already owns a RAM archive allocation");

        try
        {
            // Capacity was committed at construction; this child makes the
            // physically materialized subset observable and follows every
            // copied RAM handle until its backing sections are truly freed.
            if (!producer_device_.is_gpu())
                handle.ram_payload_memory_lease = claimHostAllocation(handle.total_bytes);
        }
        catch (const std::exception &exception)
        {
            return fail("physical RAM prefix lease rejected: " +
                        std::string(exception.what()) +
                        " materialized=" +
                        std::to_string(reservation_.materializedBytes()) +
                        " reserved=" +
                        std::to_string(reservation_.capacityBytes()));
        }

        std::shared_ptr<void> arena_payload;
        if (producer_device_.is_gpu())
        {
            handle.payload_readiness = std::make_shared<PrefixPayloadReadiness>();
            arena_payload = arena_->acquire(handle.total_bytes, handle.payload_readiness);
            if (!arena_payload)
                return fail("physical RAM prefix arena ranges busy or fragmented: requested=" +
                            std::to_string(handle.total_bytes) + " largest_available=" +
                            std::to_string(arena_->availableBytes()));
        }

        if (!allocateSection(
                kv_bytes,
                SectionWriteCoverage::Partial,
                &handle.kv_storage,
                &handle.pinned_kv_storage,
                &handle.kv_payload,
                arena_payload, 0u) ||
            !allocateSection(
                hybrid_bytes,
                SectionWriteCoverage::Complete,
                &handle.hybrid_storage,
                &handle.pinned_hybrid_storage,
                &handle.hybrid_payload,
                arena_payload, kv_bytes) ||
            !allocateSection(
                mtp_bytes,
                SectionWriteCoverage::Partial,
                &handle.mtp_storage,
                &handle.pinned_mtp_storage,
                &handle.mtp_payload,
                arena_payload, kv_bytes + hybrid_bytes) ||
            !allocateSection(
                hidden_bytes,
                SectionWriteCoverage::Partial,
                &handle.terminal_hidden_storage,
                &handle.pinned_terminal_hidden_storage,
                &handle.terminal_hidden,
                arena_payload, kv_bytes + hybrid_bytes + mtp_bytes) ||
            !allocateSection(
                logits_bytes,
                SectionWriteCoverage::Partial,
                &handle.terminal_logits_storage,
                &handle.pinned_terminal_logits_storage,
                &handle.terminal_logits,
                arena_payload, kv_bytes + hybrid_bytes + mtp_bytes + hidden_bytes))
        {
            return fail("RAM prefix payload backing allocation failed for " +
                        producer_device_.toString() +
                        " requested=" + std::to_string(handle.total_bytes));
        }

        allocations_[key] = Allocation{handle.total_bytes, payloadOwner(handle)};
        used_bytes_ += handle.total_bytes;
        if (arena_payload)
            PerfStatsCollector::addCounter("prefix_cache", "ram_arena_payload_leases",
                1.0, "allocate", producer_device_.toString());
        return handle;
    }

    bool RamPrefixStorageBackend::attachModelRuntimeState(
        PrefixBlockHandle *handle,
        std::shared_ptr<std::vector<uint8_t>> storage)
    {
        if (!handle || !handle->valid() ||
            handle->tier != PrefixStorageTier::Ram || !storage ||
            storage->empty() || handle->model_runtime_state_storage ||
            handle->ram_runtime_state_memory_lease)
        {
            return false;
        }
        auto allocation = allocations_.find(handle->key);
        if (allocation == allocations_.end() ||
            allocation->second.bytes != handle->total_bytes ||
            !allocation->second.matches(payloadOwner(*handle)))
        {
            return false;
        }
        const size_t bytes = storage->size();
        if (!canStore(bytes))
            return false;

        std::shared_ptr<void> memory_lease;
        std::shared_ptr<PrefixRuntimeStateStorage> runtime_storage;
        try
        {
            if (arena_)
            {
                auto range = arena_->acquire(bytes);
                if (!range)
                    return false;
                std::memcpy(range.get(), storage->data(), bytes);
                runtime_storage = std::make_shared<PrefixRuntimeStateStorage>(std::move(range), bytes);
            }
            else
            {
                memory_lease = claimHostAllocation(bytes);
                runtime_storage = std::make_shared<PrefixRuntimeStateStorage>(std::move(storage));
            }
        }
        catch (const std::exception &)
        {
            return false;
        }

        // Every operation after the claim is non-throwing. Publish storage
        // before accounting metadata so reverse member destruction remains
        // allocation-then-lease even if the caller immediately drops it.
        handle->model_runtime_state_storage = std::move(runtime_storage);
        handle->ram_runtime_state_memory_lease = std::move(memory_lease);
        handle->has_model_runtime_state = true;
        handle->total_bytes += bytes;
        allocation->second.bytes += bytes;
        used_bytes_ += bytes;
        return true;
    }

    bool RamPrefixStorageBackend::release(const PrefixBlockHandle &handle)
    {
        auto it = allocations_.find(handle.key);
        if (it == allocations_.end() || !it->second.matches(payloadOwner(handle)))
        {
            return false;
        }
        used_bytes_ -= std::min(used_bytes_, it->second.bytes);
        allocations_.erase(it);
        return true;
    }

} // namespace llaminar2
