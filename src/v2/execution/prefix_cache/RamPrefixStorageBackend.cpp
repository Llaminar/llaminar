#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include "backends/BackendManager.h"

#include <algorithm>
#include <cstring>
#include <exception>
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

    bool RamPrefixStorageBackend::allocateSection(
        size_t bytes,
        std::shared_ptr<std::vector<uint8_t>> *pageable_owner,
        std::shared_ptr<void> *pinned_owner,
        void **payload,
        const std::shared_ptr<PrefixPayloadReadiness> &readiness) const
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

        IBackend *backend = getBackendFor(producer_device_);
        if (!backend)
            return false;
        const int ordinal = producer_device_.gpu_ordinal();
        void *raw = backend->allocatePinned(bytes, ordinal);
        if (!raw)
            return false;
        std::memset(raw, 0, bytes);
        *pinned_owner = std::shared_ptr<void>(
            raw,
            [backend, ordinal, readiness](void *pointer)
            {
                if (pointer)
                {
                    /*
                     * An entry can be evicted while its archive DMA is still
                     * queued. Wait for that block's event, not the whole
                     * device, before returning the destination to the backend.
                     */
                    if (readiness)
                        (void)readiness->waitOnHost();
                    backend->freePinned(pointer, ordinal);
                }
            });
        *payload = raw;
        return true;
    }

    bool RamPrefixStorageBackend::canStore(size_t bytes) const
    {
        return bytes <= budget_bytes_ && bytes <= budget_bytes_ - used_bytes_;
    }

    PrefixBlockHandle RamPrefixStorageBackend::allocate(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout)
    {
        PrefixBlockHandle handle;
        handle.key = key;
        handle.tier = PrefixStorageTier::Ram;
        handle.layout = layout;
        handle.total_bytes = layout.totalBytes();

        if (!key.valid() || handle.total_bytes == 0 || !canStore(handle.total_bytes))
        {
            return {};
        }
        if (allocations_.find(key) != allocations_.end())
            return {};

        try
        {
            // Capacity was committed at construction; this child makes the
            // physically materialized subset observable and follows every
            // copied RAM handle until its backing sections are truly freed.
            handle.ram_payload_memory_lease =
                claimHostAllocation(handle.total_bytes);
        }
        catch (const std::exception &)
        {
            return {};
        }

        if (producer_device_.is_gpu())
            handle.payload_readiness = std::make_shared<PrefixPayloadReadiness>();

        const size_t kv_bytes = layout.faKVBytes();
        if (!allocateSection(
                kv_bytes,
                &handle.kv_storage,
                &handle.pinned_kv_storage,
                &handle.kv_payload,
                handle.payload_readiness) ||
            !allocateSection(
                layout.includes_hybrid_state ? layout.hybrid_state_bytes : 0,
                &handle.hybrid_storage,
                &handle.pinned_hybrid_storage,
                &handle.hybrid_payload,
                handle.payload_readiness) ||
            !allocateSection(
                layout.includes_mtp_state ? layout.mtpKVBytes() : 0,
                &handle.mtp_storage,
                &handle.pinned_mtp_storage,
                &handle.mtp_payload,
                handle.payload_readiness) ||
            !allocateSection(
                layout.includes_terminal_hidden ? layout.terminal_hidden_bytes : 0,
                &handle.terminal_hidden_storage,
                &handle.pinned_terminal_hidden_storage,
                &handle.terminal_hidden,
                handle.payload_readiness) ||
            !allocateSection(
                layout.includes_terminal_logits ? layout.terminal_logits_bytes : 0,
                &handle.terminal_logits_storage,
                &handle.pinned_terminal_logits_storage,
                &handle.terminal_logits,
                handle.payload_readiness))
        {
            return {};
        }

        allocations_[key] = handle.total_bytes;
        used_bytes_ += handle.total_bytes;
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
            allocation->second != handle->total_bytes)
        {
            return false;
        }
        const size_t bytes = storage->size();
        if (!canStore(bytes))
            return false;

        std::shared_ptr<void> memory_lease;
        try
        {
            memory_lease = claimHostAllocation(bytes);
        }
        catch (const std::exception &)
        {
            return false;
        }

        // Every operation after the claim is non-throwing. Publish storage
        // before accounting metadata so reverse member destruction remains
        // allocation-then-lease even if the caller immediately drops it.
        handle->model_runtime_state_storage = std::move(storage);
        handle->ram_runtime_state_memory_lease = std::move(memory_lease);
        handle->has_model_runtime_state = true;
        handle->total_bytes += bytes;
        allocation->second += bytes;
        used_bytes_ += bytes;
        return true;
    }

    bool RamPrefixStorageBackend::release(const PrefixBlockHandle &handle)
    {
        auto it = allocations_.find(handle.key);
        if (it == allocations_.end())
        {
            return false;
        }
        used_bytes_ -= std::min(used_bytes_, it->second);
        allocations_.erase(it);
        return true;
    }

} // namespace llaminar2
