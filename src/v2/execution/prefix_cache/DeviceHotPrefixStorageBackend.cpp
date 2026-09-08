/**
 * @file DeviceHotPrefixStorageBackend.cpp
 * @brief Pre-capture VRAM arena and event-ordered slot reuse for hot prefixes.
 *
 * Device-hot prefix storage is a performance tier, but it is still part of the
 * production graph's persistent memory topology. This implementation reserves
 * one bounded slab before capture and exposes archive-shaped aliases into fixed
 * slots. Harvest and promotion therefore perform only asynchronous copies and
 * event publication; they never ask CUDA/HIP to allocate or free memory.
 */

#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"

#include "execution/config/RuntimeConfig.h"

#include "backends/BackendManager.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Sum the serialized sections that physically live in VRAM.
         *
         * Model-runtime metadata is retained by a host vector and deliberately
         * excluded; PrefixBlockHandle::total_bytes may include those host bytes.
         */
        size_t devicePayloadBytes(const PrefixBlockHandle &handle)
        {
            const std::array<size_t, 5> sections = {
                handle.kvBytes(),
                handle.hybridBytes(),
                handle.layout.mtpKVBytes(),
                handle.terminalHiddenBytes(),
                handle.terminalLogitsBytes(),
            };
            size_t total = 0u;
            for (const size_t bytes : sections)
            {
                if (bytes > std::numeric_limits<size_t>::max() - total)
                    return std::numeric_limits<size_t>::max();
                total += bytes;
            }
            return total;
        }
    } // namespace

    /** @brief Hidden arena ownership and lease-accounting implementation. */
    struct DeviceHotPrefixStorageBackend::Impl
    {
        /**
         * @brief Shared slab owner retained by the backend and every live slot.
         *
         * If a request-scoped handle outlives PrefixStateCache teardown, its
         * SlotLease keeps this state alive. The final destruction is a genuine
         * teardown boundary, where waiting for the latest event in each slot is
         * legal before returning the slab to the backend allocator.
         */
        struct PoolState
        {
            IBackend *backend = nullptr;
            DeviceId device = DeviceId::invalid();
            void *allocation = nullptr;
            size_t allocation_bytes = 0u;
            std::optional<PhysicalMemoryAllocationLease> memory_lease;
            std::vector<std::shared_ptr<PrefixPayloadReadiness>>
                latest_slot_readiness;

            ~PoolState()
            {
                for (const auto &readiness : latest_slot_readiness)
                {
                    if (readiness)
                        (void)readiness->waitOnHost();
                }
                if (backend && allocation && device.is_gpu())
                    backend->free(allocation, device.gpu_ordinal());
                allocation = nullptr;
                allocation_bytes = 0u;

                // The byte claim must remain live through the exact backing
                // free. Retiring it earlier would let another owner believe
                // this VRAM was reusable while the backend still owns it.
                memory_lease.reset();
            }
        };

        /** @brief Alias owner proving that one slot still has live consumers. */
        struct SlotLease
        {
            std::shared_ptr<PoolState> pool;
            size_t slot_index = 0u;
        };

        /** @brief Mutable reuse state for one fixed-size slab partition. */
        struct Slot
        {
            bool assigned = false;
            std::weak_ptr<SlotLease> lease;
            std::shared_ptr<PrefixPayloadReadiness> latest_readiness;
        };

        /** @brief Backend accounting record for one installed cache key. */
        struct Allocation
        {
            size_t slot_index = 0u;
            size_t charged_bytes = 0u;
        };

        DeviceId device = DeviceId::invalid();
        size_t budget_bytes = 0u;
        size_t reserved_bytes = 0u;
        size_t slot_bytes = 0u;
        size_t used_bytes = 0u;
        std::shared_ptr<PoolState> pool;
        std::vector<Slot> slots;
        std::unordered_map<PrefixCacheKey, Allocation, PrefixCacheKeyHasher>
            allocations;
        mutable std::mutex mutex;

        /** @return First unassigned slot with no outstanding handle aliases. */
        size_t availableSlotLocked() const
        {
            for (size_t index = 0; index < slots.size(); ++index)
            {
                const Slot &slot = slots[index];
                if (!slot.assigned && slot.lease.expired())
                    return index;
            }
            return slots.size();
        }
    };

    DeviceHotPrefixStorageBackend::DeviceHotPrefixStorageBackend()
        : impl_(std::make_unique<Impl>())
    {
    }

    DeviceHotPrefixStorageBackend::~DeviceHotPrefixStorageBackend() = default;

    std::shared_ptr<DeviceHotPrefixStorageBackend>
    DeviceHotPrefixStorageBackend::create(
        DeviceId device,
        size_t budget_bytes,
        size_t slot_bytes,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        std::string *error)
    {
        const auto fail = [&](const char *message)
            -> std::shared_ptr<DeviceHotPrefixStorageBackend>
        {
            if (error)
                *error = message;
            return nullptr;
        };

        if (!device.is_gpu() || budget_bytes == 0u || slot_bytes == 0u ||
            slot_bytes > budget_bytes)
        {
            return fail(
                "device-hot arena requires a GPU and at least one complete slot");
        }
        if (!memory_authority || !memory_authority->contains(device))
        {
            return fail(
                "device-hot arena requires the admitted rank-local memory authority");
        }
        IBackend *backend = getBackendFor(device);
        if (!backend)
            return fail("device-hot arena could not resolve its GPU backend");

        const size_t reserved_bytes =
            prefixCacheWholeBlockReservationBytes(
                budget_bytes, slot_bytes);
        const size_t slot_count = reserved_bytes / slot_bytes;
        if (slot_count == 0u ||
            slot_count > std::numeric_limits<size_t>::max() / slot_bytes)
        {
            return fail("device-hot arena slot geometry is invalid");
        }

        std::optional<PhysicalMemoryAllocationLease> memory_lease;
        try
        {
            // Pre-claim before calling the backend allocator. A failed backend
            // allocation simply destroys this local lease and restores the
            // owner line; successful construction moves it beside the slab.
            memory_lease.emplace(memory_authority->claimNewAllocation(
                device,
                PhysicalMemoryOwner::PrefixDeviceTier,
                reserved_bytes));
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return nullptr;
        }
        void *allocation = backend->allocate(
            reserved_bytes, device.gpu_ordinal());
        if (!allocation)
        {
            return fail(
                "device-hot arena reservation failed before graph capture");
        }

        auto result = std::shared_ptr<DeviceHotPrefixStorageBackend>(
            new DeviceHotPrefixStorageBackend());
        auto pool = std::make_shared<Impl::PoolState>();
        pool->backend = backend;
        pool->device = device;
        pool->allocation = allocation;
        pool->allocation_bytes = reserved_bytes;
        pool->memory_lease = std::move(memory_lease);
        pool->latest_slot_readiness.resize(slot_count);

        result->impl_->device = device;
        result->impl_->budget_bytes = budget_bytes;
        result->impl_->reserved_bytes = reserved_bytes;
        result->impl_->slot_bytes = slot_bytes;
        result->impl_->pool = std::move(pool);
        result->impl_->slots.resize(slot_count);
        return result;
    }

    bool DeviceHotPrefixStorageBackend::capacityEligible(size_t bytes) const
    {
        return impl_ && impl_->device.is_gpu() && bytes > 0u &&
               bytes <= impl_->budget_bytes && bytes <= impl_->slot_bytes;
    }

    bool DeviceHotPrefixStorageBackend::canStore(size_t bytes) const
    {
        if (!capacityEligible(bytes))
            return false;
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->availableSlotLocked() < impl_->slots.size();
    }

    PrefixBlockHandle DeviceHotPrefixStorageBackend::allocate(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout)
    {
        (void)key;
        (void)layout;
        return {};
    }

    bool DeviceHotPrefixStorageBackend::allocateDeviceBlock(
        const PrefixBlockHandle &ram_archive,
        void *producer_stream,
        PrefixBlockHandle *device_handle,
        std::string *error)
    {
        const auto fail = [&](const char *message)
        {
            if (error)
                *error = message;
            return false;
        };
        if (!device_handle || !producer_stream || !impl_ ||
            !ram_archive.valid() ||
            ram_archive.tier != PrefixStorageTier::Ram ||
            !impl_->device.is_gpu())
        {
            return fail(
                "device-hot slot lease requires a RAM archive and explicit GPU stream");
        }

        const size_t payload_bytes = devicePayloadBytes(ram_archive);
        if (payload_bytes == 0u || payload_bytes > impl_->slot_bytes ||
            ram_archive.total_bytes > impl_->budget_bytes)
        {
            return fail("device-hot archive exceeds immutable arena geometry");
        }

        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->allocations.find(ram_archive.key) !=
            impl_->allocations.end())
        {
            return fail("device-hot key is already allocated");
        }
        const size_t slot_index = impl_->availableSlotLocked();
        if (slot_index >= impl_->slots.size())
            return fail("device-hot arena has no retired slot available");

        Impl::Slot &slot = impl_->slots[slot_index];
        if (slot.latest_readiness)
        {
            if (slot.latest_readiness->prepared() &&
                !slot.latest_readiness->published())
            {
                return fail(
                    "device-hot retired slot has an unpublished producer event");
            }
            if (!slot.latest_readiness->waitOnStream(producer_stream))
            {
                return fail(
                    "device-hot producer could not consume the prior slot event");
            }
        }

        auto readiness = std::make_shared<PrefixPayloadReadiness>();
        auto lease = std::make_shared<Impl::SlotLease>();
        lease->pool = impl_->pool;
        lease->slot_index = slot_index;

        PrefixBlockHandle out;
        out.key = ram_archive.key;
        out.tier = PrefixStorageTier::DeviceHot;
        out.layout = ram_archive.layout;
        out.total_bytes = ram_archive.total_bytes;
        out.has_hybrid_state = ram_archive.has_hybrid_state;
        out.has_terminal_hidden = ram_archive.has_terminal_hidden;
        out.has_terminal_logits = ram_archive.has_terminal_logits;
        out.has_model_runtime_state = ram_archive.has_model_runtime_state;
        out.model_runtime_state_storage = ram_archive.model_runtime_state_storage;
        out.ram_runtime_state_memory_lease =
            ram_archive.ram_runtime_state_memory_lease;
        out.payload_readiness = readiness;

        auto *base = static_cast<uint8_t *>(impl_->pool->allocation) +
                     slot_index * impl_->slot_bytes;
        size_t offset = 0u;
        const auto bind_section = [&](size_t bytes) -> std::shared_ptr<void>
        {
            if (bytes == 0u)
                return nullptr;
            auto owner = std::shared_ptr<void>(lease, base + offset);
            offset += bytes;
            return owner;
        };
        out.device_kv_allocation = bind_section(out.kvBytes());
        out.device_hybrid_allocation = bind_section(out.hybridBytes());
        out.device_mtp_allocation = bind_section(out.layout.mtpKVBytes());
        out.device_terminal_hidden_allocation =
            bind_section(out.terminalHiddenBytes());
        out.device_terminal_logits_allocation =
            bind_section(out.terminalLogitsBytes());
        if (offset != payload_bytes || offset > impl_->slot_bytes)
            return fail("device-hot section aliases exceed their arena slot");

        impl_->allocations.emplace(
            out.key,
            Impl::Allocation{
                .slot_index = slot_index,
                .charged_bytes = out.total_bytes,
            });
        impl_->used_bytes += out.total_bytes;
        slot.assigned = true;
        slot.lease = lease;
        slot.latest_readiness = readiness;
        impl_->pool->latest_slot_readiness[slot_index] = readiness;
        *device_handle = std::move(out);
        return true;
    }

    bool DeviceHotPrefixStorageBackend::release(
        const PrefixBlockHandle &handle)
    {
        if (!impl_)
            return false;
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto allocation = impl_->allocations.find(handle.key);
        if (allocation == impl_->allocations.end() ||
            allocation->second.slot_index >= impl_->slots.size())
        {
            return false;
        }

        Impl::Slot &slot =
            impl_->slots[allocation->second.slot_index];
        slot.assigned = false;
        impl_->used_bytes -= std::min(
            impl_->used_bytes, allocation->second.charged_bytes);
        impl_->allocations.erase(allocation);
        return true;
    }

    size_t DeviceHotPrefixStorageBackend::budgetBytes() const
    {
        return impl_ ? impl_->budget_bytes : 0u;
    }

    size_t DeviceHotPrefixStorageBackend::usedBytes() const
    {
        if (!impl_)
            return 0u;
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->used_bytes;
    }

    size_t DeviceHotPrefixStorageBackend::reservedBytes() const
    {
        return impl_ ? impl_->reserved_bytes : 0u;
    }

    size_t DeviceHotPrefixStorageBackend::slotBytes() const
    {
        return impl_ ? impl_->slot_bytes : 0u;
    }

    size_t DeviceHotPrefixStorageBackend::slotCount() const
    {
        return impl_ ? impl_->slots.size() : 0u;
    }

    DeviceId DeviceHotPrefixStorageBackend::device() const
    {
        return impl_ ? impl_->device : DeviceId::invalid();
    }
} // namespace llaminar2
