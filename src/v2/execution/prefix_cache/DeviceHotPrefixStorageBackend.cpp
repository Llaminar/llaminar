/**
 * @file DeviceHotPrefixStorageBackend.cpp
 * @brief Pure-device allocation and ownership for prefix-cache hot replicas.
 */

#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"

#include "backends/BackendManager.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Allocate one archive section in VRAM with event-safe lifetime.
         *
         * A prepared hot handle can be discarded while D2D copies are still
         * queued. The owner therefore retains the block readiness object and
         * waits only for that block's event before returning its allocation.
         */
        std::shared_ptr<void> allocateDeviceSection(
            IBackend *backend,
            DeviceId device,
            size_t bytes,
            const std::shared_ptr<PrefixPayloadReadiness> &readiness,
            std::string *error)
        {
            if (bytes == 0)
                return nullptr;
            if (!backend || !device.is_gpu())
            {
                if (error)
                    *error = "device-hot allocation requires a GPU backend";
                return nullptr;
            }

            const int ordinal = device.gpu_ordinal();
            void *raw = backend->allocate(bytes, ordinal);
            if (!raw)
            {
                if (error)
                    *error = "device-hot VRAM allocation failed";
                return nullptr;
            }

            return std::shared_ptr<void>(
                raw,
                [backend, ordinal, readiness](void *pointer)
                {
                    if (!pointer)
                        return;
                    if (readiness)
                        (void)readiness->waitOnHost();
                    backend->free(pointer, ordinal);
                });
        }
    } // namespace

    DeviceHotPrefixStorageBackend::DeviceHotPrefixStorageBackend(size_t budget_bytes)
        : budget_bytes_(budget_bytes)
    {
    }

    DeviceHotPrefixStorageBackend::DeviceHotPrefixStorageBackend(
        DeviceId device,
        size_t budget_bytes)
        : device_(device),
          budget_bytes_(budget_bytes)
    {
    }

    bool DeviceHotPrefixStorageBackend::canStore(size_t bytes) const
    {
        return device_.is_gpu() &&
               bytes > 0 &&
               bytes <= budget_bytes_ &&
               used_bytes_ <= budget_bytes_ &&
               bytes <= budget_bytes_ - used_bytes_;
    }

    PrefixBlockHandle DeviceHotPrefixStorageBackend::allocate(
        const PrefixCacheKey &key,
        const PrefixPayloadLayout &layout)
    {
        PrefixBlockHandle handle;
        handle.key = key;
        handle.tier = PrefixStorageTier::DeviceHot;
        handle.layout = layout;
        handle.total_bytes = layout.totalBytes();
        return key.valid() && canStore(handle.total_bytes)
                   ? handle
                   : PrefixBlockHandle{};
    }

    bool DeviceHotPrefixStorageBackend::allocateDeviceBlock(
        const PrefixBlockHandle &ram_archive,
        PrefixBlockHandle *device_handle,
        std::string *error)
    {
        if (!device_handle)
            return false;
        if (!ram_archive.valid() ||
            ram_archive.tier != PrefixStorageTier::Ram ||
            !device_.is_gpu() ||
            !canStore(ram_archive.total_bytes))
        {
            if (error)
                *error = "invalid RAM archive or device-hot budget exceeded";
            return false;
        }
        if (allocations_.find(ram_archive.key) != allocations_.end())
        {
            if (error)
                *error = "device-hot key is already allocated";
            return false;
        }

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
        out.payload_readiness = std::make_shared<PrefixPayloadReadiness>();

        IBackend *backend = getBackendFor(device_);
        out.device_kv_allocation = allocateDeviceSection(
            backend,
            device_,
            out.kvBytes(),
            out.payload_readiness,
            error);
        if (out.kvBytes() > 0 && !out.device_kv_allocation)
            return false;

        out.device_hybrid_allocation = allocateDeviceSection(
            backend,
            device_,
            out.hybridBytes(),
            out.payload_readiness,
            error);
        if (out.hybridBytes() > 0 && !out.device_hybrid_allocation)
            return false;

        out.device_mtp_allocation = allocateDeviceSection(
            backend,
            device_,
            out.layout.mtpKVBytes(),
            out.payload_readiness,
            error);
        if (out.layout.mtpKVBytes() > 0 && !out.device_mtp_allocation)
            return false;

        out.device_terminal_hidden_allocation = allocateDeviceSection(
            backend,
            device_,
            out.terminalHiddenBytes(),
            out.payload_readiness,
            error);
        if (out.terminalHiddenBytes() > 0 &&
            !out.device_terminal_hidden_allocation)
        {
            return false;
        }

        out.device_terminal_logits_allocation = allocateDeviceSection(
            backend,
            device_,
            out.terminalLogitsBytes(),
            out.payload_readiness,
            error);
        if (out.terminalLogitsBytes() > 0 &&
            !out.device_terminal_logits_allocation)
        {
            return false;
        }

        allocations_.emplace(out.key, out.total_bytes);
        used_bytes_ += out.total_bytes;
        *device_handle = std::move(out);
        return true;
    }

    bool DeviceHotPrefixStorageBackend::release(
        const PrefixBlockHandle &handle)
    {
        auto it = allocations_.find(handle.key);
        if (it == allocations_.end())
            return false;

        used_bytes_ -= std::min(used_bytes_, it->second);
        allocations_.erase(it);
        return true;
    }

} // namespace llaminar2
