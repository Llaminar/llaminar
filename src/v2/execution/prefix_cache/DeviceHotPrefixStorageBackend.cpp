#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"

#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2
{
    namespace
    {
        size_t tensorElementCountForBytes(size_t bytes)
        {
            return (bytes + sizeof(float) - 1) / sizeof(float);
        }

        std::shared_ptr<TensorBase> uploadBytesToDevice(
            const std::shared_ptr<std::vector<uint8_t>> &bytes,
            DeviceId device,
            std::string *error)
        {
            if (!bytes || bytes->empty())
            {
                return nullptr;
            }

            auto tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{tensorElementCountForBytes(bytes->size())},
                DeviceId::cpu());
            auto *host = static_cast<uint8_t *>(tensor->raw_mutable_data());
            if (!host)
            {
                if (error)
                    *error = "failed to allocate device-hot staging tensor";
                return nullptr;
            }
            std::memset(host, 0, tensor->size_bytes());
            std::memcpy(host, bytes->data(), bytes->size());

            auto result = TransferEngine::instance().upload(tensor.get(), device);
            if (!result.success)
            {
                if (error)
                    *error = result.error;
                return nullptr;
            }
            return tensor;
        }

        bool downloadBytesFromDevice(const std::shared_ptr<TensorBase> &tensor,
                                     std::shared_ptr<std::vector<uint8_t>> &bytes,
                                     std::string *error)
        {
            if (!bytes || bytes->empty())
            {
                return true;
            }
            if (!tensor)
            {
                if (error)
                    *error = "missing device-hot tensor";
                return false;
            }

            auto result = TransferEngine::instance().download(tensor.get());
            if (!result.success)
            {
                if (error)
                    *error = result.error;
                return false;
            }

            const auto *host = static_cast<const uint8_t *>(tensor->raw_data());
            if (!host)
            {
                if (error)
                    *error = "device-hot tensor host data unavailable";
                return false;
            }
            std::memcpy(bytes->data(), host, bytes->size());
            return true;
        }
    } // namespace

    DeviceHotPrefixStorageBackend::DeviceHotPrefixStorageBackend(size_t budget_bytes)
        : budget_bytes_(budget_bytes)
    {
    }

    DeviceHotPrefixStorageBackend::DeviceHotPrefixStorageBackend(DeviceId device, size_t budget_bytes)
        : device_(device), budget_bytes_(budget_bytes)
    {
    }

    bool DeviceHotPrefixStorageBackend::canStore(size_t bytes) const
    {
        return device_.is_gpu() && bytes > 0 &&
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
        return key.valid() && canStore(handle.total_bytes) ? handle : PrefixBlockHandle{};
    }

    bool DeviceHotPrefixStorageBackend::promoteFromRam(
        const PrefixBlockHandle &ram_handle,
        PrefixBlockHandle *device_handle,
        std::string *error)
    {
        if (!device_handle)
        {
            return false;
        }
        if (!ram_handle.valid() || !device_.is_gpu() || !canStore(ram_handle.total_bytes))
        {
            if (error)
                *error = "invalid RAM handle or device-hot budget exceeded";
            return false;
        }

        PrefixBlockHandle out = allocate(ram_handle.key, ram_handle.layout);
        if (!out.valid())
        {
            if (error)
                *error = "failed to allocate device-hot handle";
            return false;
        }
        out.has_hybrid_state = ram_handle.has_hybrid_state;
        out.has_terminal_hidden = ram_handle.has_terminal_hidden;
        out.has_terminal_logits = ram_handle.has_terminal_logits;

        out.device_kv_storage = uploadBytesToDevice(ram_handle.kv_storage, device_, error);
        if (ram_handle.kv_storage && !ram_handle.kv_storage->empty() && !out.device_kv_storage)
            return false;

        out.device_hybrid_storage = uploadBytesToDevice(ram_handle.hybrid_storage, device_, error);
        if (ram_handle.hybrid_storage && !ram_handle.hybrid_storage->empty() && !out.device_hybrid_storage)
            return false;

        out.device_mtp_storage = uploadBytesToDevice(ram_handle.mtp_storage, device_, error);
        if (ram_handle.mtp_storage && !ram_handle.mtp_storage->empty() && !out.device_mtp_storage)
            return false;

        out.device_terminal_hidden_storage = uploadBytesToDevice(ram_handle.terminal_hidden_storage, device_, error);
        if (ram_handle.terminal_hidden_storage && !ram_handle.terminal_hidden_storage->empty() &&
            !out.device_terminal_hidden_storage)
            return false;

        out.device_terminal_logits_storage = uploadBytesToDevice(ram_handle.terminal_logits_storage, device_, error);
        if (ram_handle.terminal_logits_storage && !ram_handle.terminal_logits_storage->empty() &&
            !out.device_terminal_logits_storage)
            return false;

        allocations_[out.key] = out.total_bytes;
        used_bytes_ += out.total_bytes;
        *device_handle = std::move(out);
        return true;
    }

    bool DeviceHotPrefixStorageBackend::release(const PrefixBlockHandle &handle)
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

    bool DeviceHotPrefixStorageBackend::hydrateToRam(const PrefixBlockHandle &handle,
                                                     PrefixBlockHandle *ram_handle)
    {
        RamPrefixStorageBackend ram(handle.layout.totalBytes());
        return hydrateToRamBackend(handle, ram, ram_handle, nullptr);
    }

    bool DeviceHotPrefixStorageBackend::hydrateToRamBackend(
        const PrefixBlockHandle &handle,
        IPrefixStorageBackend &ram_backend,
        PrefixBlockHandle *ram_handle,
        std::string *error)
    {
        if (!ram_handle || handle.tier != PrefixStorageTier::DeviceHot || !handle.valid())
        {
            return false;
        }

        PrefixBlockHandle out = ram_backend.allocate(handle.key, handle.layout);
        if (!out.valid())
        {
            if (error)
                *error = "failed to allocate RAM hydration handle";
            return false;
        }

        if (!downloadBytesFromDevice(handle.device_kv_storage, out.kv_storage, error) ||
            !downloadBytesFromDevice(handle.device_hybrid_storage, out.hybrid_storage, error) ||
            !downloadBytesFromDevice(handle.device_mtp_storage, out.mtp_storage, error) ||
            !downloadBytesFromDevice(handle.device_terminal_hidden_storage, out.terminal_hidden_storage, error) ||
            !downloadBytesFromDevice(handle.device_terminal_logits_storage, out.terminal_logits_storage, error))
        {
            ram_backend.release(out);
            return false;
        }

        out.kv_payload = out.kv_storage && !out.kv_storage->empty() ? out.kv_storage->data() : nullptr;
        out.hybrid_payload = out.hybrid_storage && !out.hybrid_storage->empty() ? out.hybrid_storage->data() : nullptr;
        out.mtp_payload = out.mtp_storage && !out.mtp_storage->empty() ? out.mtp_storage->data() : nullptr;
        out.terminal_hidden = out.terminal_hidden_storage && !out.terminal_hidden_storage->empty()
                                  ? out.terminal_hidden_storage->data()
                                  : nullptr;
        out.terminal_logits = out.terminal_logits_storage && !out.terminal_logits_storage->empty()
                                  ? out.terminal_logits_storage->data()
                                  : nullptr;
        out.has_hybrid_state = handle.has_hybrid_state;
        out.has_terminal_hidden = handle.has_terminal_hidden;
        out.has_terminal_logits = handle.has_terminal_logits;

        *ram_handle = std::move(out);
        return true;
    }
}
