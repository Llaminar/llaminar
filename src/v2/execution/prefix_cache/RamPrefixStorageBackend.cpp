#include "execution/prefix_cache/RamPrefixStorageBackend.h"

#include <algorithm>

namespace llaminar2
{

    RamPrefixStorageBackend::RamPrefixStorageBackend(size_t budget_bytes)
        : budget_bytes_(budget_bytes)
    {
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

        const size_t kv_bytes = layout.faKVBytes();
        if (kv_bytes > 0)
        {
            handle.kv_storage = std::make_shared<std::vector<uint8_t>>(kv_bytes, 0);
            handle.kv_payload = handle.kv_storage->data();
        }
        if (layout.includes_hybrid_state && layout.hybrid_state_bytes > 0)
        {
            handle.hybrid_storage = std::make_shared<std::vector<uint8_t>>(layout.hybrid_state_bytes, 0);
            handle.hybrid_payload = handle.hybrid_storage->data();
        }
        if (layout.includes_mtp_state && layout.mtpKVBytes() > 0)
        {
            handle.mtp_storage = std::make_shared<std::vector<uint8_t>>(layout.mtpKVBytes(), 0);
            handle.mtp_payload = handle.mtp_storage->data();
        }
        if (layout.includes_terminal_hidden && layout.terminal_hidden_bytes > 0)
        {
            handle.terminal_hidden_storage = std::make_shared<std::vector<uint8_t>>(layout.terminal_hidden_bytes, 0);
            handle.terminal_hidden = handle.terminal_hidden_storage->data();
        }
        if (layout.includes_terminal_logits && layout.terminal_logits_bytes > 0)
        {
            handle.terminal_logits_storage = std::make_shared<std::vector<uint8_t>>(layout.terminal_logits_bytes, 0);
            handle.terminal_logits = handle.terminal_logits_storage->data();
        }

        allocations_[key] = handle.total_bytes;
        used_bytes_ += handle.total_bytes;
        return handle;
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

    bool RamPrefixStorageBackend::hydrateToRam(const PrefixBlockHandle &handle,
                                               PrefixBlockHandle *ram_handle)
    {
        if (!ram_handle || handle.tier != PrefixStorageTier::Ram || !handle.valid())
        {
            return false;
        }
        *ram_handle = handle;
        return true;
    }

} // namespace llaminar2
