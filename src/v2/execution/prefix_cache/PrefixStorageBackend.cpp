#include "execution/prefix_cache/PrefixStorageBackend.h"

namespace llaminar2
{
    size_t PrefixBlockHandle::mtpKBytes() const
    {
        if (layout.mtp_layers > 0 && layout.bytes_per_mtp_layer_k > 0)
        {
            return static_cast<size_t>(layout.mtp_layers) * layout.bytes_per_mtp_layer_k;
        }
        return layout.includes_mtp_state ? layout.mtpKVBytes() / 2 : 0;
    }

    size_t PrefixBlockHandle::mtpVBytes() const
    {
        const size_t total = layout.includes_mtp_state ? layout.mtpKVBytes() : 0;
        const size_t k_bytes = mtpKBytes();
        return total > k_bytes ? total - k_bytes : 0;
    }

    uint8_t *PrefixBlockHandle::kvKData()
    {
        return kv_storage && !kv_storage->empty() ? kv_storage->data() : nullptr;
    }

    uint8_t *PrefixBlockHandle::kvVData()
    {
        if (!kv_storage || kv_storage->empty())
        {
            return nullptr;
        }
        return kv_storage->data() + kvKBytes();
    }

    const uint8_t *PrefixBlockHandle::kvKData() const
    {
        return kv_storage && !kv_storage->empty() ? kv_storage->data() : nullptr;
    }

    const uint8_t *PrefixBlockHandle::kvVData() const
    {
        if (!kv_storage || kv_storage->empty())
        {
            return nullptr;
        }
        return kv_storage->data() + kvKBytes();
    }

    uint8_t *PrefixBlockHandle::mtpKData()
    {
        return mtp_storage && !mtp_storage->empty() ? mtp_storage->data() : nullptr;
    }

    uint8_t *PrefixBlockHandle::mtpVData()
    {
        if (!mtp_storage || mtp_storage->empty())
        {
            return nullptr;
        }
        return mtp_storage->data() + mtpKBytes();
    }

    const uint8_t *PrefixBlockHandle::mtpKData() const
    {
        return mtp_storage && !mtp_storage->empty() ? mtp_storage->data() : nullptr;
    }

    const uint8_t *PrefixBlockHandle::mtpVData() const
    {
        if (!mtp_storage || mtp_storage->empty())
        {
            return nullptr;
        }
        return mtp_storage->data() + mtpKBytes();
    }

} // namespace llaminar2
