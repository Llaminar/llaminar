#pragma once

#include "execution/prefix_cache/PrefixCacheKey.h"
#include "execution/prefix_cache/PrefixPayloadLayout.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace llaminar2
{
    class TensorBase;

    enum class PrefixStorageTier
    {
        DeviceHot,
        Ram,
        Disk,
    };

    struct PrefixBlockHandle
    {
        PrefixCacheKey key;
        PrefixStorageTier tier = PrefixStorageTier::Ram;
        PrefixPayloadLayout layout;
        void *kv_payload = nullptr;
        void *hybrid_payload = nullptr;
        void *mtp_payload = nullptr;
        void *terminal_hidden = nullptr;
        void *terminal_logits = nullptr;
        size_t total_bytes = 0;
        bool has_hybrid_state = false;
        bool has_terminal_hidden = false;
        bool has_terminal_logits = false;

        std::shared_ptr<std::vector<uint8_t>> kv_storage;
        std::shared_ptr<std::vector<uint8_t>> hybrid_storage;
        std::shared_ptr<std::vector<uint8_t>> mtp_storage;
        std::shared_ptr<std::vector<uint8_t>> terminal_hidden_storage;
        std::shared_ptr<std::vector<uint8_t>> terminal_logits_storage;
        std::shared_ptr<TensorBase> device_kv_storage;
        std::shared_ptr<TensorBase> device_hybrid_storage;
        std::shared_ptr<TensorBase> device_mtp_storage;
        std::shared_ptr<TensorBase> device_terminal_hidden_storage;
        std::shared_ptr<TensorBase> device_terminal_logits_storage;

        bool valid() const { return key.valid() && total_bytes > 0; }
        size_t kvKBytes() const { return static_cast<size_t>(layout.fa_layers) * layout.bytes_per_fa_layer_k; }
        size_t kvVBytes() const { return static_cast<size_t>(layout.fa_layers) * layout.bytes_per_fa_layer_v; }
        size_t mtpKBytes() const;
        size_t mtpVBytes() const;
        uint8_t *kvKData();
        uint8_t *kvVData();
        const uint8_t *kvKData() const;
        const uint8_t *kvVData() const;
        uint8_t *mtpKData();
        uint8_t *mtpVData();
        const uint8_t *mtpKData() const;
        const uint8_t *mtpVData() const;
    };

    class IPrefixStorageBackend
    {
    public:
        virtual ~IPrefixStorageBackend() = default;
        virtual bool canStore(size_t bytes) const = 0;
        virtual PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                           const PrefixPayloadLayout &layout) = 0;
        virtual bool release(const PrefixBlockHandle &handle) = 0;
        virtual bool hydrateToRam(const PrefixBlockHandle &handle,
                                  PrefixBlockHandle *ram_handle) = 0;
    };

} // namespace llaminar2
