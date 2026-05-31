#include "execution/prefix_cache/PrefixPayloadLayout.h"

#include "kernels/IHybridKVCache.h"
#include "kernels/IKVCache.h"

namespace llaminar2
{
    namespace
    {
        int firstFullAttentionLayerForLayout(const IKVCache &kv_cache,
                                             const IHybridKVCache *hybrid)
        {
            if (!hybrid)
            {
                return kv_cache.first_layer_index();
            }

            for (int layer = 0; layer < kv_cache.n_layers(); ++layer)
            {
                if (hybrid->isFullAttentionLayer(layer))
                {
                    return layer;
                }
            }
            return -1;
        }
    } // namespace

    int firstRestorablePrefixLayer(const IKVCache &kv_cache)
    {
        const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&kv_cache);
        const int fa_layer = firstFullAttentionLayerForLayout(kv_cache, hybrid);
        return fa_layer >= 0 ? fa_layer : kv_cache.first_layer_index();
    }

    int restorablePrefixCachedTokens(const IKVCache &kv_cache, int seq_idx)
    {
        return kv_cache.get_cached_tokens(firstRestorablePrefixLayer(kv_cache), seq_idx);
    }

    size_t PrefixPayloadLayout::faKVBytes() const
    {
        return static_cast<size_t>(fa_layers) * (bytes_per_fa_layer_k + bytes_per_fa_layer_v);
    }

    size_t PrefixPayloadLayout::mtpKVBytes() const
    {
        return mtp_kv_bytes;
    }

    size_t PrefixPayloadLayout::totalBytes() const
    {
        size_t total = faKVBytes();
        if (includes_hybrid_state)
        {
            total += hybrid_state_bytes;
        }
        if (includes_mtp_state)
        {
            total += mtpKVBytes();
        }
        if (includes_terminal_hidden)
        {
            total += terminal_hidden_bytes;
        }
        if (includes_terminal_logits)
        {
            total += terminal_logits_bytes;
        }
        return total;
    }

    bool PrefixPayloadLayout::compatiblePayloadShape(const PrefixPayloadLayout &other) const
    {
        return block_size == other.block_size &&
               first_layer_index == other.first_layer_index &&
               total_layers == other.total_layers &&
               fa_layers == other.fa_layers &&
               gdn_layers == other.gdn_layers &&
               local_kv_heads == other.local_kv_heads &&
               kv_head_start == other.kv_head_start &&
               head_dim == other.head_dim &&
               k_precision == other.k_precision &&
               v_precision == other.v_precision &&
               kv_layout == other.kv_layout &&
               bytes_per_fa_layer_k == other.bytes_per_fa_layer_k &&
               bytes_per_fa_layer_v == other.bytes_per_fa_layer_v &&
               hybrid_host_state_bytes == other.hybrid_host_state_bytes &&
               hybrid_device_state_bytes == other.hybrid_device_state_bytes &&
               hybrid_state_bytes == other.hybrid_state_bytes &&
               mtp_layers == other.mtp_layers &&
               mtp_local_kv_heads == other.mtp_local_kv_heads &&
               mtp_kv_head_start == other.mtp_kv_head_start &&
               mtp_head_dim == other.mtp_head_dim &&
               mtp_k_precision == other.mtp_k_precision &&
               mtp_v_precision == other.mtp_v_precision &&
               mtp_kv_layout == other.mtp_kv_layout &&
               bytes_per_mtp_layer_k == other.bytes_per_mtp_layer_k &&
               bytes_per_mtp_layer_v == other.bytes_per_mtp_layer_v &&
               mtp_kv_bytes == other.mtp_kv_bytes &&
               terminal_hidden_bytes == other.terminal_hidden_bytes &&
               terminal_logits_bytes == other.terminal_logits_bytes &&
               includes_hybrid_state == other.includes_hybrid_state &&
               includes_mtp_state == other.includes_mtp_state &&
               includes_terminal_hidden == other.includes_terminal_hidden &&
               includes_terminal_logits == other.includes_terminal_logits;
    }

    PrefixPayloadLayout buildDensePrefixPayloadLayout(
        const IKVCache &kv_cache,
        DeviceId device,
        int block_size,
        size_t terminal_hidden_bytes,
        size_t terminal_logits_bytes)
    {
        PrefixPayloadLayout layout;
        layout.device = device;
        layout.block_size = block_size;
        layout.first_layer_index = kv_cache.first_layer_index();
        layout.total_layers = kv_cache.n_layers();
        const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&kv_cache);
        layout.fa_layers = hybrid ? hybrid->kvLayerCount() : kv_cache.n_layers();
        layout.gdn_layers = hybrid ? hybrid->gdnLayerCount() : 0;

        if (hybrid)
        {
            const HybridPrefixStateMetadata metadata = hybrid->hybridPrefixStateMetadata();
            layout.hybrid_host_state_bytes = metadata.host_bytes;
            layout.hybrid_device_state_bytes = metadata.device_bytes;
            layout.hybrid_state_bytes = metadata.host_bytes + metadata.device_bytes;
            layout.includes_hybrid_state = layout.hybrid_state_bytes > 0;
        }

        const int layout_probe_layer = firstRestorablePrefixLayer(kv_cache);
        const auto block = layout_probe_layer >= 0
                               ? kv_cache.logicalBlockLayout(layout_probe_layer, block_size)
                               : IKVCache::KVCacheLogicalBlockLayout{};
        layout.local_kv_heads = block.local_kv_heads;
        layout.kv_head_start = block.kv_head_start;
        layout.head_dim = block.head_dim;
        layout.k_precision = block.k_precision;
        layout.v_precision = block.v_precision;
        layout.kv_layout = block.layout;
        layout.bytes_per_fa_layer_k = block.k_bytes;
        layout.bytes_per_fa_layer_v = block.v_bytes;
        layout.terminal_hidden_bytes = terminal_hidden_bytes;
        layout.terminal_logits_bytes = terminal_logits_bytes;
        layout.includes_terminal_hidden = terminal_hidden_bytes > 0;
        layout.includes_terminal_logits = terminal_logits_bytes > 0;
        return layout;
    }

} // namespace llaminar2
