/**
 * @file PrefixPayloadLayout.cpp
 * @brief Derive prefix record geometry from canonical cache ownership.
 *
 * Recurrent-only pipeline shards deliberately have no attention payload. Their
 * archive is a complete state image at a hash-authenticated token frontier;
 * main or shifted attention instead requires a contiguous block sequence.
 */
#include "execution/prefix_cache/PrefixPayloadLayout.h"
#include "execution/qwen4exp/Qwen4ExpAuxiliaryState.h"

#include "kernels/IHybridKVCache.h"
#include "kernels/IKVCache.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Check exact section arithmetic before allocation or serialized extent publication. */
        size_t addBytes(size_t left, size_t right)
        {
            if (right > std::numeric_limits<size_t>::max() - left)
                throw std::overflow_error("prefix auxiliary section extent overflow");
            return left + right;
        }
    }

    size_t PrefixAuxiliaryPayloadLayout::blockBytes(int attention_layers) const
    {
        if (attention_layers < 0 || (!kind && (bytes_per_attention_layer || endpoint_bytes || includes_endpoint)) ||
            (kind && *kind != CacheAuxiliaryStateKind::QwenFlashNext))
            throw std::invalid_argument("prefix auxiliary section requires its exact supported model identity");
        if (attention_layers != 0 && bytes_per_attention_layer >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(attention_layers))
            throw std::overflow_error("prefix auxiliary block extent overflow");
        return static_cast<size_t>(attention_layers) * bytes_per_attention_layer;
    }

    size_t PrefixPayloadLayout::modelAuxiliaryBytes() const
    {
        const auto blocks = addBytes(main_auxiliary.blockBytes(fa_layers), mtp_auxiliary.blockBytes(mtp_layers));
        return addBytes(addBytes(blocks, main_auxiliary.endpointBytes()), mtp_auxiliary.endpointBytes());
    }

    void applyPrefixAuxiliaryLayout(PrefixPayloadLayout &layout, const IKVCache &main_cache,
        const IKVCache *mtp_cache, int sequence, int first_token, int token_count,
        int logical_prefix, bool include_endpoint)
    {
        if (sequence < 0 || first_token < 0 || token_count < 0 || logical_prefix < first_token ||
            token_count > logical_prefix - first_token)
            throw std::invalid_argument("prefix auxiliary range lies outside immutable logical prefix");
        const auto project = [&](const IKVCache &cache, int attention_layers, int begin, int count, int frontier)
        {
            PrefixAuxiliaryPayloadLayout result;
            const auto *owner = cache.auxiliaryState();
            if (!owner) return result;
            if (owner->device() != layout.device)
                throw std::invalid_argument("prefix auxiliary state belongs to another physical participant");
            result.kind = owner->kind();
            for (int index = 0; index < attention_layers; ++index)
            {
                const int layer = prefixFALayerForIndex(cache, index);
                if (layer < 0) throw std::invalid_argument("prefix auxiliary FA ordinal lacks exact global layer identity");
                const auto bytes = owner->blockArchiveBytes({layer, sequence, begin, count});
                if (index != 0 && bytes != result.bytes_per_attention_layer)
                    throw std::invalid_argument("prefix auxiliary pool layer strides differ within one archive");
                result.bytes_per_attention_layer = bytes;
            }
            result.endpoint_bytes = owner->endpointArchiveBytes(sequence, frontier);
            result.includes_endpoint = include_endpoint;
            (void)result.blockBytes(attention_layers);
            return result;
        };
        layout.main_auxiliary = project(main_cache, layout.fa_layers, first_token, token_count, logical_prefix);
        if(mtp_cache)layout.mtp_origin=mtp_cache->stateOwnership().shifted_origin;
        const auto shifted=mtpShiftedBlockRange(first_token,token_count,layout.mtp_origin);
        layout.mtp_auxiliary = mtp_cache ? project(*mtp_cache,layout.mtp_layers,
            shifted.first,shifted.count,mtpShiftedPrefixRows(logical_prefix,layout.mtp_origin)) : PrefixAuxiliaryPayloadLayout{};
        (void)layout.modelAuxiliaryBytes();
    }

    int prefixFALayerForIndex(const IKVCache &kv_cache, int fa_index)
    {
        if (fa_index < 0 || fa_index >= kv_cache.n_layers())
            return -1;
        const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&kv_cache);
        if (!hybrid)
            return kv_cache.first_layer_index() + fa_index;

        int seen = 0;
        for (int offset = 0; offset < kv_cache.n_layers(); ++offset)
        {
            // The cache translates global identity into its compressed FA/GDN
            // banks. Adding the offset after that query would classify a
            // different layer when a pipeline slice starts inside the model.
            const int global_layer = kv_cache.first_layer_index() + offset;
            if (hybrid->isFullAttentionLayer(global_layer))
            {
                if (seen == fa_index)
                    return global_layer;
                ++seen;
            }
        }
        return -1;
    }

    int firstRestorablePrefixLayer(const IKVCache &kv_cache)
    {
        const int fa_layer = prefixFALayerForIndex(kv_cache, 0);
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

    bool PrefixPayloadLayout::hasRestorableMainState() const
    {
        if (total_layers <= 0 || fa_layers < 0 || gdn_layers < 0 ||
            fa_layers + gdn_layers != total_layers)
            return false;
        if (fa_layers > 0 &&
            (bytes_per_fa_layer_k == 0 || bytes_per_fa_layer_v == 0))
            return false;
        if (includes_hybrid_state && device.is_cpu() && gdn_layers > 0 && hybrid_host_state_bytes > 0 &&
            hybrid_host_encoding != HybridPrefixHostEncoding::CPUFixedEpochPackedV2 &&
            hybrid_host_encoding != HybridPrefixHostEncoding::CPUFixedEpochLocalImage) return false;
        // Missing attention is valid only when the complete recurrent owner
        // supplies the state. Terminal logits alone cannot certify a cache.
        return gdn_layers == 0 ||
            (includes_hybrid_state && hybrid_state_bytes > 0 &&
             hybrid_state_bytes == hybrid_host_state_bytes + hybrid_device_state_bytes);
    }

    PrefixPayloadOrganization PrefixPayloadLayout::organization() const
    {
        return fa_layers == 0 && gdn_layers > 0 && !includes_mtp_state
            ? PrefixPayloadOrganization::RecurrentCheckpoint
            : PrefixPayloadOrganization::AttentionBlocks;
    }

    size_t PrefixPayloadLayout::totalBytes() const
    {
        if (includes_hybrid_state &&
            ((hybrid_host_state_bytes > 0 &&
              !validGDNRecurrenceStateLayout(hybrid_host_recurrence_layout)) ||
             (hybrid_device_state_bytes > 0 &&
              !validGDNRecurrenceStateLayout(hybrid_device_recurrence_layout))))
            throw std::invalid_argument("Unknown prefix payload recurrence layout");
        if (includes_hybrid_state && hybrid_host_state_bytes > 0)
        {
            const bool packed = hybrid_host_encoding == HybridPrefixHostEncoding::CPUFixedEpochPackedV2;
            const bool local = hybrid_host_encoding == HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
            if ((device.is_cpu() && gdn_layers > 0 && !packed && !local) ||
                (hybrid_host_encoding != HybridPrefixHostEncoding::None && !packed && !local) ||
                ((packed || local) && (hybrid_host_state_capacity_bytes == 0 ||
                                      hybrid_host_state_bytes > hybrid_host_state_capacity_bytes)) ||
                hybrid_host_state_bytes > std::numeric_limits<size_t>::max() - hybrid_device_state_bytes ||
                hybrid_state_bytes != hybrid_host_state_bytes + hybrid_device_state_bytes)
                throw std::invalid_argument("Invalid CPU GDN prefix host encoding/extent");
        }
        size_t total = faKVBytes();
        if (includes_hybrid_state)
        {
            total = addBytes(total, hybrid_state_bytes);
        }
        if (includes_mtp_state)
        {
            total = addBytes(total, mtpKVBytes());
        }
        if (includes_terminal_hidden)
        {
            total = addBytes(total, terminal_hidden_bytes);
        }
        if (includes_terminal_logits)
        {
            total = addBytes(total, terminal_logits_bytes);
        }
        return addBytes(total, modelAuxiliaryBytes());
    }

    bool PrefixPayloadLayout::compatiblePayloadShape(const PrefixPayloadLayout &other) const
    {
        return main_auxiliary.kind == other.main_auxiliary.kind &&
               mtp_auxiliary.kind == other.mtp_auxiliary.kind && block_size == other.block_size &&
               first_layer_index == other.first_layer_index && total_layers == other.total_layers &&
               fa_layers == other.fa_layers && gdn_layers == other.gdn_layers &&
               local_kv_heads == other.local_kv_heads && kv_head_start == other.kv_head_start &&
               head_dim == other.head_dim && k_precision == other.k_precision &&
               v_precision == other.v_precision && kv_layout == other.kv_layout &&
               bytes_per_fa_layer_k == other.bytes_per_fa_layer_k &&
               bytes_per_fa_layer_v == other.bytes_per_fa_layer_v &&
               hybrid_host_encoding == other.hybrid_host_encoding &&
               hybrid_host_state_capacity_bytes == other.hybrid_host_state_capacity_bytes &&
               (hybrid_host_encoding == HybridPrefixHostEncoding::CPUFixedEpochPackedV2
                    ? (hybrid_host_state_capacity_bytes > 0 && hybrid_host_state_bytes > 0 &&
                       other.hybrid_host_state_bytes > 0 &&
                       hybrid_host_state_bytes <= hybrid_host_state_capacity_bytes &&
                       other.hybrid_host_state_bytes <= other.hybrid_host_state_capacity_bytes &&
                       hybrid_host_state_bytes <= std::numeric_limits<size_t>::max() - hybrid_device_state_bytes &&
                       other.hybrid_host_state_bytes <= std::numeric_limits<size_t>::max() - other.hybrid_device_state_bytes &&
                       hybrid_state_bytes == hybrid_host_state_bytes + hybrid_device_state_bytes &&
                       other.hybrid_state_bytes == other.hybrid_host_state_bytes + other.hybrid_device_state_bytes)
                    : (hybrid_host_state_bytes == other.hybrid_host_state_bytes &&
                       hybrid_state_bytes == other.hybrid_state_bytes)) &&
               hybrid_device_state_bytes == other.hybrid_device_state_bytes &&
               (!(includes_hybrid_state && other.includes_hybrid_state &&
                  hybrid_host_state_bytes > 0) ||
                hybrid_host_recurrence_layout == other.hybrid_host_recurrence_layout) &&
               (!(includes_hybrid_state && other.includes_hybrid_state &&
                  hybrid_device_state_bytes > 0) ||
                hybrid_device_recurrence_layout == other.hybrid_device_recurrence_layout) &&
               mtp_origin == other.mtp_origin && mtp_layers == other.mtp_layers &&
               mtp_local_kv_heads == other.mtp_local_kv_heads &&
               mtp_kv_head_start == other.mtp_kv_head_start && mtp_head_dim == other.mtp_head_dim &&
               mtp_k_precision == other.mtp_k_precision &&
               mtp_v_precision == other.mtp_v_precision && mtp_kv_layout == other.mtp_kv_layout &&
               bytes_per_mtp_layer_k == other.bytes_per_mtp_layer_k &&
               bytes_per_mtp_layer_v == other.bytes_per_mtp_layer_v &&
               mtp_kv_bytes == other.mtp_kv_bytes &&
               terminal_hidden_bytes == other.terminal_hidden_bytes &&
               terminal_logits_bytes == other.terminal_logits_bytes &&
               includes_mtp_state == other.includes_mtp_state;
    }

    void applyHybridPrefixPayloadExtent(PrefixPayloadLayout &layout, const IKVCache &cache, int sequence)
    {
        if (!layout.includes_hybrid_state || layout.hybrid_host_state_bytes == 0) return;
        const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
        if (!hybrid) throw std::invalid_argument("Hybrid prefix extent requires its canonical cache owner");
        const auto metadata = hybrid->hybridPrefixStateMetadata();
        if (layout.hybrid_host_encoding != metadata.host_encoding ||
            layout.hybrid_host_state_capacity_bytes != metadata.host_serialized_capacity_bytes ||
            layout.hybrid_host_recurrence_layout != metadata.host_recurrence_layout)
            throw std::invalid_argument("Hybrid prefix host capacity/format identity changed");
        HybridPrefixStateDescriptor request;
        request.seq_idx = sequence;
        request.include_device_state = false;
        request.host_encoding = layout.hybrid_host_encoding;
        request.host_recurrence_layout = layout.hybrid_host_recurrence_layout;
        const auto bytes = hybrid->hybridPrefixHostPayloadBytes(request);
        if (!bytes || *bytes == 0 || *bytes > layout.hybrid_host_state_capacity_bytes)
            throw std::invalid_argument("Hybrid prefix host phase/serialized extent is invalid");
        layout.hybrid_host_state_bytes = *bytes;
        layout.hybrid_state_bytes = addBytes(*bytes, layout.hybrid_device_state_bytes);
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
            layout.hybrid_host_recurrence_layout = metadata.host_recurrence_layout;
            layout.hybrid_device_recurrence_layout = metadata.device_recurrence_layout;
            layout.hybrid_host_encoding = metadata.host_encoding;
            layout.hybrid_host_state_capacity_bytes = metadata.host_serialized_capacity_bytes;
            layout.hybrid_host_state_bytes = metadata.host_encoding == HybridPrefixHostEncoding::CPUFixedEpochPackedV2
                                                ? metadata.host_serialized_capacity_bytes : metadata.host_bytes;
            layout.hybrid_device_state_bytes = metadata.device_bytes;
            layout.hybrid_state_bytes = addBytes(layout.hybrid_host_state_bytes, metadata.device_bytes);
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
        if (kv_cache.auxiliaryState())
            applyPrefixAuxiliaryLayout(layout, kv_cache, nullptr, 0, 0, block_size, block_size, true);
        return layout;
    }

    PrefixPayloadLayout qwen4ExpPrefixCapacityLayout(const PrefixPayloadLayout &base,
        qwen4exp::Qwen4ExpArchiveGeometry main, qwen4exp::Qwen4ExpArchiveGeometry mtp,
        int context_tokens)
    {
        using Geometry = qwen4exp::Qwen4ExpArchiveGeometry;
        if (base.block_size <= 0 || context_tokens <= 0 || base.fa_layers < 0 || base.mtp_layers < 0 ||
            static_cast<size_t>(base.fa_layers) != main.sparse_layers ||
            (!mtp.empty() && static_cast<size_t>(base.mtp_layers) != mtp.sparse_layers) || mtp.ple_channels)
            throw std::invalid_argument("QFN prefix capacity requires exact cache-local sparse/lexical ownership");
        // Dense shifted attention retains ordinary KV with no model auxiliary
        // owner. Empty auxiliary geometry says nothing about that KV extent;
        // its existing base.mtp_layers/mtp_kv_bytes remain fully charged. Every
        // nonempty QSA owner still matches the complete cache-local layer bank.
        const auto project = [](Geometry geometry, size_t first, size_t count, size_t prefix)
        {
            PrefixAuxiliaryPayloadLayout result;
            if (geometry.empty()) return result;
            result.kind = CacheAuxiliaryStateKind::QwenFlashNext;
            result.bytes_per_attention_layer = Geometry::blockBytes(first, count);
            result.endpoint_bytes = geometry.endpointBytes(prefix);
            result.includes_endpoint = true;
            return result;
        };
        PrefixPayloadLayout maximum = base;
        maximum.main_auxiliary = {};
        maximum.mtp_auxiliary = {};
        auto maximum_bytes = maximum.totalBytes();
        const size_t block = base.block_size, context = context_tokens;
        // The earliest positive block for each start residue admits at least
        // as many rows as any later representative. For a fixed end residue,
        // the greatest legal count maximizes completed pool rows. This bounds
        // the exact search to20 candidates instead of walking the context.
        for (size_t ordinal = 0; ordinal <= Geometry::CompressionRatio; ++ordinal)
        {
            const size_t first = Geometry::multiply(ordinal, block);
            if (first >= context) break;
            const size_t limit = std::min(block, context - first);
            for (size_t residue = 0; residue < Geometry::CompressionRatio; ++residue)
            {
                const size_t distance = (first + limit + Geometry::CompressionRatio - residue) % Geometry::CompressionRatio;
                if (distance >= limit) continue;
                const size_t count = limit - distance, prefix = first + count;
                PrefixPayloadLayout candidate = base;
                candidate.main_auxiliary = project(main, first, count, prefix);
                const auto shifted=mtpShiftedBlockRange(static_cast<int>(first),static_cast<int>(count),base.mtp_origin);
                candidate.mtp_auxiliary=project(mtp,shifted.first,shifted.count,shifted.end);
                const auto bytes = candidate.totalBytes();
                if (bytes > maximum_bytes)
                {
                    maximum = std::move(candidate);
                    maximum_bytes = bytes;
                }
            }
        }
        return maximum;
    }

    PrefixPayloadLayout prefixPayloadCapacityLayout(const PrefixPayloadLayout &base,
        const IKVCache &main, const IKVCache *mtp)
    {
        const auto project = [](const IKVCache &cache)
        {
            const auto *owner = cache.auxiliaryState();
            if (!owner) return qwen4exp::Qwen4ExpArchiveGeometry{};
            const auto *qfn = dynamic_cast<const Qwen4ExpAuxiliaryState *>(owner);
            if (!qfn || owner->kind() != CacheAuxiliaryStateKind::QwenFlashNext)
                throw std::invalid_argument("prefix capacity owner lacks an exact typed archive geometry");
            return qfn->archiveGeometry(0);
        };
        if (!main.auxiliaryState() && (!mtp || !mtp->auxiliaryState())) return base;
        if (mtp && mtp->max_seq_len() < mtpShiftedPrefixRows(main.max_seq_len(),mtp->stateOwnership().shifted_origin))
            throw std::invalid_argument("shifted prefix capacity is smaller than the admitted main horizon");
        return qwen4ExpPrefixCapacityLayout(base, project(main),
            mtp ? project(*mtp) : qwen4exp::Qwen4ExpArchiveGeometry{}, main.max_seq_len());
    }

} // namespace llaminar2
