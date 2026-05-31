#include "execution/prefix_cache/PrefixCacheStateProbe.h"

#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/IKVCache.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr uint64_t kFnvOffset = 14695981039346656037ull;
        constexpr uint64_t kFnvPrime = 1099511628211ull;

        uint64_t fnvUpdate(uint64_t hash, unsigned char byte)
        {
            hash ^= static_cast<uint64_t>(byte);
            hash *= kFnvPrime;
            return hash;
        }

        uint64_t hashFloatVector(const std::vector<float> &values)
        {
            return hashFloatBufferForPrefixProbe(values.data(), values.size());
        }
    } // namespace

    int PrefixRuntimeStateSnapshot::totalCachedTokens() const
    {
        int total = 0;
        for (const auto &cache : kv_caches)
        {
            for (const auto &layer : cache.layers)
            {
                total += layer.cached_tokens;
            }
        }
        return total;
    }

    int PrefixRuntimeStateSnapshot::totalMTPCachedTokens() const
    {
        int total = 0;
        for (const auto &cache : mtp_kv_caches)
        {
            for (const auto &layer : cache.layers)
            {
                total += layer.cached_tokens;
            }
        }
        return total;
    }

    PrefixKVCacheProbe inspectKVCacheForPrefixProbe(
        const IKVCache &cache,
        std::string owner,
        DeviceId device,
        int sequence_count)
    {
        PrefixKVCacheProbe probe;
        probe.owner = std::move(owner);
        probe.device = device;
        probe.first_layer_index = cache.first_layer_index();
        probe.n_layers = cache.n_layers();
        probe.max_seq_len = cache.max_seq_len();
        probe.n_kv_heads = cache.n_kv_heads();
        probe.local_n_kv_heads = cache.local_n_kv_heads();
        probe.kv_head_start = cache.kv_head_start();
        probe.graph_capture_ready = cache.isGraphCaptureReady();
        probe.k_precision = cache.k_precision();
        probe.v_precision = cache.v_precision();

        const int safe_sequence_count = std::max(1, sequence_count);
        probe.layers.reserve(static_cast<size_t>(std::max(0, probe.n_layers)) *
                             static_cast<size_t>(safe_sequence_count));
        for (int layer = 0; layer < probe.n_layers; ++layer)
        {
            for (int seq = 0; seq < safe_sequence_count; ++seq)
            {
                PrefixKVLayerProbe layer_probe;
                layer_probe.cache_layer = layer;
                layer_probe.global_layer = probe.first_layer_index + layer;
                layer_probe.seq_idx = seq;
                layer_probe.cached_tokens = cache.get_cached_tokens(layer, seq);
                layer_probe.ring_head = cache.ring_head(layer, seq);
                probe.layers.push_back(layer_probe);
            }
        }

        return probe;
    }

    std::vector<PrefixGDNLayerProbe> inspectHybridGDNForPrefixProbe(
        const IKVCache &cache)
    {
        const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
        if (!hybrid)
        {
            return {};
        }

        std::vector<PrefixGDNLayerProbe> probes;
        probes.reserve(static_cast<size_t>(std::max(0, hybrid->gdnLayerCount())));
        for (int layer = 0; layer < cache.n_layers(); ++layer)
        {
            if (!hybrid->isGDNLayer(layer))
            {
                continue;
            }

            const HybridGDNLayerState *state = hybrid->getGDNState(layer);
            if (!state)
            {
                continue;
            }

            PrefixGDNLayerProbe probe;
            probe.global_layer = cache.first_layer_index() + layer;
            probe.recurrence_values = state->recurrence_state.size();
            probe.conv_values = state->conv_state.size();
            probe.recurrence_hash = hashFloatVector(state->recurrence_state);
            probe.conv_hash = hashFloatVector(state->conv_state);
            probe.recurrence_all_zero = floatBufferAllZeroForPrefixProbe(
                state->recurrence_state.data(), state->recurrence_state.size());
            probe.conv_all_zero = floatBufferAllZeroForPrefixProbe(
                state->conv_state.data(), state->conv_state.size());
            probes.push_back(probe);
        }

        return probes;
    }

    uint64_t hashFloatBufferForPrefixProbe(const float *values, size_t count)
    {
        uint64_t hash = kFnvOffset;
        if (!values || count == 0)
        {
            return hash;
        }

        for (size_t i = 0; i < count; ++i)
        {
            uint32_t bits = 0;
            std::memcpy(&bits, values + i, sizeof(bits));
            for (int byte_idx = 0; byte_idx < 4; ++byte_idx)
            {
                hash = fnvUpdate(hash, static_cast<unsigned char>((bits >> (byte_idx * 8)) & 0xffu));
            }
        }
        return hash;
    }

    bool floatBufferAllZeroForPrefixProbe(const float *values, size_t count)
    {
        if (!values)
        {
            return count == 0;
        }
        for (size_t i = 0; i < count; ++i)
        {
            if (values[i] != 0.0f)
            {
                return false;
            }
        }
        return true;
    }

} // namespace llaminar2
