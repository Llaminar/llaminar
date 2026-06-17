#include "execution/prefix_cache/PrefixCacheStateProbe.h"

#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/IKVCache.h"
#include "tensors/TensorKernels.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <utility>
#include <vector>

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

        bool envEnabled(const char *name)
        {
            return !DebugEnv::isFalseyEnv(name);
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
        int sequence_count,
        void *stream)
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
                if (layer_probe.cached_tokens > 0 &&
                    envEnabled("LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS"))
                {
                    const auto layout = cache.logicalBlockLayout(
                        layer_probe.global_layer,
                        layer_probe.cached_tokens);
                    if (layout.k_bytes > 0 && layout.v_bytes > 0)
                    {
                        std::vector<uint8_t> k_bytes(layout.k_bytes);
                        std::vector<uint8_t> v_bytes(layout.v_bytes);
                        IKVCache::KVCacheLogicalBlockDescriptor desc;
                        desc.layer = layer_probe.global_layer;
                        desc.seq_idx = seq;
                        desc.logical_token_start = 0;
                        desc.token_count = layer_probe.cached_tokens;
                        desc.stream = stream;
                        if (cache.exportLogicalBlock(
                                desc,
                                k_bytes.data(),
                                v_bytes.data()))
                        {
                            layer_probe.payload_hash_available = true;
                            layer_probe.k_payload_bytes = layout.k_bytes;
                            layer_probe.v_payload_bytes = layout.v_bytes;
                            layer_probe.k_payload_hash =
                                hashByteBufferForPrefixProbe(
                                    k_bytes.data(),
                                    k_bytes.size());
                            layer_probe.v_payload_hash =
                                hashByteBufferForPrefixProbe(
                                    v_bytes.data(),
                                    v_bytes.size());
                        }
                    }
                }
                probe.layers.push_back(layer_probe);
            }
        }

        return probe;
    }

    std::vector<PrefixGDNLayerProbe> inspectHybridGDNForPrefixProbe(
        const IKVCache &cache,
        void *stream)
    {
        const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
        if (!hybrid)
        {
            return {};
        }

        std::vector<uint8_t> device_state_bytes;
        const bool capture_device_state =
            envEnabled("LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE");
        if (capture_device_state)
        {
            const HybridPrefixStateMetadata metadata =
                hybrid->hybridPrefixStateMetadata();
            if (metadata.device_bytes > 0)
            {
                device_state_bytes.resize(metadata.device_bytes);
                HybridPrefixStateDescriptor desc;
                desc.stream = stream;
                desc.synchronize = true;
                desc.include_host_state = false;
                desc.include_device_state = true;
                if (!hybrid->exportHybridPrefixState(
                        desc,
                        device_state_bytes.data(),
                        nullptr))
                {
                    device_state_bytes.clear();
                }
            }
        }
        size_t device_offset = 0;

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
            if (!device_state_bytes.empty())
            {
                if (auto *conv_kernel =
                        const_cast<IHybridKVCache *>(hybrid)->getConvKernel(layer))
                {
                    probe.conv_device_bytes = conv_kernel->stateBytes();
                    if (device_offset + probe.conv_device_bytes <=
                        device_state_bytes.size())
                    {
                        probe.conv_device_hash =
                            hashByteBufferForPrefixProbe(
                                device_state_bytes.data() + device_offset,
                                probe.conv_device_bytes);
                        probe.device_state_hash_available = true;
                    }
                    device_offset += probe.conv_device_bytes;
                }
                if (auto *recurrence_kernel =
                        const_cast<IHybridKVCache *>(hybrid)->getRecurrenceKernel(layer))
                {
                    probe.recurrence_device_bytes =
                        recurrence_kernel->stateBytes();
                    if (device_offset + probe.recurrence_device_bytes <=
                        device_state_bytes.size())
                    {
                        probe.recurrence_device_hash =
                            hashByteBufferForPrefixProbe(
                                device_state_bytes.data() + device_offset,
                                probe.recurrence_device_bytes);
                        probe.device_state_hash_available = true;
                    }
                    device_offset += probe.recurrence_device_bytes;
                }
            }
            if (envEnabled("LLAMINAR_PREFIX_PROBE_CAPTURE_GDN_VALUES"))
            {
                probe.recurrence_sample_values = state->recurrence_state;
                probe.conv_sample_values = state->conv_state;
            }
            probes.push_back(probe);
        }

        return probes;
    }

    uint64_t hashFloatBufferForPrefixProbe(const float *values, size_t count)
    {
        return hashByteBufferForPrefixProbe(values, count * sizeof(float));
    }

    uint64_t hashByteBufferForPrefixProbe(const void *values, size_t bytes)
    {
        uint64_t hash = kFnvOffset;
        if (!values || bytes == 0)
        {
            return hash;
        }

        const auto *raw = static_cast<const unsigned char *>(values);
        for (size_t i = 0; i < bytes; ++i)
        {
            hash = fnvUpdate(hash, raw[i]);
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
