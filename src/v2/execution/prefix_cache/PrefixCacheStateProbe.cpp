/**
 * @file PrefixCacheStateProbe.cpp
 * @brief Builds compact runtime-state diagnostics for prefix-cache parity tests.
 *
 * Probes intentionally read through public cache export interfaces so failures
 * exercise the same logical ownership boundary used by prefix save/restore.
 * Environment-controlled payload hashing is kept opt-in because full KV/GDN
 * exports are expensive on long-context model runs.
 */

#include "execution/prefix_cache/PrefixCacheStateProbe.h"

#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/IKVCache.h"
#include "tensors/TensorKernels.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
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

        int envIntOrDefault(const char *name, int fallback)
        {
            const char *value = DebugEnv::envValue(name);
            if (!value || !*value)
                return fallback;
            return std::atoi(value);
        }

        std::string trimCopy(const std::string &value)
        {
            size_t first = 0;
            while (first < value.size() &&
                   std::isspace(static_cast<unsigned char>(value[first])) != 0)
            {
                ++first;
            }
            size_t last = value.size();
            while (last > first &&
                   std::isspace(static_cast<unsigned char>(value[last - 1])) != 0)
            {
                --last;
            }
            return value.substr(first, last - first);
        }

        int parseNonNegativeIntStrict(
            const std::string &value,
            const char *field_name,
            const std::string &segment_spec)
        {
            const std::string trimmed = trimCopy(value);
            if (trimmed.empty())
            {
                throw std::runtime_error(
                    std::string("empty ") + field_name +
                    " in LLAMINAR_PREFIX_PROBE_KV_SEGMENTS entry '" +
                    segment_spec + "'");
            }

            int parsed = 0;
            for (char ch : trimmed)
            {
                if (!std::isdigit(static_cast<unsigned char>(ch)))
                {
                    throw std::runtime_error(
                        std::string("non-numeric ") + field_name +
                        " in LLAMINAR_PREFIX_PROBE_KV_SEGMENTS entry '" +
                        segment_spec + "'");
                }
                parsed = parsed * 10 + (ch - '0');
            }
            return parsed;
        }

        std::vector<PrefixKVSegmentProbe> parseRequestedKVSegments()
        {
            const char *raw = DebugEnv::envValue("LLAMINAR_PREFIX_PROBE_KV_SEGMENTS");
            if (!raw || !*raw)
                return {};

            std::vector<PrefixKVSegmentProbe> segments;
            std::stringstream stream(raw);
            std::string entry;
            int unnamed_index = 0;
            while (std::getline(stream, entry, ','))
            {
                std::stringstream semi_stream(entry);
                std::string semi_entry;
                while (std::getline(semi_stream, semi_entry, ';'))
                {
                    const std::string spec = trimCopy(semi_entry);
                    if (spec.empty())
                        continue;

                    std::string name;
                    std::string range = spec;
                    const size_t equals_pos = spec.find('=');
                    if (equals_pos != std::string::npos)
                    {
                        name = trimCopy(spec.substr(0, equals_pos));
                        range = trimCopy(spec.substr(equals_pos + 1));
                        if (name.empty())
                        {
                            throw std::runtime_error(
                                "empty segment name in LLAMINAR_PREFIX_PROBE_KV_SEGMENTS entry '" +
                                spec + "'");
                        }
                    }
                    else
                    {
                        name = "segment" + std::to_string(unnamed_index++);
                    }

                    const size_t colon_pos = range.find(':');
                    if (colon_pos == std::string::npos ||
                        range.find(':', colon_pos + 1) != std::string::npos)
                    {
                        throw std::runtime_error(
                            "LLAMINAR_PREFIX_PROBE_KV_SEGMENTS entry '" + spec +
                            "' must use start:count or name=start:count");
                    }

                    PrefixKVSegmentProbe segment;
                    segment.name = std::move(name);
                    segment.token_start = parseNonNegativeIntStrict(
                        range.substr(0, colon_pos),
                        "token_start",
                        spec);
                    segment.token_count = parseNonNegativeIntStrict(
                        range.substr(colon_pos + 1),
                        "token_count",
                        spec);
                    if (segment.token_count <= 0)
                    {
                        throw std::runtime_error(
                            "LLAMINAR_PREFIX_PROBE_KV_SEGMENTS entry '" + spec +
                            "' must request a positive token_count");
                    }
                    segments.push_back(std::move(segment));
                }
            }
            return segments;
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

    PrefixProbeCapturePolicy PrefixProbeCapturePolicy::fromEnvironment()
    {
        PrefixProbeCapturePolicy policy;
        policy.hash_full_kv_payloads =
            envEnabled("LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS");
        policy.hash_default_kv_segments =
            envEnabled("LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS");
        policy.default_kv_segment_split_tokens =
            envIntOrDefault("LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT", 4);
        if (policy.hash_default_kv_segments)
            policy.requested_kv_segments = parseRequestedKVSegments();
        policy.hash_gdn_device_state =
            envEnabled("LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE");
        policy.capture_gdn_values =
            envEnabled("LLAMINAR_PREFIX_PROBE_CAPTURE_GDN_VALUES");
        return policy;
    }

    PrefixKVCacheProbe inspectKVCacheForPrefixProbe(
        const IKVCache &cache,
        std::string owner,
        DeviceId device,
        int sequence_count,
        void *stream)
    {
        return inspectKVCacheForPrefixProbe(
            cache,
            std::move(owner),
            device,
            sequence_count,
            stream,
            PrefixProbeCapturePolicy::fromEnvironment());
    }

    PrefixKVCacheProbe inspectKVCacheForPrefixProbe(
        const IKVCache &cache,
        std::string owner,
        DeviceId device,
        int sequence_count,
        void *stream,
        const PrefixProbeCapturePolicy &capture_policy)
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
                    capture_policy.hash_full_kv_payloads)
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
                const bool capture_any_segment =
                    capture_policy.hash_default_kv_segments ||
                    !capture_policy.requested_kv_segments.empty() ||
                    capture_policy.trailing_kv_tokens > 0;
                if (layer_probe.cached_tokens > 0 && capture_any_segment)
                {
                    auto hash_segment =
                        [&](int token_start,
                            int token_count,
                            bool *available,
                            size_t *k_bytes_out,
                            size_t *v_bytes_out,
                            uint64_t *k_hash_out,
                            uint64_t *v_hash_out) -> void
                    {
                        if (token_start < 0 ||
                            token_count <= 0 ||
                            token_start > layer_probe.cached_tokens ||
                            token_count > layer_probe.cached_tokens - token_start)
                        {
                            return;
                        }
                        const auto layout = cache.logicalBlockLayout(
                            layer_probe.global_layer,
                            token_count);
                        if (layout.k_bytes == 0 || layout.v_bytes == 0)
                            return;

                        std::vector<uint8_t> k_bytes(layout.k_bytes);
                        std::vector<uint8_t> v_bytes(layout.v_bytes);
                        IKVCache::KVCacheLogicalBlockDescriptor desc;
                        desc.layer = layer_probe.global_layer;
                        desc.seq_idx = seq;
                        desc.logical_token_start = token_start;
                        desc.token_count = token_count;
                        desc.stream = stream;
                        if (!cache.exportLogicalBlock(
                                desc,
                                k_bytes.data(),
                                v_bytes.data()))
                        {
                            return;
                        }

                        *available = true;
                        *k_bytes_out = layout.k_bytes;
                        *v_bytes_out = layout.v_bytes;
                        *k_hash_out = hashByteBufferForPrefixProbe(
                            k_bytes.data(),
                            k_bytes.size());
                        *v_hash_out = hashByteBufferForPrefixProbe(
                            v_bytes.data(),
                            v_bytes.size());
                    };

                    auto capture_named_segment =
                        [&](PrefixKVSegmentProbe segment) -> PrefixKVSegmentProbe
                    {
                        hash_segment(
                            segment.token_start,
                            segment.token_count,
                            &segment.hash_available,
                            &segment.k_payload_bytes,
                            &segment.v_payload_bytes,
                            &segment.k_payload_hash,
                            &segment.v_payload_hash);
                        return segment;
                    };

                    if (capture_policy.hash_default_kv_segments &&
                        layer_probe.cached_tokens > 1)
                    {
                        const int split_tokens = std::clamp(
                            capture_policy.default_kv_segment_split_tokens,
                            1,
                            layer_probe.cached_tokens - 1);
                        layer_probe.leading_segment_tokens = split_tokens;
                        hash_segment(
                            0,
                            split_tokens,
                            &layer_probe.leading_segment_hash_available,
                            &layer_probe.leading_k_payload_bytes,
                            &layer_probe.leading_v_payload_bytes,
                            &layer_probe.leading_k_payload_hash,
                            &layer_probe.leading_v_payload_hash);

                        layer_probe.trailing_segment_start = split_tokens;
                        layer_probe.trailing_segment_tokens =
                            layer_probe.cached_tokens - split_tokens;
                        hash_segment(
                            split_tokens,
                            layer_probe.trailing_segment_tokens,
                            &layer_probe.trailing_segment_hash_available,
                            &layer_probe.trailing_k_payload_bytes,
                            &layer_probe.trailing_v_payload_bytes,
                            &layer_probe.trailing_k_payload_hash,
                            &layer_probe.trailing_v_payload_hash);
                    }

                    for (const auto &requested_segment :
                         capture_policy.requested_kv_segments)
                    {
                        layer_probe.segments.push_back(
                            capture_named_segment(requested_segment));
                    }

                    if (capture_policy.trailing_kv_tokens > 0)
                    {
                        PrefixKVSegmentProbe trailing;
                        trailing.name = "diagnostic_tail";
                        trailing.token_count = std::min(
                            capture_policy.trailing_kv_tokens,
                            layer_probe.cached_tokens);
                        trailing.token_start =
                            layer_probe.cached_tokens - trailing.token_count;
                        layer_probe.segments.push_back(
                            capture_named_segment(std::move(trailing)));
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
        return inspectHybridGDNForPrefixProbe(
            cache,
            stream,
            PrefixProbeCapturePolicy::fromEnvironment());
    }

    std::vector<PrefixGDNLayerProbe> inspectHybridGDNForPrefixProbe(
        const IKVCache &cache,
        void *stream,
        const PrefixProbeCapturePolicy &capture_policy)
    {
        const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
        if (!hybrid)
        {
            return {};
        }

        std::vector<uint8_t> device_state_bytes;
        if (capture_policy.hash_gdn_device_state)
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
                /*
                 * exportHybridPrefixState serializes each GDN layer as:
                 *
                 *   local conv, optional full conv,
                 *   local recurrence, optional full recurrence.
                 *
                 * Parse those exact bank sizes.  The previous probe advanced by
                 * largestStateBytes() once per kernel; under LocalTP that
                 * hashed a range spanning the local bank and the beginning of
                 * the full bank, then misaligned every following layer.
                 */
                auto consume_bank =
                    [&](size_t byte_count,
                        size_t *reported_bytes,
                        uint64_t *reported_hash) -> bool
                {
                    *reported_bytes = byte_count;
                    if (byte_count == 0 ||
                        device_offset > device_state_bytes.size() ||
                        byte_count > device_state_bytes.size() - device_offset)
                    {
                        return false;
                    }
                    *reported_hash = hashByteBufferForPrefixProbe(
                        device_state_bytes.data() + device_offset,
                        byte_count);
                    device_offset += byte_count;
                    return true;
                };

                const bool has_conv =
                    const_cast<IHybridKVCache *>(hybrid)->getConvKernel(layer) !=
                    nullptr;
                const bool has_recurrence =
                    const_cast<IHybridKVCache *>(hybrid)
                        ->getRecurrenceKernel(layer) != nullptr;
                bool local_complete = true;
                bool full_complete = true;
                bool has_local_bank = false;
                bool has_full_bank = false;

                if (has_conv && state->local_conv_state_size > 0)
                {
                    has_local_bank = true;
                    local_complete &=
                        consume_bank(
                            static_cast<size_t>(state->local_conv_state_size) *
                                sizeof(float),
                            &probe.conv_local_device_bytes,
                            &probe.conv_local_device_hash);

                    if (state->full_conv_state_size !=
                        state->local_conv_state_size)
                    {
                        has_full_bank = true;
                        full_complete &=
                            consume_bank(
                                static_cast<size_t>(state->full_conv_state_size) *
                                    sizeof(float),
                                &probe.conv_device_bytes,
                                &probe.conv_device_hash);
                    }
                    else
                    {
                        has_full_bank = true;
                        probe.conv_device_bytes =
                            probe.conv_local_device_bytes;
                        probe.conv_device_hash =
                            probe.conv_local_device_hash;
                    }
                }
                if (has_recurrence &&
                    state->local_recurrence_state_size > 0)
                {
                    has_local_bank = true;
                    local_complete &=
                        consume_bank(
                            static_cast<size_t>(
                                state->local_recurrence_state_size) *
                                sizeof(float),
                            &probe.recurrence_local_device_bytes,
                            &probe.recurrence_local_device_hash);

                    if (state->full_recurrence_state_size !=
                        state->local_recurrence_state_size)
                    {
                        has_full_bank = true;
                        full_complete &=
                            consume_bank(
                                static_cast<size_t>(
                                    state->full_recurrence_state_size) *
                                    sizeof(float),
                                &probe.recurrence_device_bytes,
                                &probe.recurrence_device_hash);
                    }
                    else
                    {
                        has_full_bank = true;
                        probe.recurrence_device_bytes =
                            probe.recurrence_local_device_bytes;
                        probe.recurrence_device_hash =
                            probe.recurrence_local_device_hash;
                    }
                }

                probe.local_device_state_hash_available =
                    has_local_bank && local_complete;
                probe.device_state_hash_available =
                    has_full_bank && full_complete;
            }
            if (capture_policy.capture_gdn_values)
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
