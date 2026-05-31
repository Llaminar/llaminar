#pragma once

#include "backends/DeviceId.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/prefix_cache/PrefixCacheStats.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    class IKVCache;

    struct PrefixKVLayerProbe
    {
        int cache_layer = 0;
        int global_layer = 0;
        int seq_idx = 0;
        int cached_tokens = 0;
        int ring_head = 0;
    };

    struct PrefixKVCacheProbe
    {
        std::string owner;
        DeviceId device = DeviceId::cpu();
        int first_layer_index = 0;
        int n_layers = 0;
        int max_seq_len = 0;
        int n_kv_heads = 0;
        int local_n_kv_heads = 0;
        int kv_head_start = 0;
        bool graph_capture_ready = false;
        ActivationPrecision k_precision = ActivationPrecision::FP32;
        ActivationPrecision v_precision = ActivationPrecision::FP32;
        std::vector<PrefixKVLayerProbe> layers;
    };

    struct PrefixGDNLayerProbe
    {
        int global_layer = 0;
        size_t recurrence_values = 0;
        size_t conv_values = 0;
        uint64_t recurrence_hash = 0;
        uint64_t conv_hash = 0;
        bool recurrence_all_zero = true;
        bool conv_all_zero = true;
    };

    struct PrefixRuntimeStateSnapshot
    {
        bool initialized = false;
        bool prefill_logits_ready = false;
        bool has_hidden = false;
        bool has_logits = false;
        std::string architecture;
        std::string execution_path;
        DeviceId primary_device = DeviceId::cpu();
        int current_position = 0;
        uint64_t session_epoch = 0;
        bool prefix_cache_config_enabled = false;
        bool prefix_cache_ready = false;
        bool prefix_cache_bypassed = false;
        std::string prefix_cache_bypass_reason;
        uint64_t prefix_cache_lookups = 0;
        uint64_t prefix_cache_hits = 0;
        uint64_t prefix_cache_partial_hits = 0;
        uint64_t prefix_cache_misses = 0;
        uint64_t prefix_cache_matched_blocks = 0;
        uint64_t prefix_cache_matched_tokens = 0;
        uint64_t prefix_cache_stores = 0;
        uint64_t prefix_cache_inserts = 0;
        uint64_t prefix_cache_evictions = 0;
        uint64_t prefix_cache_promotions = 0;
        uint64_t prefix_cache_disk_hydrations = 0;
        uint64_t prefix_cache_terminal_state_hits = 0;
        uint64_t prefix_cache_ram_bytes = 0;
        uint64_t prefix_cache_device_bytes = 0;
        uint64_t prefix_cache_disk_bytes = 0;
        uint64_t prefix_cache_hybrid_state_bytes = 0;
        uint64_t prefix_cache_mtp_state_bytes = 0;
        uint64_t prefix_cache_bypasses = 0;
        uint64_t prefix_cache_unsupported_backend_bypasses = 0;
        uint64_t prefix_cache_fingerprint_bypasses = 0;
        uint64_t prefix_cache_terminal_state_bypasses = 0;
        bool mtp_config_enabled = false;
        bool mtp_bypassed = false;
        std::string mtp_bypass_reason;
        uint64_t mtp_draft_steps = 0;
        uint64_t mtp_accepted_tokens = 0;
        uint64_t mtp_rejected_tokens = 0;
        uint64_t mtp_rollbacks = 0;
        uint64_t mtp_bypasses = 0;
        uint64_t mtp_verifier_runs = 0;
        uint64_t mtp_verifier_token_count = 0;
        uint64_t prefill_chunk_schedules = 0;
        uint64_t prefill_chunk_successful_schedules = 0;
        uint64_t prefill_chunks = 0;
        uint64_t prefill_chunk_real_tokens = 0;
        uint64_t prefill_chunk_padded_tokens = 0;
        uint64_t prefill_chunk_failures = 0;
        PrefixCacheRequestSummary prefix_request;
        MTPRequestSummary mtp_request;
        std::vector<int> positions;
        std::vector<int> sequence_lengths;
        std::vector<PrefixKVCacheProbe> kv_caches;
        std::vector<PrefixKVCacheProbe> mtp_kv_caches;
        std::vector<PrefixGDNLayerProbe> gdn_layers;

        int totalCachedTokens() const;
        int totalMTPCachedTokens() const;
        bool hasAnyKVState() const { return totalCachedTokens() > 0; }
    };

    PrefixKVCacheProbe inspectKVCacheForPrefixProbe(
        const IKVCache &cache,
        std::string owner,
        DeviceId device,
        int sequence_count = 1);

    std::vector<PrefixGDNLayerProbe> inspectHybridGDNForPrefixProbe(
        const IKVCache &cache);

    uint64_t hashFloatBufferForPrefixProbe(const float *values, size_t count);
    bool floatBufferAllZeroForPrefixProbe(const float *values, size_t count);

} // namespace llaminar2
