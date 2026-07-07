/**
 * @file PrefixCacheStateProbe.h
 * @brief Runtime state probe records for prefix-cache, KV, GDN, and MTP diagnostics.
 *
 * The structures in this file are intentionally value-oriented snapshots:
 * callers can capture them at request boundaries, compare them in parity tests,
 * and print compact hashes without retaining ownership of live inference
 * buffers.  KV segment hashes make prefix restore failures localizable without
 * copying full model caches into ordinary test output.
 */

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

    /**
     * @brief Hash of one requested logical KV token range.
     *
     * Prefix-restore diagnostics often need to isolate a bad suffix chunk
     * without copying an entire model cache into the failure message.  This
     * structure records the logical token range that was requested plus the
     * byte hashes exported from the cache when that full range was available.
     */
    struct PrefixKVSegmentProbe
    {
        std::string name;
        int token_start = 0;
        int token_count = 0;
        bool hash_available = false;
        size_t k_payload_bytes = 0;
        size_t v_payload_bytes = 0;
        uint64_t k_payload_hash = 0;
        uint64_t v_payload_hash = 0;
    };

    /**
     * @brief Per-layer KV state captured by PrefixRuntimeStateSnapshot.
     *
     * The leading/trailing fields are kept for stable legacy diagnostics.
     * Newer failure analysis should prefer @ref segments, which can describe
     * several named logical ranges in the same cache payload.
     */
    struct PrefixKVLayerProbe
    {
        int cache_layer = 0;
        int global_layer = 0;
        int seq_idx = 0;
        int cached_tokens = 0;
        int ring_head = 0;
        bool payload_hash_available = false;
        size_t k_payload_bytes = 0;
        size_t v_payload_bytes = 0;
        uint64_t k_payload_hash = 0;
        uint64_t v_payload_hash = 0;

        bool leading_segment_hash_available = false;
        int leading_segment_tokens = 0;
        size_t leading_k_payload_bytes = 0;
        size_t leading_v_payload_bytes = 0;
        uint64_t leading_k_payload_hash = 0;
        uint64_t leading_v_payload_hash = 0;

        bool trailing_segment_hash_available = false;
        int trailing_segment_start = 0;
        int trailing_segment_tokens = 0;
        size_t trailing_k_payload_bytes = 0;
        size_t trailing_v_payload_bytes = 0;
        uint64_t trailing_k_payload_hash = 0;
        uint64_t trailing_v_payload_hash = 0;
        std::vector<PrefixKVSegmentProbe> segments;
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
        bool device_state_hash_available = false;
        size_t recurrence_device_bytes = 0;
        size_t conv_device_bytes = 0;
        uint64_t recurrence_device_hash = 0;
        uint64_t conv_device_hash = 0;
        bool local_device_state_hash_available = false;
        size_t recurrence_local_device_bytes = 0;
        size_t conv_local_device_bytes = 0;
        uint64_t recurrence_local_device_hash = 0;
        uint64_t conv_local_device_hash = 0;
        bool recurrence_all_zero = true;
        bool conv_all_zero = true;
        /**
         * @brief Optional raw state copies for deep verifier-state diagnostics.
         *
         * Normal prefix probes only carry hashes so request summaries stay
         * cheap.  Setting LLAMINAR_PREFIX_PROBE_CAPTURE_GDN_VALUES=1 asks the
         * probe to copy full FP32 GDN state into these vectors, which is useful
         * for parity tests that need tolerance-aware comparisons between
         * decode-equivalent state publication and serial decode.
         */
        std::vector<float> recurrence_sample_values;
        std::vector<float> conv_sample_values;
    };

    struct PrefillGraphRuntimeProbe
    {
        bool forward_cache_valid = false;
        bool prefill_cache_initialized = false;
        std::string phase = "cold";
        size_t cache_size = 0;
        size_t node_count = 0;
        int replay_count = 0;
        uint64_t warmup_count = 0;
        uint64_t initialized_count = 0;
        uint64_t capture_count = 0;
        uint64_t eviction_count = 0;
        bool observation_valid = false;
        int chunk_index = 0;
        int bucket_seq_len = 0;
        int real_token_start = 0;
        int real_token_count = 0;
        int real_token_end = 0;
        std::string domain_id;
        int participant_id = 0;
        uint64_t placement_epoch = 0;
        uint64_t topology_signature = 0;
        std::string capture_phase;
        std::string recapture_reason;
        std::string reject_stage_name;
        std::string reject_stage_type;
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
        uint64_t moe_runtime_movement_epoch = 0;
        uint64_t live_state_epoch = 0;
        uint64_t live_state_mutations = 0;
        std::string last_live_state_mutation_reason;
        std::string last_live_state_mutation_operation;
        uint64_t live_state_accepted_publications = 0;
        uint64_t live_state_rejected_corrections = 0;
        uint64_t live_state_prefix_restores = 0;
        uint64_t live_state_prefix_truncates = 0;
        uint64_t live_state_session_resets = 0;
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
        uint64_t mtp_stochastic_accept_tests = 0;
        uint64_t mtp_stochastic_accepts = 0;
        uint64_t mtp_stochastic_residual_samples = 0;
        uint64_t mtp_stochastic_terminal_samples = 0;
        uint64_t mtp_transaction_commits = 0;
        uint64_t mtp_transaction_rollbacks = 0;
        uint64_t mtp_transaction_validation_failures = 0;
        uint64_t mtp_unsafe_verifier_state_rejections = 0;
        uint64_t mtp_depth_policy_windows = 0;
        uint64_t mtp_depth_policy_updates = 0;
        uint64_t mtp_depth_policy_promotions = 0;
        uint64_t mtp_depth_policy_demotions = 0;
        uint64_t mtp_depth_policy_observe_recommendations = 0;
        int mtp_current_depth = 0;
        int mtp_min_depth = 0;
        int mtp_max_depth = 0;
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
        std::vector<PrefillGraphRuntimeProbe> prefill_graphs;

        int totalCachedTokens() const;
        int totalMTPCachedTokens() const;
        bool hasAnyKVState() const { return totalCachedTokens() > 0; }
    };

    PrefixKVCacheProbe inspectKVCacheForPrefixProbe(
        const IKVCache &cache,
        std::string owner,
        DeviceId device,
        int sequence_count = 1,
        void *stream = nullptr);

    std::vector<PrefixGDNLayerProbe> inspectHybridGDNForPrefixProbe(
        const IKVCache &cache,
        void *stream = nullptr);

    uint64_t hashFloatBufferForPrefixProbe(const float *values, size_t count);
    uint64_t hashByteBufferForPrefixProbe(const void *values, size_t bytes);
    bool floatBufferAllZeroForPrefixProbe(const float *values, size_t count);

} // namespace llaminar2
