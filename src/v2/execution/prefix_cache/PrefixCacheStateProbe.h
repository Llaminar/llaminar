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

        /**
         * @brief Optional canonical native-precision K bytes for deep parity.
         *
         * Ordinary probes retain only hashes.  A typed capture policy may keep
         * the bytes already exported for one bounded segment so a later state
         * comparison can prove numerical equivalence without issuing another
         * device transfer.  These bytes use the enclosing cache's declared K
         * precision and canonical logical-block layout.
         */
        std::vector<uint8_t> k_payload;

        /**
         * @brief Optional canonical native-precision V bytes for deep parity.
         *
         * Ownership and layout are identical to @ref k_payload.  Keeping K and
         * V independent makes a one-sided cache defect visible rather than
         * allowing a combined digest or metric to conceal it.
         */
        std::vector<uint8_t> v_payload;
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

    /**
     * @brief Explicitly selects expensive payloads included in a state probe.
     *
     * Ordinary request summaries need inventory and logical-position metadata,
     * but they must not copy complete GPU state to the host.  Deep parity and
     * failed-mirror diagnostics need a different contract: they select the
     * exact byte ranges that make persistent-state divergence localizable.
     *
     * Keeping this policy as a typed value prevents diagnostics from mutating
     * process-wide environment variables in the middle of inference.  The
     * legacy environment-controlled entry points remain available through
     * @ref fromEnvironment for command-line diagnostics and existing tests.
     */
    struct PrefixProbeCapturePolicy
    {
        /// Hash every logical K/V token currently retained by each FA layer.
        bool hash_full_kv_payloads = false;

        /// Hash the established leading/trailing split used by prefix tests.
        bool hash_default_kv_segments = false;

        /// Number of leading tokens in the established split.
        int default_kv_segment_split_tokens = 4;

        /// Additional fixed logical ranges to export and hash.
        std::vector<PrefixKVSegmentProbe> requested_kv_segments;

        /**
         * @brief Retain canonical bytes for explicitly requested KV segments.
         *
         * This never retains a full cache implicitly: only entries named in
         * @ref requested_kv_segments receive payload vectors.  It is intended
         * for bounded parity boundaries such as the single suffix row
         * recomputed after a partial prefix hit.
         */
        bool capture_requested_kv_segment_payloads = false;

        /**
         * @brief Hash only the newest N logical K/V tokens.
         *
         * Failed grouped-verifier diagnostics use this bounded export.  The
         * newest speculative rows reveal the first bad full-attention layer
         * without copying a multi-thousand-token cache merely to report an
         * already-failed transaction.
         */
        int trailing_kv_tokens = 0;

        /// Export and hash every GPU-resident local/full GDN state bank.
        bool hash_gdn_device_state = false;

        /**
         * @brief Hash the one-row terminal logits and MTP hidden mailboxes.
         *
         * These buffers are the bridge from a completed prefill or prefix
         * restore into the first decode/MTP transaction.  Their hashes are
         * opt-in because observing GPU-owned bytes requires a diagnostic D2H
         * result transfer.  Production inference never enables this policy.
         */
        bool hash_terminal_state = false;

        /**
         * @brief Retain the FP32 terminal-hidden payload for numerical evidence.
         *
         * Expert migration can move an otherwise identical routed expert
         * between CPU and GPU kernels.  Those kernels are required to remain
         * numerically equivalent, but they are not required to produce the
         * same floating-point bytes across device types.  Deep parity probes
         * may retain the terminal-hidden row that was already materialized for
         * hashing so a placement-aware comparison can prove equivalence
         * without issuing a second device transfer.  Ordinary runtime probes
         * leave this disabled.
         */
        bool capture_terminal_hidden_values = false;

        /**
         * @brief Retain the FP32 terminal-logits payload for numerical evidence.
         *
         * A partial prefix hit recomputes its uncached suffix.  When an expert
         * moved between device types after the serial oracle was recorded, the
         * recomputed logits can be mathematically equivalent without being
         * byte-identical.  Deep parity probes retain the already-materialized
         * logits row so that transition is certified rather than ignored.
         */
        bool capture_terminal_logits_values = false;

        /**
         * @brief Retain complete authoritative FP32 GDN values.
         *
         * CPU caches copy their host-owned state. GPU caches reuse the same
         * explicit-stream device export used for hashing and select the full
         * replicated decode bank rather than a stale host mirror.
         */
        bool capture_gdn_values = false;

        /**
         * @brief Materialize canonical GPU logical-position metadata.
         *
         * Runtime summaries deliberately leave GPU position and sequence-length
         * vectors empty: those values are device owned and must not introduce a
         * hidden D2H merely because logging, benchmark aggregation, or PerfStats
         * requested a snapshot. Focused integration diagnostics may opt in with
         * `LLAMINAR_PREFIX_PROBE_CAPTURE_DEVICE_LOGICAL_STATE=1`; callers must
         * then treat the probe as an explicit host-visible result boundary.
         */
        bool capture_device_logical_state = false;

        /**
         * @brief Build the compatibility policy selected by diagnostic env vars.
         *
         * @return Capture policy corresponding to the existing
         *         `LLAMINAR_PREFIX_PROBE_*` controls.
         */
        static PrefixProbeCapturePolicy fromEnvironment();
    };

    struct PrefixGDNLayerProbe
    {
        int global_layer = 0;
        size_t recurrence_values = 0;
        size_t conv_values = 0;
        uint64_t recurrence_hash = 0;
        uint64_t conv_hash = 0;

        /**
         * @brief Full replicated decode-bank hashes.
         *
         * LocalTP GPU caches serialize participant-local and full replicated
         * banks independently.  These fields always describe the full bank
         * selected by decode/grouped-verifier execution, never a byte range
         * spanning two adjacent serialized banks.
         */
        bool device_state_hash_available = false;
        size_t recurrence_device_bytes = 0;
        size_t conv_device_bytes = 0;
        uint64_t recurrence_device_hash = 0;
        uint64_t conv_device_hash = 0;

        /// Participant-local prefill-bank hashes from the same device export.
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
         * cheap. Setting LLAMINAR_PREFIX_PROBE_CAPTURE_GDN_VALUES=1 asks the
         * probe to copy the complete authoritative FP32 bank into these
         * vectors. On GPU this is the full replicated decode bank exported on
         * the probe's exact stream, never the potentially stale host mirror.
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
        uint64_t prefix_cache_ram_to_disk_demotions = 0;
        uint64_t prefix_cache_device_hot_promotions = 0;
        uint64_t prefix_cache_device_hot_repromotions = 0;
        uint64_t prefix_cache_device_hot_evictions = 0;
        uint64_t prefix_cache_disk_evictions = 0;
        uint64_t prefix_cache_device_hot_direct_hits = 0;
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
        int mtp_last_transaction_draft_depth = 0;
        int mtp_last_transaction_emitted_token_count = 0;
        /** Device-controller transaction ordinal paired with the observed row. */
        int mtp_observed_verifier_transaction_count = 0;
        /** Exact active draft depth paired with the observed row. */
        int mtp_observed_verifier_draft_depth = 0;
        /**
         * @brief Draft-token identity of the last committed verifier transaction.
         *
         * Populated when ordinary tensor snapshots or explicit device logical-
         * state diagnostics are enabled. The reusable verifier row is copied
         * into a fixed record by the same device lane that commits response and
         * controller state; its transaction ordinal, depth, and token prefix
         * therefore cannot name different transactions. These bytes never
         * participate in controller decisions. The vector contains exactly
         * `mtp_observed_verifier_draft_depth` draft tokens (the target token is
         * retained in the device record but intentionally omitted here).
         */
        std::vector<int32_t> mtp_observed_verifier_draft_tokens;
        /**
         * @brief Logical position consumed by the next MTP sidecar transaction.
         *
         * CPU runners publish their host-owned live position. GPU runners
         * publish the scheduler-owned transaction coordinate initialized at
         * prefill and advanced only by validated device state publication. A
         * negative value means no MTP condition boundary is currently live.
         */
        int mtp_next_condition_position = -1;
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
        bool terminal_hidden_hash_available = false;
        size_t terminal_hidden_bytes = 0;
        uint64_t terminal_hidden_hash = 0;
        /**
         * @brief Optional full FP32 terminal-hidden values in participant order.
         *
         * A rank/global aggregate concatenates child rows in the same stable
         * order used to fold their hashes.  The vector is populated only by an
         * explicit deep-probe policy and is never part of production inference.
         */
        std::vector<float> terminal_hidden_values;
        bool terminal_logits_hash_available = false;
        size_t terminal_logits_bytes = 0;
        uint64_t terminal_logits_hash = 0;
        /** Optional full FP32 terminal logits in stable participant order. */
        std::vector<float> terminal_logits_values;
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

    /**
     * @brief Inspect KV state using an invocation-scoped capture policy.
     *
     * @param cache Cache whose device-owned logical state is observed.
     * @param owner Stable diagnostic owner name.
     * @param device Device that owns @p cache.
     * @param sequence_count Number of sequence slots to inspect.
     * @param stream Explicit producer-ordered stream for GPU exports.
     * @param capture_policy Exact expensive payloads to export.
     * @return Value-owned cache inventory and selected byte hashes.
     */
    PrefixKVCacheProbe inspectKVCacheForPrefixProbe(
        const IKVCache &cache,
        std::string owner,
        DeviceId device,
        int sequence_count,
        void *stream,
        const PrefixProbeCapturePolicy &capture_policy);

    std::vector<PrefixGDNLayerProbe> inspectHybridGDNForPrefixProbe(
        const IKVCache &cache,
        void *stream = nullptr);

    /**
     * @brief Inspect hybrid recurrent state using an explicit capture policy.
     *
     * @param cache Hybrid cache whose kernel-owned banks are observed.
     * @param stream Explicit producer-ordered stream for GPU exports.
     * @param capture_policy Exact expensive payloads to export.
     * @return Per-layer host/local-device/full-device state hashes.
     */
    std::vector<PrefixGDNLayerProbe> inspectHybridGDNForPrefixProbe(
        const IKVCache &cache,
        void *stream,
        const PrefixProbeCapturePolicy &capture_policy);

    uint64_t hashFloatBufferForPrefixProbe(const float *values, size_t count);
    uint64_t hashByteBufferForPrefixProbe(const void *values, size_t bytes);
    bool floatBufferAllZeroForPrefixProbe(const float *values, size_t count);

} // namespace llaminar2
