/**
 * @file PrefillGraphCache.h
 * @brief Persistent graph capture/replay cache for monolithic single-device prefill
 *
 * Provides the state machine, preflight checks, and GPU graph lifecycle
 * (warmup → capture → replay) for prefill graph caching.
 *
 * Tier 1 Phase 4: Self-contained cache implementation.
 * Phase 5 will integrate this into ForwardExecutionEngine.
 */

#pragma once

#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../backends/DeviceId.h"
#include "../graph/ComputeGraph.h"
#include "../../compute_stages/IComputeStage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaminar2
{

    /// Key identifying a unique prefill graph cache entry.
    /// Future-proofed for Tier 2 (domain/epoch fields present but defaulted).
    struct PrefillGraphCacheKey
    {
        int seq_len = 0;                      ///< Bucket/actual seq_len
        DeviceId device_id = DeviceId::cpu(); ///< Target device
        std::string domain_id = "single";     ///< EP rebalance domain (Tier 1: always "single")
        int participant_id = 0;               ///< Domain-local participant id for segmented ExpertParallel caches
        int real_token_count = 0;             ///< Real tokens represented inside a padded bucket (0 means exact/unspecified)
        int first_layer = 0;                  ///< First layer covered by this segment cache key
        int layer_count = 0;                  ///< Number of layers covered by this segment (0 means whole graph/unspecified)
        uint64_t placement_epoch = 0;         ///< Expert placement epoch (Tier 1: always 0)
        uint64_t topology_signature = 0;      ///< Hash of stage topology (Tier 1: 0)

        bool operator==(const PrefillGraphCacheKey &other) const;
    };

    struct PrefillGraphCacheKeyHash
    {
        size_t operator()(const PrefillGraphCacheKey &k) const;
    };

    /// Phase of the prefill graph cache state machine.
    enum class PrefillGraphPhase
    {
        Disabled,    ///< Feature disabled or unsupported config
        Cold,        ///< No reusable graph or lazy-initialization state for this key
        Initialized, ///< Lazy stage/kernel resources are initialized; current request is not armed
        Warmup,      ///< Current-request warmup complete, armed for capture on next run
        Capturing,   ///< Capture run in progress (stream recording)
        Ready        ///< Graph instantiated, ready for replay
    };

    /// Controls how preflight treats padded-bucket stages that warm lazily.
    enum class PrefillGraphPreflightMode
    {
        Default,           ///< Cold padded buckets may use support-only checks; other states require readiness
        ColdPaddedSupport, ///< Validate padded-bucket support without requiring warmed capture readiness
        CaptureReady       ///< Require strict capture readiness for every stage
    };

    /// Reasons a prefill graph capture can be rejected.
    enum class PrefillGraphRejectReason
    {
        None,
        FeatureDisabled,        ///< LLAMINAR_GPU_GRAPHS=0
        SeqLenBelowMinimum,     ///< seq_len < LLAMINAR_PREFILL_GRAPH_MIN_SEQ
        NotGPUDevice,           ///< CPU device
        SnapshotsActive,        ///< ENABLE_PIPELINE_SNAPSHOTS build
        ActiveMoERebalancing,   ///< Rebalance mode is DYNAMIC or OBSERVE
        CollectiveNodesPresent, ///< Graph has TP/PP collective stages
        StageNotCapturable,     ///< One or more stages return isGraphCapturable()=false
        GDNWithPaddedBucket,    ///< GDN/short-conv state would advance through padding rows
        NoGPUContext,           ///< GPU context unavailable
        InvalidatedByPlacement, ///< Expert placement mutation since last capture
        SessionReset,           ///< Legacy name for request/session reset invalidation
        RequestStateReset       ///< Request/session reset invalidated captured live-state mutation
    };

    /// Convert PrefillGraphRejectReason to string for logging.
    const char *toString(PrefillGraphRejectReason reason);

    /// Configuration for prefill graph cache behavior.
    struct PrefillGraphConfig
    {
        bool enabled = true;                 ///< LLAMINAR_GPU_GRAPHS master flag
        int min_seq_len = 256;               ///< LLAMINAR_PREFILL_GRAPH_MIN_SEQ
        bool trace = false;                  ///< LLAMINAR_PREFILL_GRAPH_TRACE
        bool buckets_enabled = true;         ///< Bucketed capture is on by default; LLAMINAR_PREFILL_GRAPH_BUCKETS=0 opts out.
        std::vector<int> bucket_sizes;       ///< LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES
        size_t max_cached_entries = 10;      ///< LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS
    };

    /// Per-entry state in the prefill graph cache.
    struct PrefillGraphEntry
    {
        PrefillGraphCacheKey key;
        PrefillGraphPhase phase = PrefillGraphPhase::Cold;
        std::unique_ptr<IGPUGraphCapture> capture; ///< GPU graph object
        size_t node_count = 0;                     ///< Nodes in captured graph
        uint64_t capture_timestamp_ns = 0;         ///< When capture completed
        int replay_count = 0;                      ///< Number of successful replays
        uint64_t last_access_tick = 0;             ///< Monotonic LRU timestamp
    };

    /// Lifetime counters for a bucket key, retained even if the graph entry is evicted.
    struct PrefillGraphLifecycleStats
    {
        uint64_t warmup_count = 0;      ///< Number of times this key entered Warmup.
        uint64_t initialized_count = 0; ///< Number of times request reset preserved lazy init only.
        uint64_t capture_count = 0;     ///< Number of successful captures for this key.
    };

    /// Summary of request-boundary prefill graph-cache cleanup.
    struct PrefillGraphRequestResetSummary
    {
        size_t ready_preserved = 0; ///< Ready executable entries kept for replay.
        size_t initialized = 0;     ///< Entries kept as lazy-initialized, without request capture arming.
        size_t dropped = 0;         ///< Capturing/invalid entries dropped to Cold.
    };

    class PrefillGraphCache
    {
    public:
        explicit PrefillGraphCache(PrefillGraphConfig config);

        /// Get the current phase for a cache key.
        PrefillGraphPhase phase(const PrefillGraphCacheKey &key) const;

        /// Check if a ready graph exists for the given key.
        bool hasGraph(const PrefillGraphCacheKey &key) const;

        /// Run preflight checks to determine if a graph is eligible for capture.
        /// Returns None on success, or a specific reject reason.
        PrefillGraphRejectReason preflight(
            const ComputeGraph &graph,
            const PrefillGraphCacheKey &key,
            const std::unordered_set<std::string> *collective_nodes,
            bool snapshots_active,
            bool moe_rebalancing_active = false,
            int real_seq_len = 0,
            int bucket_seq_len = 0,
            PrefillGraphPreflightMode mode = PrefillGraphPreflightMode::Default) const;

        /// Mark warmup complete for a key. Transitions Cold → Warmup (arms capture).
        void markWarmedUp(const PrefillGraphCacheKey &key);

        /**
         * @brief Begin graph capture on the given explicit stream.
         *
         * Warmup entries are armed by a fresh request-local warmup. Initialized
         * entries are accepted only after the caller has run strict capture-ready
         * preflight and prepared current request metadata. The cache keeps this
         * method narrow: it validates lifecycle and stream ownership, while the
         * executor decides whether capture is semantically safe for this request.
         */
        bool beginCapture(const PrefillGraphCacheKey &key, IWorkerGPUContext *gpu_ctx, void *stream);

        /// End graph capture, instantiate the executable graph.
        /// Transitions Capturing → Ready. Returns false on failure.
        bool endCaptureAndInstantiate(const PrefillGraphCacheKey &key);

        /// Launch (replay) the cached graph.
        /// Returns false if not Ready or launch fails.
        bool launch(const PrefillGraphCacheKey &key);

        /// Invalidate all cached entries (e.g., after expert placement mutation).
        void invalidateAll(PrefillGraphRejectReason reason = PrefillGraphRejectReason::InvalidatedByPlacement);

        /**
         * @brief Preserve replay-ready entries across a request-boundary reset.
         *
         * `clear_cache()` resets live KV/GDN/short-conv contents, but replay-ready
         * bucketed prefill graph entries are designed to read refreshed graph-facing
         * buffers and then replay the same deterministic mutation sequence.
         *
         * Warmup entries are intentionally not preserved as Warmup. A warmed
         * entry has observed request-local runtime metadata, but it has also
         * performed useful lazy stage/kernel initialization. Request reset converts
         * Warmup to Initialized: the next same-key request may capture only after a
         * strict capture-ready preflight and fresh metadata preparation, or else it
         * executes a fresh warmup. This preserves lazy resources without carrying
         * a request-armed state across `clear_cache()`.
         *
         * @return Summary of preserved, initialized, and dropped entries.
         */
        PrefillGraphRequestResetSummary prepareEntriesForRequestReset();

        /// Compatibility wrapper returning only dropped entries.
        size_t preserveReadyEntriesAcrossRequestReset();

        /// Invalidate a specific entry.
        void invalidate(const PrefillGraphCacheKey &key);

        /// Get the config.
        const PrefillGraphConfig &config() const { return config_; }

        /// Get number of cached entries.
        size_t size() const;

        /// Get node count for a ready entry (0 if not ready).
        size_t nodeCount(const PrefillGraphCacheKey &key) const;

        /// Get replay count for an entry.
        int replayCount(const PrefillGraphCacheKey &key) const;

        /// Get number of entries evicted due to the configured cache cap.
        uint64_t evictionCount() const { return eviction_count_; }

        /// Get lifetime warmup count for a key, including entries later evicted.
        uint64_t warmupCount(const PrefillGraphCacheKey &key) const;

        /// Get lifetime lazy-initialized count for a key, including entries later evicted.
        uint64_t initializedCount(const PrefillGraphCacheKey &key) const;

        /// Get lifetime successful capture count for a key, including recaptures.
        uint64_t captureCount(const PrefillGraphCacheKey &key) const;

        /// Get the most recent invalidation reason for diagnostics and regression tests.
        PrefillGraphRejectReason lastInvalidationReason() const { return last_invalidation_reason_; }

    private:
        void touchEntry(PrefillGraphEntry &entry);
        void enforceCapacity(const PrefillGraphCacheKey *exempt_key = nullptr);

        PrefillGraphConfig config_;
        std::unordered_map<PrefillGraphCacheKey, PrefillGraphEntry, PrefillGraphCacheKeyHash> entries_;
        std::unordered_map<PrefillGraphCacheKey, PrefillGraphLifecycleStats, PrefillGraphCacheKeyHash> lifecycle_stats_;
        PrefillGraphRejectReason last_invalidation_reason_ = PrefillGraphRejectReason::None;
        uint64_t access_counter_ = 0;
        uint64_t eviction_count_ = 0;
    };

} // namespace llaminar2
