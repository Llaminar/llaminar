/**
 * @file PrefillGraphCache.cpp
 * @brief Implementation of PrefillGraphCache state machine and lifecycle
 */

#include "PrefillGraphCache.h"
#include "../../../utils/Logger.h"

#include <chrono>
#include <exception>
#include <functional>
#include <limits>

namespace llaminar2
{

    namespace
    {
        /// @brief Return true when a graph contains GDN state without a real-length contract.
        bool containsPaddedBucketUnsafeGDNStage(const ComputeGraph &graph)
        {
            const auto &order = graph.getExecutionOrder();
            for (const auto &name : order)
            {
                const auto *node = graph.getNode(name);
                if (!node || !node->stage)
                    continue;

                const ComputeStageType type = node->stage->type();
                if (type == ComputeStageType::GDN_RECURRENCE ||
                    type == ComputeStageType::SHORT_CONV1D)
                {
                    if (!node->stage->supportsPaddedPrefillRealLengthContract())
                        return true;
                }
            }
            return false;
        }

        /**
         * @brief Terminate after detecting an impossible prefill-capture state.
         *
         * Native graph capture is an ownership transaction, not a performance
         * hint. Seeing a request reset, invalidation, or a second warmup while
         * the transaction body is active means two control-flow owners are
         * attempting to mutate the same stream lifecycle. Continuing would make
         * it impossible to know which launches belong to the graph.
         */
        [[noreturn]] void terminatePrefillGraphLifecycle(
            const char *operation,
            const PrefillGraphCacheKey &key)
        {
            LOG_ERROR(
                "[PrefillGraphCache] Fatal prefill graph lifecycle violation"
                << " operation=" << (operation ? operation : "unknown")
                << " seq_len=" << key.seq_len
                << " device=" << key.device_id.toString()
                << " domain=" << key.domain_id
                << " participant=" << key.participant_id);
            std::terminate();
        }
    }

    // =========================================================================
    // PrefillGraphCacheKey
    // =========================================================================

    bool PrefillGraphCacheKey::operator==(const PrefillGraphCacheKey &other) const
    {
        return seq_len == other.seq_len &&
               device_id == other.device_id &&
               domain_id == other.domain_id &&
               participant_id == other.participant_id &&
               real_token_count == other.real_token_count &&
               first_layer == other.first_layer &&
               layer_count == other.layer_count &&
               placement_epoch == other.placement_epoch &&
               topology_signature == other.topology_signature;
    }

    // =========================================================================
    // PrefillGraphCacheKeyHash
    // =========================================================================

    size_t PrefillGraphCacheKeyHash::operator()(const PrefillGraphCacheKey &k) const
    {
        size_t h = std::hash<int>{}(k.seq_len);
        h ^= std::hash<int>{}(static_cast<int>(k.device_id.type)) << 1;
        h ^= std::hash<int>{}(k.device_id.ordinal) << 2;
        h ^= std::hash<std::string>{}(k.domain_id) << 3;
        h ^= std::hash<int>{}(k.participant_id) << 4;
        h ^= std::hash<int>{}(k.real_token_count) << 5;
        h ^= std::hash<int>{}(k.first_layer) << 6;
        h ^= std::hash<int>{}(k.layer_count) << 7;
        h ^= std::hash<uint64_t>{}(k.placement_epoch) << 8;
        h ^= std::hash<uint64_t>{}(k.topology_signature) << 9;
        return h;
    }

    // =========================================================================
    // PrefillGraphRejectReason toString
    // =========================================================================

    const char *toString(PrefillGraphRejectReason reason)
    {
        switch (reason)
        {
        case PrefillGraphRejectReason::None:
            return "None";
        case PrefillGraphRejectReason::FeatureDisabled:
            return "FeatureDisabled";
        case PrefillGraphRejectReason::PaddedBucketBelowMinimum:
            return "PaddedBucketBelowMinimum";
        case PrefillGraphRejectReason::NotGPUDevice:
            return "NotGPUDevice";
        case PrefillGraphRejectReason::SnapshotsActive:
            return "SnapshotsActive";
        case PrefillGraphRejectReason::ActiveMoERebalancing:
            return "ActiveMoERebalancing";
        case PrefillGraphRejectReason::CollectiveNodesPresent:
            return "CollectiveNodesPresent";
        case PrefillGraphRejectReason::StageNotCapturable:
            return "StageNotCapturable";
        case PrefillGraphRejectReason::GDNWithPaddedBucket:
            return "GDNWithPaddedBucket";
        case PrefillGraphRejectReason::HostPolicyDisabled:
            return "HostPolicyDisabled";
        case PrefillGraphRejectReason::NoGPUContext:
            return "NoGPUContext";
        case PrefillGraphRejectReason::InvalidatedByPlacement:
            return "InvalidatedByPlacement";
        case PrefillGraphRejectReason::SessionReset:
            return "SessionReset";
        case PrefillGraphRejectReason::RequestStateReset:
            return "RequestStateReset";
        }
        return "Unknown";
    }

    // =========================================================================
    // PrefillGraphCache
    // =========================================================================

    PrefillGraphCache::PrefillGraphCache(PrefillGraphConfig config)
        : config_(std::move(config))
    {
    }

    void PrefillGraphCache::touchEntry(PrefillGraphEntry &entry)
    {
        entry.last_access_tick = ++access_counter_;
    }

    void PrefillGraphCache::enforceCapacity(const PrefillGraphCacheKey *exempt_key)
    {
        if (config_.max_cached_entries == 0)
            return;

        while (entries_.size() > config_.max_cached_entries)
        {
            auto victim = entries_.end();
            uint64_t oldest_tick = std::numeric_limits<uint64_t>::max();

            // Evict the least-recently-used non-exempt entry. Ready entries are
            // safe to erase because the graph object owns its executable and the
            // next request for that bucket will explicitly warm/capture again.
            for (auto it = entries_.begin(); it != entries_.end(); ++it)
            {
                if (exempt_key && it->first == *exempt_key)
                    continue;
                if (it->second.last_access_tick < oldest_tick)
                {
                    oldest_tick = it->second.last_access_tick;
                    victim = it;
                }
            }

            if (victim == entries_.end())
                return;

            const auto evicted_key = victim->first;
            entries_.erase(victim);
            ++eviction_count_;
            LOG_INFO("[PrefillGraphCache] Evicted prefill graph bucket seq_len="
                     << evicted_key.seq_len << " device=" << evicted_key.device_id.toString()
                     << " domain=" << evicted_key.domain_id
                     << " participant=" << evicted_key.participant_id
                     << " due to cache cap=" << config_.max_cached_entries);
        }
    }

    PrefillGraphPhase PrefillGraphCache::phase(const PrefillGraphCacheKey &key) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return PrefillGraphPhase::Cold;
        return it->second.phase;
    }

    bool PrefillGraphCache::hasGraph(const PrefillGraphCacheKey &key) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return false;
        return it->second.phase == PrefillGraphPhase::Ready;
    }

    /**
     * @brief Validate whether the cached bucket can warm, capture, or replay.
     *
     * Padded buckets have two different checks. A cold warmup only needs to
     * prove that every stage supports the fixed-bucket contract. Capture needs
     * the stricter readiness check because graph capture records concrete
     * backend handles, scratch pointers, and launch topology.
     */
    PrefillGraphRejectReason PrefillGraphCache::preflight(
        const ComputeGraph &graph,
        const PrefillGraphCacheKey &key,
        const std::unordered_set<std::string> *collective_nodes,
        bool snapshots_active,
        bool moe_rebalancing_active,
        int real_seq_len,
        int bucket_seq_len,
        PrefillGraphPreflightMode mode,
        bool collectives_graph_capturable,
        bool heterogeneous_segmentation_admitted,
        PrefillMoEGraphStability moe_graph_stability,
        std::string *reject_stage_name,
        std::string *reject_stage_type) const
    {
        if (reject_stage_name)
            reject_stage_name->clear();
        if (reject_stage_type)
            reject_stage_type->clear();

        if (!config_.enabled)
            return PrefillGraphRejectReason::FeatureDisabled;

        if (!key.device_id.is_gpu())
            return PrefillGraphRejectReason::NotGPUDevice;

        (void)snapshots_active;

        const bool padded_bucket =
            real_seq_len > 0 && bucket_seq_len > 0 && real_seq_len < bucket_seq_len;
        if (padded_bucket && bucket_seq_len < config_.minimum_padded_bucket_seq_len)
            return PrefillGraphRejectReason::PaddedBucketBelowMinimum;
        const bool support_only_preflight = mode != PrefillGraphPreflightMode::CaptureReady;
        const bool cold_padded_preflight =
            padded_bucket && support_only_preflight;
        if (padded_bucket && !config_.buckets_enabled)
            return PrefillGraphRejectReason::FeatureDisabled;

        if (moe_rebalancing_active && padded_bucket &&
            moe_graph_stability != PrefillMoEGraphStability::Stable)
            return PrefillGraphRejectReason::ActiveMoERebalancing;

        if (collective_nodes && !collective_nodes->empty() &&
            !collectives_graph_capturable &&
            !heterogeneous_segmentation_admitted)
            return PrefillGraphRejectReason::CollectiveNodesPresent;

        if (padded_bucket && containsPaddedBucketUnsafeGDNStage(graph))
        {
            if (config_.trace)
            {
                LOG_INFO("[PrefillGraphCache] Rejecting padded bucket with unsupported GDN/short-conv state contract: real_seq_len="
                         << real_seq_len << " bucket_seq_len=" << bucket_seq_len);
            }
            return PrefillGraphRejectReason::GDNWithPaddedBucket;
        }

        // Cold padded-bucket preflight happens before warmup can allocate lazy
        // graph resources. It validates support only; later Warmup/Ready
        // preflight still requires isGraphCapturable() readiness.
        const auto &order = graph.getExecutionOrder();
        for (const auto &name : order)
        {
            const auto *node = graph.getNode(name);
            if (!node || !node->stage)
                continue;
            const bool stage_ok =
                cold_padded_preflight
                    ? node->stage->supportsPaddedPrefillGraphCapturePreflight()
                : support_only_preflight
                    ? node->stage->supportsLazyPrefillGraphCapturePreflight()
                    : node->stage->isGraphCapturable();
            if (!stage_ok)
            {
                const bool declared_heterogeneous_boundary =
                    heterogeneous_segmentation_admitted &&
                    (node->stage->isCollectiveStage() ||
                     node->stage->isManualGraphBoundary()) &&
                    (!padded_bucket ||
                     node->stage
                         ->supportsPaddedPrefillGraphCapturePreflight());
                if (declared_heterogeneous_boundary)
                    continue;

                if (reject_stage_name)
                    *reject_stage_name = name;
                if (reject_stage_type)
                    *reject_stage_type = computeStageTypeName(node->stage->type());
                if (config_.trace)
                {
                    LOG_INFO("[PrefillGraphCache] Stage '" << name << "' "
                                                           << (cold_padded_preflight
                                                                   ? "does not support cold padded-prefill graph preflight"
                                                                   : "is not graph-capturable"));
                }
                return PrefillGraphRejectReason::StageNotCapturable;
            }
        }

        return PrefillGraphRejectReason::None;
    }

    /**
     * @brief Mark the current request's normal execution as capture-arming warmup.
     *
     * This state is intentionally request-scoped. Request reset may preserve the
     * lazy initialization side effects, but it must not preserve the fact that
     * this particular request was next in line for capture.
     */
    void PrefillGraphCache::markWarmedUp(const PrefillGraphCacheKey &key)
    {
        auto &entry = entries_[key];
        if (entry.phase == PrefillGraphPhase::Capturing)
            terminatePrefillGraphLifecycle("markWarmedUp", key);

        entry.key = key;
        entry.capture.reset();
        entry.phase = PrefillGraphPhase::Warmup;
        lifecycle_stats_[key].warmup_count++;
        touchEntry(entry);
        enforceCapacity(&key);

        if (config_.trace)
        {
            LOG_INFO("[PrefillGraphCache] Warmup complete for seq_len="
                     << key.seq_len << " device=" << key.device_id.toString()
                     << " domain=" << key.domain_id
                     << " participant=" << key.participant_id
                     << " → armed for capture");
        }
    }

    bool PrefillGraphCache::captureAndInstantiate(
        const PrefillGraphCacheKey &key,
        IWorkerGPUContext *gpu_ctx,
        void *stream,
        const std::function<bool()> &record_graph_body,
        GraphCaptureDependencyLedger *dependency_ledger)
    {
        auto it = entries_.find(key);
        if (it == entries_.end() ||
            (it->second.phase != PrefillGraphPhase::Warmup &&
             it->second.phase != PrefillGraphPhase::Initialized))
        {
            LOG_ERROR("[PrefillGraphCache] captureAndInstantiate() called but entry not capture-armed"
                      << " (seq_len=" << key.seq_len << ")");
            return false;
        }

        if (!gpu_ctx)
        {
            LOG_ERROR("[PrefillGraphCache] captureAndInstantiate() called with null GPU context");
            return false;
        }
        if (!record_graph_body)
        {
            LOG_ERROR("[PrefillGraphCache] captureAndInstantiate() requires a graph body");
            return false;
        }

        auto &entry = it->second;
        touchEntry(entry);

        if (!stream)
        {
            LOG_ERROR("[PrefillGraphCache] captureAndInstantiate() requires an explicit capture stream");
            return false;
        }

        /*
         * Keep the in-progress backend object local. The cache entry cannot
         * expose a half-recorded graph to invalidation, replay, or request
         * reset. Only a closed and instantiated executable is published.
         */
        auto pending_capture = gpu_ctx->createGraphCapture(stream);
        if (!pending_capture)
        {
            LOG_ERROR("[PrefillGraphCache] Failed to create graph capture object");
            return false;
        }

        const std::string operation =
            "prefill capture seq_len=" + std::to_string(key.seq_len) +
            " device=" + key.device_id.toString() +
            " domain=" + key.domain_id +
            " participant=" + std::to_string(key.participant_id);
        ScopedBackendGraphCapture capture_transaction(
            *gpu_ctx,
            *pending_capture,
            operation,
            dependency_ledger);
        if (!capture_transaction.begin())
        {
            LOG_ERROR("[PrefillGraphCache] beginCapture() failed on GPU graph object");
            return false;
        }

        entry.phase = PrefillGraphPhase::Capturing;
        if (config_.trace)
            LOG_INFO("[PrefillGraphCache] Capture started for seq_len=" << key.seq_len);

        bool body_succeeded = false;
        try
        {
            body_succeeded = record_graph_body();
        }
        catch (...)
        {
            entry.phase = PrefillGraphPhase::Cold;
            throw;
        }

        /*
         * `finish()` is unconditional. Even a graph-body failure must leave
         * native stream capture before ordinary C++ control flow can inspect
         * the result. A failed backend endCapture is fatal inside the owner.
         */
        capture_transaction.finish();
        if (entry.phase != PrefillGraphPhase::Capturing)
            terminatePrefillGraphLifecycle("capture body changed cache phase", key);

        if (!body_succeeded)
        {
            LOG_ERROR(
                "[PrefillGraphCache] Graph body failed during mandatory prefill capture"
                << " seq_len=" << key.seq_len);
            entry.phase = PrefillGraphPhase::Cold;
            entry.capture.reset();
            entry.node_count = 0;
            entry.replay_count = 0;
            return false;
        }

        if (!pending_capture->instantiate())
        {
            LOG_ERROR("[PrefillGraphCache] instantiate() failed");
            entry.capture.reset();
            entry.phase = PrefillGraphPhase::Cold;
            return false;
        }

        entry.node_count = pending_capture->nodeCount();
        entry.capture = std::move(pending_capture);
        entry.phase = PrefillGraphPhase::Ready;
        entry.replay_count = 0;
        lifecycle_stats_[key].capture_count++;

        auto now = std::chrono::steady_clock::now();
        entry.capture_timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());

        LOG_INFO("[PrefillGraphCache] Captured prefill graph: seq_len=" << key.seq_len
                                                                        << ", nodes=" << entry.node_count
                                                                        << ", device=" << key.device_id.toString());

        return true;
    }

    bool PrefillGraphCache::launch(const PrefillGraphCacheKey &key)
    {
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.phase != PrefillGraphPhase::Ready)
        {
            return false;
        }

        auto &entry = it->second;
        touchEntry(entry);

        if (!entry.capture || !entry.capture->hasExecutable())
        {
            LOG_ERROR("[PrefillGraphCache] launch() called but no executable graph");
            return false;
        }

        if (!entry.capture->launch())
        {
            LOG_ERROR("[PrefillGraphCache] launch() failed");
            return false;
        }

        entry.replay_count++;
        return true;
    }

    void PrefillGraphCache::invalidateAll(PrefillGraphRejectReason reason)
    {
        size_t count = entries_.size();
        for (auto &[k, entry] : entries_)
        {
            if (entry.phase == PrefillGraphPhase::Capturing)
                terminatePrefillGraphLifecycle("invalidateAll", k);

            entry.phase = PrefillGraphPhase::Cold;
            entry.capture.reset();
            entry.node_count = 0;
            entry.replay_count = 0;
        }
        last_invalidation_reason_ = reason;

        if (count > 0)
        {
            LOG_INFO("[PrefillGraphCache] Invalidated " << count << " entries (reason: " << toString(reason) << ")");
        }
    }

    /**
     * @brief Apply the typed request-boundary policy to every prefill executable.
     *
     * Preserving reset retains only complete Ready executables. Their graph nodes
     * reference persistent device buffers, while request-local contents and
     * scalar metadata are republished before the next launch. Hard reset demotes
     * those same entries to Initialized so capture readiness must be established
     * again. Warmup is never preserved as replayable state because it represents
     * an unfinished request-local lifecycle transition.
     */
    PrefillGraphRequestResetSummary PrefillGraphCache::prepareEntriesForRequestReset(
        bool preserve_ready_executables)
    {
        PrefillGraphRequestResetSummary summary;
        for (auto &[key, entry] : entries_)
        {
            if (entry.phase == PrefillGraphPhase::Capturing)
            {
                terminatePrefillGraphLifecycle(
                    "prepareEntriesForRequestReset",
                    key);
            }

            const bool ready_for_replay =
                entry.phase == PrefillGraphPhase::Ready &&
                entry.capture &&
                entry.capture->hasExecutable();
            if (ready_for_replay)
            {
                if (preserve_ready_executables)
                {
                    ++summary.ready_preserved;
                    continue;
                }

                entry.phase = PrefillGraphPhase::Initialized;
                entry.capture.reset();
                entry.node_count = 0;
                entry.replay_count = 0;
                lifecycle_stats_[key].initialized_count++;
                ++summary.ready_demoted;
                continue;
            }

            if (entry.phase == PrefillGraphPhase::Warmup ||
                entry.phase == PrefillGraphPhase::Initialized)
            {
                // Preserve durable lazy setup, but discard any request-specific
                // graph object and replay counters from the previous prompt.
                entry.phase = PrefillGraphPhase::Initialized;
                entry.capture.reset();
                entry.node_count = 0;
                entry.replay_count = 0;
                lifecycle_stats_[key].initialized_count++;
                ++summary.initialized;
                continue;
            }

            entry.phase = PrefillGraphPhase::Cold;
            entry.capture.reset();
            entry.node_count = 0;
            entry.replay_count = 0;
            ++summary.dropped;
        }

        if (summary.ready_demoted > 0 || summary.initialized > 0 || summary.dropped > 0)
        {
            last_invalidation_reason_ = PrefillGraphRejectReason::RequestStateReset;
            LOG_INFO("[PrefillGraphCache] Request reset preserved "
                     << summary.ready_preserved
                     << " ready prefill graph executable(s), demoted "
                     << summary.ready_demoted
                     << " ready executable(s), kept "
                     << summary.initialized
                     << " entries as lazy-initialized only, and dropped "
                     << summary.dropped
                     << " stale entries");
        }
        else if (summary.ready_preserved > 0 && config_.trace)
        {
            LOG_INFO("[PrefillGraphCache] Request reset preserved "
                     << summary.ready_preserved
                     << " complete prefill graph executable(s)");
        }
        return summary;
    }

    void PrefillGraphCache::invalidate(const PrefillGraphCacheKey &key)
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return;
        if (it->second.phase == PrefillGraphPhase::Capturing)
            terminatePrefillGraphLifecycle("invalidate", key);

        it->second.phase = PrefillGraphPhase::Cold;
        it->second.capture.reset();
        it->second.node_count = 0;
        it->second.replay_count = 0;

        if (config_.trace)
        {
            LOG_INFO("[PrefillGraphCache] Invalidated entry for seq_len=" << key.seq_len);
        }
    }

    size_t PrefillGraphCache::size() const
    {
        return entries_.size();
    }

    size_t PrefillGraphCache::nodeCount(const PrefillGraphCacheKey &key) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.phase != PrefillGraphPhase::Ready)
            return 0;
        return it->second.node_count;
    }

    int PrefillGraphCache::replayCount(const PrefillGraphCacheKey &key) const
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return 0;
        return it->second.replay_count;
    }

    uint64_t PrefillGraphCache::warmupCount(const PrefillGraphCacheKey &key) const
    {
        auto it = lifecycle_stats_.find(key);
        if (it == lifecycle_stats_.end())
            return 0;
        return it->second.warmup_count;
    }

    /**
     * @brief Return how many request resets kept lazy initialization for a key.
     */
    uint64_t PrefillGraphCache::initializedCount(const PrefillGraphCacheKey &key) const
    {
        auto it = lifecycle_stats_.find(key);
        if (it == lifecycle_stats_.end())
            return 0;
        return it->second.initialized_count;
    }

    uint64_t PrefillGraphCache::captureCount(const PrefillGraphCacheKey &key) const
    {
        auto it = lifecycle_stats_.find(key);
        if (it == lifecycle_stats_.end())
            return 0;
        return it->second.capture_count;
    }

} // namespace llaminar2
