/**
 * @file PrefillGraphCache.cpp
 * @brief Implementation of PrefillGraphCache state machine and lifecycle
 */

#include "PrefillGraphCache.h"
#include "../../../utils/Logger.h"

#include <chrono>
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
        case PrefillGraphRejectReason::SeqLenBelowMinimum:
            return "SeqLenBelowMinimum";
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
        PrefillGraphPreflightMode mode) const
    {
        if (!config_.enabled)
            return PrefillGraphRejectReason::FeatureDisabled;

        if (key.seq_len < config_.min_seq_len)
            return PrefillGraphRejectReason::SeqLenBelowMinimum;

        if (!key.device_id.is_gpu())
            return PrefillGraphRejectReason::NotGPUDevice;

        if (snapshots_active)
            return PrefillGraphRejectReason::SnapshotsActive;

        if (moe_rebalancing_active)
            return PrefillGraphRejectReason::ActiveMoERebalancing;

        if (collective_nodes && !collective_nodes->empty())
            return PrefillGraphRejectReason::CollectiveNodesPresent;

        const bool padded_bucket =
            real_seq_len > 0 && bucket_seq_len > 0 && real_seq_len < bucket_seq_len;
        const bool cold_padded_preflight =
            padded_bucket &&
            (mode == PrefillGraphPreflightMode::ColdPaddedSupport ||
             (mode == PrefillGraphPreflightMode::Default && phase(key) == PrefillGraphPhase::Cold));
        if (padded_bucket && !config_.buckets_enabled)
            return PrefillGraphRejectReason::FeatureDisabled;

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
                (mode != PrefillGraphPreflightMode::CaptureReady && cold_padded_preflight)
                    ? node->stage->supportsPaddedPrefillGraphCapturePreflight()
                    : node->stage->isGraphCapturable();
            if (!stage_ok)
            {
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
        entry.key = key;
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

    /**
     * @brief Start recording a monolithic prefill GPU graph on an explicit stream.
     */
    bool PrefillGraphCache::beginCapture(const PrefillGraphCacheKey &key, IWorkerGPUContext *gpu_ctx, void *stream)
    {
        auto it = entries_.find(key);
        if (it == entries_.end() ||
            (it->second.phase != PrefillGraphPhase::Warmup &&
             it->second.phase != PrefillGraphPhase::Initialized))
        {
            LOG_ERROR("[PrefillGraphCache] beginCapture() called but entry not capture-armed"
                      << " (seq_len=" << key.seq_len << ")");
            return false;
        }

        if (!gpu_ctx)
        {
            LOG_ERROR("[PrefillGraphCache] beginCapture() called with null GPU context");
            return false;
        }

        auto &entry = it->second;
        touchEntry(entry);

        if (!stream)
        {
            LOG_ERROR("[PrefillGraphCache] beginCapture() requires an explicit capture stream");
            return false;
        }

        // Create graph capture object on the caller-managed prefill stream.
        entry.capture = gpu_ctx->createGraphCapture(stream);

        if (!entry.capture)
        {
            LOG_ERROR("[PrefillGraphCache] Failed to create graph capture object");
            return false;
        }

        // Begin stream capture
        if (!entry.capture->beginCapture())
        {
            LOG_ERROR("[PrefillGraphCache] beginCapture() failed on GPU graph object");
            entry.capture.reset();
            return false;
        }

        entry.phase = PrefillGraphPhase::Capturing;

        if (config_.trace)
        {
            LOG_INFO("[PrefillGraphCache] Capture started for seq_len=" << key.seq_len);
        }

        return true;
    }

    /**
     * @brief Finish recording and instantiate the captured prefill graph.
     */
    bool PrefillGraphCache::endCaptureAndInstantiate(const PrefillGraphCacheKey &key)
    {
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.phase != PrefillGraphPhase::Capturing)
        {
            LOG_ERROR("[PrefillGraphCache] endCaptureAndInstantiate() called but entry not in Capturing phase"
                      << " (seq_len=" << key.seq_len << ")");
            return false;
        }

        auto &entry = it->second;
        touchEntry(entry);

        // End capture
        if (!entry.capture->endCapture())
        {
            LOG_ERROR("[PrefillGraphCache] endCapture() failed");
            entry.capture.reset();
            entry.phase = PrefillGraphPhase::Cold;
            return false;
        }

        // Instantiate the graph executable
        if (!entry.capture->instantiate())
        {
            LOG_ERROR("[PrefillGraphCache] instantiate() failed");
            entry.capture.reset();
            entry.phase = PrefillGraphPhase::Cold;
            return false;
        }

        entry.node_count = entry.capture->nodeCount();
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
     * @brief Split reusable lazy initialization from request-local warmup state.
     *
     * Ready entries keep their executable graphs. Warmup/Initialized entries keep
     * only the fact that first-use stage/kernel setup has happened; they are not
     * allowed to replay, and they cannot capture until the executor prepares
     * fresh request metadata and reruns strict capture-readiness preflight.
     */
    PrefillGraphRequestResetSummary PrefillGraphCache::prepareEntriesForRequestReset()
    {
        PrefillGraphRequestResetSummary summary;
        for (auto &[key, entry] : entries_)
        {
            const bool ready_for_replay =
                entry.phase == PrefillGraphPhase::Ready &&
                entry.capture &&
                entry.capture->hasExecutable();
            if (ready_for_replay)
            {
                ++summary.ready_preserved;
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

        if (summary.initialized > 0 || summary.dropped > 0)
        {
            last_invalidation_reason_ = PrefillGraphRejectReason::RequestStateReset;
            LOG_INFO("[PrefillGraphCache] Request reset preserved "
                     << summary.ready_preserved
                     << " ready prefill graph executables, kept "
                     << summary.initialized
                     << " entries as lazy-initialized only, and dropped "
                     << summary.dropped
                     << " stale entries");
        }
        return summary;
    }

    /**
     * @brief Backward-compatible reset helper for callers that only need drops.
     */
    size_t PrefillGraphCache::preserveReadyEntriesAcrossRequestReset()
    {
        return prepareEntriesForRequestReset().dropped;
    }

    void PrefillGraphCache::invalidate(const PrefillGraphCacheKey &key)
    {
        auto it = entries_.find(key);
        if (it == entries_.end())
            return;

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
