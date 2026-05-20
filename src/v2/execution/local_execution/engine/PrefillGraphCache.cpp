/**
 * @file PrefillGraphCache.cpp
 * @brief Implementation of PrefillGraphCache state machine and lifecycle
 */

#include "PrefillGraphCache.h"
#include "../../../utils/Logger.h"

#include <chrono>
#include <functional>

namespace llaminar2
{

    // =========================================================================
    // PrefillGraphCacheKey
    // =========================================================================

    bool PrefillGraphCacheKey::operator==(const PrefillGraphCacheKey &other) const
    {
        return seq_len == other.seq_len &&
               device_id == other.device_id &&
               domain_id == other.domain_id &&
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
        h ^= std::hash<uint64_t>{}(k.placement_epoch) << 4;
        h ^= std::hash<uint64_t>{}(k.topology_signature) << 5;
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
        case PrefillGraphRejectReason::NoGPUContext:
            return "NoGPUContext";
        case PrefillGraphRejectReason::InvalidatedByPlacement:
            return "InvalidatedByPlacement";
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

    PrefillGraphRejectReason PrefillGraphCache::preflight(
        const ComputeGraph &graph,
        const PrefillGraphCacheKey &key,
        const std::unordered_set<std::string> *collective_nodes,
        bool snapshots_active,
        bool moe_rebalancing_active) const
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

        // Check all stages are capturable
        const auto &order = graph.getExecutionOrder();
        for (const auto &name : order)
        {
            const auto *node = graph.getNode(name);
            if (!node || !node->stage)
                continue;
            if (!node->stage->isGraphCapturable())
            {
                if (config_.trace)
                {
                    LOG_INFO("[PrefillGraphCache] Stage '" << name << "' is not graph-capturable");
                }
                return PrefillGraphRejectReason::StageNotCapturable;
            }
        }

        return PrefillGraphRejectReason::None;
    }

    void PrefillGraphCache::markWarmedUp(const PrefillGraphCacheKey &key)
    {
        auto &entry = entries_[key];
        entry.key = key;
        entry.phase = PrefillGraphPhase::Warmup;

        if (config_.trace)
        {
            LOG_INFO("[PrefillGraphCache] Warmup complete for seq_len="
                     << key.seq_len << " device=" << key.device_id.toString()
                     << " → armed for capture");
        }
    }

    bool PrefillGraphCache::beginCapture(const PrefillGraphCacheKey &key, IWorkerGPUContext *gpu_ctx, void *stream)
    {
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.phase != PrefillGraphPhase::Warmup)
        {
            LOG_ERROR("[PrefillGraphCache] beginCapture() called but entry not in Warmup phase"
                      << " (seq_len=" << key.seq_len << ")");
            return false;
        }

        if (!gpu_ctx)
        {
            LOG_ERROR("[PrefillGraphCache] beginCapture() called with null GPU context");
            return false;
        }

        auto &entry = it->second;

        // Create graph capture object
        if (stream)
        {
            entry.capture = gpu_ctx->createGraphCapture(stream);
        }
        else
        {
            entry.capture = gpu_ctx->createGraphCapture();
        }

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

} // namespace llaminar2
