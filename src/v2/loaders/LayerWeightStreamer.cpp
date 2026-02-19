/**
 * @file LayerWeightStreamer.cpp
 * @brief Weight streaming implementation for Option B (STREAMING mode)
 * @author David Sanftenberg
 * @date January 2026
 */

#include "LayerWeightStreamer.h"
#include "../utils/Logger.h"
#include "../tensors/Tensors.h"

#include <algorithm>
#include <sstream>

namespace llaminar2
{

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    LayerWeightStreamer::LayerWeightStreamer(
        std::shared_ptr<WeightManager> weight_manager,
        int num_layers,
        StreamingConfig config)
        : weight_manager_(std::move(weight_manager)), num_layers_(num_layers), config_(std::move(config))
    {
        if (!weight_manager_)
        {
            LOG_ERROR("[LayerWeightStreamer] WeightManager is null");
            throw std::invalid_argument("WeightManager cannot be null");
        }

        if (num_layers_ <= 0)
        {
            LOG_ERROR("[LayerWeightStreamer] Invalid num_layers: " << num_layers_);
            throw std::invalid_argument("num_layers must be positive");
        }

        // Log configuration
        LOG_DEBUG("[LayerWeightStreamer] Initialized with " << num_layers_ << " layers");
        LOG_DEBUG("[LayerWeightStreamer] GPU memory budget: "
                  << (config_.gpu_memory_budget / (1024.0 * 1024.0)) << " MB");
        LOG_DEBUG("[LayerWeightStreamer] Cache layer count: " << config_.cache_layer_count);
        LOG_DEBUG("[LayerWeightStreamer] Eviction policy: "
                  << (config_.eviction_policy == StreamingEvictionPolicy::LRU    ? "LRU"
                      : config_.eviction_policy == StreamingEvictionPolicy::FIFO ? "FIFO"
                                                                                 : "NONE"));
        LOG_DEBUG("[LayerWeightStreamer] Prefetch enabled: " << (config_.enable_prefetch ? "yes" : "no"));
    }

    LayerWeightStreamer::~LayerWeightStreamer()
    {
        // Wait for any pending transfers
        synchronize();

        // Clear cache (releases GPU memory)
        clearCache();

        if (config_.log_transfer_stats)
        {
            auto s = stats();
            LOG_INFO("[LayerWeightStreamer] Final stats:"
                     << " transfers=" << s.layers_transferred
                     << " evictions=" << s.layers_evicted
                     << " hits=" << s.cache_hits
                     << " misses=" << s.cache_misses
                     << " hit_rate=" << (s.hitRate() * 100.0) << "%"
                     << " bandwidth=" << s.averageBandwidthGBps() << " GB/s");
        }
    }

    // =========================================================================
    // Layer Management
    // =========================================================================

    bool LayerWeightStreamer::ensureLayerOnDevice(int layer_idx, DeviceId device)
    {
        // Validate layer index
        if (layer_idx < 0 || layer_idx >= num_layers_)
        {
            LOG_ERROR("[LayerWeightStreamer] Invalid layer index: " << layer_idx
                                                                    << " (num_layers=" << num_layers_ << ")");
            return false;
        }

        // CPU device always succeeds (weights are on host)
        if (device.is_cpu())
        {
            return true;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check if layer is already cached on this device
        auto it = layer_cache_.find(layer_idx);
        if (it != layer_cache_.end() && it->second.device.type == device.type &&
            it->second.device.ordinal == device.ordinal)
        {
            // Cache hit
            touchLayer(layer_idx);
            it->second.last_access = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                stats_.cache_hits++;
                if (it->second.from_prefetch)
                {
                    stats_.prefetch_hits++;
                    it->second.from_prefetch = false; // Only count once
                }
            }

            LOG_TRACE("[LayerWeightStreamer] Cache hit for layer " << layer_idx);
            return true;
        }

        // Cache miss - need to load
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.cache_misses++;
        }

        LOG_TRACE("[LayerWeightStreamer] Cache miss for layer " << layer_idx << ", loading...");

        // If layer is cached on wrong device, evict it first
        if (it != layer_cache_.end())
        {
            size_t freed = it->second.memory_bytes;
            removeLRUEntry(layer_idx);
            layer_cache_.erase(it);
            current_memory_usage_ -= freed;

            {
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                stats_.layers_evicted++;
                stats_.bytes_evicted += freed;
            }
        }

        // Load the layer
        return loadLayerToDevice(layer_idx, device);
    }

    void LayerWeightStreamer::prefetchLayer(int layer_idx, DeviceId device)
    {
        // Validate layer index
        if (layer_idx < 0 || layer_idx >= num_layers_)
        {
            return;
        }

        // CPU device - nothing to prefetch
        if (device.is_cpu())
        {
            return;
        }

        // Prefetch disabled
        if (!config_.enable_prefetch)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Already cached?
        auto it = layer_cache_.find(layer_idx);
        if (it != layer_cache_.end() && it->second.device.type == device.type &&
            it->second.device.ordinal == device.ordinal)
        {
            return; // Already there
        }

        // Check if prefetch is already in progress
        if (pending_prefetches_.find(layer_idx) != pending_prefetches_.end())
        {
            return;
        }

        LOG_TRACE("[LayerWeightStreamer] Prefetching layer " << layer_idx);

        // For now, synchronous prefetch (async deferred)
        // TODO: Use std::async or CUDA streams for true async prefetch
        if (loadLayerToDevice(layer_idx, device))
        {
            // Mark as prefetched for stats tracking
            auto cache_it = layer_cache_.find(layer_idx);
            if (cache_it != layer_cache_.end())
            {
                cache_it->second.from_prefetch = true;
            }
        }
    }

    void LayerWeightStreamer::releaseLayer(int layer_idx)
    {
        if (layer_idx < 0 || layer_idx >= num_layers_)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check if layer is cached
        auto it = layer_cache_.find(layer_idx);
        if (it == layer_cache_.end())
        {
            return; // Not cached
        }

        // Move to back of LRU (will be evicted first)
        auto lru_it = lru_map_.find(layer_idx);
        if (lru_it != lru_map_.end())
        {
            lru_order_.erase(lru_it->second);
            lru_order_.push_back(layer_idx);
            lru_it->second = std::prev(lru_order_.end());
        }

        LOG_TRACE("[LayerWeightStreamer] Released layer " << layer_idx << " (marked for early eviction)");
    }

    // =========================================================================
    // Phase Management
    // =========================================================================

    void LayerWeightStreamer::onPhaseTransition(InferencePhase old_phase, InferencePhase new_phase)
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        current_phase_ = new_phase;

        if (old_phase == InferencePhase::PREFILL && new_phase == InferencePhase::DECODE)
        {
            // Transitioning to decode - might want to preload some layers
            if (config_.prefill_preload_enabled && config_.enable_prefetch)
            {
                LOG_DEBUG("[LayerWeightStreamer] PREFILL -> DECODE transition");
                // Could preload first few layers for decode here
            }
        }
        else if (old_phase == InferencePhase::DECODE && new_phase == InferencePhase::PREFILL)
        {
            // New prompt - might clear cache for fresh start
            LOG_DEBUG("[LayerWeightStreamer] DECODE -> PREFILL transition");
            // For now, keep cache warm (most layers will be reused)
        }
    }

    // =========================================================================
    // Memory Management
    // =========================================================================

    size_t LayerWeightStreamer::currentDeviceMemoryUsage() const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return current_memory_usage_;
    }

    size_t LayerWeightStreamer::memoryBudget() const
    {
        return config_.gpu_memory_budget;
    }

    bool LayerWeightStreamer::evictLayer(int layer_idx)
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = layer_cache_.find(layer_idx);
        if (it == layer_cache_.end())
        {
            return false; // Not in cache
        }

        size_t freed = it->second.memory_bytes;

        // Remove from LRU tracking
        removeLRUEntry(layer_idx);

        // Remove from cache (shared_ptr releases GPU memory when refcount -> 0)
        layer_cache_.erase(it);
        current_memory_usage_ -= freed;

        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.layers_evicted++;
            stats_.bytes_evicted += freed;

            // If this was a prefetched layer that was never used, count it
            // (We can't tell here since we erased it, but we set from_prefetch = false on use)
        }

        LOG_TRACE("[LayerWeightStreamer] Evicted layer " << layer_idx
                                                         << " (freed " << freed << " bytes)");

        return true;
    }

    void LayerWeightStreamer::clearCache()
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        size_t layers_cleared = layer_cache_.size();
        size_t bytes_cleared = current_memory_usage_;

        layer_cache_.clear();
        lru_order_.clear();
        lru_map_.clear();
        current_memory_usage_ = 0;

        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.current_cache_size = 0;
            stats_.current_layers_cached = 0;
        }

        LOG_DEBUG("[LayerWeightStreamer] Cache cleared: " << layers_cleared << " layers, "
                                                          << (bytes_cleared / (1024.0 * 1024.0)) << " MB freed");
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    void LayerWeightStreamer::synchronize()
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Wait for all pending prefetches
        for (auto &[layer_idx, future] : pending_prefetches_)
        {
            if (future.valid())
            {
                future.wait();
            }
        }
        pending_prefetches_.clear();

        // TODO: When using CUDA streams, synchronize streams here
    }

    // =========================================================================
    // Diagnostics
    // =========================================================================

    bool LayerWeightStreamer::isLayerCached(int layer_idx, DeviceId device) const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = layer_cache_.find(layer_idx);
        if (it == layer_cache_.end())
        {
            return false;
        }

        return it->second.device.type == device.type &&
               it->second.device.ordinal == device.ordinal;
    }

    bool LayerWeightStreamer::isPrefetchInProgress(int layer_idx) const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return pending_prefetches_.find(layer_idx) != pending_prefetches_.end();
    }

    StreamingStats LayerWeightStreamer::stats() const
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

    void LayerWeightStreamer::resetStats()
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_ = StreamingStats{};

        // Update current state (not reset)
        stats_.current_cache_size = current_memory_usage_;
        stats_.current_layers_cached = layer_cache_.size();
    }

    // =========================================================================
    // Internal Methods
    // =========================================================================

    std::vector<std::string> LayerWeightStreamer::getLayerWeightNames(int layer_idx) const
    {
        // Qwen2-style weight naming convention
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        return {
            prefix + "attn_q.weight",
            prefix + "attn_k.weight",
            prefix + "attn_v.weight",
            prefix + "attn_output.weight",
            prefix + "attn_norm.weight",
            prefix + "ffn_gate.weight",
            prefix + "ffn_up.weight",
            prefix + "ffn_down.weight",
            prefix + "ffn_norm.weight"};
    }

    size_t LayerWeightStreamer::estimateLayerMemory(int layer_idx) const
    {
        size_t total = 0;

        auto weight_names = getLayerWeightNames(layer_idx);
        for (const auto &name : weight_names)
        {
            // Get the CPU weight to check its size
            auto weight = weight_manager_->getWeightForDevice(name, DeviceId::cpu(), layer_idx);
            if (weight)
            {
                total += weight->size_bytes();
            }
        }

        return total;
    }

    bool LayerWeightStreamer::loadLayerToDevice(int layer_idx, DeviceId device)
    {
        auto start_time = std::chrono::steady_clock::now();

        // Estimate memory needed
        size_t layer_size = estimateLayerMemory(layer_idx);

        // Check if we have enough budget
        if (config_.gpu_memory_budget > 0 &&
            current_memory_usage_ + layer_size > config_.gpu_memory_budget)
        {
            // Need to evict
            if (!evictToFreeMemory(layer_size))
            {
                LOG_WARN("[LayerWeightStreamer] Cannot free enough memory for layer "
                         << layer_idx << " (need " << layer_size << " bytes)");

                // If eviction policy is NONE, fail
                if (config_.eviction_policy == StreamingEvictionPolicy::NONE)
                {
                    return false;
                }

                // Otherwise, try harder - evict until we have space or cache is empty
                while (current_memory_usage_ + layer_size > config_.gpu_memory_budget &&
                       !layer_cache_.empty())
                {
                    evictLRU();
                }

                // Final check
                if (current_memory_usage_ + layer_size > config_.gpu_memory_budget)
                {
                    LOG_ERROR("[LayerWeightStreamer] Failed to free memory for layer " << layer_idx);
                    return false;
                }
            }
        }

        // Load weights
        CachedLayer cached;
        cached.device = device;
        cached.last_access = std::chrono::steady_clock::now();
        cached.memory_bytes = 0;

        auto weight_names = getLayerWeightNames(layer_idx);
        for (const auto &name : weight_names)
        {
            // Request weight on the target device
            // WeightManager will handle the CPU->GPU transfer
            auto weight = weight_manager_->getWeightForDevice(name, device, layer_idx);
            if (!weight)
            {
                LOG_WARN("[LayerWeightStreamer] Failed to load weight: " << name);
                continue; // Some weights might be optional (e.g., biases)
            }

            cached.weights.push_back(weight);
            cached.memory_bytes += weight->size_bytes();
        }

        if (cached.weights.empty())
        {
            LOG_ERROR("[LayerWeightStreamer] No weights loaded for layer " << layer_idx);
            return false;
        }

        // Update cache
        layer_cache_[layer_idx] = std::move(cached);
        current_memory_usage_ += layer_cache_[layer_idx].memory_bytes;

        // Add to LRU (most recently used = front)
        lru_order_.push_front(layer_idx);
        lru_map_[layer_idx] = lru_order_.begin();

        // Update stats
        auto end_time = std::chrono::steady_clock::now();
        auto transfer_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.layers_transferred++;
            stats_.bytes_transferred += layer_cache_[layer_idx].memory_bytes;
            stats_.total_transfer_time += transfer_time;
            if (transfer_time > stats_.max_transfer_latency)
            {
                stats_.max_transfer_latency = transfer_time;
            }
            stats_.current_cache_size = current_memory_usage_;
            stats_.current_layers_cached = layer_cache_.size();
            if (current_memory_usage_ > stats_.peak_cache_size)
            {
                stats_.peak_cache_size = current_memory_usage_;
            }
        }

        if (config_.log_transfer_stats)
        {
            LOG_DEBUG("[LayerWeightStreamer] Loaded layer " << layer_idx
                                                            << " to " << device.to_string()
                                                            << " (" << (layer_cache_[layer_idx].memory_bytes / (1024.0 * 1024.0)) << " MB"
                                                            << ", " << (transfer_time.count() / 1e6) << " ms)");
        }

        return true;
    }

    bool LayerWeightStreamer::evictToFreeMemory(size_t bytes_needed)
    {
        size_t freed = 0;

        while (freed < bytes_needed && !layer_cache_.empty())
        {
            size_t this_eviction = evictLRU();
            if (this_eviction == 0)
            {
                break; // No more layers to evict
            }
            freed += this_eviction;
        }

        return freed >= bytes_needed;
    }

    size_t LayerWeightStreamer::evictLRU()
    {
        if (lru_order_.empty())
        {
            return 0;
        }

        // Get least recently used (back of list)
        int layer_to_evict = lru_order_.back();

        auto it = layer_cache_.find(layer_to_evict);
        if (it == layer_cache_.end())
        {
            // Inconsistency - clean up LRU
            lru_order_.pop_back();
            lru_map_.erase(layer_to_evict);
            return 0;
        }

        size_t freed = it->second.memory_bytes;
        bool was_prefetched = it->second.from_prefetch;

        // Remove from cache
        layer_cache_.erase(it);
        lru_order_.pop_back();
        lru_map_.erase(layer_to_evict);
        current_memory_usage_ -= freed;

        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.layers_evicted++;
            stats_.bytes_evicted += freed;
            if (was_prefetched)
            {
                stats_.prefetch_wasted++;
            }
            stats_.current_cache_size = current_memory_usage_;
            stats_.current_layers_cached = layer_cache_.size();
        }

        LOG_TRACE("[LayerWeightStreamer] LRU evicted layer " << layer_to_evict
                                                             << " (freed " << freed << " bytes)");

        return freed;
    }

    void LayerWeightStreamer::touchLayer(int layer_idx)
    {
        auto it = lru_map_.find(layer_idx);
        if (it == lru_map_.end())
        {
            return; // Not in LRU (shouldn't happen)
        }

        // Move to front (most recently used)
        lru_order_.erase(it->second);
        lru_order_.push_front(layer_idx);
        it->second = lru_order_.begin();
    }

    void LayerWeightStreamer::removeLRUEntry(int layer_idx)
    {
        auto it = lru_map_.find(layer_idx);
        if (it != lru_map_.end())
        {
            lru_order_.erase(it->second);
            lru_map_.erase(it);
        }
    }

} // namespace llaminar2
