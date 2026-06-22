# Weight Streaming Design (Option B)

**Author:** GitHub Copilot (Claude Opus 4.5)  
**Date:** January 11, 2026  
**Status:** Design Document (Pre-Implementation)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background and Motivation](#background-and-motivation)
3. [Design Goals](#design-goals)
4. [New Types and Enums](#new-types-and-enums)
5. [IWeightStreamer Interface](#iweightstreamer-interface)
6. [Implementations](#implementations)
7. [Integration Points](#integration-points)
8. [Memory Management](#memory-management)
9. [Async Prefetch Architecture](#async-prefetch-architecture)
10. [Environment Variable Configuration](#environment-variable-configuration)
11. [File Organization](#file-organization)
12. [Test Strategy](#test-strategy)
13. [Performance Considerations](#performance-considerations)
14. [Migration Path](#migration-path)
15. [Future Extensions](#future-extensions)

---

## Executive Summary

**Option B (Weight Streaming)** addresses VRAM-constrained systems where the GPU cannot hold 100% of model weights. Unlike Option A (Selective Duplication) where GPU has full weights for prefill, Option B streams weights layer-by-layer from host memory to GPU during both PREFILL and DECODE phases.

**Key Innovation:** Overlap compute with data transfer by async-prefetching layer N+1 weights while layer N computes. This hides H2D (Host-to-Device) transfer latency behind computation.

**Trade-offs:**
| Aspect | Option A (Selective Duplication) | Option B (Weight Streaming) |
|--------|----------------------------------|----------------------------|
| GPU VRAM Requirement | 100% of weights | ~2-3 layers worth |
| Performance | Optimal | ~10-30% slower (PCIe bound) |
| Use Case | VRAM > model size | VRAM < model size |
| Complexity | Simpler | More complex (async scheduling) |

---

## Background and Motivation

### Current Architecture (Option A)

From [WeightPlacementMap.h](../../../../src/v2/loaders/WeightPlacementMap.h#L21-L33):

```cpp
/**
 * @brief Extended device information for phase-aware weight placement
 *
 * Supports the "Selective Duplication" pattern where:
 * - PREFILL: GPU has 100% weights (compute-bound, needs fast matmul)
 * - DECODE: CPU participates with a shard (~20%) (bandwidth-bound)
 */
struct WeightDeviceInfo {
    DeviceId prefill_device;              // Primary device for prefill (usually GPU)
    std::vector<DeviceId> decode_devices; // Devices participating in decode
    std::vector<float> decode_fractions;  // Fraction each decode device handles
    bool cpu_decode_participation = false;
};
```

**Problem:** Option A assumes GPU can hold 100% weights. For large models (70B+) on consumer GPUs (24GB VRAM), this is impossible.

### Option B Solution

Stream weights on-demand:
1. Weights reside primarily on CPU (pinned host memory for async transfer)
2. Before computing layer N, ensure weights are on GPU
3. While computing layer N, prefetch layer N+1 weights
4. After layer completes, optionally evict from GPU (LRU cache policy)

---

## Design Goals

1. **Interface-first**: `IWeightStreamer` for mockability and testing
2. **Zero-overhead when disabled**: `NullWeightStreamer` compiles to no-ops
3. **Async prefetch**: Overlap compute and transfer via CUDA streams
4. **Configurable memory budget**: User-specified GPU weight cache size
5. **LRU eviction**: Keep hot layers cached when budget allows
6. **Phase-aware**: Different strategies for PREFILL vs DECODE
7. **Minimal integration surface**: Clean hooks in GraphOrchestrator

---

## New Types and Enums

### WeightResidencyMode

```cpp
// src/v2/loaders/WeightResidency.h

namespace llaminar2 {

/**
 * @brief Weight residency mode for memory management
 *
 * Controls where weights primarily reside and how they're accessed.
 */
enum class WeightResidencyMode {
    /// Weights are resident on target device (no streaming needed)
    /// Use when: GPU VRAM >= model size (Option A)
    RESIDENT,
    
    /// Weights stream from host to device on demand
    /// Use when: GPU VRAM < model size (Option B)
    STREAMING,
    
    /// Weights shared across devices via unified memory
    /// Use when: Unified memory available (e.g., Apple Silicon, Grace Hopper)
    UNIFIED
};

/**
 * @brief Convert WeightResidencyMode to string for logging
 */
inline const char* toString(WeightResidencyMode mode) {
    switch (mode) {
        case WeightResidencyMode::RESIDENT:  return "RESIDENT";
        case WeightResidencyMode::STREAMING: return "STREAMING";
        case WeightResidencyMode::UNIFIED:   return "UNIFIED";
        default:                              return "UNKNOWN";
    }
}

} // namespace llaminar2
```

### StreamingConfig

```cpp
// src/v2/loaders/StreamingConfig.h

namespace llaminar2 {

/**
 * @brief Configuration for weight streaming (Option B)
 *
 * Controls memory budget, prefetch behavior, and eviction policy.
 */
struct StreamingConfig {
    // =========================================================================
    // Memory Budget
    // =========================================================================
    
    /// Maximum GPU memory (bytes) to use for weight cache
    /// Default: 0 = auto-detect (use 80% of available after activations)
    size_t gpu_memory_budget = 0;
    
    /// Minimum layers to keep in GPU cache (prevents thrashing)
    /// Default: 2 (current layer + 1 prefetch)
    int min_cached_layers = 2;
    
    /// Maximum layers to cache (0 = unlimited, bounded by budget)
    int max_cached_layers = 0;
    
    // =========================================================================
    // Prefetch Configuration
    // =========================================================================
    
    /// Number of layers to prefetch ahead
    /// Default: 1 (prefetch layer N+1 while computing layer N)
    int prefetch_depth = 1;
    
    /// Enable async prefetch (requires CUDA streams or similar)
    /// Disable for debugging to make transfers synchronous
    bool async_prefetch = true;
    
    /// Use pinned (page-locked) host memory for faster H2D transfers
    /// Significant speedup but uses more host memory
    bool use_pinned_memory = true;
    
    // =========================================================================
    // Eviction Policy
    // =========================================================================
    
    /// Eviction policy for weight cache
    enum class EvictionPolicy {
        LRU,        ///< Least Recently Used (default)
        FIFO,       ///< First In First Out
        LAYER_ORDER ///< Evict in layer order (lowest first for forward pass)
    };
    EvictionPolicy eviction_policy = EvictionPolicy::LRU;
    
    /// Aggressively evict after layer completion
    /// true: Free memory ASAP (minimize peak VRAM)
    /// false: Keep cached until budget exceeded (better for re-runs)
    bool aggressive_eviction = false;
    
    // =========================================================================
    // Phase-Specific Behavior
    // =========================================================================
    
    /// Different streaming behavior per phase
    struct PhaseConfig {
        bool enabled = true;           ///< Stream in this phase
        int prefetch_depth = 1;        ///< Phase-specific prefetch depth
        bool aggressive_eviction = false;
    };
    
    PhaseConfig prefill_config;  ///< PREFILL phase settings
    PhaseConfig decode_config;   ///< DECODE phase settings
    
    // =========================================================================
    // Debugging
    // =========================================================================
    
    /// Log streaming events (layer load/evict/prefetch)
    bool verbose_logging = false;
    
    /// Collect timing statistics
    bool collect_stats = false;
    
    // =========================================================================
    // Factory Methods
    // =========================================================================
    
    /// Create config for small GPU (aggressive streaming, minimal cache)
    static StreamingConfig forSmallGPU(size_t vram_bytes) {
        StreamingConfig cfg;
        cfg.gpu_memory_budget = vram_bytes * 0.7;  // 70% for weights
        cfg.min_cached_layers = 2;
        cfg.aggressive_eviction = true;
        return cfg;
    }
    
    /// Create config for medium GPU (balanced caching)
    static StreamingConfig forMediumGPU(size_t vram_bytes) {
        StreamingConfig cfg;
        cfg.gpu_memory_budget = vram_bytes * 0.8;  // 80% for weights
        cfg.min_cached_layers = 4;
        cfg.prefetch_depth = 2;
        return cfg;
    }
    
    /// Create from environment variables
    static StreamingConfig fromEnvironment();
};

} // namespace llaminar2
```

### StreamingStats

```cpp
// src/v2/loaders/StreamingStats.h

namespace llaminar2 {

/**
 * @brief Statistics for weight streaming operations
 *
 * Collected when StreamingConfig::collect_stats is enabled.
 */
struct StreamingStats {
    // Transfer counts
    size_t total_uploads = 0;       ///< Total H2D transfers
    size_t cache_hits = 0;          ///< Layers found in GPU cache
    size_t cache_misses = 0;        ///< Layers requiring upload
    size_t evictions = 0;           ///< Layers evicted from GPU cache
    
    // Timing (nanoseconds)
    uint64_t total_upload_time_ns = 0;      ///< Time spent uploading
    uint64_t total_wait_time_ns = 0;        ///< Time waiting for prefetch
    uint64_t max_upload_time_ns = 0;        ///< Slowest single upload
    
    // Memory
    size_t peak_gpu_cache_bytes = 0;        ///< Peak GPU memory for weights
    size_t total_bytes_transferred = 0;     ///< Total bytes uploaded
    
    // Prefetch effectiveness
    size_t prefetch_hits = 0;       ///< Prefetch completed before needed
    size_t prefetch_misses = 0;     ///< Had to wait for prefetch
    
    /// Compute cache hit rate
    float hitRate() const {
        size_t total = cache_hits + cache_misses;
        return total > 0 ? static_cast<float>(cache_hits) / total : 0.0f;
    }
    
    /// Compute prefetch effectiveness
    float prefetchEffectiveness() const {
        size_t total = prefetch_hits + prefetch_misses;
        return total > 0 ? static_cast<float>(prefetch_hits) / total : 0.0f;
    }
    
    /// Compute effective bandwidth (GB/s)
    float effectiveBandwidthGBps() const {
        if (total_upload_time_ns == 0) return 0.0f;
        double seconds = total_upload_time_ns / 1e9;
        double gb = total_bytes_transferred / 1e9;
        return static_cast<float>(gb / seconds);
    }
    
    /// Print human-readable summary
    void print() const;
    
    /// Reset all counters
    void reset();
};

} // namespace llaminar2
```

---

## IWeightStreamer Interface

```cpp
// src/v2/loaders/IWeightStreamer.h

#pragma once

#include "../backends/DeviceId.h"
#include "../execution/PlacementStrategy.h"  // For InferencePhase
#include "StreamingConfig.h"
#include "StreamingStats.h"
#include <memory>
#include <vector>
#include <string>

namespace llaminar2 {

// Forward declarations
class TensorBase;
class WeightManager;

/**
 * @brief Layer weight bundle for streaming operations
 *
 * Groups all weights for a single transformer layer.
 * Enables efficient bulk transfer and cache management.
 */
struct LayerWeightBundle {
    int layer_idx = -1;
    
    // Attention weights
    std::shared_ptr<TensorBase> attn_norm;
    std::shared_ptr<TensorBase> attn_q;
    std::shared_ptr<TensorBase> attn_k;
    std::shared_ptr<TensorBase> attn_v;
    std::shared_ptr<TensorBase> attn_output;  // Wo
    
    // FFN weights
    std::shared_ptr<TensorBase> ffn_norm;
    std::shared_ptr<TensorBase> ffn_gate;
    std::shared_ptr<TensorBase> ffn_up;
    std::shared_ptr<TensorBase> ffn_down;
    
    /// Compute total memory footprint
    size_t totalBytes() const;
    
    /// Check if all weights are present
    bool isComplete() const;
    
    /// Check if any weight is on the target device
    bool isOnDevice(DeviceId device) const;
};

/**
 * @brief Interface for weight streaming (Option B)
 *
 * Abstracts weight H2D transfer and GPU memory management.
 * Enables mocking for unit tests and different implementations
 * for various hardware configurations.
 *
 * Lifecycle:
 * 1. Create streamer with config and weight manager
 * 2. Before computing layer N: ensureLayerOnDevice(N, device)
 * 3. Optionally: prefetchLayer(N+1, device) for async overlap
 * 4. After layer N: releaseLayer(N) to allow eviction
 * 5. End of inference: synchronize() to wait for all transfers
 *
 * Thread Safety:
 * - ensureLayerOnDevice() is blocking and thread-safe
 * - prefetchLayer() is async but thread-safe
 * - Multiple threads should not call these for the same layer
 */
class IWeightStreamer {
public:
    virtual ~IWeightStreamer() = default;
    
    // =========================================================================
    // Core Streaming Operations
    // =========================================================================
    
    /**
     * @brief Ensure all weights for a layer are on the target device
     *
     * If weights are already cached, returns immediately.
     * If a prefetch is in progress, waits for it to complete.
     * Otherwise, performs synchronous H2D transfer.
     *
     * This is the primary method called before layer execution.
     *
     * @param layer_idx Layer index (0 to n_layers-1)
     * @param device Target device (GPU)
     * @return true if weights are ready, false on error
     */
    virtual bool ensureLayerOnDevice(int layer_idx, DeviceId device) = 0;
    
    /**
     * @brief Start async prefetch of a layer's weights
     *
     * Initiates H2D transfer on a separate stream. The transfer
     * proceeds while compute continues on the main stream.
     *
     * No-op if:
     * - Layer already on device
     * - Prefetch already in progress for this layer
     * - Async prefetch disabled in config
     *
     * @param layer_idx Layer index to prefetch
     * @param device Target device
     */
    virtual void prefetchLayer(int layer_idx, DeviceId device) = 0;
    
    /**
     * @brief Mark layer as complete (may trigger eviction)
     *
     * Signals that the layer's computation is done. The streamer
     * updates LRU tracking and may evict the layer if:
     * - aggressive_eviction is enabled, OR
     * - Memory budget is exceeded
     *
     * @param layer_idx Layer index to release
     */
    virtual void releaseLayer(int layer_idx) = 0;
    
    /**
     * @brief Bulk prefetch multiple layers
     *
     * Convenience method for prefetching a range of layers.
     * Useful at the start of prefill to pipeline uploads.
     *
     * @param start_layer First layer to prefetch (inclusive)
     * @param end_layer Last layer to prefetch (inclusive)
     * @param device Target device
     */
    virtual void prefetchLayers(int start_layer, int end_layer, DeviceId device) {
        for (int i = start_layer; i <= end_layer; ++i) {
            prefetchLayer(i, device);
        }
    }
    
    // =========================================================================
    // Synchronization
    // =========================================================================
    
    /**
     * @brief Wait for all pending transfers to complete
     *
     * Blocks until all async prefetches finish. Call this:
     * - At end of forward pass
     * - Before accessing weight data for debugging
     * - When switching phases
     */
    virtual void synchronize() = 0;
    
    /**
     * @brief Wait for a specific layer's transfer to complete
     *
     * Lighter-weight than full synchronize().
     *
     * @param layer_idx Layer to wait for
     * @return true if layer is ready, false if not being transferred
     */
    virtual bool waitForLayer(int layer_idx) = 0;
    
    // =========================================================================
    // Phase Management
    // =========================================================================
    
    /**
     * @brief Notify streamer of phase transition
     *
     * Allows streamer to adjust behavior (e.g., different prefetch
     * depth for PREFILL vs DECODE, cache warmup strategy).
     *
     * @param phase New inference phase
     */
    virtual void onPhaseTransition(InferencePhase phase) = 0;
    
    // =========================================================================
    // Memory Management
    // =========================================================================
    
    /**
     * @brief Get current GPU memory usage for weight cache
     *
     * @return Bytes currently used by cached weights
     */
    virtual size_t currentDeviceMemoryUsage() const = 0;
    
    /**
     * @brief Get configured GPU memory budget
     *
     * @return Maximum bytes for weight cache
     */
    virtual size_t memoryBudget() const = 0;
    
    /**
     * @brief Get number of layers currently cached
     */
    virtual int cachedLayerCount() const = 0;
    
    /**
     * @brief Explicitly evict a layer from GPU cache
     *
     * Frees GPU memory for the layer. No-op if layer not cached.
     *
     * @param layer_idx Layer to evict
     */
    virtual void evictLayer(int layer_idx) = 0;
    
    /**
     * @brief Clear entire GPU weight cache
     *
     * Evicts all layers, freeing GPU memory. Weights remain on CPU.
     */
    virtual void clearCache() = 0;
    
    // =========================================================================
    // Diagnostics
    // =========================================================================
    
    /**
     * @brief Check if layer is currently in GPU cache
     */
    virtual bool isLayerCached(int layer_idx) const = 0;
    
    /**
     * @brief Check if a prefetch is in progress for a layer
     */
    virtual bool isPrefetchInProgress(int layer_idx) const = 0;
    
    /**
     * @brief Get streaming statistics (if enabled)
     *
     * @return Reference to stats, or empty stats if disabled
     */
    virtual const StreamingStats& stats() const = 0;
    
    /**
     * @brief Reset streaming statistics
     */
    virtual void resetStats() = 0;
    
    // =========================================================================
    // Weight Access (for integration with existing weight APIs)
    // =========================================================================
    
    /**
     * @brief Get weight bundle for a layer
     *
     * Returns all weights for a layer, loading from CPU if needed.
     * Unlike ensureLayerOnDevice(), this doesn't guarantee GPU residency.
     *
     * @param layer_idx Layer index
     * @return Layer weight bundle
     */
    virtual LayerWeightBundle getLayerWeights(int layer_idx) = 0;
    
    /**
     * @brief Get device-resident pointer for a specific weight
     *
     * Returns GPU pointer if weight is cached, nullptr otherwise.
     * Use after ensureLayerOnDevice() to get compute-ready pointers.
     *
     * @param layer_idx Layer index
     * @param weight_name Weight name (e.g., "attn_q", "ffn_gate")
     * @param device Target device
     * @return TensorBase pointer on device, or nullptr if not cached
     */
    virtual std::shared_ptr<TensorBase> getDeviceWeight(
        int layer_idx,
        const std::string& weight_name,
        DeviceId device) = 0;
};

/**
 * @brief Create weight streamer based on residency mode
 *
 * Factory function for creating appropriate streamer implementation.
 *
 * @param mode Weight residency mode
 * @param config Streaming configuration (ignored for RESIDENT mode)
 * @param weight_manager Weight manager for CPU weight access
 * @param n_layers Number of transformer layers
 * @return Unique pointer to IWeightStreamer implementation
 */
std::unique_ptr<IWeightStreamer> createWeightStreamer(
    WeightResidencyMode mode,
    const StreamingConfig& config,
    std::shared_ptr<WeightManager> weight_manager,
    int n_layers);

} // namespace llaminar2
```

---

## Implementations

### NullWeightStreamer

Zero-overhead implementation for when weights are already resident:

```cpp
// src/v2/loaders/NullWeightStreamer.h

#pragma once

#include "IWeightStreamer.h"

namespace llaminar2 {

/**
 * @brief No-op weight streamer for resident weights (Option A)
 *
 * Used when GPU has sufficient VRAM to hold all weights.
 * All methods are no-ops or return trivial values.
 *
 * This implementation has zero runtime overhead - all methods
 * inline to nothing or simple returns.
 */
class NullWeightStreamer : public IWeightStreamer {
public:
    explicit NullWeightStreamer(std::shared_ptr<WeightManager> weight_manager);
    
    // Core operations - all no-ops
    bool ensureLayerOnDevice(int layer_idx, DeviceId device) override { return true; }
    void prefetchLayer(int layer_idx, DeviceId device) override {}
    void releaseLayer(int layer_idx) override {}
    void synchronize() override {}
    bool waitForLayer(int layer_idx) override { return true; }
    
    // Phase management - no-op
    void onPhaseTransition(InferencePhase phase) override {}
    
    // Memory management - trivial
    size_t currentDeviceMemoryUsage() const override { return 0; }
    size_t memoryBudget() const override { return SIZE_MAX; }
    int cachedLayerCount() const override { return 0; }
    void evictLayer(int layer_idx) override {}
    void clearCache() override {}
    
    // Diagnostics - trivial
    bool isLayerCached(int layer_idx) const override { return true; }
    bool isPrefetchInProgress(int layer_idx) const override { return false; }
    const StreamingStats& stats() const override { return empty_stats_; }
    void resetStats() override {}
    
    // Weight access - delegate to weight manager
    LayerWeightBundle getLayerWeights(int layer_idx) override;
    std::shared_ptr<TensorBase> getDeviceWeight(
        int layer_idx, const std::string& weight_name, DeviceId device) override;
    
private:
    std::shared_ptr<WeightManager> weight_manager_;
    StreamingStats empty_stats_;
};

} // namespace llaminar2
```

### LayerWeightStreamer

Full implementation with async prefetch and LRU eviction:

```cpp
// src/v2/loaders/LayerWeightStreamer.h

#pragma once

#include "IWeightStreamer.h"
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <list>
#include <thread>
#include <atomic>
#include <queue>

namespace llaminar2 {

// Forward declarations
class CUDAStream;  // Backend-specific stream wrapper

/**
 * @brief Layer cache entry for GPU weight management
 */
struct LayerCacheEntry {
    int layer_idx = -1;
    LayerWeightBundle gpu_weights;    ///< Weights on GPU
    size_t memory_bytes = 0;          ///< Memory footprint
    std::chrono::steady_clock::time_point last_access;
    bool eviction_pending = false;
    
    void updateAccessTime() {
        last_access = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Pending prefetch operation
 */
struct PrefetchOp {
    int layer_idx = -1;
    DeviceId target_device;
    std::atomic<bool> completed{false};
    std::atomic<bool> cancelled{false};
    void* stream_handle = nullptr;    ///< Backend-specific stream
};

/**
 * @brief Full weight streaming implementation (Option B)
 *
 * Features:
 * - Async H2D transfer via dedicated CUDA stream
 * - LRU eviction when memory budget exceeded
 * - Configurable prefetch depth
 * - Phase-aware behavior
 * - Statistics collection
 *
 * Thread Safety:
 * - Internal mutex protects cache state
 * - Async prefetch uses separate stream, doesn't block compute
 * - Safe to call from multiple threads (but not same layer)
 *
 * Memory Layout:
 * - CPU: Weights in pinned memory for fast DMA
 * - GPU: LRU cache with configurable budget
 */
class LayerWeightStreamer : public IWeightStreamer {
public:
    /**
     * @brief Construct layer weight streamer
     *
     * @param config Streaming configuration
     * @param weight_manager Weight manager for CPU weight access
     * @param n_layers Number of transformer layers
     */
    LayerWeightStreamer(
        const StreamingConfig& config,
        std::shared_ptr<WeightManager> weight_manager,
        int n_layers);
    
    ~LayerWeightStreamer();
    
    // Non-copyable
    LayerWeightStreamer(const LayerWeightStreamer&) = delete;
    LayerWeightStreamer& operator=(const LayerWeightStreamer&) = delete;
    
    // =========================================================================
    // IWeightStreamer Implementation
    // =========================================================================
    
    bool ensureLayerOnDevice(int layer_idx, DeviceId device) override;
    void prefetchLayer(int layer_idx, DeviceId device) override;
    void releaseLayer(int layer_idx) override;
    void synchronize() override;
    bool waitForLayer(int layer_idx) override;
    
    void onPhaseTransition(InferencePhase phase) override;
    
    size_t currentDeviceMemoryUsage() const override;
    size_t memoryBudget() const override { return config_.gpu_memory_budget; }
    int cachedLayerCount() const override;
    void evictLayer(int layer_idx) override;
    void clearCache() override;
    
    bool isLayerCached(int layer_idx) const override;
    bool isPrefetchInProgress(int layer_idx) const override;
    const StreamingStats& stats() const override { return stats_; }
    void resetStats() override { stats_.reset(); }
    
    LayerWeightBundle getLayerWeights(int layer_idx) override;
    std::shared_ptr<TensorBase> getDeviceWeight(
        int layer_idx, const std::string& weight_name, DeviceId device) override;
    
private:
    // =========================================================================
    // Internal Methods
    // =========================================================================
    
    /// Load layer weights from CPU to GPU (synchronous)
    bool uploadLayer(int layer_idx, DeviceId device);
    
    /// Evict layers until within budget
    void evictToFitBudget(size_t required_bytes);
    
    /// Get LRU candidate for eviction
    int selectEvictionCandidate() const;
    
    /// Cancel pending prefetch for a layer
    void cancelPrefetch(int layer_idx);
    
    /// Start async upload on prefetch stream
    void startAsyncUpload(int layer_idx, DeviceId device);
    
    /// Wait for async upload to complete
    bool waitForUpload(int layer_idx);
    
    /// Compute memory footprint for a layer
    size_t estimateLayerMemory(int layer_idx) const;
    
    /// Log streaming event (if verbose)
    void logEvent(const char* event, int layer_idx) const;
    
    // =========================================================================
    // State
    // =========================================================================
    
    StreamingConfig config_;
    std::shared_ptr<WeightManager> weight_manager_;
    int n_layers_;
    InferencePhase current_phase_ = InferencePhase::PREFILL;
    
    // Cache state (protected by mutex_)
    mutable std::mutex mutex_;
    std::unordered_map<int, LayerCacheEntry> cache_;
    std::list<int> lru_order_;  ///< Front = most recently used
    size_t current_memory_usage_ = 0;
    
    // Prefetch state
    std::unordered_map<int, std::unique_ptr<PrefetchOp>> pending_prefetches_;
    void* prefetch_stream_ = nullptr;  ///< Dedicated CUDA stream for prefetch
    
    // Statistics
    StreamingStats stats_;
};

} // namespace llaminar2
```

---

## Integration Points

### GraphOrchestrator Integration

The primary integration point is in `GraphOrchestrator`, which orchestrates layer execution:

```cpp
// Additions to GraphOrchestrator.h

class GraphOrchestrator : public IInferenceRunner {
public:
    // ... existing members ...
    
    // =========================================================================
    // Weight Streaming (Option B)
    // =========================================================================
    
    /**
     * @brief Set weight streamer for layer-by-layer weight management
     *
     * When set, the orchestrator will call streamer methods before/after
     * each layer execution to manage GPU weight residency.
     *
     * @param streamer Weight streamer (ownership transferred)
     */
    void setWeightStreamer(std::unique_ptr<IWeightStreamer> streamer);
    
    /**
     * @brief Get weight streamer
     * @return Pointer to streamer, or nullptr if not set
     */
    IWeightStreamer* weightStreamer() const { return weight_streamer_.get(); }
    
    /**
     * @brief Check if weight streaming is enabled
     */
    bool isWeightStreamingEnabled() const { return weight_streamer_ != nullptr; }
    
private:
    std::unique_ptr<IWeightStreamer> weight_streamer_;
};
```

### Layer Execution Hook Points

In `GraphOrchestrator::executeLayer()`:

```cpp
bool GraphOrchestrator::executeLayer(
    const Qwen2LayerWeights& layer,
    Qwen2ActivationBuffers& buffers,
    int layer_idx,
    int seq_len,
    IKVCache* kv_cache,
    const int* position_ids,
    DeviceId device) 
{
    // =========================================================================
    // WEIGHT STREAMING HOOK: Ensure current layer on device
    // =========================================================================
    if (weight_streamer_) {
        if (!weight_streamer_->ensureLayerOnDevice(layer_idx, device)) {
            LOG_ERROR("Failed to stream layer " << layer_idx << " to device");
            return false;
        }
        
        // Prefetch next layer(s) while this one computes
        const int prefetch_depth = 1;  // Or from config
        for (int i = 1; i <= prefetch_depth && layer_idx + i < n_layers_; ++i) {
            weight_streamer_->prefetchLayer(layer_idx + i, device);
        }
    }
    
    // =========================================================================
    // Execute attention and FFN (existing code)
    // =========================================================================
    if (!executeAttention(layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device)) {
        return false;
    }
    if (!executeFFN(layer, buffers, layer_idx, seq_len, device)) {
        return false;
    }
    
    // =========================================================================
    // WEIGHT STREAMING HOOK: Release layer for potential eviction
    // =========================================================================
    if (weight_streamer_) {
        weight_streamer_->releaseLayer(layer_idx);
    }
    
    return true;
}
```

### Phase Transition Integration

In `GraphOrchestrator::transitionToPhase()`:

```cpp
void GraphOrchestrator::transitionToPhase(InferencePhase phase) {
    if (current_phase_ == phase) {
        return;
    }
    
    InferencePhase old_phase = current_phase_;
    current_phase_ = phase;
    
    LOG_DEBUG("Phase transition: " << toString(old_phase) << " → " << toString(phase));
    
    // Notify weight streamer of phase transition
    if (weight_streamer_) {
        // Synchronize any pending prefetches before phase change
        weight_streamer_->synchronize();
        weight_streamer_->onPhaseTransition(phase);
        
        // DECODE phase may use different cache strategy
        // (e.g., keep all layers cached since we iterate multiple times)
    }
}
```

### Forward Pass Integration

In `GraphOrchestrator::executeForward()`:

```cpp
bool GraphOrchestrator::executeForward(
    const Qwen2ForwardInput& input,
    Qwen2ForwardOutput& output) 
{
    // Determine phase
    InferencePhase phase = input.seq_len > 1 ? InferencePhase::PREFILL 
                                              : InferencePhase::DECODE;
    transitionToPhase(phase);
    
    // For PREFILL, optionally pre-warm cache with first few layers
    if (weight_streamer_ && phase == InferencePhase::PREFILL) {
        int prewarm_count = std::min(3, n_layers_);
        weight_streamer_->prefetchLayers(0, prewarm_count - 1, device_);
    }
    
    // ... embedding, layer loop, final norm, lm_head ...
    
    // Synchronize at end of forward pass
    if (weight_streamer_) {
        weight_streamer_->synchronize();
    }
    
    return true;
}
```

---

## Memory Management

### GPU Memory Budget Calculation

```cpp
// src/v2/loaders/MemoryBudgetCalculator.h

namespace llaminar2 {

/**
 * @brief Calculate available GPU memory for weight streaming
 *
 * Accounts for:
 * - Current VRAM usage (activation buffers, KV cache)
 * - Safety margin (10%)
 * - Minimum required for other allocations
 */
struct MemoryBudgetCalculator {
    /**
     * @brief Calculate weight cache budget for a device
     *
     * @param device Target GPU device
     * @param activation_memory Estimated activation buffer size
     * @param kv_cache_memory Estimated KV cache size (for max seq len)
     * @param safety_margin_percent Safety margin (default 10%)
     * @return Available bytes for weight cache
     */
    static size_t calculate(
        DeviceId device,
        size_t activation_memory,
        size_t kv_cache_memory,
        float safety_margin_percent = 10.0f);
    
    /**
     * @brief Calculate budget assuming weights must fit N layers
     *
     * @param device Target GPU device
     * @param layer_memory Memory per layer
     * @param min_layers Minimum layers to fit
     * @param activation_memory Other memory requirements
     * @return Available bytes, or 0 if impossible
     */
    static size_t calculateForMinLayers(
        DeviceId device,
        size_t layer_memory,
        int min_layers,
        size_t activation_memory);
};

} // namespace llaminar2
```

### LRU Cache Implementation

```cpp
// Internal to LayerWeightStreamer

void LayerWeightStreamer::evictToFitBudget(size_t required_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    while (current_memory_usage_ + required_bytes > config_.gpu_memory_budget) {
        if (cache_.size() <= static_cast<size_t>(config_.min_cached_layers)) {
            LOG_WARN("Cannot evict: at minimum cached layers (" 
                     << config_.min_cached_layers << ")");
            break;
        }
        
        int victim = selectEvictionCandidate();
        if (victim < 0) {
            LOG_ERROR("No eviction candidate found");
            break;
        }
        
        evictLayerUnlocked(victim);
    }
}

int LayerWeightStreamer::selectEvictionCandidate() const {
    // LRU policy: evict least recently accessed
    // lru_order_ is maintained with most recent at front
    
    for (auto it = lru_order_.rbegin(); it != lru_order_.rend(); ++it) {
        int layer_idx = *it;
        auto cache_it = cache_.find(layer_idx);
        if (cache_it != cache_.end() && !cache_it->second.eviction_pending) {
            // Don't evict if prefetch in progress
            if (!isPrefetchInProgress(layer_idx)) {
                return layer_idx;
            }
        }
    }
    
    return -1;  // No candidate
}
```

---

## Async Prefetch Architecture

### CUDA Stream Management

```cpp
// src/v2/backends/cuda/CUDAStreamPool.h (conceptual)

/**
 * @brief Dedicated streams for weight prefetch
 *
 * Uses separate CUDA stream(s) for H2D transfers so they can
 * overlap with compute on the main stream.
 *
 * Architecture:
 * - Main stream: Layer N computation (GEMM, attention, etc.)
 * - Prefetch stream: Layer N+1 weight upload (cudaMemcpyAsync)
 *
 * Synchronization:
 * - ensureLayerOnDevice() records event on prefetch stream
 * - Main stream waits on that event before using weights
 */

class PrefetchStreamManager {
public:
    /// Get stream for prefetch operations
    cudaStream_t getPrefetchStream();
    
    /// Record completion event for a layer
    void recordLayerComplete(int layer_idx);
    
    /// Wait for layer to be ready on main stream
    void waitForLayer(int layer_idx, cudaStream_t main_stream);
    
private:
    cudaStream_t prefetch_stream_;
    std::unordered_map<int, cudaEvent_t> layer_events_;
};
```

### Overlapping Pattern

```
Timeline:
=========

Main Stream:     [Layer 0 compute]--[Layer 1 compute]--[Layer 2 compute]--...
                        |                  |                  |
                        v                  v                  v
Prefetch Stream: [Upload L1]--------[Upload L2]--------[Upload L3]----...
                 ^                  ^                  ^
                 |                  |                  |
                 Prefetch L1        Prefetch L2        Prefetch L3
                 (during L0)        (during L1)        (during L2)

Overlap benefit:
- H2D transfer (~200 GB/s PCIe 4.0) happens during compute
- Compute is GPU-bound, barely affected by background DMA
- Effective throughput approaches VRAM-resident case
```

---

## Environment Variable Configuration

Add to `src/v2/utils/DebugEnv.h`:

```cpp
struct WeightStreamingEnv {
    bool enabled = false;             ///< LLAMINAR_WEIGHT_STREAMING (0/1)
    size_t memory_budget_mb = 0;      ///< LLAMINAR_STREAM_MEMORY_MB (0 = auto)
    int prefetch_depth = 1;           ///< LLAMINAR_STREAM_PREFETCH_DEPTH
    bool async_prefetch = true;       ///< LLAMINAR_STREAM_ASYNC (0/1)
    bool use_pinned_memory = true;    ///< LLAMINAR_STREAM_PINNED (0/1)
    bool aggressive_eviction = false; ///< LLAMINAR_STREAM_AGGRESSIVE_EVICT (0/1)
    bool verbose = false;             ///< LLAMINAR_STREAM_VERBOSE (0/1)
    bool collect_stats = false;       ///< LLAMINAR_STREAM_STATS (0/1)
};
```

### Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_WEIGHT_STREAMING` | Enable weight streaming mode (1=enable) | 0 |
| `LLAMINAR_STREAM_MEMORY_MB` | GPU memory budget in MB (0=auto ~80%) | 0 |
| `LLAMINAR_STREAM_PREFETCH_DEPTH` | Layers to prefetch ahead | 1 |
| `LLAMINAR_STREAM_ASYNC` | Enable async prefetch (0=sync for debug) | 1 |
| `LLAMINAR_STREAM_PINNED` | Use pinned host memory | 1 |
| `LLAMINAR_STREAM_AGGRESSIVE_EVICT` | Evict immediately after layer | 0 |
| `LLAMINAR_STREAM_VERBOSE` | Log streaming events | 0 |
| `LLAMINAR_STREAM_STATS` | Collect timing statistics | 0 |

### CLI Integration

```bash
# Enable weight streaming with 8GB budget
./build_v2_release/llaminar2 \
    --weight-streaming \
    --stream-memory-mb 8192 \
    -m models/llama-70b-q4.gguf \
    -p "Hello, world!" \
    -n 50

# Or via environment
LLAMINAR_WEIGHT_STREAMING=1 \
LLAMINAR_STREAM_MEMORY_MB=8192 \
./build_v2_release/llaminar2 -m models/llama-70b-q4.gguf -p "Hello"
```

---

## File Organization

### New Files

```
src/v2/loaders/
├── IWeightStreamer.h              # Interface definition
├── WeightResidency.h              # WeightResidencyMode enum
├── StreamingConfig.h              # StreamingConfig struct
├── StreamingStats.h               # StreamingStats struct
├── NullWeightStreamer.h           # Zero-overhead implementation
├── NullWeightStreamer.cpp
├── LayerWeightStreamer.h          # Full streaming implementation
├── LayerWeightStreamer.cpp
├── MemoryBudgetCalculator.h       # GPU memory budget calculation
└── MemoryBudgetCalculator.cpp

tests/v2/unit/loaders/
├── Test__IWeightStreamer.cpp      # Interface tests with mock
├── Test__NullWeightStreamer.cpp   # Null implementation tests
├── Test__LayerWeightStreamer.cpp  # Full implementation unit tests
└── Test__MemoryBudgetCalculator.cpp

tests/v2/integration/streaming/
├── Test__WeightStreamingE2E.cpp   # End-to-end streaming test
├── Test__StreamingParity.cpp      # Streaming vs resident parity
└── Test__StreamingPerformance.cpp # Performance regression tests
```

### Modified Files

| File | Changes |
|------|---------|
| `src/v2/execution/GraphOrchestrator.h` | Add `IWeightStreamer` member, hook methods |
| `src/v2/execution/GraphOrchestrator.cpp` | Integrate streaming calls in layer execution |
| `src/v2/utils/DebugEnv.h` | Add `WeightStreamingEnv` struct |
| `src/v2/utils/DebugEnv.cpp` | Parse streaming environment variables |
| `src/v2/loaders/WeightManager.h` | Add `getPinnedWeight()` for pinned memory |
| `src/v2/loaders/CMakeLists.txt` | Add new source files |
| `tests/v2/CMakeLists.txt` | Add test targets |

---

## Test Strategy

### Unit Tests

```cpp
// tests/v2/unit/loaders/Test__IWeightStreamer.cpp

/**
 * @brief Mock weight streamer for testing orchestrator integration
 */
class MockWeightStreamer : public IWeightStreamer {
public:
    // Track calls for verification
    std::vector<int> ensured_layers;
    std::vector<int> prefetched_layers;
    std::vector<int> released_layers;
    int synchronize_count = 0;
    
    bool ensureLayerOnDevice(int layer_idx, DeviceId device) override {
        ensured_layers.push_back(layer_idx);
        return !fail_on_layer_.count(layer_idx);
    }
    
    void prefetchLayer(int layer_idx, DeviceId device) override {
        prefetched_layers.push_back(layer_idx);
    }
    
    void releaseLayer(int layer_idx) override {
        released_layers.push_back(layer_idx);
    }
    
    void synchronize() override {
        ++synchronize_count;
    }
    
    // Test helpers
    void failOnLayer(int layer_idx) { fail_on_layer_.insert(layer_idx); }
    void reset() { ensured_layers.clear(); prefetched_layers.clear(); released_layers.clear(); }
    
private:
    std::set<int> fail_on_layer_;
};

TEST(Test__IWeightStreamer, OrchestratorCallsStreamer) {
    auto mock = std::make_unique<MockWeightStreamer>();
    auto* mock_ptr = mock.get();
    
    GraphOrchestrator orchestrator(/*...*/);
    orchestrator.setWeightStreamer(std::move(mock));
    
    // Execute 3 layers
    for (int i = 0; i < 3; ++i) {
        orchestrator.executeLayer(/*...layer i...*/);
    }
    
    // Verify streamer was called correctly
    EXPECT_EQ(mock_ptr->ensured_layers, std::vector<int>({0, 1, 2}));
    EXPECT_EQ(mock_ptr->prefetched_layers, std::vector<int>({1, 2, 3}));
    EXPECT_EQ(mock_ptr->released_layers, std::vector<int>({0, 1, 2}));
}

TEST(Test__IWeightStreamer, FailedStreamingAbortsLayer) {
    auto mock = std::make_unique<MockWeightStreamer>();
    mock->failOnLayer(1);
    
    GraphOrchestrator orchestrator(/*...*/);
    orchestrator.setWeightStreamer(std::move(mock));
    
    EXPECT_TRUE(orchestrator.executeLayer(/*...layer 0...*/));
    EXPECT_FALSE(orchestrator.executeLayer(/*...layer 1...*/));  // Should fail
}
```

### Integration Tests

```cpp
// tests/v2/integration/streaming/Test__WeightStreamingE2E.cpp

TEST(Test__WeightStreamingE2E, StreamingProducesSameOutputAsResident) {
    // Load small model
    auto loader = loadTestModel("qwen2.5-0.5b-instruct-q4_0.gguf");
    
    // Run with resident weights (Option A)
    auto output_resident = runInference(loader, WeightResidencyMode::RESIDENT);
    
    // Run with streaming weights (Option B)
    StreamingConfig config;
    config.gpu_memory_budget = 512 * 1024 * 1024;  // 512MB (forces streaming)
    auto output_streaming = runInference(loader, WeightResidencyMode::STREAMING, config);
    
    // Outputs should be identical (deterministic)
    EXPECT_EQ(output_resident.tokens, output_streaming.tokens);
    EXPECT_NEAR_TENSORS(output_resident.logits, output_streaming.logits, 1e-5);
}

TEST(Test__WeightStreamingE2E, StreamingRespectMemoryBudget) {
    auto loader = loadTestModel("qwen2.5-7b-instruct-q4_0.gguf");
    
    StreamingConfig config;
    config.gpu_memory_budget = 2ULL * 1024 * 1024 * 1024;  // 2GB
    config.collect_stats = true;
    
    auto streamer = createWeightStreamer(
        WeightResidencyMode::STREAMING, config, loader.weight_manager(), 32);
    
    // Run inference
    runInferenceWithStreamer(loader, streamer.get());
    
    // Check peak memory stayed within budget
    EXPECT_LE(streamer->stats().peak_gpu_cache_bytes, config.gpu_memory_budget);
}
```

### Performance Tests

```cpp
// tests/v2/integration/streaming/Test__StreamingPerformance.cpp

TEST(Test__StreamingPerformance, PrefetchHidesTransferLatency) {
    auto loader = loadTestModel("qwen2.5-7b-instruct-q4_0.gguf");
    
    // Measure with prefetch enabled
    StreamingConfig config_prefetch;
    config_prefetch.async_prefetch = true;
    config_prefetch.prefetch_depth = 1;
    config_prefetch.collect_stats = true;
    
    auto time_prefetch = benchmarkInference(loader, config_prefetch);
    auto& stats_prefetch = getStats();
    
    // Measure with prefetch disabled
    StreamingConfig config_sync;
    config_sync.async_prefetch = false;
    config_sync.collect_stats = true;
    
    auto time_sync = benchmarkInference(loader, config_sync);
    auto& stats_sync = getStats();
    
    // Prefetch should be significantly faster
    LOG_INFO("Sync time: " << time_sync << "ms, Prefetch time: " << time_prefetch << "ms");
    EXPECT_LT(time_prefetch, time_sync * 0.7);  // At least 30% faster
    
    // Prefetch effectiveness should be high
    EXPECT_GT(stats_prefetch.prefetchEffectiveness(), 0.9f);  // 90%+ prefetch hits
}
```

---

## Performance Considerations

### Expected Overhead

| Scenario | Overhead vs Resident | Mitigation |
|----------|---------------------|------------|
| PCIe 4.0 x16 (32 GB/s) | ~15-25% | Prefetch overlaps 80%+ |
| PCIe 3.0 x16 (16 GB/s) | ~30-40% | Larger prefetch depth (2-3) |
| Small batch decode | ~5-10% | Memory-bound anyway |
| Large batch prefill | ~20-30% | Compute partially hides transfer |

### Optimization Opportunities

1. **Double buffering**: Ping-pong between two GPU buffers per layer
2. **Compression**: Optionally compress weights on CPU, decompress on GPU
3. **Prioritized prefetch**: Attention weights first (needed earlier in layer)
4. **Speculative prefetch**: For decode, prefetch based on predicted next tokens

---

## Migration Path

### Phase 1: Interface and Null Implementation

1. Add `IWeightStreamer` interface
2. Implement `NullWeightStreamer`
3. Add hook points in `GraphOrchestrator` (no-op by default)
4. Unit tests with mock

### Phase 2: LayerWeightStreamer Core

1. Implement synchronous `ensureLayerOnDevice()` / `releaseLayer()`
2. LRU cache management
3. Memory budget enforcement
4. Integration tests (streaming vs resident parity)

### Phase 3: Async Prefetch

1. CUDA stream management for prefetch
2. Async `prefetchLayer()` implementation
3. Event-based synchronization
4. Performance benchmarks

### Phase 4: Polish and Optimization

1. Pinned memory optimization
2. Phase-aware tuning
3. Statistics and diagnostics
4. CLI integration
5. Documentation

---

## Future Extensions

### Unified Memory (Option B')

For systems with unified memory (Apple Silicon, Grace Hopper):

```cpp
class UnifiedWeightStreamer : public IWeightStreamer {
    // Uses cudaMallocManaged() / Metal shared memory
    // No explicit transfers, just prefetch hints
    bool ensureLayerOnDevice(int layer_idx, DeviceId device) override {
        cudaMemPrefetchAsync(layer_data, layer_size, device.cuda_device());
        return true;
    }
};
```

### Multi-GPU Weight Distribution

Extend to distribute layers across multiple GPUs:

```cpp
class MultiGPUWeightStreamer : public IWeightStreamer {
    // Layer i → GPU (i % num_gpus)
    // Automatic P2P transfer for cross-GPU access
};
```

### Speculative Weight Loading

For speculative decoding:

```cpp
class SpeculativeWeightStreamer : public LayerWeightStreamer {
    // Prefetch based on draft model predictions
    void prefetchForTokens(const std::vector<int>& predicted_tokens);
};
```

---

## Conclusion

Option B (Weight Streaming) provides a path forward for VRAM-constrained systems. The interface-first design enables:

- **Clean integration** via `IWeightStreamer` abstraction
- **Zero overhead when unused** via `NullWeightStreamer`
- **Testability** via mock implementations
- **Future extensibility** for unified memory and multi-GPU

The async prefetch architecture hides most transfer latency, achieving 70-90% of resident-weight performance on typical hardware configurations.

**Next Steps:**
1. Review this design document
2. Approve interface definitions
3. Begin Phase 1 implementation (interface + null implementation)
