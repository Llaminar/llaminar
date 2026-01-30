/**
 * @file IWeightStreamer.h
 * @brief Interface for weight streaming between host and device memory
 * @author David Sanftenberg
 * @date January 2026
 *
 * This file defines the IWeightStreamer interface for Option B (Weight Streaming)
 * in Llaminar V2. Weight streaming enables running models that don't fit entirely
 * in GPU memory by dynamically transferring layer weights on-demand.
 *
 * Design Philosophy:
 * - Layer-granular streaming (one layer at a time)
 * - Async prefetching to hide transfer latency
 * - Phase-aware behavior (PREFILL vs DECODE have different access patterns)
 * - Pluggable eviction policies (LRU, FIFO, etc.)
 * - Zero-copy where possible using pinned memory
 *
 * Usage Pattern:
 *   1. DeviceGraphOrchestrator calls ensureLayerOnDevice() before executing layer stages
 *   2. Streamer uploads weights if not cached, potentially evicting old layers
 *   3. prefetchLayer() called for next layer(s) to overlap with compute
 *   4. releaseLayer() hints that a layer is no longer needed this iteration
 *
 * Memory Management:
 *   - Weights remain in pinned host memory (zero-copy source)
 *   - GPU cache holds recently-used layers within memory budget
 *   - Eviction policy determines which layers to evict when cache is full
 *
 * @see docs/v2/OPTION_B_WEIGHT_STREAMING_DESIGN.md for full design details
 */

#pragma once

#include "../backends/DeviceId.h"
#include "../execution/mpi_orchestration/PlacementStrategy.h" // For InferencePhase

#include <cstddef>
#include <cstdint>
#include <chrono>

namespace llaminar2
{

    // =========================================================================
    // Enumerations
    // =========================================================================

    /**
     * @brief Weight residency mode for a layer or model
     *
     * Determines how weights are managed between host and device memory.
     */
    enum class WeightResidencyMode
    {
        RESIDENT,  ///< Weights permanently resident on device (fits in VRAM)
        STREAMING, ///< Weights streamed on-demand from host (exceeds VRAM)
        UNIFIED    ///< Using unified/managed memory (let driver handle placement)
    };

    /**
     * @brief Eviction policy for the streaming cache
     *
     * When GPU memory is full and a new layer needs to be loaded,
     * this policy determines which cached layer to evict.
     */
    enum class StreamingEvictionPolicy
    {
        LRU,  ///< Least Recently Used - evict layer accessed longest ago
        FIFO, ///< First In First Out - evict oldest loaded layer
        NONE  ///< No eviction - fail if cache is full (for debugging)
    };

    // =========================================================================
    // Configuration Structures
    // =========================================================================

    /**
     * @brief Configuration for the weight streaming subsystem
     *
     * This struct contains all parameters needed to configure weight streaming
     * behavior. Reasonable defaults are provided for typical use cases.
     */
    struct StreamingConfig
    {
        // === Memory Budget ===

        /// Maximum GPU memory to use for weight cache (bytes)
        /// 0 = auto-detect based on available VRAM
        size_t gpu_memory_budget = 0;

        /// Number of layers to keep cached beyond the current layer
        /// Higher values reduce eviction churn but use more memory
        size_t cache_layer_count = 2;

        /// Minimum free GPU memory to maintain (bytes)
        /// Prevents OOM by leaving headroom for activations/KV cache
        size_t min_free_memory = 512 * 1024 * 1024; // 512 MB default

        // === Prefetching ===

        /// Enable async prefetching of upcoming layers
        bool enable_prefetch = true;

        /// Number of layers to prefetch ahead during decode
        size_t prefetch_depth = 1;

        /// Use separate CUDA stream for prefetch transfers
        bool use_prefetch_stream = true;

        // === Eviction ===

        /// Policy for evicting cached layers when memory is full
        StreamingEvictionPolicy eviction_policy = StreamingEvictionPolicy::LRU;

        // === Transfer Optimization ===

        /// Use pinned (page-locked) host memory for transfers
        /// Required for async transfers and optimal bandwidth
        bool use_pinned_memory = true;

        /// Overlap layer computation with next layer transfer
        bool enable_compute_transfer_overlap = true;

        // === Phase-Specific Behavior ===

        /// During PREFILL, preload all layers that fit in cache
        bool prefill_preload_enabled = true;

        /// During DECODE, use more aggressive prefetching
        bool decode_aggressive_prefetch = true;

        // === Debugging ===

        /// Log transfer timing and bandwidth statistics
        bool log_transfer_stats = false;

        /// Verify transferred data integrity (slow, for debugging)
        bool verify_transfers = false;
    };

    // =========================================================================
    // Statistics / Diagnostics
    // =========================================================================

    /**
     * @brief Runtime statistics for weight streaming operations
     *
     * Used for performance monitoring, debugging, and auto-tuning.
     * Reset between benchmark runs or inference sessions.
     */
    struct StreamingStats
    {
        // === Transfer Counts ===
        uint64_t layers_transferred = 0; ///< Total H2D layer transfers
        uint64_t layers_evicted = 0;     ///< Total layers evicted from cache
        uint64_t cache_hits = 0;         ///< Layer found in cache
        uint64_t cache_misses = 0;       ///< Layer not in cache, transfer needed
        uint64_t prefetch_hits = 0;      ///< Prefetched layer was used
        uint64_t prefetch_wasted = 0;    ///< Prefetched layer evicted before use

        // === Data Volume ===
        uint64_t bytes_transferred = 0; ///< Total bytes transferred H2D
        uint64_t bytes_evicted = 0;     ///< Total bytes evicted from cache

        // === Timing ===
        std::chrono::nanoseconds total_transfer_time{0};  ///< Cumulative transfer time
        std::chrono::nanoseconds total_wait_time{0};      ///< Time spent waiting for transfers
        std::chrono::nanoseconds max_transfer_latency{0}; ///< Worst-case single transfer

        // === Cache State ===
        size_t current_cache_size = 0;    ///< Current bytes in GPU cache
        size_t peak_cache_size = 0;       ///< Maximum bytes ever in cache
        size_t current_layers_cached = 0; ///< Number of layers currently cached

        // === Derived Metrics (computed on request) ===

        /// Cache hit rate (0.0 - 1.0)
        double hitRate() const
        {
            uint64_t total = cache_hits + cache_misses;
            return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
        }

        /// Average transfer bandwidth (GB/s)
        double averageBandwidthGBps() const
        {
            auto ns = total_transfer_time.count();
            if (ns == 0 || bytes_transferred == 0)
                return 0.0;
            double seconds = ns / 1e9;
            double gb = bytes_transferred / (1024.0 * 1024.0 * 1024.0);
            return gb / seconds;
        }

        /// Prefetch effectiveness (0.0 - 1.0)
        double prefetchEffectiveness() const
        {
            uint64_t total = prefetch_hits + prefetch_wasted;
            return total > 0 ? static_cast<double>(prefetch_hits) / total : 0.0;
        }
    };

    // =========================================================================
    // IWeightStreamer Interface
    // =========================================================================

    /**
     * @brief Interface for streaming weights between host and device memory
     *
     * The weight streamer manages a GPU-side cache of layer weights, transferring
     * them on-demand from pinned host memory. It supports async prefetching to
     * hide transfer latency and configurable eviction policies.
     *
     * Thread Safety:
     * - ensureLayerOnDevice() and prefetchLayer() may be called concurrently
     * - Statistics methods are thread-safe (atomic or snapshotted)
     * - clearCache() and resetStats() should not be called during inference
     *
     * Ownership:
     * - The streamer does NOT own the source weights (they remain in WeightManager)
     * - The streamer OWNS the GPU-side cached copies
     * - Callers must not modify weights while transfers are in progress
     */
    class IWeightStreamer
    {
    public:
        virtual ~IWeightStreamer() = default;

        // =====================================================================
        // Layer Management
        // =====================================================================

        /**
         * @brief Ensure a layer's weights are available on the specified device
         *
         * This is the primary synchronous interface. If the layer is already
         * cached, returns immediately. Otherwise, initiates a transfer and
         * blocks until complete. May evict other layers if cache is full.
         *
         * @param layer_idx Zero-based layer index
         * @param device Target device (must be a GPU for streaming mode)
         * @return true if layer is now available, false on error
         *
         * @note For CPU device, always returns true (weights are in host memory)
         * @note May block waiting for eviction or transfer completion
         */
        virtual bool ensureLayerOnDevice(int layer_idx, DeviceId device) = 0;

        /**
         * @brief Initiate async prefetch of a layer's weights
         *
         * Starts transferring the layer in the background without blocking.
         * Useful for overlapping transfer with compute on the current layer.
         * Has no effect if the layer is already cached or being transferred.
         *
         * @param layer_idx Zero-based layer index
         * @param device Target device for prefetch
         *
         * @note Call synchronize() or ensureLayerOnDevice() to wait for completion
         * @note Prefetch may be silently skipped if cache is full and cannot evict
         */
        virtual void prefetchLayer(int layer_idx, DeviceId device) = 0;

        /**
         * @brief Hint that a layer is no longer needed this iteration
         *
         * Marks the layer as eligible for eviction. Does not immediately free
         * the memory - the layer remains cached but can be evicted if space
         * is needed. Useful for LRU policy to understand access patterns.
         *
         * @param layer_idx Zero-based layer index
         *
         * @note Layer may still be in cache after this call
         * @note Calling ensureLayerOnDevice() on a released layer is valid
         */
        virtual void releaseLayer(int layer_idx) = 0;

        // =====================================================================
        // Phase Management
        // =====================================================================

        /**
         * @brief Notify the streamer of an inference phase transition
         *
         * Phase transitions trigger different streaming strategies:
         * - PREFILL → DECODE: May preload decode-critical layers
         * - DECODE → PREFILL: May clear cache for new prompt
         *
         * @param old_phase Previous inference phase
         * @param new_phase New inference phase
         */
        virtual void onPhaseTransition(InferencePhase old_phase,
                                       InferencePhase new_phase) = 0;

        // =====================================================================
        // Memory Management
        // =====================================================================

        /**
         * @brief Get current GPU memory usage by the weight cache
         *
         * @return Bytes currently allocated on GPU for cached weights
         */
        virtual size_t currentDeviceMemoryUsage() const = 0;

        /**
         * @brief Get the configured memory budget
         *
         * @return Maximum bytes allowed for weight cache
         */
        virtual size_t memoryBudget() const = 0;

        /**
         * @brief Force eviction of a specific layer from cache
         *
         * Immediately frees the GPU memory for this layer. The layer
         * will need to be re-transferred on next access.
         *
         * @param layer_idx Zero-based layer index
         * @return true if layer was evicted, false if not in cache
         */
        virtual bool evictLayer(int layer_idx) = 0;

        /**
         * @brief Clear all cached layers from GPU memory
         *
         * Frees all GPU-side weight storage. Typically called between
         * inference sessions or when switching models.
         *
         * @note Waits for any pending transfers to complete first
         */
        virtual void clearCache() = 0;

        // =====================================================================
        // Synchronization
        // =====================================================================

        /**
         * @brief Wait for all pending transfers to complete
         *
         * Blocks until all async prefetches and transfers are finished.
         * Call this before reading from potentially-prefetched layers.
         */
        virtual void synchronize() = 0;

        // =====================================================================
        // Diagnostics
        // =====================================================================

        /**
         * @brief Check if a layer is currently cached on device
         *
         * @param layer_idx Zero-based layer index
         * @param device Device to check
         * @return true if layer is cached and ready, false otherwise
         */
        virtual bool isLayerCached(int layer_idx, DeviceId device) const = 0;

        /**
         * @brief Check if a layer prefetch is in progress
         *
         * @param layer_idx Zero-based layer index
         * @return true if async transfer is in flight for this layer
         */
        virtual bool isPrefetchInProgress(int layer_idx) const = 0;

        /**
         * @brief Get current streaming statistics
         *
         * Returns a snapshot of the current statistics. The returned
         * struct is a copy, safe to use without synchronization.
         *
         * @return Copy of current StreamingStats
         */
        virtual StreamingStats stats() const = 0;

        /**
         * @brief Reset all statistics to zero
         *
         * Typically called before a benchmark run to get clean metrics.
         * Does not affect cache state or configuration.
         */
        virtual void resetStats() = 0;
    };

    // =========================================================================
    // Helper Functions
    // =========================================================================

    // Forward declaration - implementation in StreamingConfigFromEnv.h
    // (Avoids circular dependency with DebugEnv.h)
    StreamingConfig createStreamingConfigFromEnv();

} // namespace llaminar2
