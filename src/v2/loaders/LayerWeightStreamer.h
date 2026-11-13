/**
 * @file LayerWeightStreamer.h
 * @brief Weight streaming implementation for Option B (STREAMING mode)
 * @author David Sanftenberg
 * @date January 2026
 *
 * This file implements the LayerWeightStreamer class that streams layer weights
 * from CPU to GPU on demand. This is the primary streaming implementation for
 * models that exceed GPU VRAM capacity.
 *
 * Design Philosophy:
 * - Layer-granular streaming (all weights for a layer as a unit)
 * - LRU eviction when GPU memory budget is exceeded
 * - Async prefetching support (deferred to future implementation)
 * - Thread-safe statistics tracking
 * - Zero-copy from pinned host memory (when available)
 *
 * Memory Model:
 * - WeightManager owns CPU-resident weights (source of truth)
 * - LayerWeightStreamer owns GPU-side cached copies
 * - Cache is bounded by configurable memory budget
 * - LRU eviction frees oldest-accessed layers when budget exceeded
 *
 * Usage:
 *   auto weight_mgr = std::make_shared<WeightManager>(loader, mpi_ctx);
 *   StreamingConfig config;
 *   config.gpu_memory_budget = 4ULL * 1024 * 1024 * 1024; // 4GB
 *
 *   auto streamer = std::make_unique<LayerWeightStreamer>(weight_mgr, config);
 *
 *   // Before executing layer 0:
 *   streamer->ensureLayerOnDevice(0, DeviceId::cuda(0));
 *
 *   // Prefetch next layer while current executes:
 *   streamer->prefetchLayer(1, DeviceId::cuda(0));
 *
 * @see IWeightStreamer.h for the interface definition
 * @see NullWeightStreamer.h for RESIDENT mode (no streaming)
 * @see docs/v2/OPTION_B_WEIGHT_STREAMING_DESIGN.md for design details
 */

#pragma once

#include "IWeightStreamer.h"
#include "WeightManager.h"

#include <chrono>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Weight streamer that transfers layer weights from CPU to GPU on demand
     *
     * This implementation is used when model weights exceed GPU VRAM capacity.
     * It maintains a GPU-side cache of recently-used layers and uses LRU eviction
     * to stay within the configured memory budget.
     *
     * Thread Safety:
     * - ensureLayerOnDevice() and prefetchLayer() are thread-safe
     * - Statistics methods use internal locking
     * - clearCache() should not be called during inference
     *
     * Performance Characteristics:
     * - First access to a layer incurs H2D transfer latency (~ms for large layers)
     * - Subsequent accesses are cache hits (sub-microsecond)
     * - LRU eviction is O(1) using list + map iterators
     * - Prefetching overlaps transfer with compute (when async enabled)
     */
    class LayerWeightStreamer : public IWeightStreamer
    {
    public:
        /**
         * @brief Construct a layer weight streamer
         *
         * @param weight_manager Source of CPU-resident weights (must outlive streamer)
         * @param num_layers Total number of transformer layers in the model
         * @param config Streaming configuration (memory budget, eviction policy, etc.)
         */
        LayerWeightStreamer(
            std::shared_ptr<WeightManager> weight_manager,
            int num_layers,
            StreamingConfig config = StreamingConfig{});

        ~LayerWeightStreamer() override;

        // Non-copyable, non-movable (owns GPU resources)
        LayerWeightStreamer(const LayerWeightStreamer &) = delete;
        LayerWeightStreamer &operator=(const LayerWeightStreamer &) = delete;
        LayerWeightStreamer(LayerWeightStreamer &&) = delete;
        LayerWeightStreamer &operator=(LayerWeightStreamer &&) = delete;

        // =====================================================================
        // Layer Management
        // =====================================================================

        /**
         * @brief Ensure a layer's weights are available on the specified device
         *
         * If the layer is cached, returns immediately. Otherwise, loads the layer
         * from CPU to GPU, evicting LRU layers if necessary to stay within budget.
         *
         * @param layer_idx Zero-based layer index
         * @param device Target device (must be a GPU for streaming mode)
         * @return true if layer is now available, false on error
         */
        bool ensureLayerOnDevice(int layer_idx, DeviceId device) override;

        /**
         * @brief Initiate async prefetch of a layer's weights
         *
         * Currently implemented as synchronous load (async deferred).
         * Has no effect if layer is already cached or being loaded.
         *
         * @param layer_idx Zero-based layer index
         * @param device Target device for prefetch
         */
        void prefetchLayer(int layer_idx, DeviceId device) override;

        /**
         * @brief Hint that a layer is no longer needed this iteration
         *
         * Moves the layer to the end of LRU list, making it eligible for
         * early eviction. Layer remains cached until evicted.
         *
         * @param layer_idx Zero-based layer index
         */
        void releaseLayer(int layer_idx) override;

        // =====================================================================
        // Phase Management
        // =====================================================================

        /**
         * @brief Notify the streamer of an inference phase transition
         *
         * Phase transitions may trigger cache warming or clearing strategies.
         *
         * @param old_phase Previous inference phase
         * @param new_phase New inference phase
         */
        void onPhaseTransition(InferencePhase old_phase,
                               InferencePhase new_phase) override;

        // =====================================================================
        // Memory Management
        // =====================================================================

        /**
         * @brief Get current GPU memory usage by the weight cache
         *
         * @return Bytes currently allocated on GPU for cached weights
         */
        size_t currentDeviceMemoryUsage() const override;

        /**
         * @brief Get the configured memory budget
         *
         * @return Maximum bytes allowed for weight cache
         */
        size_t memoryBudget() const override;

        /**
         * @brief Force eviction of a specific layer from cache
         *
         * @param layer_idx Zero-based layer index
         * @return true if layer was evicted, false if not in cache
         */
        bool evictLayer(int layer_idx) override;

        /**
         * @brief Clear all cached layers from GPU memory
         */
        void clearCache() override;

        // =====================================================================
        // Synchronization
        // =====================================================================

        /**
         * @brief Wait for all pending transfers to complete
         */
        void synchronize() override;

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
        bool isLayerCached(int layer_idx, DeviceId device) const override;

        /**
         * @brief Check if a layer prefetch is in progress
         *
         * @param layer_idx Zero-based layer index
         * @return true if async transfer is in flight for this layer
         */
        bool isPrefetchInProgress(int layer_idx) const override;

        /**
         * @brief Get current streaming statistics
         *
         * @return Copy of current StreamingStats
         */
        StreamingStats stats() const override;

        /**
         * @brief Reset all statistics to zero
         */
        void resetStats() override;

    private:
        // =====================================================================
        // Internal Types
        // =====================================================================

        /**
         * @brief Cached layer data on GPU
         */
        struct CachedLayer
        {
            /// GPU-resident weight tensors for this layer
            std::vector<std::shared_ptr<TensorBase>> weights;

            /// Device where weights are cached
            DeviceId device;

            /// Time of last access (for stats/debugging)
            std::chrono::steady_clock::time_point last_access;

            /// Total memory footprint of this layer on GPU
            size_t memory_bytes = 0;

            /// Was this layer loaded via prefetch?
            bool from_prefetch = false;
        };

        // =====================================================================
        // Internal Methods
        // =====================================================================

        /**
         * @brief Get the weight tensor names for a layer
         *
         * Returns GGUF tensor names for all weights in the specified layer.
         * Uses Qwen2-style naming: "blk.{layer_idx}.{component}.weight"
         *
         * @param layer_idx Zero-based layer index
         * @return Vector of weight tensor names
         */
        std::vector<std::string> getLayerWeightNames(int layer_idx) const;

        /**
         * @brief Inspect the layer's original native payload extent without loading it.
         *
         * @param layer_idx Zero-based layer index
         * @return Native source bytes used by the streaming cache policy, not a PMA claim.
         */
        size_t estimateLayerMemory(int layer_idx) const;

        /**
         * @brief Load a layer's weights to GPU
         *
         * Fetches weights from WeightManager, uploads to GPU, updates cache.
         *
         * @param layer_idx Zero-based layer index
         * @param device Target GPU device
         * @return true on success, false on error
         */
        bool loadLayerToDevice(int layer_idx, DeviceId device);

        /**
         * @brief Evict layers to free memory using configured policy
         *
         * @param bytes_needed Minimum bytes to free
         * @return true if sufficient memory was freed
         */
        bool evictToFreeMemory(size_t bytes_needed);

        /**
         * @brief Evict the least recently used layer
         *
         * @return Bytes freed, or 0 if cache is empty
         */
        size_t evictLRU();

        /**
         * @brief Update LRU tracking for a layer (move to front)
         *
         * @param layer_idx Layer that was just accessed
         */
        void touchLayer(int layer_idx);

        /**
         * @brief Remove layer from LRU tracking
         *
         * @param layer_idx Layer to remove
         */
        void removeLRUEntry(int layer_idx);

        // =====================================================================
        // Data Members
        // =====================================================================

        /// Source of CPU-resident weights
        std::shared_ptr<WeightManager> weight_manager_;

        /// Total number of layers in the model
        int num_layers_;

        /// Streaming configuration
        StreamingConfig config_;

        /// Cache of GPU-resident layer weights (layer_idx -> cached data)
        std::unordered_map<int, CachedLayer> layer_cache_;

        /// Pending async prefetch operations (layer_idx -> future)
        /// Note: Currently unused (sync implementation), reserved for async
        std::unordered_map<int, std::future<bool>> pending_prefetches_;

        /// LRU order tracking: front = most recently used, back = least recently used
        std::list<int> lru_order_;

        /// Map from layer_idx to position in lru_order_ for O(1) updates
        std::unordered_map<int, std::list<int>::iterator> lru_map_;

        /// Current total GPU memory usage by cache
        size_t current_memory_usage_ = 0;

        /// Mutex for cache operations (ensures thread safety)
        mutable std::mutex cache_mutex_;

        /// Statistics tracking
        mutable std::mutex stats_mutex_;
        StreamingStats stats_;

        /// Current inference phase
        InferencePhase current_phase_ = InferencePhase::PREFILL;
    };

} // namespace llaminar2
