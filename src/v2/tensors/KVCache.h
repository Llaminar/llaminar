/**
 * @file KVCache.h
 * @brief Simple Key-Value cache for autoregressive decode (single sequence)
 * @author David Sanftenberg
 * @date October 25, 2025
 *
 * Phase 3d: KV cache for incremental decode
 * Stores K/V tensors per layer to avoid recomputing past context.
 */

#pragma once

#include "Tensors.h"
#include "TensorFactory.h"
#include "../utils/MPIContext.h"
#include <vector>
#include <memory>

namespace llaminar2
{

    /**
     * @brief Per-layer KV cache entry
     */
    struct KVCacheEntry
    {
        std::shared_ptr<FP32Tensor> K; // [past_seq_len, n_kv_heads * head_dim]
        std::shared_ptr<FP32Tensor> V; // [past_seq_len, n_kv_heads * head_dim]
        int cached_tokens = 0;         // Number of tokens currently cached
        int device_idx = -1;           // Device where this layer's cache resides
    };

    /**
     * @brief Simple KV cache for single-sequence autoregressive decode
     *
     * Stores K/V tensors for each layer to enable incremental decode:
     * - Prefill: Store K/V for all prompt tokens
     * - Decode: Append K/V for new token, use full cached context
     *
     * Design:
     * - Pre-allocates K/V buffers up to max_seq_len
     * - Supports efficient append without reallocation
     * - Thread-safe per-layer access (no concurrent layer access needed)
     */
    class KVCache
    {
    public:
        /**
         * @brief Construct KV cache for all layers
         *
         * @param mpi_ctx MPI context for NUMA-aware allocation
         * @param n_layers Number of transformer layers
         * @param max_seq_len Maximum sequence length (cache capacity)
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param device_idx Default device for all layers (-1 = CPU)
         */
        KVCache(const MPIContext &mpi_ctx, int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_idx = -1);

        /**
         * @brief Construct KV cache with per-layer attention device placement
         *
         * The KV cache is stored on the device where **attention computation** occurs
         * (i.e., where wq, wk, wv, wo weights reside). For heterogeneous execution
         * or MoE models, this may differ from where FFN or expert weights are placed.
         *
         * Example use cases:
         * - Standard heterogeneous: Layers 0-11 attention on CPU, 12-23 on GPU
         * - MoE with shared experts: Attention on CPU, experts on GPU → use CPU
         *
         * @param mpi_ctx MPI context for NUMA-aware allocation
         * @param n_layers Number of transformer layers
         * @param max_seq_len Maximum sequence length (cache capacity)
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param attention_devices Device where attention is computed per layer
         *                          (length = n_layers, -1 = CPU, ≥0 = GPU device)
         */
        KVCache(const MPIContext &mpi_ctx, int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
                const std::vector<int> &attention_devices);

        /**
         * @brief Get K cache for specific layer
         *
         * @param layer Layer index (0-based)
         * @return K tensor [cached_tokens, n_kv_heads * head_dim], or nullptr if empty
         */
        std::shared_ptr<FP32Tensor> get_k(int layer) const;

        /**
         * @brief Get V cache for specific layer
         *
         * @param layer Layer index (0-based)
         * @return V tensor [cached_tokens, n_kv_heads * head_dim], or nullptr if empty
         */
        std::shared_ptr<FP32Tensor> get_v(int layer) const;

        /**
         * @brief Get number of cached tokens for a layer
         *
         * @param layer Layer index
         * @return Number of tokens currently cached
         */
        int get_cached_tokens(int layer) const;

        /**
         * @brief Append new K/V to cache (used during prefill or decode)
         *
         * Copies new K/V data into pre-allocated cache buffers.
         * Automatically tracks cached_tokens count.
         *
         * @param layer Layer index
         * @param new_k New K tensor [new_seq_len, n_kv_heads * head_dim]
         * @param new_v New V tensor [new_seq_len, n_kv_heads * head_dim]
         * @return true on success, false if capacity exceeded
         */
        bool append_kv(int layer, const FP32Tensor *new_k, const FP32Tensor *new_v);

        /**
         * @brief Clear cache for all layers (reset to empty)
         */
        void clear();

        /**
         * @brief Clear cache for specific layer
         *
         * @param layer Layer index
         */
        void clear_layer(int layer);

        /**
         * @brief Evict oldest tokens from cache to make room for new tokens
         *
         * Shifts all cached K/V data to remove the oldest tokens, preserving
         * the most recent context. This implements a sliding window approach.
         *
         * @param tokens_to_evict Number of oldest tokens to remove
         */
        void evict_oldest(int tokens_to_evict);

        /**
         * @brief Get total number of layers
         */
        int num_layers() const { return n_layers_; }

        /**
         * @brief Get maximum sequence length (cache capacity)
         */
        int max_seq_len() const { return max_seq_len_; }

        /**
         * @brief Get device index for a specific layer's cache
         *
         * Returns the device where attention computation occurs for this layer.
         * For MoE models, this is the attention device, not necessarily where
         * FFN or expert weights reside.
         *
         * @param layer Layer index
         * @return Device index (-1 = CPU, ≥0 = GPU device)
         */
        int get_layer_device(int layer) const
        {
            if (layer >= 0 && layer < n_layers_)
            {
                return cache_[layer].device_idx;
            }
            return -1;
        }

        /**
         * @brief Get attention device for a specific layer
         *
         * Semantic alias for get_layer_device() that clarifies this returns
         * where attention computation happens, not where entire layer lives.
         *
         * @param layer Layer index
         * @return Device index where attention is computed (-1 = CPU, ≥0 = GPU)
         */
        int get_attention_device(int layer) const
        {
            return get_layer_device(layer);
        }

        /**
         * @brief Get total number of tokens evicted since last clear
         *
         * This counter accumulates whenever evict_oldest() is called.
         * The pipeline should use this to adjust position tracking after
         * sliding window eviction.
         *
         * @return Total tokens evicted since last clear()
         */
        int get_total_evicted() const { return total_evicted_; }

        /**
         * @brief Reset the eviction counter
         *
         * Called when the pipeline has processed the eviction and adjusted
         * its position tracking accordingly.
         */
        void reset_eviction_counter() { total_evicted_ = 0; }

    private:
        int n_layers_;
        int max_seq_len_;
        int n_kv_heads_;
        int head_dim_;
        int device_idx_;
        int total_evicted_ = 0; // Tracks tokens evicted for position adjustment

        std::vector<KVCacheEntry> cache_; // One entry per layer
    };

} // namespace llaminar2
