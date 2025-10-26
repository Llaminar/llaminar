/**
 * @file BatchedKVCache.h
 * @brief Batched Key-Value cache for multi-sequence autoregressive decode
 * @author David Sanftenberg
 * @date October 26, 2025
 *
 * Phase 1: Batched KV cache infrastructure
 * Stores K/V tensors per layer per sequence for batched inference.
 */

#pragma once

#include "Tensors.h"
#include <vector>
#include <memory>

namespace llaminar2
{

    /**
     * @brief Per-sequence KV cache entry (within a layer)
     */
    struct BatchedKVCacheEntry
    {
        std::shared_ptr<FP32Tensor> K; // [max_seq_len, n_kv_heads * head_dim]
        std::shared_ptr<FP32Tensor> V; // [max_seq_len, n_kv_heads * head_dim]
        int cached_tokens = 0;         // Number of tokens currently cached for this sequence
    };

    /**
     * @brief Batched KV cache for multi-sequence autoregressive decode
     *
     * Stores K/V tensors for each layer and each sequence to enable batched inference:
     * - Prefill: Store K/V for all prompt tokens across all sequences
     * - Decode: Append K/V for new token per sequence, use full cached context
     *
     * Design:
     * - Pre-allocates K/V buffers up to max_seq_len per sequence
     * - Supports efficient append without reallocation
     * - Per-sequence tracking for variable-length sequences
     * - Per-layer device placement (follows attention computation device)
     */
    class BatchedKVCache
    {
    public:
        /**
         * @brief Construct batched KV cache with uniform device placement
         *
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences in batch
         * @param max_seq_len Maximum sequence length (cache capacity per sequence)
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param device_idx Default device for all layers (-1 = CPU)
         */
        BatchedKVCache(int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int head_dim, int device_idx = -1);

        /**
         * @brief Construct batched KV cache with per-layer attention device placement
         *
         * The KV cache is stored on the device where **attention computation** occurs
         * (i.e., where wq, wk, wv, wo weights reside). For heterogeneous execution
         * or MoE models, this may differ from where FFN or expert weights are placed.
         *
         * Example use cases:
         * - Standard heterogeneous: Layers 0-11 attention on CPU, 12-23 on GPU
         * - MoE with shared experts: Attention on CPU, experts on GPU → use CPU
         *
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences in batch
         * @param max_seq_len Maximum sequence length (cache capacity per sequence)
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param attention_devices Device where attention is computed per layer
         *                          (length = n_layers, -1 = CPU, ≥0 = GPU device)
         */
        BatchedKVCache(int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int head_dim,
                       const std::vector<int> &attention_devices);

        /**
         * @brief Get K cache for specific layer and sequence
         *
         * Returns a view into the cached data up to cached_tokens.
         *
         * @param layer Layer index (0-based)
         * @param seq_idx Sequence index (0-based)
         * @return K tensor [cached_tokens, n_kv_heads * head_dim], or nullptr if empty
         */
        std::shared_ptr<FP32Tensor> get_k(int layer, int seq_idx) const;

        /**
         * @brief Get V cache for specific layer and sequence
         *
         * @param layer Layer index (0-based)
         * @param seq_idx Sequence index (0-based)
         * @return V tensor [cached_tokens, n_kv_heads * head_dim], or nullptr if empty
         */
        std::shared_ptr<FP32Tensor> get_v(int layer, int seq_idx) const;

        /**
         * @brief Get number of cached tokens for a specific sequence in a layer
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @return Number of tokens currently cached
         */
        int get_cached_tokens(int layer, int seq_idx) const;

        /**
         * @brief Append new K/V to cache for specific sequence
         *
         * Copies new K/V data into pre-allocated cache buffers.
         * Automatically tracks cached_tokens count per sequence.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param new_k New K tensor [new_seq_len, n_kv_heads * head_dim]
         * @param new_v New V tensor [new_seq_len, n_kv_heads * head_dim]
         * @return true on success, false if capacity exceeded
         */
        bool append_kv(int layer, int seq_idx, const FP32Tensor *new_k, const FP32Tensor *new_v);

        /**
         * @brief Clear cache for all layers and sequences (reset to empty)
         */
        void clear();

        /**
         * @brief Clear cache for specific sequence across all layers
         *
         * Useful for continuous batching when a sequence completes.
         *
         * @param seq_idx Sequence index
         */
        void clear_sequence(int seq_idx);

        /**
         * @brief Clear cache for specific layer across all sequences
         *
         * @param layer Layer index
         */
        void clear_layer(int layer);

        /**
         * @brief Get total number of layers
         */
        int num_layers() const { return n_layers_; }

        /**
         * @brief Get batch size
         */
        int batch_size() const { return batch_size_; }

        /**
         * @brief Get maximum sequence length (cache capacity)
         */
        int max_seq_len() const { return max_seq_len_; }

        /**
         * @brief Get device index for a specific layer's cache
         *
         * Returns the device where attention computation occurs for this layer.
         *
         * @param layer Layer index
         * @return Device index (-1 = CPU, ≥0 = GPU device)
         */
        int get_layer_device(int layer) const
        {
            if (layer >= 0 && layer < n_layers_)
            {
                return layer_devices_[layer];
            }
            return -1; // Default to CPU
        }

    private:
        int n_layers_;
        int batch_size_;
        int max_seq_len_;
        int kv_dim_; // n_kv_heads * head_dim

        // Cache storage: [n_layers][batch_size]
        std::vector<std::vector<BatchedKVCacheEntry>> entries_;

        // Device placement per layer
        std::vector<int> layer_devices_;

        // Helper: Initialize cache entries for a layer
        void initialize_layer(int layer, int device_idx);
    };

} // namespace llaminar2
