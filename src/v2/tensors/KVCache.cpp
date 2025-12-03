/**
 * @file KVCache.cpp
 * @brief KV cache implementation
 * @author David Sanftenberg
 */

#include "KVCache.h"
#include "../utils/Logger.h"
#include <cstring>
#include <map>

namespace llaminar2
{

    KVCache::KVCache(const MPIContext &mpi_ctx, int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_idx)
        : n_layers_(n_layers), max_seq_len_(max_seq_len), n_kv_heads_(n_kv_heads), head_dim_(head_dim), device_idx_(device_idx)
    {
        cache_.resize(n_layers);

        // Create TensorFactory with MPI context for NUMA-aware allocation
        TensorFactory factory(mpi_ctx);

        // Pre-allocate K/V buffers for all layers (single device)
        size_t kv_dim = n_kv_heads * head_dim;
        for (int i = 0; i < n_layers; ++i)
        {
            // Use TensorFactory for NUMA-aware allocation with device tracking
            cache_[i].K = factory.createFP32(
                std::vector<size_t>{static_cast<size_t>(max_seq_len), kv_dim}, device_idx);
            cache_[i].V = factory.createFP32(
                std::vector<size_t>{static_cast<size_t>(max_seq_len), kv_dim}, device_idx);
            cache_[i].cached_tokens = 0;
            cache_[i].device_idx = device_idx; // Store device affinity
        }

        LOG_INFO("[KVCache] Allocated cache for " << n_layers << " layers, max_seq_len=" << max_seq_len
                                                  << ", kv_dim=" << kv_dim << " on device " << device_idx
                                                  << " (" << (2 * n_layers * max_seq_len * kv_dim * sizeof(float) / 1024 / 1024) << " MB)");
    }

    // Heterogeneous constructor with per-layer device placement
    KVCache::KVCache(const MPIContext &mpi_ctx, int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
                     const std::vector<int> &layer_devices)
        : n_layers_(n_layers), max_seq_len_(max_seq_len), n_kv_heads_(n_kv_heads), head_dim_(head_dim), device_idx_(-2) // -2 = heterogeneous
    {
        if (static_cast<int>(layer_devices.size()) != n_layers)
        {
            LOG_ERROR("[KVCache] layer_devices size (" << layer_devices.size() << ") must match n_layers (" << n_layers << ")");
            throw std::invalid_argument("layer_devices size mismatch");
        }

        cache_.resize(n_layers);
        size_t kv_dim = n_kv_heads * head_dim;

        // Create TensorFactory with MPI context for NUMA-aware allocation
        TensorFactory factory(mpi_ctx);

        // Track memory usage per device
        std::map<int, size_t> device_bytes;

        // Pre-allocate K/V buffers for each layer on its assigned device
        for (int i = 0; i < n_layers; ++i)
        {
            int layer_device = layer_devices[i];
            // Use TensorFactory for NUMA-aware allocation with device tracking
            cache_[i].K = factory.createFP32(
                std::vector<size_t>{static_cast<size_t>(max_seq_len), kv_dim}, layer_device);
            cache_[i].V = factory.createFP32(
                std::vector<size_t>{static_cast<size_t>(max_seq_len), kv_dim}, layer_device);
            cache_[i].cached_tokens = 0;
            cache_[i].device_idx = layer_device;

            // Track memory per device
            size_t layer_bytes = 2 * max_seq_len * kv_dim * sizeof(float);
            device_bytes[layer_device] += layer_bytes;
        }

        // Report heterogeneous memory allocation
        LOG_INFO("[KVCache] Heterogeneous allocation for " << n_layers << " layers:");
        for (const auto &[device, bytes] : device_bytes)
        {
            double mb = bytes / (1024.0 * 1024.0);
            std::string device_name = (device == -1) ? "CPU" : "GPU " + std::to_string(device);
            LOG_INFO("  " << device_name << ": " << mb << " MB");
        }
    }

    std::shared_ptr<FP32Tensor> KVCache::get_k(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            LOG_ERROR("[KVCache] Invalid layer index: " << layer);
            return nullptr;
        }

        if (cache_[layer].cached_tokens == 0)
        {
            return nullptr; // Empty cache
        }

        // Return view of cached portion [0:cached_tokens, :]
        // For simplicity, we return the full tensor and rely on caller to use cached_tokens
        return cache_[layer].K;
    }

    std::shared_ptr<FP32Tensor> KVCache::get_v(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            LOG_ERROR("[KVCache] Invalid layer index: " << layer);
            return nullptr;
        }

        if (cache_[layer].cached_tokens == 0)
        {
            return nullptr; // Empty cache
        }

        return cache_[layer].V;
    }

    int KVCache::get_cached_tokens(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return 0;
        }
        return cache_[layer].cached_tokens;
    }

    bool KVCache::append_kv(int layer, const FP32Tensor *new_k, const FP32Tensor *new_v)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            LOG_ERROR("[KVCache] Invalid layer index: " << layer);
            return false;
        }

        if (!new_k || !new_v)
        {
            LOG_ERROR("[KVCache] Null K or V tensor");
            return false;
        }

        const auto &new_k_shape = new_k->shape();
        const auto &new_v_shape = new_v->shape();

        if (new_k_shape.size() != 2 || new_v_shape.size() != 2)
        {
            LOG_ERROR("[KVCache] K/V must be 2D tensors");
            return false;
        }

        int new_tokens = static_cast<int>(new_k_shape[0]);
        int kv_dim = static_cast<int>(new_k_shape[1]);

        if (new_v_shape[0] != new_k_shape[0] || new_v_shape[1] != new_k_shape[1])
        {
            LOG_ERROR("[KVCache] K/V shape mismatch");
            return false;
        }

        int current_cached = cache_[layer].cached_tokens;

        // Auto-evict oldest tokens if cache would overflow
        if (current_cached + new_tokens > max_seq_len_)
        {
            int tokens_to_evict = (current_cached + new_tokens) - max_seq_len_;
            // Only evict on layer 0 (evict_oldest handles all layers atomically)
            if (layer == 0)
            {
                // Warn on first eviction - sliding window is now active
                if (total_evicted_ == 0)
                {
                    LOG_DEBUG("[KVCache] Cache capacity reached (" << max_seq_len_ << " tokens). "
                                                                  << "Starting sliding window eviction. Context from oldest tokens will be lost. "
                                                                  << "Consider using a model with larger context or chunked prompts.");
                }
                LOG_TRACE("[KVCache] Cache full, evicting " << tokens_to_evict << " oldest tokens (sliding window)");
                evict_oldest(tokens_to_evict);
            }
            current_cached = cache_[layer].cached_tokens; // Re-read after eviction
        }

        // Copy new K/V into cache buffers at offset
        float *k_cache = cache_[layer].K->mutable_data();
        float *v_cache = cache_[layer].V->mutable_data();
        const float *new_k_data = new_k->data();
        const float *new_v_data = new_v->data();

        size_t offset = current_cached * kv_dim;
        size_t copy_size = new_tokens * kv_dim * sizeof(float);

        std::memcpy(k_cache + offset, new_k_data, copy_size);
        std::memcpy(v_cache + offset, new_v_data, copy_size);

        cache_[layer].cached_tokens += new_tokens;

        LOG_TRACE("[KVCache] Layer " << layer << " appended " << new_tokens << " tokens (total: " << cache_[layer].cached_tokens << ")");

        return true;
    }

    void KVCache::evict_oldest(int tokens_to_evict)
    {
        if (tokens_to_evict <= 0)
        {
            return;
        }

        size_t kv_dim = n_kv_heads_ * head_dim_;

        // Track evicted tokens for position adjustment
        total_evicted_ += tokens_to_evict;

#pragma omp parallel for
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            int current_cached = cache_[layer].cached_tokens;
            if (tokens_to_evict >= current_cached)
            {
                // Evict all tokens
                cache_[layer].cached_tokens = 0;
                continue;
            }

            // Shift remaining tokens to the beginning
            int tokens_to_keep = current_cached - tokens_to_evict;
            float *k_cache = cache_[layer].K->mutable_data();
            float *v_cache = cache_[layer].V->mutable_data();

            size_t shift_offset = tokens_to_evict * kv_dim;
            size_t keep_size = tokens_to_keep * kv_dim * sizeof(float);

            // Use memmove for overlapping regions
            std::memmove(k_cache, k_cache + shift_offset, keep_size);
            std::memmove(v_cache, v_cache + shift_offset, keep_size);

            cache_[layer].cached_tokens = tokens_to_keep;
        }

        LOG_TRACE("[KVCache] Evicted " << tokens_to_evict << " oldest tokens from all layers (total_evicted_=" << total_evicted_ << ")");
    }

    void KVCache::clear()
    {
        for (int i = 0; i < n_layers_; ++i)
        {
            cache_[i].cached_tokens = 0;
        }
        total_evicted_ = 0; // Reset eviction counter on full clear
        LOG_DEBUG("[KVCache] Cleared all layers");
    }

    void KVCache::clear_layer(int layer)
    {
        if (layer >= 0 && layer < n_layers_)
        {
            cache_[layer].cached_tokens = 0;
            LOG_DEBUG("[KVCache] Cleared layer " << layer);
        }
    }

} // namespace llaminar2
