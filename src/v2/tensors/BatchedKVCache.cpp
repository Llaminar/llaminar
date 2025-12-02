/**
 * @file BatchedKVCache.cpp
 * @brief Implementation of batched KV cache
 * @author David Sanftenberg
 * @date October 26, 2025
 */

#include "BatchedKVCache.h"
#include "../utils/Logger.h"
#include <cstring>

namespace llaminar2
{

    BatchedKVCache::BatchedKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                                   int n_kv_heads, int head_dim, int device_idx)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          kv_dim_(n_kv_heads * head_dim)
    {
        // Create TensorFactory for NUMA-aware allocation
        tensor_factory_ = std::make_unique<TensorFactory>(mpi_ctx);

        // Uniform device placement
        layer_devices_.resize(n_layers_, device_idx);

        // Initialize cache entries
        entries_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            initialize_layer(layer, device_idx);
        }
    }

    BatchedKVCache::BatchedKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                                   int n_kv_heads, int head_dim,
                                   const std::vector<int> &attention_devices)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          kv_dim_(n_kv_heads * head_dim)
    {
        // Create TensorFactory for NUMA-aware allocation
        tensor_factory_ = std::make_unique<TensorFactory>(mpi_ctx);

        // Per-layer device placement
        if (static_cast<int>(attention_devices.size()) != n_layers_)
        {
            LOG_ERROR("BatchedKVCache: attention_devices size mismatch (expected "
                      << n_layers_ << ", got " << attention_devices.size() << ")");
            // Fallback to CPU
            layer_devices_.resize(n_layers_, -1);
        }
        else
        {
            layer_devices_ = attention_devices;
        }

        // Initialize cache entries
        entries_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            initialize_layer(layer, layer_devices_[layer]);
        }
    }

    void BatchedKVCache::initialize_layer(int layer, int device_idx)
    {
        entries_[layer].resize(batch_size_);

        for (int seq = 0; seq < batch_size_; ++seq)
        {
            auto &entry = entries_[layer][seq];

            // Use TensorFactory for NUMA-aware allocation with device tracking
            entry.K = tensor_factory_->createFP32(
                std::vector<size_t>{static_cast<size_t>(max_seq_len_), static_cast<size_t>(kv_dim_)}, device_idx);

            entry.V = tensor_factory_->createFP32(
                std::vector<size_t>{static_cast<size_t>(max_seq_len_), static_cast<size_t>(kv_dim_)}, device_idx);

            entry.cached_tokens = 0;
        }
    }

    std::shared_ptr<FP32Tensor> BatchedKVCache::get_k(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("BatchedKVCache::get_k: invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return nullptr;
        }

        const auto &entry = entries_[layer][seq_idx];
        if (entry.cached_tokens == 0)
        {
            return nullptr; // No cached data
        }

        // Create a view into the cached portion [0:cached_tokens, :]
        // For simplicity, we'll create a new tensor and copy (TODO: optimize with views)
        auto k_view = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(entry.cached_tokens), static_cast<size_t>(kv_dim_)},
            entry.K->device_index());

        const float *src = entry.K->data();
        float *dst = k_view->mutable_data();
        std::memcpy(dst, src, entry.cached_tokens * kv_dim_ * sizeof(float));

        return k_view;
    }

    std::shared_ptr<FP32Tensor> BatchedKVCache::get_v(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("BatchedKVCache::get_v: invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return nullptr;
        }

        const auto &entry = entries_[layer][seq_idx];
        if (entry.cached_tokens == 0)
        {
            return nullptr; // No cached data
        }

        // Create a view into the cached portion
        auto v_view = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(entry.cached_tokens), static_cast<size_t>(kv_dim_)},
            entry.V->device_index());

        const float *src = entry.V->data();
        float *dst = v_view->mutable_data();
        std::memcpy(dst, src, entry.cached_tokens * kv_dim_ * sizeof(float));

        return v_view;
    }

    int BatchedKVCache::get_cached_tokens(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].cached_tokens;
    }

    bool BatchedKVCache::append_kv(int layer, int seq_idx, const FP32Tensor *new_k, const FP32Tensor *new_v)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("BatchedKVCache::append_kv: invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        if (!new_k || !new_v)
        {
            LOG_ERROR("BatchedKVCache::append_kv: null K/V tensors");
            return false;
        }

        auto &entry = entries_[layer][seq_idx];

        const auto &k_shape = new_k->shape();
        const auto &v_shape = new_v->shape();

        if (k_shape.size() != 2 || v_shape.size() != 2)
        {
            LOG_ERROR("BatchedKVCache::append_kv: K/V must be 2D tensors");
            return false;
        }

        int new_tokens = static_cast<int>(k_shape[0]);
        if (static_cast<int>(v_shape[0]) != new_tokens)
        {
            LOG_ERROR("BatchedKVCache::append_kv: K/V shape mismatch");
            return false;
        }

        if (static_cast<int>(k_shape[1]) != kv_dim_ || static_cast<int>(v_shape[1]) != kv_dim_)
        {
            LOG_ERROR("BatchedKVCache::append_kv: K/V dim mismatch (expected " << kv_dim_
                                                                               << ", got K=" << k_shape[1] << " V=" << v_shape[1] << ")");
            return false;
        }

        // Check capacity
        if (entry.cached_tokens + new_tokens > max_seq_len_)
        {
            LOG_ERROR("BatchedKVCache::append_kv: capacity exceeded (cached="
                      << entry.cached_tokens << " + new=" << new_tokens
                      << " > max=" << max_seq_len_ << ")");
            return false;
        }

        // Append new K/V to cache
        float *k_cache = entry.K->mutable_data();
        float *v_cache = entry.V->mutable_data();

        const float *new_k_data = new_k->data();
        const float *new_v_data = new_v->data();

        // Copy at offset cached_tokens
        std::memcpy(k_cache + entry.cached_tokens * kv_dim_, new_k_data, new_tokens * kv_dim_ * sizeof(float));
        std::memcpy(v_cache + entry.cached_tokens * kv_dim_, new_v_data, new_tokens * kv_dim_ * sizeof(float));

        // Update cached token count
        entry.cached_tokens += new_tokens;

        return true;
    }

    void BatchedKVCache::clear()
    {
#pragma omp parallel for collapse(2)
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                entries_[layer][seq].cached_tokens = 0;
            }
        }
    }

    void BatchedKVCache::clear_sequence(int seq_idx)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("BatchedKVCache::clear_sequence: invalid seq_idx=" << seq_idx);
            return;
        }

#pragma omp parallel for
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            entries_[layer][seq_idx].cached_tokens = 0;
        }
    }

    void BatchedKVCache::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            LOG_ERROR("BatchedKVCache::clear_layer: invalid layer=" << layer);
            return;
        }

#pragma omp parallel for
        for (int seq = 0; seq < batch_size_; ++seq)
        {
            entries_[layer][seq].cached_tokens = 0;
        }
    }

    void BatchedKVCache::evict_oldest(int tokens_to_evict)
    {
        if (tokens_to_evict <= 0)
        {
            return;
        }

        // Parallelize across layers and sequences
#pragma omp parallel for collapse(2)
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                auto &entry = entries_[layer][seq];
                int current_cached = entry.cached_tokens;

                if (tokens_to_evict >= current_cached)
                {
                    // Evict all tokens from this sequence
                    entry.cached_tokens = 0;
                    continue;
                }

                // Shift remaining tokens to the beginning
                int tokens_to_keep = current_cached - tokens_to_evict;
                float *k_cache = entry.K->mutable_data();
                float *v_cache = entry.V->mutable_data();

                size_t shift_offset = tokens_to_evict * kv_dim_;
                size_t keep_size = tokens_to_keep * kv_dim_ * sizeof(float);

                // Use memmove for overlapping regions
                std::memmove(k_cache, k_cache + shift_offset, keep_size);
                std::memmove(v_cache, v_cache + shift_offset, keep_size);

                entry.cached_tokens = tokens_to_keep;
            }
        }

        LOG_DEBUG("[BatchedKVCache] Evicted " << tokens_to_evict << " oldest tokens from all sequences");
    }

    void BatchedKVCache::evict_oldest_from_sequence(int seq_idx, int tokens_to_evict)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("BatchedKVCache::evict_oldest_from_sequence: invalid seq_idx=" << seq_idx);
            return;
        }

        if (tokens_to_evict <= 0)
        {
            return;
        }

        // Parallelize across layers for this sequence
#pragma omp parallel for
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            auto &entry = entries_[layer][seq_idx];
            int current_cached = entry.cached_tokens;

            if (tokens_to_evict >= current_cached)
            {
                // Evict all tokens
                entry.cached_tokens = 0;
                continue;
            }

            // Shift remaining tokens to the beginning
            int tokens_to_keep = current_cached - tokens_to_evict;
            float *k_cache = entry.K->mutable_data();
            float *v_cache = entry.V->mutable_data();

            size_t shift_offset = tokens_to_evict * kv_dim_;
            size_t keep_size = tokens_to_keep * kv_dim_ * sizeof(float);

            // Use memmove for overlapping regions
            std::memmove(k_cache, k_cache + shift_offset, keep_size);
            std::memmove(v_cache, v_cache + shift_offset, keep_size);

            entry.cached_tokens = tokens_to_keep;
        }

        LOG_DEBUG("[BatchedKVCache] Evicted " << tokens_to_evict << " oldest tokens from sequence " << seq_idx);
    }

} // namespace llaminar2
