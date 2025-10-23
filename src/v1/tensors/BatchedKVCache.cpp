/**
 * @file BatchedKVCache.cpp
 * @brief Implementation of batched Key-Value cache
 * @author David Sanftenberg
 * @date October 15, 2025
 */

#include "BatchedKVCache.h"
#include "../Logger.h"
#include <algorithm>
#include <sstream>

namespace llaminar {

BatchedKVCache::BatchedKVCache(size_t num_layers, size_t batch_size,
                               size_t max_seq_len, size_t hidden_dim)
    : num_layers_(num_layers)
    , batch_size_(batch_size)
    , max_seq_len_(max_seq_len)
    , hidden_dim_(hidden_dim)
{
    // Initialize storage structures
    k_cache_.resize(num_layers);
    v_cache_.resize(num_layers);
    seq_lengths_.resize(num_layers);
    
    for (size_t layer = 0; layer < num_layers; ++layer) {
        k_cache_[layer].resize(batch_size, nullptr);
        v_cache_[layer].resize(batch_size, nullptr);
        seq_lengths_[layer].resize(batch_size, 0);
    }
    
    LOG_DEBUG("[BatchedKVCache] Initialized: layers=" << num_layers 
              << ", batch=" << batch_size 
              << ", max_seq=" << max_seq_len
              << ", hidden=" << hidden_dim);
}

void BatchedKVCache::validate_indices(size_t layer, size_t batch_idx) const {
    if (layer >= num_layers_) {
        std::ostringstream oss;
        oss << "Layer index " << layer << " out of range [0, " << num_layers_ << ")";
        throw std::out_of_range(oss.str());
    }
    if (batch_idx >= batch_size_) {
        std::ostringstream oss;
        oss << "Batch index " << batch_idx << " out of range [0, " << batch_size_ << ")";
        throw std::out_of_range(oss.str());
    }
}

std::shared_ptr<TensorBase> BatchedKVCache::get_k(size_t layer, size_t batch_idx) const {
    validate_indices(layer, batch_idx);
    return k_cache_[layer][batch_idx];
}

std::shared_ptr<TensorBase> BatchedKVCache::get_v(size_t layer, size_t batch_idx) const {
    validate_indices(layer, batch_idx);
    return v_cache_[layer][batch_idx];
}

std::shared_ptr<TensorBase> BatchedKVCache::concatenate_seq_dim(
    const std::shared_ptr<TensorBase>& existing,
    const std::shared_ptr<TensorBase>& new_data) const 
{
    if (!existing) {
        // First time: just return new data
        return new_data->copy();
    }
    
    // Verify shapes are compatible (same num_heads and head_dim)
    const auto& existing_shape = existing->shape();
    const auto& new_shape = new_data->shape();
    
    if (existing_shape.size() != 3 || new_shape.size() != 3) {
        throw std::invalid_argument("KV cache tensors must be 3D [num_heads, seq, head_dim]");
    }
    
    if (existing_shape[0] != new_shape[0] || existing_shape[2] != new_shape[2]) {
        std::ostringstream oss;
        oss << "Incompatible KV shapes: existing=[" << existing_shape[0] << "," 
            << existing_shape[1] << "," << existing_shape[2] << "], new=["
            << new_shape[0] << "," << new_shape[1] << "," << new_shape[2] << "]";
        throw std::invalid_argument(oss.str());
    }
    
    int num_heads = existing_shape[0];
    int old_seq = existing_shape[1];
    int new_seq = new_shape[1];
    int head_dim = existing_shape[2];
    int total_seq = old_seq + new_seq;
    
    // Create concatenated tensor
    std::vector<int> concat_shape = {num_heads, total_seq, head_dim};
    auto result = std::make_shared<SimpleTensor>(concat_shape);
    
    // Copy data: for each head, concatenate along sequence dimension
    const float* existing_data = existing->data();
    const float* new_data_ptr = new_data->data();
    float* result_data = result->data();
    
    for (int h = 0; h < num_heads; ++h) {
        // Copy existing sequence
        size_t existing_offset = h * old_seq * head_dim;
        size_t result_offset = h * total_seq * head_dim;
        std::copy(existing_data + existing_offset,
                  existing_data + existing_offset + old_seq * head_dim,
                  result_data + result_offset);
        
        // Append new sequence
        size_t new_offset = h * new_seq * head_dim;
        size_t append_offset = result_offset + old_seq * head_dim;
        std::copy(new_data_ptr + new_offset,
                  new_data_ptr + new_offset + new_seq * head_dim,
                  result_data + append_offset);
    }
    
    return result;
}

void BatchedKVCache::append_kv(size_t layer, size_t batch_idx,
                               const std::shared_ptr<TensorBase>& new_k,
                               const std::shared_ptr<TensorBase>& new_v) {
    validate_indices(layer, batch_idx);
    
    if (!new_k || !new_v) {
        throw std::invalid_argument("Cannot append null K/V tensors");
    }
    
    // Check max sequence length
    size_t current_len = seq_lengths_[layer][batch_idx];
    size_t new_len = new_k->shape()[1];  // sequence dimension
    
    if (current_len + new_len > max_seq_len_) {
        std::ostringstream oss;
        oss << "Sequence length " << (current_len + new_len) 
            << " exceeds maximum " << max_seq_len_;
        throw std::runtime_error(oss.str());
    }
    
    // Concatenate
    k_cache_[layer][batch_idx] = concatenate_seq_dim(k_cache_[layer][batch_idx], new_k);
    v_cache_[layer][batch_idx] = concatenate_seq_dim(v_cache_[layer][batch_idx], new_v);
    
    // Update length
    seq_lengths_[layer][batch_idx] = current_len + new_len;
    
    LOG_TRACE("[BatchedKVCache] Appended KV: layer=" << layer 
              << ", batch=" << batch_idx 
              << ", new_seq=" << new_len
              << ", total_seq=" << seq_lengths_[layer][batch_idx]);
}

void BatchedKVCache::set_kv(size_t layer, size_t batch_idx,
                            const std::shared_ptr<TensorBase>& k,
                            const std::shared_ptr<TensorBase>& v) {
    validate_indices(layer, batch_idx);
    
    k_cache_[layer][batch_idx] = k;
    v_cache_[layer][batch_idx] = v;
    
    if (k && k->shape().size() >= 2) {
        seq_lengths_[layer][batch_idx] = k->shape()[1];  // sequence dimension
    } else {
        seq_lengths_[layer][batch_idx] = 0;
    }
}

void BatchedKVCache::reset_batch(size_t batch_idx) {
    if (batch_idx >= batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        k_cache_[layer][batch_idx] = nullptr;
        v_cache_[layer][batch_idx] = nullptr;
        seq_lengths_[layer][batch_idx] = 0;
    }
    
    LOG_DEBUG("[BatchedKVCache] Reset batch " << batch_idx);
}

void BatchedKVCache::clear_all() {
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        for (size_t batch = 0; batch < batch_size_; ++batch) {
            k_cache_[layer][batch] = nullptr;
            v_cache_[layer][batch] = nullptr;
            seq_lengths_[layer][batch] = 0;
        }
    }
    
    LOG_DEBUG("[BatchedKVCache] Cleared all caches");
}

size_t BatchedKVCache::sequence_length(size_t layer, size_t batch_idx) const {
    validate_indices(layer, batch_idx);
    return seq_lengths_[layer][batch_idx];
}

bool BatchedKVCache::is_empty(size_t layer, size_t batch_idx) const {
    validate_indices(layer, batch_idx);
    return k_cache_[layer][batch_idx] == nullptr;
}

} // namespace llaminar
