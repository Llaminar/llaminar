/**
 * @file BatchedKVCache.h
 * @brief Batched Key-Value cache for parallel multi-sequence inference
 * @author David Sanftenberg
 * @date October 15, 2025
 * 
 * Part of Option A: Full Parallel Batching Implementation
 * Day 3-4: KV Cache Restructuring
 */

#pragma once

#include "TensorBase.h"
#include "SimpleTensor.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace llaminar {

/**
 * @brief Manages per-sequence Key-Value caches for batched transformer inference
 * 
 * This class provides isolated KV cache storage for each sequence in a batch,
 * enabling parallel processing while maintaining independent context for each
 * sequence. Critical for batched attention operations.
 * 
 * Structure: [num_layers][batch_size] where each element is a tensor of shape:
 * - K cache: [num_heads, current_seq_len, head_dim]
 * - V cache: [num_heads, current_seq_len, head_dim]
 * 
 * Example usage:
 * @code
 *   auto cache = std::make_shared<BatchedKVCache>(num_layers, batch_size, max_seq, hidden);
 *   
 *   // During prefill/decode:
 *   auto k = cache->get_k(layer_idx, batch_idx);
 *   auto v = cache->get_v(layer_idx, batch_idx);
 *   
 *   // Append new KV:
 *   cache->append_kv(layer_idx, batch_idx, new_k, new_v);
 * @endcode
 */
class BatchedKVCache {
public:
    /**
     * @brief Construct a batched KV cache
     * 
     * @param num_layers Number of transformer layers
     * @param batch_size Number of sequences in the batch
     * @param max_seq_len Maximum sequence length supported
     * @param hidden_dim Hidden dimension size (for K/V tensors)
     */
    BatchedKVCache(size_t num_layers, size_t batch_size, 
                   size_t max_seq_len, size_t hidden_dim);
    
    /**
     * @brief Get K cache for specific layer and batch index
     * 
     * @param layer Layer index (0-based)
     * @param batch_idx Batch index (0-based)
     * @return Shared pointer to K cache tensor, or nullptr if empty
     * 
     * @throws std::out_of_range if layer or batch_idx is invalid
     */
    std::shared_ptr<TensorBase> get_k(size_t layer, size_t batch_idx) const;
    
    /**
     * @brief Get V cache for specific layer and batch index
     * 
     * @param layer Layer index (0-based)
     * @param batch_idx Batch index (0-based)
     * @return Shared pointer to V cache tensor, or nullptr if empty
     * 
     * @throws std::out_of_range if layer or batch_idx is invalid
     */
    std::shared_ptr<TensorBase> get_v(size_t layer, size_t batch_idx) const;
    
    /**
     * @brief Append new K/V tensors to cache for a specific sequence
     * 
     * This concatenates the new K/V along the sequence dimension. Used during
     * both prefill (append all tokens) and decode (append single token).
     * 
     * @param layer Layer index
     * @param batch_idx Batch index
     * @param new_k New K tensor to append [num_heads, new_seq_len, head_dim]
     * @param new_v New V tensor to append [num_heads, new_seq_len, head_dim]
     * 
     * @throws std::out_of_range if indices invalid
     * @throws std::invalid_argument if shapes incompatible
     * @throws std::runtime_error if max sequence length exceeded
     */
    void append_kv(size_t layer, size_t batch_idx,
                   const std::shared_ptr<TensorBase>& new_k,
                   const std::shared_ptr<TensorBase>& new_v);
    
    /**
     * @brief Set K/V cache directly (replace existing)
     * 
     * Used for initialization or full replacement. Unlike append_kv, this
     * replaces the entire cache for the sequence rather than concatenating.
     * 
     * @param layer Layer index
     * @param batch_idx Batch index
     * @param k K cache tensor
     * @param v V cache tensor
     */
    void set_kv(size_t layer, size_t batch_idx,
                const std::shared_ptr<TensorBase>& k,
                const std::shared_ptr<TensorBase>& v);
    
    /**
     * @brief Reset cache for a specific sequence (all layers)
     * 
     * Clears the cache for one sequence across all layers. Used when starting
     * a new prompt for that sequence.
     * 
     * @param batch_idx Batch index to reset
     */
    void reset_batch(size_t batch_idx);
    
    /**
     * @brief Clear all caches (all layers, all sequences)
     */
    void clear_all();
    
    /**
     * @brief Get number of layers
     */
    size_t num_layers() const { return num_layers_; }
    
    /**
     * @brief Get batch size
     */
    size_t batch_size() const { return batch_size_; }
    
    /**
     * @brief Get current sequence length for a specific cache
     * 
     * @param layer Layer index
     * @param batch_idx Batch index
     * @return Current sequence length (number of cached tokens)
     */
    size_t sequence_length(size_t layer, size_t batch_idx) const;
    
    /**
     * @brief Get maximum sequence length supported
     */
    size_t max_sequence_length() const { return max_seq_len_; }
    
    /**
     * @brief Get hidden dimension
     */
    size_t hidden_dim() const { return hidden_dim_; }
    
    /**
     * @brief Check if cache is empty for a specific sequence
     * 
     * @param layer Layer index
     * @param batch_idx Batch index
     * @return true if no tokens cached yet
     */
    bool is_empty(size_t layer, size_t batch_idx) const;

private:
    // Storage: [num_layers][batch_size] -> TensorBase
    std::vector<std::vector<std::shared_ptr<TensorBase>>> k_cache_;
    std::vector<std::vector<std::shared_ptr<TensorBase>>> v_cache_;
    
    // Metadata: current sequence length per [layer][batch]
    std::vector<std::vector<size_t>> seq_lengths_;
    
    // Configuration
    size_t num_layers_;
    size_t batch_size_;
    size_t max_seq_len_;
    size_t hidden_dim_;
    
    /**
     * @brief Validate layer and batch indices
     * @throws std::out_of_range if invalid
     */
    void validate_indices(size_t layer, size_t batch_idx) const;
    
    /**
     * @brief Concatenate two tensors along sequence dimension (dim 1)
     * 
     * @param existing Existing tensor [num_heads, seq1, head_dim]
     * @param new_data New tensor [num_heads, seq2, head_dim]
     * @return Concatenated tensor [num_heads, seq1+seq2, head_dim]
     */
    std::shared_ptr<TensorBase> concatenate_seq_dim(
        const std::shared_ptr<TensorBase>& existing,
        const std::shared_ptr<TensorBase>& new_data) const;
};

} // namespace llaminar
