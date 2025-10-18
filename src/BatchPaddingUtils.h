/**
 * @file BatchPaddingUtils.h
 * @brief Utilities for padding and batching variable-length sequences
 * @author David Sanftenberg
 * 
 * Provides utilities for creating batched tensors from variable-length token sequences,
 * including padding to uniform length and tracking of actual sequence lengths.
 */

#pragma once

#include "tensors/TensorBase.h"
#include "tensors/SimpleTensor.h"
#include <vector>
#include <memory>
#include <algorithm>

namespace llaminar
{
namespace batch
{

/**
 * @brief Result of padding variable-length sequences to uniform batch
 */
struct PaddedBatch
{
    std::shared_ptr<TensorBase> tokens;         ///< Padded token tensor [batch, max_len]
    std::vector<int> actual_lengths;             ///< Actual length of each sequence [batch]
    std::vector<int> padding_mask;               ///< Flat mask: 1=real token, 0=padding [batch * max_len]
    size_t max_length;                           ///< Maximum sequence length in batch
    size_t batch_size;                           ///< Number of sequences in batch
    
    PaddedBatch() : max_length(0), batch_size(0) {}
    
    /**
     * @brief Check if a specific position is padding
     * @param batch_idx Batch index
     * @param position Token position within sequence
     * @return true if position is padding, false if real token
     */
    bool is_padding(size_t batch_idx, size_t position) const
    {
        if (batch_idx >= batch_size || position >= max_length)
        {
            return true;
        }
        return padding_mask[batch_idx * max_length + position] == 0;
    }
};

/**
 * @brief Create padded batch from variable-length token sequences
 * 
 * Pads all sequences to the length of the longest sequence, tracks actual
 * lengths, and creates a padding mask for use in attention.
 * 
 * @param token_sequences Vector of token ID sequences (variable lengths)
 * @param pad_token_id Token ID to use for padding (default: 0)
 * @return PaddedBatch containing padded tensors and metadata
 * 
 * Example:
 *   Input: [[1,2,3], [4,5], [6,7,8,9]]
 *   Output: 
 *     tokens: [[1,2,3,0], [4,5,0,0], [6,7,8,9]]
 *     actual_lengths: [3, 2, 4]
 *     padding_mask: [1,1,1,0, 1,1,0,0, 1,1,1,1]
 *     max_length: 4
 */
PaddedBatch createPaddedBatch(
    const std::vector<std::vector<int>>& token_sequences,
    int pad_token_id = 0);

/**
 * @brief Create padding mask tensor for attention
 * 
 * Creates a mask tensor that can be used in attention to exclude padding tokens.
 * Mask value is -INFINITY for padding positions, 0.0 for real tokens.
 * 
 * @param actual_lengths Actual sequence lengths [batch]
 * @param max_length Maximum sequence length
 * @return Tensor of shape [batch, max_length] with attention mask values
 */
std::shared_ptr<TensorBase> createAttentionPaddingMask(
    const std::vector<int>& actual_lengths,
    size_t max_length);

/**
 * @brief Group sequences by length into buckets to minimize padding waste
 * 
 * Organizes sequences into length buckets to reduce padding overhead.
 * Each bucket contains sequences with similar lengths.
 * 
 * @param token_sequences All token sequences to bucket
 * @param bucket_boundaries Length boundaries for buckets (e.g., [8, 16, 32, 64, 128, 256, 512])
 * @return Vector of PaddedBatch, one per bucket
 */
std::vector<PaddedBatch> bucketSequencesByLength(
    const std::vector<std::vector<int>>& token_sequences,
    const std::vector<size_t>& bucket_boundaries = {8, 16, 32, 64, 128, 256, 512, 1024, 2048});

} // namespace batch
} // namespace llaminar
