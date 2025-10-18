/**
 * @file BatchPaddingUtils.cpp
 * @brief Implementation of batch padding utilities
 * @author David Sanftenberg
 */

#include "BatchPaddingUtils.h"
#include "Logger.h"
#include <cmath>
#include <limits>

namespace llaminar
{
namespace batch
{

PaddedBatch createPaddedBatch(
    const std::vector<std::vector<int>>& token_sequences,
    int pad_token_id)
{
    PaddedBatch result;
    
    if (token_sequences.empty())
    {
        LOG_WARN("createPaddedBatch called with empty sequence vector");
        return result;
    }
    
    result.batch_size = token_sequences.size();
    
    // Find maximum sequence length
    result.max_length = 0;
    for (const auto& seq : token_sequences)
    {
        result.max_length = std::max(result.max_length, seq.size());
    }
    
    if (result.max_length == 0)
    {
        LOG_WARN("createPaddedBatch: all sequences are empty");
        return result;
    }
    
    // Create padded token tensor [batch, max_len]
    std::vector<int> shape = {static_cast<int>(result.batch_size), 
                              static_cast<int>(result.max_length)};
    auto padded_tensor = std::make_shared<SimpleTensor>(shape);
    
    // Fill with pad token
    std::fill(padded_tensor->data(), 
              padded_tensor->data() + padded_tensor->size(), 
              static_cast<float>(pad_token_id));
    
    // Copy actual tokens and create masks
    result.actual_lengths.resize(result.batch_size);
    result.padding_mask.resize(result.batch_size * result.max_length, 0);
    
    float* token_data = padded_tensor->data();
    
    for (size_t b = 0; b < result.batch_size; ++b)
    {
        const auto& seq = token_sequences[b];
        result.actual_lengths[b] = static_cast<int>(seq.size());
        
        // Copy actual tokens
        for (size_t t = 0; t < seq.size(); ++t)
        {
            token_data[b * result.max_length + t] = static_cast<float>(seq[t]);
            result.padding_mask[b * result.max_length + t] = 1;  // Real token
        }
        // Remaining positions already filled with pad_token_id and mask=0
    }
    
    result.tokens = padded_tensor;
    
    // Log padding efficiency
    size_t total_real_tokens = 0;
    for (int len : result.actual_lengths)
    {
        total_real_tokens += len;
    }
    size_t total_positions = result.batch_size * result.max_length;
    float efficiency = 100.0f * static_cast<float>(total_real_tokens) / total_positions;
    
    LOG_DEBUG("Padded batch created: batch_size=" << result.batch_size
              << " max_len=" << result.max_length
              << " real_tokens=" << total_real_tokens
              << " total_positions=" << total_positions
              << " efficiency=" << efficiency << "%");
    
    return result;
}

std::shared_ptr<TensorBase> createAttentionPaddingMask(
    const std::vector<int>& actual_lengths,
    size_t max_length)
{
    size_t batch_size = actual_lengths.size();
    
    // Create mask tensor [batch, max_length]
    std::vector<int> shape = {static_cast<int>(batch_size), static_cast<int>(max_length)};
    auto mask = std::make_shared<SimpleTensor>(shape);
    
    float* mask_data = mask->data();
    constexpr float MASK_VALUE = -std::numeric_limits<float>::infinity();
    
    for (size_t b = 0; b < batch_size; ++b)
    {
        int actual_len = actual_lengths[b];
        
        for (size_t t = 0; t < max_length; ++t)
        {
            // Real tokens get 0.0, padding gets -inf (will be zeroed by softmax)
            mask_data[b * max_length + t] = (t < static_cast<size_t>(actual_len)) ? 0.0f : MASK_VALUE;
        }
    }
    
    return mask;
}

std::vector<PaddedBatch> bucketSequencesByLength(
    const std::vector<std::vector<int>>& token_sequences,
    const std::vector<size_t>& bucket_boundaries)
{
    if (token_sequences.empty())
    {
        return {};
    }
    
    // Create buckets: one for each boundary range + one for overflow
    size_t num_buckets = bucket_boundaries.size() + 1;
    std::vector<std::vector<std::vector<int>>> buckets(num_buckets);
    
    // Assign sequences to buckets based on length
    for (const auto& seq : token_sequences)
    {
        size_t len = seq.size();
        size_t bucket_idx = num_buckets - 1;  // Default to last bucket (overflow)
        
        // Find appropriate bucket
        for (size_t i = 0; i < bucket_boundaries.size(); ++i)
        {
            if (len <= bucket_boundaries[i])
            {
                bucket_idx = i;
                break;
            }
        }
        
        buckets[bucket_idx].push_back(seq);
    }
    
    // Create padded batches for non-empty buckets
    std::vector<PaddedBatch> result;
    
    for (size_t i = 0; i < buckets.size(); ++i)
    {
        if (!buckets[i].empty())
        {
            auto padded = createPaddedBatch(buckets[i]);
            
            std::string boundary_str;
            if (i == 0)
            {
                boundary_str = "≤" + std::to_string(bucket_boundaries[0]);
            }
            else if (i < bucket_boundaries.size())
            {
                boundary_str = std::to_string(bucket_boundaries[i-1]) + "-" + 
                              std::to_string(bucket_boundaries[i]);
            }
            else
            {
                boundary_str = ">" + std::to_string(bucket_boundaries.back());
            }
            
            LOG_INFO("Bucket [" << boundary_str << "]: " << buckets[i].size() 
                     << " sequences, max_len=" << padded.max_length);
            
            result.push_back(std::move(padded));
        }
    }
    
    return result;
}

} // namespace batch
} // namespace llaminar
