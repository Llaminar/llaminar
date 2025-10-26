/**
 * @file BatchPaddingUtils.cpp
 * @brief Implementation of batch padding utilities
 * @author David Sanftenberg
 * @date October 26, 2025
 */

#include "BatchPaddingUtils.h"
#include "Logger.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace llaminar2
{

    PaddedBatch createPaddedBatch(
        const std::vector<std::vector<int>> &token_sequences,
        int pad_token_id)
    {
        PaddedBatch result;

        // Handle empty input
        if (token_sequences.empty())
        {
            return result;
        }

        result.batch_size = token_sequences.size();

        // Find maximum sequence length
        result.max_length = 0;
        for (const auto &seq : token_sequences)
        {
            result.max_length = std::max(result.max_length, seq.size());
        }

        // Handle case where all sequences are empty
        if (result.max_length == 0)
        {
            return result;
        }

        // Create padded token tensor [batch_size, max_length]
        result.tokens = std::make_shared<FP32Tensor>(
            std::vector<size_t>{result.batch_size, result.max_length},
            -1); // CPU only for now

        float *token_data = result.tokens->mutable_data();

        // Reserve space for metadata
        result.actual_lengths.reserve(result.batch_size);
        result.padding_mask.reserve(result.batch_size * result.max_length);

        // Fill padded tensor and create mask
        for (size_t b = 0; b < result.batch_size; ++b)
        {
            const auto &seq = token_sequences[b];
            int actual_len = static_cast<int>(seq.size());
            result.actual_lengths.push_back(actual_len);

            for (size_t t = 0; t < result.max_length; ++t)
            {
                if (t < seq.size())
                {
                    // Real token
                    token_data[b * result.max_length + t] = static_cast<float>(seq[t]);
                    result.padding_mask.push_back(1);
                }
                else
                {
                    // Padding token
                    token_data[b * result.max_length + t] = static_cast<float>(pad_token_id);
                    result.padding_mask.push_back(0);
                }
            }
        }

        return result;
    }

    std::shared_ptr<FP32Tensor> createAttentionPaddingMask(
        const std::vector<int> &actual_lengths,
        size_t max_length)
    {
        size_t batch_size = actual_lengths.size();

        // Create mask tensor [batch_size, max_length]
        auto mask = std::make_shared<FP32Tensor>(
            std::vector<size_t>{batch_size, max_length},
            -1); // CPU only

        float *mask_data = mask->mutable_data();

        // Fill mask: 0.0 for real tokens, -INFINITY for padding
        for (size_t b = 0; b < batch_size; ++b)
        {
            int actual_len = actual_lengths[b];
            for (size_t t = 0; t < max_length; ++t)
            {
                if (static_cast<int>(t) < actual_len)
                {
                    // Real token: no masking
                    mask_data[b * max_length + t] = 0.0f;
                }
                else
                {
                    // Padding: mask with -INFINITY
                    mask_data[b * max_length + t] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        return mask;
    }

    std::vector<PaddedBatch> bucketSequencesByLength(
        const std::vector<std::vector<int>> &token_sequences,
        const std::vector<size_t> &bucket_boundaries)
    {
        std::vector<PaddedBatch> buckets;

        if (token_sequences.empty())
        {
            return buckets;
        }

        // Group sequences by bucket
        // bucket_map[bucket_idx] = vector of sequence indices
        std::vector<std::vector<size_t>> bucket_map;
        bucket_map.resize(bucket_boundaries.size());

        for (size_t i = 0; i < token_sequences.size(); ++i)
        {
            size_t seq_len = token_sequences[i].size();

            // Find appropriate bucket
            size_t bucket_idx = 0;
            for (size_t b = 0; b < bucket_boundaries.size(); ++b)
            {
                if (seq_len <= bucket_boundaries[b])
                {
                    bucket_idx = b;
                    break;
                }
                bucket_idx = b + 1; // Overflow to last bucket
            }

            // Clamp to valid range
            if (bucket_idx >= bucket_boundaries.size())
            {
                bucket_idx = bucket_boundaries.size() - 1;
            }

            bucket_map[bucket_idx].push_back(i);
        }

        // Create padded batch for each non-empty bucket
        for (size_t bucket_idx = 0; bucket_idx < bucket_map.size(); ++bucket_idx)
        {
            const auto &seq_indices = bucket_map[bucket_idx];
            if (seq_indices.empty())
            {
                continue; // Skip empty buckets
            }

            // Gather sequences for this bucket
            std::vector<std::vector<int>> bucket_sequences;
            bucket_sequences.reserve(seq_indices.size());
            for (size_t idx : seq_indices)
            {
                bucket_sequences.push_back(token_sequences[idx]);
            }

            // Create padded batch (will pad to longest sequence in bucket)
            PaddedBatch batch = createPaddedBatch(bucket_sequences, 0);

            // Pad up to bucket boundary if needed
            size_t boundary = bucket_boundaries[bucket_idx];
            if (batch.max_length < boundary)
            {
                // Reallocate with boundary padding
                auto new_tokens = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{batch.batch_size, boundary},
                    -1);

                float *new_data = new_tokens->mutable_data();
                const float *old_data = batch.tokens->data();

                // Copy existing data and pad
                for (size_t b = 0; b < batch.batch_size; ++b)
                {
                    for (size_t t = 0; t < boundary; ++t)
                    {
                        if (t < batch.max_length)
                        {
                            new_data[b * boundary + t] = old_data[b * batch.max_length + t];
                        }
                        else
                        {
                            new_data[b * boundary + t] = 0.0f; // pad token
                        }
                    }
                }

                // Update padding mask
                std::vector<int> new_mask;
                new_mask.reserve(batch.batch_size * boundary);
                for (size_t b = 0; b < batch.batch_size; ++b)
                {
                    int actual_len = batch.actual_lengths[b];
                    for (size_t t = 0; t < boundary; ++t)
                    {
                        new_mask.push_back(static_cast<int>(t) < actual_len ? 1 : 0);
                    }
                }

                batch.tokens = new_tokens;
                batch.max_length = boundary;
                batch.padding_mask = std::move(new_mask);
            }

            buckets.push_back(std::move(batch));
        }

        return buckets;
    }

} // namespace llaminar2
