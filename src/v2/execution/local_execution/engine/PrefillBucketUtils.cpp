/**
 * @file PrefillBucketUtils.cpp
 * @brief Implementation of host-side prefill bucket selection and chunk planning.
 *
 * These routines intentionally avoid reading environment variables or touching
 * executor state. Keeping them pure makes bucket decisions easy to test and
 * keeps fail-fast validation in the caller, where graph eligibility context is
 * available.
 */

#include "PrefillBucketUtils.h"

#include <algorithm>

namespace llaminar2
{

    std::vector<int> defaultPrefillGraphBuckets()
    {
        return {64, 128, 256, 384, 512, 544, 576, 608, 640, 672, 704, 736, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096};
    }

    std::vector<int> normalizePrefillGraphBuckets(const std::vector<int> &buckets)
    {
        std::vector<int> normalized;
        normalized.reserve(buckets.size());

        // Keep only positive boundaries; zero and negative values cannot be
        // execution lengths and usually come from malformed environment input.
        for (int bucket : buckets)
        {
            if (bucket > 0)
                normalized.push_back(bucket);
        }

        std::sort(normalized.begin(), normalized.end());
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
        return normalized;
    }

    PrefillBucketSelection selectPrefillGraphBucket(
        int real_seq_len,
        const std::vector<int> &bucket_sizes)
    {
        PrefillBucketSelection selection;
        selection.real_seq_len = real_seq_len;

        if (real_seq_len <= 0)
        {
            selection.error = "real_seq_len must be positive";
            return selection;
        }

        const std::vector<int> buckets = normalizePrefillGraphBuckets(bucket_sizes);
        if (buckets.empty())
        {
            selection.error = "no positive prefill graph buckets configured";
            return selection;
        }

        auto it = std::lower_bound(buckets.begin(), buckets.end(), real_seq_len);
        if (it == buckets.end())
        {
            selection.error = "real_seq_len exceeds largest prefill graph bucket";
            return selection;
        }

        selection.ok = true;
        selection.bucket_seq_len = *it;
        selection.exact = (selection.bucket_seq_len == real_seq_len);
        return selection;
    }

    std::vector<int> padPrefillTokensToBucket(
        const int *tokens,
        int real_seq_len,
        int bucket_seq_len,
        int pad_token_id)
    {
        if (!tokens || real_seq_len <= 0 || bucket_seq_len < real_seq_len)
            return {};

        std::vector<int> padded(static_cast<size_t>(bucket_seq_len), pad_token_id);
        std::copy(tokens, tokens + real_seq_len, padded.begin());
        return padded;
    }

    std::vector<PrefillChunkPlan> planPrefillChunks(
        int total_real_tokens,
        const std::vector<int> &bucket_sizes)
    {
        std::vector<PrefillChunkPlan> chunks;
        if (total_real_tokens <= 0)
            return chunks;

        const std::vector<int> buckets = normalizePrefillGraphBuckets(bucket_sizes);
        if (buckets.empty())
            return chunks;

        const int max_bucket = buckets.back();
        int offset = 0;
        while (offset < total_real_tokens)
        {
            const int remaining = total_real_tokens - offset;
            const int real_count = std::min(remaining, max_bucket);
            auto selected = selectPrefillGraphBucket(real_count, buckets);
            if (!selected)
                return {};

            chunks.push_back(PrefillChunkPlan{
                offset,
                real_count,
                selected.bucket_seq_len});
            offset += real_count;
        }

        return chunks;
    }

    std::vector<int> buildPrefillChunkPositionIds(
        int real_count,
        int bucket_seq_len,
        int token_offset,
        int batch_size)
    {
        if (real_count <= 0 || bucket_seq_len < real_count || token_offset < 0 || batch_size <= 0)
            return {};

        std::vector<int> position_ids(static_cast<size_t>(batch_size) * static_cast<size_t>(bucket_seq_len));
        for (int batch = 0; batch < batch_size; ++batch)
        {
            const size_t batch_offset = static_cast<size_t>(batch) * static_cast<size_t>(bucket_seq_len);
            for (int pos = 0; pos < bucket_seq_len; ++pos)
            {
                // Padding rows are intentionally initialized with monotonically
                // increasing absolute positions. The real-token row-select and
                // state gates decide later whether those rows can execute.
                position_ids[batch_offset + static_cast<size_t>(pos)] = token_offset + pos;
            }
        }

        return position_ids;
    }

    PrefillChunkExecutionInput buildPrefillChunkExecutionInput(
        const int *tokens,
        int total_real_tokens,
        const PrefillChunkPlan &chunk,
        int pad_token_id,
        int batch_size)
    {
        PrefillChunkExecutionInput input;
        input.token_offset = chunk.token_offset;
        input.real_count = chunk.real_count;
        input.bucket_seq_len = chunk.bucket_seq_len;

        if (!tokens)
        {
            input.error = "tokens must not be null";
            return input;
        }
        if (total_real_tokens <= 0)
        {
            input.error = "total_real_tokens must be positive";
            return input;
        }
        if (chunk.token_offset < 0 || chunk.real_count <= 0 || chunk.bucket_seq_len < chunk.real_count)
        {
            input.error = "invalid chunk shape";
            return input;
        }
        if (chunk.token_offset + chunk.real_count > total_real_tokens)
        {
            input.error = "chunk exceeds total token count";
            return input;
        }

        input.token_ids = padPrefillTokensToBucket(
            tokens + chunk.token_offset,
            chunk.real_count,
            chunk.bucket_seq_len,
            pad_token_id);
        input.position_ids = buildPrefillChunkPositionIds(
            chunk.real_count,
            chunk.bucket_seq_len,
            chunk.token_offset,
            batch_size);

        if (input.token_ids.empty() || input.position_ids.empty())
        {
            input.error = "failed to build chunk buffers";
            return input;
        }

        input.ok = true;
        return input;
    }

} // namespace llaminar2
