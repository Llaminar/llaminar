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
        PrefillChunkSchedulerPolicy policy;
        policy.bucket_sizes = bucket_sizes;
        policy.real_token_start = 0;
        policy.real_token_count = total_real_tokens;

        auto schedule = planPrefillChunkSchedule(policy);
        if (!schedule)
            return {};
        return schedule.chunks;
    }

    PrefillChunkSchedule planPrefillChunkSchedule(
        const PrefillChunkSchedulerPolicy &policy)
    {
        PrefillChunkSchedule schedule;

        if (policy.real_token_start < 0)
        {
            schedule.error = "real_token_start must be non-negative";
            return schedule;
        }
        if (policy.real_token_count <= 0)
        {
            schedule.error = "real_token_count must be positive";
            return schedule;
        }
        if (policy.fixed_chunk_real_tokens < 0)
        {
            schedule.error = "fixed_chunk_real_tokens must be non-negative";
            return schedule;
        }
        if (policy.min_rebalance_interval_tokens < 0 ||
            policy.max_rebalance_interval_tokens < 0)
        {
            schedule.error = "rebalance intervals must be non-negative";
            return schedule;
        }
        if (policy.min_rebalance_interval_tokens > 0 &&
            policy.max_rebalance_interval_tokens > 0 &&
            policy.min_rebalance_interval_tokens > policy.max_rebalance_interval_tokens)
        {
            schedule.error = "min rebalance interval exceeds max rebalance interval";
            return schedule;
        }

        const std::vector<int> buckets = normalizePrefillGraphBuckets(policy.bucket_sizes);
        if (buckets.empty())
        {
            schedule.error = "no positive prefill graph buckets configured";
            return schedule;
        }

        const int max_bucket = buckets.back();
        const int chunk_target =
            policy.fixed_chunk_real_tokens > 0 ? policy.fixed_chunk_real_tokens : max_bucket;
        if (chunk_target <= 0 || chunk_target > max_bucket)
        {
            schedule.error = "fixed chunk interval exceeds largest prefill graph bucket";
            return schedule;
        }

        int local_offset = 0;
        int chunk_index = 0;
        int real_tokens_since_required_boundary = 0;
        while (local_offset < policy.real_token_count)
        {
            const int remaining = policy.real_token_count - local_offset;
            const int real_count = std::min(remaining, chunk_target);
            auto selected = selectPrefillGraphBucket(real_count, buckets);
            if (!selected)
            {
                schedule.error = selected.error;
                schedule.chunks.clear();
                return schedule;
            }

            real_tokens_since_required_boundary += real_count;
            const bool rebalance_allowed =
                policy.min_rebalance_interval_tokens > 0 &&
                real_tokens_since_required_boundary >= policy.min_rebalance_interval_tokens;
            const bool rebalance_required =
                policy.max_rebalance_interval_tokens > 0 &&
                real_tokens_since_required_boundary >= policy.max_rebalance_interval_tokens;

            schedule.chunks.push_back(PrefillChunkPlan{
                policy.real_token_start + local_offset,
                real_count,
                selected.bucket_seq_len,
                chunk_index,
                rebalance_allowed || rebalance_required,
                rebalance_required});

            if (rebalance_required)
                real_tokens_since_required_boundary = 0;

            local_offset += real_count;
            ++chunk_index;
        }

        schedule.ok = true;
        return schedule;
    }

    PrefillChunkMaintenanceDecision evaluatePrefillChunkMaintenance(
        const PrefillChunkPlan &chunk,
        const PrefillChunkMaintenanceState &state)
    {
        PrefillChunkMaintenanceDecision decision;
        decision.required = chunk.rebalance_required_after;

        if (chunk.chunk_index != state.chunk_index)
        {
            decision.reason = "chunk_index_mismatch";
            return decision;
        }
        if (!state.histograms_merged)
        {
            decision.reason = "histograms_not_merged";
            return decision;
        }
        if (!state.manual_boundaries_complete)
        {
            decision.reason = "manual_boundary_incomplete";
            return decision;
        }
        if (state.graph_capture_active)
        {
            decision.reason = "graph_capture_active";
            return decision;
        }
        if (state.graph_replay_active)
        {
            decision.reason = "graph_replay_active";
            return decision;
        }
        if (!state.participants_at_same_boundary)
        {
            decision.reason = "participants_not_at_same_boundary";
            return decision;
        }
        if (!chunk.rebalance_allowed_after && !chunk.rebalance_required_after)
        {
            decision.reason = "rebalance_interval_not_ready";
            return decision;
        }
        if (!state.rebalance_requested && !chunk.rebalance_required_after)
        {
            decision.ok = true;
            decision.reason = "rebalance_not_requested";
            return decision;
        }

        decision.ok = true;
        decision.can_run = true;
        decision.reason = chunk.rebalance_required_after ? "required" : "ready";
        return decision;
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
