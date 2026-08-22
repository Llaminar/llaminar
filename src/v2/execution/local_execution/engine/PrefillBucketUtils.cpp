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

#include "utils/PrefillGraphBucketDefaults.h"

#include <algorithm>
#include <limits>

namespace llaminar2
{

    std::vector<int> defaultPrefillGraphBuckets()
    {
        return defaultPrefillGraphBucketSizes();
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

    std::vector<int> residentPrefillGraphRowCandidates(
        const std::vector<int> &buckets,
        int max_rows)
    {
        if (max_rows <= 0)
            return {};

        std::vector<int> candidates;
        for (const int bucket : normalizePrefillGraphBuckets(buckets))
        {
            if (bucket <= max_rows)
                candidates.push_back(bucket);
        }
        if (candidates.empty())
            candidates.push_back(max_rows);

        // Admission tries the throughput-favouring shape first and only
        // reduces residency when the complete physical BOM cannot cover it.
        std::reverse(candidates.begin(), candidates.end());
        return candidates;
    }

    std::vector<int> segmentedPrefillGraphRowCandidates(
        const std::vector<int> &buckets,
        int max_context_rows,
        int max_segment_rows)
    {
        if (max_context_rows <= 0 || max_segment_rows <= 0)
            return {};
        return residentPrefillGraphRowCandidates(
            buckets,
            std::min(max_context_rows, max_segment_rows));
    }

    std::vector<int> prefillGraphBucketsAtOrBelowCapacity(
        const std::vector<int> &buckets,
        int resident_graph_rows)
    {
        if (resident_graph_rows <= 0)
            return normalizePrefillGraphBuckets(buckets);

        std::vector<int> bounded;
        for (int bucket : normalizePrefillGraphBuckets(buckets))
        {
            if (bucket <= resident_graph_rows)
                bounded.push_back(bucket);
        }
        bounded.push_back(resident_graph_rows);
        std::sort(bounded.begin(), bounded.end());
        bounded.erase(std::unique(bounded.begin(), bounded.end()), bounded.end());
        return bounded;
    }

    std::vector<int> retainedPrefillGraphBucketLadder(
        const std::vector<int> &buckets,
        int resident_graph_rows,
        std::size_t maximum_bucket_count)
    {
        auto bounded = prefillGraphBucketsAtOrBelowCapacity(
            buckets, resident_graph_rows);
        if (resident_graph_rows <= 0 || maximum_bucket_count == 0u ||
            bounded.empty())
        {
            return {};
        }
        if (bounded.size() <= maximum_bucket_count)
            return bounded;
        if (maximum_bucket_count == 1u)
            return {bounded.back()};

        struct Score
        {
            long double worst_gap =
                std::numeric_limits<long double>::infinity();
            long double total_gap =
                std::numeric_limits<long double>::infinity();
            std::size_t parent = 0u;
            bool valid = false;
        };
        const std::size_t selected_count = std::min(
            maximum_bucket_count, bounded.size());
        std::vector<std::vector<Score>> scores(
            selected_count,
            std::vector<Score>(bounded.size()));
        scores[0][0] = {
            .worst_gap = 0.0L,
            .total_gap = 0.0L,
            .parent = 0u,
            .valid = true,
        };

        /* A transition i->j represents every real length just above bucket i
         * padding to bucket j. The ratio against i+1 is therefore the exact
         * worst multiplicative padding cost of that interval. */
        for (std::size_t slot = 1u; slot < selected_count; ++slot)
        {
            for (std::size_t target = 1u; target < bounded.size(); ++target)
            {
                Score best;
                for (std::size_t previous = 0u;
                     previous < target;
                     ++previous)
                {
                    const Score &prefix = scores[slot - 1u][previous];
                    if (!prefix.valid)
                        continue;
                    const long double gap =
                        static_cast<long double>(bounded[target]) /
                        static_cast<long double>(bounded[previous] + 1);
                    const long double worst =
                        std::max(prefix.worst_gap, gap);
                    const long double total = prefix.total_gap + gap;
                    if (!best.valid || worst < best.worst_gap ||
                        (worst == best.worst_gap &&
                         total < best.total_gap))
                    {
                        best = {
                            .worst_gap = worst,
                            .total_gap = total,
                            .parent = previous,
                            .valid = true,
                        };
                    }
                }
                scores[slot][target] = best;
            }
        }

        std::vector<int> result(selected_count);
        std::size_t cursor = bounded.size() - 1u;
        if (!scores[selected_count - 1u][cursor].valid)
            return {};
        for (std::size_t slot = selected_count; slot-- > 0u;)
        {
            result[slot] = bounded[cursor];
            if (slot > 0u)
                cursor = scores[slot][cursor].parent;
        }
        return result;
    }

    int effectivePrefillGraphMinimumPaddedBucketSeqLen(
        int configured_floor,
        int resident_graph_rows) noexcept
    {
        const int positive_floor = std::max(1, configured_floor);
        if (resident_graph_rows <= 0)
            return positive_floor;

        // Selection and preflight must see the same reachable lower bound.
        return std::min(positive_floor, resident_graph_rows);
    }

    std::vector<int> rawPrefillGraphBucketsForResidentCapacity(
        const std::vector<int> &configured_buckets,
        int resident_graph_rows,
        int configured_floor)
    {
        auto buckets = prefillGraphBucketsAtOrBelowCapacity(
            configured_buckets, resident_graph_rows);
        const int floor =
            effectivePrefillGraphMinimumPaddedBucketSeqLen(
                configured_floor, resident_graph_rows);
        buckets.erase(
            buckets.begin(),
            std::lower_bound(buckets.begin(), buckets.end(), floor));
        return buckets;
    }

    std::vector<int> retainedRawPrefillGraphBucketLadder(
        const std::vector<int> &configured_buckets,
        int resident_graph_rows,
        int configured_floor,
        std::size_t maximum_bucket_count)
    {
        const auto raw_buckets =
            rawPrefillGraphBucketsForResidentCapacity(
                configured_buckets,
                resident_graph_rows,
                configured_floor);
        return retainedPrefillGraphBucketLadder(
            raw_buckets,
            resident_graph_rows,
            maximum_bucket_count);
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
            /*
             * An explicit chunk interval is also a fixed physical graph
             * contract.  In particular, its final short tail must replay the
             * same captured bucket instead of silently selecting a smaller
             * graph from a multi-bucket policy.  The live row count remains
             * `real_count`; only the immutable execution width is selected
             * from `chunk_target`.
             */
            const int bucket_requirement =
                policy.fixed_chunk_real_tokens > 0
                    ? chunk_target
                    : real_count;
            auto selected =
                selectPrefillGraphBucket(bucket_requirement, buckets);
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
