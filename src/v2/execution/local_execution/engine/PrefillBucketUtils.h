/**
 * @file PrefillBucketUtils.h
 * @brief Bucket selection and chunk planning helpers for prefill graph capture.
 *
 * The helpers in this file are deliberately host-only and side-effect free.
 * They let the forward engine, server paths, and tests agree on bucket
 * selection, padding boundaries, and chunk bookkeeping before any GPU graph
 * capture starts. Stateful model execution remains responsible for deciding
 * whether padded buckets are safe for a particular graph.
 */

#pragma once

#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Result of mapping a real prompt length to a fixed prefill bucket.
     */
    struct PrefillBucketSelection
    {
        bool ok = false;              ///< True when a bucket was selected.
        int real_seq_len = 0;         ///< Number of real, non-padding tokens.
        int bucket_seq_len = 0;       ///< Fixed execution length selected for graph capture.
        bool exact = false;           ///< True when real_seq_len already equals bucket_seq_len.
        std::string error;            ///< Human-readable failure reason when ok is false.

        /// @brief Convenience conversion for success checks.
        explicit operator bool() const { return ok; }
    };

    /**
     * @brief Bookkeeping record for one chunked prefill execution.
     */
    struct PrefillChunkPlan
    {
        int token_offset = 0;                 ///< Offset of the first real token in the original prompt.
        int real_count = 0;                   ///< Real token count processed by this chunk.
        int bucket_seq_len = 0;               ///< Fixed execution length selected for this chunk.
        int chunk_index = 0;                  ///< Stable chunk ordinal within the scheduled real-token range.
        bool rebalance_allowed_after = false; ///< True when min real-token interval is satisfied at this boundary.
        bool rebalance_required_after = false; ///< True when max real-token interval forces maintenance here.
    };

    /**
     * @brief Explicit scheduler policy for graph-captured prefill chunks.
     *
     * The policy is phrased in real-token counts. Bucket padding is shape
     * material only and must not advance rebalance windows or histogram counts.
     */
    struct PrefillChunkSchedulerPolicy
    {
        std::vector<int> bucket_sizes;        ///< Candidate graph bucket lengths.
        int fixed_chunk_real_tokens = 0;      ///< Real-token interval per chunk; 0 uses the largest bucket.
        int min_rebalance_interval_tokens = 0; ///< Real tokens before rebalance may run; 0 disables optional flag.
        int max_rebalance_interval_tokens = 0; ///< Real tokens before rebalance is required; 0 disables force flag.
        int real_token_start = 0;             ///< First real token offset covered by this schedule.
        int real_token_count = 0;             ///< Number of real tokens covered by this schedule.
    };

    /**
     * @brief Result of applying PrefillChunkSchedulerPolicy.
     */
    struct PrefillChunkSchedule
    {
        bool ok = false;
        std::vector<PrefillChunkPlan> chunks;
        std::string error;

        explicit operator bool() const { return ok; }
    };

    /**
     * @brief Host-owned inputs for one future `runPrefillChunk()` execution.
     *
     * This descriptor deliberately does not execute a graph. It packages the
     * data a graph runner needs: padded token IDs, absolute positions, and the
     * real/bucket shape metadata consumed by dynamic replay callbacks.
     */
    struct PrefillChunkExecutionInput
    {
        bool ok = false;                 ///< True when all buffers were built successfully.
        int token_offset = 0;            ///< Offset of this chunk in the original prompt.
        int real_count = 0;              ///< Real tokens in this chunk.
        int bucket_seq_len = 0;          ///< Fixed graph execution length for this chunk.
        std::vector<int> token_ids;      ///< Owned padded token IDs [bucket_seq_len].
        std::vector<int> position_ids;   ///< Owned absolute position IDs [batch_size * bucket_seq_len].
        std::string error;               ///< Human-readable failure reason when ok is false.

        /// @brief Convenience conversion for success checks.
        explicit operator bool() const { return ok; }
    };

    /// @brief Returns the Tier 1 production bucket list for bounded padding overhead.
    std::vector<int> defaultPrefillGraphBuckets();

    /**
     * @brief Sort, deduplicate, and validate a bucket list.
     *
     * Non-positive entries are ignored. The returned list is strictly
     * increasing, making later selection deterministic even when an environment
     * variable lists buckets out of order.
     */
    std::vector<int> normalizePrefillGraphBuckets(const std::vector<int> &buckets);

    /**
     * @brief Select the smallest bucket that can contain real_seq_len tokens.
     *
     * Returns a failed selection when the list is empty, the real length is
     * invalid, or the prompt is longer than the largest configured bucket.
     */
    PrefillBucketSelection selectPrefillGraphBucket(
        int real_seq_len,
        const std::vector<int> &bucket_sizes);

    /**
     * @brief Return a padded token vector sized to the selected bucket.
     *
     * The first real_seq_len entries are copied from tokens. Remaining entries
     * are filled with pad_token_id. Invalid arguments return an empty vector.
     */
    std::vector<int> padPrefillTokensToBucket(
        const int *tokens,
        int real_seq_len,
        int bucket_seq_len,
        int pad_token_id);

    /**
     * @brief Plan a prompt as one or more bucket-sized prefill chunks.
     *
     * Chunks are greedily sized by the largest configured bucket so long prompts
     * can be represented as repeated fixed-topology executions. The final chunk
     * uses the smallest bucket that can contain its remainder.
     */
    std::vector<PrefillChunkPlan> planPrefillChunks(
        int total_real_tokens,
        const std::vector<int> &bucket_sizes);

    /**
     * @brief Plan a real-token range using an explicit chunk scheduling policy.
     */
    PrefillChunkSchedule planPrefillChunkSchedule(
        const PrefillChunkSchedulerPolicy &policy);

    /**
     * @brief Build absolute position IDs for one fixed-bucket prefill chunk.
     *
     * Real rows receive positions `[token_offset, token_offset + real_count)`.
     * Padding rows receive deterministic continuation positions so graph inputs
     * remain fully initialized; later execution gates decide whether padded rows
     * are safe for a model/backend.
     */
    std::vector<int> buildPrefillChunkPositionIds(
        int real_count,
        int bucket_seq_len,
        int token_offset,
        int batch_size = 1);

    /**
     * @brief Build host inputs for one bucketed prefill chunk.
     *
     * The source tokens point at the full original prompt. The chunk plan selects
     * a real-token slice by offset/count and pads it to the bucket length. This
     * is the pure data-preparation half of the future `runPrefillChunk()` path.
     */
    PrefillChunkExecutionInput buildPrefillChunkExecutionInput(
        const int *tokens,
        int total_real_tokens,
        const PrefillChunkPlan &chunk,
        int pad_token_id,
        int batch_size = 1);

} // namespace llaminar2
