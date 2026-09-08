/**
 * @file BenchmarkPrefillBucketPolicy.h
 * @brief Small policy helpers for benchmark prefill graph bucketing.
 */

#pragma once

namespace llaminar2
{
    /** @brief Reason benchmark mode must not auto-enable padded prefill graphs. */
    enum class BenchmarkPrefillBucketDisableReason
    {
        None,
        Collectives,
        DynamicMoERebalance
    };

    /**
     * @brief Resolve whether benchmark mode may enable captured prefill buckets.
     *
     * @param uses_collectives True when the production graph contains TP/PP or
     *        cross-rank collective stages.
     * @param dynamic_moe_rebalance_active True when mutable placement would
     *        invalidate a captured prefill graph.
     * @param segmented_collective_capture_authority True when a typed protocol
     *        publishes one common physical segment shape to every participant.
     * @return The first reason bucketing must remain disabled, or @ref
     *         BenchmarkPrefillBucketDisableReason::None.
     */
    inline BenchmarkPrefillBucketDisableReason benchmarkPrefillBucketDisableReason(
        bool uses_collectives,
        bool dynamic_moe_rebalance_active,
        bool segmented_collective_capture_authority = false)
    {
        if (uses_collectives && !segmented_collective_capture_authority)
            return BenchmarkPrefillBucketDisableReason::Collectives;
        if (dynamic_moe_rebalance_active)
            return BenchmarkPrefillBucketDisableReason::DynamicMoERebalance;
        return BenchmarkPrefillBucketDisableReason::None;
    }

    /**
     * @brief Return the user-facing explanation for a disable reason.
     * @param reason Resolved benchmark bucket policy result.
     * @return Static diagnostic text, or an empty string for @c None.
     */
    inline const char *benchmarkPrefillBucketDisableMessage(
        BenchmarkPrefillBucketDisableReason reason)
    {
        switch (reason)
        {
        case BenchmarkPrefillBucketDisableReason::Collectives:
            return "Multi-device (TP/PP) run detected; padded buckets "
                   "are incompatible with collective stages";
        case BenchmarkPrefillBucketDisableReason::DynamicMoERebalance:
            return "MoE dynamic rebalancing active; padded buckets are "
                   "rejected by prefill graph preflight";
        case BenchmarkPrefillBucketDisableReason::None:
        default:
            return "";
        }
    }

} // namespace llaminar2
