/**
 * @file BenchmarkPrefillBucketPolicy.h
 * @brief Small policy helpers for benchmark prefill graph bucketing.
 */

#pragma once

namespace llaminar2
{
    enum class BenchmarkPrefillBucketDisableReason
    {
        None,
        Collectives,
        DynamicMoERebalance
    };

    inline BenchmarkPrefillBucketDisableReason benchmarkPrefillBucketDisableReason(
        bool uses_collectives,
        bool dynamic_moe_rebalance_active)
    {
        if (uses_collectives)
            return BenchmarkPrefillBucketDisableReason::Collectives;
        if (dynamic_moe_rebalance_active)
            return BenchmarkPrefillBucketDisableReason::DynamicMoERebalance;
        return BenchmarkPrefillBucketDisableReason::None;
    }

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
