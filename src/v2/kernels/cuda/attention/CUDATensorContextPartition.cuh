/**
 * @file CUDATensorContextPartition.cuh
 * @brief Device-owned contiguous partition geometry for captured tensor-tile attention.
 *
 * Short contexts retain the established 256-key partition identity. Above 8K,
 * a separately compiled policy groups adjacent canonical key intervals into at
 * most four or eight contiguous sweeps. Producers and reducers share this one
 * geometry authority. Physical storage covers the 32 short-context summaries;
 * long-context spans derive from live KV length, never that smaller allocation.
 * Only live summaries are written and consumed. This policy is confined to
 * full-bucket prefill; grouped verifier reductions retain their fixed order.
 */
#pragma once

#if defined(__CUDACC__)
#define LLAMINAR_TENSOR_PARTITION_HOST_DEVICE __host__ __device__
#else
#define LLAMINAR_TENSOR_PARTITION_HOST_DEVICE
#endif

namespace llaminar2::cuda::tensor_attention_policy
{
/** @brief Captured, immutable limit on the number of long-context summaries. */
enum class SplitPolicy { Four = 4, Eight = 8 };

/** Maximum canonical short-context summaries; longer contexts retain fewer. */
inline constexpr int kMaximumSummaryCapacity = 8192 / 256;

/**
 * @brief Size physical summaries independently of the native KV capacity.
 * @param kv_capacity Positive cache row capacity.
 * @return Exact maximum summary count across every legal live prefix, or zero.
 */
[[nodiscard]] constexpr int summaryCapacity(int kv_capacity) noexcept
{
    if (kv_capacity <= 0) return 0;
    const int canonical = 1 + (kv_capacity - 1) / 256;
    return canonical < kMaximumSummaryCapacity ? canonical : kMaximumSummaryCapacity;
}

/** @brief Matching producer key span and consumer summary extent. */
struct PartitionPlan
{
    int key_span; ///< Contiguous keys owned by each summary, aligned to 256.
    int count;    ///< Live summaries; never the allocation capacity.
};

/** @brief Derive one immutable replay's live partition geometry.
 * @tparam Policy Captured contiguous split limit.
 * @param live_keys Device-published visible KV rows; zero is a valid empty replay.
 * @param capacity Admitted physical summary slots for each query/head row.
 * @return Exact key span and live summary count shared by producer and reducer.
 */
template <SplitPolicy Policy>
LLAMINAR_TENSOR_PARTITION_HOST_DEVICE constexpr PartitionPlan plan(int live_keys, int capacity)
{
    constexpr int limit = static_cast<int>(Policy);
    if (capacity <= 0) return {0, 0};
    const int canonical = live_keys > 0 ? 1 + (live_keys - 1) / 256 : 1;
    if (live_keys <= 8192) return canonical <= capacity ? PartitionPlan{256, canonical} : PartitionPlan{0, 0};
    // Allocation capacity is a stride, not a live key extent. Deriving this
    // group from capacity would silently drop keys after compact admission.
    const int group = 1 + (canonical - 1) / limit;
    const int span = group * 256;
    const int count = 1 + (live_keys - 1) / span;
    return count <= capacity ? PartitionPlan{span, count} : PartitionPlan{0, 0};
}
} // namespace llaminar2::cuda::tensor_attention_policy

#undef LLAMINAR_TENSOR_PARTITION_HOST_DEVICE
