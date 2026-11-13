/**
 * @file KVHeadSnapshotColumnGroups.h
 * @brief Pure original-head reconstruction from authenticated consumer KV assignments.
 *
 * This helper does not change schema lookup, capture or rank orchestration.
 * Each captured participant row must contain its actual contiguous KV interval
 * in original-head order. Head width comes from the concrete producer, never
 * from rank-pair arithmetic or a hardcoded model. Cache layouts require their
 * own explicit row/column view and are not certified by calling this helper.
 */
#pragma once
#include "TPSnapshot.h"
#include "../../config/TensorParallelConfig.h"

namespace llaminar2
{
/**
 * @brief Describe one complete original-head group and its exact real replicas.
 * @param assignment Opt-in canonical config retaining authenticated original Q/KV bounds.
 * @param participant_row_cols Actual captured widths in assignment participant order.
 * @param columns_per_head Actual producer width of one original KV head.
 * @return Ordered groups covering every original head once; no payload is inspected.
 * @throws std::invalid_argument For missing source identity, participants or inconsistent widths.
 */
inline std::vector<SnapshotColumnGroup> resolveKVHeadSnapshotColumnGroups(
    const TensorParallelConfig &assignment, std::span<const size_t> participant_row_cols,
    size_t columns_per_head)
{
    if (!assignment.kvHeadCoverageDeclaration() || !assignment.validate() || !columns_per_head ||
        participant_row_cols.size() != assignment.assignments().size())
        throw std::invalid_argument("KV snapshot groups require authenticated original query coverage and actual widths");
    const auto source = *assignment.kvHeadCoverageDeclaration();
    if (columns_per_head > std::numeric_limits<size_t>::max() / static_cast<size_t>(source.kv_heads))
        throw std::invalid_argument("KV snapshot original row width overflows");
    for (size_t rank = 0; rank < participant_row_cols.size(); ++rank)
    {
        const auto &a = assignment.assignments()[rank];
        if (participant_row_cols[rank] != static_cast<size_t>(a.kv_head_count) * columns_per_head)
            throw std::invalid_argument("KV snapshot actual row width differs from its original-head assignment");
    }
    std::vector<SnapshotColumnGroup> groups;
    groups.reserve(static_cast<size_t>(source.kv_heads));
    for (int head = 0; head < source.kv_heads; ++head)
    {
        SnapshotColumnGroup group;
        group.name = "KV_HEAD_" + std::to_string(head);
        group.global_cols = columns_per_head;
        group.mode = SnapshotColumnGroupMode::SUBSET_REPLICATED;
        for (const auto &a : assignment.assignments())
            group.participant_cols.push_back(head >= a.kv_head_start && head < a.kvHeadEnd() ? columns_per_head : 0);
        groups.push_back(std::move(group));
    }
    return groups;
}
} // namespace llaminar2
