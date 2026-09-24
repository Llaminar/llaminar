/**
 * @file CUDAFlashAttentionWorkspaceEnvelope.h
 * @brief Host-side family admission derived from the canonical CUDA FA2 policy.
 *
 * Metadata planning and concrete stage binding consume this single envelope.
 * It prices all serially retained prefill shapes without changing their
 * individual launch policies. Keep this host-only aggregation separate from
 * the launch header consumed by device translation units, so accounting edits
 * do not force recompilation of otherwise unchanged attention kernels.
 */
#pragma once

#include "CUDAFlashAttentionLaunchPolicy.h"

namespace llaminar2::cuda::fa2_policy
{
    /**
     * @brief Resolve the largest context-summary allocation in a serial family.
     * @param geometry Immutable device/head/KV geometry; `query_rows` is the
     *        largest admitted M, not a live request width.
     * @return A real context plan with maximal bytes, or an invalid plan when
     *         no family member selects context parallelism.
     *
     * Large query grids can select direct attention while smaller grids still
     * use context summaries. Bound the search using the persistent scheduler's
     * grid-wave ceiling and widest compiled query tile. Beyond that ceiling,
     * every grouping has only one context slot and cannot select context mode.
     * Within it, policy changes occur at 16-row tile boundaries (including the
     * explicit 17/128-row transitions). Visit each interval's largest M in
     * descending order: summary bytes grow with M, so the first context plan
     * is the exact envelope. Work is bounded by device geometry, not context
     * length. Callers reserve these bytes; they must not launch this plan in
     * place of an actual request's independently selected capture policy.
     */
    [[nodiscard]] inline constexpr FA2PrefillParallelPlan
    selectFA2GeometrySelectedWorkspaceEnvelope(
        const FA2PrefillParallelGeometry &geometry)
    {
        if (geometry.requested_axis !=
                attention::AttentionPrefillParallelAxis::GeometrySelected ||
            geometry.batch_size <= 0 || geometry.query_rows <= 0 ||
            geometry.local_query_heads <= 0 || geometry.sm_count <= 0 ||
            geometry.kv_capacity <= kFA2CanonicalContextPartitionKeys)
        {
            return {};
        }
        const int maximum_groups = maximumFA2QueryWarpGroups(geometry.head_dim);
        if (maximum_groups == 0)
            return {};

        const std::int64_t blocks_per_tile =
            static_cast<std::int64_t>(geometry.batch_size) * geometry.local_query_heads;
        const std::int64_t grid_ceiling =
            static_cast<std::int64_t>(geometry.sm_count) * kFA2ContextTargetBlockWaves;
        const std::int64_t maximum_context_tiles = (grid_ceiling - 1) / blocks_per_tile;
        const std::int64_t maximum_context_rows = maximum_context_tiles *
            maximum_groups * kFA2QueryRowsPerWarpGroup;
        // All products above fit int64 for positive int geometry. Clamp before
        // narrowing so even very large context declarations remain well-defined.
        int rows = static_cast<int>(
            maximum_context_rows < geometry.query_rows
                ? maximum_context_rows : geometry.query_rows);
        while (rows >= kFA2ContextPrefillMinimumQueryRows)
        {
            auto candidate = geometry;
            candidate.query_rows = rows;
            const auto plan = selectFA2PrefillParallelPlan(candidate);
            if (plan.usesContextParallelism())
                return plan;
            rows = ((rows - 1) / kFA2QueryRowsPerWarpGroup) * kFA2QueryRowsPerWarpGroup;
        }
        return {};
    }

} // namespace llaminar2::cuda::fa2_policy
