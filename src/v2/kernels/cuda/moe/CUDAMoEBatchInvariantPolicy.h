/**
 * @file CUDAMoEBatchInvariantPolicy.h
 * @brief Fixed arithmetic policy for byte-invariant CUDA MoE execution.
 *
 * CUDA grouped gate/up and down projections use ordered split-K reductions.
 * The partition count changes floating-point parenthesization even when launch
 * order is deterministic. A production-shaped exhaustive comparison found
 * that only the sixteen-partition tree matches the established serial-row
 * contract across grouped M. Partition count is therefore arithmetic, not a
 * tuning control. Geometry-only controls such as block width remain eligible
 * for learned dispatch because they do not alter the reduction tree.
 */

#pragma once

namespace llaminar2
{
    /**
     * @brief Compile-time CUDA MoE arithmetic invariants shared by launchers.
     *
     * Keeping these values outside `DebugEnv` makes a known non-equivalent
     * arithmetic tree unrepresentable in production. Direct down publication
     * assigns exactly one warp to each router slot; additional warps are idle
     * and strictly dominated, while fewer warps serialize independent routes.
     */
    struct CUDAMoEBatchInvariantPolicy final
    {
        static constexpr int gate_up_k_partitions = 16;
        static constexpr int down_k_partitions = 16;

        /** @return the unique non-dominated warp count for a valid top-k. */
        static constexpr int directDownWarps(int top_k) noexcept
        {
            return top_k;
        }
    };
}
