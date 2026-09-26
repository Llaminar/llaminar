/**
 * @file GroupedDp4aLaunchGeometry.h
 * @brief Shared host/device ownership for register-bounded grouped DP4A tiles.
 *
 * A logical tuning tile can contain 64 rows, but assigning all those row
 * accumulators to one lane exceeds the register file. Disjoint lane groups and
 * grid-Z blocks instead own at most 16 rows per lane, without sharing an output
 * or changing a row's K reduction. This header owns both the launcher geometry
 * and the kernel's indexing, so resource bounds cannot diverge from launch size.
 */
#pragma once

#if defined(__CUDACC__)
#define LLAMINAR_DP4A_HD __host__ __device__
#else
#define LLAMINAR_DP4A_HD
#endif

namespace llaminar2::cuda
{
/**
 * @brief Physical tiling for a logical grouped-decode candidate.
 * @tparam RequestedRows Logical row reuse selected by the dispatch policy.
 * @tparam TileColumns Columns in one original serial-M1 tile.
 * @tparam ColumnsPerThread Consecutive columns owned by one lane.
 */
template <int RequestedRows, int TileColumns, int ColumnsPerThread>
struct GroupedDp4aLaunchGeometry
{
    static_assert(RequestedRows >= 2 && RequestedRows <= 64 &&
                  (RequestedRows & (RequestedRows - 1)) == 0);
    static_assert(ColumnsPerThread > 0 && TileColumns % ColumnsPerThread == 0);
    static constexpr int rows_per_lane = RequestedRows < 16 ? RequestedRows : 16;
    static constexpr int threads_per_n_tile = TileColumns / ColumnsPerThread;
    static_assert(threads_per_n_tile == 32 || threads_per_n_tile == 64 ||
                  threads_per_n_tile == 128);
    // Shallow tiles pack independent N tiles to avoid one-warp CTA limits.
    static constexpr int n_tiles_per_block = threads_per_n_tile == 32
        ? (RequestedRows <= 8 ? 4 : (RequestedRows <= 16 ? 2 : 1))
        : (threads_per_n_tile == 64 && RequestedRows <= 8 ? 2 : 1);
    static constexpr int threads_per_row_group = threads_per_n_tile * n_tiles_per_block;
    // 512/1024-thread CTAs would impose a second, tighter register ceiling.
    // Continue oversized logical row tiles in independent grid-Z blocks.
    static constexpr int maximum_row_groups = 256 / threads_per_row_group;
    static constexpr int desired_row_groups = RequestedRows / rows_per_lane;
    static constexpr int row_groups = desired_row_groups < maximum_row_groups
        ? desired_row_groups : maximum_row_groups;
    static constexpr int rows_per_block = rows_per_lane * row_groups;
    static constexpr int threads_per_block = threads_per_row_group * row_groups;
    static constexpr int columns_per_block = TileColumns * n_tiles_per_block;

    /** @return First row owned by this lane within a grid-Z row slice. */
    LLAMINAR_DP4A_HD static constexpr int rowBase(int block_z, int thread) noexcept
    {
        // A single-group CTA has one uniform activation address for every
        // column lane. Express that uniformity directly: deriving a redundant
        // zero group from threadIdx makes NVCC retain per-lane row addresses,
        // increasing registers across the fully unrolled row loop.
        if constexpr (row_groups == 1)
            return block_z * rows_per_block;
        else
            return block_z * rows_per_block +
                   (thread / threads_per_row_group) * rows_per_lane;
    }

    /** @return First column owned by this lane within a grid-X column slice. */
    LLAMINAR_DP4A_HD static constexpr int columnBase(int block_x, int thread) noexcept
    {
        // Do not mask a lane index when no second row group exists. This keeps
        // shallow tiles identical to the original serial-column ownership.
        const int group_thread = row_groups == 1
            ? thread : thread % threads_per_row_group;
        return block_x * columns_per_block +
               (group_thread / threads_per_n_tile) * TileColumns +
               (group_thread % threads_per_n_tile) * ColumnsPerThread;
    }
};
} // namespace llaminar2::cuda

#undef LLAMINAR_DP4A_HD
