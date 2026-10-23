/**
 * @file ROCmFP32ProjectionLoadPolicy.h
 * @brief Immutable captured-grid scheduling for one ordered FP32 dot product.
 *
 * Load width changes instruction scheduling only. Every mode retains the same
 * stride-256 lane accumulators and reduction. Device live counts are excluded:
 * recording uses physical rows, output columns and this native chunk's actual
 * projection cardinality, which already determine the captured launch grid.
 */
#pragma once
#include <stdexcept>

namespace llaminar2::rocm
{
/** @brief Compiled scheduling modes; values are the exact load-pair counts. */
enum class FP32ProjectionLoadPairs : int { Scalar = 1, Four = 4, Eight = 8 };

/**
 * @brief Select scheduling from total CTAs in this exact captured native launch.
 * @param physicalRows Admitted grid-y extent, never a mutable device live count.
 * @param outputColumns Exact grid-x extent of independent output columns.
 * @param chunkProjections Actual grid-z cardinality, not the outer group total.
 * @param innerWidth Original positive K; only complete load groups are selected.
 * @return Eight for at most 64 CTAs, Four for at most 256, otherwise Scalar;
 * K below 2048 caps this at Four and K below 1024 selects Scalar.
 * @throws std::invalid_argument If any launch extent is nonpositive.
 * @details Threshold divisions bound every multiplication by 256, including
 * INT_MAX input extents. This is a gfx906 scheduling policy, not an attained
 * occupancy/cache estimate or a precision/reduction selection. Arithmetic is unchanged.
 */
[[nodiscard]] constexpr FP32ProjectionLoadPairs selectFP32ProjectionLoadPairs(
    int physicalRows, int outputColumns, int chunkProjections, int innerWidth)
{
    if (physicalRows <= 0 || outputColumns <= 0 || chunkProjections <= 0 || innerWidth <= 0)
        throw std::invalid_argument("FP32 projection scheduling requires positive grid and K extents");
    if (innerWidth < 1024)
        return FP32ProjectionLoadPairs::Scalar;
    constexpr int pipelinedLimit = 256;
    if (physicalRows > pipelinedLimit / outputColumns)
        return FP32ProjectionLoadPairs::Scalar;
    const int rowColumns = physicalRows * outputColumns;
    if (chunkProjections > pipelinedLimit / rowColumns)
        return FP32ProjectionLoadPairs::Scalar;
    const int blocks = rowColumns * chunkProjections;
    return blocks <= 64 && innerWidth >= 2048 ?
        FP32ProjectionLoadPairs::Eight : FP32ProjectionLoadPairs::Four;
}
} // namespace llaminar2::rocm
