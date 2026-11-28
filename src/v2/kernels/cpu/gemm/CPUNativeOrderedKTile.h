/**
 * @file CPUNativeOrderedKTile.h
 * @brief Canonical K-block range of one ordered K tile handed to a native CPU kernel.
 *
 * Ordered (GPU-aligned or K-parallel) projections split a row of `blocks` K
 * blocks into a fixed number of tiles of `width = ceil(blocks / tiles)` blocks.
 * When the tile count does not divide the row, trailing tiles are empty: with
 * sixteen tiles, 35 blocks give width 3, so tiles 12..15 hold no block.
 *
 * Every native CPU kernel entry admits the range it receives once, in cold code
 * (requireSuperblockWalk() in CPUNativeOrderedPartitionPlan.h): 0 <= first <=
 * last <= blocks. An empty tile is therefore handed over as [blocks, blocks),
 * never as the reversed [tile * width, blocks) that an unclamped start forms
 * (first=36 last=35 for that example). The NEON packed prefill forms its
 * partitions the same way. Clamping the start removes no block from any tile
 * and adds none, so the K traversal, the per-tile partials and the ordered fold
 * are unchanged; an empty tile still publishes positive zero (or the retained
 * accumulator).
 *
 * Kernel-internal partition intervals of a single admitted plan (the hot
 * `first = begin + p * width` of the ordered tile programs) are not entries and
 * keep their documented empty, possibly reversed form.
 */
#pragma once

#include <algorithm>

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief First K block of ordered K tile @p tile.
     * @param tile Tile index, 0 <= tile < tile count.
     * @param width Blocks per tile, ceil(blocks / tile count) >= 1.
     * @param blocks K blocks of the row.
     * @return min(tile * width, blocks); with end = min(begin + width, blocks)
     *         the tile range is never reversed, and an empty tile is [blocks, blocks).
     */
    [[nodiscard]] constexpr int orderedKTileBegin(int tile, int width, int blocks) noexcept
    {
        return std::min(tile * width, blocks);
    }
} // namespace llaminar2::cpu::native_quantised
