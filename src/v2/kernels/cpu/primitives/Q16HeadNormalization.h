/**
 * @file Q16HeadNormalization.h
 * @brief Per-head scale normalization primitives for Q16_1 integer attention
 *
 * This module implements the per-head scale normalization strategy for Q16_1
 * integer attention. The key insight is:
 *
 *   - Each Q16_1 block has its own scale factor (d)
 *   - For integer dot products, we need a unified scale per head
 *   - Normalizing to head_scale = max(|block.d|) preserves precision
 *   - After normalization, all blocks share one scale factor
 *
 * OPTIMAL PATH: When block_size == head_dim (1 block per head):
 *   - NO normalization needed! Block scale IS the head scale.
 *   - Use model-aware block sizes: 64 or 128 to hit this path.
 *   - For MLA models: use separate NOPE (128) + ROPE (64) tensors.
 *
 * Usage:
 *   // Find the maximum scale across all blocks in a head
 *   float head_scale = find_head_max_scale<Q16_1Block_128>(blocks, num_blocks);
 *
 *   // Requantize blocks to the unified head scale
 *   requantize_blocks_to_scale<Q16_1Block_128>(blocks, num_blocks, head_scale);
 *
 *   // Or use the combined normalize function
 *   float head_scale = normalize_q16_head<Q16_1Block_128>(blocks, num_blocks);
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include "../../../tensors/BlockStructures.h"

namespace llaminar2::primitives
{

    /**
     * @brief Metadata for a normalized Q16_1 head
     *
     * Stored alongside tensors to track per-head scale information
     * for the integer attention kernel.
     */
    struct Q16_1HeadMetadata
    {
        float head_scale;   ///< Unified scale for all blocks in this head
        int32_t num_blocks; ///< Number of blocks in this head
        bool is_normalized; ///< True if blocks have been normalized to head_scale

        Q16_1HeadMetadata()
            : head_scale(0.0f), num_blocks(0), is_normalized(false)
        {
        }

        Q16_1HeadMetadata(float scale, int32_t blocks, bool normalized = true)
            : head_scale(scale), num_blocks(blocks), is_normalized(normalized)
        {
        }
    };

    /**
     * @brief Find the maximum absolute scale factor across blocks in a head
     *
     * @tparam BlockType One of Q16_1Block, Q16_1Block_64, Q16_1Block_128
     * @param blocks Pointer to array of Q16_1 blocks for one head
     * @param num_blocks Number of blocks in the head (head_dim / BlockType::BLOCK_SIZE)
     * @return Maximum |block.d| value across all blocks
     *
     * @note For single-block heads (optimal path), this simply returns |blocks[0].d|
     */
    template <typename BlockType>
    float find_head_max_scale(const BlockType *blocks, size_t num_blocks);

    /**
     * @brief Requantize blocks to a unified scale factor
     *
     * For each block with scale d:
     *   - Compute ratio = d / head_scale (always <= 1.0)
     *   - Scale qs values: qs_new[i] = round(qs[i] * ratio)
     *   - Update sum_qs accordingly
     *   - Set block.d = head_scale
     *
     * @tparam BlockType One of Q16_1Block, Q16_1Block_64, Q16_1Block_128
     * @param blocks Pointer to array of Q16_1 blocks (modified in-place)
     * @param num_blocks Number of blocks to process
     * @param head_scale Target scale factor (should be >= max of all block scales)
     *
     * @note head_scale MUST be >= all block.d values, otherwise overflow may occur
     * @note For single-block heads, this is a no-op (just sets d = head_scale)
     */
    template <typename BlockType>
    void requantize_blocks_to_scale(BlockType *blocks, size_t num_blocks, float head_scale);

    /**
     * @brief Normalize all blocks in a head to a unified scale
     *
     * Combined operation: find max scale, then requantize all blocks.
     * This is the primary API for head normalization.
     *
     * @tparam BlockType One of Q16_1Block, Q16_1Block_64, Q16_1Block_128
     * @param blocks Pointer to array of Q16_1 blocks (modified in-place)
     * @param num_blocks Number of blocks in the head
     * @return The computed head_scale (max |block.d| value)
     *
     * @note For single-block heads (optimal path):
     *   - Returns blocks[0].d immediately
     *   - No requantization needed (already normalized)
     */
    template <typename BlockType>
    float normalize_q16_head(BlockType *blocks, size_t num_blocks);

    /**
     * @brief Check if a head needs normalization
     *
     * Returns false if:
     *   - num_blocks == 1 (single-block heads are always normalized)
     *   - All blocks already have the same scale (within tolerance)
     *
     * @tparam BlockType One of Q16_1Block, Q16_1Block_64, Q16_1Block_128
     * @param blocks Pointer to array of Q16_1 blocks
     * @param num_blocks Number of blocks in the head
     * @param tolerance Relative tolerance for scale comparison (default: 1e-6)
     * @return true if normalization is needed, false otherwise
     */
    template <typename BlockType>
    bool needs_normalization(const BlockType *blocks, size_t num_blocks, float tolerance = 1e-6f);

    /**
     * @brief Compute the number of blocks per head for a given head dimension and block size
     *
     * @param head_dim Dimension of attention head
     * @param block_size Size of Q16_1 block (32, 64, 128, or 192)
     * @return Number of blocks per head (head_dim / block_size)
     */
    constexpr size_t blocks_per_head(int head_dim, size_t block_size)
    {
        return static_cast<size_t>(head_dim) / block_size;
    }

    /**
     * @brief Check if head dimension is optimal for a given block size
     *
     * Optimal = exactly 1 block per head, no normalization needed.
     *
     * @param head_dim Dimension of attention head
     * @param block_size Size of Q16_1 block
     * @return true if head_dim == block_size (optimal path)
     */
    constexpr bool is_optimal_block_size(int head_dim, size_t block_size)
    {
        return static_cast<size_t>(head_dim) == block_size;
    }

    // ========================================================================
    // Explicit instantiation declarations for common block types
    // ========================================================================

    extern template float find_head_max_scale<Q16_1Block>(const Q16_1Block *, size_t);
    extern template float find_head_max_scale<Q16_1Block_64>(const Q16_1Block_64 *, size_t);
    extern template float find_head_max_scale<Q16_1Block_128>(const Q16_1Block_128 *, size_t);

    extern template void requantize_blocks_to_scale<Q16_1Block>(Q16_1Block *, size_t, float);
    extern template void requantize_blocks_to_scale<Q16_1Block_64>(Q16_1Block_64 *, size_t, float);
    extern template void requantize_blocks_to_scale<Q16_1Block_128>(Q16_1Block_128 *, size_t, float);

    extern template float normalize_q16_head<Q16_1Block>(Q16_1Block *, size_t);
    extern template float normalize_q16_head<Q16_1Block_64>(Q16_1Block_64 *, size_t);
    extern template float normalize_q16_head<Q16_1Block_128>(Q16_1Block_128 *, size_t);

    extern template bool needs_normalization<Q16_1Block>(const Q16_1Block *, size_t, float);
    extern template bool needs_normalization<Q16_1Block_64>(const Q16_1Block_64 *, size_t, float);
    extern template bool needs_normalization<Q16_1Block_128>(const Q16_1Block_128 *, size_t, float);

} // namespace llaminar2::primitives
