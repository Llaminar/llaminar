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
    // VNNI-Safe Normalization for Integer Attention
    // ========================================================================

    /**
     * @brief VNNI-safe maximum INT16 values by head dimension
     *
     * For Q16×Q16→INT32 dot products, we MUST avoid overflow.
     * The constraint is: head_dim × max_qs² ≤ INT32_MAX
     * Therefore: max_qs ≤ sqrt(INT32_MAX / head_dim)
     *
     * With INT32_MAX = 2,147,483,647:
     *   - head_dim=64:  max_qs = floor(sqrt(2147483647/64))  = 5792
     *   - head_dim=128: max_qs = floor(sqrt(2147483647/128)) = 4096
     *   - head_dim=192: max_qs = floor(sqrt(2147483647/192)) = 3344
     *   - head_dim=256: max_qs = floor(sqrt(2147483647/256)) = 2896
     *
     * We use 95% of theoretical max for headroom against rounding errors.
     */
    constexpr int16_t get_vnni_safe_max_qs(int head_dim)
    {
        // max_qs = floor(sqrt(INT32_MAX / head_dim) * 0.95)
        // Pre-computed for common head dimensions with 5% safety margin:
        if (head_dim <= 64)
            return 5500; // 95% of 5792
        else if (head_dim <= 128)
            return 3890; // 95% of 4096
        else if (head_dim <= 192)
            return 3175; // 95% of 3344
        else
            return 2750; // 95% of 2896
    }

    /**
     * @brief Result of VNNI-safe normalization
     */
    struct VNNISafeNormResult
    {
        float norm_factor; ///< Multiply attention scores by this to restore magnitude
        float new_d;       ///< New unified scale after VNNI-safe normalization
        int16_t max_qs;    ///< Maximum |qs| value in normalized output
        bool was_rescaled; ///< True if rescaling was needed
    };

    /**
     * @brief Normalize Q16 head blocks to VNNI-safe range
     *
     * After dynamic-scale RoPE, K blocks have unified d but may have |qs| > MAX_SAFE.
     * This function rescales to ensure safe VNNI dot products, returning a correction
     * factor to apply to attention scores.
     *
     * Algorithm:
     *   1. Find max(|qs|) across all elements in head
     *   2. If max(|qs|) <= get_vnni_safe_max_qs(head_dim), no rescaling needed
     *   3. Otherwise: target_max = get_vnni_safe_max_qs(head_dim)
     *      - scale_ratio = target_max / max(|qs|) (always <= 1.0)
     *      - qs_new = round(qs * scale_ratio)
     *      - d_new = d_old / scale_ratio (scale increases to compensate)
     *   4. Return norm_factor = d_old / d_new = scale_ratio
     *
     * @tparam BlockType One of Q16_1Block, Q16_1Block_64, Q16_1Block_128
     * @param blocks Pointer to array of Q16_1 blocks (modified in-place)
     * @param num_blocks Number of blocks in the head
     * @param head_dim Total dimension of the head (for VNNI limit lookup)
     * @param d_unified The unified scale from dynamic-scale RoPE
     * @return VNNISafeNormResult with norm_factor and metadata
     *
     * @note For attention: score = (Q·K^T) * norm_factor_q * norm_factor_k
     * @note Typically only K needs this; Q can use simpler normalization
     */
    template <typename BlockType>
    VNNISafeNormResult normalize_q16_head_to_vnni_safe(
        BlockType *blocks, size_t num_blocks, int head_dim, float d_unified);

    /**
     * @brief Find maximum |qs| value across all elements in a head
     *
     * Helper function for VNNI-safe normalization.
     *
     * @tparam BlockType One of Q16_1Block, Q16_1Block_64, Q16_1Block_128
     * @param blocks Pointer to array of Q16_1 blocks
     * @param num_blocks Number of blocks in the head
     * @return Maximum absolute qs value
     */
    template <typename BlockType>
    int16_t find_head_max_abs_qs(const BlockType *blocks, size_t num_blocks);

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

    // VNNI-safe normalization
    extern template int16_t find_head_max_abs_qs<Q16_1Block>(const Q16_1Block *, size_t);
    extern template int16_t find_head_max_abs_qs<Q16_1Block_64>(const Q16_1Block_64 *, size_t);
    extern template int16_t find_head_max_abs_qs<Q16_1Block_128>(const Q16_1Block_128 *, size_t);

    extern template VNNISafeNormResult normalize_q16_head_to_vnni_safe<Q16_1Block>(
        Q16_1Block *, size_t, int, float);
    extern template VNNISafeNormResult normalize_q16_head_to_vnni_safe<Q16_1Block_64>(
        Q16_1Block_64 *, size_t, int, float);
    extern template VNNISafeNormResult normalize_q16_head_to_vnni_safe<Q16_1Block_128>(
        Q16_1Block_128 *, size_t, int, float);

} // namespace llaminar2::primitives
