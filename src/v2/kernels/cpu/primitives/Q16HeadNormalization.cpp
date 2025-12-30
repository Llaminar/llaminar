/**
 * @file Q16HeadNormalization.cpp
 * @brief Implementation of per-head scale normalization primitives for Q16_1
 */

#include "Q16HeadNormalization.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace llaminar2::primitives
{

    // ========================================================================
    // find_head_max_scale - Find maximum |d| across blocks in a head
    // ========================================================================

    template <typename BlockType>
    float find_head_max_scale(const BlockType *blocks, size_t num_blocks)
    {
        if (num_blocks == 0 || blocks == nullptr)
        {
            return 0.0f;
        }

        // Single block optimization - most common case with model-aware block sizes
        if (num_blocks == 1)
        {
            return std::abs(blocks[0].d);
        }

        float max_scale = 0.0f;
        for (size_t i = 0; i < num_blocks; ++i)
        {
            float abs_scale = std::abs(blocks[i].d);
            if (abs_scale > max_scale)
            {
                max_scale = abs_scale;
            }
        }
        return max_scale;
    }

    // ========================================================================
    // requantize_blocks_to_scale - Requantize blocks to unified scale
    // ========================================================================

    template <typename BlockType>
    void requantize_blocks_to_scale(BlockType *blocks, size_t num_blocks, float head_scale)
    {
        if (num_blocks == 0 || blocks == nullptr || head_scale == 0.0f)
        {
            return;
        }

        // Single block optimization - just update the scale
        if (num_blocks == 1)
        {
            // No need to modify qs values, they're already correctly scaled
            // Just ensure d matches head_scale (may differ in sign)
            if (blocks[0].d != head_scale)
            {
                // If original scale had different sign, we need to negate qs
                if (blocks[0].d < 0.0f != head_scale < 0.0f)
                {
                    for (size_t j = 0; j < BlockType::BLOCK_SIZE; ++j)
                    {
                        blocks[0].qs[j] = static_cast<int16_t>(-blocks[0].qs[j]);
                    }
                    blocks[0].sum_qs = -blocks[0].sum_qs;
                }
                blocks[0].d = head_scale;
            }
            return;
        }

        // Multi-block case: requantize each block
        for (size_t i = 0; i < num_blocks; ++i)
        {
            BlockType &block = blocks[i];

            // Skip blocks with zero scale (all zeros)
            if (block.d == 0.0f)
            {
                block.d = head_scale;
                continue;
            }

            // Compute ratio: this is always <= 1.0 since head_scale is max
            float ratio = block.d / head_scale;

            // Early exit if ratio is 1.0 (block already at head scale)
            if (std::abs(ratio - 1.0f) < 1e-7f)
            {
                block.d = head_scale;
                continue;
            }

            // Scale each quantized value
            int32_t new_sum = 0;
            for (size_t j = 0; j < BlockType::BLOCK_SIZE; ++j)
            {
                // Round to nearest integer
                float scaled = static_cast<float>(block.qs[j]) * ratio;
                int32_t new_val = static_cast<int32_t>(std::round(scaled));

                // Clamp to INT16 range (should rarely be needed since ratio <= 1)
                new_val = std::max(static_cast<int32_t>(std::numeric_limits<int16_t>::min()),
                                   std::min(new_val, static_cast<int32_t>(std::numeric_limits<int16_t>::max())));

                block.qs[j] = static_cast<int16_t>(new_val);
                new_sum += new_val;
            }

            // Update sum and scale
            block.sum_qs = new_sum;
            block.d = head_scale;
        }
    }

    // ========================================================================
    // normalize_q16_head - Combined find + requantize operation
    // ========================================================================

    template <typename BlockType>
    float normalize_q16_head(BlockType *blocks, size_t num_blocks)
    {
        if (num_blocks == 0 || blocks == nullptr)
        {
            return 0.0f;
        }

        // Single block optimization - already normalized!
        if (num_blocks == 1)
        {
            return std::abs(blocks[0].d);
        }

        // Find max scale
        float head_scale = find_head_max_scale(blocks, num_blocks);

        // Requantize to unified scale
        if (head_scale > 0.0f)
        {
            requantize_blocks_to_scale(blocks, num_blocks, head_scale);
        }

        return head_scale;
    }

    // ========================================================================
    // needs_normalization - Check if normalization is needed
    // ========================================================================

    template <typename BlockType>
    bool needs_normalization(const BlockType *blocks, size_t num_blocks, float tolerance)
    {
        if (num_blocks <= 1 || blocks == nullptr)
        {
            // Single-block heads are always normalized by definition
            return false;
        }

        // Get reference scale from first non-zero block
        float ref_scale = 0.0f;
        for (size_t i = 0; i < num_blocks; ++i)
        {
            if (blocks[i].d != 0.0f)
            {
                ref_scale = std::abs(blocks[i].d);
                break;
            }
        }

        if (ref_scale == 0.0f)
        {
            // All blocks are zero, no normalization needed
            return false;
        }

        // Check if all blocks have the same scale within tolerance
        for (size_t i = 0; i < num_blocks; ++i)
        {
            float abs_scale = std::abs(blocks[i].d);
            if (abs_scale == 0.0f)
            {
                continue; // Skip zero blocks
            }

            float rel_diff = std::abs(abs_scale - ref_scale) / ref_scale;
            if (rel_diff > tolerance)
            {
                return true; // Scales differ, normalization needed
            }
        }

        return false; // All scales are the same, no normalization needed
    }

    // ========================================================================
    // Explicit template instantiations for all block types
    // ========================================================================

    // Q16_1Block (32 elements)
    template float find_head_max_scale<Q16_1Block>(const Q16_1Block *, size_t);
    template void requantize_blocks_to_scale<Q16_1Block>(Q16_1Block *, size_t, float);
    template float normalize_q16_head<Q16_1Block>(Q16_1Block *, size_t);
    template bool needs_normalization<Q16_1Block>(const Q16_1Block *, size_t, float);

    // Q16_1Block_64 (64 elements)
    template float find_head_max_scale<Q16_1Block_64>(const Q16_1Block_64 *, size_t);
    template void requantize_blocks_to_scale<Q16_1Block_64>(Q16_1Block_64 *, size_t, float);
    template float normalize_q16_head<Q16_1Block_64>(Q16_1Block_64 *, size_t);
    template bool needs_normalization<Q16_1Block_64>(const Q16_1Block_64 *, size_t, float);

    // Q16_1Block_128 (128 elements)
    template float find_head_max_scale<Q16_1Block_128>(const Q16_1Block_128 *, size_t);
    template void requantize_blocks_to_scale<Q16_1Block_128>(Q16_1Block_128 *, size_t, float);
    template float normalize_q16_head<Q16_1Block_128>(Q16_1Block_128 *, size_t);
    template bool needs_normalization<Q16_1Block_128>(const Q16_1Block_128 *, size_t, float);

} // namespace llaminar2::primitives
