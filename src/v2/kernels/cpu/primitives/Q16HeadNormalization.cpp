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

    // ========================================================================
    // find_head_max_abs_qs - Find maximum |qs| across all elements in a head
    // ========================================================================

    template <typename BlockType>
    int16_t find_head_max_abs_qs(const BlockType *blocks, size_t num_blocks)
    {
        if (num_blocks == 0 || blocks == nullptr)
        {
            return 0;
        }

        int16_t max_abs = 0;
        for (size_t b = 0; b < num_blocks; ++b)
        {
            for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
            {
                int16_t abs_val = static_cast<int16_t>(std::abs(blocks[b].qs[i]));
                if (abs_val > max_abs)
                {
                    max_abs = abs_val;
                }
            }
        }
        return max_abs;
    }

    // ========================================================================
    // normalize_q16_head_to_vnni_safe - Rescale to VNNI-safe range
    // ========================================================================

    template <typename BlockType>
    VNNISafeNormResult normalize_q16_head_to_vnni_safe(
        BlockType *blocks, size_t num_blocks, int head_dim, float d_unified)
    {
        VNNISafeNormResult result;
        result.norm_factor = 1.0f;
        result.new_d = d_unified;
        result.max_qs = 0;
        result.was_rescaled = false;

        if (num_blocks == 0 || blocks == nullptr || d_unified == 0.0f)
        {
            return result;
        }

        // Find max |qs| across the entire head
        int16_t max_abs_qs = find_head_max_abs_qs(blocks, num_blocks);
        result.max_qs = max_abs_qs;

        // Get the VNNI-safe limit for this head dimension
        int16_t safe_max = get_vnni_safe_max_qs(head_dim);

        // If already safe, no rescaling needed
        if (max_abs_qs <= safe_max)
        {
            // Just ensure all blocks have d = d_unified
            for (size_t b = 0; b < num_blocks; ++b)
            {
                blocks[b].d = d_unified;
            }
            return result;
        }

        // Need to rescale: target_max = safe_max
        result.was_rescaled = true;

        // scale_ratio = safe_max / max_abs_qs (always < 1.0)
        // We use Q16 fixed-point for the ratio to maintain integer-only inner loop
        float scale_ratio = static_cast<float>(safe_max) / static_cast<float>(max_abs_qs);
        int32_t ratio_q16 = static_cast<int32_t>(std::round(scale_ratio * 65536.0f));

        // Apply rescaling to all blocks
        for (size_t b = 0; b < num_blocks; ++b)
        {
            int32_t new_sum = 0;
            for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
            {
                // Integer-only rescale: qs_new = (qs * ratio_q16) >> 16
                int32_t scaled = (static_cast<int32_t>(blocks[b].qs[i]) * ratio_q16 + 32768) >> 16;

                // Clamp to safe range (should be guaranteed by construction, but be safe)
                scaled = std::max(static_cast<int32_t>(-safe_max),
                                  std::min(scaled, static_cast<int32_t>(safe_max)));

                blocks[b].qs[i] = static_cast<int16_t>(scaled);
                new_sum += scaled;
            }
            blocks[b].sum_qs = new_sum;

            // New scale: d_new = d_old / scale_ratio
            // (qs values got smaller, so scale must increase to maintain same FP32 values)
            blocks[b].d = d_unified / scale_ratio;
        }

        // Return the correction factor
        result.new_d = d_unified / scale_ratio;
        result.norm_factor = scale_ratio; // Multiply scores by this to restore magnitude

        return result;
    }

    // ========================================================================
    // Explicit template instantiations for VNNI-safe functions
    // ========================================================================

    template int16_t find_head_max_abs_qs<Q16_1Block>(const Q16_1Block *, size_t);
    template int16_t find_head_max_abs_qs<Q16_1Block_64>(const Q16_1Block_64 *, size_t);
    template int16_t find_head_max_abs_qs<Q16_1Block_128>(const Q16_1Block_128 *, size_t);

    template VNNISafeNormResult normalize_q16_head_to_vnni_safe<Q16_1Block>(
        Q16_1Block *, size_t, int, float);
    template VNNISafeNormResult normalize_q16_head_to_vnni_safe<Q16_1Block_64>(
        Q16_1Block_64 *, size_t, int, float);
    template VNNISafeNormResult normalize_q16_head_to_vnni_safe<Q16_1Block_128>(
        Q16_1Block_128 *, size_t, int, float);

} // namespace llaminar2::primitives
