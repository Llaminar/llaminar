/**
 * @file VNNISafetyConstants.h
 * @brief VNNI INT32 overflow prevention constants and helpers for Q16 attention
 *
 * This header defines the safety limits for INT16 values when using AVX-512 VNNI
 * instructions (VPDPWSSD) for Q16_1 integer attention.
 *
 * KEY INSIGHT:
 * VPDPWSSD accumulates 2 products per INT32 lane per instruction.
 * For head_dim elements, products_per_lane = head_dim / 16.
 * To prevent INT32 overflow: N × val² ≤ INT32_MAX
 * Therefore: val ≤ sqrt(INT32_MAX / N)
 *
 * See: tests/v2/unit/microkernels/Test__VNNIOverflowAnalysis.cpp for derivation.
 * See: docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md "VNNI OVERFLOW PREVENTION CONTRACT"
 *
 * @author Llaminar Team
 * @date 2025-01-01
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace llaminar2
{
    namespace vnni_safety
    {

        // ============================================================================
        // Core Constants
        // ============================================================================

        /// INT32_MAX for overflow calculations
        constexpr int64_t INT32_MAX_VAL = 2147483647LL; // 2^31 - 1

        /// INT16 maximum value
        constexpr int16_t INT16_MAX_VAL = 32767;

        // ============================================================================
        // Pre-computed MAX_SAFE_INT16 limits by head_dim
        // ============================================================================
        // Formula: MAX_SAFE_INT16 = floor(sqrt(INT32_MAX / (head_dim / 16)))
        //
        // These limits ensure that when all head_dim products are accumulated,
        // the INT32 accumulator will not overflow even in worst-case (all values at limit).

        /// head_dim=64: 4 products per INT32 lane → sqrt(2^31 / 4) = 23170
        constexpr int16_t MAX_SAFE_INT16_64 = 23170;

        /// head_dim=96: 6 products per INT32 lane → sqrt(2^31 / 6) = 18918
        constexpr int16_t MAX_SAFE_INT16_96 = 18918;

        /// head_dim=128: 8 products per INT32 lane → sqrt(2^31 / 8) = 16383
        constexpr int16_t MAX_SAFE_INT16_128 = 16383;

        /// head_dim=192 (MLA): 12 products per INT32 lane → sqrt(2^31 / 12) = 13377
        constexpr int16_t MAX_SAFE_INT16_192 = 13377;

        /// head_dim=256: 16 products per INT32 lane → sqrt(2^31 / 16) = 11585
        constexpr int16_t MAX_SAFE_INT16_256 = 11585;

        /// Conservative default: safe for head_dim ≤ 192 (covers most models)
        constexpr int16_t MAX_SAFE_INT16_DEFAULT = MAX_SAFE_INT16_192;

        // ============================================================================
        // Equivalent FP32 ranges (with kv_cache_scale=8.0)
        // ============================================================================
        // FP32_range = MAX_SAFE_INT16 × kv_cache_scale / 32767
        //
        // Values beyond these ranges will be clipped during quantization.

        /// head_dim=64: clip FP32 at ±5.66 (with scale=8)
        constexpr float SAFE_FP32_RANGE_64 = 5.66f;

        /// head_dim=96: clip FP32 at ±4.62 (with scale=8)
        constexpr float SAFE_FP32_RANGE_96 = 4.62f;

        /// head_dim=128: clip FP32 at ±4.00 (with scale=8)
        constexpr float SAFE_FP32_RANGE_128 = 4.00f;

        /// head_dim=192: clip FP32 at ±3.27 (with scale=8)
        constexpr float SAFE_FP32_RANGE_192 = 3.27f;

        /// head_dim=256: clip FP32 at ±2.83 (with scale=8)
        constexpr float SAFE_FP32_RANGE_256 = 2.83f;

        // ============================================================================
        // Helper Functions
        // ============================================================================

        /**
         * @brief Get the max safe INT16 value for a given head dimension
         *
         * @param head_dim The attention head dimension (64, 96, 128, 192, 256)
         * @return The maximum safe INT16 magnitude to prevent VNNI overflow
         */
        inline constexpr int16_t get_max_safe_int16(int head_dim)
        {
            switch (head_dim)
            {
            case 64:
                return MAX_SAFE_INT16_64;
            case 96:
                return MAX_SAFE_INT16_96;
            case 128:
                return MAX_SAFE_INT16_128;
            case 192:
                return MAX_SAFE_INT16_192;
            case 256:
                return MAX_SAFE_INT16_256;
            default:
                // For non-standard head_dim, compute dynamically
                // products_per_lane = head_dim / 16
                // max_safe = sqrt(INT32_MAX / products_per_lane)
                if (head_dim <= 64)
                    return MAX_SAFE_INT16_64;
                if (head_dim <= 96)
                    return MAX_SAFE_INT16_96;
                if (head_dim <= 128)
                    return MAX_SAFE_INT16_128;
                if (head_dim <= 192)
                    return MAX_SAFE_INT16_192;
                return MAX_SAFE_INT16_256;
            }
        }

        /**
         * @brief Compute max safe INT16 dynamically for arbitrary head_dim
         *
         * @param head_dim The attention head dimension
         * @return The maximum safe INT16 magnitude
         */
        inline int16_t compute_max_safe_int16(int head_dim)
        {
            if (head_dim <= 0)
                return INT16_MAX_VAL;

            int products_per_lane = (head_dim + 15) / 16; // Ceiling division
            if (products_per_lane <= 0)
                products_per_lane = 1;

            double max_val_squared = static_cast<double>(INT32_MAX_VAL) / products_per_lane;
            return static_cast<int16_t>(std::floor(std::sqrt(max_val_squared)));
        }

        /**
         * @brief Clip an INT32 value to the safe INT16 range for given head_dim
         *
         * This is the CRITICAL function for VNNI safety. All FP32→INT16 quantization
         * must use this function to ensure overflow safety.
         *
         * @param val The INT32 value to clip (result of FP32 * 32767 / kv_cache_scale)
         * @param head_dim The attention head dimension
         * @return Clipped INT16 value guaranteed to be within safe range
         */
        inline int16_t clip_to_safe_int16(int32_t val, int head_dim)
        {
            const int16_t limit = get_max_safe_int16(head_dim);
            const int32_t clamped = std::max(static_cast<int32_t>(-limit),
                                             std::min(val, static_cast<int32_t>(limit)));
            return static_cast<int16_t>(clamped);
        }

        /**
         * @brief Calculate the safe FP32 range for given head_dim and kv_cache_scale
         *
         * Values outside [-range, +range] will be clipped during quantization.
         *
         * @param head_dim The attention head dimension
         * @param kv_cache_scale The KV cache scale factor (typically 8.0)
         * @return The maximum safe FP32 magnitude
         */
        inline float get_safe_fp32_range(int head_dim, float kv_cache_scale)
        {
            const int16_t max_int16 = get_max_safe_int16(head_dim);
            return static_cast<float>(max_int16) * kv_cache_scale / 32767.0f;
        }

        /**
         * @brief Quantize FP32 to INT16 with fixed scale and VNNI-safe clipping
         *
         * This is the core quantization function for VNNI-safe Q16_1 attention.
         *
         * @param fp32_val The FP32 value to quantize
         * @param kv_cache_scale The KV cache scale factor (defines FP32 range)
         * @param head_dim The attention head dimension (for clipping limits)
         * @return VNNI-safe INT16 value
         */
        inline int16_t quantize_fp32_to_safe_int16(float fp32_val, float kv_cache_scale, int head_dim)
        {
            // Fixed-scale quantization: int16 = fp32 * 32767 / kv_cache_scale
            const float inv_scale = 32767.0f / kv_cache_scale;
            const int32_t int32_val = static_cast<int32_t>(std::round(fp32_val * inv_scale));

            // Clip to VNNI-safe range
            return clip_to_safe_int16(int32_val, head_dim);
        }

        // ============================================================================
        // Debug Helpers
        // ============================================================================

        /**
         * @brief Check if an INT16 value is within the safe range for given head_dim
         *
         * @param val The INT16 value to check
         * @param head_dim The attention head dimension
         * @return true if safe, false if would cause overflow
         */
        inline bool is_safe_int16(int16_t val, int head_dim)
        {
            const int16_t limit = get_max_safe_int16(head_dim);
            return std::abs(val) <= limit;
        }

        /**
         * @brief Calculate overflow margin for diagnostic purposes
         *
         * Returns positive value if within safe range, negative if would overflow.
         *
         * @param val The INT16 value
         * @param head_dim The attention head dimension
         * @return Margin (positive = safe, negative = overflow)
         */
        inline int32_t overflow_margin(int16_t val, int head_dim)
        {
            const int16_t limit = get_max_safe_int16(head_dim);
            return static_cast<int32_t>(limit) - static_cast<int32_t>(std::abs(val));
        }

    } // namespace vnni_safety
} // namespace llaminar2
