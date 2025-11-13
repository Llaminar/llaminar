/**
 * @file Q8_0Helpers.h
 * @brief Vectorized Q8_0 quantization helpers (AVX512/AVX2/scalar)
 * @author David Sanftenberg
 *
 * IMPORTANT: Must be included AFTER Tensors.h (requires Q8_0Block definition)
 *
 * Provides SIMD-optimized helpers for converting FP32/BF16/FP16 to Q8_0 format.
 * Used by FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor for to_q8_0() conversions.
 */

#pragma once

#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>

namespace llaminar2
{
    namespace simd
    {
        /**
         * @brief Find maximum absolute value in FP32 array (vectorized)
         *
         * AVX512: Process 16 floats at a time
         * AVX2: Process 8 floats at a time
         * Scalar: Process 1 float at a time
         *
         * @param data FP32 array
         * @param count Number of elements (typically 32 for Q8_0 blocks)
         * @return Maximum absolute value
         */
        inline float find_max_abs_fp32(const float *data, size_t count)
        {
            float max_abs = 0.0f;
            size_t i = 0;

#if defined(__AVX512F__)
            // AVX512: Process 16 floats at a time
            __m512 max_vec = _mm512_setzero_ps();
            constexpr size_t VEC_SIZE = 16;

            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                __m512 vec = _mm512_loadu_ps(data + i);
                __m512 abs_vec = _mm512_abs_ps(vec);
                max_vec = _mm512_max_ps(max_vec, abs_vec);
            }

            // Horizontal max reduction
            alignas(64) float temp[16];
            _mm512_store_ps(temp, max_vec);
            for (int j = 0; j < 16; ++j)
            {
                max_abs = std::max(max_abs, temp[j]);
            }

#elif defined(__AVX2__)
            // AVX2: Process 8 floats at a time
            __m256 max_vec = _mm256_setzero_ps();
            constexpr size_t VEC_SIZE = 8;

            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                __m256 vec = _mm256_loadu_ps(data + i);
                // Absolute value: clear sign bit
                __m256 abs_vec = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vec);
                max_vec = _mm256_max_ps(max_vec, abs_vec);
            }

            // Horizontal max reduction
            alignas(32) float temp[8];
            _mm256_store_ps(temp, max_vec);
            for (int j = 0; j < 8; ++j)
            {
                max_abs = std::max(max_abs, temp[j]);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(data[i]));
            }

            return max_abs;
        }

        /**
         * @brief Quantize FP32 array to INT8 with scale (vectorized)
         *
         * Formula: int8[i] = round(clamp(fp32[i] * inv_scale, -127, 127))
         *
         * AVX512: Process 16 floats → 16 INT8 at once
         * AVX2: Process 8 floats → 8 INT8 at once
         * Scalar: Process 1 at a time
         *
         * @param src Source FP32 array
         * @param dst Destination INT8 array
         * @param inv_scale Inverse scale factor (127.0 / max_abs)
         * @param count Number of elements (≤32 for Q8_0 blocks)
         */
        inline void quantize_fp32_to_int8(const float *src, int8_t *dst, float inv_scale, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__)
            // AVX512: Process 16 floats at once
            constexpr size_t VEC_SIZE = 16;
            const __m512 scale_vec = _mm512_set1_ps(inv_scale);
            const __m512 min_val = _mm512_set1_ps(-127.0f);
            const __m512 max_val = _mm512_set1_ps(127.0f);

            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load and scale
                __m512 vec = _mm512_loadu_ps(src + i);
                vec = _mm512_mul_ps(vec, scale_vec);

                // Clamp to [-127, 127]
                vec = _mm512_max_ps(vec, min_val);
                vec = _mm512_min_ps(vec, max_val);

                // Round to nearest integer
                __m512i int_vec = _mm512_cvt_roundps_epi32(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                // Convert INT32 → INT8 (saturating)
                __m128i int8_vec = _mm512_cvtsepi32_epi8(int_vec);

                // Store 16 bytes
                _mm_storeu_si128((__m128i *)(dst + i), int8_vec);
            }

#elif defined(__AVX2__)
            // AVX2: Process 8 floats at once
            constexpr size_t VEC_SIZE = 8;
            const __m256 scale_vec = _mm256_set1_ps(inv_scale);
            const __m256 min_val = _mm256_set1_ps(-127.0f);
            const __m256 max_val = _mm256_set1_ps(127.0f);

            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load and scale
                __m256 vec = _mm256_loadu_ps(src + i);
                vec = _mm256_mul_ps(vec, scale_vec);

                // Clamp
                vec = _mm256_max_ps(vec, min_val);
                vec = _mm256_min_ps(vec, max_val);

                // Round to nearest integer
                __m256i int_vec = _mm256_cvtps_epi32(_mm256_round_ps(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

                // Convert INT32 → INT16 → INT8 (manual saturation)
                __m128i int16_low = _mm256_castsi256_si128(int_vec);
                __m128i int16_high = _mm256_extracti128_si256(int_vec, 1);
                __m128i int16_packed = _mm_packs_epi32(int16_low, int16_high);
                __m128i int8_packed = _mm_packs_epi16(int16_packed, int16_packed);

                // Store 8 bytes
                _mm_storel_epi64((__m128i *)(dst + i), int8_packed);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                float val = src[i] * inv_scale;
                float clamped = std::max(-127.0f, std::min(127.0f, val));
                dst[i] = static_cast<int8_t>(std::round(clamped));
            }
        }

        /**
         * @brief Quantize FP32 block to Q8_0 format (vectorized)
         *
         * Complete FP32 → Q8_0 block conversion:
         * 1. Find max absolute value (vectorized)
         * 2. Compute FP16 scale
         * 3. Quantize to INT8 (vectorized)
         * 4. Zero-fill tail
         *
         * @param src Source FP32 array (32 elements)
         * @param block Destination Q8_0 block
         * @param count Number of valid elements (≤32)
         */
        inline void quantize_fp32_to_q8_0_block(const float *src, Q8_0Block &block, size_t count)
        {
            // Find max absolute value (vectorized)
            const float max_abs = find_max_abs_fp32(src, count);

            // Compute scale and store as FP16
            // Use 1e-6 threshold to avoid numerical issues with very small values
            // Values smaller than this are effectively zero in FP16 precision
            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            const float scale_fp32 = (max_abs > MIN_SCALE_THRESHOLD) ? (max_abs / 127.0f) : 0.0f;
            block.d = fp32_to_fp16(scale_fp32);

            // Quantize to INT8 (vectorized)
            const float inv_scale = (max_abs > MIN_SCALE_THRESHOLD) ? (127.0f / max_abs) : 0.0f;
            quantize_fp32_to_int8(src, block.qs, inv_scale, count);

            // Zero-fill tail
            for (size_t i = count; i < Q8_0Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }
        }

        /**
         * @brief Quantize BF16 block to Q8_0 format (vectorized)
         *
         * BF16 → Q8_0 conversion with minimal intermediate FP32:
         * 1. Convert BF16 → FP32 (vectorized, in-place during max-abs)
         * 2. Find max absolute value
         * 3. Convert BF16 → FP32 and quantize to INT8 (vectorized)
         *
         * @param src Source BF16 array (32 elements, uint16_t)
         * @param block Destination Q8_0 block
         * @param count Number of valid elements (≤32)
         */
        inline void quantize_bf16_to_q8_0_block(const uint16_t *src, Q8_0Block &block, size_t count)
        {
            // Convert BF16 → FP32 for max-abs and quantization
            alignas(64) float fp32_temp[Q8_0Block::BLOCK_SIZE];
            convert_bf16_to_fp32(src, fp32_temp, count);

            // Use FP32 quantization path
            quantize_fp32_to_q8_0_block(fp32_temp, block, count);
        }

        /**
         * @brief Quantize FP16 block to Q8_0 format (vectorized)
         *
         * FP16 → Q8_0 conversion with minimal intermediate FP32:
         * 1. Convert FP16 → FP32 (vectorized)
         * 2. Use FP32 quantization path
         *
         * @param src Source FP16 array (32 elements, uint16_t)
         * @param block Destination Q8_0 block
         * @param count Number of valid elements (≤32)
         */
        inline void quantize_fp16_to_q8_0_block(const uint16_t *src, Q8_0Block &block, size_t count)
        {
            // Convert FP16 → FP32
            alignas(64) float fp32_temp[Q8_0Block::BLOCK_SIZE];
            convert_fp16_to_fp32(src, fp32_temp, count);

            // Use FP32 quantization path
            quantize_fp32_to_q8_0_block(fp32_temp, block, count);
        }

    } // namespace simd
} // namespace llaminar2
