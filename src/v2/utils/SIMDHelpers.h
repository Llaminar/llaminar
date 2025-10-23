#pragma once

/**
 * @file SIMDHelpers.h
 * @brief SIMD helper functions for IQ4_NL quantization decode
 * @author David Sanftenberg
 *
 * Provides AVX512/AVX2 optimized int8→float32 conversion for IQ4_NL lookup table values.
 * Functions are inline and header-only for zero-cost abstraction.
 */

#include "CPUFeatures.h"
#include "../tensors/FP16Utils.h"
#include <cstdint>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace simd
    {

        // Re-export CPU feature detection for compatibility
        using llaminar2::cpu_supports_avx;
        using llaminar2::cpu_supports_avx2;
        using llaminar2::cpu_supports_avx512;
        using llaminar2::cpu_supports_sse41;

        // Re-export FP16 conversion for convenience
        using llaminar2::fp16_to_fp32;
        using llaminar2::fp32_to_fp16;

        // ========================================================================
        // BF16 Conversion Helpers
        // ========================================================================

        /**
         * @brief Convert BF16 to FP32 (scalar)
         *
         * BF16 format: sign(1) + exponent(8) + mantissa(7) = 16 bits
         * FP32 format: sign(1) + exponent(8) + mantissa(23) = 32 bits
         * Conversion: Just shift left by 16 bits (zero-extend mantissa)
         *
         * @param bf16 BF16 value (16-bit)
         * @return FP32 value
         */
        inline float bf16_to_fp32(uint16_t bf16)
        {
            uint32_t fp32_bits = static_cast<uint32_t>(bf16) << 16;
            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        /**
         * @brief Convert FP32 to BF16 (scalar, round-to-nearest-even)
         *
         * @param fp32 FP32 value
         * @return BF16 value (16-bit)
         */
        inline uint16_t fp32_to_bf16(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(float));

            // Round-to-nearest-even: Add 0x7FFF + LSB of result
            uint32_t rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1);
            uint32_t rounded = fp32_bits + rounding_bias;

            return static_cast<uint16_t>(rounded >> 16);
        }

#ifdef __AVX512F__
        /**
         * @brief Convert 32 BF16 values to FP32 using AVX512
         *
         * @param bf16_in Input BF16 array (32 elements)
         * @param fp32_out Output FP32 array (32 elements)
         */
        inline void convert_bf16_to_fp32_avx512(const uint16_t *bf16_in, float *fp32_out)
        {
            // Load 32 BF16 values (512 bits = 64 bytes)
            __m512i bf16_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(bf16_in));

            // Zero-extend 16-bit to 32-bit and shift left by 16
            // Process in two halves (16 elements each)
            __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0);
            __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1);

            __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
            __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);

            // Store as FP32
            _mm512_storeu_ps(fp32_out, _mm512_castsi512_ps(fp32_lo));
            _mm512_storeu_ps(fp32_out + 16, _mm512_castsi512_ps(fp32_hi));
        }
#else
        inline void convert_bf16_to_fp32_avx512(const uint16_t *, float *) {}
#endif

        // ========================================================================
        // AVX-512 Conversion Helpers (16 elements at a time)
        // ========================================================================

#ifdef __AVX512F__

        /**
         * @brief Convert 16 int8 values to float32 with scale (AVX512)
         *
         * Used by IQ4_NL decode: lookup_values[16] (int8) → output[16] (float32)
         * Formula: output[i] = input[i] * scale
         *
         * @param input Pointer to 16 int8 values
         * @param scale Scale factor to multiply
         * @param output Pointer to 16 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_avx512(const int8_t *input, float scale, float *output)
        {
            __m128i i8_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));
            __m512i i32_vec = _mm512_cvtepi8_epi32(i8_vec);
            __m512 f32_vec = _mm512_cvtepi32_ps(i32_vec);
            __m512 result = _mm512_mul_ps(f32_vec, _mm512_set1_ps(scale));
            _mm512_storeu_ps(output, result);
        }

#else
        // Fallback: not available without AVX512
        inline void convert_i8_to_f32_scaled_avx512(const int8_t *, float, float *)
        {
            // Should never be called (guarded by cpu_supports_avx512() checks)
        }
#endif

        // ========================================================================
        // AVX2 Conversion Helpers (8 elements at a time)
        // ========================================================================

#ifdef __AVX2__

        /**
         * @brief Convert 8 int8 values to float32 with scale (AVX2)
         *
         * Used by IQ4_NL decode: lookup_values[8] (int8) → output[8] (float32)
         * Formula: output[i] = input[i] * scale
         *
         * @param input Pointer to 8 int8 values
         * @param scale Scale factor to multiply
         * @param output Pointer to 8 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_avx2(const int8_t *input, float scale, float *output)
        {
            // Load 8 int8 values (8 bytes)
            __m128i i8_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(input));

            // Sign-extend int8 → int32 (8 elements)
            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);

            // Convert int32 → float32
            __m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);

            // Multiply by scale
            __m256 result = _mm256_mul_ps(f32_vec, _mm256_set1_ps(scale));

            // Store result
            _mm256_storeu_ps(output, result);
        }

#else
        // Fallback: not available without AVX2
        inline void convert_i8_to_f32_scaled_avx2(const int8_t *, float, float *)
        {
            // Should never be called (guarded by cpu_supports_avx2() checks)
        }
#endif

        // ========================================================================
        // Scalar Fallback (portable, no SIMD)
        // ========================================================================

        /**
         * @brief Convert N int8 values to float32 with scale (scalar fallback)
         *
         * Portable implementation for platforms without AVX512/AVX2.
         * Used when SIMD is not available or for small counts.
         *
         * @param input Pointer to int8 values
         * @param scale Scale factor to multiply
         * @param output Pointer to float32 output buffer
         * @param count Number of elements to convert
         */
        inline void convert_i8_to_f32_scaled_scalar(const int8_t *input, float scale, float *output, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                output[i] = static_cast<float>(input[i]) * scale;
            }
        }

    } // namespace simd
} // namespace llaminar2
