#pragma once

/**
 * @file SIMDHelpers.h
 * @brief SIMD helper functions for IQ4_NL quantization decode
 * @author David Sanftenberg
 *
 * Provides AVX512/AVX2 optimized int8→float32 conversion for IQ4_NL lookup table values.
 * Functions are inline and header-only for zero-cost abstraction.
 */

#include "BlockStructures.h" // Must be included FIRST (defines all block structures)
#include "IQQuantTables.h"   // Lookup tables: kvalues_iq4nl, iq2xxs_grid, etc.
#include "../utils/CPUFeatures.h"
#include "FP16Utils.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace simd
    {

#if defined(__GNUC__) || defined(__clang__)
        inline void prefetch_read(const void *addr)
        {
            __builtin_prefetch(addr, 0, 1);
        }
#else
        inline void prefetch_read(const void *) {}
#endif

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

        // ========================================================================
        // BF16 → FP32 Array Conversion (SIMD-optimized)
        // ========================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief Convert BF16 array to FP32 using AVX-512 (32 elements per iteration)
         *
         * BF16→FP32 is lossless: just zero-extend mantissa by shifting left 16 bits.
         * No rounding needed, making this simpler than FP32→BF16.
         *
         * Note: Regular stores used instead of streaming stores. Empirical testing showed
         * streaming stores (_mm512_stream_ps) cause 40% performance regression in GEMM
         * workloads despite 15% improvement in pure conversion benchmarks. This is because:
         * - GEMM immediately reads the converted data (cache hit important)
         * - Streaming stores bypass cache, forcing memory reads
         * - Only beneficial for write-only workloads
         *
         * @param src Source BF16 array (must have at least count elements)
         * @param dst Destination FP32 array (must have at least count elements)
         * @param count Number of elements to convert (should be multiple of 32)
         */
        inline void convert_bf16_to_fp32_avx512(const uint16_t *src, float *dst, size_t count)
        {
            for (size_t i = 0; i < count; i += 32)
            {
                // Load 32 BF16 values (512 bits)
                __m512i bf16_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i));

                // Split into two halves for processing (16 elements each)
                __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0); // Lower 16 BF16
                __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1); // Upper 16 BF16

                // Zero-extend 16-bit to 32-bit and shift left by 16
                __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
                __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);

                // Use regular stores (keep in cache for immediate GEMM consumption)
                // AlignedVector provides 64-byte alignment, so we could use _mm512_store_ps,
                // but _mm512_storeu_ps works fine and is safer
                _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(fp32_lo));
                _mm512_storeu_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi));
            }
        }
#else
        inline void convert_bf16_to_fp32_avx512(const uint16_t *, float *, size_t) {}
#endif

#if defined(__AVX2__)
        /**
         * @brief Convert BF16 array to FP32 using AVX2 (16 elements per iteration)
         *
         * BF16→FP32 is lossless: just zero-extend mantissa by shifting left 16 bits.
         *
         * @param src Source BF16 array (must have at least count elements)
         * @param dst Destination FP32 array (must have at least count elements)
         * @param count Number of elements to convert (should be multiple of 16)
         */
        inline void convert_bf16_to_fp32_avx2(const uint16_t *src, float *dst, size_t count)
        {
            for (size_t i = 0; i < count; i += 16)
            {
                // Load 16 BF16 values (256 bits)
                __m256i bf16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));

                // Split into two halves (8 elements each)
                __m128i bf16_lo = _mm256_extracti128_si256(bf16_vec, 0); // Lower 8 BF16
                __m128i bf16_hi = _mm256_extracti128_si256(bf16_vec, 1); // Upper 8 BF16

                // Zero-extend 16-bit to 32-bit and shift left by 16
                __m256i fp32_lo = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_lo), 16);
                __m256i fp32_hi = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_hi), 16);

                // Store as FP32
                _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(fp32_lo));
                _mm256_storeu_ps(dst + i + 8, _mm256_castsi256_ps(fp32_hi));
            }
        }
#else
        inline void convert_bf16_to_fp32_avx2(const uint16_t *, float *, size_t) {}
#endif

        /**
         * @brief Convert BF16 array to FP32 using scalar operations (portable fallback)
         *
         * Uses bf16_to_fp32() for each element.
         * Always available, works on any architecture.
         *
         * @param src Source BF16 array
         * @param dst Destination FP32 array
         * @param count Number of elements to convert
         */
        inline void convert_bf16_to_fp32_scalar(const uint16_t *src, float *dst, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                dst[i] = bf16_to_fp32(src[i]);
            }
        }

        // ========================================================================
        // FP32 → BF16 Array Conversion (SIMD-optimized)
        // ========================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief Convert FP32 array to BF16 using AVX-512 (32 elements per iteration)
         *
         * Uses AVX-512 instructions for maximum throughput:
         * - Processes 32 FP32 values → 32 BF16 values per iteration
         * - Round-to-nearest-even using vpaddw + vpsrlw
         * - ~8x faster than scalar for large arrays
         *
         * @param src Source FP32 array (must have at least count elements)
         * @param dst Destination BF16 array (must have at least count elements)
         * @param count Number of elements to convert (should be multiple of 32)
         */
        inline void convert_fp32_to_bf16_avx512(const float *src, uint16_t *dst, size_t count)
        {
            const __m512i rounding_bias = _mm512_set1_epi32(0x7FFF);

            for (size_t i = 0; i < count; i += 32)
            {
                // Load 32 FP32 values (2 zmm registers)
                __m512 vec0 = _mm512_loadu_ps(src + i);
                __m512 vec1 = _mm512_loadu_ps(src + i + 16);

                // Reinterpret as int32 for bit manipulation
                __m512i int0 = _mm512_castps_si512(vec0);
                __m512i int1 = _mm512_castps_si512(vec1);

                // Add rounding bias + LSB for round-to-nearest-even
                __m512i lsb0 = _mm512_srli_epi32(int0, 16);
                __m512i lsb1 = _mm512_srli_epi32(int1, 16);
                lsb0 = _mm512_and_si512(lsb0, _mm512_set1_epi32(1));
                lsb1 = _mm512_and_si512(lsb1, _mm512_set1_epi32(1));

                __m512i rounded0 = _mm512_add_epi32(int0, rounding_bias);
                __m512i rounded1 = _mm512_add_epi32(int1, rounding_bias);
                rounded0 = _mm512_add_epi32(rounded0, lsb0);
                rounded1 = _mm512_add_epi32(rounded1, lsb1);

                // Shift right to extract BF16 (upper 16 bits)
                __m512i bf16_32bit_0 = _mm512_srli_epi32(rounded0, 16);
                __m512i bf16_32bit_1 = _mm512_srli_epi32(rounded1, 16);

                // Pack 32-bit values to 16-bit (32 BF16 values)
                __m512i packed = _mm512_packus_epi32(bf16_32bit_0, bf16_32bit_1);

                // Permute to correct order and store
                const __m512i perm_idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
                packed = _mm512_permutexvar_epi64(perm_idx, packed);

                _mm512_storeu_si512(reinterpret_cast<__m512i *>(dst + i), packed);
            }
        }
#else
        inline void convert_fp32_to_bf16_avx512(const float *, uint16_t *, size_t) {}
#endif

#if defined(__AVX2__)
        /**
         * @brief Convert FP32 array to BF16 using AVX2 (16 elements per iteration)
         *
         * Uses AVX2 instructions for good throughput on older CPUs:
         * - Processes 16 FP32 values → 16 BF16 values per iteration
         * - Round-to-nearest-even using paddw + psrlw
         * - ~4x faster than scalar
         *
         * @param src Source FP32 array (must have at least count elements)
         * @param dst Destination BF16 array (must have at least count elements)
         * @param count Number of elements to convert (should be multiple of 16)
         */
        inline void convert_fp32_to_bf16_avx2(const float *src, uint16_t *dst, size_t count)
        {
            const __m256i rounding_bias = _mm256_set1_epi32(0x7FFF);

            for (size_t i = 0; i < count; i += 16)
            {
                // Load 16 FP32 values (2 ymm registers)
                __m256 vec0 = _mm256_loadu_ps(src + i);
                __m256 vec1 = _mm256_loadu_ps(src + i + 8);

                // Reinterpret as int32
                __m256i int0 = _mm256_castps_si256(vec0);
                __m256i int1 = _mm256_castps_si256(vec1);

                // Add rounding bias + LSB
                __m256i lsb0 = _mm256_srli_epi32(int0, 16);
                __m256i lsb1 = _mm256_srli_epi32(int1, 16);
                lsb0 = _mm256_and_si256(lsb0, _mm256_set1_epi32(1));
                lsb1 = _mm256_and_si256(lsb1, _mm256_set1_epi32(1));

                __m256i rounded0 = _mm256_add_epi32(int0, rounding_bias);
                __m256i rounded1 = _mm256_add_epi32(int1, rounding_bias);
                rounded0 = _mm256_add_epi32(rounded0, lsb0);
                rounded1 = _mm256_add_epi32(rounded1, lsb1);

                // Shift to extract BF16
                __m256i bf16_32bit_0 = _mm256_srli_epi32(rounded0, 16);
                __m256i bf16_32bit_1 = _mm256_srli_epi32(rounded1, 16);

                // Pack to 16-bit
                __m256i packed = _mm256_packus_epi32(bf16_32bit_0, bf16_32bit_1);

                // Permute lanes to correct order
                packed = _mm256_permute4x64_epi64(packed, 0xD8); // 0b11011000

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), packed);
            }
        }
#else
        inline void convert_fp32_to_bf16_avx2(const float *, uint16_t *, size_t) {}
#endif

        /**
         * @brief Convert FP32 array to BF16 using scalar operations (portable fallback)
         *
         * Uses fp32_to_bf16() for each element.
         * Always available, works on any architecture.
         *
         * @param src Source FP32 array
         * @param dst Destination BF16 array
         * @param count Number of elements to convert
         */
        inline void convert_fp32_to_bf16_scalar(const float *src, uint16_t *dst, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                dst[i] = fp32_to_bf16(src[i]);
            }
        }

        // ========================================================================
        // Activation Packing Helpers (row-wise max/quantize with SIMD fallbacks)
        // ========================================================================

        inline float activation_row_max_abs_scalar(const float *row, int length)
        {
            float max_abs = 0.0f;
            for (int i = 0; i < length; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(row[i]));
            }
            return max_abs;
        }

#if defined(__AVX2__)
        inline float activation_row_max_abs_avx2(const float *row, int length);
#endif

#if defined(__AVX512F__)
        inline float activation_row_max_abs_avx512(const float *row, int length)
        {
            const __m512 sign_mask = _mm512_set1_ps(-0.0f);
            __m512 max_vec0 = _mm512_setzero_ps();
            __m512 max_vec1 = _mm512_setzero_ps();
            int i = 0;

            for (; i + 32 <= length; i += 32)
            {
                prefetch_read(row + i + 64);
                __m512 v0 = _mm512_loadu_ps(row + i);
                __m512 v1 = _mm512_loadu_ps(row + i + 16);
                max_vec0 = _mm512_max_ps(max_vec0, _mm512_andnot_ps(sign_mask, v0));
                max_vec1 = _mm512_max_ps(max_vec1, _mm512_andnot_ps(sign_mask, v1));
            }

            __m512 max_vec = _mm512_max_ps(max_vec0, max_vec1);

            if (i + 16 <= length)
            {
                __m512 v = _mm512_loadu_ps(row + i);
                max_vec = _mm512_max_ps(max_vec, _mm512_andnot_ps(sign_mask, v));
                i += 16;
            }

            float max_abs = _mm512_reduce_max_ps(max_vec);
            const int remaining = length - i;
            if (remaining > 0)
            {
#if defined(__AVX2__)
                if (cpu_supports_avx2())
                {
                    const float tail = activation_row_max_abs_avx2(row + i, remaining);
                    return std::max(max_abs, tail);
                }
#endif
                for (; i < length; ++i)
                {
                    max_abs = std::max(max_abs, std::fabs(row[i]));
                }
            }
            return max_abs;
        }
#endif

#if defined(__AVX2__)
        inline float activation_row_max_abs_horizontal_max(__m256 vec)
        {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, vec);
            float max_abs = 0.0f;
            for (float val : tmp)
            {
                max_abs = std::max(max_abs, val);
            }
            return max_abs;
        }

        inline float activation_row_max_abs_avx2(const float *row, int length)
        {
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 max_vec = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= length; i += 8)
            {
                prefetch_read(row + i + 32);
                __m256 v = _mm256_loadu_ps(row + i);
                __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
                max_vec = _mm256_max_ps(max_vec, abs_v);
            }

            float max_abs = activation_row_max_abs_horizontal_max(max_vec);
            for (; i < length; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(row[i]));
            }
            return max_abs;
        }
#endif

        inline float activation_row_max_abs(const float *row, int length)
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                return activation_row_max_abs_avx512(row, length);
            }
#endif
#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                return activation_row_max_abs_avx2(row, length);
            }
#endif
            return activation_row_max_abs_scalar(row, length);
        }

        inline void quantize_activation_row_scalar(const float *src, int length, float inv_scale, int8_t *dst)
        {
            if (inv_scale == 0.0f)
            {
                std::memset(dst, 0, static_cast<size_t>(length));
                return;
            }

            for (int i = 0; i < length; ++i)
            {
                const float q = std::round(src[i] * inv_scale);
                const float clamped = std::min(127.0f, std::max(-127.0f, q));
                dst[i] = static_cast<int8_t>(clamped);
            }
        }

#if defined(__AVX2__)
        inline void quantize_activation_row_avx2(const float *src, int length, float inv_scale, int8_t *dst);
#endif

#if defined(__AVX512F__)
        inline void quantize_activation_row_avx512(const float *src, int length, float inv_scale, int8_t *dst)
        {
            if (inv_scale == 0.0f)
            {
                std::memset(dst, 0, static_cast<size_t>(length));
                return;
            }

            const __m512 scale_vec = _mm512_set1_ps(inv_scale);
            const __m512i max_vec = _mm512_set1_epi32(127);
            const __m512i min_vec = _mm512_set1_epi32(-127);
            int i = 0;
            alignas(64) int32_t tmp0[16];
            alignas(64) int32_t tmp1[16];

            for (; i + 32 <= length; i += 32)
            {
                prefetch_read(src + i + 64);
                __m512 vals0 = _mm512_loadu_ps(src + i);
                __m512 vals1 = _mm512_loadu_ps(src + i + 16);

                __m512 scaled0 = _mm512_mul_ps(vals0, scale_vec);
                __m512 scaled1 = _mm512_mul_ps(vals1, scale_vec);

                __m512i rounded0 = _mm512_cvtps_epi32(scaled0);
                __m512i rounded1 = _mm512_cvtps_epi32(scaled1);

                __m512i clamped0 = _mm512_max_epi32(min_vec, _mm512_min_epi32(max_vec, rounded0));
                __m512i clamped1 = _mm512_max_epi32(min_vec, _mm512_min_epi32(max_vec, rounded1));

                _mm512_store_si512(reinterpret_cast<__m512i *>(tmp0), clamped0);
                _mm512_store_si512(reinterpret_cast<__m512i *>(tmp1), clamped1);

                for (int lane = 0; lane < 16; ++lane)
                {
                    dst[i + lane] = static_cast<int8_t>(tmp0[lane]);
                    dst[i + lane + 16] = static_cast<int8_t>(tmp1[lane]);
                }
            }

            if (i + 16 <= length)
            {
                alignas(64) int32_t tmp_tail[16];
                __m512 vals = _mm512_loadu_ps(src + i);
                __m512 scaled = _mm512_mul_ps(vals, scale_vec);
                __m512i rounded = _mm512_cvtps_epi32(scaled);
                __m512i clamped = _mm512_max_epi32(min_vec, _mm512_min_epi32(max_vec, rounded));
                _mm512_store_si512(reinterpret_cast<__m512i *>(tmp_tail), clamped);
                for (int lane = 0; lane < 16; ++lane)
                {
                    dst[i + lane] = static_cast<int8_t>(tmp_tail[lane]);
                }
                i += 16;
            }

            const int remaining = length - i;
            if (remaining > 0)
            {
#if defined(__AVX2__)
                if (cpu_supports_avx2())
                {
                    quantize_activation_row_avx2(src + i, remaining, inv_scale, dst + i);
                    return;
                }
#endif
                quantize_activation_row_scalar(src + i, remaining, inv_scale, dst + i);
            }
        }
#endif

#if defined(__AVX2__)
        inline void quantize_activation_row_avx2(const float *src, int length, float inv_scale, int8_t *dst)
        {
            if (inv_scale == 0.0f)
            {
                std::memset(dst, 0, static_cast<size_t>(length));
                return;
            }

            const __m256 scale_vec = _mm256_set1_ps(inv_scale);
            const __m256i max_vec = _mm256_set1_epi32(127);
            const __m256i min_vec = _mm256_set1_epi32(-127);
            int i = 0;
            alignas(32) int32_t tmp[8];

            for (; i + 8 <= length; i += 8)
            {
                prefetch_read(src + i + 32);
                __m256 vals = _mm256_loadu_ps(src + i);
                __m256 scaled = _mm256_mul_ps(vals, scale_vec);
                __m256i rounded = _mm256_cvtps_epi32(scaled);
                __m256i clamped = _mm256_max_epi32(min_vec, _mm256_min_epi32(max_vec, rounded));
                _mm256_store_si256(reinterpret_cast<__m256i *>(tmp), clamped);
                for (int lane = 0; lane < 8; ++lane)
                {
                    dst[i + lane] = static_cast<int8_t>(tmp[lane]);
                }
            }

            const __m128 scale_vec128 = _mm_set1_ps(inv_scale);
            const __m128i max_vec128 = _mm_set1_epi32(127);
            const __m128i min_vec128 = _mm_set1_epi32(-127);
            alignas(16) int32_t tmp4[4];
            for (; i + 4 <= length; i += 4)
            {
                __m128 vals = _mm_loadu_ps(src + i);
                __m128 scaled = _mm_mul_ps(vals, scale_vec128);
                __m128i rounded = _mm_cvtps_epi32(scaled);
                __m128i clamped = _mm_max_epi32(min_vec128, _mm_min_epi32(max_vec128, rounded));
                _mm_store_si128(reinterpret_cast<__m128i *>(tmp4), clamped);
                for (int lane = 0; lane < 4; ++lane)
                {
                    dst[i + lane] = static_cast<int8_t>(tmp4[lane]);
                }
            }

            for (; i < length; ++i)
            {
                const float q = std::round(src[i] * inv_scale);
                const float clamped = std::min(127.0f, std::max(-127.0f, q));
                dst[i] = static_cast<int8_t>(clamped);
            }
        }
#endif

        inline void quantize_activation_row(const float *src, int length, float inv_scale, int8_t *dst)
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                quantize_activation_row_avx512(src, length, inv_scale, dst);
                return;
            }
#endif
#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                quantize_activation_row_avx2(src, length, inv_scale, dst);
                return;
            }
#endif
            quantize_activation_row_scalar(src, length, inv_scale, dst);
        }

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

        // ========================================================================
        // INT32 → FP32 Requantization Helpers
        // ========================================================================

        inline void requantize_int32_row_to_fp32_scalar(
            const int32_t *src,
            float *dst,
            int cols,
            float row_scale,
            const float *col_scales,
            const float *bias)
        {
            for (int c = 0; c < cols; ++c)
            {
                const float col_scale = col_scales ? col_scales[c] : 1.0f;
                float value = static_cast<float>(src[c]) * row_scale * col_scale;
                if (bias)
                {
                    value += bias[c];
                }
                dst[c] = value;
            }
        }

        inline void requantize_int32_row_to_fp32(
            const int32_t *src,
            float *dst,
            int cols,
            float row_scale,
            const float *col_scales,
            const float *bias)
        {
            int c = 0;

#if defined(__AVX512F__)
            const __m512 vrow = _mm512_set1_ps(row_scale);
            const __m512 vone512 = _mm512_set1_ps(1.0f);
            for (; c + 16 <= cols; c += 16)
            {
                __m512i vsrc = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + c));
                __m512 vfp = _mm512_cvtepi32_ps(vsrc);
                __m512 vcol = col_scales ? _mm512_loadu_ps(col_scales + c) : vone512;
                __m512 vout = _mm512_mul_ps(vfp, vcol);
                vout = _mm512_mul_ps(vout, vrow);
                if (bias)
                {
                    vout = _mm512_add_ps(vout, _mm512_loadu_ps(bias + c));
                }
                _mm512_storeu_ps(dst + c, vout);
            }
#endif

#if defined(__AVX2__)
            const __m256 vrow256 = _mm256_set1_ps(row_scale);
            const __m256 vone256 = _mm256_set1_ps(1.0f);
            for (; c + 8 <= cols; c += 8)
            {
                __m256i vsrc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + c));
                __m256 vfp = _mm256_cvtepi32_ps(vsrc);
                __m256 vcol = col_scales ? _mm256_loadu_ps(col_scales + c) : vone256;
                __m256 vout = _mm256_mul_ps(vfp, vcol);
                vout = _mm256_mul_ps(vout, vrow256);
                if (bias)
                {
                    vout = _mm256_add_ps(vout, _mm256_loadu_ps(bias + c));
                }
                _mm256_storeu_ps(dst + c, vout);
            }
#endif

#if defined(__SSE2__)
            const __m128 vrow128 = _mm_set1_ps(row_scale);
            const __m128 vone128 = _mm_set1_ps(1.0f);
            for (; c + 4 <= cols; c += 4)
            {
                __m128i vsrc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + c));
                __m128 vfp = _mm_cvtepi32_ps(vsrc);
                __m128 vcol = col_scales ? _mm_loadu_ps(col_scales + c) : vone128;
                __m128 vout = _mm_mul_ps(vfp, vcol);
                vout = _mm_mul_ps(vout, vrow128);
                if (bias)
                {
                    vout = _mm_add_ps(vout, _mm_loadu_ps(bias + c));
                }
                _mm_storeu_ps(dst + c, vout);
            }
#endif

            const float *col_tail = col_scales ? (col_scales + c) : nullptr;
            const float *bias_tail = bias ? (bias + c) : nullptr;
            requantize_int32_row_to_fp32_scalar(src + c, dst + c, cols - c, row_scale, col_tail, bias_tail);
        }

        inline void requantize_int32_matrix_to_fp32(
            const int32_t *src,
            float *dst,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias)
        {
#pragma omp parallel for
            for (int r = 0; r < rows; ++r)
            {
                const float row_scale = row_scales ? row_scales[r] : 1.0f;
                const size_t offset = static_cast<size_t>(r) * static_cast<size_t>(cols);
                requantize_int32_row_to_fp32(
                    src + offset,
                    dst + offset,
                    cols,
                    row_scale,
                    col_scales,
                    bias);
            }
        }

        // ========================================================================
        // High-Level Auto-Dispatch Wrappers
        // ========================================================================

        /**
         * @brief Convert BF16 array to FP32 (auto-dispatched to best ISA)
         *
         * Automatically selects:
         *   - AVX512 if available (32 elements/iter)
         *   - AVX2 if available (16 elements/iter)
         *   - Scalar fallback (1 element/iter)
         *
         * @param src Source BF16 array (uint16_t*)
         * @param dst Destination FP32 array (float*)
         * @param count Number of elements to convert
         */
        inline void convert_bf16_to_fp32(const uint16_t *src, float *dst, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
            // AVX512: Process 32 elements at a time
            constexpr size_t VEC_SIZE = 32;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_bf16_to_fp32_avx512(src + i, dst + i, VEC_SIZE);
            }
#elif defined(__AVX2__)
            // AVX2: Process 16 elements at a time
            constexpr size_t VEC_SIZE = 16;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_bf16_to_fp32_avx2(src + i, dst + i, VEC_SIZE);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                dst[i] = bf16_to_fp32(src[i]);
            }
        }

        /**
         * @brief Convert FP32 array to BF16 (auto-dispatched to best ISA)
         *
         * Automatically selects:
         *   - AVX512 if available (32 elements/iter)
         *   - AVX2 if available (16 elements/iter)
         *   - Scalar fallback (1 element/iter)
         *
         * @param src Source FP32 array (float*)
         * @param dst Destination BF16 array (uint16_t*)
         * @param count Number of elements to convert
         */
        inline void convert_fp32_to_bf16(const float *src, uint16_t *dst, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
            // AVX512: Process 32 elements at a time
            constexpr size_t VEC_SIZE = 32;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp32_to_bf16_avx512(src + i, dst + i, VEC_SIZE);
            }
#elif defined(__AVX2__)
            // AVX2: Process 16 elements at a time
            constexpr size_t VEC_SIZE = 16;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp32_to_bf16_avx2(src + i, dst + i, VEC_SIZE);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                dst[i] = fp32_to_bf16(src[i]);
            }
        }

        // ====================================================================
        // FP16 ↔ FP32 Conversion (Vectorized with Auto-Dispatch)
        // ====================================================================

        /**
         * @brief Convert FP16 array to FP32 using AVX512/AVX2/F16C (vectorized)
         *
         * Automatically dispatches to best available ISA:
         *   - AVX512F: 16 elements/iteration (native _mm512_cvtph_ps)
         *   - AVX2+F16C: 8 elements/iteration (_mm256_cvtph_ps)
         *   - Scalar fallback: 1 element/iteration
         *
         * @param src Source FP16 array (uint16_t*)
         * @param dst Destination FP32 array (float*)
         * @param count Number of elements to convert
         */
        inline void convert_fp16_to_fp32_avx512(const uint16_t *src, float *dst, size_t count)
        {
#if defined(__AVX512F__)
            constexpr size_t VEC_SIZE = 16; // 16 FP16s → 16 FP32s
            for (size_t i = 0; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load 16 FP16 values (256 bits)
                __m256i fp16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));

                // Convert FP16 → FP32 (native AVX512 instruction)
                __m512 fp32_vec = _mm512_cvtph_ps(fp16_vec);

                // Store 16 FP32 values
                _mm512_storeu_ps(dst + i, fp32_vec);
            }
#else
            (void)src;
            (void)dst;
            (void)count; // Suppress unused parameter warnings
#endif
        }

        inline void convert_fp16_to_fp32_avx2(const uint16_t *src, float *dst, size_t count)
        {
#if defined(__AVX2__) && defined(__F16C__)
            constexpr size_t VEC_SIZE = 8; // 8 FP16s → 8 FP32s
            for (size_t i = 0; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load 8 FP16 values (128 bits)
                __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));

                // Convert FP16 → FP32 (F16C instruction)
                __m256 fp32_vec = _mm256_cvtph_ps(fp16_vec);

                // Store 8 FP32 values
                _mm256_storeu_ps(dst + i, fp32_vec);
            }
#else
            (void)src;
            (void)dst;
            (void)count; // Suppress unused parameter warnings
#endif
        }

        /**
         * @brief Convert FP32 array to FP16 using AVX512/AVX2/F16C (vectorized)
         *
         * Automatically dispatches to best available ISA:
         *   - AVX512F: 16 elements/iteration (native _mm512_cvtps_ph)
         *   - AVX2+F16C: 8 elements/iteration (_mm256_cvtps_ph)
         *   - Scalar fallback: 1 element/iteration
         *
         * @param src Source FP32 array (float*)
         * @param dst Destination FP16 array (uint16_t*)
         * @param count Number of elements to convert
         */
        inline void convert_fp32_to_fp16_avx512(const float *src, uint16_t *dst, size_t count)
        {
#if defined(__AVX512F__)
            constexpr size_t VEC_SIZE = 16; // 16 FP32s → 16 FP16s
            for (size_t i = 0; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load 16 FP32 values
                __m512 fp32_vec = _mm512_loadu_ps(src + i);

                // Convert FP32 → FP16 (native AVX512 instruction, round to nearest)
                __m256i fp16_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                // Store 16 FP16 values (256 bits)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), fp16_vec);
            }
#else
            (void)src;
            (void)dst;
            (void)count; // Suppress unused parameter warnings
#endif
        }

        inline void convert_fp32_to_fp16_avx2(const float *src, uint16_t *dst, size_t count)
        {
#if defined(__AVX2__) && defined(__F16C__)
            constexpr size_t VEC_SIZE = 8; // 8 FP32s → 8 FP16s
            for (size_t i = 0; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load 8 FP32 values
                __m256 fp32_vec = _mm256_loadu_ps(src + i);

                // Convert FP32 → FP16 (F16C instruction, round to nearest)
                __m128i fp16_vec = _mm256_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                // Store 8 FP16 values (128 bits)
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), fp16_vec);
            }
#else
            (void)src;
            (void)dst;
            (void)count; // Suppress unused parameter warnings
#endif
        }

        /**
         * @brief Convert FP16 array to FP32 (auto-dispatched to best ISA)
         *
         * Automatically selects:
         *   - AVX512F if available (16 elements/iter)
         *   - AVX2+F16C if available (8 elements/iter)
         *   - Scalar fallback (1 element/iter)
         *
         * @param src Source FP16 array (uint16_t*)
         * @param dst Destination FP32 array (float*)
         * @param count Number of elements to convert
         */
        inline void convert_fp16_to_fp32(const uint16_t *src, float *dst, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__)
            // AVX512: Process 16 elements at a time
            constexpr size_t VEC_SIZE = 16;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp16_to_fp32_avx512(src + i, dst + i, VEC_SIZE);
            }
#elif defined(__AVX2__) && defined(__F16C__)
            // AVX2+F16C: Process 8 elements at a time
            constexpr size_t VEC_SIZE = 8;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp16_to_fp32_avx2(src + i, dst + i, VEC_SIZE);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                dst[i] = fp16_to_fp32(src[i]);
            }
        }

        /**
         * @brief Convert FP32 array to FP16 (auto-dispatched to best ISA)
         *
         * Automatically selects:
         *   - AVX512F if available (16 elements/iter)
         *   - AVX2+F16C if available (8 elements/iter)
         *   - Scalar fallback (1 element/iter)
         *
         * @param src Source FP32 array (float*)
         * @param dst Destination FP16 array (uint16_t*)
         * @param count Number of elements to convert
         */
        inline void convert_fp32_to_fp16(const float *src, uint16_t *dst, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__)
            // AVX512: Process 16 elements at a time
            constexpr size_t VEC_SIZE = 16;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp32_to_fp16_avx512(src + i, dst + i, VEC_SIZE);
            }
#elif defined(__AVX2__) && defined(__F16C__)
            // AVX2+F16C: Process 8 elements at a time
            constexpr size_t VEC_SIZE = 8;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp32_to_fp16_avx2(src + i, dst + i, VEC_SIZE);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                dst[i] = fp32_to_fp16(src[i]);
            }
        }

        // ==========================================
        // FP32 to Q8_0 Quantization (for TensorBase::to_q8_0())
        // ==========================================

        /**
         * @brief Quantize 32 FP32 values to Q8_0 format (scalar)
         * @param src Source FP32 values (must be at least 32 elements)
         * @param count Number of elements (ignored - always processes 32 for Q8_0)
         * @param dst_qs Destination for int8 quantized values
         * @param dst_scale_fp16 Destination for FP16 scale
         */
        inline void quantize_fp32_to_q8_0_scalar(const float *src, size_t count, int8_t *dst_qs, uint16_t *dst_scale_fp16)
        {
            // Q8_0 blocks are always 32 elements - ignore count parameter
            (void)count;

            // Find max absolute value
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Use threshold to avoid numerical issues with very small values
            // Values smaller than 1e-6 are effectively zero in FP16 precision
            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;

            // Handle zero input or extremely small values - treat as zero
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                *dst_scale_fp16 = 0;
                std::memset(dst_qs, 0, 32);
                return;
            }

            // Calculate scale (clamp to avoid overflow in FP16)
            float scale = max_abs / 127.0f;
            if (scale > 65504.0f)
            { // Max FP16 value
                scale = 65504.0f;
            }
            float inv_scale = 1.0f / scale;

            // Quantize to int8
            for (int i = 0; i < 32; ++i)
            {
                float val = src[i] * inv_scale;
                dst_qs[i] = static_cast<int8_t>(std::round(std::max(-127.0f, std::min(127.0f, val))));
            }

            // Store FP16 scale
            *dst_scale_fp16 = fp32_to_fp16(scale);
        }

        /**
         * @brief Quantize 32 FP32 values to Q8_0 format (AVX-512)
         */
        inline void quantize_fp32_to_q8_0_avx512(const float *src, size_t count, int8_t *dst_qs, uint16_t *dst_scale_fp16)
        {
#if defined(__AVX512F__)
            (void)count; // Always 32 for Q8_0

            // Load 32 FP32 values
            __m512 v0 = _mm512_loadu_ps(src);
            __m512 v1 = _mm512_loadu_ps(src + 16);

            // Compute absolute values
            __m512 abs0 = _mm512_abs_ps(v0);
            __m512 abs1 = _mm512_abs_ps(v1);

            // Find max across both vectors
            __m512 max_vec = _mm512_max_ps(abs0, abs1);
            float max_abs = _mm512_reduce_max_ps(max_vec);

            // Use threshold to avoid numerical issues with very small values
            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;

            // Handle zero input or extremely small values - treat as zero
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                *dst_scale_fp16 = 0;
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_qs), _mm256_setzero_si256());
                return;
            }

            // Calculate scale
            float scale = max_abs / 127.0f;
            if (scale > 65504.0f)
            {
                scale = 65504.0f;
            }
            float inv_scale = 1.0f / scale;
            __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);

            // Quantize: multiply by inv_scale, round, clamp to [-127, 127]
            __m512 scaled0 = _mm512_mul_ps(v0, inv_scale_vec);
            __m512 scaled1 = _mm512_mul_ps(v1, inv_scale_vec);

            // Round to nearest
            __m512 rounded0 = _mm512_roundscale_ps(scaled0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512 rounded1 = _mm512_roundscale_ps(scaled1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // Clamp to [-127, 127]
            __m512 min_val = _mm512_set1_ps(-127.0f);
            __m512 max_val = _mm512_set1_ps(127.0f);
            __m512 clamped0 = _mm512_min_ps(_mm512_max_ps(rounded0, min_val), max_val);
            __m512 clamped1 = _mm512_min_ps(_mm512_max_ps(rounded1, min_val), max_val);

            // Convert to int32
            __m512i i32_0 = _mm512_cvtps_epi32(clamped0);
            __m512i i32_1 = _mm512_cvtps_epi32(clamped1);

            // Convert int32 → int8 (pack 16+16 → 32 int8)
            __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
            __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);
            __m256i i8_result = _mm256_packs_epi16(i16_0, i16_1);

            // Permute to correct order (packs interleaves 128-bit lanes)
            const __m256i perm_idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
            i8_result = _mm256_permutevar8x32_epi32(i8_result, perm_idx);

            // Store result
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_qs), i8_result);
            *dst_scale_fp16 = fp32_to_fp16(scale);
#else
            quantize_fp32_to_q8_0_scalar(src, count, dst_qs, dst_scale_fp16);
#endif
        }

        /**
         * @brief Quantize 32 FP32 values to Q8_0 format (AVX2)
         */
        inline void quantize_fp32_to_q8_0_avx2(const float *src, size_t count, int8_t *dst_qs, uint16_t *dst_scale_fp16)
        {
#ifdef __AVX2__
            (void)count; // Always 32 for Q8_0

            // Load 32 FP32 values (4 x 8 elements)
            __m256 v0 = _mm256_loadu_ps(src);
            __m256 v1 = _mm256_loadu_ps(src + 8);
            __m256 v2 = _mm256_loadu_ps(src + 16);
            __m256 v3 = _mm256_loadu_ps(src + 24);

            // Compute absolute values
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 abs0 = _mm256_andnot_ps(sign_mask, v0);
            __m256 abs1 = _mm256_andnot_ps(sign_mask, v1);
            __m256 abs2 = _mm256_andnot_ps(sign_mask, v2);
            __m256 abs3 = _mm256_andnot_ps(sign_mask, v3);

            // Find max across all vectors
            __m256 max01 = _mm256_max_ps(abs0, abs1);
            __m256 max23 = _mm256_max_ps(abs2, abs3);
            __m256 max_all = _mm256_max_ps(max01, max23);

            // Horizontal max within 256-bit vector
            __m128 max_high = _mm256_extractf128_ps(max_all, 1);
            __m128 max_low = _mm256_castps256_ps128(max_all);
            __m128 max_hl = _mm_max_ps(max_high, max_low);
            max_hl = _mm_max_ps(max_hl, _mm_movehl_ps(max_hl, max_hl));
            max_hl = _mm_max_ss(max_hl, _mm_movehdup_ps(max_hl));
            float max_abs = _mm_cvtss_f32(max_hl);

            // Use threshold to avoid numerical issues with very small values
            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;

            // Handle zero input or extremely small values - treat as zero
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                *dst_scale_fp16 = 0;
                std::memset(dst_qs, 0, 32);
                return;
            }

            // Calculate scale
            float scale = max_abs / 127.0f;
            if (scale > 65504.0f)
            {
                scale = 65504.0f;
            }
            float inv_scale = 1.0f / scale;
            __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);

            // Quantize: multiply, round, clamp
            __m256 scaled0 = _mm256_mul_ps(v0, inv_scale_vec);
            __m256 scaled1 = _mm256_mul_ps(v1, inv_scale_vec);
            __m256 scaled2 = _mm256_mul_ps(v2, inv_scale_vec);
            __m256 scaled3 = _mm256_mul_ps(v3, inv_scale_vec);

            // Round to nearest
            __m256 rounded0 = _mm256_round_ps(scaled0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 rounded1 = _mm256_round_ps(scaled1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 rounded2 = _mm256_round_ps(scaled2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 rounded3 = _mm256_round_ps(scaled3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // Clamp to [-127, 127]
            __m256 min_val = _mm256_set1_ps(-127.0f);
            __m256 max_val = _mm256_set1_ps(127.0f);
            __m256 clamped0 = _mm256_min_ps(_mm256_max_ps(rounded0, min_val), max_val);
            __m256 clamped1 = _mm256_min_ps(_mm256_max_ps(rounded1, min_val), max_val);
            __m256 clamped2 = _mm256_min_ps(_mm256_max_ps(rounded2, min_val), max_val);
            __m256 clamped3 = _mm256_min_ps(_mm256_max_ps(rounded3, min_val), max_val);

            // Convert to int32
            __m256i i32_0 = _mm256_cvtps_epi32(clamped0);
            __m256i i32_1 = _mm256_cvtps_epi32(clamped1);
            __m256i i32_2 = _mm256_cvtps_epi32(clamped2);
            __m256i i32_3 = _mm256_cvtps_epi32(clamped3);

            // Store as int32 arrays and pack manually (AVX2 pack is tricky with lane ordering)
            int32_t temp_i32[32];
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&temp_i32[0]), i32_0);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&temp_i32[8]), i32_1);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&temp_i32[16]), i32_2);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&temp_i32[24]), i32_3);

            // Pack to int8
            for (int i = 0; i < 32; ++i)
            {
                dst_qs[i] = static_cast<int8_t>(temp_i32[i]);
            }

            *dst_scale_fp16 = fp32_to_fp16(scale);
#else
            quantize_fp32_to_q8_0_scalar(src, count, dst_qs, dst_scale_fp16);
#endif
        }

        /**
         * @brief Quantize 32 FP32 values to Q8_0 format (auto-dispatch)
         */
        inline void quantize_fp32_to_q8_0(const float *src, size_t count, int8_t *dst_qs, uint16_t *dst_scale_fp16)
        {
            const float *src_ptr = src;
            float temp_src[32];

            // Handle partial blocks by copying to temporary buffer
            if (count < 32)
            {
                std::memset(temp_src, 0, sizeof(temp_src));
                std::memcpy(temp_src, src, count * sizeof(float));
                src_ptr = temp_src;
            }

#if defined(__AVX512F__)
            quantize_fp32_to_q8_0_avx512(src_ptr, 32, dst_qs, dst_scale_fp16);
#elif defined(__AVX2__)
            quantize_fp32_to_q8_0_avx2(src_ptr, 32, dst_qs, dst_scale_fp16);
#else
            quantize_fp32_to_q8_0_scalar(src_ptr, 32, dst_qs, dst_scale_fp16);
#endif
        }

        // ==========================================
        // Q8_1 Quantization (with pre-computed sum)
        // ==========================================

        /**
         * @brief Quantize 32 FP32 values to Q8_1 format with pre-computed sum (scalar)
         * @param src Source FP32 values (must be at least 32 elements)
         * @param count Number of elements (ignored - always processes 32 for Q8_1)
         * @param dst_qs Destination for int8 quantized values
         * @param dst_scale_fp16 Destination for FP16 scale (d)
         * @param dst_sum_fp16 Destination for FP16 pre-computed sum (s = d × sum(values))
         */
        inline void quantize_fp32_to_q8_1_scalar(const float *src, size_t count, int8_t *dst_qs, uint16_t *dst_scale_fp16, uint16_t *dst_sum_fp16)
        {
            // Q8_1 blocks are always 32 elements - ignore count parameter
            (void)count;

            // Find max absolute value
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Use threshold to avoid numerical issues with very small values
            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;

            // Handle zero input or extremely small values
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                *dst_scale_fp16 = 0;
                *dst_sum_fp16 = 0;
                std::memset(dst_qs, 0, 32);
                return;
            }

            // Calculate scale (clamp to avoid overflow in FP16)
            float scale = max_abs / 127.0f;
            if (scale > 65504.0f)
            { // Max FP16 value
                scale = 65504.0f;
            }
            float inv_scale = 1.0f / scale;

            // Quantize to int8 AND compute sum simultaneously
            int32_t sum_i32 = 0; // Nov 2024: Store raw integer sum, not scaled FP sum!
            for (int i = 0; i < 32; ++i)
            {
                float val = src[i];
                float scaled = val * inv_scale;
                dst_qs[i] = static_cast<int8_t>(std::round(std::max(-127.0f, std::min(127.0f, scaled))));
                sum_i32 += static_cast<int32_t>(dst_qs[i]); // Sum quantized int8 values (CRITICAL!)
            }

            // Store FP16 scale and pre-computed INT16 sum
            *dst_scale_fp16 = fp32_to_fp16(scale);
            *dst_sum_fp16 = static_cast<uint16_t>(static_cast<int16_t>(sum_i32)); // Nov 2024: raw sum, not d × sum!
        }

        /**
         * @brief Quantize 32 FP32 values to Q8_1 format with pre-computed sum (AVX-512)
         */
        inline void quantize_fp32_to_q8_1_avx512(const float *src, size_t count, int8_t *dst_qs, uint16_t *dst_scale_fp16, uint16_t *dst_sum_fp16)
        {
#if defined(__AVX512F__)
            // Find max absolute value using AVX-512
            __m512 vmax0 = _mm512_abs_ps(_mm512_loadu_ps(src));
            __m512 vmax1 = _mm512_abs_ps(_mm512_loadu_ps(src + 16));
            __m512 vmax = _mm512_max_ps(vmax0, vmax1);
            float max_abs = _mm512_reduce_max_ps(vmax);

            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                *dst_scale_fp16 = 0;
                *dst_sum_fp16 = 0;
                std::memset(dst_qs, 0, 32);
                return;
            }

            // Calculate scale
            float scale = max_abs / 127.0f;
            if (scale > 65504.0f)
            {
                scale = 65504.0f;
            }
            const float inv_scale = 1.0f / scale;
            const __m512 vinv_scale = _mm512_set1_ps(inv_scale);

            // Load and quantize first 16 elements
            __m512 v0 = _mm512_loadu_ps(src);
            __m512 vscaled0 = _mm512_mul_ps(v0, vinv_scale);

            // Load and quantize second 16 elements
            __m512 v1 = _mm512_loadu_ps(src + 16);
            __m512 vscaled1 = _mm512_mul_ps(v1, vinv_scale);

            // Convert to int32 with rounding and clamping
            __m512i vi32_0 = _mm512_cvtps_epi32(vscaled0);
            __m512i vi32_1 = _mm512_cvtps_epi32(vscaled1);

            // Clamp to [-127, 127]
            const __m512i vmin = _mm512_set1_epi32(-127);
            const __m512i vmax_val = _mm512_set1_epi32(127);
            vi32_0 = _mm512_max_epi32(vi32_0, vmin);
            vi32_0 = _mm512_min_epi32(vi32_0, vmax_val);
            vi32_1 = _mm512_max_epi32(vi32_1, vmin);
            vi32_1 = _mm512_min_epi32(vi32_1, vmax_val);

            // Pack int32 → int16 → int8
            // Note: We need to properly pack 16+16 int32 → 32 int8
            __m256i vi16_0 = _mm512_cvtepi32_epi16(vi32_0); // 16 int32 → 16 int16 (indices 0-15)
            __m256i vi16_1 = _mm512_cvtepi32_epi16(vi32_1); // 16 int32 → 16 int16 (indices 16-31)

            // Pack 16 int16 → 16 int8 for each half
            // vi16_0: [0-7 (low), 8-15 (high)] → pack together for indices 0-15
            // vi16_1: [16-23 (low), 24-31 (high)] → pack together for indices 16-31
            __m128i vi8_first16 = _mm_packs_epi16(_mm256_castsi256_si128(vi16_0), _mm256_extracti128_si256(vi16_0, 1));
            __m128i vi8_second16 = _mm_packs_epi16(_mm256_castsi256_si128(vi16_1), _mm256_extracti128_si256(vi16_1, 1));

            // Store indices 0-15 and 16-31 sequentially
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_qs), vi8_first16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_qs + 16), vi8_second16);

            // Compute sum of QUANTIZED int8 values as INT32 (CRITICAL!)
            // Horizontal reduction of int32 vectors
            int32_t sum_i32 = _mm512_reduce_add_epi32(vi32_0) + _mm512_reduce_add_epi32(vi32_1);

            // Store scale and pre-computed INT16 sum
            *dst_scale_fp16 = fp32_to_fp16(scale);
            *dst_sum_fp16 = static_cast<uint16_t>(static_cast<int16_t>(sum_i32)); // Nov 2024: raw sum, not d × sum!
#else
            quantize_fp32_to_q8_1_scalar(src, count, dst_qs, dst_scale_fp16, dst_sum_fp16);
#endif
        }

        /**
         * @brief Quantize 32 FP32 values to Q8_1 format with pre-computed sum (AVX2)
         */
        inline void quantize_fp32_to_q8_1_avx2(const float *src, size_t count, int8_t *dst_qs, uint16_t *dst_scale_fp16, uint16_t *dst_sum_fp16)
        {
#ifdef __AVX2__
            // Find max absolute value using AVX2
            __m256 vmax0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(src));
            __m256 vmax1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(src + 8));
            __m256 vmax2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(src + 16));
            __m256 vmax3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(src + 24));

            vmax0 = _mm256_max_ps(vmax0, vmax1);
            vmax2 = _mm256_max_ps(vmax2, vmax3);
            __m256 vmax = _mm256_max_ps(vmax0, vmax2);

            // Horizontal max reduction
            __m128 vmax_hi = _mm256_extractf128_ps(vmax, 1);
            __m128 vmax_lo = _mm256_castps256_ps128(vmax);
            vmax_lo = _mm_max_ps(vmax_lo, vmax_hi);
            vmax_lo = _mm_max_ps(vmax_lo, _mm_movehl_ps(vmax_lo, vmax_lo));
            vmax_lo = _mm_max_ps(vmax_lo, _mm_shuffle_ps(vmax_lo, vmax_lo, 1));
            float max_abs = _mm_cvtss_f32(vmax_lo);

            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                *dst_scale_fp16 = 0;
                *dst_sum_fp16 = 0;
                std::memset(dst_qs, 0, 32);
                return;
            }

            // Calculate scale
            float scale = max_abs / 127.0f;
            if (scale > 65504.0f)
            {
                scale = 65504.0f;
            }
            const float inv_scale = 1.0f / scale;
            const __m256 vinv_scale = _mm256_set1_ps(inv_scale);

            // Quantize AND compute sum (4 iterations for 32 elements)
            int32_t sum_i32 = 0; // Nov 2024: Store raw integer sum, not scaled FP sum!
            for (int i = 0; i < 4; ++i)
            {
                __m256 vf = _mm256_loadu_ps(src + i * 8);
                __m256 vscaled = _mm256_mul_ps(vf, vinv_scale);

                // Convert to int32 with rounding
                __m256i vi32 = _mm256_cvtps_epi32(vscaled);

                // Clamp to [-127, 127]
                __m256i vmin = _mm256_set1_epi32(-127);
                __m256i vmax_val = _mm256_set1_epi32(127);
                vi32 = _mm256_max_epi32(vi32, vmin);
                vi32 = _mm256_min_epi32(vi32, vmax_val);

                // Pack to int8 (manual packing for 8 elements)
                alignas(32) int32_t temp[8];
                _mm256_store_si256(reinterpret_cast<__m256i *>(temp), vi32);
                for (int j = 0; j < 8; ++j)
                {
                    dst_qs[i * 8 + j] = static_cast<int8_t>(temp[j]);
                    sum_i32 += static_cast<int32_t>(dst_qs[i * 8 + j]); // Sum quantized int8 (CRITICAL!)
                }
            }

            // Store scale and pre-computed INT16 sum
            *dst_scale_fp16 = fp32_to_fp16(scale);
            *dst_sum_fp16 = static_cast<uint16_t>(static_cast<int16_t>(sum_i32)); // Nov 2024: raw sum, not d × sum!
#else
            quantize_fp32_to_q8_1_scalar(src, count, dst_qs, dst_scale_fp16, dst_sum_fp16);
#endif
        }

        /**
         * @brief Quantize 32 FP32 values to Q8_1 format with pre-computed sum (auto-dispatch)
         */
        inline void quantize_fp32_to_q8_1(const float *src, size_t count, int8_t *dst_qs, uint16_t *dst_scale_fp16, uint16_t *dst_sum_fp16)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q8_1_avx512(src, count, dst_qs, dst_scale_fp16, dst_sum_fp16);
#elif defined(__AVX2__)
            quantize_fp32_to_q8_1_avx2(src, count, dst_qs, dst_scale_fp16, dst_sum_fp16);
#else
            quantize_fp32_to_q8_1_scalar(src, count, dst_qs, dst_scale_fp16, dst_sum_fp16);
#endif
        }

        // ==========================================
        // Q6_K to Q8_0 Decode (32-element sub-blocks)
        // ==========================================

        /**
         * @brief Decode one Q6_K sub-block (32 elements) to Q8_0 format (scalar)
         */
        inline void decode_q6_k_to_q8_0_scalar(
            const Q6_KBlock &q6k_block,
            size_t sub_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (sub_idx > 7)
            {
                throw std::out_of_range("decode_q6_k_to_q8_0_scalar: sub_idx must be 0-7");
            }

            // Decode Q6_K → FP32 (using existing logic from Q6_KTensor)
            float fp32_values[32];

            // Hierarchical scaling: super-block d, then sub-block scale
            const float d = fp16_to_fp32(q6k_block.d);
            const int8_t sc = q6k_block.scales[sub_idx];
            const float block_scale = d * sc;

            const uint8_t *ql = q6k_block.ql + (sub_idx * 16); // 16 bytes for 32× 4-bit low
            const uint8_t *qh = q6k_block.qh + (sub_idx * 8);  // 8 bytes for 32× 2-bit high

            for (int j = 0; j < 32; ++j)
            {
                const int byte_idx = j / 2;
                const int shift = (j % 2) * 4;
                const uint8_t q4_low = (ql[byte_idx] >> shift) & 0x0F;

                const int qh_byte = j / 4;
                const int qh_shift = (j % 4) * 2;
                const uint8_t q2_high = (qh[qh_byte] >> qh_shift) & 0x03;

                const uint8_t q6 = (q2_high << 4) | q4_low;
                const int8_t q_signed = static_cast<int8_t>(q6) - 32;

                fp32_values[j] = block_scale * q_signed;
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_values, 32, q8_qs, q8_scale);
        }

        /**
         * @brief Q6_K to Q8_0 decode (AVX2 - currently aliased to scalar)
         */
        inline void decode_q6_k_to_q8_0_avx2(
            const Q6_KBlock &q6k_block,
            size_t sub_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            decode_q6_k_to_q8_0_scalar(q6k_block, sub_idx, q8_qs, q8_scale);
        }

        /**
         * @brief Q6_K to Q8_0 decode (AVX-512 - currently aliased to scalar)
         */
        inline void decode_q6_k_to_q8_0_avx512(
            const Q6_KBlock &q6k_block,
            size_t sub_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            decode_q6_k_to_q8_0_scalar(q6k_block, sub_idx, q8_qs, q8_scale);
        }

        /**
         * @brief Q6_K to Q8_0 decode (auto-dispatch wrapper)
         */
        inline void decode_q6_k_to_q8_0(
            const Q6_KBlock &q6k_block,
            size_t sub_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            decode_q6_k_to_q8_0_scalar(q6k_block, sub_idx, q8_qs, q8_scale);
        }

        // ==========================================
        // Stub decode functions for other formats
        // ==========================================

        /**
         * @brief Decode Q4_0 block to Q8_0 format (scalar)
         * Q4_0: 32 elements, 4-bit quantization, 1 scale
         */
        inline void decode_q4_0_to_q8_0_scalar(
            const uint8_t *qs,      // 16 bytes: packed 4-bit values
            uint16_t d_fp16,        // FP16 scale
            int8_t *output_qs,      // 32 bytes: Q8_0 quantized output
            uint16_t *output_d_fp16 // FP16 scale for Q8_0
        )
        {
            // Decode 4-bit → FP32 (32 elements)
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            for (int i = 0; i < 16; ++i)
            {
                // Extract low and high nibbles, shift to [-8, 7] range
                int8_t v0 = static_cast<int8_t>((qs[i] & 0xF) - 8);
                int8_t v1 = static_cast<int8_t>((qs[i] >> 4) - 8);

                fp32_vals[2 * i] = v0 * scale;
                fp32_vals[2 * i + 1] = v1 * scale;
            }

            // Quantize to Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_vals, 32, output_qs, output_d_fp16);
        }

        /**
         * @brief Decode Q4_0 block to Q8_0 format (AVX-512)
         */
        inline void decode_q4_0_to_q8_0_avx512(
            const uint8_t *qs,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__)
            // Decode to FP32 (scalar - nibble extraction is complex to vectorize)
            // The speedup comes from quantize_fp32_to_q8_0_avx512
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            for (int i = 0; i < 16; ++i)
            {
                int8_t v0 = (qs[i] & 0xF) - 8;
                int8_t v1 = (qs[i] >> 4) - 8;
                fp32_vals[2 * i] = v0 * scale;
                fp32_vals[2 * i + 1] = v1 * scale;
            }

            // Quantize to Q8_0 using AVX-512
            quantize_fp32_to_q8_0_avx512(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_q4_0_to_q8_0_scalar(qs, d_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q4_0 block to Q8_0 format (AVX2)
         */
        inline void decode_q4_0_to_q8_0_avx2(
            const uint8_t *qs,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#ifdef __AVX2__
            // Decode to FP32 (scalar - nibble extraction is complex to vectorize)
            // The speedup comes from quantize_fp32_to_q8_0_avx2
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            for (int i = 0; i < 16; ++i)
            {
                int8_t v0 = (qs[i] & 0xF) - 8;
                int8_t v1 = (qs[i] >> 4) - 8;
                fp32_vals[2 * i] = v0 * scale;
                fp32_vals[2 * i + 1] = v1 * scale;
            }

            // Quantize to Q8_0 using AVX2
            quantize_fp32_to_q8_0_avx2(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_q4_0_to_q8_0_scalar(qs, d_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q4_0 block to Q8_0 format (auto-dispatch)
         */
        inline void decode_q4_0_to_q8_0(
            const uint8_t *qs,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_q4_0_to_q8_0_avx512(qs, d_fp16, output_qs, output_d_fp16);
#elif defined(__AVX2__)
            decode_q4_0_to_q8_0_avx2(qs, d_fp16, output_qs, output_d_fp16);
#else
            decode_q4_0_to_q8_0_scalar(qs, d_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q4_1 block to Q8_0 format (scalar)
         * Q4_1: 32 elements, 4-bit quantization, 1 scale + 1 offset
         */
        inline void decode_q4_1_to_q8_0_scalar(
            const uint8_t *qs,      // 16 bytes: packed 4-bit values
            uint16_t d_fp16,        // FP16 scale
            uint16_t m_fp16,        // FP16 offset (min value)
            int8_t *output_qs,      // 32 bytes: Q8_0 quantized output
            uint16_t *output_d_fp16 // FP16 scale for Q8_0
        )
        {
            // Decode 4-bit → FP32 with offset (32 elements)
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);
            float offset = fp16_to_fp32(m_fp16);

            for (int i = 0; i < 16; ++i)
            {
                // Extract nibbles [0,15] (no subtraction like Q4_0)
                uint8_t v0 = qs[i] & 0x0F;
                uint8_t v1 = qs[i] >> 4;

                fp32_vals[2 * i] = v0 * scale + offset;
                fp32_vals[2 * i + 1] = v1 * scale + offset;
            }

            // Quantize to Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_vals, 32, output_qs, output_d_fp16);
        }

        /**
         * @brief Decode Q4_1 block to Q8_0 format (AVX-512)
         */
        inline void decode_q4_1_to_q8_0_avx512(
            const uint8_t *qs,
            uint16_t d_fp16,
            uint16_t m_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__)
            // Decode to FP32 (scalar - nibble extraction is complex to vectorize)
            // The speedup comes from quantize_fp32_to_q8_0_avx512
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);
            float offset = fp16_to_fp32(m_fp16);

            for (int i = 0; i < 16; ++i)
            {
                uint8_t v0 = qs[i] & 0xF; // [0, 15]
                uint8_t v1 = qs[i] >> 4;  // [0, 15]
                fp32_vals[2 * i] = v0 * scale + offset;
                fp32_vals[2 * i + 1] = v1 * scale + offset;
            }

            // Quantize to Q8_0 using AVX-512
            quantize_fp32_to_q8_0_avx512(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_q4_1_to_q8_0_scalar(qs, d_fp16, m_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q4_1 block to Q8_0 format (AVX2)
         */
        inline void decode_q4_1_to_q8_0_avx2(
            const uint8_t *qs,
            uint16_t d_fp16,
            uint16_t m_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#ifdef __AVX2__
            // Decode to FP32 (scalar - nibble extraction is complex to vectorize)
            // The speedup comes from quantize_fp32_to_q8_0_avx2
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);
            float offset = fp16_to_fp32(m_fp16);

            for (int i = 0; i < 16; ++i)
            {
                uint8_t v0 = qs[i] & 0xF; // [0, 15]
                uint8_t v1 = qs[i] >> 4;  // [0, 15]
                fp32_vals[2 * i] = v0 * scale + offset;
                fp32_vals[2 * i + 1] = v1 * scale + offset;
            }

            // Quantize to Q8_0 using AVX2
            quantize_fp32_to_q8_0_avx2(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_q4_1_to_q8_0_scalar(qs, d_fp16, m_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q4_1 block to Q8_0 format (auto-dispatch)
         */
        inline void decode_q4_1_to_q8_0(
            const uint8_t *qs,
            uint16_t d_fp16,
            uint16_t m_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_q4_1_to_q8_0_avx512(qs, d_fp16, m_fp16, output_qs, output_d_fp16);
#elif defined(__AVX2__)
            decode_q4_1_to_q8_0_avx2(qs, d_fp16, m_fp16, output_qs, output_d_fp16);
#else
            decode_q4_1_to_q8_0_scalar(qs, d_fp16, m_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q5_0 block to Q8_0 format (scalar)
         * Q5_0: 32 elements, 5-bit quantization (4 low + 1 high), 1 scale
         */
        inline void decode_q5_0_to_q8_0_scalar(
            const uint8_t *qs, // 16 bytes: 4 low bits
            const uint8_t *qh, // 4 bytes: 1 high bit per element
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            for (int i = 0; i < 32; ++i)
            {
                // Extract 4 low bits from qs
                uint8_t nibble = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;

                // Extract 1 high bit from qh
                uint8_t high_bit = (qh[i / 8] >> (i % 8)) & 1;

                // Combine: 5-bit value in [-16, 15]
                int8_t val = static_cast<int8_t>((nibble | (high_bit << 4)) - 16);

                fp32_vals[i] = val * scale;
            }

            quantize_fp32_to_q8_0_scalar(fp32_vals, 32, output_qs, output_d_fp16);
        }

        /**
         * @brief Decode Q5_0 block to Q8_0 format (AVX-512)
         */
        inline void decode_q5_0_to_q8_0_avx512(
            const uint8_t *qs,
            const uint8_t *qh,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            // Scalar unpacking (simpler and avoids lane ordering bugs)
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            for (int i = 0; i < 32; ++i)
            {
                uint8_t nibble = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
                uint8_t high_bit = (qh[i / 8] >> (i % 8)) & 1;
                int8_t val = static_cast<int8_t>((nibble | (high_bit << 4)) - 16);
                fp32_vals[i] = val * scale;
            }

            quantize_fp32_to_q8_0_avx512(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_q5_0_to_q8_0_scalar(qs, qh, d_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q5_0 block to Q8_0 format (AVX2)
         */
        inline void decode_q5_0_to_q8_0_avx2(
            const uint8_t *qs,
            const uint8_t *qh,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#ifdef __AVX2__
            // Scalar unpacking (simpler and avoids lane ordering bugs)
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            for (int i = 0; i < 32; ++i)
            {
                uint8_t nibble = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
                uint8_t high_bit = (qh[i / 8] >> (i % 8)) & 1;
                int8_t val = static_cast<int8_t>((nibble | (high_bit << 4)) - 16);
                fp32_vals[i] = val * scale;
            }

            quantize_fp32_to_q8_0_avx2(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_q5_0_to_q8_0_scalar(qs, qh, d_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q5_0 block to Q8_0 format (auto-dispatch)
         */
        inline void decode_q5_0_to_q8_0(
            const uint8_t *qs,
            const uint8_t *qh,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_q5_0_to_q8_0_avx512(qs, qh, d_fp16, output_qs, output_d_fp16);
#elif defined(__AVX2__)
            decode_q5_0_to_q8_0_avx2(qs, qh, d_fp16, output_qs, output_d_fp16);
#else
            decode_q5_0_to_q8_0_scalar(qs, qh, d_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q5_1 block to Q8_0 format (scalar)
         * Q5_1: 32 elements, 5-bit quantization (4 low + 1 high), 1 scale + 1 offset
         */
        inline void decode_q5_1_to_q8_0_scalar(
            const uint8_t *qs, // 16 bytes: 4 low bits
            const uint8_t *qh, // 4 bytes: 1 high bit per element
            uint16_t d_fp16,
            uint16_t m_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);
            float offset = fp16_to_fp32(m_fp16);

            for (int i = 0; i < 32; ++i)
            {
                // Extract 4 low bits
                uint8_t nibble = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;

                // Extract 1 high bit
                uint8_t high_bit = (qh[i / 8] >> (i % 8)) & 1;

                // Combine: 5-bit value in [0, 31] (no -16 offset like Q5_0)
                uint8_t val = nibble | (high_bit << 4);

                fp32_vals[i] = val * scale + offset;
            }

            quantize_fp32_to_q8_0_scalar(fp32_vals, 32, output_qs, output_d_fp16);
        }

        /**
         * @brief Decode Q5_1 block to Q8_0 format (AVX-512)
         */
        inline void decode_q5_1_to_q8_0_avx512(
            const uint8_t *qs,
            const uint8_t *qh,
            uint16_t d_fp16,
            uint16_t m_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            // Scalar unpacking (simpler and avoids lane ordering bugs)
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);
            float offset = fp16_to_fp32(m_fp16);

            for (int i = 0; i < 32; ++i)
            {
                uint8_t nibble = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
                uint8_t high_bit = (qh[i / 8] >> (i % 8)) & 1;
                uint8_t val = nibble | (high_bit << 4);
                fp32_vals[i] = val * scale + offset;
            }

            quantize_fp32_to_q8_0_avx512(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_q5_1_to_q8_0_scalar(qs, qh, d_fp16, m_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q5_1 block to Q8_0 format (AVX2)
         */
        inline void decode_q5_1_to_q8_0_avx2(
            const uint8_t *qs,
            const uint8_t *qh,
            uint16_t d_fp16,
            uint16_t m_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#ifdef __AVX2__
            // Scalar unpacking (simpler and avoids lane ordering bugs)
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);
            float offset = fp16_to_fp32(m_fp16);

            for (int i = 0; i < 32; ++i)
            {
                uint8_t nibble = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
                uint8_t high_bit = (qh[i / 8] >> (i % 8)) & 1;
                uint8_t val = nibble | (high_bit << 4);
                fp32_vals[i] = val * scale + offset;
            }

            quantize_fp32_to_q8_0_avx2(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_q5_1_to_q8_0_scalar(qs, qh, d_fp16, m_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode Q5_1 block to Q8_0 format (auto-dispatch)
         */
        inline void decode_q5_1_to_q8_0(
            const uint8_t *qs,
            const uint8_t *qh,
            uint16_t d_fp16,
            uint16_t m_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_q5_1_to_q8_0_avx512(qs, qh, d_fp16, m_fp16, output_qs, output_d_fp16);
#elif defined(__AVX2__)
            decode_q5_1_to_q8_0_avx2(qs, qh, d_fp16, m_fp16, output_qs, output_d_fp16);
#else
            decode_q5_1_to_q8_0_scalar(qs, qh, d_fp16, m_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode IQ4_NL block to Q8_0 format (scalar)
         * IQ4_NL: 32 elements, 4-bit indices → lookup table → scale
         */
        inline void decode_iq4nl_to_q8_0_scalar(
            const uint8_t *qs, // 16 bytes: packed 4-bit indices
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
            // IQ4_NL lookup table (from IQQuantTables.h)
            static constexpr float kvalues_iq4nl[16] = {
                -127.0f, -104.0f, -83.0f, -65.0f,
                -49.0f, -35.0f, -22.0f, -10.0f,
                1.0f, 13.0f, 25.0f, 38.0f,
                53.0f, 69.0f, 89.0f, 113.0f};

            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            // Decode 32 values via lookup table
            for (int i = 0; i < 16; ++i)
            {
                uint8_t qbyte = qs[i];
                uint8_t idx_low = qbyte & 0x0F;
                uint8_t idx_high = qbyte >> 4;

                fp32_vals[i] = scale * kvalues_iq4nl[idx_low];
                fp32_vals[i + 16] = scale * kvalues_iq4nl[idx_high];
            }

            quantize_fp32_to_q8_0_scalar(fp32_vals, 32, output_qs, output_d_fp16);
        }

        /**
         * @brief Decode IQ4_NL block to Q8_0 format (AVX-512)
         */
        inline void decode_iq4nl_to_q8_0_avx512(
            const uint8_t *qs,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            // IQ4_NL lookup table
            static constexpr float kvalues_iq4nl[16] = {
                -127.0f, -104.0f, -83.0f, -65.0f,
                -49.0f, -35.0f, -22.0f, -10.0f,
                1.0f, 13.0f, 25.0f, 38.0f,
                53.0f, 69.0f, 89.0f, 113.0f};

            // Scalar unpacking with lookup table
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            for (int i = 0; i < 16; ++i)
            {
                uint8_t qbyte = qs[i];
                uint8_t idx_low = qbyte & 0x0F;
                uint8_t idx_high = qbyte >> 4;

                fp32_vals[i] = scale * kvalues_iq4nl[idx_low];
                fp32_vals[i + 16] = scale * kvalues_iq4nl[idx_high];
            }

            quantize_fp32_to_q8_0_avx512(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_iq4nl_to_q8_0_scalar(qs, d_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode IQ4_NL block to Q8_0 format (AVX2)
         */
        inline void decode_iq4nl_to_q8_0_avx2(
            const uint8_t *qs,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#ifdef __AVX2__
            // IQ4_NL lookup table
            static constexpr float kvalues_iq4nl[16] = {
                -127.0f, -104.0f, -83.0f, -65.0f,
                -49.0f, -35.0f, -22.0f, -10.0f,
                1.0f, 13.0f, 25.0f, 38.0f,
                53.0f, 69.0f, 89.0f, 113.0f};

            // Scalar unpacking with lookup table
            float fp32_vals[32];
            float scale = fp16_to_fp32(d_fp16);

            for (int i = 0; i < 16; ++i)
            {
                uint8_t qbyte = qs[i];
                uint8_t idx_low = qbyte & 0x0F;
                uint8_t idx_high = qbyte >> 4;

                fp32_vals[i] = scale * kvalues_iq4nl[idx_low];
                fp32_vals[i + 16] = scale * kvalues_iq4nl[idx_high];
            }

            quantize_fp32_to_q8_0_avx2(fp32_vals, 32, output_qs, output_d_fp16);
#else
            decode_iq4nl_to_q8_0_scalar(qs, d_fp16, output_qs, output_d_fp16);
#endif
        }

        /**
         * @brief Decode IQ4_NL block to Q8_0 format (auto-dispatch)
         */
        inline void decode_iq4nl_to_q8_0(
            const uint8_t *qs,
            uint16_t d_fp16,
            int8_t *output_qs,
            uint16_t *output_d_fp16)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_iq4nl_to_q8_0_avx512(qs, d_fp16, output_qs, output_d_fp16);
#elif defined(__AVX2__)
            decode_iq4nl_to_q8_0_avx2(qs, d_fp16, output_qs, output_d_fp16);
#else
            decode_iq4nl_to_q8_0_scalar(qs, d_fp16, output_qs, output_d_fp16);
#endif
        }

        // =====================================================================
        // IQ4_XS → Q8_0 (256-element super-blocks with per-sub-block scales)
        // =====================================================================

        /**
         * @brief Scalar decode IQ4_XS to Q8_0 (one 32-element sub-block from 256-element super-block)
         * @param block IQ4_XS super-block (256 elements, 136 bytes)
         * @param subblock_idx Which 32-element sub-block to decode (0-7)
         * @param q8_qs Output Q8_0 quantized values (32 int8_t)
         * @param q8_scale Output Q8_0 scale (FP16)
         */
        inline void decode_iq4xs_to_q8_0_scalar(
            const IQ4_XSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq4xs_to_q8_0_scalar: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 16; // 16 bytes = 32 nibbles

            // Extract 6-bit scale: 4 bits from scales_l + 2 bits from scales_h
            const size_t ib = subblock_idx;
            const int ls = ((block.scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                           (((block.scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);

            // Decode 16 pairs of 4-bit values using kvalues_iq4nl lookup
            for (size_t j = 0; j < 16; ++j)
            {
                tmp[j] = dl * kvalues_iq4nl[qs[j] & 0xf];
                tmp[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief AVX-512 decode IQ4_XS to Q8_0
         */
        inline void decode_iq4xs_to_q8_0_avx512(
            const IQ4_XSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq4xs_to_q8_0_avx512: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 16;

            // Extract 6-bit scale
            const size_t ib = subblock_idx;
            const int ls = ((block.scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                           (((block.scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);

            // Scalar unpacking + SIMD quantization pattern
            for (size_t j = 0; j < 16; ++j)
            {
                tmp[j] = dl * kvalues_iq4nl[qs[j] & 0xf];
                tmp[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }

            quantize_fp32_to_q8_0_avx512(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

#ifdef __AVX2__
        /**
         * @brief AVX2 decode IQ4_XS to Q8_0
         */
        inline void decode_iq4xs_to_q8_0_avx2(
            const IQ4_XSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq4xs_to_q8_0_avx2: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 16;

            // Extract 6-bit scale
            const size_t ib = subblock_idx;
            const int ls = ((block.scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                           (((block.scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);

            // Scalar unpacking + SIMD quantization
            for (size_t j = 0; j < 16; ++j)
            {
                tmp[j] = dl * kvalues_iq4nl[qs[j] & 0xf];
                tmp[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }

            quantize_fp32_to_q8_0_avx2(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

        /**
         * @brief Auto-dispatch IQ4_XS to Q8_0 decode
         */
        inline void decode_iq4xs_to_q8_0(
            const IQ4_XSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_iq4xs_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
#elif defined(__AVX2__)
            decode_iq4xs_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
#else
            decode_iq4xs_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
#endif
        }

        // =====================================================================
        // IQ2_XXS → Q8_0 (256-element super-blocks with grid lookup)
        // =====================================================================

        /**
         * @brief Scalar decode IQ2_XXS to Q8_0 (one 32-element sub-block from 256-element super-block)
         * @param block IQ2_XXS super-block (256 elements, 66 bytes)
         * @param subblock_idx Which 32-element sub-block to decode (0-7)
         * @param q8_qs Output Q8_0 quantized values (32 int8_t)
         * @param q8_scale Output Q8_0 scale (FP16)
         */
        inline void decode_iq2xxs_to_q8_0_scalar(
            const IQ2_XXSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq2xxs_to_q8_0_scalar: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);

            // Extract 8 bytes (4 uint16_t) for this sub-block
            uint32_t aux32[2];
            const uint8_t *aux8 = reinterpret_cast<const uint8_t *>(aux32);
            std::memcpy(aux32, &block.qs[4 * subblock_idx], 2 * sizeof(uint32_t));

            // Sub-block scale: d * (0.5 + (aux32[1] >> 28)) * 0.25
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;

            // Decode 4 groups of 8 elements using grid lookup
            for (size_t l = 0; l < 4; ++l)
            {
                const uint64_t grid_value = iq2xxs_grid[aux8[l]];
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(&grid_value);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];

                for (size_t j = 0; j < 8; ++j)
                {
                    tmp[l * 8 + j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
                }
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief AVX-512 decode IQ2_XXS to Q8_0
         */
        inline void decode_iq2xxs_to_q8_0_avx512(
            const IQ2_XXSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq2xxs_to_q8_0_avx512: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);

            uint32_t aux32[2];
            const uint8_t *aux8 = reinterpret_cast<const uint8_t *>(aux32);
            std::memcpy(aux32, &block.qs[4 * subblock_idx], 2 * sizeof(uint32_t));

            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;

            // Scalar unpacking + SIMD quantization
            for (size_t l = 0; l < 4; ++l)
            {
                const uint64_t grid_value = iq2xxs_grid[aux8[l]];
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(&grid_value);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];

                for (size_t j = 0; j < 8; ++j)
                {
                    tmp[l * 8 + j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
                }
            }

            quantize_fp32_to_q8_0_avx512(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

#ifdef __AVX2__
        /**
         * @brief AVX2 decode IQ2_XXS to Q8_0
         */
        inline void decode_iq2xxs_to_q8_0_avx2(
            const IQ2_XXSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq2xxs_to_q8_0_avx2: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);

            uint32_t aux32[2];
            const uint8_t *aux8 = reinterpret_cast<const uint8_t *>(aux32);
            std::memcpy(aux32, &block.qs[4 * subblock_idx], 2 * sizeof(uint32_t));

            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;

            // Scalar unpacking + SIMD quantization
            for (size_t l = 0; l < 4; ++l)
            {
                const uint64_t grid_value = iq2xxs_grid[aux8[l]];
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(&grid_value);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];

                for (size_t j = 0; j < 8; ++j)
                {
                    tmp[l * 8 + j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
                }
            }

            quantize_fp32_to_q8_0_avx2(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

        /**
         * @brief Auto-dispatch IQ2_XXS to Q8_0 decode
         */
        inline void decode_iq2xxs_to_q8_0(
            const IQ2_XXSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            if (cpu_supports_avx512())
            {
                decode_iq2xxs_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                decode_iq2xxs_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

            decode_iq2xxs_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
        }

        /**
         * @brief Unpack IQ2_XXS sub-block to int8 (transcoding)
         *
         * Wrapper around decode_iq2xxs_to_q8_0 for IINT8Unpackable interface.
         */
        inline void unpack_iq2_xxs_to_int8(const IQ2_XXSBlock &block, size_t subblock_idx, int8_t *output)
        {
            uint16_t dummy_scale;
            decode_iq2xxs_to_q8_0(block, subblock_idx, output, &dummy_scale);
        }

        /**
         * @brief Get IQ2_XXS sub-block scale (transcoding)
         *
         * Wrapper around decode_iq2xxs_to_q8_0 for IINT8Unpackable interface.
         */
        inline float get_iq2_xxs_scale(const IQ2_XXSBlock &block, size_t subblock_idx)
        {
            int8_t dummy_qs[32];
            uint16_t scale_fp16;
            decode_iq2xxs_to_q8_0(block, subblock_idx, dummy_qs, &scale_fp16);
            return fp16_to_fp32(scale_fp16);
        }

        // ===================================================================
        // IQ2_XS: 2-bit extra-small IQ with grid lookup and scales
        // ===================================================================
        // Algorithm: 256-element super-block → 8 sub-blocks of 32 elements
        // Each sub-block: scale from scales[] array (two 4-bit scales per byte)
        // 4 groups of 8 elements: 9-bit grid lookup + 7-bit sign pattern
        // qs[]: 16-bit values, lower 9 bits = grid index, upper 7 bits = sign index

        /**
         * @brief Decode one IQ2_XS sub-block to Q8_0 format (scalar reference)
         * @param block IQ2_XS super-block (256 elements)
         * @param subblock_idx Sub-block index within super-block (0-7)
         * @param q8_qs Output: 32 quantized int8 values
         * @param q8_scale Output: Q8_0 scale factor (FP16 format)
         */
        inline void decode_iq2xs_to_q8_0_scalar(const IQ2_XSBlock &block, size_t subblock_idx,
                                                int8_t *q8_qs, uint16_t *q8_scale)
        {
            // Decode to FP32 intermediate buffer
            float fp32_buffer[32];

            const float d = fp16_to_fp32(block.d);
            const size_t ib32 = subblock_idx; // Each sub-block is 32 elements

            // Two sub-scales per iteration (4-bit each from scales byte)
            const float db0 = d * (0.5f + (block.scales[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (block.scales[ib32] >> 4)) * 0.25f;

            // Process 4 groups of 8 elements
            for (size_t l = 0; l < 4; ++l)
            {
                const size_t qs_idx = 4 * ib32 + l;
                const uint16_t qs_val = block.qs[qs_idx];

                // Extract 9-bit grid index (lower 9 bits)
                const uint16_t grid_idx = qs_val & 511;
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + grid_idx);

                // Extract 7-bit sign pattern (upper 7 bits)
                const uint8_t signs = ksigns_iq2xs[qs_val >> 9];

                // Select scale (db0 for l=0,1, db1 for l=2,3)
                const float db = (l < 2) ? db0 : db1;

                // Decode 8 elements with grid lookup and sign application
                for (size_t j = 0; j < 8; ++j)
                {
                    fp32_buffer[l * 8 + j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
                }
            }

            // Quantize FP32 buffer to Q8_0 format
            quantize_fp32_to_q8_0_scalar(fp32_buffer, 32, q8_qs, q8_scale);
        }

#ifdef __AVX2__
        /**
         * @brief Decode one IQ2_XS sub-block to Q8_0 format (AVX2 optimized)
         */
        inline void decode_iq2xs_to_q8_0_avx2(const IQ2_XSBlock &block, size_t subblock_idx,
                                              int8_t *q8_qs, uint16_t *q8_scale)
        {
            // Decode to FP32 intermediate buffer
            alignas(32) float fp32_buffer[32];

            const float d = fp16_to_fp32(block.d);
            const size_t ib32 = subblock_idx;

            // Two sub-scales per iteration
            const float db0 = d * (0.5f + (block.scales[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (block.scales[ib32] >> 4)) * 0.25f;

            // Process 4 groups of 8 elements
            for (size_t l = 0; l < 4; ++l)
            {
                const size_t qs_idx = 4 * ib32 + l;
                const uint16_t qs_val = block.qs[qs_idx];

                const uint16_t grid_idx = qs_val & 511;
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + grid_idx);
                const uint8_t signs = ksigns_iq2xs[qs_val >> 9];
                const float db = (l < 2) ? db0 : db1;

                // Load grid values and broadcast scale
                __m256 grid_vec = _mm256_set_ps(
                    grid[7], grid[6], grid[5], grid[4],
                    grid[3], grid[2], grid[1], grid[0]);
                __m256 scale_vec = _mm256_set1_ps(db);
                __m256 values = _mm256_mul_ps(grid_vec, scale_vec);

                // Apply signs using mask
                __m256 sign_mask = _mm256_set_ps(
                    (signs & kmask_iq2xs[7]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[6]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[5]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[4]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[3]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[2]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[1]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[0]) ? -1.0f : 1.0f);
                values = _mm256_mul_ps(values, sign_mask);

                _mm256_store_ps(&fp32_buffer[l * 8], values);
            }

            // Quantize FP32 buffer to Q8_0 format using AVX2
            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

#ifdef __AVX512F__
        /**
         * @brief Decode one IQ2_XS sub-block to Q8_0 format (AVX-512 optimized)
         */
        inline void decode_iq2xs_to_q8_0_avx512(const IQ2_XSBlock &block, size_t subblock_idx,
                                                int8_t *q8_qs, uint16_t *q8_scale)
        {
            // Decode to FP32 intermediate buffer
            alignas(64) float fp32_buffer[32];

            const float d = fp16_to_fp32(block.d);
            const size_t ib32 = subblock_idx;

            const float db0 = d * (0.5f + (block.scales[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (block.scales[ib32] >> 4)) * 0.25f;

            // Process 4 groups of 8 elements
            for (size_t l = 0; l < 4; ++l)
            {
                const size_t qs_idx = 4 * ib32 + l;
                const uint16_t qs_val = block.qs[qs_idx];

                const uint16_t grid_idx = qs_val & 511;
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + grid_idx);
                const uint8_t signs = ksigns_iq2xs[qs_val >> 9];
                const float db = (l < 2) ? db0 : db1;

                // Load grid values using AVX-512 (8 elements)
                __m256 grid_vec = _mm256_set_ps(
                    grid[7], grid[6], grid[5], grid[4],
                    grid[3], grid[2], grid[1], grid[0]);
                __m256 scale_vec = _mm256_set1_ps(db);
                __m256 values = _mm256_mul_ps(grid_vec, scale_vec);

                // Apply signs
                __m256 sign_mask = _mm256_set_ps(
                    (signs & kmask_iq2xs[7]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[6]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[5]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[4]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[3]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[2]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[1]) ? -1.0f : 1.0f,
                    (signs & kmask_iq2xs[0]) ? -1.0f : 1.0f);
                values = _mm256_mul_ps(values, sign_mask);

                _mm256_store_ps(&fp32_buffer[l * 8], values);
            }

            // Quantize FP32 buffer to Q8_0 format using AVX-512
            quantize_fp32_to_q8_0_avx512(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

        /**
         * @brief Auto-dispatch IQ2_XS decode based on CPU features
         */
        inline void decode_iq2xs_to_q8_0(const IQ2_XSBlock &block, size_t subblock_idx,
                                         int8_t *q8_qs, uint16_t *q8_scale)
        {
#ifdef __AVX512F__
            if (cpu_supports_avx512())
            {
                decode_iq2xs_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif
#ifdef __AVX2__
            if (cpu_supports_avx2())
            {
                decode_iq2xs_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif
            decode_iq2xs_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
        }

        /**
         * @brief Unpack IQ2_XS sub-block to int8 (transcoding)
         *
         * Wrapper around decode_iq2xs_to_q8_0 for IINT8Unpackable interface.
         */
        inline void unpack_iq2_xs_to_int8(const IQ2_XSBlock &block, size_t subblock_idx, int8_t *output)
        {
            uint16_t dummy_scale;
            decode_iq2xs_to_q8_0(block, subblock_idx, output, &dummy_scale);
        }

        /**
         * @brief Get IQ2_XS sub-block scale (transcoding)
         *
         * Wrapper around decode_iq2xs_to_q8_0 for IINT8Unpackable interface.
         */
        inline float get_iq2_xs_scale(const IQ2_XSBlock &block, size_t subblock_idx)
        {
            int8_t dummy_qs[32];
            uint16_t scale_fp16;
            decode_iq2xs_to_q8_0(block, subblock_idx, dummy_qs, &scale_fp16);
            return fp16_to_fp32(scale_fp16);
        }

        // ============================================================================
        // IQ3_XXS SIMD Helpers
        // ============================================================================
        // IQ3_XXS: 256 elements per super-block, grid-based with iq3xxs_grid[256]
        // Block structure: d (FP16 scale), qs[96] (grid indices + scales/signs)
        // Each super-block has 8 sub-blocks of 32 elements
        // Layout: qs[64] are grid indices, qs[64..96] are scales+signs (32 bytes)

        /**
         * @brief Decode IQ3_XXS sub-block to FP32
         */
        inline void decode_iq3xxs_subblock_to_fp32(
            const IQ3_XXSBlock &block,
            size_t subblock_idx,
            float *output)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + 8 * subblock_idx;                    // 8 grid indices
            const uint8_t *scales_and_signs = block.qs + 64 + 4 * subblock_idx; // 4 bytes

            // Extract scale (top 4 bits of uint32)
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

            for (size_t l = 0; l < 4; ++l)
            {
                // Extract sign pattern (7 bits)
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];

                // Grid lookups (2 indices → 2 × 4 elements = 8 elements)
                const uint8_t *grid1 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t *grid2 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + qs[2 * l + 1]);

                // Decode 8 elements (4 from grid1, 4 from grid2)
                for (size_t j = 0; j < 4; ++j)
                {
                    output[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                    output[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                }
                output += 8;
            }
        }

        /**
         * @brief Scalar decode IQ3_XXS sub-block to Q8_0
         * @param block IQ3_XXS super-block
         * @param subblock_idx Sub-block index (0-7)
         * @param q8_qs Output Q8_0 quantized values (32 int8_t)
         * @param q8_scale Output Q8_0 FP16 scale (pointer)
         */
        inline void decode_iq3xxs_to_q8_0_scalar(
            const IQ3_XXSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            float fp32_buffer[32];
            decode_iq3xxs_subblock_to_fp32(block, subblock_idx, fp32_buffer);

            // Quantize to Q8_0
#if defined(__AVX512F__)
            quantize_fp32_to_q8_0_avx512(fp32_buffer, 32, q8_qs, q8_scale);
#elif defined(__AVX2__)
            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, q8_qs, q8_scale);
#else
            quantize_fp32_to_q8_0_scalar(fp32_buffer, 32, q8_qs, q8_scale);
#endif
        }

#if defined(__AVX2__)
        /**
         * @brief AVX2 decode IQ3_XXS sub-block to Q8_0
         */
        inline void decode_iq3xxs_to_q8_0_avx2(
            const IQ3_XXSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            float fp32_buffer[32];
            decode_iq3xxs_subblock_to_fp32(block, subblock_idx, fp32_buffer);

            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

#if defined(__AVX512F__)
        /**
         * @brief AVX-512 decode IQ3_XXS sub-block to Q8_0
         */
        inline void decode_iq3xxs_to_q8_0_avx512(
            const IQ3_XXSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            float fp32_buffer[32];
            decode_iq3xxs_subblock_to_fp32(block, subblock_idx, fp32_buffer);

            quantize_fp32_to_q8_0_avx512(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

        /**
         * @brief Auto-dispatch IQ3_XXS decode based on CPU features
         */
        inline void decode_iq3xxs_to_q8_0(
            const IQ3_XXSBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                decode_iq3xxs_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                decode_iq3xxs_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

            decode_iq3xxs_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
        }

        /**
         * @brief Unpack IQ3_XXS sub-block to int8 (transcoding)
         *
         * Wrapper around decode_iq3xxs_to_q8_0 for IINT8Unpackable interface.
         */
        inline void unpack_iq3_xxs_to_int8(const IQ3_XXSBlock &block, size_t subblock_idx, int8_t *output)
        {
            uint16_t dummy_scale;
            decode_iq3xxs_to_q8_0(block, subblock_idx, output, &dummy_scale);
        }

        /**
         * @brief Get IQ3_XXS sub-block scale (transcoding)
         *
         * Wrapper around decode_iq3xxs_to_q8_0 for IINT8Unpackable interface.
         */
        inline float get_iq3_xxs_scale(const IQ3_XXSBlock &block, size_t subblock_idx)
        {
            int8_t dummy_qs[32];
            uint16_t scale_fp16;
            decode_iq3xxs_to_q8_0(block, subblock_idx, dummy_qs, &scale_fp16);
            return fp16_to_fp32(scale_fp16);
        }

        // ====================================================================
        // IQ2_S: 2-bit Small IQ with High Bits and Scales
        // ====================================================================

        /**
         * @brief Scalar decode IQ2_S to Q8_0 (baseline implementation)
         *
         * IQ2_S format: 256 elements per super-block, 8 sub-blocks of 32 elements each
         * - d: FP16 scale factor
         * - qs[64]: Quantized values (4 bytes per sub-block)
         * - qh[8]: High bits (1 byte per sub-block)
         * - scales[8]: Scales (1 byte per sub-block)
         *
         * Algorithm per sub-block (32 elements):
         * - scales[ib32] contains 2 scales (low 4 bits, high 4 bits)
         * - Each scale corresponds to 16 elements
         * - Grid lookup: iq2s_grid[qs[l] | (qh[ib32] << (8 - 2*l) & 0x300)]
         * - Signs from qs[32..63] (signs pointer offset)
         */
        inline void decode_iq2s_to_q8_0_scalar(
            const IQ2_SBlock &block,
            size_t subblock_idx, // 0-7
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4;         // 4 bytes per sub-block
            const uint8_t *signs = block.qs + 32 + subblock_idx * 4; // Signs in qs[32..63]
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t scale_byte = block.scales[subblock_idx];

            // Two scales per sub-block (16 elements each)
            float db[2];
            db[0] = d * (0.5f + (scale_byte & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scale_byte >> 4)) * 0.25f;

            // Decode to FP32 buffer
            alignas(32) float fp32_buffer[32];
            float *output = fp32_buffer;

            for (size_t l = 0; l < 4; ++l)
            {
                const float dl = db[l / 2]; // First 2 iterations use db[0], next 2 use db[1]

                // Construct 10-bit grid index from qs and qh high bits
                const uint16_t grid_idx = qs[l] | ((qh_byte << (8 - 2 * l)) & 0x300);
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(iq2s_grid + grid_idx);

                for (size_t j = 0; j < 8; ++j)
                {
                    output[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
                }
                output += 8;
            }

            // Quantize FP32 -> Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_buffer, 32, q8_qs, q8_scale);
        }

#ifdef __AVX2__
        /**
         * @brief AVX2 decode IQ2_S to Q8_0
         */
        inline void decode_iq2s_to_q8_0_avx2(
            const IQ2_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint8_t *signs = block.qs + 32 + subblock_idx * 4;
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t scale_byte = block.scales[subblock_idx];

            float db[2];
            db[0] = d * (0.5f + (scale_byte & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scale_byte >> 4)) * 0.25f;

            alignas(32) float fp32_buffer[32];
            float *output = fp32_buffer;

            for (size_t l = 0; l < 4; ++l)
            {
                const float dl = db[l / 2];
                const uint16_t grid_idx = qs[l] | ((qh_byte << (8 - 2 * l)) & 0x300);
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(iq2s_grid + grid_idx);

                for (size_t j = 0; j < 8; ++j)
                {
                    output[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
                }
                output += 8;
            }

            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

#ifdef __AVX512F__
        /**
         * @brief AVX-512 decode IQ2_S to Q8_0
         */
        inline void decode_iq2s_to_q8_0_avx512(
            const IQ2_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint8_t *signs = block.qs + 32 + subblock_idx * 4;
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t scale_byte = block.scales[subblock_idx];

            float db[2];
            db[0] = d * (0.5f + (scale_byte & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scale_byte >> 4)) * 0.25f;

            alignas(64) float fp32_buffer[32];
            float *output = fp32_buffer;

            for (size_t l = 0; l < 4; ++l)
            {
                const float dl = db[l / 2];
                const uint16_t grid_idx = qs[l] | ((qh_byte << (8 - 2 * l)) & 0x300);
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(iq2s_grid + grid_idx);

                for (size_t j = 0; j < 8; ++j)
                {
                    output[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
                }
                output += 8;
            }

            quantize_fp32_to_q8_0_avx512(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

        /**
         * @brief Auto-dispatch IQ2_S decode based on CPU features
         */
        inline void decode_iq2s_to_q8_0(
            const IQ2_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                decode_iq2s_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                decode_iq2s_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

            decode_iq2s_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
        }

        inline void unpack_iq2_s_to_int8(
            const IQ2_SBlock &block,
            size_t subblock_idx,
            int8_t *output)
        {
            uint16_t dummy_scale;
            decode_iq2s_to_q8_0(block, subblock_idx, output, &dummy_scale);
        }

        inline float get_iq2_s_scale(
            const IQ2_SBlock &block,
            size_t subblock_idx)
        {
            int8_t dummy_qs[32];
            uint16_t scale_f16;
            decode_iq2s_to_q8_0(block, subblock_idx, dummy_qs, &scale_f16);
            return fp16_to_fp32(scale_f16);
        }

        // ========================================================================
        // IQ3_S: 3-bit small IQ with signs[] array and scales[]
        // ========================================================================
        // Block: 110 bytes (d: FP16, qs[64], qh[8], signs[32], scales[4])
        // Structure: 256 elements → 8 sub-blocks of 32 elements
        // Grid: iq3s_grid[512] (uint32_t entries)
        // Algorithm: Process in pairs (ib32 = 0,2,4,6) with db1/db2 scales
        //   - db1 = d * (1 + 2 * (scales[ib32/2] & 0xf))
        //   - db2 = d * (1 + 2 * (scales[ib32/2] >> 4))
        //   - Each pair: 4 groups of 8 elements (2 grids × 4 elements each)
        //   - Grid index: qs[2*l+i] | ((qh[offset] << shift) & 256)
        //   - Signs from signs[] array using kmask_iq2xs[]

        /**
         * @brief Scalar decode IQ3_S sub-block to Q8_0
         *
         * IQ3_S uses iq3s_grid[512] with:
         * - qs[64]: Quantized values (8 bits per grid index)
         * - qh[8]: High bits (1 extra bit per grid index for 9-bit total)
         * - signs[32]: Sign patterns (8 sub-blocks × 4 bytes)
         * - scales[4]: Scales (2 scales per sub-block pair, 4 bits each)
         *
         * Algorithm per sub-block (32 elements):
         * - scales[ib32/2] contains 2 scales (low 4 bits, high 4 bits)
         * - Each scale: db = d * (1 + 2 * scale_nibble)
         * - 4 groups of 8 elements: 2 grids × 4 elements each
         * - Grid lookup: iq3s_grid[qs | (qh << shift & 256)]
         * - Signs from signs[] using kmask_iq2xs[]
         */
        inline void decode_iq3s_subblock_to_fp32(
            const IQ3_SBlock &block,
            size_t subblock_idx, // 0-7
            float *output)
        {
            const float d = fp16_to_fp32(block.d);

            // Sub-blocks processed in pairs: (0,1), (2,3), (4,5), (6,7)
            const size_t pair_idx = subblock_idx / 2;    // 0-3
            const size_t within_pair = subblock_idx % 2; // 0 or 1

            // Get scale for this sub-block within the pair
            const uint8_t scale_byte = block.scales[pair_idx];
            const float db = within_pair == 0
                                 ? d * (1.0f + 2.0f * (scale_byte & 0xf))
                                 : d * (1.0f + 2.0f * (scale_byte >> 4));

            // Offset into arrays
            const size_t qs_offset = subblock_idx * 8;    // 8 bytes per sub-block
            const size_t qh_offset = subblock_idx;        // 1 byte per sub-block
            const size_t signs_offset = subblock_idx * 4; // 4 bytes per sub-block

            const uint8_t *qs = block.qs + qs_offset;
            const uint8_t qh_byte = block.qh[qh_offset];
            const uint8_t *signs = block.signs + signs_offset;

            // 4 groups of 8 elements (2 grids × 4 elements each)
            for (size_t l = 0; l < 4; ++l)
            {
                // Grid index with high bit
                const uint16_t grid_idx1 = qs[2 * l + 0] | ((qh_byte << (8 - 2 * l)) & 256);
                const uint16_t grid_idx2 = qs[2 * l + 1] | ((qh_byte << (7 - 2 * l)) & 256);

                const uint8_t *grid1 = reinterpret_cast<const uint8_t *>(iq3s_grid + grid_idx1);
                const uint8_t *grid2 = reinterpret_cast<const uint8_t *>(iq3s_grid + grid_idx2);

                // First 4 elements from grid1
                for (size_t j = 0; j < 4; ++j)
                {
                    output[j + 0] = db * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                }

                // Next 4 elements from grid2
                for (size_t j = 0; j < 4; ++j)
                {
                    output[j + 4] = db * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                }

                output += 8;
            }
        }

        inline void decode_iq3s_to_q8_0_scalar(
            const IQ3_SBlock &block,
            size_t subblock_idx, // 0-7
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            // Decode to FP32 buffer (32 elements)
            alignas(32) float fp32_buffer[32];
            decode_iq3s_subblock_to_fp32(block, subblock_idx, fp32_buffer);

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_buffer, 32, q8_qs, q8_scale);
        }

#if defined(__AVX2__)
        /**
         * @brief AVX2 decode IQ3_S to Q8_0
         */
        inline void decode_iq3s_to_q8_0_avx2(
            const IQ3_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            // Decode to FP32 buffer (32 elements)
            alignas(32) float fp32_buffer[32];
            decode_iq3s_subblock_to_fp32(block, subblock_idx, fp32_buffer);

            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

#if defined(__AVX512F__)
        /**
         * @brief AVX-512 decode IQ3_S to Q8_0
         */
        inline void decode_iq3s_to_q8_0_avx512(
            const IQ3_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            // Decode to FP32 buffer (32 elements)
            alignas(64) float fp32_buffer[32];
            decode_iq3s_subblock_to_fp32(block, subblock_idx, fp32_buffer);

            quantize_fp32_to_q8_0_avx512(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

        /**
         * @brief Auto-dispatch IQ3_S decode based on CPU features
         */
        inline void decode_iq3s_to_q8_0(
            const IQ3_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                decode_iq3s_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                decode_iq3s_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

            decode_iq3s_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
        }

        /**
         * @brief Unpack IQ3_S sub-block to INT8 (transcoding via Q8_0)
         *
         * Used by IINT8Unpackable interface for QuantisedGemmKernel.
         * Decodes IQ3_S to FP32 then quantizes to Q8_0 (INT8).
         *
         * @param block Source IQ3_S super-block
         * @param subblock_idx Index of sub-block (0-7)
         * @param output Output buffer for 32 int8 values
         */
        inline void unpack_iq3_s_to_int8(
            const IQ3_SBlock &block,
            size_t subblock_idx,
            int8_t *output)
        {
            uint16_t dummy_scale;
            decode_iq3s_to_q8_0(block, subblock_idx, output, &dummy_scale);
        }

        /**
         * @brief Get scale for IQ3_S sub-block (transcoding via Q8_0)
         *
         * Used by IINT8Unpackable interface for QuantisedGemmKernel.
         * Decodes IQ3_S to FP32 then quantizes to Q8_0 to find the scale.
         *
         * @param block Source IQ3_S super-block
         * @param subblock_idx Index of sub-block (0-7)
         * @return Scale factor for the transcoded INT8 values
         */
        inline float get_iq3_s_scale(
            const IQ3_SBlock &block,
            size_t subblock_idx)
        {
            int8_t dummy_qs[32];
            uint16_t scale_fp16;
            decode_iq3s_to_q8_0(block, subblock_idx, dummy_qs, &scale_fp16);
            return fp16_to_fp32(scale_fp16);
        }

        // ============================================================================
        // IQ1_S Decode Functions (1-bit quantization with IQ1S_DELTA offset)
        // ============================================================================

        /**
         * @brief Decode one sub-block (32 elements) from IQ1_S to Q8_0 format (scalar)
         *
         * IQ1_S Format:
         * - 256 elements per super-block (8 sub-blocks of 32 elements each)
         * - Grid lookup table: iq1s_grid[2048] (uint64_t, accessed as int8_t*)
         * - Per iteration (ib=0..7, 32 elements each):
         *   - Scale: 3 bits from qh[ib] bits 12-14
         *   - Delta: Sign bit from qh[ib] bit 15 (±IQ1S_DELTA = ±0.125f)
         *   - 4 groups of 8 elements:
         *     - Grid index: qs[l] (8 bits) + qh[ib] bits (3*l to 3*l+2) → 11-bit (0-2047)
         *     - Decode: output[j] = dl * (grid[j] + delta)
         *
         * @param block IQ1_S block (50 bytes)
         * @param subblock_idx Sub-block index (0-7)
         * @param q8_qs Output quantized values (32 int8_t)
         * @param q8_scale Output scale (FP16 pointer)
         */
        __attribute__((always_inline)) inline void decode_iq1s_to_q8_0_scalar(
            const IQ1_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4; // 4 bytes per sub-block
            const uint16_t qh_val = block.qh[subblock_idx];

            // Scale: 3 bits from qh bits 12-14
            const float dl = d * (2.0f * ((qh_val >> 12) & 7) + 1.0f);

            // Delta: sign bit from qh bit 15
            const float delta = (qh_val & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

            // Decode 4 groups of 8 elements (32 elements total)
            float fp32_output[32];
            for (size_t l = 0; l < 4; ++l)
            {
                // Grid index: 11 bits (qs[l] + 3 bits from qh)
                const uint16_t grid_idx = qs[l] | (((qh_val >> (3 * l)) & 7) << 8);

                // Grid lookup: iq1s_grid is uint64_t[2048], access as int8_t*
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + grid_idx);

                for (size_t j = 0; j < 8; ++j)
                {
                    fp32_output[l * 8 + j] = dl * (static_cast<float>(grid[j]) + delta);
                }
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_output, 32, q8_qs, q8_scale);
        }

#ifdef __AVX2__
        /**
         * @brief Decode one sub-block (32 elements) from IQ1_S to Q8_0 format (AVX2)
         */
        __attribute__((always_inline)) inline void decode_iq1s_to_q8_0_avx2(
            const IQ1_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint16_t qh_val = block.qh[subblock_idx];

            const float dl = d * (2.0f * ((qh_val >> 12) & 7) + 1.0f);
            const float delta = (qh_val & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

            // Broadcast dl and delta for SIMD
            __m256 vdl = _mm256_set1_ps(dl);
            __m256 vdelta = _mm256_set1_ps(delta);

            // Decode 4 groups of 8 elements
            float fp32_output[32];
            for (size_t l = 0; l < 4; ++l)
            {
                const uint16_t grid_idx = qs[l] | (((qh_val >> (3 * l)) & 7) << 8);
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + grid_idx);

                // Load 8 int8_t values
                __m128i vgrid_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid));

                // Convert int8 → int32
                __m128i vgrid_i16 = _mm_cvtepi8_epi16(vgrid_i8);
                __m128i vgrid_lo_i32 = _mm_cvtepi16_epi32(vgrid_i16);
                __m128i vgrid_hi_i32 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(vgrid_i16, _MM_SHUFFLE(3, 2, 3, 2)));

                // Convert int32 → float
                __m256 vgrid_f32 = _mm256_set_m128(
                    _mm_cvtepi32_ps(vgrid_hi_i32),
                    _mm_cvtepi32_ps(vgrid_lo_i32));

                // Compute: dl * (grid[j] + delta)
                __m256 vresult = _mm256_mul_ps(vdl, _mm256_add_ps(vgrid_f32, vdelta));

                _mm256_storeu_ps(&fp32_output[l * 8], vresult);
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_avx2(fp32_output, 32, q8_qs, q8_scale);
        }
#endif

#ifdef __AVX512F__
        /**
         * @brief Decode one sub-block (32 elements) from IQ1_S to Q8_0 format (AVX-512)
         */
        __attribute__((always_inline)) inline void decode_iq1s_to_q8_0_avx512(
            const IQ1_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint16_t qh_val = block.qh[subblock_idx];

            const float dl = d * (2.0f * ((qh_val >> 12) & 7) + 1.0f);
            const float delta = (qh_val & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

            // Broadcast dl and delta for SIMD
            __m512 vdl = _mm512_set1_ps(dl);
            __m512 vdelta = _mm512_set1_ps(delta);

            // Decode 4 groups of 8 elements → 32 elements total (2 AVX-512 vectors)
            float fp32_output[32];
            for (size_t l = 0; l < 4; l += 2)
            {
                // Process 2 groups (16 elements) at once
                uint16_t grid_idx1 = qs[l + 0] | (((qh_val >> (3 * (l + 0))) & 7) << 8);
                uint16_t grid_idx2 = qs[l + 1] | (((qh_val >> (3 * (l + 1))) & 7) << 8);

                const int8_t *grid1 = reinterpret_cast<const int8_t *>(iq1s_grid + grid_idx1);
                const int8_t *grid2 = reinterpret_cast<const int8_t *>(iq1s_grid + grid_idx2);

                // Load 16 int8_t values (8 from each grid)
                __m128i vgrid1_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid1));
                __m128i vgrid2_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid2));
                __m128i vgrid_i8 = _mm_unpacklo_epi64(vgrid1_i8, vgrid2_i8);

                // Convert int8 → int32 → float
                __m512i vgrid_i32 = _mm512_cvtepi8_epi32(vgrid_i8);
                __m512 vgrid_f32 = _mm512_cvtepi32_ps(vgrid_i32);

                // Compute: dl * (grid[j] + delta)
                __m512 vresult = _mm512_mul_ps(vdl, _mm512_add_ps(vgrid_f32, vdelta));

                _mm512_storeu_ps(&fp32_output[l * 8], vresult);
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_avx512(fp32_output, 32, q8_qs, q8_scale);
        }
#endif

        /**
         * @brief Auto-dispatch IQ1_S decode based on CPU features
         */
        __attribute__((always_inline)) inline void decode_iq1s_to_q8_0(
            const IQ1_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                decode_iq1s_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                decode_iq1s_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

            decode_iq1s_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
        }

        /**
         * @brief Unpack IQ1_S sub-block to int8 (32 elements)
         *
         * @param block IQ1_S super-block
         * @param subblock_idx Sub-block index (0-7)
         * @param output Output buffer (32 int8_t values)
         */
        inline void unpack_iq1_s_to_int8(const IQ1_SBlock &block, size_t subblock_idx, int8_t *output)
        {
            uint16_t dummy_scale;
            decode_iq1s_to_q8_0(block, subblock_idx, output, &dummy_scale);
        }

        /**
         * @brief Get scale for IQ1_S sub-block
         *
         * @param block IQ1_S super-block
         * @param subblock_idx Sub-block index (0-7)
         * @return Scale factor
         */
        inline float get_iq1_s_scale(const IQ1_SBlock &block, size_t subblock_idx)
        {
            int8_t dummy_output[32];
            uint16_t scale_fp16;
            decode_iq1s_to_q8_0(block, subblock_idx, dummy_output, &scale_fp16);
            return fp16_to_fp32(scale_fp16);
        }

        // ============================================================================
        // IQ1_M Decode Functions (1-bit quantization with multiple scales)
        // ============================================================================

        /**
         * @brief Decode one sub-block (32 elements) from IQ1_M to Q8_0 format (scalar)
         *
         * IQ1_M Format:
         * - 256 elements per super-block (8 sub-blocks of 32 elements each)
         * - Grid lookup table: iq1s_grid[2048] (uint64_t, accessed as int8_t*)
         * - Global scale: FP16 constructed from 4 bits across scales[0-3]
         * - Per iteration (ib=0..7, 32 elements each):
         *   - Two sub-scales: dl1, dl2 (first 2 groups use dl1, last 2 use dl2)
         *   - 4 groups of 8 elements:
         *     - Grid index: qs[l] (8 bits) + qh bits → 11-bit (0-2047)
         *     - Delta: Sign bit from qh → ±IQ1S_DELTA (0.125f)
         *     - Decode: output[j] = dl1/dl2 * (grid[j] + delta[l])
         *
         * @param block IQ1_M block (56 bytes)
         * @param subblock_idx Sub-block index (0-7)
         * @param global_scale Global FP16 scale (pre-extracted)
         * @param q8_qs Output quantized values (32 int8_t)
         * @param q8_scale Output scale (FP16 pointer)
         */
        __attribute__((always_inline)) inline void decode_iq1m_to_q8_0_scalar(
            const IQ1_MBlock &block,
            size_t subblock_idx,
            float global_scale,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const uint16_t *sc = reinterpret_cast<const uint16_t *>(block.scales);
            const uint8_t *qs = block.qs + subblock_idx * 4; // 4 bytes per sub-block
            const uint8_t *qh = block.qh + subblock_idx * 2; // 2 bytes per sub-block

            // Two sub-scales per iteration
            const float dl1 = global_scale * (2.0f * ((sc[subblock_idx / 2] >> (6 * (subblock_idx % 2) + 0)) & 0x7) + 1.0f);
            const float dl2 = global_scale * (2.0f * ((sc[subblock_idx / 2] >> (6 * (subblock_idx % 2) + 3)) & 0x7) + 1.0f);

            // Grid indices for 4 groups of 8
            uint16_t idx[4];
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);

            // Delta signs for 4 groups
            float delta[4];
            delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

            // Decode 4 groups of 8 elements (32 elements total)
            float fp32_output[32];

            // First two groups use dl1
            for (size_t l = 0; l < 2; ++l)
            {
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + idx[l]);
                for (size_t j = 0; j < 8; ++j)
                {
                    fp32_output[l * 8 + j] = dl1 * (static_cast<float>(grid[j]) + delta[l]);
                }
            }

            // Last two groups use dl2
            for (size_t l = 2; l < 4; ++l)
            {
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + idx[l]);
                for (size_t j = 0; j < 8; ++j)
                {
                    fp32_output[l * 8 + j] = dl2 * (static_cast<float>(grid[j]) + delta[l]);
                }
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_output, 32, q8_qs, q8_scale);
        }

        /**
         * @brief Extract global scale from IQ1_M block
         */
        __attribute__((always_inline)) inline float extract_iq1m_global_scale(const IQ1_MBlock &block)
        {
            const uint16_t *sc = reinterpret_cast<const uint16_t *>(block.scales);
            uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                 ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            return fp16_to_fp32(scale_u16);
        }

#ifdef __AVX2__
        /**
         * @brief Decode one sub-block (32 elements) from IQ1_M to Q8_0 format (AVX2)
         */
        __attribute__((always_inline)) inline void decode_iq1m_to_q8_0_avx2(
            const IQ1_MBlock &block,
            size_t subblock_idx,
            float global_scale,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const uint16_t *sc = reinterpret_cast<const uint16_t *>(block.scales);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint8_t *qh = block.qh + subblock_idx * 2;

            const float dl1 = global_scale * (2.0f * ((sc[subblock_idx / 2] >> (6 * (subblock_idx % 2) + 0)) & 0x7) + 1.0f);
            const float dl2 = global_scale * (2.0f * ((sc[subblock_idx / 2] >> (6 * (subblock_idx % 2) + 3)) & 0x7) + 1.0f);

            // Grid indices
            uint16_t idx[4];
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);

            // Delta signs
            float delta[4];
            delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

            float fp32_output[32];

            // First two groups use dl1
            __m256 vdl1 = _mm256_set1_ps(dl1);
            for (size_t l = 0; l < 2; ++l)
            {
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + idx[l]);
                __m256 vdelta = _mm256_set1_ps(delta[l]);

                // Load 8 int8_t values
                __m128i vgrid_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid));

                // Convert int8 → int32
                __m128i vgrid_i16 = _mm_cvtepi8_epi16(vgrid_i8);
                __m128i vgrid_lo_i32 = _mm_cvtepi16_epi32(vgrid_i16);
                __m128i vgrid_hi_i32 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(vgrid_i16, _MM_SHUFFLE(3, 2, 3, 2)));

                // Convert int32 → float
                __m256 vgrid_f32 = _mm256_set_m128(
                    _mm_cvtepi32_ps(vgrid_hi_i32),
                    _mm_cvtepi32_ps(vgrid_lo_i32));

                // Compute: dl1 * (grid[j] + delta)
                __m256 vresult = _mm256_mul_ps(vdl1, _mm256_add_ps(vgrid_f32, vdelta));
                _mm256_storeu_ps(&fp32_output[l * 8], vresult);
            }

            // Last two groups use dl2
            __m256 vdl2 = _mm256_set1_ps(dl2);
            for (size_t l = 2; l < 4; ++l)
            {
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + idx[l]);
                __m256 vdelta = _mm256_set1_ps(delta[l]);

                __m128i vgrid_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid));
                __m128i vgrid_i16 = _mm_cvtepi8_epi16(vgrid_i8);
                __m128i vgrid_lo_i32 = _mm_cvtepi16_epi32(vgrid_i16);
                __m128i vgrid_hi_i32 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(vgrid_i16, _MM_SHUFFLE(3, 2, 3, 2)));

                __m256 vgrid_f32 = _mm256_set_m128(
                    _mm_cvtepi32_ps(vgrid_hi_i32),
                    _mm_cvtepi32_ps(vgrid_lo_i32));

                __m256 vresult = _mm256_mul_ps(vdl2, _mm256_add_ps(vgrid_f32, vdelta));
                _mm256_storeu_ps(&fp32_output[l * 8], vresult);
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_avx2(fp32_output, 32, q8_qs, q8_scale);
        }
#endif

#ifdef __AVX512F__
        /**
         * @brief Decode one sub-block (32 elements) from IQ1_M to Q8_0 format (AVX-512)
         */
        __attribute__((always_inline)) inline void decode_iq1m_to_q8_0_avx512(
            const IQ1_MBlock &block,
            size_t subblock_idx,
            float global_scale,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const uint16_t *sc = reinterpret_cast<const uint16_t *>(block.scales);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint8_t *qh = block.qh + subblock_idx * 2;

            const float dl1 = global_scale * (2.0f * ((sc[subblock_idx / 2] >> (6 * (subblock_idx % 2) + 0)) & 0x7) + 1.0f);
            const float dl2 = global_scale * (2.0f * ((sc[subblock_idx / 2] >> (6 * (subblock_idx % 2) + 3)) & 0x7) + 1.0f);

            // Grid indices
            uint16_t idx[4];
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);

            // Delta signs
            float delta[4];
            delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

            float fp32_output[32];

            // Process first two groups (16 elements) with dl1
            {
                const int8_t *grid0 = reinterpret_cast<const int8_t *>(iq1s_grid + idx[0]);
                const int8_t *grid1 = reinterpret_cast<const int8_t *>(iq1s_grid + idx[1]);

                // Load 16 int8_t values (8 from each grid)
                __m128i vgrid0_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid0));
                __m128i vgrid1_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid1));
                __m128i vgrid_i8 = _mm_unpacklo_epi64(vgrid0_i8, vgrid1_i8);

                // Convert int8 → int32 → float
                __m512i vgrid_i32 = _mm512_cvtepi8_epi32(vgrid_i8);
                __m512 vgrid_f32 = _mm512_cvtepi32_ps(vgrid_i32);

                // Apply deltas (first 8 elements use delta[0], next 8 use delta[1])
                __m512 vdelta = _mm512_set_ps(
                    delta[1], delta[1], delta[1], delta[1], delta[1], delta[1], delta[1], delta[1],
                    delta[0], delta[0], delta[0], delta[0], delta[0], delta[0], delta[0], delta[0]);

                // Compute: dl1 * (grid[j] + delta)
                __m512 vdl1 = _mm512_set1_ps(dl1);
                __m512 vresult = _mm512_mul_ps(vdl1, _mm512_add_ps(vgrid_f32, vdelta));
                _mm512_storeu_ps(&fp32_output[0], vresult);
            }

            // Process last two groups (16 elements) with dl2
            {
                const int8_t *grid2 = reinterpret_cast<const int8_t *>(iq1s_grid + idx[2]);
                const int8_t *grid3 = reinterpret_cast<const int8_t *>(iq1s_grid + idx[3]);

                __m128i vgrid2_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid2));
                __m128i vgrid3_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(grid3));
                __m128i vgrid_i8 = _mm_unpacklo_epi64(vgrid2_i8, vgrid3_i8);

                __m512i vgrid_i32 = _mm512_cvtepi8_epi32(vgrid_i8);
                __m512 vgrid_f32 = _mm512_cvtepi32_ps(vgrid_i32);

                __m512 vdelta = _mm512_set_ps(
                    delta[3], delta[3], delta[3], delta[3], delta[3], delta[3], delta[3], delta[3],
                    delta[2], delta[2], delta[2], delta[2], delta[2], delta[2], delta[2], delta[2]);

                __m512 vdl2 = _mm512_set1_ps(dl2);
                __m512 vresult = _mm512_mul_ps(vdl2, _mm512_add_ps(vgrid_f32, vdelta));
                _mm512_storeu_ps(&fp32_output[16], vresult);
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_avx512(fp32_output, 32, q8_qs, q8_scale);
        }
#endif

        /**
         * @brief Auto-dispatch IQ1_M decode based on CPU features
         */
        __attribute__((always_inline)) inline void decode_iq1m_to_q8_0(
            const IQ1_MBlock &block,
            size_t subblock_idx,
            float global_scale,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                decode_iq1m_to_q8_0_avx512(block, subblock_idx, global_scale, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                decode_iq1m_to_q8_0_avx2(block, subblock_idx, global_scale, q8_qs, q8_scale);
                return;
            }
#endif

            decode_iq1m_to_q8_0_scalar(block, subblock_idx, global_scale, q8_qs, q8_scale);
        }

        /**
         * @brief Unpack IQ1_M sub-block to INT8 (via Q8_0 intermediate)
         */
        __attribute__((always_inline)) inline void unpack_iq1_m_to_int8(
            const IQ1_MBlock &block,
            size_t subblock_idx,
            int8_t *output)
        {
            float global_scale = extract_iq1m_global_scale(block);
            int8_t q8_qs[32];
            uint16_t q8_scale_u16;

            decode_iq1m_to_q8_0(block, subblock_idx, global_scale, q8_qs, &q8_scale_u16);

#if defined(__AVX2__) || defined(__AVX512F__)
            _mm256_storeu_si256((__m256i *)output, _mm256_loadu_si256((const __m256i *)q8_qs));
#else
            std::memcpy(output, q8_qs, 32);
#endif
        }

        /**
         * @brief Get scale for IQ1_M sub-block (via Q8_0 intermediate)
         */
        __attribute__((always_inline)) inline float get_iq1_m_scale(
            const IQ1_MBlock &block,
            size_t subblock_idx)
        {
            float global_scale = extract_iq1m_global_scale(block);
            int8_t q8_qs[32];
            uint16_t q8_scale_u16;

            decode_iq1m_to_q8_0(block, subblock_idx, global_scale, q8_qs, &q8_scale_u16);

            return fp16_to_fp32(q8_scale_u16);
        }

        // =====================================================================
        // K-Quant Decode Functions (merged from SIMDKQuantHelpers.h)
        // =====================================================================

        // =====================================================================
        // Q2_K → Q8_0
        // =====================================================================

        inline void decode_q2_k_to_q8_0_scalar(
            const Q2_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q2_k_to_q8_0_scalar: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const size_t half = subblock_idx / 4;
            const size_t group = subblock_idx % 4;

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const uint8_t *scales = block.scales + half * 8;
            const uint8_t sc0 = scales[group * 2 + 0];
            const uint8_t sc1 = scales[group * 2 + 1];

            const float dl0 = d * static_cast<float>(sc0 & 0x0F);
            const float ml0 = dmin * static_cast<float>(sc0 >> 4);
            const float dl1 = d * static_cast<float>(sc1 & 0x0F);
            const float ml1 = dmin * static_cast<float>(sc1 >> 4);

            const uint8_t *q = block.qs + half * 32;
            const int shift = static_cast<int>(group * 2);

            for (size_t i = 0; i < 16; ++i)
            {
                const uint8_t packed0 = q[i];
                const uint8_t val0 = (packed0 >> shift) & 0x03;
                tmp[i] = dl0 * static_cast<float>(static_cast<int>(val0)) - ml0;

                const uint8_t packed1 = q[i + 16];
                const uint8_t val1 = (packed1 >> shift) & 0x03;
                tmp[i + 16] = dl1 * static_cast<float>(static_cast<int>(val1)) - ml1;
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }

#if defined(__AVX2__)
        inline void decode_q2_k_to_q8_0_avx2(
            const Q2_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q2_k_to_q8_0_avx2: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const size_t half = subblock_idx / 4;
            const size_t group = subblock_idx % 4;

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const uint8_t *scales = block.scales + half * 8;
            const uint8_t sc0 = scales[group * 2 + 0];
            const uint8_t sc1 = scales[group * 2 + 1];

            const float dl0 = d * static_cast<float>(sc0 & 0x0F);
            const float ml0 = dmin * static_cast<float>(sc0 >> 4);
            const float dl1 = d * static_cast<float>(sc1 & 0x0F);
            const float ml1 = dmin * static_cast<float>(sc1 >> 4);

            const uint8_t *q = block.qs + half * 32;
            const int shift = static_cast<int>(group * 2);
            const __m256i mask3 = _mm256_set1_epi32(0x3);

            auto process_half = [&](size_t base, float dl, float ml)
            {
                const __m256 vdl = _mm256_set1_ps(dl);
                const __m256 vml = _mm256_set1_ps(ml);

                for (size_t off = 0; off < 16; off += 8)
                {
                    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + base + off));
                    __m256i vals = _mm256_cvtepu8_epi32(bytes);
                    vals = _mm256_and_si256(_mm256_srlv_epi32(vals, _mm256_set1_epi32(shift)), mask3);
                    const __m256 vf = _mm256_cvtepi32_ps(vals);
                    const __m256 res = _mm256_sub_ps(_mm256_mul_ps(vdl, vf), vml);
                    _mm256_storeu_ps(tmp + base + off, res);
                }
            };

            process_half(0, dl0, ml0);
            process_half(16, dl1, ml1);

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
        inline void decode_q2_k_to_q8_0_avx512(
            const Q2_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q2_k_to_q8_0_avx512: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const size_t half = subblock_idx / 4;
            const size_t group = subblock_idx % 4;

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const uint8_t *scales = block.scales + half * 8;
            const uint8_t sc0 = scales[group * 2 + 0];
            const uint8_t sc1 = scales[group * 2 + 1];

            const float dl0 = d * static_cast<float>(sc0 & 0x0F);
            const float ml0 = dmin * static_cast<float>(sc0 >> 4);
            const float dl1 = d * static_cast<float>(sc1 & 0x0F);
            const float ml1 = dmin * static_cast<float>(sc1 >> 4);

            const uint8_t *q = block.qs + half * 32;
            const int shift = static_cast<int>(group * 2);

            auto process_half = [&](size_t base, float dl, float ml)
            {
                const __m512 vdl = _mm512_set1_ps(dl);
                const __m512 vml = _mm512_set1_ps(ml);

                const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + base));
                __m512i vals = _mm512_cvtepu8_epi32(bytes);
                vals = _mm512_and_si512(_mm512_srlv_epi32(vals, _mm512_set1_epi32(shift)), _mm512_set1_epi32(0x3));
                const __m512 vf = _mm512_cvtepi32_ps(vals);
                const __m512 res = _mm512_sub_ps(_mm512_mul_ps(vdl, vf), vml);
                _mm512_storeu_ps(tmp + base, res);
            };

            process_half(0, dl0, ml0);
            process_half(16, dl1, ml1);

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

        inline void decode_q2_k_to_q8_0(
            const Q2_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_q2_k_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
#elif defined(__AVX2__)
            decode_q2_k_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
#else
            decode_q2_k_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
#endif
        }

        // =====================================================================
        // Q3_K → Q8_0
        // =====================================================================

        inline void unpack_q3k_scales(const uint8_t *packed_scales, int8_t *unpacked_scales)
        {
            const uint32_t kmask1 = 0x03030303;
            const uint32_t kmask2 = 0x0f0f0f0f;

            uint32_t aux[4];
            std::memcpy(aux, packed_scales, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            std::memcpy(unpacked_scales, aux, 16);
        }

        inline void decode_q3k_subblock_to_float_scalar(
            const Q3_KBlock &block,
            size_t subblock_idx,
            float *output)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q3k_subblock_to_float: subblock_idx out of range");
            }

            int8_t scales[16];
            unpack_q3k_scales(block.scales, scales);

            const float d_all = fp16_to_fp32(block.d);

            const size_t chunk = subblock_idx / 4;
            const size_t group = subblock_idx % 4;

            // qs is split into two 32-byte chunks.
            // Within each chunk, 4 groups are interleaved (2 bits each).
            const uint8_t *q_ptr = block.qs + chunk * 32;
            const int shift = group * 2;

            // hmask is interleaved across the whole block.
            // Bit 'subblock_idx' of byte 'i' corresponds to element 'i' of that subblock.
            const uint8_t *hm_ptr = block.hmask;
            const uint8_t m = 1 << subblock_idx;

            const size_t scale_idx_0 = subblock_idx * 2;
            const size_t scale_idx_1 = subblock_idx * 2 + 1;

            const float dl0 = d_all * (scales[scale_idx_0] - 32);
            const float dl1 = d_all * (scales[scale_idx_1] - 32);

            // First 16 elements
            for (int i = 0; i < 16; ++i)
            {
                uint8_t q_val = (q_ptr[i] >> shift) & 3;
                bool high_bit_zero = (hm_ptr[i] & m) == 0;
                int8_t val = static_cast<int8_t>(q_val) - (high_bit_zero ? 4 : 0);
                output[i] = dl0 * val;
            }

            // Second 16 elements
            for (int i = 0; i < 16; ++i)
            {
                uint8_t q_val = (q_ptr[i + 16] >> shift) & 3;
                bool high_bit_zero = (hm_ptr[i + 16] & m) == 0;
                int8_t val = static_cast<int8_t>(q_val) - (high_bit_zero ? 4 : 0);
                output[i + 16] = dl1 * val;
            }
        }

#if defined(__AVX512F__)
        inline void decode_q3k_subblock_to_float_avx512(
            const Q3_KBlock &block,
            size_t subblock_idx,
            float *output)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q3k_subblock_to_float: subblock_idx out of range");
            }

            int8_t scales[16];
            unpack_q3k_scales(block.scales, scales);

            const float d_all = fp16_to_fp32(block.d);

            const size_t chunk = subblock_idx / 4;
            const size_t group = subblock_idx % 4;
            const uint8_t *q_ptr = block.qs + chunk * 32;
            const int shift = group * 2;

            const uint8_t *hm_ptr = block.hmask;

            const size_t scale_idx_0 = subblock_idx * 2;
            const size_t scale_idx_1 = subblock_idx * 2 + 1;
            const float dl0 = d_all * (scales[scale_idx_0] - 32);
            const float dl1 = d_all * (scales[scale_idx_1] - 32);

            // Load 32 bytes of qs
            __m256i q_bytes = _mm256_loadu_si256((const __m256i *)q_ptr);

            // Load 32 bytes of hmask
            __m256i h_bytes = _mm256_loadu_si256((const __m256i *)hm_ptr);

            // Process first 16
            {
                __m128i q_low = _mm256_castsi256_si128(q_bytes);
                __m512i q32 = _mm512_cvtepu8_epi32(q_low);

                __m512i shift_v = _mm512_set1_epi32(shift);
                __m512i mask3 = _mm512_set1_epi32(3);
                q32 = _mm512_and_si512(_mm512_srlv_epi32(q32, shift_v), mask3);

                __m128i h_low = _mm256_castsi256_si128(h_bytes);
                __m512i h32 = _mm512_cvtepu8_epi32(h_low);

                __m512i subblock_shift = _mm512_set1_epi32(subblock_idx);
                __m512i bit_val = _mm512_and_si512(_mm512_srlv_epi32(h32, subblock_shift), _mm512_set1_epi32(1));

                __m512i bit_inv = _mm512_xor_si512(bit_val, _mm512_set1_epi32(1));
                __m512i sub_val = _mm512_slli_epi32(bit_inv, 2); // * 4

                __m512i val_i = _mm512_sub_epi32(q32, sub_val);
                __m512 val_f = _mm512_cvtepi32_ps(val_i);

                __m512 res = _mm512_mul_ps(_mm512_set1_ps(dl0), val_f);
                _mm512_storeu_ps(output, res);
            }

            // Process second 16
            {
                __m128i q_high = _mm256_extracti128_si256(q_bytes, 1);
                __m512i q32 = _mm512_cvtepu8_epi32(q_high);

                __m512i shift_v = _mm512_set1_epi32(shift);
                __m512i mask3 = _mm512_set1_epi32(3);
                q32 = _mm512_and_si512(_mm512_srlv_epi32(q32, shift_v), mask3);

                __m128i h_high = _mm256_extracti128_si256(h_bytes, 1);
                __m512i h32 = _mm512_cvtepu8_epi32(h_high);

                __m512i subblock_shift = _mm512_set1_epi32(subblock_idx);
                __m512i bit_val = _mm512_and_si512(_mm512_srlv_epi32(h32, subblock_shift), _mm512_set1_epi32(1));

                __m512i bit_inv = _mm512_xor_si512(bit_val, _mm512_set1_epi32(1));
                __m512i sub_val = _mm512_slli_epi32(bit_inv, 2);

                __m512i val_i = _mm512_sub_epi32(q32, sub_val);
                __m512 val_f = _mm512_cvtepi32_ps(val_i);

                __m512 res = _mm512_mul_ps(_mm512_set1_ps(dl1), val_f);
                _mm512_storeu_ps(output + 16, res);
            }
        }
#endif

        inline void decode_q3k_subblock_to_float(
            const Q3_KBlock &block,
            size_t subblock_idx,
            float *output)
        {
#if defined(__AVX512F__)
            decode_q3k_subblock_to_float_avx512(block, subblock_idx, output);
#else
            decode_q3k_subblock_to_float_scalar(block, subblock_idx, output);
#endif
        }

        inline void transcode_q3_k_to_int8_scalar(const Q3_KBlock &block, int sub_block_idx, int8_t *output, float *out_scale, float *out_min)
        {
            int8_t scales[16];
            unpack_q3k_scales(block.scales, scales);

            const float d_all = fp16_to_fp32(block.d);

            const size_t chunk = sub_block_idx / 4;
            const size_t group = sub_block_idx % 4;

            const uint8_t *q_ptr = block.qs + chunk * 32;
            const int shift = group * 2;

            const uint8_t *hm_ptr = block.hmask;
            const uint8_t m = 1 << sub_block_idx;

            const size_t scale_idx_0 = sub_block_idx * 2;
            const size_t scale_idx_1 = sub_block_idx * 2 + 1;

            const float dl0 = d_all * (scales[scale_idx_0] - 32);
            const float dl1 = d_all * (scales[scale_idx_1] - 32);

            float min0, max0;
            if (dl0 >= 0)
            {
                min0 = dl0 * -4;
                max0 = dl0 * 3;
            }
            else
            {
                min0 = dl0 * 3;
                max0 = dl0 * -4;
            }

            float min1, max1;
            if (dl1 >= 0)
            {
                min1 = dl1 * -4;
                max1 = dl1 * 3;
            }
            else
            {
                min1 = dl1 * 3;
                max1 = dl1 * -4;
            }

            float global_min = std::min(min0, min1);
            float global_max = std::max(max0, max1);
            float range = global_max - global_min;

            if (range < 1e-5f)
            {
                *out_scale = 0.0f;
                *out_min = global_min;
                memset(output, -128, 32);
                return;
            }

            *out_scale = range / 255.0f;
            *out_min = global_min + 128.0f * (*out_scale);
            float inv_scale = 1.0f / (*out_scale);
            float bias = -(*out_min) * inv_scale;

            for (int i = 0; i < 16; ++i)
            {
                uint8_t q_val = (q_ptr[i] >> shift) & 3;
                bool high_bit_zero = (hm_ptr[i] & m) == 0;
                int8_t val = static_cast<int8_t>(q_val) - (high_bit_zero ? 4 : 0);
                float fval = dl0 * val;
                int32_t ival = static_cast<int32_t>(std::round(fval * inv_scale + bias));
                output[i] = static_cast<int8_t>(std::max(-128, std::min(127, ival)));
            }

            for (int i = 0; i < 16; ++i)
            {
                uint8_t q_val = (q_ptr[i + 16] >> shift) & 3;
                bool high_bit_zero = (hm_ptr[i + 16] & m) == 0;
                int8_t val = static_cast<int8_t>(q_val) - (high_bit_zero ? 4 : 0);
                float fval = dl1 * val;
                int32_t ival = static_cast<int32_t>(std::round(fval * inv_scale + bias));
                output[i + 16] = static_cast<int8_t>(std::max(-128, std::min(127, ival)));
            }
        }

#if defined(__AVX2__)
        inline void transcode_q3_k_to_int8_avx2(const Q3_KBlock &block, int sub_block_idx, int8_t *output, float *out_scale, float *out_min)
        {
            int8_t scales[16];
            unpack_q3k_scales(block.scales, scales);

            const float d_all = fp16_to_fp32(block.d);

            const size_t chunk = sub_block_idx / 4;
            const size_t group = sub_block_idx % 4;

            const uint8_t *q_ptr = block.qs + chunk * 32;
            const int shift = group * 2;

            const uint8_t *hm_ptr = block.hmask;
            const uint8_t m = 1 << sub_block_idx;

            const size_t scale_idx_0 = sub_block_idx * 2;
            const size_t scale_idx_1 = sub_block_idx * 2 + 1;

            const float dl0 = d_all * (scales[scale_idx_0] - 32);
            const float dl1 = d_all * (scales[scale_idx_1] - 32);

            float min0, max0;
            if (dl0 >= 0)
            {
                min0 = dl0 * -4;
                max0 = dl0 * 3;
            }
            else
            {
                min0 = dl0 * 3;
                max0 = dl0 * -4;
            }

            float min1, max1;
            if (dl1 >= 0)
            {
                min1 = dl1 * -4;
                max1 = dl1 * 3;
            }
            else
            {
                min1 = dl1 * 3;
                max1 = dl1 * -4;
            }

            float global_min = std::min(min0, min1);
            float global_max = std::max(max0, max1);
            float range = global_max - global_min;

            if (range < 1e-5f)
            {
                *out_scale = 0.0f;
                *out_min = global_min;
                memset(output, -128, 32);
                return;
            }

            *out_scale = range / 255.0f;
            *out_min = global_min + 128.0f * (*out_scale);
            float inv_scale = 1.0f / (*out_scale);
            float bias = -(*out_min) * inv_scale;

            int8_t lut[32];
            for (int idx = 0; idx < 8; ++idx)
            {
                int8_t val;
                if (idx < 4)
                    val = idx;
                else
                    val = (idx - 4) - 4;

                float fval0 = dl0 * val;
                int32_t i0 = static_cast<int32_t>(std::round(fval0 * inv_scale + bias));
                lut[idx] = static_cast<int8_t>(std::max(-128, std::min(127, i0)));

                float fval1 = dl1 * val;
                int32_t i1 = static_cast<int32_t>(std::round(fval1 * inv_scale + bias));
                lut[16 + idx] = static_cast<int8_t>(std::max(-128, std::min(127, i1)));
            }

            __m256i lut_vec = _mm256_loadu_si256((const __m256i *)lut);
            __m256i qs = _mm256_loadu_si256((const __m256i *)q_ptr);
            __m128i count = _mm_cvtsi32_si128(shift);
            __m256i q_indices = _mm256_srl_epi16(qs, count);
            q_indices = _mm256_and_si256(q_indices, _mm256_set1_epi8(3));

            __m256i hmask = _mm256_loadu_si256((const __m256i *)hm_ptr);
            __m256i m_vec = _mm256_set1_epi8(m);
            __m256i hm_bits = _mm256_and_si256(hmask, m_vec);
            hm_bits = _mm256_cmpeq_epi8(hm_bits, _mm256_setzero_si256());
            hm_bits = _mm256_and_si256(hm_bits, _mm256_set1_epi8(4));

            __m256i indices = _mm256_or_si256(q_indices, hm_bits);
            __m256i result = _mm256_shuffle_epi8(lut_vec, indices);
            _mm256_storeu_si256((__m256i *)output, result);
        }
#endif

        inline void transcode_q3_k_to_int8(const Q3_KBlock &block, int sub_block_idx, int8_t *output, float *out_scale, float *out_min)
        {
#if defined(__AVX2__)
            transcode_q3_k_to_int8_avx2(block, sub_block_idx, output, out_scale, out_min);
#else
            transcode_q3_k_to_int8_scalar(block, sub_block_idx, output, out_scale, out_min);
#endif
        }

        /**
         * @brief Scalar decode Q3_K to Q8_0 (one 32-element sub-block from 256-element super-block)
         * @param block Q3_K super-block (256 elements, 110 bytes)
         * @param subblock_idx Which 32-element sub-block to decode (0-7)
         * @param q8_qs Output Q8_0 quantized values (32 int8_t)
         * @param q8_scale Output Q8_0 scale (FP16)
         */
        inline void decode_q3_k_to_q8_0_scalar(
            const Q3_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];
            decode_q3k_subblock_to_float(block, subblock_idx, tmp);
            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }

        inline void decode_q3_k_to_q8_0(
            const Q3_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            // Currently only scalar implementation
            // AVX2/AVX512 would need careful handling of complex bit layout
            decode_q3_k_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
        }

        // =====================================================================
        // Q8_K → Q8_0
        // =====================================================================

        inline void decode_q8_k_to_q8_0_scalar(
            const Q8_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q8_k_to_q8_0_scalar: subblock_idx out of range");
            }

            const int8_t *src = block.qs + subblock_idx * Q8_0Block::BLOCK_SIZE;
            std::memcpy(q8_qs, src, Q8_0Block::BLOCK_SIZE * sizeof(int8_t));
            *q8_scale = fp32_to_fp16(1.0f);
        }

#if defined(__AVX2__)
        inline void decode_q8_k_to_q8_0_avx2(
            const Q8_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q8_k_to_q8_0_avx2: subblock_idx out of range");
            }

            const int8_t *src = block.qs + subblock_idx * Q8_0Block::BLOCK_SIZE;
            const __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), data);
            *q8_scale = fp32_to_fp16(1.0f);
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
        inline void decode_q8_k_to_q8_0_avx512(
            const Q8_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q8_k_to_q8_0_avx512: subblock_idx out of range");
            }

            const int8_t *src = block.qs + subblock_idx * Q8_0Block::BLOCK_SIZE;
            const __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), data);
            *q8_scale = fp32_to_fp16(1.0f);
        }
#endif

        inline void decode_q8_k_to_q8_0(
            const Q8_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_q8_k_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
#elif defined(__AVX2__)
            decode_q8_k_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
#else
            decode_q8_k_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
#endif
        }

        // =====================================================================
        // Q4_K and Q5_K Helper: Extract 6-bit scales and mins
        // =====================================================================

        /**
         * @brief Extract 6-bit scale and min values from packed Q4_K/Q5_K scales
         * Matches llama.cpp's get_scale_min_k4() function
         * @param j Scale index (0-7)
         * @param q Pointer to packed 12-byte scales array
         * @param d Output: extracted scale value (6 bits)
         * @param m Output: extracted min value (6 bits)
         */
        inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
        {
            if (j < 4)
            {
                *d = q[j] & 63;
                *m = q[j + 4] & 63;
            }
            else
            {
                *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
            }
        }

        // =====================================================================
        // Q4_K → Float (Sub-block)
        // =====================================================================

        inline void decode_q4k_subblock_to_float_scalar(
            const Q4_KBlock &block,
            size_t subblock_idx,
            float *output)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q4k_subblock_to_float_scalar: subblock_idx out of range");
            }

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *q = block.qs + group_idx * 32;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            if (is_second_half == 0)
            {
                for (size_t l = 0; l < 32; ++l)
                {
                    output[l] = dl * (q[l] & 0xF) - ml;
                }
            }
            else
            {
                for (size_t l = 0; l < 32; ++l)
                {
                    output[l] = dl * (q[l] >> 4) - ml;
                }
            }
        }

#if defined(__AVX2__)
        inline void decode_q4k_subblock_to_float_avx2(
            const Q4_KBlock &block,
            size_t subblock_idx,
            float *output)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q4k_subblock_to_float_avx2: subblock_idx out of range");
            }

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *q = block.qs + group_idx * 32;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            const __m256 dl_vec = _mm256_set1_ps(dl);
            const __m256 ml_vec = _mm256_set1_ps(ml);

            if (is_second_half == 0)
            {
                for (size_t l = 0; l < 32; l += 8)
                {
                    __m128i q_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + l));
                    __m256i q_32 = _mm256_cvtepu8_epi32(q_bytes);
                    __m256i low_bits = _mm256_and_si256(q_32, _mm256_set1_epi32(0xF));
                    __m256 q_float = _mm256_cvtepi32_ps(low_bits);
                    __m256 result = _mm256_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm256_storeu_ps(output + l, result);
                }
            }
            else
            {
                for (size_t l = 0; l < 32; l += 8)
                {
                    __m128i q_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + l));
                    __m256i q_32 = _mm256_cvtepu8_epi32(q_bytes);
                    __m256i high_bits = _mm256_srli_epi32(q_32, 4);
                    __m256 q_float = _mm256_cvtepi32_ps(high_bits);
                    __m256 result = _mm256_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm256_storeu_ps(output + l, result);
                }
            }
        }
#endif

#if defined(__AVX512F__)
        inline void decode_q4k_subblock_to_float_avx512(
            const Q4_KBlock &block,
            size_t subblock_idx,
            float *output)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q4k_subblock_to_float_avx512: subblock_idx out of range");
            }

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *q = block.qs + group_idx * 32;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            const __m512 dl_vec = _mm512_set1_ps(dl);
            const __m512 ml_vec = _mm512_set1_ps(ml);

            if (is_second_half == 0)
            {
                for (size_t l = 0; l < 32; l += 16)
                {
                    __m128i q_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + l));
                    __m512i q_32 = _mm512_cvtepu8_epi32(q_bytes);
                    __m512i low_bits = _mm512_and_si512(q_32, _mm512_set1_epi32(0xF));
                    __m512 q_float = _mm512_cvtepi32_ps(low_bits);
                    __m512 result = _mm512_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm512_storeu_ps(output + l, result);
                }
            }
            else
            {
                for (size_t l = 0; l < 32; l += 16)
                {
                    __m128i q_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + l));
                    __m512i q_32 = _mm512_cvtepu8_epi32(q_bytes);
                    __m512i high_bits = _mm512_srli_epi32(q_32, 4);
                    __m512 q_float = _mm512_cvtepi32_ps(high_bits);
                    __m512 result = _mm512_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm512_storeu_ps(output + l, result);
                }
            }
        }
#endif

        inline void decode_q4k_subblock_to_float(
            const Q4_KBlock &block,
            size_t subblock_idx,
            float *output)
        {
#if defined(__AVX512F__)
            decode_q4k_subblock_to_float_avx512(block, subblock_idx, output);
#elif defined(__AVX2__)
            decode_q4k_subblock_to_float_avx2(block, subblock_idx, output);
#else
            decode_q4k_subblock_to_float_scalar(block, subblock_idx, output);
#endif
        }

        // =====================================================================
        // Q4_K → Int8 (Transcode)
        // =====================================================================

        inline void transcode_q4_k_to_int8_scalar(
            const Q4_KBlock &block,
            size_t subblock_idx,
            int8_t *output,
            float *scale,
            float *min_val)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("transcode_q4_k_to_int8_scalar: subblock_idx out of range");
            }

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *q = block.qs + group_idx * 32;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);

            *scale = d * sc;
            *min_val = -dmin * m;

            if (is_second_half == 0)
            {
                for (size_t l = 0; l < 32; ++l)
                {
                    output[l] = (int8_t)(q[l] & 0xF);
                }
            }
            else
            {
                for (size_t l = 0; l < 32; ++l)
                {
                    output[l] = (int8_t)(q[l] >> 4);
                }
            }
        }

#if defined(__AVX2__)
        inline void transcode_q4_k_to_int8_avx2(
            const Q4_KBlock &block,
            size_t subblock_idx,
            int8_t *output,
            float *scale,
            float *min_val)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("transcode_q4_k_to_int8_avx2: subblock_idx out of range");
            }

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *q = block.qs + group_idx * 32;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);

            *scale = d * sc;
            *min_val = -dmin * m;

            if (is_second_half == 0)
            {
                // Lower 4 bits
                const __m256i mask = _mm256_set1_epi8(0xF);
                const __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q));
                const __m256i result = _mm256_and_si256(data, mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), result);
            }
            else
            {
                // Upper 4 bits
                const __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q));
                const __m256i shifted = _mm256_srli_epi16(data, 4);
                const __m256i mask = _mm256_set1_epi8(0x0F);
                const __m256i result = _mm256_and_si256(shifted, mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), result);
            }
        }
#endif

#if defined(__AVX512F__)
        inline void transcode_q4_k_to_int8_avx512(
            const Q4_KBlock &block,
            size_t subblock_idx,
            int8_t *output,
            float *scale,
            float *min_val)
        {
            // For 32 elements (32 bytes), AVX2 is sufficient and optimal.
            // AVX512 doesn't add benefit for 8-bit integer ops on 32 bytes unless we use ZMM (64 bytes).
            // But we only have 32 elements.
            transcode_q4_k_to_int8_avx2(block, subblock_idx, output, scale, min_val);
        }
#endif

        inline void transcode_q4_k_to_int8(
            const Q4_KBlock &block,
            size_t subblock_idx,
            int8_t *output,
            float *scale,
            float *min_val)
        {
#if defined(__AVX512F__)
            transcode_q4_k_to_int8_avx512(block, subblock_idx, output, scale, min_val);
#elif defined(__AVX2__)
            transcode_q4_k_to_int8_avx2(block, subblock_idx, output, scale, min_val);
#else
            transcode_q4_k_to_int8_scalar(block, subblock_idx, output, scale, min_val);
#endif
        }

        // =====================================================================
        // Q4_K → Q8_0
        // =====================================================================

        /**
         * @brief Scalar decode Q4_K to Q8_0 (one 32-element sub-block from 256-element super-block)
         * @param block Q4_K super-block (256 elements, 144 bytes)
         * @param subblock_idx Which 32-element sub-block to decode (0-7)
         * @param q8_qs Output Q8_0 quantized values (32 int8_t)
         * @param q8_scale Output Q8_0 scale (FP16)
         */
        inline void decode_q4_k_to_q8_0_scalar(
            const Q4_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q4_k_to_q8_0_scalar: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            // Q4_K processes 256 elements in 8 groups of 32 elements
            // Each group has 2 sub-groups of 16 elements with their own scale/min
            const size_t group_idx = subblock_idx / 2;      // 0-3 (which 64-element group)
            const size_t is_second_half = subblock_idx % 2; // 0 or 1 (first or second 32 within group)

            const size_t is = group_idx * 2 + is_second_half; // Scale index 0-7
            const uint8_t *q = block.qs + group_idx * 32;     // Pointer to packed 4-bit values

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            if (is_second_half == 0)
            {
                // First 32 elements: lower 4 bits
                for (size_t l = 0; l < 32; ++l)
                {
                    tmp[l] = dl * (q[l] & 0xF) - ml;
                }
            }
            else
            {
                // Second 32 elements: upper 4 bits
                for (size_t l = 0; l < 32; ++l)
                {
                    tmp[l] = dl * (q[l] >> 4) - ml;
                }
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }

#if defined(__AVX2__)
        inline void decode_q4_k_to_q8_0_avx2(
            const Q4_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q4_k_to_q8_0_avx2: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *q = block.qs + group_idx * 32;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            const __m256 dl_vec = _mm256_set1_ps(dl);
            const __m256 ml_vec = _mm256_set1_ps(ml);

            if (is_second_half == 0)
            {
                // Lower 4 bits: process 8 at a time
                for (size_t l = 0; l < 32; l += 8)
                {
                    __m128i q_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + l));
                    __m256i q_32 = _mm256_cvtepu8_epi32(q_bytes);
                    __m256i low_bits = _mm256_and_si256(q_32, _mm256_set1_epi32(0xF));
                    __m256 q_float = _mm256_cvtepi32_ps(low_bits);
                    __m256 result = _mm256_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm256_storeu_ps(tmp + l, result);
                }
            }
            else
            {
                // Upper 4 bits: process 8 at a time
                for (size_t l = 0; l < 32; l += 8)
                {
                    __m128i q_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + l));
                    __m256i q_32 = _mm256_cvtepu8_epi32(q_bytes);
                    __m256i high_bits = _mm256_srli_epi32(q_32, 4);
                    __m256 q_float = _mm256_cvtepi32_ps(high_bits);
                    __m256 result = _mm256_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm256_storeu_ps(tmp + l, result);
                }
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

#if defined(__AVX512F__)
        inline void decode_q4_k_to_q8_0_avx512(
            const Q4_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q4_k_to_q8_0_avx512: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *q = block.qs + group_idx * 32;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            const __m512 dl_vec = _mm512_set1_ps(dl);
            const __m512 ml_vec = _mm512_set1_ps(ml);

            if (is_second_half == 0)
            {
                // Lower 4 bits: process 16 at a time
                for (size_t l = 0; l < 32; l += 16)
                {
                    __m128i q_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + l));
                    __m512i q_32 = _mm512_cvtepu8_epi32(q_bytes);
                    __m512i low_bits = _mm512_and_si512(q_32, _mm512_set1_epi32(0xF));
                    __m512 q_float = _mm512_cvtepi32_ps(low_bits);
                    __m512 result = _mm512_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm512_storeu_ps(tmp + l, result);
                }
            }
            else
            {
                // Upper 4 bits: process 16 at a time
                for (size_t l = 0; l < 32; l += 16)
                {
                    __m128i q_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + l));
                    __m512i q_32 = _mm512_cvtepu8_epi32(q_bytes);
                    __m512i high_bits = _mm512_srli_epi32(q_32, 4);
                    __m512 q_float = _mm512_cvtepi32_ps(high_bits);
                    __m512 result = _mm512_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm512_storeu_ps(tmp + l, result);
                }
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

        inline void decode_q4_k_to_q8_0(
            const Q4_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__)
            decode_q4_k_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
#elif defined(__AVX2__)
            decode_q4_k_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
#else
            decode_q4_k_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
#endif
        }

        // =====================================================================
        // Q5_K → Q8_0
        // =====================================================================

        /**
         * @brief Scalar decode Q5_K to Q8_0 (one 32-element sub-block from 256-element super-block)
         * @param block Q5_K super-block (256 elements, 176 bytes)
         * @param subblock_idx Which 32-element sub-block to decode (0-7)
         * @param q8_qs Output Q8_0 quantized values (32 int8_t)
         * @param q8_scale Output Q8_0 scale (FP16)
         */
        inline void decode_q5_k_to_q8_0_scalar(
            const Q5_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q5_k_to_q8_0_scalar: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            // Q5_K processes 256 elements in 8 groups of 32 elements
            // Each group has 2 sub-groups of 16 elements with their own scale/min
            const size_t group_idx = subblock_idx / 2;      // 0-3 (which 64-element group)
            const size_t is_second_half = subblock_idx % 2; // 0 or 1 (first or second 32 within group)

            const size_t is = group_idx * 2 + is_second_half; // Scale index 0-7
            const uint8_t *ql = block.qs + group_idx * 32;    // Pointer to packed 4-bit low values
            const uint8_t *qh = block.qh + group_idx * 8;     // Pointer to high bits

            // High bit masks rotate through groups
            uint8_t u_mask = (1 << (group_idx * 2)) << is_second_half;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            if (is_second_half == 0)
            {
                // First 32 elements: lower 4 bits + high bit from u_mask
                for (size_t l = 0; l < 32; ++l)
                {
                    const uint8_t q_low = ql[l] & 0xF;
                    const uint8_t q_high = (qh[l] & u_mask) ? 16 : 0;
                    tmp[l] = dl * (q_low + q_high) - ml;
                }
            }
            else
            {
                // Second 32 elements: upper 4 bits + high bit from u_mask
                for (size_t l = 0; l < 32; ++l)
                {
                    const uint8_t q_low = ql[l] >> 4;
                    const uint8_t q_high = (qh[l] & u_mask) ? 16 : 0;
                    tmp[l] = dl * (q_low + q_high) - ml;
                }
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }

#if defined(__AVX2__)
        inline void decode_q5_k_to_q8_0_avx2(
            const Q5_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q5_k_to_q8_0_avx2: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *ql = block.qs + group_idx * 32;
            const uint8_t *qh = block.qh + group_idx * 8;

            uint8_t u_mask = (1 << (group_idx * 2)) << is_second_half;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            const __m256 dl_vec = _mm256_set1_ps(dl);
            const __m256 ml_vec = _mm256_set1_ps(ml);
            const __m256i u_mask_vec = _mm256_set1_epi32(u_mask);
            const __m256i high_bit_val = _mm256_set1_epi32(16);

            if (is_second_half == 0)
            {
                // Lower 4 bits: process 8 at a time
                for (size_t l = 0; l < 32; l += 8)
                {
                    __m128i ql_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(ql + l));
                    __m256i ql_32 = _mm256_cvtepu8_epi32(ql_bytes);
                    __m256i q_low = _mm256_and_si256(ql_32, _mm256_set1_epi32(0xF));

                    __m128i qh_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qh + l));
                    __m256i qh_32 = _mm256_cvtepu8_epi32(qh_bytes);
                    __m256i masked = _mm256_and_si256(qh_32, u_mask_vec);
                    __m256i q_high = _mm256_and_si256(
                        _mm256_cmpgt_epi32(masked, _mm256_setzero_si256()),
                        high_bit_val);

                    __m256i q_total = _mm256_add_epi32(q_low, q_high);
                    __m256 q_float = _mm256_cvtepi32_ps(q_total);
                    __m256 result = _mm256_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm256_storeu_ps(tmp + l, result);
                }
            }
            else
            {
                // Upper 4 bits: process 8 at a time
                for (size_t l = 0; l < 32; l += 8)
                {
                    __m128i ql_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(ql + l));
                    __m256i ql_32 = _mm256_cvtepu8_epi32(ql_bytes);
                    __m256i q_low = _mm256_srli_epi32(ql_32, 4);

                    __m128i qh_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qh + l));
                    __m256i qh_32 = _mm256_cvtepu8_epi32(qh_bytes);
                    __m256i masked = _mm256_and_si256(qh_32, u_mask_vec);
                    __m256i q_high = _mm256_and_si256(
                        _mm256_cmpgt_epi32(masked, _mm256_setzero_si256()),
                        high_bit_val);

                    __m256i q_total = _mm256_add_epi32(q_low, q_high);
                    __m256 q_float = _mm256_cvtepi32_ps(q_total);
                    __m256 result = _mm256_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm256_storeu_ps(tmp + l, result);
                }
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

#if defined(__AVX512F__)
        inline void decode_q5_k_to_q8_0_avx512(
            const Q5_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_q5_k_to_q8_0_avx512: subblock_idx out of range");
            }

            alignas(64) float tmp[Q8_0Block::BLOCK_SIZE];

            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            const size_t group_idx = subblock_idx / 2;
            const size_t is_second_half = subblock_idx % 2;
            const size_t is = group_idx * 2 + is_second_half;
            const uint8_t *ql = block.qs + group_idx * 32;
            const uint8_t *qh = block.qh + group_idx * 8;

            uint8_t u_mask = (1 << (group_idx * 2)) << is_second_half;

            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float dl = d * sc;
            const float ml = dmin * m;

            const __m512 dl_vec = _mm512_set1_ps(dl);
            const __m512 ml_vec = _mm512_set1_ps(ml);
            const __m512i u_mask_vec = _mm512_set1_epi32(u_mask);
            const __m512i high_bit_val = _mm512_set1_epi32(16);

            if (is_second_half == 0)
            {
                // Lower 4 bits: process 16 at a time
                for (size_t l = 0; l < 32; l += 16)
                {
                    __m128i ql_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql + l));
                    __m512i ql_32 = _mm512_cvtepu8_epi32(ql_bytes);
                    __m512i q_low = _mm512_and_si512(ql_32, _mm512_set1_epi32(0xF));

                    __m128i qh_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qh + l));
                    __m512i qh_32 = _mm512_cvtepu8_epi32(qh_bytes);
                    __mmask16 test_mask = _mm512_test_epi32_mask(
                        _mm512_and_si512(qh_32, u_mask_vec),
                        _mm512_and_si512(qh_32, u_mask_vec));
                    __m512i q_high = _mm512_mask_blend_epi32(test_mask, _mm512_setzero_si512(), high_bit_val);

                    __m512i q_total = _mm512_add_epi32(q_low, q_high);
                    __m512 q_float = _mm512_cvtepi32_ps(q_total);
                    __m512 result = _mm512_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm512_storeu_ps(tmp + l, result);
                }
            }
            else
            {
                // Upper 4 bits: process 16 at a time
                for (size_t l = 0; l < 32; l += 16)
                {
                    __m128i ql_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql + l));
                    __m512i ql_32 = _mm512_cvtepu8_epi32(ql_bytes);
                    __m512i q_low = _mm512_srli_epi32(ql_32, 4);

                    __m128i qh_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qh + l));
                    __m512i qh_32 = _mm512_cvtepu8_epi32(qh_bytes);
                    __mmask16 test_mask = _mm512_test_epi32_mask(
                        _mm512_and_si512(qh_32, u_mask_vec),
                        _mm512_and_si512(qh_32, u_mask_vec));
                    __m512i q_high = _mm512_mask_blend_epi32(test_mask, _mm512_setzero_si512(), high_bit_val);

                    __m512i q_total = _mm512_add_epi32(q_low, q_high);
                    __m512 q_float = _mm512_cvtepi32_ps(q_total);
                    __m512 result = _mm512_fmsub_ps(dl_vec, q_float, ml_vec);
                    _mm512_storeu_ps(tmp + l, result);
                }
            }

            quantize_fp32_to_q8_0(tmp, Q8_0Block::BLOCK_SIZE, q8_qs, q8_scale);
        }
#endif

        inline void decode_q5_k_to_q8_0(
            const Q5_KBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__)
            decode_q5_k_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
#elif defined(__AVX2__)
            decode_q5_k_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
#else
            decode_q5_k_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
#endif
        }

        // =====================================================================
        // Q6_K → INT8 Transcoding
        // =====================================================================

        inline void transcode_q6_k_to_int8_scalar(
            const Q6_KBlock &block,
            size_t subblock_idx,
            int8_t *out_int8,
            float *out_scale,
            float *out_min)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("transcode_q6_k_to_int8_scalar: subblock_idx out of range");
            }

            const size_t half = subblock_idx / 4;
            const size_t sub_in_half = subblock_idx % 4;

            const uint8_t *ql_base = block.ql + half * 64;
            const uint8_t *qh_base = block.qh + half * 32;
            const int8_t *sc_base = block.scales + half * 8;

            const uint8_t *ql_ptr = ql_base;
            const uint8_t *qh_ptr = qh_base;
            int ql_shift = 0;
            int qh_shift = 0;
            int sc_offset = 0;

            if (sub_in_half == 1)
            {
                ql_ptr += 32;
                qh_shift = 2;
                sc_offset = 2;
            }
            else if (sub_in_half == 2)
            {
                ql_shift = 4;
                qh_shift = 4;
                sc_offset = 4;
            }
            else if (sub_in_half == 3)
            {
                ql_ptr += 32;
                ql_shift = 4;
                qh_shift = 6;
                sc_offset = 6;
            }

            const float d = fp16_to_fp32(block.d);
            const int8_t sc0 = sc_base[sc_offset + 0];
            const int8_t sc1 = sc_base[sc_offset + 1];

            const float max_sc = std::max(std::abs(sc0), std::abs(sc1));
            if (max_sc == 0)
            {
                *out_scale = 0.0f;
                *out_min = 0.0f;
                std::memset(out_int8, 0, 32);
                return;
            }

            *out_scale = d * max_sc * 32.0f / 127.0f;
            *out_min = 0.0f;

            const float factor0 = (float)sc0 * 127.0f / (max_sc * 32.0f);
            const float factor1 = (float)sc1 * 127.0f / (max_sc * 32.0f);

            for (int l = 0; l < 32; ++l)
            {
                uint8_t ql_val = (ql_ptr[l] >> ql_shift) & 0xF;
                uint8_t qh_val = (qh_ptr[l] >> qh_shift) & 3;
                int8_t q = (int8_t)(ql_val | (qh_val << 4)) - 32;

                float f = (l < 16) ? factor0 : factor1;
                out_int8[l] = (int8_t)std::nearbyint(q * f);
            }
        }

#if defined(__AVX2__)
        inline void transcode_q6_k_to_int8_avx2(
            const Q6_KBlock &block,
            size_t subblock_idx,
            int8_t *out_int8,
            float *out_scale,
            float *out_min)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("transcode_q6_k_to_int8_avx2: subblock_idx out of range");
            }

            const size_t half = subblock_idx / 4;
            const size_t sub_in_half = subblock_idx % 4;

            const uint8_t *ql_base = block.ql + half * 64;
            const uint8_t *qh_base = block.qh + half * 32;
            const int8_t *sc_base = block.scales + half * 8;

            const uint8_t *ql_ptr = ql_base;
            const uint8_t *qh_ptr = qh_base;
            int ql_shift = 0;
            int qh_shift = 0;
            int sc_offset = 0;

            if (sub_in_half == 1)
            {
                ql_ptr += 32;
                qh_shift = 2;
                sc_offset = 2;
            }
            else if (sub_in_half == 2)
            {
                ql_shift = 4;
                qh_shift = 4;
                sc_offset = 4;
            }
            else if (sub_in_half == 3)
            {
                ql_ptr += 32;
                ql_shift = 4;
                qh_shift = 6;
                sc_offset = 6;
            }

            const float d = fp16_to_fp32(block.d);
            const int8_t sc0 = sc_base[sc_offset + 0];
            const int8_t sc1 = sc_base[sc_offset + 1];

            const float max_sc = std::max(std::abs(sc0), std::abs(sc1));
            if (max_sc == 0)
            {
                *out_scale = 0.0f;
                *out_min = 0.0f;
                std::memset(out_int8, 0, 32);
                return;
            }

            *out_scale = d * max_sc * 32.0f / 127.0f;
            *out_min = 0.0f;

            const float factor0 = (float)sc0 * 127.0f / (max_sc * 32.0f);
            const float factor1 = (float)sc1 * 127.0f / (max_sc * 32.0f);

            const __m256 f0_vec = _mm256_set1_ps(factor0);
            const __m256 f1_vec = _mm256_set1_ps(factor1);
            const __m256i offset_32 = _mm256_set1_epi32(32);
            const __m256i mask_f = _mm256_set1_epi32(0xF);
            const __m256i mask_3 = _mm256_set1_epi32(3);

            for (int l = 0; l < 32; l += 16)
            {
                __m128i ql_16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql_ptr + l));
                __m128i qh_16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qh_ptr + l));

                __m256i ql_lo = _mm256_cvtepu8_epi32(ql_16);
                __m256i ql_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(ql_16, 8));

                __m256i qh_lo = _mm256_cvtepu8_epi32(qh_16);
                __m256i qh_hi = _mm256_cvtepu8_epi32(_mm_srli_si128(qh_16, 8));

                if (ql_shift > 0)
                {
                    ql_lo = _mm256_srli_epi32(ql_lo, ql_shift);
                    ql_hi = _mm256_srli_epi32(ql_hi, ql_shift);
                }
                ql_lo = _mm256_and_si256(ql_lo, mask_f);
                ql_hi = _mm256_and_si256(ql_hi, mask_f);

                if (qh_shift > 0)
                {
                    qh_lo = _mm256_srli_epi32(qh_lo, qh_shift);
                    qh_hi = _mm256_srli_epi32(qh_hi, qh_shift);
                }
                qh_lo = _mm256_and_si256(qh_lo, mask_3);
                qh_hi = _mm256_and_si256(qh_hi, mask_3);

                __m256i q_lo = _mm256_or_si256(ql_lo, _mm256_slli_epi32(qh_lo, 4));
                __m256i q_hi = _mm256_or_si256(ql_hi, _mm256_slli_epi32(qh_hi, 4));

                q_lo = _mm256_sub_epi32(q_lo, offset_32);
                q_hi = _mm256_sub_epi32(q_hi, offset_32);

                __m256 f_vec = (l < 16) ? f0_vec : f1_vec;

                __m256 res_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(q_lo), f_vec);
                __m256 res_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(q_hi), f_vec);

                __m256i i_lo = _mm256_cvtps_epi32(_mm256_round_ps(res_lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256i i_hi = _mm256_cvtps_epi32(_mm256_round_ps(res_hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

                __m128i i_lo_128 = _mm_packs_epi32(_mm256_castsi256_si128(i_lo), _mm256_extracti128_si256(i_lo, 1));
                __m128i i_hi_128 = _mm_packs_epi32(_mm256_castsi256_si128(i_hi), _mm256_extracti128_si256(i_hi, 1));

                __m128i res_8 = _mm_packs_epi16(i_lo_128, i_hi_128);

                _mm_storeu_si128(reinterpret_cast<__m128i *>(out_int8 + l), res_8);
            }
        }
#endif

#if defined(__AVX512F__)
        inline void transcode_q6_k_to_int8_avx512(
            const Q6_KBlock &block,
            size_t subblock_idx,
            int8_t *out_int8,
            float *out_scale,
            float *out_min)
        {
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("transcode_q6_k_to_int8_avx512: subblock_idx out of range");
            }

            const size_t half = subblock_idx / 4;
            const size_t sub_in_half = subblock_idx % 4;

            const uint8_t *ql_base = block.ql + half * 64;
            const uint8_t *qh_base = block.qh + half * 32;
            const int8_t *sc_base = block.scales + half * 8;

            const uint8_t *ql_ptr = ql_base;
            const uint8_t *qh_ptr = qh_base;
            int ql_shift = 0;
            int qh_shift = 0;
            int sc_offset = 0;

            if (sub_in_half == 1)
            {
                ql_ptr += 32;
                qh_shift = 2;
                sc_offset = 2;
            }
            else if (sub_in_half == 2)
            {
                ql_shift = 4;
                qh_shift = 4;
                sc_offset = 4;
            }
            else if (sub_in_half == 3)
            {
                ql_ptr += 32;
                ql_shift = 4;
                qh_shift = 6;
                sc_offset = 6;
            }

            const float d = fp16_to_fp32(block.d);
            const int8_t sc0 = sc_base[sc_offset + 0];
            const int8_t sc1 = sc_base[sc_offset + 1];

            const float max_sc = std::max(std::abs(sc0), std::abs(sc1));
            if (max_sc == 0)
            {
                *out_scale = 0.0f;
                *out_min = 0.0f;
                std::memset(out_int8, 0, 32);
                return;
            }

            *out_scale = d * max_sc * 32.0f / 127.0f;
            *out_min = 0.0f;

            const float factor0 = (float)sc0 * 127.0f / (max_sc * 32.0f);
            const float factor1 = (float)sc1 * 127.0f / (max_sc * 32.0f);

            const __m512 f0_vec = _mm512_set1_ps(factor0);
            const __m512 f1_vec = _mm512_set1_ps(factor1);
            const __m512i offset_32 = _mm512_set1_epi32(32);

            for (int l = 0; l < 32; l += 16)
            {
                __m128i ql_16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql_ptr + l));
                __m128i qh_16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qh_ptr + l));

                __m512i ql_32 = _mm512_cvtepu8_epi32(ql_16);
                __m512i qh_32 = _mm512_cvtepu8_epi32(qh_16);

                if (ql_shift > 0)
                    ql_32 = _mm512_srli_epi32(ql_32, ql_shift);
                ql_32 = _mm512_and_si512(ql_32, _mm512_set1_epi32(0xF));

                if (qh_shift > 0)
                    qh_32 = _mm512_srli_epi32(qh_32, qh_shift);
                qh_32 = _mm512_and_si512(qh_32, _mm512_set1_epi32(3));

                __m512i q_combined = _mm512_or_si512(ql_32, _mm512_slli_epi32(qh_32, 4));
                __m512i q_centered = _mm512_sub_epi32(q_combined, offset_32);

                __m512 q_float = _mm512_cvtepi32_ps(q_centered);
                __m512 f_vec = (l < 16) ? f0_vec : f1_vec;
                __m512 res_float = _mm512_mul_ps(q_float, f_vec);

                __m512i res_int = _mm512_cvt_roundps_epi32(res_float, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                __m128i res_8 = _mm512_cvtepi32_epi8(res_int);

                _mm_storeu_si128(reinterpret_cast<__m128i *>(out_int8 + l), res_8);
            }
        }
#endif

        inline void transcode_q6_k_to_int8(
            const Q6_KBlock &block,
            size_t subblock_idx,
            int8_t *out_int8,
            float *out_scale,
            float *out_min)
        {
#if defined(__AVX512F__)
            transcode_q6_k_to_int8_avx512(block, subblock_idx, out_int8, out_scale, out_min);
#elif defined(__AVX2__)
            transcode_q6_k_to_int8_avx2(block, subblock_idx, out_int8, out_scale, out_min);
#else
            transcode_q6_k_to_int8_scalar(block, subblock_idx, out_int8, out_scale, out_min);
#endif
        }

        // =====================================================================
        // FP32 → Q8_0 (direct quantization, already in FP32 format)
        // =====================================================================

        /**
         * @brief Decode FP32 block to Q8_0 format (scalar)
         * @param fp32_data Source FP32 data (32 elements)
         * @param output_qs Output Q8_0 quantized values (32 int8_t)
         * @param output_scale Output Q8_0 scale (FP16)
         *
         * FP32 is already fully decoded - just quantize to Q8_0.
         * This is used for converting activation tensors to Q8_0 for integer GEMM.
         */
        inline void decode_fp32_to_q8_0_scalar(
            const float *fp32_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
            quantize_fp32_to_q8_0_scalar(fp32_data, 32, output_qs, output_scale);
        }

        /**
         * @brief Decode FP32 block to Q8_0 format (AVX-512)
         */
        inline void decode_fp32_to_q8_0_avx512(
            const float *fp32_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q8_0_avx512(fp32_data, 32, output_qs, output_scale);
#else
            decode_fp32_to_q8_0_scalar(fp32_data, output_qs, output_scale);
#endif
        }

        /**
         * @brief Decode FP32 block to Q8_0 format (AVX2)
         */
        inline void decode_fp32_to_q8_0_avx2(
            const float *fp32_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#ifdef __AVX2__
            quantize_fp32_to_q8_0_avx2(fp32_data, 32, output_qs, output_scale);
#else
            decode_fp32_to_q8_0_scalar(fp32_data, output_qs, output_scale);
#endif
        }

        /**
         * @brief Decode FP32 block to Q8_0 format (auto-dispatch)
         */
        inline void decode_fp32_to_q8_0(
            const float *fp32_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#if defined(__AVX512F__)
            decode_fp32_to_q8_0_avx512(fp32_data, output_qs, output_scale);
#elif defined(__AVX2__)
            decode_fp32_to_q8_0_avx2(fp32_data, output_qs, output_scale);
#else
            decode_fp32_to_q8_0_scalar(fp32_data, output_qs, output_scale);
#endif
        }

        // =====================================================================
        // FP32 → Q8_1 (quantize FP32 to Q8_1 with pre-computed sum)
        // =====================================================================

        /**
         * @brief Decode FP32 block to Q8_1 format (scalar)
         *
         * FP32 is already fully decoded - just quantize to Q8_1 with pre-computed sum.
         * This is used for converting activation tensors to Q8_1 for integer GEMM.
         */
        inline void decode_fp32_to_q8_1_scalar(
            const float *fp32_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
            quantize_fp32_to_q8_1_scalar(fp32_data, 32, output_qs, output_scale, output_sum);
        }

        /**
         * @brief Decode FP32 block to Q8_1 format (AVX-512)
         */
        inline void decode_fp32_to_q8_1_avx512(
            const float *fp32_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q8_1_avx512(fp32_data, 32, output_qs, output_scale, output_sum);
#else
            decode_fp32_to_q8_1_scalar(fp32_data, output_qs, output_scale, output_sum);
#endif
        }

        /**
         * @brief Decode FP32 block to Q8_1 format (AVX2)
         */
        inline void decode_fp32_to_q8_1_avx2(
            const float *fp32_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#ifdef __AVX2__
            quantize_fp32_to_q8_1_avx2(fp32_data, 32, output_qs, output_scale, output_sum);
#else
            decode_fp32_to_q8_1_scalar(fp32_data, output_qs, output_scale, output_sum);
#endif
        }

        /**
         * @brief Decode FP32 block to Q8_1 format (auto-dispatch)
         */
        inline void decode_fp32_to_q8_1(
            const float *fp32_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#if defined(__AVX512F__)
            decode_fp32_to_q8_1_avx512(fp32_data, output_qs, output_scale, output_sum);
#elif defined(__AVX2__)
            decode_fp32_to_q8_1_avx2(fp32_data, output_qs, output_scale, output_sum);
#else
            decode_fp32_to_q8_1_scalar(fp32_data, output_qs, output_scale, output_sum);
#endif
        }

        // =====================================================================
        // FP16 → Q8_0 (decode FP16 to FP32, then quantize to Q8_0)
        // =====================================================================

        /**
         * @brief Decode FP16 block to Q8_0 format (scalar)
         * @param fp16_data Source FP16 data (32 elements)
         * @param output_qs Output Q8_0 quantized values (32 int8_t)
         * @param output_scale Output Q8_0 scale (FP16)
         *
         * First converts FP16 → FP32, then quantizes to Q8_0.
         */
        inline void decode_fp16_to_q8_0_scalar(
            const uint16_t *fp16_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
            // Decode FP16 → FP32
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = fp16_to_fp32(fp16_data[i]);
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_buffer, 32, output_qs, output_scale);
        }

        /**
         * @brief Decode FP16 block to Q8_0 format (AVX-512)
         */
        inline void decode_fp16_to_q8_0_avx512(
            const uint16_t *fp16_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#if defined(__AVX512F__) && defined(__AVX512FP16__)
            // TODO: Use native FP16 intrinsics when available
            // For now, fall back to scalar decode + AVX512 quantize
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = fp16_to_fp32(fp16_data[i]);
            }
            quantize_fp32_to_q8_0_avx512(fp32_buffer, 32, output_qs, output_scale);
#else
            decode_fp16_to_q8_0_scalar(fp16_data, output_qs, output_scale);
#endif
        }

        /**
         * @brief Decode FP16 block to Q8_0 format (AVX2)
         */
        inline void decode_fp16_to_q8_0_avx2(
            const uint16_t *fp16_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#ifdef __AVX2__
            // Scalar decode FP16 → FP32, then AVX2 quantize
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = fp16_to_fp32(fp16_data[i]);
            }
            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, output_qs, output_scale);
#else
            decode_fp16_to_q8_0_scalar(fp16_data, output_qs, output_scale);
#endif
        }

        /**
         * @brief Decode FP16 block to Q8_0 format (auto-dispatch)
         */
        inline void decode_fp16_to_q8_0(
            const uint16_t *fp16_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#if defined(__AVX512F__) && defined(__AVX512FP16__)
            decode_fp16_to_q8_0_avx512(fp16_data, output_qs, output_scale);
#elif defined(__AVX2__)
            decode_fp16_to_q8_0_avx2(fp16_data, output_qs, output_scale);
#else
            decode_fp16_to_q8_0_scalar(fp16_data, output_qs, output_scale);
#endif
        }

        // =====================================================================
        // FP16 → Q8_1 (decode FP16 to FP32, then quantize to Q8_1)
        // =====================================================================

        /**
         * @brief Decode FP16 block to Q8_1 format (scalar)
         * @param fp16_data Source FP16 data (32 elements)
         * @param output_qs Output Q8_1 quantized values (32 int8_t)
         * @param output_scale Output Q8_1 scale (FP16)
         * @param output_sum Output Q8_1 pre-computed sum (FP16)
         *
         * First converts FP16 → FP32, then quantizes to Q8_1 with pre-computed sum.
         */
        inline void decode_fp16_to_q8_1_scalar(
            const uint16_t *fp16_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
            // Decode FP16 → FP32
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = fp16_to_fp32(fp16_data[i]);
            }

            // Quantize FP32 → Q8_1
            quantize_fp32_to_q8_1_scalar(fp32_buffer, 32, output_qs, output_scale, output_sum);
        }

        /**
         * @brief Decode FP16 block to Q8_1 format (AVX-512)
         */
        inline void decode_fp16_to_q8_1_avx512(
            const uint16_t *fp16_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#if defined(__AVX512F__) && defined(__AVX512FP16__)
            // TODO: Use native FP16 intrinsics when available
            // For now, fall back to scalar decode + AVX512 quantize
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = fp16_to_fp32(fp16_data[i]);
            }
            quantize_fp32_to_q8_1_avx512(fp32_buffer, 32, output_qs, output_scale, output_sum);
#else
            decode_fp16_to_q8_1_scalar(fp16_data, output_qs, output_scale, output_sum);
#endif
        }

        /**
         * @brief Decode FP16 block to Q8_1 format (AVX2)
         */
        inline void decode_fp16_to_q8_1_avx2(
            const uint16_t *fp16_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#ifdef __AVX2__
            // Scalar decode FP16 → FP32, then AVX2 quantize
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = fp16_to_fp32(fp16_data[i]);
            }
            quantize_fp32_to_q8_1_avx2(fp32_buffer, 32, output_qs, output_scale, output_sum);
#else
            decode_fp16_to_q8_1_scalar(fp16_data, output_qs, output_scale, output_sum);
#endif
        }

        /**
         * @brief Decode FP16 block to Q8_1 format (auto-dispatch)
         */
        inline void decode_fp16_to_q8_1(
            const uint16_t *fp16_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#if defined(__AVX512F__) && defined(__AVX512FP16__)
            decode_fp16_to_q8_1_avx512(fp16_data, output_qs, output_scale, output_sum);
#elif defined(__AVX2__)
            decode_fp16_to_q8_1_avx2(fp16_data, output_qs, output_scale, output_sum);
#else
            decode_fp16_to_q8_1_scalar(fp16_data, output_qs, output_scale, output_sum);
#endif
        }

        // =====================================================================
        // BF16 → Q8_0 (decode BF16 to FP32, then quantize to Q8_0)
        // =====================================================================

        /**
         * @brief Decode BF16 block to Q8_0 format (scalar)
         * @param bf16_data Source BF16 data (32 elements)
         * @param output_qs Output Q8_0 quantized values (32 int8_t)
         * @param output_scale Output Q8_0 scale (FP16)
         *
         * First converts BF16 → FP32, then quantizes to Q8_0.
         */
        inline void decode_bf16_to_q8_0_scalar(
            const uint16_t *bf16_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
            // Decode BF16 → FP32
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = bf16_to_fp32(bf16_data[i]);
            }

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_buffer, 32, output_qs, output_scale);
        }

        /**
         * @brief Decode BF16 block to Q8_0 format (AVX-512)
         */
        inline void decode_bf16_to_q8_0_avx512(
            const uint16_t *bf16_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            // Use vectorized BF16 → FP32 conversion
            float fp32_buffer[32];
            convert_bf16_to_fp32_avx512(bf16_data, fp32_buffer, 32);
            quantize_fp32_to_q8_0_avx512(fp32_buffer, 32, output_qs, output_scale);
#else
            decode_bf16_to_q8_0_scalar(bf16_data, output_qs, output_scale);
#endif
        }

        /**
         * @brief Decode BF16 block to Q8_0 format (AVX2)
         */
        inline void decode_bf16_to_q8_0_avx2(
            const uint16_t *bf16_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#ifdef __AVX2__
            // Use vectorized BF16 → FP32 conversion
            float fp32_buffer[32];
            convert_bf16_to_fp32_avx2(bf16_data, fp32_buffer, 32);
            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, output_qs, output_scale);
#else
            decode_bf16_to_q8_0_scalar(bf16_data, output_qs, output_scale);
#endif
        }

        /**
         * @brief Decode BF16 block to Q8_0 format (auto-dispatch)
         */
        inline void decode_bf16_to_q8_0(
            const uint16_t *bf16_data,
            int8_t *output_qs,
            uint16_t *output_scale)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_bf16_to_q8_0_avx512(bf16_data, output_qs, output_scale);
#elif defined(__AVX2__)
            decode_bf16_to_q8_0_avx2(bf16_data, output_qs, output_scale);
#else
            decode_bf16_to_q8_0_scalar(bf16_data, output_qs, output_scale);
#endif
        }

        // =====================================================================
        // BF16 → Q8_1 (decode BF16 to FP32, then quantize to Q8_1)
        // =====================================================================

        /**
         * @brief Decode BF16 block to Q8_1 format (scalar)
         * @param bf16_data Source BF16 data (32 elements)
         * @param output_qs Output Q8_1 quantized values (32 int8_t)
         * @param output_scale Output Q8_1 scale (FP16)
         * @param output_sum Output Q8_1 pre-computed sum (FP16)
         *
         * First converts BF16 → FP32, then quantizes to Q8_1 with pre-computed sum.
         */
        inline void decode_bf16_to_q8_1_scalar(
            const uint16_t *bf16_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
            // Decode BF16 → FP32
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = bf16_to_fp32(bf16_data[i]);
            }

            // Quantize FP32 → Q8_1
            quantize_fp32_to_q8_1_scalar(fp32_buffer, 32, output_qs, output_scale, output_sum);
        }

        /**
         * @brief Decode BF16 block to Q8_1 format (AVX-512)
         */
        inline void decode_bf16_to_q8_1_avx512(
            const uint16_t *bf16_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            // Scalar decode BF16 → FP32, then AVX512 quantize
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = bf16_to_fp32(bf16_data[i]);
            }
            quantize_fp32_to_q8_1_avx512(fp32_buffer, 32, output_qs, output_scale, output_sum);
#else
            decode_bf16_to_q8_1_scalar(bf16_data, output_qs, output_scale, output_sum);
#endif
        }

        /**
         * @brief Decode BF16 block to Q8_1 format (AVX2)
         */
        inline void decode_bf16_to_q8_1_avx2(
            const uint16_t *bf16_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#ifdef __AVX2__
            // Scalar decode BF16 → FP32, then AVX2 quantize
            float fp32_buffer[32];
            for (int i = 0; i < 32; ++i)
            {
                fp32_buffer[i] = bf16_to_fp32(bf16_data[i]);
            }
            quantize_fp32_to_q8_1_avx2(fp32_buffer, 32, output_qs, output_scale, output_sum);
#else
            decode_bf16_to_q8_1_scalar(bf16_data, output_qs, output_scale, output_sum);
#endif
        }

        /**
         * @brief Decode BF16 block to Q8_1 format (auto-dispatch)
         */
        inline void decode_bf16_to_q8_1(
            const uint16_t *bf16_data,
            int8_t *output_qs,
            uint16_t *output_scale,
            uint16_t *output_sum)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            decode_bf16_to_q8_1_avx512(bf16_data, output_qs, output_scale, output_sum);
#elif defined(__AVX2__)
            decode_bf16_to_q8_1_avx2(bf16_data, output_qs, output_scale, output_sum);
#else
            decode_bf16_to_q8_1_scalar(bf16_data, output_qs, output_scale, output_sum);
#endif
        }

        // ============================================================================
        // Q4_0 → INT8 Unpacking (IINT8Unpackable support)
        // ============================================================================

        /**
         * @brief Scalar implementation for unpacking Q4_0 block to int8
         * @param block Q4_0Block containing 32 values in 16 bytes
         * @param output Output buffer for 32 int8 values
         *
         * Q4_0 stores 32 4-bit values in 16 bytes:
         * - Lower nibble = first 16 values
         * - Upper nibble = last 16 values
         * Values are signed: -8..7 (4-bit signed with bias)
         */
        inline void unpack_q4_0_to_int8_scalar(const Q4_0Block &block, int8_t *output)
        {
            for (int j = 0; j < 16; ++j)
            {
                uint8_t byte = block.qs[j];
                // Lower nibble: values 0-15, Upper nibble: values 16-31
                // Q4_0 uses unsigned 0-15 representation, subtract 8 to get signed -8..7
                output[j] = static_cast<int8_t>((byte & 0xF) - 8);
                output[j + 16] = static_cast<int8_t>((byte >> 4) - 8);
            }
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief AVX512 dual-block implementation for Q4_0 → int8 unpacking
         * @param block1 First Q4_0Block
         * @param block2 Second Q4_0Block
         * @param output1 Output for first block (32 int8 values)
         * @param output2 Output for second block (32 int8 values)
         *
         * Processes 2 blocks (64 values) simultaneously for better throughput.
         */
        inline void unpack_q4_0_to_int8_avx512_dual(
            const Q4_0Block &block1, const Q4_0Block &block2,
            int8_t *output1, int8_t *output2)
        {
            // Load both blocks into a single 256-bit register
            __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block1.qs));
            __m128i raw2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block2.qs));
            __m256i combined = _mm256_set_m128i(raw2, raw1);

            // Promote to 512-bit for AVX512 operations
            __m512i raw512 = _mm512_castsi256_si512(combined);

            // Extract lower and upper nibbles
            __m512i low_mask = _mm512_set1_epi8(0x0F);
            __m512i bias = _mm512_set1_epi8(8);

            __m512i low_nibbles = _mm512_and_si512(raw512, low_mask);
            __m512i high_nibbles = _mm512_srli_epi16(raw512, 4);
            high_nibbles = _mm512_and_si512(high_nibbles, low_mask);

            // Subtract bias (8) to get signed values
            __m512i low_signed = _mm512_sub_epi8(low_nibbles, bias);
            __m512i high_signed = _mm512_sub_epi8(high_nibbles, bias);

            // Interleave low and high into final order
            // For Q4_0: output[0..15] = low nibbles, output[16..31] = high nibbles
            __m256i low_256 = _mm512_castsi512_si256(low_signed);
            __m256i high_256 = _mm512_castsi512_si256(high_signed);

            // Store block 1
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output1), _mm256_castsi256_si128(low_256));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output1 + 16), _mm256_castsi256_si128(high_256));

            // Store block 2
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output2), _mm256_extracti128_si256(low_256, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output2 + 16), _mm256_extracti128_si256(high_256, 1));
        }

        /**
         * @brief AVX512 single-block implementation for Q4_0 → int8 unpacking
         */
        inline void unpack_q4_0_to_int8_avx512(const Q4_0Block &block, int8_t *output)
        {
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);
            __m128i bias = _mm_set1_epi8(8);

            __m128i low_nibbles = _mm_and_si128(raw, low_mask);
            __m128i high_nibbles = _mm_srli_epi16(raw, 4);
            high_nibbles = _mm_and_si128(high_nibbles, low_mask);

            __m128i low_signed = _mm_sub_epi8(low_nibbles, bias);
            __m128i high_signed = _mm_sub_epi8(high_nibbles, bias);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), low_signed);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), high_signed);
        }
#endif // AVX512F + AVX512BW

#if defined(__AVX2__)
        /**
         * @brief AVX2 implementation for Q4_0 → int8 unpacking
         */
        inline void unpack_q4_0_to_int8_avx2(const Q4_0Block &block, int8_t *output)
        {
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);
            __m128i bias = _mm_set1_epi8(8);

            __m128i low_nibbles = _mm_and_si128(raw, low_mask);
            __m128i high_nibbles = _mm_srli_epi16(raw, 4);
            high_nibbles = _mm_and_si128(high_nibbles, low_mask);

            __m128i low_signed = _mm_sub_epi8(low_nibbles, bias);
            __m128i high_signed = _mm_sub_epi8(high_nibbles, bias);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), low_signed);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), high_signed);
        }
#endif // AVX2

        /**
         * @brief Auto-dispatching Q4_0 → int8 unpacker
         * @param block Q4_0Block to unpack
         * @param output Output buffer for 32 int8 values
         *
         * Selects best available SIMD implementation at compile time.
         */
        inline void unpack_q4_0_to_int8(const Q4_0Block &block, int8_t *output)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            unpack_q4_0_to_int8_avx512(block, output);
#elif defined(__AVX2__)
            unpack_q4_0_to_int8_avx2(block, output);
#else
            unpack_q4_0_to_int8_scalar(block, output);
#endif
        }

        /**
         * @brief Scalar implementation for Q5_0 → int8 unpacking
         *
         * Layout matches llama.cpp dequantize_row_q5_0():
         * - output[0..15] = low nibbles of qs[0..15] + high bits from qh
         * - output[16..31] = high nibbles of qs[0..15] + high bits from qh
         */
        inline void unpack_q5_0_to_int8_scalar(const Q5_0Block &block, int8_t *output)
        {
            // Match the layout from decodeBlockScalar / dequantize_row_q5_0
            uint32_t qh;
            std::memcpy(&qh, block.qh, sizeof(qh));

            for (int j = 0; j < 16; ++j)
            {
                // Extract high bits - same logic as decodeBlockScalar
                const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                const uint8_t xh_1 = ((qh >> (j + 12)) & 0x10);

                // Low nibble goes to output[j], high nibble to output[j+16]
                output[j + 0] = static_cast<int8_t>(((block.qs[j] & 0x0F) | xh_0) - 16);
                output[j + 16] = static_cast<int8_t>(((block.qs[j] >> 4) | xh_1) - 16);
            }
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief AVX512 implementation for Q5_0 → int8 unpacking
         * @note Currently falls back to scalar unpacking
         */
        inline void unpack_q5_0_to_int8_avx512(const Q5_0Block &block, int8_t *output)
        {
            unpack_q5_0_to_int8_scalar(block, output);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief AVX2 implementation for Q5_0 → int8 unpacking
         * @note Currently falls back to scalar unpacking
         */
        inline void unpack_q5_0_to_int8_avx2(const Q5_0Block &block, int8_t *output)
        {
            unpack_q5_0_to_int8_scalar(block, output);
        }
#endif

        /**
         * @brief Auto-dispatching Q5_0 → int8 unpacker
         */
        inline void unpack_q5_0_to_int8(const Q5_0Block &block, int8_t *output)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            unpack_q5_0_to_int8_avx512(block, output);
#elif defined(__AVX2__)
            unpack_q5_0_to_int8_avx2(block, output);
#else
            unpack_q5_0_to_int8_scalar(block, output);
#endif
        }

        /**
         * @brief Unpack Q4_1 block to plain int8 (scalar)
         * Unpacks 4-bit values to [0, 15] range.
         */
        inline void unpack_q4_1_to_int8_scalar(const Q4_1Block &block, int8_t *output)
        {
            for (int j = 0; j < 16; ++j)
            {
                uint8_t byte = block.qs[j];
                // Lower nibble: values 0-15, Upper nibble: values 16-31
                // Q4_1 uses unsigned 0-15 representation, no bias subtraction
                output[j] = static_cast<int8_t>(byte & 0xF);
                output[j + 16] = static_cast<int8_t>(byte >> 4);
            }
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief Transcode Q4_1 block directly to INT8 (affine) using AVX512
         *
         * This fuses dequantization and requantization into a single pass, avoiding FP32 expansion.
         *
         * Q4_1: v = q * d + m
         * INT8: v ~ (q_out - (-128)) * out_scale + out_min
         *
         * We compute q_out directly from q:
         * q_out = (q - min_q) * (255 / range_q) - 128
         *
         * @param block Input Q4_1 block
         * @param output Output INT8 buffer (32 elements)
         * @param out_scale Output scale
         * @param out_min Output min value
         */
        inline void transcode_q4_1_to_int8_avx512(const Q4_1Block &block, int8_t *output, float *out_scale, float *out_min)
        {
            // 1. Unpack nibbles to bytes (0-15)
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

#if defined(__AVX512VL__)
            __m256i raw_16 = _mm256_cvtepu8_epi16(raw);
            __m256i low_mask = _mm256_set1_epi16(0x0F);
            __m256i qs_low = _mm256_and_si256(raw_16, low_mask);
            __m256i qs_high = _mm256_and_si256(_mm256_srli_epi16(raw_16, 4), low_mask);

            // Pack back to 8-bit (128-bit vector)
            __m128i out_low = _mm256_cvtepi16_epi8(qs_low);
            __m128i out_high = _mm256_cvtepi16_epi8(qs_high);

            // Combine into one __m256i for processing
            __m256i v_qs = _mm256_inserti128_si256(_mm256_castsi128_si256(out_low), out_high, 1);

            // Convert to float for min/max/scale (process as two 16-element vectors)
            __m128i v_qs_lo_i = _mm256_castsi256_si128(v_qs);
            __m128i v_qs_hi_i = _mm256_extracti128_si256(v_qs, 1);

            __m512 v_qs_lo_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v_qs_lo_i));
            __m512 v_qs_hi_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v_qs_hi_i));

            float min_q = std::min(_mm512_reduce_min_ps(v_qs_lo_f), _mm512_reduce_min_ps(v_qs_hi_f));
            float max_q = std::max(_mm512_reduce_max_ps(v_qs_lo_f), _mm512_reduce_max_ps(v_qs_hi_f));

            // 3. Compute scale/bias
            float range_q = max_q - min_q;

            if (range_q < 0.5f)
            {
                // Constant value
                float d = fp16_to_fp32(block.d);
                float m = fp16_to_fp32(block.m);
                *out_scale = 0.0f;
                *out_min = min_q * d + m; // The actual value
                // Fill output with -128
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), _mm256_set1_epi8(-128));
            }
            else
            {
                float factor = 255.0f / range_q;
                float bias = -128.0f - min_q * factor;

                __m512 v_factor = _mm512_set1_ps(factor);
                __m512 v_bias = _mm512_set1_ps(bias);

                // 4. Compute output: v_out = v_qs * factor + bias
                __m512 v_out_lo_f = _mm512_fmadd_ps(v_qs_lo_f, v_factor, v_bias);
                __m512 v_out_hi_f = _mm512_fmadd_ps(v_qs_hi_f, v_factor, v_bias);

                // Convert to int32 -> int8
                __m512i v_out_lo_i = _mm512_cvtps_epi32(v_out_lo_f);
                __m512i v_out_hi_i = _mm512_cvtps_epi32(v_out_hi_f);

                __m128i v_out_lo_b = _mm512_cvtepi32_epi8(v_out_lo_i);
                __m128i v_out_hi_b = _mm512_cvtepi32_epi8(v_out_hi_i);

                _mm_storeu_si128(reinterpret_cast<__m128i *>(output), v_out_lo_b);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), v_out_hi_b);

                // 5. Compute output metadata
                float d = fp16_to_fp32(block.d);
                float m = fp16_to_fp32(block.m);
                *out_scale = range_q * d / 255.0f;
                *out_min = (min_q * d + m) + (*out_scale) * 128.0f; // Bias correction for -128 base
            }
#else
            // Fallback if no VL (pure AVX512F)
            // Just unpack and use scalar requantize for now to ensure correctness
            unpack_q4_1_to_int8_avx512(block, output);

            // We can still optimize the math part even if we unpack differently
            // But for now, let's just leave it as a TODO or fallback
            // Since we are targeting AVX512 machines which usually have VL (Skylake-X+)
            // Knights Landing is the only one without VL, and it's rare.
#endif
        }

        /**
         * @brief AVX512 implementation for Q4_1 → int8 unpacking
         */
        inline void unpack_q4_1_to_int8_avx512(const Q4_1Block &block, int8_t *output)
        {
#if defined(__AVX512VL__)
            // AVX512VL allows using 256-bit registers with AVX512 instructions
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

            // Expand to 16-bit (256-bit vector)
            __m256i raw_16 = _mm256_cvtepu8_epi16(raw);

            // Low nibbles
            __m256i low_mask = _mm256_set1_epi16(0x0F);
            __m256i low_nibbles = _mm256_and_si256(raw_16, low_mask);

            // High nibbles
            __m256i high_nibbles = _mm256_srli_epi16(raw_16, 4);

            // Pack back to 8-bit (128-bit vector)
            __m128i out_low = _mm256_cvtepi16_epi8(low_nibbles);
            __m128i out_high = _mm256_cvtepi16_epi8(high_nibbles);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), out_low);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), out_high);
#else
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);

            __m128i low_nibbles = _mm_and_si128(raw, low_mask);
            __m128i high_nibbles = _mm_srli_epi16(raw, 4);
            high_nibbles = _mm_and_si128(high_nibbles, low_mask);

            // No bias subtraction for Q4_1

            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), low_nibbles);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), high_nibbles);
#endif
        }
#endif // AVX512F + AVX512BW

#if defined(__AVX2__)
        /**
         * @brief AVX2 implementation for Q4_1 → int8 unpacking
         */
        inline void unpack_q4_1_to_int8_avx2(const Q4_1Block &block, int8_t *output)
        {
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);

            __m128i low_nibbles = _mm_and_si128(raw, low_mask);
            __m128i high_nibbles = _mm_srli_epi16(raw, 4);
            high_nibbles = _mm_and_si128(high_nibbles, low_mask);

            // No bias subtraction for Q4_1

            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), low_nibbles);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), high_nibbles);
        }
#endif // AVX2

        /**
         * @brief Auto-dispatching Q4_1 → int8 unpacker
         */
        inline void unpack_q4_1_to_int8(const Q4_1Block &block, int8_t *output)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            unpack_q4_1_to_int8_avx512(block, output);
#elif defined(__AVX2__)
            unpack_q4_1_to_int8_avx2(block, output);
#else
            unpack_q4_1_to_int8_scalar(block, output);
#endif
        }

        // ========================================================================
        // Q2_K Unpacking Helpers (for IINT8Unpackable)
        // ========================================================================

        /**
         * @brief Decode a specific 32-element sub-block of Q2_K to float
         *
         * Q2_K super-blocks (256 elements) are split into 8 sub-blocks of 32 elements.
         * This helper decodes one such sub-block to float, handling the complex
         * internal scaling logic of Q2_K.
         *
         * @param block The Q2_K super-block
         * @param sub_block_idx Index of the 32-element sub-block (0-7)
         * @param output Buffer for 32 float values
         */
        inline void decode_q2k_subblock_to_float_scalar(const Q2_KBlock &block, int sub_block_idx, float *output)
        {
            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            // Determine which 128-element chunk we are in (0 or 1)
            int chunk_idx = sub_block_idx / 4;
            // Determine which group within the chunk (0-3)
            int j = sub_block_idx % 4;

            int shift = j * 2;
            const uint8_t *q_ptr = block.qs + chunk_idx * 32;
            int scale_idx = sub_block_idx * 2;

            // First 16 elements
            {
                uint8_t sc = block.scales[scale_idx];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l)
                {
                    output[l] = dl * ((int8_t)((q_ptr[l] >> shift) & 3)) - ml;
                }
            }

            // Second 16 elements
            {
                uint8_t sc = block.scales[scale_idx + 1];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l)
                {
                    output[l + 16] = dl * ((int8_t)((q_ptr[l + 16] >> shift) & 3)) - ml;
                }
            }
        }

#if defined(__AVX512F__)
        inline void decode_q2k_subblock_to_float_avx512(const Q2_KBlock &block, int sub_block_idx, float *output)
        {
            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            int chunk_idx = sub_block_idx / 4;
            int j = sub_block_idx % 4;
            int shift = j * 2;
            const uint8_t *q_ptr = block.qs + chunk_idx * 32;
            int scale_idx = sub_block_idx * 2;

            // Load scales
            uint8_t sc0 = block.scales[scale_idx];
            uint8_t sc1 = block.scales[scale_idx + 1];

            float dl0 = d * (sc0 & 0xF);
            float ml0 = dmin * (sc0 >> 4);
            float dl1 = d * (sc1 & 0xF);
            float ml1 = dmin * (sc1 >> 4);

            // Load 32 bytes of qs
            __m256i q_bytes = _mm256_loadu_si256((const __m256i *)q_ptr);

            // First 16 elements
            __m128i q_low = _mm256_castsi256_si128(q_bytes);
            __m512i q32_0 = _mm512_cvtepu8_epi32(q_low);

            __m512i shift_v = _mm512_set1_epi32(shift);
            __m512i mask_v = _mm512_set1_epi32(3);
            q32_0 = _mm512_and_si512(_mm512_srlv_epi32(q32_0, shift_v), mask_v);

            __m512 vq0 = _mm512_cvtepi32_ps(q32_0);

            __m512 vdl0 = _mm512_set1_ps(dl0);
            __m512 vml0 = _mm512_set1_ps(ml0);
            __m512 res0 = _mm512_sub_ps(_mm512_mul_ps(vdl0, vq0), vml0);

            _mm512_storeu_ps(output, res0);

            // Second 16 elements
            __m128i q_high = _mm256_extracti128_si256(q_bytes, 1);
            __m512i q32_1 = _mm512_cvtepu8_epi32(q_high);

            q32_1 = _mm512_and_si512(_mm512_srlv_epi32(q32_1, shift_v), mask_v);
            __m512 vq1 = _mm512_cvtepi32_ps(q32_1);

            __m512 vdl1 = _mm512_set1_ps(dl1);
            __m512 vml1 = _mm512_set1_ps(ml1);
            __m512 res1 = _mm512_sub_ps(_mm512_mul_ps(vdl1, vq1), vml1);

            _mm512_storeu_ps(output + 16, res1);
        }
#endif

        inline void decode_q2k_subblock_to_float(const Q2_KBlock &block, int sub_block_idx, float *output)
        {
#if defined(__AVX512F__)
            decode_q2k_subblock_to_float_avx512(block, sub_block_idx, output);
#else
            decode_q2k_subblock_to_float_scalar(block, sub_block_idx, output);
#endif
        }

        inline void transcode_q2_k_to_int8_scalar(const Q2_KBlock &block, int sub_block_idx, int8_t *output, float *out_scale, float *out_min)
        {
            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            int chunk_idx = sub_block_idx / 4;
            int j = sub_block_idx % 4;
            int shift = j * 2;
            const uint8_t *q_ptr = block.qs + chunk_idx * 32;
            int scale_idx = sub_block_idx * 2;

            // Compute ranges for both halves
            uint8_t sc0 = block.scales[scale_idx];
            float dl0 = d * (sc0 & 0xF);
            float ml0 = dmin * (sc0 >> 4);
            float min0 = -ml0;
            float max0 = 3.0f * dl0 - ml0;

            uint8_t sc1 = block.scales[scale_idx + 1];
            float dl1 = d * (sc1 & 0xF);
            float ml1 = dmin * (sc1 >> 4);
            float min1 = -ml1;
            float max1 = 3.0f * dl1 - ml1;

            float global_min = std::min(min0, min1);
            float global_max = std::max(max0, max1);
            float range = global_max - global_min;

            if (range < 1e-5f)
            {
                *out_scale = 0.0f;
                *out_min = global_min;
                memset(output, -128, 32);
                return;
            }

            *out_scale = range / 255.0f;
            *out_min = global_min + 128.0f * (*out_scale);
            float inv_scale = 1.0f / (*out_scale);
            float bias = -(*out_min) * inv_scale;

            // Process first 16
            for (int l = 0; l < 16; ++l)
            {
                uint8_t q = (q_ptr[l] >> shift) & 3;
                float val = dl0 * q - ml0;
                int32_t i = static_cast<int32_t>(std::round(val * inv_scale + bias));
                output[l] = static_cast<int8_t>(std::max(-128, std::min(127, i)));
            }

            // Process second 16
            for (int l = 0; l < 16; ++l)
            {
                uint8_t q = (q_ptr[l + 16] >> shift) & 3;
                float val = dl1 * q - ml1;
                int32_t i = static_cast<int32_t>(std::round(val * inv_scale + bias));
                output[l + 16] = static_cast<int8_t>(std::max(-128, std::min(127, i)));
            }
        }

#if defined(__AVX2__)
        inline void transcode_q2_k_to_int8_avx2(const Q2_KBlock &block, int sub_block_idx, int8_t *output, float *out_scale, float *out_min)
        {
            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            int chunk_idx = sub_block_idx / 4;
            int j = sub_block_idx % 4;
            int shift = j * 2;
            const uint8_t *q_ptr = block.qs + chunk_idx * 32;
            int scale_idx = sub_block_idx * 2;

            uint8_t sc0 = block.scales[scale_idx];
            float dl0 = d * (sc0 & 0xF);
            float ml0 = dmin * (sc0 >> 4);
            float min0 = -ml0;
            float max0 = 3.0f * dl0 - ml0;

            uint8_t sc1 = block.scales[scale_idx + 1];
            float dl1 = d * (sc1 & 0xF);
            float ml1 = dmin * (sc1 >> 4);
            float min1 = -ml1;
            float max1 = 3.0f * dl1 - ml1;

            float global_min = std::min(min0, min1);
            float global_max = std::max(max0, max1);
            float range = global_max - global_min;

            if (range < 1e-5f)
            {
                *out_scale = 0.0f;
                *out_min = global_min;
                memset(output, -128, 32);
                return;
            }

            *out_scale = range / 255.0f;
            *out_min = global_min + 128.0f * (*out_scale);
            float inv_scale = 1.0f / (*out_scale);
            float bias = -(*out_min) * inv_scale;

            // Compute LUTs
            int8_t lut[32];
            for (int q = 0; q < 4; ++q)
            {
                float val0 = dl0 * q - ml0;
                int32_t i0 = static_cast<int32_t>(std::round(val0 * inv_scale + bias));
                lut[q] = static_cast<int8_t>(std::max(-128, std::min(127, i0)));

                float val1 = dl1 * q - ml1;
                int32_t i1 = static_cast<int32_t>(std::round(val1 * inv_scale + bias));
                lut[16 + q] = static_cast<int8_t>(std::max(-128, std::min(127, i1)));
            }

            __m256i lut_vec = _mm256_loadu_si256((const __m256i *)lut);
            __m256i qs = _mm256_loadu_si256((const __m256i *)q_ptr);

            __m128i count = _mm_cvtsi32_si128(shift);
            __m256i q_indices = _mm256_srl_epi16(qs, count);
            q_indices = _mm256_and_si256(q_indices, _mm256_set1_epi8(3));

            __m256i result = _mm256_shuffle_epi8(lut_vec, q_indices);
            _mm256_storeu_si256((__m256i *)output, result);
        }
#endif

        inline void transcode_q2_k_to_int8(const Q2_KBlock &block, int sub_block_idx, int8_t *output, float *out_scale, float *out_min)
        {
#if defined(__AVX2__)
            transcode_q2_k_to_int8_avx2(block, sub_block_idx, output, out_scale, out_min);
#else
            transcode_q2_k_to_int8_scalar(block, sub_block_idx, output, out_scale, out_min);
#endif
        }

        /**
         * @brief Re-quantize float values to affine INT8 (scale * q + min)
         *
         * Used to adapt complex formats (like Q2_K) to the simple IINT8Unpackable interface.
         * Maps the float range [min, max] to int8 range [-128, 127] (or subset).
         *
         * @param input Input float values (count elements)
         * @param count Number of elements
         * @param output Output int8 buffer
         * @param out_scale Output scale
         * @param out_min Output min value
         */
        inline void requantize_to_int8_affine_scalar(const float *input, int count, int8_t *output, float *out_scale, float *out_min)
        {
            float min_val = input[0];
            float max_val = input[0];

            for (int i = 1; i < count; ++i)
            {
                if (input[i] < min_val)
                    min_val = input[i];
                if (input[i] > max_val)
                    max_val = input[i];
            }

            float scale = (max_val - min_val) / 255.0f;
            if (scale < 1e-10f)
                scale = 0.0f; // Avoid division by zero / tiny scales

            float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            // We map min_val to -128
            // val = scale * (q + 128) + min_val
            // val = scale * q + (scale * 128 + min_val)
            // So effective min returned to kernel is (scale * 128 + min_val)
            // q = (val - min_val) / scale - 128

            for (int i = 0; i < count; ++i)
            {
                if (scale > 0.0f)
                {
                    float q_f = (input[i] - min_val) * inv_scale - 128.0f;
                    // Clamp to -128..127
                    q_f = std::max(-128.0f, std::min(127.0f, q_f));
                    output[i] = static_cast<int8_t>(std::round(q_f));
                }
                else
                {
                    output[i] = -128; // Constant value, maps to min_val
                }
            }

            *out_scale = scale;
            *out_min = scale * 128.0f + min_val;
        }

#if defined(__AVX512F__)
        inline void requantize_to_int8_affine_avx512(const float *input, int count, int8_t *output, float *out_scale, float *out_min)
        {
            // Handle 32 elements (common case for sub-blocks)
            if (count == 32)
            {
                __m512 v0 = _mm512_loadu_ps(input);
                __m512 v1 = _mm512_loadu_ps(input + 16);

                // Find min/max
                float min_val = _mm512_reduce_min_ps(v0);
                min_val = std::min(min_val, _mm512_reduce_min_ps(v1));

                float max_val = _mm512_reduce_max_ps(v0);
                max_val = std::max(max_val, _mm512_reduce_max_ps(v1));

                float scale = (max_val - min_val) / 255.0f;
                if (scale < 1e-10f)
                    scale = 0.0f;
                float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

                __m512 v_min = _mm512_set1_ps(min_val);
                __m512 v_inv_scale = _mm512_set1_ps(inv_scale);
                __m512 v_neg_128 = _mm512_set1_ps(-128.0f);

                // q = (val - min) * inv_scale - 128
                __m512 q0 = _mm512_add_ps(_mm512_mul_ps(_mm512_sub_ps(v0, v_min), v_inv_scale), v_neg_128);
                __m512 q1 = _mm512_add_ps(_mm512_mul_ps(_mm512_sub_ps(v1, v_min), v_inv_scale), v_neg_128);

                // Convert to int32 with saturation
                __m512i i0 = _mm512_cvtps_epi32(q0);
                __m512i i1 = _mm512_cvtps_epi32(q1);

#if defined(__AVX512BW__)
                // Pack 32-bit integers to 8-bit integers with saturation
                __m128i b0 = _mm512_cvtepi32_epi8(i0);
                __m128i b1 = _mm512_cvtepi32_epi8(i1);

                // Store results
                _mm_storeu_si128((__m128i *)output, b0);
                _mm_storeu_si128((__m128i *)(output + 16), b1);
#else
                // Fallback without AVX512BW: extract and pack manually
                // This is slow, but better than scalar if we are already in vector registers?
                // Actually, if we don't have BW, we might as well use scalar for the packing part or just store int32 and pack.
                // But let's just fallback to scalar implementation if BW is missing for simplicity,
                // or implement a manual pack.
                // Manual pack:
                // Extract 32-bit lanes, cast to int8_t.
                // Since we clamped in float domain (implicitly via min/max logic, but we should clamp explicitly to be safe),
                // we can just truncate.
                // But wait, we didn't clamp explicitly in float domain in the vector code above!
                // _mm512_cvtps_epi32 does not saturate to 8-bit range, it converts to 32-bit integer.
                // If the value is outside -128..127, casting to int8_t will wrap.
                // So we MUST clamp in float domain or int32 domain.

                // Let's clamp in float domain first.
                __m512 v_min_i8 = _mm512_set1_ps(-128.0f);
                __m512 v_max_i8 = _mm512_set1_ps(127.0f);
                q0 = _mm512_max_ps(v_min_i8, _mm512_min_ps(v_max_i8, q0));
                q1 = _mm512_max_ps(v_min_i8, _mm512_min_ps(v_max_i8, q1));

                i0 = _mm512_cvtps_epi32(q0);
                i1 = _mm512_cvtps_epi32(q1);

                // Now we can safely pack.
                // Without BW, we can use _mm512_mask_compressstoreu_epi8? No, that's VBMI2/BW.
                // We can extract 128-bit lanes and use SSE/AVX2 packing?
                // _mm512_extracti32x4_epi32...

                // For now, let's just use scalar fallback if no BW, as most AVX512 CPUs have BW (Skylake-X doesn't, but newer do).
                // Or just implement a simple loop to store.
                int32_t tmp[32];
                _mm512_storeu_si512(tmp, i0);
                _mm512_storeu_si512(tmp + 16, i1);
                for (int k = 0; k < 32; ++k)
                    output[k] = (int8_t)tmp[k];
#endif

                *out_scale = scale;
                *out_min = scale * 128.0f + min_val;
                return;
            }
            // Fallback for other counts
            requantize_to_int8_affine_scalar(input, count, output, out_scale, out_min);
        }
#endif

        inline void requantize_to_int8_affine(const float *input, int count, int8_t *output, float *out_scale, float *out_min)
        {
#if defined(__AVX512F__)
            requantize_to_int8_affine_avx512(input, count, output, out_scale, out_min);
#else
            requantize_to_int8_affine_scalar(input, count, output, out_scale, out_min);
#endif
        }

        /**
         * @brief Scalar implementation for Q5_1 → int8 unpacking
         */
        inline void unpack_q5_1_to_int8_scalar(const Q5_1Block &block, int8_t *output)
        {
            // Layout must match Q5_1Tensor::decodeBlockScalar():
            // output[0..15]  = low nibbles (qs[j] & 0x0F) | high bit from bits 0-15 of qh
            // output[16..31] = high nibbles (qs[j] >> 4) | high bit from bits 12-27 of qh
            // Q5_1 uses unsigned range [0,31] so we store as int8_t without sign adjustment
            uint32_t qh;
            std::memcpy(&qh, block.qh, sizeof(qh));

            for (int j = 0; j < 16; ++j)
            {
                const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10; // Bit j → position 4
                const uint8_t xh_1 = ((qh >> (j + 12)) & 0x10);     // Bit j+12 → position 4

                // Low nibble element at output[j]
                output[j] = static_cast<int8_t>((block.qs[j] & 0x0F) | xh_0);
                // High nibble element at output[j+16]
                output[j + 16] = static_cast<int8_t>((block.qs[j] >> 4) | xh_1);
            }
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief AVX512 implementation for Q5_1 → int8 unpacking
         * @note Currently falls back to scalar unpacking
         */
        inline void unpack_q5_1_to_int8_avx512(const Q5_1Block &block, int8_t *output)
        {
            unpack_q5_1_to_int8_scalar(block, output);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief AVX2 implementation for Q5_1 → int8 unpacking
         * @note Currently falls back to scalar unpacking
         */
        inline void unpack_q5_1_to_int8_avx2(const Q5_1Block &block, int8_t *output)
        {
            unpack_q5_1_to_int8_scalar(block, output);
        }
#endif

        /**
         * @brief Auto-dispatching Q5_1 → int8 unpacker
         */
        inline void unpack_q5_1_to_int8(const Q5_1Block &block, int8_t *output)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            unpack_q5_1_to_int8_avx512(block, output);
#elif defined(__AVX2__)
            unpack_q5_1_to_int8_avx2(block, output);
#else
            unpack_q5_1_to_int8_scalar(block, output);
#endif
        }

        // ========================================================================
        // Q5_K Unpacking Helpers
        // ========================================================================

        inline void get_q5_k_scale_min(const Q5_KBlock &block, int sub_block_idx, float *scale, float *min_val)
        {
            uint8_t sc, m;
            get_scale_min_k4(sub_block_idx, block.scales, &sc, &m);
            *scale = fp16_to_fp32(block.d) * sc;
            *min_val = fp16_to_fp32(block.dmin) * m;
        }

        inline void unpack_q5_k_to_int8_scalar(const Q5_KBlock &block, int sub_block_idx, int8_t *output)
        {
            const uint8_t *qs = block.qs + (sub_block_idx / 2) * 32;
            const uint8_t *qh = block.qh;
            bool is_high_nibble = (sub_block_idx % 2) != 0;
            uint8_t bit_mask = 1 << sub_block_idx;

            for (int i = 0; i < 32; ++i)
            {
                uint8_t q_val;
                if (is_high_nibble)
                {
                    q_val = (qs[i] >> 4) & 0xF;
                }
                else
                {
                    q_val = qs[i] & 0xF;
                }

                if (qh[i] & bit_mask)
                {
                    q_val += 16;
                }

                output[i] = static_cast<int8_t>(q_val);
            }
        }

#if defined(__AVX2__)
        inline void unpack_q5_k_to_int8_avx2(const Q5_KBlock &block, int sub_block_idx, int8_t *output)
        {
            const uint8_t *qs = block.qs + (sub_block_idx / 2) * 32;
            const uint8_t *qh = block.qh;
            bool is_high_nibble = (sub_block_idx % 2) != 0;

            // Broadcast bit mask for high bit extraction
            __m256i v_bit_mask = _mm256_set1_epi8(1 << sub_block_idx);
            __m256i v_mask_0F = _mm256_set1_epi8(0x0F);
            __m256i v_16 = _mm256_set1_epi8(16);
            __m256i v_zero = _mm256_setzero_si256();

            // Process 32 elements (one AVX2 register)
            __m256i v_qs = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qs));
            __m256i v_qh = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

            __m256i v_q_val;
            if (is_high_nibble)
            {
                // Shift right by 4 bits (operating on 16-bit words, then masking)
                v_q_val = _mm256_and_si256(_mm256_srli_epi16(v_qs, 4), v_mask_0F);
            }
            else
            {
                v_q_val = _mm256_and_si256(v_qs, v_mask_0F);
            }

            // Check high bit
            __m256i v_high_bit = _mm256_and_si256(v_qh, v_bit_mask);
            // Compare with zero. If != 0 (bit set), we want to add 16.
            // cmpeq returns 0xFF if equal (zero), 0x00 if not equal (set).
            __m256i v_is_zero = _mm256_cmpeq_epi8(v_high_bit, v_zero);

            // We want to add 16 where v_is_zero is FALSE (0x00).
            // andnot(a, b) = (~a) & b.
            // (~0x00) & 16 = 0xFF & 16 = 16.
            // (~0xFF) & 16 = 0x00 & 16 = 0.
            __m256i v_add_16 = _mm256_andnot_si256(v_is_zero, v_16);

            v_q_val = _mm256_add_epi8(v_q_val, v_add_16);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(output), v_q_val);
        }
#endif

        inline void unpack_q5_k_to_int8(const Q5_KBlock &block, int sub_block_idx, int8_t *output)
        {
#if defined(__AVX2__)
            unpack_q5_k_to_int8_avx2(block, sub_block_idx, output);
#else
            unpack_q5_k_to_int8_scalar(block, sub_block_idx, output);
#endif
        }

        /**
         * @brief Unpack IQ4_NL block to int8 values (using lookup table)
         *
         * @param block IQ4_NL block
         * @param output Output buffer (32 int8_t values)
         */
        inline void unpack_iq4_nl_to_int8(const IQ4_NLBlock &block, int8_t *output)
        {
#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                // Load 16 bytes of indices
                __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

                // Mask for low nibbles
                __m128i low_mask = _mm_set1_epi8(0x0F);
                __m128i low_indices = _mm_and_si128(qs, low_mask);

                // High nibbles
                __m128i high_indices = _mm_and_si128(_mm_srli_epi16(qs, 4), low_mask);

                // Load lookup table
                __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl_i8));

                // Shuffle
                __m128i low_vals = _mm_shuffle_epi8(lut, low_indices);
                __m128i high_vals = _mm_shuffle_epi8(lut, high_indices);

                // Store
                _mm_storeu_si128(reinterpret_cast<__m128i *>(output), low_vals);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), high_vals);
                return;
            }
#endif

            // Scalar fallback
            for (int i = 0; i < 16; ++i)
            {
                uint8_t val = block.qs[i];
                output[i] = kvalues_iq4nl_i8[val & 0x0F];
                output[i + 16] = kvalues_iq4nl_i8[val >> 4];
            }
        }

        /**
         * @brief Unpack IQ4_XS sub-block to int8 (32 elements)
         *
         * @param block IQ4_XS super-block
         * @param subblock_idx Sub-block index (0-7)
         * @param output Output buffer (32 int8_t values)
         */
        inline void unpack_iq4_xs_to_int8(const IQ4_XSBlock &block, size_t subblock_idx, int8_t *output)
        {
            const uint8_t *qs_ptr = block.qs + subblock_idx * 16;

#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                // Load 16 bytes of indices
                __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qs_ptr));

                // Mask for low nibbles
                __m128i low_mask = _mm_set1_epi8(0x0F);
                __m128i low_indices = _mm_and_si128(qs, low_mask);

                // High nibbles
                __m128i high_indices = _mm_and_si128(_mm_srli_epi16(qs, 4), low_mask);

                // Load lookup table
                __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl_i8));

                // Shuffle
                __m128i low_vals = _mm_shuffle_epi8(lut, low_indices);
                __m128i high_vals = _mm_shuffle_epi8(lut, high_indices);

                // Store
                _mm_storeu_si128(reinterpret_cast<__m128i *>(output), low_vals);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), high_vals);
                return;
            }
#endif

            // Scalar fallback
            for (int i = 0; i < 16; ++i)
            {
                uint8_t val = qs_ptr[i];
                output[i] = kvalues_iq4nl_i8[val & 0x0F];
                output[i + 16] = kvalues_iq4nl_i8[val >> 4];
            }
        }

        /**
         * @brief Get scale for IQ4_XS sub-block
         *
         * @param block IQ4_XS super-block
         * @param subblock_idx Sub-block index (0-7)
         * @return Scale factor
         */
        inline float get_iq4_xs_scale(const IQ4_XSBlock &block, size_t subblock_idx)
        {
            const float d = fp16_to_fp32(block.d);
            const size_t ib = subblock_idx;
            const int ls = ((block.scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                           (((block.scales_h >> 2 * ib) & 3) << 4);
            return d * (ls - 32);
        }

    } // namespace simd
} // namespace llaminar2
