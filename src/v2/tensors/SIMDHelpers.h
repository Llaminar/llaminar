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
        // SIMD Helpers
        // ========================================================================

#if defined(__AVX512F__) || defined(__AVX2__)
        inline uint8_t hmax_epu8_128(__m128i v)
        {
            __m128i max1 = _mm_max_epu8(v, _mm_srli_si128(v, 8));
            __m128i max2 = _mm_max_epu8(max1, _mm_srli_si128(max1, 4));
            __m128i max3 = _mm_max_epu8(max2, _mm_srli_si128(max2, 2));
            __m128i max4 = _mm_max_epu8(max3, _mm_srli_si128(max3, 1));
            return (uint8_t)_mm_cvtsi128_si32(max4);
        }
#endif

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
                static const bool has_avx2 = cpu_supports_avx2();
                if (has_avx2)
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
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                return activation_row_max_abs_avx512(row, length);
            }
#endif
#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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
                static const bool has_avx2 = cpu_supports_avx2();
                if (has_avx2)
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
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                quantize_activation_row_avx512(src, length, inv_scale, dst);
                return;
            }
#endif
#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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

        // ====================================================================
        // Fused Residual Add Operations (Typed Residuals Optimization)
        // ====================================================================
        // These functions perform fused dequantize + add operations for residual
        // connections in transformer blocks. By fusing dequant and add, we:
        // 1. Avoid intermediate FP32 buffer allocation
        // 2. Keep data in registers between dequant and add
        // 3. Reduce memory traffic (only read typed residual once)

        /**
         * @brief Fused FP32 residual add: output = residual + input
         *
         * @param residual FP32 residual buffer
         * @param input FP32 input to add
         * @param output FP32 output buffer
         * @param count Number of elements
         */
        inline void fused_fp32_residual_add(
            const float *residual, const float *input, float *output, size_t count)
        {
            size_t i = 0;
#ifdef __AVX512F__
            for (; i + 16 <= count; i += 16)
            {
                __m512 r = _mm512_loadu_ps(residual + i);
                __m512 x = _mm512_loadu_ps(input + i);
                __m512 sum = _mm512_add_ps(r, x);
                _mm512_storeu_ps(output + i, sum);
            }
#elif defined(__AVX2__)
            for (; i + 8 <= count; i += 8)
            {
                __m256 r = _mm256_loadu_ps(residual + i);
                __m256 x = _mm256_loadu_ps(input + i);
                __m256 sum = _mm256_add_ps(r, x);
                _mm256_storeu_ps(output + i, sum);
            }
#endif
            for (; i < count; ++i)
            {
                output[i] = residual[i] + input[i];
            }
        }

        /**
         * @brief Fused BF16 residual add: output = dequant(bf16_residual) + fp32_input
         *
         * Performs BF16→FP32 dequantization and addition in a single pass,
         * keeping intermediate FP32 values in registers.
         *
         * @param residual BF16 residual buffer (uint16_t)
         * @param input FP32 input to add
         * @param output FP32 output buffer
         * @param count Number of elements
         */
        inline void fused_bf16_residual_add(
            const uint16_t *residual, const float *input, float *output, size_t count)
        {
            size_t i = 0;
#ifdef __AVX512F__
            for (; i + 16 <= count; i += 16)
            {
                // Dequantize BF16 → FP32
                __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(residual + i));
                __m512i unpacked = _mm512_cvtepu16_epi32(bf16_vals);
                __m512i shifted = _mm512_slli_epi32(unpacked, 16);
                __m512 r = _mm512_castsi512_ps(shifted);

                // Load input and add
                __m512 x = _mm512_loadu_ps(input + i);
                __m512 sum = _mm512_add_ps(r, x);
                _mm512_storeu_ps(output + i, sum);
            }
#elif defined(__AVX2__)
            for (; i + 8 <= count; i += 8)
            {
                // Dequantize BF16 → FP32
                __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i *>(residual + i));
                __m256i unpacked = _mm256_cvtepu16_epi32(bf16_vals);
                __m256i shifted = _mm256_slli_epi32(unpacked, 16);
                __m256 r = _mm256_castsi256_ps(shifted);

                // Load input and add
                __m256 x = _mm256_loadu_ps(input + i);
                __m256 sum = _mm256_add_ps(r, x);
                _mm256_storeu_ps(output + i, sum);
            }
#endif
            // Scalar tail
            for (; i < count; ++i)
            {
                output[i] = bf16_to_fp32(residual[i]) + input[i];
            }
        }

        /**
         * @brief Fused FP16 residual add: output = dequant(fp16_residual) + fp32_input
         *
         * Performs FP16→FP32 dequantization and addition in a single pass.
         * Uses F16C hardware instructions when available.
         *
         * @param residual FP16 residual buffer (uint16_t)
         * @param input FP32 input to add
         * @param output FP32 output buffer
         * @param count Number of elements
         */
        inline void fused_fp16_residual_add(
            const uint16_t *residual, const float *input, float *output, size_t count)
        {
            size_t i = 0;
#if defined(__AVX512F__)
            for (; i + 16 <= count; i += 16)
            {
                // Dequantize FP16 → FP32
                __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(residual + i));
                __m512 r = _mm512_cvtph_ps(h);

                // Load input and add
                __m512 x = _mm512_loadu_ps(input + i);
                __m512 sum = _mm512_add_ps(r, x);
                _mm512_storeu_ps(output + i, sum);
            }
#elif defined(__AVX2__) && defined(__F16C__)
            for (; i + 8 <= count; i += 8)
            {
                // Dequantize FP16 → FP32
                __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(residual + i));
                __m256 r = _mm256_cvtph_ps(h);

                // Load input and add
                __m256 x = _mm256_loadu_ps(input + i);
                __m256 sum = _mm256_add_ps(r, x);
                _mm256_storeu_ps(output + i, sum);
            }
#endif
            // Scalar tail
            for (; i < count; ++i)
            {
                output[i] = fp16_to_fp32(residual[i]) + input[i];
            }
        }

        /**
         * @brief Fused Q8_1 residual add: output = dequant(q8_1_residual) + fp32_input
         *
         * Performs Q8_1→FP32 dequantization and addition in a single pass.
         * Processes one block (32 elements) at a time.
         *
         * @param residual Q8_1 block buffer
         * @param input FP32 input to add (must be 32-element aligned to block)
         * @param output FP32 output buffer
         * @param count Number of elements (must be multiple of 32)
         */
        inline void fused_q8_1_residual_add(
            const Q8_1Block *residual, const float *input, float *output, size_t count)
        {
            const size_t n_blocks = count / 32;

            for (size_t b = 0; b < n_blocks; ++b)
            {
                const Q8_1Block &block = residual[b];
                const float *block_input = input + b * 32;
                float *block_output = output + b * 32;

                // Get scale
                const float scale = fp16_to_fp32(block.d);

#ifdef __AVX512F__
                __m512 scale_vec = _mm512_set1_ps(scale);

                // Load 32 int8 values
                __m128i q_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
                __m128i q_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs + 16));

                // Sign-extend to int32
                __m512i i0 = _mm512_cvtepi8_epi32(q_lo);
                __m512i i1 = _mm512_cvtepi8_epi32(q_hi);

                // Convert to float, scale, and add input
                __m512 r0 = _mm512_mul_ps(_mm512_cvtepi32_ps(i0), scale_vec);
                __m512 r1 = _mm512_mul_ps(_mm512_cvtepi32_ps(i1), scale_vec);

                __m512 x0 = _mm512_loadu_ps(block_input);
                __m512 x1 = _mm512_loadu_ps(block_input + 16);

                __m512 sum0 = _mm512_add_ps(r0, x0);
                __m512 sum1 = _mm512_add_ps(r1, x1);

                _mm512_storeu_ps(block_output, sum0);
                _mm512_storeu_ps(block_output + 16, sum1);
#elif defined(__AVX2__)
                __m256 scale_vec = _mm256_set1_ps(scale);

                for (int i = 0; i < 4; ++i)
                {
                    // Load 8 int8 values
                    __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.qs + i * 8));
                    // Sign-extend to int32
                    __m256i i32 = _mm256_cvtepi8_epi32(q8);
                    // Convert to float and scale
                    __m256 r = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), scale_vec);
                    // Load input and add
                    __m256 x = _mm256_loadu_ps(block_input + i * 8);
                    __m256 sum = _mm256_add_ps(r, x);
                    // Store
                    _mm256_storeu_ps(block_output + i * 8, sum);
                }
#else
                // Scalar fallback
                for (int i = 0; i < 32; ++i)
                {
                    float r = scale * static_cast<float>(block.qs[i]);
                    block_output[i] = r + block_input[i];
                }
#endif
            }
        }

        /**
         * @brief Native Q8_1 + Q8_1 → Q8_1 addition
         *
         * Adds two Q8_1 tensors and produces a Q8_1 output.
         * This is the key operation for typed residual connections.
         *
         * Algorithm per block:
         * 1. Dequantize both blocks to FP32 (in registers)
         * 2. Add FP32 values
         * 3. Find new max_abs for quantization
         * 4. Requantize to Q8_1
         *
         * @param a First Q8_1 block buffer
         * @param b Second Q8_1 block buffer
         * @param output Output Q8_1 block buffer
         * @param count Number of elements (must be multiple of 32)
         */
        inline void q8_1_add_q8_1(
            const Q8_1Block *a, const Q8_1Block *b, Q8_1Block *output, size_t count)
        {
            const size_t n_blocks = count / 32;

            for (size_t blk = 0; blk < n_blocks; ++blk)
            {
                const Q8_1Block &block_a = a[blk];
                const Q8_1Block &block_b = b[blk];
                Q8_1Block &block_out = output[blk];

                const float scale_a = fp16_to_fp32(block_a.d);
                const float scale_b = fp16_to_fp32(block_b.d);

#ifdef __AVX512F__
                // Load and dequant A
                __m128i qa_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_a.qs));
                __m128i qa_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_a.qs + 16));
                __m512 fa0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qa_lo)), _mm512_set1_ps(scale_a));
                __m512 fa1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qa_hi)), _mm512_set1_ps(scale_a));

                // Load and dequant B
                __m128i qb_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_b.qs));
                __m128i qb_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_b.qs + 16));
                __m512 fb0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qb_lo)), _mm512_set1_ps(scale_b));
                __m512 fb1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qb_hi)), _mm512_set1_ps(scale_b));

                // Add
                __m512 sum0 = _mm512_add_ps(fa0, fb0);
                __m512 sum1 = _mm512_add_ps(fa1, fb1);

                // Find max_abs for requantization
                __m512 abs0 = _mm512_abs_ps(sum0);
                __m512 abs1 = _mm512_abs_ps(sum1);
                __m512 vmax = _mm512_max_ps(abs0, abs1);
                float max_abs = _mm512_reduce_max_ps(vmax);

                // Handle near-zero case
                if (max_abs < 1e-6f)
                {
                    block_out.d = 0;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 32);
                    continue;
                }

                // Compute scale and inverse
                float out_scale = max_abs / 127.0f;
                float inv_scale = 127.0f / max_abs;
                block_out.d = fp32_to_fp16(out_scale);

                // Quantize: round(sum * inv_scale), clamp to [-127, 127]
                __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
                __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(sum0, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(sum1, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));

                // Clamp to [-127, 127] and pack to int8
                __m512i clamped0 = _mm512_max_epi32(_mm512_min_epi32(q0, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
                __m512i clamped1 = _mm512_max_epi32(_mm512_min_epi32(q1, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));

                // Pack 32-bit to 8-bit: use cvtepi32_epi8 via packs
                __m256i packed0 = _mm512_cvtepi32_epi16(clamped0);
                __m256i packed1 = _mm512_cvtepi32_epi16(clamped1);
                __m128i bytes0 = _mm256_cvtepi16_epi8(packed0);
                __m128i bytes1 = _mm256_cvtepi16_epi8(packed1);

                _mm_storeu_si128(reinterpret_cast<__m128i *>(block_out.qs), bytes0);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(block_out.qs + 16), bytes1);

                // Compute sum_qs (raw integer sum, NOT FP16!)
                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    sum_qs += block_out.qs[i];
                }
                block_out.sum_qs = static_cast<int16_t>(sum_qs);

#elif defined(__AVX2__)
                // Dequant A and B, add, and store to temp buffer
                alignas(32) float temp[32];
                __m256 scale_a_vec = _mm256_set1_ps(scale_a);
                __m256 scale_b_vec = _mm256_set1_ps(scale_b);

                for (int i = 0; i < 4; ++i)
                {
                    __m128i qa8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block_a.qs + i * 8));
                    __m128i qb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block_b.qs + i * 8));
                    __m256i ia32 = _mm256_cvtepi8_epi32(qa8);
                    __m256i ib32 = _mm256_cvtepi8_epi32(qb8);
                    __m256 fa = _mm256_mul_ps(_mm256_cvtepi32_ps(ia32), scale_a_vec);
                    __m256 fb = _mm256_mul_ps(_mm256_cvtepi32_ps(ib32), scale_b_vec);
                    __m256 sum = _mm256_add_ps(fa, fb);
                    _mm256_store_ps(temp + i * 8, sum);
                }

                // Find max_abs
                __m256 vmax0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp));
                __m256 vmax1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 8));
                __m256 vmax2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 16));
                __m256 vmax3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 24));
                __m256 vmax_01 = _mm256_max_ps(vmax0, vmax1);
                __m256 vmax_23 = _mm256_max_ps(vmax2, vmax3);
                __m256 vmax_all = _mm256_max_ps(vmax_01, vmax_23);
                __m128 lo = _mm256_castps256_ps128(vmax_all);
                __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
                __m128 vmax128 = _mm_max_ps(lo, hi);
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
                float max_abs = _mm_cvtss_f32(vmax128);

                if (max_abs < 1e-6f)
                {
                    block_out.d = 0;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 32);
                    continue;
                }

                float out_scale = max_abs / 127.0f;
                float inv_scale = 127.0f / max_abs;
                block_out.d = fp32_to_fp16(out_scale);

                // Quantize
                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    block_out.qs[i] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                block_out.sum_qs = static_cast<int16_t>(sum_qs);

#else
                // Scalar fallback
                alignas(32) float temp[32];

                // Dequant and add
                for (int i = 0; i < 32; ++i)
                {
                    float va = scale_a * static_cast<float>(block_a.qs[i]);
                    float vb = scale_b * static_cast<float>(block_b.qs[i]);
                    temp[i] = va + vb;
                }

                // Find max_abs
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(temp[i]));
                }

                if (max_abs < 1e-6f)
                {
                    block_out.d = 0;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 32);
                    continue;
                }

                float out_scale = max_abs / 127.0f;
                float inv_scale = 127.0f / max_abs;
                block_out.d = fp32_to_fp16(out_scale);

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    block_out.qs[i] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                block_out.sum_qs = static_cast<int16_t>(sum_qs);
#endif
            }
        }

        /**
         * @brief Sum multiple Q8_1 block arrays into one (N-way reduction)
         *
         * Used for MPI allreduce operations where we need to sum Q8_1 contributions
         * from N ranks. This is more efficient than N-1 pairwise additions.
         *
         * Algorithm per block:
         * 1. Dequantize all N blocks to FP32 (accumulated in registers)
         * 2. Sum all FP32 values
         * 3. Find new max_abs for quantization
         * 4. Requantize to Q8_1
         *
         * @param inputs Array of N pointers to Q8_1 block arrays
         * @param n_inputs Number of input arrays (typically world_size)
         * @param output Output Q8_1 block buffer
         * @param n_blocks Number of blocks per array
         */
        inline void q8_1_sum_n(
            const Q8_1Block *const *inputs, size_t n_inputs, Q8_1Block *output, size_t n_blocks)
        {
            if (n_inputs == 0)
                return;
            if (n_inputs == 1)
            {
                // Copy directly (q8_1_copy may not be declared yet)
                std::memcpy(output, inputs[0], n_blocks * sizeof(Q8_1Block));
                return;
            }
            if (n_inputs == 2)
            {
                q8_1_add_q8_1(inputs[0], inputs[1], output, n_blocks * 32);
                return;
            }

            // N-way reduction (N >= 3)
            for (size_t blk = 0; blk < n_blocks; ++blk)
            {
                alignas(64) float sum_vals[32] = {0.0f};

#ifdef __AVX512F__
                // Initialize accumulators to zero
                __m512 acc0 = _mm512_setzero_ps();
                __m512 acc1 = _mm512_setzero_ps();

                // Accumulate all inputs
                for (size_t i = 0; i < n_inputs; ++i)
                {
                    const Q8_1Block &block = inputs[i][blk];
                    const float scale = fp16_to_fp32(block.d);

                    __m128i q_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
                    __m128i q_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs + 16));

                    __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q_lo)), _mm512_set1_ps(scale));
                    __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q_hi)), _mm512_set1_ps(scale));

                    acc0 = _mm512_add_ps(acc0, f0);
                    acc1 = _mm512_add_ps(acc1, f1);
                }

                // Find max_abs for requantization
                __m512 abs0 = _mm512_abs_ps(acc0);
                __m512 abs1 = _mm512_abs_ps(acc1);
                __m512 vmax = _mm512_max_ps(abs0, abs1);
                float max_abs = _mm512_reduce_max_ps(vmax);

                Q8_1Block &block_out = output[blk];

                if (max_abs < 1e-6f)
                {
                    block_out.d = 0;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 32);
                    continue;
                }

                // Compute scale and inverse
                float out_scale = max_abs / 127.0f;
                float inv_scale = 127.0f / max_abs;
                block_out.d = fp32_to_fp16(out_scale);

                // Quantize: round(sum * inv_scale), clamp to [-127, 127]
                __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
                __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(acc0, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(acc1, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));

                // Clamp to [-127, 127] and pack to int8
                __m512i clamped0 = _mm512_max_epi32(_mm512_min_epi32(q0, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
                __m512i clamped1 = _mm512_max_epi32(_mm512_min_epi32(q1, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));

                // Pack 32-bit to 8-bit
                __m256i packed0 = _mm512_cvtepi32_epi16(clamped0);
                __m256i packed1 = _mm512_cvtepi32_epi16(clamped1);
                __m128i bytes0 = _mm256_cvtepi16_epi8(packed0);
                __m128i bytes1 = _mm256_cvtepi16_epi8(packed1);

                _mm_storeu_si128(reinterpret_cast<__m128i *>(block_out.qs), bytes0);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(block_out.qs + 16), bytes1);

                // Compute sum_qs (raw integer sum)
                int32_t sum_qs = 0;
                for (int j = 0; j < 32; ++j)
                {
                    sum_qs += block_out.qs[j];
                }
                block_out.sum_qs = static_cast<int16_t>(sum_qs);

#elif defined(__AVX2__)
                // AVX2: Accumulate to temp buffer then quantize
                __m256 acc_0 = _mm256_setzero_ps();
                __m256 acc_1 = _mm256_setzero_ps();
                __m256 acc_2 = _mm256_setzero_ps();
                __m256 acc_3 = _mm256_setzero_ps();

                for (size_t i = 0; i < n_inputs; ++i)
                {
                    const Q8_1Block &block = inputs[i][blk];
                    const float scale = fp16_to_fp32(block.d);
                    __m256 scale_vec = _mm256_set1_ps(scale);

                    for (int chunk = 0; chunk < 4; ++chunk)
                    {
                        __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.qs + chunk * 8));
                        __m256i q32 = _mm256_cvtepi8_epi32(q8);
                        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), scale_vec);
                        switch (chunk)
                        {
                        case 0:
                            acc_0 = _mm256_add_ps(acc_0, f);
                            break;
                        case 1:
                            acc_1 = _mm256_add_ps(acc_1, f);
                            break;
                        case 2:
                            acc_2 = _mm256_add_ps(acc_2, f);
                            break;
                        case 3:
                            acc_3 = _mm256_add_ps(acc_3, f);
                            break;
                        }
                    }
                }

                // Store to temp buffer for max_abs and quantization
                _mm256_store_ps(sum_vals, acc_0);
                _mm256_store_ps(sum_vals + 8, acc_1);
                _mm256_store_ps(sum_vals + 16, acc_2);
                _mm256_store_ps(sum_vals + 24, acc_3);

                // Find max_abs
                __m256 abs0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), acc_0);
                __m256 abs1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), acc_1);
                __m256 abs2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), acc_2);
                __m256 abs3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), acc_3);
                __m256 vmax_01 = _mm256_max_ps(abs0, abs1);
                __m256 vmax_23 = _mm256_max_ps(abs2, abs3);
                __m256 vmax_all = _mm256_max_ps(vmax_01, vmax_23);
                __m128 lo = _mm256_castps256_ps128(vmax_all);
                __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
                __m128 vmax128 = _mm_max_ps(lo, hi);
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
                float max_abs = _mm_cvtss_f32(vmax128);

                Q8_1Block &block_out = output[blk];

                if (max_abs < 1e-6f)
                {
                    block_out.d = 0;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 32);
                    continue;
                }

                float out_scale = max_abs / 127.0f;
                float inv_scale = 127.0f / max_abs;
                block_out.d = fp32_to_fp16(out_scale);

                int32_t sum_qs_val = 0;
                for (int j = 0; j < 32; ++j)
                {
                    int32_t q = static_cast<int32_t>(std::round(sum_vals[j] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    block_out.qs[j] = static_cast<int8_t>(q);
                    sum_qs_val += q;
                }
                block_out.sum_qs = static_cast<int16_t>(sum_qs_val);

#else
                // Scalar fallback
                for (size_t i = 0; i < n_inputs; ++i)
                {
                    const Q8_1Block &block = inputs[i][blk];
                    const float scale = fp16_to_fp32(block.d);
                    for (int j = 0; j < 32; ++j)
                    {
                        sum_vals[j] += scale * static_cast<float>(block.qs[j]);
                    }
                }

                float max_abs = 0.0f;
                for (int j = 0; j < 32; ++j)
                {
                    max_abs = std::max(max_abs, std::abs(sum_vals[j]));
                }

                Q8_1Block &block_out = output[blk];

                if (max_abs < 1e-6f)
                {
                    block_out.d = 0;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 32);
                    continue;
                }

                float out_scale = max_abs / 127.0f;
                float inv_scale = 127.0f / max_abs;
                block_out.d = fp32_to_fp16(out_scale);

                int32_t sum_qs_val = 0;
                for (int j = 0; j < 32; ++j)
                {
                    int32_t q = static_cast<int32_t>(std::round(sum_vals[j] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    block_out.qs[j] = static_cast<int8_t>(q);
                    sum_qs_val += q;
                }
                block_out.sum_qs = static_cast<int16_t>(sum_qs_val);
#endif
            }
        }

        /**
         * @brief Sum multiple Q16_1 block arrays into one (N-way reduction)
         *
         * Used for MPI allreduce operations where we need to sum Q16_1 contributions
         * from N ranks. This is more efficient than N-1 pairwise additions.
         *
         * Algorithm per block:
         * 1. Dequantize all N blocks to FP32 (accumulated in registers)
         * 2. Sum all FP32 values
         * 3. Find new max_abs for quantization
         * 4. Requantize to Q16_1
         *
         * @param inputs Array of N pointers to Q16_1 block arrays
         * @param n_inputs Number of input arrays (typically world_size)
         * @param output Output Q16_1 block buffer
         * @param n_blocks Number of blocks per array
         */
        inline void q16_1_sum_n(
            const Q16_1Block *const *inputs, size_t n_inputs, Q16_1Block *output, size_t n_blocks)
        {
            if (n_inputs == 0)
                return;
            if (n_inputs == 1)
            {
                std::memcpy(output, inputs[0], n_blocks * sizeof(Q16_1Block));
                return;
            }

            // N-way reduction (N >= 2)
            for (size_t blk = 0; blk < n_blocks; ++blk)
            {
                alignas(64) float sum_vals[32] = {0.0f};

#ifdef __AVX512F__
                // Initialize accumulators to zero
                __m512 acc0 = _mm512_setzero_ps();
                __m512 acc1 = _mm512_setzero_ps();

                // Accumulate all inputs
                for (size_t i = 0; i < n_inputs; ++i)
                {
                    const Q16_1Block &block = inputs[i][blk];
                    const float scale = block.d; // Q16_1 uses FP32 scale directly
                    const __m512 vscale = _mm512_set1_ps(scale);

                    // Load 32 int16 values (two 256-bit loads)
                    __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));
                    __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs + 16));

                    // Convert int16 → int32 → float
                    __m512i i32_lo = _mm512_cvtepi16_epi32(q_lo);
                    __m512i i32_hi = _mm512_cvtepi16_epi32(q_hi);
                    __m512 f_lo = _mm512_cvtepi32_ps(i32_lo);
                    __m512 f_hi = _mm512_cvtepi32_ps(i32_hi);

                    // Scale and accumulate
                    acc0 = _mm512_fmadd_ps(f_lo, vscale, acc0);
                    acc1 = _mm512_fmadd_ps(f_hi, vscale, acc1);
                }

                // Find max_abs for requantization
                __m512 abs0 = _mm512_abs_ps(acc0);
                __m512 abs1 = _mm512_abs_ps(acc1);
                __m512 vmax = _mm512_max_ps(abs0, abs1);
                float max_abs = _mm512_reduce_max_ps(vmax);

                Q16_1Block &block_out = output[blk];

                if (max_abs < 1e-10f)
                {
                    block_out.d = 0.0f;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 64); // 32 int16 = 64 bytes
                    continue;
                }

                // Compute scale and inverse (Q16_1 uses ±32767 range)
                float out_scale = max_abs / 32767.0f;
                float inv_scale = 32767.0f / max_abs;
                block_out.d = out_scale;

                // Quantize: round(sum * inv_scale), clamp to [-32767, 32767]
                __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
                __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(
                    _mm512_mul_ps(acc0, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(
                    _mm512_mul_ps(acc1, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));

                // Clamp to [-32767, 32767]
                __m512i clamped0 = _mm512_max_epi32(_mm512_min_epi32(q0, _mm512_set1_epi32(32767)),
                                                    _mm512_set1_epi32(-32767));
                __m512i clamped1 = _mm512_max_epi32(_mm512_min_epi32(q1, _mm512_set1_epi32(32767)),
                                                    _mm512_set1_epi32(-32767));

                // Pack int32 → int16 with saturation
                __m256i packed0 = _mm512_cvtepi32_epi16(clamped0);
                __m256i packed1 = _mm512_cvtepi32_epi16(clamped1);

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block_out.qs), packed0);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block_out.qs + 16), packed1);

                // Compute sum_qs (sum of int32 before packing, fits in int32)
                int32_t sum_qs = _mm512_reduce_add_epi32(clamped0) + _mm512_reduce_add_epi32(clamped1);
                block_out.sum_qs = sum_qs;

#elif defined(__AVX2__)
                // AVX2: Process 8 elements at a time
                __m256 acc_0 = _mm256_setzero_ps();
                __m256 acc_1 = _mm256_setzero_ps();
                __m256 acc_2 = _mm256_setzero_ps();
                __m256 acc_3 = _mm256_setzero_ps();

                for (size_t i = 0; i < n_inputs; ++i)
                {
                    const Q16_1Block &block = inputs[i][blk];
                    const float scale = block.d;
                    __m256 scale_vec = _mm256_set1_ps(scale);

                    for (int chunk = 0; chunk < 4; ++chunk)
                    {
                        // Load 8 int16 values
                        __m128i q16 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(block.qs + chunk * 8));
                        // Convert int16 → int32 → float
                        __m256i q32 = _mm256_cvtepi16_epi32(q16);
                        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), scale_vec);

                        switch (chunk)
                        {
                        case 0:
                            acc_0 = _mm256_add_ps(acc_0, f);
                            break;
                        case 1:
                            acc_1 = _mm256_add_ps(acc_1, f);
                            break;
                        case 2:
                            acc_2 = _mm256_add_ps(acc_2, f);
                            break;
                        case 3:
                            acc_3 = _mm256_add_ps(acc_3, f);
                            break;
                        }
                    }
                }

                // Store to temp buffer
                _mm256_store_ps(sum_vals, acc_0);
                _mm256_store_ps(sum_vals + 8, acc_1);
                _mm256_store_ps(sum_vals + 16, acc_2);
                _mm256_store_ps(sum_vals + 24, acc_3);

                // Find max_abs
                __m256 sign_mask = _mm256_set1_ps(-0.0f);
                __m256 abs0 = _mm256_andnot_ps(sign_mask, acc_0);
                __m256 abs1 = _mm256_andnot_ps(sign_mask, acc_1);
                __m256 abs2 = _mm256_andnot_ps(sign_mask, acc_2);
                __m256 abs3 = _mm256_andnot_ps(sign_mask, acc_3);
                __m256 vmax_01 = _mm256_max_ps(abs0, abs1);
                __m256 vmax_23 = _mm256_max_ps(abs2, abs3);
                __m256 vmax_all = _mm256_max_ps(vmax_01, vmax_23);
                __m128 lo = _mm256_castps256_ps128(vmax_all);
                __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
                __m128 vmax128 = _mm_max_ps(lo, hi);
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
                float max_abs = _mm_cvtss_f32(vmax128);

                Q16_1Block &block_out = output[blk];

                if (max_abs < 1e-10f)
                {
                    block_out.d = 0.0f;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                float inv_scale = 32767.0f / max_abs;
                block_out.d = out_scale;

                int32_t sum_qs_val = 0;
                for (int j = 0; j < 32; ++j)
                {
                    int32_t q = static_cast<int32_t>(std::round(sum_vals[j] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    block_out.qs[j] = static_cast<int16_t>(q);
                    sum_qs_val += q;
                }
                block_out.sum_qs = sum_qs_val;

#else
                // Scalar fallback
                for (size_t i = 0; i < n_inputs; ++i)
                {
                    const Q16_1Block &block = inputs[i][blk];
                    const float scale = block.d;
                    for (int j = 0; j < 32; ++j)
                    {
                        sum_vals[j] += scale * static_cast<float>(block.qs[j]);
                    }
                }

                float max_abs = 0.0f;
                for (int j = 0; j < 32; ++j)
                {
                    max_abs = std::max(max_abs, std::abs(sum_vals[j]));
                }

                Q16_1Block &block_out = output[blk];

                if (max_abs < 1e-10f)
                {
                    block_out.d = 0.0f;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                float inv_scale = 32767.0f / max_abs;
                block_out.d = out_scale;

                int32_t sum_qs_val = 0;
                for (int j = 0; j < 32; ++j)
                {
                    int32_t q = static_cast<int32_t>(std::round(sum_vals[j] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    block_out.qs[j] = static_cast<int16_t>(q);
                    sum_qs_val += q;
                }
                block_out.sum_qs = sum_qs_val;
#endif
            }
        }

        // ==================== Q16_1 Native Operations ====================

        /**
         * @brief Native Q16_1 + Q16_1 addition with SIMD acceleration
         *
         * Adds two Q16_1 tensors element-wise, producing a Q16_1 result.
         * This is the key operation for high-precision typed residual connections.
         *
         * Algorithm per block:
         * 1. Dequantize both blocks to FP32 (in registers) - FP32 scale, no conversion needed
         * 2. Add FP32 values
         * 3. Find new max_abs for quantization
         * 4. Requantize to Q16_1 (int16 range: [-32767, 32767])
         *
         * Key differences from Q8_1:
         * - Uses FP32 scale directly (no FP16 conversion)
         * - int16_t quantized values (256× finer than int8)
         * - INT32 sum_qs (wider range for int16 sums)
         *
         * @param a First Q16_1 block buffer
         * @param b Second Q16_1 block buffer
         * @param output Output Q16_1 block buffer
         * @param count Number of elements (must be multiple of 32)
         */
        inline void q16_1_add_q16_1(
            const Q16_1Block *a, const Q16_1Block *b, Q16_1Block *output, size_t count)
        {
            const size_t n_blocks = count / 32;

            for (size_t blk = 0; blk < n_blocks; ++blk)
            {
                const Q16_1Block &block_a = a[blk];
                const Q16_1Block &block_b = b[blk];
                Q16_1Block &block_out = output[blk];

                // Q16_1 uses FP32 scale directly - no conversion needed!
                const float scale_a = block_a.d;
                const float scale_b = block_b.d;

#ifdef __AVX512F__
                // Load and dequant A (16 int16 values → 16 floats, 2 iterations for 32 elements)
                __m256i qa_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block_a.qs));
                __m256i qa_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block_a.qs + 16));
                __m512 fa0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qa_lo)), _mm512_set1_ps(scale_a));
                __m512 fa1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qa_hi)), _mm512_set1_ps(scale_a));

                // Load and dequant B
                __m256i qb_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block_b.qs));
                __m256i qb_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block_b.qs + 16));
                __m512 fb0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qb_lo)), _mm512_set1_ps(scale_b));
                __m512 fb1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qb_hi)), _mm512_set1_ps(scale_b));

                // Add
                __m512 sum0 = _mm512_add_ps(fa0, fb0);
                __m512 sum1 = _mm512_add_ps(fa1, fb1);

                // Find max_abs for requantization
                __m512 abs0 = _mm512_abs_ps(sum0);
                __m512 abs1 = _mm512_abs_ps(sum1);
                __m512 vmax = _mm512_max_ps(abs0, abs1);
                float max_abs = _mm512_reduce_max_ps(vmax);

                // Handle near-zero case
                if (max_abs < 1e-10f)
                {
                    block_out.d = 0.0f;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 64); // 32 × int16 = 64 bytes
                    continue;
                }

                // Compute scale and inverse (Q16_1 uses FP32 scale, int16 range)
                float out_scale = max_abs / 32767.0f;
                block_out.d = out_scale; // FP32 scale, no conversion!
                float inv_scale = 32767.0f / max_abs;

                // Quantize: round(sum * inv_scale), clamp to [-32767, 32767]
                __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
                __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(sum0, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(sum1, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));

                // Clamp to [-32767, 32767] and pack to int16
                __m512i clamped0 = _mm512_max_epi32(_mm512_min_epi32(q0, _mm512_set1_epi32(32767)), _mm512_set1_epi32(-32767));
                __m512i clamped1 = _mm512_max_epi32(_mm512_min_epi32(q1, _mm512_set1_epi32(32767)), _mm512_set1_epi32(-32767));

                // Pack 32-bit to 16-bit
                __m256i packed0 = _mm512_cvtepi32_epi16(clamped0);
                __m256i packed1 = _mm512_cvtepi32_epi16(clamped1);

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block_out.qs), packed0);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block_out.qs + 16), packed1);

                // Compute sum_qs (raw integer sum, INT32 for Q16_1)
                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    sum_qs += static_cast<int64_t>(block_out.qs[i]);
                }
                block_out.sum_qs = static_cast<int32_t>(sum_qs);

#elif defined(__AVX2__)
                // Dequant A and B, add, and store to temp buffer
                alignas(32) float temp[32];
                __m256 scale_a_vec = _mm256_set1_ps(scale_a);
                __m256 scale_b_vec = _mm256_set1_ps(scale_b);

                // Process 8 int16 values at a time (4 iterations for 32 elements)
                for (int i = 0; i < 4; ++i)
                {
                    __m128i qa16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_a.qs + i * 8));
                    __m128i qb16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_b.qs + i * 8));
                    __m256i ia32 = _mm256_cvtepi16_epi32(qa16);
                    __m256i ib32 = _mm256_cvtepi16_epi32(qb16);
                    __m256 fa = _mm256_mul_ps(_mm256_cvtepi32_ps(ia32), scale_a_vec);
                    __m256 fb = _mm256_mul_ps(_mm256_cvtepi32_ps(ib32), scale_b_vec);
                    __m256 sum = _mm256_add_ps(fa, fb);
                    _mm256_store_ps(temp + i * 8, sum);
                }

                // Find max_abs
                __m256 vmax0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp));
                __m256 vmax1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 8));
                __m256 vmax2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 16));
                __m256 vmax3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 24));
                __m256 vmax_01 = _mm256_max_ps(vmax0, vmax1);
                __m256 vmax_23 = _mm256_max_ps(vmax2, vmax3);
                __m256 vmax_all = _mm256_max_ps(vmax_01, vmax_23);
                __m128 lo = _mm256_castps256_ps128(vmax_all);
                __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
                __m128 vmax128 = _mm_max_ps(lo, hi);
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
                float max_abs = _mm_cvtss_f32(vmax128);

                if (max_abs < 1e-10f)
                {
                    block_out.d = 0.0f;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                block_out.d = out_scale;
                float inv_scale = 32767.0f / max_abs;

                // Quantize to int16
                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    block_out.qs[i] = static_cast<int16_t>(q);
                    sum_qs += static_cast<int64_t>(q);
                }
                block_out.sum_qs = static_cast<int32_t>(sum_qs);

#else
                // Scalar fallback
                alignas(32) float temp[32];

                // Dequant and add
                for (int i = 0; i < 32; ++i)
                {
                    float va = scale_a * static_cast<float>(block_a.qs[i]);
                    float vb = scale_b * static_cast<float>(block_b.qs[i]);
                    temp[i] = va + vb;
                }

                // Find max_abs
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(temp[i]));
                }

                if (max_abs < 1e-10f)
                {
                    block_out.d = 0.0f;
                    block_out.sum_qs = 0;
                    std::memset(block_out.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                block_out.d = out_scale;
                float inv_scale = 32767.0f / max_abs;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    block_out.qs[i] = static_cast<int16_t>(q);
                    sum_qs += static_cast<int64_t>(q);
                }
                block_out.sum_qs = static_cast<int32_t>(sum_qs);
#endif
            }
        }

        /**
         * @brief Q16_1 + FP32 addition (add FP32 delta to Q16_1 residual)
         *
         * Common pattern: residual (Q16_1) += attention_output (FP32)
         * This is slightly faster than converting FP32 to Q16_1 first.
         *
         * @param residual Q16_1 block buffer (input and output)
         * @param delta FP32 values to add
         * @param count Number of elements (must be multiple of 32)
         */
        inline void q16_1_add_fp32(
            Q16_1Block *residual, const float *delta, size_t count)
        {
            const size_t n_blocks = count / 32;

            for (size_t blk = 0; blk < n_blocks; ++blk)
            {
                Q16_1Block &block = residual[blk];
                const float *delta_ptr = delta + blk * 32;

                const float scale = block.d;

#ifdef __AVX512F__
                // Load and dequant residual
                __m256i qr_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));
                __m256i qr_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs + 16));
                __m512 fr0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qr_lo)), _mm512_set1_ps(scale));
                __m512 fr1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qr_hi)), _mm512_set1_ps(scale));

                // Load delta (already FP32)
                __m512 fd0 = _mm512_loadu_ps(delta_ptr);
                __m512 fd1 = _mm512_loadu_ps(delta_ptr + 16);

                // Add
                __m512 sum0 = _mm512_add_ps(fr0, fd0);
                __m512 sum1 = _mm512_add_ps(fr1, fd1);

                // Find max_abs for requantization
                __m512 abs0 = _mm512_abs_ps(sum0);
                __m512 abs1 = _mm512_abs_ps(sum1);
                __m512 vmax = _mm512_max_ps(abs0, abs1);
                float max_abs = _mm512_reduce_max_ps(vmax);

                if (max_abs < 1e-10f)
                {
                    block.d = 0.0f;
                    block.sum_qs = 0;
                    std::memset(block.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                block.d = out_scale;
                float inv_scale = 32767.0f / max_abs;

                __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
                __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(sum0, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(sum1, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));

                __m512i clamped0 = _mm512_max_epi32(_mm512_min_epi32(q0, _mm512_set1_epi32(32767)), _mm512_set1_epi32(-32767));
                __m512i clamped1 = _mm512_max_epi32(_mm512_min_epi32(q1, _mm512_set1_epi32(32767)), _mm512_set1_epi32(-32767));

                __m256i packed0 = _mm512_cvtepi32_epi16(clamped0);
                __m256i packed1 = _mm512_cvtepi32_epi16(clamped1);

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs), packed0);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + 16), packed1);

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    sum_qs += static_cast<int64_t>(block.qs[i]);
                }
                block.sum_qs = static_cast<int32_t>(sum_qs);

#elif defined(__AVX2__)
                alignas(32) float temp[32];
                __m256 scale_vec = _mm256_set1_ps(scale);

                for (int i = 0; i < 4; ++i)
                {
                    __m128i qr16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs + i * 8));
                    __m256i qr32 = _mm256_cvtepi16_epi32(qr16);
                    __m256 fr = _mm256_mul_ps(_mm256_cvtepi32_ps(qr32), scale_vec);
                    __m256 fd = _mm256_loadu_ps(delta_ptr + i * 8);
                    __m256 sum = _mm256_add_ps(fr, fd);
                    _mm256_store_ps(temp + i * 8, sum);
                }

                // Find max_abs
                __m256 vmax0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp));
                __m256 vmax1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 8));
                __m256 vmax2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 16));
                __m256 vmax3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 24));
                __m256 vmax_01 = _mm256_max_ps(vmax0, vmax1);
                __m256 vmax_23 = _mm256_max_ps(vmax2, vmax3);
                __m256 vmax_all = _mm256_max_ps(vmax_01, vmax_23);
                __m128 lo = _mm256_castps256_ps128(vmax_all);
                __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
                __m128 vmax128 = _mm_max_ps(lo, hi);
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
                float max_abs = _mm_cvtss_f32(vmax128);

                if (max_abs < 1e-10f)
                {
                    block.d = 0.0f;
                    block.sum_qs = 0;
                    std::memset(block.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                block.d = out_scale;
                float inv_scale = 32767.0f / max_abs;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    block.qs[i] = static_cast<int16_t>(q);
                    sum_qs += static_cast<int64_t>(q);
                }
                block.sum_qs = static_cast<int32_t>(sum_qs);

#else
                // Scalar fallback
                alignas(32) float temp[32];

                for (int i = 0; i < 32; ++i)
                {
                    float vr = scale * static_cast<float>(block.qs[i]);
                    temp[i] = vr + delta_ptr[i];
                }

                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(temp[i]));
                }

                if (max_abs < 1e-10f)
                {
                    block.d = 0.0f;
                    block.sum_qs = 0;
                    std::memset(block.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                block.d = out_scale;
                float inv_scale = 32767.0f / max_abs;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    block.qs[i] = static_cast<int16_t>(q);
                    sum_qs += static_cast<int64_t>(q);
                }
                block.sum_qs = static_cast<int32_t>(sum_qs);
#endif
            }
        }

        /**
         * @brief FP32 residual += Q8_1 delta (dequantize-and-add in one pass)
         *
         * Used in HybridQ16 mode when the residual stream hasn't transitioned
         * to Q16_1 yet. Avoids allocating a temporary FP32 buffer for dequantization.
         *
         * Algorithm per block:
         * 1. Dequant Q8_1 delta inline: FP32 = fp16_to_fp32(scale) × int8_qs
         * 2. Add directly to FP32 residual
         *
         * @param residual FP32 buffer (modified in-place)
         * @param delta Q8_1 block buffer to add
         * @param count Number of elements (must be multiple of 32)
         */
        inline void fp32_add_q8_1(
            float *residual, const Q8_1Block *delta, size_t count)
        {
            const size_t n_blocks = count / 32;

            for (size_t blk = 0; blk < n_blocks; ++blk)
            {
                const Q8_1Block &block_d = delta[blk];
                float *res_ptr = residual + blk * 32;

                // Q8_1 uses FP16 scale - convert to FP32
                const float scale_d = fp16_to_fp32(block_d.d);

#ifdef __AVX512F__
                // Load delta (Q8_1: int8 → FP32) and add directly to residual
                __m128i qd_bytes_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_d.qs));
                __m128i qd_bytes_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_d.qs + 16));
                __m512i qd_lo = _mm512_cvtepi8_epi32(qd_bytes_lo);
                __m512i qd_hi = _mm512_cvtepi8_epi32(qd_bytes_hi);
                __m512 fd0 = _mm512_mul_ps(_mm512_cvtepi32_ps(qd_lo), _mm512_set1_ps(scale_d));
                __m512 fd1 = _mm512_mul_ps(_mm512_cvtepi32_ps(qd_hi), _mm512_set1_ps(scale_d));

                // Load existing residual, add, store
                __m512 r0 = _mm512_loadu_ps(res_ptr);
                __m512 r1 = _mm512_loadu_ps(res_ptr + 16);
                _mm512_storeu_ps(res_ptr, _mm512_add_ps(r0, fd0));
                _mm512_storeu_ps(res_ptr + 16, _mm512_add_ps(r1, fd1));

#elif defined(__AVX2__)
                __m256 scale_d_vec = _mm256_set1_ps(scale_d);

                // Process 8 elements at a time (4 iterations for 32 elements)
                for (int i = 0; i < 4; ++i)
                {
                    // Dequant Q8_1 delta (load 8 int8 → extend to int32 → FP32)
                    __m128i qd8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block_d.qs + i * 8));
                    __m256i qd32 = _mm256_cvtepi8_epi32(qd8);
                    __m256 fd = _mm256_mul_ps(_mm256_cvtepi32_ps(qd32), scale_d_vec);

                    // Load existing residual, add, store
                    __m256 r = _mm256_loadu_ps(res_ptr + i * 8);
                    _mm256_storeu_ps(res_ptr + i * 8, _mm256_add_ps(r, fd));
                }

#else
                // Scalar fallback
                for (int i = 0; i < 32; ++i)
                {
                    float vd = scale_d * static_cast<float>(block_d.qs[i]);
                    res_ptr[i] += vd;
                }
#endif
            }
        }

        /**
         * @brief Convert Q16_1 to Q8_1 with optimized SIMD packing
         *
         * Used for Q16_1 residual → Q8_1 activation conversion before GEMM.
         * Faster than to_q8_1() for bulk conversion in the inference path.
         *
         * @param src Q16_1 source blocks
         * @param dst Q8_1 destination blocks
         * @param n_blocks Number of blocks to convert
         */
        inline void q16_1_to_q8_1_packed(
            const Q16_1Block *src, Q8_1Block *dst, size_t n_blocks)
        {
            for (size_t blk = 0; blk < n_blocks; ++blk)
            {
                const Q16_1Block &src_block = src[blk];
                Q8_1Block &dst_block = dst[blk];

                const float scale = src_block.d;

#ifdef __AVX512F__
                // Dequant Q16_1 to FP32
                __m256i q16_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src_block.qs));
                __m256i q16_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src_block.qs + 16));
                __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q16_lo)), _mm512_set1_ps(scale));
                __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q16_hi)), _mm512_set1_ps(scale));

                // Find max_abs for Q8_1 requantization
                __m512 abs0 = _mm512_abs_ps(f0);
                __m512 abs1 = _mm512_abs_ps(f1);
                __m512 vmax = _mm512_max_ps(abs0, abs1);
                float max_abs = _mm512_reduce_max_ps(vmax);

                if (max_abs < 1e-6f)
                {
                    dst_block.d = 0;
                    dst_block.sum_qs = 0;
                    std::memset(dst_block.qs, 0, 32);
                    continue;
                }

                // Q8_1 uses FP16 scale, int8 range
                float out_scale = max_abs / 127.0f;
                dst_block.d = fp32_to_fp16(out_scale);
                float inv_scale = 127.0f / max_abs;

                __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
                __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(f0, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(f1, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));

                __m512i clamped0 = _mm512_max_epi32(_mm512_min_epi32(q0, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
                __m512i clamped1 = _mm512_max_epi32(_mm512_min_epi32(q1, _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));

                // Pack 32-bit → 16-bit → 8-bit
                __m256i packed0 = _mm512_cvtepi32_epi16(clamped0);
                __m256i packed1 = _mm512_cvtepi32_epi16(clamped1);
                __m128i bytes0 = _mm256_cvtepi16_epi8(packed0);
                __m128i bytes1 = _mm256_cvtepi16_epi8(packed1);

                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_block.qs), bytes0);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_block.qs + 16), bytes1);

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    sum_qs += dst_block.qs[i];
                }
                dst_block.sum_qs = static_cast<int16_t>(sum_qs);

#elif defined(__AVX2__)
                alignas(32) float temp[32];
                __m256 scale_vec = _mm256_set1_ps(scale);

                for (int i = 0; i < 4; ++i)
                {
                    __m128i q16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src_block.qs + i * 8));
                    __m256i q32 = _mm256_cvtepi16_epi32(q16);
                    __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), scale_vec);
                    _mm256_store_ps(temp + i * 8, f);
                }

                // Find max_abs
                __m256 vmax0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp));
                __m256 vmax1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 8));
                __m256 vmax2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 16));
                __m256 vmax3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 24));
                __m256 vmax_01 = _mm256_max_ps(vmax0, vmax1);
                __m256 vmax_23 = _mm256_max_ps(vmax2, vmax3);
                __m256 vmax_all = _mm256_max_ps(vmax_01, vmax_23);
                __m128 lo = _mm256_castps256_ps128(vmax_all);
                __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
                __m128 vmax128 = _mm_max_ps(lo, hi);
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
                float max_abs = _mm_cvtss_f32(vmax128);

                if (max_abs < 1e-6f)
                {
                    dst_block.d = 0;
                    dst_block.sum_qs = 0;
                    std::memset(dst_block.qs, 0, 32);
                    continue;
                }

                float out_scale = max_abs / 127.0f;
                dst_block.d = fp32_to_fp16(out_scale);
                float inv_scale = 127.0f / max_abs;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    dst_block.qs[i] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                dst_block.sum_qs = static_cast<int16_t>(sum_qs);

#else
                // Scalar fallback
                alignas(32) float temp[32];

                for (int i = 0; i < 32; ++i)
                {
                    temp[i] = scale * static_cast<float>(src_block.qs[i]);
                }

                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(temp[i]));
                }

                if (max_abs < 1e-6f)
                {
                    dst_block.d = 0;
                    dst_block.sum_qs = 0;
                    std::memset(dst_block.qs, 0, 32);
                    continue;
                }

                float out_scale = max_abs / 127.0f;
                dst_block.d = fp32_to_fp16(out_scale);
                float inv_scale = 127.0f / max_abs;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    dst_block.qs[i] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                dst_block.sum_qs = static_cast<int16_t>(sum_qs);
#endif
            }
        }

        /**
         * @brief Q16_1 residual += Q8_1 delta (most common residual pattern)
         *
         * This is THE key operation for typed residual connections:
         *   residual (Q16_1) += layer_output (Q8_1)
         *
         * Occurs twice per layer (after attention and after FFN), so optimizing
         * this saves significant memory bandwidth vs converting Q8_1→FP32 first.
         *
         * Algorithm per block:
         * 1. Dequant Q16_1 residual: FP32 = scale_r × int16_qs
         * 2. Dequant Q8_1 delta: FP32 = fp16_to_fp32(scale_d) × int8_qs
         * 3. Add in FP32 registers
         * 4. Requantize to Q16_1
         *
         * @param residual Q16_1 block buffer (modified in-place)
         * @param delta Q8_1 block buffer to add
         * @param count Number of elements (must be multiple of 32)
         */
        inline void q16_1_add_q8_1(
            Q16_1Block *residual, const Q8_1Block *delta, size_t count)
        {
            const size_t n_blocks = count / 32;

            for (size_t blk = 0; blk < n_blocks; ++blk)
            {
                Q16_1Block &block_r = residual[blk];
                const Q8_1Block &block_d = delta[blk];

                // Q16_1 uses FP32 scale directly
                const float scale_r = block_r.d;
                // Q8_1 uses FP16 scale - convert to FP32
                const float scale_d = fp16_to_fp32(block_d.d);

#ifdef __AVX512F__
                // Load and dequant residual (Q16_1: int16 → FP32)
                __m256i qr_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block_r.qs));
                __m256i qr_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block_r.qs + 16));
                __m512 fr0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qr_lo)), _mm512_set1_ps(scale_r));
                __m512 fr1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qr_hi)), _mm512_set1_ps(scale_r));

                // Load and dequant delta (Q8_1: int8 → FP32)
                __m128i qd_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_d.qs));
                __m128i qd_bytes_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_d.qs + 16));
                __m512i qd_lo = _mm512_cvtepi8_epi32(qd_bytes);
                __m512i qd_hi = _mm512_cvtepi8_epi32(qd_bytes_hi);
                __m512 fd0 = _mm512_mul_ps(_mm512_cvtepi32_ps(qd_lo), _mm512_set1_ps(scale_d));
                __m512 fd1 = _mm512_mul_ps(_mm512_cvtepi32_ps(qd_hi), _mm512_set1_ps(scale_d));

                // Add
                __m512 sum0 = _mm512_add_ps(fr0, fd0);
                __m512 sum1 = _mm512_add_ps(fr1, fd1);

                // Find max_abs for requantization
                __m512 abs0 = _mm512_abs_ps(sum0);
                __m512 abs1 = _mm512_abs_ps(sum1);
                __m512 vmax = _mm512_max_ps(abs0, abs1);
                float max_abs = _mm512_reduce_max_ps(vmax);

                if (max_abs < 1e-10f)
                {
                    block_r.d = 0.0f;
                    block_r.sum_qs = 0;
                    std::memset(block_r.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                block_r.d = out_scale;
                float inv_scale = 32767.0f / max_abs;

                __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
                __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(sum0, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));
                __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(sum1, inv_scale_vec), _MM_FROUND_TO_NEAREST_INT));

                __m512i clamped0 = _mm512_max_epi32(_mm512_min_epi32(q0, _mm512_set1_epi32(32767)), _mm512_set1_epi32(-32767));
                __m512i clamped1 = _mm512_max_epi32(_mm512_min_epi32(q1, _mm512_set1_epi32(32767)), _mm512_set1_epi32(-32767));

                __m256i packed0 = _mm512_cvtepi32_epi16(clamped0);
                __m256i packed1 = _mm512_cvtepi32_epi16(clamped1);

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block_r.qs), packed0);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block_r.qs + 16), packed1);

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    sum_qs += static_cast<int64_t>(block_r.qs[i]);
                }
                block_r.sum_qs = static_cast<int32_t>(sum_qs);

#elif defined(__AVX2__)
                alignas(32) float temp[32];
                __m256 scale_r_vec = _mm256_set1_ps(scale_r);
                __m256 scale_d_vec = _mm256_set1_ps(scale_d);

                // Process 8 elements at a time (4 iterations for 32 elements)
                for (int i = 0; i < 4; ++i)
                {
                    // Dequant Q16_1 residual
                    __m128i qr16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_r.qs + i * 8));
                    __m256i qr32 = _mm256_cvtepi16_epi32(qr16);
                    __m256 fr = _mm256_mul_ps(_mm256_cvtepi32_ps(qr32), scale_r_vec);

                    // Dequant Q8_1 delta (load 8 int8 → extend to int32 → FP32)
                    __m128i qd8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block_d.qs + i * 8));
                    __m256i qd32 = _mm256_cvtepi8_epi32(qd8);
                    __m256 fd = _mm256_mul_ps(_mm256_cvtepi32_ps(qd32), scale_d_vec);

                    __m256 sum = _mm256_add_ps(fr, fd);
                    _mm256_store_ps(temp + i * 8, sum);
                }

                // Find max_abs
                __m256 vmax0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp));
                __m256 vmax1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 8));
                __m256 vmax2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 16));
                __m256 vmax3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_load_ps(temp + 24));
                __m256 vmax_01 = _mm256_max_ps(vmax0, vmax1);
                __m256 vmax_23 = _mm256_max_ps(vmax2, vmax3);
                __m256 vmax_all = _mm256_max_ps(vmax_01, vmax_23);
                __m128 lo = _mm256_castps256_ps128(vmax_all);
                __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
                __m128 vmax128 = _mm_max_ps(lo, hi);
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
                float max_abs = _mm_cvtss_f32(vmax128);

                if (max_abs < 1e-10f)
                {
                    block_r.d = 0.0f;
                    block_r.sum_qs = 0;
                    std::memset(block_r.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                block_r.d = out_scale;
                float inv_scale = 32767.0f / max_abs;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    block_r.qs[i] = static_cast<int16_t>(q);
                    sum_qs += static_cast<int64_t>(q);
                }
                block_r.sum_qs = static_cast<int32_t>(sum_qs);

#else
                // Scalar fallback
                alignas(32) float temp[32];

                // Dequant and add
                for (int i = 0; i < 32; ++i)
                {
                    float vr = scale_r * static_cast<float>(block_r.qs[i]);
                    float vd = scale_d * static_cast<float>(block_d.qs[i]);
                    temp[i] = vr + vd;
                }

                // Find max_abs
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(temp[i]));
                }

                if (max_abs < 1e-10f)
                {
                    block_r.d = 0.0f;
                    block_r.sum_qs = 0;
                    std::memset(block_r.qs, 0, 64);
                    continue;
                }

                float out_scale = max_abs / 32767.0f;
                block_r.d = out_scale;
                float inv_scale = 32767.0f / max_abs;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    block_r.qs[i] = static_cast<int16_t>(q);
                    sum_qs += static_cast<int64_t>(q);
                }
                block_r.sum_qs = static_cast<int32_t>(sum_qs);
#endif
            }
        }

        // ==================== End Q16_1 Native Operations ====================

        // ========================================================================
        // Q16_1 Variable Block Size Operations (for Integer Attention v2)
        // ========================================================================
        //
        // These templated functions support Q16_1Block_64 and Q16_1Block_128
        // in addition to the standard Q16_1Block (32 elements).
        // They enable per-head scale normalization for attention computation.
        //
        // All functions use BlockType::BLOCK_SIZE to determine iteration count,
        // processing in 32-element chunks to reuse the optimized SIMD paths.
        // ========================================================================

        /**
         * @brief Templated Q16 + Q16 addition for variable block sizes
         *
         * Adds two Q16 blocks element-wise: output = a + b
         * Works with Q16_1Block, Q16_1Block_64, Q16_1Block_128.
         *
         * @tparam BlockType Q16 block type (must have BLOCK_SIZE, d, sum_qs, qs members)
         * @param a First input block
         * @param b Second input block
         * @param output Output block (can alias a or b for in-place)
         */
        template <typename BlockType>
        inline void q16_add_q16(const BlockType *a, const BlockType *b, BlockType *output)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_32 = BLOCK_SIZE / 32;

            // Dequantize both inputs to FP32, add, find max, requantize
            alignas(64) float temp[BLOCK_SIZE];

            const float scale_a = a->d;
            const float scale_b = b->d;

#ifdef __AVX512F__
            // Process 32 elements per iteration (16+16 with AVX512)
            __m512 vmax_all = _mm512_setzero_ps();
            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                const int16_t *qa = a->qs + chunk * 32;
                const int16_t *qb = b->qs + chunk * 32;
                float *out = temp + chunk * 32;

                __m256i qa_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qa));
                __m256i qa_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qa + 16));
                __m512 fa0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qa_lo)), _mm512_set1_ps(scale_a));
                __m512 fa1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qa_hi)), _mm512_set1_ps(scale_a));

                __m256i qb_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qb));
                __m256i qb_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qb + 16));
                __m512 fb0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qb_lo)), _mm512_set1_ps(scale_b));
                __m512 fb1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qb_hi)), _mm512_set1_ps(scale_b));

                __m512 sum0 = _mm512_add_ps(fa0, fb0);
                __m512 sum1 = _mm512_add_ps(fa1, fb1);

                _mm512_storeu_ps(out, sum0);
                _mm512_storeu_ps(out + 16, sum1);

                vmax_all = _mm512_max_ps(vmax_all, _mm512_abs_ps(sum0));
                vmax_all = _mm512_max_ps(vmax_all, _mm512_abs_ps(sum1));
            }
            float max_abs = _mm512_reduce_max_ps(vmax_all);

#elif defined(__AVX2__)
            // Process 32 elements per iteration (8*4 with AVX2)
            __m256 vmax_all = _mm256_setzero_ps();
            __m256 scale_a_vec = _mm256_set1_ps(scale_a);
            __m256 scale_b_vec = _mm256_set1_ps(scale_b);

            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                const int16_t *qa = a->qs + chunk * 32;
                const int16_t *qb = b->qs + chunk * 32;
                float *out = temp + chunk * 32;

                for (int i = 0; i < 4; ++i)
                {
                    __m128i qa16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qa + i * 8));
                    __m128i qb16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qb + i * 8));
                    __m256i ia32 = _mm256_cvtepi16_epi32(qa16);
                    __m256i ib32 = _mm256_cvtepi16_epi32(qb16);
                    __m256 fa = _mm256_mul_ps(_mm256_cvtepi32_ps(ia32), scale_a_vec);
                    __m256 fb = _mm256_mul_ps(_mm256_cvtepi32_ps(ib32), scale_b_vec);
                    __m256 sum = _mm256_add_ps(fa, fb);
                    _mm256_storeu_ps(out + i * 8, sum);
                    vmax_all = _mm256_max_ps(vmax_all, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), sum));
                }
            }
            // Horizontal max reduction for AVX2
            __m128 lo = _mm256_castps256_ps128(vmax_all);
            __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
            __m128 vmax128 = _mm_max_ps(lo, hi);
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(vmax128);
#else
            // Scalar fallback
            float max_abs = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                float va = scale_a * static_cast<float>(a->qs[i]);
                float vb = scale_b * static_cast<float>(b->qs[i]);
                temp[i] = va + vb;
                max_abs = std::max(max_abs, std::abs(temp[i]));
            }
#endif

            // Requantize
            if (max_abs < 1e-10f)
            {
                output->d = 0.0f;
                output->sum_qs = 0;
                std::memset(output->qs, 0, BLOCK_SIZE * sizeof(int16_t));
                return;
            }

            float out_scale = max_abs / 32767.0f;
            output->d = out_scale;
            float inv_scale = 32767.0f / max_abs;

            int64_t sum_qs = 0;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                output->qs[i] = static_cast<int16_t>(q);
                sum_qs += static_cast<int64_t>(q);
            }
            output->sum_qs = static_cast<int32_t>(sum_qs);
        }

        /**
         * @brief Templated Q16 + FP32 addition for variable block sizes
         *
         * Adds FP32 delta to Q16 block in-place: block += delta
         * Common pattern: residual (Q16) += attention_output (FP32)
         *
         * @tparam BlockType Q16 block type
         * @param block Q16 block to modify in-place
         * @param delta FP32 values to add (BLOCK_SIZE elements)
         */
        template <typename BlockType>
        inline void q16_add_fp32(BlockType *block, const float *delta)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_32 = BLOCK_SIZE / 32;

            alignas(64) float temp[BLOCK_SIZE];
            const float scale = block->d;

#ifdef __AVX512F__
            __m512 vmax_all = _mm512_setzero_ps();
            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                const int16_t *qr = block->qs + chunk * 32;
                const float *dp = delta + chunk * 32;
                float *out = temp + chunk * 32;

                __m256i qr_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qr));
                __m256i qr_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qr + 16));
                __m512 fr0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qr_lo)), _mm512_set1_ps(scale));
                __m512 fr1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qr_hi)), _mm512_set1_ps(scale));

                __m512 fd0 = _mm512_loadu_ps(dp);
                __m512 fd1 = _mm512_loadu_ps(dp + 16);

                __m512 sum0 = _mm512_add_ps(fr0, fd0);
                __m512 sum1 = _mm512_add_ps(fr1, fd1);

                _mm512_storeu_ps(out, sum0);
                _mm512_storeu_ps(out + 16, sum1);

                vmax_all = _mm512_max_ps(vmax_all, _mm512_abs_ps(sum0));
                vmax_all = _mm512_max_ps(vmax_all, _mm512_abs_ps(sum1));
            }
            float max_abs = _mm512_reduce_max_ps(vmax_all);

#elif defined(__AVX2__)
            __m256 vmax_all = _mm256_setzero_ps();
            __m256 scale_vec = _mm256_set1_ps(scale);

            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                const int16_t *qr = block->qs + chunk * 32;
                const float *dp = delta + chunk * 32;
                float *out = temp + chunk * 32;

                for (int i = 0; i < 4; ++i)
                {
                    __m128i qr16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qr + i * 8));
                    __m256i qr32 = _mm256_cvtepi16_epi32(qr16);
                    __m256 fr = _mm256_mul_ps(_mm256_cvtepi32_ps(qr32), scale_vec);
                    __m256 fd = _mm256_loadu_ps(dp + i * 8);
                    __m256 sum = _mm256_add_ps(fr, fd);
                    _mm256_storeu_ps(out + i * 8, sum);
                    vmax_all = _mm256_max_ps(vmax_all, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), sum));
                }
            }
            __m128 lo = _mm256_castps256_ps128(vmax_all);
            __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
            __m128 vmax128 = _mm_max_ps(lo, hi);
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(vmax128);
#else
            float max_abs = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                temp[i] = scale * static_cast<float>(block->qs[i]) + delta[i];
                max_abs = std::max(max_abs, std::abs(temp[i]));
            }
#endif

            if (max_abs < 1e-10f)
            {
                block->d = 0.0f;
                block->sum_qs = 0;
                std::memset(block->qs, 0, BLOCK_SIZE * sizeof(int16_t));
                return;
            }

            float out_scale = max_abs / 32767.0f;
            block->d = out_scale;
            float inv_scale = 32767.0f / max_abs;

            int64_t sum_qs = 0;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                block->qs[i] = static_cast<int16_t>(q);
                sum_qs += static_cast<int64_t>(q);
            }
            block->sum_qs = static_cast<int32_t>(sum_qs);
        }

        /**
         * @brief Templated Q16 + Q8_1 addition for variable block sizes
         *
         * Adds Q8_1 delta to Q16 block in-place: block += delta
         * Common pattern: residual (Q16) += attention/FFN output (Q8_1)
         *
         * Note: Q8_1Block is always 32 elements, so we process one Q8_1 block
         * per 32-element chunk of the Q16 block.
         *
         * @tparam BlockType Q16 block type
         * @param block Q16 block to modify in-place
         * @param delta Q8_1 blocks to add (BLOCK_SIZE/32 blocks)
         */
        template <typename BlockType>
        inline void q16_add_q8(BlockType *block, const Q8_1Block *delta)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_32 = BLOCK_SIZE / 32;

            alignas(64) float temp[BLOCK_SIZE];
            const float scale_r = block->d;

#ifdef __AVX512F__
            __m512 vmax_all = _mm512_setzero_ps();
            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                const int16_t *qr = block->qs + chunk * 32;
                const Q8_1Block &block_d = delta[chunk];
                const float scale_d = fp16_to_fp32(block_d.d);
                float *out = temp + chunk * 32;

                __m256i qr_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qr));
                __m256i qr_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qr + 16));
                __m512 fr0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qr_lo)), _mm512_set1_ps(scale_r));
                __m512 fr1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qr_hi)), _mm512_set1_ps(scale_r));

                __m128i qd_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_d.qs));
                __m128i qd_bytes_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block_d.qs + 16));
                __m512i qd_lo = _mm512_cvtepi8_epi32(qd_bytes);
                __m512i qd_hi = _mm512_cvtepi8_epi32(qd_bytes_hi);
                __m512 fd0 = _mm512_mul_ps(_mm512_cvtepi32_ps(qd_lo), _mm512_set1_ps(scale_d));
                __m512 fd1 = _mm512_mul_ps(_mm512_cvtepi32_ps(qd_hi), _mm512_set1_ps(scale_d));

                __m512 sum0 = _mm512_add_ps(fr0, fd0);
                __m512 sum1 = _mm512_add_ps(fr1, fd1);

                _mm512_storeu_ps(out, sum0);
                _mm512_storeu_ps(out + 16, sum1);

                vmax_all = _mm512_max_ps(vmax_all, _mm512_abs_ps(sum0));
                vmax_all = _mm512_max_ps(vmax_all, _mm512_abs_ps(sum1));
            }
            float max_abs = _mm512_reduce_max_ps(vmax_all);

#elif defined(__AVX2__)
            __m256 vmax_all = _mm256_setzero_ps();
            __m256 scale_r_vec = _mm256_set1_ps(scale_r);

            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                const int16_t *qr = block->qs + chunk * 32;
                const Q8_1Block &block_d = delta[chunk];
                const float scale_d = fp16_to_fp32(block_d.d);
                __m256 scale_d_vec = _mm256_set1_ps(scale_d);
                float *out = temp + chunk * 32;

                for (int i = 0; i < 4; ++i)
                {
                    __m128i qr16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qr + i * 8));
                    __m256i qr32 = _mm256_cvtepi16_epi32(qr16);
                    __m256 fr = _mm256_mul_ps(_mm256_cvtepi32_ps(qr32), scale_r_vec);

                    __m128i qd8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block_d.qs + i * 8));
                    __m256i qd32 = _mm256_cvtepi8_epi32(qd8);
                    __m256 fd = _mm256_mul_ps(_mm256_cvtepi32_ps(qd32), scale_d_vec);

                    __m256 sum = _mm256_add_ps(fr, fd);
                    _mm256_storeu_ps(out + i * 8, sum);
                    vmax_all = _mm256_max_ps(vmax_all, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), sum));
                }
            }
            __m128 lo = _mm256_castps256_ps128(vmax_all);
            __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
            __m128 vmax128 = _mm_max_ps(lo, hi);
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(vmax128);
#else
            float max_abs = 0.0f;
            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                const Q8_1Block &block_d = delta[chunk];
                const float scale_d = fp16_to_fp32(block_d.d);
                for (size_t i = 0; i < 32; ++i)
                {
                    size_t idx = chunk * 32 + i;
                    float vr = scale_r * static_cast<float>(block->qs[idx]);
                    float vd = scale_d * static_cast<float>(block_d.qs[i]);
                    temp[idx] = vr + vd;
                    max_abs = std::max(max_abs, std::abs(temp[idx]));
                }
            }
#endif

            if (max_abs < 1e-10f)
            {
                block->d = 0.0f;
                block->sum_qs = 0;
                std::memset(block->qs, 0, BLOCK_SIZE * sizeof(int16_t));
                return;
            }

            float out_scale = max_abs / 32767.0f;
            block->d = out_scale;
            float inv_scale = 32767.0f / max_abs;

            int64_t sum_qs = 0;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(temp[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                block->qs[i] = static_cast<int16_t>(q);
                sum_qs += static_cast<int64_t>(q);
            }
            block->sum_qs = static_cast<int32_t>(sum_qs);
        }

        /**
         * @brief Templated N-way Q16 block sum for MPI reduction
         *
         * Sums N Q16 blocks from different MPI ranks: output = Σ inputs[i]
         * Used in allreduce operations for distributed training/inference.
         *
         * @tparam BlockType Q16 block type
         * @param inputs Array of N pointers to Q16 blocks
         * @param n_inputs Number of input blocks (typically world_size)
         * @param output Output block
         */
        template <typename BlockType>
        inline void q16_sum_n(const BlockType *const *inputs, size_t n_inputs, BlockType *output)
        {
            if (n_inputs == 0)
                return;
            if (n_inputs == 1)
            {
                std::memcpy(output, inputs[0], sizeof(BlockType));
                return;
            }

            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_32 = BLOCK_SIZE / 32;

            alignas(64) float sum_vals[BLOCK_SIZE] = {0.0f};

#ifdef __AVX512F__
            // Accumulate all inputs in FP32
            for (size_t i = 0; i < n_inputs; ++i)
            {
                const BlockType *inp = inputs[i];
                const float scale = inp->d;
                const __m512 vscale = _mm512_set1_ps(scale);

                for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
                {
                    const int16_t *qs = inp->qs + chunk * 32;
                    float *acc = sum_vals + chunk * 32;

                    __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qs));
                    __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qs + 16));
                    __m512 f_lo = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_lo));
                    __m512 f_hi = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_hi));

                    __m512 acc_lo = _mm512_loadu_ps(acc);
                    __m512 acc_hi = _mm512_loadu_ps(acc + 16);

                    acc_lo = _mm512_fmadd_ps(f_lo, vscale, acc_lo);
                    acc_hi = _mm512_fmadd_ps(f_hi, vscale, acc_hi);

                    _mm512_storeu_ps(acc, acc_lo);
                    _mm512_storeu_ps(acc + 16, acc_hi);
                }
            }

            // Find max_abs
            __m512 vmax = _mm512_setzero_ps();
            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                __m512 v0 = _mm512_loadu_ps(sum_vals + chunk * 32);
                __m512 v1 = _mm512_loadu_ps(sum_vals + chunk * 32 + 16);
                vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v0));
                vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v1));
            }
            float max_abs = _mm512_reduce_max_ps(vmax);

#elif defined(__AVX2__)
            // Accumulate all inputs
            for (size_t i = 0; i < n_inputs; ++i)
            {
                const BlockType *inp = inputs[i];
                const float scale = inp->d;
                __m256 scale_vec = _mm256_set1_ps(scale);

                for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
                {
                    const int16_t *qs = inp->qs + chunk * 32;
                    float *acc = sum_vals + chunk * 32;

                    for (int j = 0; j < 4; ++j)
                    {
                        __m128i q16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qs + j * 8));
                        __m256i q32 = _mm256_cvtepi16_epi32(q16);
                        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), scale_vec);
                        __m256 existing = _mm256_loadu_ps(acc + j * 8);
                        _mm256_storeu_ps(acc + j * 8, _mm256_add_ps(existing, f));
                    }
                }
            }

            // Find max_abs
            __m256 vmax = _mm256_setzero_ps();
            for (size_t i = 0; i < BLOCK_SIZE; i += 8)
            {
                __m256 v = _mm256_loadu_ps(sum_vals + i);
                vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v));
            }
            __m128 lo = _mm256_castps256_ps128(vmax);
            __m128 hi = _mm256_extractf128_ps(vmax, 1);
            __m128 vmax128 = _mm_max_ps(lo, hi);
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(vmax128);
#else
            // Scalar accumulation
            for (size_t i = 0; i < n_inputs; ++i)
            {
                const BlockType *inp = inputs[i];
                const float scale = inp->d;
                for (size_t j = 0; j < BLOCK_SIZE; ++j)
                {
                    sum_vals[j] += scale * static_cast<float>(inp->qs[j]);
                }
            }

            float max_abs = 0.0f;
            for (size_t j = 0; j < BLOCK_SIZE; ++j)
            {
                max_abs = std::max(max_abs, std::abs(sum_vals[j]));
            }
#endif

            // Requantize
            if (max_abs < 1e-10f)
            {
                output->d = 0.0f;
                output->sum_qs = 0;
                std::memset(output->qs, 0, BLOCK_SIZE * sizeof(int16_t));
                return;
            }

            float out_scale = max_abs / 32767.0f;
            float inv_scale = 32767.0f / max_abs;
            output->d = out_scale;

            int64_t sum_qs = 0;
            for (size_t j = 0; j < BLOCK_SIZE; ++j)
            {
                int32_t q = static_cast<int32_t>(std::round(sum_vals[j] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                output->qs[j] = static_cast<int16_t>(q);
                sum_qs += static_cast<int64_t>(q);
            }
            output->sum_qs = static_cast<int32_t>(sum_qs);
        }

        // ========================================================================
        // End Q16_1 Variable Block Size Operations
        // ========================================================================

        /**
         * @brief Sum multiple FP16 arrays into one (N-way reduction)
         *
         * Used for MPI allreduce operations where we need to sum FP16 contributions
         * from N ranks. Uses AVX512 FP16 intrinsics when available, otherwise
         * converts to FP32 for the reduction.
         *
         * @param inputs Array of N pointers to FP16 arrays (uint16_t representing FP16)
         * @param n_inputs Number of input arrays (typically world_size)
         * @param output Output FP16 buffer
         * @param count Number of FP16 elements per array
         */
        inline void fp16_sum_n(
            const uint16_t *const *inputs, size_t n_inputs, uint16_t *output, size_t count)
        {
            if (n_inputs == 0)
                return;
            if (n_inputs == 1)
            {
                std::memcpy(output, inputs[0], count * sizeof(uint16_t));
                return;
            }

#ifdef __AVX512F__
            // AVX512: Process 16 FP16 values at a time
            const size_t vec_count = count / 16;
            const size_t tail_start = vec_count * 16;

            for (size_t i = 0; i < vec_count; ++i)
            {
                // Initialize accumulator with first input (convert FP16 -> FP32)
                __m256i fp16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(inputs[0] + i * 16));
                __m512 acc = _mm512_cvtph_ps(fp16_0);

                // Accumulate remaining inputs
                for (size_t j = 1; j < n_inputs; ++j)
                {
                    __m256i fp16_j = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(inputs[j] + i * 16));
                    __m512 fp32_j = _mm512_cvtph_ps(fp16_j);
                    acc = _mm512_add_ps(acc, fp32_j);
                }

                // Convert back to FP16 and store
                __m256i result = _mm512_cvtps_ph(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i * 16), result);
            }

            // Handle tail elements
            for (size_t i = tail_start; i < count; ++i)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < n_inputs; ++j)
                {
                    sum += fp16_to_fp32(inputs[j][i]);
                }
                output[i] = fp32_to_fp16(sum);
            }

#elif defined(__AVX2__)
            // AVX2: Process 8 FP16 values at a time using F16C
            const size_t vec_count = count / 8;
            const size_t tail_start = vec_count * 8;

            for (size_t i = 0; i < vec_count; ++i)
            {
                // Initialize accumulator with first input
                __m128i fp16_0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(inputs[0] + i * 8));
                __m256 acc = _mm256_cvtph_ps(fp16_0);

                // Accumulate remaining inputs
                for (size_t j = 1; j < n_inputs; ++j)
                {
                    __m128i fp16_j = _mm_loadu_si128(reinterpret_cast<const __m128i *>(inputs[j] + i * 8));
                    __m256 fp32_j = _mm256_cvtph_ps(fp16_j);
                    acc = _mm256_add_ps(acc, fp32_j);
                }

                // Convert back to FP16 and store
                __m128i result = _mm256_cvtps_ph(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i * 8), result);
            }

            // Handle tail elements
            for (size_t i = tail_start; i < count; ++i)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < n_inputs; ++j)
                {
                    sum += fp16_to_fp32(inputs[j][i]);
                }
                output[i] = fp32_to_fp16(sum);
            }

#else
            // Scalar fallback
            for (size_t i = 0; i < count; ++i)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < n_inputs; ++j)
                {
                    sum += fp16_to_fp32(inputs[j][i]);
                }
                output[i] = fp32_to_fp16(sum);
            }
#endif
        }

        /**
         * @brief Sum multiple BF16 arrays into one (N-way reduction)
         *
         * Used for MPI allreduce operations where we need to sum BF16 contributions
         * from N ranks. Uses AVX512 BF16 intrinsics when available (with AVX512BF16),
         * otherwise uses FP32 conversion.
         *
         * @param inputs Array of N pointers to BF16 arrays (uint16_t representing BF16)
         * @param n_inputs Number of input arrays (typically world_size)
         * @param output Output BF16 buffer
         * @param count Number of BF16 elements per array
         */
        inline void bf16_sum_n(
            const uint16_t *const *inputs, size_t n_inputs, uint16_t *output, size_t count)
        {
            if (n_inputs == 0)
                return;
            if (n_inputs == 1)
            {
                std::memcpy(output, inputs[0], count * sizeof(uint16_t));
                return;
            }

#ifdef __AVX512F__
            // AVX512: Process 16 BF16 values at a time
            // BF16 to FP32: Just shift left by 16 bits (zeros in low bits)
            const size_t vec_count = count / 16;
            const size_t tail_start = vec_count * 16;

            for (size_t i = 0; i < vec_count; ++i)
            {
                // Initialize accumulator with first input (BF16 -> FP32)
                // BF16 conversion: load 16-bit, sign-extend to 32-bit, shift left 16
                __m256i bf16_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(inputs[0] + i * 16));
                __m512i bf16_32_0 = _mm512_cvtepu16_epi32(bf16_0);
                __m512i fp32_bits_0 = _mm512_slli_epi32(bf16_32_0, 16);
                __m512 acc = _mm512_castsi512_ps(fp32_bits_0);

                // Accumulate remaining inputs
                for (size_t j = 1; j < n_inputs; ++j)
                {
                    __m256i bf16_j = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(inputs[j] + i * 16));
                    __m512i bf16_32_j = _mm512_cvtepu16_epi32(bf16_j);
                    __m512i fp32_bits_j = _mm512_slli_epi32(bf16_32_j, 16);
                    __m512 fp32_j = _mm512_castsi512_ps(fp32_bits_j);
                    acc = _mm512_add_ps(acc, fp32_j);
                }

                // Convert FP32 back to BF16: round and shift right 16
                // Use round-to-nearest-even by adding rounding bias
                __m512i acc_bits = _mm512_castps_si512(acc);
                // Add rounding bias: 0x7FFF + ((bits >> 16) & 1) for round-to-nearest-even
                __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(acc_bits, 16), _mm512_set1_epi32(1));
                __m512i rounding = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
                __m512i rounded = _mm512_add_epi32(acc_bits, rounding);
                __m512i bf16_32 = _mm512_srli_epi32(rounded, 16);
                __m256i result = _mm512_cvtepi32_epi16(bf16_32);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + i * 16), result);
            }

            // Handle tail elements
            for (size_t i = tail_start; i < count; ++i)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < n_inputs; ++j)
                {
                    sum += bf16_to_fp32(inputs[j][i]);
                }
                output[i] = fp32_to_bf16(sum);
            }

#elif defined(__AVX2__)
            // AVX2: Process 8 BF16 values at a time
            const size_t vec_count = count / 8;
            const size_t tail_start = vec_count * 8;

            for (size_t i = 0; i < vec_count; ++i)
            {
                // Initialize accumulator with first input
                __m128i bf16_0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(inputs[0] + i * 8));
                __m256i bf16_32_0 = _mm256_cvtepu16_epi32(bf16_0);
                __m256i fp32_bits_0 = _mm256_slli_epi32(bf16_32_0, 16);
                __m256 acc = _mm256_castsi256_ps(fp32_bits_0);

                // Accumulate remaining inputs
                for (size_t j = 1; j < n_inputs; ++j)
                {
                    __m128i bf16_j = _mm_loadu_si128(reinterpret_cast<const __m128i *>(inputs[j] + i * 8));
                    __m256i bf16_32_j = _mm256_cvtepu16_epi32(bf16_j);
                    __m256i fp32_bits_j = _mm256_slli_epi32(bf16_32_j, 16);
                    __m256 fp32_j = _mm256_castsi256_ps(fp32_bits_j);
                    acc = _mm256_add_ps(acc, fp32_j);
                }

                // Convert FP32 back to BF16 with rounding
                __m256i acc_bits = _mm256_castps_si256(acc);
                __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(acc_bits, 16), _mm256_set1_epi32(1));
                __m256i rounding = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
                __m256i rounded = _mm256_add_epi32(acc_bits, rounding);
                __m256i bf16_32 = _mm256_srli_epi32(rounded, 16);
                // Pack 32-bit to 16-bit
                __m128i lo = _mm256_castsi256_si128(bf16_32);
                __m128i hi = _mm256_extracti128_si256(bf16_32, 1);
                __m128i result = _mm_packus_epi32(lo, hi);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i * 8), result);
            }

            // Handle tail elements
            for (size_t i = tail_start; i < count; ++i)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < n_inputs; ++j)
                {
                    sum += bf16_to_fp32(inputs[j][i]);
                }
                output[i] = fp32_to_bf16(sum);
            }

#else
            // Scalar fallback
            for (size_t i = 0; i < count; ++i)
            {
                float sum = 0.0f;
                for (size_t j = 0; j < n_inputs; ++j)
                {
                    sum += bf16_to_fp32(inputs[j][i]);
                }
                output[i] = fp32_to_bf16(sum);
            }
#endif
        }

        /**
         * @brief Scale FP16 array in-place by a scalar factor
         *
         * Used for MPI allreduce operations where we need to scale values
         * before summing (e.g., scale by 1/world_size for replicated weights).
         * Uses AVX512/AVX2 F16C instructions when available.
         *
         * @param data FP16 array (uint16_t representing FP16)
         * @param count Number of FP16 elements
         * @param scale Scalar multiplier
         */
        inline void fp16_scale_inplace(uint16_t *data, size_t count, float scale)
        {
#ifdef __AVX512F__
            // AVX512: Process 16 FP16 values at a time
            const __m512 scale_vec = _mm512_set1_ps(scale);
            const size_t vec_count = count / 16;
            const size_t tail_start = vec_count * 16;

            for (size_t i = 0; i < vec_count; ++i)
            {
                __m256i fp16_in = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i * 16));
                __m512 fp32_vals = _mm512_cvtph_ps(fp16_in);
                __m512 scaled = _mm512_mul_ps(fp32_vals, scale_vec);
                __m256i result = _mm512_cvtps_ph(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(data + i * 16), result);
            }

            // Handle tail elements
            for (size_t i = tail_start; i < count; ++i)
            {
                float val = fp16_to_fp32(data[i]);
                data[i] = fp32_to_fp16(val * scale);
            }

#elif defined(__AVX2__)
            // AVX2: Process 8 FP16 values at a time using F16C
            const __m256 scale_vec = _mm256_set1_ps(scale);
            const size_t vec_count = count / 8;
            const size_t tail_start = vec_count * 8;

            for (size_t i = 0; i < vec_count; ++i)
            {
                __m128i fp16_in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i * 8));
                __m256 fp32_vals = _mm256_cvtph_ps(fp16_in);
                __m256 scaled = _mm256_mul_ps(fp32_vals, scale_vec);
                __m128i result = _mm256_cvtps_ph(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(data + i * 8), result);
            }

            // Handle tail elements
            for (size_t i = tail_start; i < count; ++i)
            {
                float val = fp16_to_fp32(data[i]);
                data[i] = fp32_to_fp16(val * scale);
            }

#else
            // Scalar fallback
            for (size_t i = 0; i < count; ++i)
            {
                float val = fp16_to_fp32(data[i]);
                data[i] = fp32_to_fp16(val * scale);
            }
#endif
        }

        /**
         * @brief Scale BF16 array in-place by a scalar factor
         *
         * Used for MPI allreduce operations where we need to scale values
         * before summing (e.g., scale by 1/world_size for replicated weights).
         * Uses AVX512 BF16 bit manipulation when available.
         *
         * @param data BF16 array (uint16_t representing BF16)
         * @param count Number of BF16 elements
         * @param scale Scalar multiplier
         */
        inline void bf16_scale_inplace(uint16_t *data, size_t count, float scale)
        {
#ifdef __AVX512F__
            // AVX512: Process 16 BF16 values at a time
            const __m512 scale_vec = _mm512_set1_ps(scale);
            const size_t vec_count = count / 16;
            const size_t tail_start = vec_count * 16;

            for (size_t i = 0; i < vec_count; ++i)
            {
                // Load 16 BF16 values
                __m256i bf16_in = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i * 16));

                // Expand BF16 to FP32: zero-extend to 32-bit then shift left by 16
                __m512i bf16_32 = _mm512_cvtepu16_epi32(bf16_in);
                __m512i fp32_bits = _mm512_slli_epi32(bf16_32, 16);
                __m512 fp32_vals = _mm512_castsi512_ps(fp32_bits);

                // Scale
                __m512 scaled = _mm512_mul_ps(fp32_vals, scale_vec);

                // Convert FP32 back to BF16 with rounding
                __m512i scaled_bits = _mm512_castps_si512(scaled);
                __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(scaled_bits, 16), _mm512_set1_epi32(1));
                __m512i rounding = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
                __m512i rounded = _mm512_add_epi32(scaled_bits, rounding);
                __m512i bf16_result = _mm512_srli_epi32(rounded, 16);

                // Pack 32-bit to 16-bit (truncate high bits)
                __m256i result = _mm512_cvtepi32_epi16(bf16_result);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(data + i * 16), result);
            }

            // Handle tail elements
            for (size_t i = tail_start; i < count; ++i)
            {
                float val = bf16_to_fp32(data[i]);
                data[i] = fp32_to_bf16(val * scale);
            }

#elif defined(__AVX2__)
            // AVX2: Process 8 BF16 values at a time
            const __m256 scale_vec = _mm256_set1_ps(scale);
            const size_t vec_count = count / 8;
            const size_t tail_start = vec_count * 8;

            for (size_t i = 0; i < vec_count; ++i)
            {
                // Load 8 BF16 values
                __m128i bf16_in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + i * 8));

                // Expand BF16 to FP32: zero-extend to 32-bit then shift left by 16
                __m256i bf16_32 = _mm256_cvtepu16_epi32(bf16_in);
                __m256i fp32_bits = _mm256_slli_epi32(bf16_32, 16);
                __m256 fp32_vals = _mm256_castsi256_ps(fp32_bits);

                // Scale
                __m256 scaled = _mm256_mul_ps(fp32_vals, scale_vec);

                // Convert FP32 back to BF16 with rounding
                __m256i scaled_bits = _mm256_castps_si256(scaled);
                __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(scaled_bits, 16), _mm256_set1_epi32(1));
                __m256i rounding = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
                __m256i rounded = _mm256_add_epi32(scaled_bits, rounding);
                __m256i bf16_32_result = _mm256_srli_epi32(rounded, 16);

                // Pack 32-bit to 16-bit
                __m128i lo = _mm256_castsi256_si128(bf16_32_result);
                __m128i hi = _mm256_extracti128_si256(bf16_32_result, 1);
                __m128i result = _mm_packus_epi32(lo, hi);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(data + i * 8), result);
            }

            // Handle tail elements
            for (size_t i = tail_start; i < count; ++i)
            {
                float val = bf16_to_fp32(data[i]);
                data[i] = fp32_to_bf16(val * scale);
            }

#else
            // Scalar fallback
            for (size_t i = 0; i < count; ++i)
            {
                float val = bf16_to_fp32(data[i]);
                data[i] = fp32_to_bf16(val * scale);
            }
#endif
        }

        /**
         * @brief Copy Q8_1 blocks (native copy, no dequant/requant)
         *
         * Simple memcpy of Q8_1 block data. Use this instead of copy_tensor
         * for Q8_1 tensors to avoid FP32 conversion overhead.
         *
         * @param src Source Q8_1 blocks
         * @param dst Destination Q8_1 blocks
         * @param n_blocks Number of blocks to copy
         */
        inline void q8_1_copy(const Q8_1Block *src, Q8_1Block *dst, size_t n_blocks)
        {
            std::memcpy(dst, src, n_blocks * sizeof(Q8_1Block));
        }

        /**
         * @brief Dequantize Q8_1 block to FP32
         *
         * @param src Q8_1 block buffer
         * @param dst FP32 output buffer
         * @param count Number of elements (must be multiple of 32)
         */
        inline void dequantize_q8_1_to_fp32(
            const Q8_1Block *src, float *dst, size_t count)
        {
            const size_t n_blocks = count / 32;

            for (size_t b = 0; b < n_blocks; ++b)
            {
                const Q8_1Block &block = src[b];
                float *block_dst = dst + b * 32;
                const float scale = fp16_to_fp32(block.d);

#ifdef __AVX512F__
                __m512 scale_vec = _mm512_set1_ps(scale);

                __m128i q_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
                __m128i q_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs + 16));

                __m512i i0 = _mm512_cvtepi8_epi32(q_lo);
                __m512i i1 = _mm512_cvtepi8_epi32(q_hi);

                __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(i0), scale_vec);
                __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(i1), scale_vec);

                _mm512_storeu_ps(block_dst, f0);
                _mm512_storeu_ps(block_dst + 16, f1);
#elif defined(__AVX2__)
                __m256 scale_vec = _mm256_set1_ps(scale);

                for (int i = 0; i < 4; ++i)
                {
                    __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.qs + i * 8));
                    __m256i i32 = _mm256_cvtepi8_epi32(q8);
                    __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), scale_vec);
                    _mm256_storeu_ps(block_dst + i * 8, f);
                }
#else
                for (int i = 0; i < 32; ++i)
                {
                    block_dst[i] = scale * static_cast<float>(block.qs[i]);
                }
#endif
            }
        }

        /**
         * @brief Quantize FP32 to Q8_1 blocks
         *
         * Self-contained implementation that doesn't depend on other functions
         * declared later in this file.
         *
         * @param src FP32 input buffer
         * @param dst Q8_1 block output buffer
         * @param count Number of elements (must be multiple of 32)
         */
        inline void quantize_fp32_to_q8_1_blocks(
            const float *src, Q8_1Block *dst, size_t count)
        {
            const size_t n_blocks = count / 32;

            for (size_t b = 0; b < n_blocks; ++b)
            {
                const float *block_src = src + b * 32;
                Q8_1Block &block = dst[b];

                // Find max absolute value
                float max_abs = 0.0f;
#if defined(__AVX512F__)
                __m512 vmax0 = _mm512_abs_ps(_mm512_loadu_ps(block_src));
                __m512 vmax1 = _mm512_abs_ps(_mm512_loadu_ps(block_src + 16));
                __m512 vmax = _mm512_max_ps(vmax0, vmax1);
                max_abs = _mm512_reduce_max_ps(vmax);
#elif defined(__AVX2__)
                __m256 vmax0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(block_src));
                __m256 vmax1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(block_src + 8));
                __m256 vmax2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(block_src + 16));
                __m256 vmax3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_loadu_ps(block_src + 24));
                __m256 vmax_01 = _mm256_max_ps(vmax0, vmax1);
                __m256 vmax_23 = _mm256_max_ps(vmax2, vmax3);
                __m256 vmax_all = _mm256_max_ps(vmax_01, vmax_23);
                // Horizontal max in AVX2
                __m128 lo = _mm256_castps256_ps128(vmax_all);
                __m128 hi = _mm256_extractf128_ps(vmax_all, 1);
                __m128 vmax128 = _mm_max_ps(lo, hi);
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
                vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
                max_abs = _mm_cvtss_f32(vmax128);
#else
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(block_src[i]));
                }
#endif

                constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
                if (max_abs < MIN_SCALE_THRESHOLD)
                {
                    block.d = 0;
                    block.sum_qs = 0;
                    std::memset(block.qs, 0, 32);
                    continue;
                }

                // Calculate scale
                float scale = max_abs / 127.0f;
                if (scale > 65504.0f)
                {
                    scale = 65504.0f;
                }
                float inv_scale = 1.0f / scale;

                // Quantize and sum
                int32_t sum_i32 = 0;
#if defined(__AVX512F__)
                __m512 vinv = _mm512_set1_ps(inv_scale);
                __m512 vmin = _mm512_set1_ps(-127.0f);
                __m512 vmax_q = _mm512_set1_ps(127.0f);

                __m512 v0 = _mm512_loadu_ps(block_src);
                __m512 v1 = _mm512_loadu_ps(block_src + 16);

                __m512 scaled0 = _mm512_mul_ps(v0, vinv);
                __m512 scaled1 = _mm512_mul_ps(v1, vinv);

                scaled0 = _mm512_max_ps(vmin, _mm512_min_ps(vmax_q, scaled0));
                scaled1 = _mm512_max_ps(vmin, _mm512_min_ps(vmax_q, scaled1));

                __m512i i0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaled0, _MM_FROUND_TO_NEAREST_INT));
                __m512i i1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaled1, _MM_FROUND_TO_NEAREST_INT));

                // Pack to int8
                __m256i i16_0 = _mm512_cvtsepi32_epi16(i0);
                __m256i i16_1 = _mm512_cvtsepi32_epi16(i1);
                __m128i i8_0 = _mm256_cvtsepi16_epi8(i16_0);
                __m128i i8_1 = _mm256_cvtsepi16_epi8(i16_1);

                _mm_storeu_si128(reinterpret_cast<__m128i *>(block.qs), i8_0);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(block.qs + 16), i8_1);

                // Compute sum
                sum_i32 = _mm512_reduce_add_epi32(i0) + _mm512_reduce_add_epi32(i1);
#else
                for (int i = 0; i < 32; ++i)
                {
                    float scaled = block_src[i] * inv_scale;
                    block.qs[i] = static_cast<int8_t>(std::round(std::max(-127.0f, std::min(127.0f, scaled))));
                    sum_i32 += static_cast<int32_t>(block.qs[i]);
                }
#endif

                // Store scale as FP16 and sum as int16
                block.d = fp32_to_fp16(scale);
                block.sum_qs = static_cast<int16_t>(sum_i32);
            }
        }

        // ==========================================
        // Single-Block Q8_1 Quantization (SIMD-optimized)
        // ==========================================
        // These functions quantize a single 32-element block with SIMD optimization.
        // Used by FP32Tensor::quantize_to_q8_1() and QuantisedGemmKernel::quantize_activations()
        // for row-wise 2D tensor quantization with proper boundary handling.

        /**
         * @brief Quantize a single 32-element block from FP32 to Q8_1 (scalar fallback)
         *
         * Handles partial blocks at row boundaries by zero-padding.
         *
         * @param src Pointer to FP32 values
         * @param dst Output Q8_1 block
         * @param valid_elements Number of valid elements (≤32), rest are zero-padded
         */
        inline void quantize_single_block_scalar(const float *src, Q8_1Block &dst, int valid_elements)
        {
            // Find max absolute value
            float max_abs = 0.0f;
            for (int j = 0; j < valid_elements; ++j)
            {
                float val = std::abs(src[j]);
                if (val > max_abs)
                    max_abs = val;
            }

            // Handle zero/tiny values
            constexpr float MIN_SCALE_THRESHOLD = 1e-10f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                dst.d = 0;
                dst.sum_qs = 0;
                std::memset(dst.qs, 0, 32);
                return;
            }

            // Compute scale and inverse
            float d = max_abs / 127.0f;
            float id = 1.0f / d;
            dst.d = fp32_to_fp16(d);

            // Quantize values and compute sum
            int32_t sum_qs = 0;
            for (int j = 0; j < valid_elements; ++j)
            {
                int8_t q = static_cast<int8_t>(std::round(src[j] * id));
                dst.qs[j] = q;
                sum_qs += q;
            }

            // Zero-pad remaining elements
            for (int j = valid_elements; j < 32; ++j)
            {
                dst.qs[j] = 0;
            }

            dst.sum_qs = static_cast<int16_t>(sum_qs);
        }

#if defined(__AVX2__)
        /**
         * @brief Quantize a single full 32-element block from FP32 to Q8_1 (AVX2)
         *
         * Uses AVX2 intrinsics for vectorized:
         * - Absolute value computation
         * - Max reduction across 32 elements
         * - Scaling, rounding, clamping
         * - Sum accumulation
         * - INT32→INT8 packing
         *
         * @param src Pointer to 32 FP32 values (must be valid)
         * @param dst Output Q8_1 block
         *
         * @note Only call this for full 32-element blocks. Use scalar for partial blocks.
         */
        inline void quantize_single_block_avx2(const float *src, Q8_1Block &dst)
        {
            // Load all 32 floats (4 × 8)
            __m256 v0 = _mm256_loadu_ps(src);
            __m256 v1 = _mm256_loadu_ps(src + 8);
            __m256 v2 = _mm256_loadu_ps(src + 16);
            __m256 v3 = _mm256_loadu_ps(src + 24);

            // Compute absolute values using AND-NOT with sign bit mask
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 abs0 = _mm256_andnot_ps(sign_mask, v0);
            __m256 abs1 = _mm256_andnot_ps(sign_mask, v1);
            __m256 abs2 = _mm256_andnot_ps(sign_mask, v2);
            __m256 abs3 = _mm256_andnot_ps(sign_mask, v3);

            // Find max absolute value across all 32 elements
            __m256 max_01 = _mm256_max_ps(abs0, abs1);
            __m256 max_23 = _mm256_max_ps(abs2, abs3);
            __m256 max_all = _mm256_max_ps(max_01, max_23);

            // Horizontal max reduction
            __m128 lo = _mm256_castps256_ps128(max_all);
            __m128 hi = _mm256_extractf128_ps(max_all, 1);
            __m128 vmax128 = _mm_max_ps(lo, hi);
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(vmax128);

            // Handle zero/tiny values
            constexpr float MIN_SCALE_THRESHOLD = 1e-10f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                dst.d = 0;
                dst.sum_qs = 0;
                std::memset(dst.qs, 0, 32);
                return;
            }

            // Compute scale
            float d = max_abs / 127.0f;
            float id = 1.0f / d;
            dst.d = fp32_to_fp16(d);

            // Vectorized quantization
            __m256 vinv = _mm256_set1_ps(id);
            __m256 vmin = _mm256_set1_ps(-127.0f);
            __m256 vmax_q = _mm256_set1_ps(127.0f);

            // Scale and clamp
            __m256 scaled0 = _mm256_mul_ps(v0, vinv);
            __m256 scaled1 = _mm256_mul_ps(v1, vinv);
            __m256 scaled2 = _mm256_mul_ps(v2, vinv);
            __m256 scaled3 = _mm256_mul_ps(v3, vinv);

            scaled0 = _mm256_max_ps(vmin, _mm256_min_ps(vmax_q, scaled0));
            scaled1 = _mm256_max_ps(vmin, _mm256_min_ps(vmax_q, scaled1));
            scaled2 = _mm256_max_ps(vmin, _mm256_min_ps(vmax_q, scaled2));
            scaled3 = _mm256_max_ps(vmin, _mm256_min_ps(vmax_q, scaled3));

            // Round to nearest integer
            __m256 rounded0 = _mm256_round_ps(scaled0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 rounded1 = _mm256_round_ps(scaled1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 rounded2 = _mm256_round_ps(scaled2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 rounded3 = _mm256_round_ps(scaled3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // Convert to int32
            __m256i i0 = _mm256_cvtps_epi32(rounded0);
            __m256i i1 = _mm256_cvtps_epi32(rounded1);
            __m256i i2 = _mm256_cvtps_epi32(rounded2);
            __m256i i3 = _mm256_cvtps_epi32(rounded3);

            // Compute sum (horizontal add of all int32 values)
            __m256i sum_01 = _mm256_add_epi32(i0, i1);
            __m256i sum_23 = _mm256_add_epi32(i2, i3);
            __m256i sum_all = _mm256_add_epi32(sum_01, sum_23);
            __m128i sum_lo = _mm256_castsi256_si128(sum_all);
            __m128i sum_hi = _mm256_extracti128_si256(sum_all, 1);
            __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 0, 0, 1)));
            int32_t sum_i32 = _mm_cvtsi128_si32(sum128);

            // Pack int32 → int16 → int8
            // packs_epi32 interleaves lanes, need permute to fix
            __m256i i16_01 = _mm256_packs_epi32(i0, i1);
            __m256i i16_23 = _mm256_packs_epi32(i2, i3);
            i16_01 = _mm256_permute4x64_epi64(i16_01, _MM_SHUFFLE(3, 1, 2, 0));
            i16_23 = _mm256_permute4x64_epi64(i16_23, _MM_SHUFFLE(3, 1, 2, 0));

            __m256i i8_all = _mm256_packs_epi16(i16_01, i16_23);
            i8_all = _mm256_permute4x64_epi64(i8_all, _MM_SHUFFLE(3, 1, 2, 0));

            // Store 32 int8 values
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.qs), i8_all);
            dst.sum_qs = static_cast<int16_t>(sum_i32);
        }
#endif

#if defined(__AVX512F__)
        /**
         * @brief Quantize a single full 32-element block from FP32 to Q8_1 (AVX-512)
         *
         * Uses AVX-512 intrinsics for maximum throughput:
         * - Two 512-bit loads for 32 floats
         * - Vectorized abs, max reduction with _mm512_reduce_max_ps
         * - Efficient int32→int8 packing with saturating conversions
         *
         * @param src Pointer to 32 FP32 values (must be valid)
         * @param dst Output Q8_1 block
         *
         * @note Only call this for full 32-element blocks. Use scalar for partial blocks.
         */
        inline void quantize_single_block_avx512(const float *src, Q8_1Block &dst)
        {
            // Load all 32 floats (2 × 16)
            __m512 v0 = _mm512_loadu_ps(src);
            __m512 v1 = _mm512_loadu_ps(src + 16);

            // Compute absolute values and find max
            __m512 abs0 = _mm512_abs_ps(v0);
            __m512 abs1 = _mm512_abs_ps(v1);
            __m512 max_vec = _mm512_max_ps(abs0, abs1);
            float max_abs = _mm512_reduce_max_ps(max_vec);

            // Handle zero/tiny values
            constexpr float MIN_SCALE_THRESHOLD = 1e-10f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                dst.d = 0;
                dst.sum_qs = 0;
                std::memset(dst.qs, 0, 32);
                return;
            }

            // Compute scale
            float d = max_abs / 127.0f;
            float id = 1.0f / d;

            // Convert scale to FP16 using F16C intrinsic (faster than scalar bit manipulation)
            // AVX512F implies F16C support
            __m128 v_d = _mm_set_ss(d);
            __m128i v_h = _mm_cvtps_ph(v_d, _MM_FROUND_TO_NEAREST_INT);
            dst.d = (uint16_t)_mm_cvtsi128_si32(v_h);

            // Vectorized quantization
            __m512 vinv = _mm512_set1_ps(id);

            // Scale (no clamping needed: values are naturally in [-127, 127] range)
            // max_abs is the maximum absolute value in the block.
            // scaled = v * (127.0 / max_abs).
            // Since |v| <= max_abs, |scaled| <= 127.0.
            // Rounding might push 127.000...1 to 128, but vpmovdb saturates to 127.
            __m512 scaled0 = _mm512_mul_ps(v0, vinv);
            __m512 scaled1 = _mm512_mul_ps(v1, vinv);

            // Round and convert to int32
            // Note: _mm512_cvtps_epi32 rounds to nearest-even (default MXCSR), which is slightly different
            // from std::round (nearest, ties away from zero). We accept this for performance.
            // Removing _mm512_roundscale_ps saves significant latency.
            __m512i i0 = _mm512_cvtps_epi32(scaled0);
            __m512i i1 = _mm512_cvtps_epi32(scaled1);

            // Pack 32-bit integers to 8-bit using AVX-512 vpmovdb (direct int32 → int8)
            // This uses _mm512_cvtsepi32_epi8 which outputs a __m128i (16 bytes from 16 int32s)
            // Note: This saturates values > 127 to 127 and < -128 to -128
            __m128i i8_0 = _mm512_cvtsepi32_epi8(i0); // 16 int32 → 16 int8
            __m128i i8_1 = _mm512_cvtsepi32_epi8(i1); // 16 int32 → 16 int8

            // Combine into a single 256-bit register and store
            __m256i i8_all = _mm256_set_m128i(i8_1, i8_0);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.qs), i8_all);

            // Compute sum from saturated values to ensure consistency with stored data
            // This fixes a bug where rounding to 128 would be summed as 128 but stored as 127
            __m512i i32_sat_0 = _mm512_cvtepi8_epi32(i8_0);
            __m512i i32_sat_1 = _mm512_cvtepi8_epi32(i8_1);
            __m512i sum_vec = _mm512_add_epi32(i32_sat_0, i32_sat_1);
            int32_t sum_i32 = _mm512_reduce_add_epi32(sum_vec);

            dst.sum_qs = static_cast<int16_t>(sum_i32);
        }
#endif

        /**
         * @brief Quantize a single 32-element block from FP32 to Q8_1 (auto-dispatch)
         *
         * Dispatches to AVX-512, AVX2, or scalar based on runtime CPU detection.
         * Handles partial blocks at row boundaries.
         *
         * @param src Pointer to FP32 values
         * @param dst Output Q8_1 block
         * @param valid_elements Number of valid elements (≤32), rest are zero-padded
         */
        inline void quantize_single_block(const float *src, Q8_1Block &dst, int valid_elements = 32)
        {
            // Partial blocks always use scalar (rare, at row boundaries only)
            if (valid_elements < 32)
            {
                quantize_single_block_scalar(src, dst, valid_elements);
                return;
            }

            // Full blocks: dispatch to best SIMD
            static const bool has_avx512 = cpu_supports_avx512();
            static const bool has_avx2 = cpu_supports_avx2();

#if defined(__AVX512F__)
            if (has_avx512)
            {
                quantize_single_block_avx512(src, dst);
                return;
            }
#endif
#if defined(__AVX2__)
            if (has_avx2)
            {
                quantize_single_block_avx2(src, dst);
                return;
            }
#endif
            quantize_single_block_scalar(src, dst, 32);
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
#if defined(__AVX512F__) && defined(__AVX512BW__)
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq2xxs_to_q8_0_avx512: subblock_idx out of range");
            }

            const float d = fp16_to_fp32(block.d);

            // Load 8 bytes (2 uint32) from block.qs
            const uint16_t *qs_ptr = block.qs + 4 * subblock_idx;
            uint64_t raw_data = *(const uint64_t *)qs_ptr;
            uint32_t aux0 = (uint32_t)raw_data;
            uint32_t aux1 = (uint32_t)(raw_data >> 32);

            const float db = d * (0.5f + (aux1 >> 28)) * 0.25f;

            // Grid indices: aux0 (4 bytes) -> 4 indices
            // aux0 contains qs[1] << 16 | qs[0].
            // We need to extract the 4 bytes of aux0 as indices.
            __m128i vaux0 = _mm_cvtsi32_si128(aux0);
            __m128i vindices_32 = _mm_cvtepu8_epi32(vaux0);

            // Gather 4 64-bit values (32 bytes total)
            __m256i vgrid_64 = _mm256_i32gather_epi64((const long long int *)iq2xxs_grid, vindices_32, 8);

            // Find max absolute value for scaling
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid_64);
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid_64, 1);

            uint8_t max_grid_0 = hmax_epu8_128(vgrid_lo);
            uint8_t max_grid_1 = hmax_epu8_128(vgrid_hi);
            uint8_t max_grid = std::max(max_grid_0, max_grid_1);

            float max_val = db * max_grid;

            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            if (max_val < MIN_SCALE_THRESHOLD)
            {
                *q8_scale = 0;
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), _mm256_setzero_si256());
                return;
            }

            float scale = max_val / 127.0f;
            if (scale > 65504.0f)
                scale = 65504.0f;
            *q8_scale = fp32_to_fp16(scale);

            float inv_scale = 1.0f / scale;
            int32_t ratio = (int32_t)(db * inv_scale * 65536.0f + 0.5f);
            __m512i vratio = _mm512_set1_epi32(ratio);

            // Signs
            uint8_t s0 = ksigns_iq2xs[aux1 & 127];
            uint8_t s1 = ksigns_iq2xs[(aux1 >> 7) & 127];
            uint8_t s2 = ksigns_iq2xs[(aux1 >> 14) & 127];
            uint8_t s3 = ksigns_iq2xs[(aux1 >> 21) & 127];

            // Broadcast to 64-bit (8 bytes)
            uint64_t s0_64 = 0x0101010101010101ULL * s0;
            uint64_t s1_64 = 0x0101010101010101ULL * s1;
            uint64_t s2_64 = 0x0101010101010101ULL * s2;
            uint64_t s3_64 = 0x0101010101010101ULL * s3;

            __m256i vsigns = _mm256_setr_epi64x(s0_64, s1_64, s2_64, s3_64);
            __m256i vmask = _mm256_set1_epi64x(0x8040201008040201ULL); // 1, 2, 4, 8, 16, 32, 64, 128

            // Check if bit is set
            __m256i vneg = _mm256_and_si256(vsigns, vmask);
            vneg = _mm256_cmpeq_epi8(vneg, vmask); // 0xFF where negative

            // Unpack grid to 32-bit integers (split into two halves)
            __m128i vgrid_lo_128 = _mm256_castsi256_si128(vgrid_64);
            __m128i vgrid_hi_128 = _mm256_extracti128_si256(vgrid_64, 1);

            __m512i vints_lo = _mm512_cvtepu8_epi32(vgrid_lo_128);
            __m512i vints_hi = _mm512_cvtepu8_epi32(vgrid_hi_128);

            // Multiply by ratio
            __m512i vres_lo = _mm512_mullo_epi32(vints_lo, vratio);
            __m512i vres_hi = _mm512_mullo_epi32(vints_hi, vratio);

            vres_lo = _mm512_srli_epi32(vres_lo, 16);
            vres_hi = _mm512_srli_epi32(vres_hi, 16);

            // Apply signs
            uint32_t neg_mask = _mm256_movemask_epi8(vneg);
            __mmask16 kneg_lo = (__mmask16)(neg_mask & 0xFFFF);
            __mmask16 kneg_hi = (__mmask16)(neg_mask >> 16);

            // Negate where kneg is set
            vres_lo = _mm512_mask_sub_epi32(vres_lo, kneg_lo, _mm512_setzero_si512(), vres_lo);
            vres_hi = _mm512_mask_sub_epi32(vres_hi, kneg_hi, _mm512_setzero_si512(), vres_hi);

            // Pack to int8
            __m128i vq8_lo = _mm512_cvtepi32_epi8(vres_lo);
            __m128i vq8_hi = _mm512_cvtepi32_epi8(vres_hi);

            __m256i vq8 = _mm256_inserti128_si256(_mm256_castsi128_si256(vq8_lo), vq8_hi, 1);
            _mm256_storeu_si256((__m256i *)q8_qs, vq8);
#else
            // Fallback to scalar
            decode_iq2xxs_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
#endif
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
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                decode_iq2xxs_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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

        inline void unpack_iq2_xxs_superblock_to_int8_avx512(
            const IQ2_XXSBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
            const float d = fp16_to_fp32(block.d);

            // Precompute scales and signs
            float db_arr[8];
            uint32_t aux1_arr[8];

            for (int i = 0; i < 8; ++i)
            {
                const uint16_t *qs_ptr = block.qs + 4 * i;
                uint64_t raw_data = *(const uint64_t *)qs_ptr;
                uint32_t aux1 = (uint32_t)(raw_data >> 32);
                aux1_arr[i] = aux1;
                db_arr[i] = d * (0.5f + (aux1 >> 28)) * 0.25f;
            }

            for (int ib = 0; ib < 8; ib += 2)
            {
                // Load 16 bytes (2 subblocks)
                const uint16_t *qs_ptr = block.qs + 4 * ib;
                __m128i vraw = _mm_loadu_si128((const __m128i *)qs_ptr);

                // Extract indices (bytes 0-3 and 8-11)
                __m128i vshuffle = _mm_setr_epi8(
                    0, 1, 2, 3, 8, 9, 10, 11,
                    -1, -1, -1, -1, -1, -1, -1, -1);
                __m128i vindices_8 = _mm_shuffle_epi8(vraw, vshuffle);
                __m256i vindices = _mm256_cvtepu8_epi32(vindices_8);

                // Gather
                __m512i vgrid = _mm512_i32gather_epi64(vindices, iq2xxs_grid, 8);

                __m256i vgrid_ib = _mm512_castsi512_si256(vgrid);
                __m256i vgrid_ib1 = _mm512_extracti64x4_epi64(vgrid, 1);

                // Process ib
                {
                    __m128i vg0 = _mm256_castsi256_si128(vgrid_ib);
                    __m128i vg1 = _mm256_extracti128_si256(vgrid_ib, 1);

                    __m512 vf0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vg0));
                    __m512 vf1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vg1));

                    __m512 vdb = _mm512_set1_ps(db_arr[ib]);
                    vf0 = _mm512_mul_ps(vf0, vdb);
                    vf1 = _mm512_mul_ps(vf1, vdb);

                    uint32_t aux1 = aux1_arr[ib];
                    uint8_t s0 = ksigns_iq2xs[aux1 & 127];
                    uint8_t s1 = ksigns_iq2xs[(aux1 >> 7) & 127];
                    uint8_t s2 = ksigns_iq2xs[(aux1 >> 14) & 127];
                    uint8_t s3 = ksigns_iq2xs[(aux1 >> 21) & 127];

                    __mmask16 k0 = s0 | (s1 << 8);
                    __mmask16 k1 = s2 | (s3 << 8);

                    vf0 = _mm512_mask_sub_ps(vf0, k0, _mm512_setzero_ps(), vf0);
                    vf1 = _mm512_mask_sub_ps(vf1, k1, _mm512_setzero_ps(), vf1);

                    // Quantize
                    __m512 abs0 = _mm512_abs_ps(vf0);
                    __m512 abs1 = _mm512_abs_ps(vf1);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib] = scale;
                    if (mins)
                        mins[ib] = 0.0f;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + ib * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512 vq0 = _mm512_mul_ps(vf0, vinv);
                        __m512i vi0 = _mm512_cvtps_epi32(vq0);

                        __m512 vq1 = _mm512_mul_ps(vf1, vinv);
                        __m512i vi1 = _mm512_cvtps_epi32(vq1);

                        __m128i vout0 = _mm512_cvtepi32_epi8(vi0);
                        __m128i vout1 = _mm512_cvtepi32_epi8(vi1);

                        _mm_storeu_si128((__m128i *)(output + ib * 32), vout0);
                        _mm_storeu_si128((__m128i *)(output + ib * 32 + 16), vout1);
                    }
                }

                // Process ib+1
                {
                    __m128i vg0 = _mm256_castsi256_si128(vgrid_ib1);
                    __m128i vg1 = _mm256_extracti128_si256(vgrid_ib1, 1);

                    __m512 vf0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vg0));
                    __m512 vf1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vg1));

                    __m512 vdb = _mm512_set1_ps(db_arr[ib + 1]);
                    vf0 = _mm512_mul_ps(vf0, vdb);
                    vf1 = _mm512_mul_ps(vf1, vdb);

                    uint32_t aux1 = aux1_arr[ib + 1];
                    uint8_t s0 = ksigns_iq2xs[aux1 & 127];
                    uint8_t s1 = ksigns_iq2xs[(aux1 >> 7) & 127];
                    uint8_t s2 = ksigns_iq2xs[(aux1 >> 14) & 127];
                    uint8_t s3 = ksigns_iq2xs[(aux1 >> 21) & 127];

                    __mmask16 k0 = s0 | (s1 << 8);
                    __mmask16 k1 = s2 | (s3 << 8);

                    vf0 = _mm512_mask_sub_ps(vf0, k0, _mm512_setzero_ps(), vf0);
                    vf1 = _mm512_mask_sub_ps(vf1, k1, _mm512_setzero_ps(), vf1);

                    // Quantize
                    __m512 abs0 = _mm512_abs_ps(vf0);
                    __m512 abs1 = _mm512_abs_ps(vf1);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib + 1] = scale;
                    if (mins)
                        mins[ib + 1] = 0.0f;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + (ib + 1) * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512 vq0 = _mm512_mul_ps(vf0, vinv);
                        __m512i vi0 = _mm512_cvtps_epi32(vq0);

                        __m512 vq1 = _mm512_mul_ps(vf1, vinv);
                        __m512i vi1 = _mm512_cvtps_epi32(vq1);

                        __m128i vout0 = _mm512_cvtepi32_epi8(vi0);
                        __m128i vout1 = _mm512_cvtepi32_epi8(vi1);

                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32), vout0);
                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32 + 16), vout1);
                    }
                }
            }
#else
            unpack_iq2_xxs_superblock_to_int8(block, output, scales, mins);
#endif
        }

        /**
         * @brief Unpack entire IQ2_XXS super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks.
         *
         * @param block Source IQ2_XXS super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins (always 0 for IQ2_XXS)
         */
        inline void unpack_iq2_xxs_superblock_to_int8(
            const IQ2_XXSBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_iq2_xxs_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }

            for (int i = 0; i < 8; ++i)
            {
                uint16_t scale_fp16;
                decode_iq2xxs_to_q8_0(block, i, output + i * 32, &scale_fp16);
                if (scales)
                    scales[i] = fp16_to_fp32(scale_fp16);
                if (mins)
                    mins[i] = 0.0f;
            }
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
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq2xs_to_q8_0_avx512: subblock_idx out of range");
            }

            const float d = fp16_to_fp32(block.d);

            // Load 4 uint16_t values (64 bits)
            const uint16_t *qs_ptr = block.qs + 4 * subblock_idx;
            uint64_t raw_qs = *(const uint64_t *)qs_ptr;

            // Expand to 32-bit integers for gather
            __m128i vqs_128 = _mm_cvtsi64_si128(raw_qs);
            __m128i vqs_32 = _mm_cvtepu16_epi32(vqs_128); // 4 x 32-bit integers

            // Extract grid indices (low 9 bits)
            __m128i vindices = _mm_and_si128(vqs_32, _mm_set1_epi32(511));

            // Gather grid values (4 x 64-bit)
            // iq2xs_grid is uint64_t*
            __m256i vgrid_64 = _mm256_i32gather_epi64((const long long int *)iq2xs_grid, vindices, 8);

            // Calculate scales
            uint8_t scales = block.scales[subblock_idx];
            float db0 = d * (0.5f + (scales & 0xf)) * 0.25f;
            float db1 = d * (0.5f + (scales >> 4)) * 0.25f;

            // Find max grid value
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid_64);
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid_64, 1);

            uint8_t max_grid_0 = hmax_epu8_128(vgrid_lo);
            uint8_t max_grid_1 = hmax_epu8_128(vgrid_hi);

            float max_val = std::max(db0 * max_grid_0, db1 * max_grid_1);

            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            if (max_val < MIN_SCALE_THRESHOLD)
            {
                *q8_scale = 0;
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), _mm256_setzero_si256());
                return;
            }

            float scale = max_val / 127.0f;
            if (scale > 65504.0f)
                scale = 65504.0f;
            *q8_scale = fp32_to_fp16(scale);
            float inv_scale = 1.0f / scale;

            // Calculate ratios
            __m512 vscale_0 = _mm512_set1_ps(db0 * inv_scale);
            __m512 vscale_1 = _mm512_set1_ps(db1 * inv_scale);

            // Convert grid to FP32
            // vgrid_lo (16 bytes) -> 16 floats (512 bits)
            __m512 vgrid_lo_ps = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vgrid_lo));
            __m512 vgrid_hi_ps = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vgrid_hi));

            // Multiply
            __m512 vres_lo = _mm512_mul_ps(vgrid_lo_ps, vscale_0);
            __m512 vres_hi = _mm512_mul_ps(vgrid_hi_ps, vscale_1);

            // Convert to Int32
            __m512i vint_lo = _mm512_cvtps_epi32(vres_lo);
            __m512i vint_hi = _mm512_cvtps_epi32(vres_hi);

            // Signs
            // Extract sign indices: qs >> 9
            __m128i vsign_idx = _mm_srli_epi32(vqs_32, 9);

            uint32_t sign_idx_0 = _mm_extract_epi32(vsign_idx, 0);
            uint32_t sign_idx_1 = _mm_extract_epi32(vsign_idx, 1);
            uint32_t sign_idx_2 = _mm_extract_epi32(vsign_idx, 2);
            uint32_t sign_idx_3 = _mm_extract_epi32(vsign_idx, 3);

            uint8_t s0 = ksigns_iq2xs[sign_idx_0];
            uint8_t s1 = ksigns_iq2xs[sign_idx_1];
            uint8_t s2 = ksigns_iq2xs[sign_idx_2];
            uint8_t s3 = ksigns_iq2xs[sign_idx_3];

            // Broadcast signs
            uint64_t s0_64 = 0x0101010101010101ULL * s0;
            uint64_t s1_64 = 0x0101010101010101ULL * s1;
            uint64_t s2_64 = 0x0101010101010101ULL * s2;
            uint64_t s3_64 = 0x0101010101010101ULL * s3;

            __m256i vsigns = _mm256_setr_epi64x(s0_64, s1_64, s2_64, s3_64);
            __m256i vmask = _mm256_set1_epi64x(0x8040201008040201ULL); // 1, 2, 4, 8, 16, 32, 64, 128

            __m256i vneg = _mm256_and_si256(vsigns, vmask);
            vneg = _mm256_cmpeq_epi8(vneg, vmask); // 0xFF where negative

            // Split vneg to lo/hi and expand to 32-bit
            __m128i vneg_lo_128 = _mm256_castsi256_si128(vneg);
            __m128i vneg_hi_128 = _mm256_extracti128_si256(vneg, 1);

            __m512i vneg_lo_512 = _mm512_cvtepi8_epi32(vneg_lo_128); // 0xFF -> -1, 0x00 -> 0
            __m512i vneg_hi_512 = _mm512_cvtepi8_epi32(vneg_hi_128);

            // Apply signs: (x ^ -1) - (-1) = -x
            vint_lo = _mm512_sub_epi32(_mm512_xor_si512(vint_lo, vneg_lo_512), vneg_lo_512);
            vint_hi = _mm512_sub_epi32(_mm512_xor_si512(vint_hi, vneg_hi_512), vneg_hi_512);

            // Clamp to [-127, 127]
            __m512i vmin = _mm512_set1_epi32(-127);
            __m512i vmax = _mm512_set1_epi32(127);
            vint_lo = _mm512_max_epi32(vint_lo, vmin);
            vint_lo = _mm512_min_epi32(vint_lo, vmax);
            vint_hi = _mm512_max_epi32(vint_hi, vmin);
            vint_hi = _mm512_min_epi32(vint_hi, vmax);

            // Pack to 8-bit
            __m128i vq8_lo = _mm512_cvtepi32_epi8(vint_lo);
            __m128i vq8_hi = _mm512_cvtepi32_epi8(vint_hi);

            __m256i vq8 = _mm256_inserti128_si256(_mm256_castsi128_si256(vq8_lo), vq8_hi, 1);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), vq8);
        }
#endif

        /**
         * @brief Auto-dispatch IQ2_XS decode based on CPU features
         */
        inline void decode_iq2xs_to_q8_0(const IQ2_XSBlock &block, size_t subblock_idx,
                                         int8_t *q8_qs, uint16_t *q8_scale)
        {
#ifdef __AVX512F__
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                decode_iq2xs_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif
#ifdef __AVX2__
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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

        inline void unpack_iq2_xs_superblock_to_int8_avx512(
            const IQ2_XSBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
            const float d = fp16_to_fp32(block.d);

            // Precompute all 16 sub-scales (2 per subblock)
            float db_arr[16];
            for (int i = 0; i < 8; ++i)
            {
                uint8_t s = block.scales[i];
                db_arr[2 * i + 0] = d * (0.5f + (s & 0xf)) * 0.25f;
                db_arr[2 * i + 1] = d * (0.5f + (s >> 4)) * 0.25f;
            }

            // Process 2 subblocks (64 elements) per iteration
            for (int ib = 0; ib < 8; ib += 2)
            {
                // Load 8 uint16_t values (4 per subblock, 2 subblocks)
                const uint16_t *qs_ptr = block.qs + 4 * ib;
                __m128i vqs = _mm_loadu_si128((const __m128i *)qs_ptr);

                // Expand to 256-bit (8x32-bit integers)
                __m256i vqs32 = _mm256_cvtepu16_epi32(vqs);

                // Extract grid indices (low 9 bits)
                __m256i vgrid_idx = _mm256_and_si256(vqs32, _mm256_set1_epi32(511));

                // Extract sign indices (bits 9-15, 7 bits)
                __m256i vsign_idx = _mm256_srli_epi32(vqs32, 9);

                // Gather 8x64-bit grid values (fills full 512-bit register)
                __m512i vgrid = _mm512_i32gather_epi64(vgrid_idx, iq2xs_grid, 8);

                // Store sign indices to array for lookup (faster than extract)
                alignas(32) int32_t sign_indices[8];
                _mm256_store_si256((__m256i *)sign_indices, vsign_idx);

                // ===== Subblock ib: first 32 elements =====
                // Chunk 0: elements 0-15 (first half of subblock ib)
                __m128i vchunk0 = _mm512_castsi512_si128(vgrid);
                __m512 vfp0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vchunk0));
                vfp0 = _mm512_mul_ps(vfp0, _mm512_set1_ps(db_arr[2 * ib]));

                // Apply signs for chunk 0 (groups 0,1 of subblock ib)
                uint8_t s0 = ksigns_iq2xs[sign_indices[0]];
                uint8_t s1 = ksigns_iq2xs[sign_indices[1]];
                __mmask16 k0 = s0 | (s1 << 8);
                vfp0 = _mm512_mask_sub_ps(vfp0, k0, _mm512_setzero_ps(), vfp0);

                // Chunk 1: elements 16-31 (second half of subblock ib)
                __m128i vchunk1 = _mm512_extracti32x4_epi32(vgrid, 1);
                __m512 vfp1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vchunk1));
                vfp1 = _mm512_mul_ps(vfp1, _mm512_set1_ps(db_arr[2 * ib + 1]));

                // Apply signs for chunk 1 (groups 2,3 of subblock ib)
                uint8_t s2 = ksigns_iq2xs[sign_indices[2]];
                uint8_t s3 = ksigns_iq2xs[sign_indices[3]];
                __mmask16 k1 = s2 | (s3 << 8);
                vfp1 = _mm512_mask_sub_ps(vfp1, k1, _mm512_setzero_ps(), vfp1);

                // Quantize subblock ib
                {
                    __m512 abs0 = _mm512_abs_ps(vfp0);
                    __m512 abs1 = _mm512_abs_ps(vfp1);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib] = scale;
                    if (mins)
                        mins[ib] = 0.0f;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + ib * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512i vi0 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp0, vinv));
                        __m512i vi1 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp1, vinv));

                        __m128i vout0 = _mm512_cvtepi32_epi8(vi0);
                        __m128i vout1 = _mm512_cvtepi32_epi8(vi1);

                        _mm_storeu_si128((__m128i *)(output + ib * 32), vout0);
                        _mm_storeu_si128((__m128i *)(output + ib * 32 + 16), vout1);
                    }
                }

                // ===== Subblock ib+1: next 32 elements =====
                // Chunk 2: elements 0-15 (first half of subblock ib+1)
                __m128i vchunk2 = _mm512_extracti32x4_epi32(vgrid, 2);
                __m512 vfp2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vchunk2));
                vfp2 = _mm512_mul_ps(vfp2, _mm512_set1_ps(db_arr[2 * (ib + 1)]));

                // Apply signs for chunk 2 (groups 0,1 of subblock ib+1)
                uint8_t s4 = ksigns_iq2xs[sign_indices[4]];
                uint8_t s5 = ksigns_iq2xs[sign_indices[5]];
                __mmask16 k2 = s4 | (s5 << 8);
                vfp2 = _mm512_mask_sub_ps(vfp2, k2, _mm512_setzero_ps(), vfp2);

                // Chunk 3: elements 16-31 (second half of subblock ib+1)
                __m128i vchunk3 = _mm512_extracti32x4_epi32(vgrid, 3);
                __m512 vfp3 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vchunk3));
                vfp3 = _mm512_mul_ps(vfp3, _mm512_set1_ps(db_arr[2 * (ib + 1) + 1]));

                // Apply signs for chunk 3 (groups 2,3 of subblock ib+1)
                uint8_t s6 = ksigns_iq2xs[sign_indices[6]];
                uint8_t s7 = ksigns_iq2xs[sign_indices[7]];
                __mmask16 k3 = s6 | (s7 << 8);
                vfp3 = _mm512_mask_sub_ps(vfp3, k3, _mm512_setzero_ps(), vfp3);

                // Quantize subblock ib+1
                {
                    __m512 abs2 = _mm512_abs_ps(vfp2);
                    __m512 abs3 = _mm512_abs_ps(vfp3);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs2, abs3));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib + 1] = scale;
                    if (mins)
                        mins[ib + 1] = 0.0f;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + (ib + 1) * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512i vi2 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp2, vinv));
                        __m512i vi3 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp3, vinv));

                        __m128i vout2 = _mm512_cvtepi32_epi8(vi2);
                        __m128i vout3 = _mm512_cvtepi32_epi8(vi3);

                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32), vout2);
                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32 + 16), vout3);
                    }
                }
            }
#else
            unpack_iq2_xs_superblock_to_int8(block, output, scales, mins);
#endif
        }

        /**
         * @brief Unpack entire IQ2_XS super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks.
         *
         * @param block Source IQ2_XS super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins (always 0 for IQ2_XS)
         */
        inline void unpack_iq2_xs_superblock_to_int8(
            const IQ2_XSBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_iq2_xs_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }

            for (int i = 0; i < 8; ++i)
            {
                uint16_t scale_fp16;
                decode_iq2xs_to_q8_0(block, i, output + i * 32, &scale_fp16);
                if (scales)
                    scales[i] = fp16_to_fp32(scale_fp16);
                if (mins)
                    mins[i] = 0.0f;
            }
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
            if (subblock_idx >= 8)
            {
                throw std::out_of_range("decode_iq3xxs_to_q8_0_avx512: subblock_idx out of range");
            }

            const float d = fp16_to_fp32(block.d);

            // Load qs (8 bytes) -> 8 indices
            const uint8_t *qs_ptr = block.qs + 8 * subblock_idx;
            __m128i vqs_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qs_ptr));
            __m256i vindices = _mm256_cvtepu8_epi32(vqs_8);

            // Gather grid values (8 x 32-bit)
            // iq3xxs_grid is uint32_t*
            __m256i vgrid = _mm256_i32gather_epi32((const int *)iq3xxs_grid, vindices, 4);

            // Find max grid value (across 32 bytes)
            __m128i vgrid_lo_128 = _mm256_castsi256_si128(vgrid);
            __m128i vgrid_hi_128 = _mm256_extracti128_si256(vgrid, 1);
            uint8_t max_grid_0 = hmax_epu8_128(vgrid_lo_128);
            uint8_t max_grid_1 = hmax_epu8_128(vgrid_hi_128);
            uint8_t max_grid = std::max(max_grid_0, max_grid_1);

            // Calculate db
            const uint8_t *scales_ptr = block.qs + 64 + 4 * subblock_idx;
            uint32_t aux32;
            std::memcpy(&aux32, scales_ptr, 4);
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

            float max_val = db * max_grid;

            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            if (max_val < MIN_SCALE_THRESHOLD)
            {
                *q8_scale = 0;
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), _mm256_setzero_si256());
                return;
            }

            float scale = max_val / 127.0f;
            if (scale > 65504.0f)
                scale = 65504.0f;
            *q8_scale = fp32_to_fp16(scale);

            // Calculate scale factor for grid values
            // q8 = grid * 127 / max_grid
            float grid_scale = 127.0f / max_grid;
            __m512 vscale = _mm512_set1_ps(grid_scale);

            // Convert grid bytes to FP32
            __m512 vgrid_lo_ps = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vgrid_lo_128));
            __m512 vgrid_hi_ps = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vgrid_hi_128));

            // Multiply
            __m512 vres_lo = _mm512_mul_ps(vgrid_lo_ps, vscale);
            __m512 vres_hi = _mm512_mul_ps(vgrid_hi_ps, vscale);

            // Convert to Int32
            __m512i vint_lo = _mm512_cvtps_epi32(vres_lo);
            __m512i vint_hi = _mm512_cvtps_epi32(vres_hi);

            // Handle signs
            // aux32 has 4 sign indices (7 bits each).
            uint8_t s0 = ksigns_iq2xs[aux32 & 127];
            uint8_t s1 = ksigns_iq2xs[(aux32 >> 7) & 127];
            uint8_t s2 = ksigns_iq2xs[(aux32 >> 14) & 127];
            uint8_t s3 = ksigns_iq2xs[(aux32 >> 21) & 127];

            uint64_t s0_64 = 0x0101010101010101ULL * s0;
            uint64_t s1_64 = 0x0101010101010101ULL * s1;
            uint64_t s2_64 = 0x0101010101010101ULL * s2;
            uint64_t s3_64 = 0x0101010101010101ULL * s3;

            __m256i vsigns = _mm256_setr_epi64x(s0_64, s1_64, s2_64, s3_64);
            __m256i vmask = _mm256_set1_epi64x(0x8040201008040201ULL); // 1, 2, 4, 8, 16, 32, 64, 128

            __m256i vneg = _mm256_and_si256(vsigns, vmask);
            vneg = _mm256_cmpeq_epi8(vneg, vmask); // 0xFF where negative

            // Split vneg to lo/hi and expand to 32-bit
            __m128i vneg_lo_128 = _mm256_castsi256_si128(vneg);
            __m128i vneg_hi_128 = _mm256_extracti128_si256(vneg, 1);

            __m512i vneg_lo_512 = _mm512_cvtepi8_epi32(vneg_lo_128); // 0xFF -> -1, 0x00 -> 0
            __m512i vneg_hi_512 = _mm512_cvtepi8_epi32(vneg_hi_128);

            // Apply signs: (x ^ -1) - (-1) = -x
            vint_lo = _mm512_sub_epi32(_mm512_xor_si512(vint_lo, vneg_lo_512), vneg_lo_512);
            vint_hi = _mm512_sub_epi32(_mm512_xor_si512(vint_hi, vneg_hi_512), vneg_hi_512);

            // Clamp to [-127, 127]
            __m512i vmin = _mm512_set1_epi32(-127);
            __m512i vmax = _mm512_set1_epi32(127);
            vint_lo = _mm512_max_epi32(vint_lo, vmin);
            vint_lo = _mm512_min_epi32(vint_lo, vmax);
            vint_hi = _mm512_max_epi32(vint_hi, vmin);
            vint_hi = _mm512_min_epi32(vint_hi, vmax);

            // Pack to 8-bit
            __m128i vq8_lo = _mm512_cvtepi32_epi8(vint_lo);
            __m128i vq8_hi = _mm512_cvtepi32_epi8(vint_hi);

            __m256i vq8 = _mm256_inserti128_si256(_mm256_castsi128_si256(vq8_lo), vq8_hi, 1);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), vq8);
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
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                decode_iq3xxs_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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

        /**
         * @brief AVX-512 optimized IQ3_XXS superblock unpack (256 elements)
         *
         * Processes 2 subblocks (64 elements) per iteration for better throughput.
         */
        inline void unpack_iq3_xxs_superblock_to_int8_avx512(
            const IQ3_XXSBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            const float d = fp16_to_fp32(block.d);

            // Process 2 subblocks (64 elements) per iteration
            for (int ib = 0; ib < 8; ib += 2)
            {
                // ===== Load data for 2 subblocks =====
                // Load 16 bytes of qs (8 per subblock)
                const uint8_t *qs_ptr = block.qs + 8 * ib;
                __m128i vqs_16 = _mm_loadu_si128((const __m128i *)qs_ptr);

                // Expand to 16x32-bit indices
                __m512i vindices = _mm512_cvtepu8_epi32(vqs_16);

                // Gather 16 grid values (uint32_t each = 4 bytes per element)
                __m512i vgrid = _mm512_i32gather_epi32(vindices, (const int *)iq3xxs_grid, 4);

                // Load aux32 values for both subblocks (contains scales + signs)
                const uint8_t *scales_ptr = block.qs + 64 + 4 * ib;
                uint32_t aux32_0, aux32_1;
                std::memcpy(&aux32_0, scales_ptr, 4);
                std::memcpy(&aux32_1, scales_ptr + 4, 4);

                // Calculate db values
                const float db0 = d * (0.5f + (aux32_0 >> 28)) * 0.5f;
                const float db1 = d * (0.5f + (aux32_1 >> 28)) * 0.5f;

                // ===== Process subblock ib (first 32 elements) =====
                __m128i vgrid_0 = _mm512_castsi512_si128(vgrid);
                __m128i vgrid_1 = _mm512_extracti32x4_epi32(vgrid, 1);

                // Convert to FP32
                __m512 vfp0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vgrid_0));
                __m512 vfp1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vgrid_1));

                // Apply scale
                vfp0 = _mm512_mul_ps(vfp0, _mm512_set1_ps(db0));
                vfp1 = _mm512_mul_ps(vfp1, _mm512_set1_ps(db0));

                // Apply signs from aux32_0
                uint8_t s0 = ksigns_iq2xs[aux32_0 & 127];
                uint8_t s1 = ksigns_iq2xs[(aux32_0 >> 7) & 127];
                uint8_t s2 = ksigns_iq2xs[(aux32_0 >> 14) & 127];
                uint8_t s3 = ksigns_iq2xs[(aux32_0 >> 21) & 127];

                __mmask16 k0 = s0 | (s1 << 8);
                __mmask16 k1 = s2 | (s3 << 8);
                vfp0 = _mm512_mask_sub_ps(vfp0, k0, _mm512_setzero_ps(), vfp0);
                vfp1 = _mm512_mask_sub_ps(vfp1, k1, _mm512_setzero_ps(), vfp1);

                // Quantize subblock ib
                {
                    __m512 abs0 = _mm512_abs_ps(vfp0);
                    __m512 abs1 = _mm512_abs_ps(vfp1);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib] = scale;
                    if (mins)
                        mins[ib] = 0.0f;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + ib * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512i vi0 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp0, vinv));
                        __m512i vi1 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp1, vinv));

                        __m128i vout0 = _mm512_cvtepi32_epi8(vi0);
                        __m128i vout1 = _mm512_cvtepi32_epi8(vi1);

                        _mm_storeu_si128((__m128i *)(output + ib * 32), vout0);
                        _mm_storeu_si128((__m128i *)(output + ib * 32 + 16), vout1);
                    }
                }

                // ===== Process subblock ib+1 (next 32 elements) =====
                __m128i vgrid_2 = _mm512_extracti32x4_epi32(vgrid, 2);
                __m128i vgrid_3 = _mm512_extracti32x4_epi32(vgrid, 3);

                // Convert to FP32
                __m512 vfp2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vgrid_2));
                __m512 vfp3 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vgrid_3));

                // Apply scale
                vfp2 = _mm512_mul_ps(vfp2, _mm512_set1_ps(db1));
                vfp3 = _mm512_mul_ps(vfp3, _mm512_set1_ps(db1));

                // Apply signs from aux32_1
                uint8_t s4 = ksigns_iq2xs[aux32_1 & 127];
                uint8_t s5 = ksigns_iq2xs[(aux32_1 >> 7) & 127];
                uint8_t s6 = ksigns_iq2xs[(aux32_1 >> 14) & 127];
                uint8_t s7 = ksigns_iq2xs[(aux32_1 >> 21) & 127];

                __mmask16 k2 = s4 | (s5 << 8);
                __mmask16 k3 = s6 | (s7 << 8);
                vfp2 = _mm512_mask_sub_ps(vfp2, k2, _mm512_setzero_ps(), vfp2);
                vfp3 = _mm512_mask_sub_ps(vfp3, k3, _mm512_setzero_ps(), vfp3);

                // Quantize subblock ib+1
                {
                    __m512 abs2 = _mm512_abs_ps(vfp2);
                    __m512 abs3 = _mm512_abs_ps(vfp3);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs2, abs3));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib + 1] = scale;
                    if (mins)
                        mins[ib + 1] = 0.0f;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + (ib + 1) * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512i vi2 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp2, vinv));
                        __m512i vi3 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp3, vinv));

                        __m128i vout2 = _mm512_cvtepi32_epi8(vi2);
                        __m128i vout3 = _mm512_cvtepi32_epi8(vi3);

                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32), vout2);
                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32 + 16), vout3);
                    }
                }
            }
#else
            // Fallback to scalar
            for (int i = 0; i < 8; ++i)
            {
                uint16_t scale_fp16;
                decode_iq3xxs_to_q8_0(block, i, output + i * 32, &scale_fp16);
                if (scales)
                    scales[i] = fp16_to_fp32(scale_fp16);
                if (mins)
                    mins[i] = 0.0f;
            }
#endif
        }

        /**
         * @brief Unpack entire IQ3_XXS super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks.
         *
         * @param block Source IQ3_XXS super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins (always 0 for IQ3_XXS)
         */
        inline void unpack_iq3_xxs_superblock_to_int8(
            const IQ3_XXSBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_iq3_xxs_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }

            for (int i = 0; i < 8; ++i)
            {
                uint16_t scale_fp16;
                decode_iq3xxs_to_q8_0(block, i, output + i * 32, &scale_fp16);
                if (scales)
                    scales[i] = fp16_to_fp32(scale_fp16);
                if (mins)
                    mins[i] = 0.0f;
            }
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
         * @brief AVX2 decode IQ2_S sub-block to FP32
         */
        inline void decode_iq2s_subblock_to_fp32_avx2(
            const IQ2_SBlock &block,
            size_t subblock_idx,
            float *output)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint8_t *signs = block.qs + 32 + subblock_idx * 4;
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t scale_byte = block.scales[subblock_idx];

            float db[2];
            db[0] = d * (0.5f + (scale_byte & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scale_byte >> 4)) * 0.25f;

            __m256 vdb0 = _mm256_set1_ps(db[0]);
            __m256 vdb1 = _mm256_set1_ps(db[1]);

            // Compute grid indices
            int indices[4];
            for (int l = 0; l < 4; ++l)
            {
                indices[l] = qs[l] | ((qh_byte << (8 - 2 * l)) & 0x300);
            }
            __m128i vindices = _mm_loadu_si128((const __m128i *)indices);

            // Gather 4 grid values (64-bit each)
            __m256i vgrid_64 = _mm256_i32gather_epi64((const long long int *)iq2s_grid, vindices, 8);

            // Helper to create sign mask from byte
            auto get_sign_mask = [](uint8_t s) -> __m256
            {
                __m256i vs = _mm256_set1_epi32(s);
                __m256i vmask = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
                __m256i vcmp = _mm256_cmpeq_epi32(_mm256_and_si256(vs, vmask), vmask);
                return _mm256_castsi256_ps(_mm256_and_si256(vcmp, _mm256_set1_epi32(0x80000000)));
            };

            // Process first 16 elements (l=0,1)
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid_64);
            __m256i vints0 = _mm256_cvtepi8_epi32(vgrid_lo);
            __m256i vints1 = _mm256_cvtepi8_epi32(_mm_srli_si128(vgrid_lo, 8));

            __m256 vfloats0 = _mm256_cvtepi32_ps(vints0);
            __m256 vfloats1 = _mm256_cvtepi32_ps(vints1);

            vfloats0 = _mm256_xor_ps(vfloats0, get_sign_mask(signs[0]));
            vfloats1 = _mm256_xor_ps(vfloats1, get_sign_mask(signs[1]));

            vfloats0 = _mm256_mul_ps(vfloats0, vdb0);
            vfloats1 = _mm256_mul_ps(vfloats1, vdb0);

            _mm256_store_ps(output, vfloats0);
            _mm256_store_ps(output + 8, vfloats1);

            // Process next 16 elements (l=2,3)
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid_64, 1);
            __m256i vints2 = _mm256_cvtepi8_epi32(vgrid_hi);
            __m256i vints3 = _mm256_cvtepi8_epi32(_mm_srli_si128(vgrid_hi, 8));

            __m256 vfloats2 = _mm256_cvtepi32_ps(vints2);
            __m256 vfloats3 = _mm256_cvtepi32_ps(vints3);

            vfloats2 = _mm256_xor_ps(vfloats2, get_sign_mask(signs[2]));
            vfloats3 = _mm256_xor_ps(vfloats3, get_sign_mask(signs[3]));

            vfloats2 = _mm256_mul_ps(vfloats2, vdb1);
            vfloats3 = _mm256_mul_ps(vfloats3, vdb1);

            _mm256_store_ps(output + 16, vfloats2);
            _mm256_store_ps(output + 24, vfloats3);
        }

        /**
         * @brief AVX2 decode IQ2_S to Q8_0
         */
        inline void decode_iq2s_to_q8_0_avx2(
            const IQ2_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            alignas(32) float fp32_buffer[32];
            decode_iq2s_subblock_to_fp32_avx2(block, subblock_idx, fp32_buffer);
            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

#ifdef __AVX512F__
        /**
         * @brief AVX-512 decode IQ2_S sub-block to FP32
         */
        inline void decode_iq2s_subblock_to_fp32_avx512(
            const IQ2_SBlock &block,
            size_t subblock_idx,
            float *output)
        {
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint8_t *signs = block.qs + 32 + subblock_idx * 4;
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t scale_byte = block.scales[subblock_idx];

            float db[2];
            db[0] = d * (0.5f + (scale_byte & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scale_byte >> 4)) * 0.25f;

            __m512 vdb0 = _mm512_set1_ps(db[0]);
            __m512 vdb1 = _mm512_set1_ps(db[1]);

            // Compute grid indices
            int indices[4];
            for (int l = 0; l < 4; ++l)
            {
                indices[l] = qs[l] | ((qh_byte << (8 - 2 * l)) & 0x300);
            }
            __m128i vindices = _mm_loadu_si128((const __m128i *)indices);

            // Gather 4 grid values (64-bit each) -> 256 bits
            __m256i vgrid_64 = _mm256_i32gather_epi64((const long long int *)iq2s_grid, vindices, 8);

            // Helper for signs
            auto get_sign_mask_512 = [](uint8_t s0, uint8_t s1) -> __m512
            {
                __m512i vs0 = _mm512_set1_epi32(s0);
                __m512i vs1 = _mm512_set1_epi32(s1);
                // Blend: low 8 from vs0, high 8 from vs1. Mask 0xFF00 (high 8 bits set).
                __m512i vs = _mm512_mask_blend_epi32(0xFF00, vs0, vs1);

                __m512i vmask = _mm512_setr_epi32(
                    1, 2, 4, 8, 16, 32, 64, 128,
                    1, 2, 4, 8, 16, 32, 64, 128);

                __mmask16 k = _mm512_cmpeq_epi32_mask(_mm512_and_si512(vs, vmask), vmask);
                return _mm512_mask_blend_ps(k, _mm512_setzero_ps(), _mm512_set1_ps(-0.0f));
            };

            // Process first 16 elements (l=0,1)
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid_64); // g1, g0
            __m512i vints0 = _mm512_cvtepi8_epi32(vgrid_lo);
            __m512 vfloats0 = _mm512_cvtepi32_ps(vints0);

            __m512 vsign0 = get_sign_mask_512(signs[0], signs[1]);
            vfloats0 = _mm512_xor_ps(vfloats0, vsign0);
            vfloats0 = _mm512_mul_ps(vfloats0, vdb0);
            _mm512_store_ps(output, vfloats0);

            // Process next 16 elements (l=2,3)
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid_64, 1); // g3, g2
            __m512i vints1 = _mm512_cvtepi8_epi32(vgrid_hi);
            __m512 vfloats1 = _mm512_cvtepi32_ps(vints1);

            __m512 vsign1 = get_sign_mask_512(signs[2], signs[3]);
            vfloats1 = _mm512_xor_ps(vfloats1, vsign1);
            vfloats1 = _mm512_mul_ps(vfloats1, vdb1);
            _mm512_store_ps(output + 16, vfloats1);
        }

        /**
         * @brief AVX-512 decode IQ2_S to Q8_0 (Fused Transcode)
         *
         * Optimized implementation that avoids intermediate FP32 buffer storage.
         * Directly computes scale from integer grid values and transcodes.
         */
        inline void decode_iq2s_to_q8_0_avx512(
            const IQ2_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
#if defined(__AVX512F__)
            const float d = fp16_to_fp32(block.d);
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint8_t *signs = block.qs + 32 + subblock_idx * 4;
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t scale_byte = block.scales[subblock_idx];

            float db[2];
            db[0] = d * (0.5f + (scale_byte & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (scale_byte >> 4)) * 0.25f;

            // Vectorized index computation
            // Load 4 bytes of qs -> 4 ints
            uint32_t qs_u32 = *(const uint32_t *)qs;
            __m128i vqs = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(qs_u32));

            // Broadcast qh_byte and shift
            __m128i vqh = _mm_set1_epi32(qh_byte);
            __m128i vshifts = _mm_setr_epi32(8, 6, 4, 2);
            __m128i vqh_shifted = _mm_sllv_epi32(vqh, vshifts);
            __m128i vqh_masked = _mm_and_si128(vqh_shifted, _mm_set1_epi32(0x300));

            __m128i vindices = _mm_or_si128(vqs, vqh_masked);

            // Gather 4 grid values (64-bit each) -> 256 bits
            // Each 64-bit value contains 8 bytes (grid values)
            __m256i vgrid_64 = _mm256_i32gather_epi64((const long long int *)iq2s_grid, vindices, 8);

            // Find max grid value for first 16 and next 16
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid_64);
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid_64, 1);

            uint8_t max_grid_0 = hmax_epu8_128(vgrid_lo);
            uint8_t max_grid_1 = hmax_epu8_128(vgrid_hi);

            float max_val_0 = db[0] * max_grid_0;
            float max_val_1 = db[1] * max_grid_1;
            float max_abs = std::max(max_val_0, max_val_1);

            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                *q8_scale = 0;
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), _mm256_setzero_si256());
                return;
            }

            float scale = max_abs / 127.0f;
            if (scale > 65504.0f)
                scale = 65504.0f;
            *q8_scale = fp32_to_fp16(scale);

            float inv_scale = 1.0f / scale;
            // Compute integer ratios (Q16 fixed point)
            // ratio = db * inv_scale = db * 127 / max_abs
            // ratio_int = ratio * 65536
            int32_t ratio0 = (int32_t)(db[0] * inv_scale * 65536.0f + 0.5f);
            int32_t ratio1 = (int32_t)(db[1] * inv_scale * 65536.0f + 0.5f);

            __m512i vratio0 = _mm512_set1_epi32(ratio0);
            __m512i vratio1 = _mm512_set1_epi32(ratio1);

            // Process first 16 elements
            // Unpack 8-bit grid to 32-bit int
            __m512i vints0 = _mm512_cvtepu8_epi32(vgrid_lo);
            // Multiply by ratio (Q16)
            __m512i vres0 = _mm512_mullo_epi32(vints0, vratio0);
            // Shift right by 16 to get integer result
            vres0 = _mm512_srli_epi32(vres0, 16);

            // Apply signs
            __mmask16 k0 = signs[0] | (signs[1] << 8);
            // If sign bit set, negate: 0 - val
            vres0 = _mm512_mask_sub_epi32(vres0, k0, _mm512_setzero_si512(), vres0);

            // Pack to int8
            __m128i vq8_0 = _mm512_cvtepi32_epi8(vres0);
            _mm_storeu_si128((__m128i *)q8_qs, vq8_0);

            // Process next 16 elements
            __m512i vints1 = _mm512_cvtepu8_epi32(vgrid_hi);
            __m512i vres1 = _mm512_mullo_epi32(vints1, vratio1);
            vres1 = _mm512_srli_epi32(vres1, 16);

            __mmask16 k1 = signs[2] | (signs[3] << 8);
            vres1 = _mm512_mask_sub_epi32(vres1, k1, _mm512_setzero_si512(), vres1);

            __m128i vq8_1 = _mm512_cvtepi32_epi8(vres1);
            _mm_storeu_si128((__m128i *)(q8_qs + 16), vq8_1);
#else
            alignas(64) float fp32_buffer[32];
            decode_iq2s_subblock_to_fp32_avx512(block, subblock_idx, fp32_buffer);
            quantize_fp32_to_q8_0_avx512(fp32_buffer, 32, q8_qs, q8_scale);
#endif
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
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                decode_iq2s_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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

        inline void unpack_iq2_s_superblock_to_int8_avx512(
            const IQ2_SBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
            const float d = fp16_to_fp32(block.d);

            // Precompute scales
            float db_arr[16];
            for (int i = 0; i < 8; ++i)
            {
                uint8_t s = block.scales[i];
                db_arr[2 * i + 0] = d * (0.5f + (s & 0xf)) * 0.25f;
                db_arr[2 * i + 1] = d * (0.5f + (s >> 4)) * 0.25f;
            }

            const uint8_t *qs_ptr = block.qs;
            const uint8_t *signs_ptr = block.qs + 32;
            const uint8_t *qh_ptr = block.qh;

            // Process 2 subblocks (64 elements) per iteration
            for (int ib = 0; ib < 8; ib += 2)
            {
                // 1. Construct indices
                // Load 8 bytes of qs (2 subblocks * 4 bytes/subblock)
                __m128i vqs_128 = _mm_loadl_epi64((const __m128i *)(qs_ptr + ib * 4));
                __m256i vqs = _mm256_cvtepu8_epi32(vqs_128);

                // Load qh bytes
                uint8_t qh0 = qh_ptr[ib];
                uint8_t qh1 = qh_ptr[ib + 1];
                // Construct qh vector: [qh1, qh1, qh1, qh1, qh0, qh0, qh0, qh0]
                __m256i vqh = _mm256_set_epi32(qh1, qh1, qh1, qh1, qh0, qh0, qh0, qh0);

                // Shifts: 8, 6, 4, 2 repeated
                __m256i vshifts = _mm256_setr_epi32(8, 6, 4, 2, 8, 6, 4, 2);
                __m256i vqh_shifted = _mm256_sllv_epi32(vqh, vshifts);
                __m256i vqh_masked = _mm256_and_si256(vqh_shifted, _mm256_set1_epi32(0x300));

                __m256i vindices = _mm256_or_si256(vqs, vqh_masked);

                // 2. Gather grid values
                __m512i vgrid = _mm512_i32gather_epi64(vindices, iq2s_grid, 8);

                // 3. Process 4 chunks of 16 elements
                // Chunk 0: Subblock ib, first 16
                __m128i vchunk0 = _mm512_castsi512_si128(vgrid);
                __m512 vfp0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vchunk0));
                vfp0 = _mm512_mul_ps(vfp0, _mm512_set1_ps(db_arr[2 * ib]));

                uint16_t s0 = *(const uint16_t *)(signs_ptr + ib * 4);
                vfp0 = _mm512_mask_sub_ps(vfp0, s0, _mm512_setzero_ps(), vfp0);

                // Chunk 1: Subblock ib, next 16
                __m128i vchunk1 = _mm512_extracti32x4_epi32(vgrid, 1);
                __m512 vfp1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vchunk1));
                vfp1 = _mm512_mul_ps(vfp1, _mm512_set1_ps(db_arr[2 * ib + 1]));

                uint16_t s1 = *(const uint16_t *)(signs_ptr + ib * 4 + 2);
                vfp1 = _mm512_mask_sub_ps(vfp1, s1, _mm512_setzero_ps(), vfp1);

                // Quantize Block 1 (ib)
                {
                    __m512 abs0 = _mm512_abs_ps(vfp0);
                    __m512 abs1 = _mm512_abs_ps(vfp1);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib] = scale;
                    if (mins)
                        mins[ib] = 0.0f;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + ib * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512 vq0 = _mm512_mul_ps(vfp0, vinv);
                        __m512i vi0 = _mm512_cvtps_epi32(vq0);

                        __m512 vq1 = _mm512_mul_ps(vfp1, vinv);
                        __m512i vi1 = _mm512_cvtps_epi32(vq1);

                        __m128i vout0 = _mm512_cvtepi32_epi8(vi0);
                        __m128i vout1 = _mm512_cvtepi32_epi8(vi1);

                        _mm_storeu_si128((__m128i *)(output + ib * 32), vout0);
                        _mm_storeu_si128((__m128i *)(output + ib * 32 + 16), vout1);
                    }
                }

                // Chunk 2: Subblock ib+1, first 16
                __m128i vchunk2 = _mm512_extracti32x4_epi32(vgrid, 2);
                __m512 vfp2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vchunk2));
                vfp2 = _mm512_mul_ps(vfp2, _mm512_set1_ps(db_arr[2 * (ib + 1)]));

                uint16_t s2 = *(const uint16_t *)(signs_ptr + (ib + 1) * 4);
                vfp2 = _mm512_mask_sub_ps(vfp2, s2, _mm512_setzero_ps(), vfp2);

                // Chunk 3: Subblock ib+1, next 16
                __m128i vchunk3 = _mm512_extracti32x4_epi32(vgrid, 3);
                __m512 vfp3 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vchunk3));
                vfp3 = _mm512_mul_ps(vfp3, _mm512_set1_ps(db_arr[2 * (ib + 1) + 1]));

                uint16_t s3 = *(const uint16_t *)(signs_ptr + (ib + 1) * 4 + 2);
                vfp3 = _mm512_mask_sub_ps(vfp3, s3, _mm512_setzero_ps(), vfp3);

                // Quantize Block 2 (ib+1)
                {
                    __m512 abs2 = _mm512_abs_ps(vfp2);
                    __m512 abs3 = _mm512_abs_ps(vfp3);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs2, abs3));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib + 1] = scale;
                    if (mins)
                        mins[ib + 1] = 0.0f;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + (ib + 1) * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512 vq2 = _mm512_mul_ps(vfp2, vinv);
                        __m512i vi2 = _mm512_cvtps_epi32(vq2);

                        __m512 vq3 = _mm512_mul_ps(vfp3, vinv);
                        __m512i vi3 = _mm512_cvtps_epi32(vq3);

                        __m128i vout2 = _mm512_cvtepi32_epi8(vi2);
                        __m128i vout3 = _mm512_cvtepi32_epi8(vi3);

                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32), vout2);
                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32 + 16), vout3);
                    }
                }
            }
#else
            unpack_iq2_s_superblock_to_int8(block, output, scales, mins);
#endif
        }

        /**
         * @brief Unpack entire IQ2_S super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks.
         *
         * @param block Source IQ2_S super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins (always 0 for IQ2_S)
         */
        inline void unpack_iq2_s_superblock_to_int8(
            const IQ2_SBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_iq2_s_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }

            for (int i = 0; i < 8; ++i)
            {
                uint16_t scale_fp16;
                decode_iq2s_to_q8_0(block, i, output + i * 32, &scale_fp16);
                if (scales)
                    scales[i] = fp16_to_fp32(scale_fp16);
                if (mins)
                    mins[i] = 0.0f;
            }
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
        inline void decode_iq3s_subblock_to_fp32_scalar(
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
            decode_iq3s_subblock_to_fp32_scalar(block, subblock_idx, fp32_buffer);

            // Quantize FP32 → Q8_0
            quantize_fp32_to_q8_0_scalar(fp32_buffer, 32, q8_qs, q8_scale);
        }

#if defined(__AVX2__)
        /**
         * @brief AVX2 decode IQ3_S sub-block to FP32
         */
        inline void decode_iq3s_subblock_to_fp32_avx2(
            const IQ3_SBlock &block,
            size_t subblock_idx,
            float *output)
        {
            const float d = fp16_to_fp32(block.d);
            const size_t pair_idx = subblock_idx / 2;
            const uint8_t scale_byte = block.scales[pair_idx];

            // Calculate scales
            const float db_val = (subblock_idx % 2 == 0)
                                     ? d * (1.0f + 2.0f * (scale_byte & 0xf))
                                     : d * (1.0f + 2.0f * (scale_byte >> 4));
            const __m256 vdb = _mm256_set1_ps(db_val);

            const uint8_t *qs = block.qs + subblock_idx * 8;
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t *signs = block.signs + subblock_idx * 4;

            const __m256 vone = _mm256_set1_ps(1.0f);
            const __m256 vminus_one = _mm256_set1_ps(-1.0f);
            const __m256i vmask_bits = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

            // Load qs (8 bytes)
            __m128i vqs_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qs));
            __m256i vqs = _mm256_cvtepu8_epi32(vqs_8); // Expand to 8 32-bit ints (zero-extended)

            // Compute indices
            // idx[k] = qs[k] | ((qh >> k) & 1) << 8
            __m256i vqh = _mm256_set1_epi32(qh_byte);
            __m256i vqh_bits = _mm256_and_si256(vqh, vmask_bits);
            __m256i vqh_mask = _mm256_cmpgt_epi32(vqh_bits, _mm256_setzero_si256());
            __m256i vadd = _mm256_and_si256(vqh_mask, _mm256_set1_epi32(256));
            __m256i vidx = _mm256_add_epi32(vqs, vadd);

            // Gather grid values (8 uint32_t values)
            __m256i vgrid_packed = _mm256_i32gather_epi32(reinterpret_cast<const int *>(iq3s_grid), vidx, 4);

            // Split into low and high 128 bits
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid_packed);
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid_packed, 1);

            for (size_t l = 0; l < 4; ++l)
            {
                __m128i vpacked;
                if (l == 0)
                    vpacked = vgrid_lo;
                else if (l == 1)
                    vpacked = _mm_srli_si128(vgrid_lo, 8);
                else if (l == 2)
                    vpacked = vgrid_hi;
                else
                    vpacked = _mm_srli_si128(vgrid_hi, 8);

                // Convert 8 bytes to 8 int32s then to floats
                __m256i vgrid_i32 = _mm256_cvtepi8_epi32(vpacked);
                __m256 vgrid = _mm256_cvtepi32_ps(vgrid_i32);

                // Signs
                __m256i vsign_byte = _mm256_set1_epi32(signs[l]);
                __m256i vsign_bits = _mm256_and_si256(vsign_byte, vmask_bits);
                __m256i vsign_mask = _mm256_cmpgt_epi32(vsign_bits, _mm256_setzero_si256());
                __m256 vsign = _mm256_blendv_ps(vone, vminus_one, _mm256_castsi256_ps(vsign_mask));

                // Result
                __m256 vresult = _mm256_mul_ps(vdb, _mm256_mul_ps(vgrid, vsign));

                _mm256_storeu_ps(output + l * 8, vresult);
            }
        }

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
            decode_iq3s_subblock_to_fp32_avx2(block, subblock_idx, fp32_buffer);

            quantize_fp32_to_q8_0_avx2(fp32_buffer, 32, q8_qs, q8_scale);
        }
#endif

#if defined(__AVX512F__)
        /**
         * @brief AVX-512 decode IQ3_S sub-block to FP32
         */
        inline void decode_iq3s_subblock_to_fp32_avx512(
            const IQ3_SBlock &block,
            size_t subblock_idx,
            float *output)
        {
            const float d = fp16_to_fp32(block.d);
            const size_t pair_idx = subblock_idx / 2;
            const uint8_t scale_byte = block.scales[pair_idx];

            // Calculate scale
            const float db_val = (subblock_idx % 2 == 0)
                                     ? d * (1.0f + 2.0f * (scale_byte & 0xf))
                                     : d * (1.0f + 2.0f * (scale_byte >> 4));
            const __m512 vdb = _mm512_set1_ps(db_val);

            const uint8_t *qs = block.qs + subblock_idx * 8;
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t *signs = block.signs + subblock_idx * 4;

            // Load qs (8 bytes) -> 8 int32s
            __m128i vqs_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qs));
            __m256i vqs = _mm256_cvtepu8_epi32(vqs_128);

            // Compute indices: idx[k] = qs[k] | ((qh >> k) & 1) << 8
            // We use a shift vector to extract bits from qh
            __m256i vqh = _mm256_set1_epi32(qh_byte);
            __m256i vshift = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i vqh_shifted = _mm256_srlv_epi32(vqh, vshift);
            __m256i vqh_bit = _mm256_and_si256(vqh_shifted, _mm256_set1_epi32(1));
            __m256i vadd = _mm256_slli_epi32(vqh_bit, 8);
            __m256i vidx = _mm256_add_epi32(vqs, vadd);

            // Gather grid values (8 uint32_t values = 32 bytes)
            __m256i vgrid = _mm256_i32gather_epi32(reinterpret_cast<const int *>(iq3s_grid), vidx, 4);

            // Split into low and high 16 bytes
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid);
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid, 1);

            // Convert to floats (16 elements each)
            __m512 vfloats0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_lo));
            __m512 vfloats1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_hi));

            // Apply signs
            uint32_t sign_mask_bits;
            std::memcpy(&sign_mask_bits, signs, 4);
            __mmask16 m0 = sign_mask_bits & 0xFFFF;
            __mmask16 m1 = (sign_mask_bits >> 16) & 0xFFFF;

            // Negate where mask is set: 0 - val
            vfloats0 = _mm512_mask_sub_ps(vfloats0, m0, _mm512_setzero_ps(), vfloats0);
            vfloats1 = _mm512_mask_sub_ps(vfloats1, m1, _mm512_setzero_ps(), vfloats1);

            // Apply scale
            vfloats0 = _mm512_mul_ps(vfloats0, vdb);
            vfloats1 = _mm512_mul_ps(vfloats1, vdb);

            // Store
            _mm512_storeu_ps(output, vfloats0);
            _mm512_storeu_ps(output + 16, vfloats1);
        }

        /**
         * @brief AVX-512 decode IQ3_S to Q8_0 (Optimized)
         */
        __attribute__((always_inline)) inline void decode_iq3s_to_q8_0_avx512(
            const IQ3_SBlock &block,
            size_t subblock_idx,
            int8_t *q8_qs,
            uint16_t *q8_scale)
        {
            const float d = fp16_to_fp32(block.d);
            const size_t pair_idx = subblock_idx / 2;
            const uint8_t scale_byte = block.scales[pair_idx];

            // Calculate scale
            const float db_val = (subblock_idx % 2 == 0)
                                     ? d * (1.0f + 2.0f * (scale_byte & 0xf))
                                     : d * (1.0f + 2.0f * (scale_byte >> 4));
            const __m512 vdb = _mm512_set1_ps(db_val);

            const uint8_t *qs = block.qs + subblock_idx * 8;
            const uint8_t qh_byte = block.qh[subblock_idx];
            const uint8_t *signs = block.signs + subblock_idx * 4;

            // Load qs (8 bytes) -> 8 int32s
            __m128i vqs_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qs));
            __m256i vqs = _mm256_cvtepu8_epi32(vqs_128);

            // Compute indices: idx[k] = qs[k] | ((qh >> k) & 1) << 8
            __m256i vqh = _mm256_set1_epi32(qh_byte);
            __m256i vshift = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i vqh_shifted = _mm256_srlv_epi32(vqh, vshift);
            __m256i vqh_bit = _mm256_and_si256(vqh_shifted, _mm256_set1_epi32(1));
            __m256i vadd = _mm256_slli_epi32(vqh_bit, 8);
            __m256i vidx = _mm256_add_epi32(vqs, vadd);

            // Gather grid values (8 uint32_t values = 32 bytes)
            __m256i vgrid = _mm256_i32gather_epi32(reinterpret_cast<const int *>(iq3s_grid), vidx, 4);

            // Split into low and high 16 bytes
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid);
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid, 1);

            // Convert to floats (16 elements each)
            __m512 vfloats0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_lo));
            __m512 vfloats1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_hi));

            // Apply signs
            uint32_t sign_mask_bits;
            std::memcpy(&sign_mask_bits, signs, 4);
            __mmask16 m0 = sign_mask_bits & 0xFFFF;
            __mmask16 m1 = (sign_mask_bits >> 16) & 0xFFFF;

            // Negate where mask is set: 0 - val
            vfloats0 = _mm512_mask_sub_ps(vfloats0, m0, _mm512_setzero_ps(), vfloats0);
            vfloats1 = _mm512_mask_sub_ps(vfloats1, m1, _mm512_setzero_ps(), vfloats1);

            // Apply scale
            vfloats0 = _mm512_mul_ps(vfloats0, vdb);
            vfloats1 = _mm512_mul_ps(vfloats1, vdb);

            // --- Quantization Logic ---

            // Find max absolute value
            __m512 vmax_abs = _mm512_max_ps(_mm512_abs_ps(vfloats0), _mm512_abs_ps(vfloats1));
            float max_val = _mm512_reduce_max_ps(vmax_abs);

            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            if (max_val < MIN_SCALE_THRESHOLD)
            {
                *q8_scale = 0;
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), _mm256_setzero_si256());
                return;
            }

            float scale = max_val / 127.0f;
            if (scale > 65504.0f)
                scale = 65504.0f;
            *q8_scale = fp32_to_fp16(scale);
            float inv_scale = 1.0f / scale;

            __m512 vinv_scale = _mm512_set1_ps(inv_scale);

            // Quantize
            __m512i vqi_lo = _mm512_cvtps_epi32(_mm512_mul_ps(vfloats0, vinv_scale));
            __m512i vqi_hi = _mm512_cvtps_epi32(_mm512_mul_ps(vfloats1, vinv_scale));

            // Clamp
            __m512i vmin = _mm512_set1_epi32(-127);
            __m512i vmax = _mm512_set1_epi32(127);
            vqi_lo = _mm512_max_epi32(vqi_lo, vmin);
            vqi_lo = _mm512_min_epi32(vqi_lo, vmax);
            vqi_hi = _mm512_max_epi32(vqi_hi, vmin);
            vqi_hi = _mm512_min_epi32(vqi_hi, vmax);

            // Pack
            __m256i vq8 = _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm512_cvtepi32_epi8(vqi_lo)),
                _mm512_cvtepi32_epi8(vqi_hi), 1);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), vq8);
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
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                decode_iq3s_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
            {
                decode_iq3s_to_q8_0_avx2(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

            decode_iq3s_to_q8_0_scalar(block, subblock_idx, q8_qs, q8_scale);
        }

        /**
         * @brief AVX-512 optimized IQ3_S superblock unpack (256 elements)
         *
         * Processes 2 subblocks (64 elements) per iteration for better throughput.
         */
        inline void unpack_iq3_s_superblock_to_int8_avx512(
            const IQ3_SBlock &block,
            int8_t *output,
            float *scales)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            const float d = fp16_to_fp32(block.d);

            // Precompute all 8 scale values
            float db_arr[8];
            for (int i = 0; i < 4; ++i)
            {
                uint8_t scale_byte = block.scales[i];
                db_arr[2 * i + 0] = d * (1.0f + 2.0f * (scale_byte & 0xf));
                db_arr[2 * i + 1] = d * (1.0f + 2.0f * (scale_byte >> 4));
            }

            // Process 2 subblocks (64 elements) per iteration
            for (int ib = 0; ib < 8; ib += 2)
            {
                // ===== Load data for 2 subblocks =====
                // Load 16 bytes of qs (8 per subblock)
                const uint8_t *qs = block.qs + ib * 8;
                __m128i vqs_16 = _mm_loadu_si128((const __m128i *)qs);
                __m512i vqs_32 = _mm512_cvtepu8_epi32(vqs_16);

                // Load qh bytes for both subblocks
                uint8_t qh0 = block.qh[ib];
                uint8_t qh1 = block.qh[ib + 1];

                // Build qh vector: [qh1 x8, qh0 x8]
                __m512i vqh = _mm512_set_epi32(
                    qh1, qh1, qh1, qh1, qh1, qh1, qh1, qh1,
                    qh0, qh0, qh0, qh0, qh0, qh0, qh0, qh0);

                // Shifts: 0,1,2,3,4,5,6,7 repeated
                __m512i vshift = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
                __m512i vqh_shifted = _mm512_srlv_epi32(vqh, vshift);
                __m512i vqh_bit = _mm512_and_si512(vqh_shifted, _mm512_set1_epi32(1));
                __m512i vadd = _mm512_slli_epi32(vqh_bit, 8);
                __m512i vidx = _mm512_add_epi32(vqs_32, vadd);

                // Gather 16 grid values (uint32_t each = 4 bytes)
                __m512i vgrid = _mm512_i32gather_epi32(vidx, (const int *)iq3s_grid, 4);

                // Load signs for both subblocks (4 bytes each = 8 bytes total)
                const uint8_t *signs = block.signs + ib * 4;
                uint32_t sign0, sign1;
                std::memcpy(&sign0, signs, 4);
                std::memcpy(&sign1, signs + 4, 4);

                // ===== Process subblock ib (first 32 elements) =====
                __m128i vgrid_0 = _mm512_castsi512_si128(vgrid);
                __m128i vgrid_1 = _mm512_extracti32x4_epi32(vgrid, 1);

                // Convert grid (int8 values) to FP32
                __m512 vfp0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_0));
                __m512 vfp1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_1));

                // Apply scale
                vfp0 = _mm512_mul_ps(vfp0, _mm512_set1_ps(db_arr[ib]));
                vfp1 = _mm512_mul_ps(vfp1, _mm512_set1_ps(db_arr[ib]));

                // Apply signs
                __mmask16 m0 = sign0 & 0xFFFF;
                __mmask16 m1 = (sign0 >> 16) & 0xFFFF;
                vfp0 = _mm512_mask_sub_ps(vfp0, m0, _mm512_setzero_ps(), vfp0);
                vfp1 = _mm512_mask_sub_ps(vfp1, m1, _mm512_setzero_ps(), vfp1);

                // Quantize subblock ib
                {
                    __m512 abs0 = _mm512_abs_ps(vfp0);
                    __m512 abs1 = _mm512_abs_ps(vfp1);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib] = scale;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + ib * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512i vi0 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp0, vinv));
                        __m512i vi1 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp1, vinv));

                        __m128i vout0 = _mm512_cvtepi32_epi8(vi0);
                        __m128i vout1 = _mm512_cvtepi32_epi8(vi1);

                        _mm_storeu_si128((__m128i *)(output + ib * 32), vout0);
                        _mm_storeu_si128((__m128i *)(output + ib * 32 + 16), vout1);
                    }
                }

                // ===== Process subblock ib+1 (next 32 elements) =====
                __m128i vgrid_2 = _mm512_extracti32x4_epi32(vgrid, 2);
                __m128i vgrid_3 = _mm512_extracti32x4_epi32(vgrid, 3);

                // Convert grid (int8 values) to FP32
                __m512 vfp2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_2));
                __m512 vfp3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_3));

                // Apply scale
                vfp2 = _mm512_mul_ps(vfp2, _mm512_set1_ps(db_arr[ib + 1]));
                vfp3 = _mm512_mul_ps(vfp3, _mm512_set1_ps(db_arr[ib + 1]));

                // Apply signs
                __mmask16 m2 = sign1 & 0xFFFF;
                __mmask16 m3 = (sign1 >> 16) & 0xFFFF;
                vfp2 = _mm512_mask_sub_ps(vfp2, m2, _mm512_setzero_ps(), vfp2);
                vfp3 = _mm512_mask_sub_ps(vfp3, m3, _mm512_setzero_ps(), vfp3);

                // Quantize subblock ib+1
                {
                    __m512 abs2 = _mm512_abs_ps(vfp2);
                    __m512 abs3 = _mm512_abs_ps(vfp3);
                    float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs2, abs3));

                    float scale = max_abs / 127.0f;
                    if (scale > 65504.0f)
                        scale = 65504.0f;
                    if (max_abs < 1e-6f)
                        scale = 0.0f;

                    if (scales)
                        scales[ib + 1] = scale;

                    if (scale == 0.0f)
                    {
                        _mm256_storeu_si256((__m256i *)(output + (ib + 1) * 32), _mm256_setzero_si256());
                    }
                    else
                    {
                        float inv_scale = 1.0f / scale;
                        __m512 vinv = _mm512_set1_ps(inv_scale);

                        __m512i vi2 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp2, vinv));
                        __m512i vi3 = _mm512_cvtps_epi32(_mm512_mul_ps(vfp3, vinv));

                        __m128i vout2 = _mm512_cvtepi32_epi8(vi2);
                        __m128i vout3 = _mm512_cvtepi32_epi8(vi3);

                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32), vout2);
                        _mm_storeu_si128((__m128i *)(output + (ib + 1) * 32 + 16), vout3);
                    }
                }
            }
#else
            // Fallback to per-subblock
            for (int i = 0; i < 8; ++i)
            {
                uint16_t s;
                decode_iq3s_to_q8_0(block, i, output + i * 32, &s);
                if (scales)
                    scales[i] = fp16_to_fp32(s);
            }
#endif
        }

        /**
         * @brief Unpack entire IQ3_S super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks with single CPU dispatch.
         *
         * @param block Source IQ3_S super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         */
        inline void unpack_iq3_s_superblock_to_int8(
            const IQ3_SBlock &block,
            int8_t *output,
            float *scales)
        {
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_iq3_s_superblock_to_int8_avx512(block, output, scales);
                return;
            }

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
            {
                for (int i = 0; i < 8; ++i)
                {
                    uint16_t s;
                    decode_iq3s_to_q8_0_avx2(block, i, output + i * 32, &s);
                    if (scales)
                        scales[i] = fp16_to_fp32(s);
                }
                return;
            }
#endif

            // Scalar fallback
            for (int i = 0; i < 8; ++i)
            {
                uint16_t s;
                decode_iq3s_to_q8_0_scalar(block, i, output + i * 32, &s);
                if (scales)
                    scales[i] = fp16_to_fp32(s);
            }
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
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                decode_iq1s_to_q8_0_avx512(block, subblock_idx, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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

        /**
         * @brief Unpack entire IQ1_S super-block to int8 (256 elements) - AVX512 Optimized
         */
        __attribute__((always_inline)) inline void unpack_iq1_s_superblock_to_int8_avx512(
            const IQ1_SBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__)
            const float d = fp16_to_fp32(block.d);
            const __m512i vmin_clamp = _mm512_set1_epi32(-127);
            const __m512i vmax_clamp = _mm512_set1_epi32(127);
            const __m512 v127 = _mm512_set1_ps(127.0f);
            const __m512 veps = _mm512_set1_ps(1e-9f);

            // Precompute dl and delta for all 8 subblocks
            alignas(64) float dl_arr[8];
            alignas(64) float delta_arr[8];

            for (int i = 0; i < 8; ++i)
            {
                uint16_t qh_val = block.qh[i];
                dl_arr[i] = d * (2.0f * ((qh_val >> 12) & 7) + 1.0f);
                delta_arr[i] = (qh_val & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
            }

            for (int ib = 0; ib < 8; ib += 2)
            {
                // Subblock ib
                const uint8_t *qs0 = block.qs + ib * 4;
                uint16_t qh0 = block.qh[ib];

                // Subblock ib+1
                const uint8_t *qs1 = block.qs + (ib + 1) * 4;
                uint16_t qh1 = block.qh[ib + 1];

                // Indices for ib
                alignas(16) int32_t idx0[4];
                idx0[0] = qs0[0] | (((qh0 >> 0) & 7) << 8);
                idx0[1] = qs0[1] | (((qh0 >> 3) & 7) << 8);
                idx0[2] = qs0[2] | (((qh0 >> 6) & 7) << 8);
                idx0[3] = qs0[3] | (((qh0 >> 9) & 7) << 8);

                // Indices for ib+1
                alignas(16) int32_t idx1[4];
                idx1[0] = qs1[0] | (((qh1 >> 0) & 7) << 8);
                idx1[1] = qs1[1] | (((qh1 >> 3) & 7) << 8);
                idx1[2] = qs1[2] | (((qh1 >> 6) & 7) << 8);
                idx1[3] = qs1[3] | (((qh1 >> 9) & 7) << 8);

                // Gather
                __m128i vidx0 = _mm_load_si128((const __m128i *)idx0);
                __m128i vidx1 = _mm_load_si128((const __m128i *)idx1);

                __m256i vgrid0 = _mm256_i32gather_epi64((const long long int *)iq1s_grid, vidx0, 8);
                __m256i vgrid1 = _mm256_i32gather_epi64((const long long int *)iq1s_grid, vidx1, 8);

                // Process ib
                {
                    __m128i vg_lo = _mm256_castsi256_si128(vgrid0);
                    __m128i vg_hi = _mm256_extracti128_si256(vgrid0, 1);

                    __m512 vf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vg_lo));
                    __m512 vf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vg_hi));

                    __m512 vdl = _mm512_set1_ps(dl_arr[ib]);
                    __m512 vdelta = _mm512_set1_ps(delta_arr[ib]);

                    // vals = dl * (grid + delta) = dl*grid + dl*delta
                    __m512 vdl_delta = _mm512_mul_ps(vdl, vdelta);
                    __m512 vals_lo = _mm512_fmadd_ps(vdl, vf_lo, vdl_delta);
                    __m512 vals_hi = _mm512_fmadd_ps(vdl, vf_hi, vdl_delta);

                    // Quantize
                    __m512 vabs_max = _mm512_max_ps(_mm512_abs_ps(vals_lo), _mm512_abs_ps(vals_hi));
                    float max_val = _mm512_reduce_max_ps(vabs_max);
                    float scale_out = max_val / 127.0f;
                    __m512 vinv = _mm512_div_ps(v127, _mm512_add_ps(_mm512_set1_ps(max_val), veps));

                    __m512i vqi_lo = _mm512_cvtps_epi32(_mm512_mul_ps(vals_lo, vinv));
                    __m512i vqi_hi = _mm512_cvtps_epi32(_mm512_mul_ps(vals_hi, vinv));
                    vqi_lo = _mm512_max_epi32(_mm512_min_epi32(vqi_lo, vmax_clamp), vmin_clamp);
                    vqi_hi = _mm512_max_epi32(_mm512_min_epi32(vqi_hi, vmax_clamp), vmin_clamp);

                    __m256i vq8 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm512_cvtepi32_epi8(vqi_lo)),
                        _mm512_cvtepi32_epi8(vqi_hi), 1);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + ib * 32), vq8);
                    if (scales)
                        scales[ib] = scale_out;
                }

                // Process ib+1
                {
                    __m128i vg_lo = _mm256_castsi256_si128(vgrid1);
                    __m128i vg_hi = _mm256_extracti128_si256(vgrid1, 1);

                    __m512 vf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vg_lo));
                    __m512 vf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vg_hi));

                    __m512 vdl = _mm512_set1_ps(dl_arr[ib + 1]);
                    __m512 vdelta = _mm512_set1_ps(delta_arr[ib + 1]);

                    __m512 vdl_delta = _mm512_mul_ps(vdl, vdelta);
                    __m512 vals_lo = _mm512_fmadd_ps(vdl, vf_lo, vdl_delta);
                    __m512 vals_hi = _mm512_fmadd_ps(vdl, vf_hi, vdl_delta);

                    __m512 vabs_max = _mm512_max_ps(_mm512_abs_ps(vals_lo), _mm512_abs_ps(vals_hi));
                    float max_val = _mm512_reduce_max_ps(vabs_max);
                    float scale_out = max_val / 127.0f;
                    __m512 vinv = _mm512_div_ps(v127, _mm512_add_ps(_mm512_set1_ps(max_val), veps));

                    __m512i vqi_lo = _mm512_cvtps_epi32(_mm512_mul_ps(vals_lo, vinv));
                    __m512i vqi_hi = _mm512_cvtps_epi32(_mm512_mul_ps(vals_hi, vinv));
                    vqi_lo = _mm512_max_epi32(_mm512_min_epi32(vqi_lo, vmax_clamp), vmin_clamp);
                    vqi_hi = _mm512_max_epi32(_mm512_min_epi32(vqi_hi, vmax_clamp), vmin_clamp);

                    __m256i vq8 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm512_cvtepi32_epi8(vqi_lo)),
                        _mm512_cvtepi32_epi8(vqi_hi), 1);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + (ib + 1) * 32), vq8);
                    if (scales)
                        scales[ib + 1] = scale_out;
                }
            }

            if (mins)
                _mm256_storeu_ps(mins, _mm256_setzero_ps());
#endif
        }

        /**
         * @brief Unpack entire IQ1_S super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks.
         *
         * @param block Source IQ1_S super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins (always 0 for IQ1_S)
         */
        inline void unpack_iq1_s_superblock_to_int8(
            const IQ1_SBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__)
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_iq1_s_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }
#endif
            for (int i = 0; i < 8; ++i)
            {
                uint16_t scale_fp16;
                decode_iq1s_to_q8_0(block, i, output + i * 32, &scale_fp16);
                if (scales)
                    scales[i] = fp16_to_fp32(scale_fp16);
                if (mins)
                    mins[i] = 0.0f;
            }
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
            // 1. Load scales
            const uint16_t *sc = reinterpret_cast<const uint16_t *>(block.scales);
            int shift = 6 * (subblock_idx % 2);
            uint16_t sc_val = sc[subblock_idx / 2];
            float s1 = (float)((sc_val >> shift) & 0x7);
            float s2 = (float)((sc_val >> (shift + 3)) & 0x7);

            float dl1 = global_scale * (2.0f * s1 + 1.0f);
            float dl2 = global_scale * (2.0f * s2 + 1.0f);

            // 2. Load qs and qh
            const uint8_t *qs = block.qs + subblock_idx * 4;
            const uint8_t *qh = block.qh + subblock_idx * 2;

            // 3. Compute indices
            int32_t idx[4];
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);

            __m128i vindices = _mm_loadu_si128((const __m128i *)idx);

            // 4. Gather grid values
            // iq1s_grid is uint64_t*
            __m256i vgrid = _mm256_i32gather_epi64((const long long int *)iq1s_grid, vindices, 8);

            // 5. Compute Deltas
            float delta[4];
            delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

            // 6. Process in two halves (16 elements each)
            __m128i vgrid_lo = _mm256_castsi256_si128(vgrid);
            __m128i vgrid_hi = _mm256_extracti128_si256(vgrid, 1);

            __m512 vgrid_lo_ps = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_lo));
            __m512 vgrid_hi_ps = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vgrid_hi));

            __m512 vdelta_lo = _mm512_set_ps(
                delta[1], delta[1], delta[1], delta[1], delta[1], delta[1], delta[1], delta[1],
                delta[0], delta[0], delta[0], delta[0], delta[0], delta[0], delta[0], delta[0]);

            __m512 vdelta_hi = _mm512_set_ps(
                delta[3], delta[3], delta[3], delta[3], delta[3], delta[3], delta[3], delta[3],
                delta[2], delta[2], delta[2], delta[2], delta[2], delta[2], delta[2], delta[2]);

            __m512 vdl1 = _mm512_set1_ps(dl1);
            __m512 vdl2 = _mm512_set1_ps(dl2);

            __m512 vals_lo = _mm512_mul_ps(vdl1, _mm512_add_ps(vgrid_lo_ps, vdelta_lo));
            __m512 vals_hi = _mm512_mul_ps(vdl2, _mm512_add_ps(vgrid_hi_ps, vdelta_hi));

            // 7. Find max absolute value for quantization
            __m512 vmax_abs = _mm512_max_ps(_mm512_abs_ps(vals_lo), _mm512_abs_ps(vals_hi));
            float max_val = _mm512_reduce_max_ps(vmax_abs);

            constexpr float MIN_SCALE_THRESHOLD = 1e-6f;
            if (max_val < MIN_SCALE_THRESHOLD)
            {
                *q8_scale = 0;
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), _mm256_setzero_si256());
                return;
            }

            float scale = max_val / 127.0f;
            if (scale > 65504.0f)
                scale = 65504.0f;
            *q8_scale = fp32_to_fp16(scale);
            float inv_scale = 1.0f / scale;

            __m512 vinv_scale = _mm512_set1_ps(inv_scale);

            // 8. Quantize
            __m512i vqi_lo = _mm512_cvtps_epi32(_mm512_mul_ps(vals_lo, vinv_scale));
            __m512i vqi_hi = _mm512_cvtps_epi32(_mm512_mul_ps(vals_hi, vinv_scale));

            // Clamp
            __m512i vmin = _mm512_set1_epi32(-127);
            __m512i vmax = _mm512_set1_epi32(127);
            vqi_lo = _mm512_max_epi32(vqi_lo, vmin);
            vqi_lo = _mm512_min_epi32(vqi_lo, vmax);
            vqi_hi = _mm512_max_epi32(vqi_hi, vmin);
            vqi_hi = _mm512_min_epi32(vqi_hi, vmax);

            // Pack
            __m256i vq8 = _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm512_cvtepi32_epi8(vqi_lo)),
                _mm512_cvtepi32_epi8(vqi_hi), 1);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(q8_qs), vq8);
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
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                decode_iq1m_to_q8_0_avx512(block, subblock_idx, global_scale, q8_qs, q8_scale);
                return;
            }
#endif

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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

#ifdef __AVX512F__
        /**
         * @brief AVX512-optimized unpack of entire IQ1_M super-block to int8 (256 elements)
         *
         * APPROACH: 2x unroll with branchless delta and optimized reduction
         *
         * Key optimizations:
         * 1. Precompute all 16 scale factors (dl1/dl2 × 8 subblocks)
         * 2. 2x loop unroll for gather latency hiding
         * 3. Branchless delta: use mask + blend instead of branches
         * 4. FMA for scale computation: dl * (grid + delta) = fmadd(dl, grid, dl*delta)
         * 5. Shared abs/max computation before split
         *
         * @param block Source IQ1_M super-block (56 bytes)
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins (always 0.0f for IQ1_M)
         */
        __attribute__((always_inline)) inline void unpack_iq1_m_superblock_to_int8_avx512(
            const IQ1_MBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
            // ============================================================
            // Extract global scale and precompute all sub-scales
            // ============================================================
            const uint16_t *sc = reinterpret_cast<const uint16_t *>(block.scales);
            const uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                       ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            const float global_scale = fp16_to_fp32(scale_u16);
            const float gs2 = global_scale * 2.0f;

            // Precompute scales: interleaved [dl1_0, dl2_0, dl1_1, dl2_1, ...]
            alignas(64) float dl_arr[16];
            {
                const uint16_t sc0 = sc[0], sc1 = sc[1], sc2 = sc[2], sc3 = sc[3];
                dl_arr[0] = gs2 * (float)((sc0 >> 0) & 0x7) + global_scale;
                dl_arr[1] = gs2 * (float)((sc0 >> 3) & 0x7) + global_scale;
                dl_arr[2] = gs2 * (float)((sc0 >> 6) & 0x7) + global_scale;
                dl_arr[3] = gs2 * (float)((sc0 >> 9) & 0x7) + global_scale;
                dl_arr[4] = gs2 * (float)((sc1 >> 0) & 0x7) + global_scale;
                dl_arr[5] = gs2 * (float)((sc1 >> 3) & 0x7) + global_scale;
                dl_arr[6] = gs2 * (float)((sc1 >> 6) & 0x7) + global_scale;
                dl_arr[7] = gs2 * (float)((sc1 >> 9) & 0x7) + global_scale;
                dl_arr[8] = gs2 * (float)((sc2 >> 0) & 0x7) + global_scale;
                dl_arr[9] = gs2 * (float)((sc2 >> 3) & 0x7) + global_scale;
                dl_arr[10] = gs2 * (float)((sc2 >> 6) & 0x7) + global_scale;
                dl_arr[11] = gs2 * (float)((sc2 >> 9) & 0x7) + global_scale;
                dl_arr[12] = gs2 * (float)((sc3 >> 0) & 0x7) + global_scale;
                dl_arr[13] = gs2 * (float)((sc3 >> 3) & 0x7) + global_scale;
                dl_arr[14] = gs2 * (float)((sc3 >> 6) & 0x7) + global_scale;
                dl_arr[15] = gs2 * (float)((sc3 >> 9) & 0x7) + global_scale;
            }

            // Precompute constants
            const __m512i vmin_clamp = _mm512_set1_epi32(-127);
            const __m512i vmax_clamp = _mm512_set1_epi32(127);
            const __m512 vpos_delta = _mm512_set1_ps(IQ1S_DELTA);
            const __m512 vneg_delta = _mm512_set1_ps(-IQ1S_DELTA);
            const __m512 v127 = _mm512_set1_ps(127.0f);
            const __m512 veps = _mm512_set1_ps(1e-9f);

            // ============================================================
            // Main loop: 2 subblocks per iteration
            // ============================================================
            for (int ib = 0; ib < 8; ib += 2)
            {
                // ---- Extract indices for both subblocks ----
                const uint8_t *qs0 = block.qs + ib * 4;
                const uint8_t *qh0 = block.qh + ib * 2;
                const uint8_t qh0_0 = qh0[0], qh0_1 = qh0[1];

                const uint8_t *qs1 = block.qs + (ib + 1) * 4;
                const uint8_t *qh1 = block.qh + (ib + 1) * 2;
                const uint8_t qh1_0 = qh1[0], qh1_1 = qh1[1];

                // Build indices
                alignas(16) int32_t idx0[4], idx1[4];
                idx0[0] = qs0[0] | ((qh0_0 & 0x07) << 8);
                idx0[1] = qs0[1] | ((qh0_0 & 0x70) << 4);
                idx0[2] = qs0[2] | ((qh0_1 & 0x07) << 8);
                idx0[3] = qs0[3] | ((qh0_1 & 0x70) << 4);

                idx1[0] = qs1[0] | ((qh1_0 & 0x07) << 8);
                idx1[1] = qs1[1] | ((qh1_0 & 0x70) << 4);
                idx1[2] = qs1[2] | ((qh1_1 & 0x07) << 8);
                idx1[3] = qs1[3] | ((qh1_1 & 0x70) << 4);

                // ---- Issue both gathers ----
                __m128i vidx0 = _mm_load_si128((const __m128i *)idx0);
                __m128i vidx1 = _mm_load_si128((const __m128i *)idx1);

                __m256i vgrid0 = _mm256_i32gather_epi64((const long long int *)iq1s_grid, vidx0, 8);
                __m256i vgrid1 = _mm256_i32gather_epi64((const long long int *)iq1s_grid, vidx1, 8);

                // ---- Process subblock ib ----
                {
                    __m128i vg_lo = _mm256_castsi256_si128(vgrid0);
                    __m128i vg_hi = _mm256_extracti128_si256(vgrid0, 1);

                    __m512 vf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vg_lo));
                    __m512 vf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vg_hi));

                    // Build delta masks from sign bits
                    __mmask16 mask_lo = ((qh0_0 & 0x08) ? 0 : 0x00FF) | ((qh0_0 & 0x80) ? 0 : 0xFF00);
                    __mmask16 mask_hi = ((qh0_1 & 0x08) ? 0 : 0x00FF) | ((qh0_1 & 0x80) ? 0 : 0xFF00);

                    __m512 vdelta_lo = _mm512_mask_blend_ps(mask_lo, vneg_delta, vpos_delta);
                    __m512 vdelta_hi = _mm512_mask_blend_ps(mask_hi, vneg_delta, vpos_delta);

                    // Load scales and compute dl*delta for FMA
                    const float dl1 = dl_arr[ib * 2 + 0];
                    const float dl2 = dl_arr[ib * 2 + 1];
                    __m512 vdl1 = _mm512_set1_ps(dl1);
                    __m512 vdl2 = _mm512_set1_ps(dl2);

                    // Use FMA: vals = dl * grid + dl * delta = fmadd(dl, grid, dl*delta)
                    __m512 vdl_delta_lo = _mm512_mul_ps(vdl1, vdelta_lo);
                    __m512 vdl_delta_hi = _mm512_mul_ps(vdl2, vdelta_hi);
                    __m512 vals_lo = _mm512_fmadd_ps(vdl1, vf_lo, vdl_delta_lo);
                    __m512 vals_hi = _mm512_fmadd_ps(vdl2, vf_hi, vdl_delta_hi);

                    // Quantize: find max and compute inverse scale
                    __m512 vabs_max = _mm512_max_ps(_mm512_abs_ps(vals_lo), _mm512_abs_ps(vals_hi));
                    float max_val = _mm512_reduce_max_ps(vabs_max);
                    float scale_out = max_val / 127.0f;
                    __m512 vinv = _mm512_div_ps(v127, _mm512_add_ps(_mm512_set1_ps(max_val), veps));

                    __m512i vqi_lo = _mm512_cvtps_epi32(_mm512_mul_ps(vals_lo, vinv));
                    __m512i vqi_hi = _mm512_cvtps_epi32(_mm512_mul_ps(vals_hi, vinv));
                    vqi_lo = _mm512_max_epi32(_mm512_min_epi32(vqi_lo, vmax_clamp), vmin_clamp);
                    vqi_hi = _mm512_max_epi32(_mm512_min_epi32(vqi_hi, vmax_clamp), vmin_clamp);

                    __m256i vq8 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm512_cvtepi32_epi8(vqi_lo)),
                        _mm512_cvtepi32_epi8(vqi_hi), 1);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + ib * 32), vq8);
                    if (scales)
                        scales[ib] = scale_out;
                }

                // ---- Process subblock ib+1 ----
                {
                    __m128i vg_lo = _mm256_castsi256_si128(vgrid1);
                    __m128i vg_hi = _mm256_extracti128_si256(vgrid1, 1);

                    __m512 vf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vg_lo));
                    __m512 vf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(vg_hi));

                    __mmask16 mask_lo = ((qh1_0 & 0x08) ? 0 : 0x00FF) | ((qh1_0 & 0x80) ? 0 : 0xFF00);
                    __mmask16 mask_hi = ((qh1_1 & 0x08) ? 0 : 0x00FF) | ((qh1_1 & 0x80) ? 0 : 0xFF00);

                    __m512 vdelta_lo = _mm512_mask_blend_ps(mask_lo, vneg_delta, vpos_delta);
                    __m512 vdelta_hi = _mm512_mask_blend_ps(mask_hi, vneg_delta, vpos_delta);

                    const float dl1 = dl_arr[(ib + 1) * 2 + 0];
                    const float dl2 = dl_arr[(ib + 1) * 2 + 1];
                    __m512 vdl1 = _mm512_set1_ps(dl1);
                    __m512 vdl2 = _mm512_set1_ps(dl2);

                    __m512 vdl_delta_lo = _mm512_mul_ps(vdl1, vdelta_lo);
                    __m512 vdl_delta_hi = _mm512_mul_ps(vdl2, vdelta_hi);
                    __m512 vals_lo = _mm512_fmadd_ps(vdl1, vf_lo, vdl_delta_lo);
                    __m512 vals_hi = _mm512_fmadd_ps(vdl2, vf_hi, vdl_delta_hi);

                    __m512 vabs_max = _mm512_max_ps(_mm512_abs_ps(vals_lo), _mm512_abs_ps(vals_hi));
                    float max_val = _mm512_reduce_max_ps(vabs_max);
                    float scale_out = max_val / 127.0f;
                    __m512 vinv = _mm512_div_ps(v127, _mm512_add_ps(_mm512_set1_ps(max_val), veps));

                    __m512i vqi_lo = _mm512_cvtps_epi32(_mm512_mul_ps(vals_lo, vinv));
                    __m512i vqi_hi = _mm512_cvtps_epi32(_mm512_mul_ps(vals_hi, vinv));
                    vqi_lo = _mm512_max_epi32(_mm512_min_epi32(vqi_lo, vmax_clamp), vmin_clamp);
                    vqi_hi = _mm512_max_epi32(_mm512_min_epi32(vqi_hi, vmax_clamp), vmin_clamp);

                    __m256i vq8 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm512_cvtepi32_epi8(vqi_lo)),
                        _mm512_cvtepi32_epi8(vqi_hi), 1);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + (ib + 1) * 32), vq8);
                    if (scales)
                        scales[ib + 1] = scale_out;
                }
            }

            // Set mins to zero
            if (mins)
            {
                _mm256_storeu_ps(mins, _mm256_setzero_ps());
            }
        }
#endif

        /**
         * @brief Unpack entire IQ1_M super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks.
         *
         * @param block Source IQ1_M super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins (always 0 for IQ1_M)
         */
        inline void unpack_iq1_m_superblock_to_int8(
            const IQ1_MBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__)
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_iq1_m_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }
#endif
            // Fallback to per-subblock processing
            float global_scale = extract_iq1m_global_scale(block);
            for (int i = 0; i < 8; ++i)
            {
                int8_t q8_qs[32];
                uint16_t q8_scale_u16;
                decode_iq1m_to_q8_0(block, i, global_scale, q8_qs, &q8_scale_u16);

#if defined(__AVX2__) || defined(__AVX512F__)
                _mm256_storeu_si256((__m256i *)(output + i * 32), _mm256_loadu_si256((const __m256i *)q8_qs));
#else
                std::memcpy(output + i * 32, q8_qs, 32);
#endif

                if (scales)
                    scales[i] = fp16_to_fp32(q8_scale_u16);
                if (mins)
                    mins[i] = 0.0f;
            }
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

#if defined(__AVX512F__) && defined(__AVX512BW__)
        inline void unpack_q3_k_superblock_to_int8_avx512(
            const Q3_KBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
            // 1. Unpack scales (scalar is fast enough for 12 bytes)
            const float d_all = fp16_to_fp32(block.d);

            uint32_t aux[4];
            {
                const uint32_t kmask1 = 0x03030303;
                const uint32_t kmask2 = 0x0f0f0f0f;
                uint32_t tmp_aux[4];
                memcpy(tmp_aux, block.scales, 12);
                uint32_t tmp = tmp_aux[2];
                aux[2] = ((tmp_aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((tmp_aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (tmp_aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (tmp_aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            }

            // 2. Compute dl, inv_scale, bias for all 16 halves using AVX512
            __m512 v_dl, v_inv_scale, v_bias;
            __m512 v_out_s, v_out_m;

            {
                // Load 16 scale bytes
                __m128i v_sc_i8 = _mm_loadu_si128((const __m128i *)aux);
                // Convert to 16 floats
                __m512i v_sc_i32 = _mm512_cvtepi8_epi32(v_sc_i8);
                __m512 v_sc = _mm512_cvtepi32_ps(v_sc_i32);

                // dl = d_all * (sc - 32)
                __m512 v_d_all = _mm512_set1_ps(d_all);
                v_dl = _mm512_mul_ps(v_d_all, _mm512_sub_ps(v_sc, _mm512_set1_ps(32.0f)));

                // Compute min/max for each half
                __mmask16 mask_pos = _mm512_cmp_ps_mask(v_dl, _mm512_setzero_ps(), _CMP_GE_OQ);

                __m512 v_min = _mm512_mul_ps(v_dl, _mm512_mask_blend_ps(mask_pos, _mm512_set1_ps(3.0f), _mm512_set1_ps(-4.0f)));
                __m512 v_max = _mm512_mul_ps(v_dl, _mm512_mask_blend_ps(mask_pos, _mm512_set1_ps(-4.0f), _mm512_set1_ps(3.0f)));

                // Pair up to get global_min/max for each sub-block (8 sub-blocks)
                // Swap adjacent pairs: 0<->1, 2<->3...
                __m512 v_min_swap = _mm512_permute_ps(v_min, 0xB1);
                __m512 v_max_swap = _mm512_permute_ps(v_max, 0xB1);

                __m512 v_gmin = _mm512_min_ps(v_min, v_min_swap);
                __m512 v_gmax = _mm512_max_ps(v_max, v_max_swap);

                __m512 v_range = _mm512_sub_ps(v_gmax, v_gmin);

                // Check range < 1e-5
                __mmask16 mask_small = _mm512_cmp_ps_mask(v_range, _mm512_set1_ps(1e-5f), _CMP_LT_OQ);

                // out_s = range / 255.0f
                v_out_s = _mm512_div_ps(v_range, _mm512_set1_ps(255.0f));
                // out_m = gmin + 128.0f * out_s
                v_out_m = _mm512_add_ps(v_gmin, _mm512_mul_ps(_mm512_set1_ps(128.0f), v_out_s));

                // Handle small range
                v_out_s = _mm512_mask_blend_ps(mask_small, v_out_s, _mm512_setzero_ps());
                v_out_m = _mm512_mask_blend_ps(mask_small, v_out_m, v_gmin);

                // inv_scale = 1.0f / out_s
                __m512 v_safe_s = _mm512_mask_blend_ps(mask_small, v_out_s, _mm512_set1_ps(1.0f));
                v_inv_scale = _mm512_div_ps(_mm512_set1_ps(1.0f), v_safe_s);

                // bias = -out_m * inv_scale
                v_bias = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), v_out_m), v_inv_scale);

                // If small range, set bias to -128 (so result is -128)
                v_inv_scale = _mm512_mask_blend_ps(mask_small, v_inv_scale, _mm512_setzero_ps());
                v_bias = _mm512_mask_blend_ps(mask_small, v_bias, _mm512_set1_ps(-128.0f));
            }

            // Save scales/mins if needed
            if (scales || mins)
            {
                float tmp_s[16], tmp_m[16];
                _mm512_storeu_ps(tmp_s, v_out_s);
                _mm512_storeu_ps(tmp_m, v_out_m);
                for (int i = 0; i < 8; ++i)
                {
                    if (scales)
                        scales[i] = tmp_s[2 * i];
                    if (mins)
                        mins[i] = tmp_m[2 * i];
                }
            }

            // 3. Compute LUTs (128 bytes) in registers
            __m512i v_lut0, v_lut1;
            {
                __m512 v_vals = _mm512_set_ps(
                    -1, -2, -3, -4, 3, 2, 1, 0,
                    -1, -2, -3, -4, 3, 2, 1, 0);

                auto compute_lut_pair = [&](int k) -> __m128i
                {
                    __m512i v_idx = _mm512_set_epi32(
                        2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1,
                        2 * k, 2 * k, 2 * k, 2 * k, 2 * k, 2 * k, 2 * k, 2 * k);

                    __m512 v_dl_sub = _mm512_permutexvar_ps(v_idx, v_dl);
                    __m512 v_inv_sub = _mm512_permutexvar_ps(v_idx, v_inv_scale);
                    __m512 v_bias_sub = _mm512_permutexvar_ps(v_idx, v_bias);

                    __m512 v_fval = _mm512_mul_ps(v_dl_sub, v_vals);
                    __m512 v_res = _mm512_fmadd_ps(v_fval, v_inv_sub, v_bias_sub);
                    __m512i v_ires = _mm512_cvtps_epi32(v_res);
                    return _mm512_cvtepi32_epi8(v_ires);
                };

                v_lut0 = _mm512_castsi128_si512(compute_lut_pair(0));
                v_lut0 = _mm512_inserti32x4(v_lut0, compute_lut_pair(1), 1);
                v_lut0 = _mm512_inserti32x4(v_lut0, compute_lut_pair(2), 2);
                v_lut0 = _mm512_inserti32x4(v_lut0, compute_lut_pair(3), 3);

                v_lut1 = _mm512_castsi128_si512(compute_lut_pair(4));
                v_lut1 = _mm512_inserti32x4(v_lut1, compute_lut_pair(5), 1);
                v_lut1 = _mm512_inserti32x4(v_lut1, compute_lut_pair(6), 2);
                v_lut1 = _mm512_inserti32x4(v_lut1, compute_lut_pair(7), 3);
            }

            // 4. Unpack blocks using LUT
            // Hoist hmask loading
            __m256i hmask_256 = _mm256_loadu_si256((const __m256i *)block.hmask);
            __m512i hmask = _mm512_castsi256_si512(hmask_256);
            hmask = _mm512_inserti64x4(hmask, hmask_256, 1);

            // Hoist constants
            __m512i v_half_offset = _mm512_set_epi8(
                8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

            __m512i v_sb_offset = _mm512_set_epi8(
                16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

// Helper macro for unrolled iteration
#define UNPACK_Q3K_ITER(I, CHUNK_IDX, V_LUT_SRC, BASE)                               \
    {                                                                                \
        /* Load qs */                                                                \
        const uint8_t *q_ptr = block.qs + (CHUNK_IDX) * 32;                          \
        __m256i q_256 = _mm256_loadu_si256((const __m256i *)q_ptr);                  \
        __m512i qs = _mm512_castsi256_si512(q_256);                                  \
        qs = _mm512_inserti64x4(qs, q_256, 1);                                       \
                                                                                     \
        /* Shift vector */                                                           \
        /* i=0: j0=0, j1=1 -> shift0=0, shift1=2 */                                  \
        /* i=2: j0=2, j1=3 -> shift0=4, shift1=6 */                                  \
        /* i=4: j0=0, j1=1 -> shift0=0, shift1=2 */                                  \
        /* i=6: j0=2, j1=3 -> shift0=4, shift1=6 */                                  \
        int j0 = (I) % 4;                                                            \
        int shift0 = j0 * 2;                                                         \
        int j1 = ((I) + 1) % 4;                                                      \
        int shift1 = j1 * 2;                                                         \
                                                                                     \
        __m512i v_shift = _mm512_set_epi16(                                          \
            shift1, shift1, shift1, shift1, shift1, shift1, shift1, shift1,          \
            shift1, shift1, shift1, shift1, shift1, shift1, shift1, shift1,          \
            shift0, shift0, shift0, shift0, shift0, shift0, shift0, shift0,          \
            shift0, shift0, shift0, shift0, shift0, shift0, shift0, shift0);         \
                                                                                     \
        __m512i q_indices = _mm512_srlv_epi16(qs, v_shift);                          \
        q_indices = _mm512_and_si512(q_indices, _mm512_set1_epi8(3));                \
                                                                                     \
        /* Mask bits */                                                              \
        /* i=0: 1<<1, 1<<0 -> 2, 1 */                                                \
        /* i=2: 1<<3, 1<<2 -> 8, 4 */                                                \
        /* i=4: 1<<5, 1<<4 -> 32, 16 */                                              \
        /* i=6: 1<<7, 1<<6 -> 128, 64 */                                             \
        int m0 = 1 << (I);                                                           \
        int m1 = 1 << ((I) + 1);                                                     \
                                                                                     \
        __m512i m_vec = _mm512_set_epi8(                                             \
            m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1,          \
            m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1, m1,          \
            m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0,          \
            m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0, m0);         \
                                                                                     \
        __m512i hm_bits = _mm512_and_si512(hmask, m_vec);                            \
        __mmask64 is_zero = _mm512_cmpeq_epi8_mask(hm_bits, _mm512_setzero_si512()); \
        __m512i offset = _mm512_mask_set1_epi8(_mm512_setzero_si512(), is_zero, 4);  \
                                                                                     \
        __m512i indices = _mm512_or_si512(q_indices, offset);                        \
        indices = _mm512_add_epi8(indices, v_half_offset);                           \
        indices = _mm512_add_epi8(indices, v_sb_offset);                             \
                                                                                     \
        __m512i v_perm_idx = _mm512_set_epi64(                                       \
            (BASE) + 3, (BASE) + 2, (BASE) + 3, (BASE) + 2,                          \
            (BASE) + 1, (BASE) + 0, (BASE) + 1, (BASE) + 0);                         \
                                                                                     \
        __m512i v_lut_dup = _mm512_permutexvar_epi64(v_perm_idx, V_LUT_SRC);         \
        __m512i result = _mm512_shuffle_epi8(v_lut_dup, indices);                    \
        _mm512_storeu_si512(output + (I) * 32, result);                              \
    }

            UNPACK_Q3K_ITER(0, 0, v_lut0, 0);
            UNPACK_Q3K_ITER(2, 0, v_lut0, 4);
            UNPACK_Q3K_ITER(4, 1, v_lut1, 0);
            UNPACK_Q3K_ITER(6, 1, v_lut1, 4);

#undef UNPACK_Q3K_ITER
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
         * @brief Unpack entire Q3_K super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks with single CPU dispatch.
         *
         * @param block Source Q3_K super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins
         */
        inline void unpack_q3_k_superblock_to_int8(
            const Q3_KBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_q3_k_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }
#endif

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
            {
                for (int i = 0; i < 8; ++i)
                {
                    float s, m;
                    transcode_q3_k_to_int8_avx2(block, i, output + i * 32, &s, &m);
                    if (scales)
                        scales[i] = s;
                    if (mins)
                        mins[i] = m;
                }
                return;
            }
#endif

            // Scalar fallback
            for (int i = 0; i < 8; ++i)
            {
                float s, m;
                transcode_q3_k_to_int8_scalar(block, i, output + i * 32, &s, &m);
                if (scales)
                    scales[i] = s;
                if (mins)
                    mins[i] = m;
            }
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

        /**
         * @brief Unpack entire Q4_K super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks with single CPU dispatch.
         *
         * @param block Source Q4_K super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins
         */
        inline void unpack_q4_k_superblock_to_int8(
            const Q4_KBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
            {
                const float d = fp16_to_fp32(block.d);
                const float dmin = fp16_to_fp32(block.dmin);
                const __m256i mask = _mm256_set1_epi8(0xF);

                for (int i = 0; i < 4; ++i)
                {
                    const uint8_t *q = block.qs + i * 32;
                    const __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q));

                    // Sub-block 2*i (Low nibbles)
                    const __m256i low = _mm256_and_si256(data, mask);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + (2 * i) * 32), low);

                    // Sub-block 2*i+1 (High nibbles)
                    const __m256i high = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + (2 * i + 1) * 32), high);

                    // Scales and mins
                    if (scales || mins)
                    {
                        uint8_t sc, m;
                        // Sub-block 2*i
                        get_scale_min_k4(2 * i, block.scales, &sc, &m);
                        if (scales)
                            scales[2 * i] = d * sc;
                        if (mins)
                            mins[2 * i] = -dmin * m;

                        // Sub-block 2*i+1
                        get_scale_min_k4(2 * i + 1, block.scales, &sc, &m);
                        if (scales)
                            scales[2 * i + 1] = d * sc;
                        if (mins)
                            mins[2 * i + 1] = -dmin * m;
                    }
                }
                return;
            }
#endif

            // Scalar fallback
            for (int i = 0; i < 8; ++i)
            {
                float s, m;
                transcode_q4_k_to_int8_scalar(block, i, output + i * 32, &s, &m);
                if (scales)
                    scales[i] = s;
                if (mins)
                    mins[i] = m;
            }
        }

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

        /**
         * @brief Unpack entire Q6_K super-block to int8 (256 elements)
         *
         * Q6_K layout (per 128-element half):
         *   - ql[64]: lower 4 bits (2 nibbles per byte for 32 positions)
         *   - qh[32]: upper 2 bits (4 x 2-bit fields per byte)
         *   - scales[8]: per-16-element scales
         *
         * Output subblock mapping (for sequential 32-element subblocks):
         *   - Subblock 0: q = (ql[l]&0xF) | ((qh[l]>>0)&3)<<4, scales 0,1
         *   - Subblock 1: q = (ql[l+32]&0xF) | ((qh[l]>>2)&3)<<4, scales 2,3
         *   - Subblock 2: q = (ql[l]>>4) | ((qh[l]>>4)&3)<<4, scales 4,5
         *   - Subblock 3: q = (ql[l+32]>>4) | ((qh[l]>>6)&3)<<4, scales 6,7
         *
         * @param block Source Q6_K super-block
         * @param output Output buffer for 256 int8 values (8 subblocks of 32)
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins (always 0 for Q6_K)
         */
#if defined(__AVX512F__) && defined(__AVX512BW__)
        inline void unpack_q6_k_superblock_to_int8_avx512(
            const Q6_KBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
            const float d = fp16_to_fp32(block.d);

            // Precompute scale factors for all 8 subblocks (16 half-factors)
            alignas(64) int16_t factors_fixed[16]; // 8.8 fixed point
            alignas(64) float out_scales[8];

            for (int half = 0; half < 2; ++half)
            {
                const int8_t *sc_base = block.scales + half * 8;
                for (int sb = 0; sb < 4; ++sb)
                {
                    const int8_t sc0 = sc_base[sb * 2 + 0];
                    const int8_t sc1 = sc_base[sb * 2 + 1];
                    const float max_sc = std::max(std::abs((float)sc0), std::abs((float)sc1));

                    const int idx = half * 4 + sb;
                    if (max_sc == 0.0f)
                    {
                        out_scales[idx] = 0.0f;
                        factors_fixed[idx * 2 + 0] = 0;
                        factors_fixed[idx * 2 + 1] = 0;
                    }
                    else
                    {
                        out_scales[idx] = d * max_sc * 32.0f / 127.0f;
                        factors_fixed[idx * 2 + 0] = (int16_t)std::nearbyint((float)sc0 * 127.0f * 256.0f / (max_sc * 32.0f));
                        factors_fixed[idx * 2 + 1] = (int16_t)std::nearbyint((float)sc1 * 127.0f * 256.0f / (max_sc * 32.0f));
                    }
                }
            }

            if (scales)
                _mm256_storeu_ps(scales, _mm256_loadu_ps(out_scales));
            if (mins)
                _mm256_storeu_ps(mins, _mm256_setzero_ps());

            // Constants
            const __m256i v0F = _mm256_set1_epi8(0x0F);
            const __m256i v03 = _mm256_set1_epi8(0x03);
            const __m512i v32_16 = _mm512_set1_epi16(32);
            const __m512i v128_16 = _mm512_set1_epi16(128);

            // Load all data at once
            const __m512i vqh = _mm512_loadu_si512(block.qh);
            const __m256i vqh_lo = _mm512_castsi512_si256(vqh);
            const __m256i vqh_hi = _mm512_extracti64x4_epi64(vqh, 1);
            const __m512i vql_0 = _mm512_loadu_si512(block.ql);
            const __m512i vql_1 = _mm512_loadu_si512(block.ql + 64);

            // Process 2 subblocks at a time for better ILP (64 elements)
            // Each pair shares the same ql 64-byte load

            // === HALF 0: subblocks 0,1 then 2,3 ===
            {
                // Subblocks 0,1: use low nibbles of vql_0
                const __m256i vql_lo0 = _mm512_castsi512_si256(vql_0);
                const __m256i vql_lo1 = _mm512_extracti64x4_epi64(vql_0, 1);

                // sb0: low nibbles of first 32 bytes, qh shift 0
                __m256i vql_sb0 = _mm256_and_si256(vql_lo0, v0F);
                __m256i vqh_sb0 = _mm256_and_si256(vqh_lo, v03);
                __m256i vq_sb0 = _mm256_or_si256(vql_sb0, _mm256_slli_epi16(vqh_sb0, 4));

                // sb1: low nibbles of second 32 bytes, qh shift 2
                __m256i vql_sb1 = _mm256_and_si256(vql_lo1, v0F);
                __m256i vqh_sb1 = _mm256_and_si256(_mm256_srli_epi16(vqh_lo, 2), v03);
                __m256i vq_sb1 = _mm256_or_si256(vql_sb1, _mm256_slli_epi16(vqh_sb1, 4));

                // Expand both to 16-bit, center, multiply - interleaved for ILP
                __m512i vq16_sb0 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(vq_sb0), v32_16);
                __m512i vq16_sb1 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(vq_sb1), v32_16);

                __m512i vf0_0 = _mm512_set1_epi16(factors_fixed[0]);
                __m512i vf0_1 = _mm512_set1_epi16(factors_fixed[1]);
                __m512i vfactor_sb0 = _mm512_mask_blend_epi16(0xFFFF0000u, vf0_0, vf0_1);

                __m512i vf1_0 = _mm512_set1_epi16(factors_fixed[2]);
                __m512i vf1_1 = _mm512_set1_epi16(factors_fixed[3]);
                __m512i vfactor_sb1 = _mm512_mask_blend_epi16(0xFFFF0000u, vf1_0, vf1_1);

                __m512i vmul_sb0 = _mm512_srai_epi16(_mm512_add_epi16(_mm512_mullo_epi16(vq16_sb0, vfactor_sb0), v128_16), 8);
                __m512i vmul_sb1 = _mm512_srai_epi16(_mm512_add_epi16(_mm512_mullo_epi16(vq16_sb1, vfactor_sb1), v128_16), 8);

                _mm256_storeu_si256((__m256i *)(output + 0), _mm512_cvtepi16_epi8(vmul_sb0));
                _mm256_storeu_si256((__m256i *)(output + 32), _mm512_cvtepi16_epi8(vmul_sb1));

                // Subblocks 2,3: use high nibbles of vql_0
                __m256i vql_hi0 = _mm256_and_si256(_mm256_srli_epi16(vql_lo0, 4), v0F);
                __m256i vql_hi1 = _mm256_and_si256(_mm256_srli_epi16(vql_lo1, 4), v0F);

                // sb2: high nibbles of first 32 bytes, qh shift 4
                __m256i vqh_sb2 = _mm256_and_si256(_mm256_srli_epi16(vqh_lo, 4), v03);
                __m256i vq_sb2 = _mm256_or_si256(vql_hi0, _mm256_slli_epi16(vqh_sb2, 4));

                // sb3: high nibbles of second 32 bytes, qh shift 6
                __m256i vqh_sb3 = _mm256_and_si256(_mm256_srli_epi16(vqh_lo, 6), v03);
                __m256i vq_sb3 = _mm256_or_si256(vql_hi1, _mm256_slli_epi16(vqh_sb3, 4));

                __m512i vq16_sb2 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(vq_sb2), v32_16);
                __m512i vq16_sb3 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(vq_sb3), v32_16);

                __m512i vf2_0 = _mm512_set1_epi16(factors_fixed[4]);
                __m512i vf2_1 = _mm512_set1_epi16(factors_fixed[5]);
                __m512i vfactor_sb2 = _mm512_mask_blend_epi16(0xFFFF0000u, vf2_0, vf2_1);

                __m512i vf3_0 = _mm512_set1_epi16(factors_fixed[6]);
                __m512i vf3_1 = _mm512_set1_epi16(factors_fixed[7]);
                __m512i vfactor_sb3 = _mm512_mask_blend_epi16(0xFFFF0000u, vf3_0, vf3_1);

                __m512i vmul_sb2 = _mm512_srai_epi16(_mm512_add_epi16(_mm512_mullo_epi16(vq16_sb2, vfactor_sb2), v128_16), 8);
                __m512i vmul_sb3 = _mm512_srai_epi16(_mm512_add_epi16(_mm512_mullo_epi16(vq16_sb3, vfactor_sb3), v128_16), 8);

                _mm256_storeu_si256((__m256i *)(output + 64), _mm512_cvtepi16_epi8(vmul_sb2));
                _mm256_storeu_si256((__m256i *)(output + 96), _mm512_cvtepi16_epi8(vmul_sb3));
            }

            // === HALF 1: subblocks 4,5 then 6,7 ===
            {
                const __m256i vql_lo0 = _mm512_castsi512_si256(vql_1);
                const __m256i vql_lo1 = _mm512_extracti64x4_epi64(vql_1, 1);

                // sb4: low nibbles of first 32 bytes, qh shift 0
                __m256i vql_sb4 = _mm256_and_si256(vql_lo0, v0F);
                __m256i vqh_sb4 = _mm256_and_si256(vqh_hi, v03);
                __m256i vq_sb4 = _mm256_or_si256(vql_sb4, _mm256_slli_epi16(vqh_sb4, 4));

                // sb5: low nibbles of second 32 bytes, qh shift 2
                __m256i vql_sb5 = _mm256_and_si256(vql_lo1, v0F);
                __m256i vqh_sb5 = _mm256_and_si256(_mm256_srli_epi16(vqh_hi, 2), v03);
                __m256i vq_sb5 = _mm256_or_si256(vql_sb5, _mm256_slli_epi16(vqh_sb5, 4));

                __m512i vq16_sb4 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(vq_sb4), v32_16);
                __m512i vq16_sb5 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(vq_sb5), v32_16);

                __m512i vf4_0 = _mm512_set1_epi16(factors_fixed[8]);
                __m512i vf4_1 = _mm512_set1_epi16(factors_fixed[9]);
                __m512i vfactor_sb4 = _mm512_mask_blend_epi16(0xFFFF0000u, vf4_0, vf4_1);

                __m512i vf5_0 = _mm512_set1_epi16(factors_fixed[10]);
                __m512i vf5_1 = _mm512_set1_epi16(factors_fixed[11]);
                __m512i vfactor_sb5 = _mm512_mask_blend_epi16(0xFFFF0000u, vf5_0, vf5_1);

                __m512i vmul_sb4 = _mm512_srai_epi16(_mm512_add_epi16(_mm512_mullo_epi16(vq16_sb4, vfactor_sb4), v128_16), 8);
                __m512i vmul_sb5 = _mm512_srai_epi16(_mm512_add_epi16(_mm512_mullo_epi16(vq16_sb5, vfactor_sb5), v128_16), 8);

                _mm256_storeu_si256((__m256i *)(output + 128), _mm512_cvtepi16_epi8(vmul_sb4));
                _mm256_storeu_si256((__m256i *)(output + 160), _mm512_cvtepi16_epi8(vmul_sb5));

                // sb6,7: high nibbles
                __m256i vql_hi0 = _mm256_and_si256(_mm256_srli_epi16(vql_lo0, 4), v0F);
                __m256i vql_hi1 = _mm256_and_si256(_mm256_srli_epi16(vql_lo1, 4), v0F);

                __m256i vqh_sb6 = _mm256_and_si256(_mm256_srli_epi16(vqh_hi, 4), v03);
                __m256i vq_sb6 = _mm256_or_si256(vql_hi0, _mm256_slli_epi16(vqh_sb6, 4));

                __m256i vqh_sb7 = _mm256_and_si256(_mm256_srli_epi16(vqh_hi, 6), v03);
                __m256i vq_sb7 = _mm256_or_si256(vql_hi1, _mm256_slli_epi16(vqh_sb7, 4));

                __m512i vq16_sb6 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(vq_sb6), v32_16);
                __m512i vq16_sb7 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(vq_sb7), v32_16);

                __m512i vf6_0 = _mm512_set1_epi16(factors_fixed[12]);
                __m512i vf6_1 = _mm512_set1_epi16(factors_fixed[13]);
                __m512i vfactor_sb6 = _mm512_mask_blend_epi16(0xFFFF0000u, vf6_0, vf6_1);

                __m512i vf7_0 = _mm512_set1_epi16(factors_fixed[14]);
                __m512i vf7_1 = _mm512_set1_epi16(factors_fixed[15]);
                __m512i vfactor_sb7 = _mm512_mask_blend_epi16(0xFFFF0000u, vf7_0, vf7_1);

                __m512i vmul_sb6 = _mm512_srai_epi16(_mm512_add_epi16(_mm512_mullo_epi16(vq16_sb6, vfactor_sb6), v128_16), 8);
                __m512i vmul_sb7 = _mm512_srai_epi16(_mm512_add_epi16(_mm512_mullo_epi16(vq16_sb7, vfactor_sb7), v128_16), 8);

                _mm256_storeu_si256((__m256i *)(output + 192), _mm512_cvtepi16_epi8(vmul_sb6));
                _mm256_storeu_si256((__m256i *)(output + 224), _mm512_cvtepi16_epi8(vmul_sb7));
            }
        }
#endif

        inline void unpack_q6_k_superblock_to_int8(
            const Q6_KBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_q6_k_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }
#endif

            const float d = fp16_to_fp32(block.d);

            // Process 2 halves (128 elements each = 4 subblocks each)
            for (int half = 0; half < 2; ++half)
            {
                const uint8_t *ql = block.ql + half * 64;
                const uint8_t *qh = block.qh + half * 32;
                const int8_t *sc = block.scales + half * 8;
                int8_t *out = output + half * 128;

                // For each of 4 subblocks in this half
                for (int sb = 0; sb < 4; ++sb)
                {
                    // Determine bit extraction parameters based on subblock
                    const uint8_t *ql_ptr;
                    int ql_shift, qh_shift;

                    if (sb == 0)
                    {
                        ql_ptr = ql;
                        ql_shift = 0;
                        qh_shift = 0;
                    }
                    else if (sb == 1)
                    {
                        ql_ptr = ql + 32;
                        ql_shift = 0;
                        qh_shift = 2;
                    }
                    else if (sb == 2)
                    {
                        ql_ptr = ql;
                        ql_shift = 4;
                        qh_shift = 4;
                    }
                    else
                    {
                        ql_ptr = ql + 32;
                        ql_shift = 4;
                        qh_shift = 6;
                    }

                    // Scales for this subblock (2 scales per 32 elements)
                    const float scale0 = d * sc[sb * 2 + 0];
                    const float scale1 = d * sc[sb * 2 + 1];

                    // Find max for INT8 quantization (process both 16-element halves)
                    float max_abs = 0.0f;
                    for (int l = 0; l < 32; ++l)
                    {
                        uint8_t ql_val = (ql_ptr[l] >> ql_shift) & 0xF;
                        uint8_t qh_val = (qh[l] >> qh_shift) & 3;
                        int8_t q = static_cast<int8_t>(ql_val | (qh_val << 4)) - 32;
                        float s = (l < 16) ? scale0 : scale1;
                        float val = std::fabs(s * q);
                        if (val > max_abs)
                            max_abs = val;
                    }

                    // Compute output scale
                    float out_scale = max_abs / 127.0f;
                    if (out_scale < 1e-10f)
                        out_scale = 0.0f;

                    int sb_idx = half * 4 + sb;
                    if (scales)
                        scales[sb_idx] = out_scale;
                    if (mins)
                        mins[sb_idx] = 0.0f;

                    // Quantize to INT8
                    int8_t *dst = out + sb * 32;
                    if (out_scale == 0.0f)
                    {
                        std::memset(dst, 0, 32);
                    }
                    else
                    {
                        float inv_scale = 1.0f / out_scale;
                        for (int l = 0; l < 32; ++l)
                        {
                            uint8_t ql_val = (ql_ptr[l] >> ql_shift) & 0xF;
                            uint8_t qh_val = (qh[l] >> qh_shift) & 3;
                            int8_t q = static_cast<int8_t>(ql_val | (qh_val << 4)) - 32;
                            float s = (l < 16) ? scale0 : scale1;
                            float val = s * q * inv_scale;
                            int32_t qi = static_cast<int32_t>(std::nearbyint(val));
                            if (qi < -128)
                                qi = -128;
                            if (qi > 127)
                                qi = 127;
                            dst[l] = static_cast<int8_t>(qi);
                        }
                    }
                }
            }
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
         *
         * Fully vectorized: extracts high bits using SIMD bit manipulation.
         * - Low 4 bits from qs[16] (2 per byte)
         * - High bit from qh[4] (32 bits total)
         * - Result: (low4 | high_bit_in_pos4) - 16
         */
        inline void unpack_q5_0_to_int8_avx512(const Q5_0Block &block, int8_t *output)
        {
            // Load qs[16] bytes
            __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);

            // Extract low and high nibbles
            __m128i low_nibbles = _mm_and_si128(qs, low_mask);                     // Elements 0-15 (low 4 bits)
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(qs, 4), low_mask); // Elements 16-31 (high 4 bits)

            // Load qh as 32-bit integer
            uint32_t qh;
            std::memcpy(&qh, block.qh, sizeof(qh));

            // Bit mapping from scalar code analysis:
            // output[j] (j=0..15): xh_0 = ((qh >> j) << 4) & 0x10 => bit j of qh
            // output[j+16] (j=0..15): xh_1 = ((qh >> (j+12)) & 0x10 => bit (j+16) of qh (bit 4 of shifted result)
            //
            // So: low nibbles use qh bits 0-15, high nibbles use qh bits 16-31

            // Bit mask for testing each bit position within a byte
            __m128i bit_mask = _mm_set_epi8(
                static_cast<char>(0x80), 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
                static_cast<char>(0x80), 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);

            // Extract each byte of qh
            uint8_t qh_byte0 = static_cast<uint8_t>(qh);       // bits 0-7
            uint8_t qh_byte1 = static_cast<uint8_t>(qh >> 8);  // bits 8-15
            uint8_t qh_byte2 = static_cast<uint8_t>(qh >> 16); // bits 16-23
            uint8_t qh_byte3 = static_cast<uint8_t>(qh >> 24); // bits 24-31

            // Broadcast bytes for low nibbles (output 0-15 need qh bits 0-15)
            __m128i qh_lo = _mm_set_epi8(
                qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1,
                qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0);

            // Broadcast bytes for high nibbles (output 16-31 need qh bits 16-31)
            __m128i qh_hi = _mm_set_epi8(
                qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3,
                qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2);

            // Test each bit position
            __m128i bits_lo = _mm_and_si128(qh_lo, bit_mask);
            __m128i bits_hi = _mm_and_si128(qh_hi, bit_mask);

            // Compare to get 0xFF where bit was set, 0x00 where not
            bits_lo = _mm_cmpeq_epi8(bits_lo, bit_mask);
            bits_hi = _mm_cmpeq_epi8(bits_hi, bit_mask);

            // Convert 0xFF to 0x10 (bit 4 position)
            __m128i pos4_mask = _mm_set1_epi8(0x10);
            bits_lo = _mm_and_si128(bits_lo, pos4_mask);
            bits_hi = _mm_and_si128(bits_hi, pos4_mask);

            // Combine nibbles with high bits
            __m128i result_lo = _mm_or_si128(low_nibbles, bits_lo);
            __m128i result_hi = _mm_or_si128(high_nibbles, bits_hi);

            // Subtract 16 (Q5_0 is symmetric around 0)
            __m128i offset = _mm_set1_epi8(16);
            result_lo = _mm_sub_epi8(result_lo, offset);
            result_hi = _mm_sub_epi8(result_hi, offset);

            // Store results
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), result_lo);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), result_hi);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief AVX2 implementation for Q5_0 → int8 unpacking
         */
        inline void unpack_q5_0_to_int8_avx2(const Q5_0Block &block, int8_t *output)
        {
            // Load qs[16] bytes
            __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);

            // Extract low and high nibbles
            __m128i low_nibbles = _mm_and_si128(qs, low_mask);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(qs, 4), low_mask);

            // Load qh as 32-bit integer
            uint32_t qh;
            std::memcpy(&qh, block.qh, sizeof(qh));

            // Bit mask for testing each bit position within a byte
            __m128i bit_mask = _mm_set_epi8(
                static_cast<char>(0x80), 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
                static_cast<char>(0x80), 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);

            // Extract each byte of qh
            uint8_t qh_byte0 = static_cast<uint8_t>(qh);       // bits 0-7
            uint8_t qh_byte1 = static_cast<uint8_t>(qh >> 8);  // bits 8-15
            uint8_t qh_byte2 = static_cast<uint8_t>(qh >> 16); // bits 16-23
            uint8_t qh_byte3 = static_cast<uint8_t>(qh >> 24); // bits 24-31

            // Broadcast bytes for low nibbles (output 0-15 need qh bits 0-15)
            __m128i qh_lo = _mm_set_epi8(
                qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1,
                qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0);

            // Broadcast bytes for high nibbles (output 16-31 need qh bits 16-31)
            __m128i qh_hi = _mm_set_epi8(
                qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3,
                qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2);

            // Test each bit position
            __m128i bits_lo = _mm_and_si128(qh_lo, bit_mask);
            __m128i bits_hi = _mm_and_si128(qh_hi, bit_mask);

            // Compare to get 0xFF where bit was set, 0x00 where not
            bits_lo = _mm_cmpeq_epi8(bits_lo, bit_mask);
            bits_hi = _mm_cmpeq_epi8(bits_hi, bit_mask);

            // Convert 0xFF to 0x10 (bit 4 position)
            __m128i pos4_mask = _mm_set1_epi8(0x10);
            bits_lo = _mm_and_si128(bits_lo, pos4_mask);
            bits_hi = _mm_and_si128(bits_hi, pos4_mask);

            // Combine nibbles with high bits
            __m128i result_lo = _mm_or_si128(low_nibbles, bits_lo);
            __m128i result_hi = _mm_or_si128(high_nibbles, bits_hi);

            // Subtract 16 (Q5_0 is symmetric around 0)
            __m128i offset = _mm_set1_epi8(16);
            result_lo = _mm_sub_epi8(result_lo, offset);
            result_hi = _mm_sub_epi8(result_hi, offset);

            // Store results
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), result_lo);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), result_hi);
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

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief Optimized Q2_K superblock unpacking using AVX512
         *
         * Q2_K layout:
         * - 256 elements total, split into 8 sub-blocks of 32 elements each
         * - qs[64]: packed 2-bit values (4 per byte), organized as 2 chunks of 32 bytes
         * - scales[16]: each byte has scale (low 4 bits) and min (high 4 bits)
         * - Sub-block i uses: qs[chunk*32:(chunk+1)*32] with shift=(i%4)*2
         *   where chunk = i/4, and scales[i*2] for first half, scales[i*2+1] for second half
         *
         * Strategy: Vectorize everything - scale extraction, range computation, LUT building
         */
        inline void unpack_q2_k_superblock_to_int8_avx512(
            const Q2_KBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
            const float d = fp16_to_fp32(block.d);
            const float dmin = fp16_to_fp32(block.dmin);

            // 1. Extract all 16 dl/ml values using AVX512
            __m128i v_sc = _mm_loadu_si128((const __m128i *)block.scales);
            __m128i v_scale_i8 = _mm_and_si128(v_sc, _mm_set1_epi8(0x0F));
            __m128i v_min_i8 = _mm_and_si128(_mm_srli_epi16(v_sc, 4), _mm_set1_epi8(0x0F));

            __m512i v_scale_i32 = _mm512_cvtepu8_epi32(v_scale_i8);
            __m512i v_min_i32 = _mm512_cvtepu8_epi32(v_min_i8);
            __m512 v_scale_f = _mm512_cvtepi32_ps(v_scale_i32);
            __m512 v_min_f = _mm512_cvtepi32_ps(v_min_i32);

            __m512 v_d = _mm512_set1_ps(d);
            __m512 v_dmin = _mm512_set1_ps(dmin);
            __m512 v_dl = _mm512_mul_ps(v_d, v_scale_f);  // 16 dl values
            __m512 v_ml = _mm512_mul_ps(v_dmin, v_min_f); // 16 ml values

            // 2. Compute range for each sub-block (pair up halves)
            // Half i: min = -ml[i], max = 3*dl[i] - ml[i]
            __m512 v_local_min = _mm512_sub_ps(_mm512_setzero_ps(), v_ml);
            __m512 v_local_max = _mm512_fmsub_ps(_mm512_set1_ps(3.0f), v_dl, v_ml);

            // Pair: indices 0,1 → sb0; 2,3 → sb1; etc.
            __m512 v_local_min_swap = _mm512_permute_ps(v_local_min, 0xB1); // swap adjacent
            __m512 v_local_max_swap = _mm512_permute_ps(v_local_max, 0xB1);
            __m512 v_gmin = _mm512_min_ps(v_local_min, v_local_min_swap);
            __m512 v_gmax = _mm512_max_ps(v_local_max, v_local_max_swap);
            __m512 v_range = _mm512_sub_ps(v_gmax, v_gmin);

            // 3. Compute scale/bias for each half (each half in a sub-block shares the same scale)
            __mmask16 mask_small = _mm512_cmp_ps_mask(v_range, _mm512_set1_ps(1e-5f), _CMP_LT_OQ);
            __m512 v_out_s = _mm512_div_ps(v_range, _mm512_set1_ps(255.0f));
            __m512 v_out_m = _mm512_fmadd_ps(_mm512_set1_ps(128.0f), v_out_s, v_gmin);
            v_out_s = _mm512_mask_blend_ps(mask_small, v_out_s, _mm512_setzero_ps());
            v_out_m = _mm512_mask_blend_ps(mask_small, v_out_m, v_gmin);

            __m512 v_safe_s = _mm512_mask_blend_ps(mask_small, v_out_s, _mm512_set1_ps(1.0f));
            __m512 v_inv = _mm512_div_ps(_mm512_set1_ps(1.0f), v_safe_s);
            __m512 v_bias = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), v_out_m), v_inv);
            v_inv = _mm512_mask_blend_ps(mask_small, v_inv, _mm512_setzero_ps());
            v_bias = _mm512_mask_blend_ps(mask_small, v_bias, _mm512_set1_ps(-128.0f));

            // 4. Save scales/mins if requested
            if (scales || mins)
            {
                alignas(64) float tmp_s[16], tmp_m[16];
                _mm512_store_ps(tmp_s, v_out_s);
                _mm512_store_ps(tmp_m, v_out_m);
                for (int i = 0; i < 8; ++i)
                {
                    if (scales)
                        scales[i] = tmp_s[2 * i];
                    if (mins)
                        mins[i] = tmp_m[2 * i];
                }
            }

            // 5. Build all 8 LUTs using vectorized computation
            // For each half h (0-15), LUT[h,q] = round((dl[h] * q - ml[h]) * inv[h] + bias[h])
            // where q ∈ {0,1,2,3}
            alignas(64) int8_t lut_data[8 * 32]; // 8 LUTs, 32 bytes each (only first 4 and bytes 16-19 used per LUT)

            // Process 2 sub-blocks (4 halves) at a time to fit in AVX512
            __m512 v_q = _mm512_set_ps(3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);

            auto build_2_luts = [&](int sb_base)
            {
                // sb_base = 0 or 2 or 4 or 6
                // Halves: sb_base*2, sb_base*2+1, (sb_base+1)*2, (sb_base+1)*2+1
                int h0 = sb_base * 2, h1 = h0 + 1, h2 = h0 + 2, h3 = h0 + 3;

                __m512i idx = _mm512_set_epi32(h3, h3, h3, h3, h2, h2, h2, h2, h1, h1, h1, h1, h0, h0, h0, h0);
                __m512 v_dl_sel = _mm512_permutexvar_ps(idx, v_dl);
                __m512 v_ml_sel = _mm512_permutexvar_ps(idx, v_ml);
                __m512 v_inv_sel = _mm512_permutexvar_ps(idx, v_inv);
                __m512 v_bias_sel = _mm512_permutexvar_ps(idx, v_bias);

                // val = dl * q - ml
                __m512 v_val = _mm512_fmsub_ps(v_dl_sel, v_q, v_ml_sel);
                // lut = round(val * inv + bias)
                __m512 v_lut_f = _mm512_fmadd_ps(v_val, v_inv_sel, v_bias_sel);
                __m512i v_lut_i32 = _mm512_cvtps_epi32(v_lut_f);
                __m128i v_lut_i8 = _mm512_cvtepi32_epi8(v_lut_i32); // 16 int8s

                // Layout: [sb_base half0: q0,q1,q2,q3] [sb_base half1: q0,q1,q2,q3] [sb_base+1 half0: ...] [sb_base+1 half1: ...]
                alignas(16) int8_t tmp[16];
                _mm_store_si128((__m128i *)tmp, v_lut_i8);

                // Copy to proper LUT positions
                int8_t *lut0 = lut_data + sb_base * 32;
                int8_t *lut1 = lut_data + (sb_base + 1) * 32;

                // LUT0: half0 at [0-3], half1 at [16-19]
                lut0[0] = tmp[0];
                lut0[1] = tmp[1];
                lut0[2] = tmp[2];
                lut0[3] = tmp[3];
                lut0[16] = tmp[4];
                lut0[17] = tmp[5];
                lut0[18] = tmp[6];
                lut0[19] = tmp[7];

                // LUT1: half0 at [0-3], half1 at [16-19]
                lut1[0] = tmp[8];
                lut1[1] = tmp[9];
                lut1[2] = tmp[10];
                lut1[3] = tmp[11];
                lut1[16] = tmp[12];
                lut1[17] = tmp[13];
                lut1[18] = tmp[14];
                lut1[19] = tmp[15];
            };

            build_2_luts(0);
            build_2_luts(2);
            build_2_luts(4);
            build_2_luts(6);

            // 6. Apply LUTs using AVX2 vpshufb
            for (int chunk = 0; chunk < 2; ++chunk)
            {
                const uint8_t *q_ptr = block.qs + chunk * 32;
                __m256i q_256 = _mm256_loadu_si256((const __m256i *)q_ptr);

                for (int j = 0; j < 4; ++j)
                {
                    int sb = chunk * 4 + j;
                    int shift = j * 2;

                    __m256i lut_vec = _mm256_loadu_si256((const __m256i *)(lut_data + sb * 32));
                    __m128i count = _mm_cvtsi32_si128(shift);
                    __m256i q_shifted = _mm256_srl_epi16(q_256, count);
                    __m256i q_indices = _mm256_and_si256(q_shifted, _mm256_set1_epi8(3));
                    __m256i result = _mm256_shuffle_epi8(lut_vec, q_indices);
                    _mm256_storeu_si256((__m256i *)(output + sb * 32), result);
                }
            }
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
         * @brief Unpack entire Q2_K super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks with single CPU dispatch.
         *
         * @param block Source Q2_K super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins
         */
        inline void unpack_q2_k_superblock_to_int8(
            const Q2_KBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                unpack_q2_k_superblock_to_int8_avx512(block, output, scales, mins);
                return;
            }
#endif

#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
            {
                for (int i = 0; i < 8; ++i)
                {
                    float s, m;
                    transcode_q2_k_to_int8_avx2(block, i, output + i * 32, &s, &m);
                    if (scales)
                        scales[i] = s;
                    if (mins)
                        mins[i] = m;
                }
                return;
            }
#endif

            // Scalar fallback
            for (int i = 0; i < 8; ++i)
            {
                float s, m;
                transcode_q2_k_to_int8_scalar(block, i, output + i * 32, &s, &m);
                if (scales)
                    scales[i] = s;
                if (mins)
                    mins[i] = m;
            }
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
         *
         * Fully vectorized - same as Q5_0 but without the -16 offset (asymmetric).
         */
        inline void unpack_q5_1_to_int8_avx512(const Q5_1Block &block, int8_t *output)
        {
            // Load qs[16] bytes
            __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);

            // Extract low and high nibbles
            __m128i low_nibbles = _mm_and_si128(qs, low_mask);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(qs, 4), low_mask);

            // Load qh as 32-bit integer
            uint32_t qh;
            std::memcpy(&qh, block.qh, sizeof(qh));

            // Bit mapping from scalar code analysis:
            // output[j] (j=0..15): xh_0 = ((qh >> j) << 4) & 0x10 => bit j of qh
            // output[j+16] (j=0..15): xh_1 = ((qh >> (j+12)) & 0x10 => bit (j+16) of qh
            //
            // So: low nibbles use qh bits 0-15, high nibbles use qh bits 16-31

            // Bit mask for testing each bit position within a byte
            __m128i bit_mask = _mm_set_epi8(
                static_cast<char>(0x80), 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
                static_cast<char>(0x80), 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);

            // Extract each byte of qh
            uint8_t qh_byte0 = static_cast<uint8_t>(qh);       // bits 0-7
            uint8_t qh_byte1 = static_cast<uint8_t>(qh >> 8);  // bits 8-15
            uint8_t qh_byte2 = static_cast<uint8_t>(qh >> 16); // bits 16-23
            uint8_t qh_byte3 = static_cast<uint8_t>(qh >> 24); // bits 24-31

            // Broadcast bytes for low nibbles (output 0-15 need qh bits 0-15)
            __m128i qh_lo = _mm_set_epi8(
                qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1,
                qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0);

            // Broadcast bytes for high nibbles (output 16-31 need qh bits 16-31)
            __m128i qh_hi = _mm_set_epi8(
                qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3,
                qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2);

            // Test each bit position
            __m128i bits_lo = _mm_and_si128(qh_lo, bit_mask);
            __m128i bits_hi = _mm_and_si128(qh_hi, bit_mask);

            // Compare to get 0xFF where bit was set, 0x00 where not
            bits_lo = _mm_cmpeq_epi8(bits_lo, bit_mask);
            bits_hi = _mm_cmpeq_epi8(bits_hi, bit_mask);

            // Convert 0xFF to 0x10 (bit 4 position)
            __m128i pos4_mask = _mm_set1_epi8(0x10);
            bits_lo = _mm_and_si128(bits_lo, pos4_mask);
            bits_hi = _mm_and_si128(bits_hi, pos4_mask);

            // Combine nibbles with high bits (no offset for Q5_1 - asymmetric [0,31])
            __m128i result_lo = _mm_or_si128(low_nibbles, bits_lo);
            __m128i result_hi = _mm_or_si128(high_nibbles, bits_hi);

            // Store results
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), result_lo);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), result_hi);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief AVX2 implementation for Q5_1 → int8 unpacking
         */
        inline void unpack_q5_1_to_int8_avx2(const Q5_1Block &block, int8_t *output)
        {
            // Load qs[16] bytes
            __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);

            // Extract low and high nibbles
            __m128i low_nibbles = _mm_and_si128(qs, low_mask);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(qs, 4), low_mask);

            // Load qh as 32-bit integer
            uint32_t qh;
            std::memcpy(&qh, block.qh, sizeof(qh));

            // Bit mask for testing each bit position within a byte
            __m128i bit_mask = _mm_set_epi8(
                static_cast<char>(0x80), 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
                static_cast<char>(0x80), 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);

            // Extract each byte of qh
            uint8_t qh_byte0 = static_cast<uint8_t>(qh);       // bits 0-7
            uint8_t qh_byte1 = static_cast<uint8_t>(qh >> 8);  // bits 8-15
            uint8_t qh_byte2 = static_cast<uint8_t>(qh >> 16); // bits 16-23
            uint8_t qh_byte3 = static_cast<uint8_t>(qh >> 24); // bits 24-31

            // Broadcast bytes for low nibbles (output 0-15 need qh bits 0-15)
            __m128i qh_lo = _mm_set_epi8(
                qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1, qh_byte1,
                qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0, qh_byte0);

            // Broadcast bytes for high nibbles (output 16-31 need qh bits 16-31)
            __m128i qh_hi = _mm_set_epi8(
                qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3, qh_byte3,
                qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2, qh_byte2);

            // Test each bit position
            __m128i bits_lo = _mm_and_si128(qh_lo, bit_mask);
            __m128i bits_hi = _mm_and_si128(qh_hi, bit_mask);

            // Compare to get 0xFF where bit was set, 0x00 where not
            bits_lo = _mm_cmpeq_epi8(bits_lo, bit_mask);
            bits_hi = _mm_cmpeq_epi8(bits_hi, bit_mask);

            // Convert 0xFF to 0x10 (bit 4 position)
            __m128i pos4_mask = _mm_set1_epi8(0x10);
            bits_lo = _mm_and_si128(bits_lo, pos4_mask);
            bits_hi = _mm_and_si128(bits_hi, pos4_mask);

            // Combine nibbles with high bits (no offset for Q5_1 - asymmetric [0,31])
            __m128i result_lo = _mm_or_si128(low_nibbles, bits_lo);
            __m128i result_hi = _mm_or_si128(high_nibbles, bits_hi);

            // Store results
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), result_lo);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), result_hi);
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
         * @brief Unpack entire Q5_K super-block to int8 (256 elements)
         *
         * Optimized version that processes all 8 sub-blocks with single CPU dispatch.
         *
         * @param block Source Q5_K super-block
         * @param output Output buffer for 256 int8 values
         * @param scales Optional output buffer for 8 float scales
         * @param mins Optional output buffer for 8 float mins
         */
        inline void unpack_q5_k_superblock_to_int8(
            const Q5_KBlock &block,
            int8_t *output,
            float *scales,
            float *mins)
        {
#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
            {
                // Load high bits once (32 bytes) - contains high bits for all 8 sub-blocks
                __m256i v_qh = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qh));
                __m256i v_mask_0F = _mm256_set1_epi8(0x0F);
                __m256i v_16 = _mm256_set1_epi8(16);
                __m256i v_zero = _mm256_setzero_si256();

                // Process 4 pairs of sub-blocks (8 sub-blocks total)
                // Each iteration processes sub-blocks 2*i and 2*i+1
                for (int i = 0; i < 4; ++i)
                {
                    // Load 32 bytes of qs (contains sub-blocks 2*i and 2*i+1)
                    // qs has 128 bytes total. 32 bytes = 64 nibbles = 64 elements = 2 sub-blocks.
                    const uint8_t *qs_ptr = block.qs + i * 32;
                    __m256i v_qs = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qs_ptr));

                    // Sub-block 2*i (Low nibbles)
                    {
                        int sb_idx = 2 * i;
                        __m256i v_q_val = _mm256_and_si256(v_qs, v_mask_0F);

                        // High bit extraction
                        __m256i v_bit_mask = _mm256_set1_epi8(1 << sb_idx);
                        __m256i v_high_bit = _mm256_and_si256(v_qh, v_bit_mask);
                        __m256i v_is_zero = _mm256_cmpeq_epi8(v_high_bit, v_zero);
                        __m256i v_add_16 = _mm256_andnot_si256(v_is_zero, v_16);
                        v_q_val = _mm256_add_epi8(v_q_val, v_add_16);

                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + sb_idx * 32), v_q_val);

                        if (scales || mins)
                        {
                            float s, m;
                            get_q5_k_scale_min(block, sb_idx, &s, &m);
                            if (scales)
                                scales[sb_idx] = s;
                            if (mins)
                                mins[sb_idx] = -m;
                        }
                    }

                    // Sub-block 2*i+1 (High nibbles)
                    {
                        int sb_idx = 2 * i + 1;
                        // Shift right by 4 bits (operating on 16-bit words, then masking)
                        __m256i v_q_val = _mm256_and_si256(_mm256_srli_epi16(v_qs, 4), v_mask_0F);

                        // High bit extraction
                        __m256i v_bit_mask = _mm256_set1_epi8(1 << sb_idx);
                        __m256i v_high_bit = _mm256_and_si256(v_qh, v_bit_mask);
                        __m256i v_is_zero = _mm256_cmpeq_epi8(v_high_bit, v_zero);
                        __m256i v_add_16 = _mm256_andnot_si256(v_is_zero, v_16);
                        v_q_val = _mm256_add_epi8(v_q_val, v_add_16);

                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + sb_idx * 32), v_q_val);

                        if (scales || mins)
                        {
                            float s, m;
                            get_q5_k_scale_min(block, sb_idx, &s, &m);
                            if (scales)
                                scales[sb_idx] = s;
                            if (mins)
                                mins[sb_idx] = -m;
                        }
                    }
                }
                return;
            }
#endif

            // Scalar fallback
            for (int i = 0; i < 8; ++i)
            {
                unpack_q5_k_to_int8_scalar(block, i, output + i * 32);
                if (scales || mins)
                {
                    float s, m;
                    get_q5_k_scale_min(block, i, &s, &m);
                    if (scales)
                        scales[i] = s;
                    if (mins)
                        mins[i] = -m;
                }
            }
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
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
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
         * @brief Unpack entire IQ4_XS superblock (256 elements) to INT8
         *
         * Optimized implementation that processes all 8 sub-blocks at once.
         * Also computes the 8 scale factors.
         *
         * @param block IQ4_XS super-block
         * @param output Output buffer (256 int8_t values)
         * @param scales Output buffer (8 float values), optional
         */
        inline void unpack_iq4_xs_superblock_to_int8(
            const IQ4_XSBlock &block,
            int8_t *output,
            float *scales = nullptr)
        {
#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
            {
                // Load lookup table
                __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl_i8));
                __m128i low_mask = _mm_set1_epi8(0x0F);

                const __m128i *qs_ptr = reinterpret_cast<const __m128i *>(block.qs);
                __m128i *out_ptr = reinterpret_cast<__m128i *>(output);

                // Unroll loop for 8 sub-blocks (128 bytes input -> 8 x 16 bytes)
                // Each 16-byte input produces 32 bytes output (2 x 16 bytes)
                for (int i = 0; i < 8; ++i)
                {
                    __m128i qs = _mm_loadu_si128(qs_ptr + i);

                    __m128i low_indices = _mm_and_si128(qs, low_mask);
                    __m128i high_indices = _mm_and_si128(_mm_srli_epi16(qs, 4), low_mask);

                    __m128i low_vals = _mm_shuffle_epi8(lut, low_indices);
                    __m128i high_vals = _mm_shuffle_epi8(lut, high_indices);

                    _mm_storeu_si128(out_ptr + 2 * i, low_vals);
                    _mm_storeu_si128(out_ptr + 2 * i + 1, high_vals);
                }
            }
            else
#endif
            {
                // Scalar fallback
                const uint8_t *qs_ptr = block.qs;
                for (int i = 0; i < 8; ++i)
                {
                    int8_t *sub_out = output + i * 32;
                    const uint8_t *sub_qs = qs_ptr + i * 16;
                    for (int j = 0; j < 16; ++j)
                    {
                        uint8_t val = sub_qs[j];
                        sub_out[j] = kvalues_iq4nl_i8[val & 0x0F];
                        sub_out[j + 16] = kvalues_iq4nl_i8[val >> 4];
                    }
                }
            }

            // Compute scales if requested
            if (scales)
            {
                const float d = fp16_to_fp32(block.d);

                // Scalar scale computation (fast enough for 8 values)
                for (int i = 0; i < 8; ++i)
                {
                    int ls = ((block.scales_l[i / 2] >> 4 * (i % 2)) & 0xf) |
                             (((block.scales_h >> 2 * i) & 3) << 4);
                    scales[i] = d * (ls - 32);
                }
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

        /**
         * @brief Decode IQ3_S sub-block to FP32 (auto-dispatch)
         */
        inline void decode_iq3s_subblock_to_fp32(
            const IQ3_SBlock &block,
            size_t subblock_idx,
            float *output)
        {
#if defined(__AVX512F__)
            static const bool has_avx512 = cpu_supports_avx512();
            if (has_avx512)
            {
                decode_iq3s_subblock_to_fp32_avx512(block, subblock_idx, output);
                return;
            }
#endif
#if defined(__AVX2__)
            static const bool has_avx2 = cpu_supports_avx2();
            if (has_avx2)
            {
                decode_iq3s_subblock_to_fp32_avx2(block, subblock_idx, output);
                return;
            }
#endif
            decode_iq3s_subblock_to_fp32_scalar(block, subblock_idx, output);
        }

        // ==========================================
        // Q16_1 Block Decode to FP32
        // ==========================================

        /**
         * @brief Decode Q16_1 block to FP32 (scalar)
         *
         * Q16_1 format: 32 int16 values + FP32 scale
         * Decodes to 32 FP32 values: output[i] = scale * qs[i]
         *
         * @param block Q16_1 block to decode
         * @param output FP32 output buffer (must hold 32 floats)
         */
        inline void decode_q16_1_block_to_fp32_scalar(const Q16_1Block &block, float *output)
        {
            const float scale = block.d; // FP32 scale, no conversion needed
            for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                output[i] = scale * static_cast<float>(block.qs[i]);
            }
        }

#if defined(__AVX512F__)
        /**
         * @brief Decode Q16_1 block to FP32 (AVX-512)
         *
         * Uses AVX-512 to process 32 int16 values in a single pass (2 iterations unrolled).
         * Leverages ILP by interleaving loads, converts, and stores.
         *
         * @param block Q16_1 block to decode
         * @param output FP32 output buffer (must hold 32 floats)
         */
        inline void decode_q16_1_block_to_fp32_avx512(const Q16_1Block &block, float *output)
        {
            const float scale = block.d; // FP32 scale, no conversion needed
            const __m512 vscale = _mm512_set1_ps(scale);

            // Load 32 int16 values (2 x 128-bit loads -> 256 bits total)
            // Note: qs is int16_t[32], so 64 bytes total.
            // We load 16 int16s (32 bytes) at a time into YMM registers.
            const __m256i *qs_ptr = reinterpret_cast<const __m256i *>(block.qs);
            __m256i vi16_0 = _mm256_loadu_si256(qs_ptr);
            __m256i vi16_1 = _mm256_loadu_si256(qs_ptr + 1);

            // Convert int16 -> int32 (sign extension)
            __m512i vi32_0 = _mm512_cvtepi16_epi32(vi16_0);
            __m512i vi32_1 = _mm512_cvtepi16_epi32(vi16_1);

            // Convert int32 -> float
            __m512 vf_0 = _mm512_cvtepi32_ps(vi32_0);
            __m512 vf_1 = _mm512_cvtepi32_ps(vi32_1);

            // Scale
            vf_0 = _mm512_mul_ps(vf_0, vscale);
            vf_1 = _mm512_mul_ps(vf_1, vscale);

            // Store result
            _mm512_storeu_ps(output, vf_0);
            _mm512_storeu_ps(output + 16, vf_1);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Decode Q16_1 block to FP32 (AVX2)
         *
         * Uses AVX2 to process 32 int16 values in a single pass (4 iterations unrolled).
         * Leverages ILP by interleaving loads, converts, and stores.
         *
         * @param block Q16_1 block to decode
         * @param output FP32 output buffer (must hold 32 floats)
         */
        inline void decode_q16_1_block_to_fp32_avx2(const Q16_1Block &block, float *output)
        {
            const float scale = block.d; // FP32 scale, no conversion needed
            const __m256 vscale = _mm256_set1_ps(scale);

            const __m128i *qs_ptr = reinterpret_cast<const __m128i *>(block.qs);

            // Load 4 chunks of 8 int16s (128 bits each)
            __m128i vi16_0 = _mm_loadu_si128(qs_ptr);
            __m128i vi16_1 = _mm_loadu_si128(qs_ptr + 1);
            __m128i vi16_2 = _mm_loadu_si128(qs_ptr + 2);
            __m128i vi16_3 = _mm_loadu_si128(qs_ptr + 3);

            // Convert int16 -> int32 (sign extension)
            __m256i vi32_0 = _mm256_cvtepi16_epi32(vi16_0);
            __m256i vi32_1 = _mm256_cvtepi16_epi32(vi16_1);
            __m256i vi32_2 = _mm256_cvtepi16_epi32(vi16_2);
            __m256i vi32_3 = _mm256_cvtepi16_epi32(vi16_3);

            // Convert int32 -> float
            __m256 vf_0 = _mm256_cvtepi32_ps(vi32_0);
            __m256 vf_1 = _mm256_cvtepi32_ps(vi32_1);
            __m256 vf_2 = _mm256_cvtepi32_ps(vi32_2);
            __m256 vf_3 = _mm256_cvtepi32_ps(vi32_3);

            // Scale
            vf_0 = _mm256_mul_ps(vf_0, vscale);
            vf_1 = _mm256_mul_ps(vf_1, vscale);
            vf_2 = _mm256_mul_ps(vf_2, vscale);
            vf_3 = _mm256_mul_ps(vf_3, vscale);

            // Store result
            _mm256_storeu_ps(output, vf_0);
            _mm256_storeu_ps(output + 8, vf_1);
            _mm256_storeu_ps(output + 16, vf_2);
            _mm256_storeu_ps(output + 24, vf_3);
        }
#endif

        /**
         * @brief Decode Q16_1 block to FP32 (auto-dispatch)
         *
         * Automatically selects the best available SIMD implementation.
         *
         * @param block Q16_1 block to decode
         * @param output FP32 output buffer (must hold 32 floats)
         */
        inline void decode_q16_1_block_to_fp32(const Q16_1Block &block, float *output)
        {
#if defined(__AVX512F__)
            decode_q16_1_block_to_fp32_avx512(block, output);
#elif defined(__AVX2__)
            decode_q16_1_block_to_fp32_avx2(block, output);
#else
            decode_q16_1_block_to_fp32_scalar(block, output);
#endif
        }

        // ==========================================
        // Q16 Block Decode to FP32 (Templated for Variable Block Sizes)
        // ==========================================

        /**
         * @brief Decode Q16 block to FP32 (scalar, templated)
         *
         * Works with any Q16 block type (Q16_1Block, Q16_1Block_64, Q16_1Block_128).
         * Formula: output[i] = scale * qs[i]
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, qs members
         * @param block Q16 block to decode
         * @param output FP32 output buffer (must hold BLOCK_SIZE floats)
         */
        template <typename BlockType>
        inline void decode_q16_block_to_fp32_scalar(const BlockType &block, float *output)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            const float scale = block.d;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                output[i] = scale * static_cast<float>(block.qs[i]);
            }
        }

#if defined(__AVX512F__)
        /**
         * @brief Decode Q16 block to FP32 (AVX-512, templated)
         *
         * Uses AVX-512 to process 32 int16 values at a time with ILP optimization.
         * Processes in 32-element chunks, handles any block size (32, 64, 128).
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, qs members
         * @param block Q16 block to decode
         * @param output FP32 output buffer (must hold BLOCK_SIZE floats)
         */
        template <typename BlockType>
        inline void decode_q16_block_to_fp32_avx512(const BlockType &block, float *output)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_32 = BLOCK_SIZE / 32;

            const float scale = block.d;
            const __m512 vscale = _mm512_set1_ps(scale);

            const __m256i *qs_ptr = reinterpret_cast<const __m256i *>(block.qs);

            // Simple sequential processing - memory-bound operations benefit from
            // sequential access patterns that the HW prefetcher can follow
            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                // Load 32 int16s (2 x 256-bit = 64 bytes)
                __m256i vi16_0 = _mm256_loadu_si256(qs_ptr + chunk * 2);
                __m256i vi16_1 = _mm256_loadu_si256(qs_ptr + chunk * 2 + 1);

                // Convert int16 -> int32 (sign extension)
                __m512i vi32_0 = _mm512_cvtepi16_epi32(vi16_0);
                __m512i vi32_1 = _mm512_cvtepi16_epi32(vi16_1);

                // Convert int32 -> float
                __m512 vf_0 = _mm512_cvtepi32_ps(vi32_0);
                __m512 vf_1 = _mm512_cvtepi32_ps(vi32_1);

                // Scale
                vf_0 = _mm512_mul_ps(vf_0, vscale);
                vf_1 = _mm512_mul_ps(vf_1, vscale);

                // Store sequentially - allows HW prefetcher to work effectively
                _mm512_storeu_ps(output + chunk * 32, vf_0);
                _mm512_storeu_ps(output + chunk * 32 + 16, vf_1);
            }
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Decode Q16 block to FP32 (AVX2, templated)
         *
         * Uses AVX2 to process 8 int16 values at a time with ILP optimization.
         * Processes in 16-element chunks (2 x 8), handles any block size.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, qs members
         * @param block Q16 block to decode
         * @param output FP32 output buffer (must hold BLOCK_SIZE floats)
         */
        template <typename BlockType>
        inline void decode_q16_block_to_fp32_avx2(const BlockType &block, float *output)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_16 = BLOCK_SIZE / 16;

            const float scale = block.d;
            const __m256 vscale = _mm256_set1_ps(scale);

            const __m128i *qs_ptr = reinterpret_cast<const __m128i *>(block.qs);

            size_t chunk = 0;

            // Process 2 chunks (32 elements) per iteration for maximum ILP
            for (; chunk + 1 < CHUNKS_16; chunk += 2)
            {
                // Interleaved loads (4 x 8 int16s = 32 values)
                __m128i vi16_0a = _mm_loadu_si128(qs_ptr + chunk * 2);
                __m128i vi16_1a = _mm_loadu_si128(qs_ptr + (chunk + 1) * 2);
                __m128i vi16_0b = _mm_loadu_si128(qs_ptr + chunk * 2 + 1);
                __m128i vi16_1b = _mm_loadu_si128(qs_ptr + (chunk + 1) * 2 + 1);

                // Convert int16 -> int32 (interleaved)
                __m256i vi32_0a = _mm256_cvtepi16_epi32(vi16_0a);
                __m256i vi32_1a = _mm256_cvtepi16_epi32(vi16_1a);
                __m256i vi32_0b = _mm256_cvtepi16_epi32(vi16_0b);
                __m256i vi32_1b = _mm256_cvtepi16_epi32(vi16_1b);

                // Convert int32 -> float (interleaved)
                __m256 vf_0a = _mm256_cvtepi32_ps(vi32_0a);
                __m256 vf_1a = _mm256_cvtepi32_ps(vi32_1a);
                __m256 vf_0b = _mm256_cvtepi32_ps(vi32_0b);
                __m256 vf_1b = _mm256_cvtepi32_ps(vi32_1b);

                // Scale (interleaved)
                vf_0a = _mm256_mul_ps(vf_0a, vscale);
                vf_1a = _mm256_mul_ps(vf_1a, vscale);
                vf_0b = _mm256_mul_ps(vf_0b, vscale);
                vf_1b = _mm256_mul_ps(vf_1b, vscale);

                // Store results (interleaved)
                _mm256_storeu_ps(output + chunk * 16, vf_0a);
                _mm256_storeu_ps(output + (chunk + 1) * 16, vf_1a);
                _mm256_storeu_ps(output + chunk * 16 + 8, vf_0b);
                _mm256_storeu_ps(output + (chunk + 1) * 16 + 8, vf_1b);
            }

            // Handle remaining single chunk
            for (; chunk < CHUNKS_16; ++chunk)
            {
                __m128i vi16_0 = _mm_loadu_si128(qs_ptr + chunk * 2);
                __m128i vi16_1 = _mm_loadu_si128(qs_ptr + chunk * 2 + 1);

                __m256i vi32_0 = _mm256_cvtepi16_epi32(vi16_0);
                __m256i vi32_1 = _mm256_cvtepi16_epi32(vi16_1);

                __m256 vf_0 = _mm256_cvtepi32_ps(vi32_0);
                __m256 vf_1 = _mm256_cvtepi32_ps(vi32_1);

                vf_0 = _mm256_mul_ps(vf_0, vscale);
                vf_1 = _mm256_mul_ps(vf_1, vscale);

                _mm256_storeu_ps(output + chunk * 16, vf_0);
                _mm256_storeu_ps(output + chunk * 16 + 8, vf_1);
            }
        }
#endif

        /**
         * @brief Decode Q16 block to FP32 (auto-dispatch, templated)
         *
         * Automatically selects the best available SIMD implementation.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, qs members
         * @param block Q16 block to decode
         * @param output FP32 output buffer (must hold BLOCK_SIZE floats)
         */
        template <typename BlockType>
        inline void decode_q16_block_to_fp32(const BlockType &block, float *output)
        {
#if defined(__AVX512F__)
            decode_q16_block_to_fp32_avx512<BlockType>(block, output);
#elif defined(__AVX2__)
            decode_q16_block_to_fp32_avx2<BlockType>(block, output);
#else
            decode_q16_block_to_fp32_scalar<BlockType>(block, output);
#endif
        }

        // ==========================================
        // Q16_1 Block Quantization from FP32
        // ==========================================

        /**
         * @brief Quantize FP32 block to Q16_1 (scalar)
         *
         * Quantizes 32 FP32 values to a Q16_1 block:
         * 1. Find max_abs in block
         * 2. Compute scale: d = max_abs / 32767
         * 3. Quantize: qs[i] = round(src[i] / d) clamped to [-32767, 32767]
         * 4. Compute sum_qs = sum of all qs[i]
         *
         * @param src FP32 input buffer (must hold 32 floats)
         * @param block Q16_1 output block
         */
        inline void quantize_fp32_to_q16_1_block_scalar(const float *src, Q16_1Block &block)
        {
            // Find max absolute value
            float max_abs = 0.0f;
            for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Compute scale factor (d = max_abs / 32767 for int16 range)
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            // Quantize and compute sum
            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
                {
                    const float scaled = src[i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    // Use std::nearbyint for consistent rounding with SIMD (round to nearest, ties to even)
                    block.qs[i] = static_cast<int16_t>(std::nearbyint(clamped));
                    sum_i64 += static_cast<int64_t>(block.qs[i]);
                }
            }
            else
            {
                for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

#if defined(__AVX512F__)
        /**
         * @brief Quantize FP32 block to Q16_1 (AVX-512)
         *
         * Uses AVX-512 to process 32 FP32 values efficiently:
         * - 2 ZMM registers for 32 floats
         * - Vectorized max_abs, quantization, and sum computation
         *
         * @param src FP32 input buffer (must hold 32 floats, preferably 64-byte aligned)
         * @param block Q16_1 output block
         */
        inline void quantize_fp32_to_q16_1_block_avx512(const float *src, Q16_1Block &block)
        {
            // Load 32 floats (2 x 16)
            __m512 v0 = _mm512_loadu_ps(src);
            __m512 v1 = _mm512_loadu_ps(src + 16);

            // Compute absolute values
            const __m512 sign_mask = _mm512_set1_ps(-0.0f);
            __m512 abs0 = _mm512_andnot_ps(sign_mask, v0);
            __m512 abs1 = _mm512_andnot_ps(sign_mask, v1);

            // Find max absolute value
            __m512 max_vec = _mm512_max_ps(abs0, abs1);
            float max_abs = _mm512_reduce_max_ps(max_vec);

            // Compute scale
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                const __m512 vinv_d = _mm512_set1_ps(inv_d);
                const __m512i vmin = _mm512_set1_epi32(-32767);
                const __m512i vmax = _mm512_set1_epi32(32767);

                // Scale and round to nearest
                __m512 scaled0 = _mm512_mul_ps(v0, vinv_d);
                __m512 scaled1 = _mm512_mul_ps(v1, vinv_d);

                // Convert to int32 with rounding
                __m512i i32_0 = _mm512_cvtps_epi32(scaled0);
                __m512i i32_1 = _mm512_cvtps_epi32(scaled1);

                // Clamp to [-32767, 32767]
                i32_0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_0));
                i32_1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_1));

                // Pack int32 -> int16 (saturating)
                // _mm512_cvtepi32_epi16 truncates 32-bit to 16-bit (lower 16 bits)
                __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
                __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);

                // Store int16 values
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs), i16_0);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + 16), i16_1);

                // Compute sum: add all int32 values before packing
                // (more accurate than summing int16 after potential saturation)
                int32_t sum0 = _mm512_reduce_add_epi32(i32_0);
                int32_t sum1 = _mm512_reduce_add_epi32(i32_1);
                block.sum_qs = sum0 + sum1;
            }
            else
            {
                // Zero block
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs), _mm256_setzero_si256());
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + 16), _mm256_setzero_si256());
                block.sum_qs = 0;
            }
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Quantize FP32 block to Q16_1 (AVX2)
         *
         * Uses AVX2 to process 32 FP32 values efficiently:
         * - 4 YMM registers for 32 floats
         * - Vectorized max_abs, quantization, and sum computation
         *
         * @param src FP32 input buffer (must hold 32 floats, preferably 32-byte aligned)
         * @param block Q16_1 output block
         */
        inline void quantize_fp32_to_q16_1_block_avx2(const float *src, Q16_1Block &block)
        {
            // Load 32 floats (4 x 8)
            __m256 v0 = _mm256_loadu_ps(src);
            __m256 v1 = _mm256_loadu_ps(src + 8);
            __m256 v2 = _mm256_loadu_ps(src + 16);
            __m256 v3 = _mm256_loadu_ps(src + 24);

            // Compute absolute values
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 abs0 = _mm256_andnot_ps(sign_mask, v0);
            __m256 abs1 = _mm256_andnot_ps(sign_mask, v1);
            __m256 abs2 = _mm256_andnot_ps(sign_mask, v2);
            __m256 abs3 = _mm256_andnot_ps(sign_mask, v3);

            // Find max absolute value (tree reduction)
            __m256 max01 = _mm256_max_ps(abs0, abs1);
            __m256 max23 = _mm256_max_ps(abs2, abs3);
            __m256 max_vec = _mm256_max_ps(max01, max23);

            // Horizontal max within 256-bit register
            __m128 lo = _mm256_castps256_ps128(max_vec);
            __m128 hi = _mm256_extractf128_ps(max_vec, 1);
            __m128 max128 = _mm_max_ps(lo, hi);
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(max128);

            // Compute scale
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                const __m256 vinv_d = _mm256_set1_ps(inv_d);
                const __m256i vmin = _mm256_set1_epi32(-32767);
                const __m256i vmax = _mm256_set1_epi32(32767);

                // Scale
                __m256 scaled0 = _mm256_mul_ps(v0, vinv_d);
                __m256 scaled1 = _mm256_mul_ps(v1, vinv_d);
                __m256 scaled2 = _mm256_mul_ps(v2, vinv_d);
                __m256 scaled3 = _mm256_mul_ps(v3, vinv_d);

                // Round and convert to int32
                __m256i i32_0 = _mm256_cvtps_epi32(scaled0);
                __m256i i32_1 = _mm256_cvtps_epi32(scaled1);
                __m256i i32_2 = _mm256_cvtps_epi32(scaled2);
                __m256i i32_3 = _mm256_cvtps_epi32(scaled3);

                // Clamp to [-32767, 32767]
                i32_0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_0));
                i32_1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_1));
                i32_2 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_2));
                i32_3 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_3));

                // Compute sum before packing (more accurate)
                // Horizontal sum of each 256-bit register
                auto hsum_epi32 = [](__m256i v) -> int32_t
                {
                    __m128i lo = _mm256_castsi256_si128(v);
                    __m128i hi = _mm256_extracti128_si256(v, 1);
                    __m128i sum = _mm_add_epi32(lo, hi);
                    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
                    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(0, 0, 0, 1)));
                    return _mm_cvtsi128_si32(sum);
                };
                int32_t sum = hsum_epi32(i32_0) + hsum_epi32(i32_1) + hsum_epi32(i32_2) + hsum_epi32(i32_3);
                block.sum_qs = sum;

                // Pack int32 -> int16 (pairs of 8 -> 16)
                // _mm256_packs_epi32 saturates and interleaves, need permute to fix order
                __m256i packed01 = _mm256_packs_epi32(i32_0, i32_1); // [0-3,8-11,4-7,12-15]
                __m256i packed23 = _mm256_packs_epi32(i32_2, i32_3);

                // Fix lane ordering: permute 64-bit chunks
                packed01 = _mm256_permute4x64_epi64(packed01, _MM_SHUFFLE(3, 1, 2, 0));
                packed23 = _mm256_permute4x64_epi64(packed23, _MM_SHUFFLE(3, 1, 2, 0));

                // Store
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs), packed01);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + 16), packed23);
            }
            else
            {
                // Zero block
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs), _mm256_setzero_si256());
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + 16), _mm256_setzero_si256());
                block.sum_qs = 0;
            }
        }
#endif

        /**
         * @brief Quantize FP32 block to Q16_1 (auto-dispatch)
         *
         * Automatically selects the best available SIMD implementation.
         *
         * @param src FP32 input buffer (must hold 32 floats)
         * @param block Q16_1 output block
         */
        inline void quantize_fp32_to_q16_1_block(const float *src, Q16_1Block &block)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q16_1_block_avx512(src, block);
#elif defined(__AVX2__)
            quantize_fp32_to_q16_1_block_avx2(src, block);
#else
            quantize_fp32_to_q16_1_block_scalar(src, block);
#endif
        }

        // =====================================================================
        // Partial Block Quantization (< 32 elements with zero-fill)
        // =====================================================================

        /**
         * @brief Quantize partial FP32 block to Q16_1 (scalar)
         *
         * Handles blocks with fewer than 32 elements, zero-filling the remainder.
         *
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements (1-31)
         * @param block Q16_1 output block
         */
        inline void quantize_fp32_to_q16_1_block_partial_scalar(const float *src, size_t count, Q16_1Block &block)
        {
            // Find max absolute value in partial data
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Compute scale factor
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            // Quantize valid elements and compute sum
            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = src[i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    block.qs[i] = static_cast<int16_t>(std::nearbyint(clamped));
                    sum_i64 += static_cast<int64_t>(block.qs[i]);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            // Zero-fill remainder
            for (size_t i = count; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

#if defined(__AVX512F__)
        /**
         * @brief Quantize partial FP32 block to Q16_1 (AVX-512)
         *
         * Uses masked AVX-512 operations to handle partial blocks efficiently.
         * Processes up to 32 elements with mask-based loads/stores.
         *
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements (1-31)
         * @param block Q16_1 output block
         */
        inline void quantize_fp32_to_q16_1_block_partial_avx512(const float *src, size_t count, Q16_1Block &block)
        {
            // Create masks for valid elements
            const __mmask16 mask0 = (count >= 16) ? 0xFFFF : static_cast<__mmask16>((1U << count) - 1);
            const __mmask16 mask1 = (count <= 16) ? 0 : static_cast<__mmask16>((1U << (count - 16)) - 1);

            // Masked load of valid elements (zeros where invalid)
            __m512 v0 = _mm512_maskz_loadu_ps(mask0, src);
            __m512 v1 = _mm512_maskz_loadu_ps(mask1, src + 16);

            // Compute absolute values
            const __m512 sign_mask = _mm512_set1_ps(-0.0f);
            __m512 abs0 = _mm512_andnot_ps(sign_mask, v0);
            __m512 abs1 = _mm512_andnot_ps(sign_mask, v1);

            // Find max absolute value (only from valid elements via mask)
            float max0 = (mask0 != 0) ? _mm512_mask_reduce_max_ps(mask0, abs0) : 0.0f;
            float max1 = (mask1 != 0) ? _mm512_mask_reduce_max_ps(mask1, abs1) : 0.0f;
            float max_abs = std::max(max0, max1);

            // Compute scale
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                const __m512 vinv_d = _mm512_set1_ps(inv_d);
                const __m512i vmin = _mm512_set1_epi32(-32767);
                const __m512i vmax = _mm512_set1_epi32(32767);

                // Scale and round
                __m512 scaled0 = _mm512_mul_ps(v0, vinv_d);
                __m512 scaled1 = _mm512_mul_ps(v1, vinv_d);

                // Convert to int32 with rounding
                __m512i i32_0 = _mm512_cvtps_epi32(scaled0);
                __m512i i32_1 = _mm512_cvtps_epi32(scaled1);

                // Clamp
                i32_0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_0));
                i32_1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_1));

                // Pack int32 -> int16
                __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
                __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);

                // Store (full stores are fine since we'll have zeros in invalid positions)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs), i16_0);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + 16), i16_1);

                // Compute sum only from valid elements
                int32_t sum0 = (mask0 != 0) ? _mm512_mask_reduce_add_epi32(mask0, i32_0) : 0;
                int32_t sum1 = (mask1 != 0) ? _mm512_mask_reduce_add_epi32(mask1, i32_1) : 0;
                block.sum_qs = sum0 + sum1;
            }
            else
            {
                // Zero entire block
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs), _mm256_setzero_si256());
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + 16), _mm256_setzero_si256());
                block.sum_qs = 0;
            }
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Quantize partial FP32 block to Q16_1 (AVX2)
         *
         * Processes as many full 8-float chunks as possible with SIMD,
         * then handles remainder with scalar code.
         *
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements (1-31)
         * @param block Q16_1 output block
         */
        inline void quantize_fp32_to_q16_1_block_partial_avx2(const float *src, size_t count, Q16_1Block &block)
        {
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);

            // Find max absolute value using SIMD where possible
            __m256 max_vec = _mm256_setzero_ps();
            size_t i = 0;

            // Process full 8-float chunks
            for (; i + 8 <= count; i += 8)
            {
                __m256 v = _mm256_loadu_ps(src + i);
                __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
                max_vec = _mm256_max_ps(max_vec, abs_v);
            }

            // Horizontal max of SIMD result
            __m128 lo = _mm256_castps256_ps128(max_vec);
            __m128 hi = _mm256_extractf128_ps(max_vec, 1);
            __m128 max128 = _mm_max_ps(lo, hi);
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(max128);

            // Process remaining elements with scalar
            for (; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Compute scale
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                const __m256 vinv_d = _mm256_set1_ps(inv_d);
                const __m256i vmin = _mm256_set1_epi32(-32767);
                const __m256i vmax = _mm256_set1_epi32(32767);

                int32_t sum = 0;
                i = 0;

                // Lambda for horizontal sum
                auto hsum_epi32 = [](__m256i v) -> int32_t
                {
                    __m128i lo = _mm256_castsi256_si128(v);
                    __m128i hi = _mm256_extracti128_si256(v, 1);
                    __m128i s = _mm_add_epi32(lo, hi);
                    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
                    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(0, 0, 0, 1)));
                    return _mm_cvtsi128_si32(s);
                };

                // Process 16 elements at a time (2 YMM -> pack to 1 YMM of int16)
                for (; i + 16 <= count; i += 16)
                {
                    __m256 v0 = _mm256_loadu_ps(src + i);
                    __m256 v1 = _mm256_loadu_ps(src + i + 8);

                    __m256 scaled0 = _mm256_mul_ps(v0, vinv_d);
                    __m256 scaled1 = _mm256_mul_ps(v1, vinv_d);

                    __m256i i32_0 = _mm256_cvtps_epi32(scaled0);
                    __m256i i32_1 = _mm256_cvtps_epi32(scaled1);

                    i32_0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_0));
                    i32_1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_1));

                    sum += hsum_epi32(i32_0) + hsum_epi32(i32_1);

                    __m256i packed = _mm256_packs_epi32(i32_0, i32_1);
                    packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + i), packed);
                }

                // Process remaining 8 elements if present
                if (i + 8 <= count)
                {
                    __m256 v = _mm256_loadu_ps(src + i);
                    __m256 scaled = _mm256_mul_ps(v, vinv_d);
                    __m256i i32 = _mm256_cvtps_epi32(scaled);
                    i32 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32));
                    sum += hsum_epi32(i32);

                    // Pack with zeros
                    __m256i packed = _mm256_packs_epi32(i32, _mm256_setzero_si256());
                    packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
                    // Store lower 128 bits (8 int16s)
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(block.qs + i),
                                     _mm256_castsi256_si128(packed));
                    i += 8;
                }

                // Process remaining elements with scalar
                for (; i < count; ++i)
                {
                    const float scaled = src[i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    block.qs[i] = static_cast<int16_t>(std::nearbyint(clamped));
                    sum += block.qs[i];
                }

                // Zero-fill remainder
                for (; i < Q16_1Block::BLOCK_SIZE; ++i)
                {
                    block.qs[i] = 0;
                }

                block.sum_qs = sum;
            }
            else
            {
                // Zero entire block
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs), _mm256_setzero_si256());
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + 16), _mm256_setzero_si256());
                block.sum_qs = 0;
            }
        }
#endif

        /**
         * @brief Quantize partial FP32 block to Q16_1 (auto-dispatch)
         *
         * Automatically selects the best available SIMD implementation.
         *
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements (1-31)
         * @param block Q16_1 output block
         */
        inline void quantize_fp32_to_q16_1_block_partial(const float *src, size_t count, Q16_1Block &block)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q16_1_block_partial_avx512(src, count, block);
#elif defined(__AVX2__)
            quantize_fp32_to_q16_1_block_partial_avx2(src, count, block);
#else
            quantize_fp32_to_q16_1_block_partial_scalar(src, count, block);
#endif
        }

        // =====================================================================
        // Templated Variable Block Size Quantization (Q16_1Block_64, Q16_1Block_128)
        // =====================================================================

        /**
         * @brief Quantize FP32 to Q16 block (scalar, templated for any block size)
         *
         * Works with Q16_1Block (32), Q16_1Block_64 (64), Q16_1Block_128 (128).
         * Processes full blocks only.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold BlockType::BLOCK_SIZE floats)
         * @param block Output block
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_scalar(const float *src, BlockType &block)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

            // Find max absolute value
            float max_abs = 0.0f;
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Compute scale factor
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            // Quantize and compute sum
            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < BLOCK_SIZE; ++i)
                {
                    const float scaled = src[i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    block.qs[i] = static_cast<int16_t>(std::nearbyint(clamped));
                    sum_i64 += static_cast<int64_t>(block.qs[i]);
                }
            }
            else
            {
                for (size_t i = 0; i < BLOCK_SIZE; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

#if defined(__AVX512F__)
        /**
         * @brief Quantize FP32 to Q16 block (AVX-512, templated for any block size)
         *
         * Uses AVX-512 to process in 32-element chunks (2 ZMM registers).
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold BlockType::BLOCK_SIZE floats)
         * @param block Output block
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_avx512(const float *src, BlockType &block)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_32 = BLOCK_SIZE / 32;

            // Find max absolute value across all chunks
            const __m512 sign_mask = _mm512_set1_ps(-0.0f);
            __m512 max_vec = _mm512_setzero_ps();

            for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
            {
                __m512 v0 = _mm512_loadu_ps(src + chunk * 32);
                __m512 v1 = _mm512_loadu_ps(src + chunk * 32 + 16);
                max_vec = _mm512_max_ps(max_vec, _mm512_andnot_ps(sign_mask, v0));
                max_vec = _mm512_max_ps(max_vec, _mm512_andnot_ps(sign_mask, v1));
            }

            float max_abs = _mm512_reduce_max_ps(max_vec);

            // Compute scale
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                const __m512 vinv_d = _mm512_set1_ps(inv_d);
                const __m512i vmin = _mm512_set1_epi32(-32767);
                const __m512i vmax = _mm512_set1_epi32(32767);

                __m512i sum_vec = _mm512_setzero_si512();

                for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
                {
                    __m512 v0 = _mm512_loadu_ps(src + chunk * 32);
                    __m512 v1 = _mm512_loadu_ps(src + chunk * 32 + 16);

                    // Scale and round
                    __m512 scaled0 = _mm512_mul_ps(v0, vinv_d);
                    __m512 scaled1 = _mm512_mul_ps(v1, vinv_d);

                    // Convert to int32
                    __m512i i32_0 = _mm512_cvtps_epi32(scaled0);
                    __m512i i32_1 = _mm512_cvtps_epi32(scaled1);

                    // Clamp
                    i32_0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_0));
                    i32_1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_1));

                    // Accumulate sum (vectorized)
                    sum_vec = _mm512_add_epi32(sum_vec, i32_0);
                    sum_vec = _mm512_add_epi32(sum_vec, i32_1);

                    // Pack to int16
                    __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
                    __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);

                    // Combine and store 512 bits (32 int16s)
                    __m512i combined = _mm512_castsi256_si512(i16_0);
                    combined = _mm512_inserti64x4(combined, i16_1, 1);
                    _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + chunk * 32), combined);
                }

                block.sum_qs = _mm512_reduce_add_epi32(sum_vec);
            }
            else
            {
                // Zero entire block
                const __m512i zero = _mm512_setzero_si512();
                for (size_t chunk = 0; chunk < CHUNKS_32; ++chunk)
                {
                    _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + chunk * 32), zero);
                }
                block.sum_qs = 0;
            }
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Quantize FP32 to Q16 block (AVX2, templated for any block size)
         *
         * Uses AVX2 to process in 16-element chunks (2 YMM -> 1 YMM packed).
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold BlockType::BLOCK_SIZE floats)
         * @param block Output block
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_avx2(const float *src, BlockType &block)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_16 = BLOCK_SIZE / 16;

            // Find max absolute value across all chunks
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 max_vec = _mm256_setzero_ps();

            for (size_t chunk = 0; chunk < CHUNKS_16; ++chunk)
            {
                __m256 v0 = _mm256_loadu_ps(src + chunk * 16);
                __m256 v1 = _mm256_loadu_ps(src + chunk * 16 + 8);
                max_vec = _mm256_max_ps(max_vec, _mm256_andnot_ps(sign_mask, v0));
                max_vec = _mm256_max_ps(max_vec, _mm256_andnot_ps(sign_mask, v1));
            }

            // Horizontal max
            __m128 lo = _mm256_castps256_ps128(max_vec);
            __m128 hi = _mm256_extractf128_ps(max_vec, 1);
            __m128 max128 = _mm_max_ps(lo, hi);
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(max128);

            // Compute scale
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                const __m256 vinv_d = _mm256_set1_ps(inv_d);
                const __m256i vmin = _mm256_set1_epi32(-32767);
                const __m256i vmax = _mm256_set1_epi32(32767);

                // Lambda for horizontal sum
                auto hsum_epi32 = [](__m256i v) -> int32_t
                {
                    __m128i lo = _mm256_castsi256_si128(v);
                    __m128i hi = _mm256_extracti128_si256(v, 1);
                    __m128i s = _mm_add_epi32(lo, hi);
                    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
                    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(0, 0, 0, 1)));
                    return _mm_cvtsi128_si32(s);
                };

                int32_t total_sum = 0;

                for (size_t chunk = 0; chunk < CHUNKS_16; ++chunk)
                {
                    __m256 v0 = _mm256_loadu_ps(src + chunk * 16);
                    __m256 v1 = _mm256_loadu_ps(src + chunk * 16 + 8);

                    // Scale
                    __m256 scaled0 = _mm256_mul_ps(v0, vinv_d);
                    __m256 scaled1 = _mm256_mul_ps(v1, vinv_d);

                    // Convert to int32
                    __m256i i32_0 = _mm256_cvtps_epi32(scaled0);
                    __m256i i32_1 = _mm256_cvtps_epi32(scaled1);

                    // Clamp
                    i32_0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_0));
                    i32_1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_1));

                    // Accumulate sum before packing
                    total_sum += hsum_epi32(i32_0) + hsum_epi32(i32_1);

                    // Pack int32 -> int16
                    __m256i packed = _mm256_packs_epi32(i32_0, i32_1);
                    packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));

                    // Store
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + chunk * 16), packed);
                }

                block.sum_qs = total_sum;
            }
            else
            {
                // Zero entire block
                for (size_t chunk = 0; chunk < CHUNKS_16; ++chunk)
                {
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + chunk * 16), _mm256_setzero_si256());
                }
                block.sum_qs = 0;
            }
        }
#endif

        /**
         * @brief Quantize FP32 to Q16 block (auto-dispatch, templated)
         *
         * Automatically selects the best available SIMD implementation.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold BlockType::BLOCK_SIZE floats)
         * @param block Output block
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block(const float *src, BlockType &block)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q16_block_avx512<BlockType>(src, block);
#elif defined(__AVX2__)
            quantize_fp32_to_q16_block_avx2<BlockType>(src, block);
#else
            quantize_fp32_to_q16_block_scalar<BlockType>(src, block);
#endif
        }

        /**
         * @brief Quantize partial FP32 to Q16 block (scalar, templated for any block size)
         *
         * Handles blocks with fewer than BLOCK_SIZE elements, zero-filling remainder.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements
         * @param block Output block
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_partial_scalar(const float *src, size_t count, BlockType &block)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

            // Find max absolute value in partial data
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Compute scale factor
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            // Quantize valid elements and compute sum
            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = src[i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    block.qs[i] = static_cast<int16_t>(std::nearbyint(clamped));
                    sum_i64 += static_cast<int64_t>(block.qs[i]);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            // Zero-fill remainder
            for (size_t i = count; i < BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

#if defined(__AVX512F__)
        /**
         * @brief Quantize partial FP32 to Q16 block (AVX-512, templated)
         *
         * Uses masked AVX-512 operations for efficient partial block handling.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements
         * @param block Output block
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_partial_avx512(const float *src, size_t count, BlockType &block)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

            // Process full 32-element chunks first
            const size_t full_chunks = count / 32;
            const size_t remainder = count % 32;

            const __m512 sign_mask = _mm512_set1_ps(-0.0f);
            __m512 max_vec = _mm512_setzero_ps();

            // Max of full chunks
            for (size_t chunk = 0; chunk < full_chunks; ++chunk)
            {
                __m512 v0 = _mm512_loadu_ps(src + chunk * 32);
                __m512 v1 = _mm512_loadu_ps(src + chunk * 32 + 16);
                max_vec = _mm512_max_ps(max_vec, _mm512_andnot_ps(sign_mask, v0));
                max_vec = _mm512_max_ps(max_vec, _mm512_andnot_ps(sign_mask, v1));
            }

            // Max of partial remainder using masks
            if (remainder > 0)
            {
                const size_t offset = full_chunks * 32;
                const __mmask16 mask0 = (remainder >= 16) ? 0xFFFF : static_cast<__mmask16>((1U << remainder) - 1);
                const __mmask16 mask1 = (remainder <= 16) ? 0 : static_cast<__mmask16>((1U << (remainder - 16)) - 1);

                __m512 v0 = _mm512_maskz_loadu_ps(mask0, src + offset);
                __m512 v1 = _mm512_maskz_loadu_ps(mask1, src + offset + 16);
                max_vec = _mm512_max_ps(max_vec, _mm512_andnot_ps(sign_mask, v0));
                max_vec = _mm512_max_ps(max_vec, _mm512_andnot_ps(sign_mask, v1));
            }

            float max_abs = _mm512_reduce_max_ps(max_vec);

            // Compute scale
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                const __m512 vinv_d = _mm512_set1_ps(inv_d);
                const __m512i vmin = _mm512_set1_epi32(-32767);
                const __m512i vmax = _mm512_set1_epi32(32767);

                __m512i sum_vec = _mm512_setzero_si512();

                // Process full chunks
                for (size_t chunk = 0; chunk < full_chunks; ++chunk)
                {
                    __m512 v0 = _mm512_loadu_ps(src + chunk * 32);
                    __m512 v1 = _mm512_loadu_ps(src + chunk * 32 + 16);

                    __m512 scaled0 = _mm512_mul_ps(v0, vinv_d);
                    __m512 scaled1 = _mm512_mul_ps(v1, vinv_d);

                    __m512i i32_0 = _mm512_cvtps_epi32(scaled0);
                    __m512i i32_1 = _mm512_cvtps_epi32(scaled1);

                    i32_0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_0));
                    i32_1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_1));

                    // Accumulate sum
                    sum_vec = _mm512_add_epi32(sum_vec, i32_0);
                    sum_vec = _mm512_add_epi32(sum_vec, i32_1);

                    __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
                    __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);

                    // Combine and store 512 bits
                    __m512i combined = _mm512_castsi256_si512(i16_0);
                    combined = _mm512_inserti64x4(combined, i16_1, 1);
                    _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + chunk * 32), combined);
                }

                // Process partial remainder
                if (remainder > 0)
                {
                    const size_t offset = full_chunks * 32;
                    const __mmask16 mask0 = (remainder >= 16) ? 0xFFFF : static_cast<__mmask16>((1U << remainder) - 1);
                    const __mmask16 mask1 = (remainder <= 16) ? 0 : static_cast<__mmask16>((1U << (remainder - 16)) - 1);

                    __m512 v0 = _mm512_maskz_loadu_ps(mask0, src + offset);
                    __m512 v1 = _mm512_maskz_loadu_ps(mask1, src + offset + 16);

                    __m512 scaled0 = _mm512_mul_ps(v0, vinv_d);
                    __m512 scaled1 = _mm512_mul_ps(v1, vinv_d);

                    __m512i i32_0 = _mm512_cvtps_epi32(scaled0);
                    __m512i i32_1 = _mm512_cvtps_epi32(scaled1);

                    i32_0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_0));
                    i32_1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_1));

                    // Accumulate sum (masked loads are 0, so safe to add)
                    sum_vec = _mm512_add_epi32(sum_vec, i32_0);
                    sum_vec = _mm512_add_epi32(sum_vec, i32_1);

                    __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
                    __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);

                    // Combine and store 512 bits
                    __m512i combined = _mm512_castsi256_si512(i16_0);
                    combined = _mm512_inserti64x4(combined, i16_1, 1);
                    _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + offset), combined);
                }

                // Zero-fill remaining chunks
                const size_t filled_chunks = (count + 31) / 32;
                const size_t total_chunks = BLOCK_SIZE / 32;
                const __m512i zero = _mm512_setzero_si512();
                for (size_t chunk = filled_chunks; chunk < total_chunks; ++chunk)
                {
                    _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + chunk * 32), zero);
                }

                block.sum_qs = _mm512_reduce_add_epi32(sum_vec);
            }
            else
            {
                // Zero entire block
                const size_t total_chunks = BLOCK_SIZE / 32;
                const __m512i zero = _mm512_setzero_si512();
                for (size_t chunk = 0; chunk < total_chunks; ++chunk)
                {
                    _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + chunk * 32), zero);
                }
                block.sum_qs = 0;
            }
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Quantize partial FP32 to Q16 block (AVX2, templated)
         *
         * Processes full 16-element chunks with SIMD, handles remainder with scalar.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements
         * @param block Output block
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_partial_avx2(const float *src, size_t count, BlockType &block)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 max_vec = _mm256_setzero_ps();
            size_t i = 0;

            // Find max - process 8 floats at a time
            for (; i + 8 <= count; i += 8)
            {
                __m256 v = _mm256_loadu_ps(src + i);
                max_vec = _mm256_max_ps(max_vec, _mm256_andnot_ps(sign_mask, v));
            }

            // Horizontal max
            __m128 lo = _mm256_castps256_ps128(max_vec);
            __m128 hi = _mm256_extractf128_ps(max_vec, 1);
            __m128 max128 = _mm_max_ps(lo, hi);
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
            max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(max128);

            // Scalar remainder for max
            for (; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            // Compute scale
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                const __m256 vinv_d = _mm256_set1_ps(inv_d);
                const __m256i vmin = _mm256_set1_epi32(-32767);
                const __m256i vmax = _mm256_set1_epi32(32767);

                auto hsum_epi32 = [](__m256i v) -> int32_t
                {
                    __m128i lo = _mm256_castsi256_si128(v);
                    __m128i hi = _mm256_extracti128_si256(v, 1);
                    __m128i s = _mm_add_epi32(lo, hi);
                    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
                    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(0, 0, 0, 1)));
                    return _mm_cvtsi128_si32(s);
                };

                int32_t sum = 0;
                i = 0;

                // Process 16 elements at a time
                for (; i + 16 <= count; i += 16)
                {
                    __m256 v0 = _mm256_loadu_ps(src + i);
                    __m256 v1 = _mm256_loadu_ps(src + i + 8);

                    __m256 scaled0 = _mm256_mul_ps(v0, vinv_d);
                    __m256 scaled1 = _mm256_mul_ps(v1, vinv_d);

                    __m256i i32_0 = _mm256_cvtps_epi32(scaled0);
                    __m256i i32_1 = _mm256_cvtps_epi32(scaled1);

                    i32_0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_0));
                    i32_1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_1));

                    sum += hsum_epi32(i32_0) + hsum_epi32(i32_1);

                    __m256i packed = _mm256_packs_epi32(i32_0, i32_1);
                    packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + i), packed);
                }

                // Process 8 elements if present
                if (i + 8 <= count)
                {
                    __m256 v = _mm256_loadu_ps(src + i);
                    __m256 scaled = _mm256_mul_ps(v, vinv_d);
                    __m256i i32 = _mm256_cvtps_epi32(scaled);
                    i32 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32));
                    sum += hsum_epi32(i32);

                    __m256i packed = _mm256_packs_epi32(i32, _mm256_setzero_si256());
                    packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(block.qs + i),
                                     _mm256_castsi256_si128(packed));
                    i += 8;
                }

                // Scalar remainder
                for (; i < count; ++i)
                {
                    const float scaled = src[i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    block.qs[i] = static_cast<int16_t>(std::nearbyint(clamped));
                    sum += block.qs[i];
                }

                // Zero-fill remainder
                for (; i < BLOCK_SIZE; ++i)
                {
                    block.qs[i] = 0;
                }

                block.sum_qs = sum;
            }
            else
            {
                // Zero entire block
                const size_t total_chunks = BLOCK_SIZE / 16;
                for (size_t chunk = 0; chunk < total_chunks; ++chunk)
                {
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + chunk * 16), _mm256_setzero_si256());
                }
                block.sum_qs = 0;
            }
        }
#endif

        /**
         * @brief Quantize partial FP32 to Q16 block (auto-dispatch, templated)
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements
         * @param block Output block
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_partial(const float *src, size_t count, BlockType &block)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q16_block_partial_avx512<BlockType>(src, count, block);
#elif defined(__AVX2__)
            quantize_fp32_to_q16_block_partial_avx2<BlockType>(src, count, block);
#else
            quantize_fp32_to_q16_block_partial_scalar<BlockType>(src, count, block);
#endif
        }

        // =====================================================================
        // Fixed-Scale Quantization (VNNI-Safe) - Templated for Variable Block Sizes
        // =====================================================================

        /**
         * @brief Fixed-scale quantize FP32 to Q16 block (scalar, templated)
         *
         * CRITICAL: Uses a FIXED scale (not adaptive) and clips to max_safe_int16
         * to prevent INT32 overflow during VNNI dot-product accumulation.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold BlockType::BLOCK_SIZE floats)
         * @param block Output block
         * @param d Fixed scale factor (kv_cache_scale / 32767)
         * @param inv_d Inverse scale (32767 / kv_cache_scale)
         * @param max_safe_int16 Maximum safe INT16 magnitude for VNNI
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_fixed_scale_scalar(const float *src, BlockType &block,
                                                                  float d, float inv_d, int16_t max_safe_int16)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

            block.d = d;

            int64_t sum_i64 = 0;
            const float min_val = static_cast<float>(-max_safe_int16);
            const float max_val = static_cast<float>(max_safe_int16);

            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                const float scaled = src[i] * inv_d;
                const float clamped = std::max(min_val, std::min(scaled, max_val));
                const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                block.qs[i] = quantized;
                sum_i64 += static_cast<int64_t>(quantized);
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

#if defined(__AVX512F__)
        /**
         * @brief Fixed-scale quantize FP32 to Q16 block (AVX-512, templated)
         *
         * Uses AVX-512 to process in 32-element chunks with fixed scale and VNNI-safe clipping.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold BlockType::BLOCK_SIZE floats)
         * @param block Output block
         * @param d Fixed scale factor (kv_cache_scale / 32767)
         * @param inv_d Inverse scale (32767 / kv_cache_scale)
         * @param max_safe_int16 Maximum safe INT16 magnitude for VNNI
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_fixed_scale_avx512(const float *src, BlockType &block,
                                                                  float d, float inv_d, int16_t max_safe_int16)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_32 = BLOCK_SIZE / 32;

            block.d = d;

            const __m512 vinv_d = _mm512_set1_ps(inv_d);
            const __m512i vmin = _mm512_set1_epi32(-max_safe_int16);
            const __m512i vmax = _mm512_set1_epi32(max_safe_int16);

            // Use two accumulators to break dependency chains
            __m512i sum_vec0 = _mm512_setzero_si512();
            __m512i sum_vec1 = _mm512_setzero_si512();

            size_t chunk = 0;

            // Process 2 chunks (64 elements) per iteration to maximize ILP and load ports
            for (; chunk + 1 < CHUNKS_32; chunk += 2)
            {
                // Load 64 floats (4 x 16) - Interleaved loads
                __m512 v0_a = _mm512_loadu_ps(src + (chunk * 32) + 0);
                __m512 v0_b = _mm512_loadu_ps(src + (chunk * 32) + 16);
                __m512 v1_a = _mm512_loadu_ps(src + ((chunk + 1) * 32) + 0);
                __m512 v1_b = _mm512_loadu_ps(src + ((chunk + 1) * 32) + 16);

                // Scale
                __m512 s0_a = _mm512_mul_ps(v0_a, vinv_d);
                __m512 s0_b = _mm512_mul_ps(v0_b, vinv_d);
                __m512 s1_a = _mm512_mul_ps(v1_a, vinv_d);
                __m512 s1_b = _mm512_mul_ps(v1_b, vinv_d);

                // Convert to int32
                __m512i i0_a = _mm512_cvtps_epi32(s0_a);
                __m512i i0_b = _mm512_cvtps_epi32(s0_b);
                __m512i i1_a = _mm512_cvtps_epi32(s1_a);
                __m512i i1_b = _mm512_cvtps_epi32(s1_b);

                // Clamp to VNNI-safe range
                i0_a = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i0_a));
                i0_b = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i0_b));
                i1_a = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i1_a));
                i1_b = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i1_b));

                // Accumulate sums (distribute across accumulators)
                sum_vec0 = _mm512_add_epi32(sum_vec0, i0_a);
                sum_vec1 = _mm512_add_epi32(sum_vec1, i0_b);
                sum_vec0 = _mm512_add_epi32(sum_vec0, i1_a);
                sum_vec1 = _mm512_add_epi32(sum_vec1, i1_b);

                // Pack to int16
                __m256i p0_a = _mm512_cvtepi32_epi16(i0_a);
                __m256i p0_b = _mm512_cvtepi32_epi16(i0_b);
                __m256i p1_a = _mm512_cvtepi32_epi16(i1_a);
                __m256i p1_b = _mm512_cvtepi32_epi16(i1_b);

                // Combine and store
                __m512i c0 = _mm512_inserti64x4(_mm512_castsi256_si512(p0_a), p0_b, 1);
                __m512i c1 = _mm512_inserti64x4(_mm512_castsi256_si512(p1_a), p1_b, 1);

                _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + (chunk * 32)), c0);
                _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + ((chunk + 1) * 32)), c1);
            }

            // Handle remaining chunks
            for (; chunk < CHUNKS_32; ++chunk)
            {
                __m512 v0 = _mm512_loadu_ps(src + chunk * 32);
                __m512 v1 = _mm512_loadu_ps(src + chunk * 32 + 16);

                __m512 scaled0 = _mm512_mul_ps(v0, vinv_d);
                __m512 scaled1 = _mm512_mul_ps(v1, vinv_d);

                __m512i i32_0 = _mm512_cvtps_epi32(scaled0);
                __m512i i32_1 = _mm512_cvtps_epi32(scaled1);

                i32_0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_0));
                i32_1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_1));

                sum_vec0 = _mm512_add_epi32(sum_vec0, i32_0);
                sum_vec1 = _mm512_add_epi32(sum_vec1, i32_1);

                __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
                __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);

                __m512i combined = _mm512_castsi256_si512(i16_0);
                combined = _mm512_inserti64x4(combined, i16_1, 1);
                _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + chunk * 32), combined);
            }

            // Combine accumulators and reduce
            sum_vec0 = _mm512_add_epi32(sum_vec0, sum_vec1);
            block.sum_qs = _mm512_reduce_add_epi32(sum_vec0);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Fixed-scale quantize FP32 to Q16 block (AVX2, templated)
         *
         * Uses AVX2 to process in 16-element chunks with fixed scale and VNNI-safe clipping.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold BlockType::BLOCK_SIZE floats)
         * @param block Output block
         * @param d Fixed scale factor (kv_cache_scale / 32767)
         * @param inv_d Inverse scale (32767 / kv_cache_scale)
         * @param max_safe_int16 Maximum safe INT16 magnitude for VNNI
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_fixed_scale_avx2(const float *src, BlockType &block,
                                                                float d, float inv_d, int16_t max_safe_int16)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            constexpr size_t CHUNKS_16 = BLOCK_SIZE / 16;

            block.d = d;

            const __m256 vinv_d = _mm256_set1_ps(inv_d);
            const __m256i vmin = _mm256_set1_epi32(-max_safe_int16);
            const __m256i vmax = _mm256_set1_epi32(max_safe_int16);

            __m256i sum_vec = _mm256_setzero_si256();

            for (size_t chunk = 0; chunk < CHUNKS_16; ++chunk)
            {
                // Load 16 floats (2 x 8)
                __m256 v0 = _mm256_loadu_ps(src + chunk * 16);
                __m256 v1 = _mm256_loadu_ps(src + chunk * 16 + 8);

                // Scale with fixed inv_d
                __m256 scaled0 = _mm256_mul_ps(v0, vinv_d);
                __m256 scaled1 = _mm256_mul_ps(v1, vinv_d);

                // Convert to int32 (rounds to nearest)
                __m256i i32_0 = _mm256_cvtps_epi32(scaled0);
                __m256i i32_1 = _mm256_cvtps_epi32(scaled1);

                // Clamp to VNNI-safe range
                i32_0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_0));
                i32_1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_1));

                // Accumulate sum
                sum_vec = _mm256_add_epi32(sum_vec, i32_0);
                sum_vec = _mm256_add_epi32(sum_vec, i32_1);

                // Pack to int16 using packs (signed saturation, but we've already clamped)
                __m256i packed = _mm256_packs_epi32(i32_0, i32_1);
                // Fix lane ordering: packs interleaves lanes, need to permute
                packed = _mm256_permute4x64_epi64(packed, 0xD8); // 0,2,1,3

                // Store 16 int16s
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + chunk * 16), packed);
            }

            // Horizontal sum of sum_vec
            __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
            __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
            __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
            sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
            sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
            block.sum_qs = _mm_cvtsi128_si32(sum128);
        }
#endif

        /**
         * @brief Fixed-scale quantize FP32 to Q16 block (auto-dispatch, templated)
         *
         * Automatically selects the best available SIMD implementation.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold BlockType::BLOCK_SIZE floats)
         * @param block Output block
         * @param d Fixed scale factor (kv_cache_scale / 32767)
         * @param inv_d Inverse scale (32767 / kv_cache_scale)
         * @param max_safe_int16 Maximum safe INT16 magnitude for VNNI
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_fixed_scale(const float *src, BlockType &block,
                                                           float d, float inv_d, int16_t max_safe_int16)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q16_block_fixed_scale_avx512<BlockType>(src, block, d, inv_d, max_safe_int16);
#elif defined(__AVX2__)
            quantize_fp32_to_q16_block_fixed_scale_avx2<BlockType>(src, block, d, inv_d, max_safe_int16);
#else
            quantize_fp32_to_q16_block_fixed_scale_scalar<BlockType>(src, block, d, inv_d, max_safe_int16);
#endif
        }

        /**
         * @brief Fixed-scale quantize partial FP32 to Q16 block (scalar, templated)
         *
         * Handles blocks with fewer than BLOCK_SIZE elements, zero-filling remainder.
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements
         * @param block Output block
         * @param d Fixed scale factor
         * @param inv_d Inverse scale
         * @param max_safe_int16 Maximum safe INT16 magnitude for VNNI
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_partial_fixed_scale_scalar(const float *src, size_t count,
                                                                          BlockType &block, float d,
                                                                          float inv_d, int16_t max_safe_int16)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

            block.d = d;

            int64_t sum_i64 = 0;
            const float min_val = static_cast<float>(-max_safe_int16);
            const float max_val = static_cast<float>(max_safe_int16);

            for (size_t i = 0; i < count; ++i)
            {
                const float scaled = src[i] * inv_d;
                const float clamped = std::max(min_val, std::min(scaled, max_val));
                const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                block.qs[i] = quantized;
                sum_i64 += static_cast<int64_t>(quantized);
            }

            // Zero-fill remainder
            for (size_t i = count; i < BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

#if defined(__AVX512F__)
        /**
         * @brief Fixed-scale quantize partial FP32 to Q16 block (AVX-512, templated)
         *
         * Uses masked AVX-512 operations for efficient partial block handling.
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_partial_fixed_scale_avx512(const float *src, size_t count,
                                                                          BlockType &block, float d,
                                                                          float inv_d, int16_t max_safe_int16)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

            const size_t full_chunks = count / 32;
            const size_t remainder = count % 32;

            block.d = d;

            const __m512 vinv_d = _mm512_set1_ps(inv_d);
            const __m512i vmin = _mm512_set1_epi32(-max_safe_int16);
            const __m512i vmax = _mm512_set1_epi32(max_safe_int16);

            __m512i sum_vec = _mm512_setzero_si512();

            // Process full 32-element chunks
            for (size_t chunk = 0; chunk < full_chunks; ++chunk)
            {
                __m512 v0 = _mm512_loadu_ps(src + chunk * 32);
                __m512 v1 = _mm512_loadu_ps(src + chunk * 32 + 16);

                __m512 scaled0 = _mm512_mul_ps(v0, vinv_d);
                __m512 scaled1 = _mm512_mul_ps(v1, vinv_d);

                __m512i i32_0 = _mm512_cvtps_epi32(scaled0);
                __m512i i32_1 = _mm512_cvtps_epi32(scaled1);

                i32_0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_0));
                i32_1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_1));

                sum_vec = _mm512_add_epi32(sum_vec, i32_0);
                sum_vec = _mm512_add_epi32(sum_vec, i32_1);

                __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
                __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);

                __m512i combined = _mm512_castsi256_si512(i16_0);
                combined = _mm512_inserti64x4(combined, i16_1, 1);
                _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + chunk * 32), combined);
            }

            // Process partial remainder
            if (remainder > 0)
            {
                const size_t offset = full_chunks * 32;
                const __mmask16 mask0 = (remainder >= 16) ? 0xFFFF : static_cast<__mmask16>((1U << remainder) - 1);
                const __mmask16 mask1 = (remainder <= 16) ? 0 : static_cast<__mmask16>((1U << (remainder - 16)) - 1);

                __m512 v0 = _mm512_maskz_loadu_ps(mask0, src + offset);
                __m512 v1 = _mm512_maskz_loadu_ps(mask1, src + offset + 16);

                __m512 scaled0 = _mm512_mul_ps(v0, vinv_d);
                __m512 scaled1 = _mm512_mul_ps(v1, vinv_d);

                __m512i i32_0 = _mm512_cvtps_epi32(scaled0);
                __m512i i32_1 = _mm512_cvtps_epi32(scaled1);

                i32_0 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_0));
                i32_1 = _mm512_max_epi32(vmin, _mm512_min_epi32(vmax, i32_1));

                // Accumulate (masked loads give 0, so safe to add)
                sum_vec = _mm512_add_epi32(sum_vec, i32_0);
                sum_vec = _mm512_add_epi32(sum_vec, i32_1);

                __m256i i16_0 = _mm512_cvtepi32_epi16(i32_0);
                __m256i i16_1 = _mm512_cvtepi32_epi16(i32_1);

                __m512i combined = _mm512_castsi256_si512(i16_0);
                combined = _mm512_inserti64x4(combined, i16_1, 1);
                _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + offset), combined);
            }

            // Zero-fill remaining chunks
            const size_t filled_chunks = (count + 31) / 32;
            const size_t total_chunks = BLOCK_SIZE / 32;
            const __m512i zero = _mm512_setzero_si512();
            for (size_t chunk = filled_chunks; chunk < total_chunks; ++chunk)
            {
                _mm512_storeu_si512(reinterpret_cast<void *>(block.qs + chunk * 32), zero);
            }

            block.sum_qs = _mm512_reduce_add_epi32(sum_vec);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Fixed-scale quantize partial FP32 to Q16 block (AVX2, templated)
         *
         * Uses AVX2 for efficient partial block handling.
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_partial_fixed_scale_avx2(const float *src, size_t count,
                                                                        BlockType &block, float d,
                                                                        float inv_d, int16_t max_safe_int16)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;

            const size_t full_chunks = count / 16;
            const size_t remainder = count % 16;

            block.d = d;

            const __m256 vinv_d = _mm256_set1_ps(inv_d);
            const __m256i vmin = _mm256_set1_epi32(-max_safe_int16);
            const __m256i vmax = _mm256_set1_epi32(max_safe_int16);

            __m256i sum_vec = _mm256_setzero_si256();

            // Process full 16-element chunks
            for (size_t chunk = 0; chunk < full_chunks; ++chunk)
            {
                __m256 v0 = _mm256_loadu_ps(src + chunk * 16);
                __m256 v1 = _mm256_loadu_ps(src + chunk * 16 + 8);

                __m256 scaled0 = _mm256_mul_ps(v0, vinv_d);
                __m256 scaled1 = _mm256_mul_ps(v1, vinv_d);

                __m256i i32_0 = _mm256_cvtps_epi32(scaled0);
                __m256i i32_1 = _mm256_cvtps_epi32(scaled1);

                i32_0 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_0));
                i32_1 = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, i32_1));

                sum_vec = _mm256_add_epi32(sum_vec, i32_0);
                sum_vec = _mm256_add_epi32(sum_vec, i32_1);

                __m256i packed = _mm256_packs_epi32(i32_0, i32_1);
                packed = _mm256_permute4x64_epi64(packed, 0xD8);

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(block.qs + chunk * 16), packed);
            }

            // Process remainder with scalar
            if (remainder > 0)
            {
                const size_t offset = full_chunks * 16;
                const float min_val = static_cast<float>(-max_safe_int16);
                const float max_val = static_cast<float>(max_safe_int16);

                for (size_t i = 0; i < remainder; ++i)
                {
                    const float scaled = src[offset + i] * inv_d;
                    const float clamped = std::max(min_val, std::min(scaled, max_val));
                    const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                    block.qs[offset + i] = quantized;

                    // Add to sum (extract from vector first)
                    __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
                    __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
                    __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
                    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
                    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
                    int32_t current_sum = _mm_cvtsi128_si32(sum128);
                    current_sum += quantized;
                    // Store back - but actually we need to handle this differently
                }
            }

            // Zero-fill remainder
            const size_t filled = count;
            for (size_t i = filled; i < BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            // Final sum - need to handle remainder properly
            // Recompute sum with scalar for remainder portion
            __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
            __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
            __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
            sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
            sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
            int32_t total_sum = _mm_cvtsi128_si32(sum128);

            // Add remainder elements
            if (remainder > 0)
            {
                const size_t offset = full_chunks * 16;
                const float min_val = static_cast<float>(-max_safe_int16);
                const float max_val = static_cast<float>(max_safe_int16);

                for (size_t i = 0; i < remainder; ++i)
                {
                    const float scaled = src[offset + i] * inv_d;
                    const float clamped = std::max(min_val, std::min(scaled, max_val));
                    const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                    block.qs[offset + i] = quantized;
                    total_sum += quantized;
                }
            }

            block.sum_qs = total_sum;
        }
#endif

        /**
         * @brief Fixed-scale quantize partial FP32 to Q16 block (auto-dispatch, templated)
         *
         * @tparam BlockType Q16 block type with BLOCK_SIZE, d, sum_qs, qs members
         * @param src FP32 input buffer (must hold at least 'count' floats)
         * @param count Number of valid elements
         * @param block Output block
         * @param d Fixed scale factor
         * @param inv_d Inverse scale
         * @param max_safe_int16 Maximum safe INT16 magnitude for VNNI
         */
        template <typename BlockType>
        inline void quantize_fp32_to_q16_block_partial_fixed_scale(const float *src, size_t count,
                                                                   BlockType &block, float d,
                                                                   float inv_d, int16_t max_safe_int16)
        {
#if defined(__AVX512F__)
            quantize_fp32_to_q16_block_partial_fixed_scale_avx512<BlockType>(src, count, block, d, inv_d, max_safe_int16);
#elif defined(__AVX2__)
            quantize_fp32_to_q16_block_partial_fixed_scale_avx2<BlockType>(src, count, block, d, inv_d, max_safe_int16);
#else
            quantize_fp32_to_q16_block_partial_fixed_scale_scalar<BlockType>(src, count, block, d, inv_d, max_safe_int16);
#endif
        }

    } // namespace simd
} // namespace llaminar2
