#pragma once

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include <cstdint>
#include <cstring>

namespace llaminar
{
    namespace simd
    {

        // ========================================================================
        // CPU Feature Detection
        // ========================================================================

        /**
         * @brief Check if CPU supports AVX-512 (cached result)
         */
        inline bool cpu_supports_avx512()
        {
            static int cached = -1;
            if (cached < 0)
            {
                cached = __builtin_cpu_supports("avx512f") ? 1 : 0;
            }
            return cached == 1;
        }

        /**
         * @brief Check if CPU supports AVX2 (cached result)
         */
        inline bool cpu_supports_avx2()
        {
            static int cached = -1;
            if (cached < 0)
            {
                cached = __builtin_cpu_supports("avx2") ? 1 : 0;
            }
            return cached == 1;
        }

        /**
         * @brief Check if CPU supports AVX512-BF16 (cached result)
         *
         * AVX512-BF16 was introduced in Cooper Lake (3rd gen Xeon Scalable).
         * Cascade Lake does NOT support it.
         *
         * Uses CPUID to check for AVX512_BF16 support (EAX=7, ECX=1, EAX bit 5).
         */
        inline bool cpu_supports_avx512_bf16()
        {
            static int cached = -1;
            if (cached < 0)
            {
#ifdef __AVX512BF16__
                // Check CPUID directly: EAX=7, ECX=1, EAX bit 5
                unsigned int eax = 0, ebx = 0, ecx = 1, edx = 0;
#if defined(__GNUC__) || defined(__clang__)
                __asm__ __volatile__(
                    "cpuid"
                    : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                    : "a"(7), "c"(1));
                cached = (eax & (1 << 5)) ? 1 : 0; // Bit 5 = AVX512_BF16
#else
                cached = 0; // Unknown compiler, assume not supported
#endif
#else
                cached = 0; // Not compiled with AVX512BF16 support
#endif
            }
            return cached == 1;
        }

        // ========================================================================
        // AVX-512 Conversion Helpers (16 elements at a time)
        // ========================================================================

#ifdef __AVX512F__

        /**
         * @brief Convert 16 int8 values to float32 with scale
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

        /**
         * @brief Convert 16 int8 values to float32 with scale and bias
         *
         * @param input Pointer to 16 int8 values
         * @param scale Scale factor to multiply
         * @param bias Bias to add before scaling
         * @param output Pointer to 16 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_biased_avx512(const int8_t *input, float scale, float bias, float *output)
        {
            __m128i i8_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));
            __m512i i32_vec = _mm512_cvtepi8_epi32(i8_vec);
            __m512 f32_vec = _mm512_cvtepi32_ps(i32_vec);
            f32_vec = _mm512_add_ps(f32_vec, _mm512_set1_ps(bias));
            __m512 result = _mm512_mul_ps(f32_vec, _mm512_set1_ps(scale));
            _mm512_storeu_ps(output, result);
        }

        /**
         * @brief Convert 16 int8 values to float32 with scale, subtract bias
         *
         * @param input Pointer to 16 int8 values
         * @param scale Scale factor to multiply
         * @param bias Bias to subtract before scaling
         * @param output Pointer to 16 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_sub_bias_avx512(const int8_t *input, float scale, float bias, float *output)
        {
            __m128i i8_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));
            __m512i i32_vec = _mm512_cvtepi8_epi32(i8_vec);
            __m512 f32_vec = _mm512_cvtepi32_ps(i32_vec);
            f32_vec = _mm512_sub_ps(f32_vec, _mm512_set1_ps(bias));
            __m512 result = _mm512_mul_ps(f32_vec, _mm512_set1_ps(scale));
            _mm512_storeu_ps(output, result);
        }

        /**
         * @brief Unpack 16 bytes of nibbles to 32 int8 values (Q4_0 format)
         *
         * Each input byte contains 2 4-bit values (nibbles).
         * Output: 32 int8 values with nibbles extracted and bias subtracted.
         *
         * @param nibbles Pointer to 16 bytes of packed nibbles
         * @param output Pointer to 32 int8 output buffer
         */
        inline void unpack_nibbles_to_i8_avx512(const uint8_t *nibbles, int8_t *output)
        {
            __m128i nibbles_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(nibbles));

            // Extract lower nibbles (even indices)
            __m128i nibbles_even = _mm_and_si128(nibbles_vec, _mm_set1_epi8(0x0F));
            // Extract upper nibbles (odd indices)
            __m128i nibbles_odd = _mm_srli_epi16(_mm_and_si128(nibbles_vec, _mm_set1_epi8(0xF0)), 4);

            // Interleave to get proper order
            __m128i interleaved_low = _mm_unpacklo_epi8(nibbles_even, nibbles_odd);
            __m128i interleaved_high = _mm_unpackhi_epi8(nibbles_even, nibbles_odd);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(output), interleaved_low);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16), interleaved_high);
        }

        /**
         * @brief Unpack nibbles and convert to float32 in one operation (Q4_0 format)
         *
         * Unpacks 16 bytes of nibbles (32 values) and converts first 16 to float32.
         *
         * @param nibbles Pointer to 16 bytes of packed nibbles
         * @param scale Scale factor to multiply
         * @param bias Bias to subtract before scaling (typically 8.0f for Q4_0)
         * @param output Pointer to 16 float32 output buffer
         * @return __m128i containing second half of unpacked nibbles (for reuse)
         */
        inline __m128i unpack_nibbles_convert_f32_first16_avx512(
            const uint8_t *nibbles, float scale, float bias, float *output)
        {
            __m128i nibbles_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(nibbles));

            // Extract and interleave nibbles
            __m128i nibbles_even = _mm_and_si128(nibbles_vec, _mm_set1_epi8(0x0F));
            __m128i nibbles_odd = _mm_srli_epi16(_mm_and_si128(nibbles_vec, _mm_set1_epi8(0xF0)), 4);
            __m128i interleaved_low = _mm_unpacklo_epi8(nibbles_even, nibbles_odd);
            __m128i interleaved_high = _mm_unpackhi_epi8(nibbles_even, nibbles_odd);

            // Convert first 16 to float32
            __m512i i32_vec = _mm512_cvtepi8_epi32(interleaved_low);
            __m512 f32_vec = _mm512_cvtepi32_ps(i32_vec);
            f32_vec = _mm512_sub_ps(f32_vec, _mm512_set1_ps(bias));
            f32_vec = _mm512_mul_ps(f32_vec, _mm512_set1_ps(scale));
            _mm512_storeu_ps(output, f32_vec);

            return interleaved_high; // Return second half for caller to process
        }

        /**
         * @brief Convert already unpacked int8 nibbles to float32 (second half)
         *
         * @param interleaved_high XMM register with 16 unpacked int8 values
         * @param scale Scale factor to multiply
         * @param bias Bias to subtract before scaling
         * @param output Pointer to 16 float32 output buffer
         */
        inline void convert_unpacked_nibbles_f32_avx512(
            __m128i interleaved_high, float scale, float bias, float *output)
        {
            __m512i i32_vec = _mm512_cvtepi8_epi32(interleaved_high);
            __m512 f32_vec = _mm512_cvtepi32_ps(i32_vec);
            f32_vec = _mm512_sub_ps(f32_vec, _mm512_set1_ps(bias));
            f32_vec = _mm512_mul_ps(f32_vec, _mm512_set1_ps(scale));
            _mm512_storeu_ps(output, f32_vec);
        }

#endif // __AVX512F__

        // ========================================================================
        // AVX2 Conversion Helpers (8 elements at a time)
        // ========================================================================

#ifdef __AVX2__

        /**
         * @brief Convert 8 int8 values to float32 with scale
         *
         * @param input Pointer to 8 int8 values
         * @param scale Scale factor to multiply
         * @param output Pointer to 8 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_avx2(const int8_t *input, float scale, float *output)
        {
            __m128i i8_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(input));
            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);
            __m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
            __m256 result = _mm256_mul_ps(f32_vec, _mm256_set1_ps(scale));
            _mm256_storeu_ps(output, result);
        }

        /**
         * @brief Convert 8 int8 values to float32 with scale and bias
         *
         * @param input Pointer to 8 int8 values
         * @param scale Scale factor to multiply
         * @param bias Bias to add before scaling
         * @param output Pointer to 8 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_biased_avx2(const int8_t *input, float scale, float bias, float *output)
        {
            __m128i i8_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(input));
            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);
            __m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
            f32_vec = _mm256_add_ps(f32_vec, _mm256_set1_ps(bias));
            __m256 result = _mm256_mul_ps(f32_vec, _mm256_set1_ps(scale));
            _mm256_storeu_ps(output, result);
        }

        /**
         * @brief Convert 8 int8 values to float32 with scale, subtract bias
         *
         * @param input Pointer to 8 int8 values
         * @param scale Scale factor to multiply
         * @param bias Bias to subtract before scaling
         * @param output Pointer to 8 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_sub_bias_avx2(const int8_t *input, float scale, float bias, float *output)
        {
            __m128i i8_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(input));
            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);
            __m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
            f32_vec = _mm256_sub_ps(f32_vec, _mm256_set1_ps(bias));
            __m256 result = _mm256_mul_ps(f32_vec, _mm256_set1_ps(scale));
            _mm256_storeu_ps(output, result);
        }

#endif // __AVX2__

        // ========================================================================
        // Format-Specific Helpers
        // ========================================================================

        /**
         * @brief Extract 6-bit scale and min from Q5_K scales array
         *
         * Mimics get_scale_min_k4 from GGML. Extracts two 6-bit values from
         * packed byte array using hierarchical bit extraction.
         *
         * @param j Index of scale/min pair (0-7)
         * @param q Scales array (12 bytes)
         * @param d Output: extracted 6-bit scale (0-63)
         * @param m Output: extracted 6-bit min (0-63)
         */
        inline void extract_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
        {
            if (j < 4)
            {
                // First 4 pairs: scale and min in separate bytes
                *d = q[j] & 63;     // Lower 6 bits
                *m = q[j + 4] & 63; // Lower 6 bits
            }
            else
            {
                // Last 4 pairs: hierarchical extraction
                *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
            }
        }

        /**
         * @brief Extract 5-bit value from Q5_K format
         *
         * Q5_K stores 5 bits per value: 4 low bits in qs, 1 high bit in qh.
         *
         * @param qs Lower 4-bit array (2 values per byte)
         * @param qh Upper 1-bit array (8 values per byte)
         * @param idx Element index within block
         * @return Extracted 5-bit value (0-31 range)
         */
        inline uint8_t extract_q5k_value(const uint8_t *qs, const uint8_t *qh, size_t idx)
        {
            // Lower 4 bits from qs (2 values per byte)
            uint8_t q_low = (idx % 2 == 0) ? (qs[idx / 2] & 0x0F) : ((qs[idx / 2] >> 4) & 0x0F);

            // Upper 1 bit from qh (8 values per byte)
            uint8_t q_high_byte_idx = idx / 8;
            uint8_t q_high_bit_pos = idx % 8;
            uint8_t q_high = (qh[q_high_byte_idx] >> q_high_bit_pos) & 0x01;

            // Combine: 4 low bits + 1 high bit (5-bit value: 0-31)
            return q_low | (q_high << 4);
        }

        /**
         * @brief Extract 6-bit scale from Q3_K scales array
         *
         * Q3_K uses complex hierarchical packing to store 16 scales in 12 bytes.
         * This mimics the GGML dequantize_row_q3_K scale extraction logic.
         *
         * @param scales 12-byte packed scales array
         * @param scale_idx Scale index (0-15)
         * @return Extracted 6-bit scale value minus bias (range: -32 to 31)
         */
        inline int8_t extract_q3k_scale(const uint8_t *scales, int scale_idx)
        {
            // Unpack scales using GGML's algorithm
            uint32_t aux[4];
            std::memcpy(aux, scales, 12);

            const uint32_t kmask1 = 0x03030303;
            const uint32_t kmask2 = 0x0f0f0f0f;

            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            const int8_t *scales_unpacked = reinterpret_cast<const int8_t *>(aux);
            return scales_unpacked[scale_idx] - 32;
        }

        /**
         * @brief Extract 3-bit signed value from Q3_K format
         *
         * Q3_K stores 3 bits per value: 2 low bits in qs, 1 high bit in hmask.
         * The high bit acts as a sign modifier: if set, value is positive (no adjustment),
         * if clear, subtract 4 from the 2-bit value to get signed range.
         *
         * @param qs Lower 2-bit array (4 values per byte)
         * @param hmask High bit mask (8 values per byte)
         * @param idx Element index within block
         * @param shift Bit shift for extracting from qs (0, 2, 4, or 6)
         * @return Extracted 3-bit signed value (range: -4 to 3)
         */
        inline int8_t extract_q3k_value(const uint8_t *qs, const uint8_t *hmask, size_t idx, int shift)
        {
            // Extract 2 low bits from qs
            int8_t q_low = (qs[idx] >> shift) & 3;

            // Check high bit from hmask
            size_t hmask_byte_idx = idx / 8;
            size_t hmask_bit_pos = idx % 8;
            bool high_bit_set = (hmask[hmask_byte_idx] >> hmask_bit_pos) & 1;

            // If high bit is set, return q_low; otherwise return q_low - 4
            return high_bit_set ? q_low : (q_low - 4);
        }

        /**
         * @brief Extract 8 nibbles from 4 bytes (Q4_0 format)
         *
         * Each byte contains 2 4-bit values.
         *
         * @param nibble_bytes 4 bytes of packed nibbles
         * @param output 8 int8 values with bias subtracted
         * @param bias Bias to subtract (typically 8 for Q4_0)
         */
        inline void extract_nibbles_scalar(uint32_t nibble_bytes, int8_t *output, int8_t bias)
        {
            for (int i = 0; i < 4; i++)
            {
                uint8_t byte_val = (nibble_bytes >> (i * 8)) & 0xFF;
                output[i * 2] = (byte_val & 0x0F) - bias;
                output[i * 2 + 1] = (byte_val >> 4) - bias;
            }
        }

        /**
         * @brief Extract 6-bit value from Q6_K format
         *
         * @param ql Lower 4-bit array (2 values per byte)
         * @param qh Upper 2-bit array (4 values per byte)
         * @param idx Element index within block
         * @param bias Bias to subtract (typically 32 for Q6_K)
         * @return Extracted 6-bit value with bias subtracted
         */
        inline int8_t extract_q6k_value(const uint8_t *ql, const uint8_t *qh, size_t idx, int bias)
        {
            // Lower 4 bits from ql (2 values per byte)
            int q_low = (idx % 2 == 0) ? (ql[idx / 2] & 0x0F) : ((ql[idx / 2] >> 4) & 0x0F);

            // Upper 2 bits from qh (4 values per byte)
            int q_high_byte_idx = idx / 4;
            int q_high_bit_pos = (idx % 4) * 2;
            int q_high = (qh[q_high_byte_idx] >> q_high_bit_pos) & 0x03;

            // Combine and subtract bias
            return (q_low | (q_high << 4)) - bias;
        }

        /**
         * @brief Extract multiple Q6_K values into buffer
         *
         * @param ql Lower 4-bit array
         * @param qh Upper 2-bit array
         * @param start_idx Starting element index within block
         * @param count Number of elements to extract
         * @param bias Bias to subtract (typically 32)
         * @param output Buffer for extracted int8 values
         */
        inline void extract_q6k_values(const uint8_t *ql, const uint8_t *qh,
                                       size_t start_idx, int count, int bias, int8_t *output)
        {
            for (int i = 0; i < count; i++)
            {
                output[i] = extract_q6k_value(ql, qh, start_idx + i, bias);
            }
        }

        // ========================================================================
        // FP16 and BF16 Conversion
        // ========================================================================

        /**
         * @brief Convert BF16 to FP32
         *
         * BF16 is FP32 with mantissa truncated from 23 bits to 7 bits.
         * Conversion is trivial: shift left by 16 bits.
         *
         * @param bf16 BF16 value (uint16_t)
         * @return FP32 value
         */
        inline float bf16_to_fp32(uint16_t bf16)
        {
            union
            {
                uint32_t u;
                float f;
            } converter;
            converter.u = static_cast<uint32_t>(bf16) << 16;
            return converter.f;
        }

        /**
         * @brief Convert FP32 to BF16 (round-to-nearest-even)
         *
         * Truncates FP32 mantissa from 23 bits to 7 bits with rounding.
         *
         * @param fp32 FP32 value
         * @return BF16 value (uint16_t)
         */
        inline uint16_t fp32_to_bf16(float fp32)
        {
            union
            {
                float f;
                uint32_t u;
            } converter;
            converter.f = fp32;

            // Round-to-nearest-even: add rounding bias
            uint32_t rounding_bias = 0x7FFF + ((converter.u >> 16) & 1);
            return static_cast<uint16_t>((converter.u + rounding_bias) >> 16);
        }

#ifdef __AVX512F__
        /**
         * @brief Convert 32 BF16 values to FP32 using AVX512
         *
         * @param input Pointer to 32 BF16 values (uint16_t)
         * @param output Pointer to 32 FP32 output buffer
         */
        inline void convert_bf16_to_fp32_avx512(const uint16_t *input, float *output)
        {
            // Load 32 BF16 values (64 bytes)
            __m512i bf16_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input));

            // Split into two halves and shift left by 16 bits
            __m256i bf16_lo = _mm512_castsi512_si256(bf16_vec);
            __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1);

            // Zero-extend uint16 to uint32 and shift left
            __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
            __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);

            // Reinterpret as float and store
            _mm512_storeu_ps(output, _mm512_castsi512_ps(fp32_lo));
            _mm512_storeu_ps(output + 16, _mm512_castsi512_ps(fp32_hi));
        }
#endif

        /**
         * @brief Convert FP16 to FP32
         *
         * Standard IEEE 754 FP16 to FP32 conversion.
         *
         * @param h FP16 value (uint16_t)
         * @return FP32 value
         */
        inline float fp16_to_fp32(uint16_t h)
        {
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exponent = (h & 0x7C00) >> 10;
            uint32_t mantissa = (h & 0x03FF);

            if (exponent == 0)
            {
                // Subnormal or zero
                if (mantissa == 0)
                {
                    return *reinterpret_cast<const float *>(&sign);
                }
                else
                {
                    exponent = 1;
                    while ((mantissa & 0x0400) == 0)
                    {
                        mantissa <<= 1;
                        exponent--;
                    }
                    mantissa &= 0x03FF;
                    exponent += (127 - 15);
                }
            }
            else if (exponent == 0x1F)
            {
                // Inf or NaN
                exponent = 0xFF;
            }
            else
            {
                // Normalized
                exponent += (127 - 15);
            }

            uint32_t result = sign | (exponent << 23) | (mantissa << 13);
            return *reinterpret_cast<const float *>(&result);
        }

    } // namespace simd
} // namespace llaminar
