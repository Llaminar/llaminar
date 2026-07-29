/**
 * @file VNNIEmulation.h
 * @brief AVX2 emulation of AVX512-VNNI operations for GEMM/GEMV kernels.
 *
 * Provides inline functions that emulate the VNNI vpdpbusd instruction using
 * AVX2. Both AVX512 and AVX2 paths are always compiled and available for
 * runtime dispatch and testability.
 *
 * Core emulation strategy (saturation-safe even/odd byte split):
 *   AVX512-VNNI: acc = _mm512_dpbusd_epi32(acc, a_u8, b_i8)
 *     → acc[i] += sum(a_u8[4i+j] * b_i8[4i+j], j=0..3)  (all in i32, no saturation)
 *
 *   AVX2 emulation (8 i32 lanes):
 *     Split a/b into even-indexed and odd-indexed bytes within each i16 slot,
 *     widen to i16, then use _mm256_madd_epi16 (i16×i16→i32, no saturation).
 *     Two madd calls cover all 4 bytes per i32 lane.
 *
 * Why not maddubs+madd?
 *   _mm256_maddubs_epi16 saturates to INT16 (±32767). For value ranges like
 *   IQ4_NL (-127..113), pair sums can reach 255*113+255*113=57630 > 32767,
 *   causing incorrect saturation. The even/odd split avoids this entirely.
 */

#pragma once

#include <immintrin.h>
#include <cstdint>

namespace llaminar2::cpu::native_vnni::isa
{

    // =========================================================================
    // AVX2 emulation of vpdpbusd (unsigned×signed INT8 dot product + accumulate)
    // =========================================================================

    /**
     * @brief Saturation-safe AVX2 emulation of the VNNI vpdpbusd instruction.
     *
     * Computes acc[i] += dot(a_u8[4i..4i+3], b_i8[4i..4i+3]) for i=0..7.
     *
     * Strategy: split even/odd bytes, widen to i16, use madd_epi16 (i16×i16→i32).
     *   - Even bytes (0,2,4,...): zero-extend a, sign-extend b → madd → i32 products
     *   - Odd bytes (1,3,5,...): shift right → same treatment → i32 products
     *   - Sum even + odd → full 4-byte dot product per i32 lane
     *
     * This avoids the INT16 saturation of _mm256_maddubs_epi16 entirely.
     * Correct for ALL input ranges including IQ4_NL codebook values.
     */
    inline __m256i avx2_dpbusd_epi32(__m256i acc, __m256i a_u8, __m256i b_i8)
    {
        const __m256i mask_lo = _mm256_set1_epi16(0x00FF);

        // Even-indexed bytes (positions 0,2,4,...): low byte of each i16 slot
        __m256i a_even = _mm256_and_si256(a_u8, mask_lo);               // zero-extend u8→i16
        __m256i b_even = _mm256_srai_epi16(_mm256_slli_epi16(b_i8, 8), 8); // sign-extend i8→i16

        // Odd-indexed bytes (positions 1,3,5,...): high byte of each i16 slot
        __m256i a_odd = _mm256_srli_epi16(a_u8, 8);  // zero-extend u8→i16
        __m256i b_odd = _mm256_srai_epi16(b_i8, 8);  // sign-extend i8→i16

        // madd_epi16: multiply i16 pairs and sum adjacent to i32 (no saturation)
        // prod_even[i] = a_even[2i]*b_even[2i] + a_even[2i+1]*b_even[2i+1]
        //              = a[4i]*b[4i] + a[4i+2]*b[4i+2]
        __m256i prod_even = _mm256_madd_epi16(a_even, b_even);

        // prod_odd[i]  = a[4i+1]*b[4i+1] + a[4i+3]*b[4i+3]
        __m256i prod_odd = _mm256_madd_epi16(a_odd, b_odd);

        // Sum = a[4i]*b[4i] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3]
        return _mm256_add_epi32(acc, _mm256_add_epi32(prod_even, prod_odd));
    }

    // =========================================================================
    // AVX2 decode LUT for nibble→INT8 conversion via vpshufb
    // =========================================================================

    /**
     * @brief Build a 256-bit decode LUT for nibble formats (Q4_0, IQ4_NL, Q4_1).
     *
     * Broadcasts a 16-byte LUT to both 128-bit lanes of a YMM register.
     * Used with _mm256_shuffle_epi8 to decode 4-bit nibbles to signed INT8.
     */
    inline __m256i build_decode_lut_avx2(const int8_t *lut_data)
    {
        __m128i lut_128 = _mm_load_si128(reinterpret_cast<const __m128i *>(lut_data));
        return _mm256_broadcastsi128_si256(lut_128);
    }

    // =========================================================================
    // AVX2 horizontal reductions
    // =========================================================================

    /**
     * @brief Horizontal sum of 8×float in YMM → scalar float.
     */
    inline float hsum_ps_avx2(__m256 v)
    {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        lo = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, lo);
        lo = _mm_add_ss(lo, shuf);
        return _mm_cvtss_f32(lo);
    }

    /**
     * @brief Horizontal sum of 8×int32 in YMM → scalar int32.
     */
    inline int32_t hsum_epi32_avx2(__m256i v)
    {
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_hadd_epi32(lo, lo);
        lo = _mm_hadd_epi32(lo, lo);
        return _mm_extract_epi32(lo, 0);
    }

} // namespace llaminar2::cpu::native_vnni::isa
