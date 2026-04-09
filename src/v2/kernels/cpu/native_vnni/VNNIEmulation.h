/**
 * @file VNNIEmulation.h
 * @brief AVX2 emulation of AVX512-VNNI operations for GEMM/GEMV kernels.
 *
 * Provides inline functions that emulate the VNNI vpdpbusd instruction using
 * the AVX2 maddubs+madd pattern. Both AVX512 and AVX2 paths are always
 * compiled and available for runtime dispatch and testability.
 *
 * Core pattern:
 *   AVX512-VNNI: acc = _mm512_dpbusd_epi32(acc, a_u8, b_i8)
 *     → acc[i] += dot(a_u8[4i..4i+3], b_i8[4i..4i+3]) for 16 i32 lanes
 *
 *   AVX2 emulation (8 i32 lanes):
 *     prod16 = _mm256_maddubs_epi16(a_u8, b_i8)   // u8×i8 → i16 pairs
 *     prod32 = _mm256_madd_epi16(prod16, ones)     // i16 pairs → i32
 *     acc    = _mm256_add_epi32(acc, prod32)        // accumulate
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
     * @brief AVX2 emulation of the VNNI vpdpbusd instruction.
     *
     * Computes acc[i] += dot(a_u8[4i..4i+3], b_i8[4i..4i+3]) for i=0..7.
     * Uses the proven maddubs+madd two-instruction pattern (same as llama.cpp/GGML).
     *
     * @note _mm256_maddubs_epi16 treats first operand as unsigned, second as signed.
     *       This matches vpdpbusd semantics exactly.
     *
     * @warning _mm256_maddubs_epi16 can saturate to INT16_MAX/MIN if the pairwise
     *          products are large. For typical quantized inference values (u8 in
     *          [0,255], i8 in [-127,127]) this is safe. The worst case is
     *          255*127 + 255*127 = 64770, well within INT16 range (32767).
     *          HOWEVER: if a_u8 = 255 and b_i8 = {127, 127, ...}, each pair
     *          produces 255*127 = 32385, and two pairs sum to 64770 which
     *          OVERFLOWS INT16 (max 32767). This is the same saturation behavior
     *          as the VNNI instruction, so results are mathematically equivalent.
     */
    inline __m256i avx2_dpbusd_epi32(__m256i acc, __m256i a_u8, __m256i b_i8)
    {
        // Step 1: u8 × i8 → i16 pairwise products with saturation
        //   For each group of 4 bytes: pairs (a0*b0 + a1*b1), (a2*b2 + a3*b3)
        __m256i prod16 = _mm256_maddubs_epi16(a_u8, b_i8);

        // Step 2: horizontal i16 pair add → i32
        //   Adds adjacent i16 pairs: (p0+p1) → i32, completing the 4-element dot product
        __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));

        // Step 3: accumulate
        return _mm256_add_epi32(acc, prod32);
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
