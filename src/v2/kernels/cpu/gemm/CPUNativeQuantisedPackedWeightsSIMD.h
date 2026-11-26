/**
 * @file CPUNativeQuantisedPackedWeightsSIMD.h
 * @brief x86 AVX2/AVX-512 inline scale loaders for CPUNativeQuantisedPackedWeights.
 *
 * Isolates `<immintrin.h>` and the SIMD scale decoders away from
 * `CPUNativeQuantisedPackedWeights.h` so non-kernel translation units that
 * inspect prepared-weight metadata do not parse 40,000+ lines of x86 intrinsics.
 */

#pragma once

#include "CPUNativeQuantisedPackedWeights.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2::cpu::native_quantised
{
#if defined(__AVX512F__)
    /**
     * @brief Widen sixteen plain FP16 scales directly to FP32.
     * @param primary Aligned inline primary words within one 64-column unit.
     */
    [[nodiscard]] inline __m512 loadPlainFP16Scales512(const uint16_t *primary)
    {
        return _mm512_cvtph_ps(_mm256_load_si256(
            reinterpret_cast<const __m256i *>(primary)));
    }

    /**
     * @brief Load sixteen effective scales without rounding source products to half.
     * @param packed Retained CPU interleaved weight owner.
     * @param primary Aligned inline primary words within one 64-column unit.
     */
    [[nodiscard]] inline __m512 loadScales512(
        const CPUNativeQuantisedPackedWeights &packed,
        const uint16_t *primary)
    {
        const __m512 base = loadPlainFP16Scales512(primary);
        if (!packed.usesNibbleLUTSourceScale())
            return base;
        const __m512i raw = _mm512_cvtepu16_epi32(_mm256_load_si256(
            reinterpret_cast<const __m256i *>(primary + 64)));
        if (packed.codebook_id == 11)
        {
            if (_mm512_test_epi32_mask(raw, _mm512_set1_epi32(~15)) != 0)
                throw std::invalid_argument("IQ3_S coefficient contains reserved bits");
            return _mm512_mul_ps(base, _mm512_cvtepi32_ps(_mm512_add_epi32(
                _mm512_slli_epi32(raw, 1), _mm512_set1_epi32(1))));
        }
        if (_mm512_test_epi32_mask(raw, _mm512_set1_epi32(~63)) != 0)
            throw std::invalid_argument("IQ4_XS coefficient contains reserved bits");
        return _mm512_mul_ps(base, _mm512_cvtepi32_ps(
            _mm512_sub_epi32(_mm512_and_si512(raw, _mm512_set1_epi32(63)), _mm512_set1_epi32(32))));
    }
#endif

#if defined(__AVX2__)
    /**
     * @brief Load eight effective scales using the same original source equation.
     * @param packed Retained CPU interleaved weight owner.
     * @param primary Aligned inline primary words within one 64-column unit.
     */
    [[nodiscard]] inline __m256 loadScales256(
        const CPUNativeQuantisedPackedWeights &packed,
        const uint16_t *primary)
    {
        const __m256 base = _mm256_cvtph_ps(_mm_load_si128(
            reinterpret_cast<const __m128i *>(primary)));
        if (!packed.usesNibbleLUTSourceScale())
            return base;
        const __m256i raw = _mm256_cvtepu16_epi32(_mm_load_si128(
            reinterpret_cast<const __m128i *>(primary + 64)));
        if (packed.codebook_id == 11)
        {
            if (!_mm256_testz_si256(raw, _mm256_set1_epi32(~15)))
                throw std::invalid_argument("IQ3_S coefficient contains reserved bits");
            return _mm256_mul_ps(base, _mm256_cvtepi32_ps(_mm256_add_epi32(
                _mm256_slli_epi32(raw, 1), _mm256_set1_epi32(1))));
        }
        if (!_mm256_testz_si256(raw, _mm256_set1_epi32(~63)))
            throw std::invalid_argument("IQ4_XS coefficient contains reserved bits");
        return _mm256_mul_ps(base, _mm256_cvtepi32_ps(
            _mm256_sub_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(63)), _mm256_set1_epi32(32))));
    }
#endif
} // namespace llaminar2::cpu::native_quantised
