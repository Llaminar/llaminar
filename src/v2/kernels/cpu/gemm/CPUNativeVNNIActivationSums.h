/**
 * @file CPUNativeVNNIActivationSums.h
 * @brief Exact, register-bounded signed-byte reductions for native quantized dots.
 *
 * Multi-scale codebooks correct each integer dot with an activation sum before
 * applying floating-point scales. Expanding every byte to a full vector of
 * INT32 lanes makes these small corrections surprisingly expensive. These
 * shared reductions keep the work in narrow integer vectors, without changing
 * any floating-point expression, prepared format, or per-row accumulation order.
 */
#pragma once

#include <array>
#include <cstdint>
#include <immintrin.h>

namespace llaminar2::cpu::native_vnni
{
    /**
     * @brief Sum exactly sixteen signed bytes without saturation or overread.
     * @param values First of sixteen readable INT8 activation values.
     * @return Exact signed sum, including the full INT8 range [-128, 127].
     */
    inline int activationHalfSum(const std::int8_t *values) noexcept
    {
        const __m128i bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(values));
        // Reuse the unsigned-bias constant already needed by compact dots.
        // A distinct byte-ones VNNI constant otherwise spills in the grouped
        // kernel's K loop. SAD produces two exact eight-byte sums, including
        // -128 inputs, without saturating pair arithmetic or wide reductions.
        const __m128i biased = _mm_xor_si128(bytes, _mm_set1_epi8(-128));
        const __m128i sums = _mm_sad_epu8(biased, _mm_setzero_si128());
        return _mm_cvtsi128_si32(sums) + _mm_extract_epi32(sums, 2) - 2048;
    }

    /**
     * @brief Sum each eight-byte quarter of one 32-value activation block.
     * @param values First of exactly 32 readable signed activation bytes.
     * @return Four exact signed sums in increasing activation-column order.
     *
     * XOR maps signed bytes to [0,255] by adding 128. VPSADBW then sums each
     * eight-byte group; subtracting 8*128 restores the signed result. No
     * activation minimum is negated, and no saturating intermediate is used.
     */
    inline std::array<int, 4> activationQuarterSums(
        const std::int8_t *values) noexcept
    {
        const __m256i bytes = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(values));
        const __m256i biased = _mm256_xor_si256(bytes, _mm256_set1_epi8(-128));
        const __m256i sums = _mm256_sad_epu8(biased, _mm256_setzero_si256());
        const __m128i low = _mm256_castsi256_si128(sums);
        const __m128i high = _mm256_extracti128_si256(sums, 1);
        return {_mm_cvtsi128_si32(low) - 1024,
                _mm_extract_epi32(low, 2) - 1024,
                _mm_cvtsi128_si32(high) - 1024,
                _mm_extract_epi32(high, 2) - 1024};
    }
}
