/**
 * @file CPUNativeQuantisedOutputVector.h
 * @brief Load and publish the exact live columns of CPU projection vectors.
 *
 * Full-width prepared tiles keep their original vector accesses. A terminal
 * tile admits a live column count before arithmetic, so it can resume or
 * publish directly at the destination even when there is no writable padding.
 */
#pragma once

#include <bit>
#include <cstddef>
#include <cstring>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief Load a partial terminal vector without reading an adjacent row.
     * @tparam Lanes Native width of one FP32 output vector.
     * @tparam F Prepared backend vector type.
     * @param source First live value owned by the output tile.
     * @param live Number of valid values, less than the vector width.
     * @return Live values followed by positive zero in inactive SIMD lanes.
     */
    template<int Lanes, class F>
    inline F loadPartialOutput(const float *source, int live) noexcept
    {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
        if constexpr (Lanes == 8)
        {
            const auto mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(live),
                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
            return std::bit_cast<F>(_mm256_maskload_ps(source, mask));
        }
#if defined(__AVX512F__)
        if constexpr (Lanes == 16)
            return std::bit_cast<F>(_mm512_maskz_loadu_ps(static_cast<__mmask16>((1u << live) - 1u), source));
#endif
#endif
        F value{};
        std::memcpy(&value, source, static_cast<size_t>(live) * sizeof(float));
        return value;
    }

    /**
     * @brief Publish a partial terminal vector directly into its live row.
     * @tparam Lanes Native width of one FP32 output vector.
     * @tparam F Prepared backend vector type.
     * @param destination First exclusively owned live value.
     * @param value SIMD accumulator in original column order.
     * @param live Number of valid values, less than the vector width.
     */
    template<int Lanes, class F>
    inline void storePartialOutput(float *destination, F value, int live) noexcept
    {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
        if constexpr (Lanes == 8)
        {
            const auto mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(live),
                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
            return _mm256_maskstore_ps(destination, mask, std::bit_cast<__m256>(value));
        }
#if defined(__AVX512F__)
        if constexpr (Lanes == 16)
            return _mm512_mask_storeu_ps(destination, static_cast<__mmask16>((1u << live) - 1u),
                std::bit_cast<__m512>(value));
#endif
#endif
        std::memcpy(destination, &value, static_cast<size_t>(live) * sizeof(float));
    }


    /**
     * @brief Continue a native FP32 accumulator without reading unwritten columns.
     * @tparam Bounded Use the caller's terminal width. Full tiles ignore the extent.
     * @tparam Lanes FP32 columns in one vector.
     * @tparam F Actual SIMD vector type.
     * @param row Beginning of the tile's output.
     * @param columns Live columns in the tile.
     * @param offset First column of this vector.
     * @param accumulate Whether this tile owns earlier contributions.
     * @return Original live column values or positive zero.
     */
    template<bool Bounded, int Lanes, class F>
    inline F readProjectionOutputVector(const float *row, int columns, int offset, bool accumulate) noexcept
    {
        if (!accumulate)
            return F{};
        if constexpr (Bounded)
        {
            const int live = columns - offset;
            if (live <= 0)
                return F{};
            if (live < Lanes)
                return loadPartialOutput<Lanes, F>(row + offset, live);
        }
        F value;
        std::memcpy(&value, row + offset, sizeof(F));
        return value;
    }

    /**
     * @brief Store one finished vector inside the original live row.
     * @tparam Bounded Use the caller's terminal width. Full tiles ignore the extent.
     * @tparam Lanes FP32 columns in one vector.
     * @tparam F Actual SIMD vector type.
     * @param row Beginning of the tile's output.
     * @param columns Live columns in the tile.
     * @param offset First column of this vector.
     * @param value Ordered accumulator to publish.
     */
    template<bool Bounded, int Lanes, class F>
    inline void publishProjectionOutputVector(float *row, int columns, int offset, F value) noexcept
    {
        if constexpr (Bounded)
        {
            const int live = columns - offset;
            if (live <= 0)
                return;
            if (live < Lanes)
                return storePartialOutput<Lanes>(row + offset, value, live);
        }
        std::memcpy(row + offset, &value, sizeof(F));
    }
}
