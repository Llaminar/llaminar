/**
 * @file CPUFixedEpochQueryBlock.h
 * @brief Independent-query SIMD contraction for the fixed eight-update CPU epoch.
 *
 * The SIMD axis is the query, never the key reduction. Each lane executes the
 * same increasing-key FP32 FMA chain as prepareAdjustedGDNReadout. Transposed
 * bounded thread-private scratch makes those independent lanes contiguous;
 * inactive queries keep their original bytes until their own history begins.
 * This helper neither advances checkpoints nor introduces epoch boundaries.
 */
#pragma once

#include "CPUAdjustedGDNReadout.h"
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::cpu_gdn_epoch
{
    /**
     * @brief Contract a bounded query block through its own backward raw history.
     * @tparam Rows Number of independent queries, from one through four.
     * @param layout Validated key/value geometry, with key width at most 512.
     * @param history Read-only contiguous raw records containing phase+Rows updates.
     * @param q First prepared query, borrowed read-only.
     * @param q_stride Distance between prepared queries in floats.
     * @param phase History length preceding the first query, with phase+Rows<=8.
     * @param x Output contracted queries with a fixed 512-float row stride.
     * @param coefficients Output eta coefficients with a fixed eight-float stride.
     *
     * Each query visits records newest-to-oldest. The dot within a record always
     * visits keys in increasing order, with exactly one fused rounding per key.
     * Multiplication by beta and decay is separately rounded. Physical SIMD
     * width cannot change this program, its history bounds, or checkpoint state.
     */
    template<int Rows>
    inline void contractQueries(const CPUGDNStateLayout &layout, const float *history,
        const float *q, int q_stride, int phase, float *x, float *coefficients)
    {
        static_assert(Rows >= 1 && Rows <= 4);
#if defined(__AVX2__)
        if constexpr (Rows > 1)
        {
            // Four independent dots fit in one register on both shipped ISAs.
            // Padding lanes contain zero and are never published to caller state.
            alignas(16) float transposed[512 * 4];
            for (int j = 0; j < layout.keyWidth(); ++j)
                for (int r = 0; r < 4; ++r)
                    transposed[j * 4 + r] = r < Rows ? q[r * q_stride + j] : 0.0f;
            for (int p = phase + Rows - 1; p >= 0; --p)
            {
                const float *raw = history + static_cast<size_t>(p) * layout.recordFloats();
                __m128 dot = _mm_setzero_ps();
                for (int j = 0; j < layout.keyWidth(); ++j)
                    dot = _mm_fmadd_ps(_mm_set1_ps(raw[j]),
                        _mm_load_ps(transposed + j * 4), dot);
                const __m128 eta = _mm_mul_ps(dot,
                    _mm_set1_ps(raw[layout.keyWidth() + layout.valueWidth() + 1]));
                alignas(16) float lanes[4];
                _mm_store_ps(lanes, eta);
                for (int r = 0; r < Rows; ++r)
                    if (p <= phase + r)
                        coefficients[r * CPUGDNStateLayout::kPeriod + p] = lanes[r];
                const __m128 negative_eta = _mm_xor_ps(eta, _mm_set1_ps(-0.0f));
                const __m128 decay = _mm_set1_ps(raw[layout.keyWidth() + layout.valueWidth()]);
                const __m128 active = _mm_castsi128_ps(_mm_setr_epi32(
                    p <= phase ? -1 : 0, p <= phase + 1 ? -1 : 0,
                    Rows > 2 && p <= phase + 2 ? -1 : 0,
                    Rows > 3 && p <= phase + 3 ? -1 : 0));
                for (int j = 0; j < layout.keyWidth(); ++j)
                {
                    const __m128 original = _mm_load_ps(transposed + j * 4);
                    const __m128 adjusted = _mm_mul_ps(decay,
                        _mm_fmadd_ps(negative_eta, _mm_set1_ps(raw[j]), original));
                    _mm_store_ps(transposed + j * 4,
                        _mm_blendv_ps(original, adjusted, active));
                }
            }
            for (int j = 0; j < layout.keyWidth(); ++j)
                for (int r = 0; r < Rows; ++r)
                    x[r * 512 + j] = transposed[j * 4 + r];
            return;
        }
#endif
        for (int r = 0; r < Rows; ++r)
        {
            float *query = x + r * 512;
            std::memcpy(query, q + r * q_stride,
                static_cast<std::size_t>(layout.keyWidth()) * sizeof(float));
            for (int p = phase + r; p >= 0; --p)
            {
                const float *raw = history + static_cast<size_t>(p) * layout.recordFloats();
                coefficients[r * CPUGDNStateLayout::kPeriod + p] = prepareAdjustedGDNReadout(
                    query, raw, raw[layout.keyWidth() + layout.valueWidth()],
                    raw[layout.keyWidth() + layout.valueWidth() + 1], layout.keyWidth(), query);
            }
        }
    }
}
