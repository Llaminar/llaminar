/**
 * @file CPUQwen4ExpSIMD.h
 * @brief Native SIMD nonlinear arithmetic for CPU GR/PLE channel blocks.
 *
 * Each vector starts at a row-local channel boundary. A grouped call therefore
 * never turns a serial scalar tail into a vector lane. AVX2 uses the engine's
 * established exponential polynomial; AVX512 evaluates the same FP32 polynomial
 * with the same FMA order. ISA selection changes width, never reduction order.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 */
#pragma once
#include "../simd/AVX2Helpers.h"
#include "../../../utils/CPUFeatures.h"
#include <cmath>

namespace llaminar2::qwen4exp::cpu_detail
{
#if defined(__AVX512F__)
/** @brief Evaluate the established range-reduced exponential in sixteen lanes. */
inline __m512 sigmoid(__m512 value)
{
    auto x = _mm512_sub_ps(_mm512_setzero_ps(), value);
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    const auto scaled = _mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f));
    const auto integer = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const auto fraction = _mm512_sub_ps(scaled, integer);
    auto polynomial = _mm512_fmadd_ps(_mm512_set1_ps(0.001333355814642f), fraction,
                                     _mm512_set1_ps(0.009618129107629f));
    polynomial = _mm512_fmadd_ps(polynomial, fraction, _mm512_set1_ps(0.055504108664822f));
    polynomial = _mm512_fmadd_ps(polynomial, fraction, _mm512_set1_ps(0.240226506959101f));
    polynomial = _mm512_fmadd_ps(polynomial, fraction, _mm512_set1_ps(0.693147180559945f));
    polynomial = _mm512_fmadd_ps(polynomial, fraction, _mm512_set1_ps(1.0f));
    const auto exponent = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(integer),
                                                          _mm512_set1_epi32(127)), 23);
    const auto exponential = _mm512_mul_ps(polynomial, _mm512_castsi512_ps(exponent));
    return _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), exponential));
}
#endif
/** @brief Select compiled native vector width once per call; scalar is explicit. */
inline int vectorWidth()
{
#if defined(__AVX512F__)
    if (activeISALevel() == ISALevel::AVX512) return 16;
#endif
#if defined(__AVX2__)
    if (activeISALevel() >= ISALevel::AVX2) return 8;
#endif
    return 1;
}
/** @brief Apply the original ISA-specific /4 SiLU program to a channel interval.
 * @param input Original row base; exact output alias is permitted.
 * @param output Retained activated row base.
 * @param begin First admitted channel.
 * @param end One-past-last admitted channel.
 * @param vectorWidth Actual preselected CPU vector width, not a new dispatch.
 */
inline void residualDownChannels(const float* input,float* output,int begin,int end,int vectorWidth)
{
            int channel = begin;
#if defined(__AVX512F__)
            if (vectorWidth == 16)
                for (; channel + 16 <= end; channel += 16)
                {
                    const auto value = _mm512_mul_ps(_mm512_loadu_ps(input + channel), _mm512_set1_ps(0.25f));
                    _mm512_storeu_ps(output + channel, _mm512_mul_ps(value, cpu_detail::sigmoid(value)));
                }
#endif
#if defined(__AVX2__)
            if (vectorWidth >= 8)
                for (; channel + 8 <= end; channel += 8)
                {
                    const auto value = _mm256_mul_ps(_mm256_loadu_ps(input + channel), _mm256_set1_ps(0.25f));
                    _mm256_storeu_ps(output + channel, avx2::fast_silu(value));
                }
#endif
            for (; channel < end; ++channel)
            {
                const float scaled = input[channel] * 0.25f;
                output[channel] = scaled * (1.0f / (1.0f + std::exp(-scaled)));
            }
}
} // namespace llaminar2::qwen4exp::cpu_detail
