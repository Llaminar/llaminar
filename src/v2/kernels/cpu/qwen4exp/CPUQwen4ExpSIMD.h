/**
 * @file CPUQwen4ExpSIMD.h
 * @brief Native SIMD nonlinear arithmetic for CPU GR/PLE channel blocks.
 *
 * Each vector starts at a row-local channel boundary. A grouped call therefore
 * never turns a serial scalar tail into a vector lane. AVX2 uses the engine's
 * established exponential polynomial; AVX512 evaluates the same FP32 polynomial
 * with the same FMA order. AArch64 evaluates four lanes with the identical
 * range-reduction/FMA polynomial, native RNE and hardware division. ISA
 * selection changes width, never reduction order.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 */
#pragma once
#include "../simd/AVX2Helpers.h"
#include "../../../utils/CPUFeatures.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "../simd/NEONHelpers.h"
#endif
#include <algorithm>
#include <cmath>
#include <stdexcept>

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
#if defined(__aarch64__) && defined(__ARM_NEON)
    const auto isa = activeISALevel();
    if (isa == ISALevel::NEON || isa == ISALevel::NEONDotProd) return 4;
    if (isa == ISALevel::Scalar) return 1; // Explicit scalar diagnostic program, as on x86.
    throw std::invalid_argument("Native ARM HC channels require an admitted NEON or scalar ISA");
#else
#if defined(__AVX512F__)
    if (activeISALevel() == ISALevel::AVX512) return 16;
#endif
#if defined(__AVX2__)
    if (activeISALevel() >= ISALevel::AVX2) return 8;
#endif
    return 1;
#endif
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
#if defined(__aarch64__) && defined(__ARM_NEON)
            if (vectorWidth != 4 && vectorWidth != 1)
                throw std::invalid_argument("Native ARM HC channels require their four-lane or scalar program");
            if (vectorWidth == 4)
            for (; channel + 4 <= end; channel += 4)
            {
                const auto value = vmulq_n_f32(vld1q_f32(input + channel), 0.25f);
                vst1q_f32(output + channel, neon::fast_silu(value));
            }
#endif
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

/**
 * @brief Normalize one already-rounded branch through the canonical SIMD 256-lane tree.
 * @param hidden Branch width.
 * @param epsilon Authenticated positive RMS epsilon.
 * @param input Original published branch bytes.
 * @param gain Actual branch gain.
 * @param output Normalized destination.
 * @param vectorWidth Selected native CPU SIMD width.
 */
inline void normalizeBranch(int hidden, float epsilon, const float *input, const float *gain,
                            float *output, int vectorWidth)
{
    alignas(64) float partial[256]{};
#if defined(__AVX512F__)
    if (vectorWidth == 16 && (hidden % 256) == 0)
    {
        __m512 acc[16];
#pragma GCC unroll 16
        for (int b = 0; b < 16; ++b)
            acc[b] = _mm512_setzero_ps();
        for (int seg = 0; seg < hidden; seg += 256)
        {
#pragma GCC unroll 16
            for (int b = 0; b < 16; ++b)
            {
                const __m512 val = _mm512_loadu_ps(input + seg + b * 16);
                acc[b] = _mm512_fmadd_ps(val, val, acc[b]);
            }
        }
#pragma GCC unroll 8
        for (int b = 0; b < 8; ++b)
            acc[b] = _mm512_add_ps(acc[b], acc[b + 8]);
#pragma GCC unroll 4
        for (int b = 0; b < 4; ++b)
            acc[b] = _mm512_add_ps(acc[b], acc[b + 4]);
        acc[0] = _mm512_add_ps(acc[0], acc[2]);
        acc[1] = _mm512_add_ps(acc[1], acc[3]);
        acc[0] = _mm512_add_ps(acc[0], acc[1]);
        _mm512_store_ps(partial, acc[0]);
        for (int stride = 8; stride > 0; stride /= 2)
            for (int lane = 0; lane < stride; ++lane)
                partial[lane] += partial[lane + stride];
        const float inverse = 1.0f / std::sqrt(partial[0] / static_cast<float>(hidden) + epsilon);
        const __m512 vinv = _mm512_set1_ps(inverse);
        for (int c = 0; c < hidden; c += 16)
            _mm512_storeu_ps(output + c,
                _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(input + c), vinv),
                              _mm512_loadu_ps(gain + c)));
        return;
    }
#endif
    for (int segment = 0; segment < hidden; segment += 256)
    {
        const int count = std::min(256, hidden - segment);
        int lane = 0;
#if defined(__AVX512F__)
        if (vectorWidth == 16)
            for (; lane + 16 <= count; lane += 16)
            {
                const auto value = _mm512_loadu_ps(input + segment + lane);
                _mm512_store_ps(partial + lane, _mm512_fmadd_ps(value, value, _mm512_load_ps(partial + lane)));
            }
#endif
#if defined(__AVX2__)
        if (vectorWidth >= 8)
            for (; lane + 8 <= count; lane += 8)
            {
                const auto value = _mm256_loadu_ps(input + segment + lane);
                _mm256_store_ps(partial + lane, _mm256_fmadd_ps(value, value, _mm256_load_ps(partial + lane)));
            }
#endif
        for (; lane < count; ++lane)
        {
            const auto value = input[segment + lane];
            partial[lane] = std::fma(value, value, partial[lane]);
        }
    }
    for (int stride = 128; stride > 0; stride /= 2)
        for (int lane = 0; lane < stride; ++lane)
            partial[lane] += partial[lane + stride];
    const float inverse = 1.0f / std::sqrt(partial[0] / static_cast<float>(hidden) + epsilon);
    int channel = 0;
#if defined(__AVX512F__)
    if (vectorWidth == 16)
        for (; channel + 16 <= hidden; channel += 16)
            _mm512_storeu_ps(output + channel, _mm512_mul_ps(
                _mm512_mul_ps(_mm512_loadu_ps(input + channel), _mm512_set1_ps(inverse)),
                _mm512_loadu_ps(gain + channel)));
#endif
#if defined(__AVX2__)
    if (vectorWidth >= 8)
        for (; channel + 8 <= hidden; channel += 8)
            _mm256_storeu_ps(output + channel, _mm256_mul_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + channel), _mm256_set1_ps(inverse)),
                _mm256_loadu_ps(gain + channel)));
#endif
    for (; channel < hidden; ++channel)
        output[channel] = (input[channel] * inverse) * gain[channel];
}

/**
 * @brief Fused single-branch ResidualWrite + BranchRMSNorm through the canonical 256-lane tree.
 * @param hidden Branch width.
 * @param epsilon Authenticated positive RMS epsilon.
 * @param gate Precomputed scalar branch gate `2.0f * sigmoid(gateLogit * 0.25f)`.
 * @param res Prior residual branch input.
 * @param blk Block output row to inject.
 * @param gain Branch RMSNorm gain.
 * @param out Updated residual branch destination.
 * @param norm Normalized branch destination.
 * @param vectorWidth Selected native CPU SIMD width.
 */
inline void residualWriteAndNormalizeBranch(int hidden, float epsilon, float gate,
                                            const float *__restrict res, const float *__restrict blk,
                                            const float *__restrict gain, float *__restrict out,
                                            float *__restrict norm, int vectorWidth)
{
    alignas(64) float partial[256]{};
#if defined(__AVX512F__)
    if (vectorWidth == 16 && (hidden % 256) == 0)
    {
        const __m512 vgate = _mm512_set1_ps(gate);
        const __m512 vzero = _mm512_setzero_ps();
        __m512 acc[16];
#pragma GCC unroll 16
        for (int b = 0; b < 16; ++b)
            acc[b] = _mm512_setzero_ps();
        for (int seg = 0; seg < hidden; seg += 256)
        {
#pragma GCC unroll 16
            for (int b = 0; b < 16; ++b)
            {
                const int off = seg + b * 16;
                const __m512 upd = _mm512_fmadd_ps(vgate, _mm512_loadu_ps(blk + off), vzero);
                const __m512 val = _mm512_add_ps(_mm512_loadu_ps(res + off), upd);
                _mm512_storeu_ps(out + off, val);
                acc[b] = _mm512_fmadd_ps(val, val, acc[b]);
            }
        }
#pragma GCC unroll 8
        for (int b = 0; b < 8; ++b)
            acc[b] = _mm512_add_ps(acc[b], acc[b + 8]);
#pragma GCC unroll 4
        for (int b = 0; b < 4; ++b)
            acc[b] = _mm512_add_ps(acc[b], acc[b + 4]);
        acc[0] = _mm512_add_ps(acc[0], acc[2]);
        acc[1] = _mm512_add_ps(acc[1], acc[3]);
        acc[0] = _mm512_add_ps(acc[0], acc[1]);
        _mm512_store_ps(partial, acc[0]);
        for (int stride = 8; stride > 0; stride /= 2)
            for (int lane = 0; lane < stride; ++lane)
                partial[lane] += partial[lane + stride];
        const float inverse = 1.0f / std::sqrt(partial[0] / static_cast<float>(hidden) + epsilon);
        const __m512 vinv = _mm512_set1_ps(inverse);
        for (int c = 0; c < hidden; c += 16)
        {
            _mm512_storeu_ps(norm + c,
                _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(out + c), vinv),
                              _mm512_loadu_ps(gain + c)));
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if (vectorWidth >= 8)
    {
        const __m256 vgate = _mm256_set1_ps(gate);
        const __m256 vzero = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= hidden; c += 8)
        {
            const __m256 upd = _mm256_fmadd_ps(vgate, _mm256_loadu_ps(blk + c), vzero);
            _mm256_storeu_ps(out + c, _mm256_add_ps(_mm256_loadu_ps(res + c), upd));
        }
        for (; c < hidden; ++c)
        {
            const float update = std::fma(gate, blk[c], 0.0f);
            out[c] = res[c] + update;
        }
        normalizeBranch(hidden, epsilon, out, gain, norm, vectorWidth);
        return;
    }
#endif
    for (int channel = 0; channel < hidden; ++channel)
    {
        const float update = std::fma(gate, blk[channel], 0.0f);
        out[channel] = res[channel] + update;
    }
    normalizeBranch(hidden, epsilon, out, gain, norm, vectorWidth);
}
} // namespace llaminar2::qwen4exp::cpu_detail
