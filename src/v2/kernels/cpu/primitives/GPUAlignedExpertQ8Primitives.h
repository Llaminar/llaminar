/**
 * @file GPUAlignedExpertQ8Primitives.h
 * @brief CPU publication of the cross-backend GPU-aligned expert Q8 contract.
 *
 * ExpertOverlay can execute one logical expert on CPU, CUDA, or ROCm in
 * adjacent residency epochs. Generic CPU Q8 helpers are intentionally tuned
 * for ordinary tensor conversion and historically used division, MXCSR
 * rounding, and a CPU-only tiny-value threshold. This file instead mirrors the
 * device contract exactly with native AVX2/AVX-512 or AArch64 NEON max,
 * multiply, and conversion. It is used only by GPU-aligned expert activation
 * paths; the ISA cannot select a different persisted scale or activation byte.
 */

#pragma once

#include "kernels/common/DeviceHalfMetadataContract.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "PreparedHalfReciprocal.h"
#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace llaminar2::cpu::gpu_aligned_expert_q8
{
    /**
     * @brief Find the maximum magnitude in one Q8 publication block.
     *
     * Maximum is selection rather than arithmetic reduction, so SIMD grouping
     * cannot change the result for the finite activations admitted by the graph.
     * Partial tail blocks retain the scalar path to avoid out-of-range reads.
     *
     * @param source Source FP32 values.
     * @param valid_elements Number of valid values in `[1, 32]`.
     * @return Exact largest absolute source value.
     */
    inline float maximumAbsoluteValue(
        const float *source,
        int valid_elements) noexcept
    {
        if (valid_elements == static_cast<int>(Q8_1Block::BLOCK_SIZE))
        {
#if defined(__aarch64__) && defined(__ARM_NEON)
            const uint32x4_t magnitude_mask = vdupq_n_u32(0x7fffffffu);
            uint32x4_t maximum = vdupq_n_u32(0);
            for (int offset = 0; offset < 32; offset += 4)
            {
                // Unsigned magnitude words order finite values exactly and
                // retain subnormal bits independently of floating max modes.
                const uint32x4_t magnitude = vandq_u32(
                    vreinterpretq_u32_f32(vld1q_f32(source + offset)), magnitude_mask);
                maximum = vmaxq_u32(maximum, magnitude);
            }
            return device_fp32_contract::floatFromBits(vmaxvq_u32(maximum));
#endif
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                const __m512 first = _mm512_loadu_ps(source);
                const __m512 second = _mm512_loadu_ps(source + 16);
                return _mm512_reduce_max_ps(
                    _mm512_max_ps(_mm512_abs_ps(first),
                                  _mm512_abs_ps(second)));
            }
#endif
#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                const __m256 sign = _mm256_set1_ps(-0.0f);
                const __m256 first =
                    _mm256_andnot_ps(sign, _mm256_loadu_ps(source));
                const __m256 second =
                    _mm256_andnot_ps(sign, _mm256_loadu_ps(source + 8));
                const __m256 third =
                    _mm256_andnot_ps(sign, _mm256_loadu_ps(source + 16));
                const __m256 fourth =
                    _mm256_andnot_ps(sign, _mm256_loadu_ps(source + 24));
                __m256 maximum = _mm256_max_ps(
                    _mm256_max_ps(first, second),
                    _mm256_max_ps(third, fourth));
                const __m128 lower = _mm256_castps256_ps128(maximum);
                const __m128 upper = _mm256_extractf128_ps(maximum, 1);
                __m128 reduced = _mm_max_ps(lower, upper);
                reduced = _mm_max_ps(
                    reduced,
                    _mm_movehl_ps(reduced, reduced));
                reduced = _mm_max_ss(
                    reduced,
                    _mm_shuffle_ps(reduced, reduced, 0x55));
                return _mm_cvtss_f32(reduced);
            }
#endif
        }

        float maximum = 0.0f;
        for (int element = 0; element < valid_elements; ++element)
            maximum = std::max(maximum, std::fabs(source[element]));
        return maximum;
    }

    /**
     * @brief Quantize one FP32 block through the placement-invariant Q8 ABI.
     *
     * The scale is rounded through binary16 before its reciprocal is derived,
     * matching CPU `Q8_1Block::d` and the widened GPU scale sidecar. SIMD lanes
     * execute only exact multiply, clamp, and nearest-even conversion; integer
     * publication and summation preserve element order independently of ISA.
     *
     * @param source Source FP32 block.
     * @param destination Destination Q8_1 block.
     * @param valid_elements Valid values in `[1, 32]`; the tail is zero-filled.
     */
    inline void quantizeBlock(
        const float *source,
        Q8_1Block &destination,
        int valid_elements = static_cast<int>(Q8_1Block::BLOCK_SIZE))
    {
        if (!source || valid_elements <= 0 ||
            valid_elements > static_cast<int>(Q8_1Block::BLOCK_SIZE))
        {
            throw std::invalid_argument(
                "GPU-aligned expert Q8 publication requires 1..32 source values");
        }

        const float maximum = maximumAbsoluteValue(source, valid_elements);
        const float block_scale =
            device_q8_activation_contract::scale(maximum);
        destination.d = canonicalPreparedHalfBits(block_scale);
        // Persisted scale words have a finite normalized domain. This lookup
        // returns the identical five-step reciprocal without repeating that
        // scalar dependency chain for every expert activation block.
        const float inverse_scale =
            prepared_half_reciprocal::reciprocal(destination.d);

        if (valid_elements == static_cast<int>(Q8_1Block::BLOCK_SIZE))
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                const __m512 inverse = _mm512_set1_ps(inverse_scale);
                const __m512 minimum = _mm512_set1_ps(-127.0f);
                const __m512 maximum_value = _mm512_set1_ps(127.0f);
                const auto quantize_vector = [&](int offset)
                {
                    __m512 scaled = _mm512_mul_ps(
                        _mm512_loadu_ps(source + offset), inverse);
                    scaled = _mm512_max_ps(
                        minimum, _mm512_min_ps(maximum_value, scaled));
                    const __m512 rounded = _mm512_roundscale_ps(
                        scaled,
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    return _mm512_cvttps_epi32(rounded);
                };
                const __m512i first = quantize_vector(0);
                const __m512i second = quantize_vector(16);
                // All values are already in [-127,127]. Narrowing is exact,
                // and the integer sum cannot overflow in any reduction order.
                // Publish bytes directly instead of constructing a 128-byte
                // INT32 scratch row and walking it again to pack and sum.
                _mm_storeu_si128(reinterpret_cast<__m128i *>(destination.qs),
                                _mm512_cvtepi32_epi8(first));
                _mm_storeu_si128(reinterpret_cast<__m128i *>(destination.qs + 16),
                                _mm512_cvtepi32_epi8(second));
                destination.sum_qs = static_cast<std::int16_t>(
                    _mm512_reduce_add_epi32(_mm512_add_epi32(first, second)));
                return;
            }
#endif
#if defined(__AVX2__)
            if (cpu_supports_avx2())
            {
                const __m256 inverse = _mm256_set1_ps(inverse_scale);
                const __m256 minimum = _mm256_set1_ps(-127.0f);
                const __m256 maximum_value = _mm256_set1_ps(127.0f);
                const auto quantize_vector = [&](int offset)
                {
                    __m256 scaled = _mm256_mul_ps(
                        _mm256_loadu_ps(source + offset), inverse);
                    scaled = _mm256_max_ps(
                        minimum, _mm256_min_ps(maximum_value, scaled));
                    const __m256 rounded = _mm256_round_ps(
                        scaled,
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    return _mm256_cvttps_epi32(rounded);
                };
                const __m256i q0 = quantize_vector(0);
                const __m256i q1 = quantize_vector(8);
                const __m256i q2 = quantize_vector(16);
                const __m256i q3 = quantize_vector(24);
                const __m256i packed = _mm256_packs_epi16(
                    _mm256_packs_epi32(q0, q1), _mm256_packs_epi32(q2, q3));
                // AVX2 packs within each 128-bit lane. Restore the original
                // groups of four bytes before publishing the complete row.
                const __m256i ordered = _mm256_permutevar8x32_epi32(
                    packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(destination.qs), ordered);
                const __m256i lanes = _mm256_add_epi32(
                    _mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3));
                __m128i sum = _mm_add_epi32(
                    _mm256_castsi256_si128(lanes), _mm256_extracti128_si256(lanes, 1));
                sum = _mm_hadd_epi32(sum, sum);
                sum = _mm_hadd_epi32(sum, sum);
                destination.sum_qs = static_cast<std::int16_t>(_mm_cvtsi128_si32(sum));
                return;
            }
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
            const float32x4_t inverse = vdupq_n_f32(inverse_scale);
            const float32x4_t minimum = vdupq_n_f32(-127.0f);
            const float32x4_t maximum_value = vdupq_n_f32(127.0f);
            int32x4_t sum = vdupq_n_s32(0);
            for (int offset = 0; offset < 32; offset += 8)
            {
                const float32x4_t first = vmaxq_f32(minimum, vminq_f32(maximum_value,
                    vmulq_f32(vld1q_f32(source + offset), inverse)));
                const float32x4_t second = vmaxq_f32(minimum, vminq_f32(maximum_value,
                    vmulq_f32(vld1q_f32(source + offset + 4), inverse)));
                // FCVTNS is explicitly nearest-even, rather than FPCR-selected
                // conversion. Clamp to integer endpoints before conversion;
                // this is identical to RNE then saturation for finite inputs.
                const int32x4_t q0 = vcvtnq_s32_f32(first);
                const int32x4_t q1 = vcvtnq_s32_f32(second);
                const int8x8_t bytes = vmovn_s16(vcombine_s16(vmovn_s32(q0), vmovn_s32(q1)));
                vst1_s8(destination.qs + offset, bytes);
                sum = vaddq_s32(sum, vaddq_s32(q0, q1));
            }
            // Every lane is bounded by 8*127, and the complete sum by 32*127.
            // Integer reduction/narrowing is exact regardless of SIMD grouping.
            destination.sum_qs = static_cast<std::int16_t>(vaddvq_s32(sum));
            return;
#endif
        }

        std::int32_t sum = 0;
        for (int element = 0; element < valid_elements; ++element)
        {
            const std::int32_t quantized = device_q8_activation_contract::quantize(
                source[element], block_scale);
            destination.qs[element] = static_cast<std::int8_t>(quantized);
            sum += quantized;
        }
        std::memset(
            destination.qs + valid_elements,
            0,
            Q8_1Block::BLOCK_SIZE - static_cast<std::size_t>(valid_elements));
        destination.sum_qs = static_cast<std::int16_t>(sum);
    }

    /**
     * @brief Quantize two adjacent complete blocks without changing their ABI.
     * @param source Source pointer containing at least 64 FP32 values.
     * @param first Destination for values `[0, 32)`.
     * @param second Destination for values `[32, 64)`.
     */
    inline void quantizeTwoBlocks(
        const float *source,
        Q8_1Block &first,
        Q8_1Block &second)
    {
        quantizeBlock(source, first);
        quantizeBlock(source + Q8_1Block::BLOCK_SIZE, second);
    }
} // namespace llaminar2::cpu::gpu_aligned_expert_q8
