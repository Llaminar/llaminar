/**
 * @file GPUAlignedExpertQ8Primitives.h
 * @brief CPU publication of the cross-backend GPU-aligned expert Q8 contract.
 *
 * ExpertOverlay can execute one logical expert on CPU, CUDA, or ROCm in
 * adjacent residency epochs. Generic CPU Q8 helpers are intentionally tuned
 * for ordinary tensor conversion and historically used division, MXCSR
 * rounding, and a CPU-only tiny-value threshold. This file instead mirrors the
 * device contract exactly while retaining AVX2/AVX-512 max, multiply, and
 * conversion throughput. It is used only by GPU-aligned expert activation paths.
 */

#pragma once

#include "kernels/common/DeviceHalfMetadataContract.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
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
        const float inverse_scale =
            device_fp32_contract::reciprocalPositive(block_scale);
        destination.d = canonicalPreparedHalfBits(block_scale);

        alignas(64) std::int32_t quantized[Q8_1Block::BLOCK_SIZE] = {};
        bool vectorized = false;
        if (valid_elements == static_cast<int>(Q8_1Block::BLOCK_SIZE))
        {
#if defined(__AVX512F__)
            if (cpu_supports_avx512())
            {
                const __m512 inverse = _mm512_set1_ps(inverse_scale);
                const __m512 minimum = _mm512_set1_ps(-127.0f);
                const __m512 maximum_value = _mm512_set1_ps(127.0f);
                for (int offset = 0; offset < 32; offset += 16)
                {
                    __m512 scaled = _mm512_mul_ps(
                        _mm512_loadu_ps(source + offset), inverse);
                    scaled = _mm512_max_ps(
                        minimum, _mm512_min_ps(maximum_value, scaled));
                    const __m512 rounded = _mm512_roundscale_ps(
                        scaled,
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm512_store_si512(
                        quantized + offset,
                        _mm512_cvttps_epi32(rounded));
                }
                vectorized = true;
            }
#endif
#if defined(__AVX2__)
            if (!vectorized && cpu_supports_avx2())
            {
                const __m256 inverse = _mm256_set1_ps(inverse_scale);
                const __m256 minimum = _mm256_set1_ps(-127.0f);
                const __m256 maximum_value = _mm256_set1_ps(127.0f);
                for (int offset = 0; offset < 32; offset += 8)
                {
                    __m256 scaled = _mm256_mul_ps(
                        _mm256_loadu_ps(source + offset), inverse);
                    scaled = _mm256_max_ps(
                        minimum, _mm256_min_ps(maximum_value, scaled));
                    const __m256 rounded = _mm256_round_ps(
                        scaled,
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm256_store_si256(
                        reinterpret_cast<__m256i *>(quantized + offset),
                        _mm256_cvttps_epi32(rounded));
                }
                vectorized = true;
            }
#endif
        }

        if (!vectorized)
        {
            for (int element = 0; element < valid_elements; ++element)
            {
                quantized[element] =
                    device_q8_activation_contract::quantize(
                        source[element], block_scale);
            }
        }

        std::int32_t sum = 0;
        for (int element = 0; element < valid_elements; ++element)
        {
            destination.qs[element] = static_cast<std::int8_t>(
                quantized[element]);
            sum += quantized[element];
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
