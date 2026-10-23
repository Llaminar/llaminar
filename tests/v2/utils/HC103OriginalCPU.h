/**
 * @file HC103OriginalCPU.h
 * @brief Frozen pre103 CPU activation/read and native half-scale Q8 publication.
 *
 * Test-only literal arithmetic snapshots made before any production extraction.
 * Native prepared projection remains the production engine. ISA primitive dispatch
 * and hardware quantizer dispatch are deliberately separate original policies.
 */
#pragma once
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include "kernels/qwen4exp/Qwen4ExpValidation.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"
#include "utils/OpenMPUtils.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <stdexcept>
namespace llaminar2::test::hc103_original {
using namespace qwen4exp;
/** @brief Frozen original half-bit expansion. */
inline float fp16_to_fp32(uint16_t h)
    {
        uint32_t sign = (h & 0x8000u) << 16;
        int32_t exp = (h & 0x7C00u) >> 10;
        uint32_t mantissa = (h & 0x03FFu);

        uint32_t f;
        if (exp == 0)
        {
            // Zero or denormal
            if (mantissa == 0)
            {
                f = sign; // Signed zero
            }
            else
            {
                // Denormal: normalize it
                exp = 1;
                while ((mantissa & 0x400) == 0)
                {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x3FF;
                f = sign | ((static_cast<uint32_t>(exp + 112)) << 23) | (mantissa << 13);
            }
        }
        else if (exp == 0x1F)
        {
            // Infinity or NaN
            f = sign | 0x7F800000u | (mantissa << 13);
        }
        else
        {
            // Normal number
            f = sign | ((static_cast<uint32_t>(exp + 112)) << 23) | (mantissa << 13);
        }

        float result;
        std::memcpy(&result, &f, sizeof(float));
        return result;
    }
/** @brief Frozen original half rounding. */
inline uint16_t fp32_to_fp16(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));

        uint32_t sign = (bits & 0x80000000u) >> 16;
        int32_t exp = ((bits & 0x7F800000u) >> 23) - 127 + 15;
        uint32_t mantissa = (bits & 0x007FFFFFu);

        if (exp <= 0)
        {
            // Zero or denormal
            if (exp < -10)
            {
                return static_cast<uint16_t>(sign); // Underflow to zero
            }
            // Denormal: shift mantissa right and add implicit 1
            mantissa = (mantissa | 0x00800000u) >> (1 - exp);
            return static_cast<uint16_t>(sign | (mantissa >> 13));
        }
        else if (exp >= 0x1F)
        {
            // Overflow to infinity or preserve NaN
            if (exp == 0xFF - 127 + 15 && mantissa != 0)
            {
                // NaN
                return static_cast<uint16_t>(sign | 0x7C00u | (mantissa >> 13));
            }
            return static_cast<uint16_t>(sign | 0x7C00u); // Infinity
        }
        else
        {
            // Normal number: round to nearest even
            uint32_t rounding = (mantissa >> 13) & 1;
            mantissa += 0x0FFFu + rounding;
            if (mantissa >= 0x00800000u)
            {
                // Rounding caused overflow, increment exponent
                exp++;
                mantissa = 0;
            }
            return static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
        }
    }
inline float nativeVNNIFP16ScaleToFP32(uint16_t value)
    {
#if defined(__F16C__)
        if ((value & 0x7c00u) != 0x7c00u)
            return _cvtsh_ss(value);
#endif
        return fp16_to_fp32(value);
    }
inline __m256 persistRoundedAVX2(__m256 value) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : "+x"(value));
        return value;
#else
        /*
         * The project builds with GCC or Clang on x86.  Keep a correctness-
         * first implementation for another compiler instead of silently
         * dropping the numerical contract; this branch is intentionally not
         * part of the tuned production binaries.
         */
        alignas(32) volatile float rounded[8];
        _mm256_store_ps(const_cast<float *>(rounded), value);
        return _mm256_load_ps(const_cast<const float *>(rounded));
#endif
    }
namespace avx2 {
static inline __m256 fast_exp(__m256 vx)
        {
            vx = _mm256_max_ps(vx, _mm256_set1_ps(-88.0f));
            vx = _mm256_min_ps(vx, _mm256_set1_ps(88.0f));

            const __m256 vlog2e = _mm256_set1_ps(1.4426950408889634f);
            const __m256 vc0 = _mm256_set1_ps(1.0f);
            const __m256 vc1 = _mm256_set1_ps(0.693147180559945f);
            const __m256 vc2 = _mm256_set1_ps(0.240226506959101f);
            const __m256 vc3 = _mm256_set1_ps(0.055504108664822f);
            const __m256 vc4 = _mm256_set1_ps(0.009618129107629f);
            const __m256 vc5 = _mm256_set1_ps(0.001333355814642f);

            __m256 vt = _mm256_mul_ps(vx, vlog2e);
            __m256 vn = _mm256_round_ps(vt, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 vf = _mm256_sub_ps(vt, vn);

            __m256 vpoly = _mm256_fmadd_ps(vc5, vf, vc4);
            vpoly = _mm256_fmadd_ps(vpoly, vf, vc3);
            vpoly = _mm256_fmadd_ps(vpoly, vf, vc2);
            vpoly = _mm256_fmadd_ps(vpoly, vf, vc1);
            vpoly = _mm256_fmadd_ps(vpoly, vf, vc0);

            __m256i vi_n = _mm256_cvtps_epi32(vn);
            vi_n = _mm256_add_epi32(vi_n, _mm256_set1_epi32(127));
            __m256 v2n = _mm256_castsi256_ps(_mm256_slli_epi32(vi_n, 23));
            return _mm256_mul_ps(vpoly, v2n);
        }
static inline __m256 fast_sigmoid(__m256 vx)
        {
            __m256 vneg = _mm256_sub_ps(_mm256_setzero_ps(), vx);
            __m256 vexp_neg = fast_exp(vneg);
            __m256 vone = _mm256_set1_ps(1.0f);
            return _mm256_div_ps(vone, _mm256_add_ps(vone, vexp_neg));
        }
static inline __m256 fast_silu(__m256 vx)
        {
            return _mm256_mul_ps(vx, fast_sigmoid(vx));
        }
}
namespace cpu_detail {
#if defined(__AVX512F__)
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
}
int liveRows(RowExtent extent)
{
    const int count = extent.liveRows ? *extent.liveRows : extent.rows;
    qwen4exp::detail::require(count >= 0 && count <= extent.rows, "Qwen4Exp live rows exceed admitted capacity");
    return count;
}
float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }
template<class Function>
void channelBlocks(int rows, int width, Function function)
{
    const auto blocks = (static_cast<std::size_t>(width) + 255) / 256;
    const auto count = static_cast<std::size_t>(rows) * blocks;
    auto work = [&]() {
#pragma omp for schedule(static)
        for (std::size_t task = 0; task < count; ++task)
        {
            const int begin = static_cast<int>((task % blocks) * 256);
            function(task / blocks, begin, std::min(begin + 256, width));
        }
    };
    OMP_WORKSHARE_REGION(work);
}
/** @brief Frozen original residualDown arithmetic; no production nonlinear helper. */
inline void residualDown(const ResidualDownArgs &a, void *)
    {
        // Original public call supplies validated extent/storage before this oracle.
        const int vectorWidth = cpu_detail::vectorWidth();
        channelBlocks(liveRows(a.extent), a.lowRank, [&](std::size_t row, int begin, int end) {
            const auto base = row * a.lowRank;
            int channel = begin;
#if defined(__AVX512F__)
            if (vectorWidth == 16)
                for (; channel + 16 <= end; channel += 16)
                {
                    const auto value = _mm512_mul_ps(_mm512_loadu_ps(a.input + base + channel), _mm512_set1_ps(0.25f));
                    _mm512_storeu_ps(a.output + base + channel, _mm512_mul_ps(value, cpu_detail::sigmoid(value)));
                }
#endif
#if defined(__AVX2__)
            if (vectorWidth >= 8)
                for (; channel + 8 <= end; channel += 8)
                {
                    const auto value = _mm256_mul_ps(_mm256_loadu_ps(a.input + base + channel), _mm256_set1_ps(0.25f));
                    _mm256_storeu_ps(a.output + base + channel, avx2::fast_silu(value));
                }
#endif
            for (; channel < end; ++channel)
            {
                const float scaled = a.input[base + channel] * 0.25f;
                a.output[base + channel] = scaled * sigmoid(scaled);
            }
        });
    }
/** @brief Frozen original residualRead arithmetic; no production nonlinear helper. */
inline void residualRead(const ResidualReadArgs &a, void *)
    {
        // Original public call supplies validated extent/storage before this oracle.
        const auto hidden = static_cast<std::size_t>(a.hidden);
        const int vectorWidth = cpu_detail::vectorWidth();
        channelBlocks(liveRows(a.extent), a.hidden, [&](std::size_t row, int begin, int end) {
            int channel = begin;
#if defined(__AVX512F__)
            if (vectorWidth == 16)
                for (; channel + 16 <= end; channel += 16)
                {
                    auto sum = _mm512_setzero_ps();
                    for (int branch = 0; branch < 4; ++branch)
                    {
                        const auto offset = (row * 4 + branch) * hidden + channel;
                        sum = _mm512_fmadd_ps(cpu_detail::sigmoid(_mm512_loadu_ps(a.gateLogits + offset)),
                                             _mm512_loadu_ps(a.normalized + offset), sum);
                    }
                    _mm512_storeu_ps(a.output + row * hidden + channel, _mm512_mul_ps(sum, _mm512_set1_ps(0.25f)));
                }
#endif
#if defined(__AVX2__)
            if (vectorWidth >= 8)
                for (; channel + 8 <= end; channel += 8)
                {
                    auto sum = _mm256_setzero_ps();
                    for (int branch = 0; branch < 4; ++branch)
                    {
                        const auto offset = (row * 4 + branch) * hidden + channel;
                        sum = _mm256_fmadd_ps(avx2::fast_sigmoid(_mm256_loadu_ps(a.gateLogits + offset)),
                                             _mm256_loadu_ps(a.normalized + offset), sum);
                    }
                    _mm256_storeu_ps(a.output + row * hidden + channel, _mm256_mul_ps(sum, _mm256_set1_ps(0.25f)));
                }
#endif
            for (; channel < end; ++channel)
            {
                const auto base = row * 4 * hidden + channel;
                float sum = 0.0f;
                for (int branch = 0; branch < 4; ++branch)
                {
                    const auto offset = base + branch * hidden;
                    sum = std::fma(sigmoid(a.gateLogits[offset]), a.normalized[offset], sum);
                }
                a.output[row * hidden + channel] = sum * 0.25f;
            }
        });
    }
inline void quantize_single_block_scalar(const float *src, Q8_1Block &dst, int valid_elements)
        {
            // Find max absolute value
            float max_abs = 0.0f;
            for (int j = 0; j < valid_elements; ++j)
            {
                float val = std::abs(src[j]);
                if (val > max_abs)
                    max_abs = val;
            }

            // Handle zero/tiny values
            constexpr float MIN_SCALE_THRESHOLD = 1e-10f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                dst.d = 0;
                dst.sum_qs = 0;
                std::memset(dst.qs, 0, 32);
                return;
            }

            // Compute scale and inverse
            float d = max_abs / 127.0f;
            float id = 1.0f / d;
            dst.d = fp32_to_fp16(d);

            // Quantize values and compute sum
            int32_t sum_qs = 0;
            for (int j = 0; j < valid_elements; ++j)
            {
                int8_t q = static_cast<int8_t>(std::round(src[j] * id));
                dst.qs[j] = q;
                sum_qs += q;
            }

            // Zero-pad remaining elements
            for (int j = valid_elements; j < 32; ++j)
            {
                dst.qs[j] = 0;
            }

            dst.sum_qs = static_cast<int16_t>(sum_qs);
        }
#if defined(__AVX2__)
inline void quantize_single_block_avx2(const float *src, Q8_1Block &dst)
        {
            // Load all 32 floats (4 × 8)
            __m256 v0 = _mm256_loadu_ps(src);
            __m256 v1 = _mm256_loadu_ps(src + 8);
            __m256 v2 = _mm256_loadu_ps(src + 16);
            __m256 v3 = _mm256_loadu_ps(src + 24);

            // Compute absolute values using AND-NOT with sign bit mask
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 abs0 = _mm256_andnot_ps(sign_mask, v0);
            __m256 abs1 = _mm256_andnot_ps(sign_mask, v1);
            __m256 abs2 = _mm256_andnot_ps(sign_mask, v2);
            __m256 abs3 = _mm256_andnot_ps(sign_mask, v3);

            // Find max absolute value across all 32 elements
            __m256 max_01 = _mm256_max_ps(abs0, abs1);
            __m256 max_23 = _mm256_max_ps(abs2, abs3);
            __m256 max_all = _mm256_max_ps(max_01, max_23);

            // Horizontal max reduction
            __m128 lo = _mm256_castps256_ps128(max_all);
            __m128 hi = _mm256_extractf128_ps(max_all, 1);
            __m128 vmax128 = _mm_max_ps(lo, hi);
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
            vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
            float max_abs = _mm_cvtss_f32(vmax128);

            // Handle zero/tiny values
            constexpr float MIN_SCALE_THRESHOLD = 1e-10f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                dst.d = 0;
                dst.sum_qs = 0;
                std::memset(dst.qs, 0, 32);
                return;
            }

            // Compute scale
            float d = max_abs / 127.0f;
            float id = 1.0f / d;
            dst.d = fp32_to_fp16(d);

            // Scale (no explicit clamping needed: |v| ≤ max_abs ⇒ |scaled| ≤ 127,
            // and packs_epi32/packs_epi16 saturate any ±128 from FP rounding)
            __m256 vinv = _mm256_set1_ps(id);
            __m256 scaled0 = _mm256_mul_ps(v0, vinv);
            __m256 scaled1 = _mm256_mul_ps(v1, vinv);
            __m256 scaled2 = _mm256_mul_ps(v2, vinv);
            __m256 scaled3 = _mm256_mul_ps(v3, vinv);

            // Convert float → int32 (rounds to nearest-even per MXCSR)
            // No separate round_ps needed — cvtps_epi32 applies rounding.
            __m256i i0 = _mm256_cvtps_epi32(scaled0);
            __m256i i1 = _mm256_cvtps_epi32(scaled1);
            __m256i i2 = _mm256_cvtps_epi32(scaled2);
            __m256i i3 = _mm256_cvtps_epi32(scaled3);

            // Pack int32 → int16 → int8 (saturating at each step)
            // packs_epi32 interleaves 128-bit lanes, need permute to fix ordering
            __m256i i16_01 = _mm256_packs_epi32(i0, i1);
            __m256i i16_23 = _mm256_packs_epi32(i2, i3);
            i16_01 = _mm256_permute4x64_epi64(i16_01, _MM_SHUFFLE(3, 1, 2, 0));
            i16_23 = _mm256_permute4x64_epi64(i16_23, _MM_SHUFFLE(3, 1, 2, 0));

            __m256i i8_all = _mm256_packs_epi16(i16_01, i16_23);
            i8_all = _mm256_permute4x64_epi64(i8_all, _MM_SHUFFLE(3, 1, 2, 0));

            // Store 32 int8 values
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.qs), i8_all);

            // Compute sum from packed bytes using vpmaddubsw + vpmaddwd
            // (faster than tree-reduce on 32 int32 values — works on the saturated bytes directly)
            __m256i ones_u8 = _mm256_set1_epi8(1);
            __m256i paired = _mm256_maddubs_epi16(ones_u8, i8_all);
            __m256i ones_i16 = _mm256_set1_epi16(1);
            __m256i sum_i32_vec = _mm256_madd_epi16(paired, ones_i16);

            __m128i sum_lo = _mm256_castsi256_si128(sum_i32_vec);
            __m128i sum_hi = _mm256_extracti128_si256(sum_i32_vec, 1);
            __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 0, 0, 1)));

            dst.sum_qs = static_cast<int16_t>(_mm_cvtsi128_si32(sum128));
        }
#endif
#if defined(__AVX512F__)
inline void quantize_single_block_avx512(const float *src, Q8_1Block &dst)
        {
            // Load all 32 floats (2 × 16)
            __m512 v0 = _mm512_loadu_ps(src);
            __m512 v1 = _mm512_loadu_ps(src + 16);

            // Compute absolute values and find max
            __m512 abs0 = _mm512_abs_ps(v0);
            __m512 abs1 = _mm512_abs_ps(v1);
            __m512 max_vec = _mm512_max_ps(abs0, abs1);
            float max_abs = _mm512_reduce_max_ps(max_vec);

            // Handle zero/tiny values
            constexpr float MIN_SCALE_THRESHOLD = 1e-10f;
            if (max_abs < MIN_SCALE_THRESHOLD)
            {
                dst.d = 0;
                dst.sum_qs = 0;
                std::memset(dst.qs, 0, 32);
                return;
            }

            // Compute scale and convert to FP16 using F16C intrinsic
            float d = max_abs / 127.0f;
            float id = 1.0f / d;
            __m128 v_d = _mm_set_ss(d);
            __m128i v_h = _mm_cvtps_ph(v_d, _MM_FROUND_TO_NEAREST_INT);
            dst.d = (uint16_t)_mm_cvtsi128_si32(v_h);

            // Scale: no clamping needed — |v| ≤ max_abs ⇒ |scaled| ≤ 127.0
            // Rounding might push to ±128, but vpmovdb saturates to [-128,127].
            __m512 vinv = _mm512_set1_ps(id);
            __m512 scaled0 = _mm512_mul_ps(v0, vinv);
            __m512 scaled1 = _mm512_mul_ps(v1, vinv);

            // Convert float → int32 (rounds to nearest-even per MXCSR default)
            __m512i i0 = _mm512_cvtps_epi32(scaled0);
            __m512i i1 = _mm512_cvtps_epi32(scaled1);

            // Pack int32 → int8 with saturation via vpmovdb
            __m128i i8_0 = _mm512_cvtsepi32_epi8(i0);
            __m128i i8_1 = _mm512_cvtsepi32_epi8(i1);

            // Combine and store 32 quantized bytes
            __m256i i8_all = _mm256_set_m128i(i8_1, i8_0);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.qs), i8_all);

            // Compute sum of saturated bytes using vpmaddubsw + vpmaddwd
            // (eliminates costly sign-extend-back-to-i32 + 512-bit horizontal reduce)
            //
            // vpmaddubsw(ones_u8, signed_bytes): for each pair of bytes,
            //   result_i16[k] = 1*qs[2k] + 1*qs[2k+1]  (saturating, but |sum| ≤ 254 so no sat)
            // vpmaddwd(paired_i16, ones_i16): for each pair of int16,
            //   result_i32[k] = paired[2k]*1 + paired[2k+1]*1  (sum of 4 bytes per lane)
            // Then 256-bit horizontal reduce of 8 int32 values.
            __m256i ones_u8_256 = _mm256_set1_epi8(1);
            __m256i paired_i16 = _mm256_maddubs_epi16(ones_u8_256, i8_all);
            __m256i ones_i16 = _mm256_set1_epi16(1);
            __m256i sum_i32_vec = _mm256_madd_epi16(paired_i16, ones_i16);

            // Horizontal reduce 8 int32 → scalar
            __m128i lo = _mm256_castsi256_si128(sum_i32_vec);
            __m128i hi = _mm256_extracti128_si256(sum_i32_vec, 1);
            __m128i sum128 = _mm_add_epi32(lo, hi);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 0, 0, 1)));
            int32_t sum_i32 = _mm_cvtsi128_si32(sum128);

            dst.sum_qs = static_cast<int16_t>(sum_i32);
        }
#endif
inline void quantize_single_block(const float *src, Q8_1Block &dst, int valid_elements = 32)
        {
            // Partial blocks always use scalar (rare, at row boundaries only)
            if (valid_elements < 32)
            {
                quantize_single_block_scalar(src, dst, valid_elements);
                return;
            }

            // Full blocks: dispatch to best SIMD
            static const bool has_avx512 = cpu_supports_avx512();
            static const bool has_avx2 = cpu_supports_avx2();

#if defined(__AVX512F__)
            if (has_avx512)
            {
                quantize_single_block_avx512(src, dst);
                return;
            }
#endif
#if defined(__AVX2__)
            if (has_avx2)
            {
                quantize_single_block_avx2(src, dst);
                return;
            }
#endif
            quantize_single_block_scalar(src, dst, 32);
        }
inline void quantizeResidualBlock(const float*input,const Q8_1Block&primary,Q8_1Block&residual,
 llaminar2::CPUProjectionNumericalPolicy policy) {
 alignas(64)float values[32];const auto scale=_mm256_set1_ps(nativeVNNIFP16ScaleToFP32(primary.d));
 for(int i=0;i<32;i+=8) {
  const auto bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(primary.qs+i));
  const auto decoded=persistRoundedAVX2(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),scale));
  _mm256_store_ps(values+i,_mm256_sub_ps(_mm256_loadu_ps(input+i),decoded));
 }
 (void)policy; quantize_single_block(values,residual,32);
}
/** @brief Publish complete live blocks using original hardware quantizer/half residual policy. */
inline void publish(const float *activated,int live,int k,Q8_1Block *primary,Q8_1Block *residual) {
 for(int row=0;row<live;++row)for(int block=0;block<k/32;++block) {
  const auto i=std::size_t(row)*(k/32)+block;const auto *source=activated+std::size_t(row)*k+block*32;
  for(int c=0;c<32;++c)if(!std::isfinite(source[c]))throw std::invalid_argument("Frozen CPU HC activation nonfinite");
  quantize_single_block(source,primary[i],32);
  if((primary[i].d&0x7c00u)==0x7c00u)throw std::invalid_argument("Frozen CPU HC primary half scale overflow");
  quantizeResidualBlock(source,primary[i],residual[i],CPUProjectionNumericalPolicy::BackendNative);
 }
}
} // namespace llaminar2::test::hc103_original
