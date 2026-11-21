/**
 * @file CPUQwenSparseRepresentation.h
 * @brief Register-only QSA FP32-container casts with the full IEEE RNE program.
 *
 * QSA explicitly rounds half subnormals to nearest-even and canonicalizes half
 * NaNs. That is NOT NEONFP16's distinct portable truncated-subnormal storage
 * narrowing. Integer lane programs force the declared representation boundary
 * without a lane array, changed FP environment, per-head conversion or cache.
 * The enclosing synchronous owner admits original operands/native ISA and
 * rejects consumed nonfinite arithmetic; decode-only conformance remains separate.
 */
#pragma once
#include "CPUQwenSparseFP32Program.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
namespace llaminar2::cpu::qsa_native
{
#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Round four FP32 words at a QSA half/BF16 cast boundary using native integer lanes.
 * @tparam Representation One closed admitted QSA representation.
 * @param value Original four FP32 words; @return Exact rounded FP32 containers.
 */
template<qwen4exp::QwenSparseScalar Representation>
[[gnu::always_inline]] inline float32x4_t representation4(float32x4_t value)noexcept
{
    using Scalar=qwen4exp::QwenSparseScalar;
    static_assert(Representation==Scalar::FP32||Representation==Scalar::FP16||Representation==Scalar::BF16);
    if constexpr(Representation==Scalar::FP32)return value;
    else
    {
        // Ordered entry/exit dependencies keep independent half casts from
        // interleaving their large integer live sets after caller inlining.
        if constexpr(Representation==Scalar::FP16)value=neon_fp32::persistRounded(value);
        const auto bits=vreinterpretq_u32_f32(value),magnitude=vandq_u32(bits,vdupq_n_u32(0x7fffffffu));
        if constexpr(Representation==Scalar::BF16)
        {
            const auto rounded=vandq_u32(vaddq_u32(bits,vaddq_u32(vdupq_n_u32(0x7fffu),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1)))),vdupq_n_u32(0xffff0000u));
            const auto nan=vorrq_u32(vandq_u32(bits,vdupq_n_u32(0xffff0000u)),vdupq_n_u32(0x00400000u));
            return vreinterpretq_f32_u32(vbslq_u32(vcgtq_u32(magnitude,vdupq_n_u32(0x7f800000u)),nan,rounded));
        }
        else
        {
            // Keep rounded containers in FP32 instead of packing half words
            // and immediately widening them. This shortens the integer live set.
            const auto rounded=vandq_u32(vaddq_u32(magnitude,vaddq_u32(vdupq_n_u32(0xfffu),vandq_u32(vshrq_n_u32(magnitude,13),vdupq_n_u32(1)))),vdupq_n_u32(0xffffe000u));
            const auto normal=vbslq_u32(vcgeq_u32(rounded,vdupq_n_u32(0x47800000u)),vdupq_n_u32(0x7f800000u),rounded);
            const auto shift=vsubq_s32(vdupq_n_s32(126),vreinterpretq_s32_u32(vshrq_n_u32(magnitude,23)));
            const auto mantissa=vorrq_u32(vandq_u32(magnitude,vdupq_n_u32(0x7fffffu)),vdupq_n_u32(0x800000u));
            const auto truncated=vshlq_u32(mantissa,vnegq_s32(shift));
            const auto bias=vsubq_u32(vshlq_u32(vdupq_n_u32(1),vsubq_s32(shift,vdupq_n_s32(1))),vdupq_n_u32(1));
            const auto units=vshlq_u32(vaddq_u32(mantissa,vaddq_u32(bias,vandq_u32(truncated,vdupq_n_u32(1)))),vnegq_s32(shift));
            const auto subnormalMask=vcltq_u32(magnitude,vdupq_n_u32(0x38800000u));
            const auto validSubnormal=vandq_u32(subnormalMask,vcgeq_u32(magnitude,vdupq_n_u32(0x33000000u)));
            // Only integers0..1024 enter conversion, and units*2^-24 is exact
            // binary32, so neither operation depends on a floating rounding mode.
            const auto subnormal=vreinterpretq_u32_f32(vmulq_n_f32(vcvtq_f32_u32(vandq_u32(units,validSubnormal)),0x1p-24f));
            auto result=vbslq_u32(subnormalMask,subnormal,normal);
            const auto special=vorrq_u32(vdupq_n_u32(0x7f800000u),vandq_u32(vcgtq_u32(magnitude,vdupq_n_u32(0x7f800000u)),vdupq_n_u32(0x00400000u)));
            result=vbslq_u32(vcgeq_u32(magnitude,vdupq_n_u32(0x7f800000u)),special,result);
            return neon_fp32::persistRounded(vreinterpretq_f32_u32(vorrq_u32(result,vandq_u32(bits,vdupq_n_u32(0x80000000u)))));
        }
    }
}
#endif
/** @brief Round exactly eight original FP32 containers without a decoded/repacked row.
 * @tparam Representation One closed original QSA representation.
 * @param value Original native coordinates; @return Exact original cast program.
 */
template<qwen4exp::QwenSparseScalar Representation>
[[gnu::always_inline]] inline Fragment8 roundRepresentation8(Fragment8 value)noexcept
{
    using Scalar=qwen4exp::QwenSparseScalar;
    static_assert(Representation==Scalar::FP32||Representation==Scalar::FP16||Representation==Scalar::BF16);
#if LLAMINAR_COMPILED_WITH_NEON
    return {representation4<Representation>(value.low),representation4<Representation>(value.high)};
#else
    if constexpr(Representation==Scalar::FP32)return value;
    else
    {
        const auto bits=_mm256_castps_si256(value),magnitude=_mm256_and_si256(bits,_mm256_set1_epi32(0x7fffffff));
        const auto select=[](__m256i mask,__m256i yes,__m256i no){return _mm256_blendv_epi8(no,yes,mask);};
        if constexpr(Representation==Scalar::BF16)
        {
            const auto rounded=_mm256_and_si256(_mm256_add_epi32(bits,_mm256_add_epi32(_mm256_set1_epi32(0x7fff),_mm256_and_si256(_mm256_srli_epi32(bits,16),_mm256_set1_epi32(1)))),_mm256_set1_epi32(int(0xffff0000u)));
            const auto nan=_mm256_or_si256(_mm256_and_si256(bits,_mm256_set1_epi32(int(0xffff0000u))),_mm256_set1_epi32(0x00400000));
            return _mm256_castsi256_ps(select(_mm256_cmpgt_epi32(magnitude,_mm256_set1_epi32(0x7f800000)),nan,rounded));
        }
        else
        {
            const auto rounded=_mm256_and_si256(_mm256_add_epi32(magnitude,_mm256_add_epi32(_mm256_set1_epi32(0xfff),_mm256_and_si256(_mm256_srli_epi32(magnitude,13),_mm256_set1_epi32(1)))),_mm256_set1_epi32(int(0xffffe000u)));
            // Signed comparisons need a magnitude sign-bit test after a carry
            // from a maximal NaN word; special words are selected canonically.
            const auto overflow=_mm256_or_si256(_mm256_cmpgt_epi32(rounded,_mm256_set1_epi32(0x477fffff)),_mm256_cmpgt_epi32(_mm256_setzero_si256(),rounded));
            const auto normal=select(overflow,_mm256_set1_epi32(0x7f800000),rounded);
            const auto shift=_mm256_sub_epi32(_mm256_set1_epi32(126),_mm256_srli_epi32(magnitude,23));
            const auto mantissa=_mm256_or_si256(_mm256_and_si256(magnitude,_mm256_set1_epi32(0x7fffff)),_mm256_set1_epi32(0x800000));
            const auto truncated=_mm256_srlv_epi32(mantissa,shift);
            const auto bias=_mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1),_mm256_sub_epi32(shift,_mm256_set1_epi32(1))),_mm256_set1_epi32(1));
            const auto units=_mm256_srlv_epi32(_mm256_add_epi32(mantissa,_mm256_add_epi32(bias,_mm256_and_si256(truncated,_mm256_set1_epi32(1)))),shift);
            const auto subnormalMask=_mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000),magnitude);
            const auto validSubnormal=_mm256_and_si256(subnormalMask,_mm256_cmpgt_epi32(magnitude,_mm256_set1_epi32(0x32ffffff)));
            const auto subnormal=_mm256_castps_si256(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(units,validSubnormal)),_mm256_set1_ps(0x1p-24f)));
            auto result=select(subnormalMask,subnormal,normal);
            const auto special=_mm256_or_si256(_mm256_set1_epi32(0x7f800000),_mm256_and_si256(_mm256_cmpgt_epi32(magnitude,_mm256_set1_epi32(0x7f800000)),_mm256_set1_epi32(0x00400000)));
            result=select(_mm256_cmpgt_epi32(magnitude,_mm256_set1_epi32(0x7f7fffff)),special,result);
            return _mm256_castsi256_ps(_mm256_or_si256(result,_mm256_and_si256(bits,_mm256_set1_epi32(int(0x80000000u)))));
        }
    }
#endif
}
} // namespace llaminar2::cpu::qsa_native
