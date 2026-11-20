/**
 * @file CPUInt32RMSNormPublication.h
 * @brief Allocation-free original-INT32 normalization and row-INT8 publication.
 *
 * Original integer/gamma/inverse values remain caller-owned. Two native passes
 * recompute the same separately rounded pointwise products: first maximum, then
 * publication. No decoded FP32 row, TLS bank or workspace is materialized.
 * This row-scale program is distinct from block-Q8 and FP32 row-INT8: no1e-8
 * floor, zero inverse produces zero codes/scale1, native vectors use RNE while
 * bounded scalar tails retain ties-away. Enclosing borrows own complete-bank
 * exclusion and workshare lifetime; this synchronous leaf is not collective.
 */
#pragma once
#include "CPURMSNormBorrow.h"
#include "RMSNumericFatal.h"
#include "../../common/DeviceFP32NumericalContract.h"
#include "../../../utils/CPUFeatures.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#if LLAMINAR_COMPILED_WITH_NEON
#include "../../../tensors/NEONFP32.h"
#elif defined(__AVX2__) || defined(__AVX512F__)
#include "../../../tensors/X86FP32.h"
#endif

namespace llaminar2::primitives::int32_rms_publication
{
/** @brief Reject numeric corruption before a row publishes any code/scale.
 * @details Exceptions must not cross the enclosing OpenMP collective. Previously
 * retired rows are not rolled back; no alternate numeric program is selected.
 */
[[noreturn]] inline void rejectNumeric()
{
    rms_numeric::rejectNumeric(rms_numeric::NumericFailure::UnrepresentablePublication,
        "INT32 RMS row publication observed nonfinite/unrepresentable normalization or scale");
}
/** @brief Compute the original pointwise FP32 normalization with explicit round boundaries.
 * @param value Original integer; @param inverse Original row inverse.
 * @param gamma Original optional scalar gamma; @return FP32-normalized value.
 */
inline float normalizeScalar(std::int32_t value,float inverse,const float*gamma)
{
    const auto normalized=device_fp32_contract::multiply(float(value),inverse);
    return gamma?device_fp32_contract::multiply(normalized,*gamma):normalized;
}
/** @brief Scalar remainder publication retains its separate ties-away program.
 * @param value Finite original normalized value; @param inverseScale Finite row reciprocal.
 * @return One bounded signed code, never -128.
 */
inline std::int8_t scalarCode(float value,float inverseScale)
{
    const auto scaled=device_fp32_contract::multiply(value,inverseScale);
    return static_cast<std::int8_t>(std::round(std::clamp(scaled,-127.0f,127.0f)));
}
/** @brief Resolve this row program's scale without borrowing another quantizer's floor.
 * @param maximum Complete finite normalized absolute maximum.
 * @return Positive scale; unrepresentable scale/reciprocal is fatal before stores.
 */
inline float scaleForMaximum(float maximum)
{
    const float scale=maximum>0.0f?maximum/127.0f:1.0f;
    if(!std::isfinite(maximum)||!std::isfinite(scale)||scale<=0.0f||!std::isfinite(1.0f/scale))rejectNumeric();
    return scale;
}

#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Original eight-coordinate NEON publication program, represented only in registers. */
struct NEON
{
    /** @brief Two native four-lane fragments form one eight-code publication. */
    struct Vector{float32x4_t low,high;};
    using Maximum=float32x4_t;
    static constexpr std::size_t width=8; ///< RNE coordinates per complete native fragment.
    /** @brief Load/normalize original words with no decoded intermediate bank.
     * @param source Original eight integers; @param gamma Optional original weights.
     * @param inverse Original inverse; @return Separately rounded normalized registers.
     */
    static Vector normalized(const std::int32_t*source,const float*gamma,float inverse)
    {
        const auto inv=vdupq_n_f32(inverse);
        auto low=neon_fp32::multiply(vcvtq_f32_s32(vld1q_s32(source)),inv);
        auto high=neon_fp32::multiply(vcvtq_f32_s32(vld1q_s32(source+4)),inv);
        if(gamma){low=neon_fp32::multiply(low,vld1q_f32(gamma));high=neon_fp32::multiply(high,vld1q_f32(gamma+4));}
        return {low,high};
    }
    /** @return Initial nonnegative maximum lanes. */
    static Maximum zero(){return vdupq_n_f32(0);}
    /** @brief Validate every active lane, including NaNs that a maximum could otherwise hide.
     * @param value Normalized original fragment; @return Whether every lane is finite.
     */
    static bool finite(Vector value)
    {
        const auto exponent=vdupq_n_u32(0x7f800000u);
        const auto bad=vorrq_u32(vceqq_u32(vandq_u32(vreinterpretq_u32_f32(value.low),exponent),exponent),
            vceqq_u32(vandq_u32(vreinterpretq_u32_f32(value.high),exponent),exponent));
        return vmaxvq_u32(bad)==0;
    }
    /** @brief Extend the original absolute maximum; @param old Prior lanes; @param value Next fragment.
     * @return Updated lanes; no floating sum/reduction order is introduced.
     */
    static Maximum update(Maximum old,Vector value){return vmaxq_f32(old,vmaxq_f32(vabsq_f32(value.low),vabsq_f32(value.high)));}
    /** @param value Complete maximum lanes; @return Original scalar maximum. */
    static float maximum(Maximum value){return vmaxvq_f32(value);}
    /** @brief Publish one complete original eight-code fragment using RNE.
     * @param destination Disjoint eight-byte output; @param value Finite normalized lanes.
     * @param inverseScale Original finite row reciprocal.
     */
    static void publish(std::int8_t*destination,Vector value,float inverseScale)
    {
        const auto inv=vdupq_n_f32(inverseScale),lo=vdupq_n_f32(-127),hi=vdupq_n_f32(127);
        const auto a=vcvtnq_s32_f32(vminq_f32(hi,vmaxq_f32(lo,neon_fp32::multiply(value.low,inv))));
        const auto b=vcvtnq_s32_f32(vminq_f32(hi,vmaxq_f32(lo,neon_fp32::multiply(value.high,inv))));
        vst1_s8(destination,vmovn_s16(vcombine_s16(vmovn_s32(a),vmovn_s32(b))));
    }
};
#endif
#if defined(__AVX2__)
/** @brief Original eight-coordinate AVX2 normalization and signed publication. */
struct AVX2
{
    using Vector=__m256;using Maximum=__m256;
    static constexpr std::size_t width=8; ///< Complete native RNE fragment width.
    /** @brief Normalize original integers with separately retained products.
     * @param source Original eight integers; @param gamma Optional original weights.
     * @param inverse Original row inverse; @return Finite-checked by the consumer.
     */
    static Vector normalized(const std::int32_t*source,const float*gamma,float inverse)
    {
        const auto words=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
        auto value=x86_fp32::persistRounded(_mm256_mul_ps(_mm256_cvtepi32_ps(words),_mm256_set1_ps(inverse)));
        if(gamma)value=x86_fp32::persistRounded(_mm256_mul_ps(value,_mm256_loadu_ps(gamma)));
        return value;
    }
    /** @return Initial maximum lanes. */
    static Maximum zero(){return _mm256_setzero_ps();}
    /** @param value Original normalized registers; @return Whether all lanes are finite. */
    static bool finite(Vector value)
    {
        const auto bits=_mm256_castps_si256(value),exponent=_mm256_set1_epi32(0x7f800000);
        return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(bits,exponent),exponent)))==0;
    }
    /** @param old Prior maximum; @param value Next original fragment; @return Extended absolute maximum. */
    static Maximum update(Maximum old,Vector value){return _mm256_max_ps(old,_mm256_andnot_ps(_mm256_set1_ps(-0.0f),value));}
    /** @param value Complete maximum lanes; @return Original scalar maximum. */
    static float maximum(Maximum value)
    {
        auto half=_mm_max_ps(_mm256_castps256_ps128(value),_mm256_extractf128_ps(value,1));
        half=_mm_max_ps(half,_mm_movehdup_ps(half));half=_mm_max_ss(half,_mm_movehl_ps(half,half));return _mm_cvtss_f32(half);
    }
    /** @brief Publish eight codes with the existing native RNE program.
     * @param destination Complete disjoint output; @param value Finite normalized lanes.
     * @param inverseScale Original row reciprocal; caller retains default FP environment.
     */
    static void publish(std::int8_t*destination,Vector value,float inverseScale)
    {
        auto scaled=x86_fp32::persistRounded(_mm256_mul_ps(value,_mm256_set1_ps(inverseScale)));
        scaled=_mm256_min_ps(_mm256_set1_ps(127),_mm256_max_ps(_mm256_set1_ps(-127),scaled));
        const auto integers=_mm256_cvtps_epi32(scaled);
        const auto shorts=_mm_packs_epi32(_mm256_castsi256_si128(integers),_mm256_extracti128_si256(integers,1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(destination),_mm_packs_epi16(shorts,shorts));
    }
};
#endif
#if defined(__AVX512F__)
/** @brief Original sixteen-coordinate AVX512 normalization and signed publication. */
struct AVX512
{
    using Vector=__m512;using Maximum=__m512;
    static constexpr std::size_t width=16; ///< Existing native RNE fragment width.
    /** @brief Normalize original integers without rebinding data or gamma.
     * @param source Original sixteen integers; @param gamma Optional original weights.
     * @param inverse Original row inverse; @return Separately rounded normalized registers.
     */
    static Vector normalized(const std::int32_t*source,const float*gamma,float inverse)
    {
        const auto words=_mm512_loadu_si512(reinterpret_cast<const void*>(source));
        auto value=x86_fp32::persistRounded(_mm512_mul_ps(_mm512_cvtepi32_ps(words),_mm512_set1_ps(inverse)));
        if(gamma)value=x86_fp32::persistRounded(_mm512_mul_ps(value,_mm512_loadu_ps(gamma)));
        return value;
    }
    /** @return Initial maximum lanes. */
    static Maximum zero(){return _mm512_setzero_ps();}
    /** @param value Original normalized registers; @return Whether all lanes are finite. */
    static bool finite(Vector value)
    {
        const auto exponent=_mm512_set1_epi32(0x7f800000);
        return _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(value),exponent),exponent)==0;
    }
    /** @param old Prior maximum; @param value Next original fragment; @return Extended absolute maximum. */
    static Maximum update(Maximum old,Vector value){return _mm512_max_ps(old,_mm512_abs_ps(value));}
    /** @param value Complete maximum lanes; @return Original scalar maximum. */
    static float maximum(Maximum value){return _mm512_reduce_max_ps(value);}
    /** @brief Publish sixteen codes using the existing native RNE instruction.
     * @param destination Complete disjoint output; @param value Finite normalized registers.
     * @param inverseScale Original row reciprocal; no FP environment is modified.
     */
    static void publish(std::int8_t*destination,Vector value,float inverseScale)
    {
        auto scaled=x86_fp32::persistRounded(_mm512_mul_ps(value,_mm512_set1_ps(inverseScale)));
        scaled=_mm512_min_ps(_mm512_set1_ps(127),_mm512_max_ps(_mm512_set1_ps(-127),scaled));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination),_mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(scaled)));
    }
};
#endif

/** @brief Recompute original pointwise fragments for allocation-free native row publication.
 * @tparam Program Exact admitted native ISA program, including its RNE fragment width.
 * @param source Complete original integer row; @param gamma Optional original gamma.
 * @param inverse Original finite positive inverse; @param output Complete disjoint row.
 * @return Original dynamic scale, after every output byte is authored.
 * @details Full-bank ownership/alias validation belongs to the enclosing typed
 * invocation. Numeric corruption is fatal before this row's first store.
 */
template<class Program>
float nativeRow(std::span<const std::int32_t>source,const float*gamma,float inverse,std::span<std::int8_t>output)
{
    auto lanes=Program::zero();std::size_t c=0;
    for(;source.size()-c>=Program::width;c+=Program::width)
    {
        const auto values=Program::normalized(source.data()+c,gamma?gamma+c:nullptr,inverse);
        if(!Program::finite(values))rejectNumeric();lanes=Program::update(lanes,values);
    }
    float maximum=Program::maximum(lanes);
    for(;c<source.size();++c)
    {
        const auto value=normalizeScalar(source[c],inverse,gamma?gamma+c:nullptr);
        if(!std::isfinite(value))rejectNumeric();maximum=std::max(maximum,std::fabs(value));
    }
    const auto scale=scaleForMaximum(maximum);const auto reciprocal=1.0f/scale;c=0;
    for(;source.size()-c>=Program::width;c+=Program::width)
        Program::publish(output.data()+c,Program::normalized(source.data()+c,gamma?gamma+c:nullptr,inverse),reciprocal);
    for(;c<source.size();++c)output[c]=scalarCode(normalizeScalar(source[c],inverse,gamma?gamma+c:nullptr),reciprocal);
    return scale;
}
/** @brief Publish one owned original row under its already authenticated ISA.
 * @param row Typed original row descriptor, constructible only by complete-bank admission.
 * @param inverse Original row inverse; @param isa Exact native program or explicit scalar diagnostic.
 * @details Payload and scale are authored by the same row owner. Zero inverse
 * yields zero codes/scale1, without reading inactive gamma values.
 */
inline void publishRow(const rmsnorm_operands::Int32QuantizationBorrow::Row&row,float inverse,ISALevel isa)
{
    const auto source=row.source();const auto output=row.output();
    const auto weights=row.gamma();const auto *gamma=weights.empty()?nullptr:weights.data();
    const float scale=[&]() -> float
    {
    if(!std::isfinite(inverse)||inverse<0.0f)rejectNumeric();
    if(inverse==0.0f){std::fill(output.begin(),output.end(),std::int8_t(0));return 1.0f;}
    switch(isa)
    {
#if LLAMINAR_COMPILED_WITH_NEON
    case ISALevel::NEON:case ISALevel::NEONDotProd:return nativeRow<NEON>(source,gamma,inverse,output);
#endif
#if defined(__AVX2__)
    case ISALevel::AVX2:return nativeRow<AVX2>(source,gamma,inverse,output);
#endif
#if defined(__AVX512F__)
    case ISALevel::AVX512:return nativeRow<AVX512>(source,gamma,inverse,output);
#endif
    case ISALevel::Scalar:
    {
        float maximum=0;
        for(std::size_t c=0;c<source.size();++c)
        {
            const auto value=normalizeScalar(source[c],inverse,gamma?gamma+c:nullptr);
            if(!std::isfinite(value))rejectNumeric();maximum=std::max(maximum,std::fabs(value));
        }
        const auto scale=scaleForMaximum(maximum),reciprocal=1.0f/scale;
        for(std::size_t c=0;c<source.size();++c)output[c]=scalarCode(normalizeScalar(source[c],inverse,gamma?gamma+c:nullptr),reciprocal);
        return scale;
    }
    default:rejectNumeric();
    }
    }();
    row.publishScale(scale);
}
} // namespace llaminar2::primitives::int32_rms_publication
