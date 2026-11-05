/**
 * @file NEONSoftmaxPrimitives.h
 * @brief Native three-pass CPU softmax over checked original FP32/BF16/FP16/Q8 rows.
 *
 * Max and fixed four-lane FP64 summary passes finish before any in-place write.
 * Exponentials use a native split-ln2 FMA polynomial, with explicit normal and
 * subnormal scaling rather than a whole-row scalar libm substitute. Publication
 * streams bounded four-value fragments; no head mirror, allocation, hidden team
 * or decoded cache representation exists. Q8 metadata is the declared FP16 ABI.
 */
#pragma once
#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native softmax requires AArch64 ASIMD"
#endif
#include "CPUSoftmaxOperands.h"
#include "CPUQ8SoftmaxMetadata.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/NEONFP16.h"
#include <arm_neon.h>
#include <bit>
#include <cstring>

#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_SOFTMAX_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_NEON_SOFTMAX_EXACT
#endif
#if defined(__clang__)
#define LLAMINAR_NEON_SOFTMAX_ROUNDING _Pragma("clang fp contract(off)") _Pragma("clang fp reassociate(off)")
#else
#define LLAMINAR_NEON_SOFTMAX_ROUNDING
#endif

namespace llaminar2::primitives::neon_softmax
{
using softmax_detail::Storage;
using softmax_detail::Row;
static_assert(std::endian::native==std::endian::little,
    "Native Q8 softmax requires the supported little-endian CPU payload ABI");
/** @brief Select bounded original lanes, with no reads beyond a partial fragment.
 * @param count Original active coordinates in this four-lane fragment, in [0,4].
 * @return Native active-lane mask.
 */
inline uint32x4_t lanes(std::size_t count) noexcept
{
    const std::uint32_t indices[4]{0,1,2,3};
    return vcltq_u32(vld1q_u32(indices), vdupq_n_u32(std::uint32_t(count)));
}
/** @brief Evaluate nonpositive exponentials without discarding representable subnormals.
 * @param value Original shifted scores, finite <=0 or negative infinity.
 * @return Four approximate exp words; masked/-infinite and true underflow lanes are zero.
 * @details Exponents below -126 split into two normal factors. The last multiply
 * alone creates the subnormal result, instead of prematurely rounding the power
 * of two to zero. This is not the expert SwiGLU program's -87 cutoff contract.
 */
[[gnu::always_inline]] LLAMINAR_NEON_SOFTMAX_EXACT inline float32x4_t exponential(float32x4_t value) noexcept
{
    LLAMINAR_NEON_SOFTMAX_ROUNDING
    const auto bounded = vmaxq_f32(value, vdupq_n_f32(-104.0f));
    const auto exponent = vcvtnq_s32_f32(vmulq_n_f32(bounded, 0x1.715476p+0f));
    const auto exponent_float = vcvtq_f32_s32(exponent);
    auto remainder = vfmsq_n_f32(bounded, exponent_float, 0x1.62e400p-1f);
    remainder = vfmsq_n_f32(remainder, exponent_float, 0x1.7f7d1cp-20f);
    auto polynomial = vfmaq_f32(vdupq_n_f32(0x1.a01a02p-13f), vdupq_n_f32(0x1.a01a02p-16f), remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(0x1.6c16c2p-10f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(0x1.111112p-7f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(0x1.555556p-5f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(0x1.555556p-3f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(0x1.000000p-1f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f), polynomial, remainder);
    const auto normal_exponent = vmaxq_s32(exponent, vdupq_n_s32(-126));
    const auto residual_exponent = vsubq_s32(exponent, normal_exponent);
    const auto normal = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(normal_exponent, vdupq_n_s32(127)),23));
    const auto residual = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(residual_exponent, vdupq_n_s32(127)),23));
    const auto result = vmulq_f32(vmulq_f32(polynomial, residual), normal);
    return vbslq_f32(vcleq_f32(value, vdupq_n_f32(-104.0f)), vdupq_n_f32(0.0f), result);
}

/** @brief Four fixed FP64 chains, independent of worker count and surrounding row grouping. */
class Summary final
{
public:
    /** @brief Start four independent positive-exponential accumulation chains. */
    Summary() noexcept : lo_(vdupq_n_f64(0.0)), hi_(vdupq_n_f64(0.0)) {}
    /** @brief Accumulate original coordinate classes modulo four in fixed order.
     * @param values Original binary32 exponential results, with zero inactive lanes.
     */
    LLAMINAR_NEON_SOFTMAX_EXACT void add(float32x4_t values) noexcept
    {
        LLAMINAR_NEON_SOFTMAX_ROUNDING
        lo_ = vaddq_f64(lo_, vcvt_f64_f32(vget_low_f32(values)));
        hi_ = vaddq_f64(hi_, vcvt_f64_f32(vget_high_f32(values)));
    }
    /** @brief Fold adjacent pairs, then publish one rounded reciprocal.
     * @return Finite positive binary32 inverse of the fixed FP64 total.
     * @throws std::domain_error For a broken/nonfinite normalization program.
     */
    LLAMINAR_NEON_SOFTMAX_EXACT float inverse() const
    {
        LLAMINAR_NEON_SOFTMAX_ROUNDING
        const double sum = vaddvq_f64(lo_) + vaddvq_f64(hi_);
        if (!(sum > 0.0) || !std::isfinite(sum))
            throw std::domain_error("CPU native softmax has an invalid exponential summary");
        return static_cast<float>(1.0 / sum);
    }
private:
    float64x2_t lo_, hi_;
};

/** @brief Load a bounded original floating-storage fragment.
 * @tparam Format Original FP32/BF16/FP16 storage.
 * @param source Original immutable coordinates; @param count Original valid coordinate count [1,4].
 * @return Four FP32 words, with zero padding not read from original storage.
 */
template<Storage Format>
[[gnu::always_inline]] inline float32x4_t loadFloating(const softmax_detail::Unit<Format> *source, std::size_t count) noexcept
{
    static_assert(Format != Storage::Q8_1);
    if constexpr (Format == Storage::FP32)
    {
        if (count == 4) return vld1q_f32(source);
        float fragment[4]{}; for (std::size_t lane=0;lane<count;++lane) fragment[lane]=source[lane];
        return vld1q_f32(fragment);
    }
    else
    {
        uint16x4_t raw;
        if (count == 4) raw=vld1_u16(source);
        else
        {
            std::uint16_t fragment[4]{}; for(std::size_t lane=0;lane<count;++lane) fragment[lane]=source[lane];
            raw=vld1_u16(fragment);
        }
        if constexpr (Format == Storage::Half) return neon_fp16::widen(raw);
        else return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(raw),16));
    }
}
/** @brief Store a bounded original floating-storage result with canonical narrowing.
 * @tparam Format Original result storage.
 * @param destination Original in-place output fragment; @param value Four normalized results.
 * @param count Original valid result coordinates [1,4].
 */
template<Storage Format>
[[gnu::always_inline]] inline void storeFloating(softmax_detail::Unit<Format> *destination, float32x4_t value, std::size_t count) noexcept
{
    if constexpr (Format == Storage::FP32)
    {
        if(count==4) {vst1q_f32(destination,value);return;}
        float fragment[4];vst1q_f32(fragment,value);for(std::size_t lane=0;lane<count;++lane)destination[lane]=fragment[lane];
    }
    else
    {
        uint16x4_t raw;
        if constexpr(Format==Storage::Half) raw=neon_fp16::narrow(value);
        else
        {
            const auto bits=vreinterpretq_u32_f32(value);
            const auto bias=vaddq_u32(vdupq_n_u32(0x7fffu),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1)));
            raw=vmovn_u32(vshrq_n_u32(vaddq_u32(bits,bias),16));
        }
        if(count==4) {vst1_u16(destination,raw);return;}
        std::uint16_t fragment[4];vst1_u16(fragment,raw);for(std::size_t lane=0;lane<count;++lane)destination[lane]=fragment[lane];
    }
}
/** @brief Load separately rounded score multiplication and mask inactive lanes after scaling.
 * @tparam Format Original floating storage.
 * @param row Checked original row; @param offset Original coordinate; @param count Active coordinates [1,4].
 * @return Original scaled scores, with negative-infinite inactive lanes.
 */
template<Storage Format>
[[gnu::always_inline]] LLAMINAR_NEON_SOFTMAX_EXACT inline float32x4_t scaledFloating(const Row<Format> &row, std::size_t offset, std::size_t count) noexcept
{
    LLAMINAR_NEON_SOFTMAX_ROUNDING
    const auto value=vmulq_n_f32(loadFloating<Format>(row.data()+offset,count),row.scale());
    return vbslq_f32(lanes(count),value,vdupq_n_f32(-std::numeric_limits<float>::infinity()));
}
/** @brief Complete native floating-row normalization before publishing in-place fragments.
 * @tparam Format Original FP32/BF16/FP16 storage.
 * @param row Checked original row and original causal coordinate.
 * @throws std::domain_error For an active NaN/positive-infinite maximum or invalid summary.
 * @details FMAX propagates NaNs: finite lanes cannot conceal an invalid active
 * score. All-negative-infinite or fully masked rows deliberately author zeros.
 */
template<Storage Format>
LLAMINAR_NEON_SOFTMAX_EXACT inline void execute(const Row<Format> &row)
{
    LLAMINAR_NEON_SOFTMAX_ROUNDING
    static_assert(Format!=Storage::Q8_1);
    if(row.units()==0)return;
    auto maximum=vdupq_n_f32(-std::numeric_limits<float>::infinity());
    for(std::size_t offset=0;offset<row.liveColumns();offset+=4)
    {
        const auto count=std::min<std::size_t>(4,row.liveColumns()-offset);
        maximum=vmaxq_f32(maximum,scaledFloating(row,offset,count));
    }
    const float peak=vmaxvq_f32(maximum);
    if(std::isnan(peak)||peak==std::numeric_limits<float>::infinity())
        throw std::domain_error("CPU native softmax has a NaN or positive-infinite active score");
    if(peak==-std::numeric_limits<float>::infinity())
    {std::fill(row.data(),row.data()+row.units(),softmax_detail::Unit<Format>{0});return;}
    Summary summary;
    for(std::size_t offset=0;offset<row.liveColumns();offset+=4)
    {
        const auto count=std::min<std::size_t>(4,row.liveColumns()-offset);
        summary.add(exponential(vsubq_f32(scaledFloating(row,offset,count),vdupq_n_f32(peak))));
    }
    const float inverse=summary.inverse();
    for(std::size_t offset=0;offset<row.liveColumns();offset+=4)
    {
        const auto count=std::min<std::size_t>(4,row.liveColumns()-offset);
        const auto probability=vmulq_n_f32(exponential(vsubq_f32(scaledFloating(row,offset,count),vdupq_n_f32(peak))),inverse);
        storeFloating<Format>(row.data()+offset,probability,count);
    }
    std::fill(row.data()+row.liveColumns(),row.data()+row.units(),softmax_detail::Unit<Format>{0});
}

/** @brief Expand exactly four original signed bytes, including -128, without an eight-byte overread.
 * @param source Four original Q8 payload bytes.
 * @return Four exact FP32 integer coordinates.
 */
[[gnu::always_inline]] inline float32x4_t loadQ8Four(const std::int8_t *source) noexcept
{
    std::uint32_t packed;std::memcpy(&packed,source,sizeof(packed));
    const auto bytes=vreinterpret_s8_u32(vdup_n_u32(packed));
    return vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(bytes))));
}
using softmax_detail::q8Coefficient;
/** @brief Visit original active Q8 coordinates with one coefficient per original block.
 * @tparam Visitor Register-only consuming callback, inlined without allocation.
 * @param row Checked original Q8 row; @param visit Consumer of each increasing-coordinate fragment.
 */
template<class Visitor>
LLAMINAR_NEON_SOFTMAX_EXACT inline void activeQ8(const Row<Storage::Q8_1>&row,Visitor&&visit)
{
    LLAMINAR_NEON_SOFTMAX_ROUNDING
    const auto blocks=(row.liveColumns()+31)/32;
    for(std::size_t block=0;block<blocks;++block)
    {
        const auto &source=row.data()[block];const float coefficient=q8Coefficient(source,row.scale());
        const auto live=std::min<std::size_t>(32,row.liveColumns()-block*32);
        for(std::size_t offset=0;offset<live;offset+=4)
        {
            const auto count=std::min<std::size_t>(4,live-offset);
            const auto value=vmulq_n_f32(loadQ8Four(source.qs+offset),coefficient);
            visit(vbslq_f32(lanes(count),value,vdupq_n_f32(-std::numeric_limits<float>::infinity())));
        }
    }
}
/** @brief Stream a complete native Q8 row after its original maximum/summary reads finish.
 * @param row Checked original Q8 storage and causal coordinates.
 * @throws std::domain_error For corrupt active metadata, coefficient/score overflow or invalid summary.
 * @details Metadata is written last within each block, after every original
 * payload coordinate is consumed. RNE codes and exact integer sums retain the
 * native SIMD program; scalar diagnostic ties-away is a distinct oracle route.
 */
LLAMINAR_NEON_SOFTMAX_EXACT inline void execute(const Row<Storage::Q8_1>&row)
{
    LLAMINAR_NEON_SOFTMAX_ROUNDING
    if(row.units()==0)return;
    auto maximum=vdupq_n_f32(-std::numeric_limits<float>::infinity());
    activeQ8(row,[&](float32x4_t value){maximum=vmaxq_f32(maximum,value);});
    const float peak=vmaxvq_f32(maximum);
    if(std::isnan(peak)||peak==std::numeric_limits<float>::infinity())
        throw std::domain_error("CPU native Q8 softmax has a nonfinite active score");
    float inverse=1.0f;
    if(peak!=-std::numeric_limits<float>::infinity())
    {
        Summary summary;
        activeQ8(row,[&](float32x4_t value){summary.add(exponential(vsubq_f32(value,vdupq_n_f32(peak))));});
        inverse=summary.inverse();
    }
    const float new_d=softmax_detail::q8OutputScale(inverse);
    const float inverse_d=1.0f/new_d;
    const auto encoded=fp32_to_fp16(new_d);
    for(int block=0;block<row.units();++block)
    {
        auto &destination=row.data()[block];
        const auto start=std::size_t(block)*32;
        const auto live=start<row.liveColumns()?std::min<std::size_t>(32,row.liveColumns()-start):0;
        const float coefficient=live?q8Coefficient(destination,row.scale()):0.0f;
        auto sums=vdupq_n_s32(0);
        for(std::size_t offset=0;offset<32;offset+=4)
        {
            auto probability=vdupq_n_f32(0.0f);
            if(offset<live && peak!=-std::numeric_limits<float>::infinity())
            {
                const auto value=vmulq_n_f32(loadQ8Four(destination.qs+offset),coefficient);
                const auto count=std::min<std::size_t>(4,live-offset);
                const auto shifted=vsubq_f32(vbslq_f32(lanes(count),value,vdupq_n_f32(-std::numeric_limits<float>::infinity())),vdupq_n_f32(peak));
                probability=vmulq_n_f32(exponential(shifted),inverse);
            }
            auto quantized=vcvtnq_s32_f32(vmulq_n_f32(probability,inverse_d));
            quantized=vminq_s32(vdupq_n_s32(127),vmaxq_s32(vdupq_n_s32(0),quantized));
            sums=vaddq_s32(sums,quantized);
            const auto shorts=vqmovn_s32(quantized);
            const auto bytes=vqmovn_s16(vcombine_s16(shorts,vdup_n_s16(0)));
            const std::uint32_t packed=vget_lane_u32(vreinterpret_u32_s8(bytes),0);
            std::memcpy(destination.qs+offset,&packed,sizeof(packed));
        }
        destination.d=encoded;
        destination.sum_qs=static_cast<std::int16_t>(vaddvq_s32(sums));
    }
}
} // namespace llaminar2::primitives::neon_softmax
#undef LLAMINAR_NEON_SOFTMAX_EXACT
#undef LLAMINAR_NEON_SOFTMAX_ROUNDING
