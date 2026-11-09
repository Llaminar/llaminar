/**
 * @file NEONAttentionMath.h
 * @brief Native original-cache floating and integer attention arithmetic.
 *
 * Fixed one/two/four-row register programs share original query loads without
 * packing keys or mirroring decoded heads. Floating dots retain two four-lane
 * FMA chains and one fixed fold; weighted updates retain original increasing-row
 * FMA order. Integer dots accumulate in int32 under a typed Q16ReductionBound
 * whose closed form makes every partial sum exact. Nonfinite weights and query
 * codes outside the bound are fatal at the point of use, never exceptions that
 * could strand a worker team. Spans are synchronous borrows, not capacity or
 * cache-lease proofs.
 */
#pragma once
#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "Native attention arithmetic requires AArch64 ASIMD"
#endif
#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/NEONFP16.h"
#include "CPUQ16AttentionMath.h"
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace llaminar2::cpu::neon_attention
{
/** @brief Reject corrupt consumed attention data inside a worker team without unwinding.
 * @param what Exact consumed quantity, preserved in the fatal diagnostic.
 */
[[noreturn]] inline void rejectAttentionData(const char*what)noexcept
{std::fputs("ERROR: CPU native attention consumed invalid ",stderr);std::fputs(what,stderr);std::fputc('\n',stderr);std::abort();}
/** @brief Original cache scalar encoding; no decoded shadow storage is implied. */
enum class Storage : std::uint8_t { FP32,FP16,BF16,I16 };
/** @brief Exact original physical scalar type for each cache role. */
template<Storage Format>struct Scalar;
template<>struct Scalar<Storage::FP32>{using Type=float;};
template<>struct Scalar<Storage::FP16>{using Type=std::uint16_t;};
template<>struct Scalar<Storage::BF16>{using Type=std::uint16_t;};
template<>struct Scalar<Storage::I16>{using Type=std::int16_t;};
template<Storage Format>using Unit=typename Scalar<Format>::Type;

/** @brief Require a representable original synchronous span before arithmetic.
 * @tparam T Original physical scalar; @param bank Original declared borrow.
 * @throws std::invalid_argument For missing storage or a wrapping/oversized declared extent.
 */
template<class T>inline void requireSpan(std::span<T>bank)
{
    if(bank.size()>std::size_t(std::numeric_limits<int>::max())||
        (bank.size()&&(!bank.data()||reinterpret_cast<std::uintptr_t>(bank.data())>
            std::numeric_limits<std::uintptr_t>::max()-bank.size_bytes())))
        throw std::invalid_argument("CPU native attention requires a complete representable original span");
}
/** @brief Adapt original independently addressed tensor/cache rows without packing them.
 * @tparam T Original physical scalar; @tparam Rows Number of original borrows.
 * @param heads Original independently owned row addresses; @param count Declared original scalar width.
 * @return Typed synchronous spans; actual capacity/lease remain the enclosing owner's responsibility.
 * @throws std::invalid_argument For invalid width, missing storage or a wrapping declared span.
 */
template<class T,std::size_t Rows>
inline std::array<std::span<const T>,Rows> originalRows(const std::array<const T*,Rows>&heads,int count)
{
    if(count<0)throw std::invalid_argument("CPU native attention requires nonnegative original row widths");
    std::array<std::span<const T>,Rows> rows;
    for(std::size_t row=0;row<Rows;++row)
    {
        if(count&&!heads[row])throw std::invalid_argument("CPU native attention original row is missing");
        rows[row]={heads[row],std::size_t(count)};requireSpan(rows[row]);
    }
    return rows;
}
/** @brief Detect overlapping original physical bytes without overflowing end addresses.
 * @param first First bank; @param first_bytes First declared extent.
 * @param second Second bank; @param second_bytes Second declared extent.
 * @return Whether the nonempty byte envelopes overlap.
 */
inline bool overlaps(const void*first,std::size_t first_bytes,const void*second,std::size_t second_bytes)noexcept
{
    if(!first_bytes||!second_bytes)return false;
    const auto a=reinterpret_cast<std::uintptr_t>(first),b=reinterpret_cast<std::uintptr_t>(second);
    return a<=b?b-a<first_bytes:a-b<second_bytes;
}
/** @brief Expand four original cache coordinates, without a row-sized conversion bank.
 * @tparam Format Original physical encoding; @param original Four original coordinates.
 * @return Four FP32 arithmetic lanes.
 */
template<Storage Format>[[gnu::always_inline]]inline float32x4_t load4(const Unit<Format>*original)noexcept
{
    if constexpr(Format==Storage::FP32)return vld1q_f32(original);
    else if constexpr(Format==Storage::FP16)return neon_fp16::widen(vld1_u16(original));
    else if constexpr(Format==Storage::BF16)return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16(original)),16));
    else return vcvtq_f32_s32(vmovl_s16(vld1_s16(original)));
}
/** @brief Decode one bounded original tail coordinate using its canonical storage ABI.
 * @tparam Format Original physical encoding; @param original Original scalar word.
 * @return Original FP32 arithmetic value.
 */
template<Storage Format>inline float scalar(Unit<Format>original)noexcept
{
    if constexpr(Format==Storage::FP32||Format==Storage::I16)return float(original);
    else if constexpr(Format==Storage::FP16)return fp16_to_fp32(original);
    else
    {
        const auto bits=std::uint32_t(original)<<16;
        return std::bit_cast<float>(bits);
    }
}
/** @brief Share each original query load across independent original key borrows.
 * @tparam Format Original cache encoding; @tparam Rows Native register rows, one/two/four.
 * @param query Original FP32 query; @param keys Original complete key rows, possibly noncontiguous.
 * @return Original-order dot words after the same two-chain fold as native serial M1.
 * @throws std::invalid_argument For inconsistent or invalid original spans.
 */
template<Storage Format,std::size_t Rows>
inline std::array<float,Rows> dot(std::span<const float>query,const std::array<std::span<const Unit<Format>>,Rows>&keys)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    static_assert(Rows==1||Rows==2||Rows==4);
    requireSpan(query);
    for(const auto key:keys){requireSpan(key);if(key.size()!=query.size())throw std::invalid_argument("CPU native attention key/query geometry mismatch");}
    float32x4_t lo[Rows],hi[Rows];
    for(std::size_t row=0;row<Rows;++row){lo[row]=vdupq_n_f32(0.0f);hi[row]=vdupq_n_f32(0.0f);}
    std::size_t coordinate=0;
    for(;coordinate+8<=query.size();coordinate+=8)
    {
        const auto q0=vld1q_f32(query.data()+coordinate),q1=vld1q_f32(query.data()+coordinate+4);
#pragma GCC unroll 4
        for(std::size_t row=0;row<Rows;++row)
        {
            lo[row]=vfmaq_f32(lo[row],q0,load4<Format>(keys[row].data()+coordinate));
            hi[row]=vfmaq_f32(hi[row],q1,load4<Format>(keys[row].data()+coordinate+4));
        }
    }
    std::array<float,Rows> output;
    for(std::size_t row=0;row<Rows;++row)output[row]=vaddvq_f32(vaddq_f32(lo[row],hi[row]));
    for(;coordinate<query.size();++coordinate)
#pragma GCC unroll 4
        for(std::size_t row=0;row<Rows;++row)
            output[row]=std::fma(query[coordinate],scalar<Format>(keys[row][coordinate]),output[row]);
    return output;
}
/** @brief Retain the single-row native FMA program while sharing each output load/store across rows.
 * @tparam Format Original value encoding; @tparam Rows One/two/four original value rows.
 * @param output Original FP32 numerator; @param values Complete original disjoint cache borrows.
 * @param weights Original separately derived finite weights, in increasing key-row order.
 * @throws std::invalid_argument For malformed geometry, aliasing or nonfinite weights before writes.
 */
template<Storage Format,std::size_t Rows>
inline void accumulate(std::span<float>output,const std::array<std::span<const Unit<Format>>,Rows>&values,const std::array<float,Rows>&weights)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    static_assert(Rows==1||Rows==2||Rows==4);
    requireSpan(output);
    if(overlaps(output.data(),output.size_bytes(),values.data(),sizeof(values))||
        overlaps(output.data(),output.size_bytes(),weights.data(),sizeof(weights)))
        throw std::invalid_argument("CPU native attention numerator overlaps original invocation metadata");
    for(std::size_t row=0;row<Rows;++row)
    {
        requireSpan(values[row]);
        if(values[row].size()!=output.size()||
            overlaps(output.data(),output.size_bytes(),values[row].data(),values[row].size_bytes()))
            throw std::invalid_argument("CPU native attention requires disjoint complete value/numerator banks");
    }
    {
        // w-w is +0 exactly for finite w and NaN for NaN/Inf; one vector test
        // covers every row before the first numerator write.
        float lanes[4]={0.0f,0.0f,0.0f,0.0f};
        for(std::size_t row=0;row<Rows;++row)lanes[row]=weights[row];
        const auto w=vld1q_f32(lanes);
        if(vminvq_u32(vceqq_f32(vsubq_f32(w,w),vdupq_n_f32(0.0f)))==0)
            rejectAttentionData("nonfinite attention weight");
    }
    std::size_t coordinate=0;
    for(;coordinate+4<=output.size();coordinate+=4)
    {
        auto value=vld1q_f32(output.data()+coordinate);
#pragma GCC unroll 4
        for(std::size_t row=0;row<Rows;++row)value=vfmaq_n_f32(value,load4<Format>(values[row].data()+coordinate),weights[row]);
        vst1q_f32(output.data()+coordinate,value);
    }
    for(;coordinate<output.size();++coordinate)
#pragma GCC unroll 4
        for(std::size_t row=0;row<Rows;++row)output[coordinate]=std::fma(scalar<Format>(values[row][coordinate]),weights[row],output[coordinate]);
}
/** @brief Share original int16 query loads across rows with exact int32 accumulation.
 * @tparam Rows Native one/two/four-row register program.
 * @param bound Typed reduction proof whose limit quantized @p query.
 * @param query Original quantized query; @param keys Complete original native int16 key rows.
 * @return Exact original-row int32 dots, equal to the int64 sums.
 * @throws std::invalid_argument For malformed geometry or more products than the bound admits.
 * @details Under the bound, sum(|q*k|) <= bound.worstMagnitude() <= INT32_MAX,
 * so every int32 lane partial and the fold are exact. The query magnitude is
 * folded into the loop (one saturating abs/max per eight codes, shared by all
 * rows); a code beyond the bound is fatal, since the sum could have wrapped.
 */
template<std::size_t Rows>
inline std::array<std::int32_t,Rows> integerDot(const Q16ReductionBound&bound,std::span<const std::int16_t>query,
    const std::array<std::span<const std::int16_t>,Rows>&keys)
{
    static_assert(Rows==1||Rows==2||Rows==4);
    requireSpan(query);
    for(auto key:keys){requireSpan(key);if(key.size()!=query.size())throw std::invalid_argument("CPU native integer attention row geometry mismatch");}
    if(query.size()>std::size_t(bound.terms()))
        throw std::invalid_argument("CPU native integer attention reduction exceeds its proven int32 term count");
    int32x4_t sums[Rows];for(auto&sum:sums)sum=vdupq_n_s32(0);
    int16x8_t peak=vdupq_n_s16(0);
    std::size_t coordinate=0;
    for(;coordinate+8<=query.size();coordinate+=8)
    {
        const auto q=vld1q_s16(query.data()+coordinate);
        peak=vmaxq_s16(peak,vqabsq_s16(q));
#pragma GCC unroll 4
        for(std::size_t row=0;row<Rows;++row)
        {
            const auto key=vld1q_s16(keys[row].data()+coordinate);
            sums[row]=vmlal_s16(sums[row],vget_low_s16(q),vget_low_s16(key));
            sums[row]=vmlal_high_s16(sums[row],q,key);
        }
    }
    std::array<std::int32_t,Rows> result;
    for(std::size_t row=0;row<Rows;++row)result[row]=vaddvq_s32(sums[row]);
    int magnitude=vmaxvq_s16(peak);
    for(;coordinate<query.size();++coordinate)
    {
        // -32768 has no positive int16 magnitude; widening first keeps it out of bound.
        magnitude=std::max(magnitude,std::abs(int(query[coordinate])));
        for(std::size_t row=0;row<Rows;++row)result[row]+=std::int32_t(query[coordinate])*keys[row][coordinate];
    }
    if(magnitude>bound.queryLimit())rejectAttentionData("Q16 query code outside its int32 reduction bound");
    return result;
}
/** @brief Quantize one original FP32 query with the existing max/qmax, inverse, RNE program.
 * @param source Original complete query; @param destination Original disjoint int16 workspace.
 * @param limit Positive symmetric payload limit, at most 32767.
 * @return Retained FP32 query scale; exact zero for the canonical <=1e-12 maximum domain.
 * @throws std::invalid_argument For geometry/alias/limit errors before writes.
 * @throws std::domain_error For a nonfinite maximum/scale before publication.
 */
inline float quantizeQuery(std::span<const float>source,std::span<std::int16_t>destination,int limit)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    requireSpan(source);requireSpan(destination);
    if(source.size()!=destination.size()||limit<=0||limit>32767||
        overlaps(source.data(),source.size_bytes(),destination.data(),destination.size_bytes()))
        throw std::invalid_argument("CPU native attention query quantization has malformed original operands");
    auto maximum=vdupq_n_f32(0.0f);std::size_t coordinate=0;
    for(;coordinate+4<=source.size();coordinate+=4)maximum=vmaxq_f32(maximum,vabsq_f32(vld1q_f32(source.data()+coordinate)));
    float peak=vmaxvq_f32(maximum);
    for(;coordinate<source.size();++coordinate)
    {if(!std::isfinite(source[coordinate]))throw std::domain_error("CPU native attention query is nonfinite");peak=std::max(peak,std::abs(source[coordinate]));}
    if(!std::isfinite(peak))throw std::domain_error("CPU native attention query maximum is nonfinite");
    if(peak<=1e-12f){std::fill(destination.begin(),destination.end(),std::int16_t(0));return 0.0f;}
    const float scale=peak/float(limit),inverse=1.0f/scale;
    if(!std::isfinite(scale)||!std::isfinite(inverse))throw std::domain_error("CPU native attention query scale is not finite");
    coordinate=0;
    for(;coordinate+4<=source.size();coordinate+=4)
    {
        auto value=vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(source.data()+coordinate),inverse));
        value=vminq_s32(vdupq_n_s32(limit),vmaxq_s32(vdupq_n_s32(-limit),value));
        vst1_s16(destination.data()+coordinate,vqmovn_s32(value));
    }
    for(;coordinate<source.size();++coordinate)
        destination[coordinate]=std::int16_t(std::clamp(int(std::lrint(source[coordinate]*inverse)),-limit,limit));
    return scale;
}
} // namespace llaminar2::cpu::neon_attention
