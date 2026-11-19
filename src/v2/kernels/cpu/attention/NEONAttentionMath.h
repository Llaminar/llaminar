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
 * could strand a worker team. The row programs that carry no row geometry
 * checks are private members of AdmittedRows: only the owning flash-attention
 * kernel, whose public entries admit every bank before any team
 * (CPUAttentionInvocationAdmission), and the checked span overloads here can
 * call them. Row/weight metadata is passed by value, so it cannot alias an
 * output, and an integer reduction wider than its Q16ReductionBound is fatal.
 * The span overloads validate one call and then run the same programs; they
 * serve tests and cold callers. Spans are synchronous borrows, not capacity or
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

namespace llaminar2
{
enum class ActivationPrecision;
template <ActivationPrecision Precision> class CPUFlashAttentionKernelT;
} // namespace llaminar2

namespace llaminar2::cpu::neon_attention
{
/** @brief Reject corrupt consumed attention data inside a worker team without unwinding.
 * @param what Exact consumed quantity, preserved in the fatal diagnostic.
 */
[[noreturn]] inline void rejectAttentionData(const char*what)noexcept
{std::fprintf(stderr,"ERROR: CPU native attention consumed invalid %s\n",what);std::abort();} // One stdio call: no interleaving with other workers.
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
struct AdmittedRowsTestAccess;

/**
 * @brief Native row programs and their checked single-call overloads.
 * @details The bodies that trust row geometry are private. They are reachable
 * only from the owning CPUFlashAttentionKernelT, whose public entry points
 * admit every bank before any worker team, and from the checked overloads,
 * which validate one call first. AdmittedRowsTestAccess is defined only by
 * focused regressions of the private bodies' own fatal guards.
 */
class AdmittedRows final
{
public:
    /** @brief Share each original query load across independent original key borrows.
     * @tparam Format Original cache encoding; @tparam Rows Native register rows, one/two/four.
     * @param query Original FP32 query; @param keys Original complete key rows, possibly noncontiguous.
     * @return Original-order dot words after the same two-chain fold as native serial M1.
     * @throws std::invalid_argument For inconsistent or invalid original spans.
     */
    template<Storage Format,std::size_t Rows>
    static std::array<float,Rows> dot(std::span<const float>query,const std::array<std::span<const Unit<Format>>,Rows>&keys)
    {
        requireSpan(query);
        std::array<const Unit<Format>*,Rows> rows;
        for(std::size_t row=0;row<Rows;++row)
        {requireSpan(keys[row]);if(keys[row].size()!=query.size())throw std::invalid_argument("CPU native attention key/query geometry mismatch");rows[row]=keys[row].data();}
        return dotRows<Format,Rows>(query.data(),rows,int(query.size()));
    }
    /** @brief Retain the single-row native FMA program while sharing each output load/store across rows.
     * @tparam Format Original value encoding; @tparam Rows One/two/four original value rows.
     * @param output Original FP32 numerator; @param values Complete original disjoint cache borrows.
     * @param weights Original separately derived finite weights, in increasing key-row order.
     * @throws std::invalid_argument For malformed geometry or aliasing before writes.
     */
    template<Storage Format,std::size_t Rows>
    static void accumulate(std::span<float>output,const std::array<std::span<const Unit<Format>>,Rows>&values,const std::array<float,Rows>&weights)
    {
        requireSpan(output);
        if(overlaps(output.data(),output.size_bytes(),values.data(),sizeof(values))||
            overlaps(output.data(),output.size_bytes(),weights.data(),sizeof(weights)))
            throw std::invalid_argument("CPU native attention numerator overlaps original invocation metadata");
        std::array<const Unit<Format>*,Rows> rows;
        for(std::size_t row=0;row<Rows;++row)
        {
            requireSpan(values[row]);
            if(values[row].size()!=output.size()||
                overlaps(output.data(),output.size_bytes(),values[row].data(),values[row].size_bytes()))
                throw std::invalid_argument("CPU native attention requires disjoint complete value/numerator banks");
            rows[row]=values[row].data();
        }
        accumulateRows<Format,Rows>(output.data(),rows,weights,int(output.size()));
    }
    /** @brief Share original int16 query loads across rows with exact int32 accumulation.
     * @tparam Rows Native one/two/four-row register program.
     * @param bound Typed reduction proof whose limit quantized @p query.
     * @param query Original quantized query; @param keys Complete original native int16 key rows.
     * @return Exact original-row int32 dots, equal to the int64 sums.
     * @throws std::invalid_argument For malformed geometry or more products than the bound admits.
     */
    template<std::size_t Rows>
    static std::array<std::int32_t,Rows> integerDot(const Q16ReductionBound&bound,std::span<const std::int16_t>query,
        const std::array<std::span<const std::int16_t>,Rows>&keys)
    {
        requireSpan(query);
        std::array<const std::int16_t*,Rows> rows;
        for(std::size_t row=0;row<Rows;++row)
        {requireSpan(keys[row]);if(keys[row].size()!=query.size())throw std::invalid_argument("CPU native integer attention row geometry mismatch");rows[row]=keys[row].data();}
        if(query.size()>std::size_t(bound.terms()))
            throw std::invalid_argument("CPU native integer attention reduction exceeds its proven int32 term count");
        return integerDotRows<Rows>(bound,query.data(),rows,int(query.size()));
    }

private:
    template <ActivationPrecision Precision> friend class ::llaminar2::CPUFlashAttentionKernelT;
    friend struct AdmittedRowsTestAccess;

    /** @brief Shared native row program over rows an invocation admission already proved.
     * @tparam Format Original cache encoding; @tparam Rows Native register rows, one/two/four.
     * @param query First FP32 query coordinate; @param keys First coordinate of each key row.
     * @param width Admitted head width; every row holds at least @p width coordinates.
     * @return Original-order dot words after the same two-chain fold as native serial M1.
     * @details No row validation: CPUAttentionInvocationAdmission (or the checked
     * overload below) proved every row and the width before any worker team.
     */
    template<Storage Format,std::size_t Rows>
    static std::array<float,Rows> dotRows(const float*query,const std::array<const Unit<Format>*,Rows>keys,int width)noexcept
    {
    #if defined(__clang__)
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    #endif
        static_assert(Rows==1||Rows==2||Rows==4);
        const std::size_t count=std::size_t(width);
        float32x4_t lo[Rows],hi[Rows];
        for(std::size_t row=0;row<Rows;++row){lo[row]=vdupq_n_f32(0.0f);hi[row]=vdupq_n_f32(0.0f);}
        std::size_t coordinate=0;
        for(;coordinate+8<=count;coordinate+=8)
        {
            const auto q0=vld1q_f32(query+coordinate),q1=vld1q_f32(query+coordinate+4);
    #pragma GCC unroll 4
            for(std::size_t row=0;row<Rows;++row)
            {
                lo[row]=vfmaq_f32(lo[row],q0,load4<Format>(keys[row]+coordinate));
                hi[row]=vfmaq_f32(hi[row],q1,load4<Format>(keys[row]+coordinate+4));
            }
        }
        std::array<float,Rows> output;
        for(std::size_t row=0;row<Rows;++row)output[row]=vaddvq_f32(vaddq_f32(lo[row],hi[row]));
        for(;coordinate<count;++coordinate)
    #pragma GCC unroll 4
            for(std::size_t row=0;row<Rows;++row)
                output[row]=std::fma(query[coordinate],scalar<Format>(keys[row][coordinate]),output[row]);
        return output;
    }
    /** @brief Weighted value update over rows an invocation admission already proved.
     * @tparam Format Original value encoding; @tparam Rows One/two/four original value rows.
     * @param output First FP32 numerator coordinate, disjoint from every value row.
     * @param values First coordinate of each value row, by value.
     * @param weights Increasing key-row order, by value.
     * @param width Admitted head width.
     * @details Only the weights are tested: one vector test (w - w == +0 exactly
     * for finite w, NaN otherwise) before the first write; a nonfinite weight is
     * fatal, never an exception inside a team. The row addresses and weights are
     * the callee's own copies, so no numerator store can alter them.
     */
    template<Storage Format,std::size_t Rows>
    static void accumulateRows(float*output,const std::array<const Unit<Format>*,Rows>values,const std::array<float,Rows>weights,int width)noexcept
    {
    #if defined(__clang__)
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    #endif
        static_assert(Rows==1||Rows==2||Rows==4);
        {
            float lanes[4]={0.0f,0.0f,0.0f,0.0f};
            for(std::size_t row=0;row<Rows;++row)lanes[row]=weights[row];
            const auto w=vld1q_f32(lanes);
            if(vminvq_u32(vceqq_f32(vsubq_f32(w,w),vdupq_n_f32(0.0f)))==0)
                rejectAttentionData("nonfinite attention weight");
        }
        const std::size_t count=std::size_t(width);
        std::size_t coordinate=0;
        for(;coordinate+4<=count;coordinate+=4)
        {
            auto value=vld1q_f32(output+coordinate);
    #pragma GCC unroll 4
            for(std::size_t row=0;row<Rows;++row)value=vfmaq_n_f32(value,load4<Format>(values[row]+coordinate),weights[row]);
            vst1q_f32(output+coordinate,value);
        }
        for(;coordinate<count;++coordinate)
    #pragma GCC unroll 4
            for(std::size_t row=0;row<Rows;++row)output[coordinate]=std::fma(scalar<Format>(values[row][coordinate]),weights[row],output[coordinate]);
    }
    /** @brief Exact int32 integer dots over rows an invocation admission already proved.
     * @tparam Rows Native one/two/four-row register program.
     * @param bound Typed reduction proof whose limit quantized @p query.
     * @param query First query code; @param keys First code of each key row.
     * @param width Reduction width; fatal unless in [0, bound.terms()].
     * @return Exact original-row int32 dots, equal to the int64 sums.
     * @details Under the bound, sum(|q*k|) <= bound.worstMagnitude() <= INT32_MAX,
     * so every lane partial and the fold are exact. The width is part of that
     * proof, so it is checked here rather than trusted (one compare per call).
     * The query magnitude is folded into the loop (one saturating abs/max per
     * eight codes, shared by all rows); a code beyond the bound is fatal, since
     * the sum could have wrapped.
     */
    template<std::size_t Rows>
    static std::array<std::int32_t,Rows> integerDotRows(const Q16ReductionBound&bound,const std::int16_t*query,
        const std::array<const std::int16_t*,Rows>keys,int width)noexcept
    {
        static_assert(Rows==1||Rows==2||Rows==4);
        if(width<0||width>bound.terms())rejectAttentionData("Q16 reduction width outside its int32 term bound");
        const std::size_t count=std::size_t(width);
        int32x4_t sums[Rows];for(auto&sum:sums)sum=vdupq_n_s32(0);
        int16x8_t peak=vdupq_n_s16(0);
        std::size_t coordinate=0;
        for(;coordinate+8<=count;coordinate+=8)
        {
            const auto q=vld1q_s16(query+coordinate);
            peak=vmaxq_s16(peak,vqabsq_s16(q));
    #pragma GCC unroll 4
            for(std::size_t row=0;row<Rows;++row)
            {
                const auto key=vld1q_s16(keys[row]+coordinate);
                sums[row]=vmlal_s16(sums[row],vget_low_s16(q),vget_low_s16(key));
                sums[row]=vmlal_high_s16(sums[row],q,key);
            }
        }
        // Every query code is admitted before any scalar arithmetic that relies on
        // the bound: the fold and the tail run only on proven input. Vector lanes
        // above are modular SIMD adds, so invalid input cannot have caused UB yet.
        int magnitude=vmaxvq_s16(peak);
        for(std::size_t tail=coordinate;tail<count;++tail)
            magnitude=std::max(magnitude,std::abs(int(query[tail]))); // Widened: -32768 stays out of bound.
        if(magnitude>bound.queryLimit())rejectAttentionData("Q16 query code outside its int32 reduction bound");
        std::array<std::int32_t,Rows> result;
        for(std::size_t row=0;row<Rows;++row)result[row]=vaddvq_s32(sums[row]);
        for(;coordinate<count;++coordinate)
            for(std::size_t row=0;row<Rows;++row)result[row]+=std::int32_t(query[coordinate])*keys[row][coordinate];
        return result;
    }
};

/** @brief Checked single-call floating dot; see AdmittedRows::dot. */
template<Storage Format,std::size_t Rows>
inline std::array<float,Rows> dot(std::span<const float>query,const std::array<std::span<const Unit<Format>>,Rows>&keys)
{ return AdmittedRows::dot<Format,Rows>(query,keys); }
/** @brief Checked single-call weighted update; see AdmittedRows::accumulate. */
template<Storage Format,std::size_t Rows>
inline void accumulate(std::span<float>output,const std::array<std::span<const Unit<Format>>,Rows>&values,const std::array<float,Rows>&weights)
{ AdmittedRows::accumulate<Format,Rows>(output,values,weights); }
/** @brief Checked single-call exact integer dot; see AdmittedRows::integerDot. */
template<std::size_t Rows>
inline std::array<std::int32_t,Rows> integerDot(const Q16ReductionBound&bound,std::span<const std::int16_t>query,
    const std::array<std::span<const std::int16_t>,Rows>&keys)
{ return AdmittedRows::integerDot<Rows>(bound,query,keys); }
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
