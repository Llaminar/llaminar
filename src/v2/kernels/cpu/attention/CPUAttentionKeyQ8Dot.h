/**
 * @file CPUAttentionKeyQ8Dot.h
 * @brief Checked original-query AQ8 dot tiles without reconstructed head storage.
 *
 * The immutable cache block and request anchor remain the decode authorities.
 * One admitted native program shares register-only decoded fragments across up
 * to four original query rows, retaining each ISA's existing FP32 attention-dot
 * chains/fold. The separate multiply/multiply/add decode edges survive inlining;
 * dot FMAs remain intentional. These synchronous borrows create no allocation,
 * cache mirror, capacity claim or extension of the enclosing query/cache lease.
 */
#pragma once
#include "CPUAttentionKeyQ8.h"
#include <array>
#if LLAMINAR_COMPILED_WITH_NEON
#include "tensors/NEONFP32.h"
#elif defined(__AVX2__) || defined(__AVX512F__)
#include "tensors/X86FP32.h"
#endif

namespace llaminar2::cpu::attention_key_q8
{
namespace detail
{
#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Stream the exact native decode through two four-lane dot chains per query.
 * @tparam D Admitted complete head width; @tparam Rows Shared-query tile width.
 * @param queries Original query addresses; @param block Immutable compressed key.
 * @param anchor Immutable request anchor; @return Original-order attention dot words.
 */
template<int D,std::size_t Rows>
inline std::array<float,Rows> dotQueriesNEON(std::array<const float*,Rows>queries,
                                           const AttentionKeyQ8Block<D>&block,const float*anchor)
{
    std::array<float32x4_t,Rows> low{},high{};
    const auto scale=vdupq_n_f32(block.quadratic_scale);
    for(int coordinate=0;coordinate<D;coordinate+=8)
    {
        const auto shorts=vmovl_s8(vld1_s8(block.codes+coordinate));
        const auto code0=vcvtq_f32_s32(vmovl_s16(vget_low_s16(shorts)));
        const auto code1=vcvtq_f32_s32(vmovl_s16(vget_high_s16(shorts)));
        const auto residual0=neon_fp32::multiply(neon_fp32::multiply(code0,scale),vabsq_f32(code0));
        const auto residual1=neon_fp32::multiply(neon_fp32::multiply(code1,scale),vabsq_f32(code1));
        const auto key0=neon_fp32::add(residual0,vld1q_f32(anchor+coordinate));
        const auto key1=neon_fp32::add(residual1,vld1q_f32(anchor+coordinate+4));
        for(std::size_t row=0;row<Rows;++row)
        {
            low[row]=vfmaq_f32(low[row],vld1q_f32(queries[row]+coordinate),key0);
            high[row]=vfmaq_f32(high[row],vld1q_f32(queries[row]+coordinate+4),key1);
        }
    }
    std::array<float,Rows> result;
    for(std::size_t row=0;row<Rows;++row)result[row]=vaddvq_f32(vaddq_f32(low[row],high[row]));
    return result;
}
#endif
#if defined(__AVX2__)
/** @brief Stream original AQ8 fragments through the established single-chain AVX2 dot.
 * @tparam D Complete head width; @tparam Rows Number of independent original queries.
 * @param queries Checked original queries; @param block Immutable key.
 * @param anchor Original immutable anchor; @return Existing AVX2 dot words.
 */
template<int D,std::size_t Rows>
inline std::array<float,Rows> dotQueriesAVX2(std::array<const float*,Rows>queries,
                                           const AttentionKeyQ8Block<D>&block,const float*anchor)
{
    __m256 sums[Rows]{};
    const auto scale=_mm256_set1_ps(block.quadratic_scale);
    const auto mask=_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for(int coordinate=0;coordinate<D;coordinate+=8)
    {
        const auto codes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.codes+coordinate));
        const auto code=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes));
        const auto scaled=x86_fp32::persistRounded(_mm256_mul_ps(code,scale));
        const auto residual=x86_fp32::persistRounded(_mm256_mul_ps(scaled,_mm256_and_ps(code,mask)));
        const auto key=_mm256_add_ps(residual,_mm256_loadu_ps(anchor+coordinate));
        for(std::size_t row=0;row<Rows;++row)
            sums[row]=_mm256_fmadd_ps(_mm256_loadu_ps(queries[row]+coordinate),key,sums[row]);
    }
    std::array<float,Rows> result;
    for(std::size_t row=0;row<Rows;++row)
    {
        auto folded=_mm_add_ps(_mm256_castps256_ps128(sums[row]),_mm256_extractf128_ps(sums[row],1));
        folded=_mm_add_ps(folded,_mm_movehdup_ps(folded));
        result[row]=_mm_cvtss_f32(_mm_add_ss(folded,_mm_movehl_ps(folded,folded)));
    }
    return result;
}
#endif
#if defined(__AVX512F__)
/** @brief Stream original AQ8 fragments through the established sixteen-lane AVX512 dot.
 * @tparam D Complete head width; @tparam Rows Independent original query count.
 * @param queries Checked original queries; @param block Immutable compressed key.
 * @param anchor Original immutable anchor; @return Existing AVX512 reduction words.
 */
template<int D,std::size_t Rows>
inline std::array<float,Rows> dotQueriesAVX512(std::array<const float*,Rows>queries,
                                             const AttentionKeyQ8Block<D>&block,const float*anchor)
{
    __m512 sums[Rows]{};
    const auto scale=_mm512_set1_ps(block.quadratic_scale);
    const auto mask=_mm512_set1_epi32(0x7fffffff);
    for(int coordinate=0;coordinate<D;coordinate+=16)
    {
        const auto codes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes+coordinate));
        const auto code=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(codes));
        const auto magnitude=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(code),mask));
        const auto scaled=x86_fp32::persistRounded(_mm512_mul_ps(code,scale));
        const auto residual=x86_fp32::persistRounded(_mm512_mul_ps(scaled,magnitude));
        const auto key=_mm512_add_ps(residual,_mm512_loadu_ps(anchor+coordinate));
        for(std::size_t row=0;row<Rows;++row)
            sums[row]=_mm512_fmadd_ps(_mm512_loadu_ps(queries[row]+coordinate),key,sums[row]);
    }
    std::array<float,Rows> result;
    for(std::size_t row=0;row<Rows;++row)result[row]=_mm512_reduce_add_ps(sums[row]);
    return result;
}
#endif
} // namespace detail

/** @brief Admitted immutable original-query/anchor/ISA tile, valid only within caller-owned leases.
 * @tparam D Complete cache head64/128/256; @tparam Rows Physical query count1..4.
 * @details Input publication owns finite scale/code admission. This consumer
 * does not rescan cache metadata, store coefficients or reconstruct a key head.
 */
template<int D,std::size_t Rows>
class QueryDotTile final
{
    static_assert((D==64||D==128||D==256)&&Rows>=1&&Rows<=4);
public:
    /** @brief Bind complete original query heads and request basis before the K loop.
     * @param queries Original query spans, each exactly D values.
     * @param anchor Immutable original D-value request anchor.
     * @param requested Explicit native ISA or authenticated automatic policy.
     * @return A checked non-owning program; no allocation or lease is created.
     * @throws std::invalid_argument For malformed spans/address domains or unavailable ISA.
     * @throws std::runtime_error For a compiler/runtime instruction-envelope mismatch.
     */
    static QueryDotTile bind(const std::array<std::span<const float>,Rows>&queries,
                             std::span<const float>anchor,ISA requested=ISA::Automatic)
    {
        const auto valid=[](std::span<const float>head){
            return head.size()==D&&head.data()&&reinterpret_cast<std::uintptr_t>(head.data())<=
                std::numeric_limits<std::uintptr_t>::max()-D*sizeof(float);
        };
        if(!valid(anchor))throw std::invalid_argument("AQ8 dot requires a complete original anchor head");
        std::array<const float*,Rows> roots;
        for(std::size_t row=0;row<Rows;++row)
        {
            if(!valid(queries[row]))throw std::invalid_argument("AQ8 dot requires complete original query heads");
            roots[row]=queries[row].data();
        }
        return QueryDotTile(roots,anchor.data(),resolveISA(requested));
    }
    /** @return Authenticated native program, not a guessed compiler capability. */
    [[nodiscard]] ISA isa()const noexcept{return isa_;}
    /** @brief Consume one original published compressed key without a decoded head buffer.
     * @param block Original cache-owned key, whose publication validated metadata/codes.
     * @return Each original query's M=1 attention-dot word in independent fixed chains.
     * @throws std::logic_error If a bound ISA lacks a compiled body (fatal invariant).
     */
    [[nodiscard]] std::array<float,Rows> dot(const AttentionKeyQ8Block<D>&block)const
    {
        switch(isa_)
        {
#if LLAMINAR_COMPILED_WITH_NEON
        case ISA::NEON:return detail::dotQueriesNEON<D,Rows>(queries_,block,anchor_);
#endif
#if defined(__AVX2__)
        case ISA::AVX2:return detail::dotQueriesAVX2<D,Rows>(queries_,block,anchor_);
#endif
#if defined(__AVX512F__)
        case ISA::AVX512:return detail::dotQueriesAVX512<D,Rows>(queries_,block,anchor_);
#endif
        default:throw std::logic_error("AQ8 dot bound ISA has no compiled native program");
        }
    }
private:
    /** @brief Retain validated original roots and exact native policy until synchronous completion.
     * @param queries Complete original query roots; @param anchor Original immutable basis.
     * @param isa Authenticated native instruction program.
     */
    QueryDotTile(std::array<const float*,Rows>queries,const float*anchor,ISA isa)
        :queries_(queries),anchor_(anchor),isa_(isa){}
    // Immutable roots permit no post-bind rebinding. Passing only these roots
    // by value gives native leaves SSA addresses instead of repeated hot stack
    // reloads of a borrowed descriptor; no query/key/coefficient data is copied.
    const std::array<const float*,Rows> queries_;
    const float*const anchor_;
    const ISA isa_;
};
} // namespace llaminar2::cpu::attention_key_q8
