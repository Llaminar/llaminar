/**
 * @file CPUAttentionFP32DotReference.h
 * @brief Existing native FP32 attention-dot programs for AQ8 test/performance authentication.
 *
 * NEON calls its production FP32 dot helper. The x86 references transcribe the
 * private class loops, including the actual compiler intrinsic reduction tree;
 * they do not execute that private class helper. Whole decoded heads are legal
 * diagnostic oracles here, never a production cache mirror or serial fallback.
 */
#pragma once
#include "kernels/cpu/attention/CPUAttentionKeyQ8.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#if LLAMINAR_COMPILED_WITH_NEON
#include "kernels/cpu/attention/NEONAttentionMath.h"
#endif

namespace llaminar2::test
{
/** @brief Poison the entire cold dot-publication bank before correctness authentication.
 * @param output Complete fixture-owned bank; never called inside a timing interval.
 */
inline void poisonAttentionDotOutput(std::span<float>output)
{
    std::fill(output.begin(),output.end(),std::numeric_limits<float>::quiet_NaN());
}

/** @brief Accept only nonempty complete finite banks with identical original FP32 words.
 * @param actual Complete optimized publication; @param expected Complete diagnostic reference.
 * @return False for no-op/partial writers left poisoned, nonfinite data or unequal geometry/words.
 */
inline bool authenticateAttentionDotOutput(std::span<const float>actual,std::span<const float>expected)noexcept
{
    if(actual.empty()||actual.size()!=expected.size())return false;
    for(std::size_t i=0;i<actual.size();++i)
        if(!std::isfinite(actual[i])||!std::isfinite(expected[i])||
           std::bit_cast<std::uint32_t>(actual[i])!=std::bit_cast<std::uint32_t>(expected[i]))return false;
    return true;
}

/** @brief Evaluate the original ISA-specific attention dot after independent native key decoding.
 * @tparam D Admitted complete head dimension, divisible by16.
 * @param query Original FP32 query; @param key Diagnostic reconstructed original key.
 * @param isa Explicit native instruction program; @return Existing FP32 attention-dot word.
 * @throws std::invalid_argument For incomplete diagnostic geometry.
 * @throws std::logic_error For an unavailable declared reference, never a substitute.
 */
template<int D>
float cpuAttentionFP32DotReference(std::span<const float>query,std::span<const float>key,
                                  cpu::attention_key_q8::ISA isa)
{
    using cpu::attention_key_q8::ISA;
    static_assert(D==64||D==128||D==256);
    if(query.size()!=D||key.size()!=D)throw std::invalid_argument("Attention dot reference needs complete original heads");
#if LLAMINAR_COMPILED_WITH_NEON
    if(isa==ISA::NEON)
        return cpu::neon_attention::dot<cpu::neon_attention::Storage::FP32,1>(query,{key})[0];
#endif
#if defined(__AVX2__)
    if(isa==ISA::AVX2)
    {
        auto sum=_mm256_setzero_ps();
        for(int c=0;c<D;c+=8)sum=_mm256_fmadd_ps(_mm256_loadu_ps(query.data()+c),_mm256_loadu_ps(key.data()+c),sum);
        auto fold=_mm_add_ps(_mm256_castps256_ps128(sum),_mm256_extractf128_ps(sum,1));
        fold=_mm_add_ps(fold,_mm_movehdup_ps(fold));
        return _mm_cvtss_f32(_mm_add_ss(fold,_mm_movehl_ps(fold,fold)));
    }
#endif
#if defined(__AVX512F__)
    if(isa==ISA::AVX512)
    {
        auto sum=_mm512_setzero_ps();
        for(int c=0;c<D;c+=16)sum=_mm512_fmadd_ps(_mm512_loadu_ps(query.data()+c),_mm512_loadu_ps(key.data()+c),sum);
        return _mm512_reduce_add_ps(sum);
    }
#endif
    throw std::logic_error("Attention dot reference requires its declared native ISA");
}
} // namespace llaminar2::test
