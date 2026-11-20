/**
 * @file CPUQ8AttentionValue.h
 * @brief Native original-Q8 value accumulation with a separately rounded weight/scale product.
 *
 * Published cache bytes and their FP16 scale remain authoritative. Native CPU
 * leaves widen original signed codes in registers and intentionally FMA them
 * with the original numerator; no decoded value bank is constructed. The exact
 * stream of calls determines increasing-K order. This synchronous leaf neither
 * proves allocated capacity nor extends the enclosing cache/workspace lease.
 */
#pragma once
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "utils/CPUFeatures.h"
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#if LLAMINAR_COMPILED_WITH_NEON
#include "tensors/NEONFP32.h"
#elif defined(__AVX2__) || defined(__AVX512F__)
#include "tensors/X86FP32.h"
#endif

namespace llaminar2::cpu::q8_attention
{
/** @brief Accumulate one original32-code Q8 block into its disjoint numerator fragment.
 * @param output Exactly32 writable FP32 values within the caller-owned numerator.
 * @param block Original published Q8 block; import/publication owns metadata validation.
 * @param weight Original softmax weight, before multiplication by the physical FP16 scale.
 * @details The canonical active-ISA authority selects the native program;
 * scalar execution requires that authority's explicit diagnostic intent.
 * @throws std::invalid_argument For malformed geometry/address domains or a source intersection.
 * @throws std::runtime_error For a foreign or unavailable instruction program.
 * @details Native programs round weight*scale to FP32 before the intentional
 * code*combined+numerator FMA. The scalar diagnostic retains its original source
 * expression; it is never entered to replace a missing native implementation.
 */
inline void accumulateValueBlock(std::span<float>output,const Q8_1Block&block,float weight)
{
    constexpr std::size_t bytes=32*sizeof(float);
    const auto address=reinterpret_cast<std::uintptr_t>(output.data());
    if(output.size()!=32||!output.data()||address>std::numeric_limits<std::uintptr_t>::max()-bytes)
        throw std::invalid_argument("Q8 attention numerator requires a complete32-value fragment");
    const auto source=reinterpret_cast<std::uintptr_t>(&block);
    const bool intersection=address<=source?source-address<bytes:address-source<sizeof(block);
    if(intersection)throw std::invalid_argument("Q8 attention numerator aliases its original compressed value");
    const float scale=fp16_to_fp32(block.d);
    switch(activeISALevel())
    {
#if LLAMINAR_COMPILED_WITH_NEON
    case ISALevel::NEON:
    case ISALevel::NEONDotProd:
    {
        const auto combined=neon_fp32::multiply(vdupq_n_f32(weight),vdupq_n_f32(scale));
        for(int c=0;c<32;c+=8)
        {
            const auto shorts=vmovl_s8(vld1_s8(block.qs+c));
            const auto low=vcvtq_f32_s32(vmovl_s16(vget_low_s16(shorts)));
            const auto high=vcvtq_f32_s32(vmovl_s16(vget_high_s16(shorts)));
            vst1q_f32(output.data()+c,vfmaq_f32(vld1q_f32(output.data()+c),low,combined));
            vst1q_f32(output.data()+c+4,vfmaq_f32(vld1q_f32(output.data()+c+4),high,combined));
        }
        return;
    }
#endif
#if defined(__AVX2__)
    case ISALevel::AVX2:
    {
        const auto combined=x86_fp32::persistRounded(_mm256_mul_ps(_mm256_set1_ps(weight),_mm256_set1_ps(scale)));
        for(int c=0;c<32;c+=8)
        {
            const auto codes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.qs+c));
            const auto values=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes));
            _mm256_storeu_ps(output.data()+c,_mm256_fmadd_ps(values,combined,_mm256_loadu_ps(output.data()+c)));
        }
        return;
    }
#endif
#if defined(__AVX512F__)
    case ISALevel::AVX512:
    {
        const auto combined=x86_fp32::persistRounded(_mm512_mul_ps(_mm512_set1_ps(weight),_mm512_set1_ps(scale)));
        for(int c=0;c<32;c+=16)
        {
            const auto codes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(block.qs+c));
            const auto values=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(codes));
            _mm512_storeu_ps(output.data()+c,_mm512_fmadd_ps(values,combined,_mm512_loadu_ps(output.data()+c)));
        }
        return;
    }
#endif
    case ISALevel::Scalar:
    {
        const float combined=weight*scale;
        for(int c=0;c<32;++c)output[c]+=combined*float(block.qs[c]);
        return;
    }
    default:throw std::runtime_error("Q8 attention value ISA has no matching compiled native implementation");
    }
}
} // namespace llaminar2::cpu::q8_attention
