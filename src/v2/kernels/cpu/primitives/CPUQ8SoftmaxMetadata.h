/**
 * @file CPUQ8SoftmaxMetadata.h
 * @brief Original FP16 Q8 softmax metadata and sign-correct integer extrema.
 *
 * Q8_1Block has one FP16 scale ABI. A private BF16 interpretation is invalid.
 * Maximum selection uses the same separately rounded block coefficient as the
 * exponential pass, includes the legal -128 endpoint, and reverses the integer
 * extremum for negative coefficients. No coefficient bank or decoded row exists.
 */
#pragma once
#include "CPUSoftmaxOperands.h"
#include "../../../tensors/FP16Utils.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::primitives::softmax_detail
{
/** @brief Read one original FP16 coefficient, never a shadow source-scale bank.
 * @param block Original complete block; @param scale Finite invocation multiplier.
 * @return Separately rounded coefficient used by both maximum and exponential passes.
 * @throws std::domain_error For nonfinite original metadata or coefficient overflow.
 */
inline float q8Coefficient(const Q8_1Block&block,float scale)
{
    const float original=fp16_to_fp32(block.d);
    const float coefficient=original*scale;
    if(!std::isfinite(original)||!std::isfinite(coefficient))
        throw std::domain_error("CPU Q8 softmax requires finite FP16 metadata and score coefficient");
    return coefficient;
}
/** @brief Select the first-class FP16 output scale without silently publishing a zero scale.
 * @param maximum_probability Finite positive probability at an actual maximum score.
 * @return FP32 quantization scale, floored at the smallest positive physical FP16 value.
 * @throws std::domain_error If even the largest probability rounds to zero at FP16's minimum scale.
 * @details The floor retains representable long-row probabilities (for example
 * uniform 262144-column rows). It is quantizer geometry, not an ISA/precision
 * fallback. Truly unrepresentable distributions reject before output writes.
 */
inline float q8OutputScale(float maximum_probability)
{
    if(!std::isfinite(maximum_probability)||!(maximum_probability>0x1p-25f)||maximum_probability>1.0f)
        throw std::domain_error("CPU Q8 softmax maximum probability is not representable with FP16 metadata");
    return std::max(maximum_probability/127.0f,0x1p-24f);
}
/** @brief Fold a selected original endpoint without concealing a broken score domain.
 * @param maximum Previous maximum; @param code Sign-correct original integer extremum.
 * @param coefficient Original separately rounded FP16 score coefficient.
 * @return Updated maximum; negative infinity is an explicitly zero-probability score.
 * @throws std::domain_error For a positive-infinite or NaN active score.
 */
inline float foldQ8Maximum(float maximum,int code,float coefficient)
{
    const float value=float(code)*coefficient;
    if(std::isnan(value)||value==std::numeric_limits<float>::infinity())
        throw std::domain_error("CPU Q8 softmax has a NaN or positive-infinite active score");
    return std::max(maximum,value);
}
/** @brief Execute the explicit scalar diagnostic maximum over original active bytes.
 * @param row Checked original Q8 row, including its original causal prefix.
 * @return Existing diagnostic zero baseline when no finite active maximum exists.
 */
inline float q8MaximumScalar(const Row<Storage::Q8_1>&row)
{
    float maximum=-std::numeric_limits<float>::infinity();
    const auto blocks=(row.liveColumns()+31)/32;
    for(std::size_t block=0;block<blocks;++block)
    {
        const auto&source=row.data()[block];
        const auto live=std::min<std::size_t>(32,row.liveColumns()-block*32);
        const float coefficient=q8Coefficient(source,row.scale());
        const bool negative=coefficient<0.0f;
        int extremum=negative?127:-128;
        for(std::size_t lane=0;lane<live;++lane)
            extremum=negative?std::min(extremum,int(source.qs[lane])):std::max(extremum,int(source.qs[lane]));
        maximum=foldQ8Maximum(maximum,extremum,coefficient);
    }
    return maximum==-std::numeric_limits<float>::infinity()?0.0f:maximum;
}
#if defined(__AVX2__)
/** @brief Retain x86's native 32-byte maximum reduction without endpoint/sign errors.
 * @param row Checked original Q8 row, including original causal coordinates.
 * @return Existing diagnostic zero baseline when no finite active maximum exists.
 * @details The AVX-512 program also reads one original 32-byte Q8 block, so its
 * existing YMM integer reduction remains appropriate. Only the last causal
 * fragment uses bounded scalar tail comparisons, never complete-row replay.
 */
inline float q8MaximumX86(const Row<Storage::Q8_1>&row)
{
    float maximum=-std::numeric_limits<float>::infinity();
    const auto blocks=(row.liveColumns()+31)/32;
    for(std::size_t block=0;block<blocks;++block)
    {
        const auto&source=row.data()[block];
        const auto live=std::min<std::size_t>(32,row.liveColumns()-block*32);
        const float coefficient=q8Coefficient(source,row.scale());
        const bool negative=coefficient<0.0f;
        int extremum=negative?127:-128;
        if(live==32)
        {
            const auto bytes=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source.qs));
            const auto lo=_mm256_castsi256_si128(bytes),hi=_mm256_extracti128_si256(bytes,1);
            if(negative)
            {
                auto value=_mm_min_epi8(lo,hi);
                value=_mm_min_epi8(value,_mm_srli_si128(value,8));
                value=_mm_min_epi8(value,_mm_srli_si128(value,4));
                value=_mm_min_epi8(value,_mm_srli_si128(value,2));
                value=_mm_min_epi8(value,_mm_srli_si128(value,1));
                extremum=static_cast<std::int8_t>(_mm_extract_epi8(value,0));
            }
            else
            {
                auto value=_mm_max_epi8(lo,hi);
                value=_mm_max_epi8(value,_mm_srli_si128(value,8));
                value=_mm_max_epi8(value,_mm_srli_si128(value,4));
                value=_mm_max_epi8(value,_mm_srli_si128(value,2));
                value=_mm_max_epi8(value,_mm_srli_si128(value,1));
                extremum=static_cast<std::int8_t>(_mm_extract_epi8(value,0));
            }
        }
        else
            for(std::size_t lane=0;lane<live;++lane)
                extremum=negative?std::min(extremum,int(source.qs[lane])):std::max(extremum,int(source.qs[lane]));
        maximum=foldQ8Maximum(maximum,extremum,coefficient);
    }
    return maximum==-std::numeric_limits<float>::infinity()?0.0f:maximum;
}
/** @brief Author exact zero for inactive Q8 exponential lanes after polynomial evaluation.
 * @param values Computed original eight-lane exponentials; @param live Original active prefix.
 * @param start Original first coordinate of this fragment.
 * @return Exponentials with inactive coordinates exactly zero, including fully masked rows.
 */
inline __m256 maskQ8Exp(__m256 values,std::size_t live,std::size_t start) noexcept
{
    const int count=int(std::min<std::size_t>(8,live>start?live-start:0));
    const auto valid=_mm256_cmpgt_epi32(_mm256_set1_epi32(count),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
    return _mm256_and_ps(values,_mm256_castsi256_ps(valid));
}
#endif
#if defined(__AVX512F__)
/** @brief Keep original active coordinates and author exact zero in inactive AVX-512 lanes.
 * @param values Computed original sixteen-lane exponentials; @param live Original active prefix.
 * @param start Original first coordinate of this fragment.
 * @return Exactly masked exponentials, not small positive sentinel probabilities.
 */
inline __m512 maskQ8Exp(__m512 values,std::size_t live,std::size_t start) noexcept
{
    const unsigned count=unsigned(std::min<std::size_t>(16,live>start?live-start:0));
    const auto valid=static_cast<__mmask16>((std::uint32_t(1)<<count)-1);
    return _mm512_maskz_mov_ps(valid,values);
}
#endif
} // namespace llaminar2::primitives::softmax_detail
