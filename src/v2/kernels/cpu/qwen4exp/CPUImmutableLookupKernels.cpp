/**
 * @file CPUImmutableLookupKernels.cpp
 * @brief Exact CPU native-row decoding with caller-owned output and no scratch.
 *
 * Immutable table gather remains the RAM/mmap I/O authority. This kernel only
 * widens live gathered native rows, parallelizing contiguous output blocks in
 * the caller's nested-safe CPU team. Each value is independent of batch size,
 * worker count, and output partitioning.
 */
#include "../../qwen4exp/ImmutableLookupKernels.h"
#include "../../qwen4exp/ImmutableLookupDecode.h"
#include "../../../tensors/IQQuantTables.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/CPUFeatures.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Bind the canonical process-lifetime CPU IQ codebooks without copying them. */
NativeEmbeddingIQTables tables()
{
    return {iq3s_grid,iq3xxs_grid,iq2s_grid,iq2xs_grid,iq2xxs_grid,iq1s_grid};
}
/** @brief Expand original floating rows in native vectors, preserving original NaN payload bits. */
template<TensorType Format>
void floatingRows(const std::byte *source,std::size_t size,float *output)
{
    const auto isa=activeISALevel();
    auto work=[&] {
#pragma omp for schedule(static)
        for(std::size_t begin=0;begin<size;begin+=256)
        {
            const auto end=std::min(begin+256,size);auto index=begin;
            if constexpr(Format==TensorType::FP32)
            {
                std::memcpy(output+begin,source+begin*4,(end-begin)*4);
                continue;
            }
            else
            {
                const auto *words=reinterpret_cast<const std::uint16_t *>(source);
#if defined(__AVX512F__)
                if(isa==ISALevel::AVX512)
                    for(;index+16<=end;index+=16)
                    {
                        const auto packed=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(words+index));
                        const auto bits=_mm512_cvtepu16_epi32(packed);
                        __m512 result;
                        if constexpr(Format==TensorType::BF16)result=_mm512_castsi512_ps(_mm512_slli_epi32(bits,16));
                        else
                        {
                            result=_mm512_cvtph_ps(packed);
                            const auto mantissa=_mm512_and_si512(bits,_mm512_set1_epi32(1023));
                            const auto nan=_mm512_cmpeq_epi32_mask(_mm512_and_si512(bits,_mm512_set1_epi32(0x7c00)),_mm512_set1_epi32(0x7c00))&
                                _mm512_cmpneq_epi32_mask(mantissa,_mm512_setzero_si512());
                            const auto preserved=_mm512_or_si512(_mm512_set1_epi32(0x7f800000),
                                _mm512_or_si512(_mm512_slli_epi32(mantissa,13),_mm512_slli_epi32(_mm512_and_si512(bits,_mm512_set1_epi32(0x8000)),16)));
                            result=_mm512_mask_mov_ps(result,nan,_mm512_castsi512_ps(preserved));
                        }
                        _mm512_storeu_ps(output+index,result);
                    }
#endif
#if defined(__AVX2__)
                if(isa>=ISALevel::AVX2)
                    for(;index+8<=end;index+=8)
                    {
                        const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i *>(words+index));
                        const auto bits=_mm256_cvtepu16_epi32(packed);
                        __m256 result;
                        if constexpr(Format==TensorType::BF16)result=_mm256_castsi256_ps(_mm256_slli_epi32(bits,16));
                        else
                        {
                            result=_mm256_cvtph_ps(packed);
                            const auto mantissa=_mm256_and_si256(bits,_mm256_set1_epi32(1023));
                            const auto nan=_mm256_andnot_si256(_mm256_cmpeq_epi32(mantissa,_mm256_setzero_si256()),
                                _mm256_cmpeq_epi32(_mm256_and_si256(bits,_mm256_set1_epi32(0x7c00)),_mm256_set1_epi32(0x7c00)));
                            const auto preserved=_mm256_or_si256(_mm256_set1_epi32(0x7f800000),
                                _mm256_or_si256(_mm256_slli_epi32(mantissa,13),_mm256_slli_epi32(_mm256_and_si256(bits,_mm256_set1_epi32(0x8000)),16)));
                            result=_mm256_blendv_ps(result,_mm256_castsi256_ps(preserved),_mm256_castsi256_ps(nan));
                        }
                        _mm256_storeu_ps(output+index,result);
                    }
#endif
                for(;index<end;++index)output[index]=lookup_detail::decode<Format>(source,index,{});
            }
        }
    };
    OMP_WORKSHARE_REGION(work);
}
/** @brief Decode a complete IQ4_NL block once, amortizing original scale widening across thirty-two values.
 * @details Byte shuffles select the exact signed integer source grid. AVX2 and
 * AVX512 widen those same integers before one binary32 multiplication; SIMD
 * boundaries never change a value's arithmetic or depend on grouped row count.
 */
void iq4Rows(const std::byte *source,std::size_t blockCount,float *output)
{
    const auto isa=activeISALevel();const auto *blocks=reinterpret_cast<const IQ4_NLBlock *>(source);
    auto work=[&] {
#pragma omp for schedule(static)
        for(std::size_t b=0;b<blockCount;++b)
        {
            const auto &block=blocks[b];auto *destination=output+b*32;
            const float scale=native_embedding::half(block.d);
            const auto grid=_mm_setr_epi8(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);
            const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            const auto mask=_mm_set1_epi8(15);
            const auto low=_mm_shuffle_epi8(grid,_mm_and_si128(packed,mask));
            const auto high=_mm_shuffle_epi8(grid,_mm_and_si128(_mm_srli_epi16(packed,4),mask));
#if defined(__AVX512F__)
            if(isa==ISALevel::AVX512)
            {
                const auto d=_mm512_set1_ps(scale);
                _mm512_storeu_ps(destination,_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(low)),d));
                _mm512_storeu_ps(destination+16,_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(high)),d));
                continue;
            }
#endif
#if defined(__AVX2__)
            if(isa>=ISALevel::AVX2)
            {
                const auto d=_mm256_set1_ps(scale);
                _mm256_storeu_ps(destination,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low)),d));
                _mm256_storeu_ps(destination+8,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low,8))),d));
                _mm256_storeu_ps(destination+16,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high)),d));
                _mm256_storeu_ps(destination+24,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high,8))),d));
                continue;
            }
#endif
            for(int column=0;column<32;++column)
                destination[column]=native_embedding::decode<TensorType::IQ4_NL>(&block,column,{});
        }
    };
    OMP_WORKSHARE_REGION(work);
}
/** @brief Distribute contiguous requested output blocks through the existing CPU team. */
template<TensorType Format>
void decodeRows(int width,std::size_t stride,const std::byte *rows,std::size_t count,float *output)
{
    const auto grids=tables();const auto size=count*static_cast<std::size_t>(width);
    auto work=[&] {
#pragma omp for schedule(static)
        for(std::size_t begin=0;begin<size;begin+=256)
            for(auto index=begin;index<std::min(begin+256,size);++index)
                output[index]=lookup_detail::decode<Format>(rows+(index/width)*stride,index%width,grids);
    };
    OMP_WORKSHARE_REGION(work);
}
} // namespace
void decodeCPUImmutableLookupRows(TensorType format,int width,std::size_t stride,
    const std::byte *rows,std::size_t count,float *output)
{
    lookup_detail::validateGeometry(format,width,stride);
    if(count>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(width)/sizeof(float)||
       count>std::numeric_limits<std::size_t>::max()/stride||(!rows&&count)||(!output&&count)||
       (reinterpret_cast<const void *>(rows)==output&&count))
        throw std::invalid_argument("Immutable lookup CPU decode has invalid live extent or aliased storage");
    if(!count)return;
    const auto nativeBytes=count*stride,outputBytes=count*static_cast<std::size_t>(width)*sizeof(float);
    const auto nativeAddress=reinterpret_cast<std::uintptr_t>(rows),outputAddress=reinterpret_cast<std::uintptr_t>(output);
    if(nativeAddress>std::numeric_limits<std::uintptr_t>::max()-nativeBytes||
       outputAddress>std::numeric_limits<std::uintptr_t>::max()-outputBytes||
       (nativeAddress<outputAddress+outputBytes&&outputAddress<nativeAddress+nativeBytes))
        throw std::invalid_argument("Immutable lookup native source and decoded output overlap");
    const auto size=count*static_cast<std::size_t>(width);
    if(format==TensorType::FP32)return floatingRows<TensorType::FP32>(rows,size,output);
    if(format==TensorType::FP16)return floatingRows<TensorType::FP16>(rows,size,output);
    if(format==TensorType::BF16)return floatingRows<TensorType::BF16>(rows,size,output);
    if(format==TensorType::IQ4_NL)return iq4Rows(rows,size/32,output);
    switch(format)
    {
#define QFN_DECODE_CASE(F) case TensorType::F:return decodeRows<TensorType::F>(width,stride,rows,count,output);
        QFN_DECODE_CASE(FP32) QFN_DECODE_CASE(FP16) QFN_DECODE_CASE(BF16)
        QFN_LOOKUP_QUANT_FORMATS(QFN_DECODE_CASE)
#undef QFN_DECODE_CASE
    default:throw std::invalid_argument("Immutable lookup CPU decode source format is unsupported");
    }
}
} // namespace llaminar2::qwen4exp
