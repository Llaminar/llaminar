/**
 * @file CPUImmutableLookupKernels.cpp
 * @brief Exact CPU native-row decoding with caller-owned output and no scratch.
 *
 * Immutable table gather remains the RAM/mmap I/O authority. This kernel only
 * widens live gathered native rows, parallelizing contiguous output blocks in
 * the caller's nested-safe CPU team. Each value is independent of batch size,
 * worker count, and output partitioning. Floating/IQ4 SIMD leaves retain original
 * source words/codebooks; incomplete logical rows never reinterpret physical
 * padding as another row. Checked synchronous borrows do not prove capacity or
 * extend storage leases, and empty retained invocations still complete their team.
 */
#include "../../qwen4exp/ImmutableLookupKernels.h"
#include "../../qwen4exp/ImmutableLookupDecode.h"
#include "../../../tensors/IQQuantTables.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/CPUFeatures.h"
#include <algorithm>
#include <bit>
#include <cstring>
#if LLAMINAR_COMPILED_WITH_NEON
#include "../../../tensors/NEONFP16.h"
#include "../../../tensors/NEONFP32.h"
#elif defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Bind the canonical process-lifetime CPU IQ codebooks without copying them.
 * @return Original immutable table roots, with no decoded or duplicated coefficient bank.
 */
NativeEmbeddingIQTables tables()
{
    return {iq3s_grid,iq3xxs_grid,iq2s_grid,iq2xs_grid,iq2xxs_grid,iq1s_grid};
}
/** @brief Admit the canonical selected ISA without an automatic scalar substitution.
 * @return Explicit diagnostic scalar or one native implementation compiled for this architecture.
 * @throws std::runtime_error For a foreign or uncompiled native instruction program.
 */
ISALevel lookupISA()
{
    static_assert(std::endian::native==std::endian::little,
        "Immutable lookup SIMD requires the original little-endian native source ABI");
    const auto isa=activeISALevel();
    if(isa!=ISALevel::Scalar&&!isNativeCPUExecutionISAPair(compiledCPUISALevel(),isa))
        throw std::runtime_error("Immutable lookup CPU decode has no matching compiled native ISA");
    return isa;
}
/** @brief Expand original floating rows in native vectors, preserving original NaN payload bits.
 * @tparam Format Original FP32/FP16/BF16 physical representation.
 * @param source Complete original tightly packed floating rows.
 * @param size Exact live scalar count; physical row strides have already been authenticated.
 * @param output Caller-owned disjoint FP32 publication, with no intermediate converted bank.
 */
template<TensorType Format>
void floatingRows(const std::byte *source,std::size_t size,float *output)
{
    const auto isa=lookupISA();
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
#if LLAMINAR_COMPILED_WITH_NEON
                if(isa==ISALevel::NEON||isa==ISALevel::NEONDotProd)
                    for(;index+4<=end;index+=4)
                    {
                        const auto packed=vld1_u16(words+index);
                        if constexpr(Format==TensorType::BF16)
                            vst1q_f32(output+index,vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(packed),16)));
                        else
                            vst1q_f32(output+index,neon_fp16::widen(packed));
                    }
#endif
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
/** @brief Decode original IQ4 blocks without publishing padded coordinates as live rows.
 * @param source Contiguous complete physical source blocks, including each row's final padding.
 * @param width Original logical elements per row, not a rounded block capacity.
 * @param stride Authenticated complete original row bytes.
 * @param count Exact live row count.
 * @param output Disjoint requested live FP32 rows; no decoded intermediate is retained.
 * @details Register byte lookups select the original signed grid. Each coordinate
 * receives one FP32 product. Whole physical blocks distribute independently, so
 * even one long row can use the retained team. Partial rows keep original row
 * coordinates; scalar remainders are bounded below the native vector width.
 */
void iq4Rows(const std::byte *source,int width,std::size_t stride,std::size_t count,float *output)
{
    const auto isa=lookupISA();const auto *blocks=reinterpret_cast<const IQ4_NLBlock *>(source);
    const auto blocksPerRow=stride/sizeof(IQ4_NLBlock);
    const auto blockCount=count*blocksPerRow;
    const bool completeRows=width%32==0;
    auto work=[&] {
#pragma omp for schedule(static)
        for(std::size_t b=0;b<blockCount;++b)
        {
            // The common complete-row path needs no row division. A partial
            // physical final block never spills its padded codes into the next
            // logical row, including widths smaller than one source block.
            const auto column=completeRows?0:(b%blocksPerRow)*32;
            const int valid=completeRows?32:std::min<std::size_t>(32,std::size_t(width)-column);
            auto *destination=output+(completeRows?b*32:(b/blocksPerRow)*width+column);
            const auto &block=blocks[b];
            const float scale=native_embedding::half(block.d);
#if LLAMINAR_COMPILED_WITH_NEON
            if(isa==ISALevel::NEON||isa==ISALevel::NEONDotProd)
            {
                const auto packed=vld1q_u8(block.qs),mask=vdupq_n_u8(15);
                const auto grid=vld1q_s8(kvalues_iq4nl_i8);
                auto low=vqtbl1q_s8(grid,vandq_u8(packed,mask));
                auto high=vqtbl1q_s8(grid,vshrq_n_u8(packed,4));
                const auto d=vdupq_n_f32(scale);
                int c=0;
                for(;c+4<=valid;c+=4)
                {
                    const auto codes=c<16?low:high;
                    const auto integers=vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(codes))));
                    vst1q_f32(destination+c,neon_fp32::multiply(d,vcvtq_f32_s32(integers)));
                    if(c<16)low=vextq_s8(low,low,4);else high=vextq_s8(high,high,4);
                }
                for(;c<valid;++c)destination[c]=native_embedding::decode<TensorType::IQ4_NL>(&block,c,{});
                continue;
            }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
            if(isa==ISALevel::AVX2||isa==ISALevel::AVX512)
            {
                const auto grid=_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl_i8));
                const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
                const auto mask=_mm_set1_epi8(15);
                const auto low=_mm_shuffle_epi8(grid,_mm_and_si128(packed,mask));
                const auto high=_mm_shuffle_epi8(grid,_mm_and_si128(_mm_srli_epi16(packed,4),mask));
#if defined(__AVX512F__)
                if(isa==ISALevel::AVX512)
                {
                    const auto d=_mm512_set1_ps(scale);
                    const auto first=static_cast<__mmask16>(valid>=16?0xffffu:(1u<<valid)-1u);
                    _mm512_mask_storeu_ps(destination,first,_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(low)),d));
                    if(valid>16)
                    {
                        const auto last=static_cast<__mmask16>((1u<<(valid-16))-1u);
                        _mm512_mask_storeu_ps(destination+16,last,_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(high)),d));
                    }
                    continue;
                }
#endif
#if defined(__AVX2__)
                const auto d=_mm256_set1_ps(scale);
                const auto write=[&]<int Offset>(__m128i codes)
                {
                    if(valid<=Offset)return;
                    const auto result=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)),d);
                    if(valid>=Offset+8)_mm256_storeu_ps(destination+Offset,result);
                    else
                    {
                        const auto lanes=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
                        const auto live=_mm256_cmpgt_epi32(_mm256_set1_epi32(valid-Offset),lanes);
                        _mm256_maskstore_ps(destination+Offset,live,result);
                    }
                };
                write.template operator()<0>(low);
                write.template operator()<8>(_mm_srli_si128(low,8));
                write.template operator()<16>(high);
                write.template operator()<24>(_mm_srli_si128(high,8));
                continue;
#endif
            }
#endif
            // Only explicit scalar intent reaches a complete scalar block.
            for(int column=0;column<valid;++column)
                destination[column]=native_embedding::decode<TensorType::IQ4_NL>(&block,column,{});
        }
    };
    OMP_WORKSHARE_REGION(work);
}
/** @brief Distribute original coordinates for formats with the shared scalar source decoder.
 * @tparam Format Unchanged original quantized representation.
 * @param width Exact live row elements; @param stride Complete physical row bytes.
 * @param rows Caller-owned original rows; @param count Live row count.
 * @param output Disjoint requested FP32 rows, not a cached decoded mirror.
 */
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
/** @brief Authenticate original row publications, then execute their declared CPU decoder.
 * @param format Original source encoding; @param width Positive logical row elements.
 * @param stride Exact complete physical row bytes, including a partial final source block.
 * @param rows Caller-owned original live rows; @param count Exact live row count.
 * @param output Caller-owned disjoint FP32 publication.
 * @throws std::invalid_argument For malformed geometry, missing/wrapping extents or intersections.
 * @details All checks precede retained-team admission/completion. Zero work
 * rendezvous creates no standalone team; no source/output scan follows release.
 */
void decodeCPUImmutableLookupRows(TensorType format,int width,std::size_t stride,
    const std::byte *rows,std::size_t count,float *output)
{
    lookup_detail::validateGeometry(format,width,stride);
    if(count>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(width)/sizeof(float)||
       count>std::numeric_limits<std::size_t>::max()/stride||(!rows&&count)||(!output&&count)||
       (reinterpret_cast<const void *>(rows)==output&&count))
        throw std::invalid_argument("Immutable lookup CPU decode has invalid live extent or aliased storage");
    const auto nativeBytes=count*stride,outputBytes=count*static_cast<std::size_t>(width)*sizeof(float);
    const auto nativeAddress=reinterpret_cast<std::uintptr_t>(rows),outputAddress=reinterpret_cast<std::uintptr_t>(output);
    if(nativeAddress>std::numeric_limits<std::uintptr_t>::max()-nativeBytes||
       outputAddress>std::numeric_limits<std::uintptr_t>::max()-outputBytes||
       (nativeAddress<outputAddress+outputBytes&&outputAddress<nativeAddress+nativeBytes))
        throw std::invalid_argument("Immutable lookup native source and decoded output overlap");
    // Reject malformed geometry/aliases before the completion edge; never
    // inspect caller-owned publications again after the team releases them.
    if(!::llaminar::v2::beginWorkshare(count!=0))return;
    const auto size=count*static_cast<std::size_t>(width);
    if(format==TensorType::FP32)return floatingRows<TensorType::FP32>(rows,size,output);
    if(format==TensorType::FP16)return floatingRows<TensorType::FP16>(rows,size,output);
    if(format==TensorType::BF16)return floatingRows<TensorType::BF16>(rows,size,output);
    if(format==TensorType::IQ4_NL)return iq4Rows(rows,width,stride,count,output);
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
