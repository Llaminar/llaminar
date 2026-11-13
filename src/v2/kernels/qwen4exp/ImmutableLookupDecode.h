/**
 * @file ImmutableLookupDecode.h
 * @brief Shared native source decoding and exact lookup row geometry.
 *
 * Format dispatch occurs before execution. Quantized rows keep their original
 * scales and packed codes until the requested values are decoded; floating rows
 * are widened directly without a second quantizer or intermediate table copy.
 */
#pragma once
#include "../common/NativeEmbeddingDecode.h"
#include "../common/NativeEmbeddingLayout.h"
#include <limits>
#include <stdexcept>

/** @brief Complete native quantized source inventory, shared by all backends. */
#define QFN_LOOKUP_QUANT_FORMATS(M) \
    M(Q4_0) M(Q4_1) M(Q5_0) M(Q5_1) M(Q8_0) M(Q8_1) \
    M(Q2_K) M(Q3_K) M(Q4_K) M(Q5_K) M(Q6_K) M(Q8_K) \
    M(IQ1_S) M(IQ1_M) M(IQ2_XXS) M(IQ2_XS) M(IQ2_S) M(IQ3_XXS) M(IQ3_S) M(IQ4_NL) M(IQ4_XS)
#if defined(__CUDACC__) || defined(__HIPCC__)
#define QFN_LOOKUP_INLINE __device__ __forceinline__
#else
#define QFN_LOOKUP_INLINE inline
#endif
namespace llaminar2::qwen4exp::lookup_detail
{
/** @brief Authenticate exact complete source rows before binding persistent storage. */
inline void validateGeometry(TensorType format,int elements,std::size_t bytes)
{
    if(elements<=0)throw std::invalid_argument("Immutable lookup requires a positive logical row width");
    std::size_t expected=0;
    if(format==TensorType::FP32)expected=static_cast<std::size_t>(elements)*4;
    else if(format==TensorType::FP16||format==TensorType::BF16)expected=static_cast<std::size_t>(elements)*2;
    else expected=NativeEmbeddingLayout::forSource(format,1,elements).row_stride_bytes;
    if(bytes!=expected)throw std::invalid_argument("Immutable lookup native row bytes differ from its source format geometry");
}
/** @brief Decode one requested source value with the canonical original metadata arithmetic. */
template<TensorType Format>
QFN_LOOKUP_INLINE float decode(const std::byte *row,std::size_t column,const NativeEmbeddingIQTables &tables)
{
    if constexpr(Format==TensorType::FP32)
        return device_fp32_contract::floatFromBits(native_embedding::word32(row+column*4));
    else if constexpr(Format==TensorType::FP16)
        return native_embedding::half(native_embedding::word16(row+column*2));
    else if constexpr(Format==TensorType::BF16)
        return device_fp32_contract::floatFromBits(std::uint32_t(native_embedding::word16(row+column*2))<<16);
    else return native_embedding::decode<Format>(row,column,tables);
}
} // namespace llaminar2::qwen4exp::lookup_detail
#undef QFN_LOOKUP_INLINE
