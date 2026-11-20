/**
 * @file ROCmPersistentMoEFFNQuantize.h
 * @brief One original32-lane FFN Q8 publication body with exact transfer policies.
 *
 * The descending XOR16/8/4/2/1 max and integer-sum trees, q8Scale,
 * quantizeQ8FromScale and stores are shared by persistent/ordinary FFN and
 * NativeLayout. Only the target's lane-word transfer is selected at compile time.
 * Complete32-lane source participation and original output ownership are required;
 * this header owns no allocation, stream, barrier, numerical or public launch policy.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include "../moe/ROCmMoEWaveReduction.h"
#include "../ops/ROCmSwiGLUDecodeEquivalentMath.h"

/**
 * @brief Publish one canonical32-code activation and its exact integer sum.
 * @tparam Transfer Compile-time bit-transfer policy; arithmetic is unchanged.
 * @param value Original FP32 element. @param index Physical code coordinate in a complete32-lane group.
 * @param codes Destination codes. @param scales Destination scales. @param sums Destination exact sums.
 * @details Every call is made by a complete logical32-lane group. Both shuffle
 * trees are confined to32 lanes; signed code sum lies within [-4096,4064].
 */
template<llaminar2::rocm::moe_wave_contract::QuantizerXorPolicy Transfer=
    llaminar2::rocm::moe_wave_contract::QuantizerXorPolicy::Indexed>
__device__ __forceinline__ void persistentMoEFFNQuantize(float value,size_t index,
    int8_t* codes,float* scales,int32_t* sums)
{
    float maximum=fabsf(value);
    #pragma unroll
    for(int mask=16;mask>0;mask>>=1)maximum=fmaxf(maximum,llaminar2::rocm::moe_wave_contract::quantizerXor32<Transfer>(maximum,mask));
    const float scale=llaminar2::rocm::decode_equivalent::q8Scale(maximum);
    const int quantized=llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(value,scale);
    codes[index]=static_cast<int8_t>(quantized);
    int sum=quantized;
    #pragma unroll
    for(int mask=16;mask>0;mask>>=1)sum+=llaminar2::rocm::moe_wave_contract::quantizerXor32<Transfer>(sum,mask);
    if((threadIdx.x%32)==0){scales[index/32]=scale;sums[index/32]=sum;}
}
