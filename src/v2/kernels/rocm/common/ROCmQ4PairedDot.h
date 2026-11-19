/** @file ROCmQ4PairedDot.h
 * @brief Shared original affine-Q4 K32 gate/up integer dot sequence.
 */
#pragma once
#include "kernels/rocm/common/ROCmSignedDot4.h"

namespace llaminar2::rocm {
struct Q4PairDots {
    int gate,up,sum;
};

/** @brief Append one packed activation word to the original affine-Q4 pair. */
__device__ __forceinline__ void q4PairedDotStep(
    int activation,unsigned gateWord,unsigned upWord,unsigned shift,int& gate,int& up)
{
    const int gq=int((gateWord>>shift)&0x0f0f0f0fu);
    const int uq=int((upWord>>shift)&0x0f0f0f0fu);
    gate=rocm_integer_contract::signedDot4(activation,gq,gate);
    up=rocm_integer_contract::signedDot4(activation,uq,up);
}

/** @brief Decode both original nibble planes against the same eight packed activation words.
 * @tparam CountSum Whether the caller still needs the exact signed activation sum.
 */
template<bool CountSum,class Words>
__device__ __forceinline__ Q4PairDots q4PairedDots(
    const Words& gate,const Words& up,const int* activation,int sum)
{
    const unsigned gw[4]={unsigned(gate.x),unsigned(gate.y),unsigned(gate.z),unsigned(gate.w)};
    const unsigned uw[4]={unsigned(up.x),unsigned(up.y),unsigned(up.z),unsigned(up.w)};
    int gd=0,ud=0;
#pragma unroll
    for(unsigned group=0;group<8;++group) {
        const int value=activation[group];
        if constexpr(CountSum)
            sum=rocm_integer_contract::signedDot4(value,0x01010101,sum);
        q4PairedDotStep(value,gw[group&3],uw[group&3],(group>>2)*4,gd,ud);
    }
    return {gd,ud,sum};
}
} // namespace llaminar2::rocm
