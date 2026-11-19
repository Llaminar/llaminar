/**
 * @file ROCmRetainedM1QuietMagnitude.h
 * @brief Ordered gfx906 wave64 maximum for the closed retained M1 HC and FFN Q8 publishers.
 */
#pragma once

#include "kernels/rocm/ops/ROCmSwiGLUDecodeEquivalentMath.h"
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
namespace qfn_retained_m1_quantizer {

/** @brief Take the absolute value and quiet a signaling NaN before a numerical maximum.
 * @param x Original FP32 component. @return Its positive magnitude or a quiet NaN.
 */
__device__ __forceinline__ float quietAbs(float x)
{
    float out;
    asm("v_max_f32_e64 %0, |%1|, |%1|" : "=v"(out) : "v"(x));
    return out;
}

/**
 * @brief Compute the common scale maximum for the 32 values held by eight lanes.
 * @param values Four consecutive source values belonging to this lane.
 * @return The group's maximum magnitude, or a quiet NaN for an all-NaN group.
 * @details Reduce each component across the same 4, 2 and 1 lane distances,
 * then combine x/z and y/w in the original tree. The two masked row shifts
 * cross four-lane banks; the quad permutations finish the tree in registers.
 * NOPs provide the gfx906 producer-to-DPP issue distance. The bounded scale
 * consumes the result; original activation bits still supply the quantizer.
 */
__device__ __forceinline__ float quietMagnitudeMax32(float4 values)
{
    const float a=quietAbs(values.x), b=quietAbs(values.y);
    const float c=quietAbs(values.z), d=quietAbs(values.w);
    const auto componentRoot=[] __device__ (float component) {
        float root;
        asm volatile("s_nop 1\n"
                     "v_max_f32_dpp %0, %1, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                     "v_max_f32_dpp %0, %1, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                     "s_nop 1\n"
                     "v_max_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
                     "s_nop 1\n"
                     "v_max_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                     : "=&v"(root) : "v"(component));
        return root;
    };
    const float x = componentRoot(a), y = componentRoot(b);
    const float z = componentRoot(c), w = componentRoot(d);
    return fmaxf(fmaxf(x,z),fmaxf(y,w));
}
} // namespace qfn_retained_m1_quantizer
#endif

namespace qfn_retained_m1_hc_ffn_device {

/**
 * @brief Publish four adjacent norm values with the exact original32-leaf root.
 * @param value Original rounded values at logical positions4*lane through4*lane+3.
 * @param output Original aligned32-code block, shared by one complete8-lane cohort.
 * @param scale_output Original scale word; cohort lane0 is the unique publisher.
 * @param lane Logical lane0..7, with all eight peers active in the physical wave.
 * @details Offsets4/2/1 below virtualize original offsets16/8/4 independently
 * for each component. The remaining original root nodes combine0+2,1+3,then
 * those two values, before the original lane0 broadcast. No fmax association
 * changes in the portable path. The closed path uses the same partition and
 * explicitly quieted magnitudes with DPP exchanges. One unchanged reciprocal
 * serves four existing quantizeQ8 calls. Code packing changes storage issue only.
 */
template<bool NativeHalfScale>
__device__ __forceinline__ void current_hc_norm_q8_four(
    float4 value,signed char* output,float* scale_output,unsigned lane) {
    float maximum;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
    if constexpr(NativeHalfScale) {
        maximum=::qfn_retained_m1_quantizer::quietMagnitudeMax32(value);
    } else
#endif
    {
        float a=fabsf(value.x),b=fabsf(value.y),c=fabsf(value.z),d=fabsf(value.w);
#pragma unroll
        for(int offset=4;offset>0;offset>>=1) {
            a=fmaxf(a,__shfl_down(a,offset,8));
            b=fmaxf(b,__shfl_down(b,offset,8));
            c=fmaxf(c,__shfl_down(c,offset,8));
            d=fmaxf(d,__shfl_down(d,offset,8));
        }
        const float even=fmaxf(a,c),odd=fmaxf(b,d);
        const float root=fmaxf(even,odd);
        maximum=__shfl(root,0,8);
    }
    namespace q8=::llaminar2::device_q8_activation_contract;
    const float scale=NativeHalfScale?q8::publishPositiveFiniteHalfScale(q8::boundedScale(maximum)):
        ::llaminar2::rocm::decode_equivalent::q8Scale(maximum);
    const float inverse=1.0f/scale;
    if(lane==0)*scale_output=scale;
    const unsigned q0=unsigned(::llaminar2::rocm::decode_equivalent::quantizeQ8(value.x,inverse))&255u;
    const unsigned q1=unsigned(::llaminar2::rocm::decode_equivalent::quantizeQ8(value.y,inverse))&255u;
    const unsigned q2=unsigned(::llaminar2::rocm::decode_equivalent::quantizeQ8(value.z,inverse))&255u;
    const unsigned q3=unsigned(::llaminar2::rocm::decode_equivalent::quantizeQ8(value.w,inverse))&255u;
    const unsigned packed=q0|(q1<<8)|(q2<<16)|(q3<<24);
    // The original banks are byte arrays; memcpy preserves their alias contract
    // while the actual aligned four-byte ownership permits one DWORD store.
    __builtin_memcpy(output+4*lane,&packed,sizeof(packed));
}

} // namespace qfn_retained_m1_hc_ffn_device
