/**
 * @file GPUHCDownPublication.cuh
 * @brief One original HC /4-SiLU and paired-Q8 arithmetic authority.
 *
 * The standalone publisher and reducer epilogue call this same32-lane primitive.
 * No projection partition, public value, rounding edge or quantizer changes.
 */
#pragma once
#include "HCDownOutputPublicationArgs.h"
#include "DeviceFP32NumericalContract.h"
#include "DeviceQ8ActivationNumericalContract.h"
#if defined(__HIPCC__)
#include "kernels/rocm/ops/ROCmSwiGLUDecodeEquivalentMath.h"
#else
#include <cuda_runtime.h>
#endif
namespace llaminar2::hc_down_publication {
/** @brief Preserve the original backend scale policy. @param maximum Exact block maximum. @return Stored scale. */
__device__ __forceinline__ float scale(float maximum) {
#if defined(__HIPCC__)
 return rocm::decode_equivalent::q8Scale(maximum);
#else
 return maximum>0.f?maximum/127.f:1.f;
#endif
}
/** @brief Preserve native reciprocal/multiply/RNE. @param value Original value. @param stored Stored scale. @return Q8 integer. */
__device__ __forceinline__ int quantize(float value,float stored) {
 const float inverse=1.f/stored;
#if defined(__HIPCC__)
 return rocm::decode_equivalent::quantizeQ8(value,inverse);
#else
 const float scaled=value*inverse;
 return static_cast<int>(rintf(fminf(127.f,fmaxf(-127.f,scaled))));
#endif
}
/** @brief Original32-lane shuffle. @param value Lane operand. @param delta Original XOR edge. @return Exact partner. */
__device__ __forceinline__ float shuffle(float value,int delta) {
#if defined(__HIPCC__)
 return __shfl_xor(value,delta,32);
#else
 return __shfl_xor_sync(0xffffffffu,value,delta,32);
#endif
}
/** @brief Existing backend scale publication; default for every original caller. */
struct OriginalScalePublication {
 /** @brief Publish the original scale. @param maximum Original tree root. @return Original stored scale. */
 __device__ __forceinline__ float operator()(float maximum) const {return scale(maximum);}
};
/** @brief Explicit bounded positive-half publication for the qualified resident entry. */
struct NativePositiveHalfScalePublication {
 /** @brief Publish the bounded scale with native RNE half conversion.
  * @param maximum Original tree root. @return Exact widened stored half.
  * @pre The admitted native descriptor retains RNE and FP16 subnormals.
  */
 __device__ __forceinline__ float operator()(float maximum) const {
  return device_q8_activation_contract::publishPositiveFiniteHalfScale(
   device_q8_activation_contract::boundedScale(maximum));
 }
};
/**
 * @brief Publish one original value within a complete active32-column subgroup.
 * @param raw Original stored-down FP32 value, before divide-by-four SiLU.
 * @param output Exact sealed existing activated/Q8 destinations.
 * @param offset Original physical bottleneck element; subgroup aligned to32.
 * @tparam ScalePublication Explicit scale storage policy; original backend by default.
 * @tparam PublishActivated Materialize the rounded FP32 activation when the graph consumes it.
 * @tparam Output Existing typed output fields after phase row/ownership admission.
 * @param scale_publication Storage-only half publication with unchanged bounded scale math.
 */
template<class ScalePublication=OriginalScalePublication,bool PublishActivated=true,class Output>
__device__ __forceinline__ void publish(float raw,const Output& output,std::uint64_t offset,
 ScalePublication scale_publication={}) {
 const unsigned lane=offset%32;
 const float scaled=raw*0.25f;
 const float value=scaled*(1.0f/(1.0f+expf(-scaled)));
 if constexpr(PublishActivated)output.activated[offset]=value;
 if(!isfinite(value)) {
#if defined(__HIPCC__)
 __builtin_trap();
#else
 __trap();
#endif
 }
 float maximum=fabsf(value);
 for(int distance=16;distance;distance/=2)maximum=fmaxf(maximum,shuffle(maximum,distance));
 const float stored=scale_publication(maximum);
 const int primary=quantize(value,stored);
 output.primary[offset]=static_cast<signed char>(primary);
 const float residual=device_fp32_contract::add(value,-device_fp32_contract::multiply(stored,static_cast<float>(primary)));
 if(!isfinite(residual)) {
#if defined(__HIPCC__)
 __builtin_trap();
#else
 __trap();
#endif
 }
 float residualMaximum=fabsf(residual);
 for(int distance=16;distance;distance/=2)residualMaximum=fmaxf(residualMaximum,shuffle(residualMaximum,distance));
 const float residualScale=scale_publication(residualMaximum);
 output.residual[offset]=static_cast<signed char>(quantize(residual,residualScale));
 if(lane==0){output.primaryScales[offset/32]=stored;output.residualScales[offset/32]=residualScale;}
}
}
