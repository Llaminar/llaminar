/**
 * @file OriginalSingleQ8Block32.cuh
 * @brief Source-owned ordinary GPU SingleQ8 block publication shared by original and fused producers.
 *
 * One logical32-lane cohort owns the exact original backend maximum tree,
 * scale, reciprocal, rounding and saturation. Explicit32-lane HIP width makes
 * the former32-thread CTA reusable inside a larger producer; lane0 retains
 * the same original downward-tree operands and order. No global state exists.
 */
#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include "kernels/rocm/ops/ROCmSwiGLUDecodeEquivalentMath.h"
#else
#include <cuda_runtime.h>
#endif
namespace llaminar2::original_single_q8 {
/** @brief Publish one original K32 block and optional native integer sum.
 * @param value Exact original FP32 element, or zero for an invalid tail lane.
 * @param valid Whether this original lane owns a physical output byte.
 * @param output First original code byte in this logical block.
 * @param scaleOutput Original FP32 scale word.
 * @param sumOutput Optional original sum word; integer arithmetic is exact.
 * @param lane Original logical lane0..31, independent of physical wave width.
 */
__device__ __forceinline__ void publish(float value,bool valid,std::int8_t* output,
 float* scaleOutput,std::int32_t* sumOutput,unsigned lane) {
 float maximum=fabsf(value);
#if defined(__HIPCC__)
#pragma unroll
 for(int offset=16;offset>0;offset>>=1)maximum=fmaxf(maximum,__shfl_down(maximum,offset,32));
 maximum=__shfl(maximum,0,32);
 const float scale=rocm::decode_equivalent::q8Scale(maximum);
#else
#pragma unroll
 for(int mask=16;mask>0;mask>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,mask,32));
 const float scale=maximum>0.f?maximum/127.f:1.f;
#endif
 const float inverse=1.0f/scale;
 if(lane==0)*scaleOutput=scale;
#if defined(__HIPCC__)
 const int q=rocm::decode_equivalent::quantizeQ8(value,inverse);
#else
 const float qval=value*inverse;
 const int q=static_cast<int>(rintf(fminf(127.f,fmaxf(-127.f,qval))));
#endif
 if(valid)output[lane]=static_cast<std::int8_t>(q);
 if(sumOutput) {
  int sum=valid?q:0;
#pragma unroll
  for(int offset=16;offset>0;offset>>=1) {
#if defined(__HIPCC__)
   sum+=__shfl_down(sum,offset,32);
#else
   sum+=__shfl_xor_sync(0xffffffffu,sum,offset,32);
#endif
  }
  if(lane==0)*sumOutput=sum;
 }
}
}
