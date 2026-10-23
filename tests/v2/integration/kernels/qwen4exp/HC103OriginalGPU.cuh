/**
 * @file HC103OriginalGPU.cuh
 * @brief Independent frozen pre103 activation, publication and residual-read arithmetic.
 *
 * Original producer bodies and their numerical helpers remain test-owned before
 * any production extraction. Live rows use the original canonical descriptors;
 * graph/stream/storage ownership belongs to the existing native fixture.
 */
#pragma once
#include "HC103OriginalGPU.h"
#include <cstdint>
#include <cmath>
#include <stdexcept>
#if defined(HC103_CUDA)
#include <cuda_runtime.h>
#define HC103_RT(name) cuda##name
#define HC103_DOWN cudaOriginalDown
#define HC103_READ cudaOriginalRead
#define HC103_PUBLISH cudaOriginalPublish
#define HC103_NAMESPACE cuda_original
#define HC103_SHFL(x,o) __shfl_xor_sync(0xffffffffu,x,o,32)
#else
#include <hip/hip_runtime.h>
#define HC103_RT(name) hip##name
#define HC103_DOWN rocmOriginalDown
#define HC103_READ rocmOriginalRead
#define HC103_PUBLISH rocmOriginalPublish
#define HC103_NAMESPACE rocm_original
#define HC103_SHFL(x,o) __shfl_xor(x,o,32)
#endif
#include "HC103OriginalDeviceFP32NumericalContract.h"
#include "HC103OriginalDeviceHalfMetadataContract.h"
namespace llaminar2::test103 {
namespace HC103_NAMESPACE {
/** @brief Original nonfinite/live-state failure is fatal, never clamped. */
__device__ __forceinline__ void invalid() {
#if defined(HC103_CUDA)
 __trap();
#else
 __builtin_trap();
#endif
}
/** @brief Original primitive prefix authority; no host count readback. */
__device__ inline int liveRows(qwen4exp::RowExtent extent) {
 const int count=extent.liveRows?*extent.liveRows:extent.rows;
 if(count<0||count>extent.rows)invalid();return count;
}
/** @brief Original primitive expf program, without a substituted approximation. */
__device__ inline float sigmoid(float value){return 1.0f/(1.0f+expf(-value));}
/** @brief Frozen original residualDown body and256-lane launch bounds. */
__global__ __launch_bounds__(256,2) void downKernel(qwen4exp::ResidualDownArgs a) {
 const auto i=static_cast<std::size_t>(blockIdx.x)*256+threadIdx.x;
 if(i<static_cast<std::size_t>(liveRows(a.extent))*a.lowRank) {
  const float scaled=a.input[i]*0.25f;
  a.output[i]=scaled*sigmoid(scaled);
 }
}
/** @brief Frozen original ordered four-term FMA read. */
__global__ __launch_bounds__(256,2) void readKernel(qwen4exp::ResidualReadArgs a) {
 const auto i=static_cast<std::size_t>(blockIdx.x)*256+threadIdx.x;
 const auto hidden=static_cast<std::size_t>(a.hidden);
 if(i>=static_cast<std::size_t>(liveRows(a.extent))*hidden)return;
 const auto base=(i/hidden)*4*hidden+i%hidden;float sum=0.0f;
#pragma unroll
 for(int branch=0;branch<4;++branch) {
  const auto offset=base+branch*hidden;
  sum=fmaf(sigmoid(a.gateLogits[offset]),a.normalized[offset],sum);
 }
 a.output[i]=sum*0.25f;
}
/** @brief Independently frozen actual CUDA/HIP scale program. */
__device__ __forceinline__ float blockScale(float maximum) {
#if defined(HC103_CUDA)
 return maximum>0.f?maximum/127.f:1.f;
#else
 const bool nonzero=(test103_original_fp32::bitsFromFloat(maximum)&0x7fffffffu)!=0u;
 const float binary32_scale=nonzero?test103_original_fp32::multiply(maximum,0x1.020408p-7f):1.0f;
 const float finite_half_scale=fminf(65504.0f,fmaxf(0x1p-24f,binary32_scale));
 return test103_original_half::canonicalPreparedHalfValue(finite_half_scale);
#endif
}
/** @brief Original reciprocal/RNE/saturation order differs intentionally by backend. */
__device__ __forceinline__ int quantizedValue(float value,float scale) {
 const float inverse=1.f/scale;
#if defined(HC103_CUDA)
 const float scaled=value*inverse;
 return static_cast<int>(rintf(fminf(127.f,fmaxf(-127.f,scaled))));
#else
 const float rounded=rintf(value*inverse);
 const float bounded=fminf(127.f,fmaxf(-127.f,rounded));
 return static_cast<int32_t>(bounded);
#endif
}
/** @brief Frozen original XOR-max primary/residual publication body. */
__global__ __launch_bounds__(256,2) void publish(GPUQ8ResidualCorrectionArgs a) {
 const std::uint64_t task=static_cast<std::uint64_t>(blockIdx.x)*8+threadIdx.x/32;
 const auto bpr=a.weights.blocks_per_row;const int live=a.rows.activeRows();
 if(task>=static_cast<std::uint64_t>(live)*bpr)return;
 const unsigned lane=threadIdx.x%32;const auto offset=task*32+lane;
 const float value=a.input[offset];if(!isfinite(value))invalid();float maximum=fabsf(value);
 for(int distance=16;distance;distance/=2)maximum=fmaxf(maximum,HC103_SHFL(maximum,distance));
 const float stored=blockScale(maximum);const int primary=quantizedValue(value,stored);
 a.primary[offset]=static_cast<signed char>(primary);
 const float residual=test103_original_fp32::add(value,-test103_original_fp32::multiply(stored,static_cast<float>(primary)));
 if(!isfinite(residual))invalid();float residualMaximum=fabsf(residual);
 for(int distance=16;distance;distance/=2)residualMaximum=fmaxf(residualMaximum,HC103_SHFL(residualMaximum,distance));
 const float residualScale=blockScale(residualMaximum);
 a.residual[offset]=static_cast<signed char>(quantizedValue(residual,residualScale));
 if(lane==0){a.primaryScales[task]=stored;a.residualScales[task]=residualScale;}
}
/** @brief Reject default streams before any frozen-control launch. */
HC103_RT(Stream_t) stream(void *value) {
 if(!value||value==reinterpret_cast<void*>(HC103_RT(StreamLegacy))||value==reinterpret_cast<void*>(HC103_RT(StreamPerThread)))
  throw std::invalid_argument("HC103 oracle requires an exact non-default stream");
 return static_cast<HC103_RT(Stream_t)>(value);
}
/** @brief A native launch failure cannot silently certify the control. */
void checked(){const auto error=HC103_RT(GetLastError)();if(error!=HC103_RT(Success))throw std::runtime_error(HC103_RT(GetErrorString)(error));}
}
/** @brief Submit the independent original primitive on the exact producer stream. */
void HC103_DOWN(qwen4exp::ResidualDownArgs a,void *s){HC103_NAMESPACE::downKernel<<<unsigned((std::size_t(a.extent.rows)*a.lowRank+255)/256),256,0,HC103_NAMESPACE::stream(s)>>>(a);HC103_NAMESPACE::checked();}
/** @brief Submit the independent original final read on that same stream. */
void HC103_READ(qwen4exp::ResidualReadArgs a,void *s){HC103_NAMESPACE::readKernel<<<unsigned((std::size_t(a.extent.rows)*a.hidden+255)/256),256,0,HC103_NAMESPACE::stream(s)>>>(a);HC103_NAMESPACE::checked();}
/** @brief Submit the independent publisher without invoking any projection or helper. */
void HC103_PUBLISH(GPUQ8ResidualCorrectionArgs a,void *s){HC103_NAMESPACE::publish<<<unsigned((std::uint64_t(a.rows.physicalRows())*a.weights.blocks_per_row+7)/8),256,0,HC103_NAMESPACE::stream(s)>>>(a);HC103_NAMESPACE::checked();}
}
#undef HC103_RT
#undef HC103_DOWN
#undef HC103_READ
#undef HC103_PUBLISH
#undef HC103_NAMESPACE
#undef HC103_SHFL
