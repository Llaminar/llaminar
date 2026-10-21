/**
 * @file GPUQ8ResidualCorrection.cuh
 * @brief Captured CUDA/HIP two-term Q8 publication and native prepared-source dots.
 *
 * One publication launch forms two signed-byte activation banks. One projection
 * launch shares every native weight load across bounded real rows and both
 * terms. Independent increasing-K FP32 folds are added exactly once. All
 * pointers and physical geometry are immutable captured operands; live counts
 * remain on device and inactive input/output/scratch bytes are untouched.
 */
#pragma once
#if defined(Q8_RESIDUAL_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define Q8R_STREAM cudaStream_t
#define Q8R_FACTORY cudaQ8ResidualCorrectedProjection
#define Q8R_NAMESPACE cuda_q8_residual
#define Q8R_ERROR cudaGetLastError
#define Q8R_SUCCESS cudaSuccess
#define Q8R_ERROR_TEXT cudaGetErrorString
#define Q8R_SHFL(x,o) __shfl_xor_sync(0xffffffffu,x,o,32)
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/rocm/ops/ROCmSwiGLUDecodeEquivalentMath.h"
#define Q8R_STREAM hipStream_t
#define Q8R_FACTORY rocmQ8ResidualCorrectedProjection
#define Q8R_NAMESPACE rocm_q8_residual
#define Q8R_ERROR hipGetLastError
#define Q8R_SUCCESS hipSuccess
#define Q8R_ERROR_TEXT hipGetErrorString
#define Q8R_SHFL(x,o) __shfl_xor(x,o,32)
#endif
#include "Q8ResidualCorrection.h"
#include "DeviceFP32NumericalContract.h"
#include "DeviceNativeVNNIContributionContract.h"
#include "Q8ResidualGroupedGeometry.h"
#if defined(Q8_RESIDUAL_CUDA)
#include "kernels/cuda/gemm/CUDAQ8ResidualGrouped.cuh"
#else
#include "kernels/rocm/gemm/ROCmQ8ResidualGrouped.cuh"
#endif
#include <cstdint>
#include <climits>
#include <string>
namespace llaminar2 {
namespace Q8R_NAMESPACE {
/** @brief Native signed four-byte integer dot without any floating operand. */
__device__ __forceinline__ int dot4(int a,int b,int c) {
#if defined(Q8_RESIDUAL_CUDA)
 return __dp4a(a,b,c);
#else
 return __builtin_amdgcn_sdot4(a,b,c,false);
#endif
}
/** @brief Derive the exact standalone/grouped/prefill HC-up blockwise scale.
 * CUDA's production quantizer is inline in CUDAQuantisedGemmKernel_CUTLASS.cu
 * and exposes no device helper: its FP32 division is retained verbatim here.
 * ROCm directly borrows its source-owned scale helper; the native prepared API
 * fixture authenticates both policies against all actual production dispatches.
 */
__device__ __forceinline__ float blockScale(float maximum) {
#if defined(Q8_RESIDUAL_CUDA)
 return maximum>0.f?maximum/127.f:1.f;
#else
 return rocm::decode_equivalent::q8Scale(maximum);
#endif
}
/** @brief Preserve the backend's actual reciprocal/multiply/RNE/saturation program. */
__device__ __forceinline__ int quantizedValue(float value,float scale) {
 // The actual HC-up blockwise producers use this native reciprocal on both
 // backends. ROCm quantizeQ8FromScale belongs to the heterogeneous MoE path
 // and intentionally derives a different reciprocal; it is not this contract.
 const float inverse=1.f/scale;
#if defined(Q8_RESIDUAL_CUDA)
 const float scaled=value*inverse;
 return static_cast<int>(rintf(fminf(127.f,fmaxf(-127.f,scaled))));
#else
 return rocm::decode_equivalent::quantizeQ8(value,inverse);
#endif
}
/** @brief Fatal nonfinite activation detection never substitutes another precision. */
__device__ __forceinline__ void invalidActivation() {
#if defined(Q8_RESIDUAL_CUDA)
 __trap();
#else
 __builtin_trap();
#endif
}
/** @brief Quantize a primary block and its exact rounded representation residual. */
__global__ __launch_bounds__(256,2) void publish(GPUQ8ResidualCorrectionArgs a) {
 const std::uint64_t task=static_cast<std::uint64_t>(blockIdx.x)*8+threadIdx.x/32;
 const auto bpr=a.weights.blocks_per_row;
 const int live=a.rows.activeRows();
 if(task>=static_cast<std::uint64_t>(live)*bpr)return;
 const unsigned lane=threadIdx.x%32;const auto offset=task*32+lane;
 const float value=a.input[offset];if(!isfinite(value))invalidActivation();float maximum=fabsf(value);
 for(int distance=16;distance;distance/=2)maximum=fmaxf(maximum,Q8R_SHFL(maximum,distance));
 const float stored=blockScale(maximum);
 const int primary=quantizedValue(value,stored);
 a.primary[offset]=static_cast<signed char>(primary);
 const float residual=device_fp32_contract::add(value,-device_fp32_contract::multiply(stored,static_cast<float>(primary)));
 if(!isfinite(residual))invalidActivation();
 float residualMaximum=fabsf(residual);
 for(int distance=16;distance;distance/=2)residualMaximum=fmaxf(residualMaximum,Q8R_SHFL(residualMaximum,distance));
 const float residualScale=blockScale(residualMaximum);
 a.residual[offset]=static_cast<signed char>(quantizedValue(residual,residualScale));
 if(lane==0){a.primaryScales[task]=stored;a.residualScales[task]=residualScale;}
}
/** @brief Bounded real-row reuse with independent Q8 terms and one final output store.
 * @tparam Rows Fixed maximum real rows per lane; never changes a row's K order.
 */
template<int Rows>__global__ __launch_bounds__(128,2) void project(GPUQ8ResidualCorrectionArgs a) {
 const auto columns=(static_cast<unsigned>(a.weights.n)+127)/128;
 const auto rowBase=static_cast<std::uint64_t>(blockIdx.x/columns)*Rows;
 const unsigned column=(blockIdx.x%columns)*128+threadIdx.x;
 const int live=a.rows.activeRows();
 if(column>=static_cast<unsigned>(a.weights.n)||rowBase>=static_cast<unsigned>(live))return;
 float primary[Rows]{},residual[Rows]{};
 for(unsigned block=0;block<a.weights.blocks_per_row;++block) {
  const auto linear=static_cast<std::uint64_t>(block)*a.weights.n+column;
  const auto*payload=reinterpret_cast<const int4*>(a.weights.payload+linear*32);
  const int4 low=payload[0],high=payload[1];const int weights[8]{low.x,low.y,low.z,low.w,high.x,high.y,high.z,high.w};
  const auto scaleBits=static_cast<const unsigned short*>(a.weights.scales)[linear];
  const float weightScale=__half2float(__ushort_as_half(scaleBits));
#pragma unroll
  for(int r=0;r<Rows;++r) {
   const auto row=rowBase+r;if(row>=static_cast<unsigned>(live))break;
   const auto activationBlock=row*a.weights.blocks_per_row+block;
   const auto*p0=reinterpret_cast<const int*>(a.primary+activationBlock*32);
   const auto*p1=reinterpret_cast<const int*>(a.residual+activationBlock*32);
   int dot0=0,dot1=0;
#pragma unroll
   for(int group=0;group<8;++group){dot0=dot4(p0[group],weights[group],dot0);dot1=dot4(p1[group],weights[group],dot1);}
   primary[r]=device_fp32_contract::add(primary[r],device_native_vnni_contract::singleScaleBlock(dot0,weightScale,a.primaryScales[activationBlock]));
   residual[r]=device_fp32_contract::add(residual[r],device_native_vnni_contract::singleScaleBlock(dot1,weightScale,a.residualScales[activationBlock]));
  }
 }
#pragma unroll
 for(int r=0;r<Rows;++r)if(rowBase+r<static_cast<unsigned>(live))
  a.output[(rowBase+r)*a.weights.n+column]=device_fp32_contract::add(primary[r],residual[r]);
}
/** @brief Validate only static prepared identities and admitted pointer geometry. */
inline Q8R_STREAM validate(const GPUQ8ResidualCorrectionArgs&a,void*stream) {
 validateQ8ResidualCorrectionGeometry(a.rows.physicalRows(),a.weights.n,a.weights.k);
 const auto native=static_cast<Q8R_STREAM>(stream);
#if defined(Q8_RESIDUAL_CUDA)
 const bool exact=native&&native!=cudaStreamLegacy&&native!=cudaStreamPerThread;
 const bool quantizer=a.quantizer==Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal;
#else
 const bool exact=native&&native!=hipStreamLegacy&&native!=hipStreamPerThread;
 const bool quantizer=a.quantizer==Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal;
#endif
 if(!exact||!quantizer||!a.weights.valid()||a.weights.codebook_id!=native_vnni_formats::Q8_0.codebook_id||!a.weights.source_identity_present||
    a.weights.source_codebook_id!=native_vnni_formats::Q8_0.codebook_id||a.weights.source_is_superblock||a.weights.mins||a.weights.emins||
    (a.weights.allocation_payload_bytes_per_block&&a.weights.allocation_payload_bytes_per_block!=native_vnni_formats::Q8_0.payload_bytes)||
    a.weights.allocation_has_mins||a.weights.allocation_has_emins||
    a.weights.blocks_per_row!=static_cast<unsigned>(a.weights.k/32)||!a.input||!a.output||a.input==a.output||
    !a.primary||!a.residual||a.primary==a.residual||!a.primaryScales||!a.residualScales||a.primaryScales==a.residualScales||
    reinterpret_cast<std::uintptr_t>(a.weights.payload)%16||reinterpret_cast<std::uintptr_t>(a.primary)%16||
    reinterpret_cast<std::uintptr_t>(a.residual)%16)
  throw std::invalid_argument("Two-term GPU Q8 requires exact non-default stream, native Q8_0 prepared identity and distinct aligned admitted banks");
 const auto values=static_cast<std::size_t>(a.rows.physicalRows())*a.weights.k;
 const auto outputValues=static_cast<std::size_t>(a.rows.physicalRows())*a.weights.n;
 const auto weightBlocks=static_cast<std::size_t>(a.weights.n)*a.weights.blocks_per_row;
 const std::array writes{
  Q8ResidualStorageSpan{a.output,outputValues*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.primary,values,16},Q8ResidualStorageSpan{a.residual,values,16},
  Q8ResidualStorageSpan{a.primaryScales,values/32*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.residualScales,values/32*sizeof(float),alignof(float)}};
 const std::array reads{
  Q8ResidualStorageSpan{a.input,values*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.weights.payload,weightBlocks*32,16},
  Q8ResidualStorageSpan{a.weights.scales,weightBlocks*sizeof(std::uint16_t),alignof(std::uint16_t)},
  Q8ResidualStorageSpan{a.rows.countOwner(),a.rows.countOwner()?sizeof(std::int32_t):0,alignof(std::int32_t)}};
 validateQ8ResidualStorage(writes,reads);
 return native;
}
}
/** @brief Submit complete paired-Q8 arithmetic; no setup, allocation or host live-state observation. */
void Q8R_FACTORY(const GPUQ8ResidualCorrectionArgs&a,void*stream) {
 const auto native=Q8R_NAMESPACE::validate(a,stream);
 const auto tasks=static_cast<std::uint64_t>(a.rows.physicalRows())*a.weights.blocks_per_row;
#if defined(Q8_RESIDUAL_CUDA)
 const auto geometry=q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::CUDA,a.rows.physicalRows(),a.weights.n);
#else
 const auto geometry=q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::ROCm,a.rows.physicalRows(),a.weights.n);
#endif
 const auto grid=geometry.blocks;
 if((tasks+7)/8>INT_MAX||grid>INT_MAX)throw std::invalid_argument("Two-term GPU Q8 retained launch grid exceeds native capacity");
 Q8R_NAMESPACE::publish<<<static_cast<unsigned>((tasks+7)/8),256,0,native>>>(a);
 if(geometry.strategy==Q8ResidualGroupedStrategy::CooperativeMatrix)
  Q8R_NAMESPACE::projectGrouped<<<static_cast<unsigned>(grid),256,0,native>>>(a);
 else if(geometry.rowTile==1)Q8R_NAMESPACE::project<1><<<static_cast<unsigned>(grid),128,0,native>>>(a);
 else Q8R_NAMESPACE::project<4><<<static_cast<unsigned>(grid),128,0,native>>>(a);
 const auto error=Q8R_ERROR();if(error!=Q8R_SUCCESS)throw std::runtime_error(std::string("Two-term Q8 native launch failed: ")+Q8R_ERROR_TEXT(error));
}
}
#undef Q8R_STREAM
#undef Q8R_FACTORY
#undef Q8R_NAMESPACE
#undef Q8R_ERROR
#undef Q8R_SUCCESS
#undef Q8R_ERROR_TEXT
#undef Q8R_SHFL
