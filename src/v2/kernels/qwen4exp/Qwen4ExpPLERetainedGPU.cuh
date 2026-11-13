/**
 * @file Qwen4ExpPLERetainedGPU.cuh
 * @brief Capture-safe parallel PLE snapshots, live copies and exact suffix splices.
 *
 * A separate validation launch scans every accepted decision before any publication
 * kernel can run on the exact stream. A trap is fatal; no later write is permitted.
 * Each payload word has one writer, preserving integer and FP32 bits independent
 * of row grouping. Grid-stride loops bound launch geometry without capacity copies.
 */
#pragma once
#include "Qwen4ExpPLERetained.h"
namespace llaminar2::qwen4exp::PLE_RETAINED_GPU_NAMESPACE {
/** @brief Fail hard on corrupt device metadata without host observation. */
__device__ __forceinline__ void fail(){
#if defined(__HIPCC__)
 __builtin_trap();
#else
 asm volatile("trap;");
#endif
}
/** @brief Resolve an authoritative request-local live extent uniformly. */
__device__ __forceinline__ int live(DeviceRequestRowRanges rows,int r){return rows.activeRows(r);}
/** @brief Snapshot exact prior payload only for a nonempty transaction. */
__global__ __launch_bounds__(256) void snapshot(PLESnapshotPrior p){
 const int n=live(p.sourceRows,p.requestIndex);const auto total=std::size_t(2)+std::size_t(9)*p.channels;
 for(auto i=std::size_t(blockIdx.x)*256+threadIdx.x;i<total;i+=std::size_t(gridDim.x)*256)if(n){if(i<2)p.priorTokens[i]=p.acceptedTokens[i];else reinterpret_cast<std::uint32_t*>(p.priorConvolution)[i-2]=reinterpret_cast<const std::uint32_t*>(p.acceptedConvolution)[i-2];}
 if(blockIdx.x==0&&threadIdx.x==0){p.transactionLive[p.requestIndex]=n;*p.copiedBytes=n?total*4:0;}
}
/** @brief Retain only live token and normalized-key words, leaving holes untouched. */
__global__ __launch_bounds__(256) void retain(PLERetainCandidates p){
 const int n=live(p.transactionRows,p.requestIndex);const auto total=std::size_t(n)*(std::size_t(p.channels)+1);
 for(auto i=std::size_t(blockIdx.x)*256+threadIdx.x;i<total;i+=std::size_t(gridDim.x)*256){if(i<std::size_t(n))p.retainedTokens[i]=p.tokens[i];else reinterpret_cast<std::uint32_t*>(p.retainedNormalized)[i-n]=reinterpret_cast<const std::uint32_t*>(p.normalized)[i-n];}
 if(blockIdx.x==0&&threadIdx.x==0)*p.copiedBytes=total*4;
}
/** @brief Authenticate every decision before any request's accepted banks can change. */
__global__ __launch_bounds__(256) void validate(AcceptedRequestPrefixes p){for(int r=blockIdx.x*256+threadIdx.x;r<p.candidates().requests();r+=gridDim.x*256)if(p.resolveNative(r)<0)fail();}
/** @brief Restore original tails at zero acceptance or splice exact accepted suffixes. */
__global__ __launch_bounds__(256) void publish(PLEPublishAccepted p){
 if(!live(p.acceptance.candidates(),p.requestIndex))return;
 const int n=p.acceptance.resolveNative(p.requestIndex);if(n<0)fail();
 const auto total=std::size_t(2)+std::size_t(9)*p.channels;
 for(auto i=std::size_t(blockIdx.x)*256+threadIdx.x;i<total;i+=std::size_t(gridDim.x)*256){
  if(i<2){const int source=n-2+int(i);p.acceptedTokens[i]=source<0?p.priorTokens[source+2]:p.retainedTokens[source];}
  else {const int j=static_cast<int>(i-2);const int source=n-9+j/p.channels;const int column=j%p.channels;const auto *data=source<0?p.priorConvolution+std::size_t(source+9)*p.channels:p.retainedNormalized+std::size_t(source)*p.channels;reinterpret_cast<std::uint32_t*>(p.acceptedConvolution)[j]=reinterpret_cast<const std::uint32_t*>(data)[column];}
 }
}
}
