/**
 * @file GPUPooledImmutableLocalHotKernels.cuh
 * @brief Symmetric captured local-cache native mechanics with the original stable radix and row decoder.
 *
 * Probe/classification is independent of lower service. The sole LocalRead
 * produces model bytes after mapped ready1; maintenance consumes only after
 * actual read-complete2. All native copies use live complete source bytes.
 */
#pragma once
#if defined(QFN_LOCAL_HOT_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#define QFN_LH_STREAM cudaStream_t
#define QFN_LH_NAMESPACE local_hot_cuda
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define QFN_LH_STREAM hipStream_t
#define QFN_LH_NAMESPACE local_hot_rocm
#endif
#include "PooledImmutableLocalHotKernels.h"
#include "GPUImmutableStableRadix.cuh"
#include "PooledImmutableLocalHotReadyProtocol.h"
#include "ImmutableLookupDecode.h"
#include "AdaptiveImmutableRowCacheControlBorrow.h"
#include "kernels/common/DeviceGenerationControlLayout.h"
#include "kernels/common/SamplingMath.h"
#include <algorithm>
#include <string>
#include <type_traits>
namespace llaminar2::qwen4exp::QFN_LH_NAMESPACE {
namespace h=local_hot;
namespace a=adaptive_lookup;
namespace s=immutable_stable_radix;
using Action=PooledImmutableLocalHotAction;
inline constexpr unsigned Threads=256,Small=512,Bins=16;
/** @brief Original system-release primitive on the TransferEngine-bound local word. */
__device__ __forceinline__ void release(std::uint64_t* address,std::uint64_t value)
{
#if defined(QFN_LOCAL_HOT_CUDA)
    cuda::atomic_ref<std::uint64_t,cuda::thread_scope_system>(*address).store(value,cuda::memory_order_release);
#else
    __hip_atomic_store(address,value,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @return Original system-acquire snapshot; no host mirror or polling loop exists. */
__device__ __forceinline__ std::uint64_t acquireReady(std::uint64_t* address)
{
#if defined(QFN_LOCAL_HOT_CUDA)
    return cuda::atomic_ref<std::uint64_t,cuda::thread_scope_system>(*address).load(cuda::memory_order_acquire);
#else
    return __hip_atomic_load(address,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @return One original compare/exchange; ordinary publishers cannot overwrite absorbing Abort. */
__device__ __forceinline__ bool compareReady(std::uint64_t* address,std::uint64_t expected,std::uint64_t desired)
{
#if defined(QFN_LOCAL_HOT_CUDA)
    return cuda::atomic_ref<std::uint64_t,cuda::thread_scope_system>(*address).compare_exchange_strong(
        expected,desired,cuda::memory_order_acq_rel,cuda::memory_order_acquire);
#else
    return __hip_atomic_compare_exchange_strong(address,&expected,desired,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @return Original error-free forward state, including prefill before response admission. */
__device__ __forceinline__ bool healthy(const h::NativeFrame& n)
{
    if (!n.healthy || !n.error) return false;
    const int error = atomicAdd(n.error, 0);
    const int ok = atomicAdd(n.healthy, 0);
    return h::validForwardController(ok, error);
}
/** @brief Preserve the original first error and unblock the original mapped native wait. */
__device__ __forceinline__ void failure(const h::NativeFrame& n,a::Status status)
{
    auto* frame=h::at<a::FrameControl>(n.frame,n.plan.frame.control);
    atomicCAS(reinterpret_cast<unsigned*>(&frame->status),unsigned(a::Status::Success),unsigned(status));
    atomicCAS(n.error,0,int(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    atomicExch(n.healthy,0);release(n.ready,UINT64_MAX);
}
/** @return One bound source hash, with original UInt32 zero extension. */
__device__ __forceinline__ std::uint64_t original(const h::NativeQuery& n,std::uint32_t i)
{
    return n.id_width==pooled_native::OriginalIdWidth::UInt32
        ?static_cast<const std::uint32_t*>(n.ids)[i]:static_cast<const std::uint64_t*>(n.ids)[i];
}
/** @return Whether the actual native probe still owns its complete coalescing interval. */
__device__ __forceinline__ bool probing(const h::NativeFrame& n)
{
    return healthy(n) && h::at<h::FrameIdentity>(n.frame,n.plan.frame.identity)->disposition==h::Disposition::Probing &&
        h::at<a::FrameControl>(n.frame,n.plan.frame.control)->status==a::Status::Success;
}
/** @brief Uniform CTA admission around cooperative barriers, including concurrent original cancellation. */
__device__ __forceinline__ bool blockProbing(const h::NativeFrame& n)
{
    __shared__ unsigned active;
    if(threadIdx.x==0)active=probing(n)?1u:0u;
    __syncthreads();return active!=0;
}
/** @brief Uniform CTA health admission before any cooperating copy lanes. */
__device__ __forceinline__ bool blockHealthy(const h::NativeFrame& n)
{
    __shared__ unsigned active;
    if(threadIdx.x==0)active=healthy(n)?1u:0u;
    __syncthreads();return active!=0;
}
/** @brief Cold virgin array initialization; payload bytes are deliberately untouched. */
template<class Slot> static __global__ __launch_bounds__(Threads,2) void initializeArrays(h::NativeRoots n)
{
    auto v=h::view<Slot>(n);a::Operations<Slot> op(v.metadata);
    for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n.plan.slots;i+=std::uint64_t(gridDim.x)*blockDim.x){
        op.initializeSlot(unsigned(i));op.initializeNode(unsigned(i));
    }
    for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<std::uint64_t(4)*n.plan.lane_width;i+=std::uint64_t(gridDim.x)*blockDim.x){
        v.metadata.frequency_epochs[i]=0;v.metadata.frequency_counts[i]=0;
    }
}
/** @brief Only cold initialization resets model-lifetime ordinal and transition counters. */
template<class Slot> static __global__ void initializeControls(h::NativeRoots n)
{
    auto v=h::view<Slot>(n);a::Operations<Slot> op(v.metadata);op.initializeOwner();op.initializeFrame();
    *v.identity=h::FrameIdentity{};*v.counters=h::Counters{};
    *h::at<std::uint64_t>(n.frame,n.plan.frame.received_header_bytes)=0;
    *h::at<std::uint64_t>(n.frame,n.plan.frame.received_payload_bytes)=0;release(n.ready,0);
}
/** @brief Only the original new-request edge may clear a retired normal-cancellation frontier. */
template<class Slot> static __global__ void resetRequest(h::NativeRequestReset n)
{
    auto v=h::view<Slot>(n);
    if(!healthy(n) || !n.admission || !h::Operations<Slot>(v).resetRequest(*n.admission)){
        failure(n,v.metadata.frame->status==a::Status::Success?a::Status::InvalidPhase:v.metadata.frame->status);
        return;
    }
    release(n.ready,0);
}
/** @brief Start the exact next local ordinal before clearing the previous retired ready word. */
template<class Slot> __device__ __forceinline__ void openBody(h::NativeQuery n)
{
    if(!healthy(n)){release(n.ready,UINT64_MAX);return;}
    auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
    const auto count=n.original_ids.activeRows();
    if(count<0 || !op.begin(n.intent,unsigned(count))){failure(n,v.metadata.frame->status);return;}
    *h::at<std::uint64_t>(n.frame,n.plan.frame.received_header_bytes)=0;
    *h::at<std::uint64_t>(n.frame,n.plan.frame.received_payload_bytes)=0;
    if(!h::resetReadyForProbe([&]{return acquireReady(n.ready);},
        [&](std::uint64_t previous,std::uint64_t value){return compareReady(n.ready,previous,value);}))
        failure(n,a::Status::InvalidPhase);
}
/** @brief The large-query bundle retains its original single-writer opening launch. */
template<class Slot> static __global__ void open(h::NativeQuery n) { openBody<Slot>(n); }
/** @brief Open from the exact shared next-window resolver; malformed publication is absorbing failure. */
template<class Slot> __device__ __forceinline__ void openNextWindowBody(h::NativeNextWindowQuery n)
{
    if(!healthy(n)){release(n.ready,UINT64_MAX);return;}
    auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
    const auto count=n.window.originalIds();
    if(count<0 || std::uint32_t(count)>n.plan.maximum_original){failure(n,a::Status::InvalidGeometry);return;}
    if(!op.begin(n.intent,unsigned(count))){failure(n,v.metadata.frame->status);return;}
    *h::at<std::uint64_t>(n.frame,n.plan.frame.received_header_bytes)=0;
    *h::at<std::uint64_t>(n.frame,n.plan.frame.received_payload_bytes)=0;
    if(!h::resetReadyForProbe([&]{return acquireReady(n.ready);},
        [&](std::uint64_t previous,std::uint64_t value){return compareReady(n.ready,previous,value);}))
        failure(n,a::Status::InvalidPhase);
}
/** @brief Large next-window queries retain the same checked opening boundary. */
template<class Slot> static __global__ void openNextWindow(h::NativeNextWindowQuery n) { openNextWindowBody<Slot>(n); }
/** @return Stable first source occurrence in the actual sorted original query. */
__device__ __forceinline__ unsigned lowerBound(const a::KeyIndex* sorted,unsigned count,std::uint64_t row)
{
    unsigned begin=0,end=count;
    while(begin<end){const auto middle=begin+(end-begin)/2;if(sorted[middle].row<row)begin=middle+1;else end=middle;}
    return begin;
}
/** @brief One small cooperative original row sort; stable first-coordinate scan avoids a second sort. */
template<class Slot> __device__ __forceinline__ void coalesceSmallBody(h::NativeQuery n)
{
    __shared__ a::KeyIndex one[Small],two[Small];
    __shared__ unsigned histogram[Bins],cursor[Bins],warps[8*Bins],prefix[Threads],carry;
    if(!blockProbing(n))return;
    auto v=h::view<Slot>(n);const unsigned count=unsigned(v.metadata.frame->original_ids);
    auto* scan=h::at<unsigned>(n.scratch,n.plan.scratch.scan);
    for(auto i=threadIdx.x;i<count;i+=Threads){
        const auto row=original(n,i);if(row>=n.plan.source_rows)failure(n,a::Status::InvalidGeometry);
        one[i]={row,i,0};scan[i]=0;
    }
    __syncthreads();if(!blockProbing(n))return;
    s::radixSmall<false>(one,two,count,histogram,cursor,warps,pooled_native::radixPasses(n.plan.source_rows));
    for(auto i=threadIdx.x;i<count;i+=Threads){
        h::at<a::KeyIndex>(n.scratch,n.plan.scratch.keys_a)[i]=one[i];
        if(!i || one[i].row!=one[i-1].row)scan[one[i].original_coordinate]=1;
    }
    __syncthreads();if(threadIdx.x==0)carry=0;__syncthreads();
    for(unsigned first=0;first<count;first+=Threads){
        const auto i=first+threadIdx.x;const auto value=s::scan(i<count?scan[i]:0,prefix);
        if(i<count)scan[i]=carry+value;
        __syncthreads();if(threadIdx.x==0)carry+=prefix[Threads-1];__syncthreads();
    }
    for(auto i=threadIdx.x;i<count;i+=Threads){
        const auto first=one[lowerBound(one,count,one[i].row)].original_coordinate;
        const auto unique=scan[first]-1;v.original_to_unique[one[i].original_coordinate]=unique;
        if(first==one[i].original_coordinate)v.unique[unique]={one[i].row,0,first,a::None,a::None,0};
    }
    __syncthreads();if(threadIdx.x==0 && !h::Operations<Slot>(v).coalesced(carry))failure(n,v.metadata.frame->status);
}
/** @brief Large-shape parallel input initialization reads only the real original prefix. */
static __global__ __launch_bounds__(Threads,2) void initializeKeys(h::NativeQuery n)
{
    if(!probing(n))return;const auto count=h::at<a::FrameControl>(n.frame,n.plan.frame.control)->original_ids;
    auto* out=h::at<a::KeyIndex>(n.scratch,n.plan.scratch.keys_a);
    for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::uint64_t(gridDim.x)*blockDim.x){
        const auto row=original(n,unsigned(i));if(row>=n.plan.source_rows)failure(n,a::Status::InvalidGeometry);
        out[i]={row,unsigned(i),0};
    }
}
/** @brief Stable native partition histograms, sharing the original pooled vote rank. */
static __global__ __launch_bounds__(Threads,2) void radixCount(h::NativeFrame n,unsigned pass,bool from_a)
{
    if(!blockProbing(n))return;const auto count=h::at<a::FrameControl>(n.frame,n.plan.frame.control)->original_ids;
    if(std::uint64_t(blockIdx.x)*Threads>=count)return;
    __shared__ unsigned warps[8*Bins];
    const auto* input=h::at<a::KeyIndex>(n.scratch,from_a?n.plan.scratch.keys_a:n.plan.scratch.keys_b);
    const auto i=std::uint64_t(blockIdx.x)*Threads+threadIdx.x;
    const unsigned bin=i<count?unsigned((input[i].row>>(4*pass))&15):0;
    s::bucketRank(i<count,bin,warps);
    if(threadIdx.x<Bins){unsigned total=0;for(unsigned warp=0;warp<Threads/s::warpWidth();++warp)total+=warps[warp*Bins+threadIdx.x];
        h::at<unsigned>(n.scratch,n.plan.scratch.radix_counts)[blockIdx.x*Bins+threadIdx.x]=total;}
}
/** @brief Deterministic global partition offsets use only actual live partitions. */
static __global__ void radixOffsets(h::NativeFrame n)
{
    if(!probing(n) || threadIdx.x>=Bins)return;
    const auto count=h::at<a::FrameControl>(n.frame,n.plan.frame.control)->original_ids;
    const unsigned blocks=unsigned((count+Threads-1)/Threads),bin=threadIdx.x;
    const auto* hist=h::at<unsigned>(n.scratch,n.plan.scratch.radix_counts);auto* offsets=h::at<unsigned>(n.scratch,n.plan.scratch.radix_offsets);
    unsigned base=0;for(unsigned prior=0;prior<bin;++prior)for(unsigned b=0;b<blocks;++b)base+=hist[b*Bins+prior];
    for(unsigned b=0;b<blocks;++b){offsets[b*Bins+bin]=base;base+=hist[b*Bins+bin];}
}
/** @brief Parallel stable scatter retains original position in each exact row-key record. */
static __global__ __launch_bounds__(Threads,2) void radixScatter(h::NativeFrame n,unsigned pass,bool from_a)
{
    if(!blockProbing(n))return;const auto count=h::at<a::FrameControl>(n.frame,n.plan.frame.control)->original_ids;
    if(std::uint64_t(blockIdx.x)*Threads>=count)return;
    __shared__ unsigned warps[8*Bins];
    const auto* input=h::at<a::KeyIndex>(n.scratch,from_a?n.plan.scratch.keys_a:n.plan.scratch.keys_b);
    auto* output=h::at<a::KeyIndex>(n.scratch,from_a?n.plan.scratch.keys_b:n.plan.scratch.keys_a);
    const auto i=std::uint64_t(blockIdx.x)*Threads+threadIdx.x;const auto value=i<count?input[i]:a::KeyIndex{};
    const auto bin=unsigned((value.row>>(4*pass))&15);const auto rank=s::bucketRank(i<count,bin,warps);
    if(i<count)output[h::at<unsigned>(n.scratch,n.plan.scratch.radix_offsets)[blockIdx.x*Bins+bin]+rank]=value;
}
/** @brief Stable first-original flags; the canonical row-sorted bank remains read-only until retirement. */
static __global__ __launch_bounds__(Threads,2) void firstPositions(h::NativeQuery n)
{
    if(!probing(n))return;const auto count=unsigned(h::at<a::FrameControl>(n.frame,n.plan.frame.control)->original_ids);
    const auto* sorted=h::at<a::KeyIndex>(n.scratch,n.plan.scratch.keys_a);auto* inverse=h::at<unsigned>(n.frame,n.plan.frame.original_to_unique);
    auto* scan=h::at<unsigned>(n.scratch,n.plan.scratch.scan);
    for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::uint64_t(gridDim.x)*blockDim.x){
        const auto first=sorted[lowerBound(sorted,count,sorted[i].row)].original_coordinate;
        inverse[sorted[i].original_coordinate]=first;scan[sorted[i].original_coordinate]=first==sorted[i].original_coordinate?1u:0u;
    }
}
/** @brief Cooperative original-coordinate prefix sums use exact live chunks. */
static __global__ __launch_bounds__(Threads,2) void scanChunks(h::NativeFrame n)
{
    if(!blockProbing(n))return;const auto count=h::at<a::FrameControl>(n.frame,n.plan.frame.control)->original_ids;
    const auto i=std::uint64_t(blockIdx.x)*Threads+threadIdx.x;if(std::uint64_t(blockIdx.x)*Threads>=count)return;
    __shared__ unsigned prefix[Threads];auto* scan=h::at<unsigned>(n.scratch,n.plan.scratch.scan);
    const auto value=s::scan(i<count?scan[i]:0,prefix);if(i<count)scan[i]=value;
    if(threadIdx.x==0)h::at<unsigned>(n.scratch,n.plan.scratch.radix_counts)[blockIdx.x]=prefix[Threads-1];
}
/** @brief One bounded chunk-total scan, rather than a serial original-key loop. */
template<class Slot> static __global__ void scanOffsets(h::NativeFrame n)
{
    if(!probing(n))return;auto v=h::view<Slot>(n);const auto blocks=unsigned((v.metadata.frame->original_ids+Threads-1)/Threads);
    const auto* sums=h::at<unsigned>(n.scratch,n.plan.scratch.radix_counts);auto* offsets=h::at<unsigned>(n.scratch,n.plan.scratch.radix_offsets);
    unsigned total=0;for(unsigned i=0;i<blocks;++i){offsets[i]=total;total+=sums[i];}
    if(!h::Operations<Slot>(v).coalesced(total))failure(n,v.metadata.frame->status);
}
/** @brief Resolve each original inverse and unique row without another key sort. */
template<class Slot> static __global__ __launch_bounds__(Threads,2) void resolveOriginal(h::NativeQuery n)
{
    if(!probing(n))return;auto v=h::view<Slot>(n);const auto count=v.metadata.frame->original_ids;
    const auto* scan=h::at<unsigned>(n.scratch,n.plan.scratch.scan);const auto* offsets=h::at<unsigned>(n.scratch,n.plan.scratch.radix_offsets);
    for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::uint64_t(gridDim.x)*blockDim.x){
        const auto first=v.original_to_unique[i];const auto unique=scan[first]+offsets[first/Threads]-1;
        v.original_to_unique[i]=unique;if(first==i)v.unique[unique]={original(n,unsigned(i)),0,unsigned(i),a::None,a::None,0};
    }
}
/** @brief Independent genuine initial-hit probes, before the one ordered pin frontier. */
template<class Slot> static __global__ __launch_bounds__(Threads,2) void probeUnique(h::NativeFrame n)
{
    if(!probing(n))return;auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
    for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<v.metadata.frame->unique_rows;i+=std::uint64_t(gridDim.x)*blockDim.x)
        if(!op.probeUnique(unsigned(i)))failure(n,a::Status::Invariant);
}
/** @brief Publish immediate all-hit data-ready; real misses are resolved only by AcceptResult. */
template<class Slot> __device__ __forceinline__ void sealProbeBody(h::NativeFrame n)
{
    auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
    if(!healthy(n) || v.metadata.frame->status!=a::Status::Success){
        if(v.identity->disposition!=h::Disposition::Idle)op.cancel(v.identity->query,v.metadata.frame->status==a::Status::Success?a::Status::Cancelled:v.metadata.frame->status);
        release(n.ready,UINT64_MAX);return;
    }
    if(!op.pinAndCompact()){failure(n,v.metadata.frame->status);return;}
    if(!v.metadata.frame->source_rows)compareReady(n.ready,0,1);
}
/** @brief The large-query bundle preserves its original ordered final publication. */
template<class Slot> static __global__ void sealProbe(h::NativeFrame n) { sealProbeBody<Slot>(n); }
/**
 * @brief One captured CTA opens, coalesces, probes and publishes a bounded small query.
 * @tparam Slot Original native row-key representation.
 * @tparam Query Current demand or borrowed next-prefill window, with the same frame.
 * @param n Immutable original query declaration; the live count remains device-owned.
 * @note The exact original radix/first-coordinate and ordered pin code is shared.
 * CTA barriers replace only the three internal kernel boundaries. No transfer,
 * external publication dependency or cache admission decision is removed.
 */
template<class Slot,class Query> static __global__ __launch_bounds__(Threads,2) void fusedSmallProbe(Query n)
{
    if(threadIdx.x==0) {
        if constexpr(std::is_same_v<Query,h::NativeNextWindowQuery>) openNextWindowBody<Slot>(n);
        else openBody<Slot>(n);
    }
    __syncthreads();
    coalesceSmallBody<Slot>(n);
    __syncthreads();
    if(probing(n)) {
        auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
        for(unsigned index=threadIdx.x;index<v.metadata.frame->unique_rows;index+=Threads)
            if(!op.probeUnique(index))failure(n,a::Status::Invariant);
    }
    __syncthreads();
    // Run even after an invalid opening/coalescing step: the existing seal
    // cancels the frame and publishes Abort through its exact original owner.
    if(threadIdx.x==0)sealProbeBody<Slot>(n);
}
/** @brief Accept only actual native compact bytes; an all-hit late ACK never rewinds ready2 to1. */
template<class Slot> static __global__ void accept(h::NativeFrame n,const local_hot_transport::RequestHeader* own_request)
{
    auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
    if(!healthy(n)){if(v.identity->disposition!=h::Disposition::Idle)op.cancel(v.identity->query);release(n.ready,UINT64_MAX);return;}
    if(!v.metadata.frame->source_rows)return;
    if(!own_request || !op.acceptNativeResult(*own_request,n.consumer,h::result(n))){failure(n,v.metadata.frame->status);return;}
    compareReady(n.ready,0,1);
}
/** @brief Validate every private native input before the sole model output writer can run. */
template<class Slot> static __global__ __launch_bounds__(Threads,2) void validateRead(h::NativeRead n)
{
    if(!healthy(n))return;auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
    if(v.identity->intent!=PooledImmutableLocalHotAccessIntent::CurrentInvocation ||
        v.metadata.frame->original_ids>n.physical_original){failure(n,a::Status::InvalidPhase);return;}
    for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<v.metadata.frame->unique_rows;i+=std::uint64_t(gridDim.x)*blockDim.x)
        if(!op.readableUnique(unsigned(i)))failure(n,a::Status::Invariant);
}
/** @brief One producer writes every actual original model row using the unchanged source-native decoder. */
template<TensorType Format,class Slot> static __global__ __launch_bounds__(Threads,2) void read(h::NativeRead n)
{
    if(!healthy(n))return;auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
    const auto count=v.metadata.frame->original_ids*std::uint64_t(n.plan.row_elements);
    for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::uint64_t(gridDim.x)*blockDim.x){
        const auto* row=op.readableOriginal(unsigned(i/n.plan.row_elements));
        if(row)n.output[i]=lookup_detail::decode<Format>(row,i%n.plan.row_elements,n.tables);
    }
}
/** @brief Only completion after the actual read kernel publishes native frontier2. */
template<class Slot> static __global__ void completeRead(h::NativeFrame n)
{
    auto v=h::view<Slot>(n);if(!healthy(n)){release(n.ready,UINT64_MAX);return;}
    if(!h::Operations<Slot>(v).completeRead(v.identity->query)){failure(n,v.metadata.frame->status);return;}
    compareReady(n.ready,1,2);
}
/** @brief Original ordered policy writer runs only after the actual main read/consumer edge. */
template<class Slot> static __global__ void frameAction(h::NativeFrame n,Action action)
{
    auto v=h::view<Slot>(n);h::Operations<Slot> op(v);const auto query=v.identity->query;
    if(action==Action::Retire){
        if(v.identity->disposition==h::Disposition::Idle){if(!healthy(n))return;failure(n,a::Status::InvalidPhase);return;}
        if(!healthy(n) && v.identity->disposition!=h::Disposition::Cancelled)op.cancel(query);
        const auto result=op.retire(query);
        if(result.physical!=decltype(result.physical)::Retired)failure(n,result.status);
        return;
    }
    if(action==Action::Cancel || !healthy(n)){
        if(v.identity->disposition!=h::Disposition::Idle)op.cancel(query);
        release(n.ready,UINT64_MAX);return;
    }
    bool ok=false;
    switch(action){
    case Action::ConsumeDemand:ok=op.consumeDemand(query);break;
    case Action::CompleteFills:ok=op.completeFills();break;
    case Action::Discard:ok=op.discard(query);break;
    default:failure(n,a::Status::InvalidPhase);return;
    }
    if(!ok)failure(n,v.metadata.frame->status);
}
/**
 * @brief Compare the full actual ordered demand cooperatively without new persistent storage.
 * One CTA performs bounded strided comparisons and an order-independent integer
 * OR. Original identity failures poison; a normal prediction miss only disposes
 * this frame. No output, source lookup, sort or policy operation occurs here.
 */
template<class Slot> static __global__ __launch_bounds__(Threads,2) void adopt(h::NativeAdoption n)
{
    __shared__ unsigned check, comparison;
    auto v = h::view<Slot>(n);
    h::Operations<Slot> op(v);
    const auto count = n.original_ids.activeRows();
    const h::DemandIdentity identity{v.identity->query, n.admission ? n.admission->source_invocation : 0};
    if (threadIdx.x == 0)
    {
        comparison = 0;
        if (!healthy(n)) check = unsigned(h::AdoptionResult::InvalidState);
        else if (!n.admission ||
            !h::t::validSourceAdmission(*n.admission, n.plan.table, n.plan.source_generation) || count < 0)
        {
            op.fatal(a::Status::Invariant);
            check = unsigned(h::AdoptionResult::InvalidIdentity);
        }
        else check = unsigned(op.checkAdoption(identity, unsigned(count)));
    }
    __syncthreads();
    if (check == unsigned(h::AdoptionResult::Accepted))
    {
        unsigned local = 0;
        for (unsigned i = threadIdx.x; i < unsigned(count); i += blockDim.x)
            local |= unsigned(op.compareDemandOriginal(i, original(n, i)));
        atomicOr(&comparison, local); // Integer OR has the same result for every lane order.
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (!healthy(n))
        {
            if (v.identity->disposition != h::Disposition::Idle &&
                v.identity->disposition != h::Disposition::Cancelled) op.cancel(v.identity->query);
            release(n.ready, UINT64_MAX);
            return;
        }
        const auto result = check == unsigned(h::AdoptionResult::InvalidIdentity) ||
            check == unsigned(h::AdoptionResult::InvalidState)
            ? h::AdoptionResult(check) : op.finishAdoption(identity, unsigned(count), comparison);
        if (result == h::AdoptionResult::InvalidIdentity || result == h::AdoptionResult::InvalidState)
            failure(n, v.metadata.frame->status);
    }
}
/** @brief Exact complete-native bytes only; optional declined admissions perform no cache copy. */
template<class Slot> static __global__ __launch_bounds__(Threads,2) void copyFills(h::NativeFrame n)
{
    if(!blockHealthy(n))return;auto v=h::view<Slot>(n);h::Operations<Slot> op(v);
    __shared__ unsigned unique;
    for(std::uint64_t miss=blockIdx.x;miss<v.metadata.frame->source_rows;miss+=gridDim.x){
        if(threadIdx.x==0)unique=op.uniqueForMiss(unsigned(miss));
        __syncthreads();if(unique==a::None){failure(n,a::Status::Invariant);return;}
        if(v.unique[unique].flags&a::ReservedFill){
            for(std::size_t byte=threadIdx.x;byte<n.plan.row_bytes;byte+=blockDim.x){
                const auto* from=op.fillSource(unique,byte);auto* to=op.fillDestination(unique,byte);
                if(!from || !to)failure(n,a::Status::Invariant);else *to=*from;
            }
        }
        __syncthreads();
    }
}
/** @return Immutable launch geometry; no runtime/device-state query selects a mode. */
inline unsigned blocks(std::uint64_t count){return unsigned(std::max<std::uint64_t>(1,std::min<std::uint64_t>(4096,(count+Threads-1)/Threads)));}
/** @brief Reject an exact SDK submission error before another phase can hide it. */
inline void check(const char* phase)
{
#if defined(QFN_LOCAL_HOT_CUDA)
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(std::string("Local hot CUDA ")+phase+": "+cudaGetErrorString(error));
#else
    const auto error=hipGetLastError();if(error!=hipSuccess)throw std::runtime_error(std::string("Local hot HIP ")+phase+": "+hipGetErrorString(error));
#endif
}
/** @brief Submit the fixed captured native probe bundle for this original physical query shape. */
template<class Slot> void submitLargeProbeBody(const h::NativeQuery& n,QFN_LH_STREAM stream)
{
    const auto capacity=unsigned(n.original_ids.physicalRows());
    if(capacity<=Small)throw std::logic_error("Large local probe requires a multi-block query bound");
    {
        initializeKeys<<<blocks(capacity),Threads,0,stream>>>(n);check("initialize keys");
        const auto count_blocks=unsigned((capacity+Threads-1)/Threads);
        const auto passes=pooled_native::radixPasses(n.plan.source_rows);
        for(unsigned pass=0;pass<passes;++pass){
            const bool from_a=(pass%2)==0;
            radixCount<<<count_blocks,Threads,0,stream>>>(n,pass,from_a);check("radix count");
            radixOffsets<<<1,Bins,0,stream>>>(n);check("radix offsets");
            radixScatter<<<count_blocks,Threads,0,stream>>>(n,pass,from_a);check("radix scatter");
        }
        firstPositions<<<blocks(capacity),Threads,0,stream>>>(n);check("first positions");
        scanChunks<<<count_blocks,Threads,0,stream>>>(n);check("scan chunks");
        scanOffsets<Slot><<<1,1,0,stream>>>(n);check("scan offsets");
        resolveOriginal<Slot><<<blocks(capacity),Threads,0,stream>>>(n);check("original inverse");
    }
    probeUnique<Slot><<<blocks(capacity),Threads,0,stream>>>(n);check("probe");
    sealProbe<Slot><<<1,1,0,stream>>>(n);check("seal probe");
}
/** @brief Exactly one output writer bundle, with original source format specialization. */
template<TensorType Format,class Slot> void submitRead(const h::NativeRead& n,QFN_LH_STREAM stream)
{
    validateRead<Slot><<<blocks(n.physical_original),Threads,0,stream>>>(n);check("validate read");
    read<Format,Slot><<<blocks(std::uint64_t(n.physical_original)*n.plan.row_elements),Threads,0,stream>>>(n);check("read");
    completeRead<Slot><<<1,1,0,stream>>>(n);check("read completion");
}
/** @brief Decode using the SAME complete original format inventory as the pooled source. */
template<class Slot> void dispatchRead(const h::NativeRead& n,QFN_LH_STREAM stream)
{
    switch(n.plan.format){
#define QFN_LH_FORMAT(F) case TensorType::F:submitRead<TensorType::F,Slot>(n,stream);return;
        QFN_LH_FORMAT(FP32) QFN_LH_FORMAT(FP16) QFN_LH_FORMAT(BF16) QFN_LOOKUP_QUANT_FORMATS(QFN_LH_FORMAT)
#undef QFN_LH_FORMAT
    default:throw std::invalid_argument("Local hot read has no original native format decoder");
    }
}
/** @brief Terminal-only projection after the actual parent joined every local reader and auxiliary producer. */
__global__ void terminalPack(h::NativeTerminal n)
{
    const auto words=std::size_t(n.requests)*std::size_t(n.control_words);
    auto* copiedControl=reinterpret_cast<int*>(n.output+sizeof(h::TerminalObservation));
    for(std::size_t i=threadIdx.x;i<words;i+=blockDim.x)copiedControl[i]=n.control[i];
    if(threadIdx.x==0){
        const auto* identity=h::at<h::FrameIdentity>(n.frame,n.plan.frame.identity);
        const auto* frame=h::at<a::FrameControl>(n.frame,n.plan.frame.control);
        const auto* counters=h::at<h::Counters>(n.frame,n.plan.frame.counters);
        auto report=h::snapshotTerminalCounters(*counters,*identity,frame->held_pins,frame->status);
        for(int request=0;request<n.requests;++request){
            const auto* control=n.control+std::size_t(request)*n.control_words;
            if(!h::validForwardController(control[sampling_math::kDeviceGenerationControlOk],
                control[sampling_math::kDeviceGenerationControlErrorCode]))report.status=h::TerminalObservationStatus::Failed;
        }
        *reinterpret_cast<h::TerminalObservation*>(n.output)=report;
    }
}
/** @brief Stateless backend implementation; original storage owns every persistent byte and handle. */
/** @brief Select the fused small-query program or the original large-query bundle from its immutable bound. */
template<class Slot> void submitProbe(const h::NativeQuery& n,QFN_LH_STREAM stream)
{
    if(n.original_ids.physicalRows()<=Small) {
        fusedSmallProbe<Slot><<<1,Threads,0,stream>>>(n);check("fused small probe");
        return;
    }
    open<Slot><<<1,1,0,stream>>>(n);check("open");submitLargeProbeBody<Slot>(n,stream);
}
/** @brief Only the explicit next-window path resolves a derived count. */
template<class Slot> void submitNextWindowProbe(const h::NativeNextWindowQuery& n,QFN_LH_STREAM stream)
{
    if(n.original_ids.physicalRows()<=Small) {
        fusedSmallProbe<Slot><<<1,Threads,0,stream>>>(n);check("fused small next-window probe");
        return;
    }
    openNextWindow<Slot><<<1,1,0,stream>>>(n);check("open next window");submitLargeProbeBody<Slot>(n,stream);
}
class Kernels final:public IPooledImmutableLocalHotKernels {
public:
    /** @return Native data contract plus the explicit fused-probe execution-policy revision. */
    std::uint64_t codeIdentity()const noexcept override{return (h::NativeVersion << 32) | std::uint64_t{1};}
    DeviceType backend()const noexcept override{
#if defined(QFN_LOCAL_HOT_CUDA)
        return DeviceType::CUDA;
#else
        return DeviceType::ROCm;
#endif
    }
protected:
    void launchTerminal(const h::NativeTerminal& n,void* exact)const override
    {terminalPack<<<1,128,0,static_cast<QFN_LH_STREAM>(exact)>>>(n);check("whole request terminal pack");}
    NativeEmbeddingIQTables prepareTables(DeviceId device,TensorType source_format)const override
    {
        NativeEmbeddingIQTables tables{};
#if defined(QFN_LOCAL_HOT_CUDA)
        const bool valid=device.is_cuda() && prepareNativeEmbeddingTablesCUDA(device.ordinal,source_format,tables);
#else
        const bool valid=device.is_rocm() && prepareNativeEmbeddingTablesROCm(device.ordinal,source_format,tables);
#endif
        if(!valid)throw std::invalid_argument("Local hot source IQ tables changed their original backend/device");
        return tables;
    }
    void initializeNative(const h::NativeRoots& n,void* exact)const override
    {
        const auto stream=static_cast<QFN_LH_STREAM>(exact);
        const auto grid=blocks(std::max<std::uint64_t>(n.plan.slots,std::uint64_t(4)*n.plan.lane_width));
        if(n.plan.key_width==a::RowKeyWidth::UInt32){
            initializeArrays<a::Slot32><<<grid,Threads,0,stream>>>(n);check("initialize arrays");
            initializeControls<a::Slot32><<<1,1,0,stream>>>(n);
        }else{
            initializeArrays<a::Slot64><<<grid,Threads,0,stream>>>(n);check("initialize arrays");
            initializeControls<a::Slot64><<<1,1,0,stream>>>(n);
        }
        check("initialize controls");
    }
    void launchProbe(const h::NativeQuery& n,void* exact)const override
    {
        const auto stream=static_cast<QFN_LH_STREAM>(exact);
        if(n.plan.key_width==a::RowKeyWidth::UInt32)submitProbe<a::Slot32>(n,stream);
        else submitProbe<a::Slot64>(n,stream);
    }
    void launchAdoption(const h::NativeAdoption& n,void* exact)const override
    {
        const auto stream = static_cast<QFN_LH_STREAM>(exact);
        if (n.plan.key_width == a::RowKeyWidth::UInt32) adopt<a::Slot32><<<1,Threads,0,stream>>>(n);
        else adopt<a::Slot64><<<1,Threads,0,stream>>>(n);
        check("actual demand adoption");
    }
    /** @copydoc IPooledImmutableLocalHotKernels::launchNextWindowProbe */
    void launchNextWindowProbe(const h::NativeNextWindowQuery& n,void* exact)const override
    {
        const auto stream=static_cast<QFN_LH_STREAM>(exact);
        if(n.plan.key_width==a::RowKeyWidth::UInt32)submitNextWindowProbe<a::Slot32>(n,stream);
        else submitNextWindowProbe<a::Slot64>(n,stream);
    }
    void launchRead(const h::NativeRead& n,void* exact)const override
    {
        const auto stream=static_cast<QFN_LH_STREAM>(exact);
        if(n.plan.key_width==a::RowKeyWidth::UInt32)dispatchRead<a::Slot32>(n,stream);
        else dispatchRead<a::Slot64>(n,stream);
    }
    void launchRequestReset(const h::NativeRequestReset& n,void* exact)const override
    {
        const auto stream=static_cast<QFN_LH_STREAM>(exact);
        if(n.plan.key_width==a::RowKeyWidth::UInt32)resetRequest<a::Slot32><<<1,1,0,stream>>>(n);
        else resetRequest<a::Slot64><<<1,1,0,stream>>>(n);
        check("original request reset");
    }
    void launchResult(const h::NativeResult& n,void* exact)const override
    {
        const auto stream=static_cast<QFN_LH_STREAM>(exact);
        if(n.plan.key_width==a::RowKeyWidth::UInt32)accept<a::Slot32><<<1,1,0,stream>>>(n,n.own_request);
        else accept<a::Slot64><<<1,1,0,stream>>>(n,n.own_request);
        check("accept result");
    }
    void launchFrame(Action action,const h::NativeFrame& n,void* exact)const override
    {
        const auto stream=static_cast<QFN_LH_STREAM>(exact);
        if(action==Action::CopyFills){
            if(!n.plan.slots)return;
            if(n.plan.key_width==a::RowKeyWidth::UInt32)copyFills<a::Slot32><<<blocks(n.plan.maximum_original),Threads,0,stream>>>(n);
            else copyFills<a::Slot64><<<blocks(n.plan.maximum_original),Threads,0,stream>>>(n);
        }else{
            if(n.plan.key_width==a::RowKeyWidth::UInt32)frameAction<a::Slot32><<<1,1,0,stream>>>(n,action);
            else frameAction<a::Slot64><<<1,1,0,stream>>>(n,action);
        }
        check("frame phase");
    }
};
} // namespace llaminar2::qwen4exp::QFN_LH_NAMESPACE
namespace llaminar2::qwen4exp {
#if defined(QFN_LOCAL_HOT_CUDA)
std::unique_ptr<IPooledImmutableLocalHotKernels> createCUDAPooledImmutableLocalHotKernels()
#else
std::unique_ptr<IPooledImmutableLocalHotKernels> createROCmPooledImmutableLocalHotKernels()
#endif
{return std::make_unique<QFN_LH_NAMESPACE::Kernels>();}
}
#undef QFN_LH_STREAM
#undef QFN_LH_NAMESPACE
