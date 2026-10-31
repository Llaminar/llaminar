/**
 * @file PooledImmutableNativeSourceFixture.cuh
 * @brief Parallel diagnostic source/ACK and directed observation producers, never serving code.
 *
 * Every copy uses actual acquired live bytes or grants. Inactive admitted tails
 * stay poisoned. One small source frontier publishes consumption after all
 * private ACK/payload writers join, before the production owner validates/copies.
 */
#pragma once
#include "PooledImmutableNativeSourceFixture.h"
#include "kernels/qwen4exp/PooledImmutableNativeOps.h"
#if defined(QFN_FIXTURE_CUDA)
#include <cuda_runtime.h>
#define QFN_FIXTURE_STREAM cudaStream_t
#define QFN_FIXTURE_BACKEND DeviceType::CUDA
#define QFN_FIXTURE_NAMESPACE cuda_kernels
#define QFN_FIXTURE_ERROR cudaGetLastError
#define QFN_FIXTURE_SUCCESS cudaSuccess
#define QFN_FIXTURE_DEVICE cudaGetDevice
#define QFN_FIXTURE_FUNCTION launchCUDA
#else
#include <hip/hip_runtime.h>
#define QFN_FIXTURE_STREAM hipStream_t
#define QFN_FIXTURE_BACKEND DeviceType::ROCm
#define QFN_FIXTURE_NAMESPACE rocm_kernels
#define QFN_FIXTURE_ERROR hipGetLastError
#define QFN_FIXTURE_SUCCESS hipSuccess
#define QFN_FIXTURE_DEVICE hipGetDevice
#define QFN_FIXTURE_FUNCTION launchROCm
#endif
#include <stdexcept>
#include <algorithm>
namespace llaminar2::test::pooled_source_fixture::QFN_FIXTURE_NAMESPACE {
namespace p=qwen4exp::adaptive_lookup::pooled;namespace a=qwen4exp::adaptive_lookup;
/** @return The actual original owner view; no slot/occupancy/pin shadow exists. */
__device__ n::OwnerArgs owner(const Args& x){return {x.owner,x.cache,x.owner_frame,x.owner_native,{}};}
/** @brief Report incompatible decoded traces through the actual frame, never invent missing source observations. */
__device__ void fail(const Args& x){atomicExch(reinterpret_cast<unsigned*>(&n::at<p::LookupFrame>(x.coordinator_frame,x.coordinator.frame.control)->status),static_cast<unsigned>(a::Status::SourceFailure));}
/** @brief Initialize diagnostic payload poison only, before capture; leave production slot/control metadata unchanged. */
__global__ __launch_bounds__(256,2) void poison(Args x){
    for(auto i=std::uint64_t{blockIdx.x}*256+threadIdx.x;i<std::uint64_t{x.owner.owner.capacity()}*x.owner.slot_stride;i+=std::uint64_t{gridDim.x}*256)
        x.cache[x.owner.cache.payload+i]=std::byte{0xa5};
}
/** @brief Deliver only actual directed pin bytes and preserve every unused record byte. */
__global__ __launch_bounds__(256,2) void pins(Args x){
    const auto slice=n::at<n::DirectedSlice>(x.coordinator_native,x.coordinator.native.pin_slices)[0];
    const auto limit=std::uint64_t{x.owner.native.reply_capacity}*sizeof(p::OwnerCommand);
    const bool valid=slice.live_bytes<=limit && slice.live_bytes%sizeof(p::OwnerCommand)==0;
    if(!valid){if(threadIdx.x==0)fail(x);return;}
    for(auto i=std::uint64_t{threadIdx.x};i<slice.live_bytes;i+=256)x.io[x.offsets.records+i]=x.coordinator_native[x.coordinator.native.pin_commands+slice.byte_offset+i];
    if(threadIdx.x==0)*n::at<std::uint64_t>(x.io,x.offsets.record_bytes)=slice.live_bytes;
}
/** @brief Compact only an actual immutable free/victim plan, preserving its original nominal header. */
template<bool Victims> __global__ __launch_bounds__(256,2) void plan(Args x){
    using H=std::conditional_t<Victims,n::VictimPlanHeader,n::FreePlanHeader>;
    const auto h=*n::at<H>(x.coordinator_native,Victims?x.coordinator.native.victim_headers:x.coordinator.native.free_headers);
    const auto slice=n::at<n::DirectedSlice>(x.coordinator_native,Victims?x.coordinator.native.victim_slices:x.coordinator.native.free_slices)[0];
    const auto capacity=std::uint64_t{x.owner.native.reply_capacity}*sizeof(p::AllocationDecision);
    if(h.version!=n::Version || h.status!=a::Status::Success || h.decision_bytes!=slice.live_bytes || slice.live_bytes>capacity ||
        slice.live_bytes%sizeof(p::AllocationDecision) || slice.byte_offset>x.coordinator.frame.bytes ||
        slice.live_bytes>x.coordinator.frame.bytes-slice.byte_offset){if(threadIdx.x==0){fail(x);*n::at<H>(x.io,x.offsets.header)={};}return;}
    for(auto i=std::uint64_t{threadIdx.x};i<slice.live_bytes;i+=256)x.io[x.offsets.records+i]=x.coordinator_frame[slice.byte_offset+i];
    if(threadIdx.x==0)*n::at<H>(x.io,x.offsets.header)=h;
}
/** @brief Publish the exact source interval only when every requested miss has an actual native grant. */
__global__ void sourceHeader(Args x){
    const auto invocation=*n::at<n::Invocation>(x.coordinator_native,x.coordinator.native.invocation);
    const auto& frame=*n::at<p::LookupFrame>(x.coordinator_frame,x.coordinator.frame.control);
    auto* unique=n::at<p::CoordinatorUnique>(x.coordinator_frame,x.coordinator.frame.unique);bool misses=false,valid=frame.status==a::Status::Success;
    for(std::uint32_t i=0;i<frame.unique_rows;++i){
        const auto phase=unique[i].phase;valid=valid && (phase==p::UniquePhase::Read || phase==p::UniquePhase::FreeGranted || phase==p::UniquePhase::VictimGranted);
        misses=misses || phase!=p::UniquePhase::Read;
    }
    if(!valid)fail(x);
    *n::at<n::SourceFrontier>(x.trace,64)={invocation.interval,misses?p::SourceWindow{invocation.interval.table,invocation.source_storage_generation,invocation.interval.sequence}:p::SourceWindow{},
        misses?n::SourceDisposition::Published:n::SourceDisposition::Unpublished,valid?a::Status::Success:a::Status::SourceFailure,n::Version};
}
/** @brief Produce original rows and private ACKs from actual immutable owner grants; publish consumption after every writer. */
__global__ __launch_bounds__(256,2) void source(Args x){
    const auto actual=owner(x);const auto count=n::allocationFrame(actual).count;auto& frontier=*n::at<n::SourceFrontier>(x.trace,64);
    __shared__ unsigned invalid;if(threadIdx.x==0)invalid=count>x.owner.native.reply_capacity || frontier.status!=a::Status::Success;__syncthreads();
    for(auto i=threadIdx.x;i<count;i+=256)if(n::grants(actual)[i].row>=x.source_rows)atomicExch(&invalid,1u);
    __syncthreads();if(invalid){if(threadIdx.x==0){fail(x);frontier.status=a::Status::SourceFailure;}return;}
    for(auto i=threadIdx.x;i<count;i+=256){const auto grant=n::grants(actual)[i];
        n::at<n::Fill>(x.io,x.offsets.records)[i]={grant,{frontier.source,grant.destination_token}};}
    const auto bytes=std::uint64_t{count}*x.owner.row_bytes;
    for(auto i=std::uint64_t{threadIdx.x};i<bytes;i+=256){const auto row=i/x.owner.row_bytes,byte=i%x.owner.row_bytes;
        x.io[x.offsets.payload+i]=x.original_source[n::grants(actual)[row].row*x.owner.row_bytes+byte];}
    __syncthreads();if(threadIdx.x==0){
        *n::at<std::uint64_t>(x.io,x.offsets.record_bytes)=std::uint64_t{count}*sizeof(n::Fill);
        *n::at<std::uint64_t>(x.io,x.offsets.payload_bytes)=bytes;
        if(frontier.source.table)frontier.disposition=n::SourceDisposition::Consumed;
    }
}
/** @brief Actual original-position byte reader runs while the native grants still hold their physical pins. */
__global__ __launch_bounds__(256,2) void readRows(Args x){
    const auto& frame=*n::at<p::LookupFrame>(x.coordinator_frame,x.coordinator.frame.control);
    if(frame.status!=a::Status::Success || frame.phase!=p::LookupPhase::Ready)return;
    const auto* permutation=n::at<std::uint32_t>(x.coordinator_frame,x.coordinator.frame.original_to_unique);
    const auto* unique=n::at<p::CoordinatorUnique>(x.coordinator_frame,x.coordinator.frame.unique);
    const auto* slots=n::at<a::Slot32>(x.cache,x.owner.cache.slots);
    const auto bytes=std::uint64_t{frame.original_ids}*x.owner.row_bytes;
    for(auto i=std::uint64_t{blockIdx.x}*256+threadIdx.x;i<bytes;i+=std::uint64_t{gridDim.x}*256){
        const auto index=permutation[i/x.owner.row_bytes];if(index>=frame.unique_rows){fail(x);continue;}
        const auto grant=unique[index].grant;if(grant.owner || grant.slot>=x.owner.owner.capacity()){fail(x);continue;}
        const auto slot=slots[grant.slot];if(slot.row!=grant.row || slot.generation!=grant.generation || a::PhysicalSlot<a::Slot32>::state(slot)!=a::SlotState::Ready || !a::PhysicalSlot<a::Slot32>::pinned(slot)){fail(x);continue;}
        x.output[i]=x.cache[x.owner.cache.payload+std::uint64_t{grant.slot}*x.owner.slot_stride+i%x.owner.row_bytes];
    }
}
}
namespace llaminar2::test::pooled_source_fixture {
/** @brief Launch one exact diagnostic producer without allocation, readback, synchronization or hidden source I/O. */
void QFN_FIXTURE_FUNCTION(Action action,const Args& x,void* exact){
    using namespace QFN_FIXTURE_NAMESPACE;
    int current=-1;if(x.owner.device.type!=QFN_FIXTURE_BACKEND || !exact || reinterpret_cast<std::uintptr_t>(exact)<=2 || QFN_FIXTURE_DEVICE(&current)!=QFN_FIXTURE_SUCCESS || current!=x.owner.device.ordinal)
        throw std::invalid_argument("Native source fixture requires its exact current device and non-default stream");
    if(action==Action::ReadRows && x.owner.owner.rowWidth()!=a::RowKeyWidth::UInt32)throw std::invalid_argument("Native diagnostic byte reader requires its declared UInt32 source trace");
    const auto stream=static_cast<QFN_FIXTURE_STREAM>(exact);
    switch(action){
    case Action::PoisonCache:poison<<<1,256,0,stream>>>(x);break;
    case Action::DeliverPins:pins<<<1,256,0,stream>>>(x);break;
    case Action::DeliverFree:plan<false><<<1,256,0,stream>>>(x);break;
    case Action::DeliverVictims:plan<true><<<1,256,0,stream>>>(x);break;
    case Action::SourceHeader:sourceHeader<<<1,1,0,stream>>>(x);break;
    case Action::DeliverSource:source<<<1,256,0,stream>>>(x);break;
    case Action::ReadRows:readRows<<<static_cast<unsigned>(std::min<std::uint64_t>((std::uint64_t{x.maximum_original}*x.owner.row_bytes+255)/256,256)),256,0,stream>>>(x);break;
    default:throw std::invalid_argument("Unknown native source fixture operation");}
    if(QFN_FIXTURE_ERROR()!=QFN_FIXTURE_SUCCESS)throw std::runtime_error("Native source fixture launch failed");
}
}
#undef QFN_FIXTURE_STREAM
#undef QFN_FIXTURE_BACKEND
#undef QFN_FIXTURE_NAMESPACE
#undef QFN_FIXTURE_ERROR
#undef QFN_FIXTURE_SUCCESS
#undef QFN_FIXTURE_DEVICE
#undef QFN_FIXTURE_FUNCTION
