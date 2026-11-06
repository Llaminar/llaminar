/**
 * @file GPUPooledImmutableLocalHotTransport.cuh
 * @brief Symmetric native compact metadata kernels using original fixed/count receipts.
 *
 * Each fixed action is one writer downstream of its real producer or complete
 * native transfer. Failure preserves the controller's first error and exposes
 * zero payload extent. Main local readiness remains solely the L1 protocol's
 * responsibility and is never written by transport metadata.
 */
#pragma once
#include "PooledImmutableLocalHotTransportKernels.h"
#include "PooledImmutableLocalHotTerminal.h"
namespace llaminar2::qwen4exp::QFN_NATIVE_NAMESPACE {
namespace lht=local_hot_transport;
/** @return Exact fixed native launch extent after the original communicator's terminal edge. */
__device__ inline std::uint64_t compactFixedBytes(const lht::MetadataArgs& args,std::uint64_t bytes)
{
    return args.native_fixed_header.abortFlag && args.native_fixed_header.deviceCommunicator &&
        !readerSystemAcquire(args.native_fixed_header.abortFlag) && !*args.first_error ? bytes : 0;
}
/** @brief Preserve the first real error without publishing a successful compact payload. */
__device__ inline void compactFailure(const lht::MetadataArgs& args)
{
    atomicCAS(args.first_error,static_cast<int>(sampling_math::DeviceGenerationError::None),
        static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    *args.healthy=0;if(args.payload_bytes)*args.payload_bytes=0;
}
/** @brief Accumulate only after the actual receipt and all immutable identity checks succeeded. */
__device__ inline bool compactReport(std::uint64_t& value,std::uint64_t completed)
{
    return local_hot::accumulateCompletedWireCounter(value,completed);
}
/** @brief Execute one actual immutable protocol projection; no host model state is involved. */
template<lht::MetadataAction Action>
__global__ void compactMetadata(lht::MetadataArgs args)
{
    bool accepted=false;
    if constexpr(Action==lht::MetadataAction::PublishSourceAdmission){
        accepted=lht::publishSourceAdmission(args.source);
        *args.fixed_header_bytes=accepted?sizeof(lht::SourceAdmissionHeader):0;
    }
    else if constexpr(Action==lht::MetadataAction::AcceptSourceAdmission){
        const auto bytes=compactFixedBytes(args,sizeof(lht::SourceAdmissionHeader));
        accepted=lht::acceptSourceAdmission(*args.admission,args.source.table,args.source.source_storage_generation,bytes);
        *args.payload_bytes=accepted?bytes:0;
    }else if constexpr(Action==lht::MetadataAction::PrepareRequest){
        accepted=lht::prepareRequest(args.request);
        *args.fixed_header_bytes=accepted?sizeof(lht::RequestHeader):0;
    }
    else if constexpr(Action==lht::MetadataAction::AcceptRequestHeader){
        const auto bytes=compactFixedBytes(args,sizeof(lht::RequestHeader));
        accepted=*args.original_ids<=args.geometry.maximum_original && lht::acceptRequestLane(*args.received_request,*args.source_query,
            args.geometry,static_cast<std::uint32_t>(*args.original_ids),*args.admission,bytes,*args.received_lane);
        *args.fixed_header_bytes=accepted?bytes:0;
        *args.payload_bytes=accepted?args.received_request->miss_rows*sizeof(std::uint64_t):0;
    }else if constexpr(Action==lht::MetadataAction::ConfirmRequestKeys){
        const auto& request=*args.received_request;
        const auto bytes=request.miss_rows*sizeof(std::uint64_t);
        const auto aborted=readerSystemAcquire(args.native_payload_completion.abortAddress());
        accepted=args.native_payload_completion.matches(request.query.table,request.source_admission,request.query.sequence,
            bytes,args.native_payload_message,aborted,*args.first_error) &&
            *args.received_payload_bytes==bytes && *args.original_ids<=args.geometry.maximum_original &&
            lht::acceptRequest(request,*args.source_query,args.geometry,static_cast<std::uint32_t>(*args.original_ids),
                *args.admission,sizeof(lht::RequestHeader),bytes);
    }else if constexpr(Action==lht::MetadataAction::AcceptResultHeader){
        const auto bytes=compactFixedBytes(args,sizeof(lht::ResultHeader));
        accepted=lht::acceptResultHeader(*args.received_result,*args.received_request,args.geometry,bytes);
        *args.fixed_header_bytes=accepted?bytes:0;
    }else if constexpr(Action==lht::MetadataAction::PrepareResult){
        accepted=args.pool_frame->phase==adaptive_lookup::pooled::LookupPhase::Ready &&
            args.pool_frame->status==adaptive_lookup::Status::Success &&
            lht::prepareResult(*args.received_request,args.geometry,*args.admission,args.pool_frame->interval,
                *args.result,*args.payload_bytes);
    }else if constexpr(Action==lht::MetadataAction::ConfirmResult || Action==lht::MetadataAction::ConfirmAck){
        const auto& result=*args.received_result;
        const auto aborted=readerSystemAcquire(args.native_payload_completion.abortAddress());
        accepted=args.native_payload_completion.matches(result.pool_interval.table,result.pool_interval.admission,
            result.pool_interval.sequence,result.native_bytes,args.native_payload_message,aborted,*args.first_error) &&
            lht::acceptResult(result,*args.received_request,args.geometry,
                compactFixedBytes(args,sizeof(lht::ResultHeader)),*args.received_payload_bytes);
    }
    // Receive-only attribution gives every physical byte exactly one owner.
    // Fixed extents are counted only downstream of a successful actual native
    // launch, never because a configured header or capacity exists.
    if(accepted){
        if constexpr(Action==lht::MetadataAction::AcceptSourceAdmission)
            accepted=compactReport(args.reporting->request_admission_bytes,*args.payload_bytes);
        else if constexpr(Action==lht::MetadataAction::AcceptRequestHeader || Action==lht::MetadataAction::AcceptResultHeader)
            accepted=compactReport(args.reporting->lookup_control_header_bytes,*args.fixed_header_bytes);
        else if constexpr(Action==lht::MetadataAction::ConfirmRequestKeys)
            accepted=compactReport(args.reporting->request_key_payload_bytes,*args.received_payload_bytes);
        else if constexpr(Action==lht::MetadataAction::ConfirmResult)
            accepted=compactReport(args.reporting->native_result_payload_bytes,*args.received_payload_bytes);
        else if constexpr(Action==lht::MetadataAction::ConfirmAck)
            accepted=compactReport(args.reporting->acknowledgement_bytes,compactFixedBytes(args,sizeof(lht::ResultHeader)));
    }
    if(!accepted)compactFailure(args);
}
/** @brief A single writer validates immutable request headers and forms exact offsets. */
__global__ void prepareCompactUnion(lht::UnionArgs args,int* healthy,int* error)
{
    if(*error || !lht::prepareUnion(args)){
        if(args.output){args.output->status=adaptive_lookup::Status::InvalidGeometry;args.output->pool_original_ids=-1;}
        atomicCAS(error,static_cast<int>(sampling_math::DeviceGenerationError::None),static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
        atomicExch(healthy,0);
    }
}
/** @brief Disjoint live key writers reuse the actual source probe's sorted query without another sort. */
__global__ void copyCompactUnion(lht::UnionArgs args,int* healthy,int* error)
{
    if(args.output->status!=adaptive_lookup::Status::Success || args.output->pool_original_ids<0)return;
    for(std::uint64_t index=std::uint64_t{blockIdx.x}*blockDim.x+threadIdx.x;
        index<static_cast<std::uint32_t>(args.output->pool_original_ids);index+=std::uint64_t{gridDim.x}*blockDim.x){
        std::uint32_t consumer=0;
        while(consumer<args.consumer_count && index>=args.output->row_offset[consumer+1])++consumer;
        if(consumer==args.consumer_count || !lht::copyMiss(args,consumer,index-args.output->row_offset[consumer])){
            atomicCAS(error,static_cast<int>(sampling_math::DeviceGenerationError::None),static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
            atomicExch(healthy,0);
        }
    }
}
/** @brief Only this scalar terminal exposes a checked count to original pool Begin. */
__global__ void sealCompactUnion(lht::UnionArgs args,const int* error)
{
    if(*error || args.output->status!=adaptive_lookup::Status::Success){
        args.output->status=adaptive_lookup::Status::InvalidGeometry;args.output->pool_original_ids=-1;
    }
}
} // namespace llaminar2::qwen4exp::QFN_NATIVE_NAMESPACE
namespace llaminar2::qwen4exp::local_hot_transport {
#if defined(QFN_POOLED_NATIVE_CUDA)
#define QFN_COMPACT_ENQUEUE enqueueCUDAPooledLocalHotTransport
#else
#define QFN_COMPACT_ENQUEUE enqueueROCmPooledLocalHotTransport
#endif
void QFN_COMPACT_ENQUEUE(MetadataAction action,MetadataArgs args,void* exact_stream)
{
    QFN_NATIVE_NAMESPACE::require(args.device,exact_stream);
    const bool fixed=action==MetadataAction::AcceptSourceAdmission || action==MetadataAction::AcceptRequestHeader || action==MetadataAction::AcceptResultHeader || action==MetadataAction::ConfirmResult || action==MetadataAction::ConfirmAck;
    const bool counted=action==MetadataAction::ConfirmRequestKeys || action==MetadataAction::ConfirmResult || action==MetadataAction::ConfirmAck;
    if(!args.first_error || !args.healthy || !args.reporting ||
        (fixed && (!args.native_fixed_header.abortFlag || !args.native_fixed_header.deviceCommunicator)) ||
        (counted && (!args.native_payload_completion.valid() || !args.native_payload_message || !args.received_payload_bytes)) ||
        (action==MetadataAction::PublishSourceAdmission && (!args.source.output || !args.source.original_admission || !args.source.original_invocation || !args.fixed_header_bytes)) ||
        (action==MetadataAction::AcceptSourceAdmission && (!args.admission || !args.payload_bytes || !args.source.table || !args.source.source_storage_generation)) ||
        (action==MetadataAction::PrepareRequest && (!args.request.query || !args.request.original_ids || !args.request.miss_rows ||
            !args.request.source_admission || !args.request.received_admission_bytes || !args.request.output || !args.request.lane || !args.request.key_bytes || !args.fixed_header_bytes)) ||
        ((action==MetadataAction::AcceptRequestHeader || action==MetadataAction::ConfirmRequestKeys) &&
            (!args.received_request || !args.source_query || !args.original_ids || !args.admission || !valid(args.geometry))) ||
        (action==MetadataAction::AcceptRequestHeader && (!args.received_lane || !args.payload_bytes || !args.fixed_header_bytes)) ||
        (action==MetadataAction::PrepareResult && (!args.pool_frame || !args.received_request || !args.admission || !args.result || !args.payload_bytes || !valid(args.geometry))) ||
        ((action==MetadataAction::ConfirmResult || action==MetadataAction::ConfirmAck || action==MetadataAction::AcceptResultHeader) && (!args.received_result || !args.received_request || !valid(args.geometry))) ||
        (action==MetadataAction::AcceptResultHeader && !args.fixed_header_bytes))
        throw std::invalid_argument("Compact transport metadata requires its exact original producers and receipts");
    const auto stream=static_cast<QFN_NATIVE_STREAM>(exact_stream);
#define QFN_COMPACT_CASE(Name) case MetadataAction::Name: QFN_NATIVE_NAMESPACE::compactMetadata<MetadataAction::Name><<<1,1,0,stream>>>(args);break
    switch(action){
        QFN_COMPACT_CASE(PublishSourceAdmission);QFN_COMPACT_CASE(AcceptSourceAdmission);
        QFN_COMPACT_CASE(PrepareRequest);QFN_COMPACT_CASE(AcceptRequestHeader);
        QFN_COMPACT_CASE(ConfirmRequestKeys);QFN_COMPACT_CASE(AcceptResultHeader);QFN_COMPACT_CASE(PrepareResult);QFN_COMPACT_CASE(ConfirmResult);QFN_COMPACT_CASE(ConfirmAck);
        default:throw std::invalid_argument("Unknown compact metadata action");
    }
#undef QFN_COMPACT_CASE
    QFN_NATIVE_NAMESPACE::checked();
}
#if defined(QFN_POOLED_NATIVE_CUDA)
#define QFN_UNION_ENQUEUE enqueueCUDAPooledLocalHotUnion
#else
#define QFN_UNION_ENQUEUE enqueueROCmPooledLocalHotUnion
#endif
void QFN_UNION_ENQUEUE(DeviceId device,UnionArgs args,int* healthy,int* error,void* exact_stream)
{
    QFN_NATIVE_NAMESPACE::require(device,exact_stream);
    if(!healthy || !error || !args.expected || !args.source_admission || !args.output || !args.merged_keys ||
        !args.canonical.original_count || !args.canonical.sorted_original || !args.canonical.capacity ||
        !args.consumer_count || args.consumer_count>8 || !args.capacity)
        throw std::invalid_argument("Compact union requires every original source/consumer producer");
    const auto stream=static_cast<QFN_NATIVE_STREAM>(exact_stream);
    QFN_NATIVE_NAMESPACE::prepareCompactUnion<<<1,1,0,stream>>>(args,healthy,error);QFN_NATIVE_NAMESPACE::checked();
    const auto blocks=static_cast<unsigned>(std::min<std::uint64_t>(65535,(std::uint64_t{args.capacity}+255)/256));
    QFN_NATIVE_NAMESPACE::copyCompactUnion<<<blocks,256,0,stream>>>(args,healthy,error);QFN_NATIVE_NAMESPACE::checked();
    QFN_NATIVE_NAMESPACE::sealCompactUnion<<<1,1,0,stream>>>(args,error);QFN_NATIVE_NAMESPACE::checked();
}
#undef QFN_UNION_ENQUEUE
#undef QFN_COMPACT_ENQUEUE
} // namespace llaminar2::qwen4exp::local_hot_transport
