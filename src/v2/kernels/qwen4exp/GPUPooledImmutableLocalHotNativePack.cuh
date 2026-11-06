/**
 * @file GPUPooledImmutableLocalHotNativePack.cuh
 * @brief Parallel exact native-row copies from the original authenticated read image.
 *
 * One block copies one needed row, without decoding or changing reduction order.
 * The final metadata kernel alone publishes byte counts after all row readers;
 * any original pin/source/controller error vetoes the whole apparent delivery.
 */
#pragma once
#include "PooledImmutableLocalHotPackTypes.h"
#include "PooledImmutableLocalHotTransportOps.h"
namespace llaminar2::qwen4exp::QFN_SOURCE_NAMESPACE {
namespace lhpack=local_hot_transport;
/** @brief Preserve the original controller error and publish no successful native extent. */
__device__ __forceinline__ void failCompactPack(const pooled_source::DeviceArgs& a,lhpack::NativePackDescriptor p)
{
    atomicCAS(a.error,static_cast<int>(sampling_math::DeviceGenerationError::None),
        static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    atomicExch(a.healthy,0);
    *p.payload_bytes=0;if(p.local_header_bytes)*p.local_header_bytes=0;
    p.result->status=adaptive_lookup::Status::SourceFailure;
}
/** @brief Bind peer echo to the actual current source admission/interval before any row reader. */
__global__ void prepareCompactPack(pooled_source::DeviceArgs a,lhpack::NativePackDescriptor p)
{
    *p.payload_bytes=0;if(p.local_header_bytes)*p.local_header_bytes=0;
    const auto view=pooled_native::coordinatorView(a.coordinator);const auto& input=*p.union_control;
    const auto& request=*p.request;
    const lhpack::SourceAdmissionHeader admission{a.plan.owner.table,a.plan.owner.source_storage_generation,
        pooled_source::protocol(a).admissionGeneration,pooled_source::invocation(a)};
    std::uint64_t bytes=0;
    const lhpack::ConsumerGeometry geometry{p.consumer,p.consumer_count,p.maximum_rows,p.maximum_rows,
        a.plan.owner.owner.sourceRows(),a.plan.owner.row_bytes,p.native_format,p.consumer_storage_identity};
    const bool valid=pooled_source::forwardValid(a) && input.status==adaptive_lookup::Status::Success &&
        input.pool_original_ids>=0 && view.frame->original_ids==static_cast<std::uint32_t>(input.pool_original_ids) &&
        input.query.table==a.plan.owner.table && input.query.source_storage_generation==a.plan.owner.source_storage_generation &&
        p.consumer<p.consumer_count && p.consumer_count<=8 && input.row_offset[p.consumer]<=input.row_offset[p.consumer+1] &&
        request.miss_rows==input.row_offset[p.consumer+1]-input.row_offset[p.consumer] &&
        lhpack::prepareResult(request,geometry,admission,view.frame->interval,*p.result,bytes) && bytes<=p.capacity_bytes;
    if(!valid)failCompactPack(a,p);
}
/** @brief Each actual needed row has disjoint writers; inactive capacity blocks copy no bytes. */
template<class Slot> __global__ __launch_bounds__(256,2) void copyCompactRows(pooled_source::DeviceArgs a,lhpack::NativePackDescriptor p)
{
    __shared__ const std::byte* original;
    // Exactly one lane selects the block outcome. Another block may publish
    // the first error concurrently; no per-thread early return may straddle
    // this barrier, and only the final scalar seal owns result metadata.
    if(threadIdx.x==0){
        original=nullptr;
        if(blockIdx.x<p.request->miss_rows && pooled_source::forwardValid(a) && p.result->status==adaptive_lookup::Status::Success){
            original=lhpack::resultRow<Slot>(a,*p.union_control,p.consumer,p.consumer_count,blockIdx.x);
            if(!original){atomicCAS(a.error,static_cast<int>(sampling_math::DeviceGenerationError::None),
                static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));atomicExch(a.healthy,0);}
        }
    }
    __syncthreads();if(!original)return;
    const auto rowBytes=a.plan.owner.row_bytes;
    for(std::uint64_t byte=threadIdx.x;byte<rowBytes;byte+=blockDim.x)
        p.rows[std::uint64_t{blockIdx.x}*rowBytes+byte]=original[byte];
}
/** @brief Exact source-local producer frontier, after every native row read/copy has retired. */
__global__ void sealCompactPack(pooled_source::DeviceArgs a,lhpack::NativePackDescriptor p)
{
    if(!pooled_source::forwardValid(a) || p.result->status!=adaptive_lookup::Status::Success ||
        p.result->native_bytes>p.capacity_bytes){failCompactPack(a,p);return;}
    *p.payload_bytes=p.result->native_bytes;
    if(p.local_header_bytes)*p.local_header_bytes=sizeof(lhpack::ResultHeader);
}
} // namespace llaminar2::qwen4exp::QFN_SOURCE_NAMESPACE
namespace llaminar2::qwen4exp::local_hot_transport {
#if defined(QFN_POOLED_SOURCE_CUDA)
#define QFN_COMPACT_PACK enqueueCUDAPooledLocalHotNativePack
#else
#define QFN_COMPACT_PACK enqueueROCmPooledLocalHotNativePack
#endif
void QFN_COMPACT_PACK(pooled_source::DeviceArgs a,NativePackDescriptor p,void* exact_stream)
{
    QFN_SOURCE_NAMESPACE::requireContext(a.plan.coordinator.device,exact_stream);
    if(!p.union_control || !p.request || !p.result || !p.rows || !p.payload_bytes || !p.maximum_rows ||
        !p.consumer_storage_identity || !p.consumer_count || p.consumer_count>8 || p.consumer>=p.consumer_count ||
        p.capacity_bytes!=std::uint64_t{p.maximum_rows}*a.plan.owner.row_bytes || !a.healthy || !a.error)
        throw std::invalid_argument("Compact native pack requires actual complete source/consumer spans");
    const auto stream=static_cast<QFN_SOURCE_STREAM>(exact_stream);
    QFN_SOURCE_NAMESPACE::prepareCompactPack<<<1,1,0,stream>>>(a,p);QFN_SOURCE_NAMESPACE::check();
    if(a.plan.owner.owner.rowWidth()==adaptive_lookup::RowKeyWidth::UInt32)
        QFN_SOURCE_NAMESPACE::copyCompactRows<adaptive_lookup::Slot32><<<p.maximum_rows,256,0,stream>>>(a,p);
    else if(a.plan.owner.owner.rowWidth()==adaptive_lookup::RowKeyWidth::UInt64)
        QFN_SOURCE_NAMESPACE::copyCompactRows<adaptive_lookup::Slot64><<<p.maximum_rows,256,0,stream>>>(a,p);
    else throw std::invalid_argument("Compact native pack changed original source row-key width");
    QFN_SOURCE_NAMESPACE::check();
    QFN_SOURCE_NAMESPACE::sealCompactPack<<<1,1,0,stream>>>(a,p);QFN_SOURCE_NAMESPACE::check();
}
#undef QFN_COMPACT_PACK
} // namespace llaminar2::qwen4exp::local_hot_transport
