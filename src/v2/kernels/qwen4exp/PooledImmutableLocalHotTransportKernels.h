/**
 * @file PooledImmutableLocalHotTransportKernels.h
 * @brief Exact-stream metadata lowerings for the original compact pooled channels.
 *
 * NativeArgs is an internal capture descriptor minted by the checked graph
 * stage. All storage and native receipts stay with original transport, L1 and
 * source owners; this code owns no allocation, admission or mutable counter.
 */
#pragma once
#include "PooledImmutableLocalHotTransportOps.h"
#include "PooledImmutableLocalHotState.h"
#include "backends/DeviceId.h"
#include "collective/NativeCollectiveCompletion.h"
namespace llaminar2::qwen4exp::local_hot_transport {
/** @brief Fixed metadata projections; payload union/copies have separate parallel entrypoints. */
enum class MetadataAction : std::uint32_t {
    PublishSourceAdmission,AcceptSourceAdmission,PrepareRequest,AcceptRequestHeader,ConfirmRequestKeys,AcceptResultHeader,PrepareResult,ConfirmResult,ConfirmAck
};
/** @brief Complete original pointers for one checked participant-local metadata phase. */
struct MetadataArgs {
    DeviceId device;
    local_hot::Counters* reporting=nullptr; ///< SAME actual receiving participant root; source-self authors no wire count.
    SourceAdmissionArgs source{};
    RequestArgs request{}; ///< intent is the immutable expected wire purpose for every metadata action, including replies/ACKs.
    const QueryIdentity* source_query=nullptr;
    ConsumerGeometry geometry{};
    const std::uint64_t* original_ids=nullptr;
    const RequestHeader* received_request=nullptr;
    const SourceAdmissionHeader* admission=nullptr;
    RequestLaneIdentity* received_lane=nullptr;
    const ResultHeader* received_result=nullptr;
    ResultHeader* result=nullptr;
    const p::LookupFrame* pool_frame=nullptr;
    std::uint64_t* payload_bytes=nullptr;
    const std::uint64_t* received_header_bytes=nullptr;
    std::uint64_t* fixed_header_bytes=nullptr;
    const std::uint64_t* received_payload_bytes=nullptr;
    int* first_error=nullptr;
    int* healthy=nullptr;
    ncclLlaminarCompletionAuthority native_fixed_header{};
    NativeReceiveCompletionView native_payload_completion=NativeReceiveCompletionView::notRequired();
    std::uint64_t native_payload_message=0;
};
/** @brief Validate/concatenate exact peer prefixes before the unchanged original pool coalescer. */
void enqueueCUDAPooledLocalHotUnion(DeviceId,UnionArgs,int* healthy,int* first_error,void* exact_stream);
/** @brief Identical source subset/concatenation program on the exact HIP stream. */
void enqueueROCmPooledLocalHotUnion(DeviceId,UnionArgs,int* healthy,int* first_error,void* exact_stream);
/** @brief Submit one actual compact metadata kernel on the exact original CUDA stream. */
void enqueueCUDAPooledLocalHotTransport(MetadataAction,MetadataArgs,void* exact_stream);
/** @brief Submit the identical compact protocol on the exact original HIP stream. */
void enqueueROCmPooledLocalHotTransport(MetadataAction,MetadataArgs,void* exact_stream);
} // namespace llaminar2::qwen4exp::local_hot_transport
