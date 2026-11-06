/**
 * @file PooledImmutableLocalHotTransportOps.h
 * @brief Device-owned compact metadata over the original admitted source channel.
 *
 * These operations project existing producers into fixed native wire records.
 * They neither issue an admission nor advance a pool interval or L1 sequence.
 * A failed projection leaves zero payload extent and invalid metadata, so no
 * capacity bytes can become a successful counted native transfer.
 */
#pragma once
#include "PooledImmutableLocalHotUnionOps.h"
namespace llaminar2::qwen4exp::local_hot_transport {
/** @brief Exact native source publication, borrowed from the original source roots. */
struct SourceAdmissionArgs {
    std::uint64_t table=0,source_storage_generation=0;
    const std::uint64_t* original_admission=nullptr;
    const std::uint64_t* original_invocation=nullptr;
    SourceAdmissionHeader* output=nullptr;
};
/** @brief Publish the actual original I/O admission without inventing another counter. */
QFN_ADAPTIVE_HD inline bool publishSourceAdmission(const SourceAdmissionArgs& a)
{
    if(!a.output)return false;
    *a.output={};a.output->status=Status::InvalidGeometry;
    if(!a.table || !a.source_storage_generation || !a.original_admission || !a.original_invocation ||
        !*a.original_admission || !*a.original_invocation)return false;
    SourceAdmissionHeader next{};next.table=a.table;next.source_storage_generation=a.source_storage_generation;
    next.source_admission=*a.original_admission;next.source_invocation=*a.original_invocation;
    *a.output=next;return true;
}
/** @brief Actual local probe and received once-per-request source publication. */
struct RequestArgs {
    const QueryIdentity* query=nullptr;
    const std::uint64_t* original_ids=nullptr;
    const std::uint64_t* miss_rows=nullptr;
    const SourceAdmissionHeader* source_admission=nullptr;
    const std::uint64_t* received_admission_bytes=nullptr;
    ConsumerGeometry geometry{};
    PooledImmutableLocalHotAccessIntent intent=PooledImmutableLocalHotAccessIntent::CurrentInvocation;
    RequestHeader* output=nullptr;
    RequestLaneIdentity* lane=nullptr;
    std::uint64_t* key_bytes=nullptr;
};
/**
 * @brief Freeze one actual compact request after local probe and original admission.
 * Empty requests retain their ordinal and fixed metadata, with exactly zero key
 * bytes. This never opens L2 or changes the main frame's independent ready word.
 */
QFN_ADAPTIVE_HD inline bool prepareRequest(const RequestArgs& a)
{
    if(a.output){*a.output={};a.output->status=Status::InvalidGeometry;}
    if(a.lane)*a.lane={};
    if(a.key_bytes)*a.key_bytes=0;
    if(!a.output || !a.lane || !a.key_bytes || !a.query || !valid(*a.query) || !valid(a.geometry) ||
        !a.original_ids || !a.miss_rows || !a.source_admission ||
        !a.received_admission_bytes || a.intent!=PooledImmutableLocalHotAccessIntent::CurrentInvocation ||
        a.query->consumer_storage_identity!=a.geometry.consumer_storage_identity ||
        !acceptSourceAdmission(*a.source_admission,a.query->table,a.query->source_storage_generation,
            *a.received_admission_bytes) ||
        *a.original_ids>a.geometry.maximum_original || *a.original_ids%16 ||
        *a.miss_rows>static_cast<std::uint32_t>(*a.original_ids) || *a.miss_rows>a.geometry.maximum_misses)return false;
    RequestHeader next{};next.query=*a.query;next.miss_rows=*a.miss_rows;
    next.original_ids=static_cast<std::uint32_t>(*a.original_ids);next.consumer=a.geometry.coordinate;
    next.native_format=a.geometry.native_format;next.native_row_bytes=a.geometry.native_row_bytes;
    next.intent=a.intent;next.source_admission=a.source_admission->source_admission;
    next.source_invocation=a.source_admission->source_invocation;
    *a.output=next;*a.lane={next.query.table,next.source_admission,next.query.sequence};
    *a.key_bytes=next.miss_rows*sizeof(std::uint64_t);return true;
}
/** @brief Source-side request-lane projection, only after the actual header receive proof. */
QFN_ADAPTIVE_HD inline bool acceptRequestLane(const RequestHeader& request,QueryIdentity sourceQuery,
    ConsumerGeometry geometry,std::uint32_t originalIds,const SourceAdmissionHeader& admission,
    std::uint64_t receivedHeaderBytes,RequestLaneIdentity& output)
{
    output={};
    if(!acceptRequestHeader(request,sourceQuery,geometry,originalIds,admission,receivedHeaderBytes))return false;
    output={request.query.table,request.source_admission,request.query.sequence};return true;
}
/**
 * @brief Publish exact native response metadata from the original completed pool interval.
 * The payload copy remains the original readableRow/inverse-map operation; this
 * method merely bounds its counted prefix and preserves the actual peer echo.
 */
QFN_ADAPTIVE_HD inline bool prepareResult(const RequestHeader& request,ConsumerGeometry geometry,
    const SourceAdmissionHeader& admission,p::Interval actualPoolInterval,
    ResultHeader& output,std::uint64_t& nativeBytes)
{
    output={};output.status=Status::InvalidGeometry;nativeBytes=0;
    if(!acceptRequestHeader(request,request.query,geometry,request.original_ids,admission,sizeof(RequestHeader)) ||
        actualPoolInterval.table!=request.query.table || actualPoolInterval.admission!=request.source_admission ||
        !actualPoolInterval.sequence)return false;
    ResultHeader next{};next.query=request.query;next.pool_interval=actualPoolInterval;
    next.miss_rows=request.miss_rows;next.native_bytes=request.miss_rows*geometry.native_row_bytes;
    next.consumer=geometry.coordinate;next.native_format=geometry.native_format;next.intent=request.intent;
    output=next;nativeBytes=next.native_bytes;return true;
}
} // namespace llaminar2::qwen4exp::local_hot_transport
