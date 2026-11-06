/**
 * @file PooledImmutableLocalHotTransportProtocol.h
 * @brief Native compact miss records over the original addressed pooled transport.
 *
 * These are immutable wire observations, never a source ticket, pin, cache slot
 * or transport authority. Full model queries retain their sixteen-head extent;
 * compact miss counts are arbitrary prefixes. Local cache owners alone decide
 * hits and in-flight reuse. Original pooled coalescing owns union coordinates.
 */
#pragma once
#include "PooledImmutablePhysicalOwnerState.h"
#include "PooledImmutableLocalHotAccess.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaminar2::qwen4exp::local_hot_transport {
namespace p=adaptive_lookup::pooled;
using adaptive_lookup::Status;
/** @brief Versioned interpretation of these exact native byte records. */
inline constexpr std::uint32_t Version=1;
/** @brief L1 identity, independent from original source/service admission and L2 interval. */
struct QueryIdentity {
    std::uint64_t table=0,source_storage_generation=0,consumer_storage_identity=0,sequence=0;
};
/** @return Equality of complete immutable query/source identity, never readiness. */
QFN_ADAPTIVE_HD inline bool same(QueryIdentity a,QueryIdentity b)
{
    return a.table==b.table && a.source_storage_generation==b.source_storage_generation &&
        a.consumer_storage_identity==b.consumer_storage_identity && a.sequence==b.sequence;
}
/** @return Whether an identity names an actual positive original generation. */
QFN_ADAPTIVE_HD inline bool valid(QueryIdentity x)
{return x.table && x.source_storage_generation && x.consumer_storage_identity && x.sequence;}
/** @brief Fixed per-consumer envelope, derived from actual original reader geometry. */
struct ConsumerGeometry {
    std::uint32_t coordinate=0,consumers=0,maximum_original=0,maximum_misses=0;
    std::uint64_t source_rows=0,native_row_bytes=0;
    std::uint32_t native_format=0;
    std::uint64_t consumer_storage_identity=0;
};
/** @return Whether static geometry represents full queries and arbitrary compact native rows. */
QFN_ADAPTIVE_HD inline bool valid(ConsumerGeometry g)
{
    return g.consumers && g.coordinate<g.consumers && g.maximum_original &&
        g.maximum_original%16==0 && g.maximum_misses && g.maximum_misses<=g.maximum_original &&
        g.source_rows && g.native_row_bytes && g.consumer_storage_identity &&
        g.maximum_misses<=UINT64_MAX/g.native_row_bytes;
}
/** @brief Once-per-request original I/O admission, authored by the actual source native state. */
struct SourceAdmissionHeader {
    std::uint64_t table=0,source_storage_generation=0,source_admission=0,source_invocation=0;
    std::uint32_t version=Version;
    Status status=Status::Success;
};
static_assert(sizeof(SourceAdmissionHeader)==40);
/**
 * @brief Explicit request-lane identity projection for existing generic native byte operands.
 * This is NOT an L2 pool interval. Its admission is the original I/O admission,
 * sequence is the actual peer L1 ordinal, and it owns no counter or policy state.
 */
struct RequestLaneIdentity {std::uint64_t table=0,source_admission=0,sequence=0;};
static_assert(sizeof(RequestLaneIdentity)==24);
/** @return Structural original source identity; this alone is never receive proof. */
QFN_ADAPTIVE_HD inline bool validSourceAdmission(const SourceAdmissionHeader& source,
    std::uint64_t table,std::uint64_t sourceGeneration)
{
    return source.version==Version && source.status==Status::Success && source.table==table &&
        source.source_storage_generation==sourceGeneration && table && sourceGeneration &&
        source.source_admission && source.source_invocation;
}
/** @return Actual fixed-header publication after its original transfer/producer proof. */
QFN_ADAPTIVE_HD inline bool acceptSourceAdmission(const SourceAdmissionHeader& source,
    std::uint64_t table,std::uint64_t sourceGeneration,std::uint64_t actualHeaderBytes)
{return actualHeaderBytes==sizeof(SourceAdmissionHeader) && validSourceAdmission(source,table,sourceGeneration);}
/** @brief One stable unique-first-occurrence miss prefix; no capacity bytes are sent. */
struct RequestHeader {
    QueryIdentity query{};
    std::uint64_t miss_rows=0;
    std::uint32_t original_ids=0,consumer=0,native_format=0,version=Version;
    std::uint64_t native_row_bytes=0;
    Status status=Status::Success;
    PooledImmutableLocalHotAccessIntent intent=PooledImmutableLocalHotAccessIntent::CurrentInvocation;
    std::uint64_t source_admission=0,source_invocation=0;
};
/** @brief Response to precisely one consumer miss roster, in that same native row order. */
struct ResultHeader {
    QueryIdentity query{};
    p::Interval pool_interval{}; ///< Original pooled authority's actual completed read interval.
    std::uint64_t miss_rows=0,native_bytes=0;
    std::uint32_t consumer=0,native_format=0,version=Version;
    Status status=Status::Success;
    PooledImmutableLocalHotAccessIntent intent=PooledImmutableLocalHotAccessIntent::CurrentInvocation;
};
static_assert(std::is_trivially_copyable_v<RequestHeader> && std::is_trivially_copyable_v<ResultHeader>);
static_assert(sizeof(RequestHeader)==88 && sizeof(ResultHeader)==96);
/**
 * @brief Validate actual compact metadata against the original source query/reader.
 * FutureReadAhead is deliberately rejected: unchanged pooled Begin credits demand
 * and cannot authenticate speculative lookup-only/bypass-admission semantics.
 */
QFN_ADAPTIVE_HD inline bool acceptRequestHeader(const RequestHeader& h,QueryIdentity expected,
    ConsumerGeometry g,std::uint32_t expected_original_ids,const SourceAdmissionHeader& source,
    std::uint64_t received_header_bytes)
{
    return valid(g) && valid(expected) && valid(h.query) && h.query.table==expected.table &&
        h.query.source_storage_generation==expected.source_storage_generation &&
        h.query.consumer_storage_identity==g.consumer_storage_identity &&
        validSourceAdmission(source,expected.table,expected.source_storage_generation) &&
        h.source_admission==source.source_admission && h.source_invocation==source.source_invocation &&
        received_header_bytes==sizeof(RequestHeader) && h.version==Version && h.status==Status::Success &&
        h.consumer==g.coordinate && h.native_format==g.native_format && h.native_row_bytes==g.native_row_bytes &&
        h.intent==PooledImmutableLocalHotAccessIntent::CurrentInvocation &&
        h.original_ids==expected_original_ids && h.original_ids<=g.maximum_original &&
        h.original_ids%16==0 && h.miss_rows<=h.original_ids && h.miss_rows<=g.maximum_misses;
}
/** @brief Validate the actual original counted-key receipt after its separate fixed header. */
QFN_ADAPTIVE_HD inline bool acceptRequest(const RequestHeader& h,QueryIdentity expected,
    ConsumerGeometry g,std::uint32_t expected_original_ids,const SourceAdmissionHeader& source,
    std::uint64_t received_header_bytes,std::uint64_t received_key_bytes)
{
    return acceptRequestHeader(h,expected,g,expected_original_ids,source,received_header_bytes) &&
        received_key_bytes==h.miss_rows*sizeof(std::uint64_t);
}
/** @brief Validate only the received fixed native result header, never payload completion/readiness. */
QFN_ADAPTIVE_HD inline bool acceptResultHeader(const ResultHeader& h,const RequestHeader& request,
    ConsumerGeometry g,std::uint64_t received_header_bytes)
{
    return valid(g) && valid(request.query) && same(h.query,request.query) &&
        request.version==Version && request.status==Status::Success && request.consumer==g.coordinate &&
        request.intent==PooledImmutableLocalHotAccessIntent::CurrentInvocation && h.intent==request.intent &&
        request.native_format==g.native_format && request.native_row_bytes==g.native_row_bytes &&
        request.query.consumer_storage_identity==g.consumer_storage_identity &&
        request.original_ids<=g.maximum_original && request.original_ids%16==0 &&
        request.miss_rows<=request.original_ids && request.miss_rows<=g.maximum_misses &&
        received_header_bytes==sizeof(ResultHeader) && h.version==Version && h.status==Status::Success &&
        h.consumer==g.coordinate && h.native_format==g.native_format && h.miss_rows==request.miss_rows &&
        h.native_bytes==request.miss_rows*g.native_row_bytes &&
        request.source_admission && request.source_invocation &&
        h.pool_interval.table==request.query.table && h.pool_interval.admission==request.source_admission && h.pool_interval.sequence;
}
/** @brief Check the actual compact payload completion before any missing cache row becomes Ready. */
QFN_ADAPTIVE_HD inline bool acceptResult(const ResultHeader& h,const RequestHeader& request,
    ConsumerGeometry g,std::uint64_t received_header_bytes,std::uint64_t received_native_bytes)
{return acceptResultHeader(h,request,g,received_header_bytes) && received_native_bytes==h.native_bytes;}
/** @brief One endpoint's actual wire metadata; native receive extents are independent outputs. */
struct alignas(64) Control {
    SourceAdmissionHeader original_admission{};
    RequestLaneIdentity request_lane{};
    std::uint64_t admission_receive_bytes=0;
    RequestHeader request_out{},request_in{};
    ResultHeader result_out{},result_in{};
    std::uint64_t request_send_bytes=0,request_receive_bytes=0;
    std::uint64_t result_send_bytes=0,result_receive_bytes=0;
    std::uint64_t header_receive_bytes=0;
};
/** @brief One source-side concatenation projection into the original pool coalescer. */
struct alignas(64) UnionControl {
    QueryIdentity query{};
    std::uint32_t row_offset[9]{};
    std::int32_t pool_original_ids=0;
    adaptive_lookup::Status status=adaptive_lookup::Status::Success;
};

/**
 * @brief Borrow actual received header/payload publications and bounded native storage.
 * No value is downloaded or copied into a host shadow. The original transport
 * stage orders these publications before the local cache consumes this view.
 */
struct DeviceResultView {
    const ResultHeader* header=nullptr;
    const std::byte* native_rows=nullptr;
    const std::uint64_t* received_header_bytes=nullptr;
    const std::uint64_t* received_payload_bytes=nullptr;
    std::uint64_t native_capacity=0;
};
/** @return Exact native row only after its own authenticated compact result; never an allocation/capability. */
QFN_ADAPTIVE_HD inline const std::byte* receivedRow(DeviceResultView view,
    const RequestHeader& ownRequest,ConsumerGeometry geometry,std::uint64_t compactRow)
{
    if(!view.header || !view.received_header_bytes || !view.received_payload_bytes ||
        !view.native_rows || !valid(geometry) || compactRow>=ownRequest.miss_rows ||
        ownRequest.miss_rows>view.native_capacity/geometry.native_row_bytes ||
        !acceptResult(*view.header,ownRequest,geometry,*view.received_header_bytes,*view.received_payload_bytes)) return nullptr;
    return view.native_rows+compactRow*geometry.native_row_bytes;
}
} // namespace llaminar2::qwen4exp::local_hot_transport
