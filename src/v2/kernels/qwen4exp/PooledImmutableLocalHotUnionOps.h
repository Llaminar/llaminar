/**
 * @file PooledImmutableLocalHotUnionOps.h
 * @brief Compact consumer parsing into the original stable pooled coalescer.
 *
 * This adapter concatenates authenticated miss prefixes. It does not sort or
 * create a union directory: original CoordinatorOperations alone computes the
 * stable unique rows and original_to_unique inverse. The canonical query view
 * is borrowed from the actual source participant's completed local probe.
 */
#pragma once
#include "PooledImmutableLocalHotTransportProtocol.h"
#include "PooledImmutableSourceOps.h"
namespace llaminar2::qwen4exp::local_hot_transport {
/** @brief Immutable one-CTA union envelope; larger queries retain the existing partitioned DAG. */
inline constexpr unsigned CooperativeUnionRows=256;
/** @return Whether this captured union fits one CTA, independent of live counts. */
QFN_ADAPTIVE_HD constexpr bool usesCooperativeUnion(std::uint32_t capacity) noexcept
{return capacity && capacity<=CooperativeUnionRows;}
/** @return Complete immutable native union launch policy for graph capture identity. */
QFN_ADAPTIVE_HD constexpr std::uint64_t unionLaunchIdentity(std::uint32_t capacity) noexcept
{return (std::uint64_t{2}<<32)|(std::uint64_t{CooperativeUnionRows}<<1)|std::uint64_t{usesCooperativeUnion(capacity)};}
/** @brief Exact source probe's retained sorted original rows, not peer-provided identity. */
struct CanonicalQueryView {
    const adaptive_lookup::KeyIndex* sorted_original=nullptr;
    const std::uint64_t* original_count=nullptr;
    std::uint32_t capacity=0;
};
/** @brief One exact acquired remote request, or same-device local producer observation. */
struct RequestView {
    const RequestHeader* header=nullptr;
    const std::uint64_t* keys=nullptr;
    const std::uint64_t* received_header_bytes=nullptr;
    const std::uint64_t* received_key_bytes=nullptr;
};
/** @brief Capture-stable source arguments; every pointee remains with original PMA/stage owners. */
struct UnionArgs {
    const QueryIdentity* expected=nullptr;
    const SourceAdmissionHeader* source_admission=nullptr;
    CanonicalQueryView canonical;
    ConsumerGeometry consumers[8]{};
    RequestView requests[8]{};
    std::uint32_t consumer_count=0,capacity=0;
    PooledImmutableLocalHotAccessIntent expected_intent=PooledImmutableLocalHotAccessIntent::CurrentInvocation;
    UnionControl* output=nullptr;
    std::uint64_t* merged_keys=nullptr;
};
/** @return Original first coordinate for an exact source-query member, or invalid on missing/foreign keys. */
QFN_ADAPTIVE_HD inline std::uint32_t canonicalPosition(CanonicalQueryView source,std::uint64_t row)
{
    if(!source.sorted_original || !source.original_count ||
        *source.original_count>source.capacity) return adaptive_lookup::None;
    std::uint32_t first=0,last=static_cast<std::uint32_t>(*source.original_count);
    while(first<last){const auto middle=first+(last-first)/2;
        if(source.sorted_original[middle].row<row)first=middle+1;else last=middle;
    }
    return first<static_cast<std::uint32_t>(*source.original_count) && source.sorted_original[first].row==row &&
        source.sorted_original[first].original_coordinate<static_cast<std::uint32_t>(*source.original_count)
        ?source.sorted_original[first].original_coordinate:adaptive_lookup::None;
}
/**
 * @brief Validate actual fixed metadata and publish only checked concatenation offsets.
 * @return False before any key or pool state is changed for incompatible input.
 * Empty prefixes are valid, including a completely all-hit union. No source or
 * owner payload operation follows merely from allocated capacity.
 */
QFN_ADAPTIVE_HD inline bool prepareUnion(const UnionArgs& a)
{
    if(a.output){*a.output={};a.output->pool_original_ids=-1;a.output->status=Status::InvalidGeometry;}
    if(!a.output || !a.merged_keys || !a.expected || !valid(*a.expected) || !a.source_admission ||
        !validSourceAdmission(*a.source_admission,a.expected->table,a.expected->source_storage_generation) || !a.consumer_count || a.consumer_count>8 ||
        !a.canonical.original_count ||
        *a.canonical.original_count>a.canonical.capacity ||
        *a.canonical.original_count%16) return false;
    UnionControl next{};next.query=*a.expected;next.intent=a.expected_intent;std::uint32_t total=0;
    for(std::uint32_t i=0;i<a.consumer_count;++i){
        const auto& r=a.requests[i];const auto g=a.consumers[i];
        if(!r.header || !r.received_header_bytes || !r.received_key_bytes ||
            g.coordinate!=i || g.consumers!=a.consumer_count ||
            !acceptRequest(*r.header,*a.expected,g,static_cast<std::uint32_t>(*a.canonical.original_count),
                *a.source_admission,*r.received_header_bytes,*r.received_key_bytes,a.expected_intent) ||
            (r.header->miss_rows && !r.keys) || total>a.capacity || r.header->miss_rows>a.capacity-total) return false;
        next.row_offset[i]=total;total+=static_cast<std::uint32_t>(r.header->miss_rows);
    }
    if(total>static_cast<std::uint32_t>(INT32_MAX))return false;
    next.row_offset[a.consumer_count]=total;next.pool_original_ids=static_cast<std::int32_t>(total);
    *a.output=next;return true;
}
/**
 * @brief Validate and copy one actual miss, suitable for disjoint parallel GPU writers.
 * Strict first-original ordering authenticates subset, uniqueness and roster order
 * together. Binary search reuses the original probe sort, never a second sorter.
 */
QFN_ADAPTIVE_HD inline bool copyMiss(const UnionArgs& a,std::uint32_t consumer,std::uint64_t miss)
{
    if(!a.output || !a.merged_keys || consumer>=a.consumer_count || a.consumer_count>8 ||
        !a.requests[consumer].header || !a.requests[consumer].keys || a.output->status!=Status::Success ||
        miss>=a.requests[consumer].header->miss_rows || a.output->pool_original_ids<0 ||
        a.output->row_offset[consumer]>static_cast<std::uint32_t>(a.output->pool_original_ids) ||
        a.output->row_offset[consumer+1]<a.output->row_offset[consumer] ||
        miss>=a.output->row_offset[consumer+1]-a.output->row_offset[consumer] ||
        a.output->row_offset[consumer]+miss>=a.capacity) return false;
    const auto row=a.requests[consumer].keys[miss];
    if(row>=a.consumers[consumer].source_rows)return false;
    const auto position=canonicalPosition(a.canonical,row);
    if(position==adaptive_lookup::None || (miss &&
        canonicalPosition(a.canonical,a.requests[consumer].keys[miss-1])>=position))return false;
    // A singleton borrows its actual local miss roster. Do not issue a
    // redundant store while another lane validates the preceding key.
    auto* destination=a.merged_keys+a.output->row_offset[consumer]+miss;
    if(destination!=a.requests[consumer].keys+miss)*destination=row;
    return true;
}
/**
 * @brief Locate the original coalescer's exact native source row for one compact result.
 * @return Actual readable source pointer, never the expanded FP32 result or cache-capacity slice.
 * All original pins/read/source-completion checks remain in readableRow. The
 * caller copies exactly row_bytes and retires the original reader afterwards.
 */
template<class Slot> QFN_ADAPTIVE_HD inline const std::byte* resultRow(
    const pooled_source::DeviceArgs& source,const UnionControl& input,
    std::uint32_t consumer,std::uint32_t consumerCount,std::uint64_t miss)
{
    const auto view=pooled_native::coordinatorView(source.coordinator);
    if(!consumerCount || consumerCount>8 || consumer>=consumerCount || input.pool_original_ids<0 ||
        input.status!=Status::Success || !valid(input.query) ||
        ((input.intent==PooledImmutableLocalHotAccessIntent::FutureReadAhead) !=
         (view.frame->purpose==adaptive_lookup::pooled::LookupPurpose::FutureLookupOnly)) ||
        (input.intent!=PooledImmutableLocalHotAccessIntent::CurrentInvocation &&
         input.intent!=PooledImmutableLocalHotAccessIntent::FutureReadAhead) ||
        view.frame->interval.table!=input.query.table ||
        view.frame->source_storage_generation!=input.query.source_storage_generation || view.frame->original_ids!=static_cast<std::uint32_t>(input.pool_original_ids) ||
        input.row_offset[consumer]>input.row_offset[consumer+1] ||
        miss>=input.row_offset[consumer+1]-input.row_offset[consumer] ||
        input.row_offset[consumer]+miss>=view.frame->original_ids) return nullptr;
    return pooled_source::readableRow<Slot>(source,view.original_to_unique[input.row_offset[consumer]+miss]);
}
} // namespace llaminar2::qwen4exp::local_hot_transport
