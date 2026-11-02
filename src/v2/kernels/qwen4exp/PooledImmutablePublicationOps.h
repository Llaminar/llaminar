/**
 * @file PooledImmutablePublicationOps.h
 * @brief Nominal source publication/acquisition/Ready parsing over original retained roots.
 *
 * These actual CPU/GPU bodies authenticate original producers and epoch fields,
 * then fill the existing ordered metadata bank. They never touch remote roots,
 * reserve a row, release a pin or retire private source readers. Fixed graph
 * edges own acquisition/order; wire counts are exact60, never sizeof64 padding.
 */
#pragma once
#include "PooledImmutableSourceOps.h"
namespace llaminar2::qwen4exp::pooled_source {
/** @brief Preserve an original source error while rejecting a new invalid publication observation. */
QFN_ADAPTIVE_HD inline bool rejectPublication(const DeviceArgs& a,ImmutableLookupStatus status=ImmutableLookupStatus::InvalidAcknowledgement)
{return frontier(a).status==Status::Success?reject(a,status):false;}
/** @return Original protocol cursor for one closed nominal acquired metadata family. */
template<class Header> QFN_ADAPTIVE_HD std::uint32_t& publicationCursor(const DeviceArgs& a)
{
    if constexpr(std::is_same_v<Header,p::PublishHeader>)return progress(a).publication_headers;
    else if constexpr(std::is_same_v<Header,n::PublicationAcquiredHeader>)return progress(a).publication_acquired;
    else {static_assert(std::is_same_v<Header,p::ReadyHeader>);return progress(a).ready_headers;}
}
/**
 * @brief Import one exact remote60-byte nominal header into the original ordered56-byte frontier slot.
 * @param owner Original fixed participant coordinate for this captured receive/import pair.
 * @return True only for an authenticated success record; an authenticated error is retained and fails the source outcome.
 * @note Each import retires before the next serial receive. Actual error records may truthfully retain an old or zero
 * source after an unsealed failure. Its window is never accepted as authority; this only drains
 * the fixed metadata edge, and cannot permit success, retirement or a fabricated global window.
 * Local imports use original frame/kind controls through the mixed-local native boundary instead.
 */
template<class Header> QFN_ADAPTIVE_HD bool importPublicationRecord(const DeviceArgs& a,std::uint32_t owner,const Header& record,n::OwnerHeaderKind kind)
{
    static_assert(sizeof(Header)==56);
    const auto& original=*n::coordinatorView(a.coordinator).frame;const auto& outcome=frontier(a);auto& cursor=publicationCursor<Header>(a);
    if(outcome.status==Status::Success){
        if constexpr(!std::is_same_v<Header,p::PublishHeader>)
            if(progress(a).publication_headers!=a.plan.coordinator.policy.participants())return rejectPublication(a);
        if constexpr(std::is_same_v<Header,p::ReadyHeader>)
            if(progress(a).publication_acquired!=a.plan.coordinator.policy.participants())return rejectPublication(a);
    }
    if(!a.plan.offsets.seal_receipt || !a.coordinator.plan.native.ingress_frontier ||
       owner!=cursor || owner>=a.plan.coordinator.policy.participants() ||
       (stage(a)!=ResponseStage::Consumed && stage(a)!=ResponseStage::Aborted) ||
       original.phase!=p::LookupPhase::Source || original.status!=Status::Success ||
       outcome.version!=n::Version || !n::same(outcome.interval,original.interval) || !n::same(outcome.source,original.source))return rejectPublication(a);
    if(kind!=n::headerKind<Header>() || record.owner!=owner || !n::same(record.interval,original.interval) ||
       static_cast<std::uint32_t>(record.status)>static_cast<std::uint32_t>(Status::Cancelled) ||
       (record.status==Status::Success && !n::same(record.source,original.source)))return rejectPublication(a);
    n::at<Header>(a.coordinator.native,a.coordinator.plan.native.ingress_frontier)[owner]=record;++cursor;
    if(record.status!=Status::Success)return rejectPublication(a,record.status==Status::Cancelled?ImmutableLookupStatus::Cancelled:ImmutableLookupStatus::InvalidAcknowledgement);
    return true;
}
/** @brief Parse only an actual remote60-byte receive; a local owner must use its original typed publication witness. */
template<class Header> QFN_ADAPTIVE_HD bool importPublicationWire(const DeviceArgs& a,std::uint32_t owner)
{
    if(owner==a.plan.local_owner || !a.plan.offsets.seal_receipt)return rejectPublication(a);
    const auto& wire=*n::at<n::NominalHeaderWire<Header>>(a.native,a.plan.offsets.seal_receipt);
    return importPublicationRecord(a,owner,wire.record,wire.kind);
}
/**
 * @brief Import the actual local56-byte publication and device-authored kind from the original owner root.
 * @note No self transfer, envelope copy or equivalent-layout inference occurs. The immutable native plan
 * identifies the original local root; its producer kind, exact byte publication and original active frame
 * authenticate nominal intent before the shared parser can mutate the coordinator slot/cursor.
 */
template<class Header> QFN_ADAPTIVE_HD bool importLocalPublication(const DeviceArgs& a,std::uint32_t owner)
{
    if(owner!=a.plan.local_owner || owner>=a.plan.coordinator.policy.participants() || !a.owner.frame || !a.owner.native ||
       a.owner.plan.owner.coordinate()!=owner)return rejectPublication(a);
    const auto& produced=n::progress(a.owner);const auto& record=*n::at<Header>(a.owner.native,a.owner.plan.native.header);
    const auto& physical=*n::at<p::OwnerFrame>(a.owner.frame,a.owner.plan.frame.control);const auto& allocation=n::allocationFrame(a.owner);
    constexpr auto expected_phase = std::is_same_v<Header, p::PublishHeader>
        ? p::OwnerPhase::FillValidated : p::OwnerPhase::Publishable;
    if(produced.header_kind!=n::headerKind<Header>() || produced.header_bytes!=sizeof(Header) ||
       !n::same(record.interval,physical.interval) || !n::same(record.source,physical.source) ||
       (record.status==Status::Success && (physical.status!=Status::Success || physical.phase!=expected_phase ||
        allocation.phase!=p::AllocationPhase::SourceSealed || !n::same(allocation.lease.interval,physical.interval))))return rejectPublication(a);
    if(record.status==Status::Success){
        if constexpr(std::is_same_v<Header,p::ReadyHeader>){if(physical.reservations)return rejectPublication(a);}
        else if(physical.reservations!=allocation.count)return rejectPublication(a);
    }
    return importPublicationRecord(a,owner,record,produced.header_kind);
}
/**
 * @brief Authenticate complete publication/acquisition consensus without advancing original physical or reader state.
 * @return True only for every exact original success record and current truthful retired source window.
 * @note The imported count is protocol progress, never a physical row/pin counter; all private responses stay retained.
 */
template<class Header> QFN_ADAPTIVE_HD bool acquirePublicationConsensus(const DeviceArgs& a)
{
    const auto& original=*n::coordinatorView(a.coordinator).frame;const auto& outcome=frontier(a);const auto& io=protocol(a);const auto count=a.plan.coordinator.policy.participants();
    if(outcome.status!=Status::Success)return false;
    if constexpr(!std::is_same_v<Header,p::PublishHeader>)
        if(progress(a).publication_headers!=count)return rejectPublication(a);
    if constexpr(std::is_same_v<Header,p::ReadyHeader>)
        if(progress(a).publication_acquired!=count)return rejectPublication(a);
    if(stage(a)!=ResponseStage::Consumed || publicationCursor<Header>(a)!=count ||
       !a.coordinator.plan.native.ingress_frontier || original.phase!=p::LookupPhase::Source || original.status!=Status::Success ||
       outcome.version!=n::Version || !n::same(outcome.interval,original.interval) || !n::same(outcome.source,original.source) ||
       io.issuedSequence!=io.consumedSequence ||
       n::sourceGate(n::SourceCondition::AfterSuccess,&outcome,original.interval,a.plan.coordinator.table,original.source)!=n::SourceGate::Run ||
       (outcome.source.table?(io.idCount==0 || outcome.source.table!=a.plan.coordinator.table ||
           outcome.source.generation!=io.admissionGeneration || outcome.source.sequence!=io.consumedSequence):
           (io.idCount || outcome.source.generation || outcome.source.sequence)))return rejectPublication(a);
    const auto* records=n::at<Header>(a.coordinator.native,a.coordinator.plan.native.ingress_frontier);
    for(std::uint32_t owner=0;owner<count;++owner)
        if(records[owner].owner!=owner || records[owner].status!=Status::Success || !n::same(records[owner].interval,original.interval) || !n::same(records[owner].source,original.source))return rejectPublication(a);
    return true;
}
/**
 * @brief Acquire only the complete original Ready frontier before the success-only coordinator publication.
 * @param a Same original source/coordinator and canonical controller binding; private rows remain retained.
 * @param headers Actual ordered nominal Ready records from each original participant-local producer.
 * @param count Exact original cohort size, including C0 participants.
 * @return True only for successful actual source retirement and all exact Ready records.
 * @note Typed graph bindings authenticate the device-authored nominal header-kind witness before this wire parser.
 * A rejected/cancelled receipt updates only the existing source error outcome; it never fabricates Ready,
 * clears the coordinator frame, acknowledges another response or advances private-row reader retirement.
 */
QFN_ADAPTIVE_HD inline bool acquireReadyConsensus(const DeviceArgs& a,const p::ReadyHeader* headers,std::uint32_t count)
{
    const auto& original=*n::coordinatorView(a.coordinator).frame;const auto& outcome=frontier(a);const auto& io=protocol(a);
    if(outcome.status!=Status::Success)return false; // Preserve the original actual source error/abort authority.
    if(stage(a)!=ResponseStage::Consumed || !headers || count!=a.plan.coordinator.policy.participants() ||
       original.phase!=p::LookupPhase::Source || original.status!=Status::Success ||
       outcome.version!=n::Version || !n::same(outcome.interval,original.interval) ||
       !n::same(outcome.source,original.source) || io.issuedSequence!=io.consumedSequence ||
       n::sourceGate(n::SourceCondition::AfterSuccess,&outcome,original.interval,a.plan.coordinator.table,original.source)!=n::SourceGate::Run ||
       (outcome.source.table?(io.idCount==0 || outcome.source.table!=a.plan.coordinator.table ||
           outcome.source.generation!=io.admissionGeneration || outcome.source.sequence!=io.consumedSequence):
           (io.idCount || outcome.source.generation || outcome.source.sequence)))
        return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    for(std::uint32_t owner=0;owner<count;++owner){const auto& actual=headers[owner];
        if(actual.owner!=owner || !n::same(actual.interval,original.interval) ||
           !n::same(actual.source,original.source) || actual.status!=Status::Success)
            return reject(a,actual.status==Status::Cancelled?ImmutableLookupStatus::Cancelled:ImmutableLookupStatus::InvalidAcknowledgement);
    }
    return beforeReady(a);
}

} // namespace llaminar2::qwen4exp::pooled_source
