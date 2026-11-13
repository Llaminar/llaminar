/**
 * @file PooledImmutableRetirementOps.h
 * @brief Exact original release aggregation after the real distributed last-reader boundary.
 *
 * Reader retirement, graph imports and channel reads remain explicit graph nodes.
 * These participant-local bodies phase-reuse only already admitted dead banks:
 * the serial Fill bank receives Release72, pin_commands stores its dense prefix,
 * and sorted_releases remains the separate canonical stable-sort output. Original
 * physical owner release counters are never overwritten by wire observations.
 */
#pragma once
#include "PooledImmutableSourceOps.h"
namespace llaminar2::qwen4exp::pooled_source {
/** @return Whether the original unique has an actually acquired physical grant. */
QFN_ADAPTIVE_HD inline bool acquiredRelease(p::UniquePhase phase)
{return phase==p::UniquePhase::Read || phase==p::UniquePhase::FreeGranted || phase==p::UniquePhase::VictimGranted;}
/** @return Exact original aggregate receipt byte word; absent only for an admitted all-C0 cohort. */
QFN_ADAPTIVE_HD inline std::uint64_t releasedBytes(const DeviceArgs& a)
{return a.coordinator.plan.native.reply_capacity?*n::at<std::uint64_t>(a.coordinator.native,a.coordinator.plan.native.ingress_count):0;}
/**
 * @brief Reauthenticate original interval, source version, admission and actual consumed/aborted envelope.
 * @param a Original retained source and coordinator roots.
 * @return True for exactly the already retired source window; never acknowledges a pending response.
 */
QFN_ADAPTIVE_HD inline bool retirementIdentity(const DeviceArgs& a)
{
    const auto v=n::coordinatorView(a.coordinator);const auto& outcome=frontier(a);const auto& io=protocol(a);
    if(!io.admissionGeneration || !invocation(a) || outcome.version!=n::Version ||
       !n::same(outcome.interval,v.frame->interval) || !n::same(outcome.source,v.frame->source) ||
       n::sourceGate(outcome.status==Status::Success?n::SourceCondition::AfterSuccess:n::SourceCondition::AfterFailure,
           &outcome,v.frame->interval,a.plan.coordinator.table,v.frame->source)!=n::SourceGate::Run)return false;
    if(!outcome.source.table)return io.issuedSequence==io.consumedSequence;
    if(outcome.source.generation!=io.admissionGeneration)return false;
    if(outcome.disposition==n::SourceDisposition::Consumed)
        return io.issuedSequence==io.consumedSequence && outcome.source.sequence==io.consumedSequence;
    return outcome.disposition==n::SourceDisposition::Aborted &&
        (outcome.source.sequence==io.issuedSequence || (io.issuedSequence==io.consumedSequence &&
         io.issuedSequence!=~std::uint64_t{0} && outcome.source.sequence==io.issuedSequence+1));
}
/**
 * @brief Retire real successful or failed reader/output edges without clearing private rows or physical pins.
 * @param a Original source and canonical controller after the authenticated ReadCompletion.output node.
 * @return True after only the original source phase advances to ReadersComplete.
 * @note RowsReady is not a success witness: the actual output can discover cancellation or a bad
 * output span after row sealing. Both failed RowsReady and failed ReadPlanned drain through this edge.
 */
QFN_ADAPTIVE_HD inline bool retireReaders(const DeviceArgs& a)
{
    if(!a.healthy || !a.error || !retirementIdentity(a) || progress(a).read_owners!=a.plan.coordinator.policy.participants() ||
       (stage(a)!=ResponseStage::RowsReady && stage(a)!=ResponseStage::ReadPlanned) ||
       (frontier(a).status==Status::Success && forwardValid(a) && stage(a)!=ResponseStage::RowsReady))
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    if(!forwardValid(a) && frontier(a).status==Status::Success)reject(a,ImmutableLookupStatus::Cancelled);
    stage(a)=ResponseStage::ReadersComplete;return true;
}
/**
 * @brief Enter terminal bank reuse after the actual graph's output and all issued reads have retired.
 * @param a Original source/coordinator storage and device-owned reader stage, never a host count.
 * @return True after initializing only dead protocol scratch and its single original count word.
 * @note The concrete graph authenticates ReadersRetired's complete original predecessor set;
 * this body cannot infer completion from a phase flag, event from another stream, or a row count.
 */
QFN_ADAPTIVE_HD inline bool planReleases(const DeviceArgs& a)
{
    const auto v=n::coordinatorView(a.coordinator);const auto& outcome=frontier(a);
    const auto reported=*n::at<std::uint64_t>(a.native,a.plan.offsets.reported_ordinal);
    if(!retirementIdentity(a) || stage(a)!=ResponseStage::ReadersComplete || (a.plan.coordinator.policy.participants()==1 && a.plan.local_owner!=adaptive_lookup::None) ||
       !v.frame->ordinal || v.frame->ordinal<=reported ||
       (v.frame->phase!=p::LookupPhase::Ready && v.frame->phase!=p::LookupPhase::Source) || v.frame->status!=Status::Success ||
       outcome.version!=n::Version || !n::same(outcome.interval,v.frame->interval) || !n::same(outcome.source,v.frame->source) ||
       outcome.disposition==n::SourceDisposition::Published ||
       (outcome.status==Status::Success && (v.frame->phase!=p::LookupPhase::Ready ||
           (outcome.source.table?outcome.disposition!=n::SourceDisposition::Consumed:outcome.disposition!=n::SourceDisposition::Unpublished))))
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    if(a.coordinator.plan.native.reply_capacity)
        *n::at<std::uint64_t>(a.coordinator.native,a.coordinator.plan.native.ingress_count)=0;
    progress(a).packets=progress(a).seals=0;
    stage(a)=ResponseStage::ReleasesPlanned;return true;
}
/**
 * @brief Publish the exact fresh remote last-reader gate into the dead original serial64 bank.
 * @param owner Original addressed remote participant; its captured channel fixes receiver identity.
 * @return True after the actual retired source outcome and nominal kind/version are written once.
 * @note Every gate send retires before serial reverse receive reuse. No owner coordinate, padding,
 * pin shadow, count mailbox or host observation is added to this required64-byte metadata packet.
 */
QFN_ADAPTIVE_HD inline bool packReaderRetirement(const DeviceArgs& a,std::uint32_t owner)
{
    if(!a.healthy || !a.error)return reject(a,ImmutableLookupStatus::InvalidProtocol);
    if(!forwardValid(a) && frontier(a).status==Status::Success)reject(a,ImmutableLookupStatus::Cancelled);
    if(!retirementIdentity(a) || stage(a)!=ResponseStage::ReleasesPlanned || progress(a).packets ||
       owner>=a.plan.coordinator.policy.participants() || owner==a.plan.local_owner || !a.plan.offsets.seal_receipt)
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto& original=frontier(a);
    *n::at<n::ReadersRetiredGate>(a.native,a.plan.offsets.seal_receipt)={original.interval,original.source,original.status,
        original.disposition,static_cast<std::uint32_t>(n::Version),static_cast<std::uint32_t>(n::OwnerHeaderKind::ReadersRetired)};
    return true;
}
/**
 * @brief Authenticate and append one actual owner Release72 prefix before serial receive-bank reuse.
 * @param owner Original full-cohort coordinate, including C0/empty owners.
 * @param receipts Actual retained local release bank or the original acquired serial receive bank.
 * @param bytes Exact UInt64 live-byte producer, never a narrowed row count or allocation capacity.
 * @return True after all targets, duplicates, dispositions and bounds validate before aggregate writes.
 * @note sort_a is dead validation scratch here, not a physical row/pin mirror or a second ledger.
 */
QFN_ADAPTIVE_HD inline bool importReleases(const DeviceArgs& a,std::uint32_t owner,const p::ReleaseReceipt* receipts,std::uint64_t bytes)
{
    auto v=n::coordinatorView(a.coordinator);const auto capacity=a.coordinator.plan.native.reply_capacity;
    if(!retirementIdentity(a) || stage(a)!=ResponseStage::ReleasesPlanned || owner!=progress(a).packets || owner>=a.plan.coordinator.policy.participants() ||
       bytes%sizeof(p::ReleaseReceipt) || bytes/sizeof(p::ReleaseReceipt)>capacity ||
       bytes/sizeof(p::ReleaseReceipt)>v.owners[owner].slots || (bytes && !receipts))return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto count=static_cast<std::uint32_t>(bytes/sizeof(p::ReleaseReceipt));
    std::uint32_t expected=0;
    for(std::uint32_t unique=0;unique<v.frame->unique_rows;++unique){
        const auto& row=v.unique[unique];expected+=acquiredRelease(row.phase) && row.grant.owner==owner;
        v.sort_a[unique].unique=adaptive_lookup::None;
    }
    const auto previous=releasedBytes(a),maximum=std::uint64_t{capacity}*sizeof(p::ReleaseReceipt);
    if(count!=expected || previous%sizeof(p::ReleaseReceipt) || previous>maximum || bytes>maximum-previous)
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    for(std::uint32_t index=0;index<count;++index){
        const auto& receipt=receipts[index];const auto unique=receipt.original.unique;
        if(unique>=v.frame->unique_rows || v.sort_a[unique].unique!=adaptive_lookup::None)return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
        const auto& row=v.unique[unique];
        if(!acquiredRelease(row.phase) || receipt.original.owner!=owner || !n::same(receipt.original,row.grant) || receipt.reserved ||
           (receipt.disposition!=p::ReleaseDisposition::RetainedReady && receipt.disposition!=p::ReleaseDisposition::RecycledEmpty) ||
           ((row.phase==p::UniquePhase::Read || v.entries[row.entry].present) && receipt.disposition!=p::ReleaseDisposition::RetainedReady))
            return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
        v.sort_a[unique].unique=unique;
    }
    auto* output=n::at<p::ReleaseReceipt>(a.coordinator.native,a.coordinator.plan.native.pin_commands);
    for(std::uint32_t index=0;index<count;++index)output[previous/sizeof(p::ReleaseReceipt)+index]=receipts[index];
    if(capacity)*n::at<std::uint64_t>(a.coordinator.native,a.coordinator.plan.native.ingress_count)=previous+bytes;
    ++progress(a).packets;return true;
}
/** @brief Read the original exact UInt64 receive extent from the phase-reused source count word. */
QFN_ADAPTIVE_HD inline bool importRemoteReleases(const DeviceArgs& a,std::uint32_t owner)
{
    if(owner==a.plan.local_owner || owner>=a.plan.coordinator.policy.participants())return reject(a,ImmutableLookupStatus::InvalidProtocol);
    // C0 has no release endpoint or count message. Its immutable original geometry
    // proves an empty prefix; never consume the prior positive owner's serial word.
    const auto bytes=n::coordinatorView(a.coordinator).owners[owner].slots?
        *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes):0;
    return importReleases(a,owner,bytes?n::at<p::ReleaseReceipt>(a.frame,a.plan.offsets.fills):nullptr,bytes);
}
/** @brief Borrow the original local released prefix without a self-channel or copied physical release counter. */
template<class Slot> QFN_ADAPTIVE_HD inline bool importLocalReleases(const DeviceArgs& a)
{
    if(a.plan.local_owner==adaptive_lookup::None || !a.owner.frame || !a.owner.native ||
       a.owner.plan.owner.coordinate()!=a.plan.local_owner)return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto owner=n::ownerView<Slot>(a.owner);const auto& actual=n::progress(a.owner);
    if(!n::same(owner.frame->interval,n::coordinatorView(a.coordinator).frame->interval) ||
       (owner.frame->phase!=p::OwnerPhase::Publishable && owner.frame->phase!=p::OwnerPhase::Draining) ||
       owner.frame->held_pins || owner.frame->reservations ||
       actual.release_bytes!=std::uint64_t{actual.released}*sizeof(p::ReleaseReceipt))
        return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    return importReleases(a,a.plan.local_owner,actual.release_bytes?
        n::at<p::ReleaseReceipt>(a.owner.frame,a.owner.plan.frame.release_receipts):nullptr,actual.release_bytes);
}
/**
 * @brief Reconcile the complete original release set in the canonical stable order.
 * @return True only after every actual owner prefix and all original grant identities validate.
 * @note Success validates without cancellation; failure uses the original cancel authority.
 * Source disposition and window are never changed to make an unsealed owner appear sealed.
 */
QFN_ADAPTIVE_HD inline bool reconcileReleases(const DeviceArgs& a)
{
    if(!a.healthy || !a.error)return reject(a,ImmutableLookupStatus::InvalidProtocol);
    if(!forwardValid(a) && frontier(a).status==Status::Success)reject(a,ImmutableLookupStatus::Cancelled);
    if(!retirementIdentity(a) || stage(a)!=ResponseStage::ReleasesPlanned || progress(a).packets!=a.plan.coordinator.policy.participants())
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    auto original=a.coordinator;const auto bytes=releasedBytes(a);
    if(bytes%sizeof(p::ReleaseReceipt) || bytes/sizeof(p::ReleaseReceipt)>a.coordinator.plan.native.reply_capacity)
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto count=static_cast<std::uint32_t>(bytes/sizeof(p::ReleaseReceipt));
    original.input={bytes?n::at<p::ReleaseReceipt>(a.coordinator.native,a.coordinator.plan.native.pin_commands):nullptr,
        a.coordinator.plan.native.reply_capacity?&bytes:nullptr,nullptr,&frontier(a),n::SourceCondition::Always,nullptr};
    if(!n::sortedReleases(original,count))return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    auto operation=p::CoordinatorOperations(n::coordinatorView(a.coordinator));
    const auto* sorted=count?n::at<p::ReleaseReceipt>(a.coordinator.native,a.coordinator.plan.native.sorted_releases):nullptr;
    if(!operation.validateReleased(sorted,count))return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    if(frontier(a).status==Status::Success){
        if(n::coordinatorView(a.coordinator).frame->phase!=p::LookupPhase::Ready)return reject(a,ImmutableLookupStatus::InvalidProtocol);
    }else if(!operation.cancel(n::sourceRetirement(frontier(a).disposition),frontier(a).source,sorted,count))
        return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    stage(a)=ResponseStage::ReleasesComplete;return true;
}
/**
 * @brief Import one nominal actual retirement behind its original completed release prefix.
 * @param owner Original fixed full-cohort coordinate.
 * @param record Actual original header from the retained local producer or acquired remote60-byte envelope.
 * @param kind Actual UInt32 producer witness, never inferred from a record's layout or phase.
 * @return False before frontier writes/cursor advance for stale identity, kind, status or source window.
 */
QFN_ADAPTIVE_HD inline bool importRetiredMetadata(const DeviceArgs& a,std::uint32_t owner,const p::RetiredHeader& record,n::OwnerHeaderKind kind)
{
    const auto v=n::coordinatorView(a.coordinator);const bool cancelled=v.frame->phase==p::LookupPhase::Draining && v.frame->status==Status::Cancelled;
    if(!retirementIdentity(a) || stage(a)!=ResponseStage::ReleasesComplete || progress(a).packets!=a.plan.coordinator.policy.participants() ||
       owner!=progress(a).seals || owner>=a.plan.coordinator.policy.participants() || !a.coordinator.plan.native.ingress_frontier ||
       (v.frame->phase!=p::LookupPhase::Ready && !cancelled) || kind!=n::OwnerHeaderKind::Retired ||
       record.owner!=owner || !n::same(record.interval,v.frame->interval) ||
       (record.status!=Status::Success && !(cancelled && record.status==Status::Cancelled)) ||
       (!n::same(record.source,v.frame->source) && !(cancelled && record.status==Status::Cancelled && n::same(record.source,p::SourceWindow{}))))
        return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    n::at<p::RetiredHeader>(a.coordinator.native,a.coordinator.plan.native.ingress_frontier)[owner]=record;
    ++progress(a).seals;return true;
}
/** @brief Read exactly the original acquired60-byte envelope within the existing serial64 bank. */
QFN_ADAPTIVE_HD inline bool importRemoteRetired(const DeviceArgs& a,std::uint32_t owner)
{
    if(owner==a.plan.local_owner || !a.plan.offsets.seal_receipt)return reject(a,ImmutableLookupStatus::InvalidProtocol);
    return importRetiredMetadata(a,owner,*n::at<p::RetiredHeader>(a.native,a.plan.offsets.seal_receipt),
        *n::at<n::OwnerHeaderKind>(a.native,a.plan.offsets.seal_receipt+sizeof(p::RetiredHeader)));
}
/** @brief Borrow only the actual local Idle owner/allocation controls and original nominal retirement producer. */
template<class Slot> QFN_ADAPTIVE_HD inline bool importLocalRetired(const DeviceArgs& a)
{
    if(a.plan.local_owner==adaptive_lookup::None || !a.owner.frame || !a.owner.native ||
       a.owner.plan.owner.coordinate()!=a.plan.local_owner)return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto owner=n::ownerView<Slot>(a.owner);const auto& progress=n::progress(a.owner);
    if(owner.frame->phase!=p::OwnerPhase::Idle || owner.frame->held_pins || owner.frame->reservations ||
       n::allocationFrame(a.owner).phase!=p::AllocationPhase::Idle || progress.header_bytes!=sizeof(p::RetiredHeader))
        return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    const auto& record=*n::at<p::RetiredHeader>(a.owner.native,a.owner.plan.native.header);
    if(!n::same(record.interval,owner.frame->interval) || !n::same(record.source,owner.frame->source) || record.status!=owner.frame->status)
        return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    return importRetiredMetadata(a,a.plan.local_owner,record,progress.header_kind);
}
/**
 * @brief Snapshot reporting from the complete validated physical release set, retaining all actual observations.
 * @return True after writing the original pending72-byte observation and its invocation ordinal once.
 * @note Promotions are actual reservation releases which retained Ready, including partial failure;
 * no owner occupancy is inferred from coordinator presence or a successful global publication.
 */
QFN_ADAPTIVE_HD inline bool snapshotReleasedObservation(const DeviceArgs& a)
{
    const auto v=n::coordinatorView(a.coordinator);const auto bytes=releasedBytes(a);
    const auto ordinal=*n::at<std::uint64_t>(a.native,a.plan.offsets.observation_ordinal);
    const auto reported=*n::at<std::uint64_t>(a.native,a.plan.offsets.reported_ordinal);
    if(!retirementIdentity(a) || stage(a)!=ResponseStage::ReleasesComplete || !a.healthy || !a.error || !a.plan.owner.row_bytes ||
       progress(a).packets!=a.plan.coordinator.policy.participants() || !v.frame->ordinal ||
       v.frame->ordinal<=reported || ordinal>reported || bytes%sizeof(p::ReleaseReceipt) ||
       bytes/sizeof(p::ReleaseReceipt)>a.coordinator.plan.native.reply_capacity ||
       protocol(a).nativeBytes%a.plan.owner.row_bytes || protocol(a).nativeBytes/a.plan.owner.row_bytes>protocol(a).idCount)
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto count=static_cast<std::uint32_t>(bytes/sizeof(p::ReleaseReceipt));
    const auto* receipts=count?n::at<p::ReleaseReceipt>(a.coordinator.native,a.coordinator.plan.native.sorted_releases):nullptr;
    if(!p::CoordinatorOperations(v).validateReleased(receipts,count))return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    QwenAdaptiveCounterTotals actual;actual.calls=1;
    for(std::uint32_t unique=0;unique<v.frame->unique_rows;++unique){
        const auto phase=v.unique[unique].phase;
        if(phase==p::UniquePhase::Read)++actual.hit_rows;else ++actual.miss_rows;
        actual.evictions+=phase==p::UniquePhase::VictimGranted;
    }
    for(std::uint32_t index=0;index<count;++index)
        actual.promotions+=receipts[index].original.kind==p::GrantKind::Reservation && receipts[index].disposition==p::ReleaseDisposition::RetainedReady;
    actual.source_bytes=protocol(a).nativeBytes;actual.source_rows=actual.source_bytes/a.plan.owner.row_bytes;
    actual.failed_frames=!forwardValid(a) || frontier(a).status!=Status::Success || v.frame->status==Status::Cancelled;
    actual.completed_live_ids=actual.failed_frames?0:v.frame->original_ids;
    *n::at<QwenAdaptiveCounterTotals>(a.native,a.plan.offsets.observation)=actual;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.observation_ordinal)=v.frame->ordinal;
    return true;
}
/** @brief Finish only after actual original nominal retirement from every participant, including C0. */
QFN_ADAPTIVE_HD inline bool finishReleased(const DeviceArgs& a)
{
    if(!retirementIdentity(a) || stage(a)!=ResponseStage::ReleasesComplete || progress(a).seals!=a.plan.coordinator.policy.participants() ||
       progress(a).packets!=a.plan.coordinator.policy.participants())return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto* records=n::at<p::RetiredHeader>(a.coordinator.native,a.coordinator.plan.native.ingress_frontier);
    if(!p::CoordinatorOperations(n::coordinatorView(a.coordinator)).finish(records,a.plan.coordinator.policy.participants()))
        return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    stage(a)=ResponseStage::RetiredComplete;return true;
}
/**
 * @brief Commit the retained reporting tuple only after real all-owner Finish and before original reset/close.
 * @return True after the canonical controller's sole reporting writer commits exactly once.
 * @note Returning to ReadersComplete preserves the existing reset/close state boundary; the reported
 * ordinal then prevents the same source invocation from re-entering terminal reuse or double counting.
 */
QFN_ADAPTIVE_HD inline bool observeReleased(const DeviceArgs& a)
{
    const auto v=n::coordinatorView(a.coordinator);const auto ordinal=*n::at<std::uint64_t>(a.native,a.plan.offsets.observation_ordinal);
    auto& reported=*n::at<std::uint64_t>(a.native,a.plan.offsets.reported_ordinal);
    if(!retirementIdentity(a) || stage(a)!=ResponseStage::RetiredComplete || !a.control || !a.healthy || !a.error || v.frame->phase!=p::LookupPhase::Idle ||
       progress(a).seals!=a.plan.coordinator.policy.participants() || progress(a).packets!=a.plan.coordinator.policy.participants() ||
       !ordinal || ordinal!=v.frame->ordinal || ordinal<=reported)return reject(a,ImmutableLookupStatus::InvalidProtocol);
    auto actual=*n::at<QwenAdaptiveCounterTotals>(a.native,a.plan.offsets.observation);
    if(!forwardValid(a)){actual.failed_frames=1;actual.completed_live_ids=0;}
    if(!accumulateQwenAdaptiveCounters(a.control,sampling_math::kDeviceGenerationControlCount,actual))
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    reported=ordinal;stage(a)=ResponseStage::ReadersComplete;return true;
}
} // namespace llaminar2::qwen4exp::pooled_source
