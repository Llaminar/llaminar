/**
 * @file PooledImmutableSourceOps.h
 * @brief Original-provider request and private accepted-grant parsing shared by CPU replay and native kernels.
 *
 * Source operations read the existing sole owner/coordinator authorities. They
 * never change cache policy, slots, pins or accepted grant records. Mapped
 * acquire ordering is shared with deterministic actual-worker replay; backend
 * system-scope primitives still own every real completion/retirement observation.
 */
#pragma once
#include "PooledImmutableSourceGeometry.h"
#include "PooledImmutableNativeOps.h"
#include "ImmutableLookupKernels.h"
#include "ImmutableSubordinateLookupBinding.h"
#include "QwenAdaptiveTerminalCounters.h"
namespace llaminar2::qwen4exp::pooled_source {
namespace n=pooled_native;
namespace p=n::p;
using adaptive_lookup::Status;
/** @brief Native source-reader ownership; each transition is an ordered production kernel. */
enum class ResponseStage:std::uint64_t {Idle,Planned,Published,Acquired,Rejected,Private,Consumed,Aborted,ReadersComplete};
/** @brief Actual publication outcome; only Completion authorizes envelope acquisition/Consumed. */
enum class ResponsePublication {Pending,Completion,WorkerRetired};
#if defined(__CUDACC__) || defined(__HIPCC__)
#define QFN_SOURCE_ACQUISITION __device__ __forceinline__
#else
#define QFN_SOURCE_ACQUISITION inline
#endif
/**
 * @brief Acquire completion again after an exact worker retirement witness before classifying absence.
 * @param observed Backend system-scope loads, or the same actual CPU worker wire in deterministic replay.
 * @param expected Original issued source sequence, never a host count or inferred delivery.
 * @return Completion only from an actual matching publication; retirement alone never implies consumption.
 * @note The first completion load can precede CPU publication while the later retirement acquire
 * observes the worker after that publication. That acquire orders all prior worker writes, so
 * a fresh completion acquire must distinguish authenticated failure from a true no-response abort.
 */
template<class Publications>QFN_SOURCE_ACQUISITION ResponsePublication acquireResponsePublication(
    Publications& observed,std::uint64_t expected)
{
    if(observed.completion()==expected)return ResponsePublication::Completion;
    if(!observed.retirement())return ResponsePublication::Pending;
    return observed.completion()==expected?ResponsePublication::Completion:ResponsePublication::WorkerRetired;
}
#undef QFN_SOURCE_ACQUISITION
/** @brief Internal addresses accepted only by the service-issued, sealed native caller. */
struct DeviceArgs {
    PooledSourceParameters plan;
    n::OwnerArgs owner;
    n::CoordinatorArgs coordinator;
    std::byte *frame,*native;
    ImmutableSubordinateLookupDeviceView channel;
    int *healthy,*error;
    int* control=nullptr; ///< Original terminal reporting row; absent only in decoded parser-only replay.
};
/** @return Same persistent native I/O sequence authority used by the existing service boundary. */
QFN_ADAPTIVE_HD inline ImmutableLookupDeviceState& protocol(const DeviceArgs& a)
{return *n::at<ImmutableLookupDeviceState>(a.native,a.plan.offsets.protocol);}
/** @return Exact service ticket invocation published once at admission, independent of model state. */
QFN_ADAPTIVE_HD inline std::uint64_t& invocation(const DeviceArgs& a)
{return *n::at<std::uint64_t>(a.native,a.plan.offsets.invocation);}
/** @return Current source-response reader lifecycle, never an owner occupancy or pin shadow. */
QFN_ADAPTIVE_HD inline ResponseStage& stage(const DeviceArgs& a)
{return *n::at<ResponseStage>(a.native,a.plan.offsets.response_stage);}
/** @return Actual original source frontier consumed by the nominal owner/coordinator phase operations. */
QFN_ADAPTIVE_HD inline n::SourceFrontier& frontier(const DeviceArgs& a)
{return *n::at<n::SourceFrontier>(a.native,a.plan.offsets.frontier);}
/** @brief Fail only this source outcome; existing canonical request publication prevents output escape. */
QFN_ADAPTIVE_HD inline bool reject(const DeviceArgs& a,ImmutableLookupStatus status)
{
    protocol(a).status=static_cast<std::uint64_t>(status);
    frontier(a).status=status==ImmutableLookupStatus::Cancelled?Status::Cancelled:Status::SourceFailure;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes)=0;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes)=0;
    return false;
}
/** @return This unique coordinate has an actual accepted owner reservation rather than source-only delivery. */
QFN_ADAPTIVE_HD inline bool reserved(p::UniquePhase phase)
{return phase==p::UniquePhase::FreeGranted || phase==p::UniquePhase::VictimGranted;}
/**
 * @brief Produce exact first-occurrence source IDs/intents after both actual allocation waves complete.
 * @param a Original immutable roots/channel; no request publication is performed here.
 * @return True after complete validation and exact live writes, including all-hit/empty frames.
 * @note The original append-only grant bank is read in its accepted order, never rewritten to match source order.
 */
QFN_ADAPTIVE_HD inline bool planRequest(const DeviceArgs& a)
{
    auto v=n::coordinatorView(a.coordinator);auto& io=protocol(a);auto& allocation=n::allocationFrame(a.owner);
    if((stage(a)!=ResponseStage::Idle && stage(a)!=ResponseStage::ReadersComplete) || !invocation(a) ||
       !io.admissionGeneration || io.issuedSequence!=io.consumedSequence ||
       v.frame->phase!=p::LookupPhase::VictimCompleted || v.frame->status!=Status::Success || v.frame->outstanding ||
       v.frame->unique_rows>a.plan.coordinator.policy.maximumUnique() ||
       allocation.phase!=p::AllocationPhase::Executed || allocation.wave!=p::AllocationWave::Victim ||
       allocation.count>a.plan.owner.native.reply_capacity)
        return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto interval=v.frame->interval;
    if(interval.table!=a.plan.owner.table || !interval.admission || !interval.sequence)
        return reject(a,ImmutableLookupStatus::WrongTable);
    std::uint32_t misses=0,reservations=0;
    for(std::uint32_t i=0;i<v.frame->unique_rows;++i) {
        const auto& row=v.unique[i];
        if(row.row>=a.plan.coordinator.policy.sourceRows() ||
           (i && row.first_position<=v.unique[i-1].first_position) || row.first_position>=v.frame->original_ids ||
           (row.phase!=p::UniquePhase::Read && row.phase!=p::UniquePhase::SourceOnly && !reserved(row.phase)))
            return reject(a,ImmutableLookupStatus::InvalidProtocol);
        if(reserved(row.phase)) {
            const auto& grant=row.grant;
            if(grant.kind!=p::GrantKind::Reservation || grant.unique!=i || grant.row!=row.row ||
               grant.owner!=a.plan.owner.owner.coordinate() || !n::same(grant.interval,interval) ||
               grant.destination_token!=std::uint64_t{i}+1)
                return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
            ++reservations;
        }
        misses+=row.phase!=p::UniquePhase::Read;
    }
    if(reservations!=allocation.count || (misses && io.issuedSequence==~std::uint64_t{0}))
        return reject(a,ImmutableLookupStatus::InvalidExtent);
    const auto* original=n::grants(a.owner);
    for(std::uint32_t i=0;i<allocation.count;++i) {
        const auto& grant=original[i];
        if(grant.unique>=v.frame->unique_rows || !reserved(v.unique[grant.unique].phase) ||
           !n::same(grant,v.unique[grant.unique].grant))return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    }
    auto* coordinates=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates);std::uint32_t source=0;
    for(std::uint32_t i=0;i<v.frame->unique_rows;++i) {
        const auto& row=v.unique[i];coordinates[i]=adaptive_lookup::None;
        if(row.phase==p::UniquePhase::Read)continue;
        coordinates[i]=source;a.channel.base.ids[source]=static_cast<std::int64_t>(row.row);
        a.channel.intents[source]={reserved(row.phase)?ImmutableRowReadIntent::ReadAndInvalidate:ImmutableRowReadIntent::Read,
            reserved(row.phase)?row.grant.destination_token:0};++source;
    }
    io.idCount=misses;io.nativeBytes=0;io.status=static_cast<std::uint64_t>(ImmutableLookupStatus::Success);
    frontier(a)={interval,misses?p::SourceWindow{a.plan.owner.table,io.admissionGeneration,io.issuedSequence+1}:p::SourceWindow{},
        misses?n::SourceDisposition::Published:n::SourceDisposition::Unpublished,Status::Success,n::Version};
    *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes)=0;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes)=0;
    stage(a)=ResponseStage::Planned;return true;
}
/**
 * @brief Bind a rejected request plan to the still-valid original completed allocation interval.
 * @param a Actual owner/coordinator roots; accepted decisions/grants remain immutable.
 * @return True only when a fresh non-overlapping source lifetime can be sealed and then aborted.
 * @note Corrupt phase/sequence ownership cannot be repaired. Such failures retain
 * original roots and fail fatally rather than guessing that an old source retired.
 */
QFN_ADAPTIVE_HD inline bool rejectedPlan(const DeviceArgs& a)
{
    const auto v=n::coordinatorView(a.coordinator);auto& io=protocol(a);const auto& allocation=n::allocationFrame(a.owner);
    if((stage(a)!=ResponseStage::Idle && stage(a)!=ResponseStage::ReadersComplete) || !invocation(a) || !io.admissionGeneration ||
       io.issuedSequence!=io.consumedSequence || v.frame->phase!=p::LookupPhase::VictimCompleted ||
       v.frame->unique_rows>a.plan.coordinator.policy.maximumUnique() || allocation.phase!=p::AllocationPhase::Executed ||
       allocation.wave!=p::AllocationWave::Victim || v.frame->interval.table!=a.plan.owner.table ||
       !v.frame->interval.admission || !v.frame->interval.sequence)return false;
    bool misses=false;
    for(std::uint32_t i=0;i<v.frame->unique_rows;++i)misses|=v.unique[i].phase!=p::UniquePhase::Read;
    if(misses && io.issuedSequence==~std::uint64_t{0})return false;
    frontier(a)={v.frame->interval,misses?p::SourceWindow{a.plan.owner.table,io.admissionGeneration,io.issuedSequence+1}:p::SourceWindow{},
        misses?n::SourceDisposition::Published:n::SourceDisposition::Unpublished,
        io.status==static_cast<std::uint64_t>(ImmutableLookupStatus::Cancelled)?Status::Cancelled:Status::SourceFailure,n::Version};
    io.idCount=io.nativeBytes=0;stage(a)=ResponseStage::Rejected;return true;
}
/** @return Actual live request body; the backend releases publication only after both authorities seal its source window. */
QFN_ADAPTIVE_HD inline ImmutableLookupRequest requestBody(const DeviceArgs& a)
{
    const auto& io=protocol(a);ImmutableLookupRequest request;
    request.publication=frontier(a).source.sequence;request.sequenceComplement=~request.publication;
    request.generation=io.admissionGeneration;request.generationComplement=~request.generation;
    request.table=a.plan.owner.table;request.tableComplement=~request.table;
    request.idCount=io.idCount;request.idCountComplement=~request.idCount;
    request.idBytes=io.idCount*sizeof(std::int64_t);request.idBytesComplement=~request.idBytes;
    request.rowBytes=a.plan.owner.row_bytes;request.rowBytesComplement=~request.rowBytes;
    request.protocol=kImmutableLookupSubordinateProtocol;request.protocolComplement=~request.protocol;
    return request;
}
/**
 * @brief Distinguish an authenticated acquired source failure from malformed response metadata.
 * @param a Original current source request, retaining its issued sequence and live extents.
 * @param completion Exact metadata copied after the real system-scope acquire.
 * @param acquired Actual acquired publication word, not a requested or inferred sequence.
 * @return True only when the full envelope identifies the original request and reports a valid source outcome.
 * @note Failure codes and parser errors share an enum in the existing ABI. Validate
 * their original status/extent fields first, then validate the unchanged identity
 * through the common success-envelope parser; no malformed field is repaired.
 */
QFN_ADAPTIVE_HD inline bool acquiredCompletion(const DeviceArgs& a,const ImmutableLookupCompletion& completion,
    std::uint64_t acquired)
{
    const auto& io=protocol(a);
    if(completion.status>static_cast<std::uint64_t>(ImmutableLookupStatus::InvalidAcknowledgement) ||
       completion.statusComplement!=~completion.status || completion.nativeBytesComplement!=~completion.nativeBytes ||
       io.idCount>~std::uint64_t{0}/a.plan.owner.row_bytes ||
       completion.nativeBytes!=(completion.status==static_cast<std::uint64_t>(ImmutableLookupStatus::Success)?
           io.idCount*a.plan.owner.row_bytes:0))return false;
    auto identity=completion;identity.status=static_cast<std::uint64_t>(ImmutableLookupStatus::Success);
    identity.statusComplement=~identity.status;identity.nativeBytes=io.idCount*a.plan.owner.row_bytes;
    identity.nativeBytesComplement=~identity.nativeBytes;
    return ImmutableLookupProtocol::validateCompletion(identity,acquired,io.issuedSequence,io.admissionGeneration,
        a.plan.owner.table,io.idCount,a.plan.owner.row_bytes,kImmutableLookupSubordinateProtocol)==ImmutableLookupStatus::Success;
}
/**
 * @brief Parse one privately acquired live invalidation ACK against its original source intent and actual grant.
 * @param a Original source/native/accepted-grant roots.
 * @param unique Original global stable unique coordinate, never a slot coordinate.
 * @param acknowledgement Actual acquired source ACK copied into the corresponding private response coordinate.
 * @return True only for zero-token Read or exact original destination-token ReadAndInvalidate.
 */
QFN_ADAPTIVE_HD inline bool validAcknowledgement(const DeviceArgs& a,std::uint32_t unique,std::uint64_t acknowledgement)
{
    const auto v=n::coordinatorView(a.coordinator);
    if(unique>=v.frame->unique_rows)return false;
    const auto& row=v.unique[unique];const auto coordinate=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates)[unique];
    if(row.phase==p::UniquePhase::Read)return coordinate==adaptive_lookup::None;
    if(coordinate>=protocol(a).idCount)return false;
    return acknowledgement==(reserved(row.phase)?row.grant.destination_token:0);
}
/**
 * @brief Form private original-order owner Fill records only after the full source completion/ACK/payload producer.
 * @param a Original source frame and append-only accepted owner grant bank.
 * @return True after all validation; the exact ordered payload producer follows before Consumed publication.
 * @note This never copies a partial mapped response or overwrites any original accepted grant/decision record.
 */
QFN_ADAPTIVE_HD inline bool prepareDelivery(const DeviceArgs& a)
{
    auto v=n::coordinatorView(a.coordinator);auto& io=protocol(a);const auto& allocation=n::allocationFrame(a.owner);
    if(stage(a)!=ResponseStage::Acquired || io.status!=static_cast<std::uint64_t>(ImmutableLookupStatus::Success) ||
       io.nativeBytes!=io.idCount*a.plan.owner.row_bytes || v.frame->phase!=p::LookupPhase::Source ||
       !n::same(v.frame->source,frontier(a).source))return reject(a,ImmutableLookupStatus::InvalidProtocol);
    const auto* acknowledgements=n::at<ImmutableRowInvalidationAcknowledgement>(a.frame,a.plan.offsets.invalidations);
    for(std::uint32_t i=0;i<v.frame->unique_rows;++i) {
        const auto coordinate=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates)[i];
        if(v.unique[i].phase!=p::UniquePhase::Read &&
           (coordinate>=io.idCount || !validAcknowledgement(a,i,acknowledgements[coordinate].destination_grant)))
            return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    }
    auto* fills=n::at<n::Fill>(a.frame,a.plan.offsets.fills);const auto* original=n::grants(a.owner);
    for(std::uint32_t i=0;i<allocation.count;++i) {
        const auto& grant=original[i];
        if(grant.unique>=v.frame->unique_rows || !reserved(v.unique[grant.unique].phase) ||
           !n::same(grant,v.unique[grant.unique].grant))return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    }
    for(std::uint32_t i=0;i<allocation.count;++i)
        fills[i]={original[i],{frontier(a).source,original[i].destination_token}};
    *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes)=std::uint64_t{allocation.count}*sizeof(n::Fill);
    *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes)=std::uint64_t{allocation.count}*a.plan.owner.row_bytes;
    return true;
}
/** @return Unchanged private raw byte for one actual accepted grant in its original append order. */
QFN_ADAPTIVE_HD inline const std::byte* deliveryRow(const DeviceArgs& a,std::uint32_t accepted)
{
    const auto& grant=n::grants(a.owner)[accepted];
    const auto coordinate=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates)[grant.unique];
    return n::at<std::byte>(a.frame,a.plan.offsets.responses)+std::size_t{coordinate}*a.plan.owner.row_bytes;
}
/**
 * @brief Snapshot actual completed phase work before the original grant banks retire.
 * @param a Original source/coordinator/owner/control roots, all graph-ordered.
 * @return False for a duplicate, malformed or incompletely acquired observation.
 * @note These reporting values decide no policy. Pins and Ready state are read
 * directly at their actual owner, and allocation counts are executed receipts.
 */
template<class Slot> QFN_ADAPTIVE_HD inline bool snapshotObservation(const DeviceArgs& a)
{
    if(!a.native || !a.owner.frame || !a.coordinator.frame || !a.healthy || !a.plan.owner.row_bytes)return false;
    const auto v=n::coordinatorView(a.coordinator);const auto owner=n::ownerView<Slot>(a.owner);
    const auto& allocation=n::allocationFrame(a.owner);
    auto& ordinal=*n::at<std::uint64_t>(a.native,a.plan.offsets.observation_ordinal);
    const auto reported=*n::at<std::uint64_t>(a.native,a.plan.offsets.reported_ordinal);
    if(a.plan.owner.owner.participants()!=1 || a.plan.coordinator.policy.participants()!=1 ||
       !v.frame->ordinal || v.frame->ordinal<=reported || ordinal>reported ||
       v.frame->unique_rows>v.frame->original_ids || v.frame->original_ids>a.plan.coordinator.policy.maximumOriginal() ||
       allocation.free_count>allocation.count || allocation.count>a.plan.owner.native.reply_capacity ||
       (stage(a)!=ResponseStage::Consumed && stage(a)!=ResponseStage::Aborted))return false;
    QwenAdaptiveCounterTotals actual;actual.calls=1;
    for(std::uint32_t unique=0;unique<v.frame->unique_rows;++unique){
        if(v.unique[unique].phase==p::UniquePhase::Read)++actual.hit_rows;
        else ++actual.miss_rows;
    }
    p::OwnerOperations<Slot> physical(owner);
    for(std::uint32_t accepted=0;accepted<allocation.count;++accepted)
        actual.promotions+=physical.readable(v.frame->interval,n::grants(a.owner)[accepted]);
    actual.evictions=allocation.count-allocation.free_count;
    actual.source_bytes=protocol(a).nativeBytes;
    if(actual.source_bytes%a.plan.owner.row_bytes)return false;
    actual.source_rows=actual.source_bytes/a.plan.owner.row_bytes;
    actual.failed_frames=!*a.healthy || frontier(a).status!=Status::Success;
    actual.completed_live_ids=actual.failed_frames?0:v.frame->original_ids;
    *n::at<QwenAdaptiveCounterTotals>(a.native,a.plan.offsets.observation)=actual;
    ordinal=v.frame->ordinal;return true;
}
/**
 * @brief Publish reporting words only after every actual source/owner/coordinator retirement.
 * @param a Original retained source roots and canonical terminal controller.
 * @return False before reporting mutation for partial retirement, repeated observation or overflow.
 */
QFN_ADAPTIVE_HD inline bool observeRetired(const DeviceArgs& a)
{
    if(!a.native || !a.owner.frame || !a.owner.native || !a.coordinator.frame || !a.healthy || !a.control)return false;
    const auto v=n::coordinatorView(a.coordinator);
    const auto& owner=*n::at<p::OwnerFrame>(a.owner.frame,a.plan.owner.frame.control);
    const auto ordinal=*n::at<std::uint64_t>(a.native,a.plan.offsets.observation_ordinal);
    auto& reported=*n::at<std::uint64_t>(a.native,a.plan.offsets.reported_ordinal);
    const auto& receipt=*n::at<p::RetiredHeader>(a.owner.native,a.plan.owner.native.header);
    if(!a.control || !a.healthy || !ordinal || ordinal!=v.frame->ordinal || ordinal<=reported ||
       stage(a)!=ResponseStage::ReadersComplete || v.frame->phase!=p::LookupPhase::Idle ||
       owner.phase!=p::OwnerPhase::Idle || owner.held_pins || owner.reservations ||
       n::progress(a.owner).header_bytes!=sizeof(p::RetiredHeader) || !n::same(receipt.interval,v.frame->interval) ||
       !n::same(receipt.source,v.frame->source) || receipt.owner!=a.plan.owner.owner.coordinate() ||
       (receipt.status!=Status::Success && receipt.status!=Status::Cancelled))return false;
    auto actual=*n::at<QwenAdaptiveCounterTotals>(a.native,a.plan.offsets.observation);
    if(!*a.healthy){actual.failed_frames=1;actual.completed_live_ids=0;}
    if(!accumulateQwenAdaptiveCounters(a.control,sampling_math::kDeviceGenerationControlCount,actual))return false;
    reported=ordinal;return true;
}
static_assert(sizeof(QwenAdaptiveCounterTotals)==72);

/**
 * @brief Publish the canonical cancellation before any Reserved row can become Ready.
 * @param a Actual original source, owner/coordinator and canonical controller roots.
 * @return True for a successful consumed/source-free frontier; cancelled requests
 * retain the actual source-disposition witness while subsequent failure arms drain grants.
 * @note This fixed graph node follows acknowledged private/native copies and
 * acquired owner publication. It reads the original controller once and never
 * invents a host health shadow or a second source-consumption acknowledgement.
 */
QFN_ADAPTIVE_HD inline bool beforeReady(const DeviceArgs& a)
{
    if(!a.healthy || !a.error)return reject(a,ImmutableLookupStatus::InvalidProtocol);
    if(!*a.healthy && frontier(a).status==Status::Success)
        return reject(a,ImmutableLookupStatus::Cancelled);
    return frontier(a).status==Status::Success;
}
/** @return One real consumer's immutable row only after successful global Ready and actual source retirement. */
template<class Slot> QFN_ADAPTIVE_HD inline const std::byte* readableRow(const DeviceArgs& a,std::uint32_t unique)
{
    const auto v=n::coordinatorView(a.coordinator);
    if(v.frame->phase!=p::LookupPhase::Ready || v.frame->status!=Status::Success || unique>=v.frame->unique_rows ||
       n::sourceGate(n::SourceCondition::AfterSuccess,&frontier(a),v.frame->interval,a.plan.owner.table,v.frame->source)!=n::SourceGate::Run)
        return nullptr;
    const auto& row=v.unique[unique];
    if(row.phase==p::UniquePhase::SourceOnly) {
        const auto coordinate=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates)[unique];
        return coordinate<protocol(a).idCount?n::at<std::byte>(a.frame,a.plan.offsets.responses)+std::size_t{coordinate}*a.plan.owner.row_bytes:nullptr;
    }
    if(!p::OwnerOperations<Slot>(n::ownerView<Slot>(a.owner)).readable(v.frame->interval,row.grant))return nullptr;
    return a.owner.cache+a.plan.owner.cache.payload+std::size_t{row.grant.slot}*a.plan.owner.slot_stride;
}
} // namespace llaminar2::qwen4exp::pooled_source
