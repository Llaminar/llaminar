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
#include "kernels/common/SamplingMath.h"
#include "PooledImmutableNativeOps.h"
#include "ImmutableLookupKernels.h"
#include "ImmutableSubordinateLookupBinding.h"
#include "QwenAdaptiveTerminalCounters.h"
namespace llaminar2::qwen4exp::pooled_source {
namespace n=pooled_native;
namespace p=n::p;
using adaptive_lookup::Status;
/** @brief Native source-reader ownership; each transition is an ordered production kernel. */
/** @brief One original source/reader/terminal phase word; terminal reuse starts only after actual reader retirement. */
enum class ResponseStage:std::uint64_t {Idle,Planned,Published,Acquired,Verified,Rejected,Private,Consumed,Aborted,ReadersComplete,
    ReadPlanned,RowsReady,ReleasesPlanned,ReleasesComplete,RetiredComplete};
static_assert(static_cast<std::uint64_t>(ResponseStage::ReadersComplete)==9 &&
    static_cast<std::uint64_t>(ResponseStage::ReadPlanned)==10 && static_cast<std::uint64_t>(ResponseStage::RowsReady)==11 &&
    static_cast<std::uint64_t>(ResponseStage::ReleasesPlanned)==12 && static_cast<std::uint64_t>(ResponseStage::RetiredComplete)==14);
/** @brief Source protocol cursors only: authenticate ordered seal receipts and nonconcurrent packet producers.
 * Neither field shadows a physical owner grant, pin, slot, row or capacity balance.
 * The sole source-device writer resets these within the existing native metadata root.
 */
struct SourceProgress {
    std::uint32_t seals=0,packets=0;
    std::uint32_t publication_headers=0,publication_acquired=0,ready_headers=0;
    std::uint32_t read_owners=0,read_rows=0;
};
static_assert(sizeof(SourceProgress)==28);
static_assert(offsetof(SourceProgress,publication_headers)==8 && offsetof(SourceProgress,publication_acquired)==12 && offsetof(SourceProgress,ready_headers)==16);
static_assert(offsetof(SourceProgress,read_owners)==20 && offsetof(SourceProgress,read_rows)==24);
static_assert(PooledSourceOffsets{}.progress+sizeof(SourceProgress)<=PooledSourceOffsets{}.observation);
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
/** @return Original source protocol progress in declared existing metadata padding, never a residency ledger. */
QFN_ADAPTIVE_HD inline SourceProgress& progress(const DeviceArgs& a)
{return *n::at<SourceProgress>(a.native,a.plan.offsets.progress);}
/** @return Exact service ticket invocation published once at admission, independent of model state. */
QFN_ADAPTIVE_HD inline std::uint64_t& invocation(const DeviceArgs& a)
{return *n::at<std::uint64_t>(a.native,a.plan.offsets.invocation);}
/** @return Whether the original request has no first error; parent control owns forward liveness.
 * Ok==0/error==0 is valid standalone prefill before response admission. This
 * predicate never admits a controller, clears an error, or creates new state.
 */
QFN_ADAPTIVE_HD inline bool forwardValid(const DeviceArgs& a)
{return a.healthy && a.error && *a.error==static_cast<int>(sampling_math::DeviceGenerationError::None);}
/** @return Current source-response reader lifecycle, never an owner occupancy or pin shadow. */
QFN_ADAPTIVE_HD inline ResponseStage& stage(const DeviceArgs& a)
{return *n::at<ResponseStage>(a.native,a.plan.offsets.response_stage);}
/** @brief Admit only the actual new retained-worker I/O tuple, preserving sequence history and request topology. */
QFN_ADAPTIVE_HD inline bool admitProtocol(const DeviceArgs& a,std::uint64_t generation,std::uint64_t ticket)
{
    auto& io=protocol(a);
    if(!generation || !ticket || generation<=io.admissionGeneration || io.issuedSequence!=io.consumedSequence ||
       (io.status && io.status!=static_cast<std::uint64_t>(ImmutableLookupStatus::Cancelled)) ||
       (stage(a)!=ResponseStage::Idle && stage(a)!=ResponseStage::ReadersComplete))return false;
    io.admissionGeneration=io.generation=generation;io.idCount=io.nativeBytes=io.status=0;invocation(a)=ticket;return true;
}
/** @return Actual original source frontier consumed by the nominal owner/coordinator phase operations. */
QFN_ADAPTIVE_HD inline n::SourceFrontier& frontier(const DeviceArgs& a)
{return *n::at<n::SourceFrontier>(a.native,a.plan.offsets.frontier);}
/**
 * @brief Fail only this source outcome; canonical request publication prevents output escape.
 * @note Keep GPU rejection inline: an outlined cold error call would make the
 * entire source argument addressable and copy it to private memory on success.
 * Host scalar replay retains the identical body and ordinary inline semantics.
 */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define QFN_SOURCE_REJECT_INLINE __forceinline__
#else
#define QFN_SOURCE_REJECT_INLINE inline
#endif
QFN_ADAPTIVE_HD QFN_SOURCE_REJECT_INLINE bool reject(const DeviceArgs& a,ImmutableLookupStatus status)
{
    protocol(a).status=static_cast<std::uint64_t>(status);
    frontier(a).status=status==ImmutableLookupStatus::Cancelled?Status::Cancelled:Status::SourceFailure;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes)=0;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes)=0;
    return false;
}
#undef QFN_SOURCE_REJECT_INLINE
/** @brief Preserve the canonical first error while suppressing successful source publication.
 * @param a Original source/controller operands; no new health authority is created.
 * @note Native failure writers race only through the original first-error CAS.
 */
QFN_ADAPTIVE_HD inline void publishFirstError(const DeviceArgs& a)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicCAS(a.error,0,static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    atomicExch(a.healthy,0);
#else
    int expected=0;
    __atomic_compare_exchange_n(a.error,&expected,
        static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure),false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE);
    __atomic_store_n(a.healthy,0,__ATOMIC_RELEASE);
#endif
}
/** @brief Reject publication without replacing a prior I/O or canonical controller error.
 * @param a Original source/controller operands. @param status New failure if no earlier I/O error exists.
 * @return False after publishing only failure and zero transport extents.
 */
QFN_ADAPTIVE_HD inline bool rejectRequestPublication(const DeviceArgs& a,ImmutableLookupStatus status)
{
    const auto first=protocol(a).status;
    reject(a,first?static_cast<ImmutableLookupStatus>(first):status);
    stage(a)=ResponseStage::Rejected;publishFirstError(a);return false;
}
/** @brief Validate the original source frontier before any request or no-request success path.
 * @param a Original source, canonical error row and sealed allocation lifetime.
 * @param cancelledInvocation System-acquired original CPU cancellation publication.
 * @param cancelledGeneration Its generation body, loaded after the publication.
 * @return True only for a valid positive source request; empty success advances to Acquired.
 * @note Cancellation is not worker retirement. Failure leaves the original
 * source lifetime for the existing retireAborted witness/drain path.
 */
QFN_ADAPTIVE_HD inline bool prepareRequestPublication(const DeviceArgs& a,std::uint64_t cancelledInvocation,
    std::uint64_t cancelledGeneration)
{
    auto& io=protocol(a);const auto& coordinator=*n::coordinatorView(a.coordinator).frame;
    if(stage(a)==ResponseStage::Rejected){publishFirstError(a);return false;}
    if(io.status)return rejectRequestPublication(a,static_cast<ImmutableLookupStatus>(io.status));
    if(stage(a)!=ResponseStage::Planned || progress(a).seals!=a.plan.coordinator.policy.participants() ||
       coordinator.phase!=p::LookupPhase::Source || !n::same(coordinator.source,frontier(a).source))
        return rejectRequestPublication(a,ImmutableLookupStatus::InvalidProtocol);
    if(!forwardValid(a) || ImmutableLookupProtocol::matchesInvocation(cancelledInvocation,cancelledGeneration,
        io.admissionGeneration,invocation(a)))return rejectRequestPublication(a,ImmutableLookupStatus::Cancelled);
    if(!io.idCount){stage(a)=ResponseStage::Acquired;return false;}
    return true;
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
    auto v=n::coordinatorView(a.coordinator);auto& io=protocol(a);
    if((stage(a)!=ResponseStage::Idle && stage(a)!=ResponseStage::ReadersComplete) || !invocation(a) ||
       !io.admissionGeneration || io.issuedSequence!=io.consumedSequence ||
       v.frame->phase!=p::LookupPhase::VictimCompleted || v.frame->status!=Status::Success || v.frame->outstanding ||
       v.frame->unique_rows>a.plan.coordinator.policy.maximumUnique() ||
       v.frame->cancelled_completion)
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
        if(v.frame->purpose==p::LookupPurpose::FutureLookupOnly && reserved(row.phase))
            return reject(a,ImmutableLookupStatus::InvalidProtocol);
        if(reserved(row.phase)) {
            const auto& grant=row.grant;
            if(grant.kind!=p::GrantKind::Reservation || grant.unique!=i || grant.row!=row.row ||
               grant.owner>=a.plan.coordinator.policy.participants() || grant.slot>=v.owners[grant.owner].slots ||
               !grant.generation || !n::same(grant.interval,interval) ||
               grant.destination_token!=std::uint64_t{i}+1)
                return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
            ++reservations;
        }
        misses+=row.phase!=p::UniquePhase::Read;
    }
    if(misses && io.issuedSequence==~std::uint64_t{0})
        return reject(a,ImmutableLookupStatus::InvalidExtent);
    std::uint32_t actual=0;
    for(std::uint32_t owner=0;owner<a.plan.coordinator.policy.participants();++owner){const auto& allocation=v.owner_plans[owner];
        if(allocation.free_received!=allocation.free_count || allocation.victim_received!=allocation.victim_count ||
            allocation.free_received>v.owners[owner].slots || allocation.victim_received>v.owners[owner].slots-allocation.free_received)
            return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
        if(allocation.free_received+allocation.victim_received>a.plan.coordinator.policy.maximumUnique()-actual)return reject(a,ImmutableLookupStatus::InvalidExtent);
        actual+=allocation.free_received+allocation.victim_received;
    }
    if(actual!=reservations)return reject(a,ImmutableLookupStatus::InvalidAcknowledgement);
    auto* coordinates=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates);std::uint32_t source=0;
    for(std::uint32_t i=0;i<v.frame->unique_rows;++i) {
        const auto& row=v.unique[i];coordinates[i]=adaptive_lookup::None;
        if(row.phase==p::UniquePhase::Read)continue;
        coordinates[i]=source;a.channel.base.ids[source]=static_cast<std::int64_t>(row.row);
        a.channel.intents[source]={v.frame->purpose==p::LookupPurpose::FutureLookupOnly?ImmutableRowReadIntent::FutureLookupOnly:
            reserved(row.phase)?ImmutableRowReadIntent::ReadAndInvalidate:ImmutableRowReadIntent::Read,
            reserved(row.phase)?row.grant.destination_token:0};++source;
    }
    io.idCount=misses;io.nativeBytes=0;io.status=static_cast<std::uint64_t>(ImmutableLookupStatus::Success);
    frontier(a)={interval,misses?p::SourceWindow{a.plan.owner.table,io.admissionGeneration,io.issuedSequence+1}:p::SourceWindow{},
        misses?n::SourceDisposition::Published:n::SourceDisposition::Unpublished,Status::Success,n::Version};
    *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes)=0;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes)=0;
    progress(a)={};stage(a)=ResponseStage::Planned;return true;
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
    const auto v=n::coordinatorView(a.coordinator);auto& io=protocol(a);
    if((stage(a)!=ResponseStage::Idle && stage(a)!=ResponseStage::ReadersComplete) || !invocation(a) || !io.admissionGeneration ||
       io.issuedSequence!=io.consumedSequence || v.frame->phase!=p::LookupPhase::VictimCompleted ||
       v.frame->unique_rows>a.plan.coordinator.policy.maximumUnique() || v.frame->interval.table!=a.plan.owner.table ||
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
/** @brief Validate one actual ordered seal receipt before allowing the original source request publication. */
QFN_ADAPTIVE_HD inline bool acquireSeal(const DeviceArgs& a,const n::OwnerSourceSealReceipt& receipt,std::uint32_t owner)
{
    const auto& expected=frontier(a);const auto& coordinator=*n::coordinatorView(a.coordinator).frame;
    if(stage(a)!=ResponseStage::Planned || owner!=progress(a).seals || owner>=a.plan.coordinator.policy.participants() ||
       receipt.owner!=owner || receipt.status!=Status::Success || receipt.version!=n::Version ||
       !n::same(receipt.interval,expected.interval) || !n::same(receipt.source,expected.source) ||
       coordinator.phase!=p::LookupPhase::Source || !n::same(coordinator.source,expected.source)){
        reject(a,ImmutableLookupStatus::InvalidAcknowledgement);stage(a)=ResponseStage::Rejected;return false;
    }
    ++progress(a).seals;return true;
}
/** @brief Preserve authenticated private-response ownership on packet failure; retirement still consumes its actual envelope.
 * A malformed/unacquired publication remains Rejected and requires exact worker retirement instead.
 */
QFN_ADAPTIVE_HD inline bool rejectDelivery(const DeviceArgs& a,ImmutableLookupStatus status)
{
    const auto previous=stage(a);reject(a,status);
    stage(a)=(previous==ResponseStage::Acquired || previous==ResponseStage::Verified || previous==ResponseStage::Private)?ResponseStage::Private:ResponseStage::Rejected;
    return false;
}
/** @return Complete original protocol/coordinator identity of an acquired source response or all-hit no-request phase. */
QFN_ADAPTIVE_HD inline bool responseIdentity(const DeviceArgs& a)
{
    const auto& io=protocol(a);const auto& outcome=frontier(a);const auto& original=*n::coordinatorView(a.coordinator).frame;
    if(!io.admissionGeneration || !invocation(a) || original.phase!=p::LookupPhase::Source ||
        outcome.version!=n::Version || !n::same(outcome.interval,original.interval) ||
        !n::same(outcome.source,original.source) ||
        (outcome.status!=Status::Success && outcome.status!=Status::SourceFailure && outcome.status!=Status::Cancelled))return false;
    if(!io.idCount)return !outcome.source.table && !outcome.source.generation && !outcome.source.sequence &&
        outcome.disposition==n::SourceDisposition::Unpublished && io.issuedSequence==io.consumedSequence;
    return outcome.source.table==a.plan.coordinator.table && outcome.source.generation==io.admissionGeneration &&
        outcome.source.sequence==io.issuedSequence && io.consumedSequence<io.issuedSequence &&
        outcome.disposition==n::SourceDisposition::Published;
}
/** @brief Validate the complete private response ACK envelope once, before any owner packet becomes visible. */
QFN_ADAPTIVE_HD inline bool validateResponse(const DeviceArgs& a)
{
    const auto v=n::coordinatorView(a.coordinator);const auto& io=protocol(a);
    if(stage(a)!=ResponseStage::Acquired || !responseIdentity(a) || io.status!=static_cast<std::uint64_t>(ImmutableLookupStatus::Success) ||
       io.idCount>a.plan.coordinator.policy.maximumUnique() || io.nativeBytes!=io.idCount*a.plan.owner.row_bytes || v.frame->phase!=p::LookupPhase::Source ||
       !n::same(v.frame->source,frontier(a).source))return rejectDelivery(a,ImmutableLookupStatus::InvalidProtocol);
    const auto* acknowledgements=n::at<ImmutableRowInvalidationAcknowledgement>(a.frame,a.plan.offsets.invalidations);
    for(std::uint32_t i=0;i<v.frame->unique_rows;++i){const auto coordinate=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates)[i];
        if(v.unique[i].phase!=p::UniquePhase::Read && (coordinate>=io.idCount || !validAcknowledgement(a,i,acknowledgements[coordinate].destination_grant)))
            return rejectDelivery(a,ImmutableLookupStatus::InvalidAcknowledgement);
    }
    stage(a)=ResponseStage::Verified;return true;
}
/** @return Packaging order only: all remote coordinates in original order, then the one local coordinate. */
QFN_ADAPTIVE_HD inline std::uint32_t packetOwner(const DeviceArgs& a,std::uint32_t ordinal)
{
    const auto local=a.plan.local_owner,count=a.plan.coordinator.policy.participants();
    if(ordinal>=count)return adaptive_lookup::None;
    return local==adaptive_lookup::None?ordinal:ordinal==count-1?local:ordinal<local?ordinal:ordinal+1;
}
/**
 * @brief Pack only actual acquired original Grants in accepted free-then-victim order, never remote owner memory.
 * @param owner Original fixed owner coordinate for this captured producer.
 * @note Counts reset even for rejected/empty phases; source packet reuse is ordered after both previous send readers.
 */
QFN_ADAPTIVE_HD inline bool prepareDelivery(const DeviceArgs& a,std::uint32_t owner)
{
    *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes)=0;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes)=0;
    const auto v=n::coordinatorView(a.coordinator);
    if(stage(a)!=ResponseStage::Verified || protocol(a).status || owner!=packetOwner(a,progress(a).packets))return rejectDelivery(a,ImmutableLookupStatus::InvalidProtocol);
    const auto& plan=v.owner_plans[owner];const auto count=std::uint64_t{plan.free_received}+plan.victim_received;
    if(plan.free_received!=plan.free_count || plan.victim_received!=plan.victim_count || count>a.plan.packet_capacity || count>v.owners[owner].slots)
        return rejectDelivery(a,ImmutableLookupStatus::InvalidExtent);
    auto* fills=n::at<n::Fill>(a.frame,a.plan.offsets.fills);
    // Full prevalidation precedes either Fill/count publication. Original accepted order is immutable.
    for(unsigned write=0;write<2;++write){std::uint32_t accepted=0;
        for(unsigned wave=0;wave<2;++wave){const auto offset=wave?plan.victim_offset:plan.free_offset,rows=wave?plan.victim_received:plan.free_received;
            if(offset>a.plan.coordinator.policy.maximumUnique() || rows>a.plan.coordinator.policy.maximumUnique()-offset)return rejectDelivery(a,ImmutableLookupStatus::InvalidExtent);
            for(std::uint32_t i=0;i<rows;++i){const auto& decision=v.decisions[offset+i];
                if(decision.unique>=v.frame->unique_rows)return rejectDelivery(a,ImmutableLookupStatus::InvalidAcknowledgement);
                const auto& row=v.unique[decision.unique];const auto& grant=row.grant;
                if(row.phase!=(wave?p::UniquePhase::VictimGranted:p::UniquePhase::FreeGranted) || row.row!=decision.row ||
                    row.decision_index!=offset+i || row.allocation_owner!=owner ||
                    decision.kind!=(wave?p::CommandKind::ReserveVictim:p::CommandKind::ReserveFree) ||
                    row.decision.ordinal.value!=decision.ordinal.value || row.decision.victim.row!=decision.victim.row ||
                    row.decision.victim.generation!=decision.victim.generation || row.decision.victim.owner!=decision.victim.owner || row.decision.victim.slot!=decision.victim.slot ||
                    grant.unique!=decision.unique || grant.owner!=owner || grant.row!=row.row || !grant.generation ||
                    grant.kind!=p::GrantKind::Reservation || grant.slot>=v.owners[owner].slots ||
                    grant.destination_token!=std::uint64_t{decision.unique}+1 || !n::same(grant.interval,v.frame->interval))
                    return rejectDelivery(a,ImmutableLookupStatus::InvalidAcknowledgement);
                if(write)fills[accepted]={grant,{frontier(a).source,grant.destination_token}};
                ++accepted;
            }
        }
    }
    *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes)=std::uint64_t{count}*sizeof(n::Fill);
    *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes)=std::uint64_t{count}*a.plan.owner.row_bytes;
    ++progress(a).packets;return true;
}
/** @brief Distinguish private authenticated response retirement from absence requiring exact worker retirement. */
enum class ResponseRetirement {NoRequest,Consume,AwaitWorker,AlreadyAborted,Invalid};
/**
 * @brief Authorize retirement from the actual original source phase, including failed packet fanout.
 * @note A failed cursor vetoes success, not necessary zero/error sends or private-reader retirement.
 */
QFN_ADAPTIVE_HD inline ResponseRetirement retirementDecision(const DeviceArgs& a)
{
    if(stage(a)==ResponseStage::Aborted)return ResponseRetirement::AlreadyAborted;
    if(stage(a)==ResponseStage::Verified && progress(a).packets!=a.plan.coordinator.policy.participants())
        rejectDelivery(a,ImmutableLookupStatus::InvalidProtocol);
    if(stage(a)==ResponseStage::Rejected)return ResponseRetirement::AwaitWorker;
    if(stage(a)!=ResponseStage::Verified && stage(a)!=ResponseStage::Private)return ResponseRetirement::Invalid;
    if(!responseIdentity(a))return ResponseRetirement::Invalid;
    return protocol(a).idCount?ResponseRetirement::Consume:ResponseRetirement::NoRequest;
}
#if defined(__CUDACC__) || defined(__HIPCC__)
#define QFN_SOURCE_WORKER_ACQUIRE __device__ __forceinline__
#else
#define QFN_SOURCE_WORKER_ACQUIRE inline
#endif
/**
 * @brief Publish Aborted only after acquiring this original worker's actual generation/invocation retirement.
 * @param observed Native system-scope loads, or recorded/actual CPU wire publications in replay.
 * @note Caller cannot pass a success bool or a numeric SourceWindow as a retirement capability.
 */
template<class Publications> QFN_SOURCE_WORKER_ACQUIRE bool retireAborted(const DeviceArgs& a,Publications& observed)
{
    const auto state=stage(a);const auto& io=protocol(a);const auto& outcome=frontier(a);const auto& original=*n::coordinatorView(a.coordinator).frame;
    if((state!=ResponseStage::Rejected && state!=ResponseStage::Published && state!=ResponseStage::Planned) || !io.admissionGeneration || !invocation(a) ||
       outcome.version!=n::Version || !n::same(outcome.interval,original.interval) ||
       (original.phase==p::LookupPhase::Source && !n::same(outcome.source,original.source)))return false;
    if(outcome.source.table){
        if(outcome.source.table!=a.plan.coordinator.table || outcome.source.generation!=io.admissionGeneration ||
            (outcome.source.sequence!=io.issuedSequence && !(io.issuedSequence==io.consumedSequence &&
                io.issuedSequence!=~std::uint64_t{0} && outcome.source.sequence==io.issuedSequence+1)))return false;
    }else if(outcome.source.generation || outcome.source.sequence || io.issuedSequence!=io.consumedSequence)return false;
    const auto ticket=observed.retiredInvocation(),generation=observed.retiredGeneration();
    if(!ImmutableLookupProtocol::matchesInvocation(ticket,generation,io.admissionGeneration,invocation(a)))return false;
    // A matching completion that raced the first load must be acquired instead.
    if(state==ResponseStage::Published && observed.completion()==io.issuedSequence)return false;
    const auto status=io.status?static_cast<ImmutableLookupStatus>(io.status):ImmutableLookupStatus::Cancelled;
    reject(a,status);frontier(a).disposition=outcome.source.table?n::SourceDisposition::Aborted:n::SourceDisposition::Unpublished;
    stage(a)=ResponseStage::Aborted;return true;
}
#undef QFN_SOURCE_WORKER_ACQUIRE
/** @brief Commit only after the backend's actual consumedSequence release (or no-request transition). */
QFN_ADAPTIVE_HD inline bool retireConsumed(const DeviceArgs& a)
{
    const auto decision=retirementDecision(a);
    if(decision!=ResponseRetirement::Consume && decision!=ResponseRetirement::NoRequest)return false;
    auto& io=protocol(a);io.consumedSequence=io.issuedSequence;
    frontier(a).disposition=decision==ResponseRetirement::Consume?n::SourceDisposition::Consumed:n::SourceDisposition::Unpublished;
    stage(a)=ResponseStage::Consumed;return true;
}
/** @return Unchanged private raw byte for one actual accepted grant in its original append order. */
QFN_ADAPTIVE_HD inline const std::byte* deliveryRow(const DeviceArgs& a,std::uint32_t accepted)
{
    const auto& grant=n::at<n::Fill>(a.frame,a.plan.offsets.fills)[accepted].original;
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
    if(!a.native || !a.owner.frame || !a.coordinator.frame || !a.healthy || !a.error || !a.plan.owner.row_bytes)return false;
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
    actual.failed_frames=!forwardValid(a) || frontier(a).status!=Status::Success;
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
    if(!a.native || !a.owner.frame || !a.owner.native || !a.coordinator.frame || !a.healthy || !a.error || !a.control)return false;
    const auto v=n::coordinatorView(a.coordinator);
    const auto& owner=*n::at<p::OwnerFrame>(a.owner.frame,a.plan.owner.frame.control);
    const auto ordinal=*n::at<std::uint64_t>(a.native,a.plan.offsets.observation_ordinal);
    auto& reported=*n::at<std::uint64_t>(a.native,a.plan.offsets.reported_ordinal);
    const auto& receipt=*n::at<p::RetiredHeader>(a.owner.native,a.plan.owner.native.header);
    if(!a.control || !a.healthy || !a.error || !ordinal || ordinal!=v.frame->ordinal || ordinal<=reported ||
       stage(a)!=ResponseStage::ReadersComplete || v.frame->phase!=p::LookupPhase::Idle ||
       owner.phase!=p::OwnerPhase::Idle || owner.held_pins || owner.reservations ||
       n::progress(a.owner).header_bytes!=sizeof(p::RetiredHeader) || !n::same(receipt.interval,v.frame->interval) ||
       !n::same(receipt.source,v.frame->source) || receipt.owner!=a.plan.owner.owner.coordinate() ||
       (receipt.status!=Status::Success && receipt.status!=Status::Cancelled))return false;
    auto actual=*n::at<QwenAdaptiveCounterTotals>(a.native,a.plan.offsets.observation);
    if(!forwardValid(a)){actual.failed_frames=1;actual.completed_live_ids=0;}
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
    if(!forwardValid(a) && frontier(a).status==Status::Success)
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
    // An independent coordinator never dereferences remote physical roots.
    // The actual complete read program has already authenticated every original
    // Grant/Ready/pin and copied only hits after immutable private miss bytes.
    if(a.plan.coordinator.policy.participants()!=1 || a.plan.local_owner==adaptive_lookup::None){
        if(stage(a)!=ResponseStage::RowsReady || progress(a).read_owners!=a.plan.coordinator.policy.participants() ||
           progress(a).read_rows!=v.frame->unique_rows)return nullptr;
        const auto coordinate=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates)[unique];
        return coordinate<v.frame->unique_rows?n::at<std::byte>(a.frame,a.plan.offsets.responses)+std::size_t{coordinate}*a.plan.owner.row_bytes:nullptr;
    }
    const auto& row=v.unique[unique];
    if(row.phase==p::UniquePhase::SourceOnly) {
        const auto coordinate=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates)[unique];
        return coordinate<protocol(a).idCount?n::at<std::byte>(a.frame,a.plan.offsets.responses)+std::size_t{coordinate}*a.plan.owner.row_bytes:nullptr;
    }
    if(!p::OwnerOperations<Slot>(n::ownerView<Slot>(a.owner)).readable(v.frame->interval,row.grant))return nullptr;
    return a.owner.cache+a.plan.owner.cache.payload+std::size_t{row.grant.slot}*a.plan.owner.slot_stride;
}
} // namespace llaminar2::qwen4exp::pooled_source
