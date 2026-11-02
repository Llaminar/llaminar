/**
 * @file PooledImmutableReadOps.h
 * @brief Original pinned-hit validation and private original-order row assembly.
 *
 * Physical slots/pins remain OwnerOperations-owned. Read metadata reuses the
 * original command, validation, packet, response and protocol-padding banks;
 * neither decoded grants nor row coordinates become another residency ledger.
 * Complete-prefix validation precedes every payload writer. Source miss I/O
 * counts/sequences and native quantized bytes/codebooks are never rewritten.
 */
#pragma once
#include "PooledImmutableSourceOps.h"
#include <limits>
namespace llaminar2::qwen4exp::pooled_native {
/** @brief Nominal physical-read result; equal-sized earlier headers are foreign. */
struct HitReadHeader {
    p::Interval interval;
    p::SourceWindow source;
    std::uint32_t owner;
    adaptive_lookup::Status status;
};
/** @brief Exact64 wire; the actual producer witness shares the bounded schema-version tail. */
struct HitReadReceipt {
    HitReadHeader record;
    std::uint32_t version;
    OwnerHeaderKind kind;
};
static_assert(Version<=std::numeric_limits<std::uint32_t>::max());
static_assert(sizeof(HitReadHeader)==56 && sizeof(HitReadReceipt)==64);
static_assert(offsetof(HitReadReceipt,version)==56 && offsetof(HitReadReceipt,kind)==60);
static_assert(sizeof(OwnerHeaderKind)==4);
/** @return This actual header producer's original nominal kind, never inferred from equal layout. */
template<> QFN_ADAPTIVE_HD constexpr OwnerHeaderKind headerKind<HitReadReceipt>()
{return OwnerHeaderKind::HitRead;}
/** @brief Reset only the next read publication before any current/gate/parser return. */
QFN_ADAPTIVE_HD inline void resetHitRead(const OwnerArgs& a)
{
    progress(a).delivery=Delivery::None;
    if(a.plan.native.ingress)at<OwnerIngress>(a.native,a.plan.native.ingress)->payload_bytes=0;
}
/** @return Nonwrapping actual decoded prefix bounds; never a storage/admission decision. */
QFN_ADAPTIVE_HD inline bool readSpan(const void* pointer,std::uint64_t bytes)
{
    const auto value=reinterpret_cast<std::uintptr_t>(pointer);
    return !bytes || (value && bytes<=~std::uintptr_t{0}-value);
}
/** @return Exact decoded prefix overlap, treating malformed spans as invalid. */
QFN_ADAPTIVE_HD inline bool readOverlap(const void* left,std::uint64_t bytes,const void* right,std::uint64_t extent)
{
    if(!bytes || !extent)return false;
    if(!readSpan(left,bytes) || !readSpan(right,extent))return true;
    const auto a=reinterpret_cast<std::uintptr_t>(left),b=reinterpret_cast<std::uintptr_t>(right);
    return a<b+extent && b<a+bytes;
}
/**
 * @brief Read the original acquired Grant64 scalar without narrowing byte counts.
 * @param a Same original owner roots and fixed recorded input aliases.
 * @param count Receives only a checked bounded record count, never a live byte shadow.
 * @return False before reading records for partial/foreign/overlapping prefixes.
 */
QFN_ADAPTIVE_HD inline bool hitReadCount(const OwnerArgs& a,std::uint32_t& count)
{
    count=0;
    if(a.input.record_layout!=RecordLayout::Compact || a.input.allocation_slice || a.input.record_origin || a.input.record_capacity)
        return false;
    if(!a.plan.native.reply_capacity)return !a.input.records && !a.input.record_bytes;
    if(!a.input.record_bytes || reinterpret_cast<std::uintptr_t>(a.input.record_bytes)%alignof(std::uint64_t) ||
       !recordCount<p::Grant>(a,count) || (a.input.records && reinterpret_cast<std::uintptr_t>(a.input.records)%alignof(p::Grant)))return false;
    const auto bytes=std::uint64_t{count}*sizeof(p::Grant);
    if(!readSpan(a.input.records,bytes))return false;
    if(a.plan.native.ingress){
        const auto* ingress=at<OwnerIngress>(a.native,a.plan.native.ingress);
        // Transport owns this exact incoming count and original commands bank.
        if(a.input.record_bytes!=&ingress->record_bytes || a.input.records!=a.frame+a.plan.frame.commands)return false;
    }else{
        // Local requests are projected from the same coordinator native bank.
        // They cannot alias any owner output/sort/cache region while it is used.
        if(readOverlap(a.input.records,bytes,a.frame,a.plan.frame.bytes) ||
           readOverlap(a.input.records,bytes,a.native,a.plan.native.bytes) ||
           readOverlap(a.input.records,bytes,a.cache,a.plan.cache.bytes))return false;
    }
    return true;
}
/**
 * @brief Validate every requested hit against its original initial pin before any row copy.
 * @tparam Slot Original source row32/row64 slot encoding.
 * @param a Current/gate-authenticated actual owner; generic dispatcher resets output first.
 * @return True only after complete physical/read/grant validation, including zero work.
 * @note Original pin order may differ from unique order. Existing dead validation
 * scratch sorts indices, preserving the immutable pin and requested Grant banks.
 */
template<class Slot> QFN_ADAPTIVE_HD bool validateHitRead(const OwnerArgs& a)
{
    const auto view=ownerView<Slot>(a);auto physical=p::OwnerOperations<Slot>(view);
    std::uint32_t count=0;
    if(view.frame->phase!=p::OwnerPhase::Publishable || view.frame->status!=Status::Success ||
       view.frame->initial_pin_count>a.plan.native.reply_capacity || !hitReadCount(a,count) ||
       count>view.frame->initial_pin_count || !a.plan.row_bytes ||
       (count && a.plan.row_bytes>~std::uint64_t{0}/count))return fail<Slot>(a);
    const auto pins=view.frame->initial_pin_count;
    if(!pins){progress(a).delivery=Delivery::Validated;return true;}
    auto* first=at<std::uint64_t>(a.frame,a.plan.frame.validation_a);
    auto* second=at<std::uint64_t>(a.frame,a.plan.frame.validation_b);
    auto* bins=at<std::uint32_t>(a.frame,a.plan.frame.validation_counts);
    // All physical reads precede scratch changes, and scratch is phase-local.
    for(std::uint32_t i=0;i<pins;++i){
        const auto& grant=view.initial_pins[i];
        if(grant.kind!=p::GrantKind::Read || grant.unique>=a.input.invocation->unique_rows ||
           !physical.readable(a.input.invocation->interval,grant))return fail<Slot>(a);
    }
    for(std::uint32_t i=0;i<pins;++i)first[i]=(std::uint64_t{view.initial_pins[i].unique}<<32)|i;
    auto* input=first;auto* output=second;
    for(unsigned pass=4;pass<8;++pass){
        for(unsigned bin=0;bin<256;++bin)bins[bin]=0;
        for(std::uint32_t i=0;i<pins;++i)++bins[(input[i]>>(8*pass))&255];
        std::uint32_t cursor=0;
        for(unsigned bin=0;bin<256;++bin){const auto size=bins[bin];bins[bin]=cursor;cursor+=size;}
        for(std::uint32_t i=0;i<pins;++i)output[bins[(input[i]>>(8*pass))&255]++]=input[i];
        auto* swap=input;input=output;output=swap;
    }
    for(std::uint32_t i=1;i<pins;++i)if(input[i]>>32==input[i-1]>>32)return fail<Slot>(a);
    const auto* records=static_cast<const p::Grant*>(a.input.records);std::uint32_t pin=0;
    for(std::uint32_t i=0;i<count;++i){
        const auto& grant=records[i];
        if(grant.kind!=p::GrantKind::Read || (i && grant.unique<=records[i-1].unique))return fail<Slot>(a);
        while(pin<pins && (input[pin]>>32)<grant.unique)++pin;
        if(pin==pins || (input[pin]>>32)!=grant.unique ||
           !same(grant,view.initial_pins[static_cast<std::uint32_t>(input[pin])]))return fail<Slot>(a);
        ++pin;
    }
    if(a.plan.native.ingress)at<OwnerIngress>(a.native,a.plan.native.ingress)->payload_bytes=std::uint64_t{count}*a.plan.row_bytes;
    progress(a).delivery=Delivery::Validated;return true;
}
/**
 * @return Actual read-only success gate or truthful failed-source skip.
 * @param a Original owner/current invocation and retained outcome producer.
 * @note A failed source can leave an owner genuinely unsealed with window0.
 * That permits only zero outgoing payload/error metadata; it never authorizes
 * a read, changes the owner's source window, or certifies physical retirement.
 */
template<class Slot> QFN_ADAPTIVE_HD SourceGate hitReadGate(const OwnerArgs& a)
{
    const auto& frame=*ownerView<Slot>(a).frame;const auto* h=a.input.outcome;
    if(a.input.condition==SourceCondition::AfterSuccess && h && h->status!=Status::Success){
        const bool zero=!frame.source.table && !frame.source.generation && !frame.source.sequence;
        const bool window=same(h->source,frame.source) ||
            (zero && allocationFrame(a).phase!=p::AllocationPhase::SourceSealed);
        const bool source=h->source.table?(h->source.table==a.plan.table && h->source.generation && h->source.sequence &&
            (h->disposition==SourceDisposition::Consumed || h->disposition==SourceDisposition::Aborted)):
            (!h->source.generation && !h->source.sequence && h->disposition==SourceDisposition::Unpublished);
        std::uint32_t count=0;
        return a.input.invocation && h->version==Version && same(h->interval,a.input.invocation->interval) &&
            (h->status==Status::SourceFailure || h->status==Status::Cancelled) && source && window &&
            frame.phase!=p::OwnerPhase::Idle && frame.phase!=p::OwnerPhase::Uninitialized && hitReadCount(a,count)?
            SourceGate::Skip:SourceGate::Invalid;
    }
    if(a.input.condition==SourceCondition::AfterFailure)return SourceGate::Invalid;
    return a.input.invocation?sourceGate(a.input.condition,h,a.input.invocation->interval,a.plan.table,frame.source):SourceGate::Invalid;
}
/** @brief Actual ReadRows32 dispatcher branch; skip failures without poisoning a valid unsealed owner. */
template<class Slot> QFN_ADAPTIVE_HD bool beginHitRead(const OwnerArgs& a)
{
    resetHitRead(a);
    if(!current<Slot>(a))return false;
    const auto gate=hitReadGate<Slot>(a);
    if(gate==SourceGate::Invalid)return fail<Slot>(a,Status::SourceFailure);
    return gate==SourceGate::Skip || validateHitRead<Slot>(a);
}
/**
 * @return One original pinned native byte after whole-prefix validation.
 * @param a Original retained owner and Grant prefix, unchanged through final copy.
 * @param byte Exact live packed byte index; callers bound it by original live row count.
 * @note Slot stride padding is never part of this packed row address arithmetic.
 */
QFN_ADAPTIVE_HD inline const std::byte* hitReadByte(const OwnerArgs& a,std::uint64_t byte)
{
    const auto& grant=static_cast<const p::Grant*>(a.input.records)[byte/a.plan.row_bytes];
    return a.cache+a.plan.cache.payload+std::uint64_t{grant.slot}*a.plan.slot_stride+byte%a.plan.row_bytes;
}
/**
 * @brief Project truthful ReadReceipt64 only after the actual ordered payload-copy leaf.
 * @tparam Slot Original native slot encoding.
 * @param a Same retained arguments as validation/copy; no count bank is copied.
 * @return Success only from validated copy retirement and the real physical frame.
 * @note Failed/skipped native bools still produce fresh error64/count0. The
 * actual dispatch invokes this after all parallel writers on the exact stream.
 */
template<class Slot> QFN_ADAPTIVE_HD bool completeHitRead(const OwnerArgs& a)
{
    const auto& frame=*ownerView<Slot>(a).frame;
    const auto gate=hitReadGate<Slot>(a);
    const bool valid=progress(a).delivery==Delivery::Validated && frame.phase==p::OwnerPhase::Publishable &&
        frame.status==Status::Success && gate==SourceGate::Run;
    const auto status=valid?Status::Success:(frame.status!=Status::Success?frame.status:
        (gate==SourceGate::Skip?a.input.outcome->status:Status::SourceFailure));
    if(valid)progress(a).delivery=Delivery::Copied;
    else if(a.plan.native.ingress)at<OwnerIngress>(a.native,a.plan.native.ingress)->payload_bytes=0;
    header(a,HitReadReceipt{{frame.interval,frame.source,a.plan.owner.coordinate(),status},static_cast<std::uint32_t>(Version),OwnerHeaderKind::None});
    // Serialize the witness written by the one original header authority.
    at<HitReadReceipt>(a.native,a.plan.native.header)->kind=progress(a).header_kind;
    return valid;
}
} // namespace llaminar2::qwen4exp::pooled_native
namespace llaminar2::qwen4exp::pooled_source {
/** @return Original phase-reused directed hit request, never a host count projection. */
QFN_ADAPTIVE_HD inline const n::DirectedSlice& hitReadSlice(const DeviceArgs& a,std::uint32_t owner)
{return n::at<n::DirectedSlice>(a.coordinator.native,a.plan.coordinator.native.pin_slices)[owner];}
/** @return Original stable Grant64 request prefix named by that actual slice. */
QFN_ADAPTIVE_HD inline const p::Grant* hitReadRequests(const DeviceArgs& a,std::uint32_t owner)
{return reinterpret_cast<const p::Grant*>(a.coordinator.native+a.plan.coordinator.native.pin_commands+hitReadSlice(a,owner).byte_offset);}
/** @return Complete retained source/coordinator tuple before any read staging mutation. */
QFN_ADAPTIVE_HD inline bool readTuple(const DeviceArgs& a)
{
    if(!a.frame || !a.native || !a.coordinator.frame || !a.coordinator.native || !a.healthy || !a.error ||
       !invocation(a) || !protocol(a).admissionGeneration || protocol(a).generation!=protocol(a).admissionGeneration)return false;
    const auto v=n::coordinatorView(a.coordinator);const auto& f=*v.frame;const auto& h=frontier(a);
    return a.frame && a.native && a.healthy && a.error && a.plan.owner.row_bytes &&
        a.coordinator.plan.identity==a.plan.coordinator.identity &&
        f.interval.table==a.plan.coordinator.table && f.interval.admission && f.interval.sequence &&
        h.version==n::Version && n::same(h.interval,f.interval) && n::same(h.source,f.source) &&
        f.unique_rows<=a.plan.coordinator.policy.maximumUnique() && f.original_ids<=a.plan.coordinator.policy.maximumOriginal() &&
        f.unique_rows<=f.original_ids && progress(a).publication_headers==a.plan.coordinator.policy.participants() &&
        progress(a).publication_acquired==a.plan.coordinator.policy.participants() &&
        progress(a).ready_headers==a.plan.coordinator.policy.participants();
}
/**
 * @brief Pack only actual initial-hit grants after the original global Ready producer.
 * @param a Original source/coordinator roots, with every preceding command reader retired.
 * @return False for malformed/missing producers; a truthful cancelled/failed frame
 * still emits zero request extents and enters the fixed read-drain program.
 * @note Validation precedes command/map writes. Original io.idCount/nativeBytes
 * and issued/consumed sequence remain the real source-miss transaction forever.
 */
QFN_ADAPTIVE_HD inline bool planReads(const DeviceArgs& a)
{
    if((stage(a)!=ResponseStage::Consumed && stage(a)!=ResponseStage::Aborted) || !readTuple(a))return false;
    const auto v=n::coordinatorView(a.coordinator);const auto participants=a.plan.coordinator.policy.participants();
    auto* slices=n::at<n::DirectedSlice>(a.coordinator.native,a.plan.coordinator.native.pin_slices);
    auto* cursors=n::at<std::uint32_t>(a.coordinator.native,a.plan.coordinator.native.pin_cursors);
    // Previous phase readers retired at the real Ready entry. Clear usable
    // outward extents first so no normal failure can resend old PinCommands.
    for(std::uint32_t owner=0;owner<participants;++owner){slices[owner]={};cursors[owner]=0;}
    progress(a).read_owners=progress(a).read_rows=0;
    if(!forwardValid(a) || frontier(a).status!=Status::Success){
        if(frontier(a).status==Status::Success)reject(a,ImmutableLookupStatus::Cancelled);
        stage(a)=ResponseStage::ReadPlanned;return true;
    }
    const auto& io=protocol(a);
    if(v.frame->phase!=p::LookupPhase::Ready || v.frame->status!=Status::Success ||
       stage(a)!=ResponseStage::Consumed || io.status || io.issuedSequence!=io.consumedSequence ||
       io.idCount>v.frame->unique_rows || (io.idCount && a.plan.owner.row_bytes>~std::uint64_t{0}/io.idCount) ||
       io.nativeBytes!=io.idCount*a.plan.owner.row_bytes ||
       n::sourceGate(n::SourceCondition::AfterSuccess,&frontier(a),v.frame->interval,a.plan.owner.table,v.frame->source)!=n::SourceGate::Run)
        return false;
    const auto* coordinates=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates);std::uint32_t misses=0;
    for(std::uint32_t unique=0;unique<v.frame->unique_rows;++unique){
        const auto& row=v.unique[unique];
        if(row.row>=a.plan.coordinator.policy.sourceRows() || row.first_position>=v.frame->original_ids ||
           (unique && row.first_position<=v.unique[unique-1].first_position) || v.original_to_unique[row.first_position]!=unique)return false;
        if(row.phase==p::UniquePhase::Read){
            const auto& g=row.grant;
            if(coordinates[unique]!=adaptive_lookup::None || g.kind!=p::GrantKind::Read || g.unique!=unique ||
               g.row!=row.row || g.owner>=participants || g.slot>=v.owners[g.owner].slots || !g.generation ||
               g.destination_token || !n::same(g.interval,v.frame->interval))return false;
        }else{
            if((row.phase!=p::UniquePhase::SourceOnly && !reserved(row.phase)) || coordinates[unique]!=misses || misses>=io.idCount)return false;
            if(reserved(row.phase)){
                const auto& g=row.grant;
                if(g.kind!=p::GrantKind::Reservation || g.unique!=unique || g.row!=row.row ||
                   g.owner>=participants || g.slot>=v.owners[g.owner].slots || !g.generation ||
                   g.destination_token!=std::uint64_t{unique}+1 || !n::same(g.interval,v.frame->interval))return false;
            }
            ++misses;
        }
    }
    if(misses!=io.idCount)return false;
    for(std::uint32_t original=0;original<v.frame->original_ids;++original)
        if(v.original_to_unique[original]>=v.frame->unique_rows)return false;
    for(std::uint32_t unique=0;unique<v.frame->unique_rows;++unique)
        if(v.unique[unique].phase==p::UniquePhase::Read)++cursors[v.unique[unique].grant.owner];
    std::uint64_t hits=0;
    for(std::uint32_t owner=0;owner<participants;++owner){
        if(cursors[owner]>v.owners[owner].slots)return false;
        hits+=cursors[owner];
    }
    if(hits>a.plan.coordinator.native.reply_capacity)return false;
    std::uint64_t end=0;
    for(std::uint32_t owner=0;owner<participants;++owner){
        slices[owner]={end,std::uint64_t{cursors[owner]}*sizeof(p::Grant)};
        end+=slices[owner].live_bytes;cursors[owner]=0;
    }
    auto* commands=n::at<p::Grant>(a.coordinator.native,a.plan.coordinator.native.pin_commands);
    for(std::uint32_t unique=0;unique<v.frame->unique_rows;++unique)if(v.unique[unique].phase==p::UniquePhase::Read){
        const auto owner=v.unique[unique].grant.owner;
        commands[slices[owner].byte_offset/sizeof(p::Grant)+cursors[owner]++]=v.unique[unique].grant;
    }
    progress(a).read_rows=misses;stage(a)=ResponseStage::ReadPlanned;return true;
}
/** @brief Immutable addressed local read view assembled from original source/owner roots on this device. */
QFN_ADAPTIVE_HD inline n::OwnerArgs localHitRead(const DeviceArgs& a,std::uint32_t owner)
{
    auto local=a.owner;
    local.input={n::at<n::Invocation>(a.coordinator.native,a.plan.coordinator.native.invocation),
        local.plan.native.reply_capacity?hitReadRequests(a,owner):nullptr,
        local.plan.native.reply_capacity?&hitReadSlice(a,owner).live_bytes:nullptr,
        nullptr,nullptr,nullptr,n::SourceCondition::AfterSuccess,&frontier(a)};
    return local;
}
/** @return The sole actual local receipt, or the original serial transport-acquired receipt. */
QFN_ADAPTIVE_HD inline const n::HitReadReceipt& hitReadReceipt(const DeviceArgs& a,std::uint32_t owner)
{return owner==a.plan.local_owner?*n::at<n::HitReadReceipt>(a.owner.native,a.plan.owner.native.header):
    *n::at<n::HitReadReceipt>(a.native,a.plan.offsets.seal_receipt);}
/** @return Actual transport count; C0/local paths have no fictitious count receive or scalar. */
QFN_ADAPTIVE_HD inline std::uint64_t hitReadPayloadBytes(const DeviceArgs& a,std::uint32_t owner)
{
    const auto v=n::coordinatorView(a.coordinator);
    if(!v.owners[owner].slots)return 0;
    if(owner==a.plan.local_owner)return n::progress(a.owner).delivery==n::Delivery::Copied &&
        hitReadReceipt(a,owner).record.status==Status::Success?
        hitReadSlice(a,owner).live_bytes/sizeof(p::Grant)*a.plan.owner.row_bytes:0;
    return *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes);
}
/** @return The addressed immutable request/epoch geometry fits the original private row bank. */
QFN_ADAPTIVE_HD inline bool readImportDeclaration(const DeviceArgs& a,std::uint32_t owner)
{
    if(stage(a)!=ResponseStage::ReadPlanned || !readTuple(a) || owner!=progress(a).read_owners ||
       owner>=a.plan.coordinator.policy.participants() ||
       progress(a).read_rows>n::coordinatorView(a.coordinator).frame->unique_rows)return false;
    const auto v=n::coordinatorView(a.coordinator);const auto& slice=hitReadSlice(a,owner);
    const auto span=std::uint64_t{a.plan.coordinator.native.reply_capacity}*sizeof(p::Grant);
    if(slice.byte_offset>span || slice.live_bytes>span-slice.byte_offset || slice.byte_offset%sizeof(p::Grant) ||
       slice.live_bytes%sizeof(p::Grant) || slice.live_bytes/sizeof(p::Grant)>v.owners[owner].slots ||
       slice.live_bytes/sizeof(p::Grant)>a.plan.packet_capacity)return false;
    const auto count=slice.live_bytes/sizeof(p::Grant);const auto* records=hitReadRequests(a,owner);
    for(std::uint32_t i=0;i<count;++i){
        const auto& g=records[i];
        if(g.unique>=v.frame->unique_rows || g.kind!=p::GrantKind::Read || g.owner!=owner ||
           (i && g.unique<=records[i-1].unique) || v.unique[g.unique].phase!=p::UniquePhase::Read ||
           !n::same(g,v.unique[g.unique].grant))return false;
    }
    if(forwardValid(a) && frontier(a).status==Status::Success){
        if(progress(a).read_rows>v.frame->unique_rows || count>v.frame->unique_rows-progress(a).read_rows)return false;
        const auto* coordinates=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates);
        for(std::uint32_t i=0;i<count;++i)if(coordinates[records[i].unique]!=adaptive_lookup::None)return false;
    }
    return owner!=a.plan.local_owner || (a.owner.plan.identity==a.plan.owner.identity &&
        a.owner.plan.owner.coordinate()==owner && !a.owner.plan.native.ingress && a.owner.frame && a.owner.native);
}
/**
 * @brief Authenticate the complete received64/count/prefix before any remote row writer.
 * @param a Original serial packet and retained request roots, untouched until completion.
 * @param owner Immutable captured owner coordinate.
 * @return True for a complete success or truthful zero/error packet; malformed/stale data rejects atomically.
 */
QFN_ADAPTIVE_HD inline bool validateReadImport(const DeviceArgs& a,std::uint32_t owner)
{
    if(!readImportDeclaration(a,owner) || (owner!=a.plan.local_owner && !a.plan.offsets.seal_receipt))return false;
    const auto& h=hitReadReceipt(a,owner);const auto v=n::coordinatorView(a.coordinator);
    const auto bytes=hitReadPayloadBytes(a,owner);
    if(h.version!=n::Version || h.kind!=n::OwnerHeaderKind::HitRead || h.record.owner!=owner ||
       !n::same(h.record.interval,v.frame->interval) ||
       static_cast<std::uint32_t>(h.record.status)>static_cast<std::uint32_t>(Status::Cancelled))return false;
    if(owner==a.plan.local_owner && (n::progress(a.owner).header_bytes!=sizeof(n::HitReadReceipt) ||
       n::progress(a.owner).header_kind!=n::OwnerHeaderKind::HitRead))return false;
    if(h.record.status!=Status::Success){
        const bool zero=!h.record.source.table && !h.record.source.generation && !h.record.source.sequence;
        return !bytes && (n::same(h.record.source,v.frame->source) || zero);
    }
    const auto expected=hitReadSlice(a,owner).live_bytes/sizeof(p::Grant)*a.plan.owner.row_bytes;
    return n::same(h.record.source,v.frame->source) && bytes==expected;
}
/**
 * @brief Validate an actual source-local owner without inventing its absent payload/ingress banks.
 * @tparam Slot Original physical slot format on this same coordinator device.
 * @return False for malformed declaration or original grant. A truthful failed
 * source is a successful no-copy gate and still reaches receipt/drain completion.
 */
template<class Slot> QFN_ADAPTIVE_HD bool prepareLocalReadImport(const DeviceArgs& a,std::uint32_t owner)
{
    if(owner!=a.plan.local_owner || !readImportDeclaration(a,owner))return false;
    if(!forwardValid(a) && frontier(a).status==Status::Success)reject(a,ImmutableLookupStatus::Cancelled);
    auto local=localHitRead(a,owner);return n::owner<n::OwnerAction::ReadRows,Slot>(local);
}
/** @return Actual validated local copy extent; source cancellation never fabricates a byte producer. */
QFN_ADAPTIVE_HD inline std::uint64_t localHitReadCopyBytes(const DeviceArgs& a,std::uint32_t owner)
{return n::progress(a.owner).delivery==n::Delivery::Validated &&
    n::at<p::OwnerFrame>(a.owner.frame,a.plan.owner.frame.control)->status==Status::Success?
    hitReadSlice(a,owner).live_bytes/sizeof(p::Grant)*a.plan.owner.row_bytes:0;}
/** @return Whether this validated packet still contributes bytes to a successful original output. */
QFN_ADAPTIVE_HD inline bool readImportCopies(const DeviceArgs& a,std::uint32_t owner)
{return forwardValid(a) && frontier(a).status==Status::Success && hitReadReceipt(a,owner).record.status==Status::Success;}
/**
 * @brief One exact destination byte after the complete owner-prefix validation.
 * @param byte Live packed hit-row index; never a message/allocation capacity.
 * @return Private response destination strictly after immutable source-miss rows.
 */
QFN_ADAPTIVE_HD inline std::byte* hitReadDestination(const DeviceArgs& a,std::uint64_t byte)
{return a.frame+a.plan.offsets.responses+std::uint64_t{progress(a).read_rows}*a.plan.owner.row_bytes+byte;}
/** @return Exact remote packet byte or original local pinned physical byte; no local staging bank exists. */
QFN_ADAPTIVE_HD inline const std::byte* hitReadSource(const DeviceArgs& a,std::uint32_t owner,std::uint64_t byte)
{
    if(owner==a.plan.local_owner)return n::hitReadByte(localHitRead(a,owner),byte);
    return a.frame+a.plan.offsets.payload+byte;
}
/**
 * @brief Commit unique-to-private row coordinates only after all actual copy writers retire.
 * @param owner Original serial owner; the next receive follows this complete reader.
 * @return False before map/cursor changes on malformed packet. Truthful errors
 * advance only the protocol drain cursor and preserve every private miss row.
 */
QFN_ADAPTIVE_HD inline bool completeReadImport(const DeviceArgs& a,std::uint32_t owner)
{
    if(!validateReadImport(a,owner))return false;
    const auto& receipt=hitReadReceipt(a,owner);
    if(receipt.record.status!=Status::Success){
        reject(a,receipt.record.status==Status::Cancelled?ImmutableLookupStatus::Cancelled:ImmutableLookupStatus::StorageFailure);
        if(*a.error==0)*a.error=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);
        *a.healthy=0;
    }else if(readImportCopies(a,owner)){
        const auto count=hitReadSlice(a,owner).live_bytes/sizeof(p::Grant);const auto* records=hitReadRequests(a,owner);
        auto* coordinates=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates);
        for(std::uint32_t i=0;i<count;++i)coordinates[records[i].unique]=progress(a).read_rows+i;
        progress(a).read_rows+=static_cast<std::uint32_t>(count);
    }
    ++progress(a).read_owners;return true;
}
/** @brief Publish the actual local-copy receipt and commit its map only after every local byte writer. */
template<class Slot> QFN_ADAPTIVE_HD bool completeLocalReadImport(const DeviceArgs& a,std::uint32_t owner)
{
    if(owner!=a.plan.local_owner)return false;
    auto local=localHitRead(a,owner);(void)n::completeHitRead<Slot>(local);
    return completeReadImport(a,owner);
}
/**
 * @brief Validate the complete assembled map before exposing any original-order decoder.
 * @return True only after every required receipt/import retired. A failed frame
 * remains ReadPlanned for the typed all-edge drain/last-output cleanup boundary.
 */
QFN_ADAPTIVE_HD inline bool sealReadRows(const DeviceArgs& a)
{
    if(stage(a)!=ResponseStage::ReadPlanned || !readTuple(a) ||
       progress(a).read_owners!=a.plan.coordinator.policy.participants())return false;
    if(!forwardValid(a) || frontier(a).status!=Status::Success){
        if(frontier(a).status==Status::Success)reject(a,ImmutableLookupStatus::Cancelled);
        return true;
    }
    const auto v=n::coordinatorView(a.coordinator);const auto* coordinates=n::at<std::uint32_t>(a.frame,a.plan.offsets.coordinates);
    if(progress(a).read_rows!=v.frame->unique_rows)return false;
    std::uint32_t misses=0;
    for(std::uint32_t unique=0;unique<v.frame->unique_rows;++unique)if(v.unique[unique].phase!=p::UniquePhase::Read){
        if(coordinates[unique]!=misses++)return false;
    }
    if(misses!=protocol(a).idCount)return false;
    std::uint32_t next=misses;
    for(std::uint32_t owner=0;owner<a.plan.coordinator.policy.participants();++owner){
        const auto* records=hitReadRequests(a,owner);const auto count=hitReadSlice(a,owner).live_bytes/sizeof(p::Grant);
        for(std::uint32_t i=0;i<count;++i)if(coordinates[records[i].unique]!=next++)return false;
    }
    if(next!=v.frame->unique_rows)return false;
    stage(a)=ResponseStage::RowsReady;return true;
}
} // namespace llaminar2::qwen4exp::pooled_source
