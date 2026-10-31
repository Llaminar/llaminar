/**
 * @file PooledImmutableNativeOps.h
 * @brief Shared exact native phase parsers over final owner/coordinator authorities.
 *
 * CPU replay and CUDA/HIP metadata leaves execute these same operations. These
 * functions consume decoded acquired roots; they do not issue source requests,
 * launch transport, fabricate controller ownership or allocate storage. Native
 * payload copies are separate parallel producers between validation and Ready.
 */
#pragma once
#include "PooledImmutableNativeTypes.h"
#include "PooledImmutableCoordinatorOps.h"
namespace llaminar2::qwen4exp::pooled_native
{
using adaptive_lookup::Status;
/** @return Typed subrange in an original checked native root. */
template<class T> QFN_ADAPTIVE_HD T* at(std::byte* root,std::size_t offset)
{return reinterpret_cast<T*>(root+offset);}
/** @return Complete original interval equality. */
QFN_ADAPTIVE_HD inline bool same(p::Interval a,p::Interval b)
{return a.table==b.table && a.admission==b.admission && a.sequence==b.sequence;}
/** @return Exact original source window equality. */
QFN_ADAPTIVE_HD inline bool same(p::SourceWindow a,p::SourceWindow b)
{return a.table==b.table && a.generation==b.generation && a.sequence==b.sequence;}
/** @return Exact immutable grant, including global unique token and full owner-local generation. */
QFN_ADAPTIVE_HD inline bool same(const p::Grant& a,const p::Grant& b)
{
    return same(a.interval,b.interval) && a.row==b.row && a.generation==b.generation &&
        a.destination_token==b.destination_token && a.unique==b.unique && a.owner==b.owner &&
        a.slot==b.slot && a.kind==b.kind;
}
/** @return Sole persistent physical owner view; no host mirror or alternate pin/free state exists. */
template<class Slot> QFN_ADAPTIVE_HD p::OwnerView<Slot> ownerView(const OwnerArgs& a)
{
    return {a.cache?at<Slot>(a.cache,a.plan.cache.slots):nullptr,
        a.cache?at<p::PhysicalOwnerControl>(a.cache,a.plan.cache.control):nullptr,
        at<p::OwnerFrame>(a.frame,a.plan.frame.control),a.plan.table,a.plan.owner,
        at<p::Grant>(a.frame,a.plan.frame.initial_pins)};
}
/** @return Mandatory allocation frame retaining the single lease and both immutable wave prefixes. */
QFN_ADAPTIVE_HD inline p::AllocationFrame& allocationFrame(const OwnerArgs& a)
{return *at<p::AllocationFrame>(a.frame,a.plan.frame.allocation_control);}
/** @return The actual append-only grant bank. */
QFN_ADAPTIVE_HD inline p::Grant* grants(const OwnerArgs& a)
{return at<p::Grant>(a.frame,a.plan.frame.replies);}
/** @return Native-only protocol output validity; this never supplies physical occupancy or pin permission. */
QFN_ADAPTIVE_HD inline OwnerProgress& progress(const OwnerArgs& a)
{return *at<OwnerProgress>(a.native,a.plan.native.progress);}
/** @return One actual typed owner plan executor over original admitted banks. */
template<class Slot> QFN_ADAPTIVE_HD p::OwnerPlanOperations<Slot> allocation(const OwnerArgs& a)
{
    return {ownerView<Slot>(a),allocationFrame(a),
        {at<p::AllocationDecision>(a.frame,a.plan.frame.accepted_plan),grants(a),
            {at<std::uint64_t>(a.frame,a.plan.frame.validation_a),at<std::uint64_t>(a.frame,a.plan.frame.validation_b),
                at<std::uint32_t>(a.frame,a.plan.frame.validation_counts)},a.plan.owner.maximumUnique()}};
}
/** @brief Preserve the actual owner's first error and suppress only the current outward publication. */
template<class Slot> QFN_ADAPTIVE_HD bool fail(const OwnerArgs& a,Status why=Status::Invariant)
{
    auto* frame=ownerView<Slot>(a).frame;
    if(frame->status==Status::Success)frame->status=why;
    progress(a).reply_bytes=0;
    return false;
}
/** @brief Write only sizeof(Header) live bytes; the rest of the admitted 64-byte header slot remains unused. */
template<class Header> QFN_ADAPTIVE_HD void header(const OwnerArgs& a,const Header& value)
{
    static_assert(sizeof(Header)<=64);
    *at<Header>(a.native,a.plan.native.header)=value;
    progress(a).header_bytes=sizeof(Header);
}
/** @return Actual acquired original invocation is compatible before any owner mutation. */
template<class Slot> QFN_ADAPTIVE_HD bool current(const OwnerArgs& a,bool beginning=false)
{
    const auto* h=a.input.invocation;
    if(!h || h->version!=Version || h->status!=Status::Success ||
        h->source_storage_generation!=a.plan.source_storage_generation || h->interval.table!=a.plan.table ||
        !h->interval.admission || !h->interval.sequence || h->original_ids>a.plan.maximum_original ||
        h->unique_rows>a.plan.owner.maximumUnique() || h->unique_rows>h->original_ids ||
        ((h->original_ids==0)!=(h->unique_rows==0)) || progress(a).initialized!=2 ||
        progress(a).setup_identity!=a.plan.identity ||
        (!beginning && !same(h->interval,ownerView<Slot>(a).frame->interval)))return fail<Slot>(a);
    return true;
}
/** @return Exact bounded live record count from its acquired byte scalar. */
template<class Record> QFN_ADAPTIVE_HD bool recordCount(const OwnerArgs& a,std::uint32_t& count)
{
    if(!a.input.record_bytes)return false;
    const auto bytes=*a.input.record_bytes;
    if(bytes%sizeof(Record) || bytes/sizeof(Record)>a.plan.native.reply_capacity)return false;
    count=static_cast<std::uint32_t>(bytes/sizeof(Record));
    return (!count || a.input.records) && count<=a.input.invocation->unique_rows;
}
/** @return Exact completed-source branch; pointer/configuration are immutable capture operands. */
QFN_ADAPTIVE_HD inline SourceGate sourceGate(SourceCondition condition,const SourceFrontier* h,
    p::Interval interval,std::uint64_t table,p::SourceWindow expected)
{
    if(condition==SourceCondition::Always)return h?SourceGate::Invalid:SourceGate::Run;
    if((condition!=SourceCondition::AfterSuccess && condition!=SourceCondition::AfterFailure) || !h ||
       h->version!=Version || !same(h->interval,interval) || !same(h->source,expected) ||
       (h->status!=Status::Success && h->status!=Status::SourceFailure && h->status!=Status::Cancelled) ||
       (h->source.table ? h->source.table!=table || !h->source.generation || !h->source.sequence ||
           (h->disposition!=SourceDisposition::Consumed && h->disposition!=SourceDisposition::Aborted) :
           h->source.generation || h->source.sequence || h->disposition!=SourceDisposition::Unpublished) ||
       (h->disposition==SourceDisposition::Aborted && h->status==Status::Success))return SourceGate::Invalid;
    const bool success=h->status==Status::Success;
    return success==(condition==SourceCondition::AfterSuccess)?SourceGate::Run:SourceGate::Skip;
}
/** @return Nominal exact retirement; Aborted never becomes a successful acquired completion. */
QFN_ADAPTIVE_HD inline p::SourceRetirement sourceRetirement(SourceDisposition disposition)
{
    return disposition==SourceDisposition::Consumed?p::SourceRetirement::Consumed:
        disposition==SourceDisposition::Aborted?p::SourceRetirement::Aborted:
        disposition==SourceDisposition::Unpublished?p::SourceRetirement::Unpublished:p::SourceRetirement::Unknown;
}
/** @return Valid original source metadata; publication and consumption remain separate explicit producers. */
template<class Slot> QFN_ADAPTIVE_HD const SourceFrontier* source(const OwnerArgs& a,bool allow_failure=false)
{
    const auto* h=static_cast<const SourceFrontier*>(a.input.frontier);
    if(!h || h->version!=Version || (h->status!=Status::Success && (!allow_failure ||
            (h->status!=Status::SourceFailure && h->status!=Status::Cancelled))) || !same(h->interval,a.input.invocation->interval) ||
        (h->source.table ? h->source.table!=a.plan.table || !h->source.generation || !h->source.sequence ||
            h->disposition==SourceDisposition::Unpublished :
            h->source.generation || h->source.sequence || h->disposition!=SourceDisposition::Unpublished) ||
        (h->disposition!=SourceDisposition::Unpublished && h->disposition!=SourceDisposition::Published &&
            h->disposition!=SourceDisposition::Consumed &&
            (!allow_failure || h->disposition!=SourceDisposition::Aborted || h->status==Status::Success)))
    {fail<Slot>(a);return nullptr;}
    return h;
}
/** @return Exact final-reader release grant in original immutable owner log order. */
QFN_ADAPTIVE_HD inline p::Grant originalGrant(const OwnerArgs& a,std::uint32_t index)
{
    const auto pins=at<p::OwnerFrame>(a.frame,a.plan.frame.control)->initial_pin_count;
    return index<pins?at<p::Grant>(a.frame,a.plan.frame.initial_pins)[index]:grants(a)[index-pins];
}
/**
 * @brief Validate the complete actual pin batch before issuing even its first physical pin.
 * @note Integer radix duplicate checks are O(live commands); row/unique/slot
 * aliases cannot turn a later rejection into an unreported acquired prefix.
 */
template<class Slot> QFN_ADAPTIVE_HD bool validatePins(const OwnerArgs& a,const p::OwnerCommand* commands,std::uint32_t count)
{
    auto v=ownerView<Slot>(a);
    if(v.frame->phase!=p::OwnerPhase::Pinning || v.frame->initial_pin_count || v.frame->held_pins)return fail<Slot>(a);
    auto ordinal=v.frame->last_command;
    for(std::uint32_t i=0;i<count;++i){
        const auto& c=commands[i];const auto& g=c.target;
        if(c.kind!=p::CommandKind::PinInitial || c.reserved || !same(g.interval,a.input.invocation->interval) ||
            g.kind!=p::GrantKind::Read || g.destination_token || g.unique>=a.input.invocation->unique_rows ||
            g.row>=a.plan.owner.sourceRows() || g.owner!=a.plan.owner.coordinate() || g.slot>=a.plan.owner.capacity() ||
            !g.generation || c.requested_row!=g.row || c.ordinal.value<=ordinal)return fail<Slot>(a);
        const auto& slot=v.slots[g.slot];
        if(slot.row!=g.row || slot.generation!=g.generation ||
            adaptive_lookup::PhysicalSlot<Slot>::state(slot)!=adaptive_lookup::SlotState::Ready ||
            adaptive_lookup::PhysicalSlot<Slot>::pinned(slot))return fail<Slot>(a);
        ordinal=c.ordinal.value;
    }
    if(!count)return true;
    auto* first=at<std::uint64_t>(a.frame,a.plan.frame.validation_a);
    auto* second=at<std::uint64_t>(a.frame,a.plan.frame.validation_b);
    auto* bins=at<std::uint32_t>(a.frame,a.plan.frame.validation_counts);
    for(unsigned key=0;key<3;++key){
        for(std::uint32_t i=0;i<count;++i)first[i]=key==0?commands[i].target.row:key==1?commands[i].target.unique:commands[i].target.slot;
        auto* input=first;auto* output=second;
        for(unsigned pass=0;pass<(key?4u:8u);++pass){
            for(unsigned bin=0;bin<256;++bin)bins[bin]=0;
            for(std::uint32_t i=0;i<count;++i)++bins[(input[i]>>(pass*8))&255];
            std::uint32_t cursor=0;
            for(unsigned bin=0;bin<256;++bin){const auto size=bins[bin];bins[bin]=cursor;cursor+=size;}
            for(std::uint32_t i=0;i<count;++i)output[bins[(input[i]>>(pass*8))&255]++]=input[i];
            auto* swap=input;input=output;output=swap;
        }
        for(std::uint32_t i=1;i<count;++i)if(input[i]==input[i-1])return fail<Slot>(a);
    }
    return true;
}
/** @brief Initialize only native setup records; GPU slot initialization completes in a separate ordered leaf. */
template<class Slot> QFN_ADAPTIVE_HD void seedOwner(const OwnerArgs& a)
{
    *ownerView<Slot>(a).frame={};
    if(a.plan.owner.capacity())*ownerView<Slot>(a).owner={};
    allocationFrame(a)={};
    progress(a)={};
}
/** @brief Seal actual initialized physical metadata once; no payload or unused frame banks are cleared. */
template<class Slot> QFN_ADAPTIVE_HD bool sealOwner(const OwnerArgs& a)
{
    if(!p::OwnerOperations<Slot>(ownerView<Slot>(a)).initialize())return fail<Slot>(a);
    progress(a).setup_identity=a.plan.identity;progress(a).initialized=2;return true;
}
/** @brief Run one ordered native owner metadata operation; payload copy publication is a separate exact event edge. */
template<OwnerAction Action,class Slot> QFN_ADAPTIVE_HD bool owner(const OwnerArgs& a)
{
    auto view=ownerView<Slot>(a);
    auto physical=p::OwnerOperations<Slot>(view);
    auto plan=allocation<Slot>(a);
    auto& state=allocationFrame(a);
    auto& output=progress(a);
    output.header_bytes=0;
    if(!current<Slot>(a,Action==OwnerAction::Begin))return false;
    const auto gate=sourceGate(a.input.condition,a.input.outcome,a.input.invocation->interval,a.plan.table,view.frame->source);
    if(gate==SourceGate::Invalid)return fail<Slot>(a,Status::SourceFailure);
    if(gate==SourceGate::Skip)return true;
    if constexpr(Action==OwnerAction::Begin)
    {
        if(!physical.begin(a.input.invocation->interval))return fail<Slot>(a);
        output.reply_bytes=output.release_bytes=0;output.released=0;output.delivery=Delivery::None;output.retirement=p::SourceRetirement::Unpublished;
        return true;
    }
    else if constexpr(Action==OwnerAction::Pin)
    {
        output.reply_bytes=0;
        if(view.frame->phase!=p::OwnerPhase::Pinning)return fail<Slot>(a);
        std::uint32_t count=0;
        if(a.plan.native.reply_capacity?!recordCount<p::OwnerCommand>(a,count):(a.input.records || a.input.record_bytes))return fail<Slot>(a);
        const auto* commands=static_cast<const p::OwnerCommand*>(a.input.records);
        if(!validatePins<Slot>(a,commands,count))return false;
        for(std::uint32_t i=0;i<count;++i){
            const auto& c=commands[i];const auto& g=c.target;
            if(c.kind!=p::CommandKind::PinInitial || c.reserved || !same(g.interval,a.input.invocation->interval) ||
                g.unique>=a.input.invocation->unique_rows || c.requested_row!=g.row ||
                physical.pinInitial(g.interval,{g.row,g.generation,g.owner,g.slot},g.unique,c.ordinal).kind!=p::GrantKind::Read)
                return fail<Slot>(a);
        }
        output.reply_bytes=std::uint64_t{view.frame->initial_pin_count}*sizeof(p::Grant);return true;
    }
    else if constexpr(Action==OwnerAction::SealPins)
    {header(a,physical.sealPins(a.input.invocation->interval));return view.frame->status==Status::Success;}
    else if constexpr(Action==OwnerAction::AcquirePins)
    {return physical.acceptPinFrontier(a.input.invocation->interval,static_cast<const p::PinHeader*>(a.input.frontier),a.plan.owner.participants());}
    else if constexpr(Action==OwnerAction::Availability)
    {
        const bool ok=plan.acquire(a.input.invocation->interval);
        header(a,Availability{state.lease,view.frame->status,Version});return ok;
    }
    else if constexpr(Action==OwnerAction::AcceptFree || Action==OwnerAction::AcceptVictim)
    {
        const auto* h=static_cast<const std::conditional_t<Action==OwnerAction::AcceptFree,FreePlanHeader,VictimPlanHeader>*>(a.input.frontier);
        if(!h || h->version!=Version || h->status!=Status::Success || h->decision_bytes%sizeof(p::AllocationDecision) ||
            h->decision_bytes/sizeof(p::AllocationDecision)>a.plan.native.reply_capacity ||
            (h->decision_bytes && !a.input.records))return fail<Slot>(a);
        const auto count=static_cast<std::uint32_t>(h->decision_bytes/sizeof(p::AllocationDecision));
        const auto* records=static_cast<const std::byte*>(a.input.records);
        if(a.input.record_layout==RecordLayout::DirectedAllocationBank) {
            const auto* slice=a.input.allocation_slice;
            const auto capacity=std::uint64_t{a.plan.owner.maximumUnique()}*sizeof(p::AllocationDecision);
            if(!slice || !records || a.input.record_capacity!=capacity || slice->live_bytes!=h->decision_bytes ||
               slice->byte_offset<a.input.record_origin || slice->byte_offset-a.input.record_origin>capacity ||
               slice->live_bytes>capacity-(slice->byte_offset-a.input.record_origin) ||
               (slice->byte_offset-a.input.record_origin)%sizeof(p::AllocationDecision))return fail<Slot>(a);
            records+=slice->byte_offset-a.input.record_origin;
        }else if(a.input.record_layout!=RecordLayout::Compact || a.input.allocation_slice || a.input.record_origin || a.input.record_capacity)
            return fail<Slot>(a);
        const auto* decisions=reinterpret_cast<const p::AllocationDecision*>(records);
        for(std::uint32_t i=0;i<count;++i)if(decisions[i].unique>=a.input.invocation->unique_rows)return fail<Slot>(a);
        if constexpr(Action==OwnerAction::AcceptFree)return plan.accept(p::FreeAllocationPlan{h->lease,decisions,count});
        else return plan.accept(p::VictimAllocationPlan{h->lease,decisions,count});
    }
    else if constexpr(Action==OwnerAction::ExecuteFree || Action==OwnerAction::ExecuteVictim)
    {
        output.reply_bytes=0;
        const bool ok=Action==OwnerAction::ExecuteFree?plan.executeFree():plan.executeVictims();
        if(ok)output.reply_bytes=std::uint64_t{state.wave_count}*sizeof(p::Grant);
        return ok;
    }
    else if constexpr(Action==OwnerAction::FreeReceipt)
    {const auto receipt=plan.freeReceipt();header(a,receipt);return receipt.status==Status::Success || receipt.status==Status::Cancelled;}
    else if constexpr(Action==OwnerAction::AcquireFree)
    {return plan.acceptFreeFrontier(static_cast<const p::FreeAllocationReceipt*>(a.input.frontier),a.plan.owner.participants());}
    else if constexpr(Action==OwnerAction::VictimReceipt)
    {const auto receipt=plan.victimReceipt();header(a,receipt);return receipt.status==Status::Success || receipt.status==Status::Cancelled;}
    else if constexpr(Action==OwnerAction::SealSource)
    {
        const auto* h=source<Slot>(a,true);if(!h || !plan.sealSource(h->source))return false;
        output.retirement=h->source.table?p::SourceRetirement::Unknown:p::SourceRetirement::Unpublished;return true;
    }
    else if constexpr(Action==OwnerAction::ApplySourceDelivery)
    {
        output.delivery=Delivery::None;
        const auto* h=source<Slot>(a);std::uint32_t count=0;
        const bool records_ok=a.plan.native.reply_capacity?recordCount<Fill>(a,count):
            (!a.input.records && !a.input.record_bytes && !a.input.payload && !a.input.payload_bytes);
        if(!h || !records_ok || view.frame->phase!=p::OwnerPhase::Filling || count!=state.count ||
            !same(h->source,view.frame->source) || (h->source.table && h->disposition!=SourceDisposition::Consumed) ||
            (count?(!a.input.payload_bytes || *a.input.payload_bytes!=std::uint64_t{count}*a.plan.row_bytes || !a.input.payload):
                (a.input.payload_bytes && *a.input.payload_bytes)))return fail<Slot>(a,Status::SourceFailure);
        const auto* fills=static_cast<const Fill*>(a.input.records);
        for(std::uint32_t i=0;i<count;++i)
            if(!same(fills[i].original,grants(a)[i]) || !same(fills[i].acknowledgement.source,h->source) ||
                fills[i].acknowledgement.destination_token!=grants(a)[i].destination_token)
                return fail<Slot>(a,Status::SourceFailure);
        for(std::uint32_t i=0;i<count;++i)
            at<p::FillAcknowledgement>(a.frame,a.plan.frame.acknowledgements)[i]=fills[i].acknowledgement;
        output.retirement=h->source.table?p::SourceRetirement::Consumed:p::SourceRetirement::Unpublished;
        output.delivery=Delivery::Validated;return true;
    }
    else if constexpr(Action==OwnerAction::ValidateFills)
    {
        if(output.delivery!=Delivery::Copied)return fail<Slot>(a);
        const auto h=plan.validateFills(at<p::FillAcknowledgement>(a.frame,a.plan.frame.acknowledgements),state.count);
        header(a,h);return h.status==Status::Success;
    }
    else if constexpr(Action==OwnerAction::AcquirePublication)
    {return physical.acceptPublishFrontier(a.input.invocation->interval,static_cast<const p::PublishHeader*>(a.input.frontier),a.plan.owner.participants());}
    else if constexpr(Action==OwnerAction::MakeReady)
    {
        if(output.delivery!=Delivery::Copied || view.frame->phase!=p::OwnerPhase::Publishable)return fail<Slot>(a);
        for(std::uint32_t i=0;i<state.count;++i)
            if(!physical.makeReady(a.input.invocation->interval,grants(a)[i],
                at<p::FillAcknowledgement>(a.frame,a.plan.frame.acknowledgements)[i]))return false;
        return true;
    }
    else if constexpr(Action==OwnerAction::ReadyReceipt)
    {const auto h=physical.readyHeader(a.input.invocation->interval);header(a,h);return h.status==Status::Success;}
    else if constexpr(Action==OwnerAction::Release)
    {
        const auto total=view.frame->initial_pin_count+state.count;
        if((view.frame->phase!=p::OwnerPhase::Publishable && view.frame->phase!=p::OwnerPhase::Draining) ||
            total>a.plan.native.reply_capacity || output.released>total ||
            output.release_bytes!=std::uint64_t{output.released}*sizeof(p::ReleaseReceipt))return fail<Slot>(a);
        for(auto i=output.released;i<total;++i){
            const auto receipt=physical.releaseWithReceipt(a.input.invocation->interval,originalGrant(a,i));
            if(receipt.disposition==p::ReleaseDisposition::Failed)return false;
            at<p::ReleaseReceipt>(a.frame,a.plan.frame.release_receipts)[i]=receipt;
            ++output.released;output.release_bytes=std::uint64_t{output.released}*sizeof(p::ReleaseReceipt);
        }
        return true;
    }
    else if constexpr(Action==OwnerAction::CancelFree || Action==OwnerAction::CancelVictim || Action==OwnerAction::CancelSource)
    {
        output.reply_bytes=0;
        const bool accepted=state.phase==p::AllocationPhase::Accepted;
        if constexpr(Action==OwnerAction::CancelSource)
        {
            const auto* h=source<Slot>(a,true);
            if(!h || h->disposition==SourceDisposition::Published)return fail<Slot>(a,Status::SourceFailure);
            const auto retirement=sourceRetirement(h->disposition);
            if(!plan.cancel(retirement,h->source))return false;
            output.retirement=retirement;
        }
        else if(state.phase!=p::AllocationPhase::Cancelled)
        {
            if constexpr(Action==OwnerAction::CancelFree){
                if(state.wave!=p::AllocationWave::Free || state.phase==p::AllocationPhase::SourceSealed)return fail<Slot>(a);
            }else if((state.wave==p::AllocationWave::Free && state.phase!=p::AllocationPhase::FreeFrontier) ||
                state.phase==p::AllocationPhase::SourceSealed)return fail<Slot>(a);
            if(!plan.cancel(p::SourceRetirement::Unpublished))return false;
        }
        if(accepted)output.reply_bytes=std::uint64_t{state.wave_count}*sizeof(p::Grant);
        if constexpr(Action==OwnerAction::CancelFree)header(a,plan.freeReceipt());
        if constexpr(Action==OwnerAction::CancelVictim)header(a,plan.victimReceipt());
        return view.frame->status==Status::Cancelled;
    }
    else if constexpr(Action==OwnerAction::Retire)
    {
        if(output.retirement==p::SourceRetirement::Unknown)return fail<Slot>(a,Status::SourceFailure);
        const bool ok=plan.retire(output.retirement,view.frame->source);
        header(a,physical.retiredHeader(a.input.invocation->interval));return ok;
    }
}
/** @brief Mark payload-copy completion only after every actual native writer's ordered launch has retired. */
template<class Slot> QFN_ADAPTIVE_HD bool copied(const OwnerArgs& a)
{
    const auto gate=sourceGate(a.input.condition,a.input.outcome,a.input.invocation->interval,a.plan.table,ownerView<Slot>(a).frame->source);
    if(gate==SourceGate::Skip)return true;
    if(gate==SourceGate::Invalid || !current<Slot>(a) || progress(a).delivery!=Delivery::Validated)return fail<Slot>(a);
    progress(a).delivery=Delivery::Copied;return true;
}
}

namespace llaminar2::qwen4exp::adaptive_lookup::pooled
{
/**
 * @brief Internal fixed native-producer access; no public skip-coalescing enqueue exists.
 *
 * The complete native Begin bundle owns validation, stable sort/scan,
 * permutation materialization and commit on one exact stream. These private
 * lifecycle transitions expose neither transport nor source capabilities.
 */
struct CoordinatorCoalescingAccess
{
    /** @brief Open only original identity and bounds before parallel data producers. */
    template<class Row> QFN_ADAPTIVE_HD static bool open(CoordinatorOperations& op,std::uint64_t admission,const Row* rows,std::uint32_t count)
    {return op.openOriginal(admission,rows,count);}
    /** @brief Commit after every actual disjoint coalescing producer is ordered. */
    QFN_ADAPTIVE_HD static bool finish(CoordinatorOperations& op){return op.finishOriginal();}
    /** @brief Seal controls after actual independent setup producers complete. */
    QFN_ADAPTIVE_HD static bool seal(CoordinatorOperations& op){return op.initializeControls();}
    /** @brief Consume only the exact stable native run banks after their complete ordered producer. */
    QFN_ADAPTIVE_HD static bool planFree(CoordinatorOperations& op,const AvailabilityLease* leases,std::uint32_t count,OriginalObservationRuns runs)
    {return op.planFreeImpl(leases,count,&runs);}
    /** @brief Initialize one actual independent directory setup record. */
    QFN_ADAPTIVE_HD static void entry(CoordinatorOperations& op,std::uint32_t i){op.initializeEntry(i);}
    /** @brief Initialize one actual independent radix-node setup record. */
    QFN_ADAPTIVE_HD static void node(CoordinatorOperations& op,std::uint32_t i){op.initializeNode(i);}
    /** @brief Initialize one actual independent frequency setup record. */
    QFN_ADAPTIVE_HD static void frequency(CoordinatorOperations& op,std::uint64_t i){op.initializeFrequency(i);}
};
}
namespace llaminar2::qwen4exp::pooled_native
{
/** @return The sole actual coordinator view derived from exact immutable native offsets. */
template<class Args> QFN_ADAPTIVE_HD p::CoordinatorView coordinatorView(const Args& a)
{
    const auto& f=a.plan.frame;const auto& c=a.plan.cache;const auto& policy=a.plan.policy;
    return {a.cache?at<p::CoordinatorPolicyControl>(a.cache,c.control):nullptr,
        a.cache?at<p::DirectoryEntry>(a.cache,c.entries):nullptr,
        a.cache && policy.entries()>1?at<adaptive_lookup::RadixNode>(a.cache,c.nodes):nullptr,
        a.cache?at<std::uint32_t>(a.cache,c.epochs):nullptr,
        a.cache?at<std::uint16_t>(a.cache,c.counts):nullptr,
        at<p::LookupFrame>(a.frame,f.control),at<p::CoordinatorUnique>(a.frame,f.unique),
        at<p::CoordinatorOccurrence>(a.frame,f.sort_a),at<p::CoordinatorOccurrence>(a.frame,f.sort_b),
        at<std::uint32_t>(a.frame,f.radix_counts),at<std::uint32_t>(a.frame,f.original_to_unique),
        at<p::CoordinatorOwnerPlan>(a.frame,f.owner_plans),at<p::AllocationDecision>(a.frame,f.decisions),
        at<p::CoordinatorOwnerGeometry>(a.frame,f.owners),policy,policy.maximumUnique(),policy.maximumOriginal(),policy.participants(),
        a.plan.table,a.plan.source_storage_generation};
}
/** @brief Preserve actual first coordinator failure without fabricating owner or source observations. */
QFN_ADAPTIVE_HD inline bool fail(const CoordinatorArgs& a,Status why=Status::Invariant)
{
    auto v=coordinatorView(a);
    if(v.frame->status==Status::Success)v.frame->status=why;
    if(v.policy)v.policy->flags|=adaptive_lookup::OwnerFailed;
    return false;
}
/** @return Actual coordinator is in the exact required phase, including empty work. */
QFN_ADAPTIVE_HD inline bool phase(const CoordinatorArgs& a,p::LookupPhase expected)
{
    const auto& f=*coordinatorView(a).frame;
    return f.status==Status::Success && f.phase==expected?true:fail(a);
}
/** @return Exact acquired record prefix, bounded by original global U/K and never allocation capacity. */
template<class Record> QFN_ADAPTIVE_HD bool recordCount(const CoordinatorArgs& a,std::uint32_t& count)
{
    if(!a.plan.native.reply_capacity){count=0;return !a.input.records && !a.input.record_bytes;}
    if(!a.input.record_bytes)return false;
    const auto bytes=*a.input.record_bytes;
    if(bytes%sizeof(Record) || bytes/sizeof(Record)>a.plan.native.reply_capacity)return false;
    count=static_cast<std::uint32_t>(bytes/sizeof(Record));
    return (!count || a.input.records) && count<=coordinatorView(a).frame->unique_rows;
}
/** @brief Publish the actual compact immutable original invocation after coalescing completes or fails. */
QFN_ADAPTIVE_HD inline void invocation(const CoordinatorArgs& a)
{
    const auto& f=*coordinatorView(a).frame;
    *at<Invocation>(a.native,a.plan.native.invocation)={f.interval,a.plan.source_storage_generation,
        f.original_ids,f.unique_rows,f.status,Version};
}
/** @return Original consumed/unpublished source observation, rejecting a merely published response. */
QFN_ADAPTIVE_HD inline bool source(const CoordinatorArgs& a,bool consumed,bool allow_failure=false)
{
    const auto* h=a.input.source;const auto& f=*coordinatorView(a).frame;
    if(!h || h->version!=Version || (h->status!=Status::Success && (!allow_failure ||
            (h->status!=Status::SourceFailure && h->status!=Status::Cancelled))) || !same(h->interval,f.interval) ||
        (h->source.table ? h->source.table!=a.plan.table || !h->source.generation || !h->source.sequence ||
            h->disposition==SourceDisposition::Unpublished :
            h->source.generation || h->source.sequence || h->disposition!=SourceDisposition::Unpublished) ||
        (h->disposition!=SourceDisposition::Published && h->disposition!=SourceDisposition::Consumed &&
            h->disposition!=SourceDisposition::Unpublished &&
            (!allow_failure || h->disposition!=SourceDisposition::Aborted || h->status==Status::Success)) ||
        (consumed && h->disposition==SourceDisposition::Published))return fail(a,Status::SourceFailure);
    return true;
}
/** @brief Build only actual immutable per-owner plan headers/slices after the canonical policy writer succeeds. */
template<bool Victims> QFN_ADAPTIVE_HD void directedPlans(const CoordinatorArgs& a)
{
    const auto v=coordinatorView(a);
    for(std::uint32_t owner=0;owner<a.plan.policy.participants();++owner){
        const auto& plan=v.owner_plans[owner];
        const auto count=Victims?plan.victim_count:plan.free_count;
        const auto offset=Victims?plan.victim_offset:plan.free_offset;
        const auto bytes=v.frame->status==Status::Success?std::uint64_t{count}*sizeof(p::AllocationDecision):0;
        if constexpr(Victims)
            at<VictimPlanHeader>(a.native,a.plan.native.victim_headers)[owner]={plan.lease,bytes,v.frame->status,Version};
        else at<FreePlanHeader>(a.native,a.plan.native.free_headers)[owner]={plan.lease,bytes,v.frame->status,Version};
        at<DirectedSlice>(a.native,Victims?a.plan.native.victim_slices:a.plan.native.free_slices)[owner]=
            {a.plan.frame.decisions+std::uint64_t{offset}*sizeof(p::AllocationDecision),bytes};
    }
}
/** @brief Seed only virgin native coordinator controls; independent cache setup uses parallel disjoint writers. */
QFN_ADAPTIVE_HD inline void seedCoordinator(const CoordinatorSetupArgs& a)
{
    auto v=coordinatorView(a);*v.frame={};if(v.policy)*v.policy={};
    *at<Invocation>(a.native,a.plan.native.invocation)={};
}
/** @brief Stable-sort actual acquired release receipts without overwriting their original observation bank. */
QFN_ADAPTIVE_HD inline bool sortedReleases(const CoordinatorArgs& a,std::uint32_t count)
{
    auto v=coordinatorView(a);const auto* receipts=static_cast<const p::ReleaseReceipt*>(a.input.records);
    if(!count)return true;
    for(std::uint32_t i=0;i<count;++i){
        const auto unique=receipts[i].original.unique;
        if(unique>=v.frame->unique_rows)return fail(a);
        v.sort_a[i]={std::uint64_t{unique}<<32|i,0,0};
    }
    auto* input=v.sort_a;auto* output=v.sort_b;
    for(unsigned pass=4;pass<8;++pass){
        for(unsigned bin=0;bin<256;++bin)v.radix_counts[bin]=0;
        for(std::uint32_t i=0;i<count;++i)++v.radix_counts[(input[i].row>>(8*pass))&255];
        std::uint32_t cursor=0;
        for(unsigned bin=0;bin<256;++bin){const auto size=v.radix_counts[bin];v.radix_counts[bin]=cursor;cursor+=size;}
        for(std::uint32_t i=0;i<count;++i)output[v.radix_counts[(input[i].row>>(8*pass))&255]++]=input[i];
        auto* swap=input;input=output;output=swap;
    }
    for(std::uint32_t i=1;i<count;++i)if(input[i].row>>32==input[i-1].row>>32)return fail(a);
    for(std::uint32_t i=0;i<count;++i)
        at<p::ReleaseReceipt>(a.native,a.plan.native.sorted_releases)[i]=receipts[static_cast<std::uint32_t>(input[i].row)];
    return true;
}
/** @brief Execute one ordered coordinator metadata phase; Begin's independent data work is a fixed cooperative bundle. */
template<CoordinatorAction Action> QFN_ADAPTIVE_HD bool coordinator(const CoordinatorArgs& a)
{
    const auto frame=coordinatorView(a).frame;
    const auto gate=sourceGate(a.input.condition,a.input.outcome,frame->interval,a.plan.table,frame->source);
    if(gate==SourceGate::Invalid)return fail(a,Status::SourceFailure);
    if(gate==SourceGate::Skip)return true;
    auto v=coordinatorView(a);auto op=p::CoordinatorOperations(v);const auto participants=a.plan.policy.participants();
    if constexpr(Action==CoordinatorAction::PinCommands)
    {
        if(!phase(a,p::LookupPhase::Pinning))return false;
        auto* slices=at<DirectedSlice>(a.native,a.plan.native.pin_slices);
        auto* cursors=at<std::uint32_t>(a.native,a.plan.native.pin_cursors);
        for(std::uint32_t owner=0;owner<participants;++owner){slices[owner]={};cursors[owner]=0;}
        for(std::uint32_t i=0;i<v.frame->unique_rows;++i)
            if(v.unique[i].phase==p::UniquePhase::NeedsPin){
                if(!op.pinCommand(i,v.unique[i].pin))return false;
                slices[v.unique[i].pin.target.owner].live_bytes+=sizeof(p::OwnerCommand);
            }
        std::uint64_t offset=0;
        for(std::uint32_t owner=0;owner<participants;++owner){slices[owner].byte_offset=offset;offset+=slices[owner].live_bytes;}
        for(std::uint32_t i=0;i<v.frame->unique_rows;++i)if(v.unique[i].phase==p::UniquePhase::PinPending){
            const auto owner=v.unique[i].pin.target.owner;
            at<p::OwnerCommand>(a.native,a.plan.native.pin_commands)[slices[owner].byte_offset/sizeof(p::OwnerCommand)+cursors[owner]++]=v.unique[i].pin;
        }
        return true;
    }
    else if constexpr(Action==CoordinatorAction::AcquirePins || Action==CoordinatorAction::AcquireFree || Action==CoordinatorAction::AcquireVictims)
    {
        if(!phase(a,Action==CoordinatorAction::AcquirePins?p::LookupPhase::Pinning:
            Action==CoordinatorAction::AcquireFree?p::LookupPhase::FreePlanned:p::LookupPhase::VictimPlanned))return false;
        std::uint32_t count=0;if(!recordCount<p::Grant>(a,count))return fail(a);
        const auto* grants=static_cast<const p::Grant*>(a.input.records);
        for(std::uint32_t i=0;i<count;++i){
            if constexpr(Action==CoordinatorAction::AcquirePins){if(!op.acceptPin(grants[i].unique,grants[i]))return false;}
            else if(!op.acceptReservation(grants[i].unique,grants[i]))return false;
        }
        return true;
    }
    else if constexpr(Action==CoordinatorAction::SealPins)
    {return op.sealPins(static_cast<const p::PinHeader*>(a.input.frontier),participants);}
    else if constexpr(Action==CoordinatorAction::PlanFree)
    {
        if(!phase(a,p::LookupPhase::Availability) || !a.input.frontier)return fail(a);
        const auto* receipts=static_cast<const Availability*>(a.input.frontier);
        for(std::uint32_t owner=0;owner<participants;++owner)
            if(receipts[owner].version!=Version || receipts[owner].status!=Status::Success ||
                receipts[owner].lease.owner!=owner || !same(receipts[owner].lease.interval,v.frame->interval) ||
                receipts[owner].lease.free_slots>v.owners[owner].slots)return fail(a);
        auto* leases=at<p::AvailabilityLease>(a.native,a.plan.native.availability);
        for(std::uint32_t owner=0;owner<participants;++owner)leases[owner]=receipts[owner].lease;
        const bool ok=p::CoordinatorCoalescingAccess::planFree(op,leases,participants,{v.sort_a,v.sort_b});directedPlans<false>(a);return ok;
    }
    else if constexpr(Action==CoordinatorAction::CompleteFree)
    {return op.completeFree(static_cast<const p::FreeAllocationReceipt*>(a.input.frontier),participants);}
    else if constexpr(Action==CoordinatorAction::PlanVictims)
    {const bool ok=op.planVictims();directedPlans<true>(a);return ok;}
    else if constexpr(Action==CoordinatorAction::CompleteVictims)
    {return op.completeVictims(static_cast<const p::VictimAllocationReceipt*>(a.input.frontier),participants);}
    else if constexpr(Action==CoordinatorAction::SealSource)
    {return source(a,false,true) && op.sealSource(a.input.source->source);}
    else if constexpr(Action==CoordinatorAction::Publish)
    {
        if(!source(a,true))return false;
        return op.publish(static_cast<const p::ReadyHeader*>(a.input.frontier),participants,
            sourceRetirement(a.input.source->disposition),
            a.input.source->source);
    }
    else if constexpr(Action==CoordinatorAction::Cancel)
    {
        std::uint32_t count=0;
        if(!source(a,true,true) || !recordCount<p::ReleaseReceipt>(a,count) || !sortedReleases(a,count))return fail(a);
        return op.cancel(sourceRetirement(a.input.source->disposition),
            a.input.source->source,count?at<p::ReleaseReceipt>(a.native,a.plan.native.sorted_releases):nullptr,count);
    }
    else if constexpr(Action==CoordinatorAction::Finish)
    {return op.finish(static_cast<const p::RetiredHeader*>(a.input.frontier),participants);}
}
}
