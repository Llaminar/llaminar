/** @file original_PooledImmutableSourceOwner.inc @brief Exact reviewed r5 owner operation; only function name changes for isolated causal replay. */
namespace llaminar2::qwen4exp::pooled_native {
template<OwnerAction Action,class Slot> QFN_ADAPTIVE_HD bool originalSourceOutcomeOwner(const OwnerArgs& a)
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
        }else if(a.input.record_layout==RecordLayout::TransportedAllocationPrefix) {
            // The public binding retains a separate original transport count.
            // Validate that publication before plan.accept can reserve or copy
            // any decision. C0 has neither a count message nor payload pointer.
            if(a.input.allocation_slice || a.input.record_origin || a.input.record_capacity ||
               (a.plan.native.reply_capacity ? (!a.input.record_bytes ||
                   reinterpret_cast<std::uintptr_t>(a.input.record_bytes)%alignof(std::uint64_t) ||
                   *a.input.record_bytes!=h->decision_bytes) :
                   (a.input.record_bytes || a.input.records || h->decision_bytes)) ||
               (records && reinterpret_cast<std::uintptr_t>(records)%alignof(p::AllocationDecision)))return fail<Slot>(a);
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
}
