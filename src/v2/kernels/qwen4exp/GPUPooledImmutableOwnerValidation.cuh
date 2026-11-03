/**
 * @file GPUPooledImmutableOwnerValidation.cuh
 * @brief One-CTA immutable validation followed by the original ordered owner commit.
 *
 * The same native invocation owns parsing, all exact identity checks and commit.
 * CTA-local intermediates never outlive it or authorize another operation. Original
 * CPU scalar operations retain their independent validation and mutation ordering.
 */
#pragma once
#include "PooledImmutableOwnerRecordValidation.h"
namespace llaminar2::qwen4exp::pooled_native {
/** @brief Internal friend of the existing allocation owner, never a public validation bypass. */
template<class Slot> struct CooperativeOwnerValidation final {
    /** @brief Execute one supported validation-heavy operation on the exact original owner.
     * @tparam Action Pin, AcceptFree or AcceptVictim only.
     * @param a Original admitted owner and captured producer operands.
     * @return Uniform original success/failure, after any ordered commit has completed.
     */
    template<OwnerAction Action> static __device__ bool run(const OwnerArgs& a)
    {
        static_assert(Action==OwnerAction::Pin || Action==OwnerAction::AcceptFree || Action==OwnerAction::AcceptVictim);
        namespace gpu=QFN_NATIVE_NAMESPACE;
        namespace p=adaptive_lookup::pooled;
        __shared__ gpu::ExactIdentityCTA scratch;
        __shared__ unsigned state,count,total;
        __shared__ const p::OwnerCommand* commands;
        __shared__ const p::AllocationDecision* decisions;
        __shared__ p::AvailabilityLease lease;
        auto plan=allocation<Slot>(a);
        if(threadIdx.x==0){
            state=1;count=total=0;commands=nullptr;decisions=nullptr;
            auto& output=progress(a);
            output.header_bytes=0;output.header_kind=OwnerHeaderKind::None;
            if(!current<Slot>(a))state=0;
            else{
                const auto gate=sourceGate(a.input.condition,a.input.outcome,a.input.invocation->interval,
                    a.plan.table,ownerView<Slot>(a).frame->source);
                if(gate==SourceGate::Invalid){fail<Slot>(a,Status::SourceFailure);state=0;}
                else if(gate==SourceGate::Skip)state=2;
                else if constexpr(Action==OwnerAction::Pin){
                    output.reply_bytes=0;
                    if(!pinCommandSpan<Slot>(a,commands,count))state=0;
                    else{
                        const auto view=ownerView<Slot>(a);
                        if(view.frame->phase!=p::OwnerPhase::Pinning || view.frame->initial_pin_count || view.frame->held_pins){
                            fail<Slot>(a);state=0;
                        }
                    }
                }else{
                    constexpr auto wave=Action==OwnerAction::AcceptFree?p::AllocationWave::Free:p::AllocationWave::Victim;
                    if(!allocationDecisionSpan<Action,Slot>(a,lease,decisions,count) ||
                       !plan.validateWave(wave,lease,decisions,count))state=0;
                }
            }
        }
        __syncthreads();
        if(state!=1)return state==2;
        if constexpr(Action==OwnerAction::Pin){
            if(threadIdx.x==0)scratch.invalid=0;
            __syncthreads();
            for(std::uint64_t i=threadIdx.x;i<count;i+=gpu::Threads){
                const auto prior=i?commands[i-1].ordinal.value:ownerView<Slot>(a).frame->last_command;
                if(!record_validation::pin<Slot>(a,commands[i],prior))atomicOr(&scratch.invalid,1u);
            }
            __syncthreads();
            // Every immutable field rejection has the same original Invariant
            // status. Publish it once after all lanes converge, before any key
            // validation or physical pinInitial mutation can begin.
            if(threadIdx.x==0 && scratch.invalid){fail<Slot>(a);state=0;}
            __syncthreads();
            if(!state)return false;
        }
        auto* first=at<std::uint64_t>(a.frame,a.plan.frame.validation_a);
        auto* second=at<std::uint64_t>(a.frame,a.plan.frame.validation_b);
        for(unsigned key=0;key<3;++key){
            if constexpr(Action==OwnerAction::Pin){
                for(std::uint64_t i=threadIdx.x;i<count;i+=gpu::Threads)
                    first[i]=key==0?commands[i].target.row:key==1?commands[i].target.unique:commands[i].target.slot;
                if(threadIdx.x==0)total=count;
            }else{
                if(threadIdx.x==0){
                    scratch.invalid=0;
                    const auto pins=key==2?0u:plan.owner_.frame->initial_pin_count+plan.frame_.count;
                    if(pins>plan.owner_.declaration.maximumUnique()-count){plan.fail();state=0;}
                    else total=count+pins;
                }
                __syncthreads();
                if(!state)return false;
                const auto pins=total-count;
                for(std::uint64_t i=threadIdx.x;i<total;i+=gpu::Threads){
                    if(i<pins){
                        const auto initial=plan.owner_.frame->initial_pin_count;
                        const auto& pin=i<initial?plan.owner_.initial_pins[i]:plan.grants_[i-initial];
                        if(!record_validation::retained(pin,plan.frame_.lease.interval,
                            i<initial?p::GrantKind::Read:p::GrantKind::Reservation,plan.owner_.declaration.coordinate()))
                            atomicOr(&scratch.invalid,1u);
                        else first[i]=key==0?pin.row:pin.unique;
                    }else{
                        const auto& decision=decisions[i-pins];
                        first[i]=key==0?decision.row:key==1?decision.unique:
                            decision.kind==p::CommandKind::ReserveVictim?decision.victim.slot:adaptive_lookup::None;
                    }
                }
                __syncthreads();
                if(threadIdx.x==0 && scratch.invalid){plan.fail();state=0;}
            }
            __syncthreads();
            if(!state)return false;
            const auto policy=Action!=OwnerAction::Pin && key==2?
                p::DuplicateIdentityPolicy::IgnoreUnassignedSlot:p::DuplicateIdentityPolicy::AllValues;
            const bool valid=gpu::cooperativeExactIdentities(first,second,total,key?4u:8u,policy,scratch);
            if(threadIdx.x==0 && !valid){
                if constexpr(Action==OwnerAction::Pin)fail<Slot>(a);
                else plan.fail();
                state=0;
            }
            __syncthreads();
            if(!state)return false;
        }
        if(threadIdx.x==0){
            if constexpr(Action==OwnerAction::Pin)state=commitPins<Slot>(a,commands,count);
            else{
                constexpr auto wave=Action==OwnerAction::AcceptFree?p::AllocationWave::Free:p::AllocationWave::Victim;
                if(plan.owner_.declaration.capacity() && count>~std::uint64_t{0}-plan.owner_.owner->next_slot_generation){
                    plan.fail();state=0;
                }else state=plan.commitWave(wave,decisions,count);
            }
        }
        __syncthreads();
        return state!=0;
    }
};
} // namespace llaminar2::qwen4exp::pooled_native
