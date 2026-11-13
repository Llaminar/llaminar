/**
 * @file PooledImmutableParticipantLocalGraph.cpp
 * @brief Checked original same-device inputs and complete mixed cohort retention.
 *
 * All pointers remain original prepared Tensor roots. Mutable slices, metadata
 * and counts are read only by the native phase on its exact captured stream.
 * No binding materializes storage, transports to self or duplicates PMA claims.
 */
#include "PooledImmutableParticipantLocalGraph.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableSourcePhaseStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>
#include <type_traits>
#include <algorithm>

namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;namespace p=n::p;
/** @brief Declare one exact original subrange; absent C0 roles remain absent. */
void input(PooledNativeOperandSpans& spans,PooledNativeOperand role,const std::shared_ptr<TensorBase>& root,
    BufferId id,std::size_t offset,std::size_t bytes){
    if(!bytes)return;
    if(!root || root->is_view() || root->transferStorageOwner()!=root.get() ||
        offset>root->size_bytes() || bytes>root->size_bytes()-offset)
        throw std::invalid_argument("Local pooled phase lost its exact canonical root/subrange");
    spans[static_cast<unsigned>(role)]=PooledNativeTensorSpan{root,id,offset,bytes};
}

/** @return Stable complete combination, without hashing any live count. */
std::uint64_t combine(std::uint64_t a,std::uint64_t b){return ((a^b)*1099511628211ULL)|1;}
}
PooledImmutableLocalOwnerBinding::PooledImmutableLocalOwnerBinding(
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
    std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PhysicalMemoryAuthority> memory)
    :coordinator_(std::move(coordinator)),owner_(std::move(owner)),memory_(std::move(memory)),
     coordinator_identity_(coordinator_->captureIdentity()),owner_identity_(owner_->captureIdentity()){}
std::shared_ptr<const PooledImmutableLocalOwnerBinding> PooledImmutableLocalOwnerBinding::bind(
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
    std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PhysicalMemoryAuthority> memory){
    if(isGraphCaptureActive() || !coordinator || !owner || !memory)
        throw std::invalid_argument("Local pooled binding requires original cold prepared owners and PMA");
    coordinator->requireAuthority(memory);owner->requireAuthority(memory);coordinator->plan().requireOwnerPlan(owner->plan());
    if(coordinator->plan().parameters().policy.participants()<2 || !coordinator->plan().parameters().native.ingress_frontier)
        throw std::invalid_argument("Mixed local binding requires its original distributed frontier; P1 uses the existing local graph");
    const auto edge=PooledImmutablePinEdge::fromPlans(coordinator->plan(),owner->plan());
    if(edge.remote())throw std::invalid_argument("Local pooled binding requires exact same allocator identity");
    return std::shared_ptr<const PooledImmutableLocalOwnerBinding>(new PooledImmutableLocalOwnerBinding(
        std::move(coordinator),std::move(owner),std::move(memory)));
}
void PooledImmutableLocalOwnerBinding::requireCurrent()const{
    coordinator_->requireAuthority(memory_);owner_->requireAuthority(memory_);coordinator_->plan().requireOwnerPlan(owner_->plan());
    if(coordinator_->plan().allocator()!=owner_->plan().allocator() ||
        coordinator_->captureIdentity()!=coordinator_identity_ || owner_->captureIdentity()!=owner_identity_)
        throw std::logic_error("Local pooled binding changed original roots/runtime/cohort");
}
std::uint64_t PooledImmutableLocalOwnerBinding::captureIdentity()const{
    requireCurrent();return combine(combine(coordinator_identity_,owner_identity_),coordinate());
}
PooledNativeOperandSpans PooledImmutableLocalOwnerBinding::ownerInputs(n::OwnerAction action)const{
    requireCurrent();const auto& c=coordinator_->plan().parameters();const auto& o=owner_->plan().parameters();
    const auto roots=coordinator_->roots();PooledNativeOperandSpans out{};
    input(out,PooledNativeOperand::Invocation,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.invocation,sizeof(n::Invocation));
    switch(action){
    case n::OwnerAction::Pin:
        if(hasPayload()){
            input(out,PooledNativeOperand::Records,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.pin_commands,size_t{c.native.reply_capacity}*sizeof(p::OwnerCommand));
            input(out,PooledNativeOperand::AllocationSlice,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.pin_slices+size_t{coordinate()}*sizeof(n::DirectedSlice),sizeof(n::DirectedSlice));
        }break;
    case n::OwnerAction::AcquirePins:case n::OwnerAction::AcquireFree:
        input(out,PooledNativeOperand::Frontier,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.ingress_frontier,
            size_t{o.owner.participants()}*(action==n::OwnerAction::AcquirePins?sizeof(p::PinHeader):sizeof(p::FreeAllocationReceipt)));break;
    case n::OwnerAction::AcceptFree:case n::OwnerAction::AcceptVictim:{
        const bool victim=action==n::OwnerAction::AcceptVictim;
        input(out,PooledNativeOperand::Frontier,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,
            (victim?c.native.victim_headers:c.native.free_headers)+size_t{coordinate()}*sizeof(n::FreePlanHeader),sizeof(n::FreePlanHeader));
        if(hasPayload()){
            input(out,PooledNativeOperand::Records,roots.frame,BufferId::QFN_POOLED_COORDINATOR_FRAME,c.frame.decisions,size_t{c.policy.maximumUnique()}*sizeof(p::AllocationDecision));
            input(out,PooledNativeOperand::AllocationSlice,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,
                (victim?c.native.victim_slices:c.native.free_slices)+size_t{coordinate()}*sizeof(n::DirectedSlice),sizeof(n::DirectedSlice));
        }break;
    }
    case n::OwnerAction::Begin:case n::OwnerAction::SealPins:case n::OwnerAction::Availability:
    case n::OwnerAction::ExecuteFree:case n::OwnerAction::FreeReceipt:
    case n::OwnerAction::ExecuteVictim:case n::OwnerAction::VictimReceipt:break;
    default:throw std::invalid_argument("Local pre-source adapter received an unrelated owner phase");
    }
    return out;
}
PooledImmutableOwnerOperands PooledImmutableLocalOwnerBinding::ownerOperands(n::OwnerAction action)const{
    auto layout=n::RecordLayout::Compact;
    if(hasPayload()){
        if(action==n::OwnerAction::Pin)layout=n::RecordLayout::DirectedCommandBank;
        if(action==n::OwnerAction::AcceptFree || action==n::OwnerAction::AcceptVictim)layout=n::RecordLayout::DirectedAllocationBank;
    }
    return PooledImmutableOwnerOperands(action,owner_->plan(),ownerInputs(action),n::SourceCondition::Always,layout,
        layout==n::RecordLayout::DirectedCommandBank?std::optional(coordinator_->plan()):std::nullopt);
}
PooledImmutableCoordinatorOperands PooledImmutableLocalOwnerBinding::coordinatorOperands(n::CoordinatorAction action)const{
    requireCurrent();const auto& o=owner_->plan().parameters();const auto roots=owner_->roots();PooledNativeOperandSpans out{};
    if(n::localFrontierImport(action)){
        input(out,PooledNativeOperand::Frontier,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,n::localFrontierBytes(action));
        input(out,PooledNativeOperand::RecordBytes,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.progress+offsetof(n::OwnerProgress,header_bytes),sizeof(std::uint64_t));
        input(out,PooledNativeOperand::OwnerControl,roots.frame,BufferId::QFN_POOLED_OWNER_FRAME,o.frame.control,sizeof(p::OwnerFrame));
        input(out,PooledNativeOperand::AllocationControl,roots.frame,BufferId::QFN_POOLED_OWNER_FRAME,o.frame.allocation_control,sizeof(p::AllocationFrame));
        input(out,PooledNativeOperand::HeaderKind,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.progress+offsetof(n::OwnerProgress,header_kind),sizeof(n::OwnerHeaderKind));
    }else if(action==n::CoordinatorAction::AcquirePins || action==n::CoordinatorAction::AcquireFree || action==n::CoordinatorAction::AcquireVictims){
        if(!hasPayload())throw std::invalid_argument("C0 has no local Grant acquisition operation");
        input(out,PooledNativeOperand::Records,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.replies,size_t{o.native.reply_capacity}*sizeof(p::Grant));
        input(out,PooledNativeOperand::RecordBytes,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.progress+offsetof(n::OwnerProgress,reply_bytes),sizeof(std::uint64_t));
    }else throw std::invalid_argument("Local adapter received an unrelated coordinator operation");
    return PooledImmutableCoordinatorOperands(action,coordinator_->plan(),std::nullopt,std::move(out),n::SourceCondition::Always,owner_);
}
PooledImmutableMixedCohort PooledImmutableMixedCohort::fromOriginal(std::vector<PooledImmutablePinParticipant> owners){
    if(isGraphCaptureActive() || owners.empty())throw std::invalid_argument("Mixed pooled cohort requires actual cold original members");
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator;std::uint32_t local=adaptive_lookup::None;
    for(size_t i=0;i<owners.size();++i)std::visit([&](const auto& binding){
        if(!binding)throw std::invalid_argument("Mixed pooled cohort has a missing original member");binding->requireCurrent();
        using Binding=typename std::decay_t<decltype(binding)>::element_type;
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> c;std::shared_ptr<PooledImmutableOwnerPreparedStorage> o;
        if constexpr(std::is_same_v<Binding,const PooledImmutableLocalOwnerBinding>){
            if(local!=adaptive_lookup::None)throw std::invalid_argument("Mixed pooled cohort contains duplicate local owners");
            local=static_cast<std::uint32_t>(i);c=binding->coordinatorStorage();o=binding->ownerStorage();
        }else{c=binding->forward()->coordinatorStorage();o=binding->forward()->ownerStorage();}
        if(!coordinator)coordinator=c;
        if(c!=coordinator || o->plan().parameters().owner.coordinate()!=i)throw std::invalid_argument("Mixed pooled cohort changed original coordinator/member order");
        coordinator->plan().requireOwnerPlan(o->plan());
    },owners[i]);
    if(local==adaptive_lookup::None || owners.size()!=coordinator->plan().parameters().policy.participants())
        throw std::invalid_argument("Mixed pooled cohort is incomplete or lacks its actual local owner");
    return PooledImmutableMixedCohort(std::move(owners),std::move(coordinator),local);
}
void PooledImmutableMixedCohort::requireCurrent()const{
    if(owners_.size()!=coordinator_->plan().parameters().policy.participants() || local_>=owners_.size())
        throw std::logic_error("Mixed pooled cohort lost original complete membership");
    for(size_t i=0;i<owners_.size();++i)std::visit([&](const auto& binding){
        if(!binding)throw std::logic_error("Mixed pooled cohort lost original member");binding->requireCurrent();
        using Binding=typename std::decay_t<decltype(binding)>::element_type;
        if constexpr(std::is_same_v<Binding,const PooledImmutableLocalOwnerBinding>){
            if(i!=local_ || binding->coordinate()!=i || binding->coordinatorStorage()!=coordinator_)
                throw std::logic_error("Mixed pooled cohort local coordinate changed");
        }else if(i==local_ || binding->forward()->declaration().coordinate()!=i || binding->forward()->coordinatorStorage()!=coordinator_)
            throw std::logic_error("Mixed pooled cohort remote coordinate changed");
    },owners_[i]);
}
}

namespace llaminar2 {
void requirePooledOriginalDependencies(const ComputeGraph& graph,const std::string& consumer,
    std::vector<std::string> expected)
{
    if(isGraphCaptureActive())throw std::invalid_argument("Pooled phase dependency authentication must precede capture");
    const auto* node=graph.getNode(consumer);
    if(!node)throw std::invalid_argument("Pooled phase dependency authentication lost its actual consumer");
    auto actual=node->dependencies;
    if(const auto* source=dynamic_cast<const PooledImmutableSourcePhaseStage*>(node->stage.get())){
        for(const auto& trunk:source->currentDemandTrunk()){
            if(std::count(actual.begin(),actual.end(),trunk)!=1 || !graph.getNode(trunk) ||
               std::find(expected.begin(),expected.end(),trunk)!=expected.end())
                throw std::invalid_argument("Consumed source demand lost or aliased its original trunk join");
            actual.erase(std::find(actual.begin(),actual.end(),trunk));
        }
    }
    std::sort(actual.begin(),actual.end());std::sort(expected.begin(),expected.end());
    expected.erase(std::unique(expected.begin(),expected.end()),expected.end());
    if(actual!=expected)throw std::invalid_argument("Pooled phase has missing/foreign/out-of-order original predecessors");
}
}
