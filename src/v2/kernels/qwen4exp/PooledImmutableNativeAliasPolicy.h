/**
 * @file PooledImmutableNativeAliasPolicy.h
 * @brief Exact action/operand/original-root declarations used by actual native enqueue alias guards.
 *
 * These checks perform no physical preparation and grant no runtime authority.
 * Enqueue first authenticates original prepared allocations, then invokes these
 * same rules and the physical cross-root overlap guard. CPU regressions exercise
 * the rules through real CPU storage declarations, without inventing GPU roots.
 */
#pragma once
#include "PooledImmutableNativeOperands.h"
#include <array>
#include <stdexcept>
namespace llaminar2::qwen4exp::detail {
namespace alias_native=pooled_native;
/** @brief Require each canonical writable role to retain its exact original root and declared extent. */
inline void requireAliasRoots(const PooledImmutableNativeRoots& roots,std::array<size_t,3> bytes)
{
    const std::array values{roots.cache,roots.frame,roots.native};
    for(size_t i=0;i<3;++i){
        if(bool(values[i])!=(bytes[i]!=0) || (values[i] && (values[i]->is_view() ||
            values[i]->transferStorageOwner()!=values[i].get() || values[i]->size_bytes()!=bytes[i])))
            throw std::invalid_argument("Pooled alias guard requires original canonical exact-sized authority roots");
        for(size_t j=0;j<i;++j)if(values[i] && values[i]==values[j])
            throw std::invalid_argument("Pooled writable authority roles cannot share a root");
    }
}
/** @return One internal span is an exact phase input bank, never a partial overlap or physical-state alias. */
inline bool ownerAliasInput(alias_native::OwnerAction action,PooledNativeOperand role,const alias_native::OwnerParameters& p,
    size_t root,size_t offset,size_t bytes)
{
    namespace n=alias_native;namespace wire=n::p;
    if(root==2){
        if(role==PooledNativeOperand::Invocation)
            return p.native.ingress && offset==p.native.ingress+offsetof(n::OwnerIngress,invocation) && bytes==sizeof(n::Invocation);
        if(role==PooledNativeOperand::RecordBytes)
            return (action==n::OwnerAction::ReadRows || action==n::OwnerAction::Pin || action==n::OwnerAction::AcceptFree || action==n::OwnerAction::AcceptVictim || action==n::OwnerAction::ApplySourceDelivery) && p.native.ingress && offset==p.native.ingress+offsetof(n::OwnerIngress,record_bytes) && bytes==8;
        if(role==PooledNativeOperand::PayloadBytes)
            return action==n::OwnerAction::ApplySourceDelivery && p.native.ingress && offset==p.native.ingress+offsetof(n::OwnerIngress,payload_bytes) && bytes==8;
        if(role==PooledNativeOperand::Records && action==n::OwnerAction::ApplySourceDelivery)
            return p.native.source_fills && offset==p.native.source_fills && bytes==size_t{p.native.reply_capacity}*sizeof(n::Fill);
        if(role==PooledNativeOperand::Payload && action==n::OwnerAction::ApplySourceDelivery)
            return p.native.source_payload && offset==p.native.source_payload && bytes==size_t{p.native.reply_capacity}*p.row_bytes;
        if(((role==PooledNativeOperand::Outcome && (action==n::OwnerAction::ReadRows || action==n::OwnerAction::ValidateFills || action==n::OwnerAction::AcquirePublication || action==n::OwnerAction::MakeReady || action==n::OwnerAction::ReadyReceipt)) || (role==PooledNativeOperand::Frontier && (action==n::OwnerAction::ApplySourceDelivery || action==n::OwnerAction::CancelSource || action==n::OwnerAction::DrainSourceFailure))) && p.native.source_outcome && offset==p.native.source_outcome)
            return bytes==sizeof(n::SourceFrontier);
        // The outgoing Availability publication must retire before the incoming
        // plan overwrites this original64B slot. Only its exact nominal header
        // is readable; unrelated native progress/output aliases remain banned.
        if(role==PooledNativeOperand::Frontier && offset==p.native.header){
            if(action==n::OwnerAction::AcquireReaderRetirement)return bytes==sizeof(n::ReadersRetiredGate);
            if(action==n::OwnerAction::SealSource)return bytes==sizeof(n::SourceFrontier);
            if(action==n::OwnerAction::AcceptFree)return bytes==sizeof(n::FreePlanHeader);
            if(action==n::OwnerAction::AcceptVictim)return bytes==sizeof(n::VictimPlanHeader);
        }
        if(role==PooledNativeOperand::Frontier && p.owner.participants()==1 && offset==p.native.header){
            if(action==n::OwnerAction::AcquirePins)return bytes==sizeof(wire::PinHeader);
            if(action==n::OwnerAction::AcquireFree)return bytes==sizeof(wire::FreeAllocationReceipt);
            if(action==n::OwnerAction::AcquirePublication)return bytes==sizeof(wire::PublishHeader);
        }
    }
    if(root!=1)return false;
    if(role==PooledNativeOperand::Frontier){
        const auto participants=size_t{p.owner.participants()};
        if(action==n::OwnerAction::AcquirePins)return offset==p.frame.pin_headers && bytes==participants*sizeof(wire::PinHeader);
        if(action==n::OwnerAction::AcquireFree)return offset==p.frame.free_headers && bytes==participants*sizeof(wire::FreeAllocationReceipt);
        if(action==n::OwnerAction::AcquirePublication)return offset==p.frame.publish_headers && bytes==participants*sizeof(wire::PublishHeader);
    }
    // Schema validation already fixes each role's complete native extent. The
    // command bank has one start and never aliases the accepted allocation log.
    if(offset!=p.frame.commands || bytes>size_t{p.owner.maximumUnique()}*sizeof(wire::OwnerCommand))return false;
    if(role==PooledNativeOperand::Records)
        return action==n::OwnerAction::ReadRows || action==n::OwnerAction::Pin || action==n::OwnerAction::AcceptFree || action==n::OwnerAction::AcceptVictim;
    return false;
}
/** @brief Execute the actual owner's declaration/alias guard before prepared address submission. */
inline void requireOwnerAliasDeclarations(alias_native::OwnerAction action,const PooledImmutableOwnerOperands& inputs,
    const PooledImmutableNativeRoots& roots,const PooledImmutableOwnerNativePlan& original)
{
    inputs.requireDeclarations(action,original);const auto& p=original.parameters();
    requireAliasRoots(roots,{p.cache.bytes,p.frame.bytes,p.native.bytes});
    const std::array values{roots.cache,roots.frame,roots.native};
    const std::array ids{BufferId::QFN_POOLED_OWNER_CACHE,BufferId::QFN_POOLED_OWNER_FRAME,BufferId::QFN_POOLED_OWNER_NATIVE};
    for(size_t role=0;role<inputs.spans().size();++role)if(inputs.spans()[role]){
        const auto& input=*inputs.spans()[role];
        for(size_t root=0;root<3;++root){
            if((input.buffer==ids[root])!=(values[root] && input.root==values[root]))
                throw std::invalid_argument("Pooled owner input changed the original canonical authority role/root");
            if(values[root] && input.root==values[root] && !ownerAliasInput(action,static_cast<PooledNativeOperand>(role),p,root,input.offset,input.bytes))
                throw std::invalid_argument("Pooled owner operand aliases physical authority or an incompatible phase bank");
        }
    }
}
/** @return Exact reusable receive prefix/count/frontier for this actual coordinator phase. */
inline bool coordinatorAliasInput(alias_native::CoordinatorAction action,PooledNativeOperand role,const alias_native::CoordinatorParameters& p,
    size_t root,size_t offset,size_t bytes)
{
    namespace n=alias_native;namespace wire=n::p;
    if(root!=2)return false;
    const bool grants=action==n::CoordinatorAction::AcquirePins || action==n::CoordinatorAction::AcquireFree || action==n::CoordinatorAction::AcquireVictims;
    const bool releases=action==n::CoordinatorAction::Cancel;
    if(role==PooledNativeOperand::Records)return p.native.ingress_count && (grants || releases) && offset==p.native.pin_commands &&
        bytes==size_t{p.native.reply_capacity}*(releases?sizeof(wire::ReleaseReceipt):sizeof(wire::Grant));
    if(role==PooledNativeOperand::RecordBytes)return p.native.ingress_count && (grants || releases) && offset==p.native.ingress_count && bytes==8;
    if(role!=PooledNativeOperand::Frontier || !p.native.ingress_frontier || offset!=p.native.ingress_frontier)return false;
    size_t record=0;
    switch(action){
    case n::CoordinatorAction::SealPins:record=sizeof(wire::PinHeader);break;
    case n::CoordinatorAction::PlanFree:record=sizeof(n::Availability);break;
    case n::CoordinatorAction::CompleteFree:record=sizeof(wire::FreeAllocationReceipt);break;
    case n::CoordinatorAction::CompleteVictims:record=sizeof(wire::VictimAllocationReceipt);break;
    case n::CoordinatorAction::Publish:record=sizeof(wire::ReadyHeader);break;
    case n::CoordinatorAction::Finish:record=sizeof(wire::RetiredHeader);break;
    default:return false;
    }
    return bytes==size_t{p.policy.participants()}*record;
}
/** @brief Execute the actual coordinator's original-root/generation and action-specific alias declaration guard. */
inline void requireCoordinatorAliasDeclarations(alias_native::CoordinatorAction action,const PooledImmutableCoordinatorOperands& inputs,
    const PooledImmutableNativeRoots& roots,const PooledImmutableCoordinatorNativePlan& original)
{
    inputs.requireDeclarations(action,original);const auto& p=original.parameters();
    requireAliasRoots(roots,{p.cache.bytes,p.frame.bytes,p.native.bytes});
    const std::array values{roots.cache,roots.frame,roots.native};
    const std::array ids{BufferId::QFN_POOLED_COORDINATOR_POLICY,BufferId::QFN_POOLED_COORDINATOR_FRAME,BufferId::QFN_POOLED_COORDINATOR_NATIVE};
    for(size_t role=0;role<inputs.spans().size();++role)if(inputs.spans()[role]){
        const auto& input=*inputs.spans()[role];
        for(size_t root=0;root<3;++root){
            if((input.buffer==ids[root])!=(values[root] && input.root==values[root]))
                throw std::invalid_argument("Pooled coordinator input changed the original canonical authority role/root");
            if(values[root] && input.root==values[root] && !coordinatorAliasInput(action,static_cast<PooledNativeOperand>(role),p,root,input.offset,input.bytes))
                throw std::invalid_argument("Pooled coordinator input aliases policy/frame/output or an incompatible phase bank");
        }
    }
}
}
