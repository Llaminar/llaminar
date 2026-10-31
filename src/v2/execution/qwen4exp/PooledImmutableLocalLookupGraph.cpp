/**
 * @file PooledImmutableLocalLookupGraph.cpp
 * @brief Complete fixed provider/native-owner transaction with exact source branches and reader cleanup.
 *
 * Every node has a typed existing phase contract. Local directed allocation
 * consumers read bounded authoritative device offsets without copying/rewriting
 * original banks. Cross-device placement requires explicit transport instead;
 * this builder never hides a collective inside an ordinary compute stage.
 */
#include "PooledImmutableLocalLookupGraph.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>
namespace llaminar2 {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=n::p;
/** @brief Install one exact nonempty original schema span, never a dynamic communication extent. */
void input(q::PooledNativeOperandSpans& spans,q::PooledNativeOperand role,std::shared_ptr<TensorBase> root,
    BufferId buffer,std::size_t offset,std::size_t bytes){
    if(bytes)spans[static_cast<std::size_t>(role)]=q::PooledNativeTensorSpan{std::move(root),buffer,offset,bytes};
}
}
PooledImmutableLookupGraphEndpoints appendPooledImmutableLocalLookupGraph(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,
    n::OriginalRows original,q::PooledNativeOperandSpans begin,DeviceGenerationControlBinding control,
    q::PooledNativeTensorSpan output,const std::string& identity){
    if(isGraphCaptureActive() || !boundary || !boundary->isPooled() || !code || identity.empty())
        throw std::invalid_argument("Local pooled lookup graph requires its original real-provider boundary outside capture");
    const auto source=boundary->pooledStorage();const auto owner=source->owner();const auto coordinator=source->coordinator();
    const auto& o=owner->plan().parameters();const auto& c=coordinator->plan().parameters();const auto& so=source->geometry().parameters().offsets;
    if(o.owner.participants()!=1 || o.owner.coordinate()!=0 || o.device!=c.device || o.device!=boundary->device() || code->backend()!=o.device.type)
        throw std::invalid_argument("Pooled graph edge requires explicit directed owner transport; local wiring cannot represent its declared endpoints");
    graph.setRetainedStorageBoundary(boundary);
    const auto own=owner->roots(),coord=coordinator->roots(),src=source->roots();
    const auto capacity=o.native.reply_capacity;PooledImmutableLookupGraphEndpoints endpoints;std::size_t ordinal=0;
    const auto append=[&](std::unique_ptr<IComputeStage> stage){
        const auto name=identity+"/"+std::to_string(ordinal++);graph.addNode(name,std::move(stage),boundary->device());
        if(endpoints.first.empty())endpoints.first=name;
        if(!endpoints.last.empty())graph.addDependency(name,endpoints.last);
        endpoints.last=name;
    };
    const auto sourceInput=[&](q::PooledNativeOperandSpans& spans,q::PooledNativeOperand role){
        input(spans,role,src.native,BufferId::QFN_POOLED_SOURCE_NATIVE,so.frontier,sizeof(n::SourceFrontier));
    };
    const auto ownerPhase=[&](n::OwnerAction action,n::SourceCondition condition=n::SourceCondition::Always){
        q::PooledNativeOperandSpans spans{};auto layout=n::RecordLayout::Compact;
        input(spans,q::PooledNativeOperand::Invocation,coord.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.invocation,sizeof(n::Invocation));
        switch(action){
        case n::OwnerAction::Pin:
            input(spans,q::PooledNativeOperand::Records,coord.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.pin_commands,std::size_t{capacity}*sizeof(p::OwnerCommand));
            input(spans,q::PooledNativeOperand::RecordBytes,coord.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.pin_slices+offsetof(n::DirectedSlice,live_bytes),capacity?8:0);break;
        case n::OwnerAction::AcquirePins:
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::PinHeader));break;
        case n::OwnerAction::AcceptFree:case n::OwnerAction::AcceptVictim:{
            const bool victim=action==n::OwnerAction::AcceptVictim;
            input(spans,q::PooledNativeOperand::Frontier,coord.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,
                victim?c.native.victim_headers:c.native.free_headers,victim?sizeof(n::VictimPlanHeader):sizeof(n::FreePlanHeader));
            if(capacity){layout=n::RecordLayout::DirectedAllocationBank;
                input(spans,q::PooledNativeOperand::Records,coord.frame,BufferId::QFN_POOLED_COORDINATOR_FRAME,c.frame.decisions,std::size_t{o.owner.maximumUnique()}*sizeof(p::AllocationDecision));
                input(spans,q::PooledNativeOperand::AllocationSlice,coord.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,victim?c.native.victim_slices:c.native.free_slices,sizeof(n::DirectedSlice));
            }break;
        }
        case n::OwnerAction::AcquireFree:
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::FreeAllocationReceipt));break;
        case n::OwnerAction::SealSource:case n::OwnerAction::CancelSource:
            sourceInput(spans,q::PooledNativeOperand::Frontier);break;
        case n::OwnerAction::ApplySourceDelivery:
            sourceInput(spans,q::PooledNativeOperand::Frontier);
            input(spans,q::PooledNativeOperand::Records,src.frame,BufferId::QFN_POOLED_SOURCE_FRAME,so.fills,std::size_t{capacity}*sizeof(n::Fill));
            input(spans,q::PooledNativeOperand::RecordBytes,src.native,BufferId::QFN_POOLED_SOURCE_NATIVE,so.record_bytes,capacity?8:0);
            input(spans,q::PooledNativeOperand::Payload,src.frame,BufferId::QFN_POOLED_SOURCE_FRAME,so.payload,std::size_t{capacity}*o.row_bytes);
            input(spans,q::PooledNativeOperand::PayloadBytes,src.native,BufferId::QFN_POOLED_SOURCE_NATIVE,so.payload_bytes,capacity?8:0);break;
        case n::OwnerAction::AcquirePublication:
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::PublishHeader));break;
        default:break;
        }
        if(condition!=n::SourceCondition::Always && action!=n::OwnerAction::ApplySourceDelivery && action!=n::OwnerAction::CancelSource)
            sourceInput(spans,q::PooledNativeOperand::Outcome);
        q::PooledImmutableOwnerOperands operands(action,owner->plan(),std::move(spans),condition,layout);
        append(std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,owner,std::move(operands)}));
    };
    const auto coordinatorPhase=[&](n::CoordinatorAction action,n::SourceCondition condition=n::SourceCondition::Always){
        q::PooledNativeOperandSpans spans{};std::optional<n::OriginalRows> rows;
        if(action==n::CoordinatorAction::Begin){spans=begin;rows=original;}
        switch(action){
        case n::CoordinatorAction::AcquirePins:case n::CoordinatorAction::AcquireFree:case n::CoordinatorAction::AcquireVictims:
            input(spans,q::PooledNativeOperand::Records,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.replies,std::size_t{capacity}*sizeof(p::Grant));
            input(spans,q::PooledNativeOperand::RecordBytes,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.progress+offsetof(n::OwnerProgress,reply_bytes),capacity?8:0);break;
        case n::CoordinatorAction::SealPins:
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::PinHeader));break;
        case n::CoordinatorAction::PlanFree:
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(n::Availability));break;
        case n::CoordinatorAction::CompleteFree:
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::FreeAllocationReceipt));break;
        case n::CoordinatorAction::CompleteVictims:
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::VictimAllocationReceipt));break;
        case n::CoordinatorAction::SealSource:sourceInput(spans,q::PooledNativeOperand::Source);break;
        case n::CoordinatorAction::Publish:
            sourceInput(spans,q::PooledNativeOperand::Source);
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::ReadyHeader));break;
        case n::CoordinatorAction::Cancel:
            sourceInput(spans,q::PooledNativeOperand::Source);
            input(spans,q::PooledNativeOperand::Records,own.frame,BufferId::QFN_POOLED_OWNER_FRAME,o.frame.release_receipts,std::size_t{capacity}*sizeof(p::ReleaseReceipt));
            input(spans,q::PooledNativeOperand::RecordBytes,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.progress+offsetof(n::OwnerProgress,release_bytes),capacity?8:0);break;
        case n::CoordinatorAction::Finish:
            input(spans,q::PooledNativeOperand::Frontier,own.native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::RetiredHeader));break;
        default:break;
        }
        q::PooledImmutableCoordinatorOperands operands(action,coordinator->plan(),rows,std::move(spans),condition);
        append(std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{action,coordinator,std::move(operands)}));
    };
    const auto sourcePhase=[&](q::PooledSourceAction action){append(std::make_unique<PooledImmutableSourcePhaseStage>(arena,boundary,action,control,
        action==q::PooledSourceAction::Read?std::optional(output):std::nullopt));};
    coordinatorPhase(n::CoordinatorAction::Begin);ownerPhase(n::OwnerAction::Begin);
    coordinatorPhase(n::CoordinatorAction::PinCommands);ownerPhase(n::OwnerAction::Pin);ownerPhase(n::OwnerAction::SealPins);
    coordinatorPhase(n::CoordinatorAction::AcquirePins);coordinatorPhase(n::CoordinatorAction::SealPins);ownerPhase(n::OwnerAction::AcquirePins);
    ownerPhase(n::OwnerAction::Availability);coordinatorPhase(n::CoordinatorAction::PlanFree);
    ownerPhase(n::OwnerAction::AcceptFree);ownerPhase(n::OwnerAction::ExecuteFree);coordinatorPhase(n::CoordinatorAction::AcquireFree);
    ownerPhase(n::OwnerAction::FreeReceipt);coordinatorPhase(n::CoordinatorAction::CompleteFree);ownerPhase(n::OwnerAction::AcquireFree);
    coordinatorPhase(n::CoordinatorAction::PlanVictims);ownerPhase(n::OwnerAction::AcceptVictim);ownerPhase(n::OwnerAction::ExecuteVictim);
    coordinatorPhase(n::CoordinatorAction::AcquireVictims);ownerPhase(n::OwnerAction::VictimReceipt);coordinatorPhase(n::CoordinatorAction::CompleteVictims);
    sourcePhase(q::PooledSourceAction::Plan);coordinatorPhase(n::CoordinatorAction::SealSource);ownerPhase(n::OwnerAction::SealSource);
    sourcePhase(q::PooledSourceAction::Publish);sourcePhase(q::PooledSourceAction::Acquire);
    ownerPhase(n::OwnerAction::ApplySourceDelivery,n::SourceCondition::AfterSuccess);
    ownerPhase(n::OwnerAction::ValidateFills,n::SourceCondition::AfterSuccess);ownerPhase(n::OwnerAction::AcquirePublication,n::SourceCondition::AfterSuccess);
    ownerPhase(n::OwnerAction::MakeReady,n::SourceCondition::AfterSuccess);ownerPhase(n::OwnerAction::ReadyReceipt,n::SourceCondition::AfterSuccess);
    coordinatorPhase(n::CoordinatorAction::Publish,n::SourceCondition::AfterSuccess);
    sourcePhase(q::PooledSourceAction::Read);sourcePhase(q::PooledSourceAction::ReadersRetired);
    ownerPhase(n::OwnerAction::CancelSource,n::SourceCondition::AfterFailure);ownerPhase(n::OwnerAction::Release);
    coordinatorPhase(n::CoordinatorAction::Cancel,n::SourceCondition::AfterFailure);
    ownerPhase(n::OwnerAction::Retire);coordinatorPhase(n::CoordinatorAction::Finish);
    return endpoints;
}
}
