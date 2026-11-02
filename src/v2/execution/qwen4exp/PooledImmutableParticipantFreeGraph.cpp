/**
 * @file PooledImmutableParticipantFreeGraph.cpp
 * @brief Exact original Availability/free-plan/free-receipt wire bindings and participant-local graph order.
 *
 * Channel and packet storage are phase-reused only after the actual preceding
 * command/read frontier. Native phase stages remain the only reservation/state
 * writers; transfer stages execute one explicit endpoint and never coordinate
 * another participant or read a mutable device count on the host.
 */
#include "PooledImmutableParticipantFreeGraph.h"
#include "PooledImmutableDirectedOperation.h"
#include "PooledImmutableDirectedMessageGeometry.h"
#include "PooledImmutableDirectedChannelGeometry.h"
#include "PooledImmutableParticipantLocalGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <set>
#include <type_traits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;namespace p=n::p;
/** @return Checked nominal role index; invalid enum values cannot select another bank. */
size_t roleIndex(PooledFreeWireRole role){const auto i=static_cast<unsigned>(role);if(i>=6)throw std::invalid_argument("Unknown original free-wave wire action");return i;}
/** @return Whether this action travels owner→coordinator, without observing live state. */
bool reverse(PooledFreeWireRole role){(void)roleIndex(role);return role==PooledFreeWireRole::Availability || role==PooledFreeWireRole::Grants || role==PooledFreeWireRole::Receipt;}
/** @return Original root for one named role, never a view or independently allocated packet. */
std::shared_ptr<TensorBase> root(const PooledImmutableNativeRoots& roots,BufferId id){return id==BufferId::QFN_POOLED_COORDINATOR_FRAME || id==BufferId::QFN_POOLED_OWNER_FRAME?roots.frame:roots.native;}
/** @brief Retain one exact original named native input. */
void operand(PooledNativeOperandSpans& out,PooledNativeOperand role,const PooledImmutableNativeRoots& roots,PooledPinWireSpan span){
    if(!span.bytes)return;const auto value=root(roots,span.buffer);
    if(!value || value->is_view() || value->transferStorageOwner()!=value.get() || span.offset>value->size_bytes() || span.bytes>value->size_bytes()-span.offset)
        throw std::invalid_argument("Free wave lost original canonical root/subrange");
    out[static_cast<size_t>(role)]=PooledNativeTensorSpan{value,span.buffer,span.offset,span.bytes};
}
}
PooledImmutableFreeEdge PooledImmutableFreeEdge::fromOriginal(PooledImmutablePinEdge original){
    if(!original.remote())throw std::invalid_argument("Distributed free edge requires distinct original allocators; supported local P1 retains its existing direct graph");
    return PooledImmutableFreeEdge(std::move(original));
}
size_t PooledImmutableFreeEdge::maximumDecisionBytes()const noexcept{return PooledImmutableDirectedMessageGeometry::fromPlan(original_.owner()).decisions();}
PooledPinWireSpan PooledImmutableFreeEdge::source(PooledFreeWireRole role)const {
    (void)roleIndex(role);const auto& c=original_.coordinator().parameters();const auto& o=original_.owner().parameters();
    switch(role){
    case PooledFreeWireRole::Availability:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(n::Availability)};
    case PooledFreeWireRole::Plan:return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.free_headers+size_t{coordinate()}*sizeof(n::FreePlanHeader),sizeof(n::FreePlanHeader)};
    case PooledFreeWireRole::Decisions:return {BufferId::QFN_POOLED_COORDINATOR_FRAME,c.frame.decisions,size_t{c.policy.maximumUnique()}*sizeof(p::AllocationDecision)};
    case PooledFreeWireRole::Grants:return original_.replySource();
    case PooledFreeWireRole::Receipt:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(p::FreeAllocationReceipt)};
    case PooledFreeWireRole::Frontier:return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.ingress_frontier,PooledImmutableDirectedMessageGeometry::fromPlan(original_.owner()).freeReceipts()};
    }
    throw std::invalid_argument("Unknown original free source");
}
PooledPinWireSpan PooledImmutableFreeEdge::destination(PooledFreeWireRole role)const {
    (void)roleIndex(role);const auto& c=original_.coordinator().parameters();const auto& o=original_.owner().parameters();
    switch(role){
    case PooledFreeWireRole::Availability:return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.ingress_frontier+size_t{coordinate()}*sizeof(n::Availability),sizeof(n::Availability)};
    case PooledFreeWireRole::Plan:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(n::FreePlanHeader)};
    case PooledFreeWireRole::Decisions:return {BufferId::QFN_POOLED_OWNER_FRAME,o.frame.commands,maximumDecisionBytes()};
    case PooledFreeWireRole::Grants:return original_.replyDestination();
    case PooledFreeWireRole::Receipt:return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.ingress_frontier+size_t{coordinate()}*sizeof(p::FreeAllocationReceipt),sizeof(p::FreeAllocationReceipt)};
    case PooledFreeWireRole::Frontier:return {BufferId::QFN_POOLED_OWNER_FRAME,o.frame.free_headers,size_t{o.owner.participants()}*sizeof(p::FreeAllocationReceipt)};
    }
    throw std::invalid_argument("Unknown original free destination");
}
PooledPinWireSpan PooledImmutableFreeEdge::decisionSlice()const noexcept{const auto& c=original_.coordinator().parameters();return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.free_slices+size_t{coordinate()}*sizeof(n::DirectedSlice),sizeof(n::DirectedSlice)};}
size_t PooledImmutableFreeEdge::forwardChannelMaximum()const noexcept{return PooledImmutableDirectedChannelGeometry::fromPlans(original_.coordinator(),original_.owner()).forwardMaximum();}
size_t PooledImmutableFreeEdge::reverseChannelMaximum()const noexcept{return PooledImmutableDirectedChannelGeometry::fromPlans(original_.coordinator(),original_.owner()).reverseMaximum();}
uint64_t PooledImmutableFreeEdge::messageKey(PooledFreeWireRole role)const{(void)roleIndex(role);return ((original_.identity()^(16+static_cast<unsigned>(role)))*1099511628211ULL)|1;}
std::shared_ptr<const PooledImmutableCapturedFreeWave> PooledImmutableCapturedFreeWave::bind(
    std::shared_ptr<const PooledImmutableCapturedPinReturn> pin,std::shared_ptr<PhysicalMemoryAuthority> memory)
{
    if(isGraphCaptureActive() || !pin || !memory)throw std::invalid_argument("Free wave binding requires actual cold Pin owners and same PMA");
    pin->requireCurrent();const auto forward=pin->forward();forward->coordinatorStorage()->requireAuthority(memory);forward->ownerStorage()->requireAuthority(memory);
    auto d=PooledImmutableFreeEdge::fromOriginal(forward->declaration());
    if(!d.original().capturedCrossVendor() || !forward->channel_ || !pin->channel_ || memory->worldRank()!=d.original().sender().world_rank)
        throw std::invalid_argument("Free wave needs actual original cross-vendor channels; native/rank factory remains guarded");
    const auto coord=forward->coordinatorStorage()->roots(),own=forward->ownerStorage()->roots();auto& transfer=TransferEngine::instance();Bindings bindings{};
    for(unsigned index=0;index<6;++index){const auto role=static_cast<PooledFreeWireRole>(index);
        if(!d.hasPayload() && (role==PooledFreeWireRole::Decisions || role==PooledFreeWireRole::Grants))continue;
        const bool back=reverse(role);const auto& channel=back?pin->channel_:forward->channel_;const auto source=d.source(role),destination=d.destination(role);
        const auto maxBytes=role==PooledFreeWireRole::Decisions?d.maximumDecisionBytes():source.bytes;
        auto send=transfer.bindCapturedTransfer(channel,CapturedTransferEndpoint::Producer,{d.messageKey(role),maxBytes},root(back?own:coord,source.buffer),source.offset);
        auto receive=transfer.bindCapturedTransfer(channel,CapturedTransferEndpoint::Consumer,{d.messageKey(role),maxBytes},root(back?coord:own,destination.buffer),destination.offset);
        if(role==PooledFreeWireRole::Decisions){const auto slice=d.decisionSlice(),count=*d.decisionCount();
            send=transfer.bindDeviceSlicedTransfer(std::move(send),coord.native,slice.offset,source.bytes,source.offset);
            receive=transfer.bindDeviceCountedTransfer(std::move(receive),own.native,count.offset);
        }else if(role==PooledFreeWireRole::Grants){const auto count=d.grantCount(),incoming=*d.incomingGrantCount();
            send=transfer.bindDeviceCountedTransfer(std::move(send),own.native,count.offset);
            receive=transfer.bindDeviceCountedTransfer(std::move(receive),coord.native,incoming.offset);
        }
        bindings[index].emplace(std::move(send),std::move(receive));
    }
    return std::shared_ptr<const PooledImmutableCapturedFreeWave>(new PooledImmutableCapturedFreeWave(std::move(pin),std::move(memory),std::move(d),std::move(bindings)));
}
void PooledImmutableCapturedFreeWave::requireCurrent()const{pin_->requireCurrent();pin_->forward()->coordinatorStorage()->requireAuthority(memory_);pin_->forward()->ownerStorage()->requireAuthority(memory_);}
const CapturedTransferBinding& PooledImmutableCapturedFreeWave::binding(PooledFreeWireRole role,CapturedTransferEndpoint endpoint)const {
    requireCurrent();const auto index=roleIndex(role);if(endpoint!=CapturedTransferEndpoint::Producer && endpoint!=CapturedTransferEndpoint::Consumer)throw std::invalid_argument("Unknown free-wave endpoint");
    if(!bindings_[index])throw std::invalid_argument("C0 free wave has no decision/Grant payload transaction");return endpoint==CapturedTransferEndpoint::Producer?bindings_[index]->first:bindings_[index]->second;
}
PooledNativeOperandSpans PooledImmutableCapturedFreeWave::ownerInputs(n::OwnerAction action)const {
    requireCurrent();const auto original=pin_->forward();auto result=original->ownerInputs(n::OwnerAction::Begin);const auto roots=original->ownerStorage()->roots();
    switch(action){
    case n::OwnerAction::Availability:case n::OwnerAction::ExecuteFree:case n::OwnerAction::FreeReceipt:break;
    case n::OwnerAction::AcceptFree:
        operand(result,PooledNativeOperand::Frontier,roots,declaration_.destination(PooledFreeWireRole::Plan));
        if(declaration_.hasPayload()){operand(result,PooledNativeOperand::Records,roots,declaration_.destination(PooledFreeWireRole::Decisions));operand(result,PooledNativeOperand::RecordBytes,roots,*declaration_.decisionCount());}break;
    case n::OwnerAction::AcquireFree:operand(result,PooledNativeOperand::Frontier,roots,declaration_.destination(PooledFreeWireRole::Frontier));break;
    default:throw std::invalid_argument("Free wave cannot supply another owner phase");
    }
    return result;
}
PooledNativeOperandSpans PooledImmutableCapturedFreeWave::coordinatorInputs(n::CoordinatorAction action)const {
    requireCurrent();const auto& p=pin_->forward()->coordinatorStorage()->plan().parameters();const auto roots=pin_->forward()->coordinatorStorage()->roots();PooledNativeOperandSpans result{};
    if(action==n::CoordinatorAction::PlanFree)operand(result,PooledNativeOperand::Frontier,roots,{BufferId::QFN_POOLED_COORDINATOR_NATIVE,p.native.ingress_frontier,size_t{p.policy.participants()}*sizeof(n::Availability)});
    else if(action==n::CoordinatorAction::CompleteFree)operand(result,PooledNativeOperand::Frontier,roots,declaration_.source(PooledFreeWireRole::Frontier));
    else if(action==n::CoordinatorAction::AcquireFree && declaration_.hasPayload()){operand(result,PooledNativeOperand::Records,roots,declaration_.destination(PooledFreeWireRole::Grants));operand(result,PooledNativeOperand::RecordBytes,roots,*declaration_.incomingGrantCount());}
    else throw std::invalid_argument("Free wave cannot supply another coordinator phase");return result;
}
uint64_t PooledImmutableCapturedFreeWave::captureIdentity()const {
    requireCurrent();auto value=pin_->captureIdentity();for(size_t index=0;index<bindings_.size();++index){value=((value^index)*1099511628211ULL)|1;if(bindings_[index]){value=((value^bindings_[index]->first.captureIdentity())*1099511628211ULL)|1;value=((value^bindings_[index]->second.captureIdentity())*1099511628211ULL)|1;}}return value;
}
}
namespace llaminar2 {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;
/** @return True for one owner→coordinator free action; no model state is consulted. */
bool reverseFree(q::PooledFreeWireRole role){return role==q::PooledFreeWireRole::Availability || role==q::PooledFreeWireRole::Grants || role==q::PooledFreeWireRole::Receipt;}
/** @brief One explicit endpoint, never hidden cross-device orchestration or a host count observation. */
class PooledFreeTransferStage final:public IComputeStage {
public:
    /** @return Exact original semantic endpoint and native dispatch inputs before capture. */
    q::PooledImmutableDirectedTransferOperation directedTransferOperation()const{
        requireDeclarations();const auto& binding=edge_->binding(role_,endpoint_);
        return {q::PooledImmutableDirectedWireFamily::Free,static_cast<std::uint32_t>(role_),endpoint_,
            binding.message().bytes,binding.extentSource(),binding.hasSourceSlice()};
    }

    /** @brief Retain original prepared owners and their actual endpoint before freezing the capture identity. */
    PooledFreeTransferStage(BufferArena& arena,std::shared_ptr<const q::PooledImmutableCapturedFreeWave> edge,q::PooledFreeWireRole role,CapturedTransferEndpoint endpoint)
        :IComputeStage((reverseFree(role)==(endpoint==CapturedTransferEndpoint::Producer))?edge->declaration().original().receiver().device:edge->declaration().original().sender().device),arena_(&arena),edge_(std::move(edge)),role_(role),endpoint_(endpoint){requireDeclarations();identity_=graphCaptureVariantSignature();}
    /** @return Exact original retained free-wave holder/action/endpoint for subsequent wave authentication. */
    bool matches(const q::PooledImmutableCapturedFreeWave* edge,q::PooledFreeWireRole role,CapturedTransferEndpoint endpoint)const noexcept
    {return edge_.get()==edge && role_==role && endpoint_==endpoint;}
    /** @brief Acquire only the actual retained setup producer on this participant's exact stream. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override {
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<uintptr_t>(stream)<=2)throw std::invalid_argument("Free endpoint needs original participant/exact setup stream");
        requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Free endpoint capture drift");
        const bool own=reverseFree(role_)==(endpoint_==CapturedTransferEndpoint::Producer);
        if(own)edge_->pin()->forward()->ownerStorage()->acquireSetup(stream);else edge_->pin()->forward()->coordinatorStorage()->acquireSetup(stream);setGPUStream(stream);return true;
    }
    /** @brief Submit one public exact fixed/sliced/counted transfer with its original publication ownership. */
    bool execute(IDeviceContext* context)override {
        if(!context || context->deviceId()!=device())throw std::invalid_argument("Free endpoint execution moved participants");
        requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Free endpoint binding changed");TransferEngine::instance().enqueueCapturedTransfer(edge_->binding(role_,endpoint_),requireGPUStream());return true;
    }
    /** @return Explicit graph collective classification. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER;}
    /** @return This original bound endpoint executes native device-counted transfer kernels, independently from collective-runtime capture. */
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority()const noexcept override{return CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer;}
    /** @return Nominal phase name. */
    std::string name()const override{return "PooledImmutableDirectedFree";}
    /** @return Complete original binding, action and endpoint identity, excluding live device values. */
    uint64_t graphCaptureVariantSignature()const override{requireDeclarations();return q::pooledFreeEndpointCaptureIdentity(edge_->captureIdentity(),role_,endpoint_);}
    /** @return Persistent participant setup events are joined only before retained capture. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Live zero/shrinking prefixes retain the original physical row contract. */
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    /** @return Only this actual participant's native backend. */
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    /** @return No new packet, root or hot-path allocation. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Exact original payload and descriptor/count roots as graph read/write roles. */
    StageBufferContract bufferContract()const override {
        requireDeclarations();auto c=StageBufferContract::build();const auto& d=edge_->declaration();
        const bool producer=endpoint_==CapturedTransferEndpoint::Producer;const auto span=producer?d.source(role_):d.destination(role_);
        if(producer)c.addInput(span.buffer,"INT8");else c.addPreallocatedInOut(span.buffer,"INT8");
        // Decisions read a distinct original source descriptor or publish a
        // distinct owner count; neither Tensor authority is hidden by payload.
        if(role_==q::PooledFreeWireRole::Decisions){const auto id=producer?BufferId::QFN_POOLED_COORDINATOR_NATIVE:BufferId::QFN_POOLED_OWNER_NATIVE;
            if(producer)c.addInput(id,"INT8");else c.addPreallocatedInOut(id,"INT8");}
        return c;
    }
    /** @return Immutable diagnostics only; no count readback or synchronization. */
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo d;d.addScalarInt("free_wire_role",static_cast<unsigned>(role_));d.addScalarInt("free_endpoint",static_cast<unsigned>(endpoint_));return d;}
private:
    /** @brief Authenticate every original arena payload/count/descriptor root before capture and submission. */
    void requireDeclarations()const {
        if(!arena_ || !edge_)throw std::logic_error("Free endpoint lost original owners");edge_->requireCurrent();(void)edge_->binding(role_,endpoint_);
        const bool own=reverseFree(role_)==(endpoint_==CapturedTransferEndpoint::Producer);const auto roots=own?edge_->pin()->forward()->ownerStorage()->roots():edge_->pin()->forward()->coordinatorStorage()->roots();
        const auto native=own?BufferId::QFN_POOLED_OWNER_NATIVE:BufferId::QFN_POOLED_COORDINATOR_NATIVE;
        const auto frame=own?BufferId::QFN_POOLED_OWNER_FRAME:BufferId::QFN_POOLED_COORDINATOR_FRAME;
        const auto span=endpoint_==CapturedTransferEndpoint::Producer?edge_->declaration().source(role_):edge_->declaration().destination(role_);
        if(arena_->getTensor(native)!=roots.native.get() || (span.buffer==frame && arena_->getTensor(frame)!=roots.frame.get()))throw std::logic_error("Free endpoint arena replaced original native/frame authority");
    }
    BufferArena* arena_;std::shared_ptr<const q::PooledImmutableCapturedFreeWave> edge_;q::PooledFreeWireRole role_;CapturedTransferEndpoint endpoint_;uint64_t identity_=0;
};
/** @brief Validate all pending node names before declaring this participant's graph extension. */
void freeNames(const ComputeGraph& graph,std::span<const std::string> names){std::set<std::string> unique;for(const auto& name:names)if(name.empty() || graph.getNode(name) || !unique.insert(name).second)throw std::invalid_argument("Free wave has a missing/duplicate existing node name");}
}
void requirePooledFreeOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const q::PooledImmutableCapturedFreeWave>& original,const std::string& acquiredFree)
{
    if(isGraphCaptureActive() || !original || acquiredFree.empty())throw std::invalid_argument("Victim owner insertion needs original cold free completion");original->requireCurrent();const auto forward=original->pin()->forward();const auto storage=forward->ownerStorage();const auto device=storage->plan().allocator().device;
    const auto prior=[&](const ComputeNode* node){if(!node || node->device!=device || node->dependencies.size()!=1 || !graph.getNode(node->dependencies.front()))throw std::invalid_argument("Free owner chain has no exact original completed predecessor");return node->dependencies.front();};
    const auto phase=[&](const std::string& name,n::OwnerAction action){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()):nullptr;const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;
        if(!actual || node->device!=device || actual->storage!=storage || actual->action!=action)throw std::invalid_argument("Victim insertion replaced original free owner phase/storage");
        const q::PooledImmutableOwnerOperands expected(action,storage->plan(),original->ownerInputs(action),n::SourceCondition::Always,action==n::OwnerAction::AcceptFree?n::RecordLayout::TransportedAllocationPrefix:n::RecordLayout::Compact);
        const auto& wanted=expected.spans();const auto& given=actual->operands.spans();
        for(size_t i=0;i<wanted.size();++i)if(bool(wanted[i])!=bool(given[i]) || (wanted[i] && (wanted[i]->root!=given[i]->root || wanted[i]->buffer!=given[i]->buffer || wanted[i]->offset!=given[i]->offset || wanted[i]->bytes!=given[i]->bytes)))throw std::invalid_argument("Victim insertion substituted original free owner input root/subrange");
        if(actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Victim insertion changed original free owner launch policy");return node;};
    const auto endpoint=[&](const std::string& name,q::PooledFreeWireRole role,CapturedTransferEndpoint side){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledFreeTransferStage*>(node->stage.get()):nullptr;if(!stage || node->device!=device || !stage->matches(original.get(),role,side))throw std::invalid_argument("Victim insertion replaced original free endpoint/channel owner");return node;};
    auto current=prior(phase(acquiredFree,n::OwnerAction::AcquireFree));current=prior(endpoint(current,q::PooledFreeWireRole::Frontier,CapturedTransferEndpoint::Consumer));current=prior(endpoint(current,q::PooledFreeWireRole::Receipt,CapturedTransferEndpoint::Producer));current=prior(phase(current,n::OwnerAction::FreeReceipt));
    if(original->declaration().hasPayload())current=prior(endpoint(current,q::PooledFreeWireRole::Grants,CapturedTransferEndpoint::Producer));current=prior(phase(current,n::OwnerAction::ExecuteFree));current=prior(phase(current,n::OwnerAction::AcceptFree));
    if(original->declaration().hasPayload())current=prior(endpoint(current,q::PooledFreeWireRole::Decisions,CapturedTransferEndpoint::Consumer));current=prior(endpoint(current,q::PooledFreeWireRole::Plan,CapturedTransferEndpoint::Consumer));current=prior(endpoint(current,q::PooledFreeWireRole::Availability,CapturedTransferEndpoint::Producer));current=prior(phase(current,n::OwnerAction::Availability));
    requirePooledPinOwnerCompletion(graph,original->pin(),current);
}
void requirePooledFreeCoordinatorCompletion(const ComputeGraph& graph,
    std::span<const std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> owners,const std::string& sentFree)
{
    if(isGraphCaptureActive() || owners.empty() || !owners.front() || sentFree.empty())throw std::invalid_argument("Victim coordinator insertion needs original cold complete free cohort");
    const auto storage=owners.front()->pin()->forward()->coordinatorStorage();const auto device=storage->plan().allocator().device;
    if(owners.size()!=storage->plan().parameters().policy.participants())throw std::invalid_argument("Victim insertion has incomplete original free cohort");
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> pins;
    for(size_t i=0;i<owners.size();++i){if(!owners[i])throw std::invalid_argument("Missing original free owner");owners[i]->requireCurrent();if(owners[i]->declaration().coordinate()!=i || owners[i]->pin()->forward()->coordinatorStorage()!=storage)throw std::invalid_argument("Victim insertion changed original free owner/source order");pins.push_back(owners[i]->pin());}
    const auto prior=[&](const ComputeNode* node){if(!node || node->device!=device || node->dependencies.size()!=1 || !graph.getNode(node->dependencies.front()))throw std::invalid_argument("Free coordinator chain lacks exact completed predecessor");return node->dependencies.front();};
    const auto phase=[&](const std::string& name,n::CoordinatorAction action,size_t owner){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()):nullptr;const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase()):nullptr;if(!actual || node->device!=device || actual->storage!=storage || actual->action!=action)throw std::invalid_argument("Victim insertion replaced original coordinator free phase/storage");const q::PooledImmutableCoordinatorOperands expected(action,storage->plan(),{},owners[owner]->coordinatorInputs(action));const auto& wanted=expected.spans();const auto& given=actual->operands.spans();for(size_t r=0;r<wanted.size();++r)if(bool(wanted[r])!=bool(given[r]) || (wanted[r] && (wanted[r]->root!=given[r]->root || wanted[r]->buffer!=given[r]->buffer || wanted[r]->offset!=given[r]->offset || wanted[r]->bytes!=given[r]->bytes)))throw std::invalid_argument("Victim insertion substituted original coordinator free input root/subrange");if(actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Victim insertion changed original coordinator free policy");return node;};
    const auto endpoint=[&](const std::string& name,size_t owner,q::PooledFreeWireRole role,CapturedTransferEndpoint side){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledFreeTransferStage*>(node->stage.get()):nullptr;if(!stage || node->device!=device || !stage->matches(owners[owner].get(),role,side))throw std::invalid_argument("Victim insertion replaced original coordinator free endpoint/channel");return node;};
    auto current=sentFree;for(size_t i=owners.size();i-->0;)current=prior(endpoint(current,i,q::PooledFreeWireRole::Frontier,CapturedTransferEndpoint::Producer));const auto complete=current;current=prior(phase(complete,n::CoordinatorAction::CompleteFree,0));
    std::string planned;std::vector<std::string> sent(owners.size());
    for(size_t i=owners.size();i-->0;){const auto* header=endpoint(current,i,q::PooledFreeWireRole::Receipt,CapturedTransferEndpoint::Consumer);const auto* read=header;
        if(owners[i]->declaration().hasPayload()){const auto acquired=prior(header);const auto* acquiredNode=phase(acquired,n::CoordinatorAction::AcquireFree,i);read=endpoint(prior(acquiredNode),i,q::PooledFreeWireRole::Grants,CapturedTransferEndpoint::Consumer);}
        std::string previous;unsigned producers=0;std::vector<unsigned> counts(owners.size(),0);
        for(const auto& name:read->dependencies){const auto* node=graph.getNode(name);bool matched=false;if(node){const auto* native=dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get());const auto* operation=native?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&native->phase()):nullptr;
            if(operation && operation->storage==storage && operation->action==n::CoordinatorAction::PlanFree){(void)phase(name,n::CoordinatorAction::PlanFree,0);if(!planned.empty() && planned!=name)throw std::invalid_argument("Multiple original free planners");planned=name;++producers;matched=true;}
            const auto* transfer=dynamic_cast<const PooledFreeTransferStage*>(node->stage.get());
            for(size_t j=0;j<owners.size();++j)if(transfer && transfer->matches(owners[j].get(),owners[j]->declaration().hasPayload()?q::PooledFreeWireRole::Decisions:q::PooledFreeWireRole::Plan,CapturedTransferEndpoint::Producer)){if(!sent[j].empty() && sent[j]!=name)throw std::invalid_argument("Free plan reader changed during receipt gather");sent[j]=name;++counts[j];matched=true;}
            if(i && transfer && transfer->matches(owners[i-1].get(),q::PooledFreeWireRole::Receipt,CapturedTransferEndpoint::Consumer)){if(!previous.empty())throw std::invalid_argument("Duplicate original free receipt predecessor");previous=name;matched=true;}}
            if(!matched)throw std::invalid_argument("Free receipt has foreign command/read-frontier dependency");}
        if(producers!=1 || std::any_of(counts.begin(),counts.end(),[](unsigned count){return count!=1;}) || (i && previous.empty()))throw std::invalid_argument("Free receipts did not retire every original plan reader");current=previous;
    }
    // Actual fixed headers/suffix sends form one complete source-read chain;
    // preserving it also protects the free prefix from PlanVictims writes.
    for(size_t i=0;i<owners.size();++i){auto name=sent[i];if(owners[i]->declaration().hasPayload())name=prior(endpoint(name,i,q::PooledFreeWireRole::Decisions,CapturedTransferEndpoint::Producer));const auto before=prior(endpoint(name,i,q::PooledFreeWireRole::Plan,CapturedTransferEndpoint::Producer));if(before!=(i?sent[i-1]:planned))throw std::invalid_argument("Original free plan sender order/read frontier changed");}
    current=prior(phase(planned,n::CoordinatorAction::PlanFree,0));for(size_t i=owners.size();i-->0;)current=prior(endpoint(current,i,q::PooledFreeWireRole::Availability,CapturedTransferEndpoint::Consumer));requirePooledPinFrontierCompletion(graph,pins,current);
}
void wirePooledFreeAvailabilityLifetime(ComputeGraph& graph,const q::PooledImmutableCoordinatorNativePlan& original,
    std::span<const std::string> receives,const std::string& pinFrontiersComplete,const std::string& planFree)
{
    if(isGraphCaptureActive() || receives.size()!=original.parameters().policy.participants() || !graph.getNode(pinFrontiersComplete) || !graph.getNode(planFree))throw std::invalid_argument("Free Availability lifetime needs original complete producer/frontier/cohort");
    std::set<std::string> names{pinFrontiersComplete,planFree};if(names.size()!=2)throw std::invalid_argument("Free phase aliases its preceding source frontier");
    for(const auto& name:receives)if(!graph.getNode(name) || !names.insert(name).second)throw std::invalid_argument("Free Availability receiver cohort is incomplete or aliased");
    auto prior=pinFrontiersComplete;for(const auto& name:receives){graph.addDependency(name,prior);prior=name;}graph.addDependency(planFree,prior);
}
PooledPinParticipantNodes appendPooledFreeOwnerWave(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const q::PooledImmutableCapturedFreeWave> edge,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& acquiredPins,const std::string& prefix)
{
    if(isGraphCaptureActive() || !edge || !code || prefix.empty())throw std::invalid_argument("Free owner graph needs actual cold original owners/code");edge->requireCurrent();
    const auto forward=edge->pin()->forward();const auto storage=forward->ownerStorage();const auto device=storage->plan().allocator().device;
    if(code->backend()!=device.type)throw std::invalid_argument("Free owner phase changed its original backend code");
    requirePooledPinOwnerCompletion(graph,edge->pin(),acquiredPins);
    const std::array suffixes{"availability","availability_send","plan_receive","decisions_receive","accept_free","execute_free","grant_send","free_receipt","receipt_send","frontier_receive","acquire_free"};std::vector<std::string> names;for(auto s:suffixes)names.push_back(prefix+"/"+s);freeNames(graph,names);
    std::vector<std::unique_ptr<IComputeStage>> stages;
    const auto phase=[&](n::OwnerAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,storage,q::PooledImmutableOwnerOperands(action,storage->plan(),edge->ownerInputs(action),n::SourceCondition::Always,action==n::OwnerAction::AcceptFree?n::RecordLayout::TransportedAllocationPrefix:n::RecordLayout::Compact)});};
    const auto transfer=[&](q::PooledFreeWireRole role,CapturedTransferEndpoint endpoint){return std::make_unique<PooledFreeTransferStage>(arena,edge,role,endpoint);};
    stages.push_back(phase(n::OwnerAction::Availability));stages.push_back(transfer(q::PooledFreeWireRole::Availability,CapturedTransferEndpoint::Producer));stages.push_back(transfer(q::PooledFreeWireRole::Plan,CapturedTransferEndpoint::Consumer));stages.push_back(edge->declaration().hasPayload()?transfer(q::PooledFreeWireRole::Decisions,CapturedTransferEndpoint::Consumer):nullptr);
    stages.push_back(phase(n::OwnerAction::AcceptFree));stages.push_back(phase(n::OwnerAction::ExecuteFree));stages.push_back(edge->declaration().hasPayload()?transfer(q::PooledFreeWireRole::Grants,CapturedTransferEndpoint::Producer):nullptr);stages.push_back(phase(n::OwnerAction::FreeReceipt));stages.push_back(transfer(q::PooledFreeWireRole::Receipt,CapturedTransferEndpoint::Producer));stages.push_back(transfer(q::PooledFreeWireRole::Frontier,CapturedTransferEndpoint::Consumer));stages.push_back(phase(n::OwnerAction::AcquireFree));
    auto prior=acquiredPins;for(size_t i=0;i<stages.size();++i)if(stages[i]){graph.addNode(names[i],std::move(stages[i]),device);graph.addDependency(names[i],prior);prior=names[i];}return {names.front(),prior};
}
void requirePooledMixedFreeCompletion(const ComputeGraph& graph,const PooledImmutableMixedFreeCompletion& original)
{
    if(isGraphCaptureActive() || !original.previous || !original.previous->cohort)throw std::invalid_argument("Mixed Victim insertion needs the actual cold Free predecessor");
    requirePooledMixedPinCompletion(graph,*original.previous);const auto cohort=original.previous->cohort;const auto storage=cohort->coordinatorStorage();const auto device=storage->plan().allocator().device;
    const auto local=std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(cohort->owners()[cohort->localCoordinate()]);const auto count=cohort->owners().size();
    if(original.owners.size()!=count || original.availability.size()!=count || original.replies.size()!=count || original.frontiers.size()!=count || original.coordinator.last!=original.frontiers.back())throw std::invalid_argument("Mixed Free lost complete original cohort/frontier");
    const auto node=[&](const std::string& name){const auto* value=graph.getNode(name);if(!value || value->device!=device)throw std::invalid_argument("Mixed Free lost original coordinator-local node");return value;};
    const auto depends=[&](const std::string& consumer,const std::string& producer){const auto* c=node(consumer);node(producer);if(std::find(c->dependencies.begin(),c->dependencies.end(),producer)==c->dependencies.end())throw std::invalid_argument("Mixed Free lost actual original send/read/phase dependency");};
    const auto owner=[&](const std::string& name,n::OwnerAction action){const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(name)->stage.get());const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;const auto expected=local->ownerOperands(action);if(!actual || actual->action!=action || actual->storage!=local->ownerStorage() || actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Mixed Free replaced original local owner phase/inputs");};
    const auto coordinator=[&](const std::string& name,n::CoordinatorAction action,q::PooledImmutableCoordinatorOperands expected){const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(name)->stage.get());const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase()):nullptr;if(!actual || actual->action!=action || actual->storage!=storage || actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Mixed Free replaced original coordinator phase/inputs");};
    const auto full=[&](n::CoordinatorAction action){q::PooledNativeOperandSpans spans{};spans[static_cast<unsigned>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,storage->plan().parameters().native.ingress_frontier,count*(action==n::CoordinatorAction::PlanFree?sizeof(n::Availability):sizeof(n::p::FreeAllocationReceipt))};return q::PooledImmutableCoordinatorOperands(action,storage->plan(),{},std::move(spans));};
    owner(original.localAvailability,n::OwnerAction::Availability);owner(original.localAccept,n::OwnerAction::AcceptFree);owner(original.localExecute,n::OwnerAction::ExecuteFree);owner(original.localReceipt,n::OwnerAction::FreeReceipt);owner(original.localAcquire,n::OwnerAction::AcquireFree);
    coordinator(original.planned,n::CoordinatorAction::PlanFree,full(n::CoordinatorAction::PlanFree));coordinator(original.completed,n::CoordinatorAction::CompleteFree,full(n::CoordinatorAction::CompleteFree));
    depends(original.localAvailability,original.previous->coordinator.last);depends(original.localExecute,original.localAccept);
    requirePooledOriginalDependencies(graph,original.localAvailability,{original.previous->coordinator.last});
    requirePooledOriginalDependencies(graph,original.localExecute,{original.localAccept});
    std::string priorAvailability=original.previous->coordinator.last,priorCommand=original.planned,priorReceipt;
    for(size_t i=0;i<count;++i){const auto& r=original.replies[i];const bool direct=i==local->coordinate();const bool positive=direct?local->hasPayload():std::get<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>>(original.owners[i])->declaration().hasPayload();
        if(r.coordinate!=i || (r.path==PooledPinReplyOwnerNodes::Path::DirectLocal)!=direct || positive!=!r.replyAcquired.empty() || (direct && !r.replyReceived.empty()) || (!direct && positive!=!r.replyReceived.empty()))throw std::invalid_argument("Mixed Free changed original owner/C0/local phase order");
        const auto first=direct && positive?r.replyAcquired:positive?r.replyReceived:r.headerReceived;
        depends(first,original.planned);for(const auto& other:original.replies)depends(first,other.commandSent);if(!priorReceipt.empty())depends(first,priorReceipt);
        std::vector<std::string> reads{original.planned};for(const auto& other:original.replies)reads.push_back(other.commandSent);if(!priorReceipt.empty())reads.push_back(priorReceipt);
        if(direct)reads.push_back(positive?original.localExecute:original.localReceipt);
        requirePooledOriginalDependencies(graph,first,std::move(reads));
        depends(original.availability[i],priorAvailability);
        requirePooledOriginalDependencies(graph,original.availability[i],direct?std::vector<std::string>{priorAvailability,original.localAvailability}:std::vector<std::string>{priorAvailability});
        priorAvailability=original.availability[i];
        if(direct){
            if(std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(original.owners[i])!=local || r.commandSent!=original.localAccept || original.frontiers[i]!=original.localAcquire)throw std::invalid_argument("Mixed Free substituted original local binding or readers");
            coordinator(original.availability[i],n::CoordinatorAction::ImportLocalAvailability,local->coordinatorOperands(n::CoordinatorAction::ImportLocalAvailability));depends(original.availability[i],original.localAvailability);depends(original.localAccept,priorCommand);requirePooledOriginalDependencies(graph,original.localAccept,{priorCommand});
            if(positive){coordinator(r.replyAcquired,n::CoordinatorAction::AcquireFree,local->coordinatorOperands(n::CoordinatorAction::AcquireFree));depends(r.replyAcquired,original.localExecute);depends(original.localReceipt,r.replyAcquired);depends(r.headerReceived,r.replyAcquired);}
            else depends(original.localReceipt,original.localExecute);
            requirePooledOriginalDependencies(graph,original.localReceipt,{positive?r.replyAcquired:original.localExecute});
            coordinator(r.headerReceived,n::CoordinatorAction::ImportLocalFreeReceipt,local->coordinatorOperands(n::CoordinatorAction::ImportLocalFreeReceipt));depends(r.headerReceived,original.localReceipt);
            if(positive)requirePooledOriginalDependencies(graph,r.headerReceived,{original.localReceipt,r.replyAcquired});
        }else{
            const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>>(original.owners[i]);edge->requireCurrent();
            if(edge->pin()!=std::get<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>>(cohort->owners()[i]))throw std::invalid_argument("Mixed Free substituted original remote Pin owner");
            const auto endpoint=[&](const std::string& name,q::PooledFreeWireRole role,CapturedTransferEndpoint side){const auto* stage=dynamic_cast<const PooledFreeTransferStage*>(node(name)->stage.get());if(!stage || !stage->matches(edge.get(),role,side))throw std::invalid_argument("Mixed Free replaced an actual original channel endpoint");};
            endpoint(original.availability[i],q::PooledFreeWireRole::Availability,CapturedTransferEndpoint::Consumer);
            auto sent=r.commandSent;
            if(positive){endpoint(sent,q::PooledFreeWireRole::Decisions,CapturedTransferEndpoint::Producer);const auto* value=node(sent);if(value->dependencies.size()!=1)throw std::invalid_argument("Mixed Free replaced original decision-send predecessor");sent=value->dependencies.front();depends(r.commandSent,sent);endpoint(r.replyReceived,q::PooledFreeWireRole::Grants,CapturedTransferEndpoint::Consumer);coordinator(r.replyAcquired,n::CoordinatorAction::AcquireFree,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::AcquireFree,storage->plan(),{},edge->coordinatorInputs(n::CoordinatorAction::AcquireFree)));depends(r.replyAcquired,r.replyReceived);depends(r.headerReceived,r.replyAcquired);requirePooledOriginalDependencies(graph,r.replyAcquired,{r.replyReceived});requirePooledOriginalDependencies(graph,r.headerReceived,{r.replyAcquired});}
            endpoint(sent,q::PooledFreeWireRole::Plan,CapturedTransferEndpoint::Producer);depends(sent,priorCommand);requirePooledOriginalDependencies(graph,sent,{priorCommand});endpoint(r.headerReceived,q::PooledFreeWireRole::Receipt,CapturedTransferEndpoint::Consumer);endpoint(original.frontiers[i],q::PooledFreeWireRole::Frontier,CapturedTransferEndpoint::Producer);
        }
        priorCommand=r.commandSent;priorReceipt=r.headerReceived;
    }
    depends(original.planned,priorAvailability);depends(original.completed,priorReceipt);
    requirePooledOriginalDependencies(graph,original.planned,{priorAvailability});requirePooledOriginalDependencies(graph,original.completed,{priorReceipt});
    auto prior=original.completed;for(const auto& frontier:original.frontiers){depends(frontier,prior);requirePooledOriginalDependencies(graph,frontier,{prior});prior=frontier;}
}
PooledImmutableMixedFreeCompletion appendPooledFreeCoordinatorMembers(ComputeGraph& graph,BufferArena& arena,
    std::span<const q::PooledImmutableFreeParticipant> owners,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,
    const std::string& pinFrontiersComplete,const std::string& prefix,const PooledImmutableMixedPinCompletion* localPin)
{
    if(isGraphCaptureActive() || owners.empty() || !code || prefix.empty())throw std::invalid_argument("Free coordinator graph needs original complete cold owners/code");
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> storage;std::shared_ptr<const q::PooledImmutableLocalOwnerBinding> local;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> remote(owners.size());
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> pins;
    for(size_t i=0;i<owners.size();++i)std::visit([&](const auto& binding){
        if(!binding)throw std::invalid_argument("Free coordinator has no original member");binding->requireCurrent();
        using Binding=typename std::decay_t<decltype(binding)>::element_type;
        std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> c;std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> o;
        if constexpr(std::is_same_v<Binding,const q::PooledImmutableLocalOwnerBinding>){if(local)throw std::invalid_argument("Free coordinator duplicated local owner");local=binding;c=binding->coordinatorStorage();o=binding->ownerStorage();}
        else{remote[i]=binding;pins.push_back(binding->pin());c=binding->pin()->forward()->coordinatorStorage();o=binding->pin()->forward()->ownerStorage();}
        if(!storage)storage=c;if(c!=storage || o->plan().parameters().owner.coordinate()!=i)throw std::invalid_argument("Free coordinator replaced original source/member order");storage->plan().requireOwnerPlan(o->plan());
    },owners[i]);
    const auto& plan=storage->plan();const auto device=plan.allocator().device;
    if(owners.size()!=plan.parameters().policy.participants() || code->backend()!=device.type || bool(local)!=bool(localPin))throw std::invalid_argument("Free coordinator changed original cohort/backend/local predecessor");
    if(local){
        requirePooledMixedPinCompletion(graph,*localPin);
        if(localPin->coordinator.last!=pinFrontiersComplete || localPin->cohort->coordinatorStorage()!=storage || localPin->cohort->localCoordinate()!=local->coordinate() ||
            std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(localPin->cohort->owners()[local->coordinate()])!=local)
            throw std::invalid_argument("Free coordinator substituted original local Pin completion");
        for(size_t i=0;i<owners.size();++i)if(remote[i] && std::get<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>>(localPin->cohort->owners()[i])!=remote[i]->pin())throw std::invalid_argument("Free coordinator substituted original remote Pin channel");
    }else requirePooledPinFrontierCompletion(graph,pins,pinFrontiersComplete);
    PooledImmutableMixedFreeCompletion result{};if(localPin)result.previous=*localPin;result.owners.assign(owners.begin(),owners.end());
    result.planned=prefix+"/plan_free";result.completed=prefix+"/complete_free";
    std::vector<std::string> allNames{result.planned,result.completed};
    if(local){result.localAvailability=prefix+"/local_availability";result.localAccept=prefix+"/local_accept_free";result.localExecute=prefix+"/local_execute_free";result.localReceipt=prefix+"/local_free_receipt";result.localAcquire=prefix+"/local_acquire_free";
        for(const auto& name:{result.localAvailability,result.localAccept,result.localExecute,result.localReceipt,result.localAcquire})allNames.push_back(name);}
    for(size_t i=0;i<owners.size();++i){const bool direct=local && i==local->coordinate();const bool positive=direct?local->hasPayload():remote[i]->declaration().hasPayload();const auto b=prefix+"/owner"+std::to_string(i);
        result.availability.push_back(b+(direct?"/availability_local_import":"/availability_receive"));
        result.replies.push_back({static_cast<uint32_t>(i),direct?result.localAccept:positive?b+"/decisions_send":b+"/plan_send",positive && !direct?b+"/grant_receive":"",positive?b+"/grant_acquire":"",b+(direct?"/receipt_local_import":"/receipt_receive"),direct?PooledPinReplyOwnerNodes::Path::DirectLocal:PooledPinReplyOwnerNodes::Path::Remote});
        result.frontiers.push_back(direct?result.localAcquire:b+"/frontier_send");
        for(const auto& name:{result.availability.back(),result.replies.back().headerReceived})allNames.push_back(name);
        if(!direct){allNames.push_back(b+"/plan_send");if(positive){allNames.push_back(b+"/decisions_send");allNames.push_back(b+"/grant_receive");}allNames.push_back(result.frontiers.back());}
        if(positive)allNames.push_back(result.replies.back().replyAcquired);
    }
    freeNames(graph,allNames);
    struct Node{std::string name;std::unique_ptr<IComputeStage> stage;};std::vector<Node> nodes;
    const auto full=[&](n::CoordinatorAction action){q::PooledNativeOperandSpans spans{};spans[static_cast<unsigned>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,plan.parameters().native.ingress_frontier,owners.size()*(action==n::CoordinatorAction::PlanFree?sizeof(n::Availability):sizeof(n::p::FreeAllocationReceipt))};return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{action,storage,q::PooledImmutableCoordinatorOperands(action,plan,{},std::move(spans))});};
    const auto localOwner=[&](n::OwnerAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,local->ownerStorage(),local->ownerOperands(action)});};
    const auto localCoordinator=[&](n::CoordinatorAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{action,storage,local->coordinatorOperands(action)});};
    const auto transfer=[&](size_t i,q::PooledFreeWireRole role,CapturedTransferEndpoint endpoint){return std::make_unique<PooledFreeTransferStage>(arena,remote[i],role,endpoint);};
    nodes.push_back({result.planned,full(n::CoordinatorAction::PlanFree)});nodes.push_back({result.completed,full(n::CoordinatorAction::CompleteFree)});
    if(local){nodes.push_back({result.localAvailability,localOwner(n::OwnerAction::Availability)});nodes.push_back({result.localAccept,localOwner(n::OwnerAction::AcceptFree)});nodes.push_back({result.localExecute,localOwner(n::OwnerAction::ExecuteFree)});nodes.push_back({result.localReceipt,localOwner(n::OwnerAction::FreeReceipt)});nodes.push_back({result.localAcquire,localOwner(n::OwnerAction::AcquireFree)});}
    for(size_t i=0;i<owners.size();++i){const auto& r=result.replies[i];const bool direct=r.path==PooledPinReplyOwnerNodes::Path::DirectLocal;const bool positive=!r.replyAcquired.empty();const auto b=prefix+"/owner"+std::to_string(i);
        if(direct){nodes.push_back({result.availability[i],localCoordinator(n::CoordinatorAction::ImportLocalAvailability)});if(positive)nodes.push_back({r.replyAcquired,localCoordinator(n::CoordinatorAction::AcquireFree)});nodes.push_back({r.headerReceived,localCoordinator(n::CoordinatorAction::ImportLocalFreeReceipt)});}
        else{nodes.push_back({result.availability[i],transfer(i,q::PooledFreeWireRole::Availability,CapturedTransferEndpoint::Consumer)});nodes.push_back({b+"/plan_send",transfer(i,q::PooledFreeWireRole::Plan,CapturedTransferEndpoint::Producer)});
            if(positive){nodes.push_back({r.commandSent,transfer(i,q::PooledFreeWireRole::Decisions,CapturedTransferEndpoint::Producer)});nodes.push_back({r.replyReceived,transfer(i,q::PooledFreeWireRole::Grants,CapturedTransferEndpoint::Consumer)});nodes.push_back({r.replyAcquired,std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::AcquireFree,storage,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::AcquireFree,plan,{},remote[i]->coordinatorInputs(n::CoordinatorAction::AcquireFree))})});}
            nodes.push_back({r.headerReceived,transfer(i,q::PooledFreeWireRole::Receipt,CapturedTransferEndpoint::Consumer)});nodes.push_back({result.frontiers[i],transfer(i,q::PooledFreeWireRole::Frontier,CapturedTransferEndpoint::Producer)});}
    }
    for(auto& node:nodes)graph.addNode(node.name,std::move(node.stage),device);
    wirePooledFreeAvailabilityLifetime(graph,plan,result.availability,pinFrontiersComplete,result.planned);
    if(local){graph.addDependency(result.localAvailability,pinFrontiersComplete);graph.addDependency(result.availability[local->coordinate()],result.localAvailability);}
    auto prior=result.planned;
    for(size_t i=0;i<owners.size();++i){const auto& r=result.replies[i];const bool direct=r.path==PooledPinReplyOwnerNodes::Path::DirectLocal;
        if(direct){graph.addDependency(result.localAccept,prior);graph.addDependency(result.localExecute,result.localAccept);graph.addDependency(result.localReceipt,!r.replyAcquired.empty()?r.replyAcquired:result.localExecute);graph.addDependency(r.headerReceived,result.localReceipt);if(!r.replyAcquired.empty())graph.addDependency(r.replyAcquired,result.localExecute);}
        else{const auto sent=prefix+"/owner"+std::to_string(i)+"/plan_send";graph.addDependency(sent,prior);if(!r.replyReceived.empty())graph.addDependency(r.commandSent,sent);}prior=r.commandSent;
    }
    wirePooledPinReplyBankLifetime(graph,plan,result.replies,result.planned);graph.addDependency(result.completed,result.replies.back().headerReceived);
    prior=result.completed;for(const auto& sent:result.frontiers){graph.addDependency(sent,prior);prior=sent;}
    result.coordinator={local?result.localAvailability:result.availability.front(),prior};return result;
}
PooledPinParticipantNodes appendPooledFreeCoordinatorWave(ComputeGraph& graph,BufferArena& arena,
    std::span<const std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> owners,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& pinFrontiersComplete,const std::string& prefix)
{
    std::vector<q::PooledImmutableFreeParticipant> members;for(const auto& owner:owners)members.emplace_back(owner);
    return appendPooledFreeCoordinatorMembers(graph,arena,members,std::move(code),pinFrontiersComplete,prefix).coordinator;
}
}

namespace llaminar2 {
/** @brief Inspect only the exact original endpoint stage types without another object/vtable allocation. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledFreeTransfer(const IComputeStage& stage){
    if(const auto* original=dynamic_cast<const PooledFreeTransferStage*>(&stage))return original->directedTransferOperation();
    return std::nullopt;
}
}
