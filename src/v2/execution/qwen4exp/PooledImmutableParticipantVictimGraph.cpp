/**
 * @file PooledImmutableParticipantVictimGraph.cpp
 * @brief Original victim suffix/count/header bindings and explicit participant-local graph nodes.
 *
 * The first allocation wave remains immutable. Only its actual acquired free
 * frontier authorizes victim planning. Current-wave replies reuse original
 * banks after their prior readers/senders retire; no owner receipt fanout has
 * an actual consumer and none is sent.
 */
#include "PooledImmutableParticipantVictimGraph.h"
#include "PooledImmutableDirectedOperation.h"
#include "PooledImmutableParticipantLocalGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>
#include <set>
#include <type_traits>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;namespace p=n::p;
/** @return Checked nominal wire index, never inferred from a byte capacity. */
size_t index(PooledVictimWireRole role){const auto i=static_cast<unsigned>(role);if(i>=4)throw std::invalid_argument("Unknown original victim wire action");return i;}
/** @return True only for the current-wave Grant/receipt reverse direction. */
bool reverse(PooledVictimWireRole role){(void)index(role);return role==PooledVictimWireRole::Grants || role==PooledVictimWireRole::Receipt;}
/** @return Same original canonical frame/native root, without a view or packet allocation. */
std::shared_ptr<TensorBase> root(const PooledImmutableNativeRoots& roots,BufferId id){return id==BufferId::QFN_POOLED_COORDINATOR_FRAME || id==BufferId::QFN_POOLED_OWNER_FRAME?roots.frame:roots.native;}
/** @brief Retain an exact original phase input; C0 has no dummy operand. */
void operand(PooledNativeOperandSpans& out,PooledNativeOperand role,const PooledImmutableNativeRoots& roots,PooledPinWireSpan span){
    if(!span.bytes)return;const auto r=root(roots,span.buffer);if(!r || r->is_view() || r->transferStorageOwner()!=r.get() || span.offset>r->size_bytes() || span.bytes>r->size_bytes()-span.offset)throw std::invalid_argument("Victim wave lost its original canonical root/span");out[static_cast<size_t>(role)]=PooledNativeTensorSpan{r,span.buffer,span.offset,span.bytes};
}
}
PooledPinWireSpan PooledImmutableVictimEdge::source(PooledVictimWireRole role)const {
    (void)index(role);const auto& c=previous_.original().coordinator().parameters();
    switch(role){
    case PooledVictimWireRole::Plan:return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.victim_headers+size_t{coordinate()}*sizeof(n::VictimPlanHeader),sizeof(n::VictimPlanHeader)};
    case PooledVictimWireRole::Decisions:return previous_.source(PooledFreeWireRole::Decisions);
    case PooledVictimWireRole::Grants:return previous_.source(PooledFreeWireRole::Grants);
    case PooledVictimWireRole::Receipt:return {BufferId::QFN_POOLED_OWNER_NATIVE,previous_.original().owner().parameters().native.header,sizeof(p::VictimAllocationReceipt)};
    }
    throw std::invalid_argument("Unknown original victim source");
}
PooledPinWireSpan PooledImmutableVictimEdge::destination(PooledVictimWireRole role)const {
    (void)index(role);switch(role){
    case PooledVictimWireRole::Plan:return previous_.destination(PooledFreeWireRole::Plan);
    case PooledVictimWireRole::Decisions:return previous_.destination(PooledFreeWireRole::Decisions);
    case PooledVictimWireRole::Grants:return previous_.destination(PooledFreeWireRole::Grants);
    case PooledVictimWireRole::Receipt:return previous_.destination(PooledFreeWireRole::Receipt);
    }
    throw std::invalid_argument("Unknown original victim destination");
}
PooledPinWireSpan PooledImmutableVictimEdge::decisionSlice()const noexcept{const auto& c=previous_.original().coordinator().parameters();return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.victim_slices+size_t{coordinate()}*sizeof(n::DirectedSlice),sizeof(n::DirectedSlice)};}
uint64_t PooledImmutableVictimEdge::messageKey(PooledVictimWireRole role)const{(void)index(role);return ((previous_.original().identity()^(32+static_cast<unsigned>(role)))*1099511628211ULL)|1;}
std::shared_ptr<const PooledImmutableCapturedVictimWave> PooledImmutableCapturedVictimWave::bind(
    std::shared_ptr<const PooledImmutableCapturedFreeWave> previous,std::shared_ptr<PhysicalMemoryAuthority> memory)
{
    if(isGraphCaptureActive() || !previous || !memory)throw std::invalid_argument("Victim binding requires actual original cold free owners/PMA");previous->requireCurrent();const auto pin=previous->pin();const auto forward=pin->forward();forward->coordinatorStorage()->requireAuthority(memory);forward->ownerStorage()->requireAuthority(memory);auto d=PooledImmutableVictimEdge::fromFree(previous->declaration());
    if((!forward->channel_.nativeCollective() && !d.previous().original().capturedChannelTransport()) || !forward->channel_ || !pin->channel_ || memory->worldRank()!=d.previous().original().sender().world_rank)throw std::invalid_argument("Victim edge requires its original admitted transport owners");
    const auto coord=forward->coordinatorStorage()->roots(),own=forward->ownerStorage()->roots();Bindings bindings{};
    for(unsigned i=0;i<4;++i){const auto role=static_cast<PooledVictimWireRole>(i);if(!d.hasPayload() && (role==PooledVictimWireRole::Decisions || role==PooledVictimWireRole::Grants))continue;const bool back=reverse(role);const auto& channel=back?pin->channel_:forward->channel_;const auto source=d.source(role),destination=d.destination(role);const auto maximum=role==PooledVictimWireRole::Decisions?d.maximumDecisionBytes():source.bytes;
        auto send=(channel).bind(CapturedTransferEndpoint::Producer,{d.messageKey(role),maximum},root(back?own:coord,source.buffer),source.offset);auto receive=(channel).bind(CapturedTransferEndpoint::Consumer,{d.messageKey(role),maximum},root(back?coord:own,destination.buffer),destination.offset);
        if(role==PooledVictimWireRole::Decisions){const auto slice=d.decisionSlice(),count=*d.decisionCount();send=std::move(send).withDeviceSliced(coord.native,slice.offset,source.bytes,source.offset,PooledNativeCountedMessage::VictimDecisions);receive=std::move(receive).withDeviceCounted(own.native,count.offset,PooledNativeCountedMessage::VictimDecisions);}
        else if(role==PooledVictimWireRole::Grants){const auto count=d.grantCount(),incoming=*d.incomingGrantCount();send=std::move(send).withDeviceCounted(own.native,count.offset,PooledNativeCountedMessage::VictimGrants);receive=std::move(receive).withDeviceCounted(coord.native,incoming.offset,PooledNativeCountedMessage::VictimGrants);}
        bindings[i].emplace(std::move(send),std::move(receive));
    }
    return std::shared_ptr<const PooledImmutableCapturedVictimWave>(new PooledImmutableCapturedVictimWave(std::move(previous),std::move(memory),std::move(d),std::move(bindings)));
}
void PooledImmutableCapturedVictimWave::requireCurrent()const{previous_->requireCurrent();previous_->pin()->forward()->coordinatorStorage()->requireAuthority(memory_);previous_->pin()->forward()->ownerStorage()->requireAuthority(memory_);}
const PooledImmutableTransferBinding& PooledImmutableCapturedVictimWave::binding(PooledVictimWireRole role,CapturedTransferEndpoint endpoint)const{requireCurrent();const auto i=index(role);if(endpoint!=CapturedTransferEndpoint::Producer && endpoint!=CapturedTransferEndpoint::Consumer)throw std::invalid_argument("Unknown victim endpoint");if(!bindings_[i])throw std::invalid_argument("C0 victim has no decision/Grant payload transaction");return endpoint==CapturedTransferEndpoint::Producer?bindings_[i]->first:bindings_[i]->second;}
PooledNativeOperandSpans PooledImmutableCapturedVictimWave::ownerInputs(n::OwnerAction action)const {
    requireCurrent();const auto forward=previous_->pin()->forward();auto out=forward->ownerInputs(n::OwnerAction::Begin);const auto roots=forward->ownerStorage()->roots();
    if(action==n::OwnerAction::AcceptVictim){operand(out,PooledNativeOperand::Frontier,roots,declaration_.destination(PooledVictimWireRole::Plan));if(declaration_.hasPayload()){operand(out,PooledNativeOperand::Records,roots,declaration_.destination(PooledVictimWireRole::Decisions));operand(out,PooledNativeOperand::RecordBytes,roots,*declaration_.decisionCount());}}
    else if(action!=n::OwnerAction::ExecuteVictim && action!=n::OwnerAction::VictimReceipt)throw std::invalid_argument("Victim binding cannot supply another owner phase");return out;
}
PooledNativeOperandSpans PooledImmutableCapturedVictimWave::coordinatorInputs(n::CoordinatorAction action)const {
    requireCurrent();PooledNativeOperandSpans out{};const auto original=previous_->pin()->forward()->coordinatorStorage();const auto& c=original->plan().parameters();const auto roots=original->roots();
    if(action==n::CoordinatorAction::AcquireVictims && declaration_.hasPayload()){operand(out,PooledNativeOperand::Records,roots,declaration_.destination(PooledVictimWireRole::Grants));operand(out,PooledNativeOperand::RecordBytes,roots,*declaration_.incomingGrantCount());}
    else if(action==n::CoordinatorAction::CompleteVictims)operand(out,PooledNativeOperand::Frontier,roots,{BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.ingress_frontier,size_t{c.policy.participants()}*sizeof(p::VictimAllocationReceipt)});
    else if(action!=n::CoordinatorAction::PlanVictims)throw std::invalid_argument("Victim binding cannot supply another coordinator phase");return out;
}
uint64_t PooledImmutableCapturedVictimWave::captureIdentity()const{requireCurrent();auto value=previous_->captureIdentity();for(size_t i=0;i<bindings_.size();++i){value=((value^i)*1099511628211ULL)|1;if(bindings_[i]){value=((value^bindings_[i]->first.captureIdentity())*1099511628211ULL)|1;value=((value^bindings_[i]->second.captureIdentity())*1099511628211ULL)|1;}}return value;}
}
namespace llaminar2 {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;
/** @return Actual owner→coordinator direction; never guessed from a source byte count. */
bool reverseVictim(q::PooledVictimWireRole role){return role==q::PooledVictimWireRole::Grants || role==q::PooledVictimWireRole::Receipt;}
/** @brief One exact participant-local endpoint over retained original victim operands. */
class PooledVictimTransferStage final:public IComputeStage {
public:
    /** @return Exact original semantic endpoint and native dispatch inputs before capture. */
    q::PooledImmutableDirectedTransferOperation directedTransferOperation()const{
        requireDeclarations();const auto& binding=edge_->binding(role_,endpoint_);
        return {q::PooledImmutableDirectedWireFamily::Victim,static_cast<std::uint32_t>(role_),endpoint_,
            binding.message().bytes,binding.extentSource(),binding.hasSourceSlice(),binding.nativeCollective()?q::PooledImmutableDirectedTransportKind::NativeCollective:q::PooledImmutableDirectedTransportKind::CapturedChannel};
    }

    /** @brief Retain actual original free/victim owners before freezing endpoint identity. */
    PooledVictimTransferStage(BufferArena& arena,std::shared_ptr<const q::PooledImmutableCapturedVictimWave> edge,q::PooledVictimWireRole role,CapturedTransferEndpoint endpoint)
        :IComputeStage((reverseVictim(role)==(endpoint==CapturedTransferEndpoint::Producer))?edge->declaration().previous().original().receiver().device:edge->declaration().previous().original().sender().device),arena_(&arena),edge_(std::move(edge)),role_(role),endpoint_(endpoint){requireDeclarations();identity_=graphCaptureVariantSignature();native_collective_=edge_->binding(role_,endpoint_).nativeCollective();}
    /** @brief Import only the actual original setup producer on this exact retained capture stream. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override{
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<uintptr_t>(stream)<=2)throw std::invalid_argument("Victim endpoint needs original participant/exact stream");requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Victim capture identity changed");const auto forward=edge_->previous()->pin()->forward();if(reverseVictim(role_)==(endpoint_==CapturedTransferEndpoint::Producer))forward->ownerStorage()->acquireSetup(stream);else forward->coordinatorStorage()->acquireSetup(stream);edge_->binding(role_,endpoint_).prepare(stream);setGPUStream(stream);return true;}
    /** @brief Submit one public event-aware exact fixed/sliced/counted transfer without a host count read. */
    bool execute(IDeviceContext* context)override{if(!context || context->deviceId()!=device())throw std::invalid_argument("Victim endpoint changed execution participant");requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Victim binding identity drift");edge_->binding(role_,endpoint_).enqueue(requireGPUStream());return true;}
    /** @return Exact original holder/action/endpoint identity for cold successor-chain authentication. */
    bool matches(const q::PooledImmutableCapturedVictimWave* original,q::PooledVictimWireRole role,CapturedTransferEndpoint endpoint)const noexcept{return edge_.get()==original && role_==role && endpoint_==endpoint;}
    /** @return Explicit collective classification, never a hidden multi-device compute stage. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER;}
    /** @return This original bound endpoint executes native device-counted transfer kernels, independently from collective-runtime capture. */
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority()const noexcept override{return native_collective_?CollectiveGraphCaptureAuthority::RuntimeBackend:CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer;}
    /** @return Nominal second-wave name. */
    std::string name()const override{return "PooledImmutableDirectedVictim";}
    /** @return Complete immutable original binding/action/endpoint identity, excluding live device values. */
    uint64_t graphCaptureVariantSignature()const override{requireDeclarations();return q::pooledVictimEndpointCaptureIdentity(edge_->captureIdentity(),role_,endpoint_);}
    /** @return Setup imports belong before capture, never replay. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Original empty/shrinking prefix semantics retain the physical row contract. */
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    /** @return Only the actual endpoint backend. */
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    /** @return No new packet/root/hot allocation. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Actual original payload plus distinct descriptor/count roles visible to graph publication. */
    StageBufferContract bufferContract()const override{requireDeclarations();auto c=StageBufferContract::build();const bool producer=endpoint_==CapturedTransferEndpoint::Producer;const auto span=producer?edge_->declaration().source(role_):edge_->declaration().destination(role_);if(producer)c.addInput(span.buffer,"INT8");else c.addPreallocatedInOut(span.buffer,"INT8");if(role_==q::PooledVictimWireRole::Decisions){const auto id=producer?BufferId::QFN_POOLED_COORDINATOR_NATIVE:BufferId::QFN_POOLED_OWNER_NATIVE;if(producer)c.addInput(id,"INT8");else c.addPreallocatedInOut(id,"INT8");}return c;}
    /** @return Metadata-only diagnostics. */
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo d;d.addScalarInt("victim_role",static_cast<unsigned>(role_));d.addScalarInt("victim_endpoint",static_cast<unsigned>(endpoint_));return d;}
private:
    /** @brief Validate actual original arena payload/descriptor/count owners before capture or submission. */
    void requireDeclarations()const{if(!arena_ || !edge_)throw std::logic_error("Victim endpoint lost original owners");edge_->requireCurrent();(void)edge_->binding(role_,endpoint_);const auto forward=edge_->previous()->pin()->forward();const bool own=reverseVictim(role_)==(endpoint_==CapturedTransferEndpoint::Producer);const auto roots=own?forward->ownerStorage()->roots():forward->coordinatorStorage()->roots();const auto native=own?BufferId::QFN_POOLED_OWNER_NATIVE:BufferId::QFN_POOLED_COORDINATOR_NATIVE,frame=own?BufferId::QFN_POOLED_OWNER_FRAME:BufferId::QFN_POOLED_COORDINATOR_FRAME;const auto span=endpoint_==CapturedTransferEndpoint::Producer?edge_->declaration().source(role_):edge_->declaration().destination(role_);if(arena_->getTensor(native)!=roots.native.get() || (span.buffer==frame && arena_->getTensor(frame)!=roots.frame.get()))throw std::logic_error("Victim arena substituted original native/frame roots");}
    BufferArena* arena_;std::shared_ptr<const q::PooledImmutableCapturedVictimWave> edge_;q::PooledVictimWireRole role_;CapturedTransferEndpoint endpoint_;
    bool native_collective_=false; // Cold-selected original transport; capture policy never probes runtime state.
    uint64_t identity_=0;
};
/** @brief Reject every missing/duplicate existing node before graph mutation. */
void victimNames(const ComputeGraph& graph,std::span<const std::string> names){std::set<std::string> unique;for(const auto& name:names)if(name.empty() || graph.getNode(name) || !unique.insert(name).second)throw std::invalid_argument("Victim graph would replace an existing node");}
}
/** @brief Authenticate actual original owner victim execution and receipt-send retirement before source insertion. */
void requirePooledVictimOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const q::PooledImmutableCapturedVictimWave>& original,const std::string& sentVictims)
{
    if(isGraphCaptureActive() || !original || sentVictims.empty())throw std::invalid_argument("Source owner insertion needs original cold victim completion");original->requireCurrent();const auto forward=original->previous()->pin()->forward();const auto storage=forward->ownerStorage();const auto device=storage->plan().allocator().device;
    const auto prior=[&](const ComputeNode* node){if(!node || node->device!=device || node->dependencies.size()!=1 || !graph.getNode(node->dependencies.front()))throw std::invalid_argument("Victim owner chain has no exact original completed predecessor");return node->dependencies.front();};
    const auto phase=[&](const std::string& name,n::OwnerAction action){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()):nullptr;const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;
        if(!actual || node->device!=device || actual->storage!=storage || actual->action!=action)throw std::invalid_argument("Source insertion replaced original victim owner phase/storage");
        const q::PooledImmutableOwnerOperands expected(action,storage->plan(),original->ownerInputs(action),n::SourceCondition::Always,action==n::OwnerAction::AcceptVictim?n::RecordLayout::TransportedAllocationPrefix:n::RecordLayout::Compact);
        const auto& wanted=expected.spans();const auto& given=actual->operands.spans();
        for(size_t i=0;i<wanted.size();++i)if(bool(wanted[i])!=bool(given[i]) || (wanted[i] && (wanted[i]->root!=given[i]->root || wanted[i]->buffer!=given[i]->buffer || wanted[i]->offset!=given[i]->offset || wanted[i]->bytes!=given[i]->bytes)))throw std::invalid_argument("Source insertion substituted original victim owner input root/subrange");
        if(actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Source insertion changed original victim owner launch policy");return node;};
    const auto endpoint=[&](const std::string& name,q::PooledVictimWireRole role,CapturedTransferEndpoint side){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledVictimTransferStage*>(node->stage.get()):nullptr;if(!stage || node->device!=device || !stage->matches(original.get(),role,side))throw std::invalid_argument("Source insertion replaced original victim endpoint/channel owner");return node;};
    auto current=prior(endpoint(sentVictims,q::PooledVictimWireRole::Receipt,CapturedTransferEndpoint::Producer));
    current=prior(phase(current,n::OwnerAction::VictimReceipt));
    if(original->declaration().hasPayload())current=prior(endpoint(current,q::PooledVictimWireRole::Grants,CapturedTransferEndpoint::Producer));
    current=prior(phase(current,n::OwnerAction::ExecuteVictim));current=prior(phase(current,n::OwnerAction::AcceptVictim));
    if(original->declaration().hasPayload())current=prior(endpoint(current,q::PooledVictimWireRole::Decisions,CapturedTransferEndpoint::Consumer));
    current=prior(endpoint(current,q::PooledVictimWireRole::Plan,CapturedTransferEndpoint::Consumer));
    requirePooledFreeOwnerCompletion(graph,original->previous(),current);
}
/** @brief Authenticate complete actual victim source/read frontier and all original prior free-chain ownership. */
void requirePooledVictimCoordinatorCompletion(const ComputeGraph& graph,
    std::span<const std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> owners,const std::string& completeVictims)
{
    if(isGraphCaptureActive() || owners.empty() || !owners.front() || completeVictims.empty())throw std::invalid_argument("Source coordinator insertion needs original cold complete victim cohort");
    const auto storage=owners.front()->previous()->pin()->forward()->coordinatorStorage();const auto device=storage->plan().allocator().device;
    if(owners.size()!=storage->plan().parameters().policy.participants())throw std::invalid_argument("Source insertion has incomplete original victim cohort");
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> previous;
    for(size_t i=0;i<owners.size();++i){if(!owners[i])throw std::invalid_argument("Missing original victim owner");owners[i]->requireCurrent();if(owners[i]->declaration().coordinate()!=i || owners[i]->previous()->pin()->forward()->coordinatorStorage()!=storage)throw std::invalid_argument("Source insertion changed original victim owner/source order");previous.push_back(owners[i]->previous());}
    const auto prior=[&](const ComputeNode* node){if(!node || node->device!=device || node->dependencies.size()!=1 || !graph.getNode(node->dependencies.front()))throw std::invalid_argument("Victim coordinator chain lacks exact completed predecessor");return node->dependencies.front();};
    const auto phase=[&](const std::string& name,n::CoordinatorAction action,size_t owner){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()):nullptr;const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase()):nullptr;if(!actual || node->device!=device || actual->storage!=storage || actual->action!=action)throw std::invalid_argument("Source insertion replaced original coordinator victim phase/storage");const q::PooledImmutableCoordinatorOperands expected(action,storage->plan(),{},owners[owner]->coordinatorInputs(action));const auto& wanted=expected.spans();const auto& given=actual->operands.spans();for(size_t r=0;r<wanted.size();++r)if(bool(wanted[r])!=bool(given[r]) || (wanted[r] && (wanted[r]->root!=given[r]->root || wanted[r]->buffer!=given[r]->buffer || wanted[r]->offset!=given[r]->offset || wanted[r]->bytes!=given[r]->bytes)))throw std::invalid_argument("Source insertion substituted original coordinator victim input root/subrange");if(actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Source insertion changed original coordinator victim policy");return node;};
    const auto endpoint=[&](const std::string& name,size_t owner,q::PooledVictimWireRole role,CapturedTransferEndpoint side){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledVictimTransferStage*>(node->stage.get()):nullptr;if(!stage || node->device!=device || !stage->matches(owners[owner].get(),role,side))throw std::invalid_argument("Source insertion replaced original coordinator victim endpoint/channel");return node;};
    const auto complete=completeVictims;auto current=prior(phase(complete,n::CoordinatorAction::CompleteVictims,0));
    std::string planned;std::vector<std::string> sent(owners.size());
    for(size_t i=owners.size();i-->0;){const auto* header=endpoint(current,i,q::PooledVictimWireRole::Receipt,CapturedTransferEndpoint::Consumer);const auto* read=header;
        if(owners[i]->declaration().hasPayload()){const auto acquired=prior(header);const auto* acquiredNode=phase(acquired,n::CoordinatorAction::AcquireVictims,i);read=endpoint(prior(acquiredNode),i,q::PooledVictimWireRole::Grants,CapturedTransferEndpoint::Consumer);}
        std::string previous;unsigned producers=0;std::vector<unsigned> counts(owners.size(),0);
        for(const auto& name:read->dependencies){const auto* node=graph.getNode(name);bool matched=false;if(node){const auto* native=dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get());const auto* operation=native?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&native->phase()):nullptr;
            if(operation && operation->storage==storage && operation->action==n::CoordinatorAction::PlanVictims){(void)phase(name,n::CoordinatorAction::PlanVictims,0);if(!planned.empty() && planned!=name)throw std::invalid_argument("Multiple original victim planners");planned=name;++producers;matched=true;}
            const auto* transfer=dynamic_cast<const PooledVictimTransferStage*>(node->stage.get());
            for(size_t j=0;j<owners.size();++j)if(transfer && transfer->matches(owners[j].get(),owners[j]->declaration().hasPayload()?q::PooledVictimWireRole::Decisions:q::PooledVictimWireRole::Plan,CapturedTransferEndpoint::Producer)){if(!sent[j].empty() && sent[j]!=name)throw std::invalid_argument("Victim plan reader changed during receipt gather");sent[j]=name;++counts[j];matched=true;}
            if(i && transfer && transfer->matches(owners[i-1].get(),q::PooledVictimWireRole::Receipt,CapturedTransferEndpoint::Consumer)){if(!previous.empty())throw std::invalid_argument("Duplicate original victim receipt predecessor");previous=name;matched=true;}}
            if(!matched)throw std::invalid_argument("Victim receipt has foreign command/read-frontier dependency");}
        if(producers!=1 || std::any_of(counts.begin(),counts.end(),[](unsigned count){return count!=1;}) || (i && previous.empty()))throw std::invalid_argument("Victim receipts did not retire every original plan reader");current=previous;
    }
    // Actual fixed headers/suffix sends form one complete source-read chain;
    // preserving it also protects the free prefix from PlanVictims writes.
    for(size_t i=0;i<owners.size();++i){auto name=sent[i];if(owners[i]->declaration().hasPayload())name=prior(endpoint(name,i,q::PooledVictimWireRole::Decisions,CapturedTransferEndpoint::Producer));const auto before=prior(endpoint(name,i,q::PooledVictimWireRole::Plan,CapturedTransferEndpoint::Producer));if(before!=(i?sent[i-1]:planned))throw std::invalid_argument("Original victim plan sender order/read frontier changed");}
    current=prior(phase(planned,n::CoordinatorAction::PlanVictims,0));requirePooledFreeCoordinatorCompletion(graph,previous,current);
}
PooledPinParticipantNodes appendPooledVictimOwnerWave(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const q::PooledImmutableCapturedVictimWave> edge,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& acquiredFree,const std::string& prefix)
{
    if(isGraphCaptureActive() || !edge || !code || prefix.empty())throw std::invalid_argument("Victim owner needs original cold free owners/code");edge->requireCurrent();const auto storage=edge->previous()->pin()->forward()->ownerStorage();const auto device=storage->plan().allocator().device;if(code->backend()!=device.type)throw std::invalid_argument("Victim owner code changed backend");requirePooledFreeOwnerCompletion(graph,edge->previous(),acquiredFree);
    const std::array suffixes{"plan_receive","decisions_receive","accept_victim","execute_victim","grant_send","victim_receipt","receipt_send"};std::vector<std::string> names;for(auto suffix:suffixes)names.push_back(prefix+"/"+suffix);victimNames(graph,names);std::vector<std::unique_ptr<IComputeStage>> stages;
    const auto transfer=[&](q::PooledVictimWireRole role,CapturedTransferEndpoint endpoint){return std::make_unique<PooledVictimTransferStage>(arena,edge,role,endpoint);};const auto phase=[&](n::OwnerAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,storage,q::PooledImmutableOwnerOperands(action,storage->plan(),edge->ownerInputs(action),n::SourceCondition::Always,action==n::OwnerAction::AcceptVictim?n::RecordLayout::TransportedAllocationPrefix:n::RecordLayout::Compact)});};
    stages.push_back(transfer(q::PooledVictimWireRole::Plan,CapturedTransferEndpoint::Consumer));stages.push_back(edge->declaration().hasPayload()?transfer(q::PooledVictimWireRole::Decisions,CapturedTransferEndpoint::Consumer):nullptr);stages.push_back(phase(n::OwnerAction::AcceptVictim));stages.push_back(phase(n::OwnerAction::ExecuteVictim));stages.push_back(edge->declaration().hasPayload()?transfer(q::PooledVictimWireRole::Grants,CapturedTransferEndpoint::Producer):nullptr);stages.push_back(phase(n::OwnerAction::VictimReceipt));stages.push_back(transfer(q::PooledVictimWireRole::Receipt,CapturedTransferEndpoint::Producer));
    auto prior=acquiredFree;for(size_t i=0;i<stages.size();++i)if(stages[i]){graph.addNode(names[i],std::move(stages[i]),device);graph.addDependency(names[i],prior);prior=names[i];}return {names.front(),prior};
}
void requirePooledMixedVictimCompletion(const ComputeGraph& graph,const PooledImmutableMixedVictimCompletion& original)
{
    if(isGraphCaptureActive() || !original.previous || !original.previous->previous || !original.previous->previous->cohort)
        throw std::invalid_argument("Mixed source insertion requires its actual cold Victim frontier");
    requirePooledMixedFreeCompletion(graph,*original.previous);const auto cohort=original.previous->previous->cohort;const auto storage=cohort->coordinatorStorage();const auto device=storage->plan().allocator().device;
    const auto local=std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(cohort->owners()[cohort->localCoordinate()]);const auto count=cohort->owners().size();
    if(original.owners.size()!=count || original.replies.size()!=count || original.coordinator.first!=original.planned || original.coordinator.last!=original.completed)throw std::invalid_argument("Mixed Victim lost complete original boundaries");
    const auto node=[&](const std::string& name){const auto* value=graph.getNode(name);if(!value || value->device!=device)throw std::invalid_argument("Mixed Victim lost original coordinator-local node");return value;};
    const auto depends=[&](const std::string& consumer,const std::string& producer){const auto* c=node(consumer);node(producer);if(std::find(c->dependencies.begin(),c->dependencies.end(),producer)==c->dependencies.end())throw std::invalid_argument("Mixed Victim lost actual original send/read/phase dependency");};
    const auto owner=[&](const std::string& name,n::OwnerAction action){const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(name)->stage.get());const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;const auto expected=local->ownerOperands(action);if(!actual || actual->action!=action || actual->storage!=local->ownerStorage() || actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Mixed Victim replaced original local owner phase/inputs");};
    const auto coordinator=[&](const std::string& name,n::CoordinatorAction action,q::PooledImmutableCoordinatorOperands expected){const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(name)->stage.get());const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase()):nullptr;if(!actual || actual->action!=action || actual->storage!=storage || actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Mixed Victim replaced original coordinator phase/inputs");};
    coordinator(original.planned,n::CoordinatorAction::PlanVictims,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::PlanVictims,storage->plan(),{},{}));depends(original.planned,original.previous->coordinator.last);requirePooledOriginalDependencies(graph,original.planned,{original.previous->coordinator.last});
    owner(original.localAccept,n::OwnerAction::AcceptVictim);owner(original.localExecute,n::OwnerAction::ExecuteVictim);owner(original.localReceipt,n::OwnerAction::VictimReceipt);depends(original.localExecute,original.localAccept);requirePooledOriginalDependencies(graph,original.localExecute,{original.localAccept});
    std::string priorCommand=original.planned,priorReceipt;
    for(size_t i=0;i<count;++i){const auto& r=original.replies[i];const bool direct=i==local->coordinate();const bool positive=direct?local->hasPayload():std::get<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>>(original.owners[i])->declaration().hasPayload();
        if(r.coordinate!=i || (r.path==PooledPinReplyOwnerNodes::Path::DirectLocal)!=direct || positive!=!r.replyAcquired.empty() || (direct && !r.replyReceived.empty()) || (!direct && positive!=!r.replyReceived.empty()))throw std::invalid_argument("Mixed Victim changed original owner/C0/local order");
        const auto first=direct && positive?r.replyAcquired:positive?r.replyReceived:r.headerReceived;
        depends(first,original.planned);for(const auto& other:original.replies)depends(first,other.commandSent);if(!priorReceipt.empty())depends(first,priorReceipt);
        std::vector<std::string> reads{original.planned};for(const auto& other:original.replies)reads.push_back(other.commandSent);if(!priorReceipt.empty())reads.push_back(priorReceipt);
        if(direct)reads.push_back(positive?original.localExecute:original.localReceipt);
        requirePooledOriginalDependencies(graph,first,std::move(reads));
        if(direct){
            if(std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(original.owners[i])!=local || r.commandSent!=original.localAccept)throw std::invalid_argument("Mixed Victim substituted original local relation/reader");
            depends(original.localAccept,priorCommand);requirePooledOriginalDependencies(graph,original.localAccept,{priorCommand});
            if(positive){coordinator(r.replyAcquired,n::CoordinatorAction::AcquireVictims,local->coordinatorOperands(n::CoordinatorAction::AcquireVictims));depends(r.replyAcquired,original.localExecute);depends(original.localReceipt,r.replyAcquired);depends(r.headerReceived,r.replyAcquired);}else depends(original.localReceipt,original.localExecute);
            requirePooledOriginalDependencies(graph,original.localReceipt,{positive?r.replyAcquired:original.localExecute});
            coordinator(r.headerReceived,n::CoordinatorAction::ImportLocalVictimReceipt,local->coordinatorOperands(n::CoordinatorAction::ImportLocalVictimReceipt));depends(r.headerReceived,original.localReceipt);
            if(positive)requirePooledOriginalDependencies(graph,r.headerReceived,{original.localReceipt,r.replyAcquired});
        }else{
            const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>>(original.owners[i]);edge->requireCurrent();if(edge->previous()!=std::get<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>>(original.previous->owners[i]))throw std::invalid_argument("Mixed Victim substituted original remote Free relation");
            const auto endpoint=[&](const std::string& name,q::PooledVictimWireRole role,CapturedTransferEndpoint side){const auto* stage=dynamic_cast<const PooledVictimTransferStage*>(node(name)->stage.get());if(!stage || !stage->matches(edge.get(),role,side))throw std::invalid_argument("Mixed Victim replaced actual original channel endpoint");};
            auto sent=r.commandSent;if(positive){endpoint(sent,q::PooledVictimWireRole::Decisions,CapturedTransferEndpoint::Producer);const auto* value=node(sent);if(value->dependencies.size()!=1)throw std::invalid_argument("Mixed Victim replaced decision-send predecessor");sent=value->dependencies.front();depends(r.commandSent,sent);endpoint(r.replyReceived,q::PooledVictimWireRole::Grants,CapturedTransferEndpoint::Consumer);coordinator(r.replyAcquired,n::CoordinatorAction::AcquireVictims,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::AcquireVictims,storage->plan(),{},edge->coordinatorInputs(n::CoordinatorAction::AcquireVictims)));depends(r.replyAcquired,r.replyReceived);depends(r.headerReceived,r.replyAcquired);requirePooledOriginalDependencies(graph,r.replyAcquired,{r.replyReceived});requirePooledOriginalDependencies(graph,r.headerReceived,{r.replyAcquired});}
            endpoint(sent,q::PooledVictimWireRole::Plan,CapturedTransferEndpoint::Producer);depends(sent,priorCommand);requirePooledOriginalDependencies(graph,sent,{priorCommand});endpoint(r.headerReceived,q::PooledVictimWireRole::Receipt,CapturedTransferEndpoint::Consumer);
        }
        priorCommand=r.commandSent;priorReceipt=r.headerReceived;
    }
    q::PooledNativeOperandSpans spans{};spans[static_cast<unsigned>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,storage->plan().parameters().native.ingress_frontier,count*sizeof(n::p::VictimAllocationReceipt)};
    coordinator(original.completed,n::CoordinatorAction::CompleteVictims,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::CompleteVictims,storage->plan(),{},std::move(spans)));depends(original.completed,priorReceipt);requirePooledOriginalDependencies(graph,original.completed,{priorReceipt});
}
PooledImmutableMixedVictimCompletion appendPooledVictimCoordinatorMembers(ComputeGraph& graph,BufferArena& arena,
    std::span<const q::PooledImmutableVictimParticipant> owners,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,
    const std::string& sentFree,const std::string& prefix,const PooledImmutableMixedFreeCompletion* localFree)
{
    if(isGraphCaptureActive() || owners.empty() || !code || prefix.empty())throw std::invalid_argument("Victim coordinator needs original complete cold cohort/code");
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> storage;std::shared_ptr<const q::PooledImmutableLocalOwnerBinding> local;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> remote(owners.size());std::vector<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> previous;
    for(size_t i=0;i<owners.size();++i)std::visit([&](const auto& binding){
        if(!binding)throw std::invalid_argument("Victim coordinator lost original member");binding->requireCurrent();
        using Binding=typename std::decay_t<decltype(binding)>::element_type;
        std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> c;std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> o;
        if constexpr(std::is_same_v<Binding,const q::PooledImmutableLocalOwnerBinding>){if(local)throw std::invalid_argument("Victim coordinator duplicated local owner");local=binding;c=binding->coordinatorStorage();o=binding->ownerStorage();}
        else{remote[i]=binding;previous.push_back(binding->previous());c=binding->previous()->pin()->forward()->coordinatorStorage();o=binding->previous()->pin()->forward()->ownerStorage();}
        if(!storage)storage=c;if(c!=storage || o->plan().parameters().owner.coordinate()!=i)throw std::invalid_argument("Victim coordinator replaced original source/member order");storage->plan().requireOwnerPlan(o->plan());
    },owners[i]);
    const auto& plan=storage->plan();const auto device=plan.allocator().device;
    if(owners.size()!=plan.parameters().policy.participants() || code->backend()!=device.type || bool(local)!=bool(localFree))throw std::invalid_argument("Victim coordinator changed original cohort/backend/local predecessor");
    if(local){
        requirePooledMixedFreeCompletion(graph,*localFree);
        if(localFree->coordinator.last!=sentFree || localFree->previous->cohort->coordinatorStorage()!=storage ||
            localFree->previous->cohort->localCoordinate()!=local->coordinate() ||
            std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(localFree->owners[local->coordinate()])!=local)
            throw std::invalid_argument("Victim coordinator substituted original local Free completion");
        for(size_t i=0;i<owners.size();++i)if(remote[i] && std::get<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>>(localFree->owners[i])!=remote[i]->previous())throw std::invalid_argument("Victim coordinator substituted original remote Free owner");
    }else requirePooledFreeCoordinatorCompletion(graph,previous,sentFree);
    PooledImmutableMixedVictimCompletion result{};if(localFree)result.previous=*localFree;result.owners.assign(owners.begin(),owners.end());
    result.planned=prefix+"/plan_victims";result.completed=prefix+"/complete_victims";std::vector<std::string> names{result.planned,result.completed};
    if(local){result.localAccept=prefix+"/local_accept_victim";result.localExecute=prefix+"/local_execute_victim";result.localReceipt=prefix+"/local_victim_receipt";for(const auto& name:{result.localAccept,result.localExecute,result.localReceipt})names.push_back(name);}
    for(size_t i=0;i<owners.size();++i){const bool direct=local && i==local->coordinate();const bool positive=direct?local->hasPayload():remote[i]->declaration().hasPayload();const auto b=prefix+"/owner"+std::to_string(i);
        result.replies.push_back({static_cast<uint32_t>(i),direct?result.localAccept:positive?b+"/decisions_send":b+"/plan_send",positive && !direct?b+"/grant_receive":"",positive?b+"/grant_acquire":"",b+(direct?"/receipt_local_import":"/receipt_receive"),direct?PooledPinReplyOwnerNodes::Path::DirectLocal:PooledPinReplyOwnerNodes::Path::Remote});
        names.push_back(result.replies.back().headerReceived);if(positive)names.push_back(result.replies.back().replyAcquired);
        if(!direct){names.push_back(b+"/plan_send");if(positive){names.push_back(b+"/decisions_send");names.push_back(b+"/grant_receive");}}
    }
    victimNames(graph,names);struct Node{std::string name;std::unique_ptr<IComputeStage> stage;};std::vector<Node> nodes;
    const auto global=[&](n::CoordinatorAction action){q::PooledNativeOperandSpans spans{};if(action==n::CoordinatorAction::CompleteVictims)spans[static_cast<unsigned>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,plan.parameters().native.ingress_frontier,owners.size()*sizeof(n::p::VictimAllocationReceipt)};return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{action,storage,q::PooledImmutableCoordinatorOperands(action,plan,{},std::move(spans))});};
    const auto localOwner=[&](n::OwnerAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,local->ownerStorage(),local->ownerOperands(action)});};
    const auto localCoordinator=[&](n::CoordinatorAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{action,storage,local->coordinatorOperands(action)});};
    const auto transfer=[&](size_t i,q::PooledVictimWireRole role,CapturedTransferEndpoint endpoint){return std::make_unique<PooledVictimTransferStage>(arena,remote[i],role,endpoint);};
    nodes.push_back({result.planned,global(n::CoordinatorAction::PlanVictims)});nodes.push_back({result.completed,global(n::CoordinatorAction::CompleteVictims)});
    if(local){nodes.push_back({result.localAccept,localOwner(n::OwnerAction::AcceptVictim)});nodes.push_back({result.localExecute,localOwner(n::OwnerAction::ExecuteVictim)});nodes.push_back({result.localReceipt,localOwner(n::OwnerAction::VictimReceipt)});}
    for(size_t i=0;i<owners.size();++i){const auto& r=result.replies[i];const bool direct=r.path==PooledPinReplyOwnerNodes::Path::DirectLocal;const bool positive=!r.replyAcquired.empty();const auto b=prefix+"/owner"+std::to_string(i);
        if(direct){if(positive)nodes.push_back({r.replyAcquired,localCoordinator(n::CoordinatorAction::AcquireVictims)});nodes.push_back({r.headerReceived,localCoordinator(n::CoordinatorAction::ImportLocalVictimReceipt)});}
        else{nodes.push_back({b+"/plan_send",transfer(i,q::PooledVictimWireRole::Plan,CapturedTransferEndpoint::Producer)});if(positive){nodes.push_back({r.commandSent,transfer(i,q::PooledVictimWireRole::Decisions,CapturedTransferEndpoint::Producer)});nodes.push_back({r.replyReceived,transfer(i,q::PooledVictimWireRole::Grants,CapturedTransferEndpoint::Consumer)});nodes.push_back({r.replyAcquired,std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::AcquireVictims,storage,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::AcquireVictims,plan,{},remote[i]->coordinatorInputs(n::CoordinatorAction::AcquireVictims))})});}nodes.push_back({r.headerReceived,transfer(i,q::PooledVictimWireRole::Receipt,CapturedTransferEndpoint::Consumer)});}
    }
    for(auto& node:nodes)graph.addNode(node.name,std::move(node.stage),device);graph.addDependency(result.planned,sentFree);auto prior=result.planned;
    for(size_t i=0;i<owners.size();++i){const auto& r=result.replies[i];const bool direct=r.path==PooledPinReplyOwnerNodes::Path::DirectLocal;
        if(direct){graph.addDependency(result.localAccept,prior);graph.addDependency(result.localExecute,result.localAccept);graph.addDependency(result.localReceipt,!r.replyAcquired.empty()?r.replyAcquired:result.localExecute);graph.addDependency(r.headerReceived,result.localReceipt);if(!r.replyAcquired.empty())graph.addDependency(r.replyAcquired,result.localExecute);}
        else{const auto sent=prefix+"/owner"+std::to_string(i)+"/plan_send";graph.addDependency(sent,prior);if(!r.replyReceived.empty())graph.addDependency(r.commandSent,sent);}prior=r.commandSent;
    }
    wirePooledPinReplyBankLifetime(graph,plan,result.replies,result.planned);graph.addDependency(result.completed,result.replies.back().headerReceived);
    result.coordinator={result.planned,result.completed};return result;
}
PooledPinParticipantNodes appendPooledVictimCoordinatorWave(ComputeGraph& graph,BufferArena& arena,
    std::span<const std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> owners,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& sentFree,const std::string& prefix)
{
    std::vector<q::PooledImmutableVictimParticipant> members;for(const auto& owner:owners)members.emplace_back(owner);
    return appendPooledVictimCoordinatorMembers(graph,arena,members,std::move(code),sentFree,prefix).coordinator;
}
}

namespace llaminar2 {
/** @brief Inspect only the exact original endpoint stage types without another object/vtable allocation. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledVictimTransfer(const IComputeStage& stage){
    if(const auto* original=dynamic_cast<const PooledVictimTransferStage*>(&stage))return original->directedTransferOperation();
    return std::nullopt;
}
}
