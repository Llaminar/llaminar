/**
 * @file PooledImmutableParticipantSourceGraph.cpp
 * @brief Exact original source/count bindings and explicit source-fill graph DAG.
 *
 * Source compute touches only its coordinator/local source roots. Each remote
 * owner performs its own fill/copy/validation; transfer remains a graph collective.
 * The shared packet's two actual send readers retire before its next writer.
 */
#include "PooledImmutableParticipantSourceGraph.h"
#include "PooledImmutableDirectedOperation.h"
#include "PooledImmutableDirectedMessageGeometry.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <set>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;
/** @return Checked nominal message index. */
size_t roleIndex(PooledSourceWireRole role){const auto i=static_cast<unsigned>(role);if(i>=5)throw std::invalid_argument("Unknown original source message role");return i;}
/** @return Only the actual owner seal receipt travels owner to source. */
bool reverse(PooledSourceWireRole role){(void)roleIndex(role);return role==PooledSourceWireRole::SealReceipt;}
/** @return Original root matching the exact nominal source or owner arena role. */
std::shared_ptr<TensorBase> root(const PooledImmutableNativeRoots& roots,BufferId id){return id==BufferId::QFN_POOLED_SOURCE_FRAME?roots.frame:roots.native;}
/** @brief Retain one exact source-fill schema role; C0 has no dummy data/count spans. */
void operand(PooledNativeOperandSpans& out,PooledNativeOperand role,const PooledImmutableNativeRoots& roots,PooledPinWireSpan span){
    if(!span.bytes)return;const auto value=root(roots,span.buffer);
    if(!value || value->is_view() || value->transferStorageOwner()!=value.get() || span.offset>value->size_bytes() || span.bytes>value->size_bytes()-span.offset)
        throw std::invalid_argument("Source fill lost original root/subrange");
    out[static_cast<size_t>(role)]=PooledNativeTensorSpan{value,span.buffer,span.offset,span.bytes};
}
}
PooledPinWireSpan PooledImmutableCapturedSourceWave::source(PooledSourceWireRole role)const{
    (void)roleIndex(role);const auto& s=boundary_->pooledStorage()->geometry().parameters();const auto& o=previous_->previous()->pin()->forward()->ownerStorage()->plan().parameters();
    switch(role){
    case PooledSourceWireRole::InitialFrontier:case PooledSourceWireRole::TerminalFrontier:return {BufferId::QFN_POOLED_SOURCE_NATIVE,s.offsets.frontier,sizeof(n::SourceFrontier)};
    case PooledSourceWireRole::SealReceipt:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(n::OwnerSourceSealReceipt)};
    case PooledSourceWireRole::Fills:return {BufferId::QFN_POOLED_SOURCE_FRAME,s.offsets.fills,PooledImmutableDirectedMessageGeometry::fromPlan(previous_->previous()->pin()->forward()->ownerStorage()->plan()).fills()};
    case PooledSourceWireRole::Payload:return {BufferId::QFN_POOLED_SOURCE_FRAME,s.offsets.payload,PooledImmutableDirectedMessageGeometry::fromPlan(previous_->previous()->pin()->forward()->ownerStorage()->plan()).rows()};
    }throw std::invalid_argument("Unknown source role");
}
PooledPinWireSpan PooledImmutableCapturedSourceWave::destination(PooledSourceWireRole role)const{
    (void)roleIndex(role);const auto& s=boundary_->pooledStorage()->geometry().parameters();const auto& o=previous_->previous()->pin()->forward()->ownerStorage()->plan().parameters();
    switch(role){
    case PooledSourceWireRole::InitialFrontier:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(n::SourceFrontier)};
    case PooledSourceWireRole::TerminalFrontier:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.source_outcome,sizeof(n::SourceFrontier)};
    case PooledSourceWireRole::SealReceipt:return {BufferId::QFN_POOLED_SOURCE_NATIVE,s.offsets.seal_receipt,sizeof(n::OwnerSourceSealReceipt)};
    case PooledSourceWireRole::Fills:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.source_fills,PooledImmutableDirectedMessageGeometry::fromPlan(previous_->previous()->pin()->forward()->ownerStorage()->plan()).fills()};
    case PooledSourceWireRole::Payload:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.source_payload,PooledImmutableDirectedMessageGeometry::fromPlan(previous_->previous()->pin()->forward()->ownerStorage()->plan()).rows()};
    }throw std::invalid_argument("Unknown source role");
}
std::shared_ptr<const PooledImmutableCapturedSourceWave> PooledImmutableCapturedSourceWave::bind(
    std::shared_ptr<const PooledImmutableCapturedVictimWave> previous,std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,std::shared_ptr<PhysicalMemoryAuthority> memory)
{
    if(isGraphCaptureActive() || !previous || !boundary || !boundary->isPooled() || !memory)throw std::invalid_argument("Source fill binding needs original cold victim/service/PMA owners");
    previous->requireCurrent();const auto pin=previous->previous()->pin();const auto forward=pin->forward();const auto source=boundary->pooledStorage();const auto coordinate=previous->declaration().coordinate();
    forward->ownerStorage()->requireAuthority(memory);forward->coordinatorStorage()->requireAuthority(memory);
    if(source->coordinator()!=forward->coordinatorStorage() || coordinate>=source->originalOwners().size() || source->originalOwners()[coordinate]!=forward->ownerStorage() ||
        !source->geometry().parameters().offsets.seal_receipt || (!forward->channel_.nativeCollective() && !forward->declaration().capturedChannelTransport()) || !forward->channel_ || !pin->channel_)
        throw std::invalid_argument("Source fill requires original all-remote r5 cross-vendor cohort; native/rank/mixed-local factory remains guarded");
    source->coordinator()->requireAuthority(memory);source->captureIdentity();
    // The original service binding authenticated source roots against this same PMA at publication.
    // No second allocation/channel is created here.
    auto candidate=std::shared_ptr<PooledImmutableCapturedSourceWave>(new PooledImmutableCapturedSourceWave(previous,boundary,memory,{}));
    const auto src=source->roots(),own=forward->ownerStorage()->roots();const auto& offsets=source->geometry().parameters().offsets;
    for(unsigned i=0;i<5;++i){const auto role=static_cast<PooledSourceWireRole>(i);if(!candidate->hasPayload() && (role==PooledSourceWireRole::Fills || role==PooledSourceWireRole::Payload))continue;const bool back=reverse(role);const auto& channel=back?pin->channel_:forward->channel_;const auto from=candidate->source(role),to=candidate->destination(role);const auto key=((forward->declaration().identity()^(48+i))*1099511628211ULL)|1;
        auto send=(channel).bind(CapturedTransferEndpoint::Producer,{key,from.bytes},root(back?own:src,from.buffer),from.offset);
        auto receive=(channel).bind(CapturedTransferEndpoint::Consumer,{key,from.bytes},root(back?src:own,to.buffer),to.offset);
        if(role==PooledSourceWireRole::Fills || role==PooledSourceWireRole::Payload){const bool fills=role==PooledSourceWireRole::Fills;const auto& p=forward->ownerStorage()->plan().parameters();send=std::move(send).withDeviceCounted(src.native,fills?offsets.record_bytes:offsets.payload_bytes,fills?PooledNativeCountedMessage::SourceFills:PooledNativeCountedMessage::SourcePayload);receive=std::move(receive).withDeviceCounted(own.native,p.native.ingress+(fills?offsetof(n::OwnerIngress,record_bytes):offsetof(n::OwnerIngress,payload_bytes)),fills?PooledNativeCountedMessage::SourceFills:PooledNativeCountedMessage::SourcePayload);}
        candidate->bindings_[i].emplace(std::move(send),std::move(receive));
    }
    candidate->requireCurrent();return candidate;
}
void PooledImmutableCapturedSourceWave::requireCurrent()const{
    previous_->requireCurrent();const auto forward=previous_->previous()->pin()->forward();const auto storage=boundary_->pooledStorage();
    forward->ownerStorage()->requireAuthority(memory_);forward->coordinatorStorage()->requireAuthority(memory_);
    if(storage->coordinator()!=forward->coordinatorStorage() || coordinate()>=storage->originalOwners().size() || storage->originalOwners()[coordinate()]!=forward->ownerStorage())throw std::logic_error("Source fill replaced original source/owner/coordinator");
    storage->captureIdentity();
}
const PooledImmutableTransferBinding& PooledImmutableCapturedSourceWave::binding(PooledSourceWireRole role,CapturedTransferEndpoint endpoint)const{
    requireCurrent();const auto i=roleIndex(role);if(endpoint!=CapturedTransferEndpoint::Producer && endpoint!=CapturedTransferEndpoint::Consumer)throw std::invalid_argument("Invalid source endpoint");if(!bindings_[i])throw std::invalid_argument("C0 source owner has no Fill/payload transaction");return endpoint==CapturedTransferEndpoint::Producer?bindings_[i]->first:bindings_[i]->second;
}
PooledNativeOperandSpans PooledImmutableCapturedSourceWave::ownerInputs(n::OwnerAction action)const{
    requireCurrent();const auto forward=previous_->previous()->pin()->forward();auto out=forward->ownerInputs(n::OwnerAction::Begin);const auto own=forward->ownerStorage()->roots();const auto& p=forward->ownerStorage()->plan().parameters();
    switch(action){
    case n::OwnerAction::SealSource:operand(out,PooledNativeOperand::Frontier,own,destination(PooledSourceWireRole::InitialFrontier));break;
    case n::OwnerAction::ApplySourceDelivery:
        operand(out,PooledNativeOperand::Frontier,own,destination(PooledSourceWireRole::TerminalFrontier));
        if(hasPayload()){operand(out,PooledNativeOperand::Records,own,destination(PooledSourceWireRole::Fills));operand(out,PooledNativeOperand::Payload,own,destination(PooledSourceWireRole::Payload));operand(out,PooledNativeOperand::RecordBytes,own,{BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress+offsetof(n::OwnerIngress,record_bytes),8});operand(out,PooledNativeOperand::PayloadBytes,own,{BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress+offsetof(n::OwnerIngress,payload_bytes),8});}break;
    case n::OwnerAction::ValidateFills:operand(out,PooledNativeOperand::Outcome,own,destination(PooledSourceWireRole::TerminalFrontier));break;
    case n::OwnerAction::DrainSourceFailure:operand(out,PooledNativeOperand::Frontier,own,destination(PooledSourceWireRole::TerminalFrontier));break;
    default:throw std::invalid_argument("Source fill cannot supply another owner action");
    }return out;
}
uint64_t PooledImmutableCapturedSourceWave::captureIdentity()const{
    requireCurrent();auto value=previous_->captureIdentity();value=(value^boundary_->captureIdentity())*1099511628211ULL;value=(value^boundary_->pooledStorage()->captureIdentity())*1099511628211ULL;
    for(size_t i=0;i<bindings_.size();++i){value=(value^i)*1099511628211ULL;if(bindings_[i]){value=(value^bindings_[i]->first.captureIdentity())*1099511628211ULL;value=(value^bindings_[i]->second.captureIdentity())*1099511628211ULL;}}return value?value:1;
}

std::shared_ptr<PooledImmutableOwnerPreparedStorage> pooledSourceOwner(const PooledImmutableSourceParticipant& member){
    return std::visit([](const auto& original){if(!original)throw std::invalid_argument("Missing actual source participant");using T=std::decay_t<decltype(*original)>;
        if constexpr(std::is_same_v<T,PooledImmutableLocalOwnerBinding>)return original->ownerStorage();
        else return original->previous()->previous()->pin()->forward()->ownerStorage();},member);
}
void requirePooledSourceParticipant(const PooledImmutableSourceParticipant& member,const std::shared_ptr<ImmutableLookupExecutionBoundary>& boundary){
    if(!boundary || !boundary->isPooled())throw std::invalid_argument("Source participant requires its actual original source boundary");const auto source=boundary->pooledStorage();const auto owner=pooledSourceOwner(member);const auto coordinate=owner->plan().parameters().owner.coordinate();
    if(coordinate>=source->originalOwners().size() || source->originalOwners()[coordinate]!=owner)throw std::invalid_argument("Source participant changed its original ordered physical owner");
    std::visit([&](const auto& original){original->requireCurrent();using T=std::decay_t<decltype(*original)>;
        if constexpr(std::is_same_v<T,PooledImmutableLocalOwnerBinding>){
            if(original->coordinatorStorage()!=source->coordinator() || coordinate!=source->geometry().parameters().local_owner)throw std::invalid_argument("Local source participant changed original coordinator/coordinate");
        }else if(original->boundary()!=boundary || original->previous()->previous()->pin()->forward()->coordinatorStorage()!=source->coordinator() || coordinate==source->geometry().parameters().local_owner)throw std::invalid_argument("Remote source participant substituted its real source/coordinator/channel");
    },member);source->captureIdentity();
}
PooledNativeOperandSpans pooledSourceOwnerInputs(const PooledImmutableSourceParticipant& member,
    const std::shared_ptr<ImmutableLookupExecutionBoundary>& boundary,n::OwnerAction action){
    requirePooledSourceParticipant(member,boundary);const auto owner=pooledSourceOwner(member);const auto own=owner->roots();const auto& p=owner->plan().parameters();const auto source=boundary->pooledStorage();const auto src=source->roots();const auto coord=source->coordinator()->roots();const auto& c=source->coordinator()->plan().parameters();const auto& so=source->geometry().parameters().offsets;
    PooledNativeOperandSpans out=std::visit([](const auto& original){using T=std::decay_t<decltype(*original)>;if constexpr(std::is_same_v<T,PooledImmutableLocalOwnerBinding>)return original->ownerInputs(n::OwnerAction::Begin);else return original->previous()->previous()->pin()->forward()->ownerInputs(n::OwnerAction::Begin);},member);
    const bool local=std::holds_alternative<std::shared_ptr<const PooledImmutableLocalOwnerBinding>>(member);
    const auto span=[&](PooledNativeOperand role,const std::shared_ptr<TensorBase>& root,BufferId id,size_t offset,size_t bytes){if(bytes)out[static_cast<size_t>(role)]=PooledNativeTensorSpan{root,id,offset,bytes};};
    const auto outcome=[&](PooledNativeOperand role){span(role,local?src.native:own.native,local?BufferId::QFN_POOLED_SOURCE_NATIVE:BufferId::QFN_POOLED_OWNER_NATIVE,local?so.frontier:p.native.source_outcome,sizeof(n::SourceFrontier));};
    switch(action){
    case n::OwnerAction::SealSource:outcome(PooledNativeOperand::Frontier);if(!local){out={};out=std::get<std::shared_ptr<const PooledImmutableCapturedSourceWave>>(member)->ownerInputs(action);}break;
    case n::OwnerAction::DrainSourceFailure:case n::OwnerAction::ApplySourceDelivery:case n::OwnerAction::ValidateFills:
        if(!local)return std::get<std::shared_ptr<const PooledImmutableCapturedSourceWave>>(member)->ownerInputs(action);
        outcome(action==n::OwnerAction::ValidateFills?PooledNativeOperand::Outcome:PooledNativeOperand::Frontier);
        if(action==n::OwnerAction::ApplySourceDelivery && p.native.reply_capacity){span(PooledNativeOperand::Records,src.frame,BufferId::QFN_POOLED_SOURCE_FRAME,so.fills,size_t{p.native.reply_capacity}*sizeof(n::Fill));span(PooledNativeOperand::RecordBytes,src.native,BufferId::QFN_POOLED_SOURCE_NATIVE,so.record_bytes,8);span(PooledNativeOperand::Payload,src.frame,BufferId::QFN_POOLED_SOURCE_FRAME,so.payload,size_t{p.native.reply_capacity}*p.row_bytes);span(PooledNativeOperand::PayloadBytes,src.native,BufferId::QFN_POOLED_SOURCE_NATIVE,so.payload_bytes,8);}break;
    case n::OwnerAction::AcquirePublication:
        span(PooledNativeOperand::Frontier,local?coord.native:own.frame,local?BufferId::QFN_POOLED_COORDINATOR_NATIVE:BufferId::QFN_POOLED_OWNER_FRAME,local?c.native.ingress_frontier:p.frame.publish_headers,size_t{p.owner.participants()}*sizeof(n::p::PublishHeader));outcome(PooledNativeOperand::Outcome);break;
    case n::OwnerAction::MakeReady:case n::OwnerAction::ReadyReceipt:outcome(PooledNativeOperand::Outcome);break;
    case n::OwnerAction::PublishWire:case n::OwnerAction::PublicationAcquiredWire:case n::OwnerAction::ReadyWire:break;
    default:throw std::invalid_argument("Source participant cannot fabricate another owner operation");
    }return out;
}
}
namespace llaminar2 {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;
/** @return Exact original owner-versus-source participant for the nominal message endpoint. */
bool ownerEndpoint(q::PooledSourceWireRole role,CapturedTransferEndpoint side){return (role==q::PooledSourceWireRole::SealReceipt)==(side==CapturedTransferEndpoint::Producer);}
/** @brief One actual fixed/counted source message on one participant, with visible public transfer ownership. */
class PooledSourceTransferStage final:public IComputeStage {
public:
    /** @return Exact original semantic endpoint and native dispatch inputs before capture. */
    q::PooledImmutableDirectedTransferOperation directedTransferOperation()const{
        requireDeclarations();const auto& binding=edge_->binding(role_,side_);
        return {q::PooledImmutableDirectedWireFamily::Source,static_cast<std::uint32_t>(role_),side_,
            binding.message().bytes,binding.extentSource(),binding.hasSourceSlice(),binding.nativeCollective()?q::PooledImmutableDirectedTransportKind::NativeCollective:q::PooledImmutableDirectedTransportKind::CapturedChannel};
    }

    /** @brief Retain original endpoint/root/channel authority and freeze complete variant identity. */
    PooledSourceTransferStage(BufferArena& arena,std::shared_ptr<const q::PooledImmutableCapturedSourceWave> edge,q::PooledSourceWireRole role,CapturedTransferEndpoint side)
        :IComputeStage(ownerEndpoint(role,side)?edge->previous()->previous()->pin()->forward()->ownerStorage()->plan().allocator().device:edge->boundary()->device()),arena_(&arena),edge_(std::move(edge)),role_(role),side_(side){requireDeclarations();identity_=graphCaptureVariantSignature();native_collective_=edge_->binding(role_,side_).nativeCollective();}
    /** @brief Join only this participant's exact original setup events on its captured stream. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override{
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<uintptr_t>(stream)<=2)throw std::invalid_argument("Source endpoint needs exact participant/nondefault stream");requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Source endpoint capture identity drift");
        if(ownerEndpoint(role_,side_))edge_->previous()->previous()->pin()->forward()->ownerStorage()->acquireSetup(stream);else edge_->boundary()->pooledStorage()->acquireSetup(stream);edge_->binding(role_,side_).prepare(stream);setGPUStream(stream);return true;
    }
    /** @brief Submit only the actual fixed/counted transfer; no live value reaches the host. */
    bool execute(IDeviceContext* context)override{if(!context || context->deviceId()!=device())throw std::invalid_argument("Source transfer changed participant");requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Source transfer identity drift");edge_->binding(role_,side_).enqueue(requireGPUStream());return true;}
    /** @return Explicit graph collective, never a hidden multidevice compute phase. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER;}
    /** @return This original bound endpoint executes native device-counted transfer kernels, independently from collective-runtime capture. */
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority()const noexcept override{return native_collective_?CollectiveGraphCaptureAuthority::RuntimeBackend:CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer;}
    /** @return Nominal component name. */
    std::string name()const override{return "PooledImmutableDirectedSource";}
    /** @return Complete original endpoint identity with no mutable count/cursor value. */
    uint64_t graphCaptureVariantSignature()const override{requireDeclarations();return q::pooledSourceEndpointCaptureIdentity(edge_->captureIdentity(),role_,side_);}
    /** @return Exact holder/action/endpoint for subsequent cold ownership checks. */
    bool matches(const q::PooledImmutableCapturedSourceWave* original,q::PooledSourceWireRole role,CapturedTransferEndpoint side)const noexcept{return edge_.get()==original && role_==role && side_==side;}
    /** @return Imports occur before retained capture, never in graph replay. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Empty and shrinking live prefixes preserve actual original row semantics. */
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    /** @return Only this endpoint's actual backend. */
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    /** @return Existing source/owner/channel factories already declare every physical byte. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Original data/count read and receive-publication roles. */
    StageBufferContract bufferContract()const override{
        requireDeclarations();auto c=StageBufferContract::build();const bool producer=side_==CapturedTransferEndpoint::Producer;const auto span=producer?edge_->source(role_):edge_->destination(role_);if(producer)c.addInput(span.buffer,"INT8");else c.addPreallocatedInOut(span.buffer,"INT8");
        if(producer && (role_==q::PooledSourceWireRole::Fills || role_==q::PooledSourceWireRole::Payload))c.addInput(BufferId::QFN_POOLED_SOURCE_NATIVE,"INT8");return c;
    }
    /** @return Static nominal diagnostics only. */
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo d;d.addScalarInt("source_role",static_cast<unsigned>(role_));d.addScalarInt("source_endpoint",static_cast<unsigned>(side_));return d;}
private:
    /** @brief Reject original arena root replacement before cold capture or native submission. */
    void requireDeclarations()const{
        if(!arena_ || !edge_)throw std::logic_error("Source endpoint lost original owners");edge_->requireCurrent();(void)edge_->binding(role_,side_);const bool owner=ownerEndpoint(role_,side_);const auto roots=owner?edge_->previous()->previous()->pin()->forward()->ownerStorage()->roots():edge_->boundary()->pooledStorage()->roots();
        const auto native=owner?BufferId::QFN_POOLED_OWNER_NATIVE:BufferId::QFN_POOLED_SOURCE_NATIVE,frame=owner?BufferId::QFN_POOLED_OWNER_FRAME:BufferId::QFN_POOLED_SOURCE_FRAME;const auto span=side_==CapturedTransferEndpoint::Producer?edge_->source(role_):edge_->destination(role_);
        if(arena_->getTensor(native)!=roots.native.get() || (span.buffer==frame && arena_->getTensor(frame)!=roots.frame.get()))throw std::logic_error("Source endpoint arena replaced original root authority");
    }
public:
    /** @brief Seal one consumed graph-local trunk join on this exact first response receive. */
    void bindDemandJoin(const ComputeGraph& graph,const ComputeNode& node,const std::string& published,const std::vector<std::string>& trunk){
        if(!demand_trunk_.empty() || side_!=CapturedTransferEndpoint::Consumer ||
           role_!=(edge_->hasPayload()?q::PooledSourceWireRole::Fills:q::PooledSourceWireRole::TerminalFrontier) ||
           node.dependencies.size()!=1 || node.dependencies.front()!=published || trunk.empty())
            throw std::invalid_argument("Inline demand join lost its sole original seal-send predecessor");
        const auto* before=graph.getNode(published);
        const auto* stage=before?dynamic_cast<const PooledSourceTransferStage*>(before->stage.get()):nullptr;
        if(!stage || !stage->matches(edge_.get(),q::PooledSourceWireRole::SealReceipt,CapturedTransferEndpoint::Producer))
            throw std::invalid_argument("Inline demand join substituted its original seal receipt");
        for(const auto& name:trunk)if(name==published || !graph.getNode(name) || graph.getNode(name)->device!=node.device)
            throw std::invalid_argument("Inline demand join lacks actual same-participant trunk readiness");
        demand_trunk_=trunk;
    }
    /** @return Sole original protocol predecessor after checking every declared consumed trunk edge. */
    std::string protocolPredecessor(const ComputeGraph& graph,const ComputeNode& node)const{
        for(const auto& name:demand_trunk_)if(std::count(node.dependencies.begin(),node.dependencies.end(),name)!=1 || !graph.getNode(name))
            throw std::invalid_argument("Inline consumed demand lost an actual trunk dependency");
        std::string original;
        for(const auto& name:node.dependencies)if(std::find(demand_trunk_.begin(),demand_trunk_.end(),name)==demand_trunk_.end()){
            if(!original.empty() || !graph.getNode(name))throw std::invalid_argument("Source owner lost sole original protocol predecessor");original=name;
        }
        if(original.empty())throw std::invalid_argument("Source owner omitted original protocol predecessor");
        return original;
    }
private:
    std::vector<std::string> demand_trunk_; ///< Cold graph declaration; no live execution or readiness state.
    BufferArena* arena_;std::shared_ptr<const q::PooledImmutableCapturedSourceWave> edge_;q::PooledSourceWireRole role_;CapturedTransferEndpoint side_;
    bool native_collective_=false; // Cold-selected original transport; capture policy never probes runtime state.
    uint64_t identity_=0;
};
/** @brief Reject duplicate/missing node names before any new source graph mutation. */
void sourceNames(const ComputeGraph& graph,std::span<const std::string> names){std::set<std::string> unique;for(const auto& name:names)if(name.empty() || graph.getNode(name) || !unique.insert(name).second)throw std::invalid_argument("Source fill graph would replace an existing node");}
}
/** @brief Authenticate actual original source owner producers and sole ordered predecessors before a successor mutates graph. */
void requirePooledSourceOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const q::PooledImmutableCapturedSourceWave>& edge,const std::string& fillOutcome)
{
    if(isGraphCaptureActive() || !edge)throw std::invalid_argument("Publication requires original cold source owner");edge->requireCurrent();const auto storage=edge->previous()->previous()->pin()->forward()->ownerStorage();const auto device=storage->plan().allocator().device;
    const auto prior=[&](const ComputeNode* node){
        if(!node || node->device!=device)throw std::invalid_argument("Source owner lost original participant");
        if(const auto* transfer=dynamic_cast<const PooledSourceTransferStage*>(node->stage.get()))return transfer->protocolPredecessor(graph,*node);
        if(node->dependencies.size()!=1 || !graph.getNode(node->dependencies.front()))throw std::invalid_argument("Source owner lost sole original producer");
        return node->dependencies.front();};
    const auto phase=[&](const std::string& name,n::OwnerAction action){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()):nullptr;const auto* value=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;const auto condition=action==n::OwnerAction::SealSource || action==n::OwnerAction::DrainSourceFailure?n::SourceCondition::Always:n::SourceCondition::AfterSuccess;const q::PooledImmutableOwnerOperands expected(action,storage->plan(),edge->ownerInputs(action),condition);
        if(!value || value->action!=action || value->storage!=storage || value->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Source owner replaced actual original phase/operands");return node;};
    const auto endpoint=[&](const std::string& name,q::PooledSourceWireRole role,CapturedTransferEndpoint side){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledSourceTransferStage*>(node->stage.get()):nullptr;if(!stage || !stage->matches(edge.get(),role,side))throw std::invalid_argument("Source owner replaced original endpoint/channel");return node;};
    auto current=prior(phase(fillOutcome,n::OwnerAction::DrainSourceFailure));current=prior(phase(current,n::OwnerAction::ValidateFills));current=prior(phase(current,n::OwnerAction::ApplySourceDelivery));current=prior(phase(current,n::OwnerAction::DrainSourceFailure));current=prior(endpoint(current,q::PooledSourceWireRole::TerminalFrontier,CapturedTransferEndpoint::Consumer));
    if(edge->hasPayload()){current=prior(endpoint(current,q::PooledSourceWireRole::Payload,CapturedTransferEndpoint::Consumer));current=prior(endpoint(current,q::PooledSourceWireRole::Fills,CapturedTransferEndpoint::Consumer));}
    current=prior(endpoint(current,q::PooledSourceWireRole::SealReceipt,CapturedTransferEndpoint::Producer));current=prior(phase(current,n::OwnerAction::SealSource));current=prior(endpoint(current,q::PooledSourceWireRole::InitialFrontier,CapturedTransferEndpoint::Consumer));requirePooledVictimOwnerCompletion(graph,edge->previous(),current);
}
/** @brief Authenticate original complete source program, including all packet readers and local-last alias retirement. */
void requirePooledSourceCoordinatorCompletion(const ComputeGraph& graph,const PooledImmutableSourceFillCompletion& original)
{
    if(isGraphCaptureActive() || !original.boundary || !original.boundary->isPooled() || !original.control || original.owners.empty())throw std::invalid_argument("Publication requires original cold source program");const auto storage=original.boundary->pooledStorage();const auto device=original.boundary->device();const auto count=original.owners.size();
    if(count!=storage->geometry().originalOwners().size() || original.initial.size()!=count || original.sealReceived.size()!=count || original.sealAcquired.size()!=count || original.packed.size()!=count || original.fillSent.size()!=count || original.payloadSent.size()!=count || original.terminalSent.size()!=count || original.coordinator.first!=original.planned)throw std::invalid_argument("Source completion lost original full cohort/boundaries");
    const auto node=[&](const std::string& name){const auto* value=graph.getNode(name);if(!value || value->device!=device)throw std::invalid_argument("Source completion lost original participant node");return value;};
    const auto depends=[&](const std::string& name,const std::string& prior){node(name);node(prior);requirePooledOriginalDependencies(graph,name,{prior});};
    const auto source=[&](const std::string& name,q::PooledSourceAction action,uint32_t owner=q::adaptive_lookup::None){const auto* stage=dynamic_cast<const PooledImmutableSourcePhaseStage*>(node(name)->stage.get());if(!stage || !stage->matchesOriginal(original.boundary.get(),action,owner,*original.control))throw std::invalid_argument("Source completion replaced actual source/action/control/coordinate");};
    std::optional<size_t> local;std::vector<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> previous;
    const auto endpoint=[&](size_t index,const std::string& name,q::PooledSourceWireRole role){const auto& edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedSourceWave>>(original.owners[index]);const auto* stage=dynamic_cast<const PooledSourceTransferStage*>(node(name)->stage.get());if(!stage || !stage->matches(edge.get(),role,role==q::PooledSourceWireRole::SealReceipt?CapturedTransferEndpoint::Consumer:CapturedTransferEndpoint::Producer))throw std::invalid_argument("Source completion replaced original endpoint/channel");};
    for(size_t i=0;i<count;++i){q::requirePooledSourceParticipant(original.owners[i],original.boundary);if(q::pooledSourceOwner(original.owners[i])->plan().parameters().owner.coordinate()!=i)throw std::invalid_argument("Source completion reordered original members");if(const auto* remote=std::get_if<std::shared_ptr<const q::PooledImmutableCapturedSourceWave>>(&original.owners[i])){previous.push_back((*remote)->previous());endpoint(i,original.initial[i],q::PooledSourceWireRole::InitialFrontier);endpoint(i,original.sealReceived[i],q::PooledSourceWireRole::SealReceipt);endpoint(i,original.terminalSent[i],q::PooledSourceWireRole::TerminalFrontier);if((*remote)->hasPayload()){endpoint(i,original.fillSent[i],q::PooledSourceWireRole::Fills);endpoint(i,original.payloadSent[i],q::PooledSourceWireRole::Payload);}else if(!original.fillSent[i].empty() || !original.payloadSent[i].empty())throw std::invalid_argument("Source C0 gained a payload reader");}else{if(local || !original.initial[i].empty() || !original.sealReceived[i].empty() || !original.terminalSent[i].empty() || !original.fillSent[i].empty() || !original.payloadSent[i].empty())throw std::invalid_argument("Source local owner gained self transport");local=i;}source(original.sealAcquired[i],q::PooledSourceAction::AcquireSeal,i);source(original.packed[i],q::PooledSourceAction::PackOwner,i);}
    if(local){if(!original.previous || original.previous->completed!=original.predecessor)throw std::invalid_argument("Source lost actual mixed victim producer");requirePooledMixedVictimCompletion(graph,*original.previous);for(size_t i=0;i<count;++i){if(i==*local){if(std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(original.owners[i])!=std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(original.previous->owners[i]))throw std::invalid_argument("Source changed actual local victim relation");}else if(std::get<std::shared_ptr<const q::PooledImmutableCapturedSourceWave>>(original.owners[i])->previous()!=std::get<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>>(original.previous->owners[i]))throw std::invalid_argument("Source changed actual remote victim relation");}}
    else{if(original.previous)throw std::invalid_argument("Remote source gained local completion");requirePooledVictimCoordinatorCompletion(graph,previous,original.predecessor);}
    const auto localPhase=[&](const std::string& name,n::OwnerAction action){const auto owner=q::pooledSourceOwner(original.owners[*local]);const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(name)->stage.get());const auto* value=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;const auto condition=action==n::OwnerAction::SealSource || action==n::OwnerAction::DrainSourceFailure?n::SourceCondition::Always:n::SourceCondition::AfterSuccess;const q::PooledImmutableOwnerOperands expected(action,owner->plan(),q::pooledSourceOwnerInputs(original.owners[*local],original.boundary,action),condition);if(!value || value->storage!=owner || value->action!=action || value->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Source completion replaced original local owner phase/operands");};
    source(original.planned,q::PooledSourceAction::Plan);depends(original.planned,original.predecessor);source(original.published,q::PooledSourceAction::Publish);source(original.acquired,q::PooledSourceAction::AcquireResponse);source(original.retired,q::PooledSourceAction::RetireResponse);
    const auto* sealStage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(original.sealed)->stage.get());const auto* sealed=sealStage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&sealStage->phase()):nullptr;q::PooledNativeOperandSpans seal{};seal[static_cast<size_t>(q::PooledNativeOperand::Source)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,storage->geometry().parameters().offsets.frontier,sizeof(n::SourceFrontier)};const q::PooledImmutableCoordinatorOperands expected(n::CoordinatorAction::SealSource,storage->coordinator()->plan(),{},seal);if(!sealed || sealed->action!=n::CoordinatorAction::SealSource || sealed->storage!=storage->coordinator() || sealed->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Source completion replaced original coordinator Seal/source");
    depends(original.sealed,original.planned);auto prior=original.sealed;for(const auto& sent:original.initial)if(!sent.empty()){depends(sent,prior);prior=sent;}
    for(size_t i=0;i<count;++i){if(local && i==*local){localPhase(original.localSeal,n::OwnerAction::SealSource);depends(original.localSeal,prior);depends(original.sealAcquired[i],original.localSeal);}else{depends(original.sealReceived[i],prior);depends(original.sealAcquired[i],original.sealReceived[i]);}prior=original.sealAcquired[i];}depends(original.published,prior);depends(original.acquired,original.published);prior=original.acquired;
    for(size_t ordinal=0;ordinal<count;++ordinal){const auto i=local?(ordinal==count-1?*local:ordinal<*local?ordinal:ordinal+1):ordinal;depends(original.packed[i],prior);prior=original.packed[i];if(!original.fillSent[i].empty()){depends(original.fillSent[i],prior);depends(original.payloadSent[i],original.fillSent[i]);prior=original.payloadSent[i];}}depends(original.retired,prior);prior=original.retired;for(const auto& sent:original.terminalSent)if(!sent.empty()){depends(sent,prior);prior=sent;}
    if(local){localPhase(original.localDrain,n::OwnerAction::DrainSourceFailure);localPhase(original.localApply,n::OwnerAction::ApplySourceDelivery);localPhase(original.localValidate,n::OwnerAction::ValidateFills);localPhase(original.localOutcome,n::OwnerAction::DrainSourceFailure);depends(original.localDrain,prior);depends(original.localApply,original.localDrain);depends(original.localValidate,original.localApply);depends(original.localOutcome,original.localValidate);prior=original.localOutcome;}
    if(original.coordinator.last!=prior)throw std::invalid_argument("Source completion replaced actual last packet reader/outcome");
}
PooledPinParticipantNodes appendPooledSourceOwnerFill(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const q::PooledImmutableCapturedSourceWave> edge,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& sentVictims,const std::string& prefix, PooledImmutableDemandFrontier* demand)
{
    if(isGraphCaptureActive() || !edge || !code || prefix.empty())throw std::invalid_argument("Source owner graph needs original cold owners/code");edge->requireCurrent();const auto owner=edge->previous()->previous()->pin()->forward()->ownerStorage();const auto device=owner->plan().allocator().device;if(code->backend()!=device.type)throw std::invalid_argument("Source owner code changed backend");
    requirePooledVictimOwnerCompletion(graph,edge->previous(),sentVictims);
    const std::array suffixes{"initial_receive","seal_source","seal_receipt_send","fill_receive","payload_receive","terminal_receive","drain_source_failure","apply_source","validate_fills","fill_outcome"};std::vector<std::string> names;for(const auto* suffix:suffixes)names.push_back(prefix+"/"+suffix);sourceNames(graph,names);std::vector<std::unique_ptr<IComputeStage>> stages;
    const auto transfer=[&](q::PooledSourceWireRole role,CapturedTransferEndpoint side){return std::make_unique<PooledSourceTransferStage>(arena,edge,role,side);};const auto phase=[&](n::OwnerAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,owner,q::PooledImmutableOwnerOperands(action,owner->plan(),edge->ownerInputs(action),(action==n::OwnerAction::SealSource || action==n::OwnerAction::DrainSourceFailure)?n::SourceCondition::Always:n::SourceCondition::AfterSuccess)});};
    stages.push_back(transfer(q::PooledSourceWireRole::InitialFrontier,CapturedTransferEndpoint::Consumer));stages.push_back(phase(n::OwnerAction::SealSource));stages.push_back(transfer(q::PooledSourceWireRole::SealReceipt,CapturedTransferEndpoint::Producer));stages.push_back(edge->hasPayload()?transfer(q::PooledSourceWireRole::Fills,CapturedTransferEndpoint::Consumer):nullptr);stages.push_back(edge->hasPayload()?transfer(q::PooledSourceWireRole::Payload,CapturedTransferEndpoint::Consumer):nullptr);stages.push_back(transfer(q::PooledSourceWireRole::TerminalFrontier,CapturedTransferEndpoint::Consumer));stages.push_back(phase(n::OwnerAction::DrainSourceFailure));stages.push_back(phase(n::OwnerAction::ApplySourceDelivery));stages.push_back(phase(n::OwnerAction::ValidateFills));stages.push_back(phase(n::OwnerAction::DrainSourceFailure));
    auto prior=sentVictims;for(size_t i=0;i<stages.size();++i)if(stages[i]){graph.addNode(names[i],std::move(stages[i]),device);graph.addDependency(names[i],prior);prior=names[i];}
    if(demand)*demand={PooledImmutableDemandRole::InlineOwner,names[2],names[edge->hasPayload()?3:5]};
    return {names.front(),prior};
}
PooledPinParticipantNodes appendPooledSourceCoordinatorFill(ComputeGraph& graph,BufferArena& arena,
    std::span<const std::shared_ptr<const q::PooledImmutableCapturedSourceWave>> owners,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,DeviceGenerationControlBinding control,
    const std::string& completeVictims,const std::string& prefix)
{
    if(owners.empty() || !owners.front())throw std::invalid_argument("Source coordinator requires original full remote cohort");
    std::vector<q::PooledImmutableSourceParticipant> members;for(const auto& owner:owners)members.emplace_back(owner);
    return appendPooledSourceCoordinatorMembers(graph,arena,members,owners.front()->boundary(),std::move(code),control,completeVictims,prefix).coordinator;
}
PooledImmutableSourceFillCompletion appendPooledSourceCoordinatorMembers(ComputeGraph& graph,BufferArena& arena,
    std::span<const q::PooledImmutableSourceParticipant> owners,std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code,DeviceGenerationControlBinding control,
    const std::string& completeVictims,const std::string& prefix,const PooledImmutableMixedVictimCompletion* mixed)
{
    if(isGraphCaptureActive() || owners.empty() || !boundary || !boundary->isPooled() || !code || prefix.empty())throw std::invalid_argument("Source coordinator requires original cold complete participant/source/code");
    const auto storage=boundary->pooledStorage();const auto& plan=storage->coordinator()->plan();const auto& sp=storage->geometry().parameters();const auto device=boundary->device();
    if(owners.size()!=plan.parameters().policy.participants() || code->backend()!=device.type)throw std::invalid_argument("Source coordinator changed original full cohort/backend");
    control.requireFor(device);PooledImmutableSourceFillCompletion result;result.owners.assign(owners.begin(),owners.end());result.boundary=boundary;result.control=control;result.predecessor=completeVictims;
    const auto full=[&](const std::string& name){return prefix+"/"+name;};result.planned=full("plan_source");result.sealed=full("seal_coordinator");result.published=full("publish_request");result.acquired=full("acquire_response");result.retired=full("retire_response");
    for(auto* names:{&result.initial,&result.sealReceived,&result.sealAcquired,&result.packed,&result.fillSent,&result.payloadSent,&result.terminalSent})names->resize(owners.size());
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> remotePrevious;std::optional<size_t> local;
    for(size_t index=0;index<owners.size();++index){q::requirePooledSourceParticipant(owners[index],boundary);const auto owner=q::pooledSourceOwner(owners[index]);if(owner->plan().parameters().owner.coordinate()!=index)throw std::invalid_argument("Source coordinator reordered original members");const auto base="owner"+std::to_string(index)+"/";
        result.sealAcquired[index]=full(base+"acquire_seal");result.packed[index]=full(base+"pack");
        if(const auto* captured=std::get_if<std::shared_ptr<const q::PooledImmutableCapturedSourceWave>>(&owners[index])){remotePrevious.push_back((*captured)->previous());result.initial[index]=full(base+"initial_send");result.sealReceived[index]=full(base+"seal_receipt_receive");result.terminalSent[index]=full(base+"terminal_send");if((*captured)->hasPayload()){result.fillSent[index]=full(base+"fill_send");result.payloadSent[index]=full(base+"payload_send");}}
        else{if(local)throw std::invalid_argument("Source program repeats original local owner");local=index;result.localSeal=full(base+"seal_source");result.localDrain=full(base+"drain_source_failure");result.localApply=full(base+"apply_source");result.localValidate=full(base+"validate_fills");result.localOutcome=full(base+"fill_outcome");}
    }
    if(local){
        if(!mixed || mixed->completed!=completeVictims || mixed->owners.size()!=owners.size() || sp.local_owner!=*local)throw std::invalid_argument("Mixed source requires actual original completed victim descriptor");
        requirePooledMixedVictimCompletion(graph,*mixed);result.previous=*mixed;
        for(size_t index=0;index<owners.size();++index){const auto* localOriginal=std::get_if<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(&mixed->owners[index]);
            if(index==*local){const auto* actual=std::get_if<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(&owners[index]);if(!localOriginal || !actual || *actual!=*localOriginal)throw std::invalid_argument("Source substituted actual local victim relation");}
            else{const auto* previous=std::get_if<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>>(&mixed->owners[index]);const auto* actual=std::get_if<std::shared_ptr<const q::PooledImmutableCapturedSourceWave>>(&owners[index]);if(!previous || !actual || (*actual)->previous()!=*previous)throw std::invalid_argument("Source substituted actual remote victim owner");}}
    }else{if(mixed || sp.local_owner!=q::adaptive_lookup::None)throw std::invalid_argument("Source local membership has no original local producer");requirePooledVictimCoordinatorCompletion(graph,remotePrevious,completeVictims);}
    struct Node{std::string name;std::unique_ptr<IComputeStage> stage;};std::vector<Node> nodes;
    const auto source=[&](const std::string& name,q::PooledSourceAction action,uint32_t owner=q::adaptive_lookup::None){nodes.push_back({name,std::make_unique<PooledImmutableSourcePhaseStage>(arena,boundary,action,control,std::nullopt,owner)});};
    const auto transfer=[&](size_t index,const std::string& name,q::PooledSourceWireRole role){const auto& edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedSourceWave>>(owners[index]);nodes.push_back({name,std::make_unique<PooledSourceTransferStage>(arena,edge,role,role==q::PooledSourceWireRole::SealReceipt?CapturedTransferEndpoint::Consumer:CapturedTransferEndpoint::Producer)});};
    const auto localPhase=[&](const std::string& name,n::OwnerAction action){const auto owner=q::pooledSourceOwner(owners[*local]);const auto condition=action==n::OwnerAction::SealSource || action==n::OwnerAction::DrainSourceFailure?n::SourceCondition::Always:n::SourceCondition::AfterSuccess;nodes.push_back({name,std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,owner,q::PooledImmutableOwnerOperands(action,owner->plan(),q::pooledSourceOwnerInputs(owners[*local],boundary,action),condition)})});};
    source(result.planned,q::PooledSourceAction::Plan);q::PooledNativeOperandSpans seal{};seal[static_cast<size_t>(q::PooledNativeOperand::Source)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,sp.offsets.frontier,sizeof(n::SourceFrontier)};
    nodes.push_back({result.sealed,std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::SealSource,storage->coordinator(),q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::SealSource,plan,{},seal)})});source(result.published,q::PooledSourceAction::Publish);source(result.acquired,q::PooledSourceAction::AcquireResponse);source(result.retired,q::PooledSourceAction::RetireResponse);
    for(size_t index=0;index<owners.size();++index){if(local && index==*local)localPhase(result.localSeal,n::OwnerAction::SealSource);else{transfer(index,result.initial[index],q::PooledSourceWireRole::InitialFrontier);transfer(index,result.sealReceived[index],q::PooledSourceWireRole::SealReceipt);transfer(index,result.terminalSent[index],q::PooledSourceWireRole::TerminalFrontier);}source(result.sealAcquired[index],q::PooledSourceAction::AcquireSeal,index);source(result.packed[index],q::PooledSourceAction::PackOwner,index);if(!result.fillSent[index].empty()){transfer(index,result.fillSent[index],q::PooledSourceWireRole::Fills);transfer(index,result.payloadSent[index],q::PooledSourceWireRole::Payload);}}
    if(local){localPhase(result.localDrain,n::OwnerAction::DrainSourceFailure);localPhase(result.localApply,n::OwnerAction::ApplySourceDelivery);localPhase(result.localValidate,n::OwnerAction::ValidateFills);localPhase(result.localOutcome,n::OwnerAction::DrainSourceFailure);}
    std::vector<std::string> names;for(const auto& node:nodes)names.push_back(node.name);sourceNames(graph,names);
    // All original membership, predecessor and stage/operand checks precede graph mutation.
    for(auto& node:nodes)graph.addNode(node.name,std::move(node.stage),device);graph.addDependency(result.planned,completeVictims);graph.addDependency(result.sealed,result.planned);auto prior=result.sealed;
    for(const auto& sent:result.initial)if(!sent.empty()){graph.addDependency(sent,prior);prior=sent;}
    for(size_t index=0;index<owners.size();++index){if(local && index==*local){graph.addDependency(result.localSeal,prior);graph.addDependency(result.sealAcquired[index],result.localSeal);}else{graph.addDependency(result.sealReceived[index],prior);graph.addDependency(result.sealAcquired[index],result.sealReceived[index]);}prior=result.sealAcquired[index];}
    graph.addDependency(result.published,prior);graph.addDependency(result.acquired,result.published);prior=result.acquired;
    // Packaging order changes no original Grant/coordinate order. The final local packet has no later writer.
    for(size_t ordinal=0;ordinal<owners.size();++ordinal){const auto index=local?(ordinal==owners.size()-1?*local:ordinal<*local?ordinal:ordinal+1):ordinal;graph.addDependency(result.packed[index],prior);prior=result.packed[index];if(!result.fillSent[index].empty()){graph.addDependency(result.fillSent[index],prior);graph.addDependency(result.payloadSent[index],result.fillSent[index]);prior=result.payloadSent[index];}}
    graph.addDependency(result.retired,prior);prior=result.retired;for(const auto& sent:result.terminalSent)if(!sent.empty()){graph.addDependency(sent,prior);prior=sent;}
    if(local){graph.addDependency(result.localDrain,prior);graph.addDependency(result.localApply,result.localDrain);graph.addDependency(result.localValidate,result.localApply);graph.addDependency(result.localOutcome,result.localValidate);prior=result.localOutcome;}
    result.coordinator={result.planned,prior};return result;
}

}

namespace llaminar2 {
/** @brief Inspect only the exact original endpoint stage types without another object/vtable allocation. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledSourceTransfer(const IComputeStage& stage){
    if(const auto* original=dynamic_cast<const PooledSourceTransferStage*>(&stage))return original->directedTransferOperation();
    return std::nullopt;
}
}

namespace llaminar2 {
void bindPooledInlineDemandConsumption(ComputeGraph& graph,const PooledImmutableDemandFrontier& frontier,
    const std::vector<std::string>& trunk){
    if(frontier.role!=PooledImmutableDemandRole::InlineOwner)throw std::invalid_argument("Only an actual inline owner may bind its response join");
    auto* node=graph.getNode(frontier.consume);
    auto* stage=node?dynamic_cast<PooledSourceTransferStage*>(node->stage.get()):nullptr;
    if(!stage)throw std::invalid_argument("Inline demand join lost its original response endpoint");
    stage->bindDemandJoin(graph,*node,frontier.published,trunk);
    for(const auto& name:trunk)graph.addDependency(frontier.consume,name);
}
}

namespace llaminar2 {
bool isPooledPureReaderDemandHeader(const ComputeNode& node,const RetainedGraphParticipantDefinitionDomain& domain);
void validatePooledDemandFrontier(const ComputeGraph& graph,const PooledImmutableDemandFrontier& f){
    if(isGraphCaptureActive())throw std::logic_error("Pooled demand handoff must seal before native graph capture");
    if(f.role!=PooledImmutableDemandRole::Source && f.role!=PooledImmutableDemandRole::InlineOwner && f.role!=PooledImmutableDemandRole::PureReader)
        throw std::invalid_argument("Pooled demand has an unknown original role");
    const auto* consume=graph.getNode(f.consume);
    if(!consume || !consume->stage)throw std::invalid_argument("Pooled demand lost its actual delayed endpoint");
    if(f.role==PooledImmutableDemandRole::PureReader){
        if(!f.published.empty() || !graph.retainedParticipantDefinitionDomain() || !isPooledPureReaderDemandHeader(*consume,*graph.retainedParticipantDefinitionDomain()))
            throw std::invalid_argument("Pure reader demand cannot invent source publication or another receive role");
        return;
    }
    const auto* publish=graph.getNode(f.published);
    if(!publish || publish->device!=consume->device ||
       std::count(consume->dependencies.begin(),consume->dependencies.end(),f.published)!=1)
        throw std::invalid_argument("Pooled demand lost its original immediate protocol edge");
    if(f.role==PooledImmutableDemandRole::Source){
        const auto boundary=std::dynamic_pointer_cast<q::ImmutableLookupExecutionBoundary>(graph.retainedStorageBoundary());
        const auto* issued=dynamic_cast<const PooledImmutableSourcePhaseStage*>(publish->stage.get());
        const auto* delayed=dynamic_cast<const PooledImmutableSourcePhaseStage*>(consume->stage.get());
        if(!boundary || !issued || !delayed || !issued->matchesDemand(boundary.get(),q::PooledSourceAction::Publish) ||
           !delayed->matchesDemand(boundary.get(),q::PooledSourceAction::AcquireResponse))
            throw std::invalid_argument("Source demand substituted its exact Publish/AcquireResponse authority");
    }else{
        const auto* issued=dynamic_cast<const PooledSourceTransferStage*>(publish->stage.get());
        const auto* delayed=dynamic_cast<const PooledSourceTransferStage*>(consume->stage.get());
        if(!issued || !delayed)throw std::invalid_argument("Inline demand substituted its addressed transfer endpoints");
        // bindDemandJoin further authenticates the same original edge and exact first response role.
    }
}
void bindPooledDemandConsumption(ComputeGraph& graph,const PooledImmutableDemandFrontier& frontier,
    const std::vector<std::string>& trunk){
    validatePooledDemandFrontier(graph,frontier);
    if(frontier.role==PooledImmutableDemandRole::InlineOwner)bindPooledInlineDemandConsumption(graph,frontier,trunk);
    else {
        if(frontier.role==PooledImmutableDemandRole::Source){
            auto* source=dynamic_cast<PooledImmutableSourcePhaseStage*>(graph.getNode(frontier.consume)->stage.get());
            source->bindCurrentDemandTrunk(graph,*graph.getNode(frontier.consume),frontier.published,trunk);
        }
        for(const auto& name:trunk)graph.addDependency(frontier.consume,name);
    }
}
}

namespace llaminar2 {
void PooledImmutableSourcePhaseStage::bindCurrentDemandTrunk(const ComputeGraph& graph,const ComputeNode& node,
    const std::string& published,const std::vector<std::string>& trunk){
    if(!demand_trunk_.empty() || action_!=qwen4exp::PooledSourceAction::AcquireResponse ||
       graph.retainedStorageBoundary().get()!=boundary_.get() || node.stage.get()!=this || node.device!=device() ||
       node.dependencies!=std::vector<std::string>{published} || trunk.empty())
        throw std::invalid_argument("Source demand join lost its original response/Publish authority");
    const auto* before=graph.getNode(published);
    const auto* source=before?dynamic_cast<const PooledImmutableSourcePhaseStage*>(before->stage.get()):nullptr;
    if(!source || !source->matchesDemand(boundary_.get(),qwen4exp::PooledSourceAction::Publish))
        throw std::invalid_argument("Source demand join substituted its actual Publish operation");
    for(const auto& name:trunk)if(name==published || !graph.getNode(name) || graph.getNode(name)->device!=device())
        throw std::invalid_argument("Source demand join has no actual same-participant trunk producer");
    demand_trunk_=trunk;
}
}
