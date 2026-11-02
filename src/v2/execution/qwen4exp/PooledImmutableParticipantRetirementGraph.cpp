/**
 * @file PooledImmutableParticipantRetirementGraph.cpp
 * @brief Cold typed terminal bindings and participant-local graph assembly over original PMA roots.
 *
 * The source serial bank is reused only after all last-reader gate sends retire;
 * releases use original UInt64 byte producers and separate stable-sort storage.
 * Invalid metadata or a missing original consumer edge fails before graph mutation.
 */
#include "PooledImmutableParticipantRetirementGraph.h"
#include "PooledImmutableDirectedOperation.h"
#include "PooledImmutableDirectedMessageGeometry.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableSourcePhaseStage.h"
#include "kernels/qwen4exp/PooledImmutableRetirementOps.h"
#include "transfer/TransferEngine.h"
#include "memory/BufferArena.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <stdexcept>
#include <set>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;
/** @return Checked original role index. */
size_t index(PooledRetirementWireRole role){const auto i=static_cast<unsigned>(role);if(i>=3)throw std::invalid_argument("Unknown retirement role");return i;}
/** @return Original reverse channel, never a dynamically selected alternate transport. */
bool reverse(PooledRetirementWireRole role){return role!=PooledRetirementWireRole::ReadersRetired;}
/** @return Exact original retained physical root for one declared span. */
std::shared_ptr<TensorBase> root(const PooledImmutableNativeRoots& roots,BufferId id){return id==BufferId::QFN_POOLED_OWNER_FRAME || id==BufferId::QFN_POOLED_SOURCE_FRAME?roots.frame:roots.native;}
}
PooledPinWireSpan PooledImmutableCapturedRetirementWave::source(PooledRetirementWireRole role)const{
    (void)index(role);const auto source=read_->publication()->sourceWave();const auto& s=source->boundary()->pooledStorage()->geometry().parameters();const auto& o=pooledSourceOwner(source)->plan().parameters();
    switch(role){
    case PooledRetirementWireRole::ReadersRetired:return {BufferId::QFN_POOLED_SOURCE_NATIVE,s.offsets.seal_receipt,sizeof(n::ReadersRetiredGate)};
    case PooledRetirementWireRole::Releases:return {BufferId::QFN_POOLED_OWNER_FRAME,o.frame.release_receipts,PooledImmutableDirectedMessageGeometry::fromPlan(pooledSourceOwner(source)->plan()).releases()};
    case PooledRetirementWireRole::Retired:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,n::NominalHeaderWireBytes};
    }throw std::invalid_argument("Unknown retirement source");
}
PooledPinWireSpan PooledImmutableCapturedRetirementWave::destination(PooledRetirementWireRole role)const{
    (void)index(role);const auto source=read_->publication()->sourceWave();const auto& s=source->boundary()->pooledStorage()->geometry().parameters();const auto& o=pooledSourceOwner(source)->plan().parameters();
    switch(role){
    case PooledRetirementWireRole::ReadersRetired:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(n::ReadersRetiredGate)};
    case PooledRetirementWireRole::Releases:return {BufferId::QFN_POOLED_SOURCE_FRAME,s.offsets.fills,PooledImmutableDirectedMessageGeometry::fromPlan(pooledSourceOwner(source)->plan()).releases()};
    case PooledRetirementWireRole::Retired:return {BufferId::QFN_POOLED_SOURCE_NATIVE,s.offsets.seal_receipt,n::NominalHeaderWireBytes};
    }throw std::invalid_argument("Unknown retirement receiver");
}
std::shared_ptr<const PooledImmutableCapturedRetirementWave> PooledImmutableCapturedRetirementWave::bind(
    std::shared_ptr<const PooledImmutableCapturedReadWave> read,std::shared_ptr<PhysicalMemoryAuthority> memory){
    if(isGraphCaptureActive() || !read || !memory)throw std::invalid_argument("Retirement needs original cold read/PMA");read->requireCurrent();
    const auto source=read->publication()->sourceWave();const auto pin=source->previous()->previous()->pin();const auto forward=pin->forward();
    forward->ownerStorage()->requireAuthority(memory);forward->coordinatorStorage()->requireAuthority(memory);
    if(!forward->channel_ || !pin->channel_ || !forward->declaration().capturedCrossVendor())throw std::invalid_argument("Retirement native/rank transport needs complete qualified factory");
    auto out=std::shared_ptr<PooledImmutableCapturedRetirementWave>(new PooledImmutableCapturedRetirementWave(read,memory));
    const auto own=forward->ownerStorage()->roots(),src=source->boundary()->pooledStorage()->roots();const auto& o=forward->ownerStorage()->plan().parameters();const auto& s=source->boundary()->pooledStorage()->geometry().parameters();auto& transfer=TransferEngine::instance();
    for(unsigned i=0;i<3;++i){const auto role=static_cast<PooledRetirementWireRole>(i);if(role==PooledRetirementWireRole::Releases && !o.native.reply_capacity)continue;
        const auto from=out->source(role),to=out->destination(role);const bool back=reverse(role);const auto key=((forward->declaration().identity()^(160+i))*1099511628211ULL)|1;
        auto send=transfer.bindCapturedTransfer(back?pin->channel_:forward->channel_,CapturedTransferEndpoint::Producer,{key,from.bytes},root(back?own:src,from.buffer),from.offset);
        auto receive=transfer.bindCapturedTransfer(back?pin->channel_:forward->channel_,CapturedTransferEndpoint::Consumer,{key,from.bytes},root(back?src:own,to.buffer),to.offset);
        if(role==PooledRetirementWireRole::Releases){send=transfer.bindDeviceCountedTransfer(std::move(send),own.native,o.native.progress+offsetof(n::OwnerProgress,release_bytes));receive=transfer.bindDeviceCountedTransfer(std::move(receive),src.native,s.offsets.record_bytes);}
        out->bindings_[i].emplace(std::move(send),std::move(receive));
    }out->requireCurrent();return out;
}
void PooledImmutableCapturedRetirementWave::requireCurrent()const{read_->requireCurrent();const auto source=read_->publication()->sourceWave();pooledSourceOwner(source)->requireAuthority(memory_);source->boundary()->pooledStorage()->coordinator()->requireAuthority(memory_);source->boundary()->pooledStorage()->captureIdentity();}
const CapturedTransferBinding& PooledImmutableCapturedRetirementWave::binding(PooledRetirementWireRole role,CapturedTransferEndpoint side)const{requireCurrent();const auto& pair=bindings_[index(role)];if(!pair || (side!=CapturedTransferEndpoint::Producer && side!=CapturedTransferEndpoint::Consumer))throw std::invalid_argument("Retirement missing/C0 endpoint");return side==CapturedTransferEndpoint::Producer?pair->first:pair->second;}
PooledNativeOperandSpans PooledImmutableCapturedRetirementWave::ownerInputs(n::OwnerAction action)const{
    requireCurrent();const auto source=read_->publication()->sourceWave();const auto owner=pooledSourceOwner(source);const auto& o=owner->plan().parameters();
    auto out=source->previous()->previous()->pin()->forward()->ownerInputs(n::OwnerAction::Begin);
    if(action==n::OwnerAction::AcquireReaderRetirement)out[static_cast<size_t>(PooledNativeOperand::Frontier)]=PooledNativeTensorSpan{owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(n::ReadersRetiredGate)};
    else if(action==n::OwnerAction::DrainSourceFailure)out[static_cast<size_t>(PooledNativeOperand::Frontier)]=PooledNativeTensorSpan{owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,o.native.source_outcome,sizeof(n::SourceFrontier)};
    else if(action!=n::OwnerAction::Release && action!=n::OwnerAction::Retire && action!=n::OwnerAction::RetiredWire)throw std::invalid_argument("Retirement changed original owner phase");
    return out;
}
uint64_t PooledImmutableCapturedRetirementWave::captureIdentity()const{requireCurrent();auto h=read_->captureIdentity();for(size_t i=0;i<bindings_.size();++i){h=(h^i)*1099511628211ULL;if(bindings_[i]){h=(h^bindings_[i]->first.captureIdentity())*1099511628211ULL;h=(h^bindings_[i]->second.captureIdentity())*1099511628211ULL;}}return h?h:1;}
}
namespace llaminar2 {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;
/** @return Only the actual original owner endpoint for this fixed directed role. */
bool ownerEndpoint(q::PooledRetirementWireRole role,CapturedTransferEndpoint side){return (role!=q::PooledRetirementWireRole::ReadersRetired)==(side==CapturedTransferEndpoint::Producer);}
/** @brief One actual original public transfer leaf; it owns no source/row policy or alternate transport. */
class PooledRetirementTransferStage final:public IComputeStage {
public:
    /** @return Exact original semantic endpoint and native dispatch inputs before capture. */
    q::PooledImmutableDirectedTransferOperation directedTransferOperation()const{
        requireDeclarations();const auto& binding=edge_->binding(role_,side_);
        return {q::PooledImmutableDirectedWireFamily::Retirement,static_cast<std::uint32_t>(role_),side_,
            binding.message().bytes,binding.extentSource(),binding.hasSourceSlice()};
    }

    /** @brief Retain the exact owner/arena/role and freeze the original transfer identity. */
    PooledRetirementTransferStage(BufferArena& arena,std::shared_ptr<const q::PooledImmutableCapturedRetirementWave> edge,q::PooledRetirementWireRole role,CapturedTransferEndpoint side)
        :IComputeStage(ownerEndpoint(role,side)?q::pooledSourceOwner(edge->read()->publication()->sourceWave())->plan().allocator().device:edge->read()->publication()->sourceWave()->boundary()->device()),arena_(&arena),edge_(std::move(edge)),role_(role),side_(side){requireDeclarations();identity_=graphCaptureVariantSignature();}
    /** @brief Acquire only exact original setup events on the participant's nondefault stream. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override{
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<uintptr_t>(stream)<=2)throw std::invalid_argument("Retirement requires exact participant/nondefault stream");
        requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Retirement capture drift");
        const auto source=edge_->read()->publication()->sourceWave();
        if(ownerEndpoint(role_,side_))q::pooledSourceOwner(source)->acquireSetup(stream);else source->boundary()->pooledStorage()->acquireSetup(stream);
        setGPUStream(stream);return true;
    }
    /** @brief Enqueue the exact fixed/counted transfer under original event-aware publication. */
    bool execute(IDeviceContext* context)override{if(!context || context->deviceId()!=device())throw std::invalid_argument("Retirement changed participant");requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Retirement binding changed");TransferEngine::instance().enqueueCapturedTransfer(edge_->binding(role_,side_),requireGPUStream());return true;}
    /** @return Explicit participant-local graph collective role. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER;}
    /** @return This original bound endpoint executes native device-counted transfer kernels, independently from collective-runtime capture. */
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority()const noexcept override{return CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer;}
    /** @return Exact diagnostic leaf family. */
    std::string name()const override{return "PooledImmutableDirectedRetirement";}
    /** @return Setup waits belong only to retained cold capture. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Actual live byte producers serve partial/empty prefill without padding. */
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    /** @return Only the original declared native participant backend. */
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    /** @return No additional physical storage; original PMA roots remain sole owners. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Full original root/extent/count/role/endpoint identity, never a live count value. */
    uint64_t graphCaptureVariantSignature()const override{requireDeclarations();return ((((edge_->captureIdentity()^(160+static_cast<unsigned>(role_)))*1099511628211ULL)^static_cast<unsigned>(side_))*1099511628211ULL)|1;}
    /** @return Concrete original holder/role/side for mutation-free terminal-chain authentication. */
    bool matches(const q::PooledImmutableCapturedRetirementWave* edge,q::PooledRetirementWireRole role,CapturedTransferEndpoint side)const noexcept{return edge_.get()==edge && role_==role && side_==side;}
    /** @return Exact original input or receive-writer roots, with no duplicate arena claim. */
    StageBufferContract bufferContract()const override{
        requireDeclarations();auto c=StageBufferContract::build();const auto span=side_==CapturedTransferEndpoint::Producer?edge_->source(role_):edge_->destination(role_);
        if(side_==CapturedTransferEndpoint::Producer)c.addInput(span.buffer,"INT8");else c.addPreallocatedInOut(span.buffer,"INT8");
        if(role_==q::PooledRetirementWireRole::Releases){if(side_==CapturedTransferEndpoint::Producer)c.addInput(BufferId::QFN_POOLED_OWNER_NATIVE,"INT8");else c.addPreallocatedInOut(BufferId::QFN_POOLED_SOURCE_NATIVE,"INT8");}
        return c;
    }
    /** @return Only static role/endpoint diagnostics. */
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo d;d.addScalarInt("retirement_role",static_cast<unsigned>(role_));d.addScalarInt("retirement_endpoint",static_cast<unsigned>(side_));return d;}
private:
    /** @brief Reauthenticate exact original root aliases before every record/submit/identity query. */
    void requireDeclarations()const{
        if(!arena_ || !edge_)throw std::logic_error("Retirement lost original owner");edge_->requireCurrent();(void)edge_->binding(role_,side_);
        const auto span=side_==CapturedTransferEndpoint::Producer?edge_->source(role_):edge_->destination(role_);const auto source=edge_->read()->publication()->sourceWave();
        const auto roots=ownerEndpoint(role_,side_)?q::pooledSourceOwner(source)->roots():source->boundary()->pooledStorage()->roots();
        const auto original=span.buffer==BufferId::QFN_POOLED_OWNER_FRAME || span.buffer==BufferId::QFN_POOLED_SOURCE_FRAME?roots.frame:roots.native;
        if(arena_->getTensor(span.buffer)!=original.get())throw std::logic_error("Retirement replaced original payload/metadata root");
        if(role_==q::PooledRetirementWireRole::Releases){const auto role=side_==CapturedTransferEndpoint::Producer?BufferId::QFN_POOLED_OWNER_NATIVE:BufferId::QFN_POOLED_SOURCE_NATIVE;if(arena_->getTensor(role)!=roots.native.get())throw std::logic_error("Retirement replaced original UInt64 extent root");}
    }
    BufferArena* arena_;std::shared_ptr<const q::PooledImmutableCapturedRetirementWave> edge_;q::PooledRetirementWireRole role_;CapturedTransferEndpoint side_;uint64_t identity_=0;
};
/** @brief Reject every duplicate/existing target name before any graph node is added. */
void freeNames(const ComputeGraph& graph,const std::vector<std::string>& names){std::set<std::string> unique;for(const auto& name:names)if(name.empty() || graph.getNode(name) || !unique.insert(name).second)throw std::invalid_argument("Retirement changed original unique node names");}
/** @return Actual local binding or original remote read holder at its original coordinate. */
q::PooledImmutableReadParticipant readMember(const q::PooledImmutableRetirementParticipant& member){return std::visit([](const auto& value)->q::PooledImmutableReadParticipant{if(!value)throw std::invalid_argument("Retirement lost actual member");using T=std::decay_t<decltype(*value)>;if constexpr(std::is_same_v<T,q::PooledImmutableLocalOwnerBinding>)return value;else return value->read();},member);}
}
PooledPinParticipantNodes appendPooledRetirementOwner(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const q::PooledImmutableCapturedRetirementWave> original,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& lastReadSent,const std::string& prefix){
    if(isGraphCaptureActive() || !original || !code || prefix.empty())throw std::invalid_argument("Retirement owner needs original cold read/code");original->requireCurrent();
    requirePooledReadOwnerCompletion(graph,original->read(),lastReadSent);const auto owner=q::pooledSourceOwner(original->read()->publication()->sourceWave());const auto device=owner->plan().allocator().device;
    if(code->backend()!=device.type)throw std::invalid_argument("Retirement owner changed backend code");const bool positive=owner->plan().parameters().native.reply_capacity!=0;
    const std::array suffixes{"reader_gate_receive","acquire_reader_gate","drain","release","release_send","retire","retired_wire","retired_send"};std::vector<std::string> names;for(auto suffix:suffixes)names.push_back(prefix+"/"+suffix);freeNames(graph,names);
    std::vector<std::unique_ptr<IComputeStage>> stages;
    const auto wire=[&](q::PooledRetirementWireRole role,CapturedTransferEndpoint side){return std::make_unique<PooledRetirementTransferStage>(arena,original,role,side);};
    const auto phase=[&](n::OwnerAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,owner,q::PooledImmutableOwnerOperands(action,owner->plan(),original->ownerInputs(action))});};
    stages.push_back(wire(q::PooledRetirementWireRole::ReadersRetired,CapturedTransferEndpoint::Consumer));stages.push_back(phase(n::OwnerAction::AcquireReaderRetirement));stages.push_back(phase(n::OwnerAction::DrainSourceFailure));stages.push_back(phase(n::OwnerAction::Release));stages.push_back(positive?wire(q::PooledRetirementWireRole::Releases,CapturedTransferEndpoint::Producer):nullptr);stages.push_back(phase(n::OwnerAction::Retire));stages.push_back(phase(n::OwnerAction::RetiredWire));stages.push_back(wire(q::PooledRetirementWireRole::Retired,CapturedTransferEndpoint::Producer));
    auto prior=lastReadSent;for(size_t i=0;i<stages.size();++i)if(stages[i]){graph.addNode(names[i],std::move(stages[i]),device);graph.addDependency(names[i],prior);prior=names[i];}
    return {names.front(),prior};
}
PooledImmutableRetirementCompletion appendPooledRetirementCoordinator(ComputeGraph& graph,BufferArena& arena,
    const PooledImmutableReadCompletion& read,std::span<const q::PooledImmutableRetirementParticipant> owners,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& prefix){
    if(isGraphCaptureActive() || !code || prefix.empty())throw std::invalid_argument("Retirement coordinator requires original cold read/code");
    requirePooledReadCompletion(graph,read);if(!read.publication.source.control)throw std::invalid_argument("Retirement lost original canonical control");const auto boundary=read.publication.source.boundary;const auto storage=boundary->pooledStorage();const auto& p=storage->geometry().parameters();const auto device=boundary->device();
    if(owners.size()!=p.coordinator.policy.participants() || owners.size()!=read.owners.size() || code->backend()!=device.type ||
       (owners.size()==1 && p.local_owner!=q::adaptive_lookup::None))throw std::invalid_argument("Retirement changed original cohort/local-P1 path/backend");
    std::shared_ptr<const q::PooledImmutableLocalOwnerBinding> local;
    for(size_t i=0;i<owners.size();++i){if(readMember(owners[i])!=read.owners[i])throw std::invalid_argument("Retirement replaced original ordered read participant");
        if(const auto* value=std::get_if<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(&owners[i])){if(!*value || local || (*value)->coordinate()!=i || p.local_owner!=i || (*value)->coordinatorStorage()!=storage->coordinator())throw std::invalid_argument("Retirement changed actual local member/root");(*value)->requireCurrent();local=*value;}
        else {const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(owners[i]);if(!edge)throw std::invalid_argument("Retirement lost remote holder");edge->requireCurrent();if(q::pooledSourceOwner(edge->read()->publication()->sourceWave())->plan().parameters().owner.coordinate()!=i || edge->read()->publication()->sourceWave()->boundary()!=boundary)throw std::invalid_argument("Retirement changed original remote coordinate/source");}
    }
    if(bool(local)!=(p.local_owner!=q::adaptive_lookup::None))throw std::invalid_argument("Retirement lost actual local owner");
    PooledImmutableRetirementCompletion out;out.read=read;out.owners.assign(owners.begin(),owners.end());
    out.readersRetired=prefix+"/readers_retired";out.planned=prefix+"/plan_releases";out.reconciled=prefix+"/reconcile_releases";out.snapshot=prefix+"/snapshot";out.finished=prefix+"/finish";out.observed=prefix+"/observe";
    const auto count=owners.size();out.gatePacked.resize(count);out.gateSent.resize(count);out.releaseReceived.resize(count);out.releaseImported.resize(count);out.retiredReceived.resize(count);out.retiredImported.resize(count);
    struct Node {std::string name;std::unique_ptr<IComputeStage> stage;};std::vector<Node> nodes;
    const auto source=[&](const std::string& name,q::PooledSourceAction action,uint32_t owner=q::adaptive_lookup::None){nodes.push_back({name,std::make_unique<PooledImmutableSourcePhaseStage>(arena,boundary,action,*read.publication.source.control,std::nullopt,owner)});};
    const auto wire=[&](const std::string& name,size_t i,q::PooledRetirementWireRole role){nodes.push_back({name,std::make_unique<PooledRetirementTransferStage>(arena,std::get<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(owners[i]),role,role==q::PooledRetirementWireRole::ReadersRetired?CapturedTransferEndpoint::Producer:CapturedTransferEndpoint::Consumer)});};
    const auto localPhase=[&](const std::string& name,n::OwnerAction action){auto spans=local->ownerInputs(n::OwnerAction::Begin);
        if(action==n::OwnerAction::DrainSourceFailure)spans[static_cast<size_t>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,p.offsets.frontier,sizeof(n::SourceFrontier)};
        nodes.push_back({name,std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,local->ownerStorage(),q::PooledImmutableOwnerOperands(action,local->ownerStorage()->plan(),spans)})});};
    source(out.readersRetired,q::PooledSourceAction::ReadersRetired);source(out.planned,q::PooledSourceAction::PlanReleases);
    for(size_t i=0;i<count;++i){const auto stem=prefix+"/owner"+std::to_string(i);const bool direct=i==p.local_owner;
        if(!direct){out.gatePacked[i]=stem+"/pack_reader_gate";out.gateSent[i]=stem+"/reader_gate_send";source(out.gatePacked[i],q::PooledSourceAction::PackReaderRetirement,i);wire(out.gateSent[i],i,q::PooledRetirementWireRole::ReadersRetired);
            const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(owners[i]);if(q::pooledSourceOwner(edge->read()->publication()->sourceWave())->plan().parameters().native.reply_capacity){out.releaseReceived[i]=stem+"/release_receive";wire(out.releaseReceived[i],i,q::PooledRetirementWireRole::Releases);}out.retiredReceived[i]=stem+"/retired_receive";wire(out.retiredReceived[i],i,q::PooledRetirementWireRole::Retired);
        }else{out.localDrain=stem+"/drain";out.localRelease=stem+"/release";out.localRetire=stem+"/retire";localPhase(out.localDrain,n::OwnerAction::DrainSourceFailure);localPhase(out.localRelease,n::OwnerAction::Release);localPhase(out.localRetire,n::OwnerAction::Retire);}
        out.releaseImported[i]=stem+"/release_import";out.retiredImported[i]=stem+"/retired_import";source(out.releaseImported[i],q::PooledSourceAction::ImportReleases,i);source(out.retiredImported[i],q::PooledSourceAction::ImportRetired,i);
    }
    source(out.reconciled,q::PooledSourceAction::ReconcileReleases);source(out.snapshot,q::PooledSourceAction::SnapshotReleased);source(out.finished,q::PooledSourceAction::FinishRetired);source(out.observed,q::PooledSourceAction::ObserveRetired);
    std::vector<std::string> names;for(const auto& node:nodes)names.push_back(node.name);freeNames(graph,names);
    for(auto& node:nodes)graph.addNode(node.name,std::move(node.stage),device);
    graph.addDependency(out.readersRetired,read.output);graph.addDependency(out.planned,out.readersRetired);
    auto gates=out.planned;for(size_t i=0;i<count;++i)if(!out.gatePacked[i].empty()){graph.addDependency(out.gatePacked[i],gates);graph.addDependency(out.gateSent[i],out.gatePacked[i]);gates=out.gateSent[i];}
    if(local){graph.addDependency(out.localDrain,out.planned);graph.addDependency(out.localRelease,out.localDrain);}
    auto prior=gates;
    for(size_t i=0;i<count;++i){if(!out.releaseReceived[i].empty()){graph.addDependency(out.releaseReceived[i],prior);graph.addDependency(out.releaseImported[i],out.releaseReceived[i]);}
        else graph.addDependency(out.releaseImported[i],prior);
        if(i==p.local_owner)graph.addDependency(out.releaseImported[i],out.localRelease);
        prior=out.releaseImported[i];
    }
    graph.addDependency(out.reconciled,prior);graph.addDependency(out.snapshot,out.reconciled);
    if(local)graph.addDependency(out.localRetire,out.releaseImported[p.local_owner]);
    prior=out.snapshot;
    for(size_t i=0;i<count;++i){if(!out.retiredReceived[i].empty()){graph.addDependency(out.retiredReceived[i],prior);graph.addDependency(out.retiredImported[i],out.retiredReceived[i]);}
        else{graph.addDependency(out.retiredImported[i],prior);graph.addDependency(out.retiredImported[i],out.localRetire);}
        prior=out.retiredImported[i];
    }
    graph.addDependency(out.finished,prior);graph.addDependency(out.observed,out.finished);out.coordinator={out.readersRetired,out.observed};return out;
}
void requirePooledRetirementOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>& original,const std::string& retiredSent){
    if(isGraphCaptureActive() || !original || retiredSent.empty())throw std::invalid_argument("Retirement owner completion needs original cold holder");original->requireCurrent();
    const auto owner=q::pooledSourceOwner(original->read()->publication()->sourceWave());const auto device=owner->plan().allocator().device;
    const auto prior=[&](const std::string& name){const auto* node=graph.getNode(name);if(!node || node->device!=device || node->dependencies.size()!=1)throw std::invalid_argument("Retirement owner lost exact original predecessor");return node->dependencies.front();};
    const auto endpoint=[&](const std::string& name,q::PooledRetirementWireRole role,CapturedTransferEndpoint side){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledRetirementTransferStage*>(node->stage.get()):nullptr;if(!stage || !stage->matches(original.get(),role,side) || node->device!=device)throw std::invalid_argument("Retirement owner replaced original endpoint");};
    const auto phase=[&](const std::string& name,n::OwnerAction action){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()):nullptr;const auto* value=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;const q::PooledImmutableOwnerOperands expected(action,owner->plan(),original->ownerInputs(action));if(!node || node->device!=device || !value || value->action!=action || value->storage!=owner || value->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Retirement owner replaced original physical phase/operands");};
    auto current=retiredSent;endpoint(current,q::PooledRetirementWireRole::Retired,CapturedTransferEndpoint::Producer);current=prior(current);
    phase(current,n::OwnerAction::RetiredWire);current=prior(current);phase(current,n::OwnerAction::Retire);current=prior(current);
    if(owner->plan().parameters().native.reply_capacity){endpoint(current,q::PooledRetirementWireRole::Releases,CapturedTransferEndpoint::Producer);current=prior(current);}
    phase(current,n::OwnerAction::Release);current=prior(current);phase(current,n::OwnerAction::DrainSourceFailure);current=prior(current);phase(current,n::OwnerAction::AcquireReaderRetirement);current=prior(current);
    endpoint(current,q::PooledRetirementWireRole::ReadersRetired,CapturedTransferEndpoint::Consumer);current=prior(current);requirePooledReadOwnerCompletion(graph,original->read(),current);
}
void requirePooledRetirementCompletion(const ComputeGraph& graph,const PooledImmutableRetirementCompletion& original){
    if(isGraphCaptureActive())throw std::invalid_argument("Retirement completion authentication is cold-only");requirePooledReadCompletion(graph,original.read);
    if(!original.read.publication.source.control)throw std::invalid_argument("Retirement completion lost original canonical control");
    const auto boundary=original.read.publication.source.boundary;const auto storage=boundary->pooledStorage();const auto& p=storage->geometry().parameters();const auto device=boundary->device();const auto count=original.owners.size();
    if(count!=p.coordinator.policy.participants() || count!=original.read.owners.size() || !count ||
       original.gatePacked.size()!=count || original.gateSent.size()!=count || original.releaseReceived.size()!=count || original.releaseImported.size()!=count || original.retiredReceived.size()!=count || original.retiredImported.size()!=count ||
       original.coordinator.first!=original.readersRetired || original.coordinator.last!=original.observed)throw std::invalid_argument("Retirement completion lost full original cohort/phase inventory");
    const auto node=[&](const std::string& name){const auto* value=graph.getNode(name);if(!value || value->device!=device)throw std::invalid_argument("Retirement lost original coordinator-local node");return value;};
    const auto source=[&](const std::string& name,q::PooledSourceAction action,uint32_t owner=q::adaptive_lookup::None){const auto* value=node(name);const auto* phase=dynamic_cast<const PooledImmutableSourcePhaseStage*>(value->stage.get());if(!phase || !phase->matchesOriginal(boundary.get(),action,owner,*original.read.publication.source.control))throw std::invalid_argument("Retirement replaced actual source/owner/control phase");};
    const auto endpoint=[&](const std::string& name,const std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>& holder,q::PooledRetirementWireRole role,CapturedTransferEndpoint side){const auto* phase=dynamic_cast<const PooledRetirementTransferStage*>(node(name)->stage.get());if(!phase || !phase->matches(holder.get(),role,side))throw std::invalid_argument("Retirement replaced original exact channel endpoint");};
    source(original.readersRetired,q::PooledSourceAction::ReadersRetired);source(original.planned,q::PooledSourceAction::PlanReleases);source(original.reconciled,q::PooledSourceAction::ReconcileReleases);source(original.snapshot,q::PooledSourceAction::SnapshotReleased);source(original.finished,q::PooledSourceAction::FinishRetired);source(original.observed,q::PooledSourceAction::ObserveRetired);
    requirePooledOriginalDependencies(graph,original.readersRetired,{original.read.output});requirePooledOriginalDependencies(graph,original.planned,{original.readersRetired});
    std::shared_ptr<const q::PooledImmutableLocalOwnerBinding> local;auto gates=original.planned;
    for(size_t i=0;i<count;++i){if(readMember(original.owners[i])!=original.read.owners[i])throw std::invalid_argument("Retirement substituted an original ordered read participant");
        if(i==p.local_owner){if(local || !std::holds_alternative<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(original.owners[i]) || !original.gatePacked[i].empty() || !original.gateSent[i].empty() || !original.releaseReceived[i].empty() || !original.retiredReceived[i].empty())throw std::invalid_argument("Retirement created a local self-transport or substituted coordinate");local=std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(original.owners[i]);if(!local || local->coordinate()!=i || local->coordinatorStorage()!=storage->coordinator())throw std::invalid_argument("Retirement changed original local roots");local->requireCurrent();}
        else{const auto* value=std::get_if<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(&original.owners[i]);if(!value || !*value)throw std::invalid_argument("Retirement lost remote holder");(*value)->requireCurrent();source(original.gatePacked[i],q::PooledSourceAction::PackReaderRetirement,i);endpoint(original.gateSent[i],*value,q::PooledRetirementWireRole::ReadersRetired,CapturedTransferEndpoint::Producer);requirePooledOriginalDependencies(graph,original.gatePacked[i],{gates});requirePooledOriginalDependencies(graph,original.gateSent[i],{original.gatePacked[i]});gates=original.gateSent[i];}
    }
    if(bool(local)!=(p.local_owner!=q::adaptive_lookup::None))throw std::invalid_argument("Retirement changed actual local membership");
    if(local){const auto phase=[&](const std::string& name,n::OwnerAction action){const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(name)->stage.get());const auto* value=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;auto spans=local->ownerInputs(n::OwnerAction::Begin);if(action==n::OwnerAction::DrainSourceFailure)spans[static_cast<size_t>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,p.offsets.frontier,sizeof(n::SourceFrontier)};const q::PooledImmutableOwnerOperands expected(action,local->ownerStorage()->plan(),spans);if(!value || value->storage!=local->ownerStorage() || value->action!=action || value->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Retirement replaced actual local physical producer");};phase(original.localDrain,n::OwnerAction::DrainSourceFailure);phase(original.localRelease,n::OwnerAction::Release);phase(original.localRetire,n::OwnerAction::Retire);requirePooledOriginalDependencies(graph,original.localDrain,{original.planned});requirePooledOriginalDependencies(graph,original.localRelease,{original.localDrain});requirePooledOriginalDependencies(graph,original.localRetire,{original.releaseImported[p.local_owner]});}
    else if(!original.localDrain.empty() || !original.localRelease.empty() || !original.localRetire.empty())throw std::invalid_argument("Retirement invented a local phase");
    auto prior=gates;
    for(size_t i=0;i<count;++i){source(original.releaseImported[i],q::PooledSourceAction::ImportReleases,i);std::vector<std::string> expected;
        if(i==p.local_owner){if(!original.releaseReceived[i].empty())throw std::invalid_argument("Retirement local prefix cannot use a receiver");expected={prior,original.localRelease};}
        else{const auto holder=std::get<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(original.owners[i]);const bool payload=q::pooledSourceOwner(holder->read()->publication()->sourceWave())->plan().parameters().native.reply_capacity!=0;if(payload){endpoint(original.releaseReceived[i],holder,q::PooledRetirementWireRole::Releases,CapturedTransferEndpoint::Consumer);requirePooledOriginalDependencies(graph,original.releaseReceived[i],{prior});expected={original.releaseReceived[i]};}else{if(!original.releaseReceived[i].empty())throw std::invalid_argument("Retirement added C0 payload/count wire");expected={prior};}}
        requirePooledOriginalDependencies(graph,original.releaseImported[i],std::move(expected));prior=original.releaseImported[i];
    }
    requirePooledOriginalDependencies(graph,original.reconciled,{prior});requirePooledOriginalDependencies(graph,original.snapshot,{original.reconciled});prior=original.snapshot;
    for(size_t i=0;i<count;++i){source(original.retiredImported[i],q::PooledSourceAction::ImportRetired,i);
        if(i==p.local_owner)requirePooledOriginalDependencies(graph,original.retiredImported[i],{prior,original.localRetire});
        else{const auto holder=std::get<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(original.owners[i]);endpoint(original.retiredReceived[i],holder,q::PooledRetirementWireRole::Retired,CapturedTransferEndpoint::Consumer);requirePooledOriginalDependencies(graph,original.retiredReceived[i],{prior});requirePooledOriginalDependencies(graph,original.retiredImported[i],{original.retiredReceived[i]});}
        prior=original.retiredImported[i];
    }
    requirePooledOriginalDependencies(graph,original.finished,{prior});requirePooledOriginalDependencies(graph,original.observed,{original.finished});
}
} // namespace llaminar2

namespace llaminar2 {
/** @brief Inspect only the exact original endpoint stage types without another object/vtable allocation. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledRetirementTransfer(const IComputeStage& stage){
    if(const auto* original=dynamic_cast<const PooledRetirementTransferStage*>(&stage))return original->directedTransferOperation();
    return std::nullopt;
}
}
