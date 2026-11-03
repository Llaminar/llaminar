/**
 * @file PooledImmutableParticipantReadGraph.cpp
 * @brief Exact original Grant64 hit requests, native bytes and participant-local read DAG.
 *
 * All requests retire before their original command bank can be phase-reused.
 * Serial response receives retire behind their actual import/copy completion.
 * The original decoder is the last native-row reader on both success and
 * truthful failure; source worker retirement alone never releases these roots.
 */
#include "PooledImmutableParticipantReadGraph.h"
#include "PooledImmutableDirectedOperation.h"
#include "PooledImmutableDirectedMessageGeometry.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <set>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;
/** @return Checked closed wire index. */
size_t index(PooledReadWireRole role){const auto i=static_cast<unsigned>(role);if(i>=4)throw std::invalid_argument("Unknown read wire role");return i;}
/** @return Original reverse direction, independent of mutable counts. */
bool reverse(PooledReadWireRole role){(void)index(role);return role==PooledReadWireRole::Receipt || role==PooledReadWireRole::Rows;}
/** @return Exact original frame/native authority for one canonical role. */
std::shared_ptr<TensorBase> root(const PooledImmutableNativeRoots& roots,BufferId id){return id==BufferId::QFN_POOLED_SOURCE_FRAME || id==BufferId::QFN_POOLED_OWNER_FRAME?roots.frame:roots.native;}
}
PooledPinWireSpan PooledImmutableCapturedReadWave::source(PooledReadWireRole role)const{
    (void)index(role);const auto source=publication_->sourceWave();const auto storage=source->boundary()->pooledStorage();const auto& s=storage->geometry().parameters();const auto owner=pooledSourceOwner(source);const auto& o=owner->plan().parameters();
    switch(role){
    case PooledReadWireRole::Outcome:return {BufferId::QFN_POOLED_SOURCE_NATIVE,s.offsets.frontier,sizeof(n::SourceFrontier)};
    case PooledReadWireRole::Requests:return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,s.coordinator.native.pin_commands,size_t{s.coordinator.native.reply_capacity}*sizeof(n::p::Grant)};
    case PooledReadWireRole::Receipt:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.header,sizeof(n::HitReadReceipt)};
    case PooledReadWireRole::Rows:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.source_payload,PooledImmutableDirectedMessageGeometry::fromPlan(pooledSourceOwner(source)->plan()).rows()};
    }throw std::invalid_argument("Unknown read source");
}
PooledPinWireSpan PooledImmutableCapturedReadWave::destination(PooledReadWireRole role)const{
    (void)index(role);const auto source=publication_->sourceWave();const auto& s=source->boundary()->pooledStorage()->geometry().parameters();const auto& o=pooledSourceOwner(source)->plan().parameters();
    switch(role){
    case PooledReadWireRole::Outcome:return {BufferId::QFN_POOLED_OWNER_NATIVE,o.native.source_outcome,sizeof(n::SourceFrontier)};
    case PooledReadWireRole::Requests:return {BufferId::QFN_POOLED_OWNER_FRAME,o.frame.commands,PooledImmutableDirectedMessageGeometry::fromPlan(pooledSourceOwner(source)->plan()).grants()};
    case PooledReadWireRole::Receipt:return {BufferId::QFN_POOLED_SOURCE_NATIVE,s.offsets.seal_receipt,sizeof(n::HitReadReceipt)};
    case PooledReadWireRole::Rows:return {BufferId::QFN_POOLED_SOURCE_FRAME,s.offsets.payload,PooledImmutableDirectedMessageGeometry::fromPlan(pooledSourceOwner(source)->plan()).rows()};
    }throw std::invalid_argument("Unknown read destination");
}
std::shared_ptr<const PooledImmutableCapturedReadWave> PooledImmutableCapturedReadWave::bind(
    std::shared_ptr<const PooledImmutableCapturedPublicationWave> publication,std::shared_ptr<PhysicalMemoryAuthority> memory){
    if(isGraphCaptureActive() || !publication || !memory)throw std::invalid_argument("Read binding needs original cold publication/PMA");publication->requireCurrent();const auto source=publication->sourceWave();const auto pin=source->previous()->previous()->pin();const auto forward=pin->forward();forward->ownerStorage()->requireAuthority(memory);forward->coordinatorStorage()->requireAuthority(memory);
    if(!forward->channel_ || !pin->channel_ || !forward->declaration().capturedChannelTransport())throw std::invalid_argument("Read native/rank transport needs complete qualified factory");
    auto out=std::shared_ptr<PooledImmutableCapturedReadWave>(new PooledImmutableCapturedReadWave(publication,memory));const auto own=forward->ownerStorage()->roots(),coord=forward->coordinatorStorage()->roots(),src=source->boundary()->pooledStorage()->roots();const auto& o=forward->ownerStorage()->plan().parameters();const auto& c=forward->coordinatorStorage()->plan().parameters();const auto& s=source->boundary()->pooledStorage()->geometry().parameters();auto& transfer=TransferEngine::instance();
    for(unsigned i=0;i<4;++i){const auto role=static_cast<PooledReadWireRole>(i);if(!o.native.reply_capacity && (role==PooledReadWireRole::Requests || role==PooledReadWireRole::Rows))continue;
        const bool back=reverse(role);const auto from=out->source(role),to=out->destination(role);const auto channel=back?pin->channel_:forward->channel_;const auto key=((forward->declaration().identity()^(128+i))*1099511628211ULL)|1;const auto origin=back?own:role==PooledReadWireRole::Requests?coord:src;const auto target=back?src:own;
        const auto maximum=role==PooledReadWireRole::Requests?to.bytes:from.bytes;
        auto send=transfer.bindCapturedTransfer(channel,CapturedTransferEndpoint::Producer,{key,maximum},root(origin,from.buffer),from.offset);
        auto receive=transfer.bindCapturedTransfer(channel,CapturedTransferEndpoint::Consumer,{key,maximum},root(target,to.buffer),to.offset);
        if(role==PooledReadWireRole::Requests){send=transfer.bindDeviceSlicedTransfer(std::move(send),coord.native,c.native.pin_slices+size_t{o.owner.coordinate()}*sizeof(n::DirectedSlice),from.bytes,0);
            receive=transfer.bindDeviceCountedTransfer(std::move(receive),own.native,o.native.ingress+offsetof(n::OwnerIngress,record_bytes));}
        else if(role==PooledReadWireRole::Rows){send=transfer.bindDeviceCountedTransfer(std::move(send),own.native,o.native.ingress+offsetof(n::OwnerIngress,payload_bytes));
            receive=transfer.bindDeviceCountedTransfer(std::move(receive),src.native,s.offsets.payload_bytes);}
        out->bindings_[i].emplace(std::move(send),std::move(receive));
    }out->requireCurrent();return out;
}
void PooledImmutableCapturedReadWave::requireCurrent()const{publication_->requireCurrent();const auto source=publication_->sourceWave();pooledSourceOwner(source)->requireAuthority(memory_);source->boundary()->pooledStorage()->coordinator()->requireAuthority(memory_);source->boundary()->pooledStorage()->captureIdentity();}
const CapturedTransferBinding& PooledImmutableCapturedReadWave::binding(PooledReadWireRole role,CapturedTransferEndpoint side)const{requireCurrent();const auto& pair=bindings_[index(role)];if(!pair || (side!=CapturedTransferEndpoint::Producer && side!=CapturedTransferEndpoint::Consumer))throw std::invalid_argument("Read missing/C0 endpoint");return side==CapturedTransferEndpoint::Producer?pair->first:pair->second;}
PooledNativeOperandSpans PooledImmutableCapturedReadWave::ownerInputs()const{
    requireCurrent();const auto source=publication_->sourceWave();const auto owner=pooledSourceOwner(source);const auto& o=owner->plan().parameters();const auto roots=owner->roots();auto out=source->previous()->previous()->pin()->forward()->ownerInputs(pooled_native::OwnerAction::Begin);
    const auto operand=[&](PooledNativeOperand role,PooledPinWireSpan span){if(span.bytes)out[static_cast<size_t>(role)]=PooledNativeTensorSpan{root(roots,span.buffer),span.buffer,span.offset,span.bytes};};
    operand(PooledNativeOperand::Outcome,destination(PooledReadWireRole::Outcome));
    if(o.native.reply_capacity){operand(PooledNativeOperand::Records,destination(PooledReadWireRole::Requests));operand(PooledNativeOperand::RecordBytes,{BufferId::QFN_POOLED_OWNER_NATIVE,o.native.ingress+offsetof(n::OwnerIngress,record_bytes),8});}
    return out;
}
uint64_t PooledImmutableCapturedReadWave::captureIdentity()const{requireCurrent();auto h=publication_->captureIdentity();for(size_t i=0;i<bindings_.size();++i){h=(h^i)*1099511628211ULL;if(bindings_[i]){h=(h^bindings_[i]->first.captureIdentity())*1099511628211ULL;h=(h^bindings_[i]->second.captureIdentity())*1099511628211ULL;}}return h?h:1;}
}
namespace llaminar2 {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;
/** @return True only on the real physical owner endpoint. */
bool ownerEndpoint(q::PooledReadWireRole role,CapturedTransferEndpoint side){const bool back=role==q::PooledReadWireRole::Receipt || role==q::PooledReadWireRole::Rows;return back==(side==CapturedTransferEndpoint::Producer);}
/** @brief Explicit original one-participant endpoint; strong holder retains descriptor/count/payload owners. */
class PooledReadTransferStage final:public IComputeStage {
public:
    /** @return Exact original semantic endpoint and native dispatch inputs before capture. */
    q::PooledImmutableDirectedTransferOperation directedTransferOperation()const{
        requireDeclarations();const auto& binding=edge_->binding(role_,side_);
        return {q::PooledImmutableDirectedWireFamily::Read,static_cast<std::uint32_t>(role_),side_,
            binding.message().bytes,binding.extentSource(),binding.hasSourceSlice()};
    }

    /** @brief Bind original arena roles and one immutable endpoint before capture. */
    PooledReadTransferStage(BufferArena& arena,std::shared_ptr<const q::PooledImmutableCapturedReadWave> edge,q::PooledReadWireRole role,CapturedTransferEndpoint side):IComputeStage(ownerEndpoint(role,side)?q::pooledSourceOwner(edge->publication()->sourceWave())->plan().allocator().device:edge->publication()->sourceWave()->boundary()->device()),arena_(&arena),edge_(std::move(edge)),role_(role),side_(side){requireDeclarations();identity_=graphCaptureVariantSignature();}
    /** @brief Acquire exact original setup events on the actual participant stream. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override{if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<uintptr_t>(stream)<=2)throw std::invalid_argument("Read endpoint requires exact participant/stream");requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Read capture identity changed");if(ownerEndpoint(role_,side_))q::pooledSourceOwner(edge_->publication()->sourceWave())->acquireSetup(stream);else edge_->publication()->sourceWave()->boundary()->pooledStorage()->acquireSetup(stream);setGPUStream(stream);return true;}
    /** @brief Submit the actual fixed/sliced/counted message without a host count read. */
    bool execute(IDeviceContext* context)override{if(!context || context->deviceId()!=device())throw std::invalid_argument("Read endpoint changed device");requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Read endpoint changed");TransferEngine::instance().enqueueCapturedTransfer(edge_->binding(role_,side_),requireGPUStream());return true;}
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER;}
    /** @return This original bound endpoint executes native device-counted transfer kernels, independently from collective-runtime capture. */
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority()const noexcept override{return CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer;}
    std::string name()const override{return "PooledImmutableDirectedRead";}
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Full original role/root/count/descriptor/runtime identity. */
    uint64_t graphCaptureVariantSignature()const override{requireDeclarations();return ((((edge_->captureIdentity()^(128+static_cast<unsigned>(role_)))*1099511628211ULL)^static_cast<unsigned>(side_))*1099511628211ULL)|1;}
    /** @return Actual holder/role/endpoint, used by cold successor guards. */
    bool matches(const q::PooledImmutableCapturedReadWave* edge,q::PooledReadWireRole role,CapturedTransferEndpoint side)const noexcept{return edge_.get()==edge && role_==role && side_==side;}
    /** @return Every original payload, count and descriptor authority; receives publish their distinct native scalar. */
    StageBufferContract bufferContract()const override{requireDeclarations();auto c=StageBufferContract::build();const bool send=side_==CapturedTransferEndpoint::Producer;const auto span=send?edge_->source(role_):edge_->destination(role_);if(send)c.addInput(span.buffer,"INT8");else c.addPreallocatedInOut(span.buffer,"INT8");
        if((role_==q::PooledReadWireRole::Requests && !send) || (role_==q::PooledReadWireRole::Rows && !send)){
            const auto id=role_==q::PooledReadWireRole::Requests?BufferId::QFN_POOLED_OWNER_NATIVE:BufferId::QFN_POOLED_SOURCE_NATIVE;if(id!=span.buffer)c.addPreallocatedInOut(id,"INT8");}
        return c;}
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo d;d.addScalarInt("read_role",static_cast<unsigned>(role_));d.addScalarInt("read_endpoint",static_cast<unsigned>(side_));return d;}
private:
    /** @brief Authenticate both original payload/frame and descriptor/count native roots before recording. */
    void requireDeclarations()const{if(!arena_ || !edge_)throw std::logic_error("Read lost original owner");edge_->requireCurrent();(void)edge_->binding(role_,side_);const auto source=edge_->publication()->sourceWave();const bool own=ownerEndpoint(role_,side_);const auto storage=source->boundary()->pooledStorage();const auto roots=own?q::pooledSourceOwner(source)->roots():storage->roots();const auto span=side_==CapturedTransferEndpoint::Producer?edge_->source(role_):edge_->destination(role_);const auto native=own?BufferId::QFN_POOLED_OWNER_NATIVE:BufferId::QFN_POOLED_SOURCE_NATIVE;
        auto payload=span.buffer==BufferId::QFN_POOLED_COORDINATOR_NATIVE?storage->coordinator()->roots().native:(span.buffer==BufferId::QFN_POOLED_SOURCE_FRAME || span.buffer==BufferId::QFN_POOLED_OWNER_FRAME?roots.frame:roots.native);
        if(arena_->getTensor(span.buffer)!=payload.get() || arena_->getTensor(native)!=roots.native.get())throw std::logic_error("Read replaced original payload/descriptor/count root");}
    BufferArena* arena_;std::shared_ptr<const q::PooledImmutableCapturedReadWave> edge_;q::PooledReadWireRole role_;CapturedTransferEndpoint side_;uint64_t identity_=0;
};
/** @brief Reject all naming collisions before graph mutation. */
void names(const ComputeGraph& graph,std::span<const std::string> values){std::set<std::string> unique;for(const auto& value:values)if(value.empty() || graph.getNode(value) || !unique.insert(value).second)throw std::invalid_argument("Read would replace an original graph node");}
/** @return Actual publication member retained by a read member. */
q::PooledImmutablePublicationParticipant publicationMember(const q::PooledImmutableReadParticipant& member){return std::visit([](const auto& value)->q::PooledImmutablePublicationParticipant{if(!value)throw std::invalid_argument("Read lost original participant");using T=std::decay_t<decltype(*value)>;if constexpr(std::is_same_v<T,q::PooledImmutableLocalOwnerBinding>)return value;else return value->publication();},member);}
}
PooledPinParticipantNodes appendPooledReadOwner(ComputeGraph& graph,BufferArena& arena,std::shared_ptr<const q::PooledImmutableCapturedReadWave> original,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& readySent,const std::string& prefix){
    if(isGraphCaptureActive() || !original || !code || prefix.empty())throw std::invalid_argument("Read owner requires cold original publication/code");original->requireCurrent();requirePooledPublicationOwnerCompletion(graph,original->publication(),readySent);const auto owner=q::pooledSourceOwner(original->publication()->sourceWave());const auto device=owner->plan().allocator().device;if(code->backend()!=device.type)throw std::invalid_argument("Read owner changed backend");
    struct Node{std::string name;std::unique_ptr<IComputeStage> stage;};std::vector<Node> nodes;
    const auto endpoint=[&](const char* suffix,q::PooledReadWireRole role,CapturedTransferEndpoint side){nodes.push_back({prefix+"/"+suffix,std::make_unique<PooledReadTransferStage>(arena,original,role,side)});};
    endpoint("outcome_receive",q::PooledReadWireRole::Outcome,CapturedTransferEndpoint::Consumer);
    if(owner->plan().parameters().native.reply_capacity)endpoint("requests_receive",q::PooledReadWireRole::Requests,CapturedTransferEndpoint::Consumer);
    nodes.push_back({prefix+"/read_rows",std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{n::OwnerAction::ReadRows,owner,q::PooledImmutableOwnerOperands(n::OwnerAction::ReadRows,owner->plan(),original->ownerInputs(),n::SourceCondition::AfterSuccess)})});
    endpoint("receipt_send",q::PooledReadWireRole::Receipt,CapturedTransferEndpoint::Producer);
    if(owner->plan().parameters().native.reply_capacity)endpoint("rows_send",q::PooledReadWireRole::Rows,CapturedTransferEndpoint::Producer);
    std::vector<std::string> declared;for(const auto& node:nodes)declared.push_back(node.name);names(graph,declared);auto prior=readySent;for(auto& node:nodes){const auto name=node.name;graph.addNode(name,std::move(node.stage),device);graph.addDependency(name,prior);prior=name;}return {declared.front(),prior};
}
PooledImmutableReadCompletion appendPooledReadCoordinator(ComputeGraph& graph,BufferArena& arena,const PooledImmutablePublicationCompletion& publication,std::span<const q::PooledImmutableReadParticipant> owners,q::PooledNativeTensorSpan output,const std::string& prefix,const std::string& outputName){
    if(isGraphCaptureActive() || prefix.empty() || outputName.empty() || owners.size()!=publication.owners.size())throw std::invalid_argument("Read needs cold original complete publication");requirePooledPublicationCompletion(graph,publication);
    for(size_t i=0;i<owners.size();++i)if(publicationMember(owners[i])!=publication.owners[i])throw std::invalid_argument("Read substituted original ordered publication member");
    const auto boundary=publication.source.boundary;const auto device=boundary->device();PooledImmutableReadCompletion out;out.publication=publication;out.owners.assign(owners.begin(),owners.end());out.outputSpan=output;out.planned=prefix+"/planned";out.sealed=prefix+"/sealed";out.output=outputName;
    for(auto* values:{&out.outcomeSent,&out.requestSent,&out.receiptReceived,&out.rowsReceived,&out.imported})values->resize(owners.size());
    struct Node{std::string name;std::unique_ptr<IComputeStage> stage;};std::vector<Node> nodes;
    const auto phase=[&](const std::string& name,q::PooledSourceAction action,uint32_t owner=q::adaptive_lookup::None){nodes.push_back({name,std::make_unique<PooledImmutableSourcePhaseStage>(arena,boundary,action,*publication.source.control,std::nullopt,owner)});};
    const auto endpoint=[&](size_t i,const std::string& name,q::PooledReadWireRole role,CapturedTransferEndpoint side){nodes.push_back({name,std::make_unique<PooledReadTransferStage>(arena,std::get<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(owners[i]),role,side)});};
    phase(out.planned,q::PooledSourceAction::PlanReads);
    for(size_t i=0;i<owners.size();++i){const auto base=prefix+"/owner"+std::to_string(i)+"/";out.imported[i]=base+"imported";phase(out.imported[i],q::PooledSourceAction::ImportReadRows,i);
        if(const auto* remote=std::get_if<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(&owners[i])){(*remote)->requireCurrent();out.outcomeSent[i]=base+"outcome_send";endpoint(i,out.outcomeSent[i],q::PooledReadWireRole::Outcome,CapturedTransferEndpoint::Producer);out.receiptReceived[i]=base+"receipt_receive";endpoint(i,out.receiptReceived[i],q::PooledReadWireRole::Receipt,CapturedTransferEndpoint::Consumer);
            if(q::pooledSourceOwner((*remote)->publication()->sourceWave())->plan().parameters().native.reply_capacity){out.requestSent[i]=base+"requests_send";endpoint(i,out.requestSent[i],q::PooledReadWireRole::Requests,CapturedTransferEndpoint::Producer);out.rowsReceived[i]=base+"rows_receive";endpoint(i,out.rowsReceived[i],q::PooledReadWireRole::Rows,CapturedTransferEndpoint::Consumer);}}
    }
    phase(out.sealed,q::PooledSourceAction::SealReads);nodes.push_back({out.output,std::make_unique<PooledImmutableSourcePhaseStage>(arena,boundary,q::PooledSourceAction::Read,*publication.source.control,output)});
    std::vector<std::string> declared;for(const auto& node:nodes)declared.push_back(node.name);names(graph,declared);for(auto& node:nodes)graph.addNode(node.name,std::move(node.stage),device);
    graph.addDependency(out.planned,publication.ready);auto prior=out.planned;
    // Finish every original Grant-bank reader before the first serial response.
    for(size_t i=0;i<owners.size();++i){if(!out.outcomeSent[i].empty()){graph.addDependency(out.outcomeSent[i],prior);prior=out.outcomeSent[i];}if(!out.requestSent[i].empty()){graph.addDependency(out.requestSent[i],prior);prior=out.requestSent[i];}}
    for(size_t i=0;i<owners.size();++i){if(!out.receiptReceived[i].empty()){graph.addDependency(out.receiptReceived[i],prior);prior=out.receiptReceived[i];}if(!out.rowsReceived[i].empty()){graph.addDependency(out.rowsReceived[i],prior);prior=out.rowsReceived[i];}graph.addDependency(out.imported[i],prior);prior=out.imported[i];}
    graph.addDependency(out.sealed,prior);graph.addDependency(out.output,out.sealed);out.coordinator={out.planned,out.output};return out;
}
void requirePooledReadCompletion(const ComputeGraph& graph,const PooledImmutableReadCompletion& original){
    if(isGraphCaptureActive())throw std::invalid_argument("Read successor must be declared before capture");requirePooledPublicationCompletion(graph,original.publication);const auto boundary=original.publication.source.boundary;const auto device=boundary->device();const auto count=original.publication.owners.size();if(original.owners.size()!=count)throw std::invalid_argument("Read lost original cohort");
    for(const auto* values:{&original.outcomeSent,&original.requestSent,&original.receiptReceived,&original.rowsReceived,&original.imported})if(values->size()!=count)throw std::invalid_argument("Read lost issued-edge inventory");
    const auto node=[&](const std::string& name){const auto* value=graph.getNode(name);if(!value || value->device!=device)throw std::invalid_argument("Read lost actual local producer");return value;};
    const auto depends=[&](const std::string& name,const std::string& prior){node(name);node(prior);requirePooledOriginalDependencies(graph,name,{prior});};
    const auto phase=[&](const std::string& name,q::PooledSourceAction action,uint32_t owner=q::adaptive_lookup::None){const auto* value=dynamic_cast<const PooledImmutableSourcePhaseStage*>(node(name)->stage.get());if(!value || !value->matchesOriginal(boundary.get(),action,owner,*original.publication.source.control))throw std::invalid_argument("Read substituted actual source operation/control");return value;};
    const auto endpoint=[&](size_t i,const std::string& name,q::PooledReadWireRole role,CapturedTransferEndpoint side){const auto& edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(original.owners[i]);const auto* value=dynamic_cast<const PooledReadTransferStage*>(node(name)->stage.get());if(!value || !value->matches(edge.get(),role,side))throw std::invalid_argument("Read substituted original endpoint/channel");};
    phase(original.planned,q::PooledSourceAction::PlanReads);depends(original.planned,original.publication.ready);auto prior=original.planned;
    for(size_t i=0;i<count;++i){if(publicationMember(original.owners[i])!=original.publication.owners[i])throw std::invalid_argument("Read changed original ordered member");
        if(const auto* remote=std::get_if<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(&original.owners[i])){(*remote)->requireCurrent();endpoint(i,original.outcomeSent[i],q::PooledReadWireRole::Outcome,CapturedTransferEndpoint::Producer);depends(original.outcomeSent[i],prior);prior=original.outcomeSent[i];const bool data=q::pooledSourceOwner((*remote)->publication()->sourceWave())->plan().parameters().native.reply_capacity!=0;if(data){if(original.rowsReceived[i].empty())throw std::invalid_argument("Read omitted original row receive");endpoint(i,original.requestSent[i],q::PooledReadWireRole::Requests,CapturedTransferEndpoint::Producer);depends(original.requestSent[i],prior);prior=original.requestSent[i];}else if(!original.requestSent[i].empty() || !original.rowsReceived[i].empty())throw std::invalid_argument("C0 read invented a payload");}
        else if(!original.outcomeSent[i].empty() || !original.requestSent[i].empty() || !original.receiptReceived[i].empty() || !original.rowsReceived[i].empty())throw std::invalid_argument("Read invented local self transport");}
    for(size_t i=0;i<count;++i){if(std::holds_alternative<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(original.owners[i])){endpoint(i,original.receiptReceived[i],q::PooledReadWireRole::Receipt,CapturedTransferEndpoint::Consumer);depends(original.receiptReceived[i],prior);prior=original.receiptReceived[i];if(!original.rowsReceived[i].empty()){endpoint(i,original.rowsReceived[i],q::PooledReadWireRole::Rows,CapturedTransferEndpoint::Consumer);depends(original.rowsReceived[i],prior);prior=original.rowsReceived[i];}}
        phase(original.imported[i],q::PooledSourceAction::ImportReadRows,i);depends(original.imported[i],prior);prior=original.imported[i];}
    phase(original.sealed,q::PooledSourceAction::SealReads);depends(original.sealed,prior);const auto* output=phase(original.output,q::PooledSourceAction::Read);if(!output->matchesReadOutput(original.outputSpan))throw std::invalid_argument("Read substituted original FP32 consumer span");depends(original.output,original.sealed);if(original.coordinator.first!=original.planned || original.coordinator.last!=original.output)throw std::invalid_argument("Read lost actual last native-row reader/drain frontier");
}
void requirePooledReadOwnerCompletion(const ComputeGraph& graph,const std::shared_ptr<const q::PooledImmutableCapturedReadWave>& original,const std::string& rowsSent){
    if(isGraphCaptureActive() || !original)throw std::invalid_argument("Read owner successor needs cold original completion");original->requireCurrent();const auto storage=q::pooledSourceOwner(original->publication()->sourceWave());const auto device=storage->plan().allocator().device;
    const auto node=[&](const std::string& name){const auto* value=graph.getNode(name);if(!value || value->device!=device || value->dependencies.size()!=1)throw std::invalid_argument("Read owner lost original exact predecessor");return value;};
    const auto endpoint=[&](const std::string& name,q::PooledReadWireRole role,CapturedTransferEndpoint side){const auto* value=node(name);const auto* stage=dynamic_cast<const PooledReadTransferStage*>(value->stage.get());if(!stage || !stage->matches(original.get(),role,side))throw std::invalid_argument("Read owner replaced endpoint");return value->dependencies.front();};
    auto prior=rowsSent;if(storage->plan().parameters().native.reply_capacity)prior=endpoint(prior,q::PooledReadWireRole::Rows,CapturedTransferEndpoint::Producer);prior=endpoint(prior,q::PooledReadWireRole::Receipt,CapturedTransferEndpoint::Producer);
    const auto* value=node(prior);const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(value->stage.get());const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;const q::PooledImmutableOwnerOperands expected(n::OwnerAction::ReadRows,storage->plan(),original->ownerInputs(),n::SourceCondition::AfterSuccess);if(!actual || actual->storage!=storage || actual->action!=n::OwnerAction::ReadRows || actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Read owner replaced actual pinned-byte producer");prior=value->dependencies.front();
    if(storage->plan().parameters().native.reply_capacity)prior=endpoint(prior,q::PooledReadWireRole::Requests,CapturedTransferEndpoint::Consumer);prior=endpoint(prior,q::PooledReadWireRole::Outcome,CapturedTransferEndpoint::Consumer);requirePooledPublicationOwnerCompletion(graph,original->publication(),prior);
}
}

namespace llaminar2 {
/** @brief Inspect only the exact original endpoint stage types without another object/vtable allocation. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledReadTransfer(const IComputeStage& stage){
    if(const auto* original=dynamic_cast<const PooledReadTransferStage*>(&stage))return original->directedTransferOperation();
    return std::nullopt;
}
}
