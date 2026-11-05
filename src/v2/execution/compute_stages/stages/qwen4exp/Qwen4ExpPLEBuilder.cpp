/**
 * @file Qwen4ExpPLEBuilder.cpp
 * @brief Wire native lexical lookup and prepared projections as one flat model block.
 *
 * Key/value projections use the exact native source handles. Gains normalize
 * each residual branch independently. The final convolution consumes an
 * immutable accepted tail and leaves candidate rows available for the same
 * acceptance decision that commits KV/GDN state. No table, history or model
 * state crosses to a second authority while constructing or running the graph.
 */
#include "Qwen4ExpPLEBuilder.h"
#include "execution/qwen4exp/PooledImmutableRetainedParticipants.h"
#include "execution/qwen4exp/PooledImmutableParticipantExecutionPolicy.h"
#include "execution/qwen4exp/PooledImmutableReplicatedReaderGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantSourceGraph.h"
#include "backends/ReadAheadNativeResourceFamily.h"
#include "../GEMMStage.h"
#include "../NativeAllGatherStage.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "../../../local_execution/graph/ComputeGraph.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <typeinfo>

namespace llaminar2
{
namespace
{
/** @brief Refuse malformed composition before mutating the caller's graph. */
void require(bool condition,const char *message)
{
    if(!condition) throw std::invalid_argument(message);
}
/** @brief One completely constructed graph node and its explicit incoming edges. */
struct Pending
{
    std::string name;
    std::unique_ptr<IComputeStage> stage;
    std::vector<std::string> dependencies;
};
/** @brief Bind the existing prepared GEMM path without inventing a projection policy. */
std::unique_ptr<IComputeStage> project(const Qwen4ExpPLEContext &context,
    const Qwen4ExpPreparedProjection &weight,Qwen4ExpFloatBinding input,Qwen4ExpFloatBinding output,int width)
{
    require(input.byteOffset==0 && output.byteOffset==0 && weight.ref().device==context.device,
        "PLE prepared projections require exact participant identities and full arena rows");
    weight.requireGeometry(width,context.hidden);
    return std::make_unique<GEMMStage>(GEMMStage::Params{
        .device_id=context.device,.mpi_ctx=context.mpi,.A=input.tensor,.B=weight.tensor(),.C=output.tensor,
        .m=context.rows.capacity,.n=width,.k=context.hidden,.alpha=1.0f,.beta=0.0f,.transpose_B=false,
        .a_buffer_id=input.buffer,.c_buffer_id=output.buffer,
        .force_decode_equivalent_verifier_prefill=context.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
        .verifier_row_range=context.projectionRows,.prepared_ref=weight.ref(),.prepared_store=&weight.store()});
}
/** @brief Construct branch normalization through the same primitive on every backend. */
std::unique_ptr<IComputeStage> norm(const Qwen4ExpPLEContext &context,
    Qwen4ExpFloatBinding input,Qwen4ExpImmutableWeightBinding gain,Qwen4ExpFloatBinding output)
{
    return std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
        .device_id=context.device,.mpi_ctx=context.mpi,.rows=context.rows,
        .operation=Qwen4ExpPrimitiveStage::BranchRMSNorm{context.hidden,context.epsilon,input,gain,output}});
}
}
Qwen4ExpEarlyLookupFrame::Qwen4ExpEarlyLookupFrame(ComputeGraph& graph,ImmutableLookupStage::Params params,
    std::uint64_t hashIdentity,std::string producer)
    :params_(std::move(params)),handoff_(graph,std::move(producer),
        ImmutableLookupStage(params_).graphCaptureVariantSignature(),hashIdentity) {}

std::unique_ptr<Qwen4ExpEarlyLookupFrame> Qwen4ExpEarlyLookupFrame::publish(ComputeGraph& graph,
    const Qwen4ExpPLEContext& c,Qwen4ExpIntInput tokens,Qwen4ExpIntBinding prior,Qwen4ExpIntBinding ids,
    Qwen4ExpFloatBinding output,ImmutableLookupStage::GPUStorage storage)
{
    require(c.device.is_gpu() && graph.size()==0,"Early current lookup must own the original empty local graph prefix");
    const auto hash=c.prefix+"hash",published=c.prefix+"publish";
    auto stage=std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Qwen4ExpPrimitiveStage::NgramHash{tokens,prior,ids,c.hash}});
    const auto identity=stage->graphCaptureVariantSignature();
    ImmutableLookupStage::Params params{.device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,.ids=ids,.output=output,
        .format=c.nativeFormat,.rowElements=c.nativeRowElements,.rowBytes=c.nativeRowBytes,.storage=storage};
    (void)ImmutableLookupStage(params).graphCaptureVariantSignature(); // Validate all declarations before any graph mutation.
    auto publication=std::unique_ptr<ImmutableLookupStage>(new ImmutableLookupStage(params,ImmutableLookupStage::Operation::Publish));
    graph.setRetainedStorageBoundary(storage.boundary);
    graph.addNode(hash,std::move(stage),c.device);graph.addNode(published,std::move(publication),c.device);
    graph.addDependency(published,hash);
    return std::unique_ptr<Qwen4ExpEarlyLookupFrame>(new Qwen4ExpEarlyLookupFrame(graph,params,identity,published));
}
void Qwen4ExpEarlyLookupFrame::requireConsumed() const
{ handoff_.requireConsumed(); }
Qwen4ExpEarlyLookupFrame::Consumption Qwen4ExpEarlyLookupFrame::consume(ComputeGraph& graph,
    ImmutableLookupStage::Params params,std::uint64_t hashIdentity,const std::vector<std::string>& trunk)
{
    const auto* original=std::get_if<ImmutableLookupStage::GPUStorage>(&params_.storage);
    const auto* current=std::get_if<ImmutableLookupStage::GPUStorage>(&params.storage);
    require(original && current && current->boundary==original->boundary,"Early lookup consume replaced its actual source");
    auto stage=std::unique_ptr<IComputeStage>(new ImmutableLookupStage(params,ImmutableLookupStage::Operation::Consume));
    auto dependencies=handoff_.consume(graph,ImmutableLookupStage(params).graphCaptureVariantSignature(),hashIdentity,trunk);
    return {std::move(stage),std::move(dependencies)};
}

Qwen4ExpResidentReadAheadFrame::Qwen4ExpResidentReadAheadFrame(ComputeGraph& graph,
    ImmutableLookupStage::Params params,std::uint64_t hashIdentity,std::string hash,std::string gather,
    std::shared_ptr<ReadAheadNativeResourceFamily> resources)
    :params_(std::move(params)),handoff_(graph,std::move(gather),
        ImmutableLookupStage(params_).graphCaptureVariantSignature(),hashIdentity),
     hash_(std::move(hash)),resources_(std::move(resources)) {}
std::unique_ptr<Qwen4ExpResidentReadAheadFrame> Qwen4ExpResidentReadAheadFrame::publish(ComputeGraph& graph,
    const Qwen4ExpPLEContext& c,Qwen4ExpIntInput tokens,Qwen4ExpIntBinding prior,Qwen4ExpIntBinding ids,
    Qwen4ExpFloatBinding output,ImmutableLookupStage::ResidentGPUStorage storage,
    std::shared_ptr<ReadAheadNativeResourceFamily> resources)
{
    require(graph.size()==0 && c.device.is_gpu() && resources && resources->device()==c.device,
        "Resident read-ahead requires the original empty graph and admitted exact-device event family");
    const auto hash=c.prefix+"hash",gather=c.prefix+"gather";
    auto hashStage=std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Qwen4ExpPrimitiveStage::NgramHash{tokens,prior,ids,c.hash}});
    const auto identity=hashStage->graphCaptureVariantSignature();
    ImmutableLookupStage::Params params{.device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,.ids=ids,.output=output,
        .format=c.nativeFormat,.rowElements=c.nativeRowElements,.rowBytes=c.nativeRowBytes,.storage=std::move(storage)};
    auto producer=std::make_unique<ImmutableLookupStage>(params);
    auto frame=std::unique_ptr<Qwen4ExpResidentReadAheadFrame>(new Qwen4ExpResidentReadAheadFrame(
        graph,params,identity,hash,gather,std::move(resources)));
    graph.addNode(hash,std::move(hashStage),c.device);
    graph.addNode(gather,std::move(producer),c.device);graph.addDependency(gather,hash);
    return frame;
}
std::vector<std::string> Qwen4ExpResidentReadAheadFrame::consume(ComputeGraph& graph,
    ImmutableLookupStage::Params params,std::uint64_t hashIdentity,const std::vector<std::string>& trunk)
{
    const auto* original=std::get_if<ImmutableLookupStage::ResidentGPUStorage>(&params_.storage);
    const auto* current=std::get_if<ImmutableLookupStage::ResidentGPUStorage>(&params.storage);
    require(original && current && current->table==original->table,
        "Resident read-ahead consume replaced its pinned immutable source");
    return handoff_.consume(graph,ImmutableLookupStage(params).graphCaptureVariantSignature(),hashIdentity,trunk);
}
void Qwen4ExpResidentReadAheadFrame::sealBranch(ComputeGraph& graph,const std::string& consumer)
{
    handoff_.requireConsumed();require(!sealed_,"Resident read-ahead branch was sealed twice");
    graph.setReadAheadBranch({hash_,handoff_.producer(),consumer,resources_,ReadAheadJoinPolicy::FirstDeclaredCaptureFrontier,{}});sealed_=true;
}
void Qwen4ExpResidentReadAheadFrame::requireConsumed()const
{handoff_.requireConsumed();require(sealed_,"Resident read-ahead has no native Consume join");}

namespace {
/** @brief Reuse the original fully authenticated source/reader construction in both scheduling modes. */
PooledImmutableLookupGraphEndpoints appendOriginalPooledLookup(ComputeGraph& graph,const Qwen4ExpPLEContext& c,
    const Qwen4ExpPLEBuffers& b,Qwen4ExpPooledLookupDeclaration storage,const std::string& gather){
    if(auto* pooled=std::get_if<Qwen4ExpPooledLookupStorage>(&storage))
    {
        require(pooled->arena && pooled->boundary && pooled->boundary->isPooled() && pooled->code &&
            pooled->boundary->device()==c.device,"PLE split lookup lost its original model-owned source domain");
        const auto sourceStorage=pooled->boundary->pooledStorage();
        const bool needsParticipants=qwen4exp::PooledImmutableParticipantExecutionPolicy::requiresRetainedParticipants(sourceStorage->geometry());
        require(needsParticipants==bool(pooled->participants),
            "PLE shared readers and remote storage require their actual retained source participant domain");
        require(!pooled->participants || pooled->participants->boundary()==pooled->boundary,
            "PLE participant domain differs from its original model source");
        const auto& declared=pooled->boundary->pooledStorage()->geometry().parameters();
        require(declared.format==c.nativeFormat && declared.row_elements==static_cast<std::uint32_t>(c.nativeRowElements) &&
            declared.owner.row_bytes==c.nativeRowBytes,"PLE split lookup differs from original source precision");
        const auto* native=c.rows.liveRows?std::get_if<Qwen4ExpNativeIntInput>(&c.rows.liveRows->value):nullptr;
        require(!c.rows.liveRows || (native && native->owner==c.device && native->capacity==1 && native->data),
            "PLE split lookup requires the original complete native token-count producer");
        const auto original=native?DeviceRowRange::deviceCounted(c.rows.capacity,native->data):DeviceRowRange::fullyActive(c.rows.capacity);
        require(original.captureIdentity()==pooled->originalRows.captureIdentity() &&
            bool(pooled->count)==bool(original.countOwner()),"PLE split original query/count authority changed");
        const auto expanded=original.repeatEachRow(16);
        const auto ids=pooled->arena->getSharedTensor(b.ids.buffer),output=pooled->arena->getSharedTensor(b.embedding.buffer);
        require(ids && output && ids.get()==b.ids.tensor && output.get()==b.embedding.tensor,
            "PLE split lookup requires original arena-owned hash and embedding roots");
        std::optional<qwen4exp::PooledNativeTensorSpan> countSpan;
        if(pooled->count){
            const auto count=pooled->arena->getSharedTensor(pooled->count->buffer);
            require(count && count.get()==pooled->count->tensor,"PLE split original count lost its original arena root");
            countSpan=qwen4exp::PooledNativeTensorSpan{count,pooled->count->buffer,pooled->count->byteOffset,sizeof(std::int32_t)};
        }

        const auto appendSource=[&](auto&&... operands){
            if(pooled->participants)return pooled->participants->appendPLE(graph,*pooled->arena,pooled->code,
                std::forward<decltype(operands)>(operands)...);
            return appendPooledImmutablePLELookupGraph(graph,*pooled->arena,pooled->boundary,pooled->code,
                std::forward<decltype(operands)>(operands)...);
        };
        const auto edge=appendSource(
            original,qwen4exp::PooledNativeTensorSpan{ids,b.ids.buffer,b.ids.byteOffset,static_cast<std::size_t>(expanded.physicalRows())*sizeof(std::uint64_t)},
            std::move(countSpan),pooled->control,
            qwen4exp::PooledNativeTensorSpan{output,b.embedding.buffer,b.embedding.byteOffset,static_cast<std::size_t>(expanded.physicalRows())*c.nativeRowElements*sizeof(float)},
            c.prefix+"pooled",gather);
        return edge;
    }
    else if(auto* reader=std::get_if<Qwen4ExpPooledReaderLookupStorage>(&storage))
    {
        require(reader->arena && reader->participants,
            "PLE reader requires its original shared source participants and local arena");
        const auto transport=reader->participants->transport();
        require(transport && transport->readerGeometry() &&
            transport->readerGeometry()->coordinator().device!=c.device,
            "PLE reader cannot replace the original source coordinator");
        const auto* native=c.rows.liveRows?std::get_if<Qwen4ExpNativeIntInput>(&c.rows.liveRows->value):nullptr;
        require(!c.rows.liveRows || (native && native->owner==c.device && native->capacity==1 && native->data),
            "PLE reader requires its original native query-count producer");
        const auto original=native?DeviceRowRange::deviceCounted(c.rows.capacity,native->data):
            DeviceRowRange::fullyActive(c.rows.capacity);
        require(original.captureIdentity()==reader->originalRows.captureIdentity(),
            "PLE reader changed its original query-count authority");
        const auto expanded=original.repeatEachRow(16);
        const auto output=reader->arena->getSharedTensor(b.embedding.buffer);
        require(output && output.get()==b.embedding.tensor,
            "PLE reader embedding must belong to its actual participant arena");
        std::optional<qwen4exp::PooledNativeTensorSpan> count;
        require(bool(reader->count)==bool(original.countOwner()),
            "PLE reader lost its original arena-owned query-count root");
        if(reader->count)
        {
            const auto root=reader->arena->getSharedTensor(reader->count->buffer);
            require(root && root.get()==reader->count->tensor,
                "PLE reader count must belong to its actual participant arena");
            count=qwen4exp::PooledNativeTensorSpan{root,reader->count->buffer,
                reader->count->byteOffset,sizeof(std::int32_t)};
        }

        const auto edge=reader->participants->appendReader(graph,*reader->arena,
            reader->control,expanded,
            qwen4exp::PooledNativeTensorSpan{output,b.embedding.buffer,b.embedding.byteOffset,
                static_cast<std::size_t>(expanded.physicalRows())*c.nativeRowElements*sizeof(float)},
            c.prefix+"pooled_reader",gather,std::move(count));
        return edge;
    }
    throw std::invalid_argument("Pooled PLE lost its original declared role");
}
/** @return Exact original immutable declaration, including shared control-block ownership. */
bool samePooledLookup(const Qwen4ExpPooledLookupDeclaration& first,const Qwen4ExpPooledLookupDeclaration& second){
    if(first.index()!=second.index())return false;
    return std::visit([&](const auto& a){
        using T=std::decay_t<decltype(a)>;const auto& b=std::get<T>(second);
        const auto same=[](const auto& a,const auto& b){return a.get()==b.get()&&!a.owner_before(b)&&!b.owner_before(a);};
        if(a.arena!=b.arena || !same(a.participants,b.participants) || a.control.captureIdentity()!=b.control.captureIdentity() ||
           a.originalRows.captureIdentity()!=b.originalRows.captureIdentity() || bool(a.count)!=bool(b.count))return false;
        if(a.count && (a.count->tensor!=b.count->tensor || a.count->buffer!=b.count->buffer || a.count->byteOffset!=b.count->byteOffset))return false;
        if constexpr(std::is_same_v<T,Qwen4ExpPooledLookupStorage>)return same(a.boundary,b.boundary) && a.code && b.code &&
            a.code->backend()==b.code->backend() && typeid(*a.code)==typeid(*b.code);
        return true;
    },first);
}

}
namespace {
/** @return Checked actual builder descriptor before any handoff initializer dereferences it. */
const PooledImmutableDemandFrontier& requiredDemand(const PooledImmutableLookupGraphEndpoints& endpoints){
    if(!endpoints.demand)throw std::invalid_argument("Pooled current demand has no original typed split descriptor");
    return *endpoints.demand;
}
}
Qwen4ExpPooledReadAheadFrame::Qwen4ExpPooledReadAheadFrame(ComputeGraph& graph,Qwen4ExpPooledLookupDeclaration storage,
    Qwen4ExpPLEBuffers buffers,const Qwen4ExpPLEContext& context,std::uint64_t hash,PooledImmutableLookupGraphEndpoints endpoints)
    :graph_(&graph),storage_(std::move(storage)),buffers_(std::move(buffers)),device_(context.device),
     format_(context.nativeFormat),row_elements_(context.nativeRowElements),row_bytes_(context.nativeRowBytes),
     hash_(hash),endpoints_(std::move(endpoints)),hash_producer_(context.prefix+"hash"),
     handoff_(graph,requiredDemand(endpoints_),hash,hash){
    require(endpoints_.demand && graph.getNode(endpoints_.demand->consume) && graph.getNode(endpoints_.last),
        "Pooled current demand lost its actual protocol frontiers");
    require(endpoints_.demand->role==PooledImmutableDemandRole::PureReader ||
        (!endpoints_.demand->published.empty()&&graph.getNode(endpoints_.demand->published)),
        "Pooled current demand lost its actual publication");
}
std::unique_ptr<Qwen4ExpPooledReadAheadFrame> Qwen4ExpPooledReadAheadFrame::publish(ComputeGraph& graph,
    const Qwen4ExpPLEContext& c,const Qwen4ExpPLEBuffers& b,Qwen4ExpPooledLookupDeclaration storage){
    require(graph.size()==0 && c.device.is_gpu() && c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary && !c.projectionRows,
        "Pooled read-ahead requires its original ordinary empty graph prefix");
    auto hash=std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Qwen4ExpPrimitiveStage::NgramHash{b.tokens,b.priorTokens,b.ids,c.hash}});
    const auto hashIdentity=hash->graphCaptureVariantSignature();
    const auto hashName=c.prefix+"hash";
    ComputeGraph fragment;
    fragment.addNode(hashName,std::move(hash),c.device);
    const auto endpoints=appendOriginalPooledLookup(fragment,c,b,storage,c.prefix+"gather");
    fragment.addDependency(endpoints.first,hashName);
    graph.merge(std::move(fragment));
    return std::unique_ptr<Qwen4ExpPooledReadAheadFrame>(new Qwen4ExpPooledReadAheadFrame(graph,std::move(storage),b,c,hashIdentity,endpoints));
}
std::string Qwen4ExpPooledReadAheadFrame::consume(ComputeGraph& graph,const Qwen4ExpPLEContext& c,const Qwen4ExpPLEBuffers& b,
    const Qwen4ExpPooledLookupDeclaration& storage,std::uint64_t hash,const std::vector<std::string>& trunk){
    require(&graph==graph_ && hash==hash_ && samePooledLookup(storage_,storage) &&
        c.device==device_ && c.prefix+"hash"==hash_producer_ && graph.getNode(hash_producer_) &&
        graph.getNode(hash_producer_)->stage->graphCaptureVariantSignature()==hash_ && c.nativeFormat==format_ && c.nativeRowElements==row_elements_ && c.nativeRowBytes==row_bytes_ &&
        c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary && !c.projectionRows &&
        b.ids.tensor==buffers_.ids.tensor && b.ids.buffer==buffers_.ids.buffer && b.ids.byteOffset==buffers_.ids.byteOffset &&
        b.embedding.tensor==buffers_.embedding.tensor && b.embedding.buffer==buffers_.embedding.buffer &&
        b.embedding.byteOffset==buffers_.embedding.byteOffset && !trunk.empty(),
        "Pooled read-ahead consume substituted its graph/query/source or was consumed twice");
    handoff_.consume(graph,hash,hash,trunk);return endpoints_.last;
}
void Qwen4ExpPooledReadAheadFrame::requireConsumed()const{
    handoff_.requireConsumed();
}
std::string Qwen4ExpPLEBuilder::append(ComputeGraph &graph,const Qwen4ExpPLEContext &c,
    const Qwen4ExpPLEWeights &w,const Qwen4ExpPLEBuffers &b,
    LookupStorage storage,
    const std::vector<std::string> &dependencies,Qwen4ExpEarlyLookupFrame* early,
    Qwen4ExpResidentReadAheadFrame* residentEarly,Qwen4ExpPooledReadAheadFrame* pooledEarly)
{
    require(c.device.is_valid() && !c.prefix.empty() && c.rows.capacity>0 && c.hidden>0 &&
        c.hidden<=std::numeric_limits<int>::max()/4 && c.nativeRowElements>0 &&
        c.nativeRowElements==c.hidden/qwen4exp::kNgramHeads && c.hidden%qwen4exp::kNgramHeads==0 &&
        std::isfinite(c.epsilon) && c.epsilon>0,
        "PLE graph requires exact participant geometry, sixteen native rows and branch epsilon");
    require(c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary ||
        c.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,"PLE projection arithmetic is invalid");
    if(c.projectionRows)
        require(c.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
            "PLE counted projections require explicit grouped-verifier arithmetic");
    if(c.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier)
    {
        // Validate the effective descriptor, so omitting it cannot replace a
        // primitive-owned count with fully-active GEMM rows. CPU declarations
        // also borrow their original native scalar; no count value is read here.
        const auto *native=c.rows.liveRows ? std::get_if<Qwen4ExpNativeIntInput>(&c.rows.liveRows->value) : nullptr;
        require(!c.rows.liveRows || (native && native->owner==c.device && native->capacity==1 && native->data),
            "PLE verifier projections require the exact native primitive query authority");
        const auto original=native ? DeviceRowRange::deviceCounted(c.rows.capacity,native->data) :
            DeviceRowRange::fullyActive(c.rows.capacity);
        const auto effective=c.projectionRows.value_or(DeviceRowRange::fullyActive(c.rows.capacity));
        require(effective.captureIdentity()==original.captureIdentity(),
            "PLE verifier projections require the complete original primitive row identity");
    }
    const std::array views{b.embedding,b.key,b.value,b.query,b.gated,b.normalized,b.priorConvolution,b.trunk};
    for(std::size_t i=0;i<views.size();++i)
        for(std::size_t j=i+1;j<views.size();++j)
            require(views[i].tensor!=views[j].tensor && views[i].buffer!=views[j].buffer,
                "PLE projections, retained histories and residual trunk must have independent storage");
    c.hash.validate();
    using Stage=Qwen4ExpPrimitiveStage;
    const auto hash=c.prefix+"hash",gather=c.prefix+"gather",key=c.prefix+"key",value=c.prefix+"value",
        keyNorm=c.prefix+"key_norm",queryNorm=c.prefix+"query_norm",gate=c.prefix+"gate",
        convNorm=c.prefix+"conv_norm",convolution=c.prefix+"convolution";
    std::vector<Pending> nodes;
    std::optional<ComputeGraph> pooledGraph;
    std::string lookupTerminal=gather;
    auto hashStage=std::make_unique<Stage>(Stage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Stage::NgramHash{b.tokens,b.priorTokens,b.ids,c.hash}});
    const auto hashIdentity=hashStage->graphCaptureVariantSignature();
    require(!pooledEarly || std::holds_alternative<Qwen4ExpPooledLookupStorage>(storage) || std::holds_alternative<Qwen4ExpPooledReaderLookupStorage>(storage),"Pooled read-ahead cannot consume another storage role");
    require(static_cast<int>(bool(early))+static_cast<int>(bool(residentEarly))+static_cast<int>(bool(pooledEarly))<=1,"PLE cannot own two early source modes");
    std::vector<std::string> residentConsumeDependencies;
    if(!early && !residentEarly && !pooledEarly)nodes.push_back({hash,std::move(hashStage),dependencies});
    if(residentEarly){
        auto* whole=std::get_if<ImmutableLookupStage::Storage>(&storage);
        require(whole && std::holds_alternative<ImmutableLookupStage::ResidentGPUStorage>(*whole),
            "Resident early PLE requires its original resident table");
        residentConsumeDependencies=residentEarly->consume(graph,ImmutableLookupStage::Params{
            .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,.ids=b.ids,.output=b.embedding,
            .format=c.nativeFormat,.rowElements=c.nativeRowElements,.rowBytes=c.nativeRowBytes,
            .storage=*whole},hashIdentity,dependencies);
    }
    else if(early){
        require(!dependencies.empty(),"Early lookup Consume requires an actual trunk terminal");
        auto* whole=std::get_if<ImmutableLookupStage::Storage>(&storage);
        require(whole && std::holds_alternative<ImmutableLookupStage::GPUStorage>(*whole),"Early current lookup only consumes its hosted source");
        auto consumption=early->consume(graph,ImmutableLookupStage::Params{
            .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,.ids=b.ids,.output=b.embedding,
            .format=c.nativeFormat,.rowElements=c.nativeRowElements,.rowBytes=c.nativeRowBytes,
            .storage=*whole},hashIdentity,dependencies);
        nodes.push_back({gather,std::move(consumption.stage),std::move(consumption.dependencies)});
    }
    else if(std::holds_alternative<Qwen4ExpPooledLookupStorage>(storage)||std::holds_alternative<Qwen4ExpPooledReaderLookupStorage>(storage)){
        const auto original=std::holds_alternative<Qwen4ExpPooledLookupStorage>(storage)
            ?Qwen4ExpPooledLookupDeclaration{std::get<Qwen4ExpPooledLookupStorage>(storage)}
            :Qwen4ExpPooledLookupDeclaration{std::get<Qwen4ExpPooledReaderLookupStorage>(storage)};
        if(pooledEarly)lookupTerminal=pooledEarly->consume(graph,c,b,original,hashIdentity,dependencies);
        else {pooledGraph.emplace();lookupTerminal=appendOriginalPooledLookup(*pooledGraph,c,b,original,gather).last;}
    }
    else if(auto *fixed=std::get_if<Qwen4ExpFixedHeadLookupStorage>(&storage))
    {
        const auto *native=c.rows.liveRows?std::get_if<Qwen4ExpNativeIntInput>(&c.rows.liveRows->value):nullptr;
        require(!c.rows.liveRows || (native && native->owner==c.device && native->capacity==1 && native->data),
            "Fixed-head PLE requires original native query authority");
        const auto original=native?DeviceRowRange::deviceCounted(c.rows.capacity,native->data):DeviceRowRange::fullyActive(c.rows.capacity);
        require(fixed->query.device()==c.device && fixed->query.rows().captureIdentity()==original.captureIdentity(),
            "Fixed-head PLE primitive rows differ from actual query producer");
        require(fixed->localHeader.byteOffset==0 && fixed->gatheredHeaders.byteOffset==0 &&
            fixed->localNative.byteOffset==0 && fixed->gatheredNative.byteOffset==0,
            "Fixed-head native collective banks require their complete arena roots");
        if(fixed->query.isWorkspaceDeclaration())graph.markWorkspaceDeclaration();
        auto frame=Qwen4ExpFixedHeadLookupStage::frame(*fixed,b.ids,b.embedding);
        const auto &policy=frame->kernels->plan().policy();
        require(policy.multipliers==c.hash.multipliers && policy.moduli==c.hash.moduli &&
            policy.offsets==c.hash.offsets && policy.eosToken==c.hash.eosToken && frame->kernels->plan().table().format()==c.nativeFormat &&
            frame->kernels->plan().table().rowElements()==c.nativeRowElements && frame->kernels->plan().table().rowBytes()==c.nativeRowBytes,
            "Fixed-head PLE source policy/codec differs from the original hash/decode contract");
        using Lookup=Qwen4ExpFixedHeadLookupStage;
        const auto validate=c.prefix+"validate_local",headers=c.prefix+"header_allgather",consensus=c.prefix+"validate_cohort",
            pack=c.prefix+"pack_native",payload=c.prefix+"payload_allgather";
        nodes.push_back({validate,std::make_unique<Lookup>(frame,Lookup::Operation::ValidateLocal,validate),{hash}});
        nodes.push_back({headers,std::make_unique<NativeAllGatherStage>(NativeAllGatherStage::Params{
            .device_id=c.device,.mpi_ctx=c.mpi,.tp_ctx=&fixed->cohort->context(),.local_input=fixed->localHeader.tensor,
            .rank_major_output=fixed->gatheredHeaders.tensor,.bytes_per_participant=sizeof(qwen4exp::FixedHeadValidationHeader),
            .participant=fixed->query.participant(),.stage_name=headers,
            .input_buffer_id=fixed->localHeader.buffer,.output_buffer_id=fixed->gatheredHeaders.buffer,
            .arena_rows=NativeAllGatherStage::ArenaRows{fixed->transportRows.tensor,fixed->transportRows.buffer,
                fixed->transportRows.byteOffset,1,sizeof(qwen4exp::FixedHeadValidationHeader)},
            .payload_receipt=DeviceGenerationCollectiveReceipt::fixedHead(fixed->control,
                DeviceGenerationCollectiveReceiptKind::FixedHeadHeader,c.rows.capacity,
                frame->kernels->plan().heads(),c.nativeRowBytes)}),{validate}});
        nodes.push_back({consensus,std::make_unique<Lookup>(frame,Lookup::Operation::ValidateCohort,consensus),{headers}});
        nodes.push_back({pack,std::make_unique<Lookup>(frame,Lookup::Operation::Pack,pack),{consensus}});
        const auto bytes=static_cast<std::size_t>(c.rows.capacity)*frame->kernels->plan().heads()*c.nativeRowBytes;
        nodes.push_back({payload,std::make_unique<NativeAllGatherStage>(NativeAllGatherStage::Params{
            .device_id=c.device,.mpi_ctx=c.mpi,.tp_ctx=&fixed->cohort->context(),.local_input=fixed->localNative.tensor,
            .rank_major_output=fixed->gatheredNative.tensor,.bytes_per_participant=bytes,
            .participant=fixed->query.participant(),.stage_name=payload,
            .input_buffer_id=fixed->localNative.buffer,.output_buffer_id=fixed->gatheredNative.buffer,
            .arena_rows=NativeAllGatherStage::ArenaRows{fixed->transportRows.tensor,fixed->transportRows.buffer,
                fixed->transportRows.byteOffset,c.rows.capacity,static_cast<std::size_t>(frame->kernels->plan().heads())*c.nativeRowBytes},
            .payload_receipt=DeviceGenerationCollectiveReceipt::fixedHead(fixed->control,
                DeviceGenerationCollectiveReceiptKind::FixedHeadPayload,c.rows.capacity,
                frame->kernels->plan().heads(),c.nativeRowBytes)}),{pack}});
        nodes.push_back({gather,std::make_unique<Lookup>(frame,Lookup::Operation::Decode,gather),{payload}});
    }
    else
        nodes.push_back({gather,std::make_unique<ImmutableLookupStage>(ImmutableLookupStage::Params{
            .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,.ids=b.ids,.output=b.embedding,
            .format=c.nativeFormat,.rowElements=c.nativeRowElements,.rowBytes=c.nativeRowBytes,
            .storage=std::get<ImmutableLookupStage::Storage>(storage)}),{hash}});
    nodes.push_back({key,project(c,w.key,b.embedding,b.key,4*c.hidden),residentEarly?residentConsumeDependencies:std::vector<std::string>{lookupTerminal}});
    // Key is the first joined reader; value remains on that exact primary
    // lane and follows key without any extra kernel, copy or global barrier.
    nodes.push_back({value,project(c,w.value,b.embedding,b.value,c.hidden),{residentEarly?key:lookupTerminal}});
    nodes.push_back({keyNorm,norm(c,b.key,w.keyNorm,b.key),{key}});
    nodes.push_back({queryNorm,norm(c,b.trunk,w.queryNorm,b.query),dependencies});
    nodes.push_back({gate,std::make_unique<Stage>(Stage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Stage::PLEGate{c.hidden,b.key,b.query,b.value,b.gated}}),{keyNorm,queryNorm,value}});
    nodes.push_back({convNorm,norm(c,b.gated,w.convolutionNorm,b.normalized),{gate}});
    nodes.push_back({convolution,std::make_unique<Stage>(Stage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Stage::PLEConv{4*c.hidden,b.normalized,b.priorConvolution,w.convolution,b.gated,b.trunk,b.trunk}}),{convNorm}});
    for(std::size_t i=0;i<nodes.size();++i)
    {
        require(!graph.getNode(nodes[i].name),"PLE node name collides with an existing producer");
        for(const auto &edge:nodes[i].dependencies)
        {
            bool exists=graph.getNode(edge)!=nullptr || (pooledGraph && pooledGraph->getNode(edge));
            for(std::size_t j=0;j<i;++j) exists=exists || nodes[j].name==edge;
            require(exists,"PLE graph dependency is absent or points to a later node");
        }
    }
    if(pooledGraph)
        for(const auto& name:pooledGraph->getExecutionOrder())
            require(!graph.getNode(name),"PLE pooled node name collides with an existing producer");
    // merge() carries a source fragment's optional lane. A shared reader has
    // only its original participant relation and must never invent that lane.
    if(auto *whole=std::get_if<ImmutableLookupStage::Storage>(&storage))
        if(auto *gpu=std::get_if<ImmutableLookupStage::GPUStorage>(whole))
            graph.setRetainedStorageBoundary(gpu->boundary);
    for(auto &node:nodes)
    {
        graph.addNode(node.name,std::move(node.stage),c.device);
        for(const auto &edge:node.dependencies) graph.addDependency(node.name,edge);
        if(pooledGraph && node.name==hash){graph.merge(std::move(*pooledGraph),hash);pooledGraph.reset();}
    }
    if(residentEarly)residentEarly->sealBranch(graph,key);
    return convolution;
}
} // namespace llaminar2
