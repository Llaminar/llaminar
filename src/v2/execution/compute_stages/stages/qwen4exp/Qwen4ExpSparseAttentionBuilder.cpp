/**
 * @file Qwen4ExpSparseAttentionBuilder.cpp
 * @brief Wire exact split GGUF indexer projections into native sparse attention.
 *
 * Every pending stage and graph edge is validated before publishing graph
 * nodes. Persistent pools remain transaction-prefix state; this composition
 * does not commit an acceptance or advance any request/cache frontier.
 */
#include "Qwen4ExpSparseAttentionBuilder.h"
#include "../GEMMStage.h"
#include "../../../../models/GraphTypes.h"
#include "../../../local_execution/graph/ComputeGraph.h"
#include "../../../local_execution/graph/IGraphBuilder.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace llaminar2
{
namespace
{
/** @brief Reject malformed model geometry or incomplete native ownership at construction. */
void require(bool valid,const char *message) { if(!valid) throw std::invalid_argument(message); }
/** @brief An ordinary flat graph node retained until the entire extension validates. */
struct Pending
{
    std::string name;
    std::unique_ptr<IComputeStage> stage;
    std::vector<std::string> dependencies;
};
/** @brief Preserve exact source/store identity and canonical verifier arithmetic. */
std::unique_ptr<IComputeStage> projection(const AttentionConsumerBinding &c,const Qwen4ExpSparseAttentionPolicy &p,
    const Qwen4ExpPreparedProjection &weight,Qwen4ExpFloatBinding output,int width,DeviceRequestRowRanges rows)
{
    require(output.byteOffset==0,"QSA prepared projection requires a native zero-offset output tensor view");
    require(weight.ref().device==c.device,"QSA indexer projection belongs to a different participant");
    weight.requireGeometry(width,p.hidden);
    return std::make_unique<GEMMStage>(GEMMStage::Params{
        .device_id=c.device,.mpi_ctx=p.mpi,.A=&c.projectionInput.tensor,.B=weight.tensor(),.C=output.tensor,
        .m=c.queryRows,.n=width,.k=p.hidden,.alpha=1.0f,.beta=0.0f,.transpose_B=false,
        .a_buffer_id=c.projectionInput.buffer,.c_buffer_id=output.buffer,
        .force_decode_equivalent_verifier_prefill=p.projectionArithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
        .request_rows=p.projectionArithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier ?
            std::optional<DeviceRequestRowRanges>{rows} : std::nullopt,.prepared_ref=weight.ref(),.prepared_store=&weight.store()});
}
/** @brief One admitted request-local byte interval, including borrowed producer metadata. */
struct RequestLease
{
    const TensorBase *tensor; ///< Borrowed canonical admitted tensor owner.
    BufferId buffer; ///< Arena/coherence identity, shared by offset subviews.
    std::size_t offset, extent; ///< Exact byte interval retained by one request.
};
/**
 * @brief Validate all request-local lifetimes before stage construction or graph mutation.
 * @param consumer Canonical native attention geometry and compute/output tensor owners.
 * @param policy Immutable admitted indexer block capacity.
 * @param rows Sole request-count and physical-row-stride authority.
 * @param requests Explicit cache sequences and request-local arena subviews.
 * @throws std::invalid_argument If any interval exceeds storage or overlaps a sibling lifetime.
 *
 * Selection outputs, core partials, traffic counters and canonical row metadata
 * are live concurrently across independent request branches. Checking their full
 * admitted intervals here rejects aliases without consulting live host counts.
 */
void validateRequestLeases(const AttentionConsumerBinding &consumer,
    const Qwen4ExpSparseAttentionPolicy &policy,
    DeviceRequestRowRanges rows,const std::vector<Qwen4ExpSparseAttentionRequest> &requests)
{
    require(consumer.queryHeads>0 && consumer.queryHeads<=24 && policy.blockCapacity>0,
        "QSA request-local leases require positive canonical head/block geometry");
    std::vector<std::vector<RequestLease>> leases(requests.size());
    const std::size_t m=rows.rowsPerRequest();
    for(std::size_t request=0;request<requests.size();++request) {
        const auto &b=requests[request].buffers;
        const auto add=[&](const auto &view,std::size_t extent) {
            require(view.tensor && view.byteOffset<=view.tensor->size_bytes() &&
                extent<=view.tensor->size_bytes()-view.byteOffset,
                "QSA request-local lease exceeds its admitted tensor extent");
            leases[request].push_back({view.tensor,view.buffer,view.byteOffset,extent});
        };
        if(b.retained) {
            const auto prior=b.retained->transactionPrior.storage;
            require(b.replay.tensor==prior.tensor && b.replay.buffer==prior.buffer && b.replay.byteOffset==prior.byteOffset,
                "QSA retained replay must be the immutable transaction prior header");
            add(prior,sizeof(qwen4exp::QwenSparseTransactionPrior));
        } else add(b.replay,sizeof(qwen4exp::QwenSparseReplay));
        add(b.query,m*512*sizeof(float));add(b.rawKey,m*128*sizeof(float));
        add(b.priorPartial,4*128*sizeof(float));
        add(b.pooledKeys,std::size_t(policy.blockCapacity)*128*sizeof(float));
        add(b.preparedQuery,m*512*sizeof(float));add(b.scores,m*policy.blockCapacity*sizeof(float));
        add(b.selected,m*qwen4exp::QwenSparseSelected*sizeof(int));add(b.selectedCount,m*sizeof(int));
        if(b.rowParameters)add(*b.rowParameters,m*sizeof(attention::AttentionDeviceParams));
        if(b.gpuPartials)add(*b.gpuPartials,m*consumer.queryHeads*qwen4exp::QwenSparsePartitions*
            sizeof(qwen4exp::QwenSparseAttentionPartial));
        if(b.observations)add(*b.observations,m*consumer.queryHeads*sizeof(qwen4exp::QwenSparseAttentionObservation));
        const std::size_t coreOffset=request*m*consumer.queryHeads*256*sizeof(float);
        add(Qwen4ExpFloatBinding(consumer.preparedQuery.tensor,consumer.preparedQuery.buffer,coreOffset),
            m*consumer.queryHeads*256*sizeof(float));
        add(Qwen4ExpFloatBinding(consumer.output.tensor,consumer.output.buffer,coreOffset),
            m*consumer.queryHeads*256*sizeof(float));
        if(b.retained) {
            add(b.retained->rawKey,m*128*sizeof(float));add(b.retained->copiedBytes,sizeof(std::uint64_t));
        }
        for(std::size_t left=0;left<leases[request].size();++left)
            for(std::size_t right=left+1;right<leases[request].size();++right) {
                const auto &a=leases[request][left];const auto &c=leases[request][right];
                if(a.tensor==c.tensor || a.buffer==c.buffer)
                    require(a.offset>=c.offset+c.extent || c.offset>=a.offset+a.extent,
                        "QSA request-local output, partial, observation and metadata leases overlap");
            }
        for(std::size_t prior=0;prior<request;++prior) {
            require(requests[prior].sequence!=requests[request].sequence,
                "QSA requests require distinct canonical cache sequences");
            for(const auto &a:leases[prior])for(const auto &c:leases[request])
                if(a.tensor==c.tensor || a.buffer==c.buffer)
                    require(a.offset>=c.offset+c.extent || c.offset>=a.offset+a.extent,
                        "QSA requests require disjoint output, partial, observation and metadata leases");
        }
    }
}
/** @brief Check exact graph producers and unique names before any mutation. */
void validateNodes(const ComputeGraph &graph,const std::vector<Pending> &nodes)
{
    for(std::size_t index=0;index<nodes.size();++index) {
        require(!nodes[index].name.empty() && !graph.getNode(nodes[index].name),"QSA graph extension collides with an existing node name");
        for(std::size_t earlier=0;earlier<index;++earlier)
            require(nodes[earlier].name!=nodes[index].name,"QSA graph extension repeats a node identity");
        for(const auto &dependency : nodes[index].dependencies) {
            bool exists=graph.getNode(dependency)!=nullptr;
            for(std::size_t earlier=0;earlier<index;++earlier)exists=exists || nodes[earlier].name==dependency;
            require(exists,"QSA graph requires exact existing or earlier producer dependencies");
        }
    }
}
}
qwen4exp::QwenSparsePoolSelection Qwen4ExpSparseAttentionBuilder::poolSelectionForInvocation(
    ForwardExecutionPhase phase,DeviceId device,int rowCapacity)
{
    require(device.is_valid() && rowCapacity>0,
        "QSA pool selection requires an exact participant and positive captured row capacity");
    switch(phase){
    case ForwardExecutionPhase::Prefill:
    case ForwardExecutionPhase::Decode:break;
    default:throw std::invalid_argument("QSA pool selection requires a known forward phase");}
    switch(device.type){
    case DeviceType::CPU:
    case DeviceType::CUDA:
    case DeviceType::ROCm:
    case DeviceType::Vulkan:
    case DeviceType::Metal:break;
    default:throw std::invalid_argument("QSA pool selection requires a known participant backend");}
    if(phase==ForwardExecutionPhase::Decode && device.is_rocm() &&
        rowCapacity<=attention::kMaxGroupedVerifierAttentionRows)
        return qwen4exp::QwenSparsePoolSelection::Bounded1024_1024;
    return qwen4exp::QwenSparsePoolSelection::Installed;
}

std::optional<DeviceRowRange> Qwen4ExpSparseAttentionBuilder::verifierProjectionRows(
    DeviceId device,Qwen4ExpStageRows rows)
{
    if(!device.is_valid() || rows.capacity<=0)
        throw std::invalid_argument("QSA verifier projection rows require exact positive participant geometry");
    if(!rows.liveRows)return std::nullopt;
    const auto* native=std::get_if<Qwen4ExpNativeIntInput>(&rows.liveRows->value);
    if(!native || native->owner!=device || !native->data || native->capacity<1)
        throw std::invalid_argument("QSA verifier projection rows require the exact native execution-authority count");
    if(device.is_gpu())return DeviceRowRange::deviceCounted(rows.capacity,native->data);
    if(*native->data!=rows.capacity)
        throw std::invalid_argument("QSA CPU verifier projection rows must be the actual full invocation shape");
    return std::nullopt;
}

std::string Qwen4ExpSparseAttentionBuilder::append(ComputeGraph &graph,const AttentionConsumerBinding &c,
    const Qwen4ExpSparseAttentionPolicy &p,const Qwen4ExpSparseAttentionWeights &w,
    const Qwen4ExpSparseAttentionBuffers &b)
{
    require(c.requests==1,"QSA single-request composition requires one exact request");
    if(p.projectionRows)require(p.projectionRows->physicalRows()==c.queryRows &&
        p.projectionRows->capacity()==c.queryRows,"QSA single-request prefix must cover the complete matrix");
    const auto rows=p.projectionRows && p.projectionRows->countOwner() ?
        DeviceRequestRowRanges::deviceCounted(1,c.queryRows,p.projectionRows->countOwner()) :
        DeviceRequestRowRanges::fullyActive(c.queryRows);
    return appendRequests(graph,c,p,w,rows,{{0,b}}).front();
}
std::vector<std::string> Qwen4ExpSparseAttentionBuilder::appendRequests(ComputeGraph &graph,const AttentionConsumerBinding &c,
    const Qwen4ExpSparseAttentionPolicy &p,const Qwen4ExpSparseAttentionWeights &w,
    DeviceRequestRowRanges rows,const std::vector<Qwen4ExpSparseAttentionRequest> &requests)
{
    c.requireLayerIdentity();
    c.layerIdentity.requireSelectedConsumer();
    require(c.device.is_valid() && !c.prefix.empty() && c.cache && c.modelLayer>=0 && c.cacheLayer>=0 &&
        c.queryRows>0 && c.requests==rows.requests() && c.queryRows==rows.physicalRows() &&
        requests.size()==std::size_t(rows.requests()) && p.hidden>0 && p.tokenCapacity==c.cache->max_seq_len(),
        "QSA graph requires exact canonical request geometry, native cache and admitted arena views");
    require(c.kvHeads==c.cache->local_n_kv_heads() && c.kvHeadStart==c.cache->kv_head_start(),
        "QSA graph native KV shard does not match its declared cache authority");
    require(p.projectionArithmetic==Qwen4ExpProjectionArithmetic::Ordinary ||
        p.projectionArithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,"QSA projection arithmetic policy is invalid");
    if(p.projectionRows)require(rows.requests()==1 && p.projectionRows->physicalRows()==c.queryRows &&
        p.projectionRows->capacity()==c.queryRows && p.projectionRows->countOwner()==rows.lengthOwner(),
        "QSA prefix metadata must be the same complete single-request count authority");
    require(!rows.lengthOwner() || p.projectionArithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
        "Counted QSA projections require an explicitly selected decode-equivalent floating policy");
    require(p.publication==Qwen4ExpSparsePublication::AcceptAllLiveRows ||
        p.publication==Qwen4ExpSparsePublication::DeferredVerifierAcceptance,"QSA publication policy is invalid");
    // Reject policy corruption before preparing projections or publishing nodes.
    qwen4exp::validateQwenSparsePoolSelection(p.poolSelection, rows.rowsPerRequest());
    qwen4exp::validateQwenSparseAttentionArithmetic(p.attentionArithmetic,c.device.type);
    validateRequestLeases(c,p,rows,requests);
    const auto &first=requests.front().buffers;
    const auto queryName=c.prefix+"qsa_query_proj",keyName=c.prefix+"qsa_key_proj";
    std::vector<Pending> nodes;
    std::vector<std::string> projectionDependencies{c.projectionInputReady};
    for(const auto &request:requests)projectionDependencies.push_back(request.buffers.replayReady);
    nodes.push_back({queryName,projection(c,p,w.query,Qwen4ExpFloatBinding(*first.query.tensor,first.query.buffer),512,rows),projectionDependencies});
    nodes.push_back({keyName,projection(c,p,w.rawKey,Qwen4ExpFloatBinding(*first.rawKey.tensor,first.rawKey.buffer),128,rows),projectionDependencies});
    std::vector<std::string> terminals;
    using Stage=QwenSparseAttentionStage;
    const int rowCapacity=rows.rowsPerRequest();
    for(int requestIndex=0;requestIndex<rows.requests();++requestIndex) {
        const auto &request=requests[requestIndex];const auto &b=request.buffers;
        const auto firstRow=std::size_t(requestIndex)*rowCapacity;
        require(request.sequence>=0 && request.sequence<c.cache->request_capacity() &&
            !b.replayReady.empty() && (!c.device.is_gpu() || !b.rowParametersReady.empty()),
            "QSA request requires an admitted independent cache slot and exact metadata producers");
        require(b.query.tensor==first.query.tensor && b.query.buffer==first.query.buffer && b.query.byteOffset==firstRow*512*sizeof(float) &&
            b.rawKey.tensor==first.rawKey.tensor && b.rawKey.buffer==BufferId::QFN_QSA_RAW_KEY && b.rawKey.byteOffset==firstRow*128*sizeof(float),
            "QSA flat split GEMMs require exact shared projection output slices");
        require((p.publication==Qwen4ExpSparsePublication::DeferredVerifierAcceptance)==b.retained.has_value(),
            "QSA retained transaction bank must match its declared acceptance lifetime");
        const std::array retainedViews{Qwen4ExpFloatBinding(c.projectionInput.tensor,c.projectionInput.buffer),
            Qwen4ExpFloatBinding(c.preparedQuery.tensor,c.preparedQuery.buffer),Qwen4ExpFloatBinding(c.output.tensor,c.output.buffer),
            b.query,b.rawKey,b.priorPartial,b.pooledKeys,b.preparedQuery,b.scores};
        for(std::size_t left=0;left<retainedViews.size();++left)for(std::size_t right=left+1;right<retainedViews.size();++right)
            require(retainedViews[left].tensor!=retainedViews[right].tensor && retainedViews[left].buffer!=retainedViews[right].buffer,
                "QSA projections, prepared core query, pool state and scratch must retain disjoint lifetimes");
        const auto prefix=rows.requests()==1 ? c.prefix : c.prefix+"request"+std::to_string(requestIndex)+"_";
        const auto indexerName=prefix+"qsa_indexer",attentionName=prefix+"attention";
        std::optional<std::string> retentionName;
        if(b.retained) {
            retentionName=prefix+"qsa_retain_candidates";
            nodes.push_back({*retentionName,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=p.mpi,
                .rowCapacity=rowCapacity,.tokenCapacity=p.tokenCapacity,.replay=b.replay,
                .operation=Stage::RetainCandidates{b.rawKey,b.retained->rawKey,b.retained->copiedBytes,b.priorPartial,
                    requestIndex,rows.requests(),b.retained->verifierOrigins,b.retained->transactionPrior,b.retained->terminalCounters}}),{keyName,b.replayReady}});
        }
        nodes.push_back({indexerName,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=p.mpi,
            .rowCapacity=rowCapacity,.tokenCapacity=p.tokenCapacity,.replay=b.replay,
            .operation=Stage::Indexer{b.query,b.rawKey,b.priorPartial,b.pooledKeys,w.queryNorm,w.keyNorm,w.cosine,w.sine,
                b.preparedQuery,b.scores,b.selected,b.selectedCount,p.blockCapacity,p.positionCapacity,p.rotaryDimension,
                p.epsilon,p.representation,p.rotarySource,p.ropeTheta,b.retained ? std::optional{b.retained->transactionPrior} : std::nullopt,
                p.poolSelection}}),{queryName,keyName,b.replayReady}});
        const auto coreOffset=firstRow*c.queryHeads*256*sizeof(float);
        Stage::SelectedAttention attention{{*c.cache,c.cacheLayer,request.sequence,c.keyPolicy},
            Qwen4ExpFloatBinding(c.preparedQuery.tensor,c.preparedQuery.buffer,coreOffset),b.selected,b.selectedCount,
            Qwen4ExpFloatBinding(c.output.tensor,c.output.buffer,coreOffset),{},b.rowParameters,b.gpuPartials,b.observations,
            {},{},0,p.rotaryDimension,c.queryHeads,c.kvHeads,c.queryHeadStart,c.kvHeadStart};
        attention.arithmetic=p.attentionArithmetic;
        attention.keyRotarySource=p.rotarySource==qwen4exp::QwenSparseRotarySource::AnalyticTheta ?
            qwen4exp::QwenSparseRotarySource::CanonicalEngineTheta : p.rotarySource;
        if(c.keyPolicy.transformsOnRead() && p.rotarySource==qwen4exp::QwenSparseRotarySource::TablePlanes) {
            attention.keyCosine=w.cosine;attention.keySine=w.sine;attention.keyPositionCapacity=p.positionCapacity;
        }
        std::vector<std::string> dependencies{indexerName,c.preparedQueryReady,c.nativeCacheReady,b.replayReady};
        if(c.device.is_gpu())dependencies.push_back(b.rowParametersReady);
        if(retentionName)dependencies.push_back(*retentionName);
        nodes.push_back({attentionName,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=p.mpi,
            .rowCapacity=rowCapacity,.tokenCapacity=p.tokenCapacity,.replay=b.replay,.operation=attention}),std::move(dependencies)});
        auto terminal=attentionName;
        if(p.publication==Qwen4ExpSparsePublication::AcceptAllLiveRows) {
            auto live=b.replay;live.byteOffset+=sizeof(int);
            terminal=prefix+"qsa_commit_partial";
            nodes.push_back({terminal,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=p.mpi,
                .rowCapacity=rowCapacity,.tokenCapacity=p.tokenCapacity,.replay=b.replay,
                .operation=Stage::CommitPartial{b.priorPartial,b.rawKey,live,b.priorPartial}}),{attentionName}});
        }
        terminals.push_back(terminal);
    }
    validateNodes(graph,nodes);
    for(auto &node:nodes) {
        graph.addNode(node.name,std::move(node.stage),c.device);
        for(const auto &dependency:node.dependencies)graph.addDependency(node.name,dependency);
    }
    return terminals;
}
} // namespace llaminar2
