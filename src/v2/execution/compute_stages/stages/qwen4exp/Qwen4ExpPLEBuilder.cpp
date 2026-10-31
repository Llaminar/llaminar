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
#include "../GEMMStage.h"
#include "../NativeAllGatherStage.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "../../../local_execution/graph/ComputeGraph.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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
std::string Qwen4ExpPLEBuilder::append(ComputeGraph &graph,const Qwen4ExpPLEContext &c,
    const Qwen4ExpPLEWeights &w,const Qwen4ExpPLEBuffers &b,
    LookupStorage storage,
    const std::vector<std::string> &dependencies)
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
    nodes.push_back({hash,std::make_unique<Stage>(Stage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Stage::NgramHash{b.tokens,b.priorTokens,b.ids,c.hash}}),dependencies});
    if(auto* pooled=std::get_if<Qwen4ExpPooledLookupStorage>(&storage))
    {
        require(pooled->arena && pooled->boundary && pooled->boundary->isPooled() && pooled->code &&
            pooled->boundary->device()==c.device,"PLE split lookup lost its original model-owned source domain");
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
        pooledGraph.emplace();
        const auto edge=appendPooledImmutablePLELookupGraph(*pooledGraph,*pooled->arena,pooled->boundary,pooled->code,
            original,{ids,b.ids.buffer,b.ids.byteOffset,std::size_t{expanded.physicalRows()}*sizeof(std::uint64_t)},
            std::move(countSpan),pooled->control,
            {output,b.embedding.buffer,b.embedding.byteOffset,std::size_t{expanded.physicalRows()}*c.nativeRowElements*sizeof(float)},
            c.prefix+"pooled",gather);
        lookupTerminal=edge.last;
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
    nodes.push_back({key,project(c,w.key,b.embedding,b.key,4*c.hidden),{lookupTerminal}});
    nodes.push_back({value,project(c,w.value,b.embedding,b.value,c.hidden),{lookupTerminal}});
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
    if(pooledGraph)graph.setRetainedStorageBoundary(pooledGraph->retainedStorageBoundary());
    if(auto *whole=std::get_if<ImmutableLookupStage::Storage>(&storage))
        if(auto *gpu=std::get_if<ImmutableLookupStage::GPUStorage>(whole))
            graph.setRetainedStorageBoundary(gpu->boundary);
    for(auto &node:nodes)
    {
        graph.addNode(node.name,std::move(node.stage),c.device);
        for(const auto &edge:node.dependencies) graph.addDependency(node.name,edge);
        if(pooledGraph && node.name==hash){graph.merge(std::move(*pooledGraph),hash);pooledGraph.reset();}
    }
    return convolution;
}
} // namespace llaminar2
