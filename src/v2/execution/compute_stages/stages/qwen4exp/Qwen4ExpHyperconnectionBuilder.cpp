/**
 * @file Qwen4ExpHyperconnectionBuilder.cpp
 * @brief Wire typed prepared GEMMs and GR stages without model-specific orchestration.
 *
 * All nodes are constructed and validated before graph mutation. Projection
 * weights retain raw checkpoint scaling: residualDown applies /4 before SiLU,
 * and residualWrite applies /4 before 2*sigmoid. Read and injection projections
 * may overlap block compute only through explicit graph dependencies. No
 * collective, activation allocation, packing or backend selection is hidden in this builder.
 */
#include "Qwen4ExpHyperconnectionBuilder.h"
#include "models/qwen4exp/Qwen4ExpHCUpArithmetic.h"
#include "../GEMMStage.h"
#include "../../../../loaders/PreparedWeightStore.h"
#include "../../../local_execution/graph/ComputeGraph.h"
#include "../../../../tensors/Tensors.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
namespace
{
/** @brief Fail malformed graph setup before publishing any new graph nodes. */
void require(bool condition,const char *message)
{
    if(!condition)throw std::invalid_argument(message);
}
/** @brief Prepared ordinary node and explicit predecessor identities. */
struct PendingNode
{
    std::string name;
    std::unique_ptr<IComputeStage> stage;
    std::vector<std::string> dependencies;
};
/** @brief Reject names/dependencies that cannot form this participant-local extension. */
void validateGraphNames(const ComputeGraph &graph,const std::vector<PendingNode> &pending)
{
    for(std::size_t index=0;index<pending.size();++index)
    {
        require(!graph.getNode(pending[index].name),"Qwen4Exp mixer node prefix collides with an existing graph node");
        for(std::size_t earlier=0;earlier<index;++earlier)
            require(pending[earlier].name!=pending[index].name,"Qwen4Exp mixer creates duplicate node identities");
        for(const auto &dependency:pending[index].dependencies)
        {
            bool exists=graph.getNode(dependency)!=nullptr;
            for(std::size_t earlier=0;earlier<index;++earlier)exists=exists||pending[earlier].name==dependency;
            require(exists,"Qwen4Exp mixer requires an existing or earlier explicit dependency producer");
        }
    }
}
/** @brief Install a fully validated flat extension using the graph's existing node ownership. */
void install(ComputeGraph &graph,std::vector<PendingNode> pending,DeviceId device)
{
    validateGraphNames(graph,pending);
    for(auto &node:pending)
    {
        graph.addNode(node.name,std::move(node.stage),device);
        for(const auto &dependency:node.dependencies)graph.addDependency(node.name,dependency);
    }
}
/**
 * @brief Reject ambiguous lifetimes and noncanonical GEMM row authorities.
 * @param c Immutable primitive/common/up-only query declarations for this mixer.
 * @param up_arithmetic Actual operation resolved from this mixer’s original up source.
 * @throws std::invalid_argument Before graph mutation for invalid policy, geometry or different live-count owners.
 */
void validateContext(const Qwen4ExpHyperconnectionContext &c,Qwen4ExpHCUpArithmetic up_arithmetic)
{
    require(c.device.is_valid()&&!c.prefix.empty()&&c.rows.capacity>0&&c.hidden>0&&c.hidden<=std::numeric_limits<int>::max()/4&&
                c.lowRank>0&&std::isfinite(c.epsilon)&&c.epsilon>0,"Qwen4Exp mixer requires explicit participant, names and positive geometry");
    require(c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary||
                c.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,"Qwen4Exp mixer projection arithmetic policy is invalid");
    if(c.projectionRows)
    {
        require(c.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
                "Qwen4Exp mixer count-scoped projections require explicit decode-equivalent arithmetic");
        require(c.projectionRows->physicalRows()==c.rows.capacity&&c.projectionRows->capacity()==c.rows.capacity,
                "Qwen4Exp mixer projection row contract must match admitted matrix geometry");
    }
    require(!c.upProjectionRows || up_arithmetic==Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8,
            "Qwen4Exp up-only query rows require an actual two-term native quantized source operation");
    if(up_arithmetic==Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8)
    {
        // Validate the effective alternative, including legacy predictor common
        // rows and an omitted descriptor. No optional-field combination may
        // replace a counted primitive authority with fixed or expanded rows.
        const auto up=c.upProjectionRows ? *c.upProjectionRows :
            c.projectionRows.value_or(DeviceRowRange::fullyActive(c.rows.capacity));
        require(up.physicalRows()==c.rows.capacity && up.capacity()==c.rows.capacity,
                "Qwen4Exp up query rows exceed the original primitive capacity");
        const auto *native=c.rows.liveRows ? std::get_if<Qwen4ExpNativeIntInput>(&c.rows.liveRows->value) : nullptr;
        require(!c.rows.liveRows || (native && native->owner==c.device && native->capacity==1 && native->data),
                "Qwen4Exp up query must retain its exact native primitive authority");
        const auto original=native ? DeviceRowRange::deviceCounted(c.rows.capacity,native->data)
                                   : DeviceRowRange::fullyActive(c.rows.capacity);
        require(up.captureIdentity()==original.captureIdentity(),
                "Qwen4Exp up query must retain the complete original primitive row identity");
        if(c.projectionRows)
            require(c.projectionRows->captureIdentity()==up.captureIdentity(),
                    "Qwen4Exp common verifier and up query name different authorities");
    }
}
/** @brief Distinguish exact permitted aliasing from different logical arena identities. */
bool aliases(Qwen4ExpFloatBinding a,Qwen4ExpFloatBinding b)
{
    return a.tensor==b.tensor||a.buffer==b.buffer;
}
/**
 * @brief Bind one canonical prepared GEMM without reimplementing matrix arithmetic.
 * @param c Frozen participant, arithmetic and authenticated query descriptors.
 * @param weight Original native matrix and its actual prepared authority.
 * @param input Existing source arena role, with no alternate payload.
 * @param output Existing destination arena role.
 * @param n Original projection output columns.
 * @param k Original projection reduction columns.
 * @param precision Explicit operation policy; only the up operation may select TwoTermQ8.
 * @return Actual GEMMStage retaining the original source, query and arithmetic.
 * @throws std::invalid_argument For foreign preparation, noncanonical byte views or invalid stage geometry.
 */
std::unique_ptr<IComputeStage> projection(const Qwen4ExpHyperconnectionContext &c,
    const Qwen4ExpPreparedProjection &weight,Qwen4ExpFloatBinding input,Qwen4ExpFloatBinding output,int n,int k,
    QuantizedActivationOperandPrecision precision=QuantizedActivationOperandPrecision::SingleQ8)
{
    require(input.byteOffset==0&&output.byteOffset==0,
            "Qwen4Exp mixer GEMM views require zero byte offsets or prebuilt typed TensorSlice views");
    require(weight.ref().device==c.device,"Qwen4Exp mixer projection belongs to another participant");
    weight.requireGeometry(n,k);
    return std::make_unique<GEMMStage>(GEMMStage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.A=input.tensor,.B=weight.tensor(),.C=output.tensor,
        .m=c.rows.capacity,.n=n,.k=k,.alpha=1.0f,.beta=0.0f,.transpose_B=false,
        .a_buffer_id=input.buffer,.c_buffer_id=output.buffer,
        .force_decode_equivalent_verifier_prefill=c.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
        .verifier_row_range=precision==QuantizedActivationOperandPrecision::TwoTermQ8 && c.upProjectionRows
            ? c.upProjectionRows : c.projectionRows,.prepared_ref=weight.ref(),.prepared_store=&weight.store(),
        .quantized_operand_precision=precision});
}
/** @brief Construct the complete read chain before installing it into a live graph. */
std::vector<PendingNode> readNodes(const Qwen4ExpHyperconnectionContext &c,
    const Qwen4ExpHyperconnectionReadWeights &w,const Qwen4ExpHyperconnectionReadBuffers &b,
    const std::vector<std::string> &dependencies)
{
    const auto up_arithmetic=resolveQwen4ExpHCUpArithmetic(
        c.upOperandPrecision,tensorTypeName(w.up.tensor()->native_type()));
    validateContext(c,up_arithmetic);
    // Floating sources have their own native operation and never acquire Q8
    // workspace. Both quantized choices bind the unchanged native source.
    QuantizedActivationOperandPrecision up_precision;
    switch(up_arithmetic)
    {
    case Qwen4ExpHCUpArithmetic::FloatingNative:
    case Qwen4ExpHCUpArithmetic::QuantizedSingleQ8:
        up_precision=QuantizedActivationOperandPrecision::SingleQ8;
        break;
    case Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8:
        up_precision=QuantizedActivationOperandPrecision::TwoTermQ8;
        break;
    default:
        throw std::invalid_argument("Qwen4Exp mixer has an invalid source arithmetic operation");
    }
    const std::array views{b.trunk,b.normalized,b.down,b.activatedDown,b.readLogits,b.mixed};
    for(std::size_t i=0;i<views.size();++i)
        for(std::size_t j=i+1;j<views.size();++j)
            if(aliases(views[i],views[j]))
                require(i==2&&j==3&&views[i].tensor==views[j].tensor&&views[i].buffer==views[j].buffer&&
                            views[i].byteOffset==views[j].byteOffset,
                        "Qwen4Exp mixer must retain original/normalized branches and independent projection temporaries");
    using Stage=Qwen4ExpPrimitiveStage;
    const auto norm=c.prefix+"branch_norm",down=c.prefix+"read_down",activation=c.prefix+"read_activation",
               up=c.prefix+"read_up",read=c.readNodeName.empty()?c.prefix+"read":c.readNodeName;
    std::vector<PendingNode> nodes;
    nodes.push_back({norm,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Stage::BranchRMSNorm{c.hidden,c.epsilon,b.trunk,w.norm,b.normalized}}),dependencies});
    nodes.push_back({down,projection(c,w.down,b.normalized,b.down,c.lowRank,4*c.hidden),{norm}});
    nodes.push_back({activation,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Stage::ResidualDown{c.lowRank,b.down,b.activatedDown}}),{down}});
    nodes.push_back({up,projection(c,w.up,b.activatedDown,b.readLogits,4*c.hidden,c.lowRank,up_precision),{activation}});
    nodes.push_back({read,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Stage::ResidualRead{c.hidden,b.normalized,b.readLogits,b.mixed}}),{up,norm}});
    return nodes;
}
} // namespace

Qwen4ExpPreparedProjection::Qwen4ExpPreparedProjection(const WeightBinding &binding,PreparedWeightStore &store,DeviceId device)
    :tensor_(binding.tensor),store_(&store),ref_(binding.prepared.value_or(PreparedWeightRef{}))
{
    require(binding.immutable&&tensor_&&binding.prepared&&ref_.binding_id!=0&&ref_.kind!=PreparedWeightKind::None&&
                device.is_valid()&&ref_.device==device&&store.contains(ref_),
            "Qwen4Exp projection requires an immutable exact prepared binding on its participant");
    const bool kindMatches=(device.is_cpu()&&ref_.kind==PreparedWeightKind::CpuPackedGemm)||
                           (device.is_cuda()&&ref_.kind==PreparedWeightKind::CudaInt8PackedGemm)||
                           (device.is_rocm()&&ref_.kind==PreparedWeightKind::RocmInt8PackedGemm);
    require(kindMatches&&store.gemmKernel(ref_)!=nullptr,
            "Qwen4Exp projection requires an executable prepared GEMM of the selected backend kind");
    const auto owned=store.binding(ref_);
    require(owned&&owned->tensor==tensor_&&owned->binding_id==binding.binding_id&&
                owned->identity==binding.identity,"Qwen4Exp projection source/store identity does not match the frozen binding");
}
void Qwen4ExpPreparedProjection::requireGeometry(int output,int input) const
{
    require(output>0&&input>0&&tensor_->shape().size()==2&&tensor_->shape()[0]==static_cast<std::size_t>(output)&&
                tensor_->shape()[1]==static_cast<std::size_t>(input),"Qwen4Exp prepared projection requires checkpoint-native [output,input] geometry");
}
Qwen4ExpHyperconnectionBuilder::BlockRead::BlockRead(ComputeGraph &graph,Qwen4ExpHyperconnectionContext context,
    Qwen4ExpHyperconnectionReadBuffers buffers,Qwen4ExpPreparedProjection inject,std::string terminal)
    :graph_(&graph),context_(std::move(context)),buffers_(buffers),inject_(inject),terminal_(std::move(terminal)) {}
Qwen4ExpHyperconnectionBuilder::BlockRead::BlockRead(BlockRead &&other) noexcept
    :graph_(std::exchange(other.graph_,nullptr)),context_(std::move(other.context_)),buffers_(other.buffers_),
     inject_(other.inject_),terminal_(std::move(other.terminal_)) {}

Qwen4ExpHyperconnectionBuilder::BlockRead Qwen4ExpHyperconnectionBuilder::appendRead(ComputeGraph &graph,
    const Qwen4ExpHyperconnectionContext &c,const Qwen4ExpHyperconnectionBlockWeights &w,
    const Qwen4ExpHyperconnectionReadBuffers &b,const std::vector<std::string> &dependencies)
{
    validateContext(c,resolveQwen4ExpHCUpArithmetic(
        c.upOperandPrecision,tensorTypeName(w.read.up.tensor()->native_type())));
    w.inject.requireGeometry(4,4*c.hidden);
    require(w.inject.ref().device==c.device,"Qwen4Exp paired injection belongs to another participant");
    auto nodes=readNodes(c,w.read,b,dependencies);
    const auto terminal=nodes.back().name;
    install(graph,std::move(nodes),c.device);
    return BlockRead(graph,c,b,w.inject,terminal);
}
std::string Qwen4ExpHyperconnectionBuilder::appendFinalRead(ComputeGraph &graph,
    const Qwen4ExpHyperconnectionContext &c,const Qwen4ExpHyperconnectionReadWeights &w,
    const Qwen4ExpHyperconnectionReadBuffers &b,const std::vector<std::string> &dependencies)
{
    auto nodes=readNodes(c,w,b,dependencies);
    const auto terminal=nodes.back().name;
    install(graph,std::move(nodes),c.device);
    return terminal;
}
std::string Qwen4ExpHyperconnectionBuilder::appendWrite(ComputeGraph &graph,BlockRead &&read,
    const Qwen4ExpHyperconnectionWriteBuffers &b,const std::vector<std::string> &dependencies)
{
    require(read.graph_==&graph,"Qwen4Exp paired write requires an unconsumed read capability from this graph");
    const auto &c=read.context_;
    require(!dependencies.empty(),"Qwen4Exp paired write requires the block-output producer edge");
    for(auto retained:std::array{read.buffers_.trunk,read.buffers_.normalized})
    {
        require(!aliases(retained,b.block)&&!aliases(retained,b.injectionLogits),
                "Qwen4Exp paired write must retain original and normalized branches until injection completes");
        if(aliases(retained,b.outputTrunk))
            require(retained.tensor==read.buffers_.trunk.tensor&&retained.tensor==b.outputTrunk.tensor&&
                        retained.byteOffset==b.outputTrunk.byteOffset,
                    "Qwen4Exp updated trunk may alias only the exact original trunk view");
    }
    require(!aliases(b.block,b.injectionLogits)&&!aliases(b.outputTrunk,b.injectionLogits)&&!aliases(b.outputTrunk,b.block),
            "Qwen4Exp paired write requires independent block, injection and trunk-output storage");
    using Stage=Qwen4ExpPrimitiveStage;
    const auto inject=c.prefix+"write_projection",write=c.prefix+"write";
    auto writeDependencies=dependencies;writeDependencies.push_back(inject);
    std::vector<PendingNode> nodes;
    nodes.push_back({inject,projection(c,read.inject_,read.buffers_.normalized,b.injectionLogits,4,4*c.hidden),
                     {c.prefix+"branch_norm"}});
    nodes.push_back({write,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
        .operation=Stage::ResidualWrite{c.hidden,read.buffers_.trunk,b.block,b.injectionLogits,b.outputTrunk}}),writeDependencies});
    install(graph,std::move(nodes),c.device);
    read.graph_=nullptr;
    return write;
}
} // namespace llaminar2
