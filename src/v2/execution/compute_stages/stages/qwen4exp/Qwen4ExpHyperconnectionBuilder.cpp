/**
 * @file Qwen4ExpHyperconnectionBuilder.cpp
 * @brief Wire typed prepared GEMMs and GR stages without model-specific orchestration.
 *
 * All nodes are constructed and validated before graph mutation. Projection
 * weights retain raw checkpoint scaling: residualDown applies /4 before SiLU,
 * and residualWrite applies /4 before 2*sigmoid. Read and injection projections
 * may overlap block compute only through explicit graph dependencies. No
 * collective, activation allocation, packing or backend selection is hidden in this builder.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 */
#include "Qwen4ExpHyperconnectionBuilder.h"
#include "kernels/common/HCDownOutputPublication.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "kernels/common/HCBranchNormSingleQ8Publication.h"
#include "models/qwen4exp/Qwen4ExpHCUpArithmetic.h"
#include "../GEMMStage.h"
#include "../../../../loaders/PreparedWeightStore.h"
#include "../../../local_execution/graph/ComputeGraph.h"
#include "../../../../tensors/Tensors.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <unordered_set>
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmPersistentHCRead.h"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"
#include "kernels/rocm/gemm/ROCmCTAOwnedHCRead.h"
#include "transfer/TransferEngine.h"
#include "backends/BackendManager.h"
#include "utils/FNV1a.h"
#include <bit>
#include <cstring>
#endif

namespace llaminar2
{
namespace
{
#ifdef HAVE_ROCM
#include "Qwen4ExpPersistentHCReadStage.inc"
#endif
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
 * @param hc_output Optional original reducer-output publication, retaining existing up banks.
 * @param normPublication Optional original norm-produced SingleQ8 input for the same down projection.
 * @return Actual GEMMStage retaining the original source, query and arithmetic.
 * @throws std::invalid_argument For foreign preparation, noncanonical byte views or invalid stage geometry.
 */
std::unique_ptr<IComputeStage> projection(const Qwen4ExpHyperconnectionContext &c,
    const Qwen4ExpPreparedProjection &weight,Qwen4ExpFloatBinding input,Qwen4ExpFloatBinding output,int n,int k,
    QuantizedActivationOperandPrecision precision=QuantizedActivationOperandPrecision::SingleQ8,
    std::optional<GEMMStage::HCDownReducerOutput> hc_output=std::nullopt,
    std::shared_ptr<HCBranchNormSingleQ8Publication> normPublication={})
{
    require(input.byteOffset==0&&output.byteOffset==0,
            "Qwen4Exp mixer GEMM views require zero byte offsets or prebuilt typed TensorSlice views");
    require(weight.ref().device==c.device,"Qwen4Exp mixer projection belongs to another participant");
    weight.requireGeometry(n,k);
    return std::make_unique<GEMMStage>(GEMMStage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.A=input.tensor,.B=weight.tensor(),.C=output.tensor,
        .hc_down_output=std::move(hc_output),
        .hc_branch_norm_input=std::move(normPublication),
        .m=c.rows.capacity,.n=n,.k=k,.alpha=1.0f,.beta=0.0f,.transpose_B=false,
        .a_buffer_id=input.buffer,.c_buffer_id=output.buffer,
        .force_decode_equivalent_verifier_prefill=c.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
        .verifier_row_range=precision==QuantizedActivationOperandPrecision::TwoTermQ8 && c.upProjectionRows
            ? c.upProjectionRows : c.projectionRows,.prepared_ref=weight.ref(),.prepared_store=&weight.store(),
        .quantized_operand_precision=precision});
}
/** @brief Compare complete original primitive row ownership without reading live counts. */
bool sameRows(const Qwen4ExpStageRows &left,const Qwen4ExpStageRows &right)
{
    if(left.capacity!=right.capacity || left.liveRows.has_value()!=right.liveRows.has_value())return false;
    if(!left.liveRows)return true;
    if(left.liveRows->value.index()!=right.liveRows->value.index())return false;
    if(const auto* l=std::get_if<Qwen4ExpIntBinding>(&left.liveRows->value))
    {
        const auto &r=std::get<Qwen4ExpIntBinding>(right.liveRows->value);
        return l->tensor==r.tensor && l->buffer==r.buffer && l->byteOffset==r.byteOffset;
    }
    const auto &l=std::get<Qwen4ExpNativeIntInput>(left.liveRows->value);
    const auto &r=std::get<Qwen4ExpNativeIntInput>(right.liveRows->value);
    return l.owner==r.owner && l.data==r.data && l.capacity==r.capacity;
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
    const bool normQ8=c.branchNormExecution==Qwen4ExpHCBranchNormExecution::PublishOriginalSingleQ8;
    require(normQ8||c.branchNormExecution==Qwen4ExpHCBranchNormExecution::Staged,"Unknown HC branch-norm publication policy");
    std::shared_ptr<HCBranchNormSingleQ8Publication> normPublication;
    if(normQ8) {
        require(c.device.is_gpu()&&c.rows.capacity==1&&!c.rows.liveRows&&c.hidden%32==0&&
            c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&!c.projectionRows&&
            b.trunk.byteOffset==0&&b.normalized.byteOffset==0,
            "HC norm SingleQ8 candidate requires exact fully-active ordinary GPU M1 source rows");
        normPublication=std::make_shared<HCBranchNormSingleQ8Publication>(c.device,c.hidden,c.epsilon,
            *b.trunk.tensor,*w.norm.tensor,*b.normalized.tensor,*b.down.tensor);
    }
    const bool shortRead=c.readExecution==Qwen4ExpHCReadExecution::ProducerGridShortK;
    if((c.readExecution==Qwen4ExpHCReadExecution::PersistentOriginalPhases||c.readExecution==Qwen4ExpHCReadExecution::CTAOwnedOriginalTail)) {
#ifdef HAVE_ROCM
        require(c.device.is_rocm()&&c.rows.capacity==1&&c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&
            up_arithmetic==Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8,
            "Persistent HC read requires explicit original ROCm ordinary M1 two-term arithmetic");
#else
        throw std::invalid_argument("Persistent HC read requires a compiled ROCm backend");
#endif
    }
    require(c.readExecution==Qwen4ExpHCReadExecution::Staged||shortRead||c.readExecution==Qwen4ExpHCReadExecution::PersistentOriginalPhases||c.readExecution==Qwen4ExpHCReadExecution::CTAOwnedOriginalTail,"Unknown HC read execution policy");
    if(shortRead) {
        DeviceNativeQuantisedMatrixDesc source;
        auto* engine=w.up.store().gemmKernel(w.up.ref());
        require(c.device.is_gpu()&&c.rows.capacity==1&&c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&
            up_arithmetic==Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8&&
            c.lowRank>=64&&c.lowRank<=512&&4*c.hidden>=4*c.lowRank&&
            b.normalized.byteOffset==0&&b.mixed.byteOffset==0&&engine&&
            engine->inspectPreparedNativeQuantisedMatrixDesc(source)&&source.codebook_id==19,
            "HC producer-grid read requires the original prepared GPU ordinary M1 signed-Q8 short-K region");
    }
    const bool downPublish=c.downPublication==Qwen4ExpHCDownPublication::ReducerEpilogue;
    require(c.downPublication==Qwen4ExpHCDownPublication::Staged||downPublish,"Unknown HC down publication policy");
    require(c.downProducerProgram==HCDownProducerProgram::Original||
        (c.downProducerProgram==HCDownProducerProgram::TwoK32Lookahead&&downPublish),
        "HC down lookahead requires its explicitly selected reducer-publication purpose");
    std::shared_ptr<HCDownOutputPublication> downPublication;
    if(downPublish) {
        DeviceNativeQuantisedMatrixDesc source;
        auto* engine=w.down.store().gemmKernel(w.down.ref());
        require(c.device.is_rocm()&&c.rows.capacity==1&&c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&
            up_arithmetic==Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8&&!c.projectionRows&&
            c.lowRank>=64&&c.lowRank<=512&&c.lowRank%32==0&&c.hidden%8==0&&engine&&
            engine->inspectPreparedNativeQuantisedMatrixDesc(source)&&source.codebook_id==19&&
            source.source_identity_present&&canonicalDeviceVnniCodebookId(source.source_codebook_id)==19,
            "HC down epilogue requires original ROCm native-Q8 physicalM1 projection and complete32-column publisher");
        const auto rows=c.upProjectionRows.value_or(DeviceRowRange::fullyActive(1));
        downPublication=std::make_shared<HCDownOutputPublication>(c.device,*b.down.tensor,*b.activatedDown.tensor,c.lowRank,rows,c.downProducerProgram);
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
        .operation=Stage::BranchRMSNorm{c.hidden,c.epsilon,b.trunk,w.norm,b.normalized,
            Stage::DiagnosticInputPublication::None,Stage::BranchNormInputRole::Ordinary,normPublication}}),dependencies});
    nodes.push_back({down,projection(c,w.down,b.normalized,b.down,c.lowRank,4*c.hidden,
        QuantizedActivationOperandPrecision::SingleQ8,downPublication?
            std::optional{GEMMStage::HCDownReducerOutput{downPublication,b.activatedDown.buffer}}:std::nullopt,normPublication),{norm}});
    if(up_arithmetic==Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8) {
        w.up.requireGeometry(4*c.hidden,c.lowRank);
        require(w.up.ref().device==c.device && b.down.byteOffset==0 &&
                    b.activatedDown.byteOffset==0 && b.readLogits.byteOffset==0,
                "HC fused publication requires exact participant and zero-offset FP32 matrices");
        // One prepared up invocation owns activation, both banks and projection.
        // The same original row authority is validated above before graph mutation.
        nodes.push_back({shortRead?read:up,std::make_unique<GEMMStage>(GEMMStage::Params{
            .device_id=c.device,.mpi_ctx=c.mpi,.A=b.down.tensor,.B=w.up.tensor(),.C=b.readLogits.tensor,
            .hc_prepublished_down=downPublication,
            .hc_residual_down=GEMMStage::HCResidualDownPublication{b.activatedDown.tensor,b.activatedDown.buffer},
            .hc_short_k_read=shortRead?std::optional{GEMMStage::HCResidualShortKRead{
                b.normalized.tensor,b.normalized.buffer,b.mixed.tensor,b.mixed.buffer}}:std::nullopt,
            .m=c.rows.capacity,.n=4*c.hidden,.k=c.lowRank,.alpha=1.f,.beta=0.f,
            .a_buffer_id=b.down.buffer,.c_buffer_id=b.readLogits.buffer,
            .force_decode_equivalent_verifier_prefill=c.arithmetic==Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
            .verifier_row_range=c.upProjectionRows?c.upProjectionRows:c.projectionRows,
            .prepared_ref=w.up.ref(),.prepared_store=&w.up.store(),
            .quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8}),shortRead?std::vector<std::string>{down,norm}:std::vector<std::string>{down}});
    } else {
        nodes.push_back({activation,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
            .operation=Stage::ResidualDown{c.lowRank,b.down,b.activatedDown}}),{down}});
        nodes.push_back({up,projection(c,w.up,b.activatedDown,b.readLogits,4*c.hidden,c.lowRank,up_precision),{activation}});
    }
    if(!shortRead)nodes.push_back({read,std::make_unique<Stage>(Stage::Params{.device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows,
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
    if((c.readExecution==Qwen4ExpHCReadExecution::PersistentOriginalPhases||c.readExecution==Qwen4ExpHCReadExecution::CTAOwnedOriginalTail)) {
#ifdef HAVE_ROCM
        const auto normName=c.prefix+"branch_norm",downName=c.prefix+"read_down",upName=c.prefix+"read_up";
        const auto* norm=dynamic_cast<const Qwen4ExpPrimitiveStage*>(graph.getNode(normName)->stage.get());
        const auto* down=dynamic_cast<const GEMMStage*>(graph.getNode(downName)->stage.get());
        const auto* up=dynamic_cast<const GEMMStage*>(graph.getNode(upName)->stage.get());
        const auto* injection=dynamic_cast<const GEMMStage*>(graph.getNode(inject)->stage.get());
        const auto* originalRead=dynamic_cast<const Qwen4ExpPrimitiveStage*>(graph.getNode(read.terminal_)->stage.get());
        require(norm&&down&&up&&injection&&originalRead,"Persistent HC lowering lost its original prepared stages");
        const auto* readOperation=std::get_if<Qwen4ExpPrimitiveStage::ResidualRead>(&originalRead->getParams().operation);
        require(readOperation&&sameRows(originalRead->getParams().rows,c.rows)&&sameRows(norm->getParams().rows,c.rows)&&
            readOperation->hidden==c.hidden&&readOperation->normalized.tensor==read.buffers_.normalized.tensor&&
            readOperation->normalized.buffer==read.buffers_.normalized.buffer&&readOperation->normalized.byteOffset==0&&
            readOperation->gateLogits.tensor==read.buffers_.readLogits.tensor&&readOperation->gateLogits.buffer==read.buffers_.readLogits.buffer&&
            readOperation->gateLogits.byteOffset==0&&readOperation->output.tensor==read.buffers_.mixed.tensor&&
            readOperation->output.buffer==read.buffers_.mixed.buffer&&readOperation->output.byteOffset==0,
            "Persistent HC lowering changed the original four-term read or its row authority");
        const auto* operation=std::get_if<Qwen4ExpPrimitiveStage::BranchRMSNorm>(&norm->getParams().operation);
        require(operation,"Persistent HC lowering requires the original branch norm");
        // Retain the actual original stage quote in its existing execution
        // order. A different merge order changes the final packed bank extent
        // even when every name/size is equal; PMA owns that exact physical byte.
        WorkspaceRequirements workspaceRequirements;
        for(const auto& name:graph.getExecutionOrder()) {
            const auto* stage=graph.getNode(name)->stage.get();
            if(stage!=down&&stage!=up&&stage!=injection)continue;
            const auto* projection=static_cast<const GEMMStage*>(stage);
            const auto& params=projection->getParams();
            workspaceRequirements.merge(projection->getWorkspaceRequirements(params.m,params.n,params.k));
        }
        auto program=std::make_unique<PersistentHCReadStage>(c,*operation,down->getParams(),up->getParams(),
            injection->getParams(),read.buffers_,b.injectionLogits,read.terminal_,std::move(workspaceRequirements));
        const std::array<LocalRegionExport,1> exports{{{.producer=inject,.consumers={write}}}};
        if(c.readExecution==Qwen4ExpHCReadExecution::CTAOwnedOriginalTail)
            graph.replaceClosedLocalRegion({downName,upName,read.terminal_},read.terminal_,std::move(program));
        else graph.replaceClosedLocalRegion({normName,downName,upName,read.terminal_,inject},read.terminal_,std::move(program),exports);
#endif
    }
    read.graph_=nullptr;
    return write;
}
/** @copydoc Qwen4ExpHyperconnectionBuilder::describeFullNativeReadRegion */
Qwen4ExpFullNativeHCRegion Qwen4ExpHyperconnectionBuilder::describeFullNativeReadRegion(
    ComputeGraph& graph,const std::string& previous_write,const std::string& read_name)
{
    const auto* read_node=graph.getNode(read_name);
    const auto* read=read_node?dynamic_cast<const Qwen4ExpPrimitiveStage*>(read_node->stage.get()):nullptr;
    const auto* r=read?std::get_if<Qwen4ExpPrimitiveStage::ResidualRead>(&read->getParams().operation):nullptr;
    require(r,"Full HC lowering requires the named original residual-read stage");
    const Qwen4ExpPrimitiveStage* norm=nullptr;const GEMMStage* up=nullptr;const GEMMStage* down=nullptr;
    std::string norm_name,up_name,down_name,injection_name;
    for(const auto& dependency:read_node->dependencies) {
        const auto* node=graph.getNode(dependency);
        const auto* primitive=dynamic_cast<const Qwen4ExpPrimitiveStage*>(node->stage.get());
        const auto* n=primitive?std::get_if<Qwen4ExpPrimitiveStage::BranchRMSNorm>(&primitive->getParams().operation):nullptr;
        if(n&&n->output.tensor==r->normalized.tensor&&n->output.buffer==r->normalized.buffer) {
            require(!norm,"Full HC read has ambiguous original normalization");norm=primitive;norm_name=dependency;
        }
        const auto* projection=dynamic_cast<const GEMMStage*>(node->stage.get());
        if(projection&&projection->getParams().C==r->gateLogits.tensor) {
            require(!up,"Full HC read has ambiguous original up projection");up=projection;up_name=dependency;
        }
    }
    require(norm&&up,"Full HC read lost its direct norm/up declarations");
    for(const auto& dependency:graph.getNode(up_name)->dependencies) {
        const auto* projection=dynamic_cast<const GEMMStage*>(graph.getNode(dependency)->stage.get());
        if(projection&&projection->getParams().C==up->getParams().A) {
            require(!down,"Full HC up has ambiguous original down projection");down=projection;down_name=dependency;
        }
    }
    require(down,"Full HC up lost its direct down producer");
    const auto& n=std::get<Qwen4ExpPrimitiveStage::BranchRMSNorm>(norm->getParams().operation);
    // Only an actual independent prepared four-logit projection sharing this
    // exact norm input is an injection export. A final read has none.
    for(const auto& name:graph.getExecutionOrder()) {
        const auto* node=graph.getNode(name);const auto* p=dynamic_cast<const GEMMStage*>(node->stage.get());
        if(!p||p==down||p==up)continue;
        const auto& a=p->getParams();
        if(a.A==n.output.tensor&&a.n==4&&a.k==4*n.hidden&&
            std::find(node->dependencies.begin(),node->dependencies.end(),norm_name)!=node->dependencies.end()) {
            require(injection_name.empty(),"Full HC norm has ambiguous independent injection exports");injection_name=name;
        }
    }
    const auto& dp=down->getParams();const auto& up_params=up->getParams();
    require(!dp.force_decode_equivalent_verifier_prefill&&!up_params.force_decode_equivalent_verifier_prefill&&
        !dp.request_rows&&!up_params.request_rows&&!dp.hc_short_k_read&&!up_params.hc_short_k_read&&
        !dp.gated_norm&&!up_params.gated_norm,"Full HC initial implementation requires original ordinary HC projection declarations");
    Qwen4ExpHyperconnectionContext c{.device=read->device(),.mpi=up_params.mpi_ctx,.prefix=read_name+"_full_",
        .rows=read->getParams().rows,.hidden=n.hidden,.lowRank=dp.n,.epsilon=n.epsilon,
        .arithmetic=Qwen4ExpProjectionArithmetic::Ordinary,.projectionRows=dp.verifier_row_range,
        .readNodeName=read_name,.upOperandPrecision=up_params.quantized_operand_precision,
        .upProjectionRows=up_params.verifier_row_range};
    return {std::move(c),previous_write,norm_name,down_name,up_name,read_name,injection_name};
}
/** @copydoc Qwen4ExpHyperconnectionBuilder::retainFullNativeReadPhase */
std::shared_ptr<Qwen4ExpHCReadPhaseOwner> Qwen4ExpHyperconnectionBuilder::retainFullNativeReadPhase(
    ComputeGraph& graph,const Qwen4ExpFullNativeHCRegion& region)
{
#ifdef HAVE_ROCM
    const auto& c=region.context;
    require(c.device.is_rocm()&&c.rows.capacity==1&&c.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&
        c.readExecution==Qwen4ExpHCReadExecution::Staged,
        "Full native HC requires explicit original staged ROCm ordinary M1 declarations");
    const auto primitive=[&](const std::string& name) {
        const auto* node=graph.getNode(name);
        const auto* stage=node?dynamic_cast<const Qwen4ExpPrimitiveStage*>(node->stage.get()):nullptr;
        require(stage&&stage->device()==c.device&&sameRows(stage->getParams().rows,c.rows),
            "Full HC primitive lost its original participant/row authority");return stage;
    };
    const auto projection=[&](const std::string& name) {
        const auto* node=graph.getNode(name);
        const auto* stage=node?dynamic_cast<const GEMMStage*>(node->stage.get()):nullptr;
        require(stage&&stage->device()==c.device,"Full HC projection lost its original participant");
        const auto& p=stage->getParams();
        require(p.m==1&&!p.needs_allreduce&&p.output_range.empty()&&!p.force_decode_equivalent_verifier_prefill&&
            !p.request_rows&&!p.projection_row_work&&!p.hc_short_k_read&&!p.gated_norm&&p.prepared_store&&p.prepared_ref&&p.prepared_store->contains(*p.prepared_ref)&&
            !p.do_swiglu,
            "Full HC projection requires its original prepared plain matrix identity");
        const auto binding=p.prepared_store->binding(*p.prepared_ref);
        require(binding&&binding->tensor==p.B,"Full HC projection prepared reference differs from its original source");
        return stage;
    };
    const auto* write=primitive(region.previous_write);
    const auto* norm=primitive(region.branch_norm);
    const auto* read=primitive(region.read);
    const auto* down=projection(region.down);const auto* up=projection(region.up);
    const auto* injection=region.injection.empty()?nullptr:projection(region.injection);
    const auto* w=std::get_if<Qwen4ExpPrimitiveStage::ResidualWrite>(&write->getParams().operation);
    const auto* n=std::get_if<Qwen4ExpPrimitiveStage::BranchRMSNorm>(&norm->getParams().operation);
    const auto* r=std::get_if<Qwen4ExpPrimitiveStage::ResidualRead>(&read->getParams().operation);
    require(w&&n&&r&&w->hidden==c.hidden&&r->hidden==c.hidden&&
        w->output.tensor==n->input.tensor&&w->output.buffer==n->input.buffer&&w->output.byteOffset==0&&
        r->normalized.tensor==n->output.tensor&&r->normalized.buffer==n->output.buffer&&
        r->normalized.byteOffset==0&&r->gateLogits.tensor==up->getParams().C,
        "Full HC write/norm/read original public boundaries differ");
    require(up->getParams().hc_residual_down.has_value(),"Full HC up requires original activated-down publication");
    const auto& publication=*up->getParams().hc_residual_down;
    require(down->getParams().a_buffer_id&&*down->getParams().a_buffer_id==n->output.buffer&&
        down->getParams().c_buffer_id&&up->getParams().a_buffer_id&&up->getParams().c_buffer_id&&
        down->getParams().c_buffer_id==up->getParams().a_buffer_id&&
        r->gateLogits.buffer==*up->getParams().c_buffer_id,
        "Full HC projection public BufferIds differ");
    auto* raw_down=dynamic_cast<TensorBase*>(down->getParams().C);
    require(raw_down&&publication.activated,"Full HC public projection outputs require original tensor owners");
    Qwen4ExpHyperconnectionReadBuffers buffers{n->input,n->output,
        {*raw_down,*down->getParams().c_buffer_id},
        {*publication.activated,publication.buffer},r->gateLogits,r->output};
    const auto edge=[&](const std::string& producer,const std::string& consumer) {
        const auto* node=graph.getNode(consumer);
        require(node&&std::find(node->dependencies.begin(),node->dependencies.end(),producer)!=node->dependencies.end(),
            "Full HC region lacks an original direct producer edge");
    };
    edge(region.previous_write,region.branch_norm);edge(region.branch_norm,region.down);
    edge(region.down,region.up);edge(region.up,region.read);edge(region.branch_norm,region.read);
    if(injection) {
        require(injection->getParams().a_buffer_id&&*injection->getParams().a_buffer_id==n->output.buffer&&
            injection->getParams().c_buffer_id.has_value(),"Full HC injection requires its original input/output BufferIds");
        require(injection->getParams().verifier_row_range.value_or(DeviceRowRange::fullyActive(1)).captureIdentity()==
            down->getParams().verifier_row_range.value_or(DeviceRowRange::fullyActive(1)).captureIdentity(),
            "Full HC injection and down must retain the original shared projection row authority");
        edge(region.branch_norm,region.injection);
    }
    WorkspaceRequirements requirements;
    for(const auto& name:graph.getExecutionOrder()) {
        if(name!=region.down&&name!=region.up&&name!=region.injection)continue;
        const auto* stage=static_cast<const GEMMStage*>(graph.getNode(name)->stage.get());
        const auto& p=stage->getParams();requirements.merge(stage->getWorkspaceRequirements(p.m,p.n,p.k));
    }
    std::optional<Qwen4ExpFloatBinding> injection_binding;
    if(injection) {
        auto* output=dynamic_cast<TensorBase*>(injection->getParams().C);
        require(output,"Full HC injection requires its original tensor owner");
        injection_binding.emplace(*output,*injection->getParams().c_buffer_id);
    }
    return std::make_shared<PersistentHCReadStage>(c,*n,down->getParams(),up->getParams(),
        injection?std::optional<GEMMStage::Params>{injection->getParams()}:std::nullopt,buffers,
        injection_binding,region.read,std::move(requirements),*w);
#else
    (void)graph;(void)region;
    throw std::invalid_argument("Full native HC requires the compiled ROCm backend");
#endif
}
bool Qwen4ExpHyperconnectionBuilder::lowerAdjacentWriteNorm(ComputeGraph &graph,const std::string &producer)
{
    using Stage=Qwen4ExpPrimitiveStage;
    const auto* writeNode=graph.getNode(producer);
    const auto* writeStage=writeNode?dynamic_cast<const Stage*>(writeNode->stage.get()):nullptr;
    const auto* write=writeStage?std::get_if<Stage::ResidualWrite>(&writeStage->getParams().operation):nullptr;
    require(write,"Qwen4Exp write/next-norm lowering requires its original residual write terminal");
    const auto order=graph.getExecutionOrder();
    const ComputeNode *normNode=nullptr;const Stage *normStage=nullptr;const Stage::BranchRMSNorm *norm=nullptr;
    for(const auto &name:order)
    {
        const auto* node=graph.getNode(name);
        if(node->dependencies!=std::vector<std::string>{producer})continue;
        const auto* stage=dynamic_cast<const Stage*>(node->stage.get());
        const auto* operation=stage?std::get_if<Stage::BranchRMSNorm>(&stage->getParams().operation):nullptr;
        if(!operation)continue;
        require(!norm,"Qwen4Exp write/next-norm lowering has ambiguous normalization successors");
        normNode=node;normStage=stage;norm=operation;
    }
    require(!norm||!norm->single_q8,"HC write/norm lowering cannot discard a required SingleQ8 publication");
    require(norm && normNode->device==writeNode->device && write->hidden==norm->hidden &&
            sameRows(writeStage->getParams().rows,normStage->getParams().rows) &&
            write->output.tensor==norm->input.tensor && write->output.buffer==norm->input.buffer &&
            write->output.byteOffset==norm->input.byteOffset &&
            norm->diagnostic_input_publication==Stage::DiagnosticInputPublication::None &&
            norm->input_role==Stage::BranchNormInputRole::Ordinary,
            "Qwen4Exp write/next-norm lowering requires an exact direct ordinary input/row/participant binding");
    // Prove the original dependency DAG orders every other user of the reused
    // normalization arena before the write or after the norm. Lifting its store
    // across an unordered reader/writer would otherwise change a real lifetime.
    std::unordered_set<std::string> ancestors;
    const auto visitAncestors=[&](auto&& self,const std::string &name)->void {
        if(!ancestors.insert(name).second)return;
        for(const auto &edge:graph.getNode(name)->dependencies)self(self,edge);
    };
    visitAncestors(visitAncestors,producer);
    std::unordered_set<std::string> descendants{normNode->name};
    for(const auto &name:order)
    {
        const auto* node=graph.getNode(name);
        if(std::any_of(node->dependencies.begin(),node->dependencies.end(),
            [&](const auto &edge){return descendants.contains(edge);}))descendants.insert(name);
    }
    const auto touches=[&](const auto &bindings){return std::any_of(bindings.begin(),bindings.end(),
        [&](const auto &binding){return binding.id==norm->output.buffer;});};
    for(const auto &name:order)
    {
        if(ancestors.contains(name) || descendants.contains(name))continue;
        const auto *stage=graph.getNode(name)->stage.get();
        require(stage,"Qwen4Exp write/next-norm lowering cannot authenticate a missing stage contract");
        const auto contract=stage->bufferContract();
        require(!touches(contract.inputs) && !touches(contract.outputs) && !touches(contract.inouts),
                "Qwen4Exp write/next-norm lowering found an unordered normalized-arena lifetime");
    }
    const auto successor=normNode->name;
    auto replacement=std::make_unique<Stage>(Stage::Params{
        .device_id=writeNode->device,.mpi_ctx=writeStage->getParams().mpi_ctx,.rows=writeStage->getParams().rows,
        .operation=Stage::ResidualWriteBranchRMSNorm{write->hidden,norm->epsilon,write->residual,write->block,
            write->gateLogits,write->output,norm->gain,norm->output}});
    graph.replaceProducerAndBypassSuccessor(producer,successor,std::move(replacement));
    return true;
}
} // namespace llaminar2
