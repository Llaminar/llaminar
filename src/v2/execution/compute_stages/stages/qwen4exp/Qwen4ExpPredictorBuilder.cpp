/**
 * @file Qwen4ExpPredictorBuilder.cpp
 * @brief Reuse native norm, bit-preserving branch concat and prepared GEMM for Dense0.
 *
 * Immutable source/cache/count validation precedes node insertion. Reshapes
 * describe complete existing allocations and are owned by the GEMM stage. The
 * recursive4H result remains in MTP_HIDDEN while terminal H-wide head scratch
 * reuses MTP_NORM_EMBEDDING after its input-mix consumers have completed.
 */
#include "Qwen4ExpPredictorBuilder.h"
#include "Qwen4ExpBranchConcatStage.h"
#include "../GEMMArenaReshapeStage.h"
#include "../RMSNormStage.h"
#include "../../ComputeStageFactory.h"
#include "../../../../memory/BufferArena.h"
#include "../../../../models/GraphTypes.h"
#include "../../../../execution/local_execution/graph/ComputeGraph.h"
#include <cmath>
namespace llaminar2 {
namespace {
/** @brief Borrow one exact admitted ordinary root without triggering native preparation. */
Qwen4ExpFloatBinding buffer(BufferArena &arena, BufferId id, DeviceId device)
{
    auto root=arena.getSharedTensor(id);
    if (!root || root->transferStorageOwner()!=root.get() || root->is_view() || root->isMapped() ||
        root->is_raw_data_released() || root->memoryResidency()!=MemoryResidency::STANDARD ||
        root->native_type()!=TensorType::FP32 || root->home_device()!=device || arena.getTensor(id)!=root.get())
        throw std::invalid_argument("Dense0 composition requires its exact owned ordinary FP32 arena root");
    return {*root,id};
}
/** @brief Reject policy/source/count contradictions before graph mutation or stage preparation. */
void requireContext(const Qwen4ExpPredictorContext &c,const MTPDepthWeightBindings &weights)
{
    Qwen4ExpPredictorResidualBinding::requireFrozenSource(c.layer.source(),weights);
    c.rows.requireCompactCPUQuery();
    if (c.layer.consumer()!=GraphAttentionConsumer::Dense || c.rows.device()!=c.device ||
        c.hidden<=0 || c.hidden>std::numeric_limits<int>::max()/8 || c.low_rank<=0 ||
        !std::isfinite(c.epsilon) || c.epsilon<=0 || c.prefix.empty())
        throw std::invalid_argument("Dense0 builder requires paired Dense0 geometry, participant and original rows");
}
/** @brief Require the unchanged imported FP32 gain owner for one exact source48 role. */
Qwen4ExpImmutableWeightBinding gain(const WeightBinding *binding,int source,int elements)
{
    if (!binding || !binding->immutable || !binding->tensor || binding->identity.layer!=source ||
        binding->tensor->native_type()!=TensorType::FP32 || binding->tensor->numel()!=static_cast<size_t>(elements))
        throw std::invalid_argument("Dense0 gain must retain its exact immutable source48 identity");
    return Qwen4ExpImmutableWeightBinding(*binding->tensor);
}
/** @brief Capture predictor-only serial-row projection arithmetic with the original count pointer. */
Qwen4ExpHyperconnectionContext hcContext(const Qwen4ExpPredictorContext &c)
{
    return {.device=c.device,.mpi=c.mpi,.prefix=c.prefix+"hc_head_",.rows=c.rows.primitives(),
        .hidden=c.hidden,.lowRank=c.low_rank,.epsilon=c.epsilon,
        .arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier,
        .projectionRows=c.rows.tokens(),.readNodeName=c.prefix+"final_norm"};
}
} // namespace
std::string Qwen4ExpPredictorBuilder::appendEmbedding(ComputeGraph &graph,BufferArena &arena,
    const Qwen4ExpPredictorContext &c,const Qwen4ExpPredictorGlobalBindings &globals,
    const MTPForwardInput &input,PreparedWeightStore &store,std::unique_ptr<IComputeStage> reconstruction)
{
    c.rows.requireCompactCPUQuery();
    if (input.batch_size!=1 || input.seq_len!=c.rows.tokens().physicalRows() ||
        (c.device.is_gpu() ? !input.draft_token_ids_device : !input.draft_token_ids) ||
        static_cast<bool>(reconstruction)!=globals.reconstructEmbedding())
        throw std::invalid_argument("Dense0 target embedding requires exact native token rows and its declared reconstruction");
    auto output=buffer(arena,BufferId::MTP_EMBEDDING,c.device);
    const auto &source=globals.embedding();
    auto embedding=ComputeStageFactory::createEmbedding({.device_id=c.device,.mpi_ctx=c.mpi,
        .embed_table=source.tensor,.token_ids=input.draft_token_ids,.token_ids_device=input.draft_token_ids_device,
        .output=output.tensor,.num_tokens=c.rows.tokens().physicalRows(),.d_model=c.hidden,
        .vocab_size=globals.vocabulary(),
        .vocab_offset=globals.embeddingOffset(),.local_vocab_size=globals.embeddingRows(),
        .output_buffer_id=BufferId::MTP_EMBEDDING,.prepared_ref=source.prepared,.prepared_store=&store});
    graph.addNode(c.prefix+"embedding",std::move(embedding),c.device);
    if (!reconstruction) return c.prefix+"embedding";
    graph.addNode(c.prefix+"embedding_allreduce",std::move(reconstruction),c.device);
    graph.addDependency(c.prefix+"embedding_allreduce",c.prefix+"embedding");
    return c.prefix+"embedding_allreduce";
}
ActivationBuffers Qwen4ExpPredictorBuilder::activationBuffers(BufferArena &arena,
    const MTPForwardOutput &output,DeviceId device,bool kv_only)
{
    const auto require=[&](TensorBase *actual,BufferId role) {
        auto expected=buffer(arena,role,device);
        if(actual!=expected.tensor)throw std::invalid_argument("Dense0 output descriptors must preserve canonical arena roles");
        return actual;
    };
    ActivationBuffers result;
    result.current_hidden=require(output.projected,BufferId::MTP_PROJECTED);
    // Input embedding normalization is finished at FC. Reuse this H-wide root
    // for the mixed attention/FFN input, never reinterpret the4H hnorm root.
    result.normalized=require(output.norm_embedding,BufferId::MTP_NORM_EMBEDDING);
    result.K=require(output.k,BufferId::MTP_K_PROJ);result.V=require(output.v,BufferId::MTP_V_PROJ);
    result.binding_ids={{BufferId::HIDDEN_STATE,BufferId::MTP_PROJECTED},{BufferId::NORMALIZED,BufferId::MTP_NORM_EMBEDDING},
        {BufferId::K_PROJ,BufferId::MTP_K_PROJ},{BufferId::V_PROJ,BufferId::MTP_V_PROJ},
        {BufferId::K_FULL_PREFILL,BufferId::MTP_K_FULL_PREFILL},{BufferId::V_FULL_PREFILL,BufferId::MTP_V_FULL_PREFILL},
        {BufferId::ATTN_OUTPUT,BufferId::MTP_ATTN_OUTPUT},{BufferId::ATTN_PROJ,BufferId::MTP_ATTN_PROJ},
        {BufferId::GATE_PROJ,BufferId::MTP_GATE_PROJ},{BufferId::UP_PROJ,BufferId::MTP_UP_PROJ},{BufferId::FFN_OUTPUT,BufferId::MTP_FFN_OUTPUT}};
    if(output.k_full_prefill)result.K_full_prefill=require(output.k_full_prefill,BufferId::MTP_K_FULL_PREFILL);
    if(output.v_full_prefill)result.V_full_prefill=require(output.v_full_prefill,BufferId::MTP_V_FULL_PREFILL);
    for (const auto id:{BufferId::QFN_HC_NORMALIZED,BufferId::QFN_HC_DOWN,BufferId::QFN_HC_READ_LOGITS,BufferId::QFN_HC_WRITE_LOGITS})
        result.extensions.emplace(id,buffer(arena,id,device).tensor);
    if (!kv_only) {
        (void)require(output.hidden,BufferId::MTP_HIDDEN);
        result.Q=require(output.q,BufferId::MTP_Q_PROJ);result.attn_output=require(output.attn_output,BufferId::MTP_ATTN_OUTPUT);
        result.attn_proj=require(output.attn_proj,BufferId::MTP_ATTN_PROJ);result.gate=require(output.gate,BufferId::MTP_GATE_PROJ);
        result.up=require(output.up,BufferId::MTP_UP_PROJ);result.ffn_output=require(output.ffn_output,BufferId::MTP_FFN_OUTPUT);
        result.extensions.emplace(BufferId::MTP_FA_Q_RAW,require(output.q_raw,BufferId::MTP_FA_Q_RAW));
        result.extensions.emplace(BufferId::MTP_FA_GATE,require(output.q_gate,BufferId::MTP_FA_GATE));
        result.binding_ids.emplace(BufferId::Q_PROJ,BufferId::MTP_Q_PROJ);
        result.binding_ids.emplace(BufferId::FA_Q_RAW,BufferId::MTP_FA_Q_RAW);
        result.binding_ids.emplace(BufferId::FA_GATE,BufferId::MTP_FA_GATE);
        if(!output.moe)throw std::invalid_argument("Dense0 routed graph requires its existing policy-selected MoE activation binding");
        for(const auto &[id,tensor]:output.moe->entries()) {
            if(tensor!=arena.getTensor(id))throw std::invalid_argument("Dense0 MoE activation descriptor must borrow its admitted canonical arena root");
            result.extensions.emplace(id,tensor);
        }
    }
    return result;
}
std::string Qwen4ExpPredictorBuilder::appendVocabulary(ComputeGraph &graph,BufferArena &arena,
    const Qwen4ExpPredictorContext &c,const Qwen4ExpPredictorGlobalBindings &globals,PreparedWeightStore &store,
    int serial_partition_width,const std::string &head_ready)
{
    c.rows.requireCompactCPUQuery();
    if (head_ready.empty() || (serial_partition_width!=0 && !globals.selection().mirrored_head))
        throw std::invalid_argument("Dense0 vocabulary stage requires its dedicated HC read and canonical mirrored arithmetic policy");
    const auto &head=globals.head();
    graph.addNode(c.prefix+"lm_head",ComputeStageFactory::createLMHead({.device_id=c.device,.mpi_ctx=c.mpi,
        .hidden_states=buffer(arena,BufferId::MTP_NORM_EMBEDDING,c.device).tensor,.lm_head_weight=head.tensor,
        .logits=buffer(arena,BufferId::MTP_LOGITS,c.device).tensor,.seq_len=c.rows.tokens().physicalRows(),
        .d_model=c.hidden,.vocab_size=globals.headRows(),.serial_equivalent_partition_width=serial_partition_width,
        .use_prefill_replay_row_offset=false,.compute_all_positions=true,
        .force_decode_equivalent_verifier_prefill=c.rows.tokens().physicalRows()>1,
        .input_buffer_id=BufferId::MTP_NORM_EMBEDDING,.output_buffer_id=BufferId::MTP_LOGITS,
        .prepared_ref=head.prepared,.prepared_store=&store}),c.device);
    graph.addDependency(c.prefix+"lm_head",head_ready);
    return c.prefix+"lm_head";
}
std::string Qwen4ExpPredictorBuilder::appendInputMix(ComputeGraph &graph,BufferArena &arena,
    const Qwen4ExpPredictorContext &c,const MTPDepthWeightBindings &weights,PreparedWeightStore &store,
    Qwen4ExpFloatBinding hidden,Qwen4ExpPredictorInvocation invocation,
    const std::string &embedding_ready,const std::string &hidden_ready)
{
    requireContext(c,weights);
    if (invocation != Qwen4ExpPredictorInvocation::Draft && invocation != Qwen4ExpPredictorInvocation::ShiftedPrefill)
        throw std::invalid_argument("Dense0 input mix requires its authenticated invocation role");
    const bool valid_hidden_role=hidden.buffer==BufferId::HIDDEN_STATE || hidden.buffer==BufferId::PREFIX_TERMINAL_HIDDEN ||
        hidden.buffer==BufferId::MTP_HIDDEN || hidden.buffer==BufferId::MTP_NORM_HIDDEN;
    if (!valid_hidden_role || !hidden.tensor || buffer(arena,hidden.buffer,c.device).tensor!=hidden.tensor ||
        embedding_ready.empty() || hidden.byteOffset!=0)
        throw std::invalid_argument("Dense0 input mix requires original4H hidden and reconstructed target embedding producer");
    const auto embedding=buffer(arena,BufferId::MTP_EMBEDDING,c.device);
    const auto normalized_hidden=buffer(arena,BufferId::MTP_NORM_HIDDEN,c.device);
    const auto normalized_embedding=buffer(arena,BufferId::MTP_NORM_EMBEDDING,c.device);
    const auto concat=buffer(arena,BufferId::MTP_CONCAT,c.device);
    const auto projected=buffer(arena,BufferId::MTP_PROJECTED,c.device);
    const auto geometry=Qwen4ExpPredictorFCGeometry::admit(concat.tensor->shape(),projected.tensor->shape(),c.rows.tokens().physicalRows(),c.hidden);
    const auto hidden_gain=gain(weights.pre_fc_norm_hidden,c.layer.source().sourceLayer(),c.hidden*4);
    const auto embedding_gain=gain(weights.pre_fc_norm_embedding,c.layer.source().sourceLayer(),c.hidden);
    if (!weights.fc || !weights.fc->immutable || weights.fc->identity.layer!=c.layer.source().sourceLayer())
        throw std::invalid_argument("Dense0 FC must retain the original immutable source48 matrix");
    const Qwen4ExpPreparedProjection fc(*weights.fc,store,c.device);
    fc.requireGeometry(c.hidden,c.hidden*2);
    // Admit whole-storage metadata before normal GPU preparation. The stage,
    // not a graph side-list, retains both roots and seals addresses at capture.
    auto input_view=FP32ArenaReshape::fromArena(arena,BufferId::MTP_CONCAT,
        {geometry.input[0],geometry.input[1]},c.device);
    auto output_view=FP32ArenaReshape::fromArena(arena,BufferId::MTP_PROJECTED,
        {geometry.output[0],geometry.output[1]},c.device);
    using Primitive=Qwen4ExpPrimitiveStage;
    auto hidden_norm=std::make_unique<Primitive>(Primitive::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows.primitives(),
        .operation=Primitive::BranchRMSNorm::predictorHidden(c.hidden,c.epsilon,hidden,hidden_gain,normalized_hidden,
            invocation == Qwen4ExpPredictorInvocation::Draft
                ? Primitive::DiagnosticInputPublication::MTPTerminalHidden
                : Primitive::DiagnosticInputPublication::None)});
    auto embedding_norm=std::make_unique<RMSNormStage>(RMSNormStage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.input=embedding.tensor,.output=normalized_embedding.tensor,
        .gamma=weights.pre_fc_norm_embedding->tensor,.eps=c.epsilon,.subtract_one=false,
        .seq_len=c.rows.tokens().physicalRows(),.input_buffer_id=embedding.buffer,.output_buffer_id=normalized_embedding.buffer});
    auto concatenation=std::make_unique<Qwen4ExpBranchConcatStage>(Qwen4ExpBranchConcatStage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.rows=c.rows.tokens(),.hidden_dim=c.hidden,
        .hidden=normalized_hidden,.embedding=normalized_embedding,.output=concat});
    auto projection=std::make_unique<GEMMArenaReshapeStage>(GEMMStage::Params{
        .device_id=c.device,.mpi_ctx=c.mpi,.B=fc.tensor(),
        .m=Qwen4ExpPredictorProjectionGeometry::perBranchRows(c.rows.tokens().physicalRows(),qwen4exp::kResidualBranches),.n=c.hidden,.k=c.hidden*2,
        .force_decode_equivalent_verifier_prefill=true,.verifier_row_range=c.rows.branches(),
        .prepared_ref=fc.ref(),.prepared_store=&store},std::move(input_view),std::move(output_view));
    // Constructor validation for every operation completes before adding any node.
    graph.addNode(c.prefix+"norm_hidden",std::move(hidden_norm),c.device);
    graph.addNode(c.prefix+"norm_embedding",std::move(embedding_norm),c.device);
    graph.addNode(c.prefix+"concat",std::move(concatenation),c.device);
    graph.addNode(c.prefix+"fc",std::move(projection),c.device);
    if (!hidden_ready.empty()) graph.addDependency(c.prefix+"norm_hidden",hidden_ready);
    graph.addDependency(c.prefix+"norm_embedding",embedding_ready);
    graph.addDependency(c.prefix+"concat",c.prefix+"norm_hidden");
    graph.addDependency(c.prefix+"concat",c.prefix+"norm_embedding");
    graph.addDependency(c.prefix+"fc",c.prefix+"concat");
    return c.prefix+"fc";
}
std::string Qwen4ExpPredictorBuilder::appendHeadRead(ComputeGraph &graph,BufferArena &arena,
    const Qwen4ExpPredictorContext &c,const MTPDepthWeightBindings &weights,PreparedWeightStore &store,
    const std::string &trunk_ready)
{
    requireContext(c,weights);
    if (trunk_ready.empty()) throw std::invalid_argument("Dense0 head requires the direct completed recursive4H write");
    const auto &head=weights.hyper_connection_head;
    const auto norm=gain(head.norm,c.layer.source().sourceLayer(),c.hidden*4);
    if (head.inject || !head.down || !head.up || head.down->identity.layer!=c.layer.source().sourceLayer() ||
        head.up->identity.layer!=c.layer.source().sourceLayer())
        throw std::invalid_argument("Dense0 terminal HC head requires its dedicated immutable source48 read-only weights");
    const auto down=buffer(arena,BufferId::QFN_HC_DOWN,c.device);
    return Qwen4ExpHyperconnectionBuilder::appendFinalRead(graph,hcContext(c),
        {norm,Qwen4ExpPreparedProjection(*head.down,store,c.device),Qwen4ExpPreparedProjection(*head.up,store,c.device)},
        {buffer(arena,BufferId::MTP_HIDDEN,c.device),buffer(arena,BufferId::QFN_HC_NORMALIZED,c.device),
         down,down,buffer(arena,BufferId::QFN_HC_READ_LOGITS,c.device),buffer(arena,BufferId::MTP_NORM_EMBEDDING,c.device)},
        {trunk_ready});
}
} // namespace llaminar2
