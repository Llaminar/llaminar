/**
 * @file Qwen4ExpPredictorGraph.cpp
 * @brief Declare the R1 Dense0/PerBranch sidecar using shared native graph bodies.
 *
 * Predictor source48, shifted cache0, target globals and existing arena roots are
 * explicit immutable inputs. The source manifest remains loader-owned. This
 * adapter creates neither state storage nor a second controller: it composes
 * shared branch mixing, dense FA, routed FFN and terminal ownership contracts.
 */
#include "Qwen4ExpGraph.h"
#include "Qwen4ExpSidecarManifest.h"
#include "../../execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorBuilder.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/IModelLoader.h"
#include "../../memory/BufferArena.h"
#include <limits>

namespace llaminar2
{
ComputeGraph Qwen4ExpGraph::buildMTPGraph(int, const MTPDepthWeights &,
    const MTPForwardInput &, MTPForwardOutput &)
{
    throw std::invalid_argument("QFN shared predictor requires frozen native source bindings and an authenticated separate-source manifest");
}

ComputeGraph Qwen4ExpGraph::buildMTPGraph(int depth, const MTPDepthWeightBindings &weights,
    const MTPForwardInput &input, MTPForwardOutput &output)
{
    if (!config_.flash_next || !model_ctx_ || !model_ctx_->loader() || !arena_ ||
        !prepared_weight_store_ || !retainsMTPGraphCapacity(config_.mtp))
        throw std::invalid_argument("QFN shared predictor requires exact metadata, retained policy, canonical model/prepared authority and admitted arena");
    const auto *manifest = model_ctx_->loader()->predictorManifest();
    if (!manifest || depth != 0 || !input.device.is_valid() || input.batch_size != 1 ||
        input.first_sequence_index != 0 || input.seq_len <= 0 || (!input.kv_cache_only && input.seq_len != 1) ||
        !input.terminal_hidden || !input.kv_cache ||
        (input.device.is_gpu() ? !input.position_ids_device : !input.position_ids))
        throw std::invalid_argument("QFN Dense0 composer requires an explicitly paired source, R1/depth0/cache0, native positions and exact draft or compact shifted-prefill geometry");

    const DeviceId device = input.device;
    const auto rows = Qwen4ExpPredictorRows::admit(input.batch_size, input.seq_len, device,
        input.sequence_lengths, input.sequence_lengths_device);
    // CPU KV/attention consume exact host-owned seq_len. Authenticate it before
    // adding any graph node; a native GPU count is deliberately never read here.
    rows.requireCompactCPUQuery();
    const auto &frozen = mtpBindingsForPolicy(depth, weights);
    const auto source = LayerGraphSourceIdentity::qwen4ExpSharedPredictor(
        static_cast<int>(config_.flash_next->mainLayers()), manifest->predictorMetadata(),
        frozen.source_layer_index, depth, 0);
    const auto identity = FullAttentionLayerIdentity::qwen4ExpSharedPredictor(source);
    const int participants = config_.tp_config ? config_.tp_config->worldSize() : 1;
    const auto layout = Qwen4ExpPredictorProjectionLayout::fromFrozen(
        resolveMTPShiftedKVHeadLayout(config_.mtp, config_.dense_tp_enabled, participants),
        config_, frozen, device);
    // Shifted prefill consumes HC reads and K/V only. Recursive draft roots
    // retain their compact admitted capacity and are never borrowed as unused
    // bucket-wide write targets. Draft keeps both strict direct destinations.
    const auto invocation = input.kv_cache_only ? Qwen4ExpPredictorInvocation::ShiftedPrefill
                                               : Qwen4ExpPredictorInvocation::Draft;
    const auto residual = Qwen4ExpPredictorResidualBinding::fromArena(invocation,
        source, frozen, rows, layout, *arena_, config_.residualWidth());
    const auto globals = Qwen4ExpPredictorGlobalBindings::fromFrozen(config_, weight_bindings_,
        decode_replicated_dense_weight_bindings_, *prepared_weight_store_, device, invocation);
    Qwen4ExpPredictorContext context{identity, rows, config_.d_model,
        static_cast<int>(config_.flash_next->hyperConnections().gate_rank), config_.rms_norm_eps,
        device, mpi_ctx_.get(), "mtp0_"};
    auto buffers = Qwen4ExpPredictorBuilder::activationBuffers(*arena_, output, device, input.kv_cache_only);
    for (const auto &[actual, role] : {
            std::pair{output.embedding, BufferId::MTP_EMBEDDING},
            std::pair{output.norm_hidden, BufferId::MTP_NORM_HIDDEN},
            std::pair{output.concat, BufferId::MTP_CONCAT}})
        if (!actual || actual != arena_->getTensor(role))
            throw std::invalid_argument("QFN predictor output descriptors must borrow their exact declared input-mix arena roles");
    if (!input.kv_cache_only && (!output.logits || output.logits != arena_->getTensor(BufferId::MTP_LOGITS)))
        throw std::invalid_argument("QFN predictor terminal logits must preserve their admitted target-head role");

    std::unique_ptr<IComputeStage> reconstruction;
    if (globals.reconstructEmbedding())
    {
        std::optional<NativeAllreduceRequestRows> native_rows;
        if (device.is_gpu() && rows.tokens().countOwner())
            native_rows.emplace(1, input.seq_len, config_.d_model, rows.tokens().countOwner());
        reconstruction = createTPAllreduceStage(output.embedding,
            static_cast<size_t>(input.seq_len) * config_.d_model, device, source.sourceLayer(), true,
            context.prefix + "embedding_allreduce", BufferId::MTP_EMBEDDING, {}, std::nullopt,
            native_rows, ForwardExecutionPhase::Decode);
        if (!reconstruction)
            throw std::invalid_argument("QFN predictor declared sharded target embedding requires its canonical reconstruction stage");
    }
    ComputeGraph graph;
    const auto embedding_ready = Qwen4ExpPredictorBuilder::appendEmbedding(graph, *arena_, context,
        globals, input, *prepared_weight_store_, std::move(reconstruction));
    const auto mixed = Qwen4ExpPredictorBuilder::appendInputMix(graph, *arena_, context, frozen,
        *prepared_weight_store_, { *input.terminal_hidden, input.terminal_hidden_buffer_id }, invocation, embedding_ready, {});
    const auto layer = toLegacyLayerWeights(frozen.fa_block);
    if (input.kv_cache_only)
    {
        auto append = buildMTPFAKVCacheAppendGraphForLayer(layer, buffers, identity, input.seq_len, 1,
            input.kv_cache, input.position_ids, input.position_ids_device, input.sequence_lengths_device,
            device, source.prefix(), 0, &residual);
        if (append.size() == 0)
            throw std::logic_error("QFN shifted-prefill native K/V fragment is incomplete");
        const auto terminal = append.terminalNode();
        graph.merge(std::move(append), mixed);
        sealInferenceTransactionGraph(graph, terminal);
        return graph;
    }

    auto attention = buildFAAttentionGraphForLayer(layer, buffers, identity, input.seq_len, 1,
        input.kv_cache, input.position_ids, input.position_ids_device, device,
        input.sequence_lengths, input.sequence_lengths_device, source.prefix(), &residual);
    if (attention.size() == 0)
        throw std::logic_error("QFN predictor native dense attention fragment is incomplete");
    const auto attention_ready = attention.terminalNode();
    graph.merge(std::move(attention), mixed);
    auto ffn = buildFFNGraphForLayer(layer, buffers, source, input.seq_len, 1, device,
        input.device_state_publication_stream, input.sequence_lengths_device,
        static_cast<const int32_t *>(input.position_ids_device), &residual);
    if (ffn.size() == 0)
        throw std::logic_error("QFN predictor native routed FFN fragment is incomplete");
    const auto ffn_ready = ffn.terminalNode();
    graph.merge(std::move(ffn), attention_ready);
    const auto head_ready = Qwen4ExpPredictorBuilder::appendHeadRead(graph, *arena_, context,
        frozen, *prepared_weight_store_, ffn_ready);
    const auto logits_ready = Qwen4ExpPredictorBuilder::appendVocabulary(graph, *arena_, context,
        globals, *prepared_weight_store_, serialEquivalentLMHeadPartitionWidth(
            globals.selection().terminal_layout == MTPTerminalLogitsLayout::VocabularyShardPerParticipant,
            globals.selection().mirrored_head ? FinalHeadPolicy::MirroredMTPFullVocabulary
                                             : FinalHeadPolicy::PrimaryFullVocabulary), head_ready);
    const bool spans_global = config_.tp_ctx && !config_.tp_ctx->isLocal() && config_.tp_ctx->degree() > 1;
    const auto collective = resolveMTPTerminalLogitsCollective({
        .layout = globals.selection().terminal_layout, .sidecar_produces_logits = true,
        .spans_multiple_global_ranks = spans_global, .native_local_gpu_tp = config_.hasNativeLocalVocabularyGather()});
    if (collective != MTPTerminalLogitsCollective::None &&
        (!output.gathered_logits || output.gathered_logits != arena_->getTensor(BufferId::MTP_LOGITS_GATHERED)))
        throw std::invalid_argument("QFN predictor terminal gather requires its exact admitted target vocabulary role");
    const auto terminal = addTerminalLogitsCollectiveToGraph(graph, logits_ready, {
        .collective = collective, .local_logits = output.logits, .full_logits = output.gathered_logits,
        .rows = rows.tokens(), .input_id = BufferId::MTP_LOGITS, .output_id = BufferId::MTP_LOGITS_GATHERED,
        .device = device, .stage_name = context.prefix + "lm_head_allgather"});
    sealInferenceTransactionGraph(graph, terminal);
    return graph;
}
} // namespace llaminar2
