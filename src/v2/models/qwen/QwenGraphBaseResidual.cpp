/**
 * @file QwenGraphBaseResidual.cpp
 * @brief Bind declarative Flash Next branch mixers to the shared Qwen graph.
 *
 * This adapter selects immutable semantic weight roles and admitted arena
 * storage. Qwen4ExpHyperconnectionBuilder owns the reusable flat wiring. It
 * never changes the projection width of attention or MoE: those blocks consume
 * and publish H-wide rows, while residual and pipeline transport retain 4H.
 */
#include "QwenGraphBase.h"
#include "../qwen4exp/Qwen4ExpConfig.h"
#include "../qwen4exp/Qwen4ExpHCUpArithmetic.h"
#include "../deepseek4/DeepSeekV4Config.h"
#include "../../execution/compute_stages/stages/deepseek4/DeepSeekV4PrimitiveStage.h"
#include "../../execution/compute_stages/ComputeStageFactory.h"
#include <stdexcept>
#include <utility>

namespace llaminar2
{
namespace
{
/** @brief Require an admitted activation role before any graph node is added. */
Qwen4ExpFloatBinding branchBuffer(ActivationBuffers &buffers, BufferId role)
{
    const auto id = buffers.idFor(role);
    auto *tensor = buffers.get(id);
    if (!tensor)
        throw std::invalid_argument("Flash Next is missing admitted arena buffer " +
                                    std::string(bufferIdName(id)));
    return {*tensor, id};
}

/** @brief Authenticate the three frozen model-owned read weights together. */
Qwen4ExpHyperconnectionReadWeights readWeights(
    const HyperConnectionWeights<const WeightBinding *> &weights,
    PreparedWeightStore *store, DeviceId device)
{
    if (!store || !weights.norm || !weights.norm->immutable || !weights.norm->tensor ||
        !weights.down || !weights.up)
        throw std::invalid_argument("Flash Next branch mixer requires frozen norm/down/up weights and their prepared owner");
    return {Qwen4ExpImmutableWeightBinding(*weights.norm->tensor),
            Qwen4ExpPreparedProjection(*weights.down, *store, device),
            Qwen4ExpPreparedProjection(*weights.up, *store, device)};
}

/**
 * @brief Select query-scoped correction from one actual immutable HC-up source.
 * @param config Frozen model-level quantized activation policy.
 * @param up Authenticated prepared source consumed by the same graph builder.
 * @return Whether this source executes the two-term quantized operation.
 * @throws std::invalid_argument For unsupported source or invalid policy metadata.
 */
bool usesHCUpCorrection(const GraphConfig &config,const Qwen4ExpPreparedProjection &up)
{
    return resolveQwen4ExpHCUpArithmetic(config.qfn_hc_up_activation,
        tensorTypeName(up.tensor()->native_type()))==Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8;
}

/** @brief Keep projection scratch distinct from both original and normalized branches. */
Qwen4ExpHyperconnectionReadBuffers readBuffers(ActivationBuffers &buffers,
    TensorBase *trunk, BufferId trunkId, TensorBase *mixed, BufferId mixedId)
{
    if (!trunk || !mixed)
        throw std::invalid_argument("Flash Next branch mixer requires admitted trunk and compute rows");
    const auto down = branchBuffer(buffers, BufferId::QFN_HC_DOWN);
    return {{*trunk, trunkId}, branchBuffer(buffers, BufferId::QFN_HC_NORMALIZED),
            down, down, branchBuffer(buffers, BufferId::QFN_HC_READ_LOGITS), {*mixed, mixedId}};
}

/** @brief Capture all immutable mixer policy and preserve the grouped verifier arithmetic. */
Qwen4ExpHyperconnectionContext mixerContext(const GraphConfig &config,
    const IMPIContext *mpi, const std::string &prefix, const std::string &terminal,
    Qwen4ExpStageRows rows, DeviceId device,std::optional<DeviceRowRange> projectionRows)
{
    if (!config.flash_next)
        throw std::logic_error("Branch mixing requires an authenticated Flash Next configuration");
    return {.device = device, .mpi = mpi, .prefix = prefix,
            .rows = rows, .hidden = config.d_model,
            .lowRank = static_cast<int>(config.flash_next->hyperConnections().gate_rank),
            .epsilon = config.rms_norm_eps,
            .arithmetic = config.usesMTPGroupedDecodeEquivalentRows() && rows.capacity > 1
                              ? Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier
                              : Qwen4ExpProjectionArithmetic::Ordinary,
            .projectionRows = projectionRows, .readNodeName = terminal,
            .upOperandPrecision = config.qfn_hc_up_activation};
}
} // namespace

Qwen4ExpStageRows QwenGraphBase::modelExtensionRows(int capacity,DeviceId device) const
{
    Qwen4ExpStageRows rows{capacity,std::nullopt};
    if(!forward_input_construction_) return rows;
    const auto &input=*forward_input_construction_;
    if(input.batch_size!=1)
        throw std::invalid_argument("Flash Next extension requires explicit per-request row views for a padded request batch");
    if(capacity!=input.seq_len)
        throw std::invalid_argument("Flash Next primitive rows do not match their complete forward invocation");
    const int32_t *count=nullptr;
    if(device.is_gpu())
    {
        if(auto prefix=prefillCollectiveRows(device,capacity)) count=prefix->countOwner();
        else if(!config_.live_mtp_request_batch_condition &&
                (input.execution_phase==ForwardExecutionPhase::Prefill || config_.grouped_mtp_verifier))
            count=input.sequence_lengths_device;
    }
    else if(input.sequence_lengths)
    {
        if(input.sequence_lengths->size()!=1)
            throw std::invalid_argument("Flash Next CPU row count must name the exact request slot");
        count=input.sequence_lengths->data();
    }
    if(count) rows.liveRows=Qwen4ExpNativeIntInput{device,count,1};
    return rows;
}

DeviceRowRange QwenGraphBase::singleRequestProjectionQueryRows(int capacity, DeviceId device) const
{
    const auto &input = modelForwardInput();
    if (capacity <= 0 || input.seq_len != capacity || input.batch_size != 1 || input.device != device ||
        (input.execution_phase != ForwardExecutionPhase::Prefill && input.execution_phase != ForwardExecutionPhase::Decode) ||
        (input.execution_role != ForwardExecutionRole::MainInference &&
         input.execution_role != ForwardExecutionRole::GroupedMTPVerifier &&
         input.execution_role != ForwardExecutionRole::MTPCondition))
        throw std::invalid_argument("Counted projection requires its exact single-request forward query geometry and role");
    const auto primitives = modelExtensionRows(capacity, device);
    const Qwen4ExpNativeIntInput *native = primitives.liveRows
        ? std::get_if<Qwen4ExpNativeIntInput>(&primitives.liveRows->value) : nullptr;
    if (primitives.liveRows && (!native || native->owner != device || !native->data || native->capacity != 1))
        throw std::invalid_argument("Counted projection lost its original primitive query-count authority");
    if(input.graph_submission_intent==ForwardGraphSubmissionIntent::WorkspaceDeclaration)
        return native ? DeviceRowRange::deviceCounted(capacity,native->data) : DeviceRowRange::fullyActive(capacity);
    if(!executableForwardSubmissionIntent(input.graph_submission_intent))
        throw std::invalid_argument("Counted projection has an unknown construction intent");
    if(input.device_request_query)
    {
        requireDeviceRequestQueryOwnsInput(input);
        if(!native || native->data!=input.device_request_query->lengths())
            throw std::invalid_argument("Counted projection lost the original request query count");
        return DeviceRowRange::deviceCounted(capacity,native->data);
    }
    if (auto query = prefillCollectiveRows(device, capacity))
    {
        if (input.execution_phase != ForwardExecutionPhase::Prefill ||
            input.execution_role != ForwardExecutionRole::MainInference ||
            !devicePrefillChunkOwnsInput(input) || !native ||
            query->countOwner() != native->data ||
            query->countOwner() != input.device_prefill_chunk->chunk_real_rows_device)
            throw std::invalid_argument("Counted projection has a foreign native prefill query authority");
        return *query;
    }
    if (input.device_prefill_chunk)
        throw std::invalid_argument("Counted projection is missing its authenticated native prefill scope");
    if (device.is_cpu())
    {
        if (input.sequence_lengths_device ||
            (input.real_seq_len != 0 && input.real_seq_len != capacity) ||
            (input.bucket_seq_len != 0 && input.bucket_seq_len != capacity) ||
            (input.sequence_lengths && (input.sequence_lengths->size() != 1 || (*input.sequence_lengths)[0] != capacity)))
            throw std::invalid_argument("Counted CPU projection requires exact compact host query rows");
        return native ? DeviceRowRange::deviceCounted(capacity, native->data) : DeviceRowRange::fullyActive(capacity);
    }
    if (device.is_gpu() && input.execution_phase == ForwardExecutionPhase::Decode &&
        input.execution_role == ForwardExecutionRole::GroupedMTPVerifier)
    {
        if (!config_.grouped_mtp_verifier || !config_.usesMTPGroupedDecodeEquivalentRows() ||
            !native || native->data != input.sequence_lengths_device)
            throw std::invalid_argument("Counted projection requires its frozen grouped verifier and original query count");
        // Preserve counted M1 as well as larger verifier prefixes; this is the
        // same native primitive scalar, never a second query publication.
        return DeviceRowRange::deviceCounted(capacity, native->data);
    }
    if (device.is_gpu() && input.execution_phase == ForwardExecutionPhase::Decode && capacity == 1 &&
        (input.execution_role == ForwardExecutionRole::MainInference || input.execution_role == ForwardExecutionRole::MTPCondition) && !native)
        return DeviceRowRange::fullyActive(1);
    throw std::invalid_argument("Counted projection has no authenticated ordinary query or grouped verifier authority");
}

ResidualBlockRead QwenGraphBase::addBranchBlockRead(
    ComputeGraph &graph, const std::string &prefix, ActivationBuffers &buffers,
    int layer_idx, int rows, DeviceId device, ResidualBlock block,
    const Qwen4ExpPredictorResidualBinding *predictor)
{
    if (config_.deepseek_v4)
    {
        if (predictor)
            throw std::invalid_argument("DeepSeek-V4 has no Qwen4Exp predictor residual binding");
        return ResidualBlockRead(addDeepSeekV4BlockRead(graph, prefix, buffers, layer_idx, rows, device, block));
    }
    if(!predictor && forward_input_construction_)applyForwardGraphConstructionIntent(graph,*forward_input_construction_);
    const auto binding = predictor ? predictor->weights().fa_block.flash_next
                                   : layerWeightBindingsForGraph(layer_idx).flash_next;
    if (predictor && (predictor->rows().device() != device ||
                      predictor->rows().tokens().physicalRows() != rows))
        throw std::invalid_argument("Dense0 residual construction must retain its original participant and query rows");
    const bool attention = block == ResidualBlock::Attention;
    const auto &weights = attention ? binding.attention : binding.ffn;
    if (!weights.inject || !prepared_weight_store_)
        throw std::invalid_argument("Flash Next block mixer requires its paired injection projection");
    const auto prepared_read=readWeights(weights,prepared_weight_store_,device);
    const std::string role = attention ? "attn" : "ffn";
    // The existing main path keeps its original primitive/projection policy.
    // Predictor rows instead arrive explicitly from MTPForwardInput, and every
    // HC projection uses that same original scalar with serial-row arithmetic.
    auto context = mixerContext(config_, mpi_ctx_.get(), prefix + "hc_" + role + "_",
        prefix + role + "_norm", predictor ? predictor->rows().primitives() : modelExtensionRows(rows,device), device,
        predictor ? std::optional{predictor->rows().tokens()} :
            (config_.usesMTPGroupedDecodeEquivalentRows() ?
                projectionVerifierRows(device,rows,1,forward_input_construction_ ?
                    forward_input_construction_->sequence_lengths_device : nullptr) : std::nullopt));
    if (predictor)
        context.arithmetic = Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
    else if (usesHCUpCorrection(config_,prepared_read.up))
        context.upProjectionRows = singleRequestProjectionQueryRows(rows, device);
    return ResidualBlockRead(Qwen4ExpHyperconnectionBuilder::appendRead(graph, context,
        {prepared_read,
         Qwen4ExpPreparedProjection(*weights.inject, *prepared_weight_store_, device)},
        readBuffers(buffers, buffers.current_hidden, buffers.idFor(BufferId::HIDDEN_STATE),
                    buffers.normalized, buffers.idFor(BufferId::NORMALIZED)), {}));
}

std::string QwenGraphBase::addBranchBlockWrite(ComputeGraph &graph,
    ResidualBlockRead &&read, ActivationBuffers &buffers,
    const std::string &dependency, const Qwen4ExpPredictorTrunkTarget *predictor_output)
{
    if (!buffers.attn_proj || !buffers.current_hidden || dependency.empty())
        throw std::invalid_argument("Flash Next branch write requires a completed block and original trunk");
    if (auto *dsv4 = std::get_if<DeepSeekV4BlockRead>(&read.value()))
    {
        if (predictor_output)
            throw std::invalid_argument("DeepSeek-V4 has no Qwen4Exp predictor trunk target");
        return addDeepSeekV4BlockWrite(graph, std::move(*dsv4), buffers, dependency);
    }
    return Qwen4ExpHyperconnectionBuilder::appendWrite(graph,
        std::move(std::get<Qwen4ExpHyperconnectionBuilder::BlockRead>(read.value())),
        {{*buffers.attn_proj, buffers.idFor(BufferId::ATTN_PROJ)},
         branchBuffer(buffers, BufferId::QFN_HC_WRITE_LOGITS),
         predictor_output ? predictor_output->binding()
                          : Qwen4ExpFloatBinding{*buffers.current_hidden, buffers.idFor(BufferId::HIDDEN_STATE)}}, {dependency});
}

void QwenGraphBase::addBranchAttentionReadOnly(ComputeGraph &graph,
    const std::string &prefix, ActivationBuffers &buffers, int layer_idx,
    int rows, DeviceId device, const Qwen4ExpPredictorResidualBinding *predictor)
{
    if(!predictor && forward_input_construction_)applyForwardGraphConstructionIntent(graph,*forward_input_construction_);
    if (predictor && (predictor->rows().device()!=device || predictor->rows().tokens().physicalRows()!=rows))
        throw std::invalid_argument("Dense0 KV-only HC read must retain its original rows and participant");
    const auto prepared_read=readWeights(predictor ? predictor->weights().fa_block.flash_next.attention :
        layerWeightBindingsForGraph(layer_idx).flash_next.attention,prepared_weight_store_,device);
    auto context=mixerContext(config_, mpi_ctx_.get(), prefix + "hc_attn_", prefix + "attn_norm",
        predictor ? predictor->rows().primitives() : modelExtensionRows(rows,device),device,
        predictor ? std::optional{predictor->rows().tokens()} : std::nullopt);
    if (predictor) context.arithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
    else if (usesHCUpCorrection(config_,prepared_read.up))
        context.upProjectionRows = singleRequestProjectionQueryRows(rows, device);
    Qwen4ExpHyperconnectionBuilder::appendFinalRead(graph,context,
        prepared_read,
        readBuffers(buffers, buffers.current_hidden, buffers.idFor(BufferId::HIDDEN_STATE),
                    buffers.normalized, buffers.idFor(BufferId::NORMALIZED)), {});
}

std::string QwenGraphBase::addBranchHeadRead(ComputeGraph &graph, TensorBase *hidden,
    TensorBase *normalized_out, const std::string &dependency,
    int rows, DeviceId device, BufferId input_buffer_id)
{
    if (config_.deepseek_v4)
        return addDeepSeekV4HeadRead(graph, hidden, normalized_out, dependency, rows, device, input_buffer_id);
    if(forward_input_construction_)applyForwardGraphConstructionIntent(graph,*forward_input_construction_);
    const auto prepared_read=readWeights(modelHyperConnectionHeadBindings(),prepared_weight_store_,device);
    auto context = mixerContext(config_, mpi_ctx_.get(), "hc_head_", "final_norm",
        modelExtensionRows(rows,device),device,
        config_.usesMTPGroupedDecodeEquivalentRows() ?
            projectionVerifierRows(device,rows,1,forward_input_construction_ ?
                forward_input_construction_->sequence_lengths_device : nullptr) : std::nullopt);
    if (usesHCUpCorrection(config_,prepared_read.up))
        context.upProjectionRows = singleRequestProjectionQueryRows(rows, device);
    return Qwen4ExpHyperconnectionBuilder::appendFinalRead(graph, context,
        prepared_read,
        readBuffers(buffers_.layer_buffers, hidden, input_buffer_id,
                    normalized_out, BufferId::NORMALIZED),
        dependency.empty() ? std::vector<std::string>{} : std::vector<std::string>{dependency});
}

std::string QwenGraphBase::addEmbeddingTrunkExpansion(ComputeGraph &graph,
    TensorBase *embedding, TensorBase *trunk, const std::string &dependency,
    int rows, DeviceId device)
{
    if (!config_.usesResidualTrunk())
        return dependency;
    if (!embedding || !trunk || dependency.empty())
        throw std::invalid_argument("Flash Next embedding expansion requires reconstructed compute rows and trunk storage");
    if (config_.deepseek_v4)
    {
        using Stage = DeepSeekV4PrimitiveStage;
        graph.addNode("embedding_expand", std::make_unique<Stage>(Stage::Params{
            .device_id = device, .mpi_ctx = mpi_ctx_.get(), .rows = modelExtensionRows(rows, device),
            .operation = Stage::HCExpand{config_.deepseek_v4->hcMult(), config_.d_model,
                {*embedding, BufferId::QFN_EMBEDDING_ROWS}, {*trunk, BufferId::HIDDEN_STATE}}}), device);
        graph.addDependency("embedding_expand", dependency);
        return "embedding_expand";
    }
    using Stage = Qwen4ExpPrimitiveStage;
    graph.addNode("embedding_expand", std::make_unique<Stage>(Stage::Params{
        .device_id = device, .mpi_ctx = mpi_ctx_.get(), .rows = modelExtensionRows(rows,device),
        .operation = Stage::ResidualExpand{config_.d_model,
            {*embedding, BufferId::QFN_EMBEDDING_ROWS}, {*trunk, BufferId::HIDDEN_STATE}}}), device);
    graph.addDependency("embedding_expand", dependency);
    return "embedding_expand";
}

TensorBase *QwenGraphBase::embeddingDestination(TensorBase *ordinary_trunk) const
{
    if (!config_.usesResidualTrunk())
        return ordinary_trunk;
    auto *tensor = buffers_.layer_buffers.get(BufferId::QFN_EMBEDDING_ROWS);
    if (!tensor)
        throw std::invalid_argument("Flash Next vocabulary lookup requires QFN_EMBEDDING_ROWS");
    return tensor;
}

// ---------------------------------------------------------------------------
// DeepSeek-V4 Sinkhorn hyper-connections
// ---------------------------------------------------------------------------
namespace
{
/** @brief Require an admitted arena role for a DeepSeek-V4 mixer stage. */
Qwen4ExpFloatBinding dsv4Buffer(ActivationBuffers &buffers, BufferId role)
{
    const auto id = buffers.idFor(role);
    auto *tensor = buffers.get(id);
    if (!tensor)
        throw std::invalid_argument("DeepSeek-V4 is missing admitted arena buffer " + std::string(bufferIdName(id)));
    return {*tensor, id};
}

/** @brief Require a frozen native FP32 operator weight (norm gain, Sinkhorn base/scale). */
Qwen4ExpImmutableWeightBinding dsv4Gain(const WeightBinding *binding, const char *role)
{
    if (!binding || !binding->tensor)
        throw std::invalid_argument(std::string("DeepSeek-V4 hyper-connection requires its frozen ") + role);
    return Qwen4ExpImmutableWeightBinding(*binding->tensor);
}
} // namespace

std::string QwenGraphBase::addDeepSeekV4MixGEMM(ComputeGraph &graph, const std::string &node,
    const WeightBinding *fn, TensorBase *trunk, BufferId trunk_id, int rows, DeviceId device)
{
    if (!fn || !fn->tensor || !trunk)
        throw std::invalid_argument("DeepSeek-V4 hyper-connection requires its frozen fn projection and trunk");
    auto &buffers = buffers_.layer_buffers;
    TensorBase *mixes = buffers.get(buffers.idFor(BufferId::DSV4_HC_MIXES));
    if (!mixes)
        throw std::invalid_argument("DeepSeek-V4 is missing admitted DSV4_HC_MIXES");
    const int n = static_cast<int>(fn->tensor->shape()[0]);
    const int k = static_cast<int>(fn->tensor->shape()[1]);
    if (k != config_.residualWidth())
        throw std::invalid_argument("DeepSeek-V4 hyper-connection fn width differs from the trunk width");
    GEMMStage::Params params{
        .device_id = device,
        .A = trunk,
        .B = fn->tensor,
        .C = mixes,
        .m = rows,
        .n = n,
        .k = k,
        .a_buffer_id = trunk_id,
        .c_buffer_id = buffers.idFor(BufferId::DSV4_HC_MIXES),
        .prepared_ref = preparedRefForGraphWeight(fn, device),
        .prepared_store = prepared_weight_store_,
    };
    graph.addNode(node, ComputeStageFactory::createGEMM(params), device);
    return node;
}

DeepSeekV4BlockRead QwenGraphBase::addDeepSeekV4BlockRead(ComputeGraph &graph, const std::string &prefix,
    ActivationBuffers &buffers, int layer_idx, int rows, DeviceId device, ResidualBlock block)
{
    if (forward_input_construction_) applyForwardGraphConstructionIntent(graph, *forward_input_construction_);
    const auto &model = *config_.deepseek_v4;
    const auto bindings = layerWeightBindingsForGraph(layer_idx);
    const bool attention = block == ResidualBlock::Attention;
    const auto &site = attention ? bindings.deepseek_v4.hc_attn : bindings.deepseek_v4.hc_ffn;
    const std::string role = attention ? "attn" : "ffn";
    const std::string mix_prefix = prefix + "hc_" + role + "_";
    const auto stage_rows = modelExtensionRows(rows, device);
    const std::string mixes = addDeepSeekV4MixGEMM(graph, mix_prefix + "mixes", site.fn, buffers.current_hidden,
                                                   buffers.idFor(BufferId::HIDDEN_STATE), rows, device);
    using Stage = DeepSeekV4PrimitiveStage;
    Stage::HCRead read{
        .hc = model.hcMult(), .hidden = config_.d_model, .head = false,
        .normEps = model.normEps(), .hcEps = model.hcEps(), .outputEps = model.normEps(),
        .sinkhornIters = model.sinkhornIters(),
        .trunk = {*buffers.current_hidden, buffers.idFor(BufferId::HIDDEN_STATE)},
        .mixes = dsv4Buffer(buffers, BufferId::DSV4_HC_MIXES),
        .scale = dsv4Gain(site.scale, "scale"),
        .base = dsv4Gain(site.base, "base"),
        .gain = dsv4Gain(attention ? bindings.attn_norm : bindings.ffn_norm,
                         attention ? "attn_norm (layer read)" : "ffn_norm (layer read)"),
        .mixed = std::nullopt,
        .output = {*buffers.normalized, buffers.idFor(BufferId::NORMALIZED)},
        .post = dsv4Buffer(buffers, BufferId::DSV4_HC_POST),
        .comb = dsv4Buffer(buffers, BufferId::DSV4_HC_COMB),
    };
    const std::string terminal = prefix + role + "_norm";
    graph.addNode(terminal, std::make_unique<Stage>(Stage::Params{
        .device_id = device, .mpi_ctx = mpi_ctx_.get(), .rows = stage_rows, .operation = std::move(read)}), device);
    graph.addDependency(terminal, mixes);
    return DeepSeekV4BlockRead(terminal, mix_prefix, device, stage_rows);
}

std::string QwenGraphBase::addDeepSeekV4BlockWrite(ComputeGraph &graph, DeepSeekV4BlockRead &&read,
    ActivationBuffers &buffers, const std::string &dependency)
{
    using Stage = DeepSeekV4PrimitiveStage;
    const Qwen4ExpFloatBinding trunk{*buffers.current_hidden, buffers.idFor(BufferId::HIDDEN_STATE)};
    Stage::HCWrite write{
        .hc = config_.deepseek_v4->hcMult(), .hidden = config_.d_model,
        .block = {*buffers.attn_proj, buffers.idFor(BufferId::ATTN_PROJ)},
        .post = dsv4Buffer(buffers, BufferId::DSV4_HC_POST),
        .comb = dsv4Buffer(buffers, BufferId::DSV4_HC_COMB),
        .trunk = trunk,
        .output = trunk,
    };
    const std::string node = read.prefix() + "write";
    graph.addNode(node, std::make_unique<Stage>(Stage::Params{
        .device_id = read.device(), .mpi_ctx = mpi_ctx_.get(), .rows = read.rows(), .operation = std::move(write)}),
        read.device());
    graph.addDependency(node, dependency);
    return node;
}

std::string QwenGraphBase::addDeepSeekV4HeadRead(ComputeGraph &graph, TensorBase *hidden,
    TensorBase *normalized_out, const std::string &dependency, int rows, DeviceId device, BufferId input_buffer_id)
{
    if (forward_input_construction_) applyForwardGraphConstructionIntent(graph, *forward_input_construction_);
    const auto &model = *config_.deepseek_v4;
    const auto &head = weight_bindings_.deepseek_v4_head;
    auto &buffers = buffers_.layer_buffers;
    const std::string mixes = addDeepSeekV4MixGEMM(graph, "hc_head_mixes", head.fn, hidden, input_buffer_id, rows, device);
    if (!dependency.empty()) graph.addDependency(mixes, dependency);
    using Stage = DeepSeekV4PrimitiveStage;
    Stage::HCRead read{
        .hc = model.hcMult(), .hidden = config_.d_model, .head = true,
        .normEps = model.normEps(), .hcEps = model.hcEps(), .outputEps = model.normEps(),
        .sinkhornIters = model.sinkhornIters(),
        .trunk = {*hidden, input_buffer_id},
        .mixes = dsv4Buffer(buffers, BufferId::DSV4_HC_MIXES),
        .scale = dsv4Gain(head.scale, "head scale"),
        .base = dsv4Gain(head.base, "head base"),
        .gain = dsv4Gain(modelFinalNormBinding(), "output norm"),
        .mixed = std::nullopt,
        .output = {*normalized_out, BufferId::NORMALIZED},
        .post = std::nullopt,
        .comb = std::nullopt,
    };
    graph.addNode("final_norm", std::make_unique<Stage>(Stage::Params{
        .device_id = device, .mpi_ctx = mpi_ctx_.get(), .rows = modelExtensionRows(rows, device),
        .operation = std::move(read)}), device);
    graph.addDependency("final_norm", mixes);
    return "final_norm";
}
} // namespace llaminar2
