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
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "execution/moe/MoEIntermediateShardStaticGraphAdmission.h"
#include "../qwen4exp/Qwen4ExpConfig.h"
#include "../qwen4exp/Qwen4ExpHCUpArithmetic.h"
#include "../qwen4exp/Qwen4ExpRetainedM1HCFFNPolicy.h"
#ifdef HAVE_ROCM
#include "../qwen4exp/Qwen4ExpRetainedGDNPreAttentionPolicy.h"
#endif
#include "../../loaders/PreparedWeightStore.h"
#include "../../tensors/NativeQuantisedFormatInfo.h"
#include "../../utils/DebugEnv.h"
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

/** @brief Return whether a prepared projection is a native signed-Q8 (codebook 19) matrix. */
bool isOriginalSignedQ8Projection(const Qwen4ExpPreparedProjection &projection)
{
    DeviceNativeQuantisedMatrixDesc source{};
    const auto *engine = projection.store().gemmKernel(projection.ref());
    return engine && engine->inspectPreparedNativeQuantisedMatrixDesc(source) &&
           source.codebook_id == 19 && source.source_identity_present &&
           canonicalDeviceVnniCodebookId(source.source_codebook_id) == 19;
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
            .upOperandPrecision = config.qfn_hc_up_activation,
            .verifierKernelPolicy = projectionRows && rows.capacity == 4 && config.moe.routed_expert_plan
                ? config.moe.routed_expert_plan->local_gpu_expert_kernel_policy : LocalGPUExpertKernelPolicy::Generic};
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

Qwen4ExpHyperconnectionBuilder::BlockRead QwenGraphBase::addBranchBlockRead(
    ComputeGraph &graph, const std::string &prefix, ActivationBuffers &buffers,
    int layer_idx, int rows, DeviceId device, ResidualBlock block,
    const Qwen4ExpPredictorResidualBinding *predictor)
{
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
    if(!predictor&&config_.qfn_hc_norm_q8_candidate&&device.is_gpu()&&rows==1&&
       forward_input_construction_&&forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
        context.branchNormExecution=Qwen4ExpHCBranchNormExecution::PublishOriginalSingleQ8;
    if(!predictor&&config_.qfn_hc_short_k_read_candidate&&device.is_gpu()&&rows==1&&
       forward_input_construction_&&forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
        context.readExecution=Qwen4ExpHCReadExecution::ProducerGridShortK;
    if(!predictor&&config_.qfn_hc_down_publish_candidate&&rows==1&&forward_input_construction_&&
       forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
    {
        context.downPublication=Qwen4ExpHCDownPublication::ReducerEpilogue;
        if(config_.qfn_hc_down_lookahead_candidate)
            context.downProducerProgram=HCDownProducerProgram::TwoK32Lookahead;
    }
    if(!predictor&&device.is_gpu()&&rows==1&&forward_input_construction_&&
       forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference&&
       context.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&!context.projectionRows&&
       usesHCUpCorrection(config_,prepared_read.up)&&
       context.readExecution==Qwen4ExpHCReadExecution::Staged&&
       context.branchNormExecution==Qwen4ExpHCBranchNormExecution::Staged&&
       context.downPublication==Qwen4ExpHCDownPublication::Staged)
    {
        const bool rocm_retained_ffn = !attention && device.is_rocm() &&
            useQwen4ExpRetainedM1HCFFN(
                qfnRetainedM1HCFFNEnabled(config_.moe.routed_expert_plan &&
                    config_.moe.routed_expert_plan->usesGFX906MI50Kernels() &&
                    !hasMappedGPUCPUIntermediateShardTopology(*config_.moe.routed_expert_plan) &&
                    config_.flash_next->hasRetainedM1HCFFNSources(layer_idx)),
                forward_input_construction_->execution_phase,
                forward_input_construction_->execution_role,
                forward_input_construction_->seq_len,
                forward_input_construction_->batch_size);
        const bool rocm_mapped_retained_ffn = !attention && device.is_rocm() &&
            config_.moe.routed_expert_plan &&
            config_.moe.routed_expert_plan->usesGFX906MI50Kernels() &&
            hasMappedGPUCPUIntermediateShardTopology(*config_.moe.routed_expert_plan) &&
            qfnROCmMappedRetainedM1HCFFNCandidate();
        const bool cuda_retained_ffn = !attention && device.is_cuda() &&
            config_.moe.routed_expert_plan &&
            hasCUDAMappedRetainedM1OverlayTopology(*config_.moe.routed_expert_plan) &&
            qfnCUDAMappedRetainedM1HCFFNCandidate();
#ifdef HAVE_ROCM
        const bool rocm_retained_gdn_pre_attention = attention && device.is_rocm() &&
            config_.flash_next->mixers().at(static_cast<std::size_t>(layer_idx)) == Qwen4ExpMixer::GatedDeltaNet &&
            usesRetainedGDNPreAttention(
                qfnRetainedGDNPreAttentionEnabled(
                    config_.moe.routed_expert_plan &&
                    config_.moe.routed_expert_plan->usesGFX906MI50Kernels() &&
                    config_.flash_next->supportsRetainedGDN(
                        static_cast<std::size_t>(layer_idx), debugEnv().gemm.gdn_projection_arithmetic)),
                device, config_.moe.routed_compute_policy,
                forward_input_construction_->execution_phase,
                forward_input_construction_->execution_role,
                forward_input_construction_->seq_len,
                forward_input_construction_->batch_size);
        const bool rocm_retained_qsa_pre_attention = attention && device.is_rocm() &&
            config_.flash_next->mixers().at(static_cast<std::size_t>(layer_idx)) == Qwen4ExpMixer::SparseAttention &&
            usesRetainedROCmQSAAttention(
                qfnRetainedROCmQSAAttentionEnabled(
                    config_.moe.routed_expert_plan &&
                    config_.moe.routed_expert_plan->usesGFX906MI50Kernels()),
                device,
                forward_input_construction_->execution_phase,
                forward_input_construction_->execution_role,
                forward_input_construction_->seq_len,
                forward_input_construction_->batch_size);
        if(device.is_rocm()&&!rocm_retained_ffn&&!rocm_mapped_retained_ffn&&
           !rocm_retained_gdn_pre_attention&&!rocm_retained_qsa_pre_attention&&
           context.hidden%32==0&&context.lowRank>=64&&context.lowRank<=512&&context.lowRank%32==0&&
           isOriginalSignedQ8Projection(prepared_read.down)&&
           isOriginalSignedQ8Projection(prepared_read.up)&&
           weights.inject->tensor&&weights.inject->tensor->native_type()==TensorType::FP32)
        {
            // Fuse all five participant-local HC read+inject stages into one
            // exact resident 1024-thread kernel across all admitted ROCm topologies.
            context.readExecution=Qwen4ExpHCReadExecution::PersistentOriginalPhases;
        }
#endif
        (void)cuda_retained_ffn;
    }
    return Qwen4ExpHyperconnectionBuilder::appendRead(graph, context,
        {prepared_read,
         Qwen4ExpPreparedProjection(*weights.inject, *prepared_weight_store_, device)},
        readBuffers(buffers, buffers.current_hidden, buffers.idFor(BufferId::HIDDEN_STATE),
                    buffers.normalized, buffers.idFor(BufferId::NORMALIZED)), {});
}

std::string QwenGraphBase::addBranchBlockWrite(ComputeGraph &graph,
    Qwen4ExpHyperconnectionBuilder::BlockRead &&read, ActivationBuffers &buffers,
    const std::string &dependency, const Qwen4ExpPredictorTrunkTarget *predictor_output)
{
    if (!buffers.attn_proj || !buffers.current_hidden || dependency.empty())
        throw std::invalid_argument("Flash Next branch write requires a completed block and original trunk");
    return Qwen4ExpHyperconnectionBuilder::appendWrite(graph, std::move(read),
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
    if(!predictor&&config_.qfn_hc_norm_q8_candidate&&device.is_gpu()&&rows==1&&
       forward_input_construction_&&forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
        context.branchNormExecution=Qwen4ExpHCBranchNormExecution::PublishOriginalSingleQ8;
    if(!predictor&&config_.qfn_hc_short_k_read_candidate&&device.is_gpu()&&rows==1&&
       forward_input_construction_&&forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
        context.readExecution=Qwen4ExpHCReadExecution::ProducerGridShortK;
    if(!predictor&&config_.qfn_hc_down_publish_candidate&&rows==1&&forward_input_construction_&&
       forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
    {
        context.downPublication=Qwen4ExpHCDownPublication::ReducerEpilogue;
        if(config_.qfn_hc_down_lookahead_candidate)
            context.downProducerProgram=HCDownProducerProgram::TwoK32Lookahead;
    }
    if(!predictor&&device.is_gpu()&&rows==1&&forward_input_construction_&&
       forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference&&
       context.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&!context.projectionRows&&
       usesHCUpCorrection(config_,prepared_read.up)&&
       context.readExecution==Qwen4ExpHCReadExecution::Staged&&
       context.branchNormExecution==Qwen4ExpHCBranchNormExecution::Staged&&
       context.downPublication==Qwen4ExpHCDownPublication::Staged)
    {
        if(device.is_rocm()&&!context.rows.liveRows&&context.hidden%32==0)
            context.branchNormExecution=Qwen4ExpHCBranchNormExecution::PublishOriginalSingleQ8;
        if(device.is_rocm()&&context.lowRank>=64&&context.lowRank<=512&&4*context.hidden>=4*context.lowRank&&
           isOriginalSignedQ8Projection(prepared_read.up))
            context.readExecution=Qwen4ExpHCReadExecution::ProducerGridShortK;
        if(device.is_rocm()&&context.lowRank>=64&&context.lowRank<=512&&context.lowRank%32==0&&
           context.hidden%8==0&&isOriginalSignedQ8Projection(prepared_read.down))
            context.downPublication=Qwen4ExpHCDownPublication::ReducerEpilogue;
    }
    Qwen4ExpHyperconnectionBuilder::appendFinalRead(graph,context,
        prepared_read,
        readBuffers(buffers, buffers.current_hidden, buffers.idFor(BufferId::HIDDEN_STATE),
                    buffers.normalized, buffers.idFor(BufferId::NORMALIZED)), {});
}

std::string QwenGraphBase::addBranchHeadRead(ComputeGraph &graph, TensorBase *hidden,
    TensorBase *normalized_out, const std::string &dependency,
    int rows, DeviceId device, BufferId input_buffer_id)
{
    if(forward_input_construction_)applyForwardGraphConstructionIntent(graph,*forward_input_construction_);
    const auto prepared_read=readWeights(modelHyperConnectionHeadBindings(),prepared_weight_store_,device);
    auto context = mixerContext(config_, mpi_ctx_.get(), "hc_head_", "final_norm",
        modelExtensionRows(rows,device),device,
        config_.usesMTPGroupedDecodeEquivalentRows() ?
            projectionVerifierRows(device,rows,1,forward_input_construction_ ?
                forward_input_construction_->sequence_lengths_device : nullptr) : std::nullopt);
    if (usesHCUpCorrection(config_,prepared_read.up))
        context.upProjectionRows = singleRequestProjectionQueryRows(rows, device);
    if(config_.qfn_hc_norm_q8_candidate&&device.is_gpu()&&rows==1&&
       forward_input_construction_&&forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
        context.branchNormExecution=Qwen4ExpHCBranchNormExecution::PublishOriginalSingleQ8;
    if(config_.qfn_hc_short_k_read_candidate&&device.is_gpu()&&rows==1&&
       forward_input_construction_&&forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
        context.readExecution=Qwen4ExpHCReadExecution::ProducerGridShortK;
    if(config_.qfn_hc_down_publish_candidate&&rows==1&&forward_input_construction_&&
       forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference)
    {
        context.downPublication=Qwen4ExpHCDownPublication::ReducerEpilogue;
        if(config_.qfn_hc_down_lookahead_candidate)
            context.downProducerProgram=HCDownProducerProgram::TwoK32Lookahead;
    }
    if(device.is_gpu()&&rows==1&&forward_input_construction_&&
       forward_input_construction_->execution_phase==ForwardExecutionPhase::Decode&&
       forward_input_construction_->execution_role==ForwardExecutionRole::MainInference&&
       context.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&!context.projectionRows&&
       usesHCUpCorrection(config_,prepared_read.up)&&
       context.readExecution==Qwen4ExpHCReadExecution::Staged&&
       context.branchNormExecution==Qwen4ExpHCBranchNormExecution::Staged&&
       context.downPublication==Qwen4ExpHCDownPublication::Staged)
    {
        if(device.is_rocm()&&!context.rows.liveRows&&context.hidden%32==0)
            context.branchNormExecution=Qwen4ExpHCBranchNormExecution::PublishOriginalSingleQ8;
        if(device.is_rocm()&&context.lowRank>=64&&context.lowRank<=512&&4*context.hidden>=4*context.lowRank&&
           isOriginalSignedQ8Projection(prepared_read.up))
            context.readExecution=Qwen4ExpHCReadExecution::ProducerGridShortK;
        if(device.is_rocm()&&context.lowRank>=64&&context.lowRank<=512&&context.lowRank%32==0&&
           context.hidden%8==0&&isOriginalSignedQ8Projection(prepared_read.down))
            context.downPublication=Qwen4ExpHCDownPublication::ReducerEpilogue;
    }
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
    if (!config_.flash_next)
        return dependency;
    if (!embedding || !trunk || dependency.empty())
        throw std::invalid_argument("Flash Next embedding expansion requires reconstructed compute rows and trunk storage");
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
    if (!config_.flash_next)
        return ordinary_trunk;
    auto *tensor = buffers_.layer_buffers.get(BufferId::QFN_EMBEDDING_ROWS);
    if (!tensor)
        throw std::invalid_argument("Flash Next vocabulary lookup requires QFN_EMBEDDING_ROWS");
    return tensor;
}
} // namespace llaminar2
