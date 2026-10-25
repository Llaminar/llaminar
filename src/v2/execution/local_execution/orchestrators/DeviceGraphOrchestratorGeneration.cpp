/**
 * @file DeviceGraphOrchestratorGeneration.cpp
 * @brief Ordinary captured generation and terminal request retirement using the existing request/controller owner.
 *
 * Prefill sampling consumes no KV row. Each subsequent retained transaction
 * consumes the pending device token, then publishes one response and its next
 * frontier. No host token, speculative verifier or fake forward participates.
 * Sampler recordings borrow arena storage; their parent retires first. HIP
 * retains one complete transaction, never a host-submitted list of model stages.
 */
#include "DeviceGraphOrchestrator.h"
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "PipelineForwardGraphEdges.h"
#include "backends/BackendManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "utils/Logger.h"
#include <exception>
#include <stdexcept>

namespace llaminar2
{
std::size_t DeviceGraphOrchestrator::sparseQueryProgramCount() const
{
    if (!graph_builder_ || !graph_builder_->config().flash_next) return 1u;
    const auto &model = *graph_builder_->config().flash_next;
    const auto &mixers = model.mixers();
    if (model.mainLayers() > mixers.size())
        throw std::logic_error("Sparse-query program inventory lost validated main-layer geometry");
    return std::any_of(mixers.begin(), mixers.begin() + model.mainLayers(),
        [](auto mixer) { return mixer == Qwen4ExpMixer::SparseAttention; }) ? kSparseQueryExecutionProgramCount : 1u;
}

void DeviceGraphOrchestrator::admitSparseQueryProgram(const DeviceGenerationAdmissionRequest &admission,
    SparseQueryAdmissionBoundary boundary)
{
    if (!admission.valid() || compiling_sparse_query_program_ || sparseQueryParentLaunched())
        throw std::logic_error("Sparse-query program selection requires a valid idle generation admission");
    // A local forced controller has no response budget authority. Preserve
    // the already-proved outer host response rather than deriving a program
    // from its budget0. Standalone proved forced admissions resolve P+1 in
    // the same common contract; missing proof retains the complete program.
    if (boundary == SparseQueryAdmissionBoundary::ResidentGeneration &&
        admission.depth_policy.isForwardOnly() && admitted_sparse_query_program_ &&
        admitted_sparse_query_program_->boundary == SparseQueryAdmissionBoundary::HostControlledResponse)
        return;
    admitted_sparse_query_program_ = SparseQueryProgramAdmission{
        sparseQueryProgramCount() == 1u ? SparseQueryExecutionProgram::RankingPossible :
            resolveSparseQueryExecutionProgram(admission), boundary};
}

bool DeviceGraphOrchestrator::admitSparseQueryExecutionProgram(const DeviceGenerationAdmissionRequest &admission)
{
    if (!admission.valid() || device_generation_storage_.active_request_count != 0 ||
        active_device_generation_admission_ || compiling_sparse_query_program_ || sparseQueryParentLaunched() ||
        (admitted_sparse_query_program_ &&
         admitted_sparse_query_program_->boundary == SparseQueryAdmissionBoundary::HostControlledResponse))
        return false;
    admitSparseQueryProgram(admission, SparseQueryAdmissionBoundary::HostControlledResponse);
    return true;
}

void DeviceGraphOrchestrator::endSparseQueryExecutionProgram() noexcept
{
    if (admitted_sparse_query_program_ &&
        admitted_sparse_query_program_->boundary == SparseQueryAdmissionBoundary::HostControlledResponse)
        admitted_sparse_query_program_.reset();
}

std::optional<ForwardGraphSignature> &DeviceGraphOrchestrator::pairedVerifierForwardSignature(
    MTPVerifierPreparationGraphCache &cache) const
{
    return cache.paired_forward_signatures[sparseQueryExecutionProgramIndex(sparseQueryGraphProgram())];
}

void DeviceGraphOrchestrator::prepareRetainedStorageBinding()
{
    const auto declared=graph_builder_?graph_builder_->retainedStorageBoundary(state_.device_id):nullptr;
    if(retained_storage_boundary_&&declared!=retained_storage_boundary_)
        throw std::logic_error("Immutable storage lane changed while captured generation retained its addresses");
    retained_storage_boundary_=declared;
    if(declared)
    {
        if(declared->device()!=state_.device_id)
            throw std::logic_error("Immutable storage admission names another GPU participant");
        declared->bindGenerationControl({device_generation_storage_.control_device+sampling_math::kDeviceGenerationControlOk,
            device_generation_storage_.control_device+sampling_math::kDeviceGenerationControlErrorCode});
    }
}
bool DeviceGraphOrchestrator::beginRetainedStorageAdmission(void *stream)
{
    prepareRetainedStorageBinding();
    if(retained_storage_boundary_)retained_storage_boundary_->beginAdmission(session_epoch_,stream);
    return true;
}

std::span<const DeviceControlledLoopEntryFragment> DeviceGraphOrchestrator::retainedStorageTerminalFragments()
{
    if(!retained_storage_boundary_)return {};
    void *stream=deviceGenerationLoopGraph().stream.get();
    if(!stream)throw std::logic_error("Immutable storage terminal requires its exact generation stream");
    if(retained_storage_terminal_capture_&&retained_storage_terminal_capture_->executionStream()!=stream)
        throw std::logic_error("Immutable storage terminal stream changed before retained graph retirement");
    if(!retained_storage_terminal_capture_)
    {
        auto &worker=GPUDeviceContextPool::instance().getContext(state_.device_id);
        retained_storage_terminal_capture_.bind(deviceGenerationLoopGraph().stream,worker.createGraphCapture(stream));
        ScopedBackendGraphCapture recording(worker,*retained_storage_terminal_capture_.get(),"immutable_storage_generation_close");
        if(!recording.begin())throw std::runtime_error("Immutable storage terminal recording failed");
        retained_storage_boundary_->recordTerminalClose(stream);recording.finish();
        if(!retained_storage_terminal_capture_->instantiate())
            throw std::runtime_error("Immutable storage terminal executable materialization failed");
        retained_storage_terminal_fragment_[0]={"immutable storage terminal close",retained_storage_terminal_capture_.get()};
    }
    return retained_storage_terminal_fragment_;
}

bool DeviceGraphOrchestrator::submitRetainedStorageTerminal()
{
    if(!retained_storage_boundary_)return true;
    retained_storage_boundary_->requireArmedForLaunch();
    if(!retained_storage_terminal_capture_||!retained_storage_terminal_capture_->hasExecutable())
        throw std::logic_error("Hosted terminal ticket names an unprepared immutable storage close branch");
    return retained_storage_terminal_capture_->launchOnStream(deviceGenerationLoopGraph().stream.get());
}
void DeviceGraphOrchestrator::releaseRetainedStorageGenerationTopology() noexcept
{
    for (auto &parent : sparse_query_generation_parents_) parent.release();
    compiling_sparse_query_program_.reset();
    admitted_sparse_query_program_.reset();
    // The terminal owner retained the original stream while the parent and its
    // child family retired. Its own recording now retires before the last stream.
    retained_storage_terminal_fragment_={};
    retained_storage_terminal_capture_.reset();
}
void DeviceGraphOrchestrator::retireRetainedStorageForRequestReset(void *stream)
{
    if(!retained_storage_boundary_ || !retained_storage_boundary_->hasActiveInterval())return;
    auto *backend=getBackendFor(state_.device_id);
    if(!state_.device_id.is_gpu() || !backend || !stream ||
        reinterpret_cast<std::uintptr_t>(stream)<=2 || !device_generation_terminal_host_ready_event_)
        throw std::logic_error("Immutable storage reset terminal requires its exact GPU stream and setup-owned terminal event");
    // All prior forward/publication producers have joined this reset stream.
    // A standalone prefill has no complete generation parent to publish close;
    // this is its terminal request boundary, before protocol/session mutation.
    // The DGO's request API owns this event exclusively: response observation
    // and reset cannot overlap. Reusing it creates no new allocation or transfer.
    // Do not requireHealthy here: a failed storage worker still owes its join.
    retained_storage_boundary_->recordTerminalClose(stream);
    PerfStatsCollector::addCounter("immutable_lookup","reset_terminal_closes",1.0,
        "request_reset",state_.device_id.toString());
    if(!backend->recordEvent(device_generation_terminal_host_ready_event_.get(),state_.device_id.ordinal,stream) ||
        !backend->waitForEvent(device_generation_terminal_host_ready_event_.get(),state_.device_id.ordinal))
        throw std::runtime_error("Immutable storage reset terminal event publication/completion failed");
    // Service await authenticates the mapped close immediately. Only this exact
    // terminal event (not a stream/device-wide sync) makes that publication safe
    // to observe. No token, stage or retained-transaction boundary waits here.
    PerfStatsCollector::addCounter("immutable_lookup","reset_terminal_waits",1.0,
        "request_reset",state_.device_id.toString());
    awaitRetainedStorageTerminal();
}
void DeviceGraphOrchestrator::awaitRetainedStorageTerminal()
{
    if(!retained_storage_boundary_)return;
    std::exception_ptr failure;
    try{retained_storage_boundary_->awaitTerminal();}catch(...){failure=std::current_exception();}
    // Both successful and failed workers have joined before counters become
    // observable. Recording adds no device read, transfer or synchronization.
    try{retained_storage_perf_stats_.publish(retained_storage_boundary_->terminalObservation(),state_.device_id.toString());}
    catch(...){if(!failure)throw;} // Preserve a more precise terminal-lifecycle failure when no joined observation exists.
    if(failure)std::rethrow_exception(failure);
}

bool DeviceGraphOrchestrator::observeOrdinaryGenerationForwardCompletion(int64_t completed_invocations)
{
    const auto &loop = deviceGenerationLoopGraph();
    std::string error;
    if (loop.ordinary_hosted_forward &&
        loop.execution_kind == MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedDispatchTicketPublisher)
    {
        /*
         * Hosted ExpertOverlay forwards pass through
         * replayRetainedDecodeGraph(), which records every real replay as it is
         * submitted. The terminal only authenticates that the same retained
         * cache entry survived; publishing composed-parent evidence here would
         * double-count work that did not execute inside a parent.
         */
        const auto retained = forward_engine_ && loop.ordinary_forward_identity
            ? forward_engine_->retainedDecodeGraph(*loop.ordinary_forward_identity, &error) : std::nullopt;
        if (!retained || !loop.ordinary_hosted_forward ||
            *retained != *loop.ordinary_hosted_forward || !loop.source_fragments.empty())
        {
            LOG_ERROR("Ordinary ExpertOverlay terminal lost its retained forward identity: " << error);
            return false;
        }
        return completed_invocations >= 0;
    }
    if (!forward_engine_ || !loop.ordinary_forward_identity || loop.source_fragments.size() != 1 ||
        !forward_engine_->observeComposedForwardCompletion(*loop.ordinary_forward_identity,
            loop.source_fragments.front().capture, completed_invocations, &error))
    {
        LOG_ERROR("Ordinary generation terminal lost its captured forward identity: " << error);
        return false;
    }
    return true;
}

OrdinaryForwardRetentionKind DeviceGraphOrchestrator::ordinaryForwardRetentionKind() const
{
    if (!graph_builder_)
        throw std::logic_error("Ordinary forward retention requires its admitted graph builder");
    const auto execution = deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth);
    const bool boundary = hasHeterogeneousCollectiveExecutionDomain() ||
        static_cast<bool>(moe_overlay_inference_transaction_coordinator_);
    OrdinaryForwardDomain domain = OrdinaryForwardDomain::ParticipantLocal;
    if (boundary && graph_builder_ && graph_builder_->config().moe.routed_expert_plan &&
        graph_builder_->config().moe.routed_expert_plan->usesExpertOverlayAuthority())
    {
        const auto &moe = graph_builder_->config().moe;
        // The builder owns the admitted residency registry and its exact epoch
        // arena. Placement snapshots may legitimately change during maintenance;
        // they are not graph/export identity and are never read to select replay.
        const auto declared = graph_builder_->deviceMoEOverlayEpochExecutionBinding(state_.device_id);
        if (!moe.expert_overlay_residency_authority || !moe.expert_overlay_participant_residency ||
            !declared || declared.arena != moe_overlay_epoch_execution_binding_.arena ||
            declared.request_slot != moe_overlay_epoch_execution_binding_.request_slot ||
            !moe_overlay_epoch_execution_binding_ ||
            moe_overlay_epoch_execution_binding_.arena->deviceId() != state_.device_id ||
            moe_overlay_epoch_execution_binding_.request_slot >=
                moe_overlay_epoch_execution_binding_.arena->requestSlotCapacity())
            throw std::logic_error("Hosted ordinary forward lacks its exact admitted overlay and epoch owner");
        domain = OrdinaryForwardDomain::HostedExpertOverlay;
    }
    return resolveOrdinaryForwardRetention(execution, domain);
}

std::optional<DeviceGraphOrchestrator::OrdinaryGenerationForward>
DeviceGraphOrchestrator::materializeOrdinaryGenerationForward(
    const int32_t *condition, OrdinaryForwardRetentionKind retention, std::string *error)
{
    const auto fail = [&](const std::string &reason) -> std::optional<OrdinaryGenerationForward> {
        if (error) *error = reason;
        return std::nullopt;
    };
    // Resolve the requested ownership before preparing any model nodes. A
    // failed native export never selects the hosted arm.
    const auto admitted = ordinaryForwardRetentionKind();
    if (retention == OrdinaryForwardRetentionKind::HostedExpertOverlay && retention != admitted)
        return fail("Hosted ordinary forward is outside its admitted ExpertOverlay domain");
    if (retention != OrdinaryForwardRetentionKind::NativeComposable &&
        retention != OrdinaryForwardRetentionKind::HostedExpertOverlay)
        return fail("Unknown ordinary forward retention kind");
    auto *const backend = getBackendFor(state_.device_id);
    void *const stream = deviceGenerationLoopGraph().stream.get();
    if (!backend || !stream || !condition || !state_.kv_cache || !request_position_ids_dev_ ||
        condition != device_resident_logical_sequence_state_storage_.next_condition_tokens_device)
        return fail("Resident ordinary forward requires its exact logical token owner, stream and owned KV position");
    // A receive writes directly into the participant's existing hidden bank.
    // This is deliberately a self-binding: the engine must not insert a host
    // handoff or copy/rebind the preceding participant's tensor during replay.
    ForwardInput input;
    input.external_hidden_state = pp_stage_config_ && !pp_stage_config_->has_embedding
        ? state_.hidden.get() : nullptr;
    input.token_ids_device = condition;
    input.position_ids_device = request_position_ids_dev_;
    input.device_decode_position = DeviceDecodePositionBinding{
        backend, state_.kv_cache->deviceSequenceCachedTokenCountPtr(0),
        static_cast<int32_t *>(request_position_ids_dev_)};
    input.position_policy = ForwardPositionPolicy::ExplicitRows;
    input.execution_role = ForwardExecutionRole::MainInference;
    input.execution_phase = ForwardExecutionPhase::Decode;
    input.sparse_query_execution_program = sparseQueryForwardProgram(input.execution_phase);
    input.graph_submission_intent = ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;
    if (pipeline_forward_edges_ && !pipeline_forward_edges_->hasHeterogeneousBoundary())
        input.pipeline_activation_ownership = ForwardPipelineActivationOwnership::NativeGenerationParent;
    input.batch_size = input.seq_len = input.real_seq_len = 1;
    input.position_offset = input.token_offset = 1;
    input.device_state_publication_stream = stream;
    input.device = state_.device_id;
    input.kv_cache = state_.kv_cache.get();
    if (!bindShiftedMTPPrefillTransaction(input, 1) || !bindMTPMainTerminalHiddenTransaction(input, 1))
        return fail("Ordinary forward cannot bind the retained model-state producers");
    setBuffers(state_.toModelBuffers());
    ensureForwardEngine();
    ForwardOutput output;
    output.logits = state_.logits.get();
    output.hidden = state_.hidden.get();
    std::optional<ForwardGraphSignature> signature;
    if (!forward_engine_->execute(input, output, *this, &signature) || !signature)
        return fail("Ordinary resident-input forward could not materialize its exact identity");
    std::string detail;
    if (retention == OrdinaryForwardRetentionKind::HostedExpertOverlay)
    {
        const auto forward = forward_engine_->retainedDecodeGraph(*signature, &detail);
        if (!forward)
            return fail("Ordinary hosted ExpertOverlay requires its complete retained body: " + detail);
        PerfStatsCollector::addCounter("forward_graph", "ordinary_hosted_forward_exports", 1.0,
            "graph_setup", state_.device_id.toString());
        return OrdinaryGenerationForward{*forward, *signature};
    }
    const auto forward = forward_engine_->deviceLoopGraphTemplate(*signature, &detail);
    if (!forward)
        return fail("Ordinary generation requires a complete native-composable forward: " + detail);
    return OrdinaryGenerationForward{*forward, *signature};
}

bool DeviceGraphOrchestrator::materializeOrdinaryDeviceGenerationLoopGraph(
    DeviceGenerationSamplingMode sampling_mode, std::string *error,
    OrdinaryGenerationComposition composition)
{
    using namespace sampling_math;
    using SamplerStage = OrdinaryGenerationSamplingStage;
    using Plan = OrdinaryGenerationGraphPlan;
    auto &loop = deviceGenerationLoopGraph();
    const auto fail = [&](const std::string &reason) {
        // An invalid second materialization must not destroy a graph which
        // already owns an in-flight request. Its existing terminal/reset edge
        // remains responsible for retirement, even on the failure path.
        if (!loop.launched)
            loop.invalidateGraph();
        if (error) *error = reason;
        return false;
    };
    if (error) error->clear();
    const bool native_no_peer_tail = composition == OrdinaryGenerationComposition::PipelineNoPeerOverlayTail;
    const bool pipeline_tail = composition == OrdinaryGenerationComposition::PipelineTail || native_no_peer_tail;
    const bool expert_overlay = composition == OrdinaryGenerationComposition::ExpertOverlay ||
        composition == OrdinaryGenerationComposition::PipelineDomainExpertOverlayTail || native_no_peer_tail;
    const bool pipeline_domain_tail = composition == OrdinaryGenerationComposition::PipelineDomainTail ||
        composition == OrdinaryGenerationComposition::PipelineDomainExpertOverlayTail;
    if (native_no_peer_tail)
    {
        if (!pp_stage_config_ || !graph_builder_ || !moe_overlay_epoch_execution_binding_ ||
            moe_overlay_epoch_execution_binding_.arena->deviceId() != state_.device_id ||
            moe_overlay_epoch_execution_binding_.request_slot >=
                moe_overlay_epoch_execution_binding_.arena->requestSlotCapacity() ||
            (pipeline_forward_edges_ && pipeline_forward_edges_->hasHeterogeneousBoundary()) ||
            usesParticipantLocalDeviceMoERebalanceController())
            return fail("Native no-peer pipeline tail has no exact epoch, scope or parent ownership");
        const auto &config = graph_builder_->config();
        const auto &authority = config.moe.expert_overlay_residency_authority;
        if (!authority || config.pp_layer_offset != pp_stage_config_->first_layer ||
            config.n_layers != pp_stage_config_->last_layer - pp_stage_config_->first_layer)
            return fail("Native no-peer pipeline tail has no exact admitted model-layer authority");
        try
        {
            MoEOverlayNoPeerResidency::requireNativePipelineTail(
                authority->noPeerResidency(), *pp_stage_config_, state_.device_id, config.moe.num_experts);
        }
        catch (const std::exception &exception)
        {
            return fail(exception.what());
        }
    }
    // This compiler accepts a complete participant-local graph. A pipeline or
    // sparse boundary needs its explicit transfer/follower composition, not an
    // apparently successful loop around only this participant's model shard.
    if (!active_device_generation_admission_ ||
        !active_device_generation_admission_->depth_policy.isOrdinary() ||
        active_device_generation_admission_->request_count != 1 ||
        !active_device_generation_admission_->ordinary_sampling ||
        !state_.kv_cache || !state_.logits || !request_position_ids_dev_ ||
        activeMainLogitsAreColumnParallel() ||
        (pp_stage_config_ && (!pp_stage_config_->has_lm_head ||
            (!pp_stage_config_->has_embedding && !pipeline_tail &&
             !pipeline_domain_tail))) ||
        (pipeline_domain_tail &&
            (!pipeline_forward_edges_ || !pipeline_forward_edges_->hasHeterogeneousBoundary())) ||
        expert_overlay !=
            static_cast<bool>(moe_overlay_epoch_execution_binding_))
        return fail("Ordinary generation requires an admitted scalar sampler and an exact local, pipeline-tail, or ExpertOverlay composition");

    const auto &admission = *active_device_generation_admission_;
    const auto &law = *admission.ordinary_sampling;
    const bool greedy = law.is_greedy();
    if ((greedy ? DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic) != sampling_mode)
        return fail("Ordinary materialization differs from the admitted sampling law");
    if (law.presence_penalty != mtp_request_penalty_policy_.presence_penalty ||
        law.frequency_penalty != mtp_request_penalty_policy_.frequency_penalty ||
        law.repetition_penalty != mtp_request_penalty_policy_.repetition_penalty)
        return fail("Ordinary sampling penalties differ from the published request policy");

    const auto generation = workspaceGeneration(state_.device_id);
    auto *const backend = getBackendFor(state_.device_id);
    const auto execution = deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::FixedDepth);
    const bool hosted = execution == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    if (!generation || !backend || !loop.stream || !loop.capture || loop.launched ||
        execution == DeviceGenerationExecutionPolicy::Unsupported)
        return fail("Ordinary generation has no idle, admitted native graph owner");
    auto &context = GPUDeviceContextPool::instance().getContext(state_.device_id);
    void *const stream = loop.stream.get();
    auto &logical = device_resident_logical_sequence_state_storage_;
    if (!logical.validFor(1))
        return fail("Ordinary generation has no persistent logical frontier");

    const auto retention = (pipeline_tail || !expert_overlay)
        ? OrdinaryForwardRetentionKind::NativeComposable : ordinaryForwardRetentionKind();
    const auto prepared = materializeOrdinaryGenerationForward(logical.next_condition_tokens_device, retention, error);
    if (!prepared) return false;
    const auto &signature = prepared->signature;
    const auto *native_forward = prepared->native();
    const auto *hosted_forward = prepared->hosted();
    const bool hosted_overlay = retention == OrdinaryForwardRetentionKind::HostedExpertOverlay;
    if ((hosted_overlay && (!hosted || !expert_overlay || pipeline_tail)) ||
        hosted_overlay != static_cast<bool>(hosted_forward) ||
        (!hosted_overlay && !native_forward))
        return fail("Ordinary generation export differs from its explicit native/hosted composition");
    std::string detail;
    const bool owns_maintenance =
        usesParticipantLocalDeviceMoERebalanceController();
    const uint32_t *maintenance_due = nullptr;
    const IGPUGraphCapture *maintenance_capture = nullptr;
    if (owns_maintenance)
    {
        const auto &maintenance = device_moe_rebalance_maintenance_graph_;
        const auto *const controller =
            deviceMoERebalanceControllerStateDevice();
        if (!expert_overlay ||
            !controller || !maintenance.graph ||
            maintenance.workspace_generation != generation)
            return fail("Ordinary ExpertOverlay has no current device maintenance owner");
        maintenance_due = reinterpret_cast<const uint32_t *>(
            reinterpret_cast<const std::byte *>(controller) +
            offsetof(DeviceMoERebalanceGraphControllerState,
                     maintenance_due));
        const auto retained =
            maintenance.segment_cache.deviceLoopGraphTemplate(
                *maintenance.graph, &detail);
        if (!retained || !retained->capture)
            return fail("Ordinary ExpertOverlay has no retained maintenance graph: " + detail);
        maintenance_capture = retained->capture;
    }

    const auto seeds = arena_->getSharedTensor(BufferId::SAMPLING_REQUEST_SEEDS);
    SamplerStage::Params params;
    params.device_id = state_.device_id;
    params.backend = backend;
    params.logits = static_cast<float *>(state_.logits->gpu_data_ptr());
    params.vocab_size = params.logits_row_stride = state_.vocab_size;
    if (!greedy)
        params.policy = SamplerStage::Stochastic{law.top_k, law.top_p, law.temperature,
            seeds ? static_cast<const uint64_t *>(seeds->gpu_data_ptr()) : nullptr};
    params.workspace = {
        static_cast<float *>(stochastic_target_probs_dev_),
        static_cast<int32_t *>(stochastic_target_token_ids_dev_), kMaxTopK,
        static_cast<float *>(stochastic_topk_partial_vals_dev_),
        static_cast<int32_t *>(stochastic_topk_partial_idxs_dev_), stochastic_topk_partial_capacity_};
    params.publication = {
        .request_count = 1,
        .sampled_tokens = static_cast<int32_t *>(stochastic_target_sample_tokens_dev_),
        .stop_tokens = {static_cast<const int32_t *>(mtp_verifier_stop_tokens_dev_), kSpeculativeBatchMaxStopTokens, 0},
        .source = OrdinaryGenerationSampleSource::PrefillLogits,
        .response_tokens = device_generation_storage_.response_tokens_device,
        .response_token_stride = device_generation_storage_.response_token_stride,
        .control = device_generation_storage_.control_device,
        .control_stride = device_generation_storage_.control_stride,
        .frontier = {logical.target_cached_tokens_device, logical.next_condition_tokens_device,
            logical.stopped_flags_device, logical.publication_ok_flags_device, 1, state_.max_seq_len},
        .history = {static_cast<int32_t *>(mtp_generated_token_counts_dev_), state_.vocab_size,
            state_.vocab_size, 1}};
    params.penalties = law.has_penalties()
        ? static_cast<const MTPGreedyPenaltyPolicy *>(mtp_greedy_penalty_policy_dev_) : nullptr;

    if (loop.valid && loop.workspace_generation == generation &&
        loop.ordinary_sampling_identity == params && loop.ordinary_forward_identity == signature &&
        loop.ordinary_composition == composition &&
        loop.ordinary_maintenance_capture == maintenance_capture &&
        loop.ordinary_maintenance_due == maintenance_due &&
        (hosted_overlay
            ? (loop.source_fragments.empty() && loop.ordinary_hosted_forward == *hosted_forward)
            : (!loop.ordinary_hosted_forward && loop.source_fragments.size() == 1 &&
               loop.source_fragments.front().capture == native_forward->capture)))
    {
        // Leading-row ownership is request data, not graph topology. CUDA
        // reads the admitted control word; HIP uses this immutable admission
        // to submit its first complete transaction before observing tickets.
        loop.ordinary_leading = admission.initial_leading_row_disposition;
        PerfStatsCollector::addCounter("generation", "ordinary_graph_reuses", 1.0,
            "graph_setup", state_.device_id.toString());
        return true;
    }

    loop.invalidateGraph();
    // RAII closes capture even if a backend reports an asynchronous error.
    // These recordings are cloned by the complete parent, not launched as
    // independent per-stage work. Only HIP's initial sampler is executable.
    for (const auto index : {Plan::prefill_sampler, Plan::decode_sampler})
    {
        auto &child = loop.ordinary_children[index];
        child = context.createGraphCapture(stream);
        if (!child) return fail("Ordinary sampler has no capture owner");
        auto binding = params;
        binding.publication.source = index == Plan::prefill_sampler
            ? OrdinaryGenerationSampleSource::PrefillLogits : OrdinaryGenerationSampleSource::DecodeLogits;
        SamplerStage stage(binding);
        stage.setGPUStream(stream);
        bool recorded = false;
        {
            ScopedBackendGraphCapture recording(context, *child, "ordinary_generation_sampler");
            if (!recording.begin()) return fail("Ordinary sampler capture could not begin");
            recorded = stage.execute(getDeviceContext(state_.device_id));
            recording.finish();
        }
        if (!recorded) return fail("Ordinary sampler could not record its exact policy");
    }
    const DeviceGenerationGraphControl control{admission.depth_policy,
        device_generation_storage_.control_device, device_generation_storage_.control_stride, 1};
    const DeviceControlledLoopFragment initialization[] = {{
        .name = "ordinary_prefill_sample", .capture = loop.ordinary_children[Plan::prefill_sampler].get(),
        .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordZero,
        .condition_word_device = reinterpret_cast<const uint32_t *>(
            control.rows + kDeviceGenerationControlNextLeadingCommittedOutputCount)}};
    /*
     * A MainInference forward is the owner of the ExpertOverlay reader lease:
     * appendCapturedMainForwardMoEOverlayEpochTransaction decorates this exact
     * retained forward during materializeOrdinaryGenerationForward.  Keeping
     * the boundary in that graph is essential because it spans the real sparse
     * collective body.  Adding a second acquire/release around the sampler
     * would both duplicate the lease and borrow an independently invalidated
     * boundary cache after a normal decode recapture.
     */
    if (pipeline_tail)
    {
        // The rank installs activation/command edges around these recordings.
        // A local-only tail loop would advance KV without upstream layers.
    }
    else if (!hosted)
    {
        std::array<DeviceControlledLoopFragment, 3> iteration = {{
            {.name = "ordinary_resident_forward", .capture = native_forward->capture},
            {.name = "ordinary_decode_sample", .capture = loop.ordinary_children[Plan::decode_sampler].get()},
            {}}};
        if (owns_maintenance)
        {
            // The model's final routed layer publishes the serial commit clock.
            // Maintenance consumes that due word only after the sampled response
            // and the forward's captured epoch release have completed.
            iteration[2] = {
                .name = "ordinary_device_moe_maintenance",
                .capture = maintenance_capture,
                .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero,
                .condition_word_device = maintenance_due};
        }
        if (!DeviceGenerationGraphProgram::native(*loop.capture, control,
                {.initialization = initialization,
                 .iteration = std::span<const DeviceControlledLoopFragment>(
                     iteration.data(), 2u + static_cast<size_t>(owns_maintenance)),
                 .terminal=retainedStorageTerminalFragments()},
                detail))
            return fail(detail);
    }
    else if (hosted_overlay)
    {
        std::string retained_error;
        if (!forward_engine_->retainedDecodeGraph(signature, &retained_error) ||
            !loop.ordinary_children[Plan::prefill_sampler]->instantiate() ||
            !loop.ordinary_children[Plan::decode_sampler]->instantiate() ||
            !DeviceGenerationGraphProgram::ticketPublisher(*loop.capture, context, *backend, state_.device_id,
                control, maintenance_due, device_generation_storage_.dispatch_tickets_device, detail,retained_storage_boundary_.get()))
        {
            return fail("Ordinary ExpertOverlay hosted graph compilation failed: " +
                        (retained_error.empty() ? detail : retained_error));
        }
        loop.hosted_fragments.push_back({
            .name = "ordinary_expert_overlay_transaction",
            .kind = HostedDeviceGenerationFragment::Kind::OrdinaryExpertOverlayReplay,
            .capture = loop.ordinary_children[Plan::decode_sampler].get(),
            .forward_signature = signature,
        });
    }
    else
    {
        auto &transaction = loop.ordinary_children[Plan::hosted_transaction];
        transaction = context.createGraphCapture(stream);
        const GPUOrderedTimelineStep steps[] = {
            {.name = "ordinary_resident_forward", .capture = native_forward->capture},
            {.name = "ordinary_decode_sample", .capture = loop.ordinary_children[Plan::decode_sampler].get()}};
        if (!transaction || !transaction->buildOrderedTimelineTransaction(steps) || !transaction->instantiate() ||
            !loop.ordinary_children[Plan::prefill_sampler]->instantiate() ||
            !DeviceGenerationGraphProgram::ticketPublisher(*loop.capture, context, *backend, state_.device_id,
                control, maintenance_due, device_generation_storage_.dispatch_tickets_device, detail,retained_storage_boundary_.get()))
            return fail("Ordinary hosted graph compilation failed: " + detail);
        loop.hosted_fragments.push_back({.name = "ordinary_complete_transaction",
            .kind = HostedDeviceGenerationFragment::Kind::CapturedLocal, .capture = transaction.get()});
    }
    if (hosted && !pipeline_tail)
    {
        if (owns_maintenance)
        {
            // Hosted tickets retire this conditional tail before they admit
            // the next body; a period boundary can never be crossed early.
            loop.hosted_fragments.push_back({
                .name = "ordinary_device_moe_maintenance",
                .kind = HostedDeviceGenerationFragment::Kind::CapturedLocal,
                .capture = maintenance_capture,
                .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero,
                .condition_word_device = maintenance_due});
        }
        loop.branch_fragment_counts[0] = loop.hosted_fragments.size();
    }
    // The forward identity and bindings are immutable; all mutable counters,
    // seeds and frontier contents stay in their existing arena-owned banks.
    // Completion attribution belongs to the retained model forward, never to
    // the overlay lease child that happens to precede it in this composition.
    if (hosted_overlay)
        loop.ordinary_hosted_forward = *hosted_forward;
    else
        loop.source_fragments.push_back(
            {.name = "ordinary_resident_forward", .capture = native_forward->capture});
    loop.ordinary_sampling_identity = params;
    loop.ordinary_forward_identity = signature;
    loop.ordinary_composition = composition;
    loop.ordinary_maintenance_capture = maintenance_capture;
    loop.ordinary_maintenance_due = maintenance_due;
    loop.ordinary_leading = admission.initial_leading_row_disposition;
    loop.workspace_generation = generation;
    loop.request_count = 1;
    loop.depth_policy_mode = static_cast<int>(DeviceGenerationPolicyMode::Ordinary);
    loop.sampling_mode = sampling_mode;
    loop.fragment_count = hosted
        ? loop.branch_fragment_counts[0]
        : 3u + static_cast<size_t>(owns_maintenance);
    loop.execution_kind = hosted ? MTPDeviceGenerationLoopGraphCache::ExecutionKind::HostedDispatchTicketPublisher
                                : MTPDeviceGenerationLoopGraphCache::ExecutionKind::NativeConditionalParent;
    loop.valid = true;
    if (native_no_peer_tail)
        PerfStatsCollector::addCounter("generation", "ordinary_native_pipeline_nopeer_tail_materializations", 1.0,
            "graph_setup", state_.device_id.toString());
    PerfStatsCollector::addCounter("generation", "ordinary_graph_materializations", 1.0,
        "graph_setup", state_.device_id.toString());
    return true;
}

bool DeviceGraphOrchestrator::replayHostedOrdinaryExpertOverlayTransaction(
    const ForwardGraphSignature &signature,
    const IGPUGraphCapture *sampler,
    void **out_producer_stream,
    std::string *error)
{
    const auto fail = [&](const std::string &reason)
    {
        if (error)
            *error = reason;
        return false;
    };
    if (error)
        error->clear();
    if (out_producer_stream)
        *out_producer_stream = nullptr;

    auto &loop = deviceGenerationLoopGraph();
    if (!forward_engine_ || !sampler || !sampler->hasExecutable() ||
        !loop.stream ||
        (loop.ordinary_composition != OrdinaryGenerationComposition::ExpertOverlay &&
         loop.ordinary_composition != OrdinaryGenerationComposition::PipelineDomainExpertOverlayTail) ||
        !loop.ordinary_forward_identity ||
        *loop.ordinary_forward_identity != signature)
    {
        LOG_ERROR("[DeviceGraphOrchestrator] Hosted ordinary ExpertOverlay ownership mismatch"
                  << " device=" << state_.device_id.toString()
                  << " forward_engine=" << static_cast<bool>(forward_engine_)
                  << " sampler=" << static_cast<bool>(sampler)
                  << " sampler_executable="
                  << (sampler && sampler->hasExecutable())
                  << " stream=" << static_cast<bool>(loop.stream)
                  << " composition="
                  << (loop.ordinary_composition
                          ? static_cast<int>(*loop.ordinary_composition)
                          : -1)
                  << " forward_identity="
                  << static_cast<bool>(loop.ordinary_forward_identity)
                  << " signature_match="
                  << (loop.ordinary_forward_identity &&
                      *loop.ordinary_forward_identity == signature)
                  << " coordinator="
                  << static_cast<bool>(
                         moe_overlay_inference_transaction_coordinator_));
        return fail(
            "hosted ordinary ExpertOverlay replay has incomplete or divergent retained ownership");
    }

    std::string retained_error;
    const auto retained =
        forward_engine_->retainedDecodeGraph(signature, &retained_error);
    if (!retained)
        return fail(retained_error);
    if (!loop.ordinary_hosted_forward || *loop.ordinary_hosted_forward != *retained ||
        !loop.source_fragments.empty() || ordinaryForwardRetentionKind() != OrdinaryForwardRetentionKind::HostedExpertOverlay)
        return fail("Hosted ordinary ExpertOverlay retained owner changed before replay");

    MoEOverlayInferenceExecutionDescriptor descriptor{
        .graph_role = MoEOverlayInferenceGraphRole::MainDecode,
        .logical_step_id = 0,
        .placement_epoch =
            moe_overlay_inference_transaction_coordinator_
                ? moe_overlay_inference_transaction_coordinator_
                      ->currentPlacementEpoch()
                : 0,
        .request_count = 1,
        .logical_rows_per_request = 1,
        .physical_rows_per_request = 1,
        .draft_depth = -1,
        .sidecar_depth = -1,
    };
    MoEOverlayInferenceParticipantGraphScope graph_scope(
        moe_overlay_inference_transaction_coordinator_,
        descriptor,
        moe_overlay_inference_transaction_participant_index_);
    if (!graph_scope.ready() ||
        (moe_overlay_inference_transaction_coordinator_ &&
         !graph_scope.active()))
    {
        return fail(
            graph_scope.error().empty()
                ? "hosted ordinary ExpertOverlay replay has no active transaction binding"
                : graph_scope.error());
    }

    const auto request_identity = currentMoESparseRequestIdentity();
    const auto &binding = graph_scope.binding();
    if (!request_identity ||
        (graph_scope.active() &&
         (request_identity->generation() != binding.request_generation ||
          binding.descriptor.graph_role !=
              MoEOverlayInferenceGraphRole::MainDecode ||
          binding.descriptor.draft_depth != -1 ||
          binding.descriptor.sidecar_depth != -1)))
    {
        return fail(
            "hosted ordinary ExpertOverlay binding disagrees with its request generation or serial decode role");
    }
    const IComputeStage::MoEOverlayCollectiveRuntimeParams sparse_params{
        .generation_id = graph_scope.active()
                             ? binding.request_generation
                             : request_identity->generation(),
        .step_id = graph_scope.active()
                       ? binding.descriptor.logical_step_id
                       : 0,
        .execution_semantics =
            IComputeStage::MoEOverlayCollectiveRuntimeParams::
                ExecutionSemantics::Decode,
        .mtp_depth = -1,
        .placement_epoch = graph_scope.active()
                               ? binding.descriptor.placement_epoch
                               : 0,
    };
    if (!sparse_params.valid() || !sparse_params.hasExecutionSemantics())
    {
        return fail(
            "hosted ordinary ExpertOverlay replay produced an invalid sparse wire identity");
    }

    DeviceGraphExecutor::GraphLaunchDependencyHook launch_dependency =
        [&graph_scope, device = state_.device_id](
            DeviceGraphExecutor::GraphExecutableLaunchPhase,
            void *execution_stream) -> bool
    {
        // The complete main forward already captures its acquire/release
        // around sparse work. Only a remote follower needs a host-side ticket.
        if (!graph_scope.ownsTicketAuthority())
            return true;
        std::string arm_error;
        if (!graph_scope.armForExecutableLaunch(
                execution_stream, &arm_error))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Hosted ordinary decode could not arm its ExpertOverlay follower at the retained executable launch edge on "
                << device.toString() << ": " << arm_error);
            return false;
        }
        return true;
    };

    IDeviceContext *const ctx = getDeviceContext(state_.device_id);
    void *forward_stream = nullptr;
    bool submitted = forward_engine_->replayRetainedDecodeGraph(
        signature,
        ctx,
        sparse_params,
        launch_dependency,
        *this,
        loop.stream.get(),
        &forward_stream,
        &retained_error);
    if (submitted)
        submitted = sampler->launchOnStream(loop.stream.get());

    std::string finish_error;
    if (!graph_scope.finish(submitted, &finish_error))
        return fail(finish_error);
    if (!submitted)
    {
        return fail(
            retained_error.empty()
                ? "hosted ordinary ExpertOverlay sampler submission failed"
                : retained_error);
    }
    if (forward_stream != retained->stream)
    {
        return fail(
            "hosted ordinary ExpertOverlay forward published an invalid completion stream");
    }
    if (out_producer_stream)
        *out_producer_stream = loop.stream.get();
    return true;
}
}
